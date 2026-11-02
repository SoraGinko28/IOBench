"""Tensor-free FIFO permit admission shared by save writers and load readers.

Permits only: no shared slots, tensors, addresses or buffer ownership. A rank
sends a request (or 'empty' for an empty local plan) and later a release, never
requesting another permit while one is pending or held. The launcher grants at most `limit` concurrent
permits in arrival order, starting once every rank has reported in once.
An optional preparation handshake establishes the memory baseline before the
checkpoint call, while the real IO requests still establish permit order.
"""

import collections
import json
import math
import threading
from time import perf_counter, perf_counter_ns

GIB = 1024**3


class PermitClient:
    def __init__(self, connection, abort, timeout):
        self.connection, self.abort, self.timeout = connection, abort, timeout
        self.request_index = 0
        self.last_acquire_timing = None
        self.preparation_timing = None

    def prepare(self):
        """Wait for the safe group baseline before starting the checkpoint timer.

        This handshake is separate from permit requests: it does not consume a
        request index or establish the FIFO order of the real IO requests.
        """
        if self.preparation_timing is not None or self.request_index:
            raise RuntimeError("Admission preparation must precede the first request")
        start = perf_counter()
        self.connection.send(("prepare", None))
        while True:
            if self.abort.is_set():
                raise RuntimeError("Permit queue aborted by peer or supervisor")
            if self.connection.poll(0.1):
                kind, timing = self.connection.recv()
                if kind != "prepared":
                    raise RuntimeError("Admission preparation rejected")
                self.preparation_timing = dict(timing, client_prepared_ns=perf_counter_ns())
                return self.preparation_timing
            if perf_counter() - start > self.timeout:
                self.fail()
                raise TimeoutError("Admission preparation timeout")

    def acquire(self, empty=False):
        start = perf_counter()
        self.request_index += 1
        request_ns = perf_counter_ns()
        self.last_acquire_timing = None
        self.connection.send(
            (
                "empty" if empty else "request",
                dict(request_send_ns=request_ns, client_request_index=self.request_index),
            )
        )
        while True:
            if self.abort.is_set():
                raise RuntimeError("Permit queue aborted by peer or supervisor")
            if self.connection.poll(0.1):
                kind, token = self.connection.recv()
                if kind != "grant":
                    raise RuntimeError("Permit rejected")
                received_ns = perf_counter_ns()
                timing = token.get("timing", {}) if isinstance(token, dict) else {}
                self.last_acquire_timing = dict(
                    timing,
                    request_send_ns=request_ns,
                    client_request_index=self.request_index,
                    client_receive_ns=received_ns,
                    empty=empty,
                )
                return token["token"] if isinstance(token, dict) else token
            if perf_counter() - start > self.timeout:
                self.fail()
                raise TimeoutError("Permit timeout")

    def release(self, token, completed):
        self.connection.send(("release", (token, completed)))

    def fail(self):
        self.abort.set()
        try:
            self.connection.send(("error", perf_counter()))
        except (EOFError, OSError):
            pass


class PermitQueue:
    """One fixed FIFO permit per nonempty rank plan, independent of worker threads."""

    def __init__(
        self,
        connections,
        abort,
        limit,
        output,
        role,
        soft_pss_bytes=20 * GIB,
        soft_available_bytes=7 * GIB,
        max_sample_age_seconds=0.5,
    ):
        if limit < 1:
            raise ValueError(f"{role} limit must be positive")
        if role not in ("writer", "reader", "io"):
            raise ValueError("Unknown permit role")
        if not math.isfinite(max_sample_age_seconds) or max_sample_age_seconds <= 0:
            raise ValueError("Sample freshness interval must be positive and finite")
        self.connections, self.abort, self.limit, self.role = connections, abort, limit, role
        self.soft_pss_bytes, self.soft_available_bytes = soft_pss_bytes, soft_available_bytes
        self.max_sample_age_ns = round(max_sample_age_seconds * 1_000_000_000)
        if self.max_sample_age_ns <= 0:
            raise ValueError("Sample freshness interval must be at least one nanosecond")
        self.pending = collections.deque()
        self.active, self.seen = set(), set()
        self.prepared = set()
        self.prepare_generation = 0
        self.all_prepare_ready_ns = None
        self.rounds = collections.Counter()
        self.baseline = None
        self.baseline_ready_ns = None
        self.baseline_sample = None
        self.request_generation = 0
        self.live_generation = 0
        self.all_request_ready_ns = None
        self.requests = {}
        self.sample_needed = threading.Event()
        self.work_available = threading.Event()
        self.error = None
        self.lock = threading.RLock()
        self.stop = threading.Event()
        self.log = (output / f"{role}-queue.jsonl").open("w", buffering=1)
        self.thread = threading.Thread(target=self.run, daemon=True)

    def event(self, kind, **details):
        with self.lock:
            now_ns = perf_counter_ns()
            record = dict(
                time=now_ns / 1_000_000_000,
                time_ns=now_ns,
                event=kind,
                role=self.role,
                limit=self.limit,
                active=len(self.active),
                waiting=len(self.pending),
                baseline_bytes=self.baseline,
                request_generation=self.request_generation,
                live_generation=self.live_generation,
            )
            record.update(details)
            self.log.write(json.dumps(record) + "\n")

    def _live_ranks(self):
        return [rank for rank, conn in enumerate(self.connections) if conn is not None]

    def _refresh_initial_request_set(self):
        """Keep the all-rank request gate independent of pre-call preparation."""
        if self.baseline is not None and self.all_request_ready_ns is not None:
            return
        live = self._live_ranks()
        self.all_request_ready_ns = (
            perf_counter_ns() if live and all(rank in self.seen for rank in live) else None
        )
        if self.baseline is None and not self.prepared and self.all_request_ready_ns is not None:
            self.event(
                "sample_requested",
                all_request_ready_ns=self.all_request_ready_ns,
                required_ranks=live,
            )
            self.sample_needed.set()

    def _refresh_preparation_set(self):
        if self.baseline is not None or not self.prepared:
            return
        live = self._live_ranks()
        self.all_prepare_ready_ns = (
            perf_counter_ns() if live and all(rank in self.prepared for rank in live) else None
        )
        if self.all_prepare_ready_ns is not None:
            self.event(
                "sample_requested",
                all_prepare_ready_ns=self.all_prepare_ready_ns,
                required_ranks=live,
            )
            self.sample_needed.set()

    def begin_sample(self, sample_start_ns=None):
        """Snapshot and clear notifications atomically, before reading /proc outside the lock."""
        with self.lock:
            self.sample_needed.clear()
            ticket = dict(
                request_generation=self.request_generation,
                live_generation=self.live_generation,
                prepare_generation=self.prepare_generation,
                required_ranks=self._live_ranks(),
                all_request_ready_ns=self.all_request_ready_ns,
                all_prepare_ready_ns=self.all_prepare_ready_ns,
                sample_start_ns=perf_counter_ns() if sample_start_ns is None else sample_start_ns,
            )
            self.event("sample_begin", sample_ticket=ticket)
            return ticket

    def check_memory(self, sample):
        """Threshold check alone cannot establish a baseline or grant a permit."""
        with self.lock:
            if (
                sample["total_pss_bytes"] >= self.soft_pss_bytes
                or sample["available_bytes"] < self.soft_available_bytes
            ):
                self.error = f"Fixed {self.role} admission stopped at soft memory threshold"
                self.abort.set()
                self.sample_needed.set()
                self.work_available.set()
                return False
            return not self.abort.is_set() and not self.stop.is_set()

    def update(self, sample, completed_bytes=0):
        with self.lock:
            if not self.check_memory(sample):
                return False
            if self.baseline is not None:
                return True
            before_timer = bool(self.prepared)
            ready_ns = self.all_prepare_ready_ns if before_timer else self.all_request_ready_ns
            if ready_ns is None:
                return False
            ticket = sample.get("admission_sample_ticket")
            start_ns, end_ns = sample.get("sample_start_ns"), sample.get("sample_end_ns")
            now_ns = perf_counter_ns()
            valid_clock = (
                type(start_ns) is int
                and type(end_ns) is int
                and ready_ns <= start_ns <= end_ns <= now_ns
                and now_ns - end_ns <= self.max_sample_age_ns
            )
            valid_ticket = (
                isinstance(ticket, dict)
                and ticket.get("sample_start_ns") == start_ns
                and ticket.get("request_generation") == self.request_generation
                and ticket.get("live_generation") == self.live_generation
                and ticket.get("prepare_generation") == self.prepare_generation
                and ticket.get("required_ranks") == self._live_ranks()
                and ticket.get("all_request_ready_ns") == self.all_request_ready_ns
                and ticket.get("all_prepare_ready_ns") == self.all_prepare_ready_ns
            )
            if not valid_clock or not valid_ticket:
                self.event(
                    "sample_rejected", reason="stale_clock_or_generation", sample_ticket=ticket
                )
                self.sample_needed.set()
                return False
            self.baseline = sample["total_pss_bytes"]
            self.baseline_ready_ns = perf_counter_ns()
            self.baseline_sample = dict(
                sample_start_ns=start_ns,
                sample_end_ns=end_ns,
                sample_request_generation=self.request_generation,
                sample_live_generation=self.live_generation,
                sample_pss_bytes=sample["total_pss_bytes"],
                sample_available_bytes=sample["available_bytes"],
                all_request_ready_ns=self.all_request_ready_ns,
                all_prepare_ready_ns=self.all_prepare_ready_ns,
                admission_baseline_before_timer=before_timer,
            )
            self.event("baseline", baseline_ready_ns=self.baseline_ready_ns, **self.baseline_sample)
            if before_timer:
                for rank in self._live_ranks():
                    self.connections[rank].send(
                        ("prepared", dict(self.baseline_sample, baseline_ready_ns=self.baseline_ready_ns))
                    )
            # The queue need not wait for its periodic pipe poll after a safety sample.
            self.work_available.set()
            return True

    def snapshot(self):
        with self.lock:
            return {
                f"active_{self.role}s": len(self.active),
                f"queued_{self.role}s": len(self.pending),
                f"{self.role}_limit": self.limit,
                "baseline_bytes": self.baseline,
                "baseline_ready_ns": self.baseline_ready_ns,
                "all_request_ready_ns": self.all_request_ready_ns,
                "all_prepare_ready_ns": self.all_prepare_ready_ns,
                "admission_baseline_before_timer": bool(self.prepared),
                "request_generation": self.request_generation,
                "live_generation": self.live_generation,
                "admission_kind": "permit_only",
            }

    def _grant(self, rank, token, *, empty=False):
        if self.stop.is_set() or self.abort.is_set():
            return False
        timing = dict(
            self.baseline_sample or {},
            **self.requests[rank],
            grant_ns=perf_counter_ns(),
            baseline_ready_ns=self.baseline_ready_ns,
            all_request_ready_ns=self.all_request_ready_ns,
            queue_round=self.rounds[rank],
            rank=rank,
            role=self.role,
            empty=empty,
        )
        self.event("empty" if empty else "grant", **timing)
        self.connections[rank].send(("grant", dict(token=token, timing=timing)))
        return True

    def handle(self, rank, kind, value):
        if kind == "prepare":
            if rank in self.prepared or self.seen or self.baseline is not None:
                raise RuntimeError("Admission preparation must precede all requests")
            self.prepared.add(rank)
            self.prepare_generation += 1
            self.event("prepare", rank=rank)
            self._refresh_preparation_set()
        elif kind in ("request", "empty"):
            if self.prepared and self.baseline is None:
                raise RuntimeError("IO request before admission preparation completed")
            if rank in self.pending or rank in self.active:
                raise RuntimeError(f"Duplicate {self.role} request")
            self.seen.add(rank)
            self.rounds[rank] += 1
            self.request_generation += 1
            self.requests[rank] = dict(
                server_receive_ns=perf_counter_ns(),
                server_enqueue_ns=perf_counter_ns(),
                request_send_ns=value.get("request_send_ns") if isinstance(value, dict) else None,
                client_request_index=value.get("client_request_index")
                if isinstance(value, dict)
                else None,
            )
            if kind == "empty":
                self._grant(rank, None, empty=True)
            else:
                self.pending.append(rank)
                self.event("enqueue", rank=rank, round=self.rounds[rank], **self.requests[rank])
            self._refresh_initial_request_set()
        elif kind == "release":
            token, completed = value
            if rank not in self.active or token != rank:
                raise RuntimeError(f"Invalid {self.role} release")
            self.active.remove(rank)
            self.event("release", rank=rank, completed_time=completed, round=self.rounds[rank])
        else:
            raise RuntimeError(f"Worker reported {self.role} failure")

    def run(self):
        try:
            while not self.stop.is_set() and not self.abort.is_set():
                self.work_available.clear()
                with self.lock:
                    for rank, conn in enumerate(self.connections):
                        if conn is None:
                            continue
                        while conn.poll():
                            try:
                                kind, value = conn.recv()
                            except EOFError:
                                self.connections[rank] = None
                                self.live_generation += 1
                                # A rank that never needed a permit (no save happened) may
                                # exit; worker crashes are caught by the supervisor's exit codes.
                                if rank in self.active or rank in self.pending:
                                    raise RuntimeError(
                                        f"Worker disconnected while holding or awaiting {self.role}"
                                    ) from None
                                self._refresh_initial_request_set()
                                self._refresh_preparation_set()
                                break
                            self.handle(rank, kind, value)
                    while (
                        self.baseline is not None
                        and self.all_request_ready_ns is not None
                        and self.pending
                        and not self.abort.is_set()
                        and not self.stop.is_set()
                        and len(self.active) < self.limit
                    ):
                        rank = self.pending.popleft()
                        self.active.add(rank)
                        if not self._grant(rank, rank):
                            self.active.remove(rank)
                            self.pending.appendleft(rank)
                self.work_available.wait(0.01)
        except BaseException as exc:
            self.error = repr(exc)
            self.abort.set()
            self.sample_needed.set()
            self.work_available.set()
        finally:
            if self.abort.is_set():
                self.event("abort", reason=self.error or "Peer aborted")
                self.sample_needed.set()

    def close(self):
        with self.lock:
            self.stop.set()
            self.sample_needed.set()
            self.work_available.set()
        self.thread.join(timeout=2)
        self.log.close()
        for conn in self.connections:
            if conn:
                conn.close()
