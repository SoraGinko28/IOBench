"""Buffered-file API instrumentation for DCP; never monkey-patches torch or os.

Offsets are logical file offsets, not block-device addresses. The proxy keeps a
cursor so instrumentation does not add tell/lseek calls to the measured stream.
"""

import hashlib
import json
import os
import pickle
import resource
import sys
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, perf_counter_ns

from metasl.procmem import process_counters

DATA_OPS = {"read", "readinto", "readline", "write"}
DATA_SUFFIXES = (".distcp", ".safetensors")


def proc_io():
    return {
        key: int(value)
        for key, value in (
            line.split(":") for line in Path("/proc/self/io").read_text().splitlines()
        )
    }


def rss_bytes():
    return int(Path("/proc/self/statm").read_text().split()[1]) * os.sysconf("SC_PAGE_SIZE")


def distribution(values):
    if not values:
        return {"count": 0}
    values = sorted(values)
    return {
        "count": len(values),
        "sum": sum(values),
        "min": values[0],
        "mean": sum(values) / len(values),
        "p50": values[(len(values) - 1) // 2],
        "p95": values[int((len(values) - 1) * 0.95)],
        "max": values[-1],
    }


def union_size(intervals):
    total = 0
    end = 0
    for start, stop in sorted(intervals):
        total += max(0, stop - max(start, end))
        end = max(end, stop)
    return total


class IORecorder:
    def __init__(self):
        self.files = {}
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.origin_ns = perf_counter_ns()
        self.in_flight = 0
        self.sealed = False

    def begin_data_call(self, is_data):
        with self.lock:
            if self.sealed:
                raise ValueError("Cannot record into a sealed IO recorder")
            self.in_flight += 1
            if is_data:
                self.active += 1
                self.max_active = max(self.max_active, self.active)

    def end_data_call(self, is_data):
        if is_data:
            with self.lock:
                self.active -= 1

    def end_recording_call(self):
        with self.lock:
            self.in_flight -= 1

    def add_file(self, name, mode):
        events = []
        with self.lock:
            if self.sealed:
                raise ValueError("Cannot add a file to a sealed IO recorder")
            if name in self.files:
                raise ValueError(f"Expected one open per phase: {name}")
            self.files[name] = {"mode": mode, "events": events, "origin_ns": self.origin_ns}
        return events




    def observation(self, deadline_ns=None):
        """Small API byte/boundary summary; unknown native IO is handled by callers."""
        if deadline_ns is not None and (type(deadline_ns) is not int or deadline_ns < 0):
            raise ValueError("deadline_ns must be a nonnegative integer or None")
        counts = Counter()
        first, last = {"read": None, "write": None}, {"read": None, "write": None}
        with self.lock:
            for name, entry in self.files.items():
                origin = entry.get("origin_ns", self.origin_ns)
                events = entry["events"]
                if isinstance(events, bytes):
                    events = pickle.loads(events)
                category = "data" if name.endswith(DATA_SUFFIXES) else "metadata"
                for event in events:
                    op = event["op"]
                    kind = "write" if op == "write" else "read" if op in DATA_OPS else None
                    if kind is None:
                        continue
                    counts[f"app_{category}_{kind}_bytes"] += event["completed"]
                    counts[f"app_{category}_{kind}_calls"] += 1
                    start = origin + event["start_ns"]
                    end = start + event["duration_ns"]
                    if deadline_ns is not None:
                        period = "within_T" if end <= deadline_ns else "after_T"
                        counts[f"app_{category}_{kind}_bytes_{period}"] += event["completed"]
                        if (
                            kind == "write"
                            and event["completed"] > 0
                            and start <= deadline_ns < end
                        ):
                            counts[f"crossing_{category}_write_calls"] += 1
                    if category == "data" and event["completed"] > 0:
                        first[kind] = start if first[kind] is None else min(first[kind], start)
                        last[kind] = end if last[kind] is None else max(last[kind], end)
        result = {
            f"app_{category}_{kind}_{metric}": counts[f"app_{category}_{kind}_{metric}"]
            for category in ("data", "metadata")
            for kind in ("read", "write")
            for metric in ("bytes", "calls")
        }
        return {
            **result,
            **{
                f"app_{category}_{kind}_bytes_{period}": (
                    counts[f"app_{category}_{kind}_bytes_{period}"]
                    if deadline_ns is not None
                    else None
                )
                for category in ("data", "metadata")
                for kind in ("read", "write")
                for period in ("within_T", "after_T")
            },
            "crossing_write_calls": counts["crossing_data_write_calls"]
            if deadline_ns is not None
            else None,
            "crossing_write_calls_scope": "nonzero recorded data writes",
            "crossing_metadata_write_calls": counts["crossing_metadata_write_calls"]
            if deadline_ns is not None
            else None,
            "recorder_origin_ns": self.origin_ns,
            **{f"first_data_{kind}_start_ns": first[kind] for kind in ("read", "write")},
            **{f"last_data_{kind}_end_ns": last[kind] for kind in ("read", "write")},
            "max_simultaneous_data_calls": self.max_active,
        }

    def pack_file(self, name):
        """Release closed metadata streams' event objects before metadata pruning.

        Events are allocated between the unpickler's metadata objects. Keeping
        even a few hundred dictionaries alive pins almost every pymalloc arena
        from the full metadata graph, defeating the reader's heap release. A
        single byte string preserves the complete trace without those survivors.
        Clear the original list too: the stream proxy may still reference it.
        """
        with self.lock:
            entry = self.files[name]
            events = entry["events"]
            if isinstance(events, bytes):
                return
            packed = pickle.dumps(events, protocol=pickle.HIGHEST_PROTOCOL)
            events.clear()
            entry["events"] = packed

    def summary(self, trace_path=None):
        # Called after DCP returns, when the reader's full metadata is gone.
        with self.lock:
            for entry in self.files.values():
                if isinstance(entry["events"], bytes):
                    entry["events"] = pickle.loads(entry["events"])
        if trace_path is not None:
            with Path(trace_path).open("w") as out:
                for name, entry in sorted(self.files.items()):
                    for event in entry["events"]:
                        origin = {"origin_ns": entry["origin_ns"]} if "component" in entry else {}
                        out.write(json.dumps({"file": name, **origin, **event}) + "\n")
        output = {
            "origin_ns": self.origin_ns,
            "clock": "monotonic_perf_counter_ns",
            "layer": "Python buffered-file API (not syscalls)",
            "max_simultaneous_data_calls": self.max_active,
            "event_origin_policy": "per_file_component"
            if any("component" in entry for entry in self.files.values())
            else "recorder_origin",
            "file_origins_ns": {
                name: entry.get("origin_ns", self.origin_ns) for name, entry in self.files.items()
            },
            "files": {},
        }
        totals = Counter()
        all_sizes = {"read": [], "write": []}
        all_durations = {"read": [], "write": []}
        all_threads = set()
        digest = hashlib.sha256()
        for name, entry in sorted(self.files.items()):
            events = entry["events"]
            is_data = name.endswith(DATA_SUFFIXES)
            counts = Counter(e["op"] for e in events)
            stats = {
                "mode": entry["mode"],
                "origin_ns": entry.get("origin_ns", self.origin_ns),
                "component": entry.get("component"),
                "calls": dict(counts),
            }
            for kind in ("read", "write"):
                selected = [
                    e
                    for e in events
                    if (
                        e["op"] == "write"
                        if kind == "write"
                        else e["op"] in ("read", "readinto", "readline")
                    )
                ]
                sizes = [e["completed"] for e in selected]
                intervals = [
                    (e["offset"], e["offset"] + e["completed"]) for e in selected if e["completed"]
                ]
                transitions = Counter()
                previous_end = None
                for start, end in intervals:
                    if previous_end is not None:
                        transitions[
                            "contiguous"
                            if start == previous_end
                            else "backward_or_overlap"
                            if start < previous_end
                            else "forward_gap"
                        ] += 1
                    previous_end = end
                unique = union_size(intervals)
                stats[kind] = {
                    "bytes": sum(sizes),
                    "calls": len(selected),
                    "unique_bytes": unique,
                    "repeated_bytes": sum(sizes) - unique,
                    "size_bytes": distribution(sizes),
                    "size_histogram": dict(sorted(Counter(sizes).items())),
                    "offset_transitions": dict(transitions),
                    "call_duration_ns": distribution([e["duration_ns"] for e in selected]),
                }
                totals[f"{'data' if is_data else 'metadata'}_{kind}_bytes"] += sum(sizes)
                if is_data:
                    totals[f"data_{kind}_calls"] += len(selected)
                    totals[f"data_{kind}_repeated_bytes"] += sum(sizes) - unique
                    all_sizes[kind].extend(sizes)
                    all_durations[kind].extend(e["duration_ns"] for e in selected)
                    all_threads.update(e["thread_id"] for e in selected)
                    for key, value in transitions.items():
                        totals[f"data_{kind}_{key}"] += value
            seeks = [e for e in events if e["op"] == "seek"]
            stats["seek_directions"] = dict(
                Counter(
                    "backward"
                    if e["completed"] < e["offset"]
                    else "forward"
                    if e["completed"] > e["offset"]
                    else "same"
                    for e in seeks
                )
            )
            signature = hashlib.sha256()
            for event in events:
                if event.get("request") is False:
                    continue
                # Exclude clocks/thread IDs/payload: compare per-file access patterns.
                signature.update(
                    json.dumps(
                        [
                            event["op"],
                            event["offset"],
                            event["requested"],
                            event["completed"],
                            event.get("whence"),
                        ]
                    ).encode()
                )
            stats["pattern_sha256"] = signature.hexdigest()
            if is_data:
                digest.update((name + signature.hexdigest()).encode())
                totals["data_seek_calls"] += len(seeks)
                totals["data_backward_seeks"] += stats["seek_directions"].get("backward", 0)
            output["files"][name] = stats
        output.update(totals)
        observation = self.observation()
        output["first_data_write_start_ns"] = observation["first_data_write_start_ns"]
        output["last_data_write_end_ns"] = observation["last_data_write_end_ns"]
        output["first_data_read_start_ns"] = observation["first_data_read_start_ns"]
        output["last_data_read_end_ns"] = observation["last_data_read_end_ns"]
        output["data_threads_observed"] = len(all_threads)
        output["data_pattern_sha256"] = digest.hexdigest()
        for kind in ("read", "write"):
            output[f"data_{kind}_size_bytes"] = distribution(all_sizes[kind])
            output[f"data_{kind}_size_histogram"] = dict(sorted(Counter(all_sizes[kind]).items()))
            output[f"data_{kind}_call_duration_ns"] = distribution(all_durations[kind])
        return output




class RecordedStream:
    """Duck-typed stream proxy; every operation still uses the original stream."""

    def __init__(self, stream, recorder, name, mode, *, capture_lifecycle=False):
        self.stream = stream
        self.recorder = recorder
        self.events = recorder.add_file(name, mode)
        self.cursor = 0  # DCP opens new rb/wb streams at offset zero.
        self.is_data = name.endswith(DATA_SUFFIXES)
        self.capture_lifecycle = capture_lifecycle

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def _call(self, op, requested, function, *args, whence=None):
        offset = self.cursor
        active = self.is_data and op in DATA_OPS
        self.recorder.begin_data_call(active)
        try:
            started = perf_counter_ns()
            try:
                value = function(*args)
            finally:
                ended = perf_counter_ns()
                self.recorder.end_data_call(active)
            count = len(value) if isinstance(value, bytes) else value
            if op == "seek":
                self.cursor = count
            elif op in DATA_OPS:
                self.cursor += count
            event = dict(
                op=op,
                offset=offset,
                requested=requested,
                completed=count,
                start_ns=started - self.recorder.origin_ns,
                duration_ns=ended - started,
                thread_id=threading.get_native_id(),
            )
            if whence is not None:
                event["whence"] = whence
            self.events.append(event)
            return value
        finally:
            self.recorder.end_recording_call()

    def read(self, size=-1):
        return self._call("read", size, self.stream.read, size)

    def readinto(self, buffer):
        return self._call("readinto", memoryview(buffer).nbytes, self.stream.readinto, buffer)

    def readline(self, size=-1):
        return self._call("readline", size, self.stream.readline, size)

    def write(self, data):
        return self._call("write", memoryview(data).nbytes, self.stream.write, data)

    def seek(self, offset, whence=os.SEEK_SET):
        return self._call("seek", offset, self.stream.seek, offset, whence, whence=whence)

    def _lifecycle(self, op, function, *args):
        if not self.capture_lifecycle:
            return function(*args)
        self.recorder.begin_data_call(False)
        started, successful = perf_counter_ns(), False
        try:
            result = function(*args)
            successful = True
            return result
        finally:
            ended = perf_counter_ns()
            self.events.append(
                dict(
                    op=op,
                    request=False,
                    offset=0,
                    requested=0,
                    completed=0,
                    start_ns=started - self.recorder.origin_ns,
                    duration_ns=ended - started,
                    success=int(successful),
                    thread_id=threading.get_native_id(),
                )
            )
            self.recorder.end_recording_call()

    def flush(self):
        return self._lifecycle("flush", self.stream.flush)

    def close(self):
        return self._lifecycle("close", self.stream.close)

    def fsync(self):
        return self._lifecycle("fsync", fsync_stream, self.stream)


def fsync_stream(stream):
    """Fsync the same descriptor, measuring it when the stream opts into tracing."""
    method = getattr(stream, "fsync", None)
    if method is not None:
        return method()
    return os.fsync(stream.fileno())


class RecordedFileSystem:
    """Wrap an instance's filesystem; no torch class/module is modified."""

    def __init__(self, fs, recorder, *, capture_lifecycle=False, capture_open=False):
        self.fs, self.recorder = fs, recorder
        self.capture_lifecycle = capture_lifecycle
        self.capture_open = capture_open

    def __getattr__(self, name):
        return getattr(self.fs, name)

    @contextmanager
    def create_stream(self, path, mode):
        recorded = None
        name = Path(path).name
        try:
            context = self.fs.create_stream(path, mode)
            opened_started = perf_counter_ns()
            stream = context.__enter__()
            opened_ended = perf_counter_ns()
            try:
                recorded = RecordedStream(
                    stream, self.recorder, name, mode, capture_lifecycle=self.capture_lifecycle
                )
                if self.capture_open:
                    recorded.events.append(dict(
                        op="open", request=False, io_layer="lifecycle", offset=0,
                        requested=0, completed=0, start_ns=opened_started - self.recorder.origin_ns,
                        duration_ns=opened_ended - opened_started, success=True,
                        thread_id=threading.get_native_id(), mode=mode,
                    ))
            except BaseException:
                context.__exit__(*sys.exc_info())
                raise
            try:
                yield recorded
            except BaseException:
                if not recorded._lifecycle("close", context.__exit__, *sys.exc_info()):
                    raise
            else:
                recorded._lifecycle("close", context.__exit__, None, None, None)
        finally:
            if recorded is not None and not recorded.is_data:
                self.recorder.pack_file(name)


def measure_call(call, checkpoint_bytes=None):
    """Time the call; sample current RSS at 10 ms and collect process counters."""
    stop = threading.Event()
    peak = [rss_bytes()]

    def sample():
        while not stop.wait(0.01):
            peak[0] = max(peak[0], rss_bytes())

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    before_usage = resource.getrusage(resource.RUSAGE_SELF)
    before_io = proc_io()
    before_rss = rss_bytes()
    started = perf_counter()
    try:
        value = call()
    finally:
        seconds = perf_counter() - started
        after_usage = resource.getrusage(resource.RUSAGE_SELF)
        after_io = proc_io()
        after_rss = rss_bytes()
        stop.set()
        sampler.join()
    peak[0] = max(peak[0], before_rss, after_rss)
    fields = (
        "ru_utime",
        "ru_stime",
        "ru_minflt",
        "ru_majflt",
        "ru_nvcsw",
        "ru_nivcsw",
        "ru_inblock",
        "ru_oublock",
    )
    counters = {key: getattr(after_usage, key) - getattr(before_usage, key) for key in fields}
    return value, {
        "seconds": seconds,
        "resource_delta": counters,
        "cpu_seconds": counters["ru_utime"] + counters["ru_stime"],
        "average_cpu_cores": (counters["ru_utime"] + counters["ru_stime"]) / seconds,
        "proc_io_delta": {key: after_io[key] - before_io[key] for key in before_io},
        "rss_start_bytes": before_rss,
        "rss_end_bytes": after_rss,
        "sampled_peak_rss_bytes": peak[0],
        "rss_sample_interval_ms": 10,
        "process_high_water_rss_bytes": after_usage.ru_maxrss * 1024,
    }


def advise_evict_checkpoint(path):
    """Best-effort eviction of this checkpoint only; no system-wide drop_caches."""
    files = []
    path = Path(path)
    for file in checkpoint_files(path):
        with file.open("rb") as stream:
            os.fsync(stream.fileno())
            os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
        files.append(file.relative_to(path).as_posix())
    return {"method": "POSIX_FADV_DONTNEED", "files": files, "guaranteed_cold": False}


def warm_checkpoint(path):
    """Read all data files before starting workers; never writes or drops global cache."""
    started = time.perf_counter()
    before = process_counters(os.getpid())
    buffer = bytearray(16 * 1024**2)
    count = 0
    files = [file for file in checkpoint_files(path) if file.name.endswith(DATA_SUFFIXES)]
    for file in files:
        with file.open("rb", buffering=0) as stream:
            while size := stream.readinto(buffer):
                count += size
    after = process_counters(os.getpid())
    if count == 0:
        raise ValueError("No nonempty checkpoint data files to warm")
    return dict(
        seconds=time.perf_counter() - started,
        data_bytes=count,
        process_read_bytes=after["read_bytes"] - before["read_bytes"],
    )


def checkpoint_files(path):
    """Regular checkpoint payload and metadata files.

    Completion validation rejects symlinks before cache controls run; traversal
    itself also excludes file links and does not follow directory links.
    """
    return sorted(
        file for file in Path(path).rglob("*") if not file.is_symlink() and file.is_file()
    )
