"""Export reference-shaped client traces and compare measured per-rank requests.

The recorder exports events grouped by filename. Actual invocation order is
reconstructed from its nanosecond clock, independently for each rank. Exporting
never reads checkpoint payloads and never invents filesystem RPCs or syscalls.
"""

import hashlib
import itertools
import json
import posixpath
from collections import Counter
from pathlib import Path

SCHEMA_VERSION = 1
OPERATIONS = {"read", "readinto", "readline", "write", "seek", "pread", "pwrite"}
AUXILIARY_OPERATIONS = {"fsync", "flush", "close", "open", "ftruncate", "barrier"}
REFERENCE_FIELDS = ("op", "t_us", "path", "offset", "size", "cost_us", "success")
SIGNATURE_FIELDS = ("rank", "path", "op", "offset", "requested", "completed", "whence", "success")


class TraceEvidenceError(ValueError):
    """Missing, malformed, or incompatible evidence cannot establish equality."""


def _integer(value, label, *, minimum=0):
    if type(value) is not int or (minimum is not None and value < minimum):
        raise TraceEvidenceError(f"{label} must be an integer >= {minimum}")
    return value


def _json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _read_jsonl(path):
    path = Path(path)
    raw = path.read_bytes()
    rows = []
    for number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except (ValueError, UnicodeError) as error:
            raise TraceEvidenceError(f"Invalid JSON at {path}:{number}: {error}") from error
        if not isinstance(row, dict):
            raise TraceEvidenceError(f"Expected an event object at {path}:{number}")
        rows.append(row)
    return rows, dict(
        path=str(path.resolve()), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()
    )


def _path(value, checkpoint_root):
    if not isinstance(value, str) or not value or "\x00" in value:
        raise TraceEvidenceError("Event file must be a nonempty path")
    value = posixpath.normpath(value)
    if value.startswith("/"):
        if checkpoint_root is None:
            raise TraceEvidenceError("An absolute event path requires checkpoint_root")
        root = posixpath.normpath(str(checkpoint_root))
        if not root.startswith("/") or not value.startswith(root.rstrip("/") + "/"):
            raise TraceEvidenceError("Event file is outside checkpoint_root")
        value = posixpath.relpath(value, root)
    if value == "." or value == ".." or value.startswith("../"):
        raise TraceEvidenceError("Event file escapes the checkpoint")
    return value


def _request(row, *, rank, checkpoint_root):
    op = row.get("op")
    if op not in OPERATIONS | AUXILIARY_OPERATIONS:
        raise TraceEvidenceError(f"Unsupported measured operation: {op!r}")
    request = op in OPERATIONS
    if type(row.get("request", request)) is not bool or row.get("request", request) != request:
        raise TraceEvidenceError("request must identify ordinary requests versus lifecycle events")
    requested = _integer(row.get("requested"), "requested", minimum=None)
    completed = _integer(row.get("completed"), "completed")
    whence = row.get("whence")
    if op == "seek":
        if type(whence) is not int or whence not in (0, 1, 2):
            raise TraceEvidenceError("Seek requires whence=0, 1, or 2")
    elif whence is not None or requested < -1 or (requested >= 0 and completed > requested):
        raise TraceEvidenceError("Impossible data request length or whence")
    offset = _integer(row.get("offset"), "offset")
    if not request and (offset != 0 or requested != 0 or completed != 0):
        raise TraceEvidenceError("Lifecycle events cannot contain request bytes or offsets")
    success = row.get("success", 1)
    if type(success) is bool:
        success = int(success)
    if type(success) is not int or success not in (0, 1):
        raise TraceEvidenceError("success must be 0 or 1")
    return dict(
        rank=rank,
        path=_path(row.get("file", row.get("path")), checkpoint_root),
        op=op,
        request=request,
        offset=offset,
        requested=requested,
        completed=completed,
        whence=whence,
        success=success,
    )


def _clock_order(rows):
    ordered = sorted(rows, key=lambda row: (row["start_ns"], row["seq"]))
    ties = overlaps = 0
    last_start, end = None, -1
    for row in ordered:
        if row["op"] in AUXILIARY_OPERATIONS:
            continue
        ties += row["start_ns"] == last_start
        overlaps += row["start_ns"] < end
        end = max(end, row["start_ns"] + row["duration_ns"])
        last_start = row["start_ns"]
    return ordered, dict(
        start_timestamp_ties=ties,
        overlapping_call_starts=overlaps,
        serial_rank_order_verifiable=any(row["op"] in OPERATIONS for row in rows)
        and not ties
        and not overlaps,
    )


def _signatures(rows):
    signatures = [
        tuple(row[field] for field in SIGNATURE_FIELDS) for row in rows if row["op"] in OPERATIONS
    ]
    digest = hashlib.sha256()
    for signature in signatures:
        digest.update(_json_bytes(signature) + b"\n")
    return signatures, digest.hexdigest()


def export_rank_trace(
    raw_trace,
    output_dir,
    *,
    rank,
    phase,
    origin_ns=None,
    checkpoint_root=None,
    payload_events=None,
):
    """Export an IORecorder JSONL plus optional genuinely measured payload calls.

    ``origin_ns`` is the recorder summary's origin when raw events do not carry
    one. Optional ``payload_events`` use *absolute* perf_counter_ns ``start_ns``
    (or ``absolute_start_ns``), plus duration_ns and the ordinary request fields.
    Both sources must use the same process's monotonic clock. No raw trace may
    be omitted; a missing trace is missing evidence, even if payload calls exist.
    The returned manifest contains the exact output path for the matrix driver.
    """
    _integer(rank, "rank")
    if phase not in ("save", "load"):
        raise TraceEvidenceError("phase must be save or load")
    if origin_ns is not None:
        _integer(origin_ns, "origin_ns")
    raw_rows, raw_source = _read_jsonl(raw_trace)
    payload_events = list(payload_events or [])
    if origin_ns is None and len({"origin_ns" in row for row in raw_rows}) > 1:
        raise TraceEvidenceError("Mixed explicit and missing recorder origins require origin_ns")
    if payload_events and any("origin_ns" not in row for row in raw_rows) and origin_ns is None:
        raise TraceEvidenceError("Combining absolute payload calls requires recorder origin_ns")
    rows = []

    # New gdtensor raw traces contain the positioned calls as well. Older
    # reports still supply those same absolute events separately; include each
    # exactly once and keep staging spans out of the filesystem request export.
    def payload_key(event, start):
        return tuple(
            event.get(key)
            for key in (
                "file",
                "op",
                "offset",
                "requested",
                "completed",
                "duration_ns",
                "thread_id",
            )
        ) + (start,)

    raw_payload = {
        payload_key(event, event.get("origin_ns", origin_ns or 0) + event["start_ns"])
        for event in raw_rows
        if event.get("io_layer") in ("payload", "container_patch")
    }
    for source, events in (("buffered_file_api", raw_rows), ("payload_api", payload_events)):
        for event in events:
            if not isinstance(event, dict):
                raise TraceEvidenceError("Payload calls must be event objects")
            if source == "buffered_file_api" and event.get("io_layer") == "staging":
                continue
            if (
                source == "payload_api"
                and payload_key(event, event.get("absolute_start_ns", event.get("start_ns")))
                in raw_payload
            ):
                continue
            row = _request(event, rank=rank, checkpoint_root=checkpoint_root)
            if source == "payload_api":
                start = event.get("absolute_start_ns", event.get("start_ns"))
            else:
                start = _integer(event.get("origin_ns", origin_ns or 0), "event origin_ns")
                start += _integer(event.get("start_ns"), "start_ns")
            row.update(
                start_ns=_integer(start, "absolute start_ns"),
                duration_ns=_integer(event.get("duration_ns"), "duration_ns"),
                seq=len(rows),
                source="payload_api"
                if event.get("io_layer") in ("payload", "container_patch")
                else source,
            )
            rows.append(row)
    ordered, clocks = _clock_order(rows)
    first_ns = min((row["start_ns"] for row in ordered), default=0)
    for index, row in enumerate(ordered):
        relative_start = row["start_ns"] - first_ns
        row.update(
            seq=index,
            start_ns=relative_start,
            t_us=relative_start // 1000,
            size=0 if row["op"] == "seek" else row["completed"],
            cost_us=row["duration_ns"] // 1000,
        )
        if not row["request"]:
            row["end_ns"] = relative_start + row["duration_ns"]
    _, signature = _signatures(ordered)
    clients = Path(output_dir) / "clients"
    clients.mkdir(parents=True, exist_ok=True)
    output = clients / f"rank-{rank:05d}.jsonl"
    encoded = b"".join(_json_bytes(row) + b"\n" for row in ordered)
    output.write_bytes(encoded)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        rank=rank,
        phase=phase,
        layer="Python buffered-file API plus explicitly instrumented payload calls",
        reference_fields=list(REFERENCE_FIELDS),
        signature_fields=list(SIGNATURE_FIELDS),
        time_basis=(
            "microseconds since this rank's first measured event; exact nanoseconds retained"
        ),
        first_event_absolute_ns=first_ns,
        raw_source=raw_source,
        payload_events_count=len(payload_events),
        payload_events_sha256=hashlib.sha256(_json_bytes(payload_events)).hexdigest(),
        output_path=str(output.resolve()),
        output_sha256=hashlib.sha256(encoded).hexdigest(),
        event_count=len(ordered),
        request_event_count=sum(row["request"] for row in ordered),
        auxiliary_event_count=sum(not row["request"] for row in ordered),
        auxiliary_operations_excluded_from_signature=sorted(AUXILIARY_OPERATIONS),
        operations=dict(Counter(row["op"] for row in ordered)),
        successful=all(row["success"] == 1 for row in ordered),
        clock_order=clocks,
        pattern_sha256=signature,
        limitations=[
            "Operation names describe actual instrumented calls; no RPCs are synthesized.",
            "The buffered recorder records returned calls; thrown calls lack raw events.",
            "Payload call instrumentation may use a different API layer, identified by source.",
            "Measured lifecycle/setup/barrier intervals are auxiliary, excluded from request identity.",
            "No physical block requests, checkpoint payload hashes, or values are observed here.",
        ],
    )
    manifest_path = clients / f"rank-{rank:05d}.trace.json"
    manifest["manifest_path"] = str(manifest_path.resolve())
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def _load_export(path):
    rows, source = _read_jsonl(path)
    result, ranks = [], set()
    for index, event in enumerate(rows):
        rank = _integer(event.get("rank"), "rank")
        ranks.add(rank)
        if event.get("seq") != index or type(event.get("seq")) is not int:
            raise TraceEvidenceError("Export seq must match its contiguous JSONL position")
        row = _request(event, rank=rank, checkpoint_root=None)
        start = _integer(event.get("start_ns"), "start_ns")
        duration = _integer(event.get("duration_ns"), "duration_ns")
        if not row["request"] and event.get("end_ns") != start + duration:
            raise TraceEvidenceError("Lifecycle end_ns is inconsistent with its exact event")
        size = 0 if row["op"] == "seek" else row["completed"]
        for field, expected in (
            ("size", size),
            ("t_us", start // 1000),
            ("cost_us", duration // 1000),
        ):
            if type(event.get(field)) is not int or event[field] != expected:
                raise TraceEvidenceError(f"Export {field} is inconsistent with its exact event")
        row.update(start_ns=start, duration_ns=duration, seq=index)
        result.append(row)
    if len(ranks) > 1:
        raise TraceEvidenceError("One client trace contains multiple ranks")
    return result, source, next(iter(ranks), None)


def _first_difference(left, right):
    for index in range(max(len(left), len(right))):
        a = left[index] if index < len(left) else None
        b = right[index] if index < len(right) else None
        if a == b:
            continue
        field = "event"
        if a is not None and b is not None:
            position = next(
                i for i, pair in enumerate(zip(a, b, strict=True)) if pair[0] != pair[1]
            )
            field, a, b = SIGNATURE_FIELDS[position], a[position], b[position]
        return dict(index=index, field=field, left=a, right=b)
    return None


def prepare_rank_trace(path):
    """Validate one export once and retain its exact ordered request signatures.

    Shared-reference comparisons can reuse this compact representation without
    reparsing files or retaining their much larger per-event dictionaries.
    """
    rows, source, rank = _load_export(path)
    rows, clocks = _clock_order(rows)
    signatures, signature_hash = _signatures(rows)
    return dict(
        rank=rank,
        source=source,
        clock_order=clocks,
        signatures=signatures,
        signature_sha256=signature_hash,
        successful=all(row["success"] == 1 for row in rows if row["request"]),
        auxiliary_event_count=sum(not row["request"] for row in rows),
        auxiliary_successful=all(row["success"] == 1 for row in rows if not row["request"]),
    )


def compare_prepared_rank_traces(left, right):
    """Compare validated exports with the same evidence rules as file comparison."""
    left_rank, right_rank = left["rank"], right["rank"]
    if left_rank is not None and right_rank is not None and left_rank != right_rank:
        raise TraceEvidenceError("Cannot compare different rank identities")
    left_clock, right_clock = left["clock_order"], right["clock_order"]
    a, b = left["signatures"], right["signatures"]
    equal = a == b
    verifiable = (
        left_clock["serial_rank_order_verifiable"] and right_clock["serial_rank_order_verifiable"]
    )
    successful = left["successful"] and right["successful"]
    status = "unknown" if not verifiable else "identical" if equal else "different"
    if not successful:
        status = "different"
    return dict(
        schema_version=SCHEMA_VERSION,
        status=status,
        rank=left_rank if left_rank is not None else right_rank,
        signature_fields=list(SIGNATURE_FIELDS),
        signatures=dict(left=left["signature_sha256"], right=right["signature_sha256"]),
        event_counts=dict(left=len(a), right=len(b)),
        sequence_equal=equal,
        geometry_multiset_equal=equal or Counter(a) == Counter(b),
        successful=successful,
        auxiliary_event_counts=dict(
            left=left["auxiliary_event_count"],
            right=right["auxiliary_event_count"],
        ),
        auxiliary_successful=left["auxiliary_successful"] and right["auxiliary_successful"],
        serial_rank_order_verifiable=verifiable,
        clock_order=dict(left=left_clock, right=right_clock),
        first_difference=None if equal else _first_difference(a, b),
        sources=dict(left=left["source"], right=right["source"]),
        cross_rank_interleaving_compared=False,
        time_or_duration_equivalence_tested=False,
    )


def compare_rank_traces(left_path, right_path):
    """Compare exported client traces, excluding clocks and cross-rank interleaving.

    A tied or overlapping invocation clock cannot prove a serial order, even
    when stable sorting happens to produce equal signatures. Empty traces are
    unknown. Call failures always prevent an identical result.
    """
    return compare_prepared_rank_traces(
        prepare_rank_trace(left_path), prepare_rank_trace(right_path)
    )


def compare_trace_sets(traces_by_device, *, world_size):
    """Compare complete device/rank mappings such as {'meta': {0: path}, ...}.

    Missing or extra ranks, fewer than two devices, empty files, and unreadable
    evidence produce unknown. Pair details preserve exact first differences.
    No cross-rank scheduling order is compared.
    """
    _integer(world_size, "world_size", minimum=1)
    expected = set(range(world_size))
    completeness, errors, traces = {}, [], {}
    for device, mapping in traces_by_device.items():
        ranks = {}
        for rank, path in mapping.items():
            if type(rank) is not int or rank < 0:
                raise TraceEvidenceError("Device rank mapping keys must be nonnegative integers")
            ranks[rank] = path
        traces[device] = ranks
        completeness[device] = dict(
            present=sorted(ranks),
            missing=sorted(expected - ranks.keys()),
            extra=sorted(ranks.keys() - expected),
        )
        if set(ranks) != expected:
            errors.append(f"{device}: rank set differs from expected 0..{world_size - 1}")
    if len(traces) < 2:
        errors.append("At least two devices are required")
    comparisons = []
    for left, right in itertools.combinations(traces, 2):
        for rank in sorted(expected & traces[left].keys() & traces[right].keys()):
            try:
                result = compare_rank_traces(traces[left][rank], traces[right][rank])
                if result["rank"] is not None and result["rank"] != rank:
                    raise TraceEvidenceError("Event rank differs from device/rank mapping")
            except (OSError, ValueError) as error:
                errors.append(f"{left}/{right} rank {rank}: {error}")
                result = dict(status="unknown", error=str(error), rank=rank)
            comparisons.append(dict(left_device=left, right_device=right, **result))
    statuses = {row["status"] for row in comparisons}
    status = (
        "unknown"
        if errors or "unknown" in statuses or not comparisons
        else ("different" if "different" in statuses else "identical")
    )
    return dict(
        schema_version=SCHEMA_VERSION,
        status=status,
        world_size=world_size,
        devices=list(traces),
        rank_completeness=completeness,
        errors=errors,
        comparisons=comparisons,
        cross_rank_interleaving_compared=False,
        time_or_duration_equivalence_tested=False,
    )
