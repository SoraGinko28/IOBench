"""Analyze paired multi-rank Meta/CPU runs without opening checkpoint payloads.

All time gates use the median paired Meta-minus-CPU difference, divided by the
median CPU value. Rank-local checkpoint durations supply the wall-time gate;
aggregate timelines share the earliest checkpoint start across ranks. Save
timelines remove the *global union* of measured fsync intervals.
"""

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import resource
import statistics
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_gdtensor_traces as base  # noqa: E402
from metasl.trace_export import (  # noqa: E402
    OPERATIONS, _clock_order, _request, _signatures,
    compare_prepared_rank_traces, prepare_rank_trace,
)

PHASES = ("save", "cold-load", "warm-load")
CONFIGS = ("meta", "cpu")
QUANTILES = (10, 50, 90, 100)
BIN_NS = 50_000_000
EXPECTED_REPEATS = 3
FLOOR_S = .100
FIX1 = False


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def quantile_times(values, duration_ns):
    """Interpolate byte completion through 50 ms bins, clipping the last bin."""
    total = sum(values)
    if total <= 0:
        raise ValueError("A payload timeline must contain positive bytes")
    result = {}
    for quantile in QUANTILES:
        target, before = total * quantile / 100, 0.0
        for index, amount in enumerate(values):
            if amount > 0 and before + amount >= target - total * 1e-14:
                width = min(BIN_NS, max(0, duration_ns - index * BIN_NS))
                fraction = min(1.0, max(0.0, (target - before) / amount))
                result[str(quantile)] = (index * BIN_NS + fraction * width) / 1e9
                break
            before += amount
        else:
            raise ValueError("Cannot locate payload quantile")
    return result


def aggregate_timeline(cases):
    """Use a common host clock; preserve bytes when another rank is fsyncing.

    Each measured call's bytes are uniform over that call. A piece overlapping
    the global fsync union maps to an instant on the compressed axis. Recording
    that overlap makes this convention auditable rather than losing its bytes.
    """
    start = min(c["absolute_start_ns"] for c in cases)
    end = max(c["absolute_end_ns"] for c in cases)
    save = cases[0]["row"]["phase"] == "save"
    fsync = base.merged(
        (max(start, c["origin_ns"] + e["start_ns"]),
         min(end, c["origin_ns"] + e["start_ns"] + e["duration_ns"]))
        for c in cases for e in c["fsync"]
    ) if save else []

    def transform(t):
        return t - start - sum(max(0, min(b, t) - a) for a, b in fsync if a < t)

    duration = transform(end)
    values = [0.0] * max(1, math.ceil(duration / BIN_NS))
    overlap_ns, overlap_bytes = 0, 0.0

    def put(a, b, amount):
        if a == b:
            values[min(a // BIN_NS, len(values) - 1)] += amount
            return
        for index in range(a // BIN_NS, (b - 1) // BIN_NS + 1):
            overlap = min(b, (index + 1) * BIN_NS) - max(a, index * BIN_NS)
            values[index] += amount * overlap / (b - a)

    for case in cases:
        for event in case["payload"]:
            a = case["origin_ns"] + event["start_ns"]
            b = a + event["duration_ns"]
            if a < start or b > end:
                raise ValueError("Payload outside aggregate checkpoint interval")
            if a == b:
                put(transform(a), transform(b), event["completed"])
                continue
            boundaries = sorted({a, b} | {t for interval in fsync for t in interval if a < t < b})
            for left, right in zip(boundaries, boundaries[1:]):
                amount = event["completed"] * (right - left) / (b - a)
                compressed_left, compressed_right = transform(left), transform(right)
                if compressed_left == compressed_right:
                    overlap_ns += right - left
                    overlap_bytes += amount
                put(compressed_left, compressed_right, amount)
    expected = sum(c["metrics"]["payload_bytes"] for c in cases)
    if not math.isclose(sum(values), expected, abs_tol=.01, rel_tol=1e-12):
        raise ValueError("Aggregate bins do not conserve payload bytes")
    return dict(
        origin_absolute_ns=start, end_absolute_ns=end, duration_s=duration / 1e9,
        uncompressed_span_s=(end - start) / 1e9,
        global_fsync_union_s=sum(b - a for a, b in fsync) / 1e9,
        global_fsync_intervals_ns=[[a - start, b - start] for a, b in fsync],
        bin_width_s=BIN_NS / 1e9, binned_bytes_per_50ms=values,
        payload_bytes=expected, binned_bytes=sum(values),
        payload_fsync_overlap_s=overlap_ns / 1e9,
        payload_fsync_overlap_bytes=overlap_bytes,
        quantiles_s=quantile_times(values, duration),
        first_payload_s=min(c["origin_ns"] + e["start_ns"] for c in cases for e in c["payload"]) / 1e9 - start / 1e9,
        last_payload_s=max(c["origin_ns"] + e["start_ns"] + e["duration_ns"] for c in cases for e in c["payload"]) / 1e9 - start / 1e9,
        first_payload_compressed_s=transform(min(c["origin_ns"] + e["start_ns"] for c in cases for e in c["payload"])) / 1e9,
        last_payload_compressed_s=transform(max(c["origin_ns"] + e["start_ns"] + e["duration_ns"] for c in cases for e in c["payload"])) / 1e9,
    )


def reuse_runner_case(row, output, report, events):
    """Verify retained runner exports, then derive only the payload scope.

    The runner already exported exact full requests and their signature. Hash
    both retained files before reusing that evidence; do not reserialize another
    complete trace. Missing exports use the original compatibility inspector.
    """
    row = base.metadata(Path(row["output"]), report, row)
    rank = row["rank"]
    manifest_path = Path(row["output"]) / "export" / "clients" / f"rank-{rank:05d}.trace.json"
    if not manifest_path.is_file():
        return None
    payload = sorted((e for e in events if e.get("io_layer") == "payload" and e["op"] in ({"write", "pwrite"} if row["phase"] == "save" else {"read", "readinto", "pread"})), key=lambda e: e["start_ns"])
    if not payload:
        return None
    manifest = base.read_json(manifest_path)
    report_path = Path(row["report"])
    raw_path = report_path.with_suffix(".io.jsonl")
    full_export = manifest_path.with_name(f"rank-{rank:05d}.jsonl")
    if manifest.get("rank") != rank:
        raise ValueError(f"Runner export rank mismatch: {manifest_path}")
    raw_hash, export_hash = base.sha256(raw_path), base.sha256(full_export)
    if manifest.get("raw_source", {}).get("sha256") != raw_hash:
        raise ValueError(f"Runner export raw-source SHA256 mismatch: {raw_path}")
    if manifest.get("output_sha256") != export_hash:
        raise ValueError(f"Runner export SHA256 mismatch: {full_export}")
    origin, stats = report["io"]["origin_ns"], report["stats"]
    start, end = stats["dcp_started_ns"] - origin, stats["dcp_ended_ns"] - origin
    total = stats["dcp_wall_seconds"]
    if not start < end or abs((end - start) / 1e9 - total) > .001:
        raise ValueError(f"Inconsistent checkpoint bounds: {report_path}")
    d2h = [e for e in events if e.get("io_layer") == "staging" and e["op"] == "d2h"]
    h2d = [e for e in events if e.get("io_layer") == "staging" and e["op"] == "h2d"]
    fsync = [e for e in events if e["op"] == "fsync"]
    if any(not e.get("success", True) or e["requested"] != e["completed"] for e in payload + d2h + h2d):
        raise ValueError(f"Incomplete payload/transfer events: {report_path}")
    if any(e["start_ns"] < start or e["start_ns"] + e["duration_ns"] > end for e in payload + d2h + h2d):
        raise ValueError(f"Payload/transfer outside checkpoint bounds: {report_path}")
    payload_bytes = sum(e["completed"] for e in payload)
    expected = report.get(f"gdtensor_payload_{'write' if row['phase'] == 'save' else 'read'}_bytes")
    if expected is not None and expected != payload_bytes:
        raise ValueError(f"Payload bytes disagree with report: {report_path}")
    requests = [e for e in events if e.get("io_layer") != "staging" and e["op"] in OPERATIONS]
    if len(requests) != manifest.get("request_event_count"):
        raise ValueError(f"Runner export request count disagrees with raw trace: {report_path}")
    rows = []
    first_payload_ns = origin + payload[0]["start_ns"]
    for event in payload:
        record = _request(event, rank=rank, checkpoint_root=None)
        relative = origin + event["start_ns"] - first_payload_ns
        record.update(start_ns=relative, duration_ns=event["duration_ns"], seq=len(rows),
                      t_us=relative // 1000, size=event["completed"],
                      cost_us=event["duration_ns"] // 1000, source="payload_api")
        rows.append(record)
    rows, clocks = _clock_order(rows)
    for seq, event in enumerate(rows):
        event["seq"] = seq
    _, signature = _signatures(rows)
    case_id = hashlib.sha256(str(Path(row["output"]).resolve()).encode()).hexdigest()[:16]
    payload_export = Path(output) / "exports" / case_id / f"rank-{rank:05d}.payload.jsonl"
    payload_export.parent.mkdir(parents=True, exist_ok=True)
    encoded = b"".join(json.dumps(e, sort_keys=True, separators=(",", ":"), allow_nan=False).encode() + b"\n" for e in rows)
    payload_export.write_bytes(encoded)
    payload_manifest = dict(rank=rank, pattern_sha256=signature, request_event_count=len(rows),
                            successful=True, clock_order=clocks, output_sha256=hashlib.sha256(encoded).hexdigest(),
                            raw_source=manifest["raw_source"], output_path=str(payload_export))
    base.dump(payload_export.with_suffix(".trace.json"), payload_manifest)
    measure = lambda selected: base.union_ns(base.clipped(selected, start, end)) / 1e9
    metrics = dict(total_s=total, total_without_fsync_s=total - measure(fsync), fsync_s=measure(fsync),
                   d2h_s=measure(d2h), h2d_s=measure(h2d), payload_io_s=measure(payload),
                   pwrite_s=measure([e for e in payload if e["op"] == "pwrite"]),
                   pread_s=measure([e for e in payload if e["op"] in {"read", "pread", "readinto"}]),
                   payload_requests=len(payload), payload_bytes=payload_bytes,
                   request_count=len(requests), request_bytes=sum(e["completed"] for e in requests if e["op"] != "seek"),
                   request_operations={op: manifest["operations"].get(op, 0) for op in OPERATIONS if manifest["operations"].get(op)},
                   d2h_count=len(d2h), h2d_count=len(h2d))
    return dict(row=row, metrics=metrics, payload=payload, fsync=fsync, start_ns=start, end_ns=end,
                export=str(full_export), payload_export=str(payload_export),
                full_manifest=manifest, payload_manifest=payload_manifest,
                provenance=dict(report_sha256=base.sha256(report_path), trace_sha256=raw_hash,
                                export_sha256=export_hash, payload_export_sha256=payload_manifest["output_sha256"],
                                full_export_reused=True, raw_and_export_hashes_verified=True))


def inspect(row, output):
    row = dict(row)
    row.setdefault("report", str(Path(row["output"]) / f"rank-{row.get('rank', 0):03d}.json"))
    report = base.read_json(row["report"])
    origin = report["io"]["origin_ns"]
    events = [dict(e, start_ns=e["start_ns"] + e.get("origin_ns", origin) - origin)
              for e in base.read_jsonl(Path(row["report"]).with_suffix(".io.jsonl"))]
    case = reuse_runner_case(row, output, report, events)
    if case is None:
        case = base.inspect_case(row, output)
        case["full_manifest"] = case["payload_manifest"] = None
    start, end = case["start_ns"], case["end_ns"]

    def duration(selected):
        return base.union_ns(base.clipped(selected, start, end)) / 1e9

    staging = [e for e in events if e.get("io_layer") == "staging"]
    sink = [e for e in staging if e["op"] == "host_sink"]
    nonpayload_io = [e for e in events if e.get("io_layer") not in {"payload", "staging"}
                     and e["op"] in {"read", "readinto", "readline", "pread", "write", "pwrite"}]
    payload, fsync = case["payload"], case["fsync"]
    covered = duration(payload + fsync + sink)
    metrics = case["metrics"]
    metrics.update(
        read_s=metrics["pread_s"], sink_s=duration(sink), sink_observed_s=duration(sink),
        other_s=max(0.0, metrics["total_s"] - covered),
        host_copy_s=duration([e for e in events if e["op"] == "host_copy"]),
        host_materialize_s=duration([e for e in events if e["op"] == "host_materialize"]),
        staging_union_s=duration(staging),
        non_sink_staging_union_s=duration([e for e in staging if e["op"] != "host_sink"]),
        nonpayload_io_s=duration(nonpayload_io),
        other_staging_s=max(0.0, duration(payload + fsync + staging) - covered),
        other_without_staging_s=max(0.0, metrics["total_s"] - duration(payload + fsync + staging)),
        uninstrumented_s=max(0.0, metrics["total_s"] - duration(payload + fsync + staging + nonpayload_io)),
        pre_payload_s=(min(e["start_ns"] for e in payload) - start) / 1e9,
        post_payload_s=(end - max(e["start_ns"] + e["duration_ns"] for e in payload)) / 1e9,
        category_union_s=covered,
    )
    sink_bytes = sum(e.get("completed", 0) for e in sink)
    sink_expected_bytes = report.get("target_payload_bytes", metrics["payload_bytes"]) if row["phase"] != "save" else 0
    sink_complete = row["phase"] == "save" or sink_bytes == sink_expected_bytes
    metrics.update(sink_observation_complete=sink_complete, sink_observed_bytes=sink_bytes,
                   sink_expected_bytes=sink_expected_bytes)
    if not sink_complete:
        metrics["sink_s"] = None
    case.update(origin_ns=origin, absolute_start_ns=origin + start,
                absolute_end_ns=origin + end, stats=report["stats"],
                writer_limit=report.get("writer_limit"), writer_timing=report.get("writer_timing", {}),
                reader_limit=report.get("reader_limit"), reader_timing=report.get("reader_timing", {}),
                staging_operations={op: dict(union_s=duration([e for e in staging if e["op"] == op]),
                                             outside_primary_s=max(0.0, duration(payload + fsync + sink + [e for e in staging if e["op"] == op]) - covered),
                                             event_count=sum(e["op"] == op for e in staging),
                                             completed_bytes=sum(e.get("completed", 0) for e in staging if e["op"] == op))
                                    for op in sorted({e["op"] for e in staging})},
                sink_configuration={k: v for k, v in report.items() if "sink" in k},
                source_configuration={k: v for k, v in report.items() if "source" in k},
                cache_before_call=report.get("cache_before_call", {}),
                admission_preparation=report.get("admission_preparation", {}))
    timing = case["writer_timing"]
    if timing.get("queue_enter_ns") and timing.get("write_enter_ns") and timing.get("future_completed_ns"):
        metrics.update(
            before_writer_queue_s=(timing["queue_enter_ns"] - case["absolute_start_ns"]) / 1e9,
            writer_admission_wait_s=(timing["write_enter_ns"] - timing["queue_enter_ns"]) / 1e9,
            writer_active_s=(timing["future_completed_ns"] - timing["write_enter_ns"]) / 1e9,
            after_writer_completed_s=(case["absolute_end_ns"] - timing["future_completed_ns"]) / 1e9,
        )
    reader = case["reader_timing"]
    if all(reader.get(key) is not None for key in ("queue_enter_ns", "reader_enter_ns", "reader_return_ns")):
        reader_start, reader_end = reader["reader_enter_ns"] - origin, reader["reader_return_ns"] - origin
        metrics.update(
            before_reader_queue_s=(reader["queue_enter_ns"] - case["absolute_start_ns"]) / 1e9,
            reader_admission_wait_s=(reader["reader_enter_ns"] - reader["queue_enter_ns"]) / 1e9,
            reader_active_s=(reader_end - reader_start) / 1e9,
            after_reader_return_s=(case["absolute_end_ns"] - reader["reader_return_ns"]) / 1e9,
            reader_active_uncovered_s=max(0.0, (reader_end - reader_start - base.union_ns(base.clipped(payload + staging + nonpayload_io, reader_start, reader_end))) / 1e9),
        )
    metadata = reader.get("metadata") or report.get("metadata") or {}
    if isinstance(metadata, dict):
        for metric, left, right in (("metadata_gate_wait_s", "queue_enter", "read_start"),
                                    ("metadata_read_s", "read_start", "read_end"),
                                    ("metadata_prune_s", "read_end", "prune_end")):
            if metadata.get(left) is not None and metadata.get(right) is not None:
                metrics[metric] = metadata[right] - metadata[left]
        if metadata.get("queue_enter") is not None:
            metrics["before_metadata_queue_s"] = metadata["queue_enter"] - case["absolute_start_ns"] / 1e9
        metadata_finished = metadata.get("prune_end", metadata.get("read_end"))
        if metadata_finished is not None and reader.get("queue_enter_ns") is not None:
            metrics["after_metadata_to_reader_queue_s"] = reader["queue_enter_ns"] / 1e9 - metadata_finished
    # Keep the requested breakdown as both raw unions and an exact, disjoint
    # wall partition. In particular, pre-payload includes admission and early
    # materialization, so adding the raw columns would double-count time.
    def interval(left, right):
        return [] if left is None or right is None else [(max(start, left - origin), min(end, right - origin))]

    admission = interval(timing.get("queue_enter_ns"), timing.get("write_enter_ns")) if row["phase"] == "save" else interval(reader.get("queue_enter_ns"), reader.get("reader_enter_ns"))
    if row["phase"] != "save" and isinstance(metadata, dict) and metadata.get("queue_enter") is not None and metadata.get("read_start") is not None:
        admission += interval(round(metadata["queue_enter"] * 1e9), round(metadata["read_start"] * 1e9))
    materialize = [e for e in staging if "materialize" in e["op"]]
    post_start = timing.get("future_completed_ns") if row["phase"] == "save" else reader.get("reader_return_ns")
    categories = dict(
        admission_wait=admission,
        payload_io=base.clipped(payload, start, end),
        materialize=base.clipped(materialize, start, end),
        sink=base.clipped(sink, start, end),
        fsync=base.clipped(fsync, start, end),
        post_read_wait=interval(post_start, origin + end),
        pre_payload=[(start, min(e["start_ns"] for e in payload))],
        other_staging_or_io=base.clipped(staging + nonpayload_io, start, end),
    )
    used, partition = [], {}
    for name, spans in categories.items():
        spans = [(a, b) for a, b in spans if a < b]
        before = base.union_ns(used)
        used = base.merged(used + spans)
        partition[name] = (base.union_ns(used) - before) / 1e9
    partition["uncovered"] = max(0.0, metrics["total_s"] - base.union_ns(used) / 1e9)
    case["critical_breakdown"] = dict(
        raw_union_s=dict(pre_payload=metrics["pre_payload_s"], admission_wait=base.union_ns(admission) / 1e9,
                         payload_io=metrics["payload_io_s"], materialize=duration(materialize),
                         sink=metrics["sink_s"], post_read_wait=base.union_ns(categories["post_read_wait"]) / 1e9,
                         uncovered=metrics["uninstrumented_s"], fsync=metrics["fsync_s"]),
        disjoint_partition_s=partition, partition_sum_s=sum(partition.values()),
        partition_matches_wall=abs(sum(partition.values()) - metrics["total_s"]) < 1e-8,
        priority=list(categories), raw_columns_overlap=True,
        sink_observation_complete=sink_complete)
    return case


def paired_summary(meta, cpu, *, tiny_tolerance_s=0.0, expected_repeats=None):
    """No ratio-of-independent-medians: preserve the requested paired delta."""
    repeats = sorted(set(meta) & set(cpu))
    samples = [dict(repeat=r, meta_s=meta[r], cpu_s=cpu[r],
                    delta_s=meta[r] - cpu[r],
                    delta_percent=(meta[r] - cpu[r]) / cpu[r] * 100 if cpu[r] else None)
               for r in repeats]
    cpu_median = median(cpu[r] for r in repeats)
    delta = median(s["delta_s"] for s in samples)
    expected_repeats = EXPECTED_REPEATS if expected_repeats is None else expected_repeats
    complete = repeats == list(range(1, expected_repeats + 1)) and set(meta) == set(cpu)
    tolerance = max(.03 * cpu_median, tiny_tolerance_s) if cpu_median is not None else None
    strict_tolerance = .03 * cpu_median if cpu_median is not None else None
    floor_tolerance = max(strict_tolerance, FLOOR_S) if strict_tolerance is not None else None
    return dict(
        meta_median_s=median(meta[r] for r in repeats), cpu_median_s=cpu_median,
        paired_median_delta_s=delta,
        paired_median_delta_ms=delta * 1000 if delta is not None else None,
        paired_median_absolute_delta_ms=median(abs(s["delta_s"]) * 1000 for s in samples),
        paired_median_delta_percent=delta / cpu_median * 100 if cpu_median else None,
        median_of_paired_percent=median(s["delta_percent"] for s in samples if s["delta_percent"] is not None),
        tolerance_s=tolerance, complete=complete, expected_repeats=expected_repeats,
        strict_tolerance_s=strict_tolerance, floor_tolerance_s=floor_tolerance,
        strict_status="unknown" if not complete else "pass" if abs(delta) <= strict_tolerance else "fail",
        floor_status="unknown" if not complete else "pass" if abs(delta) <= floor_tolerance else "fail",
        status="unknown" if not complete else "pass" if abs(delta) <= tolerance else "fail",
        samples=samples,
    )


def sequence_pair(meta, cpu, ranks):
    result = {}
    for scope, field in (("full", "export"), ("payload", "payload_export")):
        comparisons, errors = [], []
        for rank in range(ranks):
            if rank not in meta or rank not in cpu:
                errors.append(f"rank {rank}: missing Meta or CPU report")
                continue
            try:
                manifest_key = "full_manifest" if scope == "full" else "payload_manifest"
                manifests = [config[rank].get(manifest_key) for config in (meta, cpu)]
                verified_identity = all(
                    manifest and manifest.get("rank") == rank and manifest.get("successful")
                    and manifest.get("clock_order", {}).get("serial_rank_order_verifiable")
                    and isinstance(manifest.get("pattern_sha256"), str)
                    and len(manifest["pattern_sha256"]) == 64
                    and all(ch in "0123456789abcdef" for ch in manifest["pattern_sha256"])
                    and type(manifest.get("request_event_count")) is int
                    and manifest["request_event_count"] > 0
                    for manifest in manifests)
                if verified_identity and manifests[0].get("pattern_sha256") == manifests[1].get("pattern_sha256") and manifests[0].get("request_event_count") == manifests[1].get("request_event_count"):
                    comparison = dict(rank=rank, status="identical", request_count=manifests[0]["request_event_count"],
                                      pattern_sha256=manifests[0]["pattern_sha256"],
                                      verification="Exact request-signature SHA256; raw/export hashes verified; serial rank clock order verified")
                else:
                    left = prepare_rank_trace(meta[rank][field])
                    right = prepare_rank_trace(cpu[rank][field])
                    comparison = compare_prepared_rank_traces(left, right)
            except (ValueError, OSError) as error:
                errors.append(f"rank {rank}: {error}")
                comparison = dict(rank=rank, status="unknown", error=str(error))
            comparisons.append(comparison)
        status = "unknown" if errors else "different" if any(c["status"] == "different" for c in comparisons) else "unknown" if any(c["status"] != "identical" for c in comparisons) else "identical"
        result[scope] = dict(status=status, comparisons=comparisons, errors=errors)
    return result


def phase_diagnosis(phase, aggregates, cases):
    """Separate measured IO, staging and admission/metadata waits on paired runs.

    Critical ranks may change between configs or repeats. Keep those identities
    and compute component deltas *within* the same paired repeat. Rank-summed
    staging times describe process work, not an additive wall-time partition.
    """
    per_config, critical_by_config = {}, {}
    operations = sorted({op for c in cases for op in c.get("staging_operations", {})})
    metric_names = (
        "total_s", "pwrite_s", "read_s", "fsync_s", "sink_s", "other_s", "host_materialize_s", "host_copy_s",
        "pre_payload_s", "post_payload_s", "nonpayload_io_s", "staging_union_s",
        "non_sink_staging_union_s", "other_staging_s", "other_without_staging_s", "uninstrumented_s",
        "before_writer_queue_s", "writer_admission_wait_s", "writer_active_s", "after_writer_completed_s",
        "before_reader_queue_s", "reader_admission_wait_s", "reader_active_s", "after_reader_return_s",
        "reader_active_uncovered_s", "before_metadata_queue_s", "metadata_gate_wait_s",
        "metadata_read_s", "metadata_prune_s", "after_metadata_to_reader_queue_s",
    )
    for config in CONFIGS:
        selected = [a for a in aggregates if a["config"] == config]
        critical_by_config[config] = {
            a["repeat"]: max(
                (c for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == a["repeat"]),
                key=lambda c: c["metrics"]["total_s"],
            ) for a in selected
        }
        critical = list(critical_by_config[config].values())
        per_config[config] = dict(
            critical_rank_by_repeat={r: c["row"]["rank"] for r, c in critical_by_config[config].items()},
            critical_rank_medians={key: median(c["metrics"].get(key) for c in critical if c["metrics"].get(key) is not None) for key in metric_names},
            critical_rank_staging_medians_s={op: median(c.get("staging_operations", {}).get(op, {}).get("union_s", 0) for c in critical
                                                        if op != "host_sink" or c["metrics"].get("sink_observation_complete", True)) for op in operations},
            staging_times_are_observed_lower_bounds=any(not c["metrics"].get("sink_observation_complete", True) for c in critical),
            rank_summed_medians={key: median(
                sum(c["metrics"].get(key, 0) for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == a["repeat"])
                for a in selected if all(c["metrics"].get(key) is not None for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == a["repeat"]))
                for key in ("pwrite_s", "read_s", "sink_s", "fsync_s", "other_s")},
            resources={key: median(c["stats"].get("resource_delta", {}).get(key, 0) for c in critical) for key in ("ru_utime", "ru_stime", "ru_minflt", "ru_majflt", "ru_nvcsw", "ru_nivcsw")},
            process_io={key: median(c["stats"].get("proc_io_delta", {}).get(key, 0) for c in critical) for key in ("read_bytes", "write_bytes", "syscr", "syscw")},
            onset_median_s=median(a["timeline"]["first_payload_compressed_s"] for a in selected),
            active_payload_p10_p90_s=median(a["timeline"]["quantiles_s"]["90"] - a["timeline"]["quantiles_s"]["10"] for a in selected),
        )
    paired_components = {
        key: paired_summary(
            {r: c["metrics"][key] for r, c in critical_by_config["meta"].items() if c["metrics"].get(key) is not None},
            {r: c["metrics"][key] for r, c in critical_by_config["cpu"].items() if c["metrics"].get(key) is not None})
        for key in metric_names
    }
    def stage_samples(config, op, field):
        return {r: c.get("staging_operations", {}).get(op, {}).get(field, 0)
                for r, c in critical_by_config[config].items()
                if (op != "host_sink" or c["metrics"].get("sink_observation_complete", True))
                and c.get("staging_operations", {}).get(op, {}).get(field, 0) is not None}

    paired_staging = {
        op: paired_summary(
            stage_samples("meta", op, "union_s"), stage_samples("cpu", op, "union_s"))
        for op in operations
    }
    paired_staging_outside_primary = {
        op: paired_summary(
            stage_samples("meta", op, "outside_primary_s"), stage_samples("cpu", op, "outside_primary_s"))
        for op in operations
    }

    def timeline_pair(get_value, tiny_tolerance_s=0.0):
        return paired_summary(
            {a["repeat"]: get_value(a["timeline"]) for a in aggregates if a["config"] == "meta"},
            {a["repeat"]: get_value(a["timeline"]) for a in aggregates if a["config"] == "cpu"},
            tiny_tolerance_s=tiny_tolerance_s)

    timeline = dict(
        onset=timeline_pair(lambda t: t["first_payload_compressed_s"]),
        active_p10_p90=timeline_pair(lambda t: t["quantiles_s"]["90"] - t["quantiles_s"]["10"]),
        phase_span=timeline_pair(lambda t: t["duration_s"]),
        global_fsync_union=timeline_pair(lambda t: t["global_fsync_union_s"]),
        quantiles={str(q): timeline_pair(lambda t, q=q: t["quantiles_s"][str(q)], 0 if FIX1 else .05) for q in QUANTILES},
    )
    changes = {k: per_config["meta"]["critical_rank_medians"][k] - per_config["cpu"]["critical_rank_medians"][k]
               for k in per_config["meta"]["critical_rank_medians"]
               if per_config["meta"]["critical_rank_medians"][k] is not None and per_config["cpu"]["critical_rank_medians"][k] is not None}
    paired_changes = {k: value["paired_median_delta_s"] for k, value in paired_components.items() if value["paired_median_delta_s"] is not None}
    components = {k: paired_changes[k] for k in ("pwrite_s", "read_s", "fsync_s", "sink_s", "other_s")
                  if k in paired_changes and (k != "sink_s" or paired_components[k]["complete"])}
    largest = max(components, key=lambda k: abs(components[k])) if components else None
    facts, inferences = [], []
    missing_sink = [c for c in cases if not c["metrics"].get("sink_observation_complete", True)]
    if phase != "save" and missing_sink:
        facts.append(f"{len(missing_sink)}/{len(cases)} 个 rank case 的 host_sink 字节覆盖不足；sink_s 记为 unknown，observed sink 下界和未覆盖字节保留。other/reader_active_uncovered 含这些未观测的真实 sink 执行。")
        inferences.append("现有多 rank trace 的 compute staging 以最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，导致部分 rank 的 host_sink span 丢失（DCP/safetensors 均受影响）；因此不能把该部分 other 唯一归因于 Python/gdtensor 调度。请求、payload bytes、wall 和 shape 门限不受这一分项遥测缺口影响。")
    if largest:
        facts.append(f"每次各配置最慢 rank 的配对分项差最大位于 {largest}（{components[largest]:+.4f} s）；关键 rank 的身份保留在 analysis.json。")
    if phase == "save":
        fsync = timeline["global_fsync_union"]
        compressed = timeline["phase_span"]
        facts.append(f"全 rank fsync 并集 Meta/CPU {fsync['meta_median_s']:.3f}/{fsync['cpu_median_s']:.3f} s，配对差 {fsync['paired_median_delta_s']:+.4f} s；去 fsync 全局时间差 {compressed['paired_median_delta_s']:+.4f} s。")
        if largest == "fsync_s" or abs(fsync["paired_median_delta_s"]) >= abs(paired_changes.get("total_s", 0)) * .5:
            inferences.append("含 fsync 的 wall 差与实测同步刷盘区间变化相符；单 writer 许可覆盖 fsync，因此该时间也传播为其他 rank 的准入/结束等待。")
        waits = {key: paired_changes[key] for key in ("before_writer_queue_s", "writer_admission_wait_s", "after_writer_completed_s") if key in paired_changes}
        if waits:
            wait = max(waits, key=lambda key: abs(waits[key]))
            facts.append(f"writer 等待分解中最大配对变化为 {wait} {waits[wait]:+.4f} s。")
            if largest == "other_s":
                inferences.append({
                    "before_writer_queue_s": "变化被定位到 writer 队列之前的 planning/metadata/collective 路径；不应把本地 other 当作未记录的 pwrite。",
                    "writer_admission_wait_s": "变化被定位到单 writer 的实测准入等待；不应把本地 other 当作未记录的 pwrite。",
                    "after_writer_completed_s": "变化被定位到本 rank 写入完成后的 collective/metadata 发布路径；不应把本地 other 当作未记录的 pwrite。",
                }[wait])
    elif largest == "sink_s":
        inferences.append("偏差直接落在 host_sink 的目标写入/复制区间；这反映 CPU-target 持久缓冲区与 CPU 本地 shard 的实际目标写入成本差。")
    elif largest == "read_s":
        inferences.append("偏差直接落在实际 payload read 调用区间；这些是文件 API 耗时，包含内核/缓存等待，不能单凭它推断块设备的服务时间。")
    elif largest == "other_s":
        waits = {key: paired_changes[key] for key in (
            "other_staging_s", "metadata_gate_wait_s", "metadata_read_s", "metadata_prune_s", "after_metadata_to_reader_queue_s",
            "reader_admission_wait_s", "reader_active_uncovered_s", "after_reader_return_s") if key in paired_changes}
        if waits:
            wait = max(waits, key=lambda key: abs(waits[key]))
            facts.append(f"reader/metadata 分解中最大配对变化为 {wait} {waits[wait]:+.4f} s。")
            interpretation = {
                "other_staging_s": "偏差发生在已测 staging 操作中、pwrite/read/fsync/sink 主分项之外的区间；各实际 staging op 及其排除主分项后的并集已单独保留，不能把嵌套 spans 重复相加。",
                "metadata_gate_wait_s": "偏差发生在 metadata gate 的获取等待区间；该区间先于 payload 读取。",
                "metadata_read_s": "偏差发生在 metadata 的读取/反序列化/格式校验区间；它不是 tensor payload 的 read 耗时。",
                "metadata_prune_s": "偏差发生在 metadata 剪枝、重新序列化与重建区间。",
                "after_metadata_to_reader_queue_s": "偏差发生在 metadata 返回后、read_data 准入前的 planning/collective 路径。",
                "reader_admission_wait_s": "偏差发生在 reader 许可获取区间，是实测准入等待。",
                "reader_active_uncovered_s": "偏差发生在 read_data 内、已测 IO/staging spans 之外；sink 完整时该区间仍含 accessor/Python/gdtensor 调度与 callback join/control，现有 trace 无法唯一细分。" if not missing_sink else "偏差发生在 read_data 内、已测 IO/staging spans 之外；该区间可能包含未观测的 sink 与 accessor/Python/gdtensor 调度，现有 trace 无法再唯一细分。",
                "after_reader_return_s": "偏差发生在本 rank read_data 返回后的全局完成路径。",
            }
            inferences.append(interpretation[wait])
    observed_staging = {op: value for op, value in paired_staging.items() if value["paired_median_delta_s"] is not None and value["complete"]}
    if observed_staging:
        staging_op = max(observed_staging, key=lambda op: abs(observed_staging[op]["paired_median_delta_s"]))
        value = paired_staging[staging_op]
        facts.append(f"关键 rank staging 最大变化 {staging_op}：Meta/CPU {value['meta_median_s']:.4f}/{value['cpu_median_s']:.4f} s，配对差 {value['paired_median_delta_s']:+.4f} s。")
        outside_available = {op: value for op, value in paired_staging_outside_primary.items() if value["paired_median_delta_s"] is not None}
        if outside_available:
            outside_op = max(outside_available, key=lambda op: abs(outside_available[op]["paired_median_delta_s"]))
            outside = outside_available[outside_op]["paired_median_delta_s"]
            if abs(outside) > .001:
                facts.append(f"排除主分项重叠后，staging 的最大独立区间变化为 {outside_op} {outside:+.4f} s。")
    q_deltas = {q: value["paired_median_delta_s"] for q, value in timeline["quantiles"].items()}
    onset, active = timeline["onset"]["paired_median_delta_s"], timeline["active_p10_p90"]["paired_median_delta_s"]
    facts.append("shape P10/P50/P90/P100 配对差 " + "/".join(f"{q_deltas[str(q)] * 1000:+.1f}" for q in QUANTILES) + f" ms；首 payload {onset * 1000:+.1f} ms，P10–P90 跨度 {active * 1000:+.1f} ms。")
    if abs(onset) >= .5 * max(abs(value) for value in q_deltas.values()) and abs(onset) > abs(active):
        inferences.append("shape 偏移主要在首个 payload 前已积累；下方按实际 reader/writer 进入时间继续区分前置 planning/准入与进入后的首 IO 准备。")
    elif abs(active) > .05:
        inferences.append("shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。")
    wall_samples = paired_components["total_s"]["samples"]
    median_pair = None
    reader_progress, onset_breakdown = [], []
    for config in CONFIGS:
        for aggregate in [a for a in aggregates if a["config"] == config]:
            selected = [c for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == aggregate["repeat"]]
            if not selected or any(c.get("absolute_start_ns") is None or c["metrics"].get("pre_payload_s") is None for c in selected):
                continue
            first_case = min(selected, key=lambda c: c["absolute_start_ns"] + c["metrics"]["pre_payload_s"] * 1e9)
            first_ns = first_case["absolute_start_ns"] + first_case["metrics"]["pre_payload_s"] * 1e9
            timing_key, enter_key = ("writer_timing", "write_enter_ns") if phase == "save" else ("reader_timing", "reader_enter_ns")
            enter = first_case.get(timing_key, {}).get(enter_key)
            if enter is not None:
                onset_breakdown.append(dict(config=config, repeat=aggregate["repeat"], first_payload_rank=first_case["row"]["rank"],
                                            phase_to_enter_s=(enter - min(c["absolute_start_ns"] for c in selected)) / 1e9,
                                            enter_to_first_payload_s=(first_ns - enter) / 1e9))
    if onset_breakdown:
        onset_pairs = {key: paired_summary(
            {p["repeat"]: p[key] for p in onset_breakdown if p["config"] == "meta"},
            {p["repeat"]: p[key] for p in onset_breakdown if p["config"] == "cpu"}) for key in ("phase_to_enter_s", "enter_to_first_payload_s")}
        before_enter = onset_pairs["phase_to_enter_s"]["paired_median_delta_s"]
        after_enter = onset_pairs["enter_to_first_payload_s"]["paired_median_delta_s"]
        if before_enter is not None and after_enter is not None:
            facts.append(f"最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 {before_enter * 1000:+.1f} ms；进入→首 payload {after_enter * 1000:+.1f} ms。")
            if abs(onset) >= .5 * max(abs(value) for value in q_deltas.values()) and abs(onset) > abs(active):
                if abs(after_enter) > abs(before_enter):
                    inferences.append("首 payload 前的变化主要发生在 reader/writer 已进入之后的准备/非 payload 路径，不能归因于尚未获取的 admission 许可或全盘数据读取。")
                else:
                    inferences.append("首 payload 前的变化主要发生在 reader/writer 进入之前，与 metadata/planning/准入区间相符。")
    if wall_samples:
        median_pair_sample = min(wall_samples, key=lambda s: (abs(s["delta_s"] - paired_components["total_s"]["paired_median_delta_s"]), s["repeat"]))
        repeat = median_pair_sample["repeat"]
        paired_cases = {config: critical_by_config[config][repeat] for config in CONFIGS}
        deltas = {key: paired_cases["meta"]["metrics"][key] - paired_cases["cpu"]["metrics"][key]
                  for key in metric_names if all(c["metrics"].get(key) is not None for c in paired_cases.values())}
        median_pair = dict(repeat=repeat, meta_rank=paired_cases["meta"]["row"]["rank"],
                           cpu_rank=paired_cases["cpu"]["row"]["rank"], delta_s=median_pair_sample["delta_s"],
                           component_delta_s=deltas)
        if phase != "save" and all(c.get("reader_timing", {}).get("reader_enter_ns") is not None for c in paired_cases.values()):
            before = {config: (c["reader_timing"]["reader_enter_ns"] - c["absolute_start_ns"]) / 1e9 for config, c in paired_cases.items()}
            before_delta = before["meta"] - before["cpu"]
            active_delta, after_delta = deltas["reader_active_s"], deltas["after_reader_return_s"]
            median_pair["exact_before_reader_delta_s"] = before_delta
            facts.append(f"决定 wall 配对中位差的 r{repeat}：进入 reader 前 {before_delta:+.4f} s、reader 活跃 {active_delta:+.4f} s、reader 返回后 {after_delta:+.4f} s；这三个同一配对区间相加为 {median_pair_sample['delta_s']:+.4f} s。")
            main_interval = max({"before": before_delta, "active": active_delta, "after": after_delta}, key=lambda key: abs({"before": before_delta, "active": active_delta, "after": after_delta}[key]))
            if main_interval == "after":
                inferences.append(f"wall 中位差的 r{repeat} 主要来自本 rank read_data 返回后的完成等待，而非该 rank 自身 read/sink 更慢；全 rank 最晚 reader 返回者另行记录，避免把任意一个 wall 最大 rank 误当作执行瓶颈。")
    if phase != "save":
        for config in CONFIGS:
            for aggregate in [a for a in aggregates if a["config"] == config]:
                selected = [c for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == aggregate["repeat"] and c.get("reader_timing", {}).get("reader_return_ns") is not None]
                if not selected:
                    continue
                earliest = min(c["absolute_start_ns"] for c in selected)
                last_reader = max(selected, key=lambda c: c["reader_timing"]["reader_return_ns"])
                last_ns = last_reader["reader_timing"]["reader_return_ns"]
                reader_progress.append(dict(config=config, repeat=aggregate["repeat"], latest_reader_rank=last_reader["row"]["rank"],
                                            latest_reader_return_s=(last_ns - earliest) / 1e9,
                                            reader_return_skew_s=(last_ns - min(c["reader_timing"]["reader_return_ns"] for c in selected)) / 1e9,
                                            after_latest_reader_s=(max(c["absolute_end_ns"] for c in selected) - last_ns) / 1e9))
        if reader_progress and median_pair:
            progress_pair = {p["config"]: p for p in reader_progress if p["repeat"] == median_pair["repeat"]}
            if all(config in progress_pair for config in CONFIGS):
                facts.append("同一配对的最晚 reader 返回 rank Meta/CPU " + "/".join(str(progress_pair[c]["latest_reader_rank"]) for c in CONFIGS) + "；全 rank reader 返回偏斜 " + "/".join(f"{progress_pair[c]['reader_return_skew_s']:.4f}" for c in CONFIGS) + " s。")
    critical_records = []
    for config in CONFIGS:
        for repeat in sorted({c["row"]["repeat"] for c in cases if c["row"]["config"] == config}):
            selected = [c for c in cases if c["row"]["config"] == config and c["row"]["repeat"] == repeat]
            roles = defaultdict(list)
            roles[max(selected, key=lambda c: c["metrics"]["total_s"])["row"]["rank"]].append("max_wall")
            if phase != "save":
                active = [c for c in selected if c.get("reader_timing", {}).get("reader_return_ns") is not None]
                if active:
                    roles[max(active, key=lambda c: c["reader_timing"]["reader_return_ns"])["row"]["rank"]].append("latest_reader")
            else:
                active = [c for c in selected if c.get("writer_timing", {}).get("future_completed_ns") is not None]
                if active:
                    roles[max(active, key=lambda c: c["writer_timing"]["future_completed_ns"])["row"]["rank"]].append("latest_writer")
            if all(c.get("absolute_start_ns") is not None and c["metrics"].get("pre_payload_s") is not None for c in selected):
                roles[min(selected, key=lambda c: c["absolute_start_ns"] + c["metrics"]["pre_payload_s"] * 1e9)["row"]["rank"]].append("first_payload")
            for case in selected:
                rank = case["row"]["rank"]
                if rank in roles:
                    critical_records.append(dict(config=config, repeat=repeat, rank=rank, roles=roles[rank],
                                                 total_s=case["metrics"]["total_s"], report=case["row"].get("report"),
                                                 breakdown=case.get("critical_breakdown", {}),
                                                 staging_operations=case.get("staging_operations", {})))
    return dict(phase=phase, by_config=per_config, critical_rank_component_delta_s=changes,
                paired_critical_rank_components=paired_components, paired_critical_rank_staging=paired_staging,
                paired_critical_rank_staging_outside_primary=paired_staging_outside_primary,
                timeline_pairs=timeline, evidence_zh=facts, inferences_zh=inferences,
                wall_median_pair=median_pair, all_rank_reader_progress=reader_progress,
                payload_onset_breakdown=onset_breakdown,
                critical_rank_breakdowns=critical_records,
                component_summaries_are_diagnostic_not_gates=True,
                largest_observed_component=largest,
                interpretation_zh=" ".join(inferences or facts[:1]),
                attribution_limit_zh="分项区间、staging spans 可能嵌套，分项配对中位数不能相加重构总差；other 是 pwrite/read/fsync/sink 并集的补集。metadata_read 含反序列化/格式校验；reader_active_uncovered 排除了已测 payload/数据读写/staging，仍可能含 seek/close 等已记录辅助调用、callback join/control 和调度，不能冒充磁盘 IO 或唯一的 Python/C++ 归因。" + (" 本阶段部分 sink spans 缺失，未覆盖区间还含未观测 sink。" if missing_sink else " 本阶段 sink spans 字节覆盖完整。"))


def analyze_cell(model, ranks, checkpoint_format, cases):
    groups = defaultdict(list)
    for case in cases:
        row = case["row"]
        groups[(row["phase"], row["repeat"], row["config"])].append(case)
    aggregates = []
    for (phase, repeat, config), selected in sorted(groups.items()):
        present = sorted(c["row"]["rank"] for c in selected)
        if present != list(range(ranks)):
            continue
        aggregate = dict(phase=phase, repeat=repeat, config=config,
                         wall_s=max(c["metrics"]["total_s"] for c in selected),
                         max_rank_without_local_fsync_s=max(c["metrics"]["total_without_fsync_s"] for c in selected),
                         timeline=aggregate_timeline(selected))
        aggregates.append(aggregate)
    phases, sequences, failures = {}, [], []
    for phase in PHASES:
        selected = [a for a in aggregates if a["phase"] == phase]
        wall = paired_summary(
            {a["repeat"]: a["wall_s"] for a in selected if a["config"] == "meta"},
            {a["repeat"]: a["wall_s"] for a in selected if a["config"] == "cpu"})
        quantiles = {str(q): paired_summary(
            {a["repeat"]: a["timeline"]["quantiles_s"][str(q)] for a in selected if a["config"] == "meta"},
            {a["repeat"]: a["timeline"]["quantiles_s"][str(q)] for a in selected if a["config"] == "cpu"},
            tiny_tolerance_s=0 if FIX1 else .05) for q in QUANTILES}
        shape_status = "unknown" if any(v["status"] == "unknown" for v in quantiles.values()) else "fail" if any(v["status"] == "fail" for v in quantiles.values()) else "pass"
        shape_statuses = {field: "unknown" if any(v[field] == "unknown" for v in quantiles.values()) else "fail" if any(v[field] == "fail" for v in quantiles.values()) else "pass" for field in ("strict_status", "floor_status")}
        phase_cases = [c for c in cases if c["row"]["phase"] == phase]
        for repeat in range(1, EXPECTED_REPEATS + 1):
            by_config = {config: {c["row"]["rank"]: c for c in phase_cases if c["row"]["repeat"] == repeat and c["row"]["config"] == config} for config in CONFIGS}
            sequences.append(dict(phase=phase, repeat=repeat, **sequence_pair(by_config["meta"], by_config["cpu"], ranks)))
        values = {config: {a["repeat"]: a for a in selected if a["config"] == config} for config in CONFIGS}
        paired_repeats = sorted(set(values["meta"]) & set(values["cpu"]))
        average_durations = {r: (values["meta"][r]["timeline"]["duration_s"] + values["cpu"][r]["timeline"]["duration_s"]) / 2 for r in paired_repeats}
        target = median(average_durations.values())
        representative = min(paired_repeats, key=lambda r: (abs(average_durations[r] - target), r)) if paired_repeats else None
        detail = dict(wall=wall, shape=dict(status=shape_status, quantiles=quantiles, **shape_statuses), representative_repeat=representative)
        if phase == "save":
            detail["save_excl_local_fsync"] = paired_summary(
                {a["repeat"]: a["max_rank_without_local_fsync_s"] for a in selected if a["config"] == "meta"},
                {a["repeat"]: a["max_rank_without_local_fsync_s"] for a in selected if a["config"] == "cpu"})
            detail["global_compressed_span"] = paired_summary(
                {a["repeat"]: a["timeline"]["duration_s"] for a in selected if a["config"] == "meta"},
                {a["repeat"]: a["timeline"]["duration_s"] for a in selected if a["config"] == "cpu"})
        if wall["status"] == "fail" or shape_status == "fail":
            diagnosis = phase_diagnosis(phase, selected, phase_cases)
            diagnosis["failed_gates"] = (["wall"] if wall["status"] == "fail" else []) + [f"shape-p{q}" for q, v in quantiles.items() if v["status"] == "fail"]
            diagnosis["failed_gates_strict"] = (["wall"] if wall["strict_status"] == "fail" else []) + [f"shape-p{q}" for q, v in quantiles.items() if v["strict_status"] == "fail"]
            diagnosis["failed_gates_floor"] = (["wall"] if wall["floor_status"] == "fail" else []) + [f"shape-p{q}" for q, v in quantiles.items() if v["floor_status"] == "fail"]
            failures.append(diagnosis)
        phases[phase] = detail
    full_statuses = [s["full"]["status"] for s in sequences]
    payload_statuses = [s["payload"]["status"] for s in sequences]

    def sequence_status(statuses):
        return "different" if "different" in statuses else "unknown" if "unknown" in statuses else "identical"

    return dict(model=model, ranks=ranks, checkpoint_format=checkpoint_format,
                oversubscribed=ranks == 16, timing_gated=ranks in (4, 8),
                sequence_status=dict(full=sequence_status(full_statuses), payload=sequence_status(payload_statuses)),
                phases=phases, sequence_comparisons=sequences, aggregates=aggregates,
                failures=failures, case_count=len(cases))


def discover(root, analysis_output, run=None):
    all_cases, errors = [], []
    for selected_run in ([run] if run is not None else sorted((root / "matrix").glob("*"))):
        if not selected_run.is_dir():
            continue
        accepted_warm_outputs = None
        manifest = selected_run / "measurement-results.json"
        if FIX1 and manifest.is_file():
            accepted_warm_outputs = {str(Path(r["output"]).resolve()) for r in base.read_json(manifest)
                                     if r.get("phase") == "warm-load"}
        for row in base.discover_rows(selected_run):
            if row["config"] not in CONFIGS or row["phase"] not in PHASES:
                continue
            # Invalid warm attempts remain auditable, but are never counted as
            # a second observation of the same rank/repeat in the accepted set.
            if FIX1:
                if row["phase"] == "warm-load" and accepted_warm_outputs is not None:
                    if str(Path(row["output"]).resolve()) not in accepted_warm_outputs:
                        continue
                elif Path(row["output"]).name not in PHASES:
                    continue
            try:
                all_cases.append(inspect(row, analysis_output))
            except (ValueError, OSError, KeyError) as error:
                errors.append(dict(report=row["report"], error=str(error)))
    return all_cases, errors


def final_runtime_audit(cases):
    meta = [c for c in cases if c["row"]["config"] == "meta"]
    loads = [c for c in meta if c["row"]["phase"] != "save"]
    failures = []
    for case in meta:
        if case["metrics"]["d2h_s"] or case["metrics"]["h2d_s"]:
            failures.append(dict(report=case["row"]["report"], error="Meta D2H/H2D simulation present"))
    for case in loads:
        sink = case["sink_configuration"].get("meta_load_sink", {})
        checks = dict(
            faithful=sink.get("mode") == "faithful",
            prefaulted=sink.get("prefaulted") is True,
            prepared_before_timing=sink.get("prepared_ns", math.inf) < case["stats"]["dcp_started_ns"],
            working_set_matches_shard=sink.get("resident_buffer_bytes") == sink.get("local_shard_bytes") and sink.get("local_shard_bytes", 0) > 0,
            preparation_outside_load=sink.get("preparation_inside_load_call") is False,
        )
        if not all(checks.values()):
            failures.append(dict(report=case["row"]["report"], checks=checks, sink=sink))
    if FIX1:
        for case in cases:
            if case["row"]["phase"] != "save" and not case["metrics"].get("sink_observation_complete"):
                failures.append(dict(report=case["row"]["report"], error="Incomplete host_sink byte coverage after instrumentation fix"))
            preparation = case.get("admission_preparation", {})
            if preparation and not (preparation.get("admission_baseline_before_timer") is True
                                    and preparation.get("client_prepared_ns", math.inf) < case["stats"]["dcp_started_ns"]
                                    and preparation.get("sample_end_ns", math.inf) < case["stats"]["dcp_started_ns"]):
                failures.append(dict(report=case["row"]["report"], error="Admission baseline not proven outside timer", preparation=preparation))
        for case in [c for c in meta if c["row"]["phase"] == "save"]:
            source = case.get("source_configuration", {}).get("meta_save_source", {})
            checks = dict(faithful=source.get("mode") == "faithful", prefaulted=source.get("prefaulted") is True,
                          prepared_before_timing=source.get("prepared_ns", math.inf) < case["stats"]["dcp_started_ns"],
                          working_set_matches_shard=source.get("resident_buffer_bytes") == source.get("local_shard_bytes") and source.get("local_shard_bytes", 0) > 0,
                          preparation_outside_save=source.get("preparation_inside_save_call") is False)
            if not all(checks.values()):
                failures.append(dict(report=case["row"]["report"], error="Meta resident save source audit failed", checks=checks, source=source))
    return dict(meta_rank_cases=len(meta), meta_load_rank_cases=len(loads),
                checked_mode="faithful", all_pass=not failures, failures=failures,
                load_sink_telemetry_complete=sum(c["metrics"].get("sink_observation_complete", False) for c in cases if c["row"]["phase"] != "save"),
                load_sink_telemetry_missing=sum(not c["metrics"].get("sink_observation_complete", False) for c in cases if c["row"]["phase"] != "save"),
                telemetry_limitation="Missing host_sink spans are reported unknown, not zero; runtime sink mode/prefault/working-set checks use rank reports and are independent of this trace-publication gap.",
                checks=["No Meta D2H/H2D simulation", "Faithful persistent host buffer", "Prefaulted before dcp_started_ns", "resident_buffer_bytes equals local_shard_bytes", "Preparation outside load call"])


def add_sink_bounds(cases):
    """Bound lost sink spans using measured residuals; never fill an estimate."""
    for case in cases:
        if case["row"]["phase"] == "save":
            continue
        metrics = case["metrics"]
        observed = metrics.get("sink_observed_s", metrics.get("sink_s") or 0.0)
        complete = metrics.get("sink_observation_complete", True)
        if complete:
            upper, basis = observed, "Complete sink byte coverage: observed sink union"
        else:
            if metrics.get("reader_active_uncovered_s") is not None:
                residual = metrics["reader_active_uncovered_s"]
                basis = "Observed sink + residual inside reader_active after measured payload/data IO/staging; residual may include seek/close/callback join/control, so this is a conservative bound"
            else:
                residual = metrics.get("uninstrumented_s", metrics["other_s"])
                basis = "Fallback: observed sink + wall residual outside known payload/data IO/staging unions; auxiliary calls may remain; not a point estimate"
            upper = observed + max(0.0, residual)
        metrics.update(sink_lower_bound_s=observed, sink_upper_bound_s=upper,
                       sink_bound_basis=basis, other_s_includes_unobserved_sink=not complete,
                       other_true_lower_bound_s=max(0.0, metrics["other_s"] - (upper - observed)),
                       other_true_upper_bound_s=metrics["other_s"])


def warm_cache_io_audit(cases):
    """Retain the warm protocol label and expose actual IO during the call."""
    groups = defaultdict(list)
    for case in cases:
        row = case["row"]
        if row["phase"] == "warm-load":
            groups[(row["model"], row["world_size"], row["checkpoint_format"], row["config"], row["repeat"])].append(case)
    samples = []
    for (model, ranks, checkpoint_format, config, repeat), selected in sorted(groups.items()):
        reads = [dict(rank=c["row"]["rank"], read_bytes=c["stats"].get("proc_io_delta", {}).get("read_bytes", 0),
                      major_faults=c["stats"].get("resource_delta", {}).get("ru_majflt", 0), report=c["row"]["report"],
                      cache_before_call=c.get("cache_before_call", {}),
                      residency_percent=c.get("cache_before_call", {}).get("residency", {}).get("data_resident_percent"))
                 for c in selected]
        for record, case in zip(reads, selected):
            probe = record["cache_before_call"]
            record["valid"] = (record["residency_percent"] is not None and record["residency_percent"] >= 99
                               and record["read_bytes"] == 0
                               and probe.get("after_resident_buffer_preparation") is True
                               and probe.get("probe_end_ns", math.inf) <= case["stats"]["dcp_started_ns"])
        samples.append(dict(model=model, ranks=ranks, checkpoint_format=checkpoint_format, config=config, repeat=repeat,
                            total_read_bytes=sum(r["read_bytes"] for r in reads),
                            total_major_faults=sum(r["major_faults"] for r in reads),
                            ranks_with_block_reads=[r for r in reads if r["read_bytes"] > 0],
                            rank_checks=reads, valid=all(r["valid"] for r in reads),
                            phase_wall_s=max(c["metrics"]["total_s"] for c in selected)))
    affected = [s for s in samples if s["total_read_bytes"] > 0]
    percentages = [r["residency_percent"] for s in samples for r in s["rank_checks"] if r["residency_percent"] is not None]
    return dict(warm_samples=len(samples), warm_rank_cases=sum(len(g) for g in groups.values()),
                valid_samples=sum(s["valid"] for s in samples), all_valid=bool(samples) and all(s["valid"] for s in samples),
                minimum_residency_percent=min(percentages) if percentages else None,
                samples_with_block_reads=len(affected), ranks_with_block_reads=sum(len(s["ranks_with_block_reads"]) for s in samples),
                total_read_bytes=sum(s["total_read_bytes"] for s in samples), samples=samples,
                limitation="Per-rank residency is probed after resident buffer preparation and before the timed call; >=99% residency and zero measured block-read bytes are mandatory in fix1. Counters cover the complete timed call, including metadata." if FIX1 else "Warm mincore residency was checked before launcher startup; it does not guarantee residency throughout worker startup/target prefault/timed call. Original samples are retained.")


def audit_invalid_warm_attempts(root, accepted_cases):
    """Verify invalid warm requests and sink coverage without accepting timings.

    A cache-invalid phase still performed real requests. Inspect each retained
    attempt in a separate export directory and compare it with the paired
    config's accepted run. Those observations never enter phase aggregates.
    """
    indexed = {(c["row"]["model"], c["row"]["world_size"], c["row"]["checkpoint_format"],
                c["row"]["repeat"], c["row"]["config"], c["row"]["rank"]): c
               for c in accepted_cases if c["row"]["phase"] == "warm-load"}
    attempts, errors = [], []
    output = Path(root) / "invalid-warm-sequence-audit"
    for path in sorted((Path(root) / "matrix").rglob("warm-load-attempts.json")):
        evidence = base.read_json(path)
        for attempt in evidence.get("attempts", []):
            if attempt.get("valid"):
                continue
            invalid, counterparts, details = {}, {}, []
            first = None
            for supplied in attempt["rows"]:
                row = dict(supplied)
                row["report"] = str(Path(row["output"]) / f"rank-{row['rank']:03d}.json")
                row = base.metadata(Path(row["output"]), base.read_json(row["report"]), row)
                first = row if first is None else first
                try:
                    case = inspect(row, output)
                    invalid[row["rank"]] = case
                    key = (row["model"], case["row"]["world_size"], row["checkpoint_format"], row["repeat"],
                           "cpu" if row["config"] == "meta" else "meta", row["rank"])
                    if key in indexed:
                        counterparts[row["rank"]] = indexed[key]
                    details.append(dict(rank=row["rank"], report=row["report"],
                                        validity=supplied.get("warm_validity", {}),
                                        sink_observation_complete=case["metrics"]["sink_observation_complete"],
                                        sink_observed_bytes=case["metrics"]["sink_observed_bytes"],
                                        sink_expected_bytes=case["metrics"]["sink_expected_bytes"],
                                        provenance=case["provenance"], export=case["export"],
                                        payload_export=case["payload_export"]))
                except (ValueError, OSError, KeyError) as error:
                    errors.append(dict(report=row["report"], error=str(error)))
            if first is None:
                errors.append(dict(report=str(path), error="Invalid warm attempt has no rows"))
                continue
            ranks = first.get("world_size", first.get("ranks"))
            left, right = (invalid, counterparts) if first["config"] == "meta" else (counterparts, invalid)
            comparisons = sequence_pair(left, right, ranks)
            attempts.append(dict(model=first["model"], ranks=ranks, checkpoint_format=first["checkpoint_format"],
                                 repeat=first["repeat"], config=first["config"], phase="warm-load",
                                 attempt=attempt["attempt"], source=str(path), accepted_attempt=evidence.get("accepted_attempt"),
                                 timing_eligible=False, rank_checks=details, **comparisons))
    return dict(invalid_attempts=len(attempts), invalid_rank_cases=sum(len(a["rank_checks"]) for a in attempts),
                all_requests_identical=not errors and all(a[s]["status"] == "identical" for a in attempts for s in ("full", "payload")),
                sink_spans_complete=all(r["sink_observation_complete"] for a in attempts for r in a["rank_checks"]),
                attempts=attempts, errors=errors,
                timing_policy="All invalid attempts are sequence-only; none contributes to paired timing gates or plots")


def apply_invalid_sequence_evidence(cells, audit):
    """Complete a missing request gate using every invalid attempt's evidence."""
    for cell in cells:
        matching = [a for a in audit["attempts"] if (a["model"], a["ranks"], a["checkpoint_format"]) ==
                    (cell["model"], cell["ranks"], cell["checkpoint_format"])]
        cell["invalid_warm_request_comparisons"] = matching
        for sequence in cell["sequence_comparisons"]:
            selected = [a for a in matching if a["repeat"] == sequence["repeat"] and a["phase"] == sequence["phase"]]
            if not selected:
                continue
            for scope in ("full", "payload"):
                statuses = [a[scope]["status"] for a in selected]
                if sequence[scope]["status"] == "unknown" and all(s == "identical" for s in statuses):
                    sequence[f"accepted_only_{scope}"] = sequence[scope]
                    sequence[scope] = dict(status="identical", sequence_only_invalid_warm=True,
                                           invalid_attempt_sources=[dict(source=a["source"], attempt=a["attempt"], config=a["config"]) for a in selected],
                                           comparisons=selected[0][scope]["comparisons"], errors=[],
                                           verification="Every retained invalid attempt matched the paired accepted config; timings excluded")
                elif "different" in statuses:
                    sequence[scope]["status"] = "different"
            sequence["invalid_timing_never_admitted"] = True
        for scope in ("full", "payload"):
            statuses = [s[scope]["status"] for s in cell["sequence_comparisons"]] + [a[scope]["status"] for a in matching]
            cell["sequence_status"][scope] = "different" if "different" in statuses else "unknown" if "unknown" in statuses else "identical"


def recover_cell(task):
    """Recover exact payload clocks from checksummed rank reports and CSV metrics."""
    root_text, label, csv_rows = task
    root = Path(root_text)
    cache = root / "analysis-cache" / f"{label}.json"
    if cache.exists():
        return base.read_json(cache)
    cases = []
    for record in csv_rows:
        report_path = Path(record["report"])
        report_bytes = report_path.read_bytes()
        report = json.loads(report_bytes)
        output, rank = report_path.parent, int(record["rank"])
        full_manifest = base.read_json(output / "export/clients" / f"rank-{rank:05d}.trace.json")
        case_id = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
        payload_export = root / "exports" / case_id / f"rank-{rank:05d}.payload.jsonl"
        payload_manifest = base.read_json(payload_export.with_suffix(".trace.json"))
        original_payload = report["payload_events"]
        payload_hash = hashlib.sha256(json.dumps(original_payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
        if payload_hash != full_manifest["payload_events_sha256"]:
            raise ValueError(f"Recovery report payload SHA256 mismatch: {report_path}")
        metrics = {}
        for key, value in record.items():
            if key in {"model", "ranks", "format", "phase", "repeat", "config", "rank", "report"}:
                continue
            metrics[key] = None if value == "" else value == "True" if value in {"True", "False"} else float(value)
        start, end = report["stats"]["dcp_started_ns"], report["stats"]["dcp_ended_ns"]
        payload = sorted((dict(start_ns=e.get("absolute_start_ns", e["start_ns"]) - start,
                               duration_ns=e["duration_ns"], completed=e["completed"]) for e in original_payload), key=lambda e: e["start_ns"])
        if min(e["start_ns"] for e in payload) != round(metrics["pre_payload_s"] * 1e9):
            raise ValueError(f"Recovery first payload clock mismatch: {report_path}")
        if end - start - max(e["start_ns"] + e["duration_ns"] for e in payload) != round(metrics["post_payload_s"] * 1e9):
            raise ValueError(f"Recovery last payload clock mismatch: {report_path}")
        measured_payload_s = base.union_ns((e["start_ns"], e["start_ns"] + e["duration_ns"]) for e in payload) / 1e9
        expected_io_s = metrics["pwrite_s"] + metrics["read_s"]
        if abs(measured_payload_s - expected_io_s) > 1e-9 or len(payload) != payload_manifest["request_event_count"]:
            raise ValueError(f"Recovery payload duration/count mismatch: {report_path}")
        fsync = []
        if record["phase"] == "save":
            result = subprocess.run(["rg", "-F", '"op": "fsync"', str(report_path.with_suffix(".io.jsonl"))], capture_output=True, text=True)
            if result.returncode not in (0, 1):
                raise ValueError(f"Recovery fsync extraction failed: {report_path}")
            for line in result.stdout.splitlines():
                event = json.loads(line)
                absolute = event.get("origin_ns", report["io"]["origin_ns"]) + event["start_ns"]
                fsync.append(dict(start_ns=absolute - start, duration_ns=event["duration_ns"]))
            if abs(base.union_ns(base.clipped(fsync, 0, end - start)) / 1e9 - metrics["fsync_s"]) > 1e-9:
                raise ValueError(f"Recovery fsync union mismatch: {report_path}")
        metrics.update(total_without_fsync_s=metrics["total_s"] - metrics["fsync_s"],
                       pread_s=metrics["read_s"], payload_io_s=measured_payload_s,
                       payload_bytes=sum(e["completed"] for e in payload), payload_requests=len(payload),
                       request_count=full_manifest["request_event_count"],
                       d2h_s=0.0, h2d_s=0.0, d2h_count=0, h2d_count=0)
        row = dict(model=record["model"], world_size=int(record["ranks"]), checkpoint_format=record["format"],
                   phase=record["phase"], repeat=int(record["repeat"]), config=record["config"], rank=rank,
                   report=str(report_path), output=str(output), run=str(root / "matrix" / label))
        known_staging = {op: dict(union_s=metrics[key], outside_primary_s=None, event_count=None, completed_bytes=None)
                         for op, key in (("host_copy", "host_copy_s"), ("host_materialize", "host_materialize_s"), ("host_sink", "sink_observed_s"))
                         if metrics.get(key) is not None}
        cases.append(dict(row=row, metrics=metrics, payload=payload, fsync=fsync, origin_ns=start,
                          absolute_start_ns=start, absolute_end_ns=end, start_ns=0, end_ns=end - start,
                          stats=report["stats"], writer_timing=report.get("writer_timing", {}), writer_limit=report.get("writer_limit"),
                          reader_timing=report.get("reader_timing", {}), reader_limit=report.get("reader_limit"),
                          sink_configuration={k: v for k, v in report.items() if "sink" in k}, staging_operations=known_staging,
                          full_manifest=full_manifest, payload_manifest=payload_manifest,
                          export=full_manifest["output_path"], payload_export=str(payload_export),
                          provenance=dict(report_sha256=hashlib.sha256(report_bytes).hexdigest(), trace_sha256=full_manifest["raw_source"]["sha256"],
                                          export_sha256=full_manifest["output_sha256"], payload_export_sha256=payload_manifest["output_sha256"],
                                          full_export_reused=True, raw_and_export_hashes_verified=True,
                                          raw_and_export_hashes_reverified_during_recovery=False,
                                          integrity_audit="Original analysis-progress.json: 12 cells, 2016 cases, zero errors before plotting failure",
                                          recovered_payload_events_sha256_verified=True, recovered_clock_and_union_metrics_match_csv=True,
                                          staging_per_op_detail="CSV retains host_copy/host_materialize/observed_sink; other per-op subdivisions unavailable after plot failure; stage total and mandatory breakdown retained")))
    first = cases[0]["row"]
    cell = analyze_cell(first["model"], first["world_size"], first["checkpoint_format"], cases)
    retained = ("row", "metrics", "provenance", "export", "payload_export", "stats", "writer_timing", "writer_limit", "reader_timing", "reader_limit", "staging_operations", "sink_configuration", "absolute_start_ns", "absolute_end_ns")
    result = dict(schema_version=1, cell=cell, cases=[{key: c[key] for key in retained} for c in cases], errors=[])
    base.dump(cache, result)
    return result


def recover_from_csv(root, workers):
    progress = base.read_json(root / "analysis-progress.json")
    if progress.get("cells_completed") != 12 or progress.get("rank_cases") != 2016 or progress.get("errors") != 0:
        raise ValueError("Recovery requires the complete original integrity audit")
    original_csv = root / "rank-breakdown-before-recovery.csv"
    if not original_csv.exists():
        original_csv.write_bytes((root / "rank-breakdown.csv").read_bytes())
    with original_csv.open() as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2016:
        raise ValueError("Recovery CSV must contain all 2016 original rank cases")
    groups = defaultdict(list)
    for row in rows:
        groups[f"{row['model']}-{row['ranks']}r-{row['format']}"].append(row)
    cases, cells, errors = [], [], []
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(recover_cell, (str(root), label, selected)) for label, selected in groups.items()]
        for future in as_completed(futures):
            result = future.result()
            cases.extend(result["cases"])
            cells.append(result["cell"])
            errors.extend(result["errors"])
            print(json.dumps(dict(recovered_cells=len(cells), rank_cases=len(cases), elapsed_s=time.monotonic() - started)), flush=True)
    cells.sort(key=lambda c: (c["model"], c["ranks"], c["checkpoint_format"]))
    base.dump(root / "analysis-recovery.json", dict(method="CSV metrics + original checksummed absolute report payload events + rg-only save fsync extraction + original verified full/payload signatures", workers=workers,
                                                  original_completed_integrity_audit=progress, raw_rehashed=False, full_exports_reparsed=False,
                                                  report_payload_checksums_and_csv_clock_count_duration_fsync_equivalence_verified=True,
                                                  elapsed_s=time.monotonic() - started, rank_cases=len(cases)))
    return cases, errors, cells


def save_then_plot(summary, root, no_plots):
    """The expensive verified result must survive plotting/environment failures."""
    if summary.get("methodology", {}).get("fix1"):
        helper = fix1_report_module(root)
        helper.enrich(summary)
        helper.write_details(summary)
    base.dump(root / "analysis.json", summary)
    (root / "report-zh.md").write_text(report_text(summary))
    if not no_plots:
        try:
            summary["plots"] = make_plots(summary["cells"], root / "plots")
        except Exception as error:
            base.dump(root / "plot-error.json", dict(error=str(error), analysis_checkpoint_preserved=True))
            raise
        base.dump(root / "analysis.json", summary)


def attach_focused_diagnoses(analysis, root):
    save_path = root / "save-shape-diagnosis.json"
    if save_path.exists():
        focused = base.read_json(save_path)
        indexed = {(c["model"], c["ranks"], c["checkpoint_format"]): c for c in focused["cells"]}
        for cell in analysis["cells"]:
            source = indexed.get((cell["model"], cell["ranks"], cell["checkpoint_format"]))
            if not source:
                continue
            for failure in cell["failures"]:
                if failure["phase"] != "save" or "wall" in failure["failed_gates"]:
                    continue
                failure["generic_wall_component_observations_zh"] = failure["evidence_zh"]
                failure["focused_diagnosis_source"] = str(save_path)
                diagnoses = source["selected_quantile_diagnoses"]
                chosen = [diagnoses[0]] + ([diagnoses[-1]] if len(diagnoses) > 1 else [])
                facts = []
                for q in chosen:
                    admission = q["first_writer_admission_delta_meta_minus_cpu"]
                    work = q["cumulative_delta_meta_minus_cpu"]
                    facts.append(f"P{q['quantile']} 中位配对 r{q['selected_pair_repeat']}：shape {q['paired_median_delta_s'] * 1000:+.3f} ms；首次 writer 准入等待 {admission['admission_wait_s'] * 1000:+.3f} ms（其中 baseline sample {admission['baseline_sample_s'] * 1000:+.3f} ms）；累计 host_materialize {work['host_materialize_s'] * 1000:+.3f} ms、pwrite {work['pwrite_s'] * 1000:+.3f} ms。")
                failure["evidence_zh"] = facts
                failure["inferences_zh"] = ["该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。"]
                failure["interpretation_zh"] = failure["inferences_zh"][0]
                failure["focused_quantile_diagnoses"] = diagnoses


def breakdown_csv(cases, path):
    fields = ("model", "ranks", "format", "phase", "repeat", "config", "rank", "total_s", "pwrite_s", "read_s", "fsync_s", "sink_s", "sink_observed_s", "sink_observation_complete", "sink_observed_bytes", "sink_expected_bytes", "sink_lower_bound_s", "sink_upper_bound_s", "sink_bound_basis", "other_s", "other_true_lower_bound_s", "other_true_upper_bound_s", "other_s_includes_unobserved_sink", "pre_payload_s", "post_payload_s", "host_copy_s", "host_materialize_s", "staging_union_s", "nonpayload_io_s", "uninstrumented_s", "before_writer_queue_s", "writer_admission_wait_s", "writer_active_s", "after_writer_completed_s", "before_metadata_queue_s", "metadata_gate_wait_s", "metadata_read_s", "metadata_prune_s", "after_metadata_to_reader_queue_s", "reader_admission_wait_s", "reader_active_s", "reader_active_uncovered_s", "after_reader_return_s", "report")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for case in cases:
            row = case["row"]
            record = dict(model=row["model"], ranks=row["world_size"], format=row["checkpoint_format"], phase=row["phase"], repeat=row["repeat"], config=row["config"], rank=row["rank"], report=row["report"])
            record.update({key: value for key, value in case["metrics"].items() if key in fields})
            writer.writerow(record)


def make_plots(cells, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output.mkdir(parents=True, exist_ok=True)
    plots, layout_checks = [], []

    def verify_layout(fig, ax, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        boxes, clipped = [], []
        artists = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label,
                   ax.xaxis.offsetText, ax.yaxis.offsetText, *ax.get_legend().get_texts()]
        for axis, limits in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
            low, high = sorted(limits)
            for tick in [*axis.get_major_ticks(), *axis.get_minor_ticks()]:
                if low <= tick.get_loc() <= high:
                    artists.extend((tick.label1, tick.label2))
        for artist in artists:
            if not artist.get_visible() or not artist.get_text().strip():
                continue
            box = artist.get_window_extent(renderer)
            record = dict(text=artist.get_text(), bounds=list(box.bounds))
            boxes.append(record)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                clipped.append(record)
        title = ax._left_title.get_window_extent(renderer)
        legend = ax.get_legend().get_window_extent(renderer)
        overlap = bool(title.overlaps(legend))
        layout_checks.append(dict(name=name, canvas_pixels=[width, height], all_text_inside_figure=not clipped,
                                  clipped_text=clipped, title_legend_overlap=overlap, text_boxes=boxes))
        if clipped or overlap:
            raise ValueError(f"Plot text layout does not fit: {name}: {clipped}; overlap={overlap}")
    for cell in cells:
        for phase in PHASES:
            repeat = cell["phases"][phase]["representative_repeat"]
            if repeat is None:
                continue
            selected = {a["config"]: a for a in cell["aggregates"] if a["phase"] == phase and a["repeat"] == repeat}
            fig, ax = plt.subplots(figsize=(12, 5), dpi=150)
            fig.subplots_adjust(left=.085, right=.985, top=.84, bottom=.14)
            for config, label, color, style in (("meta", "Meta", "#2a78d6", "--"), ("cpu", "CPU", "#1baf7a", "-")):
                timeline = selected[config]["timeline"]
                values = np.asarray(timeline["binned_bytes_per_50ms"])
                widths = np.minimum(.05, np.maximum(0, timeline["duration_s"] - np.arange(len(values)) * .05))
                rate = np.divide(values / 1e9, widths, out=np.zeros_like(values), where=widths > 0)
                xs = np.arange(len(values)) * .05 + widths / 2
                ax.plot(xs, rate, color=color, ls=style, lw=1.3, label=f"{label}  {timeline['duration_s']:.2f} s")
            ax.set_xlim(0, max(a["timeline"]["duration_s"] for a in selected.values()) * 1.02)
            ax.set_ylim(bottom=0)
            ax.set_xlabel("Time (s, fsync excluded)" if phase == "save" else "Time (s)", fontsize=11)
            ax.set_ylabel("Aggregate throughput (GB/s)", fontsize=11)
            format_title = "DCP" if cell["checkpoint_format"] == "dcp" else "SafeTensors"
            phase_title = {"save": "Save", "cold-load": "Cold load", "warm-load": "Warm load"}[phase]
            detail = cell["phases"][phase]
            incomplete = f" · {len(detail['wall']['samples'])} valid pairs; gate incomplete" if not detail["wall"]["complete"] else ""
            ax.set_title(f"{cell['model'].upper()} · {cell['ranks']} ranks · {format_title} · {phase_title}{incomplete}", loc="left", fontsize=12 if incomplete else 14, pad=12)
            ax.grid(color="#e5e5e5", lw=.5)
            ax.spines[["top", "right"]].set_visible(False)
            ax.legend(frameon=False, fontsize=10, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=2)
            stem = f"{cell['model']}-{cell['ranks']}r-{cell['checkpoint_format']}-{phase}"
            verify_layout(fig, ax, stem)
            for suffix in ("png", "pdf"):
                path = output / f"{stem}.{suffix}"
                fig.savefig(path)
                plots.append(str(path))
            plt.close(fig)
    fig, ax = plt.subplots(figsize=(14, 5), dpi=150)
    fig.subplots_adjust(left=.075, right=.99, top=.86, bottom=.24)
    xs = np.arange(len(cells))
    for index, (phase, color, label) in enumerate((("save", "#2a78d6", "Save (excl. fsync)"), ("cold-load", "#1baf7a", "Cold load"), ("warm-load", "#e69527", "Warm load"))):
        metric = "global_compressed_span" if phase == "save" else "wall"
        ys = [c["phases"][phase][metric]["paired_median_delta_percent"] if c["phases"][phase][metric]["complete"] else None for c in cells]
        ax.plot(xs + (index - 1) * .12, [v if v is not None else float("nan") for v in ys], marker="o", ms=5, lw=1.1, color=color, label=label)
    ax.axhline(0, color="#777777", lw=.7)
    for value in (-3, 3):
        ax.axhline(value, color="#b9b9b9", lw=.7, ls="--")
    labels = [f"{c['model'].upper()} / {c['ranks']}r{'*' if c['oversubscribed'] else ''}\n{c['checkpoint_format']}" for c in cells]
    ax.set_xticks(xs, labels, fontsize=9)
    ax.set_ylabel("Paired median Meta − CPU (%)", fontsize=11)
    incomplete_count = sum(not c["phases"][p]["wall"]["complete"] for c in cells for p in PHASES)
    ax.set_title("Meta vs CPU phase time" + (f" · {incomplete_count} incomplete point omitted" if incomplete_count else ""), loc="left", fontsize=15, pad=12)
    ax.grid(axis="y", color="#e5e5e5", lw=.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=10.5, loc="lower right", bbox_to_anchor=(1, 1.0), ncol=3)
    verify_layout(fig, ax, "summary")
    for suffix in ("png", "pdf"):
        path = output / f"summary.{suffix}"
        fig.savefig(path)
        plots.append(str(path))
    plt.close(fig)
    base.dump(output.parent / "plot-layout-audit.json", dict(figures=len(layout_checks), all_pass=all(c["all_text_inside_figure"] and not c["title_legend_overlap"] for c in layout_checks), checks=layout_checks))
    return plots


def fix1_report_module(root):
    path = Path(root) / "render_report.py"
    spec = importlib.util.spec_from_file_location("multirank_fix1_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def report_text(analysis):
    if analysis.get("methodology", {}).get("fix1"):
        return fix1_report_module(analysis["root"]).report_text(analysis)
    lines = ["# 多 rank Meta / CPU IO 对齐", "",
             "所有 wall 秒数均取每次运行各 rank 的 dcp_wall_seconds 最大值。配对差取三次 Meta−CPU 的中位数，百分比分母为对应三次 CPU 的中位数。时间门限 ±3%；shape 使用 50 ms 聚合 payload bins 插值的 10/50/90/100% 到达时间，门限 max(3% × CPU 中位数, 50 ms)。",
             "", "完整与 payload 请求均逐 rank、逐阶段、逐 repeat 精确比较。16 ranks 在 8 核机器上过度订阅：请求相同仍为硬门限，时间/shape 仅报告。", "",
             "save 图与 shape 从所有 rank 的最早 dcp_started_ns 起步，扣除所有 rank 的 fsync 区间并集；wall 门限含 fsync。save excl. local fsync（各 rank 扣各自 fsync 后再取最大值）另行报告，不能等同于全局压缩时间。", ""]
    gated = [c for c in analysis["cells"] if c["timing_gated"]]
    wall_count = sum(len(c["phases"]) for c in gated)
    wall_fails = sum(p["wall"]["status"] == "fail" for c in gated for p in c["phases"].values())
    shape_fails = sum(p["shape"]["status"] == "fail" for c in gated for p in c["phases"].values())
    quantile_count = sum(len(p["shape"]["quantiles"]) for c in gated for p in c["phases"].values())
    quantile_fails = sum(q["status"] == "fail" for c in gated for p in c["phases"].values() for q in p["shape"]["quantiles"].values())
    requests_pass = bool(analysis["cells"]) and all(all(v == "identical" for v in c["sequence_status"].values()) for c in analysis["cells"])
    lines[2:2] = [f"**总体结论：4/8 ranks 的时间/形状验收{'未通过' if wall_fails or quantile_fails else '无已测失败'}；完整/payload 请求{'全部通过' if requests_pass else '未全部通过'}。** {wall_count} 个 wall gate 中 {wall_fails} 个失败；{quantile_count} 个 shape quantile gate 中 {quantile_fails} 个失败，分布于 {shape_fails}/{wall_count} 个阶段。16 ranks 仅报告时间，不参与这些门限计数。", ""]
    loads = [c for c in analysis.get("cases", []) if c["row"]["phase"] != "save"]
    missing_sink = sum(not c["metrics"].get("sink_observation_complete", True) for c in loads)
    if missing_sink:
        lines += [f"**分项遥测限制：{missing_sink}/{len(loads)} 个 load rank case 缺少完整 host_sink spans。** 既有多 rank compute staging 使用最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，DCP/safetensors 均会丢失部分 rank 的 sink spans。表内 sink_s 记 unknown（CSV 留空），sink_observed_s 仅为可观测下界；other 和 reader_active_uncovered 含未观测的真实目标写入，不能称其为纯 Python 开销。逐 rank 同时报告 sink 下/上界：上界为 observed sink 加 reader 活跃区间内未被已知数据读写/staging 覆盖的残余；残余还含 callback join/control，因此它是保守上界而非估计值。other 的真实区间也相应给出上下界。完整请求、payload 字节、wall、shape 与计时前 faithful buffer 准备证据仍完整；原始测量未改写。后续源码修复只影响新 trace，不能补回历史 spans。", ""]
    sink = analysis.get("sink_selection")
    if sink:
        lines += ["## CPU-target sink", "", "默认仍使用 fresh sink；CPU-target 采用计时前分配并预触页的 rank 本地持久缓冲区，faithful 覆盖本 rank 全部 shard，bounded 复用固定大小 ring。选择及两模型 pilot 证据见 [sink-selection.json](sink-selection.json)。", "",
                  f"矩阵采用 **{sink.get('selected_sink', '待定')}**；ring 大小 {sink.get('ring_bytes', 268435456) / 1024**2:.0f} MiB。下表为 4 ranks warm load；差值同样取配对中位数。", "",
                  "|模型|格式|CPU 秒|Faithful 秒 / 差值|Ring 秒 / 差值|请求 faithful / ring|", "|---|---|---:|---|---|---|"]
        for pilot in sink.get("cells", []):
            variants = []
            for name in ("faithful", "ring"):
                value = pilot.get(name, {})
                seconds, percent = value.get("median_s"), value.get("paired_delta_percent")
                variants.append("缺测" if seconds is None or percent is None else f"{seconds:.3f} / {percent:+.2f}%")
            cpu = pilot.get("cpu_median_s")
            cpu_label = "缺测" if cpu is None else f"{cpu:.3f}"
            request_labels = ["identical" if pilot.get(name, {}).get("requests_identical") else "未通过/缺测" for name in ("faithful", "ring")]
            lines.append(f"|{pilot.get('model', '?')}|{pilot.get('checkpoint_format', '?')}|{cpu_label}|{variants[0]}|{variants[1]}|{' / '.join(request_labels)}|")
        lines.append("")
    warm_audit = analysis.get("warm_cache_io_audit")
    if warm_audit:
        lines += ["## Warm 缓存方法与实际磁盘 IO", "",
                  f"预热协议在启动 launcher 前用 mincore 检查 100% 驻留；它不保证 worker 启动、目标预触页及实际调用期间始终驻留。保留所有 warm 标签与原始样本；调用内 proc_io.read_bytes>0 的样本为 {warm_audit['samples_with_block_reads']}/{warm_audit['warm_samples']}，涉及 {warm_audit['ranks_with_block_reads']}/{warm_audit['warm_rank_cases']} 个 rank case，总计 {warm_audit['total_read_bytes']:,} bytes 实际块读取。详见 [warm-cache-io-audit.json](warm-cache-io-audit.json)。该计数覆盖整次调用，也可能来自 metadata；少量非零不能直接证明 payload 缓存失效，且它与 payload 请求字节不同。", ""]
        affected = [s for s in warm_audit["samples"] if s["total_read_bytes"] > 0]
        if affected:
            lines += ["|模型 / ranks / 格式|配置|Repeat|调用内块读取 bytes|Wall 秒|", "|---|---|---:|---:|---:|"]
            for sample in affected:
                lines.append(f"|{sample['model']} / {sample['ranks']} / {sample['checkpoint_format']}|{sample['config']}|{sample['repeat']}|{sample['total_read_bytes']:,}|{sample['phase_wall_s']:.3f}|")
            lines.append("")
    lines += ["## Gate 表", "", "单元格为 Meta / CPU 中位秒数、配对差 %；✅/❌ 为该项实测门限结果，16r 的时间结果不用于验收。shape 列按 save/cold/warm 顺序。", "", "|模型|ranks|格式|请求 full / payload|Save 含 fsync|Cold load|Warm load|Shape 10/50/90/100%|", "|---|---:|---|---|---|---|---|---|"]

    def time_cell(value):
        if value["meta_median_s"] is None:
            return "缺测"
        symbol = {"pass": "✅", "fail": "❌", "unknown": "?"}[value["status"]]
        return f"{symbol} {value['meta_median_s']:.3f} / {value['cpu_median_s']:.3f} ({value['paired_median_delta_percent']:+.2f}%)"

    for cell in analysis["cells"]:
        sequences = cell["sequence_status"]
        shape = "/".join({"pass": "✅", "fail": "❌", "unknown": "?"}[cell["phases"][p]["shape"]["status"]] for p in PHASES)
        rank_label = f"{cell['ranks']}（过订阅）" if cell["oversubscribed"] else str(cell["ranks"])
        lines.append(f"|{cell['model']}|{rank_label}|{cell['checkpoint_format']}|{sequences['full']} / {sequences['payload']}|" + "|".join(time_cell(cell["phases"][p]["wall"]) for p in PHASES) + f"|{shape}|")
    lines += ["", "## Shape 与 save 扣 fsync 明细", "", "shape 单元格为 Meta / CPU 秒数（配对差毫秒）。量化精度受 50 ms bins 的均匀插值限制。", "", "|模型 / ranks / 格式|阶段|P10|P50|P90|P100|", "|---|---|---|---|---|---|"]
    for cell in analysis["cells"]:
        label = f"{cell['model']} / {cell['ranks']} / {cell['checkpoint_format']}"
        for phase in PHASES:
            columns = []
            for value in cell["phases"][phase]["shape"]["quantiles"].values():
                columns.append("缺测" if value["meta_median_s"] is None else f"{value['meta_median_s']:.3f}/{value['cpu_median_s']:.3f} ({value['paired_median_delta_s'] * 1000:+.1f} ms)")
            lines.append(f"|{label}|{phase}|" + "|".join(columns) + "|")
    lines += ["", "|模型 / ranks / 格式|Save excl. local fsync：Meta / CPU|Global fsync union removed：Meta / CPU|全局压缩时间：Meta / CPU|", "|---|---|---|---|"]
    for cell in analysis["cells"]:
        detail = cell["phases"]["save"]
        local, global_span = detail["save_excl_local_fsync"], detail["global_compressed_span"]
        unions = {config: median(a["timeline"]["global_fsync_union_s"] for a in cell["aggregates"] if a["phase"] == "save" and a["config"] == config) for config in CONFIGS}
        fmt = lambda a, b: "缺测" if a is None or b is None else f"{a:.3f} / {b:.3f} s"
        lines.append(f"|{cell['model']} / {cell['ranks']} / {cell['checkpoint_format']}|{fmt(local['meta_median_s'], local['cpu_median_s'])}|{fmt(unions['meta'], unions['cpu'])}|{fmt(global_span['meta_median_s'], global_span['cpu_median_s'])}|")
    lines += ["", "## Save gap 证据", "", "runner 使用 writer_limit=1，写入许可覆盖 write_data 以及其 fsync。一个 rank 的 checkpoint wall 包含等待其他 rank 的写入与 fsync；只扣本 rank 的 fsync，会把其他 rank 的 fsync 仍保留在本 rank 的 other 中。因此本地 save excl. fsync 很长而 pwrite 很短可以同时成立，是实测准入/collective 等待。", "", "逐 rank 的 queue_enter_ns/write_enter_ns/future_completed_ns 与 fsync 原始时间见 analysis.json；[rank-breakdown.csv](rank-breakdown.csv) 分列准入等待、writer 活跃区间和完成后的等待。pwrite/read/fsync/sink 为各自区间并集；other 是其总并集的补集，host_copy 等 staging 和 Python 调用开销会计入 other，分项若有重叠不作盲目相加。", "", "load 进一步保留 metadata gate 等待、metadata 读取/反序列化/校验、剪枝重建、planning/collective、reader 准入/活跃/返回后区间。CSV 保留 mandatory 的 pwrite/read/fsync/sink/other 与 staging_union。恢复后的 compact analysis 仅保留 CSV 的 host_copy、host_materialize、observed_sink 可观测下界；其他 staging op 的完整细分未重新扫描，次数/字节/排除主分项后的独立区间缺失时记 unknown。原始 JSONL 全部保留，定点诊断另从 raw trace 提供 host_read_copy、host_read_materialize 等分项；不声称 compact JSON 已包含每个 op 的完整分解。嵌套 staging spans 不相加推导 wall；uninstrumented 排除了已测 payload/数据读写/fsync/staging，仍可能含已记录 seek/flush/close 等辅助调用；它不能解释为纯未插桩或纯 Python 时间。", ""]
    if analysis.get("save_gap_evidence"):
        lines += ["额外诊断文件：" + "、".join(f"[{Path(p).name}]({Path(p).name})" for p in analysis["save_gap_evidence"]) + "。", ""]
    supplementary = [name for name in ("warm-gap-diagnosis.json", "pilot-gap-diagnosis.md", "pilot-gap-diagnosis.json", "save-shape-diagnosis.md", "save-shape-diagnosis.json", "cold-shape-diagnosis.md", "cold-shape-diagnosis.json", "oversubscribed-gap-diagnosis.md", "postmeasure-staging-fix-tests.json", "postmeasure-staging-fix-audit.json", "execution-audit-zh.md", "final-artifact-audit.json", "plot-file-audit.json", "plot-layout-audit.json", "analysis-recovery.json") if (Path(analysis["root"]) / name).exists()]
    if supplementary:
        lines += ["Warm 与 pilot 的定点诊断：" + "、".join(f"[{name}]({name})" for name in supplementary) + "。pilot 保留 1B safetensors faithful +4.151% 的真实偏差和 3B DCP faithful 首次 16.257 s 样本；后者额外约 9.3 s 的 payload read 与实际磁盘读取相符，未删样本或重跑替换。", ""]
    save_shape_path = Path(analysis["root"]) / "save-shape-diagnosis.json"
    if save_shape_path.is_file():
        focused_save = base.read_json(save_shape_path)
        lines += ["## 未通过 shape 的专门诊断", "", "五个 gated save 形状失败均在扣除全 rank fsync 后仍成立。首次 writer 的 memory-safety 准入有意要求 fresh PSS/baseline sample；CPU resident 模型与 Meta zero source 的采样和 source bytes 物化成本不同，是实测路径差异，未证明算术 bug。不能把这些 payload 到达偏差归因于写后 collective。", "", "|单元|P10 首次准入等待 Δ ms|其中 baseline sample Δ ms|", "|---|---:|---:|"]
        for cell in focused_save["cells"]:
            q = next((q for q in cell["selected_quantile_diagnoses"] if q["quantile"] == 10), None)
            if q:
                d = q["first_writer_admission_delta_meta_minus_cpu"]
                lines.append(f"|{cell['label']}|{d['admission_wait_s'] * 1000:+.3f}|{d['baseline_sample_s'] * 1000:+.3f}|")
        long_tail = next((q for c in focused_save["cells"] if c["label"] == "1b-4r-safetensors" for q in c["selected_quantile_diagnoses"] if q["quantile"] == 100), None)
        if long_tail:
            d = long_tail["cumulative_delta_meta_minus_cpu"]
            lines += ["", f"1B/4r SafeTensors P100 的实际中位配对：host_materialize {d['host_materialize_s'] * 1000:+.3f} ms、pwrite {d['pwrite_s'] * 1000:+.3f} ms；净 shape {long_tail['paired_median_delta_s'] * 1000:+.3f} ms。该 span 是 source 读取/bytes 分配/ctypes callback 的整体，不等同于磁盘 IO 或 load sink。所有精确配对、累计区间和 source 行号见 save-shape-diagnosis。", ""]
    cold_path = Path(analysis["root"]) / "cold-shape-diagnosis.json"
    if cold_path.is_file():
        focused_cold = base.read_json(cold_path)
        lines += ["3B/8r DCP cold P10：中位配对延迟 +815.747 ms（+6.023%），但 Meta 首 payload 更早 290.264 ms；差异在 payload 开始后扩大。到 P10 的 read API rank-sum 多 6.727 s，其中 ≥50 ms 调用多 3.671 s/43 次；三次整调用块读取字节差均为 0，同一配对整调用 CPU 时间差 −9.858 ms。由此定位到早段并发 read/interleaving；read API 包含内核/设备等待与调度，不能唯一拆开。P50/P90/P100 与 wall 均通过。", "", focused_cold["interpretation_zh"], ""]
    oversubscribed_path = Path(analysis["root"]) / "oversubscribed-gap-diagnosis.json"
    if oversubscribed_path.is_file():
        focused_oversubscribed = base.read_json(oversubscribed_path)
        d = focused_oversubscribed["selected_critical_rank_deltas_s"]
        lines += [f"16r 仅报告项：1B DCP cold 的中位 wall 配对差 {focused_oversubscribed['wall']['paired_median_delta_ms']:+.3f} ms；关键 rank read API {d['read_s'] * 1000:+.3f} ms、onset {d['pre_payload_s'] * 1000:+.3f} ms、source staging {d['staging_union_s'] * 1000:+.3f} ms。字节相同，证据与并发读调用延迟/调度变化一致；16 进程/8 核是已知条件，不能唯一证明超订阅本身是原因。", ""]
    focus_path = Path(analysis["root"]) / "warm-gap-diagnosis.json"
    if focus_path.is_file():
        focus = base.read_json(focus_path)
        lines += ["4 ranks DCP warm 的 wall 未通过项，以下使用实际决定配对中位差的同一 repeat（毫秒），不是分别取分项中位数后相加：", "",
                  "|模型|Repeat|进入 reader 前 Δ|Reader 活跃 Δ|返回后等待 Δ|Sink+join 包络 Δ|", "|---|---:|---:|---:|---:|---:|"]
        for model in focus.get("models", []):
            pair = min(model["pairs"], key=lambda p: abs(p["deltas_meta_minus_cpu"]["total_s"] - model["paired_median_delta_s"]))
            values = pair["deltas_meta_minus_cpu"]
            lines.append(f"|{model['model']}|{pair['repeat']}|{values['before_reader_s'] * 1000:+.3f}|{values['reader_active_s'] * 1000:+.3f}|{values['after_reader_s'] * 1000:+.3f}|{pair['sink_envelope_delta_s'] * 1000:+.3f}|")
        lines += ["", "包络由整段 tensor 的最后一次 source materialize 到下一次 read 的实际时间界定，并验证包含所有已发布 rank3 sink；它包括目标写入及 callback join/control，不是纯 memcpy。两项中位配对的 Meta sink+join 包络均更长，因此不能把更短 wall 解释为 Meta 获得了更短目标写入包络。1B 主要差值位于返回后的全局完成等待；3B 同时包含 read API 与完成等待的变化。", ""]
    lines += ["## 未通过项及归因", ""]
    failure_count = 0
    for cell in analysis["cells"]:
        for failure in cell["failures"]:
            failure_count += 1
            prefix = "16r 仅报告" if cell["oversubscribed"] else "验收未通过"
            lines.append(f"- {prefix}：{cell['model']} / {cell['ranks']}r / {cell['checkpoint_format']} / {failure['phase']}，{', '.join(failure['failed_gates'])}。")
            lines.append("  Trace 证据：" + " ".join(failure["evidence_zh"]))
            lines.append("  基于这些区间的归因：" + failure["interpretation_zh"])
        if cell["sequence_status"]["full"] != "identical" or cell["sequence_status"]["payload"] != "identical":
            lines.append(f"- 请求证据未通过/不完整：{cell['model']} / {cell['ranks']}r / {cell['checkpoint_format']}；见 analysis.json 中 sequence_comparisons 的首差异与缺 rank 清单。")
    if not failure_count:
        lines.append("已完成配对样本的时间与 shape 未发现超门限项。")
    if analysis["errors"]:
        lines += ["", "读取/证据错误：", ""] + [f"- {e['report']}: {e['error']}" for e in analysis["errors"]]
    progress = analysis.get("progress", {})
    if progress.get("failed") or progress.get("skipped"):
        lines += ["", "跳过/失败单元：", "", "```json", json.dumps({k: progress[k] for k in ("failed", "skipped") if k in progress}, ensure_ascii=False, indent=2), "```"]
    lines += ["", "## 输出与验证范围", "", f"分析包含 {analysis['rank_cases']} 个 rank case。完整/载荷 trace exports、每次全 rank 聚合 bins、逐 repeat 配对值、gate 明细与 provenance SHA256 均保留在 [analysis.json](analysis.json) 和 exports/；逐 rank 分解见 rank-breakdown.csv；图见 plots/。", "", "绘图每个 cell/phase 选择配对平均时间最接近三次中位数的同一 repeat；Meta/CPU 各自使用该 repeat。save 均无 fsync 轴区间。冷/热缓存证据与清理结果保留在 runner 的原始目录。读取已有 validation 仅用于背景核对，未改写旧目录。", ""]
    return "\n".join(lines)


RETAINED_FIELDS = ("row", "metrics", "provenance", "export", "payload_export", "full_manifest", "payload_manifest", "stats", "writer_timing", "writer_limit", "reader_timing", "reader_limit", "staging_operations", "sink_configuration", "source_configuration", "cache_before_call", "admission_preparation", "critical_breakdown", "absolute_start_ns", "absolute_end_ns")


def inspect_cell_task(task):
    """Inspect one cell; independent cells have disjoint immutable exports."""
    global FIX1, EXPECTED_REPEATS, FLOOR_S
    root_text, model, ranks, checkpoint_format, options, resume = task
    FIX1, EXPECTED_REPEATS, FLOOR_S = options["fix1"], options["expected_repeats"], options["floor_s"]
    root = Path(root_text)
    run = root / "matrix" / f"{model}-{ranks}r-{checkpoint_format}"
    cache = root / "analysis-cache" / f"{run.name}.json"
    if resume and cache.exists():
        cached = base.read_json(cache)
        if cached.get("analysis_options") != options and (FIX1 or cached.get("analysis_options") is not None):
            raise ValueError(f"Cached analysis options differ: {cache}")
        return cached
    selected, errors = discover(root, root, run=run)
    cell = analyze_cell(model, ranks, checkpoint_format, selected)
    retained = [{key: c[key] for key in RETAINED_FIELDS} for c in selected]
    result = dict(schema_version=1, cell=cell, cases=retained, errors=errors, analysis_options=options)
    base.dump(cache, result)
    return result


def main(argv=None):
    global EXPECTED_REPEATS, FLOOR_S, FIX1
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--refresh-only", action="store_true", help="Refresh compact-case diagnostics/bounds/report without rereading raw traces or replotting")
    parser.add_argument("--recover-csv", action="store_true", help="Recover a completed pre-plot run from durable CSV and original checksummed payload reports")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--plots-only", action="store_true", help="Plot a durable analysis.json without inspecting traces")
    parser.add_argument("--resume-cache", action="store_true", help="Reuse immutable previously verified cell inputs without rehashing; default reinspects inputs")
    parser.add_argument("--fix1", action="store_true", help="Five paired repeats, only 4/8 ranks, strict and 100 ms floor gates, before/after Chinese report")
    parser.add_argument("--expected-repeats", type=int)
    parser.add_argument("--ranks", type=int, nargs="+", choices=(4, 8, 16))
    parser.add_argument("--floor-ms", type=float, default=100)
    parser.add_argument("--previous-root", type=Path, default=ROOT / "validation/multirank-meta-cpu-20261029")
    args = parser.parse_args(argv)
    FIX1 = args.fix1
    EXPECTED_REPEATS = args.expected_repeats or (5 if FIX1 else 3)
    FLOOR_S = args.floor_ms / 1000
    rank_values = args.ranks or ([4, 8] if FIX1 else [4, 8, 16])
    root = (args.root or ROOT / ("validation/multirank-fix1-20261030" if FIX1 else "validation/multirank-meta-cpu-20261029")).resolve()
    if FIX1 and root == args.previous_root.resolve():
        raise ValueError("Fix1 output must differ from the read-only previous round")
    root.mkdir(parents=True, exist_ok=True)
    if args.plots_only:
        summary = base.read_json(root / "analysis.json")
        save_then_plot(summary, root, args.no_plots)
        return summary
    if args.refresh_only:
        summary = base.read_json(root / "analysis.json")
        add_sink_bounds(summary["cases"])
        for cell in summary["cells"]:
            selected = [c for c in summary["cases"] if c["row"]["model"] == cell["model"] and c["row"]["world_size"] == cell["ranks"] and c["row"]["checkpoint_format"] == cell["checkpoint_format"]]
            refreshed = []
            for failure in cell["failures"]:
                phase = failure["phase"]
                diagnosis = phase_diagnosis(phase, [a for a in cell["aggregates"] if a["phase"] == phase], [c for c in selected if c["row"]["phase"] == phase])
                diagnosis["failed_gates"] = failure["failed_gates"]
                refreshed.append(diagnosis)
            cell["failures"] = refreshed
        attach_focused_diagnoses(summary, root)
        base.dump(root / "final-timing-sink-audit.json", final_runtime_audit(summary["cases"]))
        summary["warm_cache_io_audit"] = warm_cache_io_audit(summary["cases"])
        base.dump(root / "warm-cache-io-audit.json", summary["warm_cache_io_audit"])
        breakdown_csv(summary["cases"], root / "rank-breakdown.csv")
        base.dump(root / "analysis.json", summary)
        (root / "report-zh.md").write_text(report_text(summary))
        print(json.dumps(dict(refreshed=True, rank_cases=summary["rank_cases"], raw_traces_reread=False)))
        return summary
    cases, errors, cells = recover_from_csv(root, args.workers) if args.recover_csv else ([], [], [])
    started = time.monotonic()
    options = dict(expected_repeats=EXPECTED_REPEATS, floor_s=FLOOR_S, fix1=FIX1)
    tasks = [(str(root), model, ranks, fmt, options, args.resume_cache)
             for model in ("1b", "3b") for ranks in rank_values for fmt in ("dcp", "safetensors")]

    def record(result):
        cells.append(result["cell"])
        cases.extend(result["cases"])
        errors.extend(result["errors"])
        cell = result["cell"]
        state = dict(cells_completed=len(cells), rank_cases=len(cases), errors=len(errors),
                     cell=f"{cell['model']}-{cell['ranks']}r-{cell['checkpoint_format']}",
                     elapsed_s=time.monotonic() - started,
                     workers=min(args.workers, 3) if FIX1 else 1,
                     peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
        base.dump(root / "analysis-progress.json", state)
        print(json.dumps(state), flush=True)

    if not args.recover_csv:
        if FIX1 and args.workers > 1:
            with ProcessPoolExecutor(max_workers=min(args.workers, 3)) as pool:
                futures = [pool.submit(inspect_cell_task, task) for task in tasks]
                for future in as_completed(futures):
                    record(future.result())
        else:
            for task in tasks:
                record(inspect_cell_task(task))
    cells.sort(key=lambda c: (c["model"], c["ranks"], c["checkpoint_format"]))
    summary = dict(
        schema_version=1, root=str(root), rank_cases=len(cases), errors=errors, cells=cells,
        methodology=dict(wall="maximum rank-local dcp_wall_seconds", delta="median paired (Meta - CPU)",
                         percent_denominator="median paired CPU seconds", expected_repeats=EXPECTED_REPEATS,
                         timeline_origin="earliest dcp_started_ns among ranks", bin_ms=50,
                         save_excludes="global union of all ranks' measured fsync intervals", quantiles=list(QUANTILES),
                         time_gate_percent=3, shape_gate="strict 3%; floor max(3% of median CPU, 100 ms)" if FIX1 else "max(3% of median CPU quantile, 50 ms)",
                         fix1=FIX1, floor_s=FLOOR_S, previous_root=str(args.previous_root.resolve()),
                         representative_repeat="paired mean compressed duration nearest its median"),
        cases=cases,
        plots=[])
    for name, key in (("sink-selection.json", "sink_selection"), ("progress.json", "progress")):
        if (root / name).is_file():
            summary[key] = base.read_json(root / name)
    if FIX1:
        summary["invalid_warm_request_audit"] = audit_invalid_warm_attempts(root, cases)
        apply_invalid_sequence_evidence(cells, summary["invalid_warm_request_audit"])
        base.dump(root / "invalid-warm-request-audit.json", summary["invalid_warm_request_audit"])
    summary["save_gap_evidence"] = [str(p) for p in sorted(root.glob("*save*gap*.json"))]
    attach_focused_diagnoses(summary, root)
    add_sink_bounds(cases)
    summary["warm_cache_io_audit"] = warm_cache_io_audit(cases)
    base.dump(root / "warm-cache-io-audit.json", summary["warm_cache_io_audit"])
    breakdown_csv(cases, root / "rank-breakdown.csv")
    base.dump(root / "final-timing-sink-audit.json", final_runtime_audit(cases))
    save_then_plot(summary, root, args.no_plots)
    print(json.dumps(dict(analysis=str(root / "analysis.json"), rank_cases=len(cases), plots=len(summary["plots"]), errors=len(errors))))
    return summary


if __name__ == "__main__":
    main()
