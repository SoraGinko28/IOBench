"""Compare per-rank checkpoint requests and timing from any gdtensor run directory.

Reads reports/raw traces only. Writes fresh exports, analysis.json and report.md
under --output; checkpoint payloads are never opened. Multiple run directories
can be compared together, including saved baseline runs.
"""

import argparse
import hashlib
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metasl.trace_export import (  # noqa: E402
    compare_prepared_rank_traces,
    export_rank_trace,
    prepare_rank_trace,
)

PHASES = ("save", "cold-load", "warm-load", "load")
ALIASES = {"cuda": "gpu", "pageable": "gpu", "meta-sim-nt": "meta-sim"}
BIN_NS = 50_000_000


def read_json(path):
    return json.loads(Path(path).read_text())


def read_jsonl(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def merged(intervals):
    result = []
    for start, end in sorted((a, b) for a, b in intervals if b > a):
        if result and start <= result[-1][1]:
            result[-1] = (result[-1][0], max(end, result[-1][1]))
        else:
            result.append((start, end))
    return result


def union_ns(intervals):
    return sum(b - a for a, b in merged(intervals))


def clipped(events, start, end):
    return [(max(start, e["start_ns"]), min(end, e["start_ns"] + e["duration_ns"])) for e in events]


def summarize(values):
    values = list(values)
    return dict(median=statistics.median(values), samples=values, minimum=min(values), maximum=max(values))


def metadata(output, report, row=None):
    """Recognize both retained runner layouts without requiring their manifests."""
    output = Path(output).resolve()
    result = dict(row or {})
    phase_match = re.match(r"^(cold-load|warm-load|save|load)(?:-(.+))?$", output.name)
    phase = result.get("phase") or (phase_match[1] if phase_match else report.get("action"))
    config = result.get("config") or result.get("device")
    repeat = result.get("repeat", 1)
    model = result.get("model")
    for parent in (output, *output.parents):
        case_match = re.fullmatch(r"r(\d+)-(.+)", parent.name)
        if case_match:
            repeat = int(case_match[1])
            config = config or case_match[2]
            model = model or parent.parent.name
            break
        matrix_match = re.fullmatch(r"(.+)-r\d+", parent.name)
        if matrix_match:
            model = model or matrix_match[1]
            break
    config = config or (phase_match[2] if phase_match else None) or report.get("state_device", "unknown")
    config = ALIASES.get(str(config).split(":")[0], config)
    result.update(
        output=str(output), model=model or "unknown", config=config, repeat=int(repeat), phase=phase,
        checkpoint_format=report.get("checkpoint_format", "dcp"),
        rank=report.get("rank", 0), world_size=report.get("world_size", 1),
        pattern=report.get("pattern", report.get("data_pattern", result.get("pattern", "unknown"))),
        location=result.get("location", "unknown"),
    )
    return result


def discover_rows(run, *, models=None, repeats=None):
    """Yield one row per measured rank, deduplicating reports across manifests."""
    run = Path(run).resolve()
    indexed = {}
    manifests = list(run.rglob("measurement-results.json")) + list(run.rglob("results.json"))
    for manifest in manifests:
        data = read_json(manifest)
        if not isinstance(data, list):
            continue
        for row in data:
            if isinstance(row, dict) and row.get("output"):
                output = Path(row["output"])
                if not output.is_absolute():
                    candidate = Path.cwd() / output
                    output = candidate if candidate.exists() else manifest.parent / output
                indexed[output.resolve()] = row
    for report_path in sorted(run.rglob("rank-*.json")):
        if not re.fullmatch(r"rank-\d+\.json", report_path.name):
            continue
        if not report_path.with_suffix(".io.jsonl").exists():
            continue
        hint = metadata(report_path.parent, {}, indexed.get(report_path.parent))
        if models and hint["model"] != "unknown" and hint["model"] not in models or repeats and hint["repeat"] not in repeats:
            continue
        report = read_json(report_path)
        if "stats" not in report or "dcp_started_ns" not in report["stats"]:
            continue
        row = metadata(report_path.parent, report, indexed.get(report_path.parent))
        row["run"] = str(run)
        row["report"] = str(report_path)
        yield row


def select_payload_exports(payload, exported_rows, raw_origin, export_origin):
    """Use geometry and exact clocks to distinguish DCP payload from header patches."""
    def identity(event, origin, path_key):
        return (event[path_key], event["op"], event["offset"], event["requested"], event["completed"], origin + event["start_ns"])

    remaining = Counter(identity(e, raw_origin, "file") for e in payload)
    selected = []
    for event in exported_rows:
        key = identity(event, export_origin, "path")
        if event.get("source") == "payload_api" and remaining[key]:
            selected.append(event)
            remaining[key] -= 1
    if any(remaining.values()):
        raise ValueError("Raw payload events are missing from the measured export")
    return selected


def inspect_case(row, analysis_dir, rank=0):
    """Measure the complete checkpoint call, including publication/header IO."""
    output = Path(row["output"])
    rank = row.get("rank", rank)
    report_path = Path(row.get("report", output / f"rank-{rank:03d}.json"))
    raw_path = report_path.with_suffix(".io.jsonl")
    report = read_json(report_path)
    row = metadata(output, report, row)
    origin = report["io"]["origin_ns"]
    events = [dict(e, start_ns=e["start_ns"] + e.get("origin_ns", origin) - origin) for e in read_jsonl(raw_path)]
    stats = report["stats"]
    start, end = stats["dcp_started_ns"] - origin, stats["dcp_ended_ns"] - origin
    total = stats["dcp_wall_seconds"]
    if not start < end or abs((end - start) / 1e9 - total) > 0.001:
        raise ValueError(f"Inconsistent checkpoint bounds: {report_path}")
    save = row["phase"] == "save"
    ops = {"pwrite", "write"} if save else {"pread", "read", "readinto"}
    payload = sorted((e for e in events if e.get("io_layer") == "payload" and e["op"] in ops), key=lambda e: e["start_ns"])
    # Earlier DCP traces kept absolute payload events only in the rank report.
    if not payload:
        payload = [dict(e, start_ns=e.get("absolute_start_ns", e["start_ns"]) - origin) for e in report.get("payload_events", []) if e["op"] in ops]
    d2h = [e for e in events if e.get("io_layer") == "staging" and e["op"] == "d2h"]
    h2d = [e for e in events if e.get("io_layer") == "staging" and e["op"] == "h2d"]
    fsync = [e for e in events if e["op"] == "fsync"]
    if row["config"] == "meta" and any(e.get("timing") == "simulated" for e in d2h + h2d):
        row["config"] = "meta-sim"
    if not payload or any(not e.get("success", True) or e["requested"] != e["completed"] for e in payload + d2h + h2d):
        raise ValueError(f"Missing or incomplete payload/transfer events: {report_path}")
    if any(e["start_ns"] < start or e["start_ns"] + e["duration_ns"] > end for e in payload + d2h + h2d):
        raise ValueError(f"Payload/transfer outside checkpoint bounds: {report_path}")
    payload_bytes = sum(e["completed"] for e in payload)
    expected = report.get(f"gdtensor_payload_{'write' if save else 'read'}_bytes")
    if expected is not None and expected != payload_bytes:
        raise ValueError(f"Payload bytes disagree with report: {report_path}")
    case_id = hashlib.sha256(str(output.resolve()).encode()).hexdigest()[:16]
    export_dir = Path(analysis_dir) / "exports" / case_id
    exported = export_rank_trace(raw_path, export_dir, rank=rank, phase="save" if save else "load", origin_ns=origin, payload_events=report.get("payload_events", []) + report.get("container_patch_events", []))
    exported_rows = read_jsonl(exported["output_path"])
    selected = select_payload_exports(payload, exported_rows, origin, exported["first_event_absolute_ns"])
    payload_export = export_dir / f"rank-{rank:05d}.payload.jsonl"
    payload_export.write_text("".join(json.dumps(dict(e, seq=i), sort_keys=True) + "\n" for i, e in enumerate(selected)))
    requests = [e for e in exported_rows if e["request"]]
    measure = lambda selected: union_ns(clipped(selected, start, end)) / 1e9
    metrics = dict(
        total_s=total, total_without_fsync_s=total - measure(fsync), fsync_s=measure(fsync),
        d2h_s=measure(d2h), h2d_s=measure(h2d), payload_io_s=measure(payload),
        pwrite_s=measure([e for e in payload if e["op"] == "pwrite"]),
        pread_s=measure([e for e in payload if e["op"] in {"pread", "read", "readinto"}]),
        payload_requests=len(payload), payload_bytes=payload_bytes,
        request_count=len(requests), request_bytes=sum(e["completed"] for e in requests if e["op"] != "seek"),
        request_operations=dict(Counter(e["op"] for e in requests)),
        d2h_count=len(d2h), h2d_count=len(h2d),
    )
    return dict(row=row, metrics=metrics, payload=payload, fsync=fsync, start_ns=start, end_ns=end,
                export=exported["output_path"], payload_export=str(payload_export),
                provenance=dict(report_sha256=sha256(report_path), trace_sha256=exported["raw_source"]["sha256"], export_sha256=exported["output_sha256"]))


def bin_payload(case):
    """Prorate measured payload bytes into 50 ms bins; save axes remove fsync unions."""
    start, end = case["start_ns"], case["end_ns"]
    intervals = merged(clipped(case["fsync"], start, end)) if case["row"]["phase"] == "save" else []

    def transform(t):
        return t - start - sum(max(0, min(b, t) - a) for a, b in intervals if a < t)

    duration = transform(end)
    values = [0.0] * max(1, math.ceil(duration / BIN_NS))
    for event in case["payload"]:
        a, b = transform(event["start_ns"]), transform(event["start_ns"] + event["duration_ns"])
        if b - a != event["duration_ns"]:
            raise ValueError("Payload overlaps fsync; cannot remove fsync without distorting IO")
        if a == b:
            values[min(a // BIN_NS, len(values) - 1)] += event["completed"]
        else:
            for index in range(a // BIN_NS, (b - 1) // BIN_NS + 1):
                overlap = min(b, (index + 1) * BIN_NS) - max(a, index * BIN_NS)
                values[index] += event["completed"] * overlap / (b - a)
    if not math.isclose(sum(values), case["metrics"]["payload_bytes"], rel_tol=1e-12, abs_tol=0.01):
        raise ValueError("Plot bins lost payload bytes")
    return dict(duration_s=duration / 1e9, binned_bytes_per_50ms=values, binned_bytes=sum(values))


def group_key(row):
    return tuple(row.get(k) for k in ("model", "checkpoint_format", "world_size", "pattern", "location", "phase"))


def compare_cases(cases):
    """Compare one common reference per group/rank; never compare cross-rank order."""
    groups = defaultdict(list)
    for case in cases:
        groups[group_key(case["row"])].append(case)
    results = []
    for key, selected in sorted(groups.items(), key=lambda item: str(item[0])):
        world_size = key[2]
        scopes = {}
        for scope, field in (("full", "export"), ("payload", "payload_export")):
            variants = defaultdict(dict)
            for case in selected:
                row = case["row"]
                label = f"{row.get('run', '')}:{row['config']}:r{row['repeat']}:{Path(row['output']).name}"
                if row["rank"] in variants[label]:
                    raise ValueError(f"Duplicate rank in comparison variant {label}")
                variants[label][row["rank"]] = case[field]
            expected = set(range(world_size))
            completeness = {label: dict(present=sorted(ranks), missing=sorted(expected - ranks.keys()), extra=sorted(ranks.keys() - expected)) for label, ranks in variants.items()}
            errors = [f"{label}: incomplete ranks" for label, ranks in variants.items() if set(ranks) != expected]
            if len(variants) < 2:
                errors.append("At least two variants are required")
            comparisons = []
            labels = list(variants)
            for rank in sorted(expected):
                reference = None
                for label in labels:
                    if rank not in variants[label]:
                        continue
                    try:
                        prepared = prepare_rank_trace(variants[label][rank])
                        if prepared["rank"] != rank:
                            raise ValueError("Event rank differs from report")
                        if label == labels[0]:
                            reference = prepared
                            continue
                        if reference is None:
                            raise ValueError("Missing reference rank")
                        comparison = compare_prepared_rank_traces(reference, prepared)
                    except (OSError, ValueError) as error:
                        errors.append(f"{label} rank {rank}: {error}")
                        comparison = dict(status="unknown", rank=rank, error=str(error))
                    comparisons.append(dict(left_device=labels[0], right_device=label, **comparison))
            statuses = {c["status"] for c in comparisons}
            status = "unknown" if errors or not comparisons or "unknown" in statuses else "different" if "different" in statuses else "identical"
            scopes[scope] = dict(status=status, rank_completeness=completeness, comparisons=comparisons, errors=errors)
        results.append(dict(group=dict(zip(("model", "checkpoint_format", "world_size", "pattern", "location", "phase"), key)), **scopes))
    return results


def paired_differences(cases):
    groups = defaultdict(dict)
    for case in cases:
        row = case["row"]
        key = group_key(row) + (row["rank"], row.get("run", ""), row["repeat"])
        groups[key][row["config"]] = case["metrics"]
    values = defaultdict(list)
    for key, configs in groups.items():
        if "meta-sim" not in configs or "gpu" not in configs:
            continue
        metric = "total_without_fsync_s" if key[5] == "save" else "total_s"
        simulated, real = configs["meta-sim"][metric], configs["gpu"][metric]
        values[key[:7] + (metric,)].append(dict(repeat=key[-1], seconds=simulated - real, percent=(simulated / real - 1) * 100 if real else None))
    return [dict(group=dict(zip(("model", "checkpoint_format", "world_size", "pattern", "location", "phase", "rank", "metric"), key)), seconds=summarize(v["seconds"] for v in samples), percent=summarize(v["percent"] for v in samples if v["percent"] is not None), paired_samples=samples) for key, samples in sorted(values.items(), key=lambda item: str(item[0]))]


def report_text(analysis):
    lines = ["# gdtensor trace analysis", "", "Times are rank-local seconds. Save excludes the union of fsync spans; loads use the complete checkpoint call. Payload/header requests are compared separately. No cross-rank scheduling order is compared.", "", "|Model / format / rank|Config|Phase|Total|Save excl. fsync|D2H|H2D|pwrite|Read|fsync|", "|---|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case in analysis["cases"]:
        row, metrics = case["row"], case["metrics"]
        values = [metrics[name] for name in ("total_s", "total_without_fsync_s", "d2h_s", "h2d_s", "pwrite_s", "pread_s", "fsync_s")]
        lines.append(f"|{row['model']} / {row['checkpoint_format']} / {row['rank']}|{row['config']} r{row['repeat']}|{row['phase']}|" + "|".join(f"{v:.4f}" for v in values) + "|")
    lines += ["", "|Model / format / ranks|Phase|Full sequence|Payload sequence|", "|---|---|---|---|"]
    for sequence in analysis["sequences"]:
        group = sequence["group"]
        lines.append(f"|{group['model']} / {group['checkpoint_format']} / {group['world_size']}|{group['phase']}|{sequence['full']['status']}|{sequence['payload']['status']}|")
    lines += ["", "Meta+sim minus GPU (paired repeats, per rank):", "", "|Model / format / rank|Phase|Median seconds|Median percent|", "|---|---|---:|---:|"]
    for delta in analysis["paired_differences"]:
        group = delta["group"]
        lines.append(f"|{group['model']} / {group['checkpoint_format']} / {group['rank']}|{group['phase']}|{delta['seconds']['median']:+.4f}|{delta['percent']['median']:+.2f}%|")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", nargs="+", type=Path, help="Run roots containing rank reports and raw IO JSONL")
    parser.add_argument("--output", type=Path, required=True, help="Directory for analysis and fresh exports")
    parser.add_argument("--models", nargs="+", help="Optional model labels, e.g. 1b 3b")
    parser.add_argument("--configs", nargs="+", help="Optional config labels: meta meta-sim cpu gpu")
    parser.add_argument("--repeats", nargs="+", type=int, help="Optional repeat numbers")
    args = parser.parse_args(argv)
    cases, seen = [], set()
    for run in args.run:
        for row in discover_rows(run, models=args.models, repeats=args.repeats):
            if row["report"] in seen or args.models and row["model"] not in args.models or args.repeats and row["repeat"] not in args.repeats:
                continue
            seen.add(row["report"])
            case = inspect_case(row, args.output)
            if args.configs and case["row"]["config"] not in args.configs:
                continue
            case["bins"] = bin_payload(case)
            for field in ("payload", "fsync", "start_ns", "end_ns"):
                case.pop(field)
            cases.append(case)
    if not cases:
        raise ValueError("No matching rank reports with raw traces")
    analysis = dict(schema_version=1, runs=[str(p.resolve()) for p in args.run], sequences=compare_cases(cases), paired_differences=paired_differences(cases), cases=[{k: c[k] for k in ("row", "metrics", "provenance", "export", "payload_export", "bins")} for c in cases])
    dump(args.output / "analysis.json", analysis)
    (args.output / "report.md").write_text(report_text(analysis))
    print(json.dumps(dict(analysis=str(args.output / "analysis.json"), cases=len(cases), groups=len(analysis["sequences"]))))


if __name__ == "__main__":
    main()
