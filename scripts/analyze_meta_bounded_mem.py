"""Analyze the bounded-memory sweep using only the owner's acceptance gates.

Request identity covers every ordinary file request in invocation order for
each rank, phase, and paired repetition. Wall time is the largest rank's full
checkpoint call (including save fsync). Shape metrics are deliberately absent.
The script never reads checkpoint payloads or mutates previous validations.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "validation/meta-bounded-mem-20261030"
PHASES = ("save", "cold-load", "warm-load")
CONFIGS = ("meta", "cpu")
SIZES_MIB = (64, 256, 1024)
WARM_BLOCK_READ_TOLERANCE = 64 * 1024
GIB = 1024**3
ANALYSIS_VERSION = 3


def read_json(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024**2), b""):
            result.update(chunk)
    return result.hexdigest()


def median(values):
    values = list(values)
    return statistics.median(values) if values else None


def paired_summary(meta, cpu, expected_repeats):
    repeats = sorted(set(meta) & set(cpu))
    samples = [
        dict(repeat=r, meta_s=meta[r], cpu_s=cpu[r], delta_s=meta[r] - cpu[r]) for r in repeats
    ]
    baseline = median(cpu[r] for r in repeats)
    delta = median(s["delta_s"] for s in samples)
    complete = set(meta) == set(cpu) == set(range(1, expected_repeats + 1))
    tolerance = max(0.03 * baseline, 0.1) if baseline is not None else None
    return dict(
        status="unknown" if not complete else "pass" if abs(delta) <= tolerance else "fail",
        complete=complete,
        expected_repeats=expected_repeats,
        meta_median_s=median(meta[r] for r in repeats),
        cpu_median_s=baseline,
        paired_median_delta_s=delta,
        paired_median_delta_percent=delta / baseline * 100 if baseline else None,
        tolerance_s=tolerance,
        samples=samples,
    )


def expected_repetitions(run, rows):
    for name in ("cell-config.json", "progress.json"):
        path = run / name
        if path.is_file():
            config = read_json(path)
            for field in ("expected_repeats", "repeats"):
                if type(config.get(field)) is int:
                    return config[field]
    if "confirm" in run.parts:
        return 3 if rows and rows[0].get("checkpoint_format") == "dcp" else 5
    return 3


def verified_trace(row):
    """Reuse exact signatures only after checking raw and exported file hashes."""
    exported = row["export"]
    manifest = read_json(exported["manifest_path"])
    rank = row["rank"]
    if manifest.get("rank") != rank or not manifest.get("successful"):
        raise ValueError("trace rank/success evidence invalid")
    if not manifest.get("clock_order", {}).get("serial_rank_order_verifiable"):
        raise ValueError("serial rank request order unverifiable")
    signature = manifest.get("pattern_sha256", "")
    if not re.fullmatch(r"[0-9a-f]{64}", signature):
        raise ValueError("request signature missing")
    count = manifest.get("request_event_count")
    if type(count) is not int or count < 1:
        raise ValueError("request count missing")
    raw = manifest["raw_source"]
    if sha256(raw["path"]) != raw["sha256"]:
        raise ValueError("raw trace SHA256 changed")
    if sha256(manifest["output_path"]) != manifest["output_sha256"]:
        raise ValueError("exported trace SHA256 changed")
    if exported.get("pattern_sha256") != signature or exported.get("request_event_count") != count:
        raise ValueError("measurement manifest and rank export disagree")
    return dict(
        rank=rank,
        request_event_count=count,
        pattern_sha256=signature,
        raw_trace_sha256=raw["sha256"],
        export_sha256=manifest["output_sha256"],
        manifest_path=exported["manifest_path"],
    )


def warm_check(report, row):
    before = report.get("cache_before_call", {})
    residency = before.get("residency", {})
    percent = residency.get("data_resident_percent")
    read_bytes = report.get("stats", {}).get("proc_io_delta", {}).get("read_bytes")
    prepared = before.get("after_resident_buffer_preparation") is True
    before_timer = before.get("probe_end_ns", math.inf) <= report["stats"]["dcp_started_ns"]
    reasons = []
    if percent is None or percent < 99 or residency.get("fully_warm_data_residency") is not True:
        reasons.append("checkpoint residency below 99% or missing")
    if not prepared or not before_timer:
        reasons.append("residency not verified after buffer preparation and before timing")
    if read_bytes is None or read_bytes > WARM_BLOCK_READ_TOLERANCE:
        reasons.append("timed block reads exceed 64 KiB per rank or missing")
    if row.get("warm_attempt", 1) not in (1, 2):
        reasons.append("more than one warm retry")
    return dict(
        rank=row["rank"],
        repeat=row["repeat"],
        config=row["config"],
        output=row["output"],
        valid=not reasons,
        reasons=reasons,
        residency_percent=percent,
        block_read_bytes=read_bytes,
        after_resident_buffer_preparation=prepared,
        before_timer=before_timer,
        warm_attempt=row.get("warm_attempt", 1),
    )


def report_buffer(report):
    source = report.get("meta_save_source") or {}
    sink = report.get("meta_load_sink") or {}
    configurations = [value for value in (source, sink) if value]
    # Reports normally have one direction. A shared object must count once if
    # a future combined report retains both source and sink configurations.
    return max((value.get("resident_buffer_bytes", 0) for value in configurations), default=0)


def memory_report(report):
    """Discard potentially large payload-event arrays before retaining reports."""
    fields = ("rank", "pid", "stats", "meta_save_source", "meta_load_sink", "gdtensor_buffer_bytes")
    return {field: report[field] for field in fields if field in report}


def memory_run(path, reports, *, accepted=True):
    """Peak of simultaneous rank PSS, not the sum of independent rank peaks."""
    samples = []
    with path.open() as stream:
        for line in stream:
            if line.strip():
                samples.append(json.loads(line))
    ranks_by_pid = {str(report["pid"]): report["rank"] for report in reports}
    sampled = {str(report["rank"]): 0 for report in reports}
    for sample in samples:
        for pid, worker in sample.get("workers", {}).items():
            if pid in ranks_by_pid:
                rank = str(ranks_by_pid[pid])
                sampled[rank] = max(sampled[rank], worker.get("rss", 0))
    rank_memory = []
    for report in sorted(reports, key=lambda value: value["rank"]):
        stats, rank = report.get("stats", {}), str(report["rank"])
        measured = max(
            sampled[rank],
            stats.get("sampled_peak_rss_bytes", 0),
            stats.get("rss_start_bytes", 0),
            stats.get("rss_end_bytes", 0),
        )
        high_water = stats.get("process_high_water_rss_bytes", 0)
        rank_memory.append(
            dict(
                rank=report["rank"],
                sampled_peak_rss_bytes=measured,
                process_high_water_rss_bytes=high_water,
                peak_rss_bytes=max(measured, high_water),
                meta_buffer_bytes=report_buffer(report),
                gdtensor_staging_bytes=report.get("gdtensor_buffer_bytes"),
            )
        )
    return dict(
        output=str(path.parent),
        accepted=accepted,
        sample_count=len(samples),
        peak_worker_pss_bytes=max((s.get("worker_pss_bytes", 0) for s in samples), default=0),
        peak_total_pss_bytes=max((s.get("total_pss_bytes", 0) for s in samples), default=0),
        peak_worker_rss_bytes=max((s.get("worker_rss_bytes", 0) for s in samples), default=0),
        meta_buffer_bytes=sum(r["meta_buffer_bytes"] for r in rank_memory),
        ranks=rank_memory,
        memory_path=str(path),
    )


def memory_summary(records):
    result = {}
    for config in CONFIGS:
        selected = [record for record in records if record["config"] == config]
        result[config] = dict(
            peak_worker_pss_bytes=max((r["peak_worker_pss_bytes"] for r in selected), default=0),
            peak_total_pss_bytes=max((r["peak_total_pss_bytes"] for r in selected), default=0),
            peak_meta_buffer_bytes=max((r["meta_buffer_bytes"] for r in selected), default=0),
            per_rank_peak_rss_bytes={
                str(rank): max(
                    (
                        r["peak_rss_bytes"]
                        for sample in selected
                        for r in sample["ranks"]
                        if r["rank"] == rank
                    ),
                    default=0,
                )
                for rank in sorted({r["rank"] for sample in selected for r in sample["ranks"]})
            },
            run_count=len(selected),
        )
    result["records"] = records
    result["scope"] = (
        "Primary PSS is the peak simultaneous sum for all rank workers; total PSS also includes "
        "controller/helpers. RSS retains sampled peak and process high water separately. "
        "Peaks cover the whole process lifetime, including untimed preparation; "
        "rejected warm attempts are retained."
    )
    return result


def rejected_warm_checks(run, indexed, accepted_outputs):
    checks, warm, errors = [], [], []
    for path in sorted((run / "measurements").glob("*/r*-*/warm-load-attempts.json")):
        match = re.fullmatch(r"r(\d+)-(meta|cpu)", path.parent.name)
        if not match:
            continue
        repeat = int(match[1])
        attempts = read_json(path)
        for attempt in attempts.get("attempts", []):
            for original in attempt.get("rows", []):
                row = dict(original, repeat=repeat, warm_attempt=attempt["attempt"])
                if str(Path(row["output"]).resolve()) in accepted_outputs:
                    continue
                config, rank = row["config"], row["rank"]
                other = "cpu" if config == "meta" else "meta"
                comparison = dict(
                    phase="warm-load",
                    repeat=repeat,
                    rank=rank,
                    accepted=False,
                    warm_attempt=attempt["attempt"],
                    rejected_config=config,
                    output=row["output"],
                )
                try:
                    trace = verified_trace(row)
                    peer = indexed.get(("warm-load", repeat, other, rank))
                    identical = peer and all(
                        trace[field] == peer["trace"][field]
                        for field in ("request_event_count", "pattern_sha256")
                    )
                    comparison.update(
                        status="identical" if identical else "different" if peer else "unknown",
                        traces={config: trace, **({other: peer["trace"]} if peer else {})},
                    )
                    report = read_json(Path(row["output"]) / f"rank-{rank:03d}.json")
                    warm.append(dict(warm_check(report, row), accepted=False))
                except (OSError, ValueError, KeyError, TypeError) as error:
                    comparison.update(status="unknown", error=str(error))
                    errors.append(
                        dict(output=row["output"], error=f"Rejected warm request audit: {error}")
                    )
                checks.append(comparison)
    return checks, warm, errors


def resolve_missing_warm_requests(checks, rejected_checks):
    """Rejected cache timings can still prove the actual file request stream."""
    for check in checks:
        if check["phase"] != "warm-load" or check["status"] != "unknown":
            continue
        missing = set(CONFIGS) - set(check["traces"])
        evidence = [
            rejected
            for rejected in rejected_checks
            if rejected["repeat"] == check["repeat"]
            and rejected["rank"] == check["rank"]
            and rejected["rejected_config"] in missing
        ]
        covered = {item["rejected_config"] for item in evidence if item["status"] == "identical"}
        if (
            missing
            and missing <= covered
            and all(item["status"] == "identical" for item in evidence)
        ):
            check.update(
                status="identical",
                resolved_from_rejected_attempts=True,
                rejected_attempt_evidence=evidence,
                resolution=(
                    "Request identity only, verified against the paired accepted opposite-config "
                    "rank; rejected timing and warm validity remain excluded"
                ),
            )


def warm_attempt_failures(warm, rejected_warm, ranks, repeats):
    failures = []
    for repeat in range(1, repeats + 1):
        for config in CONFIGS:
            accepted = [
                sample
                for sample in warm
                if sample["repeat"] == repeat and sample["config"] == config
            ]
            if {sample["rank"] for sample in accepted} == set(range(ranks)) and all(
                sample["valid"] for sample in accepted
            ):
                continue
            rejected = [
                sample
                for sample in rejected_warm
                if sample["repeat"] == repeat and sample["config"] == config
            ]
            attempts = []
            for attempt in sorted({sample["warm_attempt"] for sample in rejected}):
                samples = [sample for sample in rejected if sample["warm_attempt"] == attempt]
                percentages = [
                    sample["residency_percent"]
                    for sample in samples
                    if sample["residency_percent"] is not None
                ]
                reads = [
                    sample["block_read_bytes"]
                    for sample in samples
                    if sample["block_read_bytes"] is not None
                ]
                attempts.append(
                    dict(
                        attempt=attempt,
                        rank_count=len(samples),
                        min_residency_percent=min(percentages) if percentages else None,
                        max_block_read_bytes=max(reads) if reads else None,
                        valid=all(sample["valid"] for sample in samples),
                    )
                )
            two_block_failures = {item["attempt"] for item in attempts} == {1, 2} and all(
                item["max_block_read_bytes"] is not None
                and item["max_block_read_bytes"] > WARM_BLOCK_READ_TOLERANCE
                for item in attempts
            )
            resident = [item["min_residency_percent"] for item in attempts]
            if two_block_failures and all(value is not None and value >= 99 for value in resident):
                detail = "; ".join(
                    f"{'first' if item['attempt'] == 1 else 'redo'} max "
                    f"{item['max_block_read_bytes'] / 1024**2:.2f} MiB/rank"
                    for item in attempts
                )
                cause = (
                    f"{config.upper()} r{repeat} warm load: both permitted attempts exceeded "
                    f"64 KiB timed block reads per rank despite {min(resident):g}% checkpoint "
                    f"residency ({detail}); no accepted timing and no further retry"
                )
            else:
                cause = (
                    f"{config.upper()} r{repeat} warm load: missing or invalid accepted rank "
                    "evidence after permitted attempts"
                )
            failures.append(dict(config=config, repeat=repeat, attempts=attempts, cause=cause))
    return failures


def analyze_cell(run, *, expected_repeats=None, write=True):
    run = Path(run).resolve()
    rows = read_json(run / "measurement-results.json")
    if not rows:
        raise ValueError(f"No measurements: {run}")
    model, ranks = rows[0]["model"], rows[0]["ranks"]
    fmt = rows[0]["checkpoint_format"]
    repeats = expected_repeats or expected_repetitions(run, rows)
    errors, indexed, reports_by_output = [], {}, defaultdict(dict)
    checks, warm = [], []
    for row in rows:
        identity = row["phase"], row["repeat"], row["config"], row["rank"]
        try:
            if identity in indexed:
                raise ValueError("duplicate accepted phase/repeat/config/rank")
            report = read_json(Path(row["output"]) / f"rank-{row['rank']:03d}.json")
            reports_by_output[str(Path(row["output"]).resolve())][row["rank"]] = memory_report(
                report
            )
            stats = report["stats"]
            wall = stats["dcp_wall_seconds"]
            if not isinstance(wall, (int, float)) or not math.isfinite(wall) or wall <= 0:
                raise ValueError("invalid dcp_wall_seconds")
            if abs((stats["dcp_ended_ns"] - stats["dcp_started_ns"]) / 1e9 - wall) > 0.001:
                raise ValueError("checkpoint bounds and wall time disagree")
            trace = verified_trace(row)
            indexed[identity] = dict(row=row, trace=trace, wall_s=wall)
            if row["phase"] == "warm-load":
                warm.append(warm_check(report, row))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(dict(identity=list(identity), output=row.get("output"), error=str(error)))
    phases = {}
    for phase in PHASES:
        times = {config: {} for config in CONFIGS}
        for repeat in range(1, repeats + 1):
            for config in CONFIGS:
                selected = [indexed.get((phase, repeat, config, rank)) for rank in range(ranks)]
                if all(selected):
                    times[config][repeat] = max(value["wall_s"] for value in selected)
            for rank in range(ranks):
                pair = [indexed.get((phase, repeat, config, rank)) for config in CONFIGS]
                identical = all(pair) and all(
                    pair[0]["trace"][field] == pair[1]["trace"][field]
                    for field in ("request_event_count", "pattern_sha256")
                )
                checks.append(
                    dict(
                        phase=phase,
                        repeat=repeat,
                        rank=rank,
                        status="identical"
                        if identical
                        else "different"
                        if all(pair)
                        else "unknown",
                        traces={
                            config: value["trace"]
                            for config, value in zip(CONFIGS, pair, strict=True)
                            if value
                        },
                    )
                )
        phases[phase] = paired_summary(times["meta"], times["cpu"], repeats)
    accepted_outputs = {str(Path(row["output"]).resolve()) for row in rows}
    rejected_checks, rejected_warm, rejected_errors = rejected_warm_checks(
        run, indexed, accepted_outputs
    )
    resolve_missing_warm_requests(checks, rejected_checks)
    checks.extend(rejected_checks)
    errors.extend(rejected_errors)
    records = []
    for path in sorted((run / "measurements").glob("**/memory.jsonl")):
        output = str(path.parent.resolve())
        accepted = output in reports_by_output
        reports = (
            list(reports_by_output[output].values())
            if accepted
            else [
                memory_report(read_json(p))
                for p in sorted(path.parent.glob("rank-[0-9][0-9][0-9].json"))
            ]
        )
        match = re.fullmatch(r"r(\d+)-(meta|cpu)", path.parent.parent.name)
        if not match:
            continue
        record = memory_run(path, reports, accepted=accepted)
        record.update(repeat=int(match[1]), config=match[2], phase=path.parent.name)
        records.append(record)
    sequence_status = (
        "different"
        if any(c["status"] == "different" for c in checks)
        else "unknown"
        if any(c["status"] != "identical" for c in checks)
        else "identical"
    )
    warm_valid = len(warm) == repeats * 2 * ranks and all(value["valid"] for value in warm)
    timing_pass = all(p["status"] == "pass" for p in phases.values())
    result = dict(
        analysis_version=ANALYSIS_VERSION,
        model=model,
        ranks=ranks,
        checkpoint_format=fmt,
        run=str(run),
        expected_repeats=repeats,
        sequence_status=sequence_status,
        timing_pass=timing_pass,
        overall_pass=sequence_status == "identical" and timing_pass and warm_valid and not errors,
        phases=phases,
        request_comparisons=checks,
        warm_validity=dict(
            all_valid=warm_valid,
            samples=warm,
            rejected_samples=rejected_warm,
            failures=warm_attempt_failures(warm, rejected_warm, ranks, repeats),
            timed_block_read_tolerance_bytes_per_rank=WARM_BLOCK_READ_TOLERANCE,
        ),
        memory=memory_summary(records),
        errors=errors,
        timing_method=(
            "max per-rank dcp_wall_seconds, including save fsync; median paired Meta-minus-CPU "
            "delta; tolerance max(3% median CPU, 0.1 seconds)"
        ),
        request_method=(
            "Exact full ordinary-request signatures per rank, with serial invocation order and "
            "raw/export SHA256 verified; lifecycle events excluded"
        ),
    )
    if write:
        dump(run / "analysis.json", result)
    return result


def cached_cell(run):
    path = Path(run) / "analysis.json"
    cached = read_json(path) if path.is_file() else None
    return (
        cached
        if cached and cached.get("analysis_version") == ANALYSIS_VERSION
        else analyze_cell(run)
    )


def summarize_sweep(root):
    root = Path(root)
    sizes = []
    for size in SIZES_MIB:
        cells, missing = [], []
        for model in ("1b", "3b"):
            for ranks in (4, 8):
                run = root / "sweep" / f"{size}mib" / f"{model}-{ranks}r-safetensors"
                if (run / "measurement-results.json").is_file():
                    cells.append(cached_cell(run))
                else:
                    missing.append(str(run))
        sizes.append(
            dict(
                buffer_size_mib=size,
                overall_pass=not missing and all(c["overall_pass"] for c in cells),
                complete=not missing,
                cells=cells,
                missing=missing,
            )
        )
    chosen = next((s["buffer_size_mib"] for s in sizes if s["overall_pass"]), None)
    result = dict(
        chosen_size_mib=chosen,
        sizes=sizes,
        selection_rule=(
            "Smallest size passing every shared-safetensors request, full phase wall-time, and "
            "warm validity check; memory assessed separately without a new numeric gate"
        ),
    )
    dump(root / "sweep-summary.json", result)
    return result


def markdown_table(columns, rows):
    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
        + ["| " + " | ".join(str(value) for value in row) + " |" for row in rows]
    )


def label(cell):
    return f"{cell['model']}-{cell['ranks']}r-{cell.get('checkpoint_format', cell.get('format'))}"


def fmt(value, precision=3):
    return "—" if value is None else f"{value:.{precision}f}"


def phase_row(size, cell, phase):
    result = cell["phases"][phase]
    warm = cell["warm_validity"]["all_valid"]
    time_status = (
        "invalid/incomplete"
        if phase == "warm-load" and not result["complete"] and not warm
        else result["status"]
    )
    return [
        size,
        label(cell),
        phase,
        fmt(result["meta_median_s"]),
        fmt(result["cpu_median_s"]),
        fmt(result["paired_median_delta_s"]),
        fmt(result["paired_median_delta_percent"], 2),
        time_status,
        cell["sequence_status"],
        "valid" if warm else "invalid",
    ]


def write_csv(path, header, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def plots(root, cells):
    out = root / "plots"
    out.mkdir(exist_ok=True)
    outputs = []
    for cell in cells:
        if (cell["model"], cell["ranks"]) != ("3b", 8):
            continue
        measurements = Path(cell["run"]) / "measurements" / cell["model"]
        for phase in ("save", "cold-load"):
            target = out / f"{label(cell)}-{phase}.png"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts/plot_realtime_io.py"),
                    str(measurements),
                    phase,
                    str(target),
                    f"3B · 8 ranks · {cell['checkpoint_format']} · {phase}",
                ],
                check=True,
            )
            outputs.append(str(target))
    return outputs


def memory_plot(root, rows, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, ax = plt.subplots(figsize=(max(8, len(rows) * 1.65), 4.8), dpi=150)
    for offset, field, text, color in (
        (-0.26, 1, "Faithful Meta (previous run)", "#a6a6a6"),
        (0, 2, "Bounded Meta", "#2a78d6"),
        (0.26, 3, "CPU", "#1baf7a"),
    ):
        ax.bar(
            [index + offset for index in range(len(rows))],
            [row[field] if row[field] is not None else float("nan") for row in rows],
            width=0.25,
            label=text,
            color=color,
        )
    ax.set_xticks(range(len(rows)), [row[0] for row in rows], rotation=15, ha="right")
    ax.set_ylabel("Peak total rank PSS (GiB)")
    ax.set_title(title, loc="left")
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#e5e5e5", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    target = root / "plots/memory-pss.png"
    target.parent.mkdir(exist_ok=True)
    figure.savefig(target)
    plt.close(figure)
    return str(target)


def finalize(root, *, make_plots=True):
    root = Path(root).resolve()
    sweep = summarize_sweep(root)
    confirmations = [
        cached_cell(path)
        for path in sorted((root / "confirm").glob("*"))
        if (path / "measurement-results.json").is_file()
    ]
    sweep_cells = [c for size in sweep["sizes"] for c in size["cells"]]
    selected = next(
        (
            size["cells"]
            for size in sweep["sizes"]
            if size["buffer_size_mib"] == sweep["chosen_size_mib"]
        ),
        [],
    )
    fallback = next((size for size in sweep["sizes"] if size["complete"]), None)
    chart_cells = confirmations or selected or (fallback["cells"] if fallback else [])
    chart_size = sweep["chosen_size_mib"] or (fallback["buffer_size_mib"] if fallback else None)
    chart_stage = "confirmation" if confirmations else "sweep"
    baseline_path = root / "baseline-memory.json"
    baseline = read_json(baseline_path) if baseline_path.is_file() else {}
    baseline = baseline.get("cells", baseline)
    if isinstance(baseline, list):
        baseline = {label(cell): cell for cell in baseline}
    headers = [
        "MiB/rank",
        "Cell",
        "Phase",
        "Meta s",
        "CPU s",
        "Paired Δ s",
        "Paired Δ %",
        "Time",
        "Requests",
        "Warm",
    ]
    sweep_rows = [
        phase_row(size["buffer_size_mib"], cell, phase)
        for size in sweep["sizes"]
        for cell in size["cells"]
        for phase in PHASES
    ]
    confirm_rows = [
        phase_row(sweep["chosen_size_mib"], cell, phase)
        for cell in confirmations
        for phase in PHASES
    ]
    memory_rows, chart_rows = [], []
    memory_cells = [
        ("sweep", size["buffer_size_mib"], cell)
        for size in sweep["sizes"]
        for cell in size["cells"]
    ]
    memory_cells += [("confirm", sweep["chosen_size_mib"], cell) for cell in confirmations]
    for stage, size, cell in memory_cells:
        reference = baseline.get(label(cell), {})
        before = reference.get(
            "faithful_peak_worker_pss_bytes",
            reference.get("configs", {}).get("meta", {}).get("peak_worker_pss_bytes"),
        )
        meta, cpu = [cell["memory"][config] for config in CONFIGS]
        meta_pss, cpu_pss = meta["peak_worker_pss_bytes"], cpu["peak_worker_pss_bytes"]
        if cell in chart_cells:
            chart_rows.append(
                [
                    label(cell),
                    before / GIB if before is not None else None,
                    meta_pss / GIB,
                    cpu_pss / GIB,
                ]
            )
        warm_samples = cell["warm_validity"]["samples"] + cell["warm_validity"].get(
            "rejected_samples", []
        )
        residency = []
        reads = []
        for config in CONFIGS:
            chosen_warm = [sample for sample in warm_samples if sample["config"] == config]
            percentages = [
                sample["residency_percent"]
                for sample in chosen_warm
                if sample["residency_percent"] is not None
            ]
            read_counts = [
                sample["block_read_bytes"]
                for sample in chosen_warm
                if sample["block_read_bytes"] is not None
            ]
            residency.append(fmt(min(percentages) if percentages else None, 2))
            reads.append(fmt(max(read_counts) / 1024 if read_counts else None, 1))
        memory_rows.append(
            [
                stage,
                size,
                label(cell),
                fmt(before / GIB if before is not None else None),
                fmt(meta_pss / GIB),
                fmt(cpu_pss / GIB),
                fmt(meta["peak_meta_buffer_bytes"] / GIB),
                ", ".join(
                    f"r{rank}: {rss / GIB:.3f}"
                    for rank, rss in meta["per_rank_peak_rss_bytes"].items()
                ),
                ", ".join(
                    f"r{rank}: {rss / GIB:.3f}"
                    for rank, rss in cpu["per_rank_peak_rss_bytes"].items()
                ),
                " / ".join(residency),
                " / ".join(reads),
            ]
        )
    memory_headers = [
        "Stage",
        "MiB/rank",
        "Cell",
        "Faithful Meta PSS GiB (before)",
        "Bounded Meta PSS GiB (after)",
        "CPU PSS GiB",
        "Meta buffer GiB (sum ranks)",
        "Meta per-rank peak RSS GiB",
        "CPU per-rank peak RSS GiB",
        "Warm min residency % Meta / CPU (all attempts)",
        "Warm max block reads KiB Meta / CPU (all attempts)",
    ]
    for name, columns, rows in (
        ("sweep-table", headers, sweep_rows),
        ("confirmation-gates", headers, confirm_rows),
        ("memory-table", memory_headers, memory_rows),
    ):
        (root / f"{name}.md").write_text(markdown_table(columns, rows) + "\n")
        write_csv(root / f"{name}.csv", columns, rows)
    failures = []
    for cell in sweep_cells + confirmations:
        location = str(Path(cell["run"]).relative_to(root))
        warm_failures = cell["warm_validity"].get("failures", [])
        for phase, result in cell["phases"].items():
            if result["status"] != "pass":
                if phase == "warm-load" and result["status"] == "unknown" and warm_failures:
                    continue
                failures.append(
                    f"{location}/{phase}: {result['status']}; paired Δ "
                    f"{fmt(result['paired_median_delta_s'], 6)} s exceeds "
                    f"±{fmt(result['tolerance_s'], 6)} s"
                    if result["status"] == "fail"
                    else f"{location}/{phase}: incomplete paired rank evidence"
                )
        if cell["sequence_status"] != "identical":
            failures.append(f"{location}: full request sequence {cell['sequence_status']}")
        if not cell["warm_validity"]["all_valid"]:
            failures.extend(f"{location}: {failure['cause']}" for failure in warm_failures)
            if not warm_failures:
                failures.append(
                    f"{location}: accepted warm evidence failed residency/block-read validity"
                )
        failures.extend(f"{location}: {error['error']}" for error in cell["errors"])
    chosen = sweep["chosen_size_mib"]
    if chosen is None:
        choice = (
            "No buffer size passed every sweep cell. Per the brief, confirmation and default "
            "changes stop after the sweep."
        )
    else:
        choice = (
            f"Chosen buffer: {chosen} MiB per rank, capped at the local shard size. It is the "
            "smallest swept size passing all four shared-safetensors cells."
        )
    small_fast = [
        f"{size['buffer_size_mib']} MiB {label(cell)} {phase}"
        for size in sweep["sizes"]
        for cell in size["cells"]
        for phase, value in cell["phases"].items()
        if value["status"] == "fail"
        and value["paired_median_delta_s"] is not None
        and value["paired_median_delta_s"] < 0
    ]
    explanation = (
        "Smaller reused mappings can reduce cache and TLB pressure compared with CPU's distinct "
        "shard pages. The 64 MiB ring also splits the largest 3B compute ranges into multiple "
        "callbacks, while 256 MiB holds those ranges without splitting. Extra callbacks and "
        "chunk assembly may offset the reuse benefit, so timing need not vary monotonically "
        "with capacity. This is a hypothesis based on the implementation and timings, not "
        "hardware-counter proof. The request signatures verify unchanged file offsets, sizes "
        "and order; the owner's symmetric timing gate decides acceptance."
    )
    if small_fast:
        explanation += " Some bounded phases remain faster than the allowed tolerance and fail."
    plot_paths = []
    if make_plots and chart_cells:
        plot_paths = plots(root, chart_cells)
        plot_paths.append(memory_plot(root, chart_rows, f"{chart_size} MiB/rank · {chart_stage}"))
    expected_confirm = {
        "1b-4r-safetensors",
        "1b-8r-safetensors",
        "3b-4r-safetensors",
        "3b-8r-safetensors",
        "3b-8r-dcp",
    }
    confirmation_complete = {label(cell) for cell in confirmations} == expected_confirm
    result = dict(
        chosen_size_mib=chosen,
        sweep=sweep,
        confirmations=confirmations,
        confirmation_complete=confirmation_complete,
        confirmation_pass=confirmation_complete and all(c["overall_pass"] for c in confirmations),
        failures=failures,
        plots=plot_paths,
    )
    dump(root / "analysis.json", result)
    buffer_bound = (
        "At the selected 64 MiB capacity, total resident Meta ring bytes are exactly "
        "0.25 GiB for four ranks and 0.50 GiB for eight ranks for both 1B and 3B. "
        if chosen == 64
        else "Each rank's resident Meta ring is capped at its configured capacity and local shard. "
    )
    buffer_bound += (
        "Whole-process PSS also includes largest-range executor copies/staging, runtime state, "
        "and trace overhead. Those components vary with the model and rank count, so the "
        "fixed ring bound does not imply exactly flat total PSS."
    )
    report = [
        "# Bounded Meta host memory",
        "",
        choice,
        "",
        explanation,
        "",
        "## Design and files",
        "",
        "`MetaLoadSink` and `MetaSaveSource` reuse one writable, pre-faulted ring per rank for "
        "zero-pattern save/load. Preparation occurs before the checkpoint timer; callback "
        "pieces reuse the mapping without changing file requests. Faithful mode retains the "
        "full-shard reference; position data does not use the zero-pattern ring.",
        "",
        "Changed files: `metasl/gdt_storage.py`, `scripts/run_gdt_safetensors.py`, "
        "`scripts/run_meta_bounded_mem.py`, `scripts/analyze_meta_bounded_mem.py`, `README.md`, "
        "`tests/test_meta_cpu_target_sink.py`, `tests/test_shared_hf_storage.py`, "
        "`tests/test_gdt_measurement_runner.py`, `tests/test_meta_bounded_analysis.py`, "
        "and `tests/test_meta_bounded_runner.py`.",
        "",
        (
            "The runner's automatic zero-pattern CPU-target Meta path selects ring mode at "
            f"{chosen} MiB. Automatic Meta-sim and position-pattern paths select fresh mode. "
            "The low-level fresh default and explicit ring capacity default of 256 MiB stay "
            "unchanged. Historical matrices explicitly selecting faithful remain references."
            if chosen is not None
            else "No bounded runner default was selected because no capacity passed the sweep."
        ),
        "",
        "## Method",
        "",
        "Paired/interleaved Meta versus CPU. The sweep uses three repetitions; confirmation "
        "uses five shared-safetensors repetitions and three DCP repetitions. Save includes "
        "fsync. For each repetition, phase time is the maximum rank's `dcp_wall_seconds`; "
        "the median paired Meta-minus-CPU difference must fall within max(3% of median CPU, "
        "0.1 s). Every rank/phase/repetition must have identical full request signatures. "
        "Warm loads require >=99% residency after buffer preparation and before timing, "
        "with <=64 KiB timed block reads per rank and one retry permitted.",
        "",
        "Validation: 96 focused tests passed: 42 core, 41 runner-default, nine analysis, and "
        "four orchestration checks. Logs and JUnit records are retained in this output.",
        "",
        "## Sweep",
        "",
        markdown_table(headers, sweep_rows),
        "",
        "## Confirmation gates",
        "",
        markdown_table(headers, confirm_rows) if confirm_rows else "Not run: no swept size passed.",
        "",
        "## Memory",
        "",
        markdown_table(memory_headers, memory_rows),
        "",
        buffer_bound,
        "",
        "PSS is the peak simultaneous total across rank workers over the whole run, including "
        "preparation. Controller/helper-inclusive PSS and each rank's sampled RSS/process "
        "high water, Meta buffer bytes, and all warm residency samples are retained in cell "
        "`analysis.json`. Rejected warm attempts remain in memory and residency accounting; "
        "their verified request streams can establish request identity but never supply "
        "accepted phase timing or warm validity. Faithful references come from the previous "
        "shared-safetensors run, with previous fix1 DCP numbers used for DCP; those directories "
        "remain read-only. A missing staging-byte field is reported as unknown, not zero.",
        "",
        "## Failures",
        "",
    ]
    report += ["- " + value for value in failures] or ["None."]
    if plot_paths:
        report += [
            "",
            "## Plots",
            "",
            f"IO and memory plots use {chart_size} MiB/rank {chart_stage} data"
            + (" (the smallest swept capacity; no capacity passed)." if chosen is None else "."),
            "",
        ]
        report += [f"![{Path(path).stem}]({Path(path).relative_to(root)})" for path in plot_paths]
    (root / "report.md").write_text("\n".join(report) + "\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--cell", type=Path)
    parser.add_argument("--sweep-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    result = (
        analyze_cell(args.cell)
        if args.cell
        else summarize_sweep(args.root)
        if args.sweep_only
        else finalize(args.root, make_plots=not args.no_plots)
    )
    print(
        json.dumps(
            {
                field: result[field]
                for field in ("overall_pass", "chosen_size_mib", "confirmation_pass")
                if field in result
            }
        )
    )


if __name__ == "__main__":
    main()
