"""Analyze the save-only concurrent experiment and its immutable serial baseline.

Run only after measurements finish. --plots-only reads durable aggregate bins,
never traces; the CPU plotting environment can therefore be used separately.
"""

import argparse
import csv
import gzip
import hashlib
import json
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_multirank_meta_cpu as core  # noqa: E402

DEFAULT_ROOT = ROOT / "validation/multirank-concurrent-save-20261030"
DEFAULT_SERIAL = ROOT / "validation/multirank-fix1-20261030"
QUANTILES = ("10", "50", "90", "100")


def configure():
    core.PHASES = ("save",)
    core.FIX1 = True
    core.EXPECTED_REPEATS = 5
    core.FLOOR_S = .1


def intersection_ns(left, right):
    """Exact intersection of sorted, disjoint, half-open interval unions."""
    i = j = total = 0
    while i < len(left) and j < len(right):
        a, b = left[i]
        c, d = right[j]
        total += max(0, min(b, d) - max(a, c))
        if b <= d:
            i += 1
        else:
            j += 1
    return total


def concurrency(intervals_by_rank):
    """Count simultaneously active ranks, resolving equal boundaries together."""
    changes = defaultdict(int)
    for intervals in intervals_by_rank.values():
        for a, b in core.base.merged(intervals):
            if a < b:
                changes[a] += 1
                changes[b] -= 1
    active, previous, maximum = 0, None, 0
    histogram = defaultdict(int)
    for t, delta in sorted(changes.items()):
        if previous is not None and active:
            histogram[active] += t - previous
        active += delta
        maximum = max(maximum, active)
        previous = t
    union = sum(histogram.values())
    return dict(max_active_ranks=maximum,
                active_rank_seconds=sum(k * v for k, v in histogram.items()) / 1e9,
                union_s=union / 1e9,
                mean_active_ranks=sum(k * v for k, v in histogram.items()) / union if union else 0,
                multi_rank_union_s=sum(v for k, v in histogram.items() if k >= 2) / 1e9,
                seconds_by_active_ranks={str(k): v / 1e9 for k, v in sorted(histogram.items())})


def interval_diagnostics(records):
    """Per-rank overlap comes from exact API intervals on the common host clock."""
    writes = {r["rank"]: core.base.merged(r["pwrite_absolute_intervals_ns"]) for r in records}
    syncs = {r["rank"]: core.base.merged(r["fsync_absolute_intervals_ns"]) for r in records}
    writers = {r["rank"]: r.get("writer_absolute_intervals_ns", []) for r in records}
    write_union = core.base.merged([s for values in writes.values() for s in values])
    sync_union = core.base.merged([s for values in syncs.values() for s in values])
    per_rank = []
    for record in records:
        rank = record["rank"]
        peer_writes = core.base.merged([s for other, values in writes.items() if other != rank for s in values])
        peer_syncs = core.base.merged([s for other, values in syncs.items() if other != rank for s in values])
        per_rank.append(dict(
            rank=rank, pwrite_union_s=core.base.union_ns(writes[rank]) / 1e9,
            fsync_union_s=core.base.union_ns(syncs[rank]) / 1e9,
            pwrite_overlap_peer_pwrite_s=intersection_ns(writes[rank], peer_writes) / 1e9,
            fsync_overlap_peer_fsync_s=intersection_ns(syncs[rank], peer_syncs) / 1e9,
            pwrite_overlap_peer_fsync_s=intersection_ns(writes[rank], peer_syncs) / 1e9,
            fsync_overlap_peer_pwrite_s=intersection_ns(syncs[rank], peer_writes) / 1e9,
            raw_trace=record["raw_trace"], raw_sha256=record["raw_sha256"]))
    return dict(pwrite=concurrency(writes), fsync=concurrency(syncs), writer=concurrency(writers),
                pwrite_fsync_overlap_union_s=intersection_ns(write_union, sync_union) / 1e9,
                per_rank=per_rank)


def interval_record(case, *, raw=False):
    """Retain exact merged absolute intervals and verify baseline raw provenance."""
    raw_path = Path(case["row"]["report"]).with_suffix(".io.jsonl")
    start, end = case["absolute_start_ns"], case["absolute_end_ns"]
    if raw:
        report = core.base.read_json(Path(case["row"]["report"]))
        origin = report["io"]["origin_ns"]
        selected, digest = [], hashlib.sha256()
        with raw_path.open("rb") as stream:
            for line in stream:
                digest.update(line)
                # Raw traces are JSON with either compact or spaced separators.
                if b'"pwrite"' not in line and b'"fsync"' not in line:
                    continue
                event = json.loads(line)
                if event["op"] == "fsync" or event["op"] == "pwrite" and event.get("io_layer") == "payload":
                    selected.append((event["op"], event.get("origin_ns", origin) + event["start_ns"], event["duration_ns"]))
        actual_hash = digest.hexdigest()
        expected_hash = case["provenance"]["trace_sha256"]
        if actual_hash != expected_hash:
            raise ValueError(f"Serial raw SHA256 mismatch: {raw_path}")
    else:
        origin = case["origin_ns"]
        selected = [(e["op"], origin + e["start_ns"], e["duration_ns"])
                    for e in case["payload"] if e["op"] == "pwrite"]
        selected += [("fsync", origin + e["start_ns"], e["duration_ns"]) for e in case["fsync"]]
        actual_hash = case["provenance"]["trace_sha256"]
    intervals = {op: core.base.merged([(max(start, a), min(end, a + duration))
                                     for name, a, duration in selected if name == op and max(start, a) < min(end, a + duration)])
                 for op in ("pwrite", "fsync")}
    for op, key in (("pwrite", "pwrite_s"), ("fsync", "fsync_s")):
        observed = core.base.union_ns(intervals[op]) / 1e9
        if abs(observed - case["metrics"][key]) > 1e-8:
            raise ValueError(f"{op} intervals disagree with verified rank metrics: {raw_path}")
    timing = case.get("writer_timing", {})
    writer = []
    if timing.get("write_enter_ns") is not None and timing.get("future_completed_ns") is not None:
        writer = [(timing["write_enter_ns"], timing["future_completed_ns"])]
    return dict(rank=case["row"]["rank"], raw_trace=str(raw_path), raw_sha256=actual_hash,
                pwrite_absolute_intervals_ns=intervals["pwrite"],
                fsync_absolute_intervals_ns=intervals["fsync"], writer_absolute_intervals_ns=writer)


def inspect_cell(task):
    """A cell owns its exports/cache; PHASES is set explicitly in every worker."""
    configure()
    root_text, model, ranks, fmt, resume = task
    root = Path(root_text)
    label = f"{model}-{ranks}r-{fmt}"
    cache = root / "analysis-cache" / f"{label}.concurrent-save.json"
    if resume and cache.exists():
        cached = core.base.read_json(cache)
        rewritten = sanitize_concurrent_diagnoses([cached["cell"]])
        enriched = augment_tail_diagnoses(cached["cell"], cached["cases"])
        if rewritten or enriched:
            core.base.dump(cache, cached)
        return cached
    selected, errors = core.discover(root, root, run=root / "matrix" / label)
    cell = core.analyze_cell(model, ranks, fmt, selected)
    sanitize_concurrent_diagnoses([cell])
    augment_tail_diagnoses(cell, selected)
    by_pair = defaultdict(list)
    for case in selected:
        by_pair[(case["row"]["repeat"], case["row"]["config"])].append(interval_record(case))
    overlap = []
    archive = root / "overlap-intervals" / f"concurrent-{label}.jsonl.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(archive, "wt") as stream:
        for (repeat, config), records in sorted(by_pair.items()):
            if sorted(r["rank"] for r in records) != list(range(ranks)):
                continue
            for record in records:
                stream.write(json.dumps(dict(repeat=repeat, config=config, **record), separators=(",", ":")) + "\n")
            overlap.append(dict(repeat=repeat, config=config, **interval_diagnostics(records)))
    retained = [{key: case[key] for key in core.RETAINED_FIELDS} for case in selected]
    result = dict(cell=cell, cases=retained, errors=errors, overlap=overlap,
                  absolute_intervals_archive=str(archive), absolute_intervals_archive_sha256=core.base.sha256(archive))
    core.base.dump(cache, result)
    return result


def serial_overlap_task(task):
    root_text, cell, cases = task
    root = Path(root_text)
    label = cell_label(cell)
    cache = root / "analysis-cache" / f"{label}.serial-overlap.json"
    if cache.exists():
        return core.base.read_json(cache)
    grouped = defaultdict(list)
    for case in cases:
        grouped[(case["row"]["repeat"], case["row"]["config"])].append(interval_record(case, raw=True))
    archive = root / "overlap-intervals" / f"serial-{label}.jsonl.gz"
    archive.parent.mkdir(parents=True, exist_ok=True)
    overlap = []
    with gzip.open(archive, "wt") as stream:
        for (repeat, config), records in sorted(grouped.items()):
            if sorted(r["rank"] for r in records) != list(range(cell["ranks"])):
                raise ValueError(f"Missing baseline ranks: {label}/{repeat}/{config}")
            for record in records:
                stream.write(json.dumps(dict(repeat=repeat, config=config, **record), separators=(",", ":")) + "\n")
            overlap.append(dict(repeat=repeat, config=config, **interval_diagnostics(records)))
    result = dict(overlap=overlap, absolute_intervals_archive=str(archive),
                  absolute_intervals_archive_sha256=core.base.sha256(archive))
    core.base.dump(cache, result)
    return result


def cell_key(cell):
    return cell["model"], cell["ranks"], cell["checkpoint_format"]


def cell_label(cell):
    return f"{cell['model']}-{cell['ranks']}r-{cell['checkpoint_format']}"


def gate_counts(cells):
    result = {}
    for mode in ("strict", "floor"):
        result[mode] = {}
        for kind in ("wall", "shape"):
            values = [c["phases"]["save"]["wall"] for c in cells] if kind == "wall" else [v for c in cells for v in c["phases"]["save"]["shape"]["quantiles"].values()]
            statuses = [v[mode + "_status"] for v in values]
            result[mode][kind] = dict(passed=statuses.count("pass"), failed=statuses.count("fail"),
                                      unknown=statuses.count("unknown"), total=len(statuses))
    return result


def failure_cause(failure):
    """Describe measured intervals without inventing unique function attribution."""
    pieces = []
    if "wall" in failure.get("failed_gates_strict", []):
        pair = failure.get("wall_median_pair") or {}
        selected = {r["config"]: r for r in failure.get("critical_rank_breakdowns", [])
                    if r["repeat"] == pair.get("repeat") and "max_wall" in r["roles"]}
        if set(selected) == {"meta", "cpu"}:
            partitions = {c: r["breakdown"]["disjoint_partition_s"] for c, r in selected.items()}
            differences = {key: partitions["meta"][key] - partitions["cpu"][key] for key in partitions["meta"]}
            aligned = {k: v for k, v in differences.items() if v * pair["delta_s"] > 0}
            key = max(aligned or differences, key=lambda k: abs(differences[k]))
            pieces.append(f"wall中位配对r{pair['repeat']}关键rank Meta/CPU={selected['meta']['rank']}/{selected['cpu']['rank']}，{key} {differences[key] * 1000:+.1f}ms，互斥分区合计 {sum(differences.values()) * 1000:+.1f}ms")
    failed_q = [g for g in failure.get("failed_gates_strict", []) if g.startswith("shape-")]
    if failed_q:
        timeline = failure["timeline_pairs"]
        pieces.append("/".join(f"{g} {timeline['quantiles'][g.split('p')[-1]]['paired_median_delta_ms']:+.1f}ms" for g in failed_q)
                      + f"；首payload {timeline['onset']['paired_median_delta_ms']:+.1f}ms、P10–P90跨度 {timeline['active_p10_p90']['paired_median_delta_ms']:+.1f}ms")
        diagnoses = failure.get("selected_quantile_diagnoses", [])
        tail = next((d for d in diagnoses if d["quantile"] == "100"), diagnoses[-1] if diagnoses else None)
        if tail:
            pieces.append(f"P{tail['quantile']}中位配对r{tail['selected_pair_repeat']}：末payload相对各自本次最早checkpoint的完成Δ{tail['raw_last_payload_delta_s'] * 1000:+.1f}ms，减去此前fsync并集删除量Δ{tail['removed_before_last_payload_delta_s'] * 1000:+.1f}ms，得到压缩末payloadΔ{tail['compressed_last_payload_delta_s'] * 1000:+.1f}ms（该配对P100分bin差{tail['same_pair_p100_delta_s'] * 1000:+.1f}ms）；尾部shape差与跨rank pwrite/fsync重叠及压缩边界差相符")
        else:
            pieces.append("定位为聚合payload进度变化")
    return "；".join(pieces) + "；API spans含内核等待和调度，未唯一归因到设备服务时间。"


def sanitize_concurrent_diagnoses(cells):
    """Correct the shared analyzer's legacy serial-only prose, never its metrics."""
    replacements = {
        "含 fsync 的 wall 差与实测同步刷盘区间变化相符；单 writer 许可覆盖 fsync，因此该时间也传播为其他 rank 的准入/结束等待。":
            "含 fsync 的 wall 差与实测 fsync API 区间变化相符；本轮 writer limit 等于 world size，许可活跃和跨 rank fsync/pwrite 重叠另有绝对区间证据，不能据此推断其他 rank 的准入等待。",
        "变化被定位到单 writer 的实测准入等待；不应把本地 other 当作未记录的 pwrite。":
            "变化被定位到 writer 的实测准入等待；该等待仍受并发许可和内存安全准入约束，不能仅由本地 other 推断 pwrite 耗时。",
    }
    changed = 0

    def rewrite(value):
        nonlocal changed
        for before, after in replacements.items():
            if before in value:
                value = value.replace(before, after)
                changed += 1
        return value

    for cell in cells:
        for failure in cell.get("failures", []):
            for field in ("evidence_zh", "inferences_zh"):
                if field in failure:
                    failure[field] = [rewrite(value) for value in failure[field]]
            for field in ("interpretation_zh", "attribution_limit_zh"):
                if field in failure:
                    failure[field] = rewrite(failure[field])
    return changed


def selected_quantile_diagnosis(cell, quantile):
    """Keep all terms of the compression identity in one selected paired run."""
    metric = cell["phases"]["save"]["shape"]["quantiles"][quantile]
    sample = min(metric["samples"], key=lambda s: (abs(s["delta_s"] - metric["paired_median_delta_s"]), s["repeat"]))
    repeat = sample["repeat"]
    pair = {a["config"]: a["timeline"] for a in cell["aggregates"] if a["phase"] == "save" and a["repeat"] == repeat}
    by_config = {}
    for config, timeline in pair.items():
        by_config[config] = dict(raw_last_payload_s=timeline["last_payload_s"],
                                 compressed_last_payload_s=timeline["last_payload_compressed_s"],
                                 removed_before_last_payload_s=timeline["last_payload_s"] - timeline["last_payload_compressed_s"],
                                 uncompressed_span_s=timeline["uncompressed_span_s"],
                                 global_fsync_union_s=timeline["global_fsync_union_s"],
                                 compressed_span_s=timeline["duration_s"])
    delta = {name + "_delta_s": by_config["meta"][name] - by_config["cpu"][name]
             for name in by_config["meta"]}
    same_p100 = next(s["delta_s"] for s in cell["phases"]["save"]["shape"]["quantiles"]["100"]["samples"] if s["repeat"] == repeat)
    return dict(quantile=quantile, selected_pair_repeat=repeat,
                paired_median_delta_s=metric["paired_median_delta_s"], binned_quantile_delta_s=sample["delta_s"],
                by_config=by_config, **delta, same_pair_p100_delta_s=same_p100,
                p100_bin_interpolation_delta_error_s=same_p100 - delta["compressed_last_payload_s_delta_s"],
                raw_last_payload_delta_s=delta["raw_last_payload_s_delta_s"],
                removed_before_last_payload_delta_s=delta["removed_before_last_payload_s_delta_s"],
                compressed_last_payload_delta_s=delta["compressed_last_payload_s_delta_s"],
                tail_identity_error_s=delta["raw_last_payload_s_delta_s"] - delta["removed_before_last_payload_s_delta_s"] - delta["compressed_last_payload_s_delta_s"],
                span_identity_error_s=delta["uncompressed_span_s_delta_s"] - delta["global_fsync_union_s_delta_s"] - delta["compressed_span_s_delta_s"],
                time_origin="Each run's earliest checkpoint start; no absolute-clock subtraction between Meta and CPU runs",
                attribution="Observed cross-rank pwrite/fsync overlap and global-union compression boundary timing; API spans do not uniquely identify physical device service time")


def augment_tail_diagnoses(cell, cases):
    """Include the true last-payload rank, including on resumed compact caches."""
    if not cell.get("failures") or cell.get("concurrent_tail_diagnosis_version") == 1:
        return False
    groups = defaultdict(list)
    for case in cases:
        if "payload" in case:
            last = max(case["origin_ns"] + event["start_ns"] + event["duration_ns"] for event in case["payload"])
        else:
            report_path = Path(case["row"]["report"])
            encoded = report_path.read_bytes()
            if hashlib.sha256(encoded).hexdigest() != case["provenance"]["report_sha256"]:
                raise ValueError(f"Cached tail report SHA256 mismatch: {report_path}")
            report = json.loads(encoded)
            events = report["payload_events"]
            if sum(e["completed"] for e in events) != case["metrics"]["payload_bytes"]:
                raise ValueError(f"Cached tail report payload-byte mismatch: {report_path}")
            last = max(event["start_ns"] + event["duration_ns"] for event in events)
        if not case["absolute_start_ns"] <= last <= case["absolute_end_ns"]:
            raise ValueError(f"Tail payload outside checkpoint: {case['row']['report']}")
        groups[(case["row"]["repeat"], case["row"]["config"])].append((last, case))
    for failure in cell["failures"]:
        failure["selected_quantile_diagnoses"] = [selected_quantile_diagnosis(cell, gate.split("p")[-1])
                                                   for gate in failure["failed_gates_strict"] if gate.startswith("shape-")]
        existing = {(r["repeat"], r["config"], r["rank"]): r for r in failure["critical_rank_breakdowns"]}
        tails = []
        for (repeat, config), selected in sorted(groups.items()):
            last, case = max(selected, key=lambda value: (value[0], -value[1]["row"]["rank"]))
            rank = case["row"]["rank"]
            key = repeat, config, rank
            if key not in existing:
                record = dict(config=config, repeat=repeat, rank=rank, roles=[], total_s=case["metrics"]["total_s"],
                              report=case["row"]["report"], breakdown=case["critical_breakdown"], staging_operations=case["staging_operations"])
                failure["critical_rank_breakdowns"].append(record)
                existing[key] = record
            if "latest_payload" not in existing[key]["roles"]:
                existing[key]["roles"].append("latest_payload")
            start = min(c["absolute_start_ns"] for _, c in selected)
            aggregate = next(a for a in cell["aggregates"] if a["phase"] == "save" and a["repeat"] == repeat and a["config"] == config)
            if abs((last - start) / 1e9 - aggregate["timeline"]["last_payload_s"]) > 1e-8:
                raise ValueError(f"Cached latest payload disagrees with aggregate: {cell_label(cell)}/{repeat}/{config}")
            tails.append(dict(repeat=repeat, config=config, rank=rank, last_payload_absolute_ns=last,
                              last_payload_relative_s=(last - start) / 1e9, report=case["row"]["report"],
                              report_sha256=case["provenance"]["report_sha256"]))
        failure["latest_payload_by_repeat"] = tails
        failure["critical_rank_breakdowns"].sort(key=lambda r: (r["repeat"], r["config"], r["rank"]))
    cell["concurrent_tail_diagnosis_version"] = 1
    return True


def critical_csv(cells, path):
    fields = ("model", "ranks", "format", "phase", "repeat", "config", "rank", "roles", "failed_strict", "failed_floor", "total_s", "pre_payload", "admission_wait", "payload_io", "materialize", "sink", "post_read_wait", "uncovered", "fsync", "other_staging_or_io", "partition_sum_s", "partition_matches_wall", "report")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        for cell in cells:
            for failure in cell["failures"]:
                failure["cause_zh"] = failure_cause(failure)
                for rank in failure["critical_rank_breakdowns"]:
                    part = rank["breakdown"]
                    writer.writerow(dict(model=cell["model"], ranks=cell["ranks"], format=cell["checkpoint_format"], phase="save",
                                         repeat=rank["repeat"], config=rank["config"], rank=rank["rank"], roles=";".join(rank["roles"]),
                                         failed_strict=";".join(failure["failed_gates_strict"]), failed_floor=";".join(failure["failed_gates_floor"]),
                                         total_s=rank["total_s"], **part["disjoint_partition_s"], partition_sum_s=part["partition_sum_s"],
                                         partition_matches_wall=part["partition_matches_wall"], report=rank["report"]))


def round_metrics(cell, config):
    aggregates = [a for a in cell["aggregates"] if a["phase"] == "save" and a["config"] == config]
    values = dict(wall_incl_fsync_s=core.median(a["wall_s"] for a in aggregates),
                  compressed_span_s=core.median(a["timeline"]["duration_s"] for a in aggregates),
                  global_fsync_union_s=core.median(a["timeline"]["global_fsync_union_s"] for a in aggregates),
                  payload_fsync_overlap_bytes=core.median(a["timeline"]["payload_fsync_overlap_bytes"] for a in aggregates))
    values["shape_quantiles_s"] = {q: core.median(a["timeline"]["quantiles_s"][q] for a in aggregates) for q in QUANTILES}
    return values


def overlap_summary(groups, config):
    selected = [g for g in groups if g["config"] == config]
    result = {}
    for op in ("pwrite", "fsync", "writer"):
        result[op] = {key: core.median(g[op][key] for g in selected)
                      for key in ("max_active_ranks", "active_rank_seconds", "union_s", "mean_active_ranks", "multi_rank_union_s")}
    result["pwrite_fsync_overlap_union_s"] = core.median(g["pwrite_fsync_overlap_union_s"] for g in selected)
    return result


def comparison_cell(serial, concurrent, serial_overlap, concurrent_overlap):
    by_config = {}
    for config in ("meta", "cpu"):
        before, after = round_metrics(serial, config), round_metrics(concurrent, config)
        speedup = {name: before[name] / after[name] for name in ("wall_incl_fsync_s", "compressed_span_s")}
        reduction = {name: 100 * (1 - after[name] / before[name]) for name in ("wall_incl_fsync_s", "compressed_span_s")}
        by_config[config] = dict(serial=before, concurrent=after, speedup_ratio=speedup, reduction_percent=reduction,
                                 shape_speedup_ratio={q: before["shape_quantiles_s"][q] / after["shape_quantiles_s"][q] for q in QUANTILES},
                                 overlap_serial=overlap_summary(serial_overlap["overlap"], config),
                                 overlap_concurrent=overlap_summary(concurrent_overlap["overlap"], config))
    return dict(model=serial["model"], ranks=serial["ranks"], checkpoint_format=serial["checkpoint_format"], by_config=by_config,
                serial_paired_alignment=serial["phases"]["save"], concurrent_paired_alignment=concurrent["phases"]["save"],
                alignment_change_percentage_points=dict(
                    wall=concurrent["phases"]["save"]["wall"]["paired_median_delta_percent"] - serial["phases"]["save"]["wall"]["paired_median_delta_percent"],
                    shape={q: concurrent["phases"]["save"]["shape"]["quantiles"][q]["paired_median_delta_percent"] - serial["phases"]["save"]["shape"]["quantiles"][q]["paired_median_delta_percent"] for q in QUANTILES}),
                serial_overlap=serial_overlap, concurrent_overlap=concurrent_overlap)


def write_tables(summary):
    root = Path(summary["root"])
    lines = ["# Concurrent save gates", "", "Δ = median of five paired Meta − CPU samples; percent denominator = median CPU. Wall includes fsync; shape removes the global union of measured fsync intervals. strict=±3%; floor=max(3%×CPU median,100 ms).", "", "|Cell|Metric|Meta s|CPU s|Paired Δ ms|Paired Δ %|Strict|Floor|", "|---|---|---:|---:|---:|---:|---|---|"]
    for cell in summary["cells"]:
        detail = cell["phases"]["save"]
        metrics = [("wall", detail["wall"])] + [(f"P{q}", detail["shape"]["quantiles"][q]) for q in QUANTILES]
        for name, value in metrics:
            def fmt(key, digits):
                return "unknown" if value[key] is None else f"{value[key]:.{digits}f}"
            lines.append(f"|{cell_label(cell)}|{name}|{fmt('meta_median_s', 4)}|{fmt('cpu_median_s', 4)}|{fmt('paired_median_delta_ms', 2)}|{fmt('paired_median_delta_percent', 2)}|{value['strict_status']}|{value['floor_status']}|")
    (root / "gate-table.md").write_text("\n".join(lines) + "\n")
    lines = ["# Serial vs concurrent save", "", "Each round uses five paired Meta/CPU samples. Cross-round speedup is a descriptive ratio of medians: the rounds were measured separately and are not temporally paired. API interval overlaps do not measure device concurrency.", "", "|Cell|Config|Wall serial→concurrent s|Speedup|Compressed span serial→concurrent s|Speedup|pwrite max ranks serial→concurrent|fsync max ranks serial→concurrent|pwrite∩fsync union serial→concurrent s|", "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in summary["serial_comparison"]["cells"]:
        for config, values in cell["by_config"].items():
            before, after = values["serial"], values["concurrent"]
            old_overlap, new_overlap = values["overlap_serial"], values["overlap_concurrent"]
            lines.append(f"|{cell_label(cell)}|{config}|{before['wall_incl_fsync_s']:.3f}→{after['wall_incl_fsync_s']:.3f}|{values['speedup_ratio']['wall_incl_fsync_s']:.2f}×|{before['compressed_span_s']:.3f}→{after['compressed_span_s']:.3f}|{values['speedup_ratio']['compressed_span_s']:.2f}×|{old_overlap['pwrite']['max_active_ranks']:g}→{new_overlap['pwrite']['max_active_ranks']:g}|{old_overlap['fsync']['max_active_ranks']:g}→{new_overlap['fsync']['max_active_ranks']:g}|{old_overlap['pwrite_fsync_overlap_union_s']:.3f}→{new_overlap['pwrite_fsync_overlap_union_s']:.3f}|")
    lines += ["", "Complete per-repeat and per-rank intervals/overlap, shape quantiles, paired alignment and percentage-point changes are retained in serial-comparison.json. Exact merged absolute intervals and verified raw-trace SHA256 are retained under overlap-intervals/."]
    (root / "serial-comparison.md").write_text("\n".join(lines) + "\n")


def plot(summary):
    """Every time axis uses the fsync-compressed aggregate timeline."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    root = Path(summary["root"])
    output = root / "plots"
    output.mkdir(parents=True, exist_ok=True)
    files, audits = [], []

    def draw(ax, timeline, label, color, style):
        values = np.asarray(timeline["binned_bytes_per_50ms"])
        widths = np.minimum(.05, np.maximum(0, timeline["duration_s"] - np.arange(len(values)) * .05))
        rate = np.divide(values / 1e9, widths, out=np.zeros_like(values), where=widths > 0)
        ax.plot(np.arange(len(values)) * .05 + widths / 2, rate, label=label, color=color, ls=style, lw=1.2)

    def finish(fig, axes, name):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        width, height = fig.canvas.get_width_height()
        clipped = []
        for artist in fig.findobj():
            if not isinstance(artist, matplotlib.text.Text) or not artist.get_visible() or not artist.get_text().strip():
                continue
            # Matplotlib creates out-of-view tick artists; only rendered ticks matter.
            if any(artist in [tick.label1, tick.label2] and not min(ax.get_xlim()) <= tick.get_loc() <= max(ax.get_xlim())
                   for ax in axes for tick in ax.xaxis.get_major_ticks()):
                continue
            if any(artist in [tick.label1, tick.label2] and not min(ax.get_ylim()) <= tick.get_loc() <= max(ax.get_ylim())
                   for ax in axes for tick in ax.yaxis.get_major_ticks()):
                continue
            box = artist.get_window_extent(renderer)
            if box.x0 < -1 or box.y0 < -1 or box.x1 > width + 1 or box.y1 > height + 1:
                clipped.append(artist.get_text())
        if clipped:
            raise ValueError(f"Plot text clipped: {name}: {clipped}")
        for suffix in ("png", "pdf"):
            target = output / f"{name}.{suffix}"
            fig.savefig(target)
            files.append(str(target))
        audits.append(dict(name=name, all_text_inside_figure=True, fsync_excluded=True))
        plt.close(fig)

    for cell in summary["cells"]:
        repeat = cell["phases"]["save"]["representative_repeat"]
        selected = {a["config"]: a["timeline"] for a in cell["aggregates"] if a["phase"] == "save" and a["repeat"] == repeat}
        if len(selected) != 2:
            continue
        fig, ax = plt.subplots(figsize=(12, 4.8), dpi=150)
        fig.subplots_adjust(left=.085, right=.985, bottom=.15, top=.83)
        for config, color, style in (("meta", "#2868b2", "--"), ("cpu", "#28936c", "-")):
            draw(ax, selected[config], config.upper() if config == "cpu" else "Meta", color, style)
        ax.set(title=f"{cell['model'].upper()} · {cell['ranks']} ranks · {cell['checkpoint_format']} · concurrent save",
               xlabel="Time (s, global fsync union excluded)", ylabel="Aggregate write throughput (GB/s)", ylim=(0, None))
        ax.set_xlim(0, max(t["duration_s"] for t in selected.values()) * 1.01)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e7e7e7", lw=.5)
        ax.legend(frameon=False, loc="lower right", bbox_to_anchor=(1, 1.08), ncol=2)
        finish(fig, [ax], cell_label(cell) + "-save")
    serial = {cell_key(c): c for c in summary["serial_baseline_cells"]}
    fig, axes = plt.subplots(4, 2, figsize=(15, 14), dpi=150)
    fig.subplots_adjust(left=.065, right=.985, bottom=.065, top=.925, hspace=.58, wspace=.22)
    for ax, cell in zip(axes.flat, summary["cells"]):
        previous = serial[cell_key(cell)]
        for source, prefix, style in ((previous, "Serial", "--"), (cell, "Concurrent", "-")):
            repeat = source["phases"]["save"]["representative_repeat"]
            selected = {a["config"]: a["timeline"] for a in source["aggregates"] if a["phase"] == "save" and a["repeat"] == repeat}
            for config, color in (("meta", "#2868b2"), ("cpu", "#28936c")):
                draw(ax, selected[config], f"{prefix} {'Meta' if config == 'meta' else 'CPU'}", color, style)
        ax.set(title=f"{cell['model'].upper()} · {cell['ranks']} ranks · {cell['checkpoint_format']}",
               xlabel="Time (s, global fsync union excluded)", ylabel="Write throughput (GB/s)", ylim=(0, None))
        ax.set_xlim(left=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color="#e7e7e7", lw=.5)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(.5, .98))
    finish(fig, list(axes.flat), "serial-vs-concurrent-save")
    core.base.dump(root / "plot-layout-audit.json", dict(all_pass=True, figures=len(audits), checks=audits))
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--serial-root", type=Path, default=DEFAULT_SERIAL)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args(argv)
    root, serial_root = args.root.resolve(), args.serial_root.resolve()
    if root == serial_root:
        raise ValueError("Concurrent output must differ from read-only baseline")
    configure()
    if args.plots_only:
        summary = core.base.read_json(root / "analysis.json")
        summary["plots"] = plot(summary)
        core.base.dump(root / "analysis.json", summary)
        print(json.dumps(dict(plots=len(summary["plots"]), raw_traces_reread=False)), flush=True)
        return summary
    root.mkdir(parents=True, exist_ok=True)
    workers = max(1, min(args.workers, 3))
    results = []
    tasks = [(str(root), model, ranks, fmt, args.resume_cache) for model in ("1b", "3b") for ranks in (4, 8) for fmt in ("dcp", "safetensors")]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(inspect_cell, task) for task in tasks]):
            result = future.result()
            results.append(result)
            progress = dict(stage="concurrent-inspection", completed_cells=len(results), rank_cases=sum(len(r["cases"]) for r in results),
                            elapsed_s=time.monotonic() - started, cell=cell_label(result["cell"]))
            core.base.dump(root / "analysis-progress.json", progress)
            print(json.dumps(progress), flush=True)
    results.sort(key=lambda r: cell_key(r["cell"]))
    cells = [r["cell"] for r in results]
    sanitize_concurrent_diagnoses(cells)
    cases = [case for r in results for case in r["cases"]]
    baseline_path = serial_root / "analysis.json"
    baseline = core.base.read_json(baseline_path)
    old_index = {cell_key(c): c for c in baseline["cells"]}
    serial_overlaps = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {}
        for cell in cells:
            selected = [case for case in baseline["cases"] if case["row"]["phase"] == "save" and (case["row"]["model"], case["row"]["world_size"], case["row"]["checkpoint_format"]) == cell_key(cell)]
            task = (str(root), old_index[cell_key(cell)], selected)
            futures[pool.submit(serial_overlap_task, task)] = cell_key(cell)
        for future in as_completed(futures):
            serial_overlaps[futures[future]] = future.result()
            progress = dict(stage="serial-overlap", completed_cells=len(serial_overlaps), elapsed_s=time.monotonic() - started)
            core.base.dump(root / "analysis-progress.json", progress)
            print(json.dumps(progress), flush=True)
    comparison = dict(methodology="Within each round: median of paired Meta-minus-CPU samples. Across rounds: descriptive ratios of medians, not paired observations.",
                      serial_root=str(serial_root), serial_analysis_sha256=core.base.sha256(baseline_path),
                      serial_gate_counts=gate_counts([old_index[cell_key(c)] for c in cells]), concurrent_gate_counts=gate_counts(cells),
                      intervals="Exact merged half-open pwrite payload/fsync API spans on the same host monotonic clock; baseline raw SHA256 verified; concurrent raw hashes verified by shared inspector.",
                      cells=[comparison_cell(old_index[cell_key(r["cell"])], r["cell"], serial_overlaps[cell_key(r["cell"])],
                                             {k: r[k] for k in ("overlap", "absolute_intervals_archive", "absolute_intervals_archive_sha256")}) for r in results])
    core.breakdown_csv(cases, root / "rank-breakdown.csv")
    critical_csv(cells, root / "critical-rank-breakdown.csv")
    runtime_audit = core.final_runtime_audit(cases)
    runtime_audit["writer_limit_observed_by_world_size"] = {
        str(ranks): sorted({case["writer_limit"] for case in cases if case["row"]["world_size"] == ranks})
        for ranks in (4, 8)}
    for case in cases:
        if case["writer_limit"] != case["row"]["world_size"]:
            runtime_audit["failures"].append(dict(report=case["row"]["report"], error="Writer limit does not equal world size", observed=case["writer_limit"]))
    runtime_audit["all_pass"] = not runtime_audit["failures"]
    core.base.dump(root / "runtime-mode-audit.json", runtime_audit)
    errors = [e for r in results for e in r["errors"]]
    summary = dict(schema_version=1, root=str(root), rank_cases=len(cases), expected_rank_cases=480, errors=errors,
                   cells=cells, cases=cases, gate_counts=gate_counts(cells), runtime_mode_audit=runtime_audit,
                   request_hard_gate=all(c["sequence_status"] == dict(full="identical", payload="identical") for c in cells),
                   complete=len(cells) == 8 and len(cases) == 480 and not errors and all(c["phases"]["save"]["wall"]["complete"] for c in cells),
                   methodology=dict(phases=["save"], expected_repeats=5, wall="maximum rank-local checkpoint wall including fsync",
                                    delta="median of paired Meta - CPU seconds", percent_denominator="median paired CPU seconds",
                                    strict="absolute paired median delta <= 3% CPU median", floor="absolute paired median delta <= max(3% CPU median,100 ms)",
                                    shape="P10/P50/P90/P100 interpolated through 50 ms aggregate byte bins after deleting global fsync union",
                                    overlapping_payload_policy="Payload bytes inside any rank's fsync map to an instant on compressed axis; conserved, not discarded. Throughput is compressed API progress, not physical device throughput.",
                                    representative_repeat="same paired repeat with average compressed duration nearest its median",
                                    source="Meta CPU-target faithful resident local-shard source, prepared before timer",
                                    requested_writer_limit=0, effective_writer_limits={"4": 4, "8": 8},
                                    observed_writer_limits=runtime_audit["writer_limit_observed_by_world_size"]),
                   serial_comparison=comparison,
                   serial_baseline_cells=[dict(model=c["model"], ranks=c["ranks"], checkpoint_format=c["checkpoint_format"],
                                               phases={"save": c["phases"]["save"]},
                                               aggregates=[a for a in c["aggregates"] if a["phase"] == "save"]) for c in (old_index[cell_key(c)] for c in cells)], plots=[])
    core.base.dump(root / "serial-comparison.json", comparison)
    core.base.dump(root / "analysis.json", summary)
    write_tables(summary)
    print(json.dumps(dict(analysis=str(root / "analysis.json"), rank_cases=len(cases), errors=len(errors), gates=summary["gate_counts"])), flush=True)
    return summary


if __name__ == "__main__":
    main()
