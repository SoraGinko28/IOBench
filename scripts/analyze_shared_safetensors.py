"""Analyze shared safetensors gates and immutable per-rank baselines.

Raw files are read only after measurements finish. Existing multi-rank analysis
supplies request signatures, paired gates, global-fsync compression and plots.
Every new artifact, including baseline comparison caches, goes under --root.
"""

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import analyze_concurrent_save as concurrent  # noqa: E402
from scripts import analyze_multirank_meta_cpu as core  # noqa: E402

DEFAULT_ROOT = ROOT / "validation/shared-safetensors-20261030"
DEFAULT_FIX1 = ROOT / "validation/multirank-fix1-20261030"
DEFAULT_CONCURRENT = ROOT / "validation/multirank-concurrent-save-20261030"
PHASES = ("save", "cold-load", "warm-load")
QUANTILES = ("10", "50", "90", "100")
READ_OPS = {"read", "pread", "readinto", "readline"}
WRITE_OPS = {"write", "pwrite"}
WARM_TOLERANCE = 64 * 1024


def configure(repeats=5):
    core.PHASES, core.EXPECTED_REPEATS = PHASES, repeats
    core.FIX1, core.FLOOR_S = True, .1


def key(cell):
    return cell["model"], cell["ranks"], cell["checkpoint_format"]


def label(cell):
    return f"{cell['model']}-{cell['ranks']}r-{cell['checkpoint_format']}"


def gate_counts(cells):
    result = {}
    for mode in ("strict", "floor"):
        result[mode] = {}
        for kind in ("wall", "shape"):
            values = [p["wall"] for c in cells for p in c["phases"].values()] if kind == "wall" else [q for c in cells for p in c["phases"].values() for q in p["shape"]["quantiles"].values()]
            statuses = [v[mode + "_status"] for v in values]
            result[mode][kind] = dict(passed=statuses.count("pass"), failed=statuses.count("fail"),
                                      unknown=statuses.count("unknown"), total=len(statuses))
    return result


def add_tail_roles(cell, cases):
    """Track the actual final payload rank for save and both load phases."""
    for failure in cell["failures"]:
        phase = failure["phase"]
        groups = defaultdict(list)
        for case in cases:
            if case["row"]["phase"] == phase:
                last = max(case["origin_ns"] + e["start_ns"] + e["duration_ns"] for e in case["payload"])
                groups[(case["row"]["repeat"], case["row"]["config"])].append((last, case))
        existing = {(r["repeat"], r["config"], r["rank"]): r for r in failure["critical_rank_breakdowns"]}
        tails = []
        for (repeat, config), selected in sorted(groups.items()):
            last, case = max(selected, key=lambda value: (value[0], -value[1]["row"]["rank"]))
            rank = case["row"]["rank"]
            if (repeat, config, rank) not in existing:
                record = dict(repeat=repeat, config=config, rank=rank, roles=[],
                              total_s=case["metrics"]["total_s"], report=case["row"]["report"],
                              breakdown=case["critical_breakdown"], staging_operations=case["staging_operations"])
                existing[(repeat, config, rank)] = record
                failure["critical_rank_breakdowns"].append(record)
            existing[(repeat, config, rank)]["roles"].append("latest_payload")
            tails.append(dict(repeat=repeat, config=config, rank=rank,
                              last_payload_absolute_ns=last, report=case["row"]["report"]))
        failure["latest_payload_by_repeat"] = tails
        if phase == "save":
            failure["selected_quantile_diagnoses"] = [concurrent.selected_quantile_diagnosis(cell, g.split("p")[-1])
                                                       for g in failure["failed_gates_strict"] if g.startswith("shape-")]
        failure["cause_zh"] = concurrent.failure_cause(failure)


def inspect_cell(task):
    root_text, model, ranks, resume, stage, repeats = task
    configure(repeats)
    root = Path(root_text)
    run = root / stage / f"{model}-{ranks}r-safetensors"
    cache = root / "analysis-cache" / f"{stage}-{run.name}.json"
    if resume and cache.exists():
        return core.base.read_json(cache)
    cases, errors = core.discover(root, root, run=run)
    cell = core.analyze_cell(model, ranks, "safetensors", cases)
    cell.update(layout="shared", measurement_stage=stage, expected_repeats=repeats)
    concurrent.sanitize_concurrent_diagnoses([cell])
    add_tail_roles(cell, cases)
    retained = [{field: case[field] for field in core.RETAINED_FIELDS} for case in cases]
    result = dict(cell=cell, cases=retained, errors=errors)
    core.base.dump(cache, result)
    return result


def warm_audit(cases):
    """Keep every byte of block-read evidence while applying the 64 KiB rule."""
    audit = core.warm_cache_io_audit(cases)
    for sample in audit["samples"]:
        for rank in sample["rank_checks"]:
            rank["valid"] = (rank["residency_percent"] is not None and rank["residency_percent"] >= 99
                             and rank["cache_before_call"].get("residency", {}).get("fully_warm_data_residency") is True
                             and rank["read_bytes"] <= WARM_TOLERANCE
                             and rank["cache_before_call"].get("after_resident_buffer_preparation") is True
                             and rank["cache_before_call"].get("probe_end_ns", float("inf"))
                             <= next(c["stats"]["dcp_started_ns"] for c in cases if c["row"]["report"] == rank["report"]))
        sample["valid"] = all(r["valid"] for r in sample["rank_checks"])
    audit.update(valid_samples=sum(s["valid"] for s in audit["samples"]),
                 all_valid=bool(audit["samples"]) and all(s["valid"] for s in audit["samples"]),
                 timed_block_read_tolerance_bytes_per_rank=WARM_TOLERANCE,
                 limitation="Per-rank residency >=99% after target/source preparation and before timing; timed block reads <=64 KiB accepted and still recorded; one redo only.")
    return audit


def profile_case(case):
    """Count file/rank relationships and actual request sizes from raw traces."""
    row = case["row"]
    raw = Path(row["report"]).with_suffix(".io.jsonl")
    digest = hashlib.sha256()
    payload_histogram, operations, io_layers = Counter(), Counter(), Counter()
    files = defaultdict(lambda: dict(payload_read_requests=0, payload_write_requests=0,
                                    payload_read_bytes=0, payload_write_bytes=0,
                                    header_requests=0, header_bytes=0, fsyncs=0, observed_opens=0))
    headers, lifecycle = [], []
    with raw.open("rb") as stream:
        for line in stream:
            digest.update(line)
            if not line.strip():
                continue
            event = json.loads(line)
            op, layer = event["op"], event.get("io_layer", "buffered_file_api")
            if layer == "staging":
                continue
            operations[op] += 1
            io_layers[layer] += 1
            name = Path(event.get("file", event.get("path", "unknown"))).name
            if layer == "lifecycle":
                lifecycle.append({k: event[k] for k in ("op", "file", "offset", "requested", "completed", "target", "start_ns", "duration_ns", "details") if k in event})
            if not name.endswith(".safetensors"):
                continue
            record = files[name]
            if op == "fsync":
                record["fsyncs"] += 1
            if op in {"open", "file_open", "open_file"} or event.get("target") == "file_open":
                record["observed_opens"] += 1
            if layer == "payload" and op in READ_OPS | WRITE_OPS:
                direction = "read" if op in READ_OPS else "write"
                record[f"payload_{direction}_requests"] += 1
                record[f"payload_{direction}_bytes"] += event.get("completed", 0)
                payload_histogram[str(event["requested"])] += 1
            elif op in READ_OPS | WRITE_OPS:
                record["header_requests"] += 1
                record["header_bytes"] += event.get("completed", 0)
                headers.append(dict(file=name, op=op, offset=event.get("offset"),
                                    requested=event.get("requested"), completed=event.get("completed"), layer=layer))
    actual = digest.hexdigest()
    if actual != case["provenance"]["trace_sha256"]:
        raise ValueError(f"Raw trace changed since analysis: {raw}")
    return dict(model=row["model"], ranks=row["world_size"], phase=row["phase"],
                config=row["config"], repeat=row["repeat"], rank=row["rank"],
                trace=str(raw), trace_sha256=actual, files=dict(files),
                payload_request_size_histogram=dict(payload_histogram), operations=dict(operations),
                io_layers=dict(io_layers), header_requests=headers, lifecycle=lifecycle,
                inventory_file_count=row.get("layout", {}).get("file_count"))


def aggregate_profiles(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record["model"], record["ranks"], record["phase"], record["config"], record["repeat"])].append(record)
    samples = []
    for (model, ranks, phase, config, repeat), selected in sorted(groups.items()):
        files, hist, operations = {}, Counter(), Counter()
        for record in selected:
            hist.update(record["payload_request_size_histogram"])
            operations.update(record["operations"])
            for name, values in record["files"].items():
                target = files.setdefault(name, dict(reader_ranks=[], writer_ranks=[], header_ranks=[],
                                                     payload_read_requests=0, payload_write_requests=0,
                                                     payload_read_bytes=0, payload_write_bytes=0,
                                                     header_requests=0, header_bytes=0, fsyncs=0, observed_opens=0))
                for direction, field in (("read", "reader_ranks"), ("write", "writer_ranks")):
                    if values[f"payload_{direction}_requests"]:
                        target[field].append(record["rank"])
                if values["header_requests"]:
                    target["header_ranks"].append(record["rank"])
                for field, value in values.items():
                    target[field] += value
        observed_opens = sum(f["observed_opens"] for f in files.values())
        # Old traces did not record open(). A count of touched (rank,file)
        # pairs is a conservative source-independent lower bound, not an
        # invented exact syscall count.
        touched_pairs = sum(len(r["files"]) for r in selected)
        samples.append(dict(model=model, ranks=ranks, phase=phase, config=config, repeat=repeat,
                            complete=sorted(r["rank"] for r in selected) == list(range(ranks)),
                            payload_file_count=len(files), files=files,
                            payload_request_size_histogram=dict(sorted(hist.items(), key=lambda item: int(item[0]))),
                            operations=dict(operations), observed_payload_file_opens=observed_opens or None,
                            distinct_rank_file_pairs=touched_pairs,
                            open_counts_are_lower_bounds=True,
                            header_counts_are_observed_only=True,
                            open_count_limitation="Observed opens are a lower bound: shared save rank0 finish reopens files for header validation outside the recorder. Old per-rank traces do not instrument opens, so their exact count is unknown. Distinct rank/file pairs are a separate lower bound on handles used, not an exact open count.",
                            header_count_limitation="Shared save finish header-validation reads/reopens are untraced; old per-rank load header parsing is also untraced. Header counts describe recorded requests only.",
                            header_requests=sum(f["header_requests"] for f in files.values()),
                            header_bytes=sum(f["header_bytes"] for f in files.values()),
                            fsyncs=operations.get("fsync", 0),
                            lifecycle_events=sum(len(r["lifecycle"]) for r in selected)))
    return samples


def io_profiles(root, name, cases, *, resume):
    cache = root / "analysis-cache" / f"io-{name}.json"
    hashes = {c["row"]["report"]: c["provenance"]["trace_sha256"] for c in cases}
    if resume and cache.exists():
        cached = core.base.read_json(cache)
        if cached["raw_trace_hashes"] != hashes:
            raise ValueError(f"IO cache inputs changed: {cache}")
        return cached
    records = [profile_case(case) for case in cases]
    result = dict(name=name, raw_trace_hashes=hashes, rank_records=records,
                  samples=aggregate_profiles(records))
    core.base.dump(cache, result)
    return result


def median_profile(profiles, model, ranks, phase, config):
    selected = [s for s in profiles["samples"] if (s["model"], s["ranks"], s["phase"], s["config"]) == (model, ranks, phase, config)]
    fields = ("payload_file_count", "distinct_rank_file_pairs", "header_requests", "header_bytes", "fsyncs", "lifecycle_events")
    result = {field: core.median(s[field] for s in selected) for field in fields}
    opens = [s["observed_payload_file_opens"] for s in selected]
    result["observed_payload_file_opens"] = core.median(opens) if opens and all(v is not None for v in opens) else None
    result["open_counts_are_lower_bounds"] = True
    result["header_counts_are_observed_only"] = True
    result["reader_ranks_per_file"] = sorted({len(f["reader_ranks"]) for s in selected for f in s["files"].values() if f["reader_ranks"]})
    result["writer_ranks_per_file"] = sorted({len(f["writer_ranks"]) for s in selected for f in s["files"].values() if f["writer_ranks"]})
    hist = Counter()
    for sample in selected:
        hist.update(sample["payload_request_size_histogram"])
    result["payload_request_size_histogram_all_repeats"] = dict(sorted(hist.items(), key=lambda item: int(item[0])))
    result["repeats"] = sorted(s["repeat"] for s in selected)
    return result


def compare_baselines(root, cells, cases, baselines, *, resume):
    current_io = io_profiles(root, "shared", cases, resume=resume)
    result = dict(methodology="Across-run descriptive median ratios; only within each run are Meta and CPU paired. Baselines remain read-only.",
                  shared_io=current_io, baselines=[], comparisons=[])
    keys = {key(cell) for cell in cells}
    for name, baseline_root in baselines:
        source = baseline_root / "analysis.json"
        baseline = core.base.read_json(source)
        old_cells = {key(c): c for c in baseline["cells"] if key(c) in keys}
        old_cases = [c for c in baseline["cases"] if (c["row"]["model"], c["row"]["world_size"], c["row"]["checkpoint_format"]) in keys]
        old_io = io_profiles(root, name, old_cases, resume=resume)
        result["baselines"].append(dict(name=name, root=str(baseline_root), analysis_sha256=core.base.sha256(source), io=old_io))
        for cell in cells:
            previous = old_cells[key(cell)]
            for phase in PHASES:
                if phase not in previous["phases"]:
                    continue
                for config in ("meta", "cpu"):
                    metric = config + "_median_s"
                    previous_wall, shared_wall = previous["phases"][phase]["wall"], cell["phases"][phase]["wall"]
                    # Preserve the original partial median as evidence, but do
                    # not compare an incomplete baseline against five pairs.
                    before = previous_wall[metric] if previous_wall["complete"] else None
                    after = shared_wall[metric] if shared_wall["complete"] else None
                    old_profile = median_profile(old_io, cell["model"], cell["ranks"], phase, config)
                    new_profile = median_profile(current_io, cell["model"], cell["ranks"], phase, config)
                    result["comparisons"].append(dict(baseline=name, model=cell["model"], ranks=cell["ranks"],
                        phase=phase, config=config, per_rank_wall_s=before, shared_wall_s=after,
                        per_rank_observed_wall_median_s=previous_wall[metric],
                        shared_observed_wall_median_s=shared_wall[metric],
                        per_rank_paired_repeats=len(previous_wall["samples"]),
                        shared_paired_repeats=len(shared_wall["samples"]),
                        comparison_status="complete" if previous_wall["complete"] and shared_wall["complete"] else "unknown",
                        incomplete_baseline_reason=(None if previous_wall["complete"] else "Baseline lacks the required five valid pairs; partial median retained only as evidence"),
                        shared_wall_change_percent=100 * (after / before - 1) if before and after else None,
                        wall_speedup=before / after if before and after else None,
                        per_rank_io=old_profile, shared_io=new_profile,
                        per_rank_gate_complete=previous["phases"][phase]["wall"]["complete"],
                        shared_gate_complete=cell["phases"][phase]["wall"]["complete"]))
    core.base.dump(root / "layout-comparison.json", result)
    return result


def critical_csv(cells, path):
    fields = ("model", "ranks", "phase", "repeat", "config", "rank", "roles", "failed_strict", "failed_floor", "total_s", "pre_payload", "admission_wait", "payload_io", "materialize", "sink", "post_read_wait", "uncovered", "fsync", "other_staging_or_io", "partition_sum_s", "partition_matches_wall", "report")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in cells:
            for failure in cell["failures"]:
                for record in failure["critical_rank_breakdowns"]:
                    breakdown = record["breakdown"]
                    writer.writerow(dict(model=cell["model"], ranks=cell["ranks"], phase=failure["phase"],
                        repeat=record["repeat"], config=record["config"], rank=record["rank"], roles=";".join(record["roles"]),
                        failed_strict=";".join(failure["failed_gates_strict"]), failed_floor=";".join(failure["failed_gates_floor"]),
                        total_s=record["total_s"], **breakdown["disjoint_partition_s"],
                        partition_sum_s=breakdown["partition_sum_s"], partition_matches_wall=breakdown["partition_matches_wall"], report=record["report"]))


def write_tables(summary, root):
    fields = ("model", "ranks", "phase", "metric", "meta_median_s", "cpu_median_s", "paired_median_delta_ms", "paired_median_delta_percent", "strict_tolerance_s", "floor_tolerance_s", "strict_status", "floor_status", "complete")
    lines = ["|单元|阶段|Wall Meta/CPU s|Wall Δ%|Wall严格/floor|Shape Δ% P10/P50/P90/P100|Shape严格/floor|",
             "|---|---|---:|---:|---|---|---|"]
    with (root / "gate-table.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for cell in summary["cells"]:
            for phase, detail in cell["phases"].items():
                for metric, value in [("wall", detail["wall"])] + [("shape-p" + q, detail["shape"]["quantiles"][q]) for q in QUANTILES]:
                    writer.writerow(dict(model=cell["model"], ranks=cell["ranks"], phase=phase, metric=metric,
                                         **{f: value[f] for f in fields if f in value}))
                wall, shape = detail["wall"], detail["shape"]["quantiles"]
                fmt = lambda value, digits=2: "unknown" if value is None else f"{value:+.{digits}f}"
                seconds = "unknown" if wall["meta_median_s"] is None else f"{wall['meta_median_s']:.3f}/{wall['cpu_median_s']:.3f}"
                statuses = "/".join(wall[m + "_status"] for m in ("strict", "floor"))
                passes = " → ".join(f"{sum(v[m + '_status'] == 'pass' for v in shape.values())}/4" for m in ("strict", "floor"))
                lines.append(f"|{cell['model'].upper()}/{cell['ranks']}r|{phase}|{seconds}|{fmt(wall['paired_median_delta_percent'])}|{statuses}|" + "/".join(fmt(shape[q]["paired_median_delta_percent"]) for q in QUANTILES) + f"|{passes}|")
    (root / "gate-table.md").write_text("\n".join(lines) + "\n")
    comparisons = summary["layout_comparison"]["comparisons"]
    compare_fields = ("baseline", "model", "ranks", "phase", "config", "per_rank_wall_s", "shared_wall_s", "shared_wall_change_percent", "wall_speedup", "per_rank_files", "shared_files", "per_rank_fsyncs", "shared_fsyncs", "per_rank_observed_opens", "shared_observed_opens", "per_rank_rank_file_pairs", "shared_rank_file_pairs")
    with (root / "layout-comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=compare_fields)
        writer.writeheader()
        for record in comparisons:
            row = {f: record[f] for f in compare_fields if f in record}
            for layout in ("per_rank", "shared"):
                io = record[layout + "_io"]
                row.update({layout + "_files": io["payload_file_count"], layout + "_fsyncs": io["fsyncs"],
                            layout + "_observed_opens": io["observed_payload_file_opens"],
                            layout + "_rank_file_pairs": io["distinct_rank_file_pairs"]})
            writer.writerow(row)


def report_text(summary):
    counts = summary["gate_counts"]
    requests = all(v == "identical" for c in summary["cells"] for v in c["sequence_status"].values())
    count_text = lambda mode, kind: f"{counts[mode][kind]['passed']}/{counts[mode][kind]['total']}（fail {counts[mode][kind]['failed']}，unknown {counts[mode][kind]['unknown']}）"
    correctness = summary.get("correctness", {})
    correctness_results = summary.get("correctness_results", [])
    correctness_counts = Counter(r.get("gate", "unknown") for r in correctness_results if r.get("passed"))
    correctness_text = (f"位置pattern正确性：状态{correctness.get('status', 'unknown')}，"
                        f"通过{sum(r.get('passed') is True for r in correctness_results)}/{len(correctness_results)}项；"
                        + "、".join(f"{name} {count}" for name, count in sorted(correctness_counts.items()))
                        + "。" if correctness_results else "正确性结果尚未汇入，不能从零值测量推导正确性。")
    lines = ["# 共享标准 safetensors：多 rank Meta / CPU", "",
        "采用一个标准完整 tensor header，每个 rank 通过 gdtensor 并发写入自身 FSDP 行区间；header/文件大小由 rank 0 建立，barrier 后各 rank 写入互不重叠的 payload。默认 model.safetensors；按完整 tensor 拆文件时使用 HF index。没有每 rank 文件或合并读写回合。运行时细节及 setup/header/barrier 证据保留在原始 IO trace。", "",
        f"4个主单元、每格5次Meta/CPU配对并交错，save/cold/warm。完整及payload请求硬门禁{'全部通过' if requests else '未全部通过'}。严格wall {count_text('strict', 'wall')}、shape {count_text('strict', 'shape')}；100ms floor wall {count_text('floor', 'wall')}、shape {count_text('floor', 'shape')}。", "",
        "wall为各rank checkpoint调用最大值（save含fsync）。Δ=median(逐repeat Meta−CPU)，百分比分母为CPU中位秒；严格±3%，floor为±max(3%×CPU中位秒,100ms)。shape使用50ms聚合byte bins插值P10/P50/P90/P100。所有save shape及图均删除全rank fsync区间并集，同时保留与其他rank fsync重叠的payload字节。", "",
        correctness_text + "证据见[correctness/progress.json](correctness/progress.json)及[correctness/results.json](correctness/results.json)。", "",
        "变更：metasl/gdt_shared_hf_storage.py实现共享layout，hf_storage.py及run.py/phase.py接入；共享实现复用未改动的gdt_hf_storage.py会话；iostats.py/trace_export.py保留实际open、setup、barrier辅助证据；scripts/run_gdt_safetensors.py保留旧per-rank默认，新增shared和warm 64KiB容差；新run_shared_safetensors.py、validate_shared_safetensors.py、analyze_shared_safetensors.py负责编排、正确性及分析。", "",
        (Path(summary["root"]) / "gate-table.md").read_text(), "",
        "门禁逐秒、毫秒差和容限见[gate-table.csv](gate-table.csv)。warm保持≥99%计时前residency；每rank计时内≤64KiB块读取有效但仍逐字节记录，最多重做一次。" +
        f"本轮有效warm {summary['warm_cache_io_audit']['valid_samples']}/{summary['warm_cache_io_audit']['warm_samples']}，最低residency {summary['warm_cache_io_audit']['minimum_residency_percent']}%，块读取合计 {summary['warm_cache_io_audit']['total_read_bytes']} bytes。", "",
        "共享与per-rank使用两轮既有只读基线：并发save对比multirank-concurrent-save，cold/warm及串行save背景对比fix1。跨轮比值是描述性中位数比较，没有把旧新数据视作时间配对。", "",
        "|基线|单元/阶段|Wall per-rank→shared s（Meta / CPU）|共享变化% M/C|文件数 per-rank→shared|fsync次数 per-rank→shared（M/C）|",
        "|---|---|---|---:|---|---|"]
    grouped = defaultdict(dict)
    for comparison in summary["layout_comparison"]["comparisons"]:
        grouped[(comparison["baseline"], comparison["model"], comparison["ranks"], comparison["phase"])][comparison["config"]] = comparison
    for (baseline, model, ranks, phase), pair in sorted(grouped.items()):
        m, c = pair["meta"], pair["cpu"]
        def fmt_s(record):
            before = (f"{record['per_rank_wall_s']:.3f}" if record["per_rank_wall_s"] is not None
                      else f"unknown(n={record['per_rank_paired_repeats']})")
            after = f"{record['shared_wall_s']:.3f}" if record["shared_wall_s"] is not None else "unknown"
            return f"{before}→{after}"
        fmt_pct = lambda r: "unknown" if r["shared_wall_change_percent"] is None else f"{r['shared_wall_change_percent']:+.1f}"
        lines.append(f"|{baseline}|{model.upper()}/{ranks}r/{phase}|{fmt_s(m)} / {fmt_s(c)}|{fmt_pct(m)}/{fmt_pct(c)}|{m['per_rank_io']['payload_file_count']}→{m['shared_io']['payload_file_count']}|" + "/".join(f"{r['per_rank_io']['fsyncs']}→{r['shared_io']['fsyncs']}" for r in (m, c)) + "|")
    lines += ["", "请求size histogram、每文件reader/writer rank数、header读写字节与请求、文件/handle关系及原始hash见[layout-comparison.json](layout-comparison.json)，简表见[layout-comparison.csv](layout-comparison.csv)。新共享路径的observed opens同样是下界：rank0 finish为校验header再次打开文件，这些open/read未插桩；header计数仅指已记录请求。历史路径没有open调用插桩，exact opens记unknown；另列实际触达的(rank,file)对数作为handle使用下界。旧per-rank cold/warm的header解析也未进入原始IO trace，因此旧header读取次数记未观测。", "",
        "fix1基线1B/8r safetensors warm只有4个有效配对，比较wall、差百分比及加速比保留unknown；原始4对中位数仅作为证据存入layout-comparison.json，不与本轮5对结果构成完整比较。", "",
        "生产加载关联：共享路径先读取8字节header长度和JSON header，再对目标tensor范围pread；实际沿用64KiB块，与生产所述4MiB pread粒度不同，tensor/shard尾部还会更小。文件组织和header→range-read顺序接近标准HF生产路径；吞吐不能直接外推到4MiB请求配置。", "",
        "剩余失败（每项一句原因，详细互斥分区见[critical-rank-breakdown.csv](critical-rank-breakdown.csv)）：", ""]
    failures = 0
    for cell in summary["cells"]:
        for failure in cell["failures"]:
            failures += 1
            lines.append(f"- {cell['model'].upper()}/{cell['ranks']}r/{failure['phase']}，strict {','.join(failure['failed_gates_strict'])}；floor {','.join(failure['failed_gates_floor']) or '无'}：{failure['cause_zh']}")
        for phase, detail in cell["phases"].items():
            if not detail["wall"]["complete"]:
                failures += 1
                lines.append(f"- {cell['model'].upper()}/{cell['ranks']}r/{phase}：配对不完整（n={len(detail['wall']['samples'])}），时间gate保留unknown，未补造样本。")
    if not failures:
        lines.append("- 无剩余时间门禁失败。")
    lines += ["", "每个失败保留逐repeat max-wall、latest-reader/writer、first/last-payload关键rank及互斥wall分区。分区可相加重构该rank wall；跨repeat分项中位数不可相加。IO API时长含内核等待和调度，不唯一归因于磁盘服务时间。", "",
              f"3B按5GiB整tensor拆文件的1-repeat smoke保留在[{summary.get('split_smoke_stage', 'split-smoke')}/]({summary.get('split_smoke_stage', 'split-smoke')}/)与[split-smoke-analysis.json](split-smoke-analysis.json)，不混入5-repeat主门禁。最初split-smoke的index文本写入失败及修复/重新执行记录见progress.json与[progress-first-pass.json](progress-first-pass.json)；主矩阵单文件路径不受该index分支修复影响。", "",
              "图见[plots/summary.png](plots/summary.png)及plots/中的12个Meta/CPU聚合吞吐图（PNG/PDF）。代表repeat由同一配对平均压缩时长最接近5次中位数选取；save去fsync。summary的save点同样用去fsync全局span，cold/warm点用wall。布局检查见[plot-layout-audit.json](plot-layout-audit.json)。", "",
              "每phase启动前foreign metasl.run/fsbench、CPU/本地盘空闲检查见[idle-guards.jsonl](idle-guards.jsonl)，进度见[progress.json](progress.json)，每格cleanup.json记录checkpoint清理；无旧validation写入、git操作、NFS或系统设置变更。", ""]
    if summary["errors"]:
        lines += ["读取/证据错误：", ""] + [f"- {e}" for e in summary["errors"]]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--fix1-root", type=Path, default=DEFAULT_FIX1)
    parser.add_argument("--concurrent-root", type=Path, default=DEFAULT_CONCURRENT)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume-cache", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    if root in {args.fix1_root.resolve(), args.concurrent_root.resolve()}:
        raise ValueError("Shared output must differ from read-only baselines")
    configure()
    if args.plots_only:
        summary = core.base.read_json(root / "analysis.json")
        if not summary.get("cells") or any(c.get("layout") != "shared" for c in summary["cells"]):
            raise ValueError("Refusing to write plots into a previous non-shared validation directory")
        summary["plots"] = core.make_plots(summary["cells"], root / "plots")
        core.base.dump(root / "analysis.json", summary)
        return summary
    progress = core.base.read_json(root / "progress.json")
    if progress.get("layout") != "shared":
        raise ValueError("Refusing to write analysis into a previous non-shared validation directory")
    if progress["status"] not in {"measurements-complete", "complete"}:
        raise ValueError("Analysis must wait for all measurements to finish")
    results, started = [], time.monotonic()
    tasks = [(str(root), model, ranks, args.resume_cache, "matrix", 5) for model in ("1b", "3b") for ranks in (4, 8)]
    with ProcessPoolExecutor(max_workers=max(1, min(args.workers, 3))) as pool:
        for future in as_completed([pool.submit(inspect_cell, task) for task in tasks]):
            results.append(future.result())
            state = dict(stage="shared-inspection", completed_cells=len(results),
                         rank_cases=sum(len(r["cases"]) for r in results), elapsed_s=time.monotonic() - started)
            core.base.dump(root / "analysis-progress.json", state)
            print(json.dumps(state), flush=True)
    results.sort(key=lambda r: key(r["cell"]))
    cells, cases = [r["cell"] for r in results], [case for r in results for case in r["cases"]]
    configure()
    invalid = core.audit_invalid_warm_attempts(root, cases)
    core.apply_invalid_sequence_evidence(cells, invalid)
    core.base.dump(root / "invalid-warm-request-audit.json", invalid)
    core.add_sink_bounds(cases)
    summary = dict(schema_version=1, root=str(root), cells=cells, cases=cases, rank_cases=len(cases),
                   errors=[e for r in results for e in r["errors"]], progress=progress, plots=[],
                   methodology=dict(expected_repeats=5, floor_s=.1, warm_tolerance_bytes=WARM_TOLERANCE,
                       wall="max rank-local checkpoint wall including fsync", delta="median paired Meta-CPU",
                       save_axis="global union of all rank fsync intervals removed; overlap bytes conserved"),
                   invalid_warm_request_audit=invalid, gate_counts=gate_counts(cells), warm_cache_io_audit=warm_audit(cases))
    for name, field in (("progress.json", "correctness"), ("results.json", "correctness_results")):
        path = root / "correctness" / name
        if path.is_file():
            summary[field] = core.base.read_json(path)
    core.base.dump(root / "warm-cache-io-audit.json", summary["warm_cache_io_audit"])
    core.base.dump(root / "final-timing-sink-audit.json", core.final_runtime_audit(cases))
    core.breakdown_csv(cases, root / "rank-breakdown.csv")
    critical_csv(cells, root / "critical-rank-breakdown.csv")
    summary["layout_comparison"] = compare_baselines(root, cells, cases,
        [("fix1", args.fix1_root.resolve()), ("concurrent-save", args.concurrent_root.resolve())], resume=args.resume_cache)
    smoke_stage = "split-smoke-retry" if (root / "split-smoke-retry").is_dir() else "split-smoke"
    smoke = [inspect_cell((str(root), "3b", ranks, args.resume_cache, smoke_stage, 1))
             for ranks in (4, 8) if (root / smoke_stage / f"3b-{ranks}r-safetensors").is_dir()]
    summary["split_smoke_stage"] = smoke_stage
    core.base.dump(root / "split-smoke-analysis.json", dict(stage=smoke_stage, cells=[r["cell"] for r in smoke],
        rank_cases=sum(len(r["cases"]) for r in smoke), errors=[e for r in smoke for e in r["errors"]],
        warm_cache_io_audit=warm_audit([case for r in smoke for case in r["cases"]])))
    configure()
    write_tables(summary, root)
    core.base.dump(root / "analysis.json", summary)
    (root / "report-zh.md").write_text(report_text(summary))
    if not args.no_plots:
        summary["plots"] = core.make_plots(cells, root / "plots")
        core.base.dump(root / "analysis.json", summary)
    print(json.dumps(dict(analysis=str(root / "analysis.json"), rank_cases=len(cases), gate_counts=summary["gate_counts"], errors=len(summary["errors"])), ensure_ascii=False), flush=True)
    return summary


if __name__ == "__main__":
    main()
