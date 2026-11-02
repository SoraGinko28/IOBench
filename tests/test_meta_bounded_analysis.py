"""Focused owner-gate checks using tiny traces, without workers or benchmarks."""

import json
from pathlib import Path

import pytest

from metasl.trace_export import export_rank_trace
from scripts import analyze_meta_bounded_mem as analysis


def make_cell(tmp_path, *, changed_offset=False):
    run = tmp_path / "cell"
    rows = []
    for phase in analysis.PHASES:
        for config in analysis.CONFIGS:
            output = run / "measurements/1b" / f"r1-{config}" / phase
            output.mkdir(parents=True)
            memory_workers = {}
            for rank in range(2):
                wall = 1.02 if config == "meta" else 1.0
                origin = 1_000_000_000
                report = dict(
                    rank=rank,
                    pid=100 + rank,
                    stats=dict(
                        dcp_wall_seconds=wall,
                        dcp_started_ns=origin,
                        dcp_ended_ns=origin + int(wall * 1e9),
                        process_high_water_rss_bytes=250 + rank,
                        sampled_peak_rss_bytes=200 + rank,
                        proc_io_delta=dict(read_bytes=65536),
                    ),
                    cache_before_call=dict(
                        after_resident_buffer_preparation=True,
                        probe_end_ns=origin - 1,
                        residency=dict(data_resident_percent=99, fully_warm_data_residency=True),
                    ),
                    meta_save_source=dict(resident_buffer_bytes=64)
                    if config == "meta" and phase == "save"
                    else None,
                    meta_load_sink=dict(resident_buffer_bytes=64)
                    if config == "meta" and phase != "save"
                    else None,
                )
                raw = output / f"rank-{rank:03d}.io.jsonl"
                event = dict(
                    file="model.safetensors",
                    op="pwrite" if phase == "save" else "pread",
                    io_layer="payload",
                    offset=rank * 4 + int(changed_offset and config == "meta" and phase == "save"),
                    requested=4,
                    completed=4,
                    start_ns=10,
                    duration_ns=10,
                )
                raw.write_text(json.dumps(event) + "\n")
                trace = export_rank_trace(
                    raw,
                    output / "export",
                    rank=rank,
                    phase="save" if phase == "save" else "load",
                    origin_ns=origin,
                )
                analysis.dump(output / f"rank-{rank:03d}.json", report)
                rows.append(
                    dict(
                        model="1b",
                        config=config,
                        phase=phase,
                        rank=rank,
                        ranks=2,
                        checkpoint_format="safetensors",
                        output=str(output),
                        repeat=1,
                        export=trace,
                    )
                )
                memory_workers[str(100 + rank)] = dict(rss=150 + rank, pss=100)
            (output / "memory.jsonl").write_text(
                json.dumps(
                    dict(
                        worker_pss_bytes=200,
                        total_pss_bytes=300,
                        worker_rss_bytes=301,
                        workers=memory_workers,
                    )
                )
                + "\n"
            )
    analysis.dump(run / "measurement-results.json", rows)
    return run


def test_full_cell_gate_uses_owner_floor_and_accounts_all_rank_memory(tmp_path):
    result = analysis.analyze_cell(make_cell(tmp_path), expected_repeats=1)
    assert result["overall_pass"]
    assert result["sequence_status"] == "identical"
    assert len(result["request_comparisons"]) == 6
    assert all(phase["tolerance_s"] == 0.1 for phase in result["phases"].values())
    assert result["memory"]["meta"]["peak_worker_pss_bytes"] == 200
    assert result["memory"]["meta"]["peak_total_pss_bytes"] == 300
    assert result["memory"]["meta"]["peak_meta_buffer_bytes"] == 128
    assert result["memory"]["meta"]["per_rank_peak_rss_bytes"] == {"0": 250, "1": 251}


def test_full_request_offset_difference_is_a_hard_failure(tmp_path):
    result = analysis.analyze_cell(make_cell(tmp_path, changed_offset=True), expected_repeats=1)
    assert result["sequence_status"] == "different"
    assert result["timing_pass"]
    assert not result["overall_pass"]


def test_changed_raw_trace_cannot_reuse_old_identity_signature(tmp_path):
    run = make_cell(tmp_path)
    raw = run / "measurements/1b/r1-meta/save/rank-000.io.jsonl"
    raw.write_text(raw.read_text() + "\n")
    result = analysis.analyze_cell(run, expected_repeats=1)
    assert not result["overall_pass"]
    assert any("raw trace SHA256 changed" in error["error"] for error in result["errors"])


def test_rejected_warm_attempt_request_difference_remains_a_hard_failure(tmp_path):
    run = make_cell(tmp_path)
    accepted = next(
        row
        for row in analysis.read_json(run / "measurement-results.json")
        if row["phase"] == "warm-load" and row["config"] == "meta" and row["rank"] == 0
    )
    rejected_output = run / "measurements/1b/r1-meta/warm-load-rejected"
    rejected_output.mkdir()
    raw = rejected_output / "rank-000.io.jsonl"
    raw.write_text(
        json.dumps(
            dict(
                file="model.safetensors",
                op="pread",
                io_layer="payload",
                offset=8,
                requested=4,
                completed=4,
                start_ns=10,
                duration_ns=10,
            )
        )
        + "\n"
    )
    exported = export_rank_trace(
        raw, rejected_output / "export", rank=0, phase="load", origin_ns=1_000_000_000
    )
    report = analysis.read_json(Path(accepted["output"]) / "rank-000.json")
    report["cache_before_call"]["residency"]["data_resident_percent"] = 1
    analysis.dump(rejected_output / "rank-000.json", report)
    rejected_row = dict(accepted, output=str(rejected_output), export=exported, warm_attempt=1)
    analysis.dump(
        rejected_output.parent / "warm-load-attempts.json",
        dict(attempts=[dict(attempt=1, valid=False, rows=[rejected_row])], accepted_attempt=2),
    )
    result = analysis.analyze_cell(run, expected_repeats=1)
    assert result["warm_validity"]["all_valid"]
    assert result["sequence_status"] == "different"
    assert not result["overall_pass"]
    assert not result["warm_validity"]["rejected_samples"][0]["valid"]
    assert any(
        check.get("accepted") is False and check["status"] == "different"
        for check in result["request_comparisons"]
    )


def test_both_invalid_warm_attempts_prove_requests_without_admitting_timing(tmp_path):
    run = make_cell(tmp_path)
    all_rows = analysis.read_json(run / "measurement-results.json")
    rejected = [row for row in all_rows if row["config"] == "cpu" and row["phase"] == "warm-load"]
    redo_rows = []
    for row in rejected:
        original_output = Path(row["output"])
        rank = row["rank"]
        report = analysis.read_json(original_output / f"rank-{rank:03d}.json")
        report["cache_before_call"]["residency"]["data_resident_percent"] = 100
        report["stats"]["proc_io_delta"]["read_bytes"] = 8 * 1024**2
        analysis.dump(original_output / f"rank-{rank:03d}.json", report)
        redo = original_output.with_name("warm-load-redo1")
        redo.mkdir(exist_ok=True)
        raw = redo / f"rank-{rank:03d}.io.jsonl"
        raw.write_bytes((original_output / raw.name).read_bytes())
        exported = export_rank_trace(
            raw, redo / "export", rank=rank, phase="load", origin_ns=1_000_000_000
        )
        report["stats"]["proc_io_delta"]["read_bytes"] = int(3.4 * 1024**2)
        analysis.dump(redo / f"rank-{rank:03d}.json", report)
        redo_rows.append(dict(row, output=str(redo), export=exported, warm_attempt=2))
        (redo / "memory.jsonl").write_bytes((original_output / "memory.jsonl").read_bytes())
    analysis.dump(
        Path(rejected[0]["output"]).parent / "warm-load-attempts.json",
        dict(
            attempts=[
                dict(attempt=1, valid=False, rows=rejected),
                dict(attempt=2, valid=False, rows=redo_rows),
            ],
            accepted_attempt=None,
        ),
    )
    analysis.dump(
        run / "measurement-results.json", [row for row in all_rows if row not in rejected]
    )
    result = analysis.analyze_cell(run, expected_repeats=1)
    assert result["sequence_status"] == "identical"
    assert not result["overall_pass"]
    assert not result["warm_validity"]["all_valid"]
    assert not result["phases"]["warm-load"]["complete"]
    assert result["phases"]["warm-load"]["samples"] == []
    assert result["phases"]["warm-load"]["status"] == "unknown"
    assert analysis.phase_row(256, result, "warm-load")[-3:] == [
        "invalid/incomplete",
        "identical",
        "invalid",
    ]
    resolved = [
        check
        for check in result["request_comparisons"]
        if check.get("resolved_from_rejected_attempts")
    ]
    assert len(resolved) == 2
    assert all(len(check["rejected_attempt_evidence"]) == 2 for check in resolved)
    assert result["memory"]["cpu"]["run_count"] == 4
    assert len([record for record in result["memory"]["records"] if not record["accepted"]]) == 2
    cause = result["warm_validity"]["failures"][0]["cause"]
    assert "both permitted attempts exceeded 64 KiB" in cause
    assert "despite 100% checkpoint residency" in cause
    assert "first max 8.00 MiB/rank; redo max 3.40 MiB/rank" in cause


def test_paired_gate_uses_median_of_pair_differences_and_accepts_both_directions():
    # Independent medians differ by ten seconds, but paired median delta is .05.
    result = analysis.paired_summary({1: 1.05, 2: 20, 3: 30.05}, {1: 1, 2: 10, 3: 30}, 3)
    assert result["paired_median_delta_s"] == pytest.approx(0.05)
    assert result["status"] == "pass"
    assert analysis.paired_summary({1: 0.89}, {1: 1}, 1)["status"] == "fail"
    assert analysis.paired_summary({1: 1.11}, {1: 1}, 1)["status"] == "fail"
    assert analysis.paired_summary({1: 1}, {1: 1}, 3)["status"] == "unknown"


def test_warm_validity_requires_prep_order_and_allows_only_one_retry():
    report = dict(
        stats=dict(dcp_started_ns=20, proc_io_delta=dict(read_bytes=65536)),
        cache_before_call=dict(
            probe_end_ns=19,
            after_resident_buffer_preparation=True,
            residency=dict(data_resident_percent=99, fully_warm_data_residency=True),
        ),
    )
    row = dict(rank=0, repeat=1, config="meta", output="tiny", warm_attempt=2)
    assert analysis.warm_check(report, row)["valid"]
    assert not analysis.warm_check(report, dict(row, warm_attempt=3))["valid"]
    report["cache_before_call"]["probe_end_ns"] = 21
    assert not analysis.warm_check(report, row)["valid"]


def test_memory_pss_is_peak_simultaneous_rank_sum(tmp_path):
    path = tmp_path / "memory.jsonl"
    samples = [
        dict(
            worker_pss_bytes=110,
            total_pss_bytes=150,
            workers={"1": dict(pss=100, rss=120), "2": dict(pss=10, rss=20)},
        ),
        dict(
            worker_pss_bytes=120,
            total_pss_bytes=160,
            workers={"1": dict(pss=20, rss=30), "2": dict(pss=100, rss=130)},
        ),
    ]
    path.write_text("".join(json.dumps(sample) + "\n" for sample in samples))
    result = analysis.memory_run(path, [dict(pid=1, rank=0), dict(pid=2, rank=1)])
    assert result["peak_worker_pss_bytes"] == 120
    assert result["peak_total_pss_bytes"] == 160
    assert [rank["peak_rss_bytes"] for rank in result["ranks"]] == [120, 130]


def test_sweep_selects_smallest_complete_passing_size_without_extra_memory_gate(tmp_path):
    for size in (64, 256):
        for model in ("1b", "3b"):
            for ranks in (4, 8):
                run = tmp_path / "sweep" / f"{size}mib" / f"{model}-{ranks}r-safetensors"
                analysis.dump(run / "measurement-results.json", [])
                analysis.dump(
                    run / "analysis.json",
                    dict(analysis_version=analysis.ANALYSIS_VERSION, overall_pass=size == 256),
                )
    result = analysis.summarize_sweep(tmp_path)
    assert result["chosen_size_mib"] == 256
    assert not result["sizes"][2]["complete"]
