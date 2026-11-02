"""Cross-rank clocks, paired differences and request completeness are hard gates."""

import json

import pytest

from scripts import analyze_multirank_meta_cpu as analysis
from test_gdtensor_trace_analysis import fixture_case


def timeline_case(rank, start, end, payload, fsync=(), phase="save", origin=0):
    return dict(row=dict(rank=rank, phase=phase), origin_ns=origin,
                absolute_start_ns=origin + start, absolute_end_ns=origin + end,
                payload=[dict(start_ns=a, duration_ns=b - a, completed=size) for a, b, size in payload],
                fsync=[dict(start_ns=a, duration_ns=b - a) for a, b in fsync],
                metrics=dict(payload_bytes=sum(size for _, _, size in payload)))


def test_global_save_compression_removes_peer_fsync_and_conserves_payload():
    ns = 1_000_000
    cases = [timeline_case(0, 0, 400 * ns, [(0, 100 * ns, 100)], [(100 * ns, 200 * ns)]),
             timeline_case(1, 10 * ns, 410 * ns, [(200 * ns, 300 * ns, 100)], [(300 * ns, 400 * ns)])]
    result = analysis.aggregate_timeline(cases)
    assert result["global_fsync_union_s"] == .2
    assert result["duration_s"] == .21
    assert result["binned_bytes_per_50ms"] == [50, 50, 50, 50, 0]
    assert result["quantiles_s"]["100"] == .2
    assert result["quantiles_s"]["50"] == .1
    assert result["binned_bytes"] == 200


def test_common_origin_respects_rank_start_skew_and_absolute_clock_offsets():
    ns = 1_000_000
    cases = [timeline_case(0, 0, 200 * ns, [(0, 50 * ns, 10)], phase="warm-load", origin=1000 * ns),
             timeline_case(1, 0, 150 * ns, [(0, 50 * ns, 10)], phase="warm-load", origin=1100 * ns)]
    result = analysis.aggregate_timeline(cases)
    assert result["duration_s"] == .25
    assert result["binned_bytes_per_50ms"] == [10, 0, 10, 0, 0]
    assert result["quantiles_s"]["90"] == .14


def test_payload_overlapping_peer_fsync_is_collapsed_without_losing_bytes():
    ns = 1_000_000
    cases = [timeline_case(0, 0, 200 * ns, [(0, 100 * ns, 100)], [(50 * ns, 150 * ns)]),
             timeline_case(1, 0, 200 * ns, [(150 * ns, 200 * ns, 100)])]
    result = analysis.aggregate_timeline(cases)
    assert result["duration_s"] == .1
    assert result["binned_bytes"] == 200
    assert result["payload_fsync_overlap_s"] == .05
    assert result["payload_fsync_overlap_bytes"] == 50
    assert result["binned_bytes_per_50ms"] == [50, 150]


def test_quantiles_clip_the_last_partial_bin_and_ignore_empty_tail():
    assert analysis.quantile_times([0, 100], 75_000_000) == {"10": .0525, "50": .0625, "90": .0725, "100": .075}
    assert analysis.quantile_times([100, 0, 0], 150_000_000)["100"] == .05


def test_paired_gate_uses_paired_delta_median_divided_by_cpu_median():
    value = analysis.paired_summary({1: 11, 2: 8, 3: 30}, {1: 10, 2: 10, 3: 20})
    assert value["paired_median_delta_s"] == 1
    assert value["paired_median_delta_percent"] == 10
    assert value["status"] == "fail"
    assert analysis.paired_summary({1: .14, 2: .14, 3: .14}, {1: .1, 2: .1, 3: .1}, tiny_tolerance_s=.05)["status"] == "pass"
    assert analysis.paired_summary({1: 10}, {1: 10})["status"] == "unknown"


def test_sequence_pair_requires_every_rank_for_every_scope(tmp_path):
    cases = {config: {rank: analysis.inspect(fixture_case(tmp_path / "run", config, rank=rank, world_size=2), tmp_path / "analysis") for rank in range(2)} for config in ("meta", "cpu")}
    result = analysis.sequence_pair(cases["meta"], cases["cpu"], 2)
    assert result["full"]["status"] == result["payload"]["status"] == "identical"
    cases["cpu"].pop(1)
    assert analysis.sequence_pair(cases["meta"], cases["cpu"], 2)["full"]["status"] == "unknown"


def test_sink_and_other_breakdown_cover_wall_without_double_counting(tmp_path):
    row = fixture_case(tmp_path / "run", "meta", "warm-load")
    raw = tmp_path / "run/measurements/1b/r1-meta/warm-load/rank-000.io.jsonl"
    with raw.open("a") as stream:
        stream.write(json.dumps(dict(file="data", op="host_sink", offset=0, requested=64, completed=64, io_layer="staging", start_ns=300, duration_ns=100, request=False, success=True)) + "\n")
    case = analysis.inspect(row, tmp_path / "analysis")
    assert case["metrics"]["sink_s"] == pytest.approx(1e-7)
    assert case["metrics"]["other_s"] == pytest.approx(8.5e-7)
    assert case["metrics"]["category_union_s"] == pytest.approx(1.5e-7)


def test_reader_metadata_and_active_timing_are_separate_from_payload(tmp_path):
    row = fixture_case(tmp_path / "run", "meta", "warm-load")
    report_path = tmp_path / "run/measurements/1b/r1-meta/warm-load/rank-000.json"
    report = json.loads(report_path.read_text())
    report["reader_timing"] = dict(
        queue_enter_ns=1500, reader_enter_ns=1600, reader_return_ns=1800,
        metadata=dict(queue_enter=1.15e-6, read_start=1.25e-6, read_end=1.35e-6, prune_end=1.45e-6),
    )
    report_path.write_text(json.dumps(report))
    case = analysis.inspect(row, tmp_path / "analysis")
    metrics = case["metrics"]
    assert metrics["metadata_gate_wait_s"] == pytest.approx(1e-7)
    assert metrics["metadata_read_s"] == pytest.approx(1e-7)
    assert metrics["metadata_prune_s"] == pytest.approx(1e-7)
    assert metrics["after_metadata_to_reader_queue_s"] == pytest.approx(5e-8)
    assert metrics["reader_admission_wait_s"] == pytest.approx(1e-7)
    assert metrics["reader_active_s"] == pytest.approx(2e-7)
    assert metrics["reader_active_uncovered_s"] == pytest.approx(2e-7)
    assert "d2h" not in case["staging_operations"]
    assert case["staging_operations"]["h2d"]["event_count"] == 1


def test_failed_load_diagnosis_attributes_measured_metadata_and_onset_waits():
    cases, aggregates = [], []
    for config in ("meta", "cpu"):
        extra = .4 if config == "meta" else 0
        for repeat in (1, 2, 3):
            cases.append(dict(
                row=dict(config=config, repeat=repeat, rank=0), stats={},
                metrics=dict(total_s=2 + extra, pwrite_s=0, read_s=.8, fsync_s=0, sink_s=.2,
                             other_s=1 + extra, metadata_gate_wait_s=.1 + extra),
                staging_operations={"host_sink": dict(union_s=.2, outside_primary_s=None, event_count=None, completed_bytes=None)},
            ))
            aggregates.append(dict(config=config, repeat=repeat, timeline=dict(
                first_payload_compressed_s=.5 + extra,
                quantiles_s={str(q): q / 100 + .5 + extra for q in analysis.QUANTILES},
                duration_s=2 + extra, global_fsync_union_s=0,
            )))
    diagnosis = analysis.phase_diagnosis("warm-load", aggregates, cases)
    assert diagnosis["largest_observed_component"] == "other_s"
    assert diagnosis["paired_critical_rank_components"]["metadata_gate_wait_s"]["paired_median_delta_s"] == pytest.approx(.4)
    assert "metadata gate" in diagnosis["interpretation_zh"]
    assert "首个 payload 前" in diagnosis["interpretation_zh"]
    assert diagnosis["timeline_pairs"]["quantiles"]["50"]["paired_median_delta_ms"] == pytest.approx(400)


def test_reuse_runner_exports_verifies_provenance_and_all_requests(tmp_path):
    rows = {config: fixture_case(tmp_path / "run", config, phase="warm-load") for config in ("meta", "cpu")}
    for row in rows.values():
        report_path = tmp_path / f"run/measurements/1b/r1-{row['config']}/warm-load/rank-000.json"
        report = json.loads(report_path.read_text())
        analysis.base.export_rank_trace(report_path.with_suffix(".io.jsonl"), report_path.parent / "export", rank=0, phase="load", origin_ns=report["io"]["origin_ns"])
    cases = {config: analysis.inspect(row, tmp_path / "analysis") for config, row in rows.items()}
    for case in cases.values():
        assert case["provenance"]["full_export_reused"]
        assert case["provenance"]["raw_and_export_hashes_verified"]
        assert "/warm-load/export/clients/" in case["export"]
    result = analysis.sequence_pair({0: cases["meta"]}, {0: cases["cpu"]}, 1)
    assert result["full"]["status"] == result["payload"]["status"] == "identical"
    assert "SHA256" in result["full"]["comparisons"][0]["verification"]
    rows["cpu"] = fixture_case(tmp_path / "run", "cpu", phase="warm-load", offset=129)
    report_path = tmp_path / "run/measurements/1b/r1-cpu/warm-load/rank-000.json"
    report = json.loads(report_path.read_text())
    analysis.base.export_rank_trace(report_path.with_suffix(".io.jsonl"), report_path.parent / "export", rank=0, phase="load", origin_ns=report["io"]["origin_ns"])
    changed = analysis.inspect(rows["cpu"], tmp_path / "analysis")
    difference = analysis.sequence_pair({0: cases["meta"]}, {0: changed}, 1)
    assert difference["full"]["status"] == difference["payload"]["status"] == "different"
    for config in ("meta", "cpu"):
        manifest_path = tmp_path / f"run/measurements/1b/r1-{config}/warm-load/export/clients/rank-00000.trace.json"
        manifest = json.loads(manifest_path.read_text())
        manifest.pop("pattern_sha256")
        manifest_path.write_text(json.dumps(manifest))
    no_signature = {config: analysis.inspect(row, tmp_path / "analysis") for config, row in rows.items()}
    fallback = analysis.sequence_pair({0: no_signature["meta"]}, {0: no_signature["cpu"]}, 1)
    assert fallback["full"]["status"] == "different"
    export = tmp_path / "run/measurements/1b/r1-cpu/warm-load/export/clients/rank-00000.jsonl"
    with export.open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="export SHA256 mismatch"):
        analysis.inspect(rows["cpu"], tmp_path / "analysis")


def test_missing_sink_spans_are_unknown_instead_of_zero(tmp_path):
    row = fixture_case(tmp_path / "run", "meta", phase="warm-load")
    case = analysis.inspect(row, tmp_path / "analysis")
    assert case["metrics"]["sink_s"] is None
    assert case["metrics"]["sink_observed_s"] == 0
    assert case["metrics"]["sink_observation_complete"] is False
    assert case["metrics"]["sink_observed_bytes"] == 0
    assert case["metrics"]["sink_expected_bytes"] == 64


def test_missing_sink_bounds_use_reader_residual_without_point_estimate():
    case = dict(row=dict(phase="warm-load"), metrics=dict(
        sink_s=None, sink_observed_s=.1, sink_observation_complete=False,
        reader_active_uncovered_s=.4, other_s=.7,
    ))
    analysis.add_sink_bounds([case])
    assert case["metrics"]["sink_s"] is None
    assert case["metrics"]["sink_lower_bound_s"] == .1
    assert case["metrics"]["sink_upper_bound_s"] == .5
    assert case["metrics"]["other_true_lower_bound_s"] == pytest.approx(.3)
    assert case["metrics"]["other_true_upper_bound_s"] == .7


def test_plot_failure_preserves_verified_analysis_checkpoint(tmp_path, monkeypatch):
    summary = dict(schema_version=1, rank_cases=2016, cells=[], cases=[], plots=[])
    monkeypatch.setattr(analysis, "report_text", lambda value: "verified report\n")
    def fail_plot(*args):
        raise RuntimeError("plot environment unavailable")
    monkeypatch.setattr(analysis, "make_plots", fail_plot)
    with pytest.raises(RuntimeError, match="plot environment"):
        analysis.save_then_plot(summary, tmp_path, False)
    assert json.loads((tmp_path / "analysis.json").read_text())["rank_cases"] == 2016
    assert json.loads((tmp_path / "plot-error.json").read_text())["analysis_checkpoint_preserved"] is True


def test_stale_cell_cache_requires_explicit_resume_opt_in(tmp_path, monkeypatch):
    cache = tmp_path / "analysis-cache/1b-4r-dcp.json"
    cell = analysis.analyze_cell("1b", 4, "dcp", [])
    cell["sequence_status"] = dict(full="identical", payload="identical")
    cached = dict(schema_version=1, cell=cell, cases=[], errors=[])
    analysis.base.dump(cache, cached)
    monkeypatch.setattr(analysis, "discover", lambda *args, **kwargs: ([], []))
    monkeypatch.setattr(analysis, "save_then_plot", lambda *args: None)
    fresh = analysis.main(["--root", str(tmp_path), "--no-plots"])
    assert fresh["cells"][0]["sequence_status"]["full"] == "unknown"
    analysis.base.dump(cache, cached)
    resumed = analysis.main(["--root", str(tmp_path), "--no-plots", "--resume-cache"])
    assert resumed["cells"][0]["sequence_status"]["full"] == "identical"


def test_fix1_reports_strict_and_floor_without_accepting_four_pairs():
    meta = {r: 2.09 for r in range(1, 6)}
    cpu = {r: 2. for r in range(1, 6)}
    result = analysis.paired_summary(meta, cpu, expected_repeats=5)
    assert result["strict_status"] == "fail"
    assert result["floor_status"] == "pass"
    cpu.pop(4)
    incomplete = analysis.paired_summary(meta, cpu, expected_repeats=5)
    assert incomplete["complete"] is False
    assert incomplete["strict_status"] == incomplete["floor_status"] == "unknown"
    assert [s["repeat"] for s in incomplete["samples"]] == [1, 2, 3, 5]


def test_fix1_discovery_uses_selected_redo_and_excludes_invalid_original(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "FIX1", True)
    run = tmp_path / "matrix/1b-8r-safetensors"
    original = run / "measurements/1b/r4-cpu/warm-load"
    redo = original.with_name("warm-load-redo1")
    rows = [dict(config="cpu", phase="warm-load", output=str(p), report=str(p / "rank-000.json")) for p in (original, redo)]
    analysis.base.dump(run / "measurement-results.json", [rows[1]])
    monkeypatch.setattr(analysis.base, "discover_rows", lambda _: iter(rows))
    monkeypatch.setattr(analysis, "inspect", lambda row, _: row)
    selected, errors = analysis.discover(tmp_path, tmp_path / "analysis", run=run)
    assert errors == []
    assert [r["output"] for r in selected] == [str(redo)]


def test_sequence_only_invalid_attempts_never_fill_missing_timing_pair(monkeypatch):
    monkeypatch.setattr(analysis, "FIX1", True)
    monkeypatch.setattr(analysis, "EXPECTED_REPEATS", 5)
    cases = []
    for config in ("meta", "cpu"):
        for repeat in range(1, 6):
            for phase in analysis.PHASES:
                if config == "cpu" and repeat == 4 and phase == "warm-load":
                    continue
                case = timeline_case(0, 0, 1_000_000_000, [(100_000_000, 900_000_000, 64)], phase=phase)
                case["row"].update(config=config, repeat=repeat)
                case["metrics"].update(total_s=1., total_without_fsync_s=1.)
                cases.append(case)
    monkeypatch.setattr(analysis, "sequence_pair", lambda meta, cpu, ranks: {
        scope: dict(status="identical" if meta and cpu else "unknown", comparisons=[], errors=[])
        for scope in ("full", "payload")})
    cell = analysis.analyze_cell("1b", 1, "safetensors", cases)
    phase = cell["phases"]["warm-load"]
    assert phase["wall"]["complete"] is False
    assert phase["shape"]["strict_status"] == phase["shape"]["floor_status"] == "unknown"
    assert phase["representative_repeat"] in (1, 2, 3, 5)
    assert cell["sequence_status"]["full"] == "unknown"
    audit = dict(attempts=[dict(model="1b", ranks=1, checkpoint_format="safetensors", repeat=4,
                                config="cpu", phase="warm-load", attempt=attempt, source="attempts.json",
                                full=dict(status="identical", comparisons=[]), payload=dict(status="identical", comparisons=[]))
                          for attempt in (1, 2)])
    analysis.apply_invalid_sequence_evidence([cell], audit)
    assert cell["sequence_status"] == dict(full="identical", payload="identical")
    assert phase["wall"]["complete"] is False
    assert len(phase["wall"]["samples"]) == 4
    assert phase["wall"]["strict_status"] == phase["wall"]["floor_status"] == "unknown"
    audit["attempts"][1]["payload"]["status"] = "different"
    analysis.apply_invalid_sequence_evidence([cell], audit)
    assert cell["sequence_status"]["payload"] == "different"


def test_invalid_attempt_audit_preserves_rank_ids_and_separate_exports(tmp_path):
    run = tmp_path / "matrix/1b-4r-dcp"
    accepted = [analysis.inspect(fixture_case(run, "meta", phase="warm-load", rank=rank, world_size=2), tmp_path)
                for rank in range(2)]
    invalid = [fixture_case(run, "cpu", phase="warm-load", rank=rank, world_size=2) for rank in range(2)]
    for row in invalid:
        row["warm_validity"] = dict(valid=False, reasons=["timed block-read bytes > 0"], residency_percent=100., block_read_bytes=4096)
    analysis.base.dump(run / "measurements/1b/r1-cpu/warm-load-attempts.json",
                       dict(attempts=[dict(attempt=1, valid=False, rows=invalid)], accepted_attempt=None))
    result = analysis.audit_invalid_warm_attempts(tmp_path, accepted)
    assert result["errors"] == []
    assert result["invalid_attempts"] == 1
    assert result["invalid_rank_cases"] == 2
    assert result["all_requests_identical"] is True
    assert result["attempts"][0]["timing_eligible"] is False
    assert [r["rank"] for r in result["attempts"][0]["rank_checks"]] == [0, 1]
    assert all("invalid-warm-sequence-audit" in r["payload_export"] for r in result["attempts"][0]["rank_checks"])
