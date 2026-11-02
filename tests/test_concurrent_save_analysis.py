"""Exact overlap and paired-round semantics for the save-only analysis."""

import pytest

from scripts import analyze_concurrent_save as analysis


def test_intersection_preserves_disjoint_union_and_half_open_boundaries():
    assert analysis.intersection_ns([(0, 10), (20, 30)], [(5, 25)]) == 10
    assert analysis.intersection_ns([(0, 10)], [(10, 20)]) == 0
    assert analysis.intersection_ns([], [(10, 20)]) == 0


def test_concurrency_resolves_ties_without_false_overlap_and_merges_rank_calls():
    result = analysis.concurrency({0: [(0, 10), (8, 15)], 1: [(15, 20)], 2: [(10, 18)]})
    assert result["max_active_ranks"] == 2
    assert result["union_s"] == 20e-9
    assert result["multi_rank_union_s"] == 8e-9
    assert result["active_rank_seconds"] == 28e-9
    assert result["mean_active_ranks"] == pytest.approx(1.4)


def test_peer_overlap_excludes_self_and_counts_api_concurrency_exactly():
    records = [dict(rank=0, pwrite_absolute_intervals_ns=[(0, 10)], fsync_absolute_intervals_ns=[(10, 20)], raw_trace="a", raw_sha256="a"),
               dict(rank=1, pwrite_absolute_intervals_ns=[(5, 15)], fsync_absolute_intervals_ns=[(15, 25)], raw_trace="b", raw_sha256="b")]
    result = analysis.interval_diagnostics(records)
    assert result["pwrite"]["max_active_ranks"] == 2
    assert result["fsync"]["max_active_ranks"] == 2
    assert result["pwrite_fsync_overlap_union_s"] == 5e-9
    by_rank = {r["rank"]: r for r in result["per_rank"]}
    assert by_rank[0]["pwrite_overlap_peer_fsync_s"] == 0
    assert by_rank[1]["pwrite_overlap_peer_fsync_s"] == 5e-9
    assert by_rank[0]["fsync_overlap_peer_pwrite_s"] == 5e-9


def test_serial_raw_interval_reuse_verifies_hash_and_metrics(tmp_path):
    import hashlib
    import json
    report = tmp_path / "rank-000.json"
    report.write_text(json.dumps(dict(io=dict(origin_ns=100))))
    raw = report.with_suffix(".io.jsonl")
    raw.write_text(json.dumps(dict(op="pwrite", io_layer="payload", start_ns=10, duration_ns=10)) + "\n"
                   + json.dumps(dict(op="fsync", start_ns=20, duration_ns=20)) + "\n")
    case = dict(row=dict(report=str(report), rank=0), absolute_start_ns=100, absolute_end_ns=150,
                provenance=dict(trace_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()),
                metrics=dict(pwrite_s=10e-9, fsync_s=20e-9), writer_timing={})
    result = analysis.interval_record(case, raw=True)
    assert result["pwrite_absolute_intervals_ns"] == [(110, 120)]
    assert result["fsync_absolute_intervals_ns"] == [(120, 140)]
    case["provenance"]["trace_sha256"] = "changed"
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        analysis.interval_record(case, raw=True)


def test_wall_failure_uses_one_paired_disjoint_partition_not_median_components():
    parts = dict(pre_payload=.1, admission_wait=.2, payload_io=.7, materialize=0, sink=0, post_read_wait=0, uncovered=0, fsync=0, other_staging_or_io=0)
    failure = dict(failed_gates_strict=["wall"], wall_median_pair=dict(repeat=3, delta_s=.2),
                   critical_rank_breakdowns=[dict(config="meta", rank=1, repeat=3, roles=["max_wall"], breakdown=dict(disjoint_partition_s={**parts, "payload_io": .9})),
                                             dict(config="cpu", rank=2, repeat=3, roles=["max_wall"], breakdown=dict(disjoint_partition_s=parts))])
    text = analysis.failure_cause(failure)
    assert "r3" in text and "=1/2" in text
    assert "payload_io +200.0ms" in text
    assert "分区合计 +200.0ms" in text


def test_paired_core_configuration_is_save_only_and_five_complete_pairs():
    previous = analysis.core.PHASES, analysis.core.FIX1, analysis.core.EXPECTED_REPEATS, analysis.core.FLOOR_S
    try:
        analysis.configure()
        assert analysis.core.PHASES == ("save",)
        meta, cpu = {i: value for i, value in enumerate((11, 8, 30, 10, 10), 1)}, {i: value for i, value in enumerate((10, 10, 20, 10, 10), 1)}
        result = analysis.core.paired_summary(meta, cpu)
        assert result["paired_median_delta_s"] == 0
        assert result["complete"]
        assert result["floor_tolerance_s"] == .3
    finally:
        analysis.core.PHASES, analysis.core.FIX1, analysis.core.EXPECTED_REPEATS, analysis.core.FLOOR_S = previous


def test_concurrent_diagnosis_corrects_serial_assumptions_preserving_evidence():
    serial_sync = "含 fsync 的 wall 差与实测同步刷盘区间变化相符；单 writer 许可覆盖 fsync，因此该时间也传播为其他 rank 的准入/结束等待。"
    serial_admission = "变化被定位到单 writer 的实测准入等待；不应把本地 other 当作未记录的 pwrite。"
    failure = dict(inferences_zh=[serial_sync, serial_admission], interpretation_zh=serial_sync + " " + serial_admission,
                   evidence_zh=["实测 fsync 并集 +0.3000 s"], cause_zh="保留实际互斥分区归因",
                   wall_median_pair=dict(repeat=3, delta_s=.3))
    assert analysis.sanitize_concurrent_diagnoses([dict(failures=[failure])]) == 4
    assert all("单 writer" not in value for value in failure["inferences_zh"])
    assert "单 writer" not in failure["interpretation_zh"]
    assert "writer limit 等于 world size" in failure["interpretation_zh"]
    assert "内存安全准入" in failure["interpretation_zh"]
    assert failure["wall_median_pair"] == dict(repeat=3, delta_s=.3)
    assert failure["evidence_zh"] == ["实测 fsync 并集 +0.3000 s"]
    assert failure["cause_zh"] == "保留实际互斥分区归因"
    assert analysis.sanitize_concurrent_diagnoses([dict(failures=[failure])]) == 0


def test_same_pair_tail_compression_can_reverse_an_earlier_raw_completion():
    metric = dict(paired_median_delta_s=1, samples=[dict(repeat=1, delta_s=1)])
    cell = dict(phases=dict(save=dict(shape=dict(quantiles={"100": metric}))), aggregates=[
        dict(phase="save", repeat=1, config="meta", timeline=dict(last_payload_s=9, last_payload_compressed_s=6,
             uncompressed_span_s=12, global_fsync_union_s=5, duration_s=7)),
        dict(phase="save", repeat=1, config="cpu", timeline=dict(last_payload_s=10, last_payload_compressed_s=5,
             uncompressed_span_s=12, global_fsync_union_s=6, duration_s=6))])
    result = analysis.selected_quantile_diagnosis(cell, "100")
    assert result["selected_pair_repeat"] == 1
    assert result["raw_last_payload_delta_s"] == -1
    assert result["removed_before_last_payload_delta_s"] == -2
    assert result["compressed_last_payload_delta_s"] == 1
    assert result["tail_identity_error_s"] == 0
    assert result["span_identity_error_s"] == 0
    assert result["p100_bin_interpolation_delta_error_s"] == 0
