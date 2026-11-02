"""Trace comparisons retain requests and reject incomplete ordering evidence."""

import hashlib
import json
from pathlib import Path

import pytest

from metasl.trace_export import (
    REFERENCE_FIELDS,
    TraceEvidenceError,
    compare_prepared_rank_traces,
    compare_rank_traces,
    compare_trace_sets,
    export_rank_trace,
    prepare_rank_trace,
)


def event(file="__0_0.distcp", *, start=1000, duration=100, **changes):
    result = dict(
        file=file,
        op="readinto",
        offset=0,
        requested=16,
        completed=16,
        start_ns=start,
        duration_ns=duration,
        thread_id=91,
    )
    result.update(changes)
    return result


def export(tmp_path, name, rows, *, rank=0, **options):
    raw = tmp_path / f"{name}.raw.jsonl"
    raw.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return export_rank_trace(raw, tmp_path / name, rank=rank, phase="load", **options)


def read(manifest):
    return [json.loads(line) for line in Path(manifest["output_path"]).read_text().splitlines()]


def test_export_reconstructs_actual_order_and_preserves_reference_schema(tmp_path):
    result = export(
        tmp_path,
        "meta",
        [
            event("a.distcp", start=4000),
            event("b.distcp", start=1000),
            event("b.distcp", start=7000),
        ],
        origin_ns=100_000,
    )
    rows = read(result)
    assert [row["path"] for row in rows] == ["b.distcp", "a.distcp", "b.distcp"]
    assert [row["seq"] for row in rows] == [0, 1, 2]
    assert [row["t_us"] for row in rows] == [0, 3, 6]
    assert all(set(REFERENCE_FIELDS) <= row.keys() for row in rows)
    assert all(row["op"] == "readinto" and row["size"] == 16 for row in rows)
    assert result["first_event_absolute_ns"] == 101_000
    raw = Path(result["raw_source"]["path"]).read_bytes()
    assert result["raw_source"]["sha256"] == hashlib.sha256(raw).hexdigest()
    assert Path(result["manifest_path"]).is_file()


def test_shared_setup_events_remain_visible_outside_request_identity(tmp_path):
    lifecycle = [event(start=i * 100, op=op, request=False, offset=0, requested=0,
                       completed=0, success=1)
                 for i, op in enumerate(("open", "ftruncate", "barrier"))]
    left = export(tmp_path, "shared", lifecycle + [event(start=1000)])
    right = export(tmp_path, "plain", [event(start=2000)])
    assert [row["op"] for row in read(left)] == ["open", "ftruncate", "barrier", "readinto"]
    assert compare_rank_traces(left["output_path"], right["output_path"])["status"] == "identical"


def test_merge_genuinely_measured_pwrite_uses_same_absolute_clock(tmp_path):
    result = export(
        tmp_path,
        "save",
        [event(start=10), event(start=300, duration=5)],
        origin_ns=10_000,
        payload_events=[
            event(start=10_150, duration=5, op="pwrite", offset=16, success=1),
        ],
    )
    rows = read(result)
    assert [row["source"] for row in rows] == [
        "buffered_file_api",
        "payload_api",
        "buffered_file_api",
    ]
    assert [row["start_ns"] for row in rows] == [0, 140, 290]
    assert rows[1]["op"] == "pwrite"
    assert result["clock_order"]["serial_rank_order_verifiable"]


def test_absolute_payload_requires_recorder_clock_origin(tmp_path):
    with pytest.raises(TraceEvidenceError, match="requires recorder origin_ns"):
        export(tmp_path, "bad", [event()], payload_events=[event(op="pwrite")])


def test_raw_positioned_calls_are_not_duplicated_and_staging_is_auxiliary(tmp_path):
    positioned = event(start=150, duration=5, op="pwrite", io_layer="payload")
    result = export(
        tmp_path,
        "new-save",
        [
            event(start=10),
            positioned,
            event(start=30, op="d2h", io_layer="staging", request=False),
        ],
        origin_ns=10_000,
        payload_events=[dict(positioned, start_ns=10_150)],
    )
    rows = read(result)
    assert [row["op"] for row in rows] == ["readinto", "pwrite"]
    assert result["clock_order"]["serial_rank_order_verifiable"]


@pytest.mark.parametrize("phase,payload_op", [("save", "pwrite"), ("load", "pread")])
def test_safetensors_header_payload_and_manifest_keep_the_existing_request_schema(
    tmp_path, phase, payload_op,
):
    traces = {}
    for device, origin in (("meta", 10_000), ("cpu", 20_000), ("cuda", 30_000)):
        checkpoint = tmp_path / device / "checkpoint"
        payload = event(
            str(checkpoint / "shard-00001-model-00001.safetensors"),
            start=150, duration=5, op=payload_op, io_layer="payload", offset=128,
        )
        raw = tmp_path / f"{device}.raw.jsonl"
        raw.write_text("".join(json.dumps(row) + "\n" for row in (
            event(str(checkpoint / "shard-00001-model-00001.safetensors"), start=10,
                  duration=5, op="write" if phase == "save" else "read"),
            payload,
            event(str(checkpoint / "shard-00001-model-00001.safetensors"), start=30,
                  op="d2h", io_layer="staging", request=False),
            event(str(checkpoint / ".metasl-complete.json"), start=200, duration=5,
                  op="write" if phase == "save" else "read"),
        )))
        result = export_rank_trace(
            raw, tmp_path / f"export-{device}", rank=0, phase=phase,
            origin_ns=origin, checkpoint_root=checkpoint,
            payload_events=[dict(payload, start_ns=origin + 150)],
        )
        traces[device] = {0: result["output_path"]}
        rows = read(result)
        assert [row["op"] for row in rows] == [
            "write" if phase == "save" else "read", payload_op,
            "write" if phase == "save" else "read",
        ]
        assert rows[1]["offset"] == 128
        assert all(set(REFERENCE_FIELDS) <= row.keys() for row in rows)
    assert compare_trace_sets(traces, world_size=1)["status"] == "identical"


def test_per_event_origins_are_used_for_component_records(tmp_path):
    result = export(
        tmp_path,
        "components",
        [event("a.distcp", start=100, origin_ns=5000), event("b.distcp", start=0, origin_ns=1000)],
    )
    assert [row["path"] for row in read(result)] == ["b.distcp", "a.distcp"]
    assert read(result)[1]["start_ns"] == 4100


def test_mixed_clock_origins_require_explicit_common_recorder_origin(tmp_path):
    rows = [event("a.distcp", origin_ns=5000), event("b.distcp")]
    with pytest.raises(TraceEvidenceError, match="Mixed explicit and missing"):
        export(tmp_path, "mixed", rows)
    result = export(tmp_path, "bound", rows, origin_ns=1000)
    assert [row["path"] for row in read(result)] == ["b.distcp", "a.distcp"]


def test_seek_negative_displacement_does_not_become_data_bytes(tmp_path):
    result = export(
        tmp_path,
        "seek",
        [event(op="seek", requested=-16, completed=32, offset=48, whence=1)],
    )
    row = read(result)[0]
    assert (row["size"], row["requested"], row["completed"], row["whence"]) == (0, -16, 32, 1)


def test_clocks_threads_and_explicit_checkpoint_prefix_are_excluded(tmp_path):
    left = export(
        tmp_path, "meta", [event("/meta/ckpt/__0_0.distcp")], checkpoint_root="/meta/ckpt"
    )
    right = export(
        tmp_path,
        "cpu",
        [{**event("/cpu/ckpt/__0_0.distcp", start=50_000, duration=700), "thread_id": 1200}],
        checkpoint_root="/cpu/ckpt",
    )
    comparison = compare_rank_traces(left["output_path"], right["output_path"])
    assert comparison["status"] == "identical"
    assert comparison["signatures"]["left"] == comparison["signatures"]["right"]
    assert comparison["first_difference"] is None
    assert not comparison["time_or_duration_equivalence_tested"]


@pytest.mark.parametrize(
    "change,field",
    [
        ({"offset": 2}, "offset"),
        ({"completed": 15}, "completed"),
        ({"file": "__1_0.distcp"}, "path"),
        ({"op": "read"}, "op"),
    ],
)
def test_real_request_differences_are_explained(tmp_path, change, field):
    left = export(tmp_path, "meta", [event()])
    right = export(tmp_path, "cpu", [{**event(), **change}])
    comparison = compare_rank_traces(left["output_path"], right["output_path"])
    assert comparison["status"] == "different"
    assert comparison["first_difference"]["field"] == field


def test_actual_cross_file_order_is_part_of_each_ranks_signature(tmp_path):
    left = export(tmp_path, "meta", [event("a.distcp", start=1000), event("b.distcp", start=2000)])
    right = export(tmp_path, "cpu", [event("a.distcp", start=2000), event("b.distcp", start=1000)])
    comparison = compare_rank_traces(left["output_path"], right["output_path"])
    assert comparison["status"] == "different"
    assert comparison["geometry_multiset_equal"]
    assert not comparison["sequence_equal"]


@pytest.mark.parametrize("second_start", [1000, 1050])
def test_tied_or_overlapping_clocks_never_prove_serial_equality(tmp_path, second_start):
    rows = [event("a.distcp", start=1000), event("b.distcp", start=second_start)]
    left = export(tmp_path, "meta", rows)
    right = export(tmp_path, "cpu", rows)
    comparison = compare_rank_traces(left["output_path"], right["output_path"])
    assert comparison["sequence_equal"]
    assert comparison["status"] == "unknown"


def test_nanosecond_precision_avoids_false_ties_from_microsecond_export(tmp_path):
    rows = [event("a.distcp", start=1000, duration=10), event("b.distcp", start=1100, duration=10)]
    left = export(tmp_path, "meta", rows)
    right = export(tmp_path, "cpu", rows)
    assert [row["t_us"] for row in read(left)] == [0, 0]
    assert compare_rank_traces(left["output_path"], right["output_path"])["status"] == "identical"


def test_identical_failed_requests_do_not_pass_acceptance(tmp_path):
    left = export(tmp_path, "meta", [event(success=0, completed=0)])
    right = export(tmp_path, "cpu", [event(success=0, completed=0)])
    comparison = compare_rank_traces(left["output_path"], right["output_path"])
    assert comparison["sequence_equal"]
    assert comparison["status"] == "different"
    assert not comparison["successful"]


def test_empty_traces_and_missing_ranks_are_unknown(tmp_path):
    empty = export(tmp_path, "empty", [])
    assert compare_rank_traces(empty["output_path"], empty["output_path"])["status"] == "unknown"
    result = compare_trace_sets({"meta": {0: empty["output_path"]}, "cpu": {}}, world_size=2)
    assert result["status"] == "unknown"
    assert result["rank_completeness"]["meta"]["missing"] == [1]
    assert result["rank_completeness"]["cpu"]["missing"] == [0, 1]


def test_matrix_compares_every_pair_and_rank_without_cross_rank_order(tmp_path):
    traces = {}
    for device in ("meta", "cpu", "cuda"):
        traces[device] = {}
        for rank in range(2):
            result = export(
                tmp_path, f"{device}-{rank}", [event(start=(rank + 1) * 1000)], rank=rank
            )
            traces[device][rank] = result["output_path"]
    result = compare_trace_sets(traces, world_size=2)
    assert result["status"] == "identical"
    assert len(result["comparisons"]) == 6
    assert not result["cross_rank_interleaving_compared"]
    traces["cpu"][1] = traces["cpu"][0]
    result = compare_trace_sets(traces, world_size=2)
    assert result["status"] == "unknown"
    assert any("rank" in error for error in result["errors"])


@pytest.mark.parametrize(
    "changes",
    [
        {"requested": 3},
        {"whence": 0},
        {"offset": -1},
        {"op": "nn_getfileinfo"},
        {"duration_ns": -1},
    ],
)
def test_impossible_or_fabricated_source_events_are_rejected(tmp_path, changes):
    with pytest.raises(TraceEvidenceError):
        export(tmp_path, "invalid", [{**event(), **changes}])


def test_truncated_jsonl_and_tampered_export_fail_closed(tmp_path):
    raw = tmp_path / "truncated.jsonl"
    raw.write_text('{"op": "read"')
    with pytest.raises(TraceEvidenceError, match="Invalid JSON"):
        export_rank_trace(raw, tmp_path / "out", rank=0, phase="load")
    valid = export(tmp_path, "valid", [event()])
    tampered = read(valid)
    tampered[0]["size"] = 999
    path = Path(valid["output_path"])
    path.write_text(json.dumps(tampered[0]) + "\n")
    with pytest.raises(TraceEvidenceError, match="size"):
        compare_rank_traces(path, path)


def test_absolute_paths_cannot_silently_drop_unknown_components(tmp_path):
    with pytest.raises(TraceEvidenceError, match="requires checkpoint_root"):
        export(tmp_path, "absolute", [event("/unbound/__0_0.distcp")])
    with pytest.raises(TraceEvidenceError, match="outside"):
        export(tmp_path, "outside", [event("/outside/__0_0.distcp")], checkpoint_root="/checkpoint")


def test_lifecycle_intervals_are_retained_but_excluded_from_request_identity(tmp_path):
    requests = [event(start=1000), event(start=2000)]
    reference = export(tmp_path, "reference", requests)
    lifecycle = [
        event(
            start=1050,
            duration=5000,
            op=op,
            request=False,
            offset=0,
            requested=0,
            completed=0,
        )
        for op in ("fsync", "flush", "close")
    ]
    traced = export(tmp_path, "traced", requests + lifecycle)
    rows = read(traced)
    auxiliary = [row for row in rows if not row["request"]]
    assert len(auxiliary) == 3
    assert traced["event_count"] == 5 and traced["request_event_count"] == 2
    assert traced["auxiliary_event_count"] == 3
    assert traced["pattern_sha256"] == reference["pattern_sha256"]
    for row in auxiliary:
        assert row["size"] == 0
        assert row["end_ns"] == row["start_ns"] + row["duration_ns"] == 5050
        assert row["t_us"] == 0 and row["cost_us"] == 5
    comparison = compare_rank_traces(reference["output_path"], traced["output_path"])
    assert comparison["status"] == "identical"
    assert comparison["serial_rank_order_verifiable"]
    assert comparison["event_counts"] == {"left": 2, "right": 2}
    assert comparison["auxiliary_event_counts"] == {"left": 0, "right": 3}


@pytest.mark.parametrize("changes", [{"request": True}, {"completed": 1}, {"offset": 1}])
def test_lifecycle_events_cannot_masquerade_as_requests(tmp_path, changes):
    row = event(op="fsync", request=False, offset=0, requested=0, completed=0)
    with pytest.raises(TraceEvidenceError):
        export(tmp_path, "invalid-sync", [{**row, **changes}])


def test_lifecycle_only_trace_cannot_establish_request_identity(tmp_path):
    traced = export(
        tmp_path,
        "sync-only",
        [event(op="fsync", request=False, requested=0, completed=0)],
    )
    assert compare_rank_traces(traced["output_path"], traced["output_path"])["status"] == "unknown"


@pytest.mark.parametrize(
    "rows",
    [
        [event()],
        [event(offset=1)],
        [event(success=0, completed=0)],
        [event(start=1000), event(start=1050)],
    ],
)
def test_prepared_comparisons_retain_all_evidence_without_reopening_files(tmp_path, rows):
    left = export(tmp_path, "reference", [event()])
    right = export(tmp_path, "candidate", rows)
    expected = compare_rank_traces(left["output_path"], right["output_path"])
    prepared_left = prepare_rank_trace(left["output_path"])
    prepared_right = prepare_rank_trace(right["output_path"])
    Path(left["output_path"]).unlink()
    Path(right["output_path"]).unlink()
    assert compare_prepared_rank_traces(prepared_left, prepared_right) == expected
