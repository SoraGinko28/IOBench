"""Consolidated analysis validates timings, byte conservation and per-rank equality."""

import json
from pathlib import Path

import pytest

from scripts import analyze_gdtensor_traces as analysis
from scripts.fit_pageable_d2h_knots import THRESHOLD, fit as fit_d2h, leave_repeat_out
from scripts.plot_gdtensor_traces import representative


def fixture_case(root, config="meta", phase="save", rank=0, world_size=1, checkpoint_format="safetensors", offset=128):
    output = root / "measurements" / "1b" / f"r1-{config}" / phase
    output.mkdir(parents=True, exist_ok=True)
    save = phase == "save"
    op = "pwrite" if save else "read" if checkpoint_format == "dcp" else "pread"
    event = dict(file="data", op=op, offset=offset, requested=64, completed=64, io_layer="payload", start_ns=200, duration_ns=50, success=True)
    transfer = dict(event, op="d2h" if save else "h2d", io_layer="staging", start_ns=260, timing="simulated" if config == "meta-sim" else "blocking_copy_host_clock")
    report = dict(rank=rank, world_size=world_size, state_device="cuda:0" if config == "gpu" else config, pattern="zero", action="save" if save else "load", checkpoint_format=checkpoint_format, io=dict(origin_ns=1000), stats=dict(dcp_started_ns=1100, dcp_ended_ns=2100, dcp_wall_seconds=.000001), **{f"gdtensor_payload_{'write' if save else 'read'}_bytes": 64})
    report_path = output / f"rank-{rank:03d}.json"
    report_path.write_text(json.dumps(report))
    report_path.with_suffix(".io.jsonl").write_text("".join(json.dumps(e) + "\n" for e in (event, transfer)))
    return dict(output=str(output), model="1b", config=config, phase=phase, repeat=1, rank=rank, run=str(root))


def test_bins_remove_only_save_fsync_union_and_conserve_bytes():
    case = dict(start_ns=0, end_ns=300_000_000, fsync=[dict(start_ns=50_000_000, duration_ns=100_000_000), dict(start_ns=100_000_000, duration_ns=100_000_000)], payload=[dict(start_ns=200_000_000, duration_ns=100_000_000, completed=100)], row=dict(phase="save"), metrics=dict(payload_bytes=100))
    result = analysis.bin_payload(case)
    assert result["duration_s"] == .15
    assert result["binned_bytes_per_50ms"] == [0., 50., 50.]
    case["row"]["phase"] = "cold-load"
    assert analysis.bin_payload(case)["duration_s"] == .3
    case["row"]["phase"] = "save"
    case["payload"][0]["start_ns"] = 190_000_000
    with pytest.raises(ValueError, match="overlaps fsync"):
        analysis.bin_payload(case)


@pytest.mark.parametrize("checkpoint_format,op", [("dcp", "read"), ("safetensors", "pread")])
def test_inspect_load_and_discovery_use_measured_format_and_rank(tmp_path, checkpoint_format, op):
    row = fixture_case(tmp_path / "run", "meta-sim", "cold-load", rank=1, world_size=4, checkpoint_format=checkpoint_format)
    discovered = list(analysis.discover_rows(tmp_path / "run"))
    assert len(discovered) == 1
    assert discovered[0]["model"] == "1b"
    assert discovered[0]["rank"] == 1
    case = analysis.inspect_case(discovered[0], tmp_path / "analysis")
    assert case["metrics"]["payload_bytes"] == 64
    assert case["metrics"]["h2d_s"] == .00000005
    assert case["payload"][0]["op"] == op
    assert case["row"]["config"] == row["config"]
    assert Path(case["export"]).is_relative_to(tmp_path / "analysis")


def test_every_rank_compares_independently_and_missing_rank_stays_unknown(tmp_path):
    cases = [analysis.inspect_case(fixture_case(tmp_path / "run", config, rank=rank, world_size=4), tmp_path / "analysis") for config in ("meta", "cpu") for rank in range(4)]
    result = analysis.compare_cases(cases)[0]
    assert result["full"]["status"] == "identical"
    assert [c["rank"] for c in result["full"]["comparisons"]] == list(range(4))
    missing = analysis.compare_cases(cases[:-1])[0]
    assert missing["payload"]["status"] == "unknown"
    changed = analysis.inspect_case(fixture_case(tmp_path / "run", "cpu", rank=3, world_size=4, offset=129), tmp_path / "analysis")
    cases[-1] = changed
    assert analysis.compare_cases(cases)[0]["full"]["status"] == "different"


def test_payload_export_selection_keeps_exact_clock_and_excludes_dcp_patches():
    geometry = dict(op="pwrite", offset=100, requested=64, completed=64)
    payload = [dict(file="data", start_ns=100, **geometry)]
    exported = [dict(path="data", start_ns=t, source=source, **geometry) for t, source in ((201, "payload_api"), (200, "buffered_file_api"), (200, "payload_api"))]
    assert analysis.select_payload_exports(payload, exported, 1000, 900) == [exported[-1]]
    with pytest.raises(ValueError, match="missing"):
        analysis.select_payload_exports(payload, exported, 1002, 900)


def test_paired_differences_compute_percent_per_repeat_and_rank():
    cases = [dict(row=dict(model="1b", checkpoint_format="dcp", world_size=1, pattern="zero", location="local", phase="warm-load", rank=0, repeat=repeat, config=config), metrics=dict(total_s=value)) for config, values in (("meta-sim", [11., 8., 30.]), ("gpu", [10., 10., 20.])) for repeat, value in enumerate(values, 1)]
    difference = analysis.paired_differences(cases)[0]
    assert difference["seconds"]["samples"] == [1., -2., 10.]
    assert difference["percent"]["median"] == pytest.approx(10.)
    representatives = [dict(row=dict(repeat=r), bins=dict(duration_s=work)) for r, work in ((1, 5.), (2, 6.), (3, 4.))]
    assert representative(representatives)["row"]["repeat"] == 1


def test_current_d2h_knots_stay_continuous_monotone_and_hold_out_whole_repeats():
    rows = [dict(bytes=size, seconds=seconds + repeat * .001, repeat=repeat, model="7b", format="dcp") for repeat in (1, 2, 3) for size, seconds in ((THRESHOLD + 100, .7), (THRESHOLD + 200, .6))]
    profile = fit_d2h(rows)
    assert len(profile["knots"]) == 3
    assert profile["knots"][1]["seconds"] < profile["knots"][2]["seconds"]
    predictions, folds = leave_repeat_out(rows)
    assert len(predictions) == 6
    assert [fold["train_samples"] for fold in folds] == [4, 4, 4]
