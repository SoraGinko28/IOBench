"""H2D profile fit, held-out repeat isolation, and measured-event selection."""

import json

import pytest

from metasl.meta_h2d_profile import (
    DEFAULT_PROFILE,
    duration_seconds,
    load_profile,
)
from scripts.fit_pageable_h2d import fit, leave_repeat_out, metrics, read_samples


def sample(size, seconds, repeat=1, phase="cold-load", model="1b", format="safetensors"):
    return dict(bytes=size, seconds=seconds, repeat=repeat, phase=phase, model=model, format=format)


def test_default_is_fresh_and_has_expected_blocking_budget():
    profile = load_profile()
    assert profile == DEFAULT_PROFILE
    assert duration_seconds(profile, 0) == DEFAULT_PROFILE["a_seconds"]
    assert duration_seconds(profile, 2**30) == pytest.approx(0.2344698679251225)
    profile["a_seconds"] = 0.9
    assert load_profile() == DEFAULT_PROFILE


def test_override_supports_continuous_size_nodes(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(
        json.dumps(
            dict(
                a_seconds=0.1,
                bandwidth_bytes_per_s=1000,
                knots=[dict(bytes=100, seconds=0.2), dict(bytes=200, seconds=0.4)],
                evidence="ignored by runtime",
            )
        )
    )
    profile = load_profile(path)
    assert duration_seconds(profile, 50) == pytest.approx(0.15)
    assert duration_seconds(profile, 150) == pytest.approx(0.3)
    assert duration_seconds(profile, 300) == pytest.approx(0.6)


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {},
        dict(a_seconds=-1, bandwidth_bytes_per_s=1000),
        dict(a_seconds=0, bandwidth_bytes_per_s=0),
        dict(a_seconds=True, bandwidth_bytes_per_s=1000),
    ],
)
def test_invalid_overrides_report_h2d_context(tmp_path, value):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Invalid Meta simulated H2D profile"):
        load_profile(path)


def test_affine_fit_recovers_latency_and_bandwidth():
    rows = [sample(size, 0.0001 + size / 4e9) for size in (512, 4096, 2**20, 2**30)]
    profile = fit(rows)
    assert profile["a_seconds"] == pytest.approx(0.0001)
    assert profile["bandwidth_bytes_per_s"] == pytest.approx(4e9)
    for row in rows:
        assert duration_seconds(profile, row["bytes"]) == pytest.approx(row["seconds"])


@pytest.mark.parametrize(
    "rows",
    [
        [],
        [sample(1, 1)],
        [sample(1, 1), sample(1, 2)],
        [sample(1, 2), sample(2, 1)],
        [sample(1, 1), sample(2, float("nan"))],
    ],
)
def test_fit_rejects_insufficient_or_nonphysical_samples(rows):
    with pytest.raises(ValueError):
        fit(rows)


def test_leave_repeat_out_removes_all_phases_models_and_formats():
    rows = [
        sample(size, intercept + size / 1000, repeat, phase, model, format)
        for repeat, intercept in ((1, 0.1), (2, 0.2), (3, 0.3))
        for phase in ("cold-load", "warm-load")
        for model in ("1b", "7b")
        for format in ("safetensors", "dcp")
        for size in (100, 200)
    ]
    predictions, folds = leave_repeat_out(rows)
    assert len(predictions) == len(rows)
    assert len(folds) == 3
    assert [fold["train_samples"] for fold in folds] == [32, 32, 32]
    assert [fold["profile"]["a_seconds"] for fold in folds] == pytest.approx([0.25, 0.2, 0.15])
    for row in predictions:
        intercept = {1: 0.25, 2: 0.2, 3: 0.15}[row["repeat"]]
        assert row["predicted_seconds"] == pytest.approx(intercept + row["bytes"] / 1000)


def test_metrics_report_ms_and_mean_absolute_relative_error():
    rows = [
        dict(sample(100, 0.1), predicted_seconds=0.101),
        dict(sample(200, 0.2), predicted_seconds=0.198),
    ]
    result = metrics(rows)
    assert result["median_abs_ms"] == pytest.approx(1.5)
    assert result["p95_abs_ms"] == pytest.approx(1.95)
    assert result["mean_abs_percent"] == pytest.approx(1)
    assert result["signed_mean_ms"] == pytest.approx(-0.5)


def test_read_samples_only_local_pageable_successful_measured_complete_ranges(tmp_path):
    output = tmp_path / "r1-gpu" / "cold-load"
    output.mkdir(parents=True)
    event = dict(
        io_layer="staging",
        op="h2d",
        timing="blocking_copy_host_clock",
        success=True,
        requested=4096,
        completed=4096,
        duration_ns=50_000,
        file="shard.safetensors",
    )
    events = [
        event,
        dict(event, timing="simulated"),
        dict(event, success=False),
        dict(event, completed=2048),
        dict(event, op="d2h"),
        dict(event, duration_ns=0),
    ]
    path = output / "rank-000.io.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in events))
    selected = dict(
        device="cuda",
        mode="pageable",
        phase="cold-load",
        repeat=1,
        model="1b",
        output="r1-gpu/cold-load",
    )
    result_path = tmp_path / "results.json"
    result_path.write_text(
        json.dumps(
            [
                selected,
                dict(selected, mode="pinned", output="missing"),
                dict(selected, location="nfs", output="missing"),
            ]
        )
    )
    rows, sources = read_samples([result_path])
    assert len(rows) == 1
    assert rows[0]["format"] == "safetensors"
    assert rows[0]["seconds"] == pytest.approx(0.00005)
    assert rows[0]["line"] == 1
    assert sources[0]["samples"] == 1
    assert sources[0]["format"] == "safetensors"
    assert len(sources[0]["sha256"]) == 64
