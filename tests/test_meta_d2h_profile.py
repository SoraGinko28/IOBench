"""Budget continuity, legacy compatibility, and malformed calibration rejection."""

import json

import pytest

from metasl.meta_d2h_profile import (
    LEGACY_PROFILE,
    duration_seconds,
    load_profile,
    validate_profile,
)


def example():
    return dict(
        a_seconds=0.1,
        bandwidth_bytes_per_s=1000,
        knots=[
            dict(bytes=100, seconds=0.2),
            dict(bytes=200, seconds=0.4),
            dict(bytes=400, seconds=0.7),
        ],
    )


@pytest.mark.parametrize(
    "size,seconds",
    [
        (0, 0.1),
        (50, 0.15),
        (100, 0.2),
        (150, 0.3),
        (200, 0.4),
        (300, 0.55),
        (400, 0.7),
        (500, 0.85),
    ],
)
def test_piecewise_budget_boundaries_and_extrapolation(size, seconds):
    assert duration_seconds(validate_profile(example()), size) == pytest.approx(seconds)


@pytest.mark.parametrize(
    "knots",
    [
        [],
        [dict(bytes=100, seconds=0.2)],
        "bad",
        [None, None],
        [dict(bytes=True, seconds=0.2), dict(bytes=200, seconds=0.4)],
        [dict(bytes=100, seconds=0.21), dict(bytes=200, seconds=0.4)],
        [dict(bytes=100, seconds=0.2), dict(bytes=100, seconds=0.4)],
        [dict(bytes=100, seconds=0.2), dict(bytes=200, seconds=0.1)],
        [dict(bytes=100, seconds=0.2), dict(bytes=200, seconds=float("inf"))],
        [dict(bytes=100, seconds=0.2), dict(bytes=200, seconds=True)],
        [dict(bytes=100, seconds=0.2), dict(bytes=200.5, seconds=0.4)],
        [dict(bytes=100, seconds=0.2), dict(bytes=200)],
    ],
)
def test_invalid_calibration_profile(tmp_path, knots):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(dict(example(), knots=knots)))
    with pytest.raises(ValueError, match="Invalid Meta simulated D2H profile"):
        load_profile(path)


def test_default_preserves_1b_3b_and_legacy_remains_selectable():
    default, old = load_profile(), load_profile("legacy")
    assert old == LEGACY_PROFILE
    for size in (1024, 4096, 2**20, 8 * 2**20, 125 * 2**20, 135790592, 622329856):
        assert duration_seconds(default, size) == duration_seconds(old, size)
    assert default["knots"][0]["bytes"] == 622329856
    assert default["knots"][-1]["bytes"] == 2**31
    assert duration_seconds(default, 1089994752) > duration_seconds(old, 1089994752)
    default["knots"][0]["seconds"] = 10
    assert load_profile()["knots"][0]["seconds"] != 10
