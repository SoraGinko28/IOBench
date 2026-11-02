"""Pageable D2H budgets: legacy affine model or calibrated size knots.

The affine branch is preserved through the first knot. Above it, interpolate
duration in bytes; beyond the last measured size extrapolate the last segment.
No CUDA or filesystem activity is performed when evaluating a budget.
"""

import bisect
import json
import math

LEGACY_PROFILE = {"a_seconds": 0.00021, "bandwidth_bytes_per_s": 1.6e9}
# V100 fresh-anonymous pageable calibration, 2026-10-28. The old branch is
# unchanged through the largest 3B range; 29 held-out repeat groups qualify this
# default. Regression evidence: validation/gdt-meta-d2h-repeats-20261028/.
DEFAULT_KNOTS = (
    (622329856, 0.38916616),
    (671088640, 0.46168982049999996),
    (738197504, 0.5066564915),
    (805306368, 0.5541491519999999),
    (872415232, 0.6016629335),
    (939524096, 0.6452419764999999),
    (1006632960, 0.691472139),
    (1089994752, 0.7386827705),
    (1140850688, 0.7857493335),
    (1207959552, 0.8356480275),
    (1342177280, 0.92007279),
    (1476395008, 1.0140541645),
    (1610612736, 1.0974673105),
    (1879048192, 1.2770811235),
    (2147483648, 1.4526768964999999),
)


def validate_profile(profile):
    if not isinstance(profile, dict):
        raise ValueError("expected a JSON object")
    result = {}
    for key in ("a_seconds", "bandwidth_bytes_per_s"):
        value = profile[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{key} must be a number")
        value = float(value)
        positive = key == "bandwidth_bytes_per_s"
        if not math.isfinite(value) or value < 0 or (positive and not value):
            bound = "positive" if positive else "nonnegative"
            raise ValueError(f"{key} must be finite and {bound}")
        result[key] = value
    if "knots" in profile:
        knots = profile["knots"]
        if not isinstance(knots, list) or len(knots) < 2:
            raise ValueError("knots must be a list with at least two calibration points")
        normalized = []
        for knot in knots:
            if not isinstance(knot, dict):
                raise ValueError("each knot must be an object with bytes and seconds")
            size, seconds = knot["bytes"], knot["seconds"]
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise ValueError("knot bytes must be a positive integer")
            if (
                isinstance(seconds, bool)
                or not isinstance(seconds, (int, float))
                or not math.isfinite(seconds)
                or seconds <= 0
            ):
                raise ValueError("knot seconds must be finite and positive")
            if normalized and (
                size <= normalized[-1]["bytes"] or seconds <= normalized[-1]["seconds"]
            ):
                raise ValueError("knot bytes and seconds must be strictly increasing")
            normalized.append(dict(bytes=size, seconds=float(seconds)))
        first = normalized[0]
        base = result["a_seconds"] + first["bytes"] / result["bandwidth_bytes_per_s"]
        if not math.isclose(first["seconds"], base, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError("first knot must meet the affine branch continuously")
        result["knots"] = normalized
    return result


def load_profile(path=None):
    try:
        if not path:
            profile = dict(LEGACY_PROFILE)
            if DEFAULT_KNOTS:
                profile["knots"] = [
                    dict(bytes=size, seconds=value) for size, value in DEFAULT_KNOTS
                ]
        elif str(path) == "legacy":
            profile = dict(LEGACY_PROFILE)
        else:
            with open(path, encoding="utf-8") as stream:
                profile = json.load(stream)
        return validate_profile(profile)
    except (OSError, ValueError, KeyError, OverflowError, TypeError) as error:
        raise ValueError(f"Invalid Meta simulated D2H profile {path!r}: {error}") from error


def duration_seconds(profile, size):
    knots = profile.get("knots")
    if not knots or size <= knots[0]["bytes"]:
        return profile["a_seconds"] + size / profile["bandwidth_bytes_per_s"]
    index = min(bisect.bisect_left([k["bytes"] for k in knots], size), len(knots) - 1)
    left, right = knots[index - 1], knots[index]
    fraction = (size - left["bytes"]) / (right["bytes"] - left["bytes"])
    return left["seconds"] + fraction * (right["seconds"] - left["seconds"])
