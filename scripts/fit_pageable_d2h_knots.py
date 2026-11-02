"""Refit current monotone D2H size knots from real pageable GPU save traces.

Keeps the affine branch through 593.5 MiB, then fits weighted isotonic size
medians. Inputs may be run roots, results JSON lists or raw rank IO JSONL files.
Outputs profile.json plus leave-one-repeat-out predictions; no GPU is needed.
"""

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metasl.meta_d2h_profile import LEGACY_PROFILE, duration_seconds, load_profile, validate_profile  # noqa: E402
from scripts.fit_pageable_h2d import metrics, read_samples  # noqa: E402

THRESHOLD = int(593.5 * 2**20)


def fit(rows):
    """The current knot model: unchanged small-range affine plus monotone medians."""
    if not rows or any(row["bytes"] <= 0 or row["seconds"] <= 0 or not math.isfinite(row["seconds"]) for row in rows):
        raise ValueError("D2H samples require positive sizes and finite positive durations")
    values = defaultdict(list)
    for row in rows:
        if row["bytes"] > THRESHOLD:
            values[row["bytes"]].append(row["seconds"])
    if not values:
        return validate_profile(dict(LEGACY_PROFILE))
    blocks = []
    for size, samples in sorted(values.items()):
        blocks.append(dict(sizes=[size], total=statistics.median(samples) * len(samples), n=len(samples)))
        while len(blocks) >= 2 and blocks[-2]["total"] / blocks[-2]["n"] > blocks[-1]["total"] / blocks[-1]["n"]:
            right, left = blocks.pop(), blocks.pop()
            blocks.append(dict(sizes=left["sizes"] + right["sizes"], total=left["total"] + right["total"], n=left["n"] + right["n"]))
    knots = [dict(bytes=THRESHOLD, seconds=duration_seconds(LEGACY_PROFILE, THRESHOLD))]
    for block in blocks:
        for size in block["sizes"]:
            seconds = max(block["total"] / block["n"], knots[-1]["seconds"] + (size - knots[-1]["bytes"]) * 1e-15)
            knots.append(dict(bytes=size, seconds=seconds))
    return validate_profile(dict(LEGACY_PROFILE, knots=knots))


def evaluate(rows):
    by_cohort = defaultdict(list)
    for row in rows:
        by_cohort[f"{row['format']}:{row['model']}"].append(row)
    return dict(pooled=metrics(rows), by_cohort={key: metrics(selected) for key, selected in sorted(by_cohort.items())}, large_by_size={str(size): metrics([r for r in rows if r["bytes"] == size]) for size in sorted({r["bytes"] for r in rows if r["bytes"] >= 512 * 2**20})})


def leave_repeat_out(rows):
    repeats = sorted({r["repeat"] for r in rows})
    if len(repeats) < 2:
        raise ValueError("Leave-one-repeat-out validation requires at least two repeats")
    predictions, folds = [], []
    for repeat in repeats:
        training = [r for r in rows if r["repeat"] != repeat]
        profile = fit(training)
        held = [dict(r, predicted_seconds=duration_seconds(profile, r["bytes"])) for r in rows if r["repeat"] == repeat]
        folds.append(dict(held_out_repeat=repeat, train_samples=len(training), profile=profile, **evaluate(held)))
        predictions.extend(held)
    return predictions, folds


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True, help="Run roots, results JSON lists, or raw GPU trace JSONL files")
    parser.add_argument("--output", type=Path, required=True, help="New directory for the profile and validation evidence")
    args = parser.parse_args(argv)
    rows, sources = read_samples(args.input, direction="d2h")
    profile = fit(rows)
    predictions, folds = leave_repeat_out(rows)
    held_out = evaluate(predictions)
    current = evaluate([dict(r, predicted_seconds=duration_seconds(load_profile(), r["bytes"])) for r in rows])
    args.output.mkdir(parents=True, exist_ok=False)
    dump(args.output / "profile.json", dict(profile, sources=sources, threshold_bytes=THRESHOLD, method="Size medians with weighted monotone isotonic adjustment; affine through 593.5 MiB", sample_count=len(rows)))
    dump(args.output / "cross-validation.json", dict(leave_repeat_out=held_out, current_default=current, folds=folds, sources=sources, limitations="All models, formats and ranks from a repeat are held out together. Only observed sizes and this GPU are validated. The small-range affine branch stays fixed; scheduling and end-to-end save time are not fitted."))
    with (args.output / "held-out-predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(dict(profile=profile, leave_repeat_out=held_out["pooled"]), indent=2))


if __name__ == "__main__":
    main()
