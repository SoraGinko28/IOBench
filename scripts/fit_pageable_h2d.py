"""Fit measured pageable H2D ranges and hold out entire interleaved repeats.

No GPU, tensor payload, or checkpoint mutation is required. Inputs are results
JSON files, run directories, or explicit rank event JSONL files. Both checkpoint formats and all recorded ranks are supported.
"""

import argparse
import csv
import hashlib
import json
import math
import re
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from metasl.meta_h2d_profile import duration_seconds, validate_profile  # noqa: E402

LARGE_BYTES = 512 * 2**20


def quantile(values, fraction):
    ordered = sorted(values)
    if not ordered:
        raise ValueError("Cannot evaluate an empty sample set")
    position = (len(ordered) - 1) * fraction
    lower = math.floor(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def fit(samples):
    """Ordinary least squares: fixed overhead plus bytes / bandwidth."""
    if len(samples) < 2:
        raise ValueError("At least two H2D samples with distinct sizes are required")
    x = [sample["bytes"] for sample in samples]
    y = [sample["seconds"] for sample in samples]
    if any(size <= 0 for size in x) or any(
        seconds <= 0 or not math.isfinite(seconds) for seconds in y
    ):
        raise ValueError("H2D samples require positive sizes and finite positive durations")
    mean_x, mean_y = st.fmean(x), st.fmean(y)
    variance = sum((size - mean_x) ** 2 for size in x)
    if not variance:
        raise ValueError("H2D samples must contain distinct sizes")
    slope = (
        sum((size - mean_x) * (seconds - mean_y) for size, seconds in zip(x, y, strict=True))
        / variance
    )
    intercept = mean_y - slope * mean_x
    if slope <= 0 or intercept < 0 or not all(math.isfinite(v) for v in (slope, intercept)):
        raise ValueError(f"Invalid nonnegative H2D fit: a={intercept}, slope={slope}")
    return validate_profile(dict(a_seconds=intercept, bandwidth_bytes_per_s=1 / slope))


def metrics(rows, prediction="predicted_seconds"):
    errors = [row[prediction] - row["seconds"] for row in rows]
    absolute = [abs(error) * 1000 for error in errors]
    return dict(
        samples=len(rows),
        signed_mean_ms=st.fmean(errors) * 1000,
        mean_abs_ms=st.fmean(absolute),
        median_abs_ms=quantile(absolute, 0.5),
        p95_abs_ms=quantile(absolute, 0.95),
        rms_ms=math.sqrt(st.fmean(error**2 for error in errors)) * 1000,
        mean_abs_percent=st.fmean(
            abs(row[prediction] - row["seconds"]) / row["seconds"] * 100 for row in rows
        ),
    )


def evaluate(rows):
    groups = defaultdict(list)
    large_sizes = defaultdict(list)
    for row in rows:
        groups[f"{row['format']}:{row['model']}:{row['phase']}"].append(row)
        if row["bytes"] >= LARGE_BYTES:
            large_sizes[row["bytes"]].append(row)
    return dict(
        pooled=metrics(rows),
        by_cohort={name: metrics(selected) for name, selected in sorted(groups.items())},
        by_size={
            str(size): metrics([row for row in rows if row["bytes"] == size])
            for size in sorted({row["bytes"] for row in rows})
        },
        large_by_size={
            str(size): metrics(selected) for size, selected in sorted(large_sizes.items())
        },
        large_by_cohort_and_size={
            name: {
                str(size): metrics([row for row in selected if row["bytes"] == size])
                for size in sorted(
                    {row["bytes"] for row in selected if row["bytes"] >= LARGE_BYTES}
                )
            }
            for name, selected in sorted(groups.items())
        },
    )


def leave_repeat_out(samples):
    """Remove the repeat from every model, phase, and format together."""
    predictions, folds = [], []
    repeats = sorted({row["repeat"] for row in samples}, key=str)
    if len(repeats) < 2:
        raise ValueError("Leave-one-repeat-out evaluation requires at least two repeats")
    for repeat in repeats:
        train = [row for row in samples if row["repeat"] != repeat]
        held = [row for row in samples if row["repeat"] == repeat]
        profile = fit(train)
        predicted = [
            dict(row, predicted_seconds=duration_seconds(profile, row["bytes"])) for row in held
        ]
        folds.append(
            dict(
                held_out_repeat=repeat,
                train_samples=len(train),
                profile=profile,
                **evaluate(predicted),
            )
        )
        predictions.extend(predicted)
    return predictions, folds


def event_sources(inputs, direction="h2d"):
    """Select real pageable GPU save/load traces, including every measured rank."""
    from scripts.analyze_gdtensor_traces import discover_rows

    sources = {}

    def add_rows(rows, parent):
        for row in rows:
            phase = row.get("phase", "save")
            device = row.get("device", row.get("config"))
            if (
                device not in ("cuda", "gpu")
                or (direction == "h2d" and "load" not in phase)
                or (direction == "d2h" and phase != "save")
                or row.get("mode") not in (None, "pageable")
                or row.get("location", "local") not in ("local", "unknown")
            ):
                continue
            output = Path(row["output"])
            if not output.is_absolute():
                candidate = Path.cwd() / output
                output = candidate if candidate.exists() else parent / output
            for path in output.glob("rank-*.io.jsonl"):
                sources[path.resolve()] = dict(
                    repeat=int(row.get("repeat", 1)), model=row.get("model", "unknown"),
                    phase=phase, format=row.get("checkpoint_format"),
                )

    for value in inputs:
        source = Path(value).resolve()
        if source.is_dir():
            manifests = list(source.rglob("measurement-results.json")) + list(source.rglob("results.json"))
            for manifest in manifests:
                data = json.loads(manifest.read_text())
                if isinstance(data, list):
                    add_rows(data, manifest.parent)
            if not manifests:
                add_rows(discover_rows(source), source)
        elif source.suffix == ".jsonl":
            match = re.search(r"(?:^|[-/])r(\d+)(?:[-/]|$)", str(source))
            sources[source] = dict(repeat=int(match[1]) if match else 1, model="unknown", phase=source.parent.name)
        else:
            data = json.loads(source.read_text())
            if not isinstance(data, list):
                raise ValueError(f"Expected a results list: {source}")
            add_rows(data, source.parent)
    return sorted(sources.items(), key=lambda item: str(item[0]))


def read_samples(inputs, direction="h2d"):
    rows, sources = [], []
    for path, metadata in event_sources(inputs, direction):
        digest, count = hashlib.sha256(), 0
        with path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if ('"' + direction + '"').encode() not in line:
                    continue
                event = json.loads(line)
                if (
                    event.get("io_layer") != "staging"
                    or event.get("op") != direction
                    or event.get("timing") != "blocking_copy_host_clock"
                    or not event.get("success", True)
                ):
                    continue
                size = event.get("completed", event.get("requested", 0))
                duration_ns = event["duration_ns"]
                if size <= 0 or size != event.get("requested", size) or duration_ns <= 0:
                    continue
                checkpoint_format = metadata.get("format") or (
                    "safetensors" if event.get("file", "").endswith(".safetensors") else "dcp"
                )
                metadata["format"] = checkpoint_format
                rows.append(dict(metadata, format=checkpoint_format, bytes=size, seconds=duration_ns / 1e9, source=str(path), line=line_number))
                count += 1
        if not count:
            raise ValueError(f"No successful measured blocking pageable {direction.upper()} ranges in {path}")
        sources.append(dict(path=str(path), sha256=digest.hexdigest(), samples=count, **metadata))
    if not rows:
        raise ValueError(f"No local pageable GPU {direction.upper()} traces selected")
    return rows, sources


def dump(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        nargs="+",
        type=Path,
        required=True,
        help="Run roots, results JSON lists, or raw GPU trace JSONL files",
    )
    parser.add_argument("--output", type=Path, required=True, help="New fit artifact directory")
    args = parser.parse_args()
    rows, sources = read_samples(args.input)
    profile = fit(rows)
    in_sample = evaluate(
        [dict(row, predicted_seconds=duration_seconds(profile, row["bytes"])) for row in rows]
    )
    predictions, folds = leave_repeat_out(rows)
    held_out = evaluate(predictions)
    eligible = held_out["pooled"]["median_abs_ms"] <= 0.5 and all(
        item["mean_abs_percent"] <= 3 for item in held_out["large_by_size"].values()
    )
    args.output.mkdir(parents=True, exist_ok=False)
    dump(
        args.output / "profile.json",
        dict(
            profile,
            model="duration_seconds = a_seconds + bytes / bandwidth_bytes_per_s",
            method="Ordinary least squares on measured blocking local pageable H2D ranges",
            timing="blocking_copy_host_clock",
            size_min_bytes=min(row["bytes"] for row in rows),
            size_max_bytes=max(row["bytes"] for row in rows),
            distinct_sizes=len({row["bytes"] for row in rows}),
            sample_count=len(rows),
            sources=sources,
            size_knots_required=False,
            size_knots_reason=(
                "Affine leave-repeat-out passes both gates; no size nodes needed"
                if eligible
                else "Affine leave-repeat-out fails gates; investigate size nodes"
            ),
        ),
    )
    dump(
        args.output / "cross-validation.json",
        dict(
            eligible_for_default=eligible,
            large_threshold_bytes=LARGE_BYTES,
            in_sample=in_sample,
            leave_repeat_out=held_out,
            folds=folds,
            sources=sources,
            limitations="All cold/warm/model/format rows from a repeat are held out together. "
            "Other sizes/GPUs are unvalidated; held-out repeats only cover observed sizes. "
            "Predictions exclude scheduling overruns and do not establish end-to-end agreement.",
        ),
    )
    with (args.output / "held-out-predictions.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
        writer.writeheader()
        writer.writerows(predictions)
    print(
        json.dumps(
            dict(
                profile=profile,
                eligible_for_default=eligible,
                in_sample=in_sample["pooled"],
                leave_repeat_out=held_out["pooled"],
                large_by_size=held_out["large_by_size"],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
