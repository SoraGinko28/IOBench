"""Recover complete pilot summaries after the measurement driver has exited.

Only the new multirank output's derived JSON summaries are changed. Measurements,
raw traces and original error evidence are preserved; no checkpoint is rerun.
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_multirank_meta_cpu import paired_summary, warm_request_identity  # noqa: E402

SUMMARY_ERROR = "dict() got multiple values for keyword argument 'rank'"
PILOT_LABELS = tuple(f"{model}-{fmt}" for model in ("1b", "3b")
                     for fmt in ("dcp", "safetensors"))
CONFIGS = ("cpu", "meta-faithful", "meta-ring")


def read(path):
    return json.loads(path.read_text())


def atomic_write(path, value):
    temporary = path.with_name(path.name + ".repair.tmp")
    with temporary.open("x") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    temporary.replace(path)


def require_driver_stopped(progress):
    pid = progress.get("pid")
    if type(pid) is int and pid > 0:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError(f"Measurement PID {pid} is still alive; repair only after it exits")


def matching_error(error):
    return (error.get("stage") == "sink-pilot"
            and error.get("cell") in PILOT_LABELS
            and error.get("error") == SUMMARY_ERROR)


def reconstruct_pilot(directory, label):
    model, fmt = label.split("-", 1)
    progress = read(directory / "progress.json")
    if progress.get("status") == "failed" and progress.get("error") != SUMMARY_ERROR:
        raise ValueError(f"{label}: failure is not the known summary-only error")
    cleanup = read(directory / "cleanup.json")
    if not cleanup.get("checkpoints_removed") or cleanup.get("remaining"):
        raise ValueError(f"{label}: checkpoint cleanup is incomplete")
    rows = read(directory / "measurement-results.json")
    expected = {(config, repeat, rank) for config in CONFIGS
                for repeat in range(1, 4) for rank in range(4)}
    identities = [(row["config"], row["repeat"], row["rank"]) for row in rows]
    if len(identities) != len(expected) or set(identities) != expected:
        raise ValueError(f"{label}: expected exactly all 36 config/repeat/rank rows")
    for row in rows:
        if (row.get("model") != model or row.get("checkpoint_format") != fmt
                or row.get("phase") != "warm-load" or row.get("ranks") != 4):
            raise ValueError(f"{label}: mismatched row metadata")
        if not math.isfinite(row["dcp_seconds"]) or row["dcp_seconds"] <= 0:
            raise ValueError(f"{label}: invalid measured duration")
        export = row["export"]
        trace = Path(export["output_path"])
        if hashlib.sha256(trace.read_bytes()).hexdigest() != export["output_sha256"]:
            raise ValueError(f"{label}: exported trace checksum changed: {trace}")
    times = {config: [max(row["dcp_seconds"] for row in rows
                         if row["config"] == config and row["repeat"] == repeat)
                      for repeat in range(1, 4)] for config in CONFIGS}
    result = dict(model=model, checkpoint_format=fmt,
                  cpu_median_s=statistics.median(times["cpu"]),
                  cpu_samples_s=times["cpu"])
    for sink in ("faithful", "ring"):
        config = "meta-" + sink
        result[sink] = dict(**paired_summary(times[config], times["cpu"]),
                            **warm_request_identity(rows, config))
    return result, progress, cleanup


def repair(root):
    root = root.resolve()
    if root != (ROOT / "validation/multirank-meta-cpu-20261029").resolve():
        raise ValueError("Repair is restricted to the new multirank-meta-cpu-20261029 output")
    root_progress = read(root / "progress.json")
    require_driver_stopped(root_progress)
    correction_path = root / "pilot-summary-correction.json"
    if correction_path.exists():
        raise ValueError("Correction evidence already exists; refusing to overwrite history")
    if root_progress.get("stage") == "sink-pilot":
        raise ValueError("Driver did not finish its pilot stage")
    original_selection = read(root / "sink-selection.json")
    originals, reconstructed, progress_updates = {}, [], {}
    for label in PILOT_LABELS:
        directory = root / "sink-pilot" / label
        result, old_progress, cleanup = reconstruct_pilot(directory, label)
        originals[label] = dict(progress=old_progress, cleanup=cleanup,
                                pilot_result=read(directory / "pilot-result.json")
                                if (directory / "pilot-result.json").exists() else None)
        updated = dict(old_progress, status="stage-complete", current=None,
                       updated_unix=time.time(),
                       recovered_from=dict(error=old_progress.get("error"),
                                           correction=str(correction_path)))
        updated.pop("error", None)
        progress_updates[label] = updated
        reconstructed.append(result)
    ring_pass = all(cell["ring"]["requests_identical"]
                    and abs(cell["ring"]["paired_delta_percent"]) <= 3
                    for cell in reconstructed)
    recommendation = "ring" if ring_pass else "faithful"
    # The matrix has already run with its recorded selection. Preserve that
    # measured configuration even if reconstructed pilot advice differs.
    selection = dict(original_selection, cells=reconstructed,
                     failed_pilots=[error for error in original_selection.get("failed_pilots", [])
                                    if not matching_error(error)],
                     pilot_recommended_sink=recommendation,
                     original_selected_sink=original_selection["selected_sink"],
                     selected_sink=original_selection["selected_sink"],
                     recovered_from=str(correction_path))
    updated_progress = dict(root_progress)
    updated_progress["failed"] = [error for error in root_progress.get("failed", [])
                                  if not matching_error(error)]
    updated_progress["completed"] = list(root_progress.get("completed", []))
    for label in PILOT_LABELS:
        key = "pilot-" + label
        if key not in updated_progress["completed"]:
            updated_progress["completed"].append(key)
    updated_progress.update(updated_unix=time.time(),
                            pilot_summary_recovery=str(correction_path))
    correction = dict(schema_version=1, status="verified-before-summary-writes",
                      created_unix=time.time(),
                      explanation="Measurements succeeded; pilot postprocessing failed because the comparison already returned rank. Complete saved rows and exact trace comparisons reconstruct the summaries without rerunning measurements.",
                      original_root_progress=root_progress,
                      original_sink_selection=original_selection,
                      original_pilots=originals,
                      reconstructed_pilot_results=reconstructed,
                      removed_summary_errors=[error for error in root_progress.get("failed", [])
                                              if matching_error(error)],
                      measured_sink=original_selection["selected_sink"],
                      reconstructed_recommended_sink=recommendation,
                      rerun_measurements=False,
                      selected_sink_matches_recommendation=(original_selection["selected_sink"] == recommendation))
    # Evidence is written first. If a write is interrupted, original failures
    # remain available and the correction's status reports partial completion.
    atomic_write(correction_path, correction)
    for label, result in zip(PILOT_LABELS, reconstructed, strict=True):
        directory = root / "sink-pilot" / label
        atomic_write(directory / "pilot-result.json", result)
        atomic_write(directory / "progress.json", progress_updates[label])
    atomic_write(root / "sink-selection.json", selection)
    atomic_write(root / "progress.json", updated_progress)
    correction.update(status="complete", completed_unix=time.time())
    atomic_write(correction_path, correction)
    return dict(status="complete", correction=str(correction_path),
                measured_sink=original_selection["selected_sink"],
                reconstructed_recommended_sink=recommendation,
                recovered_pilots=list(PILOT_LABELS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/multirank-meta-cpu-20261029")
    args = parser.parse_args()
    print(json.dumps(repair(args.output), ensure_ascii=False))


if __name__ == "__main__":
    main()
