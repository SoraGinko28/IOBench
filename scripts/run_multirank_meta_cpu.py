"""Serial CPU-target sink pilot and paired multi-rank Meta/CPU measurements.

Only the fresh output directory is mutated. Checkpoints are owned by each
sub-run and removed on success or failure. No global cache or system settings
are changed: cold loads use per-file POSIX_FADV_DONTNEED and verify mincore.
"""

import argparse
import hashlib
import json
import os
import platform
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gdt_safetensors import Runner  # noqa: E402
from scripts.run_gdtensor_trace_matrix import dump, load, warm_load_checkpoint  # noqa: E402
from metasl.trace_export import (  # noqa: E402
    compare_prepared_rank_traces, prepare_rank_trace,
)


def paired_summary(meta, cpu):
    baseline = statistics.median(cpu)
    delta = statistics.median(a - b for a, b in zip(meta, cpu, strict=True))
    return dict(median_s=statistics.median(meta), samples_s=meta,
                paired_delta_s=delta, paired_delta_percent=100 * delta / baseline)


def warm_request_identity(rows, config, repeats=3, ranks=4):
    evidence = []
    for repeat in range(1, repeats + 1):
        for rank in range(ranks):
            pair = {}
            for label in (config, "cpu"):
                row = next(r for r in rows if r["config"] == label
                           and r["repeat"] == repeat and r["rank"] == rank)
                pair[label] = prepare_rank_trace(row["export"]["output_path"])
            result = compare_prepared_rank_traces(pair[config], pair["cpu"])
            evidence.append(dict(result, repeat=repeat, rank=rank))
    return dict(requests_identical=all(r["status"] == "identical" for r in evidence),
                request_comparisons=evidence)


class PilotRunner(Runner):
    """Same checkpoint and resident-cache procedure for all three pilot targets."""

    def measurements(self):
        model = self.args.models[0]
        checkpoint = self.ckroot / "pilot-shared-checkpoint"
        self.verify_ownership()
        checkpoint.mkdir()
        rows = []
        try:
            self.phase(self.root / "setup", checkpoint, model, "cpu", "save")
            for repeat in range(1, 4):
                labels = ["meta-faithful", "meta-ring", "cpu"]
                offset = (repeat - 1) % len(labels)
                for label in labels[offset:] + labels[:offset]:
                    self.args.meta_load_sink = label.removeprefix("meta-") if label != "cpu" else "fresh"
                    case = self.root / "measurements" / model / f"r{repeat}-{label}"
                    cache = warm_load_checkpoint(checkpoint)
                    if not cache["target_met"]:
                        raise RuntimeError("Pilot warm-cache residency failed")
                    dump(case / "warm-load-cache.json", cache)
                    measured = self.phase(case, checkpoint, model,
                                          "cpu" if label == "cpu" else "meta", "warm-load")
                    measured = [dict(r, config=label, repeat=repeat) for r in measured]
                    rows.extend(measured)
                    dump(case / "completed.json", dict(rows=measured))
                    dump(self.root / "measurement-results.json", rows)
                    self.progress["completed"].append(f"r{repeat}-{label}")
                    self.update()
            medians = {}
            for label in ("cpu", "meta-faithful", "meta-ring"):
                medians[label] = [max(r["dcp_seconds"] for r in rows
                                     if r["repeat"] == repeat and r["config"] == label)
                                  for repeat in range(1, 4)]
            result = dict(model=model, checkpoint_format=self.args.checkpoint_format,
                          cpu_median_s=statistics.median(medians["cpu"]),
                          cpu_samples_s=medians["cpu"])
            for sink in ("faithful", "ring"):
                label = "meta-" + sink
                result[sink] = dict(**paired_summary(medians[label], medians["cpu"]),
                                    **warm_request_identity(rows, label))
            dump(self.root / "pilot-result.json", result)
        finally:
            self.verify_ownership()
            shutil.rmtree(checkpoint)


def arguments(output, model, ranks, fmt, sink, *, writer_limit=1, save_only=False):
    return SimpleNamespace(output=output, checkpoint_root=None, ranks=ranks,
                           checkpoint_format=fmt, safetensors_max_file_mib=None,
                           models=[model], repeats=3, configs=["meta", "cpu"],
                           bundle=None, bundles=None, model_configs=ROOT / "configs",
                           data_pattern="zero", validate_values=False,
                           meta_load_sink=sink, meta_load_sink_mib=256,
                           writer_limit=writer_limit, save_only=save_only)


def inventory():
    return {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for folder in ("metasl", "scripts", "tests")
            for p in sorted((ROOT / folder).glob("*.py"))}


def run_fix1(root, writer_limit=1):
    """Eight gated cells only, five alternating Meta/CPU pairs, 1B first."""
    if (root / "progress.json").exists() or (root / "matrix").exists():
        raise ValueError("A fresh fix1 measurement output is required")
    root.mkdir(parents=True, exist_ok=True)
    progress = dict(status="running", started_unix=time.time(), pid=os.getpid(),
                    completed=[], failed=[], current=None, matrix_cells=8,
                    repeats=5, selected_sink="faithful", selected_source="resident-local-shards",
                    phase_policy="one measurement at a time; no tests during measurements",
                    warm_redo_policy="one redo only for <99% pre-call residency or nonzero timed block reads")

    def update(**fields):
        progress.update(updated_unix=time.time(), **fields)
        dump(root / "progress.json", progress)

    class TrackedRunner(Runner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

    update()
    import torch
    import gdtensor

    dump(root / "environment.json", dict(
        python=sys.version, executable=sys.executable, torch=torch.__version__,
        gdtensor_package=str(gdtensor.__path__[0]), platform=platform.platform(),
        cpu_count=os.cpu_count(), load_average=os.getloadavg(),
        meminfo=Path("/proc/meminfo").read_text(),
        method="serial; OMP_NUM_THREADS=1; CPU-only; local disk; per-file fadvise cold; zero pattern",
    ))
    dump(root / "source-before.json", inventory())
    dump(root / "sink-selection.json", dict(selected_sink="faithful",
         selected_source="resident-local-shards", ring_retained=True, pilot_run=False,
         reason="fix1 brief gates CPU-target faithful mode only; ring is not a gate"))
    for model in ("1b", "3b"):
        for ranks in (4, 8):
            for fmt in ("dcp", "safetensors"):
                label = f"{model}-{ranks}r-{fmt}"
                update(stage="matrix", current=label)
                case_args = arguments(root / "matrix" / label, model, ranks, fmt, "faithful",
                                      writer_limit=writer_limit)
                case_args.repeats = 5
                try:
                    TrackedRunner(case_args).run()
                    progress["completed"].append("matrix-" + label)
                except Exception as error:
                    progress["failed"].append(dict(stage="matrix", cell=label, error=str(error)))
                update()
    dump(root / "source-after-measurements.json", inventory())
    update(status="measurements-complete", stage="analysis-pending", current=None,
           completed_unix=time.time())
    print(json.dumps(dict(status=progress["status"], failed=progress["failed"])), flush=True)


def run_concurrent_save(root, writer_limit=1):
    """Eight save-only cells; zero selects world-size admission to retain safety checks."""
    if writer_limit < 0:
        raise ValueError("writer-limit must be nonnegative")
    if (root / "progress.json").exists() or (root / "matrix").exists():
        raise ValueError("A fresh concurrent-save measurement output is required")
    root.mkdir(parents=True, exist_ok=True)
    limits = {str(ranks): writer_limit or ranks for ranks in (4, 8)}
    policy = ("requested 0 resolves to world size per cell: all ranks may write concurrently; "
              "PermitQueue and pre-timer baseline memory safety checks remain enabled")
    progress = dict(status="running", started_unix=time.time(), pid=os.getpid(),
                    completed=[], failed=[], current=None, matrix_cells=8, repeats=5,
                    phases=["save"], selected_sink="faithful",
                    selected_source="resident-local-shards", requested_writer_limit=writer_limit,
                    effective_writer_limits=limits, writer_limit_policy=policy,
                    phase_policy="one measurement at a time; no tests during measurements")

    def update(**fields):
        progress.update(updated_unix=time.time(), **fields)
        dump(root / "progress.json", progress)

    class TrackedRunner(Runner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

    update()
    try:
        import torch
        import gdtensor

        dump(root / "environment.json", dict(
            python=sys.version, executable=sys.executable, torch=torch.__version__,
            gdtensor_package=str(gdtensor.__path__[0]), platform=platform.platform(),
            cpu_count=os.cpu_count(), load_average=os.getloadavg(),
            meminfo=Path("/proc/meminfo").read_text(),
            method="serial measurements; OMP_NUM_THREADS=1; CPU-only; local disk; zero pattern; save only",
            requested_writer_limit=writer_limit, effective_writer_limits=limits,
            writer_limit_policy=policy,
        ))
        dump(root / "source-before.json", inventory())
        dump(root / "sink-selection.json", dict(
            selected_sink="faithful", selected_source="resident-local-shards",
            ring_retained=True, pilot_run=False,
            reason="concurrent-save brief uses fix1 CPU-target resident source and faithful sink"))
        for model in ("1b", "3b"):
            for ranks in (4, 8):
                for fmt in ("dcp", "safetensors"):
                    label = f"{model}-{ranks}r-{fmt}"
                    effective = limits[str(ranks)]
                    update(stage="matrix", current=label, effective_writer_limit=effective,
                           phase_progress=None)
                    case_args = arguments(root / "matrix" / label, model, ranks, fmt, "faithful",
                                          writer_limit=effective, save_only=True)
                    case_args.repeats = 5
                    case_args.requested_writer_limit = writer_limit
                    try:
                        TrackedRunner(case_args).run()
                        progress["completed"].append("matrix-" + label)
                    except Exception as error:
                        progress["failed"].append(dict(stage="matrix", cell=label, error=str(error)))
                    update()
        dump(root / "source-after-measurements.json", inventory())
        update(status="measurements-complete", stage="analysis-pending", current=None,
               completed_unix=time.time())
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(json.dumps(dict(status=progress["status"], failed=progress["failed"])), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/multirank-meta-cpu-20261029")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--fix1", action="store_true",
                       help="Run only the eight 4/8-rank gates with five repeats and resident sources")
    modes.add_argument("--concurrent-save", action="store_true",
                       help="Run the eight 4/8-rank save-only gates with five alternating Meta/CPU pairs")
    parser.add_argument("--writer-limit", type=int, default=1,
                        help="Save writer concurrency (default: 1); concurrent-save maps 0 to world size")
    args = parser.parse_args()
    if args.writer_limit < 0:
        parser.error("writer-limit must be nonnegative")
    root = args.output.resolve()
    root.relative_to(ROOT / "validation")
    if args.fix1:
        if root == ROOT / "validation/multirank-meta-cpu-20261029":
            root = ROOT / "validation/multirank-fix1-20261030"
        run_fix1(root, writer_limit=args.writer_limit)
        return
    if args.concurrent_save:
        if root == ROOT / "validation/multirank-meta-cpu-20261029":
            root = ROOT / "validation/multirank-concurrent-save-20261030"
        run_concurrent_save(root, writer_limit=args.writer_limit)
        return
    # Read-only diagnostic notes may already exist; measurement records may not.
    if (root / "progress.json").exists() or (root / "matrix").exists() or (root / "sink-pilot").exists():
        raise ValueError("A fresh measurement output is required")
    root.mkdir(parents=True, exist_ok=True)
    progress = dict(status="running", started_unix=time.time(), pid=os.getpid(),
                    completed=[], failed=[], current=None, matrix_cells=12)

    def update(**fields):
        progress.update(updated_unix=time.time(), **fields)
        dump(root / "progress.json", progress)

    class TrackedRunner(Runner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

    class TrackedPilot(PilotRunner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

    update()
    import torch
    import gdtensor

    env = dict(python=sys.version, executable=sys.executable, torch=torch.__version__,
               gdtensor_package=str(gdtensor.__path__[0]), platform=platform.platform(),
               cpu_count=os.cpu_count(), load_average=os.getloadavg(),
               meminfo=Path("/proc/meminfo").read_text(),
               processes=subprocess.run(["ps", "-eo", "pid,pcpu,pmem,comm", "--sort=-pcpu"],
                                        capture_output=True, text=True, check=True).stdout,
               method="serial; OMP_NUM_THREADS=1; CPU-only; per-file fadvise cold eviction; zero pattern")
    dump(root / "environment.json", env)
    dump(root / "source-before.json", inventory())
    pilot_cells = []
    for model in ("1b", "3b"):
        for fmt in ("dcp", "safetensors"):
            label = f"{model}-{fmt}"
            update(stage="sink-pilot", current=label)
            try:
                output = root / "sink-pilot" / label
                TrackedPilot(arguments(output, model, 4, fmt, "faithful",
                                       writer_limit=args.writer_limit)).run()
                pilot_cells.append(load(output / "pilot-result.json"))
                progress["completed"].append("pilot-" + label)
            except Exception as error:
                progress["failed"].append(dict(stage="sink-pilot", cell=label, error=str(error)))
            update()
    ring_pass = len(pilot_cells) == 4 and all(
        c["ring"]["requests_identical"] and abs(c["ring"]["paired_delta_percent"]) <= 3
        for c in pilot_cells)
    sink = "ring" if ring_pass else "faithful"
    selection = dict(selected_sink=sink, ring_bytes=256 * 1024**2,
                     criterion="ring only if all 1B/3B x both formats warm paired medians within +/-3% and requests identical",
                     cells=pilot_cells, failed_pilots=progress["failed"].copy())
    dump(root / "sink-selection.json", selection)
    for model in ("1b", "3b"):
        for ranks in (4, 8, 16):
            for fmt in ("dcp", "safetensors"):
                label = f"{model}-{ranks}r-{fmt}"
                update(stage="matrix", current=label, selected_sink=sink)
                try:
                    TrackedRunner(arguments(root / "matrix" / label, model, ranks, fmt, sink,
                                            writer_limit=args.writer_limit)).run()
                    progress["completed"].append("matrix-" + label)
                except Exception as error:
                    progress["failed"].append(dict(stage="matrix", cell=label, error=str(error)))
                update()
    dump(root / "source-after-measurements.json", inventory())
    update(status="measurements-complete", stage="analysis-pending", current=None,
           completed_unix=time.time())
    print(json.dumps(dict(status=progress["status"], failed=progress["failed"], sink=sink)), flush=True)


if __name__ == "__main__":
    main()
