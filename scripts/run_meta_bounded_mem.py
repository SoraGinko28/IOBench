"""Guarded, serial bounded Meta buffer sweep and selected-size confirmation.

All payloads belong to an individual Runner and are removed by its ownership
checked cleanup. Existing validation trees are read only. Reinvocation skips
finished cells; it never reuses or overwrites an unfinished measurement cell.
"""

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gdt_safetensors import Runner  # noqa: E402
from scripts.run_gdtensor_trace_matrix import dump, load  # noqa: E402
from scripts.run_multirank_meta_cpu import arguments, inventory  # noqa: E402
from scripts.run_shared_safetensors import IdleGuard  # noqa: E402

DEFAULT_OUTPUT = ROOT / "validation/meta-bounded-mem-20261030"
SIZES_MIB = (64, 256, 1024)
CELLS = tuple((model, ranks) for model in ("1b", "3b") for ranks in (4, 8))


def run_stage(root, options):
    root.mkdir(parents=True, exist_ok=True)
    progress_path = root / "progress.json"
    progress = (
        load(progress_path)
        if progress_path.exists()
        else dict(started_unix=time.time(), completed=[], failed=[])
    )
    progress.update(
        status="running",
        stage=options.stage,
        pid=os.getpid(),
        sweep_sizes_mib=list(SIZES_MIB),
        sweep_repeats=3,
        confirmation_repeats=5,
        dcp_confirmation_repeats=3,
        selected_sink="ring",
        requested_writer_limit=0,
        warm_block_read_tolerance_bytes=64 * 1024,
        warm_redo_policy=(
            "one redo for <99% pre-call residency or >64 KiB timed block reads per rank"
        ),
        phase_policy=(
            "one phase at a time; idle/foreign-job guard before every phase; "
            "no tests during measurements"
        ),
    )

    def update(**fields):
        progress.update(updated_unix=time.time(), **fields)
        dump(progress_path, progress)

    update()
    guard = IdleGuard(root, options, update)

    class BoundedRunner(Runner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

        def _phase_once(self, case, checkpoint, model, config, action, *, attempt=1):
            guard.wait(
                dict(
                    cell=self.root.name,
                    stage=options.stage,
                    buffer_mib=self.args.meta_load_sink_mib,
                    model=model,
                    config=config,
                    action=action,
                    attempt=attempt,
                    output=str(case),
                )
            )
            return super()._phase_once(case, checkpoint, model, config, action, attempt=attempt)

    import gdtensor
    import torch

    dump(
        root / f"{options.stage}-environment.json",
        dict(
            python=sys.version,
            executable=sys.executable,
            torch=torch.__version__,
            gdtensor_package=str(gdtensor.__path__[0]),
            platform=platform.platform(),
            cpu_count=os.cpu_count(),
            meminfo=Path("/proc/meminfo").read_text(),
            method=(
                "paired/interleaved; OMP_NUM_THREADS=1; zero; concurrent writers; "
                "per-file fadvise cold; pre-call warm residency"
            ),
            writer_limit_policy="world size with PermitQueue and pre-timer memory checks retained",
            idle_policy=vars(options),
            local_device=list(guard.device),
        ),
    )
    dump(root / f"{options.stage}-source-before.json", inventory())
    if options.stage == "sweep":
        cells = [
            (f"sweep/{size}mib", size, model, ranks, "safetensors", 3)
            for size in SIZES_MIB
            for model, ranks in CELLS
        ]
    else:
        selection = load(root / "sweep-summary.json")
        size = selection["chosen_size_mib"]
        if size is None:
            raise ValueError(
                "No buffer size passed the sweep; confirmation is not authorized by the brief"
            )
        if options.buffer_mib is not None and options.buffer_mib != size:
            raise ValueError("Confirmation buffer must equal the smallest passing sweep size")
        update(chosen_size_mib=size)
        cells = [("confirm", size, model, ranks, "safetensors", 5) for model, ranks in CELLS]
        cells.append(("confirm", size, "3b", 8, "dcp", 3))

    try:
        for stage, size, model, ranks, fmt, repeats in cells:
            label = f"{model}-{ranks}r-{fmt}"
            run = root / stage / label
            identifier = f"{stage}/{label}"
            update(current=identifier, buffer_mib=size, phase_progress=None)
            if run.exists():
                previous = load(run / "progress.json")
                if previous["status"] != "stage-complete":
                    raise ValueError(
                        f"Unfinished cell requires inspection; refusing overwrite: {run}"
                    )
            else:
                args = arguments(run, model, ranks, fmt, "ring", writer_limit=ranks)
                args.repeats, args.meta_load_sink_mib = repeats, size
                args.requested_writer_limit = 0
                args.safetensors_layout = "shared" if fmt == "safetensors" else "per-rank"
                args.warm_block_read_tolerance_bytes = 64 * 1024
                try:
                    BoundedRunner(args).run()
                except Exception as error:
                    progress["failed"].append(
                        dict(cell=identifier, error=f"{type(error).__name__}: {error}")
                    )
                    update()
                    continue
            if identifier not in progress["completed"]:
                progress["completed"].append(identifier)
            update()

        # Analysis begins only after measurement workers have exited.
        from scripts.analyze_meta_bounded_mem import analyze_cell, summarize_sweep

        for stage, _size, model, ranks, fmt, _repeats in cells:
            run = root / stage / f"{model}-{ranks}r-{fmt}"
            if (run / "measurement-results.json").exists():
                analyze_cell(run)
        if options.stage == "sweep":
            result = summarize_sweep(root)
            update(
                status="sweep-complete",
                chosen_size_mib=result["chosen_size_mib"],
                current=None,
                stage="awaiting-default-change" if result["chosen_size_mib"] else "no-size-passed",
            )
        else:
            update(status="confirmation-complete", current=None, stage="analysis-pending")
        dump(root / f"{options.stage}-source-after.json", inventory())
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(
        json.dumps(
            dict(
                status=progress["status"],
                failed=progress["failed"],
                chosen_size_mib=progress.get("chosen_size_mib"),
            )
        ),
        flush=True,
    )
    return progress


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("sweep", "confirm"), required=True)
    parser.add_argument("--buffer-mib", type=int, choices=SIZES_MIB)
    parser.add_argument("--idle-samples", type=int, default=2)
    parser.add_argument("--idle-sample-seconds", type=float, default=1)
    parser.add_argument("--idle-poll-seconds", type=float, default=10)
    parser.add_argument("--idle-cpu-percent", type=float, default=15)
    parser.add_argument("--idle-disk-percent", type=float, default=5)
    parser.add_argument("--idle-timeout", type=float, default=0)
    options = parser.parse_args(argv)
    if (
        options.idle_samples < 1
        or options.idle_sample_seconds <= 0
        or options.idle_poll_seconds < 0
    ):
        parser.error(
            "idle samples and sample seconds must be positive; poll seconds must be nonnegative"
        )
    root = options.output.resolve()
    root.relative_to(ROOT / "validation")
    options.output = str(root)
    return root, options


if __name__ == "__main__":
    run_stage(*parse_args())
