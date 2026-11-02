"""Run the shared-file safetensors matrix, serially, on an idle local host.

The existing Runner owns phase commands, paired/interleaved ordering, residency
probes, trace exports and checkpoint cleanup. This entry point adds the fixed
shared-layout matrix, concurrent admission policy and a guard before every
phase (including a permitted warm retry). No global cache controls are used.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gdt_safetensors import Runner  # noqa: E402
from scripts.run_gdtensor_trace_matrix import dump  # noqa: E402
from scripts.run_multirank_meta_cpu import arguments, inventory  # noqa: E402

DEFAULT_OUTPUT = ROOT / "validation/shared-safetensors-20261030"
WARM_BLOCK_READ_TOLERANCE = 64 * 1024


def measurement_process(argv):
    """Recognize actual worker programs, excluding pgrep and shell wrappers."""
    if not argv:
        return False
    executable = Path(argv[0]).name
    python = executable.startswith("python")
    metasl = python and any(argv[i:i + 2] == ["-m", "metasl.run"]
                            for i in range(len(argv) - 1))
    fsbench = (any("fsbench" in Path(part).name.lower() for part in argv[:2])
               or python and any(argv[i] == "-m" and argv[i + 1].startswith("fsbench")
                                 for i in range(len(argv) - 1)))
    return metasl or fsbench


def foreign_measurements():
    # Explicitly perform the requested check. Reading cmdline below prevents
    # shell command strings that merely mention metasl.run from blocking us.
    probe = subprocess.run(["pgrep", "-f", "metasl.run"], text=True,
                           capture_output=True, check=False)
    if probe.returncode not in (0, 1):
        raise RuntimeError(f"pgrep failed: {probe.stderr.strip()}")
    found = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        try:
            pid = int(path.parent.name)
            argv = [p.decode(errors="replace") for p in path.read_bytes().split(b"\0") if p]
        except (OSError, ValueError):
            continue
        if pid != os.getpid() and measurement_process(argv):
            found.append(dict(pid=pid, argv=argv))
    return dict(pgrep_pids=[int(p) for p in probe.stdout.split()], jobs=found)


def host_counters(device):
    fields = [int(v) for v in Path("/proc/stat").read_text().splitlines()[0].split()[1:]]
    # guest is already included in user/nice; do not count it twice.
    total = sum(fields[:8])
    idle = fields[3] + fields[4]
    disk_busy = None
    for line in Path("/proc/diskstats").read_text().splitlines():
        columns = line.split()
        if (int(columns[0]), int(columns[1])) == device:
            disk_busy = int(columns[12])
            break
    mem = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
    return dict(monotonic=time.monotonic(), cpu_total=total, cpu_idle=idle,
                disk_busy_ms=disk_busy, available_bytes=int(mem["MemAvailable"].split()[0]) * 1024)


def idle_delta(before, after):
    elapsed = after["monotonic"] - before["monotonic"]
    ticks = after["cpu_total"] - before["cpu_total"]
    disk = (100 * (after["disk_busy_ms"] - before["disk_busy_ms"]) / (1000 * elapsed)
            if before["disk_busy_ms"] is not None and after["disk_busy_ms"] is not None else None)
    return dict(sample_seconds=elapsed,
                cpu_busy_percent=100 * (1 - (after["cpu_idle"] - before["cpu_idle"]) / ticks) if ticks else 0,
                disk_busy_percent=disk, available_bytes=after["available_bytes"])


class IdleGuard:
    def __init__(self, output, options, update):
        self.output, self.options, self.update = output, options, update
        device = os.stat(output).st_dev
        self.device = os.major(device), os.minor(device)

    def wait(self, context):
        started, consecutive = time.monotonic(), 0
        while consecutive < self.options.idle_samples:
            jobs = foreign_measurements()
            before = host_counters(self.device)
            time.sleep(self.options.idle_sample_seconds)
            after = host_counters(self.device)
            sample = idle_delta(before, after)
            # Recheck immediately before allowing the block to start.
            last = foreign_measurements()
            reasons = []
            if jobs["jobs"] or last["jobs"]:
                reasons.append("foreign metasl.run/fsbench")
            if sample["cpu_busy_percent"] > self.options.idle_cpu_percent:
                reasons.append("CPU busy")
            if sample["disk_busy_percent"] is None:
                reasons.append("local filesystem diskstats unavailable")
            elif sample["disk_busy_percent"] > self.options.idle_disk_percent:
                reasons.append("local disk busy")
            if sample["available_bytes"] < 5 * 1024**3:
                reasons.append("MemAvailable below 5 GiB")
            consecutive = 0 if reasons else consecutive + 1
            evidence = dict(unix=time.time(), context=context, device=list(self.device),
                            pgrep_pids=last["pgrep_pids"], foreign_jobs=last["jobs"],
                            sample=sample, reasons=reasons, consecutive_idle_samples=consecutive,
                            required_idle_samples=self.options.idle_samples,
                            elapsed_wait_s=time.monotonic() - started)
            with (self.output / "idle-guards.jsonl").open("a") as stream:
                stream.write(json.dumps(evidence) + "\n")
            self.update(idle_guard=evidence)
            if reasons:
                print(json.dumps(dict(waiting_for_idle=evidence)), flush=True)
                if self.options.idle_timeout and time.monotonic() - started > self.options.idle_timeout:
                    raise TimeoutError("Host did not become idle before idle-timeout")
                time.sleep(self.options.idle_poll_seconds)
        return evidence


def run_matrix(root, options):
    if (root / "progress.json").exists() or (root / "matrix").exists():
        raise ValueError("A fresh shared-safetensors measurement output is required")
    root.mkdir(parents=True, exist_ok=True)
    progress = dict(status="running", pid=os.getpid(), started_unix=time.time(),
                    completed=[], failed=[], current=None, matrix_cells=4, repeats=5,
                    phases=["save", "cold-load", "warm-load"], layout="shared",
                    pattern="zero", selected_sink="faithful",
                    selected_source="resident-local-shards", requested_writer_limit=0,
                    effective_writer_limits={str(n): n for n in (4, 8)},
                    warm_block_read_tolerance_bytes=WARM_BLOCK_READ_TOLERANCE,
                    warm_redo_policy="one redo for <99% residency or >64 KiB timed block reads per rank",
                    phase_policy="one phase at a time; idle/foreign-job guard before every phase; no tests during measurements")

    def update(**fields):
        progress.update(updated_unix=time.time(), **fields)
        dump(root / "progress.json", progress)

    update()
    correctness = root / "correctness" / "progress.json"
    if correctness.is_file():
        progress["correctness_before_measurements"] = json.loads(correctness.read_text())
        update()
    guard = IdleGuard(root, options, update)

    class SharedRunner(Runner):
        def update(self, **fields):
            super().update(**fields)
            update(phase_progress=self.progress)

        def _phase_once(self, case, checkpoint, model, config, action, *, attempt=1):
            guard.wait(dict(cell=self.root.name, model=model, config=config,
                            action=action, attempt=attempt, output=str(case)))
            return super()._phase_once(case, checkpoint, model, config, action, attempt=attempt)

    try:
        import torch
        import gdtensor

        dump(root / "environment.json", dict(
            python=sys.version, executable=sys.executable, torch=torch.__version__,
            gdtensor_package=str(gdtensor.__path__[0]), platform=platform.platform(),
            cpu_count=os.cpu_count(), meminfo=Path("/proc/meminfo").read_text(),
            method="paired/interleaved; OMP_NUM_THREADS=1; CPU-target Meta and CPU; local fadvise cold; zero pattern",
            requested_writer_limit=0, effective_writer_limits=progress["effective_writer_limits"],
            writer_limit_policy="0 resolves to world size, retaining PermitQueue and pre-timer memory safety checks",
            warm_block_read_tolerance_bytes=WARM_BLOCK_READ_TOLERANCE,
            idle_policy=vars(options), local_device=list(guard.device)))
        dump(root / "source-before.json", inventory())
        dump(root / "sink-selection.json", dict(selected_sink="faithful",
             selected_source="resident-local-shards", ring_retained=True, pilot_run=False,
             reason="fixed shared safetensors brief: CPU-target faithful mode"))
        cells = [("matrix", model, ranks, 5, None) for model in ("1b", "3b") for ranks in (4, 8)]
        cells += [("split-smoke", "3b", ranks, 1, 5) for ranks in (4, 8)]
        for stage, model, ranks, repeats, max_gib in cells:
            label = f"{model}-{ranks}r-safetensors"
            update(stage=stage, current=label, phase_progress=None)
            args = arguments(root / stage / label, model, ranks, "safetensors", "faithful",
                             writer_limit=ranks)
            args.repeats, args.requested_writer_limit = repeats, 0
            args.safetensors_layout, args.safetensors_max_file_gib = "shared", max_gib
            args.warm_block_read_tolerance_bytes = WARM_BLOCK_READ_TOLERANCE
            try:
                SharedRunner(args).run()
                progress["completed"].append(f"{stage}-{label}")
            except Exception as error:
                progress["failed"].append(dict(stage=stage, cell=label,
                                               error=f"{type(error).__name__}: {error}"))
            update()
        dump(root / "source-after-measurements.json", inventory())
        update(status="measurements-complete", stage="analysis-pending", current=None,
               completed_unix=time.time())
    except BaseException as error:
        update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    print(json.dumps(dict(status=progress["status"], failed=progress["failed"])), flush=True)
    return progress


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--idle-samples", type=int, default=2)
    parser.add_argument("--idle-sample-seconds", type=float, default=1)
    parser.add_argument("--idle-poll-seconds", type=float, default=10)
    parser.add_argument("--idle-cpu-percent", type=float, default=15)
    parser.add_argument("--idle-disk-percent", type=float, default=5)
    parser.add_argument("--idle-timeout", type=float, default=0,
                        help="Seconds to wait per phase; 0 waits indefinitely for foreign measurements")
    options = parser.parse_args(argv)
    if options.idle_samples < 1 or options.idle_sample_seconds <= 0 or options.idle_poll_seconds < 0:
        parser.error("idle samples and sample seconds must be positive; poll seconds must be nonnegative")
    root = options.output.resolve()
    root.relative_to(ROOT / "validation")
    # JSON environment evidence must contain ordinary serializable values.
    options.output = str(root)
    return run_matrix(root, options)


if __name__ == "__main__":
    main()
