"""Serial Meta/CPU/CUDA gdtensor save, cold-load and warm-load measurements.

The default layouts use one payload file per rank; shared safetensors writes
full tensors directly into shared files. Each case owns its checkpoint and
removes it after exporting the trace and completion metadata. Each invocation
requires a fresh output directory; measurement runs do not resume.
"""

import argparse
import fcntl
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metasl.cache_control import checkpoint_residency  # noqa: E402
from metasl.iostats import advise_evict_checkpoint  # noqa: E402
from metasl.trace_export import export_rank_trace  # noqa: E402
from scripts.run_gdtensor_trace_matrix import (  # noqa: E402
    dump, load, phase_command, runtime_environment, validate_phase,
)

DEFAULT_OUTPUT = ROOT / "validation/gdtensor-measurements"
BUNDLES = {
    "1b": ROOT / "validation/repo-slim-20261029/bundles/1b",
    "3b": ROOT / "validation/gdt-meta-sim-scale-20261028/bundles/3b",
    "7b": ROOT / "validation/gdt-meta-sim-scale-20261028/bundles/7b",
}
CONFIGS = {"meta": "meta", "meta-sim": "meta", "cpu": "cpu", "gpu": "cuda"}
DEFAULT_META_RING_MIB = 64


def verify_layout(checkpoint, max_file_mib, checkpoint_format="safetensors", ranks=1,
                  safetensors_layout="per-rank", max_file_gib=None):
    suffix = ".distcp" if checkpoint_format == "dcp" else ".safetensors"
    files = sorted(p.name for p in checkpoint.glob("*" + suffix))
    metadata = ".metadata" if checkpoint_format == "dcp" else ".metasl-complete.json"
    if not (checkpoint / metadata).is_file():
        raise RuntimeError(f"Missing completion metadata: {checkpoint / metadata}")
    expected = 1 if checkpoint_format == "safetensors" and safetensors_layout == "shared" else ranks
    if max_file_mib is None and max_file_gib is None and len(files) != expected:
        raise RuntimeError(f"Expected {expected} payload files for {safetensors_layout}: {files}")
    return dict(checkpoint_format=checkpoint_format, payload_files=files,
                file_count=len(files), max_file_mib=max_file_mib)


def archive_metadata(checkpoint, case, checkpoint_format):
    if checkpoint_format == "safetensors":
        dump(case / "manifest.json", load(checkpoint / ".metasl-complete.json"))
    else:
        shutil.copyfile(checkpoint / ".metadata", case / "dcp.metadata")


def warm_validity(report, block_read_tolerance_bytes=0):
    residency = report.get("cache_before_call", {}).get("residency", {})
    block_read_bytes = report["stats"]["proc_io_delta"]["read_bytes"]
    reasons = []
    if not residency.get("fully_warm_data_residency", False):
        reasons.append("checkpoint residency below 99% or missing")
    if block_read_bytes > block_read_tolerance_bytes:
        reasons.append(f"timed block-read bytes > {block_read_tolerance_bytes}")
    return dict(valid=not reasons, reasons=reasons,
                residency_percent=residency.get("data_resident_percent"),
                block_read_bytes=block_read_bytes)


class Runner:
    def __init__(self, args):
        self.args = args
        if getattr(args, "writer_limit", 1) < 0:
            raise ValueError("writer-limit must be nonnegative")
        self.root = args.output.resolve()
        self.root.relative_to(ROOT / "validation")
        if self.root.exists():
            raise ValueError("Output must be a new directory; measurement runs do not resume")
        self.ckroot = (args.checkpoint_root or self.root / "checkpoints").resolve()
        if self.ckroot == self.root:
            raise ValueError("checkpoint-root must differ from the report output")
        if self.ckroot.exists() and (
            not self.ckroot.is_dir() or any(self.ckroot.iterdir())
        ):
            raise ValueError("checkpoint-root must be a new or empty directory owned by this run")
        self.lock = open("/tmp/metasl-gdt-save-gap.lock", "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.root.mkdir(parents=True)
        self.ckroot.mkdir(parents=True, exist_ok=True)
        self.ownership = dict(run_id=uuid.uuid4().hex, output=str(self.root))
        self.owner_path = self.ckroot / ".metasl-owner.json"
        with self.owner_path.open("x") as stream:
            json.dump(self.ownership, stream)
        dump(self.root / "checkpoint-ownership.json", self.ownership)
        self.progress_path = self.root / "progress.json"
        self.progress = {}
        self.progress.update(status="running", pid=os.getpid(), ranks=args.ranks,
                             checkpoint_format=args.checkpoint_format,
                             safetensors_max_file_mib=args.safetensors_max_file_mib,
                             safetensors_layout=getattr(args, "safetensors_layout", "per-rank"),
                             safetensors_max_file_gib=getattr(args, "safetensors_max_file_gib", None),
                             writer_limit=getattr(args, "writer_limit", 1),
                             requested_writer_limit=getattr(args, "requested_writer_limit",
                                                            getattr(args, "writer_limit", 1)),
                             save_only=getattr(args, "save_only", False))
        self.progress.pop("error", None)
        self.progress.setdefault("completed", [])
        self.update()

    def verify_ownership(self):
        if load(self.owner_path) != self.ownership:
            raise RuntimeError("Checkpoint directory ownership changed; refusing cleanup")

    def update(self, **fields):
        self.progress.update(updated_unix=time.time(), **fields)
        dump(self.progress_path, self.progress)

    def bundle(self, model):
        bundle = self.args.bundle or (
            self.args.bundles / model if self.args.bundles else BUNDLES[model]
        )
        if not (bundle / "model_manifest.json").is_file():
            from metasl.skeleton import prepare_bundle

            config = self.args.model_configs / model
            if not (config / "config.json").is_file():
                raise ValueError(f"Missing bundle and offline config for {model}: {config}")
            bundle = self.root / "bundles" / model
            if not (bundle / "model_manifest.json").is_file():
                prepare_bundle(str(config), "bf16", bundle, local_files_only=True)
        return bundle

    def phase(self, case, checkpoint, model, config, action):
        attempts = []
        for attempt in (1, 2) if action == "warm-load" else (1,):
            rows = self._phase_once(case, checkpoint, model, config, action, attempt=attempt)
            if action != "warm-load":
                return rows
            valid = all(row["warm_validity"]["valid"] for row in rows)
            attempts.append(dict(attempt=attempt, valid=valid, rows=rows))
            dump(case / "warm-load-attempts.json", dict(
                policy=("one documented redo only for residency <99% or timed block reads >"
                        + str(getattr(self.args, "warm_block_read_tolerance_bytes", 0))),
                attempts=attempts, accepted_attempt=attempt if valid else None,
            ))
            if valid:
                return rows
            self.progress.setdefault("invalid_warm_attempts", []).append(dict(
                model=model, config=config, case=str(case), attempt=attempt,
                ranks=[dict(rank=row["rank"], **row["warm_validity"]) for row in rows],
            ))
            self.update()
        raise RuntimeError(f"Warm load invalid after the one permitted redo: {case}")

    def _phase_once(self, case, checkpoint, model, config, action, *, attempt=1):
        device = CONFIGS[config]
        phase_name = action if attempt == 1 else f"{action}-redo1"
        output = case / phase_name
        command = phase_command(
            sys.executable, self.bundle(model), checkpoint, output, self.args.ranks,
            "save" if action == "save" else "load", device,
            data_pattern=self.args.data_pattern, validate_values=self.args.validate_values,
            checkpoint_format=self.args.checkpoint_format,
            safetensors_max_file_mib=self.args.safetensors_max_file_mib,
            writer_limit=getattr(self.args, "writer_limit", 1),
            safetensors_layout=getattr(self.args, "safetensors_layout", "per-rank"),
            safetensors_max_file_gib=getattr(self.args, "safetensors_max_file_gib", None),
        )
        if not self.args.validate_values:
            command += ["--content-audit", "none"]
        if action == "warm-load":
            command += ["--warm-cache", "--measure-cache-residency"]
        env = runtime_environment()
        for key in list(env):
            if key.startswith("METASL_GDTENSOR_"):
                env.pop(key)
        if config == "meta-sim":
            env["METASL_GDTENSOR_META_SIM_D2H"] = "pageable"
            env["METASL_GDTENSOR_META_SIM_D2H_POPULATE"] = "non-temporal"
        if device == "meta":
            sink = getattr(self.args, "meta_load_sink", None)
            if sink is None:
                sink = "ring" if config == "meta" and self.args.data_pattern == "zero" else "fresh"
            env["METASL_GDTENSOR_META_LOAD_SINK"] = sink
            env["METASL_GDTENSOR_META_LOAD_SINK_BYTES"] = str(
                getattr(self.args, "meta_load_sink_mib", DEFAULT_META_RING_MIB) * 1024**2
            )
        if config == "gpu":
            env["METASL_GDTENSOR_CUDA_STAGING_MODE"] = "pageable"
        case.mkdir(parents=True, exist_ok=True)
        dump(case / f"{phase_name}-command.json", command)
        dump(case / f"{phase_name}-environment.json",
             {k: v for k, v in env.items() if k.startswith("METASL_GDTENSOR_")})
        self.update(current=dict(model=model, config=config, action=action, output=str(output)))
        print(f"{model} {case.name} {phase_name} ranks={self.args.ranks}", flush=True)
        started = time.monotonic()
        with (case / f"{phase_name}.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                    stderr=subprocess.STDOUT, timeout=2100)
        if result.returncode:
            raise RuntimeError(f"Phase failed ({result.returncode}): {case / (phase_name + '.log')}")
        _, reports = validate_phase(
            output, action="save" if action == "save" else "load", device=device,
            ranks=self.args.ranks, data_pattern=self.args.data_pattern,
            validate_values=self.args.validate_values,
            checkpoint_format=self.args.checkpoint_format,
        )
        layout = verify_layout(checkpoint, self.args.safetensors_max_file_mib,
                               self.args.checkpoint_format, self.args.ranks,
                               getattr(self.args, "safetensors_layout", "per-rank"),
                               getattr(self.args, "safetensors_max_file_gib", None))
        dump(case / f"{phase_name}-layout.json", layout)
        rows = []
        for report in reports:
            rank = report["rank"]
            exported = export_rank_trace(
                output / f"rank-{rank:03d}.io.jsonl", output / "export", rank=rank,
                phase="save" if action == "save" else "load",
                origin_ns=report["io"]["origin_ns"], checkpoint_root=str(checkpoint),
                payload_events=report.get("payload_events", []),
            )
            row = dict(
                model=model, config=config, device=device, phase=action, rank=rank,
                ranks=self.args.ranks, backend="gdtensor", pattern=self.args.data_pattern,
                checkpoint_format=self.args.checkpoint_format, output=str(output),
                process_seconds=time.monotonic() - started, export=exported,
                dcp_seconds=report["stats"]["dcp_wall_seconds"],
                numerical_values_verified=report.get("numerical_values_verified"), layout=layout,
            )
            if action == "warm-load":
                row.update(warm_validity=warm_validity(
                    report, getattr(self.args, "warm_block_read_tolerance_bytes", 0)),
                    warm_attempt=attempt)
            rows.append(row)
        dump(case / f"{phase_name}-result.json", rows[0] if len(rows) == 1 else rows)
        self.update(last_completed_phase=dict(model=model, config=config, action=action,
                                              output=str(output)))
        return rows

    def measurements(self):
        rows = []
        for model in self.args.models:
            for repeat in range(1, self.args.repeats + 1):
                offset = (repeat - 1) % len(self.args.configs)
                order = self.args.configs[offset:] + self.args.configs[:offset]
                for config in order:
                    case = self.root / "measurements" / model / f"r{repeat}-{config}"
                    done = case / "completed.json"
                    label = config.ljust(8, "-") if self.args.checkpoint_format == "dcp" else config
                    checkpoint = self.ckroot / f"measure-{model}-r{repeat}-{label}"
                    self.verify_ownership()
                    checkpoint.mkdir()
                    case_rows = []
                    try:
                        actions = ("save",) if getattr(self.args, "save_only", False) else (
                            "save", "cold-load", "warm-load")
                        for action in actions:
                            if action != "save":
                                if action == "cold-load":
                                    cache = dict(eviction=advise_evict_checkpoint(checkpoint),
                                                 residency_before_load=checkpoint_residency(checkpoint))
                                    if not cache["residency_before_load"]["near_zero_data_residency"]:
                                        raise RuntimeError(f"Cold residency failed: {case}")
                                else:
                                    cache = dict(mode="warm", warmup_deferred_to_workers=True,
                                                 after_resident_buffer_preparation=True)
                                dump(case / f"{action}-cache.json", cache)
                            phase_rows = self.phase(case, checkpoint, model, config, action)
                            case_rows.extend(dict(row, repeat=repeat) for row in phase_rows)
                            dump(self.root / "measurement-results.json", rows + case_rows)
                        sizes = {p.name: p.stat().st_size for p in checkpoint.iterdir()}
                        archive_metadata(checkpoint, case, self.args.checkpoint_format)
                        self.verify_ownership()
                        shutil.rmtree(checkpoint)
                        dump(done, dict(rows=case_rows, inventory_bytes=sizes, checkpoint_removed=True))
                        rows.extend(case_rows)
                        self.progress["completed"].append(f"measurement-{model}-r{repeat}-{config}")
                        self.update()
                        dump(self.root / "measurement-results.json", rows)
                    finally:
                        if checkpoint.exists():
                            self.verify_ownership()
                            shutil.rmtree(checkpoint)
        dump(self.root / "measurement-results.json", rows)

    def run(self):
        try:
            self.measurements()
            self.update(status="stage-complete", current=None)
        except BaseException as error:
            self.update(status="failed", error=str(error))
            raise
        finally:
            try:
                self.verify_ownership()
                remaining = [p for p in self.ckroot.iterdir() if p != self.owner_path]
                dump(self.root / "cleanup.json", dict(checkpoint_root=str(self.ckroot),
                     remaining=[str(p) for p in remaining], checkpoints_removed=not remaining))
                if not remaining:
                    self.owner_path.unlink()
                    self.ckroot.rmdir()
            finally:
                self.lock.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("measurements",), default="measurements")
    parser.add_argument("--models", nargs="+", choices=tuple(BUNDLES), default=["1b"])
    parser.add_argument("--bundle", type=Path, help="Prepared bundle for a single --models entry")
    parser.add_argument("--bundles", type=Path, help="Directory containing prepared MODEL bundles")
    parser.add_argument("--model-configs", type=Path, default=ROOT / "configs")
    parser.add_argument("--checkpoint-root", type=Path, help="Payload parent, e.g. /mnt/nfs-ckpt/run")
    parser.add_argument("--ranks", "--nproc", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--configs", nargs="+", choices=tuple(CONFIGS), default=list(CONFIGS))
    parser.add_argument("--data-pattern", choices=("zero", "position"), default="zero")
    parser.add_argument("--validate-values", action="store_true")
    parser.add_argument("--checkpoint-format", choices=("safetensors", "dcp"), default="safetensors")
    parser.add_argument("--safetensors-max-file-mib", type=int)
    parser.add_argument("--safetensors-layout", choices=("per-rank", "shared"), default="per-rank")
    parser.add_argument("--safetensors-max-file-gib", type=float)
    parser.add_argument("--writer-limit", type=int, default=1,
                        help="Maximum concurrent save writers; 0 disables writer admission (default: 1)")
    parser.add_argument("--save-only", action="store_true",
                        help="Measure saves only; skip cold and warm loads")
    parser.add_argument("--meta-load-sink", choices=("fresh", "faithful", "ring"), default=None,
                        help="Meta host buffer: defaults to ring for zero-pattern meta runs, "
                             "fresh for meta-sim and position; faithful keeps the resident-shard reference")
    parser.add_argument("--meta-load-sink-mib", type=int, default=DEFAULT_META_RING_MIB,
                        help="Resident ring capacity in MiB (default: 64)")
    args = parser.parse_args(argv)
    if args.writer_limit < 0:
        parser.error("writer-limit must be nonnegative")
    if min(args.repeats, args.ranks) <= 0:
        parser.error("repeats and ranks must be positive")
    if args.meta_load_sink_mib <= 0:
        parser.error("meta-load-sink-mib must be positive")
    if args.bundle and len(args.models) != 1:
        parser.error("--bundle requires exactly one model")
    if len(set(args.configs)) != len(args.configs):
        parser.error("configs must be unique")
    if args.safetensors_max_file_mib is not None and (
        args.checkpoint_format != "safetensors" or args.safetensors_max_file_mib <= 0
    ):
        parser.error("safetensors-max-file-mib requires safetensors and a positive integer")
    if args.safetensors_layout == "shared" and args.checkpoint_format != "safetensors":
        parser.error("safetensors-layout shared requires safetensors")
    if args.safetensors_max_file_gib is not None and (
        args.checkpoint_format != "safetensors" or args.safetensors_layout != "shared"
        or not math.isfinite(args.safetensors_max_file_gib) or args.safetensors_max_file_gib <= 0
    ):
        parser.error("safetensors-max-file-gib requires shared safetensors and a finite positive size")
    if args.safetensors_max_file_gib is not None and args.safetensors_max_file_mib is not None:
        parser.error("choose one safetensors maximum file size unit")
    return args


def main():
    Runner(parse_args()).run()


if __name__ == "__main__":
    main()
