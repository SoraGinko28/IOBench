"""Sequential position-pattern correctness gates for standard shared safetensors.

The native reference and each verifier run in separate processes. At no point
does a full reference state dict coexist with a distributed worker group. The
only full-model allocation is the independent, single-process save_file oracle
or a standard loader. Checkpoints are owned by this run and removed on exit.
"""

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.run_gdtensor_trace_matrix import (  # noqa: E402
    dump,
    load,
    phase_command,
    runtime_environment,
    validate_phase,
)
from scripts.run_shared_safetensors import IdleGuard  # noqa: E402

DEFAULT_BUNDLES = {
    "1b": ROOT / "validation/repo-slim-20261029/bundles/1b",
    "3b": ROOT / "validation/gdt-meta-sim-scale-20261028/bundles/3b",
}
LOAD_MODES = ("cpu", "meta-fresh", "meta-faithful")


def compare_files(actual, expected, block_bytes=8 * 1024**2):
    """Compare every byte with bounded memory and retain independent SHA256s."""
    actual, expected = Path(actual), Path(expected)
    if block_bytes <= 0:
        raise ValueError("comparison block size must be positive")
    if actual.stat().st_size != expected.stat().st_size:
        raise ValueError(f"File size differs: {actual} vs {expected}")
    digests = [hashlib.sha256(), hashlib.sha256()]
    offset = 0
    with actual.open("rb") as left, expected.open("rb") as right:
        while True:
            a, b = left.read(block_bytes), right.read(block_bytes)
            if a != b:
                first = next(
                    (i for i, (x, y) in enumerate(zip(a, b, strict=False)) if x != y),
                    min(len(a), len(b)),
                )
                raise ValueError(f"File byte mismatch at offset {offset + first}: {actual}")
            if not a:
                break
            digests[0].update(a)
            digests[1].update(b)
            offset += len(a)
    return dict(
        byte_identical=True,
        bytes=offset,
        actual_sha256=digests[0].hexdigest(),
        reference_sha256=digests[1].hexdigest(),
    )


def read_standard_header(path):
    """Independently reject DCP metadata and read the on-disk standard schema."""
    with Path(path).open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError("Truncated safetensors prefix")
        size = struct.unpack("<Q", prefix)[0]
        if size > 100_000_000 or size > Path(path).stat().st_size - 8:
            raise ValueError("Invalid safetensors header length")
        raw = stream.read(size)
    header = json.loads(raw)
    if "DCP_SHARDING_INFO" in header.get("__metadata__", {}):
        raise ValueError("Shared checkpoint has DCP_SHARDING_INFO")
    return header


def state_specs(bundle):
    """Use the prepared full state dict's names, shapes, dtypes and alias salt."""
    from metasl.sharding import alias_names
    from metasl.skeleton import build_model

    model, manifest = build_model(Path(bundle), "meta")
    aliases = alias_names(manifest["structure"])
    specs = [
        (name, tuple(tensor.shape), tensor.dtype, aliases.get(name, [name])[0])
        for name, tensor in model.state_dict().items()
    ]
    del model
    return specs


def native_reference(bundle, checkpoint):
    """Call the public native serializer on an independently built full dict."""
    import torch
    from safetensors.torch import save_file

    from metasl.sharding import fill_or_check

    torch.set_num_threads(1)
    state = {}
    for name, shape, dtype, salt in state_specs(bundle):
        tensor = torch.empty(shape, dtype=dtype, device="cpu")
        fill_or_check(tensor, (0,) * len(shape), shape, salt, "position")
        # Distinct buffers preserve every HF state key, including tied aliases.
        state[name] = tensor
    checkpoint = Path(checkpoint)
    checkpoint.mkdir()
    save_file(state, checkpoint / "model.safetensors")
    return dict(
        serializer="safetensors.torch.save_file",
        full_state_dict=True,
        tensors=len(state),
        payload_bytes=sum(tensor.nbytes for tensor in state.values()),
        metadata=None,
    )


def verify_standard_loader(bundle, checkpoint, *, transformers=False):
    import torch

    from metasl.sharding import fill_or_check

    torch.set_num_threads(1)
    expected = {name: (shape, dtype, salt) for name, shape, dtype, salt in state_specs(bundle)}
    checkpoint = Path(checkpoint)
    checked = set()
    payload_bytes = 0

    def verify(state):
        nonlocal payload_bytes
        for name, tensor in state.items():
            if name in checked or name not in expected:
                raise ValueError(f"Unexpected or duplicate loaded state key: {name}")
            shape, dtype, salt = expected[name]
            if tuple(tensor.shape) != shape or tensor.dtype != dtype or tensor.device.type != "cpu":
                raise ValueError(f"Standard loader changed shape, dtype or device: {name}")
            fill_or_check(tensor, (0,) * len(shape), shape, salt, "position", check=True)
            checked.add(name)
            payload_bytes += tensor.nbytes

    if transformers:
        from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

        config = AutoConfig.from_pretrained(bundle, local_files_only=True)
        dtype = next(iter(expected.values()))[1]
        model, info = AutoModelForCausalLM.from_pretrained(
            checkpoint,
            config=config,
            generation_config=GenerationConfig.from_model_config(config),
            dtype=dtype,
            local_files_only=True,
            output_loading_info=True,
        )
        for field in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if info.get(field):
                raise ValueError(f"Transformers {field}: {info[field]}")
        verify(model.state_dict())
        loader = "transformers.AutoModelForCausalLM.from_pretrained"
    else:
        from safetensors.torch import load_file

        files = sorted(checkpoint.glob("*.safetensors"))
        if not files:
            raise ValueError("No standard safetensors files")
        for path in files:
            read_standard_header(path)
            state = load_file(path, device="cpu")
            verify(state)
            del state
        loader = "safetensors.torch.load_file"
    if checked != set(expected):
        raise ValueError(f"Standard loader omitted keys: {sorted(set(expected) - checked)}")
    return dict(
        loader=loader,
        cpu=True,
        tensors=len(checked),
        payload_bytes=payload_bytes,
        all_values_verified=True,
        pattern="position",
    )


def gate_plan(ranks):
    """Rank-count changes are storage-independent and cover both load sinks."""
    pairs = [(saved, loaded) for saved, loaded in ((4, 8), (8, 4)) if saved in ranks]
    return [
        dict(save_ranks=saved, load_ranks=loaded, mode=mode)
        for saved, loaded in pairs
        for mode in LOAD_MODES
    ]


class GateRunner:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        self.output.relative_to(ROOT / "validation")
        if self.output.exists():
            raise ValueError("Correctness output must be a new directory")
        self.lock = open("/tmp/metasl-gdt-save-gap.lock", "a")
        fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.output.mkdir(parents=True)
        self.checkpoints = self.output / "checkpoints"
        self.checkpoints.mkdir()
        self.owner = dict(pid=os.getpid(), output=str(self.output), created_ns=time.time_ns())
        dump(self.checkpoints / ".owner.json", self.owner)
        self.results = []
        self.progress = dict(
            status="running", pattern="position", ranks=args.ranks, models=args.models, gates=[]
        )
        self.update()
        self.guard = IdleGuard(self.output, args, self.update)

    def update(self, **fields):
        self.progress.update(updated_unix=time.time(), **fields)
        dump(self.output / "progress.json", self.progress)

    def owned(self):
        if load(self.checkpoints / ".owner.json") != self.owner:
            raise RuntimeError("Correctness checkpoint ownership changed")

    def idle(self, label):
        # Called before EVERY correctness process, including the native oracle.
        # No other metasl.run group may overlap these memory/IO-heavy gates.
        self.guard.wait(dict(stage="correctness", process=label))

    def process(self, label, command, *, environment=None):
        self.idle(label)
        self.update(current=label)
        print(label, flush=True)
        dump(self.output / f"{label}-command.json", command)
        started = time.monotonic()
        with (self.output / f"{label}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=ROOT,
                env=environment or runtime_environment(),
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=2400,
            )
        if result.returncode:
            raise RuntimeError(f"Correctness subprocess failed: {self.output / (label + '.log')}")
        return time.monotonic() - started

    def worker(self, label, action, model, checkpoint):
        result = self.output / f"{label}.json"
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            action,
            "--bundle",
            str(self.bundle(model)),
            "--checkpoint",
            str(checkpoint),
            "--result",
            str(result),
        ]
        self.process(label, command)
        return load(result)

    def bundle(self, model):
        bundle = self.args.bundle or (
            self.args.bundles / model if self.args.bundles else DEFAULT_BUNDLES[model]
        )
        if not (bundle / "model_manifest.json").is_file():
            raise ValueError(f"Missing prepared model bundle: {bundle}")
        return bundle

    def phase(self, label, model, checkpoint, ranks, action, mode):
        device = "cpu" if mode == "cpu" else "meta"
        output = self.output / label
        command = phase_command(
            sys.executable,
            self.bundle(model),
            checkpoint,
            output,
            ranks,
            action,
            device,
            data_pattern="position",
            validate_values=True,
            checkpoint_format="safetensors",
            safetensors_layout="shared",
            writer_limit=ranks,
        )
        environment = runtime_environment()
        for key in list(environment):
            if key.startswith("METASL_GDTENSOR_"):
                environment.pop(key)
        if device == "meta":
            environment["METASL_GDTENSOR_META_LOAD_SINK"] = mode.removeprefix("meta-")
        self.process(label, command, environment=environment)
        _, reports = validate_phase(
            output,
            action=action,
            device=device,
            ranks=ranks,
            data_pattern="position",
            validate_values=True,
            checkpoint_format="safetensors",
        )
        return dict(
            all_ranks_completed=True,
            ranks=ranks,
            mode=mode,
            all_values_verified=all(row.get("numerical_values_verified") is True for row in reports)
            if action == "load"
            else None,
            output=str(output),
        )

    def record(self, row):
        self.results.append(row)
        self.progress["gates"].append(row)
        dump(self.output / "results.json", self.results)
        self.update()

    def remove_checkpoint(self, checkpoint):
        self.owned()
        shutil.rmtree(checkpoint)

    def compare(self, label, actual, reference):
        self.idle(label)
        self.update(current=label)
        result = compare_files(actual, reference)
        dump(self.output / f"{label}.json", result)
        return result

    def run(self):
        try:
            for model in self.args.models:
                reference = self.checkpoints / f"{model}-native"
                oracle = self.worker(f"{model}-native-reference", "reference", model, reference)
                for ranks in self.args.ranks:
                    checkpoint = self.checkpoints / f"{model}-{ranks}r-cpu"
                    saved = self.phase(
                        f"{model}-{ranks}r-save-cpu", model, checkpoint, ranks, "save", "cpu"
                    )
                    files = sorted(path.name for path in checkpoint.glob("*.safetensors"))
                    if files != ["model.safetensors"]:
                        raise ValueError(f"Expected one shared file: {files}")
                    equality = self.compare(
                        f"{model}-{ranks}r-byte-compare",
                        checkpoint / files[0],
                        reference / files[0],
                    )
                    standard = self.worker(
                        f"{model}-{ranks}r-load-file", "load-file", model, checkpoint
                    )
                    row = dict(
                        gate="native-byte-equality-and-load-file",
                        model=model,
                        save_ranks=ranks,
                        passed=True,
                        save=saved,
                        oracle=oracle,
                        equality=equality,
                        standard_loader=standard,
                    )
                    if model == "1b":
                        row["transformers"] = self.worker(
                            f"{model}-{ranks}r-transformers", "transformers", model, checkpoint
                        )
                    self.record(row)
                    for spec in gate_plan(self.args.ranks):
                        if spec["save_ranks"] != ranks:
                            continue
                        loaded = self.phase(
                            f"{model}-save{ranks}-load{spec['load_ranks']}-{spec['mode']}",
                            model,
                            checkpoint,
                            spec["load_ranks"],
                            "load",
                            spec["mode"],
                        )
                        self.record(
                            dict(
                                gate="cross-rank-load",
                                model=model,
                                passed=True,
                                **spec,
                                load=loaded,
                            )
                        )
                    self.remove_checkpoint(checkpoint)
                # Two Meta source/sink implementations also produce the native
                # 1B file. A full gate is stronger than a tiny integration probe.
                if model == "1b" and not self.args.skip_meta_save:
                    for mode in ("meta-fresh", "meta-faithful"):
                        ranks = max(self.args.ranks)
                        checkpoint = self.checkpoints / f"{model}-{ranks}r-{mode}"
                        saved = self.phase(
                            f"{model}-{ranks}r-save-{mode}", model, checkpoint, ranks, "save", mode
                        )
                        equality = self.compare(
                            f"{model}-{ranks}r-{mode}-byte-compare",
                            checkpoint / "model.safetensors",
                            reference / "model.safetensors",
                        )
                        self.record(
                            dict(
                                gate="meta-native-byte-equality",
                                model=model,
                                save_ranks=ranks,
                                mode=mode,
                                passed=True,
                                save=saved,
                                equality=equality,
                            )
                        )
                        self.remove_checkpoint(checkpoint)
                self.remove_checkpoint(reference)
            self.update(status="complete", current=None, all_gates_passed=True)
        except BaseException as error:
            self.update(status="failed", error=str(error))
            raise
        finally:
            self.owned()
            for checkpoint in self.checkpoints.iterdir():
                if checkpoint.name != ".owner.json":
                    if not checkpoint.is_dir() or checkpoint.is_symlink():
                        raise RuntimeError(f"Unexpected correctness checkpoint: {checkpoint}")
                    shutil.rmtree(checkpoint)
            (self.checkpoints / ".owner.json").unlink()
            self.checkpoints.rmdir()
            dump(
                self.output / "cleanup.json",
                dict(checkpoints_removed=True, checkpoint_parent_removed=True),
            )
            self.lock.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(DEFAULT_BUNDLES), default=list(DEFAULT_BUNDLES)
    )
    parser.add_argument("--ranks", nargs="+", type=int, choices=(1, 4, 8), default=[1, 4, 8])
    parser.add_argument("--bundles", type=Path)
    parser.add_argument("--bundle", type=Path, help="Single-model prepared bundle override")
    parser.add_argument("--skip-meta-save", action="store_true")
    parser.add_argument("--idle-samples", type=int, default=2)
    parser.add_argument("--idle-sample-seconds", type=float, default=1)
    parser.add_argument("--idle-poll-seconds", type=float, default=10)
    parser.add_argument("--idle-cpu-percent", type=float, default=15)
    parser.add_argument("--idle-disk-percent", type=float, default=5)
    parser.add_argument("--idle-timeout", type=float, default=0)
    parser.add_argument(
        "--worker", choices=("reference", "load-file", "transformers"), help=argparse.SUPPRESS
    )
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker:
        if not all((args.bundle, args.checkpoint, args.result)):
            parser.error("worker requires bundle, checkpoint and result")
    else:
        if not args.output:
            parser.error("output is required")
        if len(set(args.models)) != len(args.models) or len(set(args.ranks)) != len(args.ranks):
            parser.error("models and ranks must be unique")
        if args.bundle and len(args.models) != 1:
            parser.error("bundle override requires exactly one model")
        if args.idle_samples < 1 or args.idle_sample_seconds <= 0 or args.idle_poll_seconds < 0:
            parser.error("idle samples and sample seconds must be positive; poll is nonnegative")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.worker:
        if args.worker == "reference":
            result = native_reference(args.bundle, args.checkpoint)
        else:
            result = verify_standard_loader(
                args.bundle, args.checkpoint, transformers=args.worker == "transformers"
            )
        dump(args.result, result)
    else:
        GateRunner(args).run()


if __name__ == "__main__":
    main()
