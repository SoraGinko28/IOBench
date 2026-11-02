"""Run an offline FSDP Meta/CPU/CUDA gdtensor save/load trace matrix.

Each case independently saves the same deterministic BF16 pattern
state on Meta, resident CPU, and resident CUDA, compares write traces and file
inventories, then loads the Meta-produced checkpoint into all three devices.
GPU ranks explicitly share visible CUDA device 0.
"""

import argparse
import hashlib
import itertools
import json
import math
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metasl.cache_control import checkpoint_residency  # noqa: E402
from metasl.iostats import (  # noqa: E402
    DATA_SUFFIXES,
    advise_evict_checkpoint,
    checkpoint_files,
    warm_checkpoint,
)
from metasl.trace_export import (  # noqa: E402
    REFERENCE_FIELDS,
    SIGNATURE_FIELDS,
    compare_trace_sets,
    export_rank_trace,
)

GIB = 1024**3
DEVICES = ("meta", "cpu", "cuda")
# DCP pickles the checkpoint path into .metadata. Equal-length directory names
# avoid introducing an artificial metadata write-size difference between devices.
CHECKPOINT_LABELS = {"meta": "meta", "cpu": "cpu0", "cuda": "cuda"}
MODEL_IDS = {
    "1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "3b": "Qwen/Qwen2.5-3B",
    "7b": "Qwen/Qwen2.5-7B",
}


def now():
    return datetime.now(UTC).isoformat()


def dump(path, value):
    """Replace a JSON record atomically so an interruption leaves readable progress."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def load(path):
    return json.loads(Path(path).read_text())


def cache_mode(args):
    return getattr(args, "cache_mode", None) or (
        "cold" if getattr(args, "cache_control", False) else "none"
    )


def warm_load_checkpoint(checkpoint):
    """Read payloads and metadata, then observe residency without faulting pages."""
    started = time.monotonic()
    warmup = warm_checkpoint(checkpoint)
    metadata_bytes = 0
    buffer = bytearray(16 * 1024**2)
    for path in checkpoint_files(checkpoint):
        if not path.name.endswith(DATA_SUFFIXES):
            with path.open("rb", buffering=0) as stream:
                while size := stream.readinto(buffer):
                    metadata_bytes += size
    residency = checkpoint_residency(checkpoint)
    pages = sum(row["pages"] for row in residency["files"])
    resident_pages = sum(row["resident_pages"] for row in residency["files"])
    percent = 100 * resident_pages / pages if pages else 0.0
    return dict(
        mode="warm", method="sequential read of all checkpoint data and metadata",
        warmup=warmup, metadata_bytes=metadata_bytes,
        total_bytes=warmup["data_bytes"] + metadata_bytes,
        seconds=time.monotonic() - started,
        residency_before_load=residency, resident_percent=percent,
        target_resident_percent=99.9,
        target_met=bool(pages) and all(
            row["resident_percent"] >= 99.9
            for row in residency["files"] if row["pages"]
        ),
    )


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 * 1024**2):
            digest.update(block)
    return digest.hexdigest()


def runtime_environment():
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", OMP_NUM_THREADS="1")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo")
    env.setdefault("MALLOC_ARENA_MAX", "2")
    env["PYTHONPATH"] = str(ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    return env


def environment_evidence(python):
    probe = """
import importlib.metadata as m, json, platform, sys
import torch, gdtensor
versions = {}
for name in ('torch', 'transformers', 'gdtensor', 'numpy'):
    try: versions[name] = m.version(name)
    except m.PackageNotFoundError: versions[name] = None
gpus = []
for index in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(index)
    gpus.append(dict(index=index, name=p.name, total_memory=p.total_memory,
                     capability=list(torch.cuda.get_device_capability(index))))
print(json.dumps(dict(python=sys.version, executable=sys.executable,
    platform=platform.platform(), versions=versions, torch_version=torch.__version__,
    cuda_available=torch.cuda.is_available(), cuda_version=torch.version.cuda,
    cuda_devices=gpus, gdtensor_package=str(gdtensor.__path__[0]))))
"""
    result = subprocess.run(
        [str(python), "-c", probe],
        cwd=ROOT,
        env=runtime_environment(),
        text=True,
        capture_output=True,
        check=True,
        timeout=90,
    )
    evidence = json.loads(result.stdout)
    evidence.update(
        hostname=platform.node(),
        environment={
            key: runtime_environment().get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "GLOO_SOCKET_IFNAME",
                "OMP_NUM_THREADS",
                "MALLOC_ARENA_MAX",
                "HF_HUB_OFFLINE",
                "TRANSFORMERS_OFFLINE",
            )
        },
        actual_gpu_mode="all ranks share visible CUDA device 0 explicitly",
    )
    if not evidence["cuda_available"] or not evidence["cuda_devices"]:
        raise RuntimeError("CUDA is unavailable; this three-device matrix cannot substitute CPU")
    return evidence


def source_inventory(gdtensor_package):
    files = {str(path.relative_to(ROOT)): path for path in (ROOT / "metasl").rglob("*.py")}
    for path in (
        Path(__file__), ROOT / "scripts/analyze_gdtensor_traces.py",
        ROOT / "scripts/plot_gdtensor_traces.py",
        ROOT / "requirements.txt", ROOT / "pyproject.toml",
    ):
        files[str(path.relative_to(ROOT))] = path
    package = Path(gdtensor_package)
    for path in package.rglob("*.py"):
        files["gdtensor-package/" + str(path.relative_to(package))] = path
    return {
        name: dict(source=str(path.resolve()), sha256=sha256(path))
        for name, path in sorted(files.items())
    }


def verify_artifacts(artifacts):
    for item in artifacts:
        path = Path(item["path"])
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Missing or linked completed evidence: {path}")
        if path.stat().st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Completed evidence changed: {path}")


def artifacts(paths):
    return [
        dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha256(path))
        for path in sorted(set(map(Path, paths)))
        if path.is_file()
    ]


def parameter_facts(manifest):
    """Count parameter identities once, using the meta model's explicit aliases."""
    structure = manifest["structure"]
    aliases = {}
    for group in structure.get("aliases", []):
        parameters = [name.split(":", 1)[1] for name in group if name.startswith("parameter:")]
        for name in parameters:
            aliases[name] = min(parameters)
    seen, unique, state_entries = set(), 0, 0
    for tensor in structure["tensors"]:
        if tensor["kind"] != "parameter":
            continue
        elements = math.prod(tensor["shape"])
        state_entries += elements
        identity = aliases.get(tensor["name"], tensor["name"])
        if identity not in seen:
            unique += elements
            seen.add(identity)
    return dict(
        unique_parameters=unique,
        unique_parameters_billion=unique / 1e9,
        parameter_state_entry_elements=state_entries,
        unique_parameter_bf16_bytes=unique * 2,
        estimated_checkpoint_tensor_bytes=state_entries * 2,
        count_basis="meta model manifest parameter identities with tied aliases counted once",
    )


def phase_command(
    python, bundle, checkpoint, output, ranks, action, device="meta", *,
    data_pattern="position", validate_values=True, cache_control=False,
    checkpoint_format="dcp", safetensors_max_file_mib=None, writer_limit=1,
    safetensors_layout="per-rank", safetensors_max_file_gib=None,
):
    if writer_limit < 0:
        raise ValueError("writer-limit must be nonnegative")
    command = [
        str(python),
        "-m",
        "metasl.run",
        "--action",
        action,
        "--io-backend",
        "gdtensor",
        "--checkpoint-state",
        "model",
        "--checkpoint-format",
        checkpoint_format,
        "--dcp-file-layout",
        "rank",
        "--bundle",
        str(bundle),
        "--checkpoint",
        str(checkpoint),
        "--output",
        str(output),
        "--ranks",
        str(ranks),
        "--implementation",
        "simulated" if device == "meta" else "standard",
        "--sharding",
        "fsdp2",
        "--parallel",
        f"dp_shard={ranks}",
        "--threads",
        "1",
        "--compute-dtype",
        "bf16",
        "--data-pattern",
        data_pattern,
        "--writer-limit" if action == "save" else "--reader-limit",
        str(writer_limit) if action == "save" else "0",
        "--metadata-limit",
        "4",
        "--max-pss-gib",
        "26",
        "--io-soft-pss-gib",
        "26",
        "--min-available-gib",
        "3",
        "--timeout",
        "1800",
    ]
    if safetensors_max_file_mib is not None:
        command += ["--safetensors-max-file-mib", str(safetensors_max_file_mib)]
    if safetensors_layout != "per-rank":
        command += ["--safetensors-layout", safetensors_layout]
    if safetensors_max_file_gib is not None:
        command += ["--safetensors-max-file-gib", str(safetensors_max_file_gib)]
    command += ["--trace-io"]
    if action == "load" and cache_control:
        command += ["--evict-cache", "--measure-cache-residency"]
    if action == "load" and validate_values:
        command += ["--validate-values"]
    if device == "cpu":
        command += ["--standard-device", "cpu"]
    if device == "cuda":
        command += [
            "--standard-device",
            "cuda",
            "--cuda-share-devices",
            "--cuda-devices",
            ",".join(["0"] * ranks),
        ]
    return command


def validate_phase(
    output, *, action, device, ranks, data_pattern="position", validate_values=True,
    checkpoint_format="dcp",
):
    summary = load(Path(output) / "summary.json")
    if (
        summary.get("completed") != ranks
        or summary.get("errors")
        or summary.get("action") != action
    ):
        raise RuntimeError(f"{action}/{device}: phase did not complete all {ranks} ranks")
    configuration = summary["configuration"]
    expected = dict(
        io_backend="gdtensor",
        ranks=ranks,
        checkpoint_state="model",
        checkpoint_format=checkpoint_format,
        sharding="fsdp2",
        compute_dtype="bf16",
        data_pattern=data_pattern,
        validate_values=action == "load" and validate_values,
        threads=1,
    )
    if any(configuration.get(key) != value for key, value in expected.items()):
        raise RuntimeError(f"{action}/{device}: unexpected phase configuration")
    paths = sorted(Path(output).glob("rank-*.json"))
    reports = [load(path) for path in paths if re.fullmatch(r"rank-\d+\.json", path.name)]
    if len(reports) != ranks or sorted(report.get("rank") for report in reports) != list(
        range(ranks)
    ):
        raise RuntimeError(f"{action}/{device}: rank reports are incomplete or duplicated")
    for report in reports:
        actual = report.get("state_device")
        expected_device = "cuda:0" if device == "cuda" else device
        if (
            report.get("world_size") != ranks
            or report.get("io_backend") != "gdtensor"
            or actual != expected_device
        ):
            raise RuntimeError(
                f"{action}/{device}: wrong backend, world size, or actual state device"
            )
        if report.get("cpu_state_resident") is not (device == "cpu"):
            raise RuntimeError(f"{action}/{device}: CPU residency evidence differs")
        if report.get("cuda_state_resident") is not (device == "cuda"):
            raise RuntimeError(f"{action}/{device}: CUDA residency evidence differs")
        if any(name != expected_device for name in report.get("local_tensor_devices", {})):
            raise RuntimeError(f"{action}/{device}: local tensors reside on an unexpected device")
        if device == "cuda" and report.get("stats", {}).get("cuda_synchronized") is not True:
            raise RuntimeError(f"CUDA {action} lacks synchronized timing evidence")
        if (
            not report.get("io")
            or not (Path(output) / f"rank-{report['rank']:03d}.io.jsonl").is_file()
        ):
            raise RuntimeError(f"{action}/{device}: raw recorder trace is missing")
        if action == "load":
            for field in (
                "complete_load_verified",
                "coverage_crosscheck",
            ):
                if report.get(field) is not True:
                    raise RuntimeError(f"load/{device}: {field} was not verified")
            if report.get("numerical_values_verified") is not validate_values:
                raise RuntimeError(
                    f"load/{device}: numerical_values_verified differs from requested mode"
                )
            requests = report["requests"]
            if (
                report.get("completed_items") != report.get("expected_items")
                or report["completed_items"] != len(requests)
                or report["completed_elements"]
                != sum(math.prod(row["lengths"]) for row in requests)
            ):
                raise RuntimeError(f"load/{device}: incomplete item/element coverage")
        elif len(report.get("completed_write_indices", [])) != report.get("assigned_items"):
            raise RuntimeError(f"save/{device}: producer did not finish its assigned write items")
    return summary, sorted(reports, key=lambda report: report["rank"])


def checkpoint_inventory(checkpoint):
    checkpoint = Path(checkpoint)
    entries = []
    for path in sorted(checkpoint.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Unexpected checkpoint entry: {path}")
        stat = path.stat()
        entries.append(
            dict(
                path=path.name,
                bytes=stat.st_size,
                sha256=sha256(path),
                device=stat.st_dev,
                inode=stat.st_ino,
                mtime_ns=stat.st_mtime_ns,
            )
        )
    dcp = any(row["path"].endswith(".distcp") for row in entries)
    safetensors = any(row["path"].endswith(".safetensors") for row in entries)
    if dcp == safetensors:
        raise RuntimeError("Producer checkpoint requires exactly one payload format")
    metadata = ".metadata" if dcp else ".metasl-complete.json"
    if not (checkpoint / metadata).is_file():
        raise RuntimeError(f"Producer checkpoint lacks {metadata}")
    return dict(checkpoint=str(checkpoint.resolve()), entries=entries, captured_at=now())


def compare_checkpoint_inventories(inventories_by_device):
    """Compare published names/sizes and payload bytes independently of metadata IDs."""
    comparisons = []
    for left, right in itertools.combinations(inventories_by_device, 2):
        a = {row["path"]: row for row in inventories_by_device[left]["entries"]}
        b = {row["path"]: row for row in inventories_by_device[right]["entries"]}
        differences = []
        for name in sorted(a.keys() | b.keys()):
            if name not in a or name not in b:
                differences.append(dict(path=name, field="presence", left=name in a, right=name in b))
                continue
            for field in ("bytes", "sha256"):
                if a[name][field] != b[name][field]:
                    differences.append(
                        dict(path=name, field=field, left=a[name][field], right=b[name][field])
                    )
        geometry_equal = not any(row["field"] != "sha256" for row in differences)
        payload_equal = not any(row["path"].endswith(DATA_SUFFIXES) for row in differences)
        metadata_equal = not any(not row["path"].endswith(DATA_SUFFIXES) for row in differences)
        comparisons.append(
            dict(
                left_device=left,
                right_device=right,
                status="identical" if geometry_equal and payload_equal else "different",
                names_and_sizes_status="identical" if geometry_equal else "different",
                data_byte_identity="identical" if payload_equal else "different",
                metadata_byte_identity="identical" if metadata_equal else "different",
                first_difference=next(
                    (row for row in differences if row["field"] != "sha256"
                     or row["path"].endswith(DATA_SUFFIXES)),
                    None,
                ),
                differences=differences,
                total_bytes=dict(
                    left=sum(row["bytes"] for row in a.values()),
                    right=sum(row["bytes"] for row in b.values()),
                ),
            )
        )
    result = dict(schema_version=1, devices=list(inventories_by_device), comparisons=comparisons)
    for field in ("status", "names_and_sizes_status", "data_byte_identity", "metadata_byte_identity"):
        result[field] = (
            "unknown" if not comparisons
            else "identical" if all(row[field] == "identical" for row in comparisons)
            else "different"
        )
    result["metadata_note"] = (
        "Metadata SHA256 is reported separately: independent DCP saves may contain distinct "
        "checkpoint identifiers. Overall identity requires all file names/sizes and data SHA256."
    )
    return result


def guard_checkpoint_capacity(case_dir, facts, *, remaining_checkpoints=3):
    required = math.ceil(facts["estimated_checkpoint_tensor_bytes"] * 1.1) * remaining_checkpoints + GIB
    free = shutil.disk_usage(case_dir).free
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space for {remaining_checkpoints} checkpoint(s) plus 1 GiB reserve: "
            f"need {required} bytes, free {free}"
        )
    return dict(required_bytes=required, free_bytes=free, remaining_checkpoints=remaining_checkpoints)


def verify_checkpoint(inventory, *, allow_deleted=False):
    checkpoint = Path(inventory["checkpoint"])
    if checkpoint.is_symlink():
        raise RuntimeError("Checkpoint became a symlink")
    for item in inventory["entries"]:
        path = checkpoint / item["path"]
        if allow_deleted and item["path"].endswith(DATA_SUFFIXES) and not path.exists():
            continue
        if path.is_symlink() or not path.is_file():
            raise RuntimeError(f"Checkpoint entry is missing or linked: {path}")
        stat = path.stat()
        actual = (stat.st_size, stat.st_dev, stat.st_ino, stat.st_mtime_ns)
        expected = (item["bytes"], item["device"], item["inode"], item["mtime_ns"])
        if actual != expected:
            raise RuntimeError(f"Checkpoint file identity/size/mtime changed: {path}")


def cleanup_checkpoint(case_dir, registration_id, inventory, *, device=None):
    """Delete only this runner's successful case's inventoried payload files."""
    case_dir = Path(case_dir)
    checkpoint = Path(inventory["checkpoint"])
    prefix = "checkpoint" if device is None else f"checkpoint-{device}"
    owner = load(case_dir / f"{prefix}-owner.json")
    if (
        owner.get("registration_id") != registration_id
        or owner.get("checkpoint") != str(checkpoint.resolve())
        or checkpoint.parent.resolve() != case_dir.resolve()
        or case_dir.is_symlink()
        or checkpoint.is_symlink()
    ):
        raise RuntimeError("Checkpoint deletion ownership check failed")
    journal_path = case_dir / f"{prefix}-deletion.json"
    previous = load(journal_path) if journal_path.exists() else None
    if previous and previous.get("checkpoint") != str(checkpoint.resolve()):
        raise RuntimeError("Deletion journal refers to a different checkpoint")
    verify_checkpoint(inventory, allow_deleted=previous is not None)
    selected = [row for row in inventory["entries"] if row["path"].endswith(DATA_SUFFIXES)]
    if any(Path(row["path"]).name != row["path"] for row in selected):
        raise RuntimeError("Payload inventory escapes its checkpoint directory")
    journal = dict(
        registration_id=registration_id,
        checkpoint=str(checkpoint.resolve()),
        state="deleting",
        planned=selected,
        deleted=[],
        started_at=now(),
    )
    dump(journal_path, journal)
    for item in selected:
        path = checkpoint / item["path"]
        if path.exists():
            path.unlink()
        journal["deleted"].append(item["path"])
        dump(journal_path, journal)
    journal.update(
        state="complete",
        completed_at=now(),
        deleted_bytes=sum(row["bytes"] for row in selected),
        metadata_retained=True,
    )
    dump(journal_path, journal)
    return journal


def phase_metrics(summary, reports, exports, *, action, ranks, output):
    """Summarize measured calls without reading checkpoint payloads again."""
    rows = [
        json.loads(line)
        for exported in exports
        for line in Path(exported["output_path"]).read_text().splitlines()
        if line.strip()
    ]
    operations = {"write", "pwrite"} if action == "save" else {"read", "readinto", "readline", "pread"}
    kind = "write" if action == "save" else "read"
    selected = [row for row in rows if row["op"] in operations]
    data = [row for row in selected if row["path"].endswith(DATA_SUFFIXES)]
    result = dict(
        ranks=ranks,
        file_api_events=len(rows),
        state_devices=[report["state_device"] for report in reports],
        max_rank_dcp_seconds=max(report["stats"]["seconds"] for report in reports),
        peak_sampled_total_pss_bytes=summary.get("peak_sampled_total_pss_bytes", 0),
        peak_sampled_worker_rss_bytes=summary.get("peak_sampled_worker_rss_bytes", 0),
        max_rank_process_high_water_rss_bytes=max(
            report["stats"].get("process_high_water_rss_bytes", 0) for report in reports
        ),
        max_rank_cuda_peak_allocated_bytes=max(
            report["stats"].get("cuda_memory", {}).get("peak_allocated_bytes", 0) for report in reports
        ),
        max_rank_cuda_peak_reserved_bytes=max(
            report["stats"].get("cuda_memory", {}).get("peak_reserved_bytes", 0) for report in reports
        ),
        output=str(output),
    )
    result.update({
        f"{kind}_calls": len(selected),
        f"{kind}_bytes": sum(row["completed"] for row in selected),
        f"data_{kind}_calls": len(data),
        f"data_{kind}_bytes": sum(row["completed"] for row in data),
        f"metadata_{kind}_calls": len(selected) - len(data),
        f"metadata_{kind}_bytes": sum(row["completed"] for row in selected) - sum(row["completed"] for row in data),
        f"file_api_{kind}_calls": sum(report["io"].get(f"data_{kind}_calls", 0) for report in reports),
        f"file_api_data_{kind}_bytes": sum(report["io"].get(f"data_{kind}_bytes", 0) for report in reports),
        f"gdtensor_payload_{kind}_bytes": sum(report[f"gdtensor_payload_{kind}_bytes"] for report in reports),
        "logical_payload_requests": sum(len(report["payload_requests"]) for report in reports),
    })
    if action == "save":
        result["payload_pwrite_calls"] = sum(row["op"] == "pwrite" for row in selected)
        result["complete_write_coverage_verified"] = True
    else:
        for field in ("complete_load_verified", "numerical_values_verified", "coverage_crosscheck"):
            result[field] = all(report.get(field) is True for report in reports)
    return result



class Matrix:
    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve()
        # Resolving a virtualenv's interpreter symlink would discard its environment.
        self.python = args.python.absolute()
        self.options = dict(
            models=args.models,
            ranks=args.ranks,
            configs=str(args.configs.resolve()),
            python=str(self.python),
            keep_checkpoints=args.keep_checkpoints,
            data_pattern=args.data_pattern,
            validate_values=args.validate_values,
            checkpoint_format=getattr(args, "checkpoint_format", "dcp"),
            safetensors_max_file_mib=getattr(args, "safetensors_max_file_mib", None),
            writer_limit=getattr(args, "writer_limit", 1),
            cache_control=getattr(args, "cache_control", False),
            cache_mode=cache_mode(args),
        )
        self.environment = environment_evidence(self.python)
        self.sources = source_inventory(self.environment["gdtensor_package"])
        self.configs = {
            model: dict(
                path=str((args.configs / model / "config.json").resolve()),
                sha256=sha256(args.configs / model / "config.json"),
            )
            for model in args.models
        }
        registration_path = self.output / "registration.json"
        if args.resume:
            self.registration = load(registration_path)
            for field, actual in (
                ("options", self.options),
                ("sources", self.sources),
                ("configs", self.configs),
                ("environment", self.environment),
            ):
                if self.registration[field] != actual:
                    raise RuntimeError(f"Resume {field} differs from the registered experiment")
            for name, item in self.sources.items():
                if sha256(self.output / "source" / name) != item["sha256"]:
                    raise RuntimeError(f"Archived source changed: {name}")
            for model, item in self.configs.items():
                if sha256(self.output / "input-configs" / model / "config.json") != item["sha256"]:
                    raise RuntimeError(f"Archived model configuration changed: {model}")
            self.progress = load(self.output / "progress.json")
        else:
            self.output.mkdir(parents=True, exist_ok=False)
            self.registration = dict(
                schema_version=2,
                registration_id=str(uuid.uuid4()),
                created_at=now(),
                options=self.options,
                environment=self.environment,
                sources=self.sources,
                configs=self.configs,
                fixture=f"configuration-only model; deterministic BF16 {args.data_pattern} values",
                pretrained_weights_downloaded=False,
                comparison_scope=(
                    "independent deterministic gdtensor model saves; per-rank save/load calls "
                    "and order; file names/sizes and data SHA256; same-checkpoint loads"
                ),
                reference_schema=list(REFERENCE_FIELDS),
                signature_fields=list(SIGNATURE_FIELDS),
                reference_layer="reference CFS client API; this run records Python file API",
                rank_schedule=(
                    f"serial cases; save writer_limit={self.options['writer_limit']}; "
                    "raw writer_limit=0 disables writer admission; "
                    "load reader_limit=0 (unlimited); "
                    "metadata_limit=4"
                ),
                cache_policy=(
                    "POSIX_FADV_DONTNEED after save inventory and before every load; "
                    "per-file fincore residency measured; VM host cache unobservable"
                    if self.options["cache_mode"] == "cold"
                    else "read all checkpoint data and metadata before every load; "
                    "verify >=99.9% per-file fincore/mincore residency before launch"
                    if self.options["cache_mode"] == "warm"
                    else "uncontrolled local page cache; time equality is not tested"
                ),
            )
            for name, item in self.sources.items():
                target = self.output / "source" / name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item["source"], target)
            for model, item in self.configs.items():
                target = self.output / "input-configs" / model / "config.json"
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(item["path"], target)
            dump(registration_path, self.registration)
            self.progress = dict(
                schema_version=2,
                registration_id=self.registration["registration_id"],
                status="running",
                stages={},
                cases={
                    f"{model}-r{rank:02d}": dict(model=model, ranks=rank, status="pending")
                    for model in args.models
                    for rank in args.ranks
                },
            )
        self.update("matrix registered")

    def update(self, message):
        self.progress["updated_at"] = now()
        dump(self.output / "progress.json", self.progress)
        dump(
            self.output / "summary.json",
            dict(
                schema_version=2,
                status=self.progress["status"],
                updated_at=self.progress["updated_at"],
                registration=str(self.output / "registration.json"),
                comparison_scope=(
                    "independent deterministic gdtensor FSDP model saves and same-checkpoint "
                    "loads, per-rank calls/order; checkpoint names/sizes and data SHA256"
                ),
                timing_equivalence_tested=False,
                cross_rank_interleaving_compared=False,
                actual_gpu_mode=self.environment["actual_gpu_mode"],
                cases=list(self.progress["cases"].values()),
            ),
        )
        print(f"[{datetime.now(UTC).strftime('%H:%M:%S')} UTC] {message}", flush=True)

    def verify_sources(self):
        if source_inventory(self.environment["gdtensor_package"]) != self.sources:
            raise RuntimeError("Runtime source changed after experiment registration")
        for name, item in self.sources.items():
            if sha256(self.output / "source" / name) != item["sha256"]:
                raise RuntimeError(f"Archived source changed after registration: {name}")
        for model, item in self.configs.items():
            archived = self.output / "input-configs" / model / "config.json"
            if sha256(item["path"]) != item["sha256"] or sha256(archived) != item["sha256"]:
                raise RuntimeError(f"Model configuration changed after registration: {model}")

    def stage(self, key, base_output, build_command):
        self.verify_sources()
        previous = self.progress["stages"].get(key)
        if previous and previous["status"] == "complete":
            verify_artifacts(previous["artifacts"])
            self.update(f"{key}: verified completed evidence")
            return Path(previous["output"])
        attempt = (previous.get("attempt", 0) if previous else 0) + 1
        output = (
            Path(base_output) if attempt == 1 else Path(str(base_output) + f"-attempt{attempt:02d}")
        )
        log_path = output.parent / f"{output.name}.console.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        command = build_command(output)
        stage = dict(
            status="running",
            attempt=attempt,
            output=str(output),
            command=command,
            console_log=str(log_path),
            started_at=now(),
        )
        if previous:
            stage["previous_attempts"] = previous.get("previous_attempts", []) + [
                {k: v for k, v in previous.items() if k != "previous_attempts"}
            ]
        self.progress["stages"][key] = stage
        self.update(f"{key}: started (attempt {attempt})")
        started = time.monotonic()
        try:
            if (
                cache_mode(self.args) == "warm"
                and "--action" in command
                and command[command.index("--action") + 1] == "load"
            ):
                checkpoint = Path(command[command.index("--checkpoint") + 1])
                cache_path = output.parent / "cache-control" / f"{output.name}.json"
                cache = warm_load_checkpoint(checkpoint)
                dump(cache_path, cache)
                stage["cache_control"] = str(cache_path)
                self.update(
                    f"{key}: warmed {cache['total_bytes']} bytes including metadata; "
                    f"pre-load residency {cache['resident_percent']:.3f}%"
                )
                if not cache["target_met"]:
                    raise RuntimeError(f"{key}: warm-cache residency target was not met")
            with log_path.open("w") as log:
                process = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    env=runtime_environment(),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    returncode = process.wait(timeout=1920)
                except BaseException:
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
                    raise
        except BaseException:
            stage.update(
                status="failed", ended_at=now(), elapsed_seconds=time.monotonic() - started
            )
            raise
        finally:
            if stage["status"] == "failed":
                self.update(f"{key}: interrupted; checkpoint and evidence retained")
        stage.update(
            returncode=returncode, ended_at=now(), elapsed_seconds=time.monotonic() - started
        )
        if returncode:
            stage["status"] = "failed"
            self.update(f"{key}: failed with exit {returncode}; see {log_path}")
            raise RuntimeError(f"{key} exited {returncode}; evidence retained at {output}")
        self.verify_sources()
        stage_artifacts = [*output.rglob("*"), log_path]
        if stage.get("cache_control"):
            stage_artifacts.append(Path(stage["cache_control"]))
        stage.update(status="complete", artifacts=artifacts(stage_artifacts))
        self.update(f"{key}: complete")
        return output

    def prepare(self, model):
        bundle = self.stage(
            f"prepare:{model}",
            self.output / "bundles" / model,
            lambda output: [
                str(self.python),
                "-m",
                "metasl.prepare",
                "--model-id",
                self.configs[model]["path"].rsplit("/", 1)[0],
                "--dtype",
                "bf16",
                "--local-files-only",
                "--bundle",
                str(output),
            ],
        )
        facts = parameter_facts(load(bundle / "model_manifest.json"))
        facts["model_id"] = MODEL_IDS.get(model, "local configuration " + model)
        facts["config_sha256"] = self.configs[model]["sha256"]
        return bundle, facts

    def run_case(self, model, ranks, bundle, facts):
        name = f"{model}-r{ranks:02d}"
        case = self.progress["cases"][name]
        case_dir = self.output / "cases" / name
        case_dir.mkdir(parents=True, exist_ok=True)
        if case.get("completed_at"):
            verify_artifacts(case["artifacts"])
            for key, stage in self.progress["stages"].items():
                if key.startswith(name + ":") and stage["status"] == "complete":
                    verify_artifacts(stage["artifacts"])
            if load(case_dir / "compare.json")["status"] != case["status"]:
                raise RuntimeError(f"{name}: stored comparison differs from the recorded outcome")
            for device in DEVICES:
                verify_checkpoint(
                    load(case_dir / f"checkpoint-{device}-inventory.json"),
                    allow_deleted=(case_dir / f"checkpoint-{device}-deletion.json").exists(),
                )
            self.update(f"{name}: verified completed case")
            return
        if any((case_dir / f"checkpoint-{device}-deletion.json").exists() for device in DEVICES):
            # An interrupted authorized cleanup can finish without re-reading deleted payloads.
            if not case.get("comparison_artifacts"):
                raise RuntimeError("Deletion recovery requires sealed comparison evidence")
            verify_artifacts(case["comparison_artifacts"])
            if load(case_dir / "compare.json")["status"] != "identical" or case["status"] != "identical":
                raise RuntimeError("Deletion recovery requires identical save/load and payload evidence")
            for phase in ("save", "load"):
                for device in DEVICES:
                    stage = self.progress["stages"][f"{name}:{phase}-{device}"]
                    if stage["status"] != "complete":
                        raise RuntimeError("Deletion recovery requires all six successful phases")
                    verify_artifacts(stage["artifacts"])
                    validate_phase(
                        stage["output"], action=phase, device=device, ranks=ranks,
                        data_pattern=self.args.data_pattern,
                        validate_values=self.args.validate_values,
                        checkpoint_format=getattr(self.args, "checkpoint_format", "dcp"),
                    )
            case["checkpoint_deletions"] = {
                device: cleanup_checkpoint(
                    case_dir, self.registration["registration_id"],
                    load(case_dir / f"checkpoint-{device}-inventory.json"), device=device,
                )
                for device in DEVICES
            }
            self.seal_case(case, case_dir)
            self.update(f"{name}: completed interrupted payload cleanup")
            return
        case.update(
            status="running",
            actual_parameters=facts,
            actual_gpu_mode=dict(
                shared_visible_device=0,
                rank_devices=["cuda:0"] * ranks,
                sharing_enabled=True,
                rank_count=ranks,
                visible_physical_gpus=len(self.environment["cuda_devices"]),
            ),
        )
        traces = {phase: {} for phase in ("save", "load")}
        metrics = {phase: {} for phase in ("save", "load")}
        export_manifests, inventories, inventory_paths = [], {}, {}
        checkpoints = case.setdefault("checkpoints", {})

        def record_phase(phase, device, output, checkpoint):
            summary, reports = validate_phase(
                output, action=phase, device=device, ranks=ranks,
                data_pattern=self.args.data_pattern, validate_values=self.args.validate_values,
                checkpoint_format=getattr(self.args, "checkpoint_format", "dcp"),
            )
            traces[phase][device], device_exports = {}, []
            for report in reports:
                rank = report["rank"]
                exported = export_rank_trace(
                    output / f"rank-{rank:03d}.io.jsonl",
                    case_dir / f"export-{phase}-{device}",
                    rank=rank, phase=phase,
                    origin_ns=report["io"]["origin_ns"], checkpoint_root=checkpoint,
                    payload_events=(
                        report.get("payload_events", []) + report.get("container_patch_events", [])
                        if phase == "save" else None
                    ),
                )
                traces[phase][device][rank] = exported["output_path"]
                device_exports.append(exported)
            export_manifests.extend(device_exports)
            metrics[phase][device] = phase_metrics(
                summary, reports, device_exports, action=phase, ranks=ranks, output=output,
            )
            dump(case_dir / "export-manifests.json", export_manifests)
            self.update(f"{name}: {phase}/{device} trace exported and coverage verified")

        for device in DEVICES:
            save_key = f"{name}:save-{device}"
            save_stage = self.progress["stages"].get(save_key)
            if not save_stage or save_stage["status"] != "complete":
                remaining = sum(
                    self.progress["stages"].get(f"{name}:save-{dev}", {}).get("status") != "complete"
                    for dev in DEVICES
                )
                case["disk_admission"] = guard_checkpoint_capacity(
                    case_dir, facts, remaining_checkpoints=remaining,
                )
            label = CHECKPOINT_LABELS[device]
            checkpoint = Path(checkpoints.get(device, str(case_dir / f"checkpoint-{label}")))
            if save_stage and save_stage["status"] != "complete":
                checkpoint = case_dir / f"checkpoint-{label}-attempt{save_stage.get('attempt', 0) + 1:02d}"
            checkpoints[device] = str(checkpoint)
            dump(
                case_dir / f"checkpoint-{device}-owner.json",
                dict(
                    registration_id=self.registration["registration_id"],
                    checkpoint=str(checkpoint.resolve()), producer=f"gdtensor {device} save", case=name,
                ),
            )
            cache_path = case_dir / "cache-control" / f"save-{device}.json"
            cache_enabled = cache_mode(self.args) == "cold"
            if cache_enabled and (not save_stage or save_stage["status"] != "complete"):
                before = checkpoint_residency(checkpoint)
                if checkpoint.exists() and any(checkpoint.iterdir()):
                    raise RuntimeError("Controlled save requires an empty target directory")
                dump(cache_path, dict(before_save=before))
            output = self.stage(
                save_key, case_dir / f"save-{device}",
                lambda output, device=device, checkpoint=checkpoint: phase_command(
                    self.python, bundle, checkpoint, output, ranks, "save", device,
                    data_pattern=self.args.data_pattern, validate_values=self.args.validate_values,
                    checkpoint_format=getattr(self.args, "checkpoint_format", "dcp"),
                    safetensors_max_file_mib=getattr(self.args, "safetensors_max_file_mib", None),
                    writer_limit=getattr(self.args, "writer_limit", 1),
                ),
            )
            if cache_enabled:
                cache_evidence = load(cache_path)
                if "after_save" not in cache_evidence:
                    if save_stage and save_stage["status"] == "complete":
                        raise RuntimeError("Cannot reconstruct original post-save residency on resume")
                    cache_evidence["after_save"] = checkpoint_residency(checkpoint)
                    dump(cache_path, cache_evidence)
            validate_phase(
                output, action="save", device=device, ranks=ranks,
                data_pattern=self.args.data_pattern, validate_values=self.args.validate_values,
                checkpoint_format=getattr(self.args, "checkpoint_format", "dcp"),
            )
            inventory_paths[device] = case_dir / f"checkpoint-{device}-inventory.json"
            if inventory_paths[device].exists():
                inventory = load(inventory_paths[device])
                if inventory["checkpoint"] != str(checkpoint.resolve()):
                    raise RuntimeError("Stored inventory refers to a different checkpoint")
                verify_checkpoint(inventory)
            else:
                self.update(f"{name}: preserving {device} checkpoint SHA inventory")
                inventory = checkpoint_inventory(checkpoint)
                dump(inventory_paths[device], inventory)
            inventories[device] = inventory
            record_phase("save", device, output, checkpoint)
            if cache_enabled:
                # SHA inventory reads every byte, so eviction must follow hashing.
                cache_evidence["eviction_after_inventory"] = advise_evict_checkpoint(checkpoint)
                cache_evidence["after_eviction"] = checkpoint_residency(checkpoint)
                dump(cache_path, cache_evidence)
                self.progress["stages"][save_key]["cache_control"] = str(cache_path)
                self.update(f"{name}: {device} saved checkpoint evicted and residency measured")

        save_comparison = compare_trace_sets(traces["save"], world_size=ranks)
        save_comparison.update(io_backend="gdtensor", phase="save", independent_checkpoints=True)
        checkpoint_comparison = compare_checkpoint_inventories(inventories)
        dump(case_dir / "compare-save.json", save_comparison)
        dump(case_dir / "compare-checkpoints.json", checkpoint_comparison)

        checkpoint = Path(checkpoints["meta"])
        case["load_checkpoint"] = str(checkpoint)
        for device in DEVICES:
            verify_checkpoint(inventories["meta"])
            output = self.stage(
                f"{name}:load-{device}", case_dir / f"load-{device}",
                lambda output, device=device: phase_command(
                    self.python, bundle, checkpoint, output, ranks, "load", device,
                    data_pattern=self.args.data_pattern, validate_values=self.args.validate_values,
                    cache_control=cache_mode(self.args) == "cold",
                    checkpoint_format=getattr(self.args, "checkpoint_format", "dcp"),
                    safetensors_max_file_mib=getattr(self.args, "safetensors_max_file_mib", None),
                ),
            )
            if cache_mode(self.args) == "cold":
                cache_path = output / "cache.json"
                cache = load(cache_path)
                if "residency_after_eviction" not in cache:
                    raise RuntimeError("Controlled load lacks measured residency after eviction")
                self.progress["stages"][f"{name}:load-{device}"]["cache_control"] = str(cache_path)
            elif cache_mode(self.args) == "warm":
                stage = self.progress["stages"][f"{name}:load-{device}"]
                cache = load(stage["cache_control"])
                if not cache.get("target_met") or "residency_before_load" not in cache:
                    raise RuntimeError("Warm load lacks verified pre-load residency")
            verify_checkpoint(inventories["meta"])
            record_phase("load", device, output, checkpoint)
        load_comparison = compare_trace_sets(traces["load"], world_size=ranks)
        load_comparison.update(
            checkpoint_inventory_sha256=sha256(inventory_paths["meta"]),
            same_physical_checkpoint=True, io_backend="gdtensor", phase="load",
        )
        dump(case_dir / "compare-load.json", load_comparison)
        statuses = {row["status"] for row in (save_comparison, load_comparison, checkpoint_comparison)}
        comparison = dict(
            schema_version=2,
            status="unknown" if "unknown" in statuses else "different" if "different" in statuses else "identical",
            save=save_comparison, load=load_comparison, checkpoints=checkpoint_comparison,
            timing_equivalence_tested=False, cross_rank_interleaving_compared=False,
        )
        dump(case_dir / "compare.json", comparison)
        case.update(
            status=comparison["status"], metrics=metrics,
            compare=str(case_dir / "compare.json"),
            checkpoint_inventories={dev: str(path) for dev, path in inventory_paths.items()},
        )
        immutable = [
            case_dir / filename
            for filename in ("compare.json", "compare-save.json", "compare-load.json", "compare-checkpoints.json", "export-manifests.json")
        ]
        immutable.extend(inventory_paths.values())
        if cache_mode(self.args) != "none":
            immutable.extend((case_dir / "cache-control").glob("*.json"))
        for exported in export_manifests:
            immutable.extend([Path(exported["output_path"]), Path(exported["manifest_path"])])
        case["comparison_artifacts"] = artifacts(immutable)
        self.update(f"{name}: save/load and checkpoint comparisons written")
        if comparison["status"] != "identical":
            self.seal_case(case, case_dir)
            self.update(f"{name}: comparison {comparison['status']}; checkpoint retained")
            return
        if not self.args.keep_checkpoints:
            self.update(f"{name}: save/load evidence complete; deleting owned payload files")
            case["checkpoint_deletions"] = {
                device: cleanup_checkpoint(
                    case_dir, self.registration["registration_id"], inventories[device], device=device,
                )
                for device in DEVICES
            }
        self.seal_case(case, case_dir)
        self.update(f"{name}: IDENTICAL across Meta/CPU/CUDA ({ranks} ranks)")

    def seal_case(self, case, case_dir):
        proof = [
            path
            for path in case_dir.rglob("*")
            if path.is_file()
            and not any(
                part.startswith("checkpoint")
                and part != "checkpoint-owner.json"
                and part != "checkpoint-inventory.json"
                and part != "checkpoint-deletion.json"
                for part in path.relative_to(case_dir).parts[:-1]
            )
        ]
        case["artifacts"] = artifacts(proof)
        case["completed_at"] = now()

    def run(self):
        try:
            for model in self.args.models:
                bundle, facts = self.prepare(model)
                for ranks in self.args.ranks:
                    self.run_case(model, ranks, bundle, facts)
            equal = all(case["status"] == "identical" for case in self.progress["cases"].values())
            self.progress["status"] = "complete" if equal else "complete_with_differences"
            self.update(
                "matrix complete; all cases IDENTICAL"
                if equal
                else "matrix complete; different/unknown cases and checkpoints retained"
            )
            return equal
        except BaseException as error:
            self.progress.update(status="failed", error=f"{type(error).__name__}: {error}")
            self.update("matrix stopped; failed evidence and checkpoint retained")
            raise


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--configs", type=Path, default=ROOT / "configs"
    )
    parser.add_argument("--models", default="1b,3b,7b")
    parser.add_argument("--ranks", default="1,4,8,16")
    parser.add_argument("--python", type=Path, default=Path("/tmp/metasl-gdt-cuda-venv/bin/python"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    parser.add_argument("--checkpoint-format", choices=("dcp", "safetensors"), default="dcp")
    parser.add_argument("--safetensors-max-file-mib", type=int)
    parser.add_argument("--writer-limit", type=int, default=1,
                        help="Maximum concurrent save writers; 0 disables writer admission (default: 1)")
    parser.add_argument(
        "--cache-control", action="store_true",
        help="measure empty save targets and memory, evict after saves and before each load; "
        "measure per-file residency with fincore (default: off)",
    )
    parser.add_argument(
        "--cache-mode", choices=("none", "cold", "warm"),
        help="cache policy (default: none); cold is also selected by --cache-control; "
        "warm reads data and metadata and verifies fincore residency before each load",
    )
    parser.add_argument("--data-pattern", choices=("position", "zero"), default="position")
    parser.add_argument(
        "--validate-values", action=argparse.BooleanOptionalAction, default=True,
        help="verify load values (default: enabled); completeness and coverage are always checked",
    )
    args = parser.parse_args(argv)
    if args.writer_limit < 0:
        parser.error("writer-limit must be nonnegative")
    if args.safetensors_max_file_mib is not None and (
        args.checkpoint_format != "safetensors" or args.safetensors_max_file_mib <= 0
    ):
        parser.error("safetensors-max-file-mib requires safetensors and a positive integer")
    if args.cache_control and args.cache_mode not in (None, "cold"):
        parser.error("--cache-control selects cold and conflicts with --cache-mode")
    args.cache_mode = cache_mode(args)
    args.cache_control = args.cache_mode == "cold"
    args.models = args.models.split(",")
    if len(set(args.models)) != len(args.models) or any(
        not re.fullmatch(r"[A-Za-z0-9_-]+", model) for model in args.models
    ):
        parser.error("models must be unique safe configuration directory names")
    try:
        args.ranks = [int(value) for value in args.ranks.split(",")]
    except ValueError:
        parser.error("ranks must be comma-separated positive integers")
    if len(set(args.ranks)) != len(args.ranks) or not args.ranks or min(args.ranks) < 1:
        parser.error("ranks must be unique positive integers")
    if args.resume and not (args.output / "registration.json").is_file():
        parser.error("resume requires this runner's registered output directory")
    if not args.resume and args.output.exists():
        parser.error("output already exists; choose a new directory or use --resume")
    for model in args.models:
        if not (args.configs / model / "config.json").is_file():
            parser.error(f"missing offline config: {args.configs / model / 'config.json'}")
    return args


def main(argv=None):
    if not Matrix(parse_args(argv)).run():
        raise SystemExit(1)


if __name__ == "__main__":
    main()
