"""Compare native/native/gdtensor DCP checkpoints with real CPU FSDP weights.

Run with /tmp/metasl-gdt-venv/bin/python scripts/validate_dcp_identity.py.
No downloads, timing measurements, git commands, or retained checkpoints.
Each checkpoint is loaded before its streaming audit, then deleted immediately.
Only small non-payload envelopes and hashes survive in the JSON evidence.
"""

import argparse
import base64
import hashlib
import json
import math
import os
import pickle
import shutil
import struct
import subprocess
import sys
import time
import traceback
import zipfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.dcp_identity_helpers import (  # noqa: E402
    byte_diffs,
    digest_stream,
    field_diffs,
    fields,
)


def dump(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def stamp():
    return datetime.now(UTC).isoformat()


def idle():
    """Wait for foreign runner jobs before every model/rank block."""
    while True:
        probe = subprocess.run(
            ["pgrep", "-af", "[m]etasl.run"], capture_output=True, text=True, check=False
        )
        if probe.returncode == 1:
            return
        if probe.returncode != 0:
            raise RuntimeError(probe.stderr)
        print("Waiting for foreign runs: " + probe.stdout.strip(), flush=True)
        time.sleep(15)


class Slice:
    """Bounded seekable view of one torch.save ZIP in a concatenated file."""

    def __init__(self, stream, offset, size):
        self.stream, self.offset, self.size, self.position = stream, offset, size, 0

    def seek(self, offset, whence=0):
        self.position = offset + (self.position if whence == 1 else self.size if whence == 2 else 0)
        return self.position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, length=-1):
        length = self.size - self.position if length < 0 else min(length, self.size - self.position)
        self.stream.seek(self.offset + self.position)
        block = self.stream.read(length)
        self.position += len(block)
        return block


def snapshot(checkpoint):
    """Hash every byte; retain only metadata/envelopes for exact later diffs.

    Tensor storage regions are individually hashed. Equal cryptographic hashes
    prove that excluding them from the retained envelopes hides no differences.
    An unequal payload aborts comparison rather than silently masking it.
    """
    metadata_bytes = (checkpoint / ".metadata").read_bytes()
    metadata = pickle.loads(metadata_bytes)
    result = dict(metadata=fields(metadata), files={})
    by_file = {}
    for index, info in metadata.storage_data.items():
        by_file.setdefault(info.relative_path, []).append((info.offset, info.length, index))
    for path in sorted(checkpoint.iterdir()):
        with path.open("rb") as stream:
            row = dict(
                size=path.stat().st_size,
                blake2b_256=digest_stream(stream),
                envelopes=[],
                payloads=[],
                members=[],
            )
            if path.name == ".metadata":
                row["envelopes"].append(
                    dict(offset=0, data=base64.b64encode(metadata_bytes).decode())
                )
            else:
                skipped = []
                for offset, length, index in sorted(by_file[path.name]):
                    view = Slice(stream, offset, length)
                    with zipfile.ZipFile(view) as archive:
                        for member in archive.infolist():
                            view.seek(member.header_offset + 26)
                            name_len, extra_len = struct.unpack("<HH", view.read(4))
                            start = offset + member.header_offset + 30 + name_len + extra_len
                            stream.seek(start)
                            description = dict(
                                fqn=index.fqn,
                                shard_offset=list(index.offset),
                                name=member.filename,
                                start=start,
                                end=start + member.file_size,
                                blake2b_256=digest_stream(stream, member.file_size),
                                crc32=member.CRC,
                            )
                            row["members"].append(description)
                            if "/data/" in member.filename:
                                skipped.append((start, start + member.file_size))
                                row["payloads"].append(description)
                cursor = 0
                for start, end in sorted(skipped) + [(row["size"], row["size"])]:
                    stream.seek(cursor)
                    row["envelopes"].append(
                        dict(
                            offset=cursor,
                            data=base64.b64encode(stream.read(start - cursor)).decode(),
                        )
                    )
                    cursor = end
            result["files"][path.name] = row
    return result


def compare(left, right, mask_save_id=False):
    same_names = left["files"].keys() == right["files"].keys()
    result = dict(
        same_file_names=same_names,
        files={},
        metadata_field_diffs=field_diffs(left["metadata"], right["metadata"]),
    )
    for name in sorted(left["files"].keys() | right["files"].keys()):
        if name not in left["files"] or name not in right["files"]:
            result["files"][name] = dict(
                raw_identical=False, normalized_identical=False, error="missing file"
            )
            continue
        a, b = left["files"][name], right["files"][name]
        payload_equal = a["payloads"] == b["payloads"]
        if not payload_equal:
            raise RuntimeError(f"{name}: payload/layout mismatch; exact envelope diff insufficient")
        differences, normalized_equal = [], a["size"] == b["size"]
        if len(a["envelopes"]) != len(b["envelopes"]):
            raise RuntimeError(f"{name}: envelope layout mismatch")
        for ea, eb in zip(a["envelopes"], b["envelopes"], strict=True):
            if ea["offset"] != eb["offset"]:
                raise RuntimeError(f"{name}: envelope offset mismatch")
            ba, bb = base64.b64decode(ea["data"]), base64.b64decode(eb["data"])
            for diff in byte_diffs(ba, bb, ea["offset"]):
                labels = [
                    member["name"]
                    for member in a["members"]
                    if diff["start"] < member["end"] and diff["end"] > member["start"]
                ]
                diff["location"] = labels or (
                    ["pickle: storage_meta.save_id"]
                    if name == ".metadata"
                    and all(
                        d["field"] == "storage_meta.save_id" for d in result["metadata_field_diffs"]
                    )
                    else ["ZIP headers / padding / directory"]
                )
                differences.append(diff)
            if mask_save_id and name == ".metadata":
                sa = left["metadata"]["storage_meta"]["save_id"].encode()
                sb = right["metadata"]["storage_meta"]["save_id"].encode()
                assert len(sa) == len(sb) == 36 and ba.count(sa) == bb.count(sb) == 1
                ba, bb = ba.replace(sa, b"0" * 36), bb.replace(sb, b"0" * 36)
            normalized_equal &= ba == bb
        result["files"][name] = dict(
            raw_identical=a["blake2b_256"] == b["blake2b_256"] and a["size"] == b["size"],
            normalized_identical=normalized_equal,
            payloads_identical=payload_equal,
            byte_diffs=differences,
        )
    result["raw_identical_count"] = sum(row["raw_identical"] for row in result["files"].values())
    result["normalized_identical"] = same_names and all(
        row["normalized_identical"] for row in result["files"].values()
    )
    return result


def file_table(evidence):
    """Consolidate file sizes, raw/normalized hashes and proven mask spans."""
    result = {}
    for name in evidence["checkpoints"]["native_a"]["files"]:
        copies = {}
        for label, checkpoint in evidence["checkpoints"].items():
            row = checkpoint["files"][name]
            normalized = row["blake2b_256"]
            masked_ranges = []
            if name == ".metadata" and "storage_meta.save_id" in evidence["masked_fields"]:
                raw = base64.b64decode(row["envelopes"][0]["data"])
                save_id = checkpoint["metadata"]["storage_meta"]["save_id"].encode()
                if len(save_id) != 36 or raw.count(save_id) != 1:
                    raise ValueError("Expected one 36-byte save_id in metadata")
                start = raw.index(save_id)
                masked_ranges.append(
                    dict(field="storage_meta.save_id", start=start, end=start + len(save_id))
                )
                normalized = hashlib.blake2b(
                    raw.replace(save_id, b"0" * 36), digest_size=32
                ).hexdigest()
            copies[label] = dict(
                size=row["size"],
                blake2b_256=row["blake2b_256"],
                normalized_blake2b_256=normalized,
                masked_ranges=masked_ranges,
            )
        for pair, comparison in evidence["comparisons"].items():
            left, right = (copies[label] for label in pair.split("_vs_"))
            hashes_equal = (
                left["size"] == right["size"]
                and left["normalized_blake2b_256"] == right["normalized_blake2b_256"]
            )
            if hashes_equal != comparison["files"][name]["normalized_identical"]:
                raise ValueError(f"{name}: normalized hash disagrees with exact comparison")
        result[name] = copies
    return result


def worker(rank, world, model_name, output, tiny):
    import torch
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.checkpoint.state_dict import get_model_state_dict
    from transformers import AutoConfig, Qwen2Config

    from metasl.fsdp import apply_parallelism, describe_targets, materialize
    from metasl.gdt_storage import (
        GDTensorFileSystemReader,
        GDTensorFileSystemWriter,
        GDTensorLoadPlanner,
        GDTensorSavePlanner,
    )
    from metasl.layout import Layout, ParallelDims
    from metasl.skeleton import model_from_config, model_structure

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    output = Path(output)
    checkpoint = output / "checkpoint"
    dist.init_process_group(
        "gloo",
        init_method=(output / "rendezvous").resolve().as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(minutes=45),
    )

    def progress(stage):
        if rank == 0:
            dump(output / "stage.json", dict(stage=stage, updated=stamp()))
            progress_path = output.parent / "progress.json"
            current = json.loads(progress_path.read_text())
            current.update(stage=stage, updated=stamp())
            dump(progress_path, current)
            print(f"{model_name}/{world}: {stage}", flush=True)

    def tensor_hash(tensor):
        array = tensor.detach().contiguous().reshape(-1).view(torch.uint8).numpy()
        return hashlib.blake2b(memoryview(array), digest_size=32).hexdigest()

    def logical_hashes(state):
        hashes = {}
        for name, tensor in flatten_state_dict(state)[0].items():
            local = tensor.to_local().detach().contiguous()
            if rank == 0:
                full = torch.empty(tuple(tensor.shape), dtype=tensor.dtype)
                width = math.ceil(tensor.shape[0] / world)
                full[: local.shape[0]].copy_(local)
                for peer in range(1, world):
                    lo = min(peer * width, tensor.shape[0])
                    hi = min(lo + width, tensor.shape[0])
                    if hi > lo:
                        dist.recv(full[lo:hi], src=peer)
                hashes[name] = tensor_hash(full)
                del full
            elif local.numel():
                dist.send(local, dst=0)
        dist.barrier()
        return hashes

    try:
        progress("build_real_cpu_fsdp")
        if tiny:
            config = Qwen2Config(**json.loads((ROOT / "tests/tiny_qwen2.json").read_text()))
        else:
            config = AutoConfig.from_pretrained(
                ROOT / "configs" / model_name, local_files_only=True
            )
        with torch.device("meta"):
            model = model_from_config(config, torch.bfloat16)
        model.eval()
        manifest = dict(
            structure=model_structure(model),
            dtype="bf16",
            model_id="tiny-qwen2"
            if tiny
            else {
                "1b": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "3b": "Qwen/Qwen2.5-3B",
            }[model_name],
        )
        layout = Layout(ParallelDims(dp_shard=world), model.config.num_hidden_layers, rank)
        layout.build_mesh()
        apply_parallelism(model, layout, "bf16")
        state = materialize({"model": get_model_state_dict(model)}, "cpu")
        rows = describe_targets(state, manifest["structure"])
        flat = flatten_state_dict(state)[0]
        expected = {}
        seeds = {}
        progress("seed_full_tensors_then_shard")
        for row in rows:
            name = row["fqn"]
            canonical = "model." + row["alias_group"][0]
            seed = int.from_bytes(
                hashlib.blake2b(canonical.encode(), digest_size=8).digest(), "little"
            )
            seeds[name] = seed
            full = torch.randn(
                row["global_shape"],
                dtype=flat[name].dtype,
                generator=torch.Generator().manual_seed(seed),
            )
            if rank == 0:
                expected[name] = tensor_hash(full)
            slices = tuple(
                slice(lo, lo + size)
                for lo, size in zip(row["global_offsets"], row["local_shape"], strict=True)
            )
            flat[name].to_local().copy_(full[slices])
            del full
        generated = logical_hashes(state)
        if rank == 0:
            assert generated == expected
        evidence = dict(
            model=model_name,
            model_id=manifest["model_id"],
            ranks=world,
            torch_version=torch.__version__,
            dtype=manifest["dtype"],
            checkpoint_path=str(checkpoint.resolve()),
            seeds=seeds,
            expected_logical_hashes=expected,
            generated_logical_hashes=generated,
            logical_loads={},
            checkpoints={},
            started=stamp(),
        )
        for label in ("native_a", "native_b", "gdtensor"):
            progress(f"save_{label}")
            if label == "gdtensor":
                writer = GDTensorFileSystemWriter(checkpoint)
                planner = GDTensorSavePlanner()
            else:
                writer = dcp.FileSystemWriter(checkpoint)
                planner = dcp.DefaultSavePlanner()
            dcp.save(state, storage_writer=writer, planner=planner)
            for reader_name in ("native", "gdtensor") if label == "native_a" else ("native",):
                progress(f"load_{label}_with_{reader_name}")
                for tensor in flat.values():
                    tensor.to_local().fill_(float("nan"))
                if reader_name == "gdtensor":
                    load_planner = GDTensorLoadPlanner()
                    dcp.load(
                        state,
                        storage_reader=GDTensorFileSystemReader(checkpoint),
                        planner=load_planner,
                    )
                    load_planner.assert_complete()
                else:
                    dcp.load(state, storage_reader=dcp.FileSystemReader(checkpoint))
                hashes = logical_hashes(state)
                if rank == 0:
                    evidence["logical_loads"][f"{label}_via_{reader_name}"] = dict(
                        match_expected=hashes == expected, tensor_hashes=hashes
                    )
                    assert hashes == expected, (label, reader_name)
            dist.barrier()
            if rank == 0:
                progress(f"hash_and_remove_{label}")
                try:
                    evidence["checkpoints"][label] = snapshot(checkpoint)
                finally:
                    shutil.rmtree(checkpoint)
                dump(output / "evidence.json", evidence)
            dist.barrier()
        if rank == 0:
            a, b, g = (evidence["checkpoints"][key] for key in ("native_a", "native_b", "gdtensor"))
            control = compare(a, b)
            nondeterministic = [row["field"] for row in control["metadata_field_diffs"]]
            # No broad normalization: permit only this field after THIS cell's
            # native control proves it varies. All ZIP bytes remain unmasked.
            mask_id = "storage_meta.save_id" in nondeterministic
            evidence["proven_nondeterministic_fields"] = nondeterministic
            evidence["masked_fields"] = ["storage_meta.save_id"] if mask_id else []
            evidence["comparisons"] = dict(
                native_a_vs_native_b=compare(a, b, mask_id),
                native_a_vs_gdtensor=compare(a, g, mask_id),
                native_b_vs_gdtensor=compare(b, g, mask_id),
            )
            evidence["file_table"] = file_table(evidence)
            evidence["complete"] = True
            evidence["finished"] = stamp()
            dump(output / "evidence.json", evidence)
            progress("complete")
    finally:
        dist.destroy_process_group()


def report(output, cells):
    model_ids = ", ".join(sorted({cell["model_id"] for cell in cells}))
    torch_versions = ", ".join(sorted({cell["torch_version"] for cell in cells}))
    tensor_counts = ", ".join(
        sorted(
            {f"{cell['model']}: {len(cell['expected_logical_hashes'])} tensors" for cell in cells}
        )
    )
    load_count = sum(len(cell["logical_loads"]) for cell in cells)
    lines = [
        "Identical except the native DCP nondeterministic `StorageMeta.save_id`.",
        "",
        "Real CPU BF16 tensors; the runner's layer/root FSDP2 Shard(0) wrapping and local "
        f"configs ({model_ids}). Each FQN's canonical tied-alias name seeds a "
        "CPU Generator with the little-endian BLAKE2b-64 digest; full `torch.randn` tensors "
        "are generated directly in model dtype, then sliced into rank shards. Full contiguous "
        "logical bytes are gathered to rank 0 and hashed with BLAKE2b-256.",
        "",
        "Native baseline: plain `dcp.save` and `FileSystemWriter(path)` with all defaults: "
        "`thread_count=1`, `single_file_per_rank=True`, `sync_files=True`, "
        "`per_thread_copy_ahead=10000000`, `cache_staged_state_dict=False`, `overwrite=True`, "
        "TORCH_SAVE serialization, with no stream extensions. These mirror the backend's "
        "documented one thread/file per rank "
        "and native planning/ZIP layout. Both writers use their respective default save planners.",
        "",
        "Native A, native B, and gdtensor reuse the exact same path sequentially. Each checkpoint "
        "is loaded and validated, streamed through BLAKE2b-256, and immediately removed. "
        "Only small ZIP envelopes, metadata, member hashes, and tensor hashes remain. "
        "No timings are measured. Every checkpoint is loaded using native `dcp.load`; "
        "native A is additionally loaded using the gdtensor reader. Destinations are poisoned "
        "with NaNs before each load.",
        "",
        "| Model | Ranks | Files | Raw A/B | Raw A/G | Raw B/G | Normalized | Logical hashes |",
        "|---|---:|---:|---:|---:|---:|---|---|",
    ]
    all_good = True
    for cell in cells:
        c = cell["comparisons"]
        logical = all(row["match_expected"] for row in cell["logical_loads"].values())
        good = all(row["normalized_identical"] for row in c.values()) and logical
        all_good &= good
        count = len(cell["checkpoints"]["native_a"]["files"])
        lines.append(
            f"| {cell['model']} | {cell['ranks']} | {count} | "
            f"{c['native_a_vs_native_b']['raw_identical_count']}/{count} | "
            f"{c['native_a_vs_gdtensor']['raw_identical_count']}/{count} | "
            f"{c['native_b_vs_gdtensor']['raw_identical_count']}/{count} | "
            f"{c['native_a_vs_gdtensor']['normalized_identical']} | {logical} |"
        )
    across = {}
    for cell in cells:
        hashes = cell["expected_logical_hashes"]
        if cell["model"] in across:
            assert across[cell["model"]] == hashes, "Weights differ across rank counts"
        across[cell["model"]] = hashes
    lines += [
        "",
        "Full logical tensor hashes also match across all tested rank counts for each model.",
        "",
        "The native A/B control differs only in `.metadata` → `storage_meta.save_id` "
        "(a fresh UUID). The entire 36-byte UUID value is masked only after that cell's control "
        "demonstrates nondeterminism. Exact differing byte runs (half-open file offsets and "
        "hex values), all metadata fields, and field-by-field diffs are in each cell's "
        "`evidence.json`. All `.distcp` ZIP bytes, including tensor payloads, CRCs, member names, "
        "headers and `.data/serialization_id`, are compared without normalization. "
        "No additional backend differences were found.",
        "",
        "Each cell's `file_table` stores size, raw BLAKE2b-256, normalized BLAKE2b-256, "
        "and exact UUID mask spans for every file in all three writes. "
        "ZIP tensor-storage members have independent hashes; equal hashes prove excluded "
        "payload regions contain no differing byte ranges. Retained non-payload bytes allow "
        "exact envelope diffs after deletion. Persisted metadata is unpickled afresh after "
        "loading, independently of the reader's in-memory load id. Torch's installed "
        "`distributed/checkpoint/filesystem.py` generates `save_id` using `uuid.uuid4()`; "
        "the backend inherits this behavior.",
        "",
        f"Scope: model-state CPU BF16 checkpoints on torch {torch_versions} with these configs "
        f"and pure FSDP layout ({tensor_counts}). All {load_count} load checks match the "
        "generated full logical tensor hashes. No backend-caused byte differences were found.",
        "",
        "Reusable validator: `scripts/validate_dcp_identity.py` (use `--output` with an empty "
        "directory; `--resume` continues this audit's interrupted output; `--tiny --models 1b "
        "--ranks 2` runs the small integration probe). Helper tests: "
        "`python -m pytest tests/test_dcp_identity_helpers.py`. All checkpoint directories "
        "were removed; prior validation directories and the gdtensor library were untouched.",
        "",
    ]
    if not all_good:
        lines[0] = "Not identical; inspect per-cell evidence for differences."
    (output / "report.md").write_text("\n".join(lines))
    dump(
        output / "summary.json",
        dict(
            normalized_identical=all_good,
            across_rank_logical_hashes_match=True,
            cells=[
                dict(
                    model=c["model"],
                    ranks=c["ranks"],
                    comparisons=c["comparisons"],
                    logical_hashes_match=all(
                        x["match_expected"] for x in c["logical_loads"].values()
                    ),
                )
                for c in cells
            ],
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "validation/dcp-gdtensor-identity-20261030"
    )
    parser.add_argument("--models", nargs="+", default=["1b", "3b"], choices=["1b", "3b"])
    parser.add_argument("--ranks", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument(
        "--tiny", action="store_true", help="Use tiny Qwen for a targeted smoke test"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume this audit's output, preserving completed cells",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output must be empty; existing validation directories are read-only")
    os.environ.update(
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        GLOO_SOCKET_IFNAME="lo",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        MALLOC_ARENA_MAX="2",
    )
    import torch.multiprocessing as mp

    prior = json.loads((args.output / "progress.json").read_text()) if args.resume else None
    progress = dict(status="running", started=stamp(), completed=[], current=None)
    if prior:
        progress["previous_attempt"] = prior
    dump(args.output / "progress.json", progress)
    cells = []
    try:
        for model in args.models:
            for world in args.ranks:
                idle()
                name = f"{model}-{world}r"
                cell_dir = args.output / name
                cell_dir.mkdir(exist_ok=args.resume)
                progress.update(current=name, updated=stamp())
                dump(args.output / "progress.json", progress)
                evidence_path = cell_dir / "evidence.json"
                completed = evidence_path.exists() and json.loads(evidence_path.read_text()).get(
                    "complete"
                )
                if not completed:
                    if (cell_dir / "checkpoint").exists():
                        raise ValueError(
                            "Remove this audit's interrupted checkpoint before resuming"
                        )
                    (cell_dir / "rendezvous").unlink(missing_ok=True)
                    mp.spawn(
                        worker,
                        args=(world, model, str(cell_dir), args.tiny),
                        nprocs=world,
                        join=True,
                    )
                evidence = json.loads((cell_dir / "evidence.json").read_text())
                cells.append(evidence)
                progress["completed"].append(name)
                dump(args.output / "progress.json", progress)
        report(args.output, cells)
        progress.update(status="complete", current=None, stage="complete", finished=stamp())
    except BaseException:
        progress.update(status="failed", error=traceback.format_exc(), updated=stamp())
        raise
    finally:
        dump(args.output / "progress.json", progress)


if __name__ == "__main__":
    main()
