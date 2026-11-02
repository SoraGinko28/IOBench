"""Native distributed safetensors IO with admission and a completion manifest.

Tensor serialization and copying remain in Torch's HuggingFace storage classes.
The manifest binds the complete set of shard headers, since this writer does not
produce DCP's .metadata file. It is deliberately a model-tensor-only format.
"""

import dataclasses
import hashlib
import json
import math
import os
import struct
import tempfile
from contextlib import nullcontext
from pathlib import Path
from threading import get_native_id
from time import perf_counter, perf_counter_ns

import torch
from safetensors.torch import _getdtype
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter
from torch.distributed.checkpoint._hf_utils import _HFStorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata

from metasl.procmem import memory
from metasl.sharding import check_partition, element_size
from metasl.storage import AdmittedFileSystemReader, AdmittedFileSystemWriter

COMPLETION_MARKER = ".metasl-complete.json"


def canonical_safetensors_header(header):
    """Keep native tensor order/layout, making Rust's metadata map deterministic.

    safetensors serializes its text metadata from a randomly seeded HashMap.
    Ordering that map changes no offsets or lengths, but makes identical native
    and gdtensor saves reproducible. Only the small JSON header is constructed.
    """
    header = dict(header)
    if "__metadata__" in header:
        header["__metadata__"] = dict(sorted(header["__metadata__"].items()))
    raw = json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    raw += b" " * (-len(raw) % 8)
    return struct.pack("<Q", len(raw)) + raw


def safetensors_file_buckets(rows, max_file_mib):
    """Assign complete local tensor items to deterministic native HF files.

    The limit applies to data bytes. A tensor larger than the limit occupies its
    own file; the safetensors prefix and header are additional file bytes.
    """
    if max_file_mib is not None and (type(max_file_mib) is not int or max_file_mib <= 0):
        raise ValueError("safetensors-max-file-mib must be a positive integer")
    limit = max_file_mib * 1024**2 if max_file_mib is not None else None
    mapping, payloads = {}, []
    for row in sorted(rows, key=lambda row: row["fqn"]):
        name, size = row["fqn"], row["payload_bytes"]
        if not name.startswith("model.") or name in mapping:
            raise ValueError("Safetensors file buckets require unique model tensor keys")
        shape = _integer_list(row["local_shape"], "bucket local shape")
        dtype = getattr(torch, row["dtype"].removeprefix("torch."))
        if type(size) is not int or size < 0 or size != math.prod(shape) * element_size(dtype):
            raise ValueError("Safetensors bucket payload disagrees with local shape and dtype")
        if not payloads or (limit is not None and payloads[-1] + size > limit):
            payloads.append(0)
        mapping[name] = len(payloads)
        payloads[-1] += size
    largest = max((row["payload_bytes"] for row in rows), default=0)
    return (
        mapping if limit is not None else None,
        dict(
            max_file_mib=max_file_mib,
            data_limit_bytes=limit,
            assignment=(
                "sorted-fqn-next-fit-v1" if limit is not None else "native-one-file-per-rank"
            ),
            largest_item_payload_bytes=largest,
            data_upper_bound_bytes=max(largest, limit) if limit is not None else sum(payloads),
            estimated_local_bucket_payload_bytes=payloads,
            header_bytes_are_additional=True,
        ),
    )


@dataclasses.dataclass
class SafetensorsStorageInfo(_HFStorageInfo):
    """Native HF storage information plus real tensor extents for reporting."""

    offset: int = 0
    length: int = 0


def validate_empty_checkpoint(path):
    """Reject reusing a checkpoint; ranks call this before DCP's plan collective."""
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError("Safetensors save requires an empty or absent checkpoint directory")


def _integer_list(value, field):
    if not isinstance(value, list) or any(type(n) is not int or n < 0 for n in value):
        raise ValueError(f"Invalid safetensors {field}")
    return value


def _read_header(file, *, shared=False, stream=None):
    """Read only the header, retaining offsets into the original native file."""
    size = file.stat().st_size
    with file.open("rb") if stream is None else nullcontext(stream) as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors header: {file.name}")
        header_size = struct.unpack("<Q", prefix)[0]
        if header_size > min(size - 8, 100_000_000):
            raise ValueError(f"Invalid safetensors header length: {file.name}")
        raw = stream.read(header_size)
    try:
        header = json.loads(raw)
        extra = header.get("__metadata__", {})
        if shared:
            if not isinstance(extra, dict) or "DCP_SHARDING_INFO" in extra:
                raise ValueError("Shared safetensors must use standard full-tensor headers")
            sharding = None
        else:
            if not isinstance(extra, dict) or extra.get("DCP_VERSION") != "1.0":
                raise ValueError("Unsupported safetensors DCP header version")
            if extra.get("format") != "pt":
                raise ValueError("Unsupported safetensors tensor format")
            sharding = json.loads(extra["DCP_SHARDING_INFO"])
            if not isinstance(sharding, dict):
                raise ValueError("Invalid safetensors sharding metadata")
        entries = {name: entry for name, entry in header.items() if name != "__metadata__"}
        if sharding is not None and set(entries) != set(sharding):
            raise ValueError("Safetensors keys and sharding metadata disagree")
        records = []
        payload_start = 8 + header_size
        for name, entry in entries.items():
            if not shared and not name.startswith("model."):
                raise ValueError("Safetensors supports model tensor entries only")
            shape = _integer_list(entry["shape"], "shape")
            offsets = (
                [0] * len(shape)
                if shared
                else _integer_list(sharding[name]["saved_offsets"], "saved offsets")
            )
            interval = _integer_list(entry["data_offsets"], "data offsets")
            if len(offsets) != len(shape) or len(interval) != 2:
                raise ValueError("Invalid safetensors tensor dimensions or extent")
            dtype = _getdtype(entry["dtype"])
            start, end = interval
            if end < start or end - start != math.prod(shape) * element_size(dtype):
                raise ValueError("Safetensors tensor extent does not match shape and dtype")
            if payload_start + end > size:
                raise ValueError("Safetensors tensor exceeds its file")
            records.append(
                dict(
                    fqn=f"model.{name}" if shared else name,
                    offsets=offsets,
                    shape=shape,
                    dtype=str(dtype),
                    file=file.name,
                    file_offset=payload_start + start,
                    file_bytes=end - start,
                )
            )
        cursor = payload_start
        for record in sorted(records, key=lambda r: (r["file_offset"], r["file_bytes"])):
            if record["file_offset"] != cursor:
                raise ValueError("Safetensors payload has a gap or overlapping tensor data")
            cursor += record["file_bytes"]
        if cursor != size:
            raise ValueError("Safetensors payload has unreferenced bytes")
    except (KeyError, TypeError, AttributeError, json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"Invalid safetensors header: {file.name}") from error
    return (
        dict(
            path=file.name,
            size_bytes=size,
            header_sha256=hashlib.sha256(prefix + raw).hexdigest(),
        ),
        records,
    )


def _scan_checkpoint(path, *, shared=False, header_streams=None):
    path = Path(path)
    files, records = [], []
    for file in sorted(path.iterdir()):
        if file.name == COMPLETION_MARKER:
            continue
        if shared and file.name == "model.safetensors.index.json":
            if file.is_symlink() or not file.is_file():
                raise ValueError("Safetensors index must be a regular file")
            continue
        if file.is_symlink() or not file.is_file() or file.suffix != ".safetensors":
            raise ValueError(f"Unexpected safetensors checkpoint entry: {file.name}")
        info, tensors = _read_header(
            file,
            shared=shared,
            stream=None if header_streams is None else header_streams[file.name],
        )
        files.append(info)
        records.extend(tensors)
    if not files:
        raise ValueError("Safetensors checkpoint has no data files")
    if shared:
        index_path = path / "model.safetensors.index.json"
        if len(files) > 1 and not index_path.exists():
            raise ValueError("Split shared safetensors checkpoint requires an HF index")
        if index_path.exists():
            try:
                index = json.loads(index_path.read_text())
                expected_map = {r["fqn"].removeprefix("model."): r["file"] for r in records}
                if index.get("weight_map") != expected_map or index.get("metadata", {}).get(
                    "total_size"
                ) != sum(r["file_bytes"] for r in records):
                    raise ValueError("Shared safetensors index disagrees with tensor headers")
            except (OSError, json.JSONDecodeError, AttributeError) as error:
                raise ValueError("Invalid shared safetensors index") from error
    tensors = {}
    indices = set()
    for record in records:
        key = (record["fqn"], tuple(record["offsets"]))
        if key in indices:
            raise ValueError("Duplicate safetensors tensor chunk")
        indices.add(key)
        tensor = tensors.setdefault(
            record["fqn"],
            dict(
                fqn=record["fqn"],
                dtype=record["dtype"],
                shape=[0] * len(record["shape"]),
                chunks=[],
            ),
        )
        if tensor["dtype"] != record["dtype"] or len(tensor["shape"]) != len(record["shape"]):
            raise ValueError("Safetensors chunks disagree on tensor dtype or dimensions")
        tensor["shape"] = [
            max(current, offset + extent)
            for current, offset, extent in zip(
                tensor["shape"], record["offsets"], record["shape"], strict=True
            )
        ]
        tensor["chunks"].append(dict(offsets=record["offsets"], sizes=record["shape"]))
    schema = []
    for name in sorted(tensors):
        tensor = tensors[name]
        tensor["chunks"].sort(key=lambda c: (c["offsets"], c["sizes"]))
        check_partition(tensor["shape"], [(c["offsets"], c["sizes"]) for c in tensor["chunks"]])
        schema.append(tensor)
    return files, records, schema


def _metadata_schema(metadata):
    tensors = []
    for name, item in sorted(metadata.state_dict_metadata.items()):
        if not name.startswith("model.") or not isinstance(item, TensorStorageMetadata):
            raise ValueError("Safetensors supports model tensor entries only")
        tensors.append(
            dict(
                fqn=name,
                dtype=str(item.properties.dtype),
                shape=list(item.size),
                chunks=sorted(
                    (dict(offsets=list(c.offsets), sizes=list(c.sizes)) for c in item.chunks),
                    key=lambda c: (c["offsets"], c["sizes"]),
                ),
            )
        )
    return tensors


def _validated_checkpoint(path, *, header_streams=None):
    path = Path(path)
    try:
        marker = path / COMPLETION_MARKER
        if marker.is_symlink():
            raise ValueError("Safetensors completion marker must be a regular file")
        manifest = json.loads(marker.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("Missing or invalid safetensors completion marker") from error
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        raise ValueError("Unsupported safetensors completion manifest version")
    if manifest.get("checkpoint_format") != "safetensors":
        raise ValueError("Checkpoint completion manifest format mismatch")
    if type(manifest.get("world_size")) is not int or manifest["world_size"] < 1:
        raise ValueError("Invalid safetensors completion manifest rank count")
    layout = manifest.get("safetensors_layout", "per-rank")
    if layout not in ("per-rank", "shared"):
        raise ValueError("Unsupported safetensors checkpoint layout")
    files, records, tensors = _scan_checkpoint(
        path, shared=layout == "shared", header_streams=header_streams
    )
    if files != manifest.get("files"):
        raise ValueError(
            "Safetensors file inventory, sizes or header hashes disagree with manifest"
        )
    if tensors != manifest.get("tensors"):
        raise ValueError("Safetensors tensor structure disagrees with completion manifest")
    return manifest, records


def validate_safetensors_checkpoint(path):
    """Validate completion, the exact file inventory, headers and global geometry."""
    return _validated_checkpoint(path)[0]


def _enrich_metadata(metadata, path, records):
    storage = {}
    for record in records:
        index = MetadataIndex(record["fqn"], torch.Size(record["offsets"]))
        storage[index] = SafetensorsStorageInfo(
            relative_path=str((Path(path) / record["file"]).resolve()),
            shape=torch.Size(record["shape"]),
            dtype=getattr(torch, record["dtype"].removeprefix("torch.")),
            offset=record["file_offset"],
            length=record["file_bytes"],
        )
    metadata.storage_data = storage
    return metadata


def _write_manifest(path, manifest, sync_files, recorder=None):
    temporary = None
    events = (
        recorder.add_file(COMPLETION_MARKER, "w") if sync_files and recorder is not None else None
    )

    def sync(descriptor, target):
        started, success = perf_counter_ns(), False
        try:
            os.fsync(descriptor)
            success = True
        finally:
            if events is not None:
                events.append(
                    dict(
                        op="fsync",
                        offset=0,
                        requested=0,
                        completed=0,
                        start_ns=started - recorder.origin_ns,
                        duration_ns=perf_counter_ns() - started,
                        thread_id=get_native_id(),
                        success=success,
                        request=False,
                        io_layer="lifecycle",
                        target=target,
                    )
                )

    try:
        with tempfile.NamedTemporaryFile(
            mode="w", dir=path, prefix=".metasl-complete-", suffix=".tmp", delete=False
        ) as stream:
            temporary = Path(stream.name)
            json.dump(manifest, stream, indent=2)
            stream.write("\n")
            stream.flush()
            if sync_files:
                sync(stream.fileno(), "completion_manifest")
        os.replace(temporary, Path(path) / COMPLETION_MARKER)
        temporary = None
        if sync_files:
            descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
            try:
                sync(descriptor, "checkpoint_directory")
            finally:
                os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class AdmittedHuggingFaceStorageWriter(AdmittedFileSystemWriter, HuggingFaceStorageWriter):
    """Use the existing admission wrapper around native HF write_data."""

    def __init__(
        self,
        path,
        *,
        client=None,
        abort=None,
        status_path=None,
        thread_count=1,
        sync_files=True,
        overwrite=False,
        fqn_to_index_mapping=None,
    ):
        if thread_count != 1:
            raise ValueError("Safetensors currently requires one writer thread")
        if overwrite:
            raise ValueError("Safetensors checkpoint overwrite is unsupported")
        validate_empty_checkpoint(path)
        HuggingFaceStorageWriter.__init__(
            self,
            str(path),
            thread_count=1,
            save_distributed=True,
            enable_consolidation=False,
            fqn_to_index_mapping=fqn_to_index_mapping,
        )
        self.sync_files, self.overwrite = sync_files, False
        self.client, self.abort, self.status_path = client, abort, status_path
        self.timing = {}
        self.completed_write_indices = []
        self.results = None

    def prepare_local_plan(self, plan):
        plan = HuggingFaceStorageWriter.prepare_local_plan(self, plan)
        if self.fqn_to_index_mapping is None:
            return plan
        # Native HF global planning otherwise broadcasts the coordinator's map.
        # Local chunks may differ in size, including empty chunks on tail ranks.
        return dataclasses.replace(
            plan, storage_data={"metasl_local_file_mapping": self.fqn_to_index_mapping}
        )

    def prepare_global_plan(self, plans):
        mappings = [
            plan.storage_data.get("metasl_local_file_mapping")
            if isinstance(plan.storage_data, dict)
            else None
            for plan in plans
        ]
        prepared = HuggingFaceStorageWriter.prepare_global_plan(self, plans)
        if any(mapping is not None for mapping in mappings):
            if any(mapping is None for mapping in mappings):
                raise ValueError("Safetensors file bucketing must be configured on every rank")
            prepared = [
                dataclasses.replace(
                    plan, storage_data={**plan.storage_data, "fqn_to_index_mapping": mapping}
                )
                for plan, mapping in zip(prepared, mappings, strict=True)
            ]
        return prepared

    def write_data(self, plan, planner):
        if any(
            item.tensor_data is None or not item.index.fqn.startswith("model.")
            for item in plan.items
        ):
            if self.abort is not None:
                self.abort.set()
            if self.client:
                self.client.fail()
            raise ValueError("Safetensors supports model tensor entries only")
        return super().write_data(plan, planner)

    def finish(self, metadata, results):
        HuggingFaceStorageWriter.finish(self, metadata, results)
        files, records, tensors = _scan_checkpoint(self.path)
        expected = {
            (item.index.fqn, tuple(item.index.offset or ()), item.storage_data.relative_path)
            for group in results
            for item in group
        }
        actual = {(r["fqn"], tuple(r["offsets"]), r["file"]) for r in records}
        if expected != actual or sum(len(group) for group in results) != len(records):
            raise ValueError("Safetensors headers do not match completed write results")
        if tensors != _metadata_schema(metadata):
            raise ValueError("Safetensors headers do not match the complete save plan")
        _enrich_metadata(metadata, self.path, records)
        manifest = dict(
            schema_version=1,
            checkpoint_format="safetensors",
            world_size=len(results),
            files=files,
            tensors=tensors,
        )
        _write_manifest(
            self.path, manifest, self.sync_files, recorder=getattr(self.fs, "recorder", None)
        )


class AdmittedHuggingFaceStorageReader(AdmittedFileSystemReader, HuggingFaceStorageReader):
    """Native serial HF copies with admission, strict metadata and real extents."""

    def __init__(
        self,
        path,
        *,
        client=None,
        abort=None,
        status_path=None,
        rows=None,
        gate=None,
        thread_count=1,
    ):
        if thread_count != 1:
            raise ValueError("Safetensors currently requires one reader thread")
        HuggingFaceStorageReader.__init__(self, str(Path(path).resolve()), thread_count=1)
        self.client, self.abort, self.status_path, self.rows = client, abort, status_path, rows
        self.gate = gate
        self.timing = {}
        self.requests = []
        self.byte_items = 0

    def read_metadata(self):
        stats = dict(gate=self.gate is not None, pruned=False, queue_enter=perf_counter())
        self.phase("metadata_queue" if self.gate is not None else "metadata")
        self.acquire_gate()
        stats["read_start"] = perf_counter()
        stats["pss_before_read"] = memory(os.getpid())["pss"]
        try:
            self.phase("metadata")
            manifest, records = _validated_checkpoint(self.path)
            metadata = HuggingFaceStorageReader.read_metadata(self)
            if _metadata_schema(metadata) != manifest["tensors"]:
                raise ValueError("Native safetensors metadata disagrees with completion manifest")
            if self.rows is not None:
                targets = {r["fqn"]: r for r in self.rows}
                if len(targets) != len(self.rows) or set(targets) != {
                    t["fqn"] for t in manifest["tensors"]
                }:
                    raise ValueError("Safetensors checkpoint and target tensor keys disagree")
                for tensor in manifest["tensors"]:
                    target = targets[tensor["fqn"]]
                    if (
                        target["global_shape"] != tensor["shape"]
                        or target["dtype"] != tensor["dtype"]
                    ):
                        raise ValueError("Safetensors checkpoint and target dtype/shape mismatch")
            return _enrich_metadata(metadata, self.path, records)
        finally:
            stats["read_end"] = perf_counter()
            self.timing["metadata"] = stats
            if self.gate is not None:
                self.gate.release()
