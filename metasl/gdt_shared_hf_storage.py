"""One standard HF safetensors layout, independent of the FSDP compute mesh.

Every rank derives the same native dtype/name order from zero-length tensors.
The coordinator opens each final file once during global planning, writes and
flushes its header, and ftruncates its full size. DCP's subsequent plan scatter
publishes that completed setup before any worker can enter write_data. The
coordinator retains those descriptors for its payload; other ranks open r+b.
gdtensor maps each compute shard to its disjoint range of a full storage tensor.
There is no per-rank container, consolidation, or payload read on save.

Disk keys remove exactly the outer DCP ``model.`` nesting component, so these
files also load directly through safetensors and transformers. Load retains
each header descriptor through payload reads, with the existing 64 KiB storage
buffer and positioned reads. Compute and saved rank counts need not agree.
"""

import dataclasses
import hashlib
import json
import math
import os
import struct
from contextlib import ExitStack
from pathlib import Path
from threading import get_native_id
from time import perf_counter, perf_counter_ns

import torch
import torch.distributed as dist
from gdtensor.builder import Ok
from gdtensor.io import rank_load, rank_save
from safetensors.torch import save as safetensors_save
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    StorageMeta,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.tensor import DTensor
from torch.futures import Future

from metasl.gdt_hf_storage import (
    GDTensorHuggingFaceStorageReader,
    _SafetensorsLoadSession,
    _SafetensorsSaveSession,
)
from metasl.gdt_storage import (
    ComputeAccessor,
    PayloadSlot,
    TensorGeometry,
    _gdt_value,
    _local_byte_start,
    _publish_payload_trace,
    assign_payload_offsets,
    make_views,
)
from metasl.hf_storage import (
    AdmittedHuggingFaceStorageWriter,
    _enrich_metadata,
    _scan_checkpoint,
    _validated_checkpoint,
    _write_manifest,
    canonical_safetensors_header,
)
from metasl.iostats import fsync_stream
from metasl.procmem import memory
from metasl.sharding import check_partition, element_size


def _disk_name(name):
    if not name.startswith("model.") or not name.removeprefix("model."):
        raise ValueError("Shared safetensors supports model tensor entries only")
    return name.removeprefix("model.")


def _shared_layout(rows, max_file_gib=None):
    if max_file_gib is not None and (
        isinstance(max_file_gib, bool)
        or not isinstance(max_file_gib, (int, float))
        or not math.isfinite(max_file_gib)
        or max_file_gib <= 0
    ):
        raise ValueError("safetensors-max-file-gib must be a positive finite number")
    limit = None if max_file_gib is None else int(max_file_gib * 1024**3)
    if limit == 0:
        raise ValueError("safetensors-max-file-gib must allow at least one data byte")
    buckets, sizes, seen = [], [], set()
    for row in sorted(rows, key=lambda r: _disk_name(r["fqn"])):
        name = _disk_name(row["fqn"])
        shape = row["global_shape"]
        if name in seen or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("Shared safetensors requires unique keys and valid global shapes")
        seen.add(name)
        dtype = getattr(torch, row["dtype"].removeprefix("torch."))
        size = math.prod(shape) * element_size(dtype)
        if not buckets or (limit is not None and sizes[-1] + size > limit):
            buckets.append([])
            sizes.append(0)
        buckets[-1].append((name, tuple(shape), dtype))
        sizes[-1] += size
    if not buckets:
        raise ValueError("Shared safetensors requires at least one model tensor")
    headers, extents, files = {}, {}, []
    split = len(buckets) > 1
    for index, bucket in enumerate(buckets, 1):
        file = (
            f"model-{index:05d}-of-{len(buckets):05d}.safetensors" if split else "model.safetensors"
        )
        # Native serialization supplies the exact dtype/name ordering. No model
        # payload is allocated, including for multi-billion parameter layouts.
        native = safetensors_save({name: torch.empty(0, dtype=dtype) for name, _, dtype in bucket})
        header_size = struct.unpack("<Q", native[:8])[0]
        header = json.loads(native[8 : 8 + header_size])
        specs = {name: (shape, dtype) for name, shape, dtype in bucket}
        cursor = 0
        relative = {}
        for name, entry in header.items():
            shape, dtype = specs[name]
            size = math.prod(shape) * element_size(dtype)
            entry["shape"] = list(shape)
            entry["data_offsets"] = [cursor, cursor + size]
            relative[f"model.{name}"] = (cursor, size)
            cursor += size
        prefix = canonical_safetensors_header(header)
        headers[file] = prefix
        for name, (offset, size) in relative.items():
            extents[name] = (file, len(prefix) + offset, size)
        files.append(
            dict(
                path=file,
                header_bytes=len(prefix),
                data_bytes=cursor,
                size_bytes=len(prefix) + cursor,
            )
        )
    digest = hashlib.sha256()
    for file, prefix in headers.items():
        digest.update(file.encode())
        digest.update(prefix)
    report = dict(
        safetensors_layout="shared",
        assignment="global-sorted-hf-key-next-fit-v1",
        max_file_gib=max_file_gib,
        data_limit_bytes=limit,
        files=files,
        global_payload_bytes=sum(sizes),
        header_bytes_are_additional=True,
        disk_key_mapping="strip-one-outer-model-component",
        header_order="native-safetensors-dtype-then-name",
        setup_order="rank0 header-write/flush/ftruncate -> DCP plan scatter -> concurrent payload",
        consolidation=False,
        layout_sha256=digest.hexdigest(),
    )
    return headers, extents, report


def shared_safetensors_layout(rows, max_file_gib=None):
    """Return the JSON-compatible, global whole-tensor bucketing report."""
    return _shared_layout(rows, max_file_gib)[2]


def _lifecycle(stream, op, started=None, **details):
    recorder = getattr(stream, "recorder", None)
    if recorder is None:
        return
    ended = perf_counter_ns()
    started = ended if started is None else started
    stream.events.append(
        dict(
            op=op,
            request=False,
            io_layer="lifecycle",
            offset=0,
            requested=0,
            completed=0,
            start_ns=started - recorder.origin_ns,
            duration_ns=ended - started,
            success=True,
            thread_id=get_native_id(),
            **details,
        )
    )


class _SharedWriterCore(HuggingFaceStorageWriter):
    def write_data(self, plan, planner):
        results = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        if plan.storage_data["shared_layout_sha256"] != self.layout_report["layout_sha256"]:
            raise ValueError("Shared safetensors ranks computed different global headers")
        items = {item.index.fqn: item for item in plan.items}
        try:
            for file in self.headers:
                if file not in self._streams:
                    path = self.fs.concat_path(self.path, file)
                    self._streams[file] = self._stream_stack.enter_context(
                        self.fs.create_stream(path, "r+b")
                    )
                stream = self._streams[file]
                _lifecycle(
                    stream,
                    "barrier",
                    self._plan_setup_started_ns,
                    stage="ready",
                    setup_barrier="DCP_global_plan_scatter",
                )
                names = [name for name, extent in self.extents.items() if extent[0] == file]
                tensors = {
                    name: planner.state_dict[name]
                    for name in names
                    if isinstance(planner.state_dict[name], DTensor)
                }
                geometries, full_slots = {}, []
                for name in tensors:
                    tensor = tensors[name]
                    _, offset, length = self.extents[name]
                    geometries[name] = TensorGeometry(tuple(tensor.shape), tensor.dtype, 1)
                    full_slots.append(PayloadSlot(name, 0, file, length, 0, file_offset=offset))
                assign_payload_offsets(full_slots, list(geometries))
                local_slots = []
                for slot in full_slots:
                    tensor = tensors[slot.name]
                    start = _local_byte_start(tensor, rank)
                    local_slots.append(
                        PayloadSlot(
                            slot.name,
                            0,
                            file,
                            tensor.to_local().nbytes,
                            start,
                            file_offset=slot.file_offset + start,
                            payload_offset=slot.payload_offset + start,
                        )
                    )
                # DCP deduplicates replicas and identical empty tail chunks.
                # Results retain the compute-shard index and its true byte span.
                for name in names:
                    if name not in items:
                        continue
                    item = items[name]
                    _, offset, _ = self.extents[name]
                    size = math.prod(item.tensor_data.chunk.sizes) * element_size(
                        item.tensor_data.properties.dtype
                    )
                    start = (
                        item.tensor_data.chunk.offsets[0]
                        * math.prod(item.tensor_data.size[1:])
                        * element_size(item.tensor_data.properties.dtype)
                        if item.tensor_data.size
                        else 0
                    )
                    results.append(
                        WriteResult(item.index, size, _StorageInfo(file, offset + start, size))
                    )
                    if name not in tensors:
                        value = planner.resolve_data(item).detach().cpu().contiguous()
                        stream.seek(offset + start)
                        stream.write(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
                stream.flush()
                if tensors:
                    session = _SafetensorsSaveSession(
                        stream,
                        local_slots,
                        self.sync_files,
                        self.abort,
                        self.payload_events if self.capture_payload_events else None,
                        self.staging_events if self.capture_payload_events else None,
                    )
                    compute = ComputeAccessor(
                        tensors,
                        rank,
                        value_pattern=getattr(planner, "pattern", "zero"),
                        alias_groups={
                            r["fqn"]: r["alias_group"] for r in getattr(planner, "rows", []) or []
                        },
                        abort=self.abort,
                        staging_events=self.staging_events if self.capture_payload_events else None,
                        staging_slots=local_slots,
                        meta_save_source=getattr(planner, "meta_save_source", None),
                    )
                    cv, sv = make_views(
                        tensors,
                        geometries,
                        full_slots,
                        compute,
                        begin_save=lambda view, r, session=session: Ok(session),
                    )
                    groups = [("payload", self.payload_events), ("staging", self.staging_events)]
                    begins = [len(events) for _, events in groups]
                    try:
                        _gdt_value(rank_save(rank, cv, sv))
                    finally:
                        _publish_payload_trace(
                            stream,
                            [
                                (layer, events[begin:])
                                for (layer, events), begin in zip(groups, begins, strict=True)
                            ],
                        )
                    self.payload_requests.extend(
                        dict(file=name, file_offset=offset, length=size)
                        for name, offset, size in session.accessor.requests
                    )
                elif self.sync_files:
                    fsync_stream(stream)
        finally:
            self._stream_stack.close()
        future = Future()
        future.set_result(results)
        return future


class SharedGDTensorHuggingFaceStorageWriter(AdmittedHuggingFaceStorageWriter, _SharedWriterCore):
    """Admitted DCP plans, standard shared files, direct gdtensor shard writes."""

    buffer_bytes = 2 * 65536
    payload_header_is_canonical = True

    def __init__(self, path, *, max_file_gib=None, **kwargs):
        super().__init__(path, **kwargs)
        if self.fqn_to_index_mapping is not None:
            raise ValueError("Shared layout uses global max_file_gib, not local file mappings")
        self.max_file_gib = max_file_gib
        self.payload_requests, self.payload_events = [], []
        self.container_patch_events, self.staging_events = [], []
        self.capture_payload_events = False
        self._stream_stack, self._streams = ExitStack(), {}

    def prepare_local_plan(self, plan):
        rows = []
        for item in plan.items:
            if item.tensor_data is None:
                raise ValueError("Shared safetensors supports model tensor entries only")
            _disk_name(item.index.fqn)
            rows.append(
                dict(
                    fqn=item.index.fqn,
                    global_shape=list(item.tensor_data.size),
                    dtype=str(item.tensor_data.properties.dtype),
                )
            )
        self.headers, self.extents, self.layout_report = _shared_layout(rows, self.max_file_gib)
        self._plan_setup_started_ns = perf_counter_ns()
        plan = super().prepare_local_plan(plan)
        return dataclasses.replace(plan, storage_data={"shared_rows": rows})

    def prepare_global_plan(self, plans):
        reference = sorted(plans[0].storage_data["shared_rows"], key=lambda row: row["fqn"])
        if any(
            sorted(plan.storage_data["shared_rows"], key=lambda row: row["fqn"]) != reference
            for plan in plans
        ):
            raise ValueError("Shared safetensors ranks disagree on global tensor geometry")
        try:
            for info in self.layout_report["files"]:
                file = info["path"]
                stream = self._stream_stack.enter_context(
                    self.fs.create_stream(self.fs.concat_path(self.path, file), "w+b")
                )
                self._streams[file] = stream
                stream.write(self.headers[file])
                stream.flush()
                started = perf_counter_ns()
                os.ftruncate(stream.fileno(), info["size_bytes"])
                _lifecycle(stream, "ftruncate", started, size_bytes=info["size_bytes"])
                _lifecycle(
                    stream, "barrier", stage="publish", setup_barrier="DCP_global_plan_scatter"
                )
        except BaseException:
            self._stream_stack.close()
            raise
        return [
            dataclasses.replace(
                plan,
                storage_data={
                    "shared_layout_sha256": self.layout_report["layout_sha256"],
                },
            )
            for plan in plans
        ]

    def finish(self, metadata, results):
        if len(self.headers) > 1:
            index = dict(
                metadata=dict(total_size=self.layout_report["global_payload_bytes"]),
                weight_map={
                    _disk_name(name): extent[0] for name, extent in sorted(self.extents.items())
                },
            )
            path = self.fs.concat_path(self.path, "model.safetensors.index.json")
            encoded_index = (json.dumps(index, indent=2) + "\n").encode("utf-8")
            with self.fs.create_stream(path, "wb") as stream:
                stream.write(encoded_index)
                stream.flush()
                if self.sync_files:
                    fsync_stream(stream)
        files, records, tensors = _scan_checkpoint(self.path, shared=True)
        for name, md in metadata.state_dict_metadata.items():
            if not isinstance(md, TensorStorageMetadata):
                raise ValueError("Shared safetensors supports tensors only")
            check_partition(md.size, [(c.offsets, c.sizes) for c in md.chunks])
            saved = next((t for t in tensors if t["fqn"] == name), None)
            if (
                saved is None
                or saved["shape"] != list(md.size)
                or saved["dtype"] != str(md.properties.dtype)
            ):
                raise ValueError("Shared safetensors headers disagree with complete save plan")
        if set(metadata.state_dict_metadata) != {t["fqn"] for t in tensors}:
            raise ValueError("Shared safetensors tensor inventory disagrees with save plan")
        storage = {item.index: item.storage_data for group in results for item in group}
        expected = {
            MetadataIndex(name, chunk.offsets)
            for name, md in metadata.state_dict_metadata.items()
            for chunk in md.chunks
        }
        if expected != set(storage) or sum(map(len, results)) != len(storage):
            raise ValueError("Shared safetensors write results disagree with save plan")
        metadata.storage_data = storage
        manifest = dict(
            schema_version=1,
            checkpoint_format="safetensors",
            safetensors_layout="shared",
            world_size=len(results),
            files=files,
            tensors=tensors,
            layout=self.layout_report,
        )
        _write_manifest(
            self.path, manifest, self.sync_files, recorder=getattr(self.fs, "recorder", None)
        )


class SharedGDTensorHuggingFaceStorageReader(GDTensorHuggingFaceStorageReader):
    """Read standard full-tensor storage into any pure FSDP Shard(0) mesh."""

    def read_metadata(self):
        stats = dict(gate=self.gate is not None, pruned=False, queue_enter=perf_counter())
        self.phase("metadata_queue" if self.gate is not None else "metadata")
        self.acquire_gate()
        stats["read_start"] = perf_counter()
        stats["pss_before_read"] = memory(os.getpid())["pss"]
        self._header_stack, self._header_streams = ExitStack(), {}
        try:
            self.phase("metadata")
            for file in sorted(Path(self.path).glob("*.safetensors")):
                self._header_streams[file.name] = self._header_stack.enter_context(
                    self.fs.create_stream(str(file), "rb")
                )
            manifest, records = _validated_checkpoint(
                self.path, header_streams=self._header_streams
            )
            if manifest.get("safetensors_layout") != "shared":
                raise ValueError("Shared reader requires a shared safetensors checkpoint")
            if self.rows is not None:
                targets = {row["fqn"]: row for row in self.rows}
                if len(targets) != len(self.rows) or set(targets) != {r["fqn"] for r in records}:
                    raise ValueError("Shared checkpoint and target tensor keys disagree")
                for record in records:
                    target = targets[record["fqn"]]
                    if (
                        target["global_shape"] != record["shape"]
                        or target["dtype"] != record["dtype"]
                    ):
                        raise ValueError("Shared checkpoint and target dtype/shape mismatch")
            state = {}
            self.saved_geometries, self.saved_slots = {}, []
            for record in records:
                name, shape = record["fqn"], record["shape"]
                dtype = getattr(torch, record["dtype"].removeprefix("torch."))
                state[name] = TensorStorageMetadata(
                    TensorProperties(dtype=dtype),
                    torch.Size(shape),
                    [ChunkStorageMetadata(torch.Size([0] * len(shape)), torch.Size(shape))],
                )
                if shape:
                    self.saved_geometries[name] = TensorGeometry(tuple(shape), dtype, 1)
                    self.saved_slots.append(
                        PayloadSlot(
                            name,
                            0,
                            record["file"],
                            record["file_bytes"],
                            0,
                            file_offset=record["file_offset"],
                        )
                    )
            return _enrich_metadata(
                Metadata(
                    state_dict_metadata=state,
                    storage_meta=StorageMeta(load_id=self.load_id),
                ),
                self.path,
                records,
            )
        except BaseException:
            self._header_stack.close()
            raise
        finally:
            stats["read_end"] = perf_counter()
            self.timing["metadata"] = stats
            if self.gate is not None:
                self.gate.release()

    def _read_safetensors_data(self, plan, planner):
        self.capture_payload_events = getattr(self.fs, "recorder", None) is not None
        tensors = {
            name: value for name, value in planner.state_dict.items() if isinstance(value, DTensor)
        }
        slots = [slot for slot in self.saved_slots if slot.name in tensors]
        geometries = {name: self.saved_geometries[name] for name in tensors}
        # The session indexes packed starts at construction; make_views repeats
        # the same assignment, but doing it only there leaves that index stale.
        assign_payload_offsets(slots, list(tensors))
        rank = dist.get_rank() if dist.is_initialized() else 0
        accessor = ComputeAccessor(
            tensors,
            rank,
            value_pattern=planner.pattern,
            alias_groups={r["fqn"]: r["alias_group"] for r in planner.rows or []},
            abort=self.abort,
            staging_events=self.staging_events if self.capture_payload_events else None,
            staging_slots=slots,
            meta_load_sink=planner.meta_load_sink,
        )
        session = _SafetensorsLoadSession(self, slots)
        session.stack.close()
        session.stack, session.streams = self._header_stack, self._header_streams
        try:
            for item in plan.items:
                if item.dest_index.fqn in tensors:
                    continue
                info = self.storage_data[item.storage_index]
                shape = info.shape
                stride, offset = 1, 0
                for dimension, start in zip(
                    reversed(shape), reversed(item.storage_offsets), strict=True
                ):
                    offset += start * stride
                    stride *= dimension
                length = math.prod(item.lengths) * element_size(info.dtype)
                if length:
                    raw = session.read(
                        Path(info.relative_path).name,
                        info.offset + offset * element_size(info.dtype),
                        length,
                    )
                    value = torch.frombuffer(bytearray(raw), dtype=info.dtype).reshape(item.lengths)
                else:
                    value = torch.empty(item.lengths, dtype=info.dtype)
                target = planner.resolve_tensor(item)
                target.copy_(value)
                planner.commit_tensor(item, target)
            if tensors:
                cv, sv = make_views(
                    tensors, geometries, slots, accessor, begin_load=lambda view, r: Ok(session)
                )
                _gdt_value(rank_load(rank, cv, sv))
            planner.complete_gdtensor(accessor)
        finally:
            session.finish()
        future = Future()
        future.set_result(None)
        return future
