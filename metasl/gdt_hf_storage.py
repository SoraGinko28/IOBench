"""Native HF safetensors layout with gdtensor DTensor payload transport.

Torch still assigns files and plans chunks. A zero-payload native safetensors
serialization supplies its dtype/name ordering; only the resulting small header
is retained. Real payloads go directly to their final header offsets. This
format has no CRC or serialization ID, so there is no patch or consolidation IO.
"""

import dataclasses
import json
import math
import os
import struct
from contextlib import ExitStack
from pathlib import Path
from threading import get_native_id
from time import perf_counter_ns

import torch
import torch.distributed as dist
from gdtensor.builder import Ok
from gdtensor.io import rank_load, rank_save
from safetensors.torch import save as safetensors_save
from torch.distributed.checkpoint import HuggingFaceStorageReader, HuggingFaceStorageWriter
from torch.distributed.checkpoint._hf_utils import (
    CUSTOM_METADATA_KEY,
    DCP_VERSION_KEY,
    FORMAT_KEY,
    FORMAT_VALUE,
    HF_DCP_VERSION,
    _gen_file_name,
)
from torch.distributed.checkpoint.filesystem import _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.tensor import DTensor
from torch.futures import Future

from metasl.gdt_storage import (
    ComputeAccessor,
    GDTensorLoadPlanner,
    PayloadSlot,
    StorageAccessor,
    TensorGeometry,
    _gdt_value,
    _LoadSession,
    _local_byte_start,
    _publish_payload_trace,
    _ReadFileSystem,
    _SaveSession,
    assign_payload_offsets,
    infer_shard_grid,
    make_views,
)
from metasl.hf_storage import (
    AdmittedHuggingFaceStorageReader,
    AdmittedHuggingFaceStorageWriter,
    canonical_safetensors_header,
)
from metasl.iostats import fsync_stream
from metasl.sharding import element_size


def _native_item_order(items, writer):
    """Match Torch's CPU loader, including its accelerator size ordering."""
    ordered = list(items)
    custom = getattr(torch, torch._C._get_privateuse1_backend_name(), None)
    if writer.per_thread_copy_ahead > 0 and (
        torch.cuda.is_available() or (custom is not None and custom.is_available())
    ):
        ordered.sort(
            key=lambda item: (
                math.prod(item.tensor_data.size) * element_size(item.tensor_data.properties.dtype)
            )
        )
    return ordered


def _native_header(items):
    """Ask safetensors for its layout without allocating any model payload."""
    sharding = {item.index.fqn: {"saved_offsets": item.tensor_data.chunk.offsets} for item in items}
    extra = {
        CUSTOM_METADATA_KEY: json.dumps(sharding),
        DCP_VERSION_KEY: str(HF_DCP_VERSION),
        FORMAT_KEY: FORMAT_VALUE,
    }
    empty = {
        item.index.fqn: torch.empty(0, dtype=item.tensor_data.properties.dtype) for item in items
    }
    native = safetensors_save(empty, metadata=extra)
    size = struct.unpack("<Q", native[:8])[0]
    header = json.loads(native[8 : 8 + size])
    by_name = {item.index.fqn: item for item in items}
    cursor, extents = 0, {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        item = by_name[name]
        shape = list(item.tensor_data.chunk.sizes)
        length = math.prod(shape) * element_size(item.tensor_data.properties.dtype)
        entry["shape"] = shape
        entry["data_offsets"] = [cursor, cursor + length]
        extents[name] = (cursor, length)
        cursor += length
    prefix = canonical_safetensors_header(header)
    return prefix, {
        name: (len(prefix) + offset, length) for name, (offset, length) in extents.items()
    }


class _SafetensorsSaveSession(_SaveSession):
    """Reuse positioned payload writes, without DCP CRC observation or patching."""

    def __init__(self, stream, slots, sync_files, abort, payload_events, staging_events):
        self.stream, self.slots, self.sync_files = stream, slots, sync_files
        self.payload_events = payload_events
        self.received = {id(slot): 0 for slot in slots}
        self.accessor = StorageAccessor(
            slots,
            write=self.write,
            on_write=self.observe,
            abort=abort,
            staging_events=staging_events,
        )

    def observe(self, slot, offset, data):
        if offset != self.received[id(slot)] or offset + len(data) > slot.size:
            raise ValueError("Out-of-order or overflowing gdtensor safetensors save range")
        self.received[id(slot)] += len(data)

    def commit(self):
        for slot in self.slots:
            if self.received[id(slot)] != slot.size:
                raise ValueError(f"Incomplete gdtensor safetensors payload for {slot.name}")
        self.stream.flush()
        if self.sync_files:
            fsync_stream(self.stream)
        return Ok(None)


class _GDTensorHuggingFaceWriterCore(HuggingFaceStorageWriter):
    def write_data(self, plan, planner):
        # Native HF's super()._write_data bypasses subclass overrides. Retain
        # its exact file planning here, dispatching to the payload implementation.
        storage = plan.storage_data
        mapping = storage.get("fqn_to_index_mapping")
        shard = storage.get("shard_index")
        highest = max(mapping.values()) if mapping is not None else 1
        buckets = self._split_by_storage_plan(mapping, plan.items) if plan.items else {}
        return self._write_safetensors_data(planner, buckets, highest, shard)


class GDTensorHuggingFaceStorageWriter(
    AdmittedHuggingFaceStorageWriter, _GDTensorHuggingFaceWriterCore
):
    """Admitted native HF plans/manifest, with gdtensor payload writes."""

    buffer_bytes = 2 * 65536
    payload_header_is_canonical = True

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.payload_requests, self.payload_events = [], []
        self.container_patch_events, self.staging_events = [], []
        self.capture_payload_events = False

    def _write_safetensors_data(self, planner, buckets, highest, shard):
        results = []
        rank = dist.get_rank() if dist.is_initialized() else 0
        for index, items in buckets.items():
            file = _gen_file_name(index, highest, shard)
            ordered = _native_item_order(items, self)
            header, extents = _native_header(ordered)
            tensors = {
                item.index.fqn: planner.state_dict[item.index.fqn]
                for item in ordered
                if isinstance(planner.state_dict[item.index.fqn], DTensor)
            }
            geometries, slots = {}, []
            path = self.fs.concat_path(self.path, file)
            with self.fs.create_stream(path, "wb") as stream:
                stream.write(header)
                for item in ordered:
                    name = item.index.fqn
                    offset, length = extents[name]
                    results.append(
                        WriteResult(item.index, length, _StorageInfo(file, offset, length))
                    )
                    if name in tensors:
                        tensor = tensors[name]
                        geometries[name] = TensorGeometry(
                            tuple(tensor.shape), tensor.dtype, tensor.device_mesh.size()
                        )
                        slots.append(
                            PayloadSlot(
                                name,
                                rank,
                                file,
                                length,
                                _local_byte_start(tensor, rank),
                                file_offset=offset,
                            )
                        )
                    else:
                        # Only non-DTensor model tensors take the existing
                        # native materialization path (e.g. small buffers).
                        value = planner.resolve_data(item).detach().cpu().contiguous()
                        stream.seek(offset)
                        stream.write(memoryview(value.reshape(-1).view(torch.uint8).numpy()))
                stream.flush()
                if tensors:
                    assign_payload_offsets(slots, list(geometries))
                    session = _SafetensorsSaveSession(
                        stream,
                        slots,
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
                            r["fqn"]: r["alias_group"] for r in getattr(planner, "rows", [])
                        },
                        abort=self.abort,
                        staging_events=self.staging_events if self.capture_payload_events else None,
                        staging_slots=slots,
                        meta_save_source=getattr(planner, "meta_save_source", None),
                    )
                    cv, sv = make_views(
                        tensors,
                        geometries,
                        slots,
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
                        {"file": name, "file_offset": offset, "length": size}
                        for name, offset, size in session.accessor.requests
                    )
                elif self.sync_files:
                    fsync_stream(stream)
        future = Future()
        future.set_result(results)
        return future


class _GDTensorHuggingFaceReaderCore(HuggingFaceStorageReader):
    def read_data(self, plan, planner):
        if not isinstance(planner, GDTensorLoadPlanner):
            raise ValueError("gdtensor safetensors reader requires GDTensorLoadPlanner")
        filesystem = self.fs
        with ExitStack() as stack:
            self.fs = _ReadFileSystem(filesystem, stack)
            try:
                return self._read_safetensors_data(plan, planner)
            finally:
                self.fs = filesystem

    def _read_safetensors_data(self, plan, planner):
        self.capture_payload_events = getattr(self.fs, "recorder", None) is not None
        tensors = {
            name: value for name, value in planner.state_dict.items() if isinstance(value, DTensor)
        }
        native = dataclasses.replace(
            plan, items=[item for item in plan.items if item.dest_index.fqn not in tensors]
        )
        if native.items:
            HuggingFaceStorageReader.read_data(self, native, planner).wait()
        slots = [slot for slot in self.saved_slots if slot.name in tensors]
        geometries = {name: self.saved_geometries[name] for name in tensors}
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
        try:
            if tensors:
                cv, sv = make_views(
                    tensors,
                    geometries,
                    slots,
                    accessor,
                    begin_load=lambda view, r: Ok(session),
                )
                _gdt_value(rank_load(rank, cv, sv))
            planner.complete_gdtensor(accessor)
        finally:
            session.finish()
        future = Future()
        future.set_result(None)
        return future


class _SafetensorsLoadSession(_LoadSession):
    def read(self, file, offset, size):
        if file not in self.streams:
            path = self.reader.fs.concat_path(self.reader.path, file)
            self.streams[file] = self.stack.enter_context(self.reader.fs.create_stream(path, "rb"))
        stream = self.streams[file]
        started, data = perf_counter_ns(), b""
        try:
            data = os.pread(stream.fileno(), size, offset)
        finally:
            if self.reader.capture_payload_events:
                self.reader.payload_events.append(
                    dict(
                        file=file,
                        op="pread",
                        offset=offset,
                        requested=size,
                        completed=len(data),
                        start_ns=started,
                        duration_ns=perf_counter_ns() - started,
                        thread_id=get_native_id(),
                        success=len(data) == size,
                    )
                )
        if len(data) != size:
            raise EOFError("Short gdtensor safetensors payload read")
        self.reader.payload_requests.append({"file": file, "file_offset": offset, "length": size})
        return data

    def finish(self):
        if not getattr(self, "_finished", False):
            for file, stream in self.streams.items():
                _publish_payload_trace(
                    stream,
                    [("payload", [e for e in self.reader.payload_events if e["file"] == file])],
                )
        return super().finish()


class GDTensorHuggingFaceStorageReader(
    AdmittedHuggingFaceStorageReader, _GDTensorHuggingFaceReaderCore
):
    """Validate native HF headers/manifest, then range-read final payload offsets."""

    buffer_bytes = 2 * 65536

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.payload_requests, self.payload_events, self.staging_events = [], [], []
        self.capture_payload_events = False

    def read_metadata(self):
        metadata = super().read_metadata()
        self.saved_geometries, self.saved_slots = {}, []
        for name, md in metadata.state_dict_metadata.items():
            if not isinstance(md, TensorStorageMetadata) or not md.size:
                continue
            columns = infer_shard_grid(name, md)
            self.saved_geometries[name] = TensorGeometry(
                tuple(md.size), md.properties.dtype, columns
            )
            row_bytes = math.prod(md.size[1:]) * element_size(md.properties.dtype)
            for column, chunk in enumerate(sorted(md.chunks, key=lambda c: tuple(c.offsets))):
                info = metadata.storage_data[MetadataIndex(name, chunk.offsets)]
                self.saved_slots.append(
                    PayloadSlot(
                        name,
                        column,
                        Path(info.relative_path).name,
                        math.prod(chunk.sizes) * element_size(md.properties.dtype),
                        chunk.offsets[0] * row_bytes,
                        file_offset=info.offset,
                    )
                )
        return metadata
