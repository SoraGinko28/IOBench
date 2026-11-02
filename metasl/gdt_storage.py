"""DCP containers and planning with public gdtensor DTensor payload IO.

V1 uses pure FSDP Shard(0), local files, and one writer thread/file per rank.
Plain tensors and BYTE_IO retain the
native torch.save/load path. Only DTensor payloads use rank_save/rank_load.
Simulated plain tensor values follow ScratchSavePlanner's zero/position fixture
and small native load views are discarded; BYTE_IO values remain real objects.

Compute accessors hand off one whole requested range by default. Meta zero
sources reuse untouched, read-only anonymous zero pages; load sinks reuse one
anonymous mapping and MADV_DONTNEED it after every range, including failed
callbacks. Each mapping is bounded by the largest range seen, never total model
size. Optional position generation/checking uses bounded temporary blocks.
METASL_GDTENSOR_META_LOAD_SINK=faithful instead targets one pre-faulted resident
rank buffer with disjoint local tensor offsets; "ring" uses one resident buffer
for either save sources or load destinations, bounded by
METASL_GDTENSOR_META_LOAD_SINK_BYTES (default 256 MiB), capped at local shard size.
The phase runner write-faults the buffer before the phase clock. Ring supports
zero/default patterns and splits only compute callbacks, preserving file IO
requests even when a range exceeds the buffer. Faithful also selects a prepared
resident source for saves, sized to all local Meta shards, to match CPU source
page residency. The default "fresh" retains discard behavior and zero-page saves.
METASL_GDTENSOR_META_SIM_D2H=pageable makes Meta zero/default save sources use
writable pageable scratch, first-touch zero filling, and a simulated blocking D2H
delay. The same switch adds a simulated blocking H2D after each Meta load callback.
METASL_GDTENSOR_META_SIM_H2D_PROFILE selects an optional H2D profile override.
METASL_GDTENSOR_META_SIM_D2H_POPULATE defaults to non-temporal: fenced
SSE2 streaming zero stores, requiring x86_64 and cc. Other choices are memset
(previous behavior), memset-flush, populate-write, and populate-write-evict.
Population and any cache work count toward the same D2H time budget.
METASL_GDTENSOR_META_SIM_D2H_PROFILE selects a JSON profile containing
a_seconds, bandwidth_bytes_per_s, and optional size knots, or "legacy" for the
old affine profile. The pageable simulation default preserves 0.00021 + bytes /
1.6e9 through 593.5 MiB, then uses calibrated V100 size knots through 2 GiB.
The position correctness fixture does not support simulated D2H on save.
CUDA defaults to one pageable anonymous host scratch mapping sized to the largest
requested range. Each save does a blocking D2H copy then one callback; each load
does one callback then a blocking H2D copy. Both directions MADV_DONTNEED the
scratch after every range, including failed callbacks, so reused addresses pay
host first-touch costs again. Set ComputeAccessor cuda_staging_mode, or
METASL_GDTENSOR_CUDA_STAGING_MODE, to "pinned" for the former pinned-reuse mode,
or "double-buffer" for two 8 MiB pinned buffers with event-protected async copies.
The cuda_staging_bytes option or METASL_GDTENSOR_CUDA_STAGING_BYTES overrides the
double-buffer capacity and also selects that mode when no mode is specified.
Callbacks always receive host pointers. CPU retains its direct pointer path.

Storage staging stays at 64 KiB and envelopes contain only a few KiB of metadata.
In addition to compute staging, gdtensor owns up to approximately twice the
largest emitted range (callback pieces plus joined bytes), with O(R/B) Python
bookkeeping on multipart reads. A 128 MiB range measured 268,703,193 traced bytes
on load. Thus default Meta resident memory scales with the largest range, not the sum
of model tensors; whole-range staging does not make executor memory constant.
The bounds describe live buffers; PyTorch may cache released pinned allocations.

The native writer still names files, plans/deduplicates, and publishes .metadata.
Sparse torch.save envelopes preserve the exact zip layout. Sessions patch CRCs
and serialization IDs as each payload completes, while its pages are still hot.
Empty payloads are patched at commit; completed nonempty payloads are patched
immediately.
"""

from __future__ import annotations

import bisect
import collections
import ctypes
import dataclasses
import io
import math
import mmap
import os
import pickle
import struct
import sys
import warnings
import zipfile
import zlib
from collections import defaultdict
from contextlib import ExitStack, contextmanager
from threading import get_native_id
from time import perf_counter_ns, sleep

import gdtensor.builder as gb
import torch
import torch.distributed as dist
from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import Ok
from gdtensor.io import rank_load, rank_save
from gdtensor.session import StorageLoadSession, StorageSaveSession
from gdtensor.tag import ConstantOne, TensorCount, WorldSize
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultSavePlanner
from torch.distributed.checkpoint.filesystem import CURRENT_DCP_VERSION, _metadata_fn, _StorageInfo
from torch.distributed.checkpoint.metadata import MetadataIndex, TensorStorageMetadata
from torch.distributed.checkpoint.planner import LoadItemType, WriteItemType
from torch.distributed.checkpoint.storage import WriteResult
from torch.distributed.tensor import DTensor, Shard
from torch.futures import Future

from metasl import meta_h2d_profile
from metasl.iostats import fsync_stream
from metasl.meta_d2h_profile import duration_seconds, load_profile
from metasl.scratch import MetaLoadPlanner, OptionalKeysLoadPlanner
from metasl.sharding import audit_local, element_size, fill_or_check, pattern_blocks, read_record
from metasl.storage import AdmittedFileSystemReader, AdmittedFileSystemWriter


def _meta_d2h_profile(path):
    return load_profile(path)


def _meta_h2d_profile(path):
    return meta_h2d_profile.load_profile(path)


@contextmanager
def _staging_event(events, op, file, offset, size, **details):
    """Time host work on the payload clock; never add a CUDA synchronization."""
    if events is None:
        yield
        return
    started, success = perf_counter_ns(), False
    try:
        yield
        success = True
    finally:
        ended = perf_counter_ns()
        events.append(
            dict(
                file=file,
                op=op,
                offset=offset,
                requested=size,
                completed=size if success else 0,
                start_ns=started,
                end_ns=ended,
                duration_ns=ended - started,
                thread_id=get_native_id(),
                success=success,
                request=False,
                io_layer="staging",
                **details,
            )
        )


def _publish_payload_trace(stream, groups):
    """Translate absolute accessor clocks once into the stream recorder's clock.

    Keep the report arrays absolute for existing consumers. Appending leaves
    every original buffered-file event (and its order) untouched; consumers
    reconstruct invocation order from start_ns, as for the existing traces.
    """
    recorder = getattr(stream, "recorder", None)
    if recorder is None:
        return
    for layer, events in groups:
        for event in events:
            translated = dict(event, io_layer=layer)
            translated["start_ns"] -= recorder.origin_ns
            if "end_ns" in translated:
                translated["end_ns"] -= recorder.origin_ns
            stream.events.append(translated)


class _SparseEnvelope:
    """Keep only torch's small writes; skip_data seeks past the tensor storage."""

    def __init__(self):
        self.position = self.length = 0
        self.segments = []

    def write(self, data):
        self.segments.append((self.position, bytes(data)))
        self.position += len(data)
        self.length = max(self.length, self.position)
        return len(data)

    def seek(self, offset, whence=0):
        self.position = offset + (
            self.position if whence == 1 else self.length if whence == 2 else 0
        )
        if self.position < 0:
            raise ValueError("Negative container seek")
        self.length = max(self.length, self.position)
        return self.position

    def tell(self):
        return self.position

    def flush(self):
        pass

    def seekable(self):
        return True

    def read(self, size=-1):
        if size < 0:
            size = self.length - self.position
        size = min(size, max(0, self.length - self.position))
        result = bytearray(size)
        for offset, data in self.segments:
            lo, hi = max(self.position, offset), min(self.position + size, offset + len(data))
            if lo < hi:
                result[lo - self.position : hi - self.position] = data[lo - offset : hi - offset]
        self.position += size
        return bytes(result)


class _ContainerSlice:
    """Expose exactly one concatenated torch.save archive to zipfile."""

    def __init__(self, stream, offset, length):
        if offset < 0 or length < 0:
            raise ValueError("Invalid torch.save container bounds")
        self.stream, self.offset, self.length = stream, offset, length
        self.position = 0

    def seek(self, offset, whence=0):
        position = offset + (self.position if whence == 1 else self.length if whence == 2 else 0)
        if position < 0 or position > self.length:
            raise ValueError("torch.save container seek exceeds its bounds")
        self.position = position
        return position

    def tell(self):
        return self.position

    def seekable(self):
        return True

    def read(self, size=-1):
        size = self.length - self.position if size < 0 else min(size, self.length - self.position)
        self.stream.seek(self.offset + self.position)
        data = self.stream.read(size)
        if len(data) != size:
            raise ValueError("Truncated torch.save container")
        self.position += size
        return data


@dataclasses.dataclass(frozen=True)
class _ContainerRecord:
    name: str
    payload_offset: int
    size: int
    crc: int
    flags: int
    central_offset: int


@dataclasses.dataclass(frozen=True)
class Container:
    offset: int
    size: int
    payload_offset: int
    payload_size: int
    records: tuple
    serialization_prefix: bytes
    envelope_bytes: int

    @property
    def end(self):
        return self.offset + self.size


def _container_zip(stream):
    """Require the native footer so zipfile never searches through tensor bytes."""
    stream.seek(-22, 2)
    footer = stream.read(22)
    if footer[:4] != b"PK\x05\x06" or footer[-2:] != b"\0\0":
        raise ValueError("Unsupported torch.save ZIP footer or archive comment")
    disk, central_disk, disk_count, count, central_size, _ = struct.unpack_from(
        "<HHHHII", footer, 4
    )
    if disk or central_disk or disk_count != count or count > 32 or central_size > 65536:
        raise ValueError("Unsupported torch.save ZIP directory")
    return zipfile.ZipFile(stream)


def _container_records(stream, archive, base=0):
    """Read only local headers and the central directory, never tensor storage."""
    records = []
    central_offset = archive.start_dir
    for info in archive.infolist():
        if info.compress_type != zipfile.ZIP_STORED or info.flag_bits & 1:
            raise ValueError("gdtensor requires uncompressed, unencrypted tensor records")
        stream.seek(info.header_offset)
        header = stream.read(30)
        if header[:4] != b"PK\x03\x04":
            raise ValueError("Invalid torch.save local record header")
        flags, method = struct.unpack_from("<HH", header, 6)
        name_length, extra_length = struct.unpack_from("<HH", header, 26)
        if flags != info.flag_bits or method != zipfile.ZIP_STORED:
            raise ValueError("torch.save local and central record headers disagree")
        name = stream.read(name_length)
        if name.decode("utf-8") != info.filename:
            raise ValueError("torch.save local and central record names disagree")
        payload_offset = info.header_offset + 30 + name_length + extra_length
        if payload_offset + info.file_size > archive.start_dir:
            raise ValueError("torch.save tensor record exceeds its data region")
        records.append(
            _ContainerRecord(
                info.filename,
                base + payload_offset,
                info.file_size,
                info.CRC,
                flags,
                base + central_offset,
            )
        )
        stream.seek(central_offset)
        central = stream.read(46)
        if central[:4] != b"PK\x01\x02":
            raise ValueError("Invalid torch.save central record header")
        name_length, extra_length, comment_length = struct.unpack_from("<HHH", central, 28)
        central_offset += 46 + name_length + extra_length + comment_length
    return tuple(records)


def begin_container(stream, shape, dtype, stride=None):
    """Write a native envelope with a seek hole, retaining at most a few KiB."""
    with FakeTensorMode():
        tensor = (
            torch.empty(tuple(shape), dtype=dtype, device="cpu")
            if stride is None
            else torch.empty_strided(tuple(shape), tuple(stride), dtype=dtype, device="cpu")
        )
        if not tensor.is_contiguous():
            raise ValueError("gdtensor requires contiguous tensor stride")
    envelope = _SparseEnvelope()
    with torch.serialization.skip_data(materialize_fake_tensors=True):
        torch.save(tensor, envelope)
    offset = stream.tell()
    with _container_zip(envelope) as archive:
        records = _container_records(envelope, archive, offset)
        payload = next(record for record in records if record.name == "archive/data/0")
        serialization_prefix = archive.read("archive/.data/serialization_id")[:20]
    for relative, data in envelope.segments:
        stream.seek(offset + relative)
        stream.write(data)
    stream.seek(offset + envelope.length)
    return Container(
        offset,
        envelope.length,
        payload.payload_offset,
        payload.size,
        records,
        serialization_prefix,
        sum(len(data) for _, data in envelope.segments),
    )


def _container_pwrite(target, offset, data, *, write_events=None, file=None):
    if isinstance(target, int):
        descriptor = target
    else:
        target.flush()
        try:
            descriptor = target.fileno()
        except (AttributeError, io.UnsupportedOperation):
            position = target.tell()
            target.seek(offset)
            target.write(data)
            target.seek(position)
            return
    view = memoryview(data)
    while view:
        written = 0
        started = perf_counter_ns() if write_events is not None else None
        try:
            written = os.pwrite(descriptor, view, offset)
        finally:
            ended = perf_counter_ns() if started is not None else None
            if write_events is not None:
                write_events.append(
                    dict(
                        file=file,
                        op="pwrite",
                        offset=offset,
                        requested=len(view),
                        completed=written,
                        start_ns=started,
                        duration_ns=ended - started,
                        success=written > 0,
                        thread_id=get_native_id(),
                    )
                )
        if written <= 0:
            raise OSError("Failed to patch torch.save container")
        view = view[written:]
        offset += written


def _combine_container_crc(seed, value):
    return (seed ^ (value + 0x9E3779B9 + (seed << 6) + (seed >> 2))) & ((1 << 64) - 1)


def patch_container(target, container, crc, first_bytes=b"", *, write_events=None, file=None):
    """Commit accessor's incremental CRC and torch's fixed-size serialization ID.

    PyTorch hashes CRCs only for records that have a data descriptor. Its write
    callback additionally mistakes a payload beginning with the descriptor magic
    for a descriptor, so preserve those first eight bytes when present.
    """
    payload = next(record for record in container.records if record.name == "archive/data/0")
    id_record = next(
        record for record in container.records if record.name == "archive/.data/serialization_id"
    )
    combined = 0
    for record in container.records:
        if record is id_record:
            break
        if record is payload and len(first_bytes) >= 8 and first_bytes[:4] == b"PK\x07\x08":
            combined = _combine_container_crc(combined, struct.unpack("<I", first_bytes[4:8])[0])
        if record.flags & 8:
            combined = _combine_container_crc(combined, crc if record is payload else record.crc)
    serialization_id = container.serialization_prefix + f"{combined:020d}".encode("ascii")
    if id_record.size != len(serialization_id):
        raise ValueError("Unexpected torch.save serialization ID size")
    _container_pwrite(
        target, id_record.payload_offset, serialization_id, write_events=write_events, file=file
    )
    for record, value in ((payload, crc), (id_record, zlib.crc32(serialization_id))):
        crc_bytes = struct.pack("<I", value)
        if record.flags & 8:
            _container_pwrite(
                target,
                record.payload_offset + record.size + 4,
                crc_bytes,
                write_events=write_events,
                file=file,
            )
        _container_pwrite(
            target, record.central_offset + 16, crc_bytes, write_events=write_events, file=file
        )


@dataclasses.dataclass(frozen=True)
class _StorageHeader:
    dtype: torch.dtype
    key: str
    location: str
    count: int


@dataclasses.dataclass(frozen=True)
class _TensorHeader:
    storage: _StorageHeader
    storage_offset: int
    shape: tuple
    stride: tuple


class _TensorHeaderUnpickler(pickle.Unpickler):
    """Interpret the small single-tensor pickle without allocating its storage."""

    storage_types = {
        "DoubleStorage": torch.float64,
        "FloatStorage": torch.float32,
        "HalfStorage": torch.float16,
        "BFloat16Storage": torch.bfloat16,
        "LongStorage": torch.int64,
        "IntStorage": torch.int32,
        "ShortStorage": torch.int16,
        "CharStorage": torch.int8,
        "ByteStorage": torch.uint8,
        "BoolStorage": torch.bool,
        "ComplexFloatStorage": torch.complex64,
        "ComplexDoubleStorage": torch.complex128,
    }

    def find_class(self, module, name):
        if module == "torch" and name in self.storage_types:
            return self.storage_types[name]
        if module == "torch._utils" and name in ("_rebuild_tensor", "_rebuild_tensor_v2"):
            return self.rebuild_tensor
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        raise ValueError(f"Unsupported torch.save tensor pickle global: {module}.{name}")

    def persistent_load(self, identity):
        if not isinstance(identity, tuple) or len(identity) != 5 or identity[0] != "storage":
            raise ValueError("Unsupported torch.save storage identity")
        _, dtype, key, location, count = identity
        if not isinstance(dtype, torch.dtype) or key != "0" or location != "cpu":
            raise ValueError("gdtensor requires one native CPU tensor storage")
        if type(count) is not int or count < 0:
            raise ValueError("Invalid torch.save storage size")
        return _StorageHeader(dtype, key, location, count)

    @staticmethod
    def rebuild_tensor(storage, storage_offset, shape, stride, *extra):
        if not isinstance(storage, _StorageHeader):
            raise ValueError("Invalid torch.save tensor storage")
        if len(extra) > 3 or (len(extra) == 3 and extra[2]):
            raise ValueError("Unsupported torch.save tensor metadata")
        return _TensorHeader(storage, storage_offset, tuple(shape), tuple(stride))


def read_container(stream, container_offset, length, shape, dtype):
    """Locate validated native tensor storage using only the envelope's bytes."""
    position = stream.tell()
    try:
        bounded = _ContainerSlice(stream, container_offset, length)
        with _container_zip(bounded) as archive:
            records = _container_records(bounded, archive)
            names = [record.name for record in records]
            if len(names) != len(set(names)) or not names:
                raise ValueError("Duplicated or empty torch.save record table")
            prefix = names[0].split("/", 1)[0] + "/"
            relative = {name.removeprefix(prefix) for name in names if name.startswith(prefix)}
            supported = {
                "data.pkl",
                ".format_version",
                ".storage_alignment",
                "byteorder",
                "data/0",
                "version",
                ".data/version",
                ".data/serialization_id",
            }
            required = {"data.pkl", "byteorder", "data/0"}
            if (
                len(relative) != len(names)
                or not relative <= supported
                or not required <= relative
                or not relative & {"version", ".data/version"}
            ):
                raise ValueError("gdtensor requires a single native torch.save tensor storage")
            pickle_info = archive.getinfo(prefix + "data.pkl")
            if pickle_info.file_size > 65536:
                raise ValueError("Oversized torch.save tensor header")
            tensor = _TensorHeaderUnpickler(io.BytesIO(archive.read(pickle_info))).load()
            if not isinstance(tensor, _TensorHeader):
                raise ValueError("torch.save item is not a supported tensor")
            shape = tuple(shape)
            if any(type(value) is not int or value < 0 for value in tensor.shape):
                raise ValueError("Invalid torch.save tensor shape")
            if tensor.storage.dtype != dtype or tensor.shape != shape:
                raise ValueError("torch.save tensor dtype or shape disagrees with DCP metadata")
            if (
                type(tensor.storage_offset) is not int
                or len(tensor.stride) != len(shape)
                or any(type(value) is not int or value < 0 for value in tensor.stride)
                or tensor.storage_offset != 0
            ):
                raise ValueError("gdtensor requires contiguous tensor stride and storage offset 0")
            if math.prod(shape):
                stride = 1
                for dimension, actual_stride in zip(
                    reversed(shape), reversed(tensor.stride), strict=True
                ):
                    if dimension != 1 and actual_stride != stride:
                        raise ValueError("gdtensor requires contiguous tensor stride")
                    stride *= dimension
            if tensor.storage.count != math.prod(shape):
                raise ValueError("gdtensor requires tensor storage numel == tensor numel")
            byteorder = archive.getinfo(prefix + "byteorder")
            if byteorder.file_size > 16:
                raise ValueError("Oversized torch.save byte order record")
            if archive.read(byteorder) != sys.byteorder.encode("ascii"):
                raise ValueError("Unsupported torch.save tensor byte order")
            payload = next(record for record in records if record.name == prefix + "data/0")
            if payload.size != math.prod(shape) * torch.empty((), dtype=dtype).element_size():
                raise ValueError("torch.save tensor data record has an inconsistent byte length")
            return container_offset + payload.payload_offset
    except (zipfile.BadZipFile, KeyError, EOFError, pickle.UnpicklingError, struct.error) as error:
        raise ValueError(f"Invalid torch.save tensor container: {error}") from error
    finally:
        stream.seek(position)


ACCESSOR_BUFFER_BYTES = 65536


_GDT_DTYPES = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
}


def _gdt_value(result):
    if type(result) is not gb.Ok:
        raise ValueError(f"gdtensor operation failed: {result!r}")
    return result.value


def shard_bounds(shape, columns, column):
    """Return global byte-independent leading-row bounds for torch.chunk Shard(0)."""
    if not shape or columns < 1 or column < 0 or column >= columns:
        raise ValueError("gdtensor IO requires a non-scalar Shard(0) grid")
    width = (shape[0] + columns - 1) // columns
    lo = min(column * width, shape[0])
    return lo, min(lo + width, shape[0])


@dataclasses.dataclass
class TensorGeometry:
    shape: tuple
    dtype: torch.dtype
    columns: int


@dataclasses.dataclass
class PayloadSlot:
    name: str
    column: int
    file: str
    size: int
    global_offset: int
    file_offset: int = 0
    payload_offset: int = 0
    container: object = None


def assign_payload_offsets(slots, names):
    """Pack nonempty slots in tensor table order, then storage-column order per file."""
    order = {name: index for index, name in enumerate(names)}
    offsets = defaultdict(int)
    seen = set()
    for slot in sorted(slots, key=lambda slot: (order[slot.name], slot.column)):
        key = (slot.name, slot.column)
        if key in seen:
            raise ValueError(f"Duplicate gdtensor payload slot: {key}")
        seen.add(key)
        if slot.size < 0 or slot.file_offset < 0 or slot.global_offset < 0:
            raise ValueError("Negative gdtensor payload slot bounds")
        slot.payload_offset = offsets[slot.file]
        offsets[slot.file] += slot.size
    return dict(offsets)


def _compute_geometry(tensor):
    if not isinstance(tensor, DTensor) or tensor.ndim == 0:
        raise ValueError("gdtensor payload IO supports only non-scalar DTensors")
    mesh = tensor.device_mesh
    if mesh.ndim != 1 or len(tensor.placements) != 1:
        raise ValueError("gdtensor IO requires a one-dimensional pure FSDP mesh")
    placement = tensor.placements[0]
    if not isinstance(placement, Shard) or placement.dim != 0:
        raise ValueError("gdtensor IO requires pure FSDP Shard(0) placement")
    ranks = mesh.mesh.reshape(-1).tolist()
    if ranks != list(range(len(ranks))):
        raise ValueError("gdtensor IO requires consecutive FSDP ranks starting at zero")
    local = tensor.to_local()
    if local.device.type not in ("cpu", "meta", "cuda") or not local.is_contiguous():
        raise ValueError("gdtensor IO requires contiguous CPU, CUDA or meta local tensors")
    if tensor.dtype not in _GDT_DTYPES:
        raise ValueError(f"gdtensor IO does not support dtype {tensor.dtype}")
    return TensorGeometry(tuple(tensor.shape), tensor.dtype, len(ranks))


class MetaLoadSink:
    """Resident Meta CPU-target buffer, prepared before the phase clock.

    Faithful mode assigns every local tensor a disjoint byte interval. Ring mode
    reuses one bounded mapping and may split compute callbacks. The executor
    assembles a complete range before storage writes, and reads the complete
    range before compute copies, so file IO offsets, sizes, and order stay fixed.
    """

    DEFAULT_RING_BYTES = 256 * 1024**2

    def __init__(self, tensor_bytes, mode="faithful", capacity=None):
        if mode not in ("faithful", "ring"):
            raise ValueError(f"Unknown resident Meta load sink mode: {mode}")
        if any(not isinstance(size, int) or size < 0 for size in tensor_bytes.values()):
            raise ValueError("Meta load sink tensor sizes must be nonnegative integers")
        if capacity is not None and (not isinstance(capacity, int) or capacity < 8):
            raise ValueError("Meta load sink capacity must be at least 8 bytes")
        if mode == "faithful" and capacity is not None:
            raise ValueError("Meta load sink capacity requires ring mode")
        self.mode = mode
        self.tensor_bytes = dict(tensor_bytes)
        self.tensor_offsets = {}
        total = 0
        for name, size in self.tensor_bytes.items():
            self.tensor_offsets[name] = total
            total += size
        self.local_shard_bytes = total
        self.capacity = total if mode == "faithful" else min(
            total, self.DEFAULT_RING_BYTES if capacity is None else capacity
        )
        self.buffer = None
        self.address = self.cursor = 0
        started = perf_counter_ns()
        if self.capacity:
            self.buffer = mmap.mmap(
                -1, self.capacity, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS
            )
            self.address = ctypes.addressof(ctypes.c_char.from_buffer(self.buffer))
            # A read-only touch would map the shared zero page. Writable memset
            # establishes private resident destination pages, as CPU tensors do.
            ctypes.memset(self.address, 0, self.capacity)
        self.prepared_ns = perf_counter_ns()
        self.prepare_duration_ns = self.prepared_ns - started

    def report(self):
        return dict(
            mode=self.mode,
            local_shard_bytes=self.local_shard_bytes,
            resident_buffer_bytes=self.capacity,
            prepared_ns=self.prepared_ns,
            prepare_duration_ns=self.prepare_duration_ns,
            prefaulted=True,
            preparation_inside_load_call=False,
        )

    def pieces(self, name, local_offset, size, itemsize):
        """Yield (range-relative offset, mapping offset, length) without IO."""
        if name not in self.tensor_bytes or not (
            0 <= local_offset <= local_offset + size <= self.tensor_bytes[name]
        ):
            raise ValueError(f"Meta load sink range lies outside prepared tensor: {name}")
        if not size:
            return
        if self.mode == "faithful":
            yield 0, self.tensor_offsets[name] + local_offset, size
            return
        # Keep each callback contiguous. Align chunk sizes and addresses to the
        # destination dtype, including rings whose configured size is not aligned.
        capacity = self.capacity // itemsize * itemsize
        if capacity < itemsize:
            raise ValueError("Meta load sink ring cannot hold one tensor element")
        relative = 0
        while relative < size:
            self.cursor = (self.cursor + itemsize - 1) // itemsize * itemsize
            length = min(size - relative, capacity)
            if self.cursor + length > capacity:
                self.cursor = 0
            offset = self.cursor
            self.cursor += length
            yield relative, offset, length
            relative += length


class MetaSaveSource(MetaLoadSink):
    """Write-faulted CPU-target source: disjoint faithful pages or one ring."""

    def report(self):
        report = super().report()
        report.pop("preparation_inside_load_call")
        report["preparation_inside_save_call"] = False
        return report


class ComputeAccessor(DTensorDataAccessor):
    """Serial whole-range CPU/Meta/CUDA hand-off; optional pinned CUDA staging.

    buffer_bytes bounds position fixtures/checks, not the default range hand-off.
    CUDA defaults to pageable scratch discarded after every range. The "pinned"
    mode reuses one pinned buffer; "double-buffer" uses two bounded pinned buffers
    (8 MiB each unless cuda_staging_bytes is given). Supplying just a capacity
    selects double buffering. Host buffers remain valid until callbacks and all
    associated CUDA copies have completed.
    METASL_GDTENSOR_META_SIM_D2H=pageable simulates both pageable transfer
    directions. Only zero/default save sources support D2H simulation. Direct
    accessor callers should call prepare_save() before starting their save timer;
    normal DCP/safetensors callers prepare through GDTensorSavePlanner instead.
    """

    def __init__(
        self,
        tensors,
        rank,
        buffer_bytes=ACCESSOR_BUFFER_BYTES,
        value_pattern=None,
        alias_groups=None,
        abort=None,
        cuda_staging_bytes=None,
        cuda_staging_mode=None,
        staging_events=None,
        staging_slots=(),
        meta_load_sink=None,
        meta_save_source=None,
    ):
        if buffer_bytes < 8:
            raise ValueError("gdtensor accessor buffer must hold at least one tensor element")
        if value_pattern not in (None, "zero", "position"):
            raise ValueError(f"Unknown gdtensor value pattern: {value_pattern}")
        if cuda_staging_bytes is None:
            configured = os.environ.get("METASL_GDTENSOR_CUDA_STAGING_BYTES")
            if configured is not None:
                cuda_staging_bytes = int(configured)
        if cuda_staging_bytes is not None and cuda_staging_bytes < 8:
            raise ValueError("CUDA staging buffer must hold at least one tensor element")
        if cuda_staging_mode is None:
            cuda_staging_mode = os.environ.get("METASL_GDTENSOR_CUDA_STAGING_MODE")
        if cuda_staging_mode is None:
            cuda_staging_mode = "double-buffer" if cuda_staging_bytes is not None else "pageable"
        if cuda_staging_mode not in ("pageable", "pinned", "double-buffer"):
            raise ValueError(f"Unknown CUDA staging mode: {cuda_staging_mode}")
        if cuda_staging_mode == "double-buffer":
            if cuda_staging_bytes is None:
                cuda_staging_bytes = 8 * 1024**2
        elif cuda_staging_bytes is not None:
            raise ValueError("CUDA staging bytes requires double-buffer mode")
        self.meta_sim_d2h_mode = os.environ.get("METASL_GDTENSOR_META_SIM_D2H") or "off"
        if self.meta_sim_d2h_mode not in ("off", "pageable"):
            raise ValueError(f"Unknown Meta simulated D2H mode: {self.meta_sim_d2h_mode}")
        self.meta_sim_d2h_profile = (
            _meta_d2h_profile(os.environ.get("METASL_GDTENSOR_META_SIM_D2H_PROFILE"))
            if self.meta_sim_d2h_mode == "pageable"
            else None
        )
        self.meta_sim_h2d_profile = (
            _meta_h2d_profile(os.environ.get("METASL_GDTENSOR_META_SIM_H2D_PROFILE"))
            if self.meta_sim_d2h_mode == "pageable"
            else None
        )
        self.meta_sim_d2h_population = None
        self._meta_d2h_populate = None
        self._meta_d2h_prepared = False
        if self.meta_sim_d2h_mode == "pageable":
            from metasl.meta_d2h_cache import DEFAULT_POPULATION, validate_population

            self.meta_sim_d2h_population = os.environ.get(
                "METASL_GDTENSOR_META_SIM_D2H_POPULATE", DEFAULT_POPULATION
            )
            validate_population(self.meta_sim_d2h_population)
        self.tensors = dict(tensors)
        self.rank = rank
        self.geometries = {name: _compute_geometry(tensor) for name, tensor in tensors.items()}
        self.buffer_bytes = buffer_bytes
        self.buffer = bytearray(buffer_bytes)
        self.value_pattern = value_pattern
        self.alias_groups = alias_groups or {}
        self.abort = abort
        self.requests = []
        self.cuda_staging_bytes = cuda_staging_bytes
        self.cuda_staging_mode = cuda_staging_mode
        self._meta_source = self._meta_sink = None
        self._meta_source_address = self._meta_sink_address = 0
        self._cuda_buffers = []
        self._cuda_scratch = None
        self._cuda_scratch_address = 0
        self.staging_events = staging_events
        self.staging_slots = defaultdict(list)
        for slot in staging_slots:
            self.staging_slots[slot.name].append(slot)
        self.meta_load_sink = meta_load_sink
        self.meta_save_source = meta_save_source
        if value_pattern == "position" and any(
            buffer is not None and buffer.mode == "ring"
            for buffer in (meta_load_sink, meta_save_source)
        ):
            raise ValueError(
                "Meta ring buffers support only zero/default value patterns; "
                "use faithful or fresh for position"
            )

    def prepare_save(self):
        """Prepare optional Meta save population before timing a direct accessor.

        Load-only accessors never call this method, even with simulation enabled.
        GDTensorSavePlanner already prepares the same cached helper before DCP.
        """
        if self.meta_sim_d2h_mode == "pageable" and not self._meta_d2h_prepared:
            from metasl.meta_d2h_cache import population_function

            self._meta_d2h_populate = population_function(self.meta_sim_d2h_population)
            self._meta_d2h_prepared = True

    def _stage(self, rng, op, size=None, relative=0, **details):
        offset = rng.global_tensor_offset + relative
        length = rng.size if size is None else size
        # Load metadata can retain every saved shard of one tensor. Selecting
        # by name alone labels other ranks' events with the final saved file,
        # then session publication drops them. Use the actual global interval,
        # including callback-relative chunks, rather than the compute rank: a
        # resharded load can read a different saved column from its local rank.
        slot = next(
            (
                candidate
                for candidate in self.staging_slots.get(rng.global_tensor_id, ())
                if candidate.global_offset <= offset
                and (
                    not hasattr(candidate, "size")
                    or (
                        offset + length <= candidate.global_offset + candidate.size
                        and (length or offset < candidate.global_offset + candidate.size)
                    )
                )
            ),
            None,
        )
        if slot is not None:
            offset += slot.file_offset - slot.global_offset
        return _staging_event(
            self.staging_events,
            op,
            slot.file if slot else rng.global_tensor_id,
            offset,
            length,
            tensor=rng.global_tensor_id,
            **details,
        )

    def _range(self, rng):
        if self.abort is not None and self.abort.is_set():
            raise RuntimeError("A peer failed during gdtensor IO")
        geometry = self.geometries[rng.global_tensor_id]
        tensor = self.tensors[rng.global_tensor_id].to_local()
        itemsize = tensor.element_size()
        lo, hi = shard_bounds(geometry.shape, geometry.columns, self.rank)
        row_bytes = math.prod(geometry.shape[1:]) * itemsize
        global_start = lo * row_bytes
        offset = rng.global_tensor_offset - global_start
        size = rng.size
        if (
            offset < 0
            or size < 0
            or offset + size > tensor.numel() * itemsize
            or tensor.numel() * itemsize != (hi - lo) * row_bytes
            or offset % itemsize
            or size % itemsize
        ):
            raise ValueError(f"gdtensor compute range lies outside local shard: {rng!r}")
        self.requests.append((rng.global_tensor_id, rng.global_tensor_offset, size))
        return tensor, offset, size

    def _meta_buffer(self, size, write):
        name = "_meta_sink" if write else "_meta_source"
        return self._anonymous_buffer(name, size, readonly=not write)

    def _anonymous_buffer(self, name, size, *, readonly=False):
        """Grow a single mapping without touching its pages or retaining old maps."""
        buffer = getattr(self, name)
        if buffer is None or len(buffer) < size:
            if buffer is not None:
                buffer.close()
            setattr(self, name, None)
            buffer = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
            address = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
            if readonly:
                # Obtain the pointer before making the untouched mapping truly
                # read-only. ctypes.from_buffer cannot export a read-only mmap.
                protect = ctypes.CDLL(None, use_errno=True).mprotect
                if protect(ctypes.c_void_p(address), ctypes.c_size_t(size), mmap.PROT_READ):
                    error = ctypes.get_errno()
                    buffer.close()
                    raise OSError(error, os.strerror(error))
            setattr(self, name, buffer)
            setattr(self, name + "_address", address)
        return buffer, getattr(self, name + "_address")

    def _meta_position_source(self, rng, tensor, size, callback):
        """Keep bounded pattern generation for the non-timing correctness fixture."""
        itemsize = tensor.element_size()
        block_size = self.buffer_bytes // itemsize * itemsize
        address = ctypes.addressof((ctypes.c_char * self.buffer_bytes).from_buffer(self.buffer))
        for offset in range(0, size, block_size):
            if self.abort is not None and self.abort.is_set():
                raise RuntimeError("A peer failed during gdtensor IO")
            length = min(block_size, size - offset)
            actual = torch.frombuffer(self.buffer, dtype=tensor.dtype, count=length // itemsize)
            with self._stage(rng, "meta_generate", length, offset):
                actual.copy_(self._meta_expected(rng, tensor, offset, actual.numel()))
            _gdt_value(callback(address, length))
        return gb.Ok(None)

    def _meta_expected(self, rng, tensor, offset, count):
        name = rng.global_tensor_id
        aliases = self.alias_groups.get(name, (name,))
        return next(
            pattern_blocks(
                (count,),
                ((rng.global_tensor_offset + offset) // tensor.element_size(),),
                (math.prod(self.geometries[name].shape),),
                aliases[0],
                tensor.dtype,
                block=count,
            )
        )[1]

    def _check_meta(self, rng, tensor, size, buffer, *, relative=0, buffer_offset=0):
        if self.value_pattern is None:
            return
        itemsize = tensor.element_size()
        block_size = self.buffer_bytes // itemsize * itemsize
        for offset in range(0, size, block_size):
            length = min(block_size, size - offset)
            actual = torch.frombuffer(
                buffer, dtype=tensor.dtype, count=length // itemsize, offset=buffer_offset + offset
            )
            if self.value_pattern == "position":
                expected = self._meta_expected(rng, tensor, relative + offset, actual.numel())
                if not torch.equal(actual, expected):
                    raise ValueError(
                        f"numerical mismatch: {rng.global_tensor_id}, "
                        f"global element {(rng.global_tensor_offset + relative + offset) // itemsize}"
                    )
            elif torch.count_nonzero(actual).item():
                raise ValueError("nonzero checkpoint value")

    def _call_meta(self, rng, callback, write):
        tensor, local_offset, size = self._range(rng)
        if not size:
            return callback(ctypes.addressof(ctypes.c_char.from_buffer(self.buffer)), 0)
        if write and self.meta_load_sink is not None:
            sink = self.meta_load_sink
            for relative, offset, length in sink.pieces(
                rng.global_tensor_id, local_offset, size, tensor.element_size()
            ):
                _gdt_value(callback(sink.address + offset, length))
                self._check_meta(
                    rng, tensor, length, sink.buffer, relative=relative, buffer_offset=offset
                )
            if self.meta_sim_d2h_mode == "pageable":
                with self._stage(rng, "h2d", timing="simulated"):
                    self._simulate_meta_h2d(size)
            return gb.Ok(None)
        if not write and self.meta_save_source is not None:
            source = self.meta_save_source
            for _, offset, length in source.pieces(
                rng.global_tensor_id, local_offset, size, tensor.element_size()
            ):
                _gdt_value(callback(source.address + offset, length))
            return gb.Ok(None)
        if not write and self.value_pattern == "position":
            if self.meta_sim_d2h_mode == "pageable":
                raise ValueError(
                    "Meta simulated D2H save supports only zero/default value patterns"
                )
            return self._meta_position_source(rng, tensor, size, callback)
        if not write and self.meta_sim_d2h_mode == "pageable":
            self.prepare_save()
            buffer, address = self._anonymous_buffer("_cuda_scratch", size)
            try:
                with self._stage(
                    rng, "d2h", timing="simulated", population=self.meta_sim_d2h_population
                ):
                    self._simulate_meta_d2h(address, size)
                _gdt_value(callback(address, size))
            finally:
                # Mirror CUDA pageable staging: every reused range pays real
                # writable-page first-touch costs again, including on failure.
                buffer.madvise(mmap.MADV_DONTNEED)
            return gb.Ok(None)
        with self._stage(rng, "meta_generate", demand_zero=True):
            buffer, address = self._meta_buffer(size, write)
        if not write:
            return callback(address, size)
        try:
            _gdt_value(callback(address, size))
            if self.meta_sim_d2h_mode == "pageable":
                # CUDA's blocking H2D reads the completed host sink here, before
                # the next range and before MADV_DONTNEED drops its pages.
                with self._stage(rng, "h2d", timing="simulated"):
                    self._simulate_meta_h2d(size)
            self._check_meta(rng, tensor, size, buffer)
        finally:
            # Match ScratchPool.discard: the next range must pay destination
            # demand-zero faults even when it reuses exactly the same address.
            buffer.madvise(mmap.MADV_DONTNEED)
        return gb.Ok(None)

    def _simulate_meta_d2h(self, address, size):
        started = perf_counter_ns()
        if self._meta_d2h_populate is None:
            ctypes.memset(address, 0, size)
        else:
            self._meta_d2h_populate(address, size)
        self._wait_transfer_deadline(
            started + round(duration_seconds(self.meta_sim_d2h_profile, size) * 1e9)
        )

    def _simulate_meta_h2d(self, size):
        # The callback has already faulted and written every source page. CUDA
        # only reads them; they are discarded immediately after this transfer,
        # so an extra source scan would add work without reproducing page faults.
        deadline = perf_counter_ns() + round(
            meta_h2d_profile.duration_seconds(self.meta_sim_h2d_profile, size) * 1e9
        )
        self._wait_transfer_deadline(deadline)

    @staticmethod
    def _wait_transfer_deadline(deadline):
        while (remaining := deadline - perf_counter_ns()) > 0:
            if remaining >= 1_000_000:
                sleep(remaining / 1e9)
            # Short tails spin on the host clock. If zero fill exceeded the
            # profile target, this loop never waits.

    def _call_cuda(self, rng, callback, write):
        tensor, offset, size = self._range(rng)
        if not size:
            return callback(ctypes.addressof(ctypes.c_char.from_buffer(self.buffer)), 0)
        itemsize = tensor.element_size()
        flat = tensor.detach().view(-1)
        if self.cuda_staging_mode == "pageable":
            buffer, address = self._anonymous_buffer("_cuda_scratch", size)
            try:
                staged = torch.frombuffer(buffer, dtype=tensor.dtype, count=size // itemsize)
                target = flat.narrow(0, offset // itemsize, size // itemsize)
                if not write:
                    # copy_(non_blocking=False) already waits for the real D2H
                    # transfer. Extra synchronize() would change stream overlap.
                    with self._stage(rng, "d2h", timing="blocking_copy_host_clock"):
                        staged.copy_(target, non_blocking=False)
                _gdt_value(callback(address, size))
                if write:
                    with self._stage(rng, "h2d", timing="blocking_copy_host_clock"):
                        target.copy_(staged, non_blocking=False)
            finally:
                # Both blocking copies finish before dropping the host pages.
                # Reused ranges must pay the same first-touch cost as Meta sinks.
                buffer.madvise(mmap.MADV_DONTNEED)
            return gb.Ok(None)
        capacity = size if self.cuda_staging_bytes is None else min(size, self.cuda_staging_bytes)
        if not self._cuda_buffers or self._cuda_buffers[0].numel() < capacity:
            # Previous calls drain all copies before returning. Release old
            # ownership before growing, including on resharded partial ranges.
            with self._stage(
                rng, "pinned_alloc", capacity, buffers=(1 if self.cuda_staging_bytes is None else 2)
            ):
                self._cuda_buffers.clear()
                count = 1 if self.cuda_staging_bytes is None else 2
                self._cuda_buffers = [
                    torch.empty(capacity, dtype=torch.uint8, device="cpu", pin_memory=True)
                    for _ in range(count)
                ]
        if self.cuda_staging_bytes is None:
            staged = self._cuda_buffers[0][:size].view(tensor.dtype)
            target = flat.narrow(0, offset // itemsize, size // itemsize)
            if not write:
                with self._stage(rng, "d2h", timing="blocking_copy_host_clock"):
                    staged.copy_(target, non_blocking=False)
            _gdt_value(callback(staged.data_ptr(), size))
            if write:
                with self._stage(rng, "h2d", timing="blocking_copy_host_clock"):
                    target.copy_(staged, non_blocking=False)
            return gb.Ok(None)
        return self._call_cuda_bounded(rng, callback, write, tensor, flat, offset, size)

    def _call_cuda_bounded(self, rng, callback, write, tensor, flat, offset, size):
        """Overlap host callbacks with copies, protecting both buffer lifetimes."""
        itemsize = tensor.element_size()
        block_size = self._cuda_buffers[0].numel() // itemsize * itemsize
        timed = self.staging_events is not None
        events = [torch.cuda.Event(enable_timing=timed) for _ in range(2)]
        pending = [False, False]
        copy_starts = [None, None]
        anchor, anchor_host = None, None

        def finish_copy(slot):
            if not timed or copy_starts[slot] is None:
                return
            index, started, enqueued, begin = copy_starts[slot]
            _, _, length = block(index)
            with self._stage(
                rng,
                "h2d" if write else "d2h",
                length,
                index * block_size,
                timing="enqueue_to_existing_wait_upper_bound",
                cuda_duration_ns=round(begin.elapsed_time(events[slot]) * 1e6),
                cuda_start_relative_ns=round(anchor.elapsed_time(begin) * 1e6),
                cuda_end_relative_ns=round(anchor.elapsed_time(events[slot]) * 1e6),
                cuda_anchor_host_ns=anchor_host,
                enqueue_end_ns=enqueued,
                chunk=index,
                buffer_slot=slot,
            ):
                pass
            event = self.staging_events[-1]
            event["start_ns"] = started
            event["duration_ns"] = event["end_ns"] - started
            copy_starts[slot] = None

        def wait_copy(slot, reason):
            if not pending[slot]:
                return
            index = copy_starts[slot][0] if timed else slot
            _, _, length = block(index)
            with self._stage(
                rng,
                "event_wait",
                length,
                index * block_size,
                reason=reason,
                chunk=index,
                buffer_slot=slot,
            ):
                events[slot].synchronize()
            finish_copy(slot)
            pending[slot] = False

        def enqueue(index, staged, target):
            if timed:
                started = perf_counter_ns()
                begin = torch.cuda.Event(enable_timing=True)
                begin.record()
            if write:
                target.copy_(staged, non_blocking=True)
            else:
                staged.copy_(target, non_blocking=True)
            events[index % 2].record()
            if timed:
                copy_starts[index % 2] = (index, started, perf_counter_ns(), begin)
            pending[index % 2] = True

        def block(index):
            start = index * block_size
            length = min(block_size, size - start)
            staged = self._cuda_buffers[index % 2][:length].view(tensor.dtype)
            target = flat.narrow(0, (offset + start) // itemsize, length // itemsize)
            return staged, target, length

        def stage_read(index):
            staged, target, _ = block(index)
            enqueue(index, staged, target)

        blocks = (size + block_size - 1) // block_size
        with torch.cuda.device(tensor.device):
            if timed:
                # Device offsets are exact; placement on the host clock is an
                # estimate anchored before this record (no extra sync). Retain
                # enqueue/observed-wait bounds separately for honest overlap analysis.
                anchor_host = perf_counter_ns()
                anchor = torch.cuda.Event(enable_timing=True)
                anchor.record()
            try:
                if not write:
                    for index in range(min(2, blocks)):
                        stage_read(index)
                for index in range(blocks):
                    if self.abort is not None and self.abort.is_set():
                        raise RuntimeError("A peer failed during gdtensor IO")
                    slot = index % 2
                    wait_copy(
                        slot,
                        "buffer_reuse"
                        if write
                        else ("first_chunk" if index == 0 else "chunk_ready"),
                    )
                    staged, target, length = block(index)
                    _gdt_value(callback(staged.data_ptr(), length))
                    if write:
                        enqueue(index, staged, target)
                    elif index + 2 < blocks:
                        stage_read(index + 2)
            finally:
                # Preserve the original waits, including failure cleanup.
                for slot in range(2):
                    wait_copy(slot, "final_drain")
        return gb.Ok(None)

    def read_to_buffer(self, rng, callback):
        if self.staging_events is not None:
            original_callback = callback
            consumed = 0

            def callback(address, size):
                nonlocal consumed
                # gdtensor's take() materializes the source via string_at().
                # This includes demand-zero faults for Meta's untouched pages.
                with self._stage(rng, "host_materialize", size, consumed):
                    result = original_callback(address, size)
                consumed += size
                return result

        tensor = self.tensors[rng.global_tensor_id].to_local()
        if tensor.is_meta:
            return self._call_meta(rng, callback, False)
        if tensor.device.type == "cuda":
            return self._call_cuda(rng, callback, False)
        with self._stage(rng, "cpu_view"):
            tensor, offset, size = self._range(rng)
        return callback(tensor.data_ptr() + offset, size)

    def write_from_buffer(self, rng, callback):
        if self.staging_events is not None:
            original_callback = callback
            consumed = 0

            def callback(address, size):
                nonlocal consumed
                with self._stage(rng, "host_sink", size, consumed):
                    result = original_callback(address, size)
                consumed += size
                return result

        tensor = self.tensors[rng.global_tensor_id].to_local()
        if tensor.is_meta:
            return self._call_meta(rng, callback, True)
        if tensor.device.type == "cuda":
            return self._call_cuda(rng, callback, True)
        tensor, offset, size = self._range(rng)
        return callback(tensor.data_ptr() + offset, size)


class StorageAccessor(DTensorDataAccessor):
    """Map public packed payload ranges to container records, with bounded buffers."""

    def __init__(
        self,
        slots,
        read=None,
        write=None,
        get_file_offset=None,
        on_write=None,
        buffer_bytes=ACCESSOR_BUFFER_BYTES,
        abort=None,
        staging_events=None,
    ):
        if buffer_bytes < 1:
            raise ValueError("gdtensor storage buffer must be nonempty")
        self.slots = defaultdict(list)
        for slot in slots:
            if slot.size:
                self.slots[slot.file].append(slot)
        for file_slots in self.slots.values():
            file_slots.sort(key=lambda slot: slot.payload_offset)
        self.starts = {
            path: [slot.payload_offset for slot in file_slots]
            for path, file_slots in self.slots.items()
        }
        self.read = read
        self.write = write
        self.get_file_offset = get_file_offset
        self.on_write = on_write
        self._buffer = None
        self.buffer_bytes = buffer_bytes
        self.abort = abort
        self.requests = []
        self.staging_events = staging_events

    @property
    def buffer(self):
        # The view-level accessor is only a required metadata handle; the
        # executor uses the session accessor, so allocate on its first IO.
        if self._buffer is None:
            self._buffer = bytearray(self.buffer_bytes)
        return self._buffer

    def _range(self, rng):
        if self.abort is not None and self.abort.is_set():
            raise RuntimeError("A peer failed during gdtensor IO")
        index = bisect.bisect_right(
            self.starts.get(rng.file_path, ()), rng.contiguous_payload_offset
        )
        if index < 1:
            raise ValueError(f"Unknown gdtensor payload range: {rng!r}")
        slot = self.slots[rng.file_path][index - 1]
        relative = rng.contiguous_payload_offset - slot.payload_offset
        if relative < 0 or rng.size < 0 or relative + rng.size > slot.size:
            raise ValueError(f"gdtensor payload range crosses a slot boundary: {rng!r}")
        base = slot.file_offset if self.get_file_offset is None else self.get_file_offset(slot)
        return slot, relative, base + relative

    def read_to_buffer(self, rng, callback):
        slot, _, file_offset = self._range(rng)
        if self.read is None:
            raise ValueError("gdtensor storage read accessor is not bound to an open session")
        address = ctypes.addressof((ctypes.c_char * self.buffer_bytes).from_buffer(self.buffer))
        for offset in range(0, rng.size, self.buffer_bytes):
            if self.abort is not None and self.abort.is_set():
                raise RuntimeError("A peer failed during gdtensor IO")
            length = min(self.buffer_bytes, rng.size - offset)
            data = self.read(slot.file, file_offset + offset, length)
            if len(data) != length:
                raise OSError("Short gdtensor container payload read")
            with _staging_event(
                self.staging_events, "host_read_copy", slot.file, file_offset + offset, length
            ):
                ctypes.memmove(address, data, length)
            self.requests.append((slot.file, file_offset + offset, length))
            with _staging_event(
                self.staging_events,
                "host_read_materialize",
                slot.file,
                file_offset + offset,
                length,
            ):
                _gdt_value(callback(address, length))
        return gb.Ok(None)

    def write_from_buffer(self, rng, callback):
        slot, relative, file_offset = self._range(rng)
        if self.write is None:
            raise ValueError("gdtensor storage write accessor is not bound to an open session")
        address = ctypes.addressof((ctypes.c_char * self.buffer_bytes).from_buffer(self.buffer))
        for offset in range(0, rng.size, self.buffer_bytes):
            if self.abort is not None and self.abort.is_set():
                raise RuntimeError("A peer failed during gdtensor IO")
            length = min(self.buffer_bytes, rng.size - offset)
            if self.staging_events is None:
                _gdt_value(callback(address, length))
            else:
                with _staging_event(
                    self.staging_events,
                    "host_copy",
                    slot.file,
                    file_offset + offset,
                    length,
                ):
                    _gdt_value(callback(address, length))
            data = memoryview(self.buffer)[:length]
            self.write(slot.file, file_offset + offset, data)
            if self.on_write is not None:
                self.on_write(slot, relative + offset, data)
            self.requests.append((slot.file, file_offset + offset, length))
        return gb.Ok(None)


def make_views(tensors, geometries, slots, compute_accessor, begin_save=None, begin_load=None):
    """Build independent storage/compute meshes from pure FSDP geometries."""
    names = list(geometries)
    row_order = {name: row for row, name in enumerate(names)}
    if set(tensors) != set(names):
        raise ValueError("gdtensor compute and storage tensor sets differ")
    assign_payload_offsets(slots, names)
    compute_table, storage_table = [], []
    for name in names:
        target = _compute_geometry(tensors[name])
        source = geometries[name]
        if target.shape != tuple(source.shape) or target.dtype != source.dtype:
            raise ValueError(f"gdtensor source and destination metadata differ: {name}")
        meta = gb.GlobalTensorMeta(
            list(source.shape), _GDT_DTYPES[source.dtype], tensors[name].element_size()
        )
        compute_table.append(
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(meta, [gb.Shard(0)], gb.ArrangedMesh(WorldSize(target.columns))),
            )
        )
        storage_table.append(
            gb.NamedDTensor(
                name,
                gb.NaiveDTensor(
                    meta,
                    [gb.Replicate(), gb.Shard(0)],
                    gb.SlotOuterProductMesh(
                        gb.ArrangedMesh(ConstantOne(1)), gb.ArrangedMesh(WorldSize(source.columns))
                    ),
                ),
            )
        )
    grouped = defaultdict(list)
    for slot in slots:
        grouped[slot.file].append(slot)
    entries, paths = {}, {}
    for path, file_slots in grouped.items():
        rows_by_column = defaultdict(list)
        for slot in file_slots:
            source = geometries[slot.name]
            lo, hi = shard_bounds(source.shape, source.columns, slot.column)
            row_bytes = (
                math.prod(source.shape[1:]) * torch.empty((), dtype=source.dtype).element_size()
            )
            if slot.global_offset != lo * row_bytes or slot.size != (hi - lo) * row_bytes:
                raise ValueError(f"gdtensor slot does not match its Shard(0) grid: {slot.name}")
            rows_by_column[(source.columns, slot.column)].append(row_order[slot.name])
        cells = []
        for (columns, column), rows in sorted(rows_by_column.items()):
            ordered_rows = sorted(rows)
            # A rank file normally owns the same column of every tensor.
            # Describe each contiguous run once, rather than attaching one
            # full tensor-table mesh per cell to every tensor entry. Literal
            # content slots also avoid resolving unrelated storage columns.
            start = previous = ordered_rows[0]
            for row in ordered_rows[1:] + [None]:
                if row is not None and row == previous + 1:
                    previous = row
                    continue
                cells.append(
                    gb.SlicedMesh(
                        gb.SlotOuterProductMesh(
                            gb.ArrangedMesh(TensorCount(len(names))),
                            gb.LiteralMesh([column], WorldSize(columns)),
                        ),
                        [gb.Slice(start, previous + 1), gb.Slice(column, column + 1)],
                    )
                )
                start = previous = row
        group = gb.DisjointStorageMeshList(cells, names)
        leader = gb.leader_tensor_component_pos_of(group)
        if isinstance(leader, gb.Err):
            leader = _gdt_value(leader)
        paths[leader] = path
        for slot in file_slots:
            entries[gb.TensorComponentPos(slot.name, slot.column)] = group
    compute_view = gb.ComputeDTensorsView(names, gb.DTensorTable(compute_table), compute_accessor)

    def unavailable(view, rank):
        raise ValueError("Requested gdtensor storage operation is not available in this session")

    storage_view = gb.StorageDTensorsView(
        names,
        gb.DTensorTable(storage_table),
        StorageAccessor(slots),
        gb.PhysicalLayoutTable("distcp", entries, paths),
        begin_save or unavailable,
        begin_load or unavailable,
    )
    _gdt_value(compute_view.check_constraints())
    _gdt_value(storage_view.check_constraints())
    return compute_view, storage_view


class GDTensorSavePlanner(DefaultSavePlanner):
    """Default DCP planning; DTensor data is consumed directly by the accessor."""

    def __init__(self, simulated=False, pattern="zero", rows=None):
        super().__init__()
        self.simulated, self.pattern = simulated, pattern
        self.rows = rows or []
        self.meta_save_source = None
        self.meta_save_source_report = dict(
            mode="demand-zero", resident_buffer_bytes=0, prefaulted=False,
            preparation_inside_save_call=False,
        )
        self._meta_save_prepared = False
        if simulated and os.environ.get("METASL_GDTENSOR_META_SIM_D2H") == "pageable":
            # Planner construction precedes the measured DCP API call. Build
            # the helper here; only real fill/flush work belongs in D2H events.
            from metasl.meta_d2h_cache import DEFAULT_POPULATION, population_function

            population_function(
                os.environ.get("METASL_GDTENSOR_META_SIM_D2H_POPULATE", DEFAULT_POPULATION)
            )

    def prepare_save(self, tensors):
        """Prepare CPU-target source pages before the dcp.save clock.

        Faithful and ring select both directions of the CPU comparison. Ring
        reuses one fixed-size buffer for all source ranges, capped at shard size.
        Pass the same flat FQNs that DCP uses; repeated preparation is a no-op.
        """
        if self._meta_save_prepared:
            return
        mode = os.environ.get("METASL_GDTENSOR_META_LOAD_SINK") or "fresh"
        if mode not in ("fresh", "faithful", "ring"):
            raise ValueError(f"Unknown Meta load sink mode: {mode}")
        if self.simulated and mode != "fresh":
            if (os.environ.get("METASL_GDTENSOR_META_SIM_D2H") or "off") != "off":
                raise ValueError("Resident Meta CPU-target sources require simulated D2H off")
            if mode == "ring" and self.pattern == "position":
                raise ValueError(
                    "Meta ring buffers support only zero/default value patterns; "
                    "use faithful or fresh for position"
                )
            capacity = None
            if mode == "ring":
                configured = os.environ.get("METASL_GDTENSOR_META_LOAD_SINK_BYTES")
                capacity = (
                    MetaSaveSource.DEFAULT_RING_BYTES if configured is None else int(configured)
                )
            local_tensors = {
                name: tensor for name, tensor in tensors.items()
                if isinstance(tensor, DTensor) and tensor.to_local().is_meta
            }
            source = MetaSaveSource(
                {name: tensor.to_local().nbytes for name, tensor in local_tensors.items()},
                mode=mode,
                capacity=capacity,
            )
            started = source.prepared_ns - source.prepare_duration_ns
            if self.pattern == "position":
                rank = dist.get_rank() if dist.is_initialized() else 0
                aliases = {row["fqn"]: row["alias_group"][0] for row in self.rows}
                for name, tensor in local_tensors.items():
                    local = tensor.to_local()
                    if not local.nbytes:
                        continue
                    target = torch.frombuffer(
                        source.buffer, dtype=local.dtype, count=local.numel(),
                        offset=source.tensor_offsets[name],
                    ).reshape(local.shape)
                    row_bytes = math.prod(tensor.shape[1:]) * local.element_size()
                    offsets = (_local_byte_start(tensor, rank) // row_bytes,) + (
                        0,
                    ) * (tensor.ndim - 1)
                    fill_or_check(
                        target, offsets, tuple(tensor.shape), aliases.get(name, name), "position"
                    )
            elif self.pattern not in (None, "zero"):
                raise ValueError(f"Unknown gdtensor value pattern: {self.pattern}")
            source.prepared_ns = perf_counter_ns()
            source.prepare_duration_ns = source.prepared_ns - started
            self.meta_save_source = source
            self.meta_save_source_report = source.report()
        self._meta_save_prepared = True

    def set_up_planner(self, state_dict, storage_meta=None, is_coordinator=False):
        if (
            self.simulated
            and os.environ.get("METASL_GDTENSOR_META_LOAD_SINK") in ("faithful", "ring")
            and not self._meta_save_prepared
        ):
            raise ValueError(
                "Resident Meta save sources require planner.prepare_save(flat_state_dict) "
                "before starting the dcp.save timer"
            )
        return super().set_up_planner(state_dict, storage_meta, is_coordinator)

    def resolve_data(self, item):
        if item.type == WriteItemType.BYTE_IO or not self.simulated:
            return super().resolve_data(item)
        # Only small plain tensors reach this native serialization path.
        data = item.tensor_data
        tensor = torch.zeros(data.chunk.sizes, dtype=data.properties.dtype)
        if self.pattern == "position":
            row = next(r for r in self.rows if r["fqn"] == item.index.fqn)
            fill_or_check(
                tensor, data.chunk.offsets, row["global_shape"], row["alias_group"][0], "position"
            )
        return tensor


def infer_shard_grid(name, md, columns=None):
    """Validate a full checkpoint tensor's torch.chunk (ceil) Shard(0) grid."""
    shape = tuple(md.size)
    if not shape:
        raise ValueError(f"{name}: scalar storage is native, not a Shard(0) grid")
    columns = columns or max(1, len(md.chunks))
    width = (shape[0] + columns - 1) // columns
    expected = {}
    for column in range(columns):
        start = min(column * width, shape[0])
        extent = min(width, shape[0] - start)
        expected[(start,) + (0,) * (len(shape) - 1)] = (extent,) + shape[1:]
    actual = {tuple(c.offsets): tuple(c.sizes) for c in md.chunks}
    # Dictionaries collapse the identical trailing empty slots DCP deduplicates.
    if actual != expected or len(actual) != len(md.chunks):
        raise ValueError(f"{name}: irregular chunk grid; expected torch.chunk ceil Shard(0)")
    return columns


class GDTensorLoadPlanner(OptionalKeysLoadPlanner):
    """Default ReadItems with independently checked gdtensor element coverage.

    Simulated targets stay meta. Small native tensors alone use temporary CPU
    views; DTensors use the bounded accessor sink instead of scratch views.
    """

    def __init__(
        self, rows=None, simulated=False, pattern=None, optional_keys=(), abort=None, progress=None
    ):
        super().__init__(optional_keys=optional_keys)
        self.rows, self.simulated, self.pattern = rows, simulated, pattern
        self.abort, self.progress = abort, progress
        self.expected_items = self.completed_items = 0
        self.expected_elements, self.completed_elements = {}, {}
        self.completed_native = []
        self.copy_events = []
        self.meta_load_sink = None
        self.meta_load_sink_report = dict(
            mode="fresh", resident_buffer_bytes=0, prefaulted=False,
            preparation_inside_load_call=False,
        )
        self._meta_load_prepared = False

    def prepare_load(self, tensors):
        """Allocate and write-fault a CPU-target sink before measuring dcp.load.

        Pass a flat state dictionary, with the same FQNs DCP will use. The normal
        phase runner calls this before its barrier and phase clock; direct DCP
        callers enabling a resident sink must do the same. CPU/CUDA targets keep
        their direct destination path. Repeated preparation is a no-op.
        """
        if self._meta_load_prepared:
            return
        mode = os.environ.get("METASL_GDTENSOR_META_LOAD_SINK") or "fresh"
        if mode not in ("fresh", "faithful", "ring"):
            raise ValueError(f"Unknown Meta load sink mode: {mode}")
        if self.simulated and mode != "fresh":
            if mode == "ring" and self.pattern == "position":
                raise ValueError(
                    "Meta ring buffers support only zero/default value patterns; "
                    "use faithful or fresh for position"
                )
            capacity = None
            if mode == "ring":
                configured = os.environ.get("METASL_GDTENSOR_META_LOAD_SINK_BYTES")
                capacity = MetaLoadSink.DEFAULT_RING_BYTES if configured is None else int(configured)
            tensor_bytes = {
                name: tensor.to_local().nbytes
                for name, tensor in tensors.items()
                if isinstance(tensor, DTensor) and tensor.to_local().is_meta
            }
            self.meta_load_sink = MetaLoadSink(tensor_bytes, mode=mode, capacity=capacity)
            self.meta_load_sink_report = self.meta_load_sink.report()
        self._meta_load_prepared = True

    def set_up_planner(self, state_dict, metadata=None, is_coordinator=False):
        mode = os.environ.get("METASL_GDTENSOR_META_LOAD_SINK") or "fresh"
        if self.simulated and mode != "fresh" and not self._meta_load_prepared:
            raise ValueError(
                "Resident Meta load sinks require planner.prepare_load(flat_state_dict) "
                "before starting the dcp.load timer"
            )
        if self.simulated:
            MetaLoadPlanner.set_up_planner(self, state_dict, metadata, is_coordinator)
        else:
            super().set_up_planner(state_dict, metadata, is_coordinator)
        for name, value in self.state_dict.items():
            if isinstance(value, torch.Tensor):
                md = metadata.state_dict_metadata[name]
                if md.size != value.size() or md.properties.dtype != value.dtype:
                    raise ValueError(f"Checkpoint structure mismatch for {name}")

    def finish_plan(self, plan):
        plan = super().finish_plan(plan)
        self.tensor_items = [i for i in plan.items if i.type != LoadItemType.BYTE_IO]
        self.expected_items = len(self.tensor_items)
        self.expected_elements = collections.Counter()
        for item in self.tensor_items:
            self.expected_elements[item.dest_index.fqn] += math.prod(item.lengths)
        self.requests = [read_record(i, self.metadata) for i in self.tensor_items]
        if self.rows is not None:
            audit_local(self.rows, self.requests)
        return plan

    def resolve_tensor(self, item):
        if self.simulated:
            return torch.empty(item.lengths, dtype=self.state_dict[item.dest_index.fqn].dtype)
        return super().resolve_tensor(item)

    def commit_tensor(self, item, tensor):
        self.completed_native.append(item)
        if not self.simulated:
            super().commit_tensor(item, tensor)
        if self.pattern is not None:
            record = read_record(item, self.metadata)
            row = next(r for r in self.rows if r["fqn"] == item.dest_index.fqn)
            fill_or_check(
                tensor,
                record["global_offsets"],
                row["global_shape"],
                row["alias_group"][0],
                self.pattern,
                check=True,
            )

    def complete_gdtensor(self, accessor):
        actual = collections.defaultdict(list)
        for name, offset, size in accessor.requests:
            actual[name].append((offset, offset + size))
        expected = collections.defaultdict(list)
        for item in self.tensor_items:
            name = item.dest_index.fqn
            tensor = self.state_dict[name]
            if not isinstance(tensor, DTensor):
                continue
            if tuple(item.lengths[1:]) != tuple(tensor.shape[1:]):
                raise ValueError(f"{name}: non-FSDP ReadItem")
            width = tensor.element_size() * math.prod(tensor.shape[1:])
            begin = (item.storage_index.offset[0] + item.storage_offsets[0]) * width
            end = begin + math.prod(item.lengths) * tensor.element_size()
            if begin != end:
                expected[name].append((begin, end))
        names = {name for name, tensor in self.state_dict.items() if isinstance(tensor, DTensor)}
        for name in names | set(actual) | set(expected):
            if _merge_intervals(actual[name]) != _merge_intervals(expected[name]):
                raise ValueError(f"{name}: gdtensor coverage differs from DCP ReadItems")
            tensor = self.state_dict[name]
            local = tensor.to_local()
            begin = _local_byte_start(tensor, dist.get_rank() if dist.is_initialized() else 0)
            wanted = [(begin, begin + local.nbytes)] if local.nbytes else []
            if _merge_intervals(actual[name]) != wanted:
                raise ValueError(f"{name}: gdtensor did not write every local element exactly once")
        native = collections.Counter(_read_item_key(i) for i in self.completed_native)
        required = collections.Counter(
            _read_item_key(i)
            for i in self.tensor_items
            if not isinstance(self.state_dict[i.dest_index.fqn], DTensor)
        )
        if native != required:
            raise ValueError("Incomplete native tensor load")
        self.completed_items = self.expected_items
        self.completed_elements = dict(self.expected_elements)
        self.coverage_verified = True
        if self.progress is not None:
            with self.progress.get_lock():
                self.progress.value += sum(
                    self.expected_elements[n] * self.state_dict[n].element_size()
                    for n in self.expected_elements
                )

    def assert_complete(self):
        if not getattr(self, "coverage_verified", False):
            raise ValueError("gdtensor load coverage was not checked")
        if self.completed_items != self.expected_items:
            raise ValueError("Incomplete gdtensor load")


def _read_item_key(item):
    return (
        item.dest_index.fqn,
        tuple(item.dest_index.offset or ()),
        tuple(item.storage_index.offset or ()),
        tuple(item.dest_offsets),
        tuple(item.storage_offsets),
        tuple(item.lengths),
    )


def _merge_intervals(intervals):
    result = []
    for start, end in sorted(intervals):
        if start == end:
            continue
        if result and start < result[-1][1]:
            raise ValueError("gdtensor copied an element more than once")
        if result and start == result[-1][1]:
            result[-1] = (result[-1][0], end)
        else:
            result.append((start, end))
    return result


def _local_byte_start(tensor, rank):
    world = tensor.device_mesh.size()
    rows = (tensor.shape[0] + world - 1) // world
    return min(rank * rows, tensor.shape[0]) * math.prod(tensor.shape[1:]) * tensor.element_size()


class _SaveSession(StorageSaveSession):
    def __init__(
        self,
        stream,
        slots,
        sync_files,
        abort=None,
        *,
        payload_events=None,
        container_patch_events=None,
        staging_events=None,
    ):
        self.stream, self.slots, self.sync_files = stream, slots, sync_files
        self.payload_events = payload_events
        self.container_patch_events = container_patch_events
        self.patched = set()
        self.crcs = {id(slot): 0 for slot in slots}
        self.first_bytes = {id(slot): bytearray() for slot in slots}
        self.received = {id(slot): 0 for slot in slots}
        self.accessor = StorageAccessor(
            slots,
            write=self.write,
            on_write=self.observe,
            abort=abort,
            staging_events=staging_events,
        )

    def write(self, file, offset, data):
        # The header stream has already been flushed; payload uses positioned IO.
        done = 0
        while done < len(data):
            count = 0
            started = perf_counter_ns() if self.payload_events is not None else None
            try:
                count = os.pwrite(self.stream.fileno(), data[done:], offset + done)
            finally:
                if self.payload_events is not None:
                    self.payload_events.append(
                        dict(
                            file=file,
                            op="pwrite",
                            offset=offset + done,
                            requested=len(data) - done,
                            completed=count,
                            start_ns=started,
                            duration_ns=perf_counter_ns() - started,
                            success=count > 0,
                            thread_id=get_native_id(),
                        )
                    )
            if count <= 0:
                raise OSError("Short gdtensor payload write")
            done += count

    def observe(self, slot, offset, data):
        if offset != self.received[id(slot)]:
            raise ValueError("Out-of-order or duplicated gdtensor save range")
        if offset + len(data) > slot.size:
            raise ValueError("gdtensor save range exceeds payload size")
        prefix = self.first_bytes[id(slot)]
        if len(prefix) < 8:
            prefix.extend(data[: 8 - len(prefix)])
        self.crcs[id(slot)] = zlib.crc32(data, self.crcs[id(slot)])
        self.received[id(slot)] += len(data)
        if self.received[id(slot)] == slot.size:
            # StorageAccessor calls observe only after write's synchronous
            # pwrites return. The writer flushed all envelopes before creating
            # this session and does no buffered writes during rank_save. Patch
            # offsets lie outside payload ranges; later slots cannot overwrite
            # them. _container_pwrite also flushes before using the descriptor.
            self._patch(slot)

    def _patch(self, slot):
        if id(slot) not in self.patched:
            patch_container(
                self.stream,
                slot.container,
                self.crcs[id(slot)],
                bytes(self.first_bytes[id(slot)]),
                write_events=self.container_patch_events,
                file=slot.file,
            )
            self.patched.add(id(slot))

    def get_file_offset(self, pos):
        return Ok(
            next(
                slot.file_offset
                for slot in self.slots
                if slot.name == pos.tensor_id and slot.column == pos.rank
            )
        )

    def init_data_accessor(self):
        return Ok(self.accessor)

    def commit(self):
        for slot in self.slots:
            if self.received[id(slot)] != slot.size:
                raise ValueError(f"Incomplete gdtensor payload for {slot.name}")
            self._patch(slot)
        self.stream.flush()
        if self.sync_files:
            fsync_stream(self.stream)
        return Ok(None)

    def abort(self):
        return Ok(None)


class GDTensorFileSystemWriter(AdmittedFileSystemWriter):
    """Admitted native DCP planning/finish, gdtensor DTensor payload movement."""

    buffer_bytes = 2 * 65536

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        if self.thread_count != 1 or not self.single_file_per_rank:
            raise ValueError("gdtensor writer requires threads=1 and single_file_per_rank=True")
        self.payload_requests = []
        self.capture_payload_events = False
        self.payload_events = []
        self.container_patch_events = []
        self.staging_events = []

    def finish(self, metadata, results):
        """Keep native metadata publication, routing fsync through its trace proxy.

        This is torch's FileSystemWriter.finish sequence: the same pickle,
        flush/fsync/close, destination removal and atomic rename. Its explicit
        fsync must use the proxy because a stream's fileno() cannot observe the
        duration of a subsequent os.fsync call.
        """
        metadata.version = CURRENT_DCP_VERSION
        storage_md = {}
        for wr_list in results:
            storage_md.update({wr.index: wr.storage_data for wr in wr_list})
        metadata.storage_data = storage_md
        metadata.storage_meta = self.storage_meta()
        tmp_filename = (
            f"__{self.rank}{_metadata_fn}.tmp"
            if not self.use_collectives and self.rank is not None
            else f"{_metadata_fn}.tmp"
        )
        tmp_path = self.fs.concat_path(self.path, tmp_filename)
        with self.fs.create_stream(tmp_path, "wb") as metadata_file:
            pickle.dump(metadata, metadata_file)
            if self.sync_files:
                metadata_file.flush()
                try:
                    fsync_stream(metadata_file)
                except (AttributeError, io.UnsupportedOperation) as error:
                    warnings.warn(
                        f"fsync not supported for this stream, relying on flush(): {error}",
                        stacklevel=2,
                    )
        metadata_path = (
            self._get_metadata_path(self.rank)
            if not self.use_collectives and self.rank is not None
            else self._get_metadata_path()
        )
        if self.fs.exists(metadata_path):
            self.fs.rm_file(metadata_path)
        self.fs.rename(tmp_path, metadata_path)

    def _write_data(self, planner, file_queue):
        results = []
        custom_device = getattr(torch, torch._C._get_privateuse1_backend_name(), None)
        # Native DCP's overlapping CPU loader stably sorts tensors by global
        # size whenever an accelerator and copy-ahead are available. Match its
        # container order without materializing or prefetching entire shards.
        native_size_order = (
            self.thread_count == 1
            and self.per_thread_copy_ahead > 0
            and (
                torch.cuda.is_available()
                or (custom_device is not None and custom_device.is_available())
            )
        )
        while not file_queue.empty():
            path, file, items = file_queue.get_nowait()
            with self.fs.create_stream(path, "wb") as stream:
                tensor_items = [i for i in items if i.type != WriteItemType.BYTE_IO]
                if native_size_order:
                    tensor_items.sort(
                        key=lambda item: (
                            math.prod(item.tensor_data.size)
                            * element_size(item.tensor_data.properties.dtype)
                        )
                    )
                ordered = [i for i in items if i.type == WriteItemType.BYTE_IO] + tensor_items
                tensors = {
                    i.index.fqn: planner.state_dict[i.index.fqn]
                    for i in items
                    if isinstance(planner.state_dict[i.index.fqn], DTensor)
                }
                geometries, slots = {}, []
                rank = dist.get_rank() if dist.is_initialized() else 0
                for item in ordered:
                    start = stream.tell()
                    if item.index.fqn not in tensors:
                        value = planner.resolve_data(item)
                        if item.type == WriteItemType.BYTE_IO:
                            stream.write(value.getbuffer())
                        else:
                            value = value.detach().cpu()
                            if value.untyped_storage().nbytes() != value.nbytes:
                                value = value.clone()
                            torch.save(value, stream)
                    else:
                        tensor = tensors[item.index.fqn]
                        geometry = TensorGeometry(
                            tuple(tensor.shape), tensor.dtype, tensor.device_mesh.size()
                        )
                        geometries[item.index.fqn] = geometry
                        container = begin_container(
                            stream,
                            item.tensor_data.chunk.sizes,
                            tensor.dtype,
                            stride=tensor.to_local().stride(),
                        )
                        slots.append(
                            PayloadSlot(
                                item.index.fqn,
                                rank,
                                file,
                                math.prod(item.tensor_data.chunk.sizes) * tensor.element_size(),
                                _local_byte_start(tensor, rank),
                                file_offset=container.payload_offset,
                                container=container,
                            )
                        )
                    length = stream.tell() - start
                    results.append(
                        WriteResult(item.index, length, _StorageInfo(file, start, length))
                    )
                stream.flush()
                if tensors:
                    # The session caches packed offsets when its accessor is
                    # constructed. Use the view's tensor-table order now;
                    # native accelerator ordering may differ from plan order.
                    assign_payload_offsets(slots, list(geometries))
                    session = _SaveSession(
                        stream,
                        slots,
                        self.sync_files,
                        self.abort,
                        payload_events=self.payload_events if self.capture_payload_events else None,
                        container_patch_events=(
                            self.container_patch_events if self.capture_payload_events else None
                        ),
                        staging_events=self.staging_events if self.capture_payload_events else None,
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
                    # Publish only this file's events, including partial failures.
                    groups = [
                        ("payload", self.payload_events),
                        ("container_patch", self.container_patch_events),
                        ("staging", self.staging_events),
                    ]
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
                        {"file": file, "file_offset": offset, "length": size}
                        for file, offset, size in session.accessor.requests
                    )
                else:
                    stream.flush()
                    if self.sync_files:
                        fsync_stream(stream)
        future = Future()
        future.set_result(results)
        return future


class _ReadFileSystem:
    """Share native-item, header and payload reads within one admitted call."""

    def __init__(self, fs, stack):
        self.fs, self.stack = fs, stack
        self.streams = {}

    def __getattr__(self, name):
        return getattr(self.fs, name)

    @contextmanager
    def create_stream(self, path, mode):
        if mode != "rb":
            raise ValueError("gdtensor load streams must be read-only")
        key = str(path)
        if key not in self.streams:
            self.streams[key] = self.stack.enter_context(self.fs.create_stream(path, mode))
        yield self.streams[key]


class _LoadSession(StorageLoadSession):
    def __init__(self, reader, slots):
        self.reader, self.slots = reader, slots
        self.streams = {}
        self.stack = ExitStack()
        self.accessor = StorageAccessor(
            slots,
            read=self.read,
            abort=reader.abort,
            staging_events=reader.staging_events if reader.capture_payload_events else None,
        )

    def read(self, file, offset, size):
        if file not in self.streams:
            path = self.reader.fs.concat_path(self.reader.path, file)
            self.streams[file] = self.stack.enter_context(self.reader.fs.create_stream(path, "rb"))
        stream = self.streams[file]
        stream.seek(offset)
        data = stream.read(size)
        if self.reader.capture_payload_events:
            # Annotate the existing buffered read instead of duplicating it:
            # recorder byte/call totals and the original request order stay intact.
            event = stream.events[-1]
            assert event["op"] == "read" and event["offset"] == offset
            event.update(io_layer="payload", api="buffered_stream_read")
            self.reader.payload_events.append(
                dict(
                    event,
                    file=file,
                    start_ns=event["start_ns"] + stream.recorder.origin_ns,
                )
            )
        if len(data) != size:
            raise EOFError("Short gdtensor payload read")
        self.reader.payload_requests.append({"file": file, "file_offset": offset, "length": size})
        return data

    def get_file_offset(self, pos):
        return Ok(
            next(
                s.file_offset
                for s in self.slots
                if s.name == pos.tensor_id and s.column == pos.rank
            )
        )

    def init_data_accessor(self):
        return Ok(self.accessor)

    def finish(self):
        if getattr(self, "_finished", False):
            return Ok(None)
        self._finished = True
        # Preserve every span even when its saved slot's file was not opened on
        # this rank (or a callback crossed saved slots). Trace publication must
        # not open another file or change the executor's read sequence. Events
        # retain their own file field when carried on a different open stream.
        fallback = next(iter(self.streams), None)
        by_file = defaultdict(list)
        for event in self.reader.staging_events:
            file = event["file"] if event["file"] in self.streams else fallback
            if file is not None:
                by_file[file].append(event)
        for file, events in by_file.items():
            _publish_payload_trace(
                self.streams[file], [("staging", events)],
            )
        self.stack.close()
        return Ok(None)


class _GDTensorReaderCore(FileSystemReader):
    """MRO hook inside the existing admission/pruning wrapper."""

    def read_metadata(self):
        metadata = super().read_metadata()
        # This executes inside AdmittedFileSystemReader's metadata gate, before
        # its unchanged pruning. Keep geometry/offset descriptors, not Metadata.
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
                        info.relative_path,
                        math.prod(chunk.sizes) * element_size(md.properties.dtype),
                        chunk.offsets[0] * row_bytes,
                        container=(info, tuple(chunk.sizes)),
                    )
                )
        return metadata

    def read_data(self, plan, planner):
        if not isinstance(planner, GDTensorLoadPlanner):
            raise ValueError("gdtensor reader requires GDTensorLoadPlanner")
        filesystem = self.fs
        with ExitStack() as stack:
            self.fs = _ReadFileSystem(filesystem, stack)
            try:
                return self._read_data(plan, planner)
            finally:
                self.fs = filesystem

    def _read_data(self, plan, planner):
        self.capture_payload_events = getattr(self.fs, "recorder", None) is not None
        tensors = {
            name: value for name, value in planner.state_dict.items() if isinstance(value, DTensor)
        }
        # Native BYTE_IO and plain tensors use exactly the native reader.
        native = dataclasses.replace(
            plan,
            items=[
                i
                for i in plan.items
                if i.type == LoadItemType.BYTE_IO or i.dest_index.fqn not in tensors
            ],
        )
        super().read_data(native, planner).wait()
        slots = [s for s in self.saved_slots if s.name in tensors]
        geometries = {name: self.saved_geometries[name] for name in tensors}
        needed = {
            (i.storage_index.fqn, tuple(i.storage_index.offset))
            for i in plan.items
            if i.type != LoadItemType.BYTE_IO and i.dest_index.fqn in tensors
        }
        for slot in slots:
            info, shape = slot.container
            offset = (
                slot.global_offset
                // (
                    math.prod(geometries[slot.name].shape[1:])
                    * element_size(geometries[slot.name].dtype)
                )
                if slot.size
                else geometries[slot.name].shape[0]
            )
            key = (slot.name, (offset,) + (0,) * (len(shape) - 1))
            if key in needed:
                if info.transform_descriptors:
                    raise ValueError("gdtensor reader does not support transformed containers")
                path = self.fs.concat_path(self.path, slot.file)
                with self.fs.create_stream(path, "rb") as stream:
                    slot.file_offset = read_container(
                        stream, info.offset, info.length, shape, geometries[slot.name].dtype
                    )
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
        session = _LoadSession(self, slots)
        try:
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


class GDTensorFileSystemReader(AdmittedFileSystemReader, _GDTensorReaderCore):
    """Native metadata/bytes and admitted gdtensor payload range reads."""

    buffer_bytes = 2 * 65536

    def __init__(self, path, **kwargs):
        super().__init__(path, **kwargs)
        self.payload_requests = []
        self.capture_payload_events = False
        self.payload_events = []
        self.staging_events = []
