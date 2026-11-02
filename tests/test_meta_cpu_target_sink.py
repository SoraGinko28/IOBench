"""Resident Meta CPU-target destinations preserve IO ranges and local geometry."""

import ctypes
import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from gdtensor.builder import Ok  # noqa: E402

from metasl import gdt_storage  # noqa: E402
from metasl.gdt_hf_storage import (  # noqa: E402
    GDTensorHuggingFaceStorageReader,
    GDTensorHuggingFaceStorageWriter,
)
from metasl.gdt_shared_hf_storage import (  # noqa: E402
    SharedGDTensorHuggingFaceStorageReader,
    SharedGDTensorHuggingFaceStorageWriter,
)
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.sharding import fill_or_check, manifest_row  # noqa: E402


@pytest.fixture(autouse=True)
def sink_environment(monkeypatch):
    monkeypatch.delenv("METASL_GDTENSOR_META_LOAD_SINK", raising=False)
    monkeypatch.delenv("METASL_GDTENSOR_META_LOAD_SINK_BYTES", raising=False)
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "off")


def resident_pages(pointer, size):
    page = os.sysconf("SC_PAGESIZE")
    start = pointer // page * page
    count = (pointer - start + size + page - 1) // page
    vector = (ctypes.c_ubyte * count)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    assert libc.mincore(start, pointer - start + size, vector) == 0
    return sum(value & 1 for value in vector), count


def make_accessor(monkeypatch, sink):
    # Rank 1 owns the second half, so offsets must be local, not global.
    locals_ = {
        "weight": torch.empty(2**14, dtype=torch.float32, device="meta"),
        "bias": torch.empty(2**12, dtype=torch.float32, device="meta"),
    }
    monkeypatch.setattr(
        gdt_storage,
        "_compute_geometry",
        lambda tensor: gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 2),
    )
    tensors = {
        name: SimpleNamespace(
            shape=(local.numel() * 2,), dtype=local.dtype, to_local=lambda local=local: local
        )
        for name, local in locals_.items()
    }
    return gdt_storage.ComputeAccessor(tensors, 1, meta_load_sink=sink), locals_


def rng_for(name, tensor, offset, size):
    return SimpleNamespace(
        global_tensor_id=name, global_tensor_offset=tensor.nbytes + offset, size=size
    )


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux mincore")
def test_faithful_sink_prefaults_and_preserves_each_local_tensor_offset(monkeypatch):
    sizes = {"weight": 2**16, "bias": 2**14}
    sink = gdt_storage.MetaLoadSink(sizes)
    assert sink.capacity == sum(sizes.values())
    assert (
        resident_pages(sink.address, sink.capacity)[0]
        == resident_pages(sink.address, sink.capacity)[1]
    )
    accessor, locals_ = make_accessor(monkeypatch, sink)
    requests = [("weight", 4096, 8192, 19), ("bias", 256, 4096, 37)]
    for name, offset, size, value in requests:
        rng = rng_for(name, locals_[name], offset, size)

        def give(pointer, length):
            assert pointer == sink.address + sink.tensor_offsets[name] + offset
            assert length == size
            ctypes.memset(pointer, value, length)
            return Ok(None)

        assert isinstance(accessor.write_from_buffer(rng, give), Ok)
    for name, offset, size, value in requests:
        tensor_start = sink.address + sink.tensor_offsets[name]
        assert ctypes.string_at(tensor_start, offset) == bytes(offset)
        assert ctypes.string_at(tensor_start + offset, size) == bytes([value]) * size
        after = sizes[name] - offset - size
        assert ctypes.string_at(tensor_start + offset + size, after) == bytes(after)
    assert len(accessor.requests) == 2
    assert accessor._meta_sink is None
    assert sink.report()["prefaulted"] is True
    assert sink.report()["preparation_inside_load_call"] is False


@pytest.mark.parametrize("capacity", [1024, 1027, 2**17])
def test_ring_wraps_and_splits_callbacks_without_changing_range_requests(monkeypatch, capacity):
    sizes = {"weight": 2**16, "bias": 2**14}
    sink = gdt_storage.MetaLoadSink(sizes, mode="ring", capacity=capacity)
    accessor, locals_ = make_accessor(monkeypatch, sink)
    pointers, delivered, expected_requests = [], [], []
    for name, offset, size in [("weight", 16, 600), ("bias", 128, 1000), ("weight", 0, 2**16)]:
        rng = rng_for(name, locals_[name], offset, size)
        expected_requests.append((name, rng.global_tensor_offset, size))
        delivered.clear()

        def give(pointer, length):
            assert sink.address <= pointer < pointer + length <= sink.address + sink.capacity
            assert (pointer - sink.address) % 4 == length % 4 == 0
            pointers.append(pointer)
            delivered.append(length)
            ctypes.memset(pointer, 0, length)
            return Ok(None)

        accessor.write_from_buffer(rng, give)
        assert sum(delivered) == size
    assert accessor.requests == expected_requests
    assert sink.capacity == min(capacity, sum(sizes.values()))
    assert accessor._meta_sink is None
    if capacity < 2**16:
        assert len(delivered) > 1
        assert len(set(pointers)) < len(pointers)


def test_resident_sink_keeps_written_pages_after_failed_callback(monkeypatch):
    sink = gdt_storage.MetaLoadSink({"weight": 2**16, "bias": 2**14})
    accessor, locals_ = make_accessor(monkeypatch, sink)

    def give(pointer, length):
        ctypes.memset(pointer, 7, length)
        raise RuntimeError("injected sink failure")

    with pytest.raises(RuntimeError, match="injected sink failure"):
        accessor.write_from_buffer(rng_for("weight", locals_["weight"], 0, 4096), give)
    assert ctypes.string_at(sink.address, 4096) == bytes([7]) * 4096


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux mincore")
def test_resident_source_reads_prefaulted_disjoint_local_offsets(monkeypatch):
    source = gdt_storage.MetaSaveSource({"weight": 2**16, "bias": 2**14})
    accessor, locals_ = make_accessor(monkeypatch, None)
    accessor.meta_save_source = source
    assert resident_pages(source.address, source.capacity) == (
        source.capacity // os.sysconf("SC_PAGESIZE"),
        source.capacity // os.sysconf("SC_PAGESIZE"),
    )
    pointers = []
    for name, offset, size in [("weight", 4096, 8192), ("bias", 256, 4096)]:
        ctypes.memset(source.address + source.tensor_offsets[name] + offset, 17, size)

        def take(pointer, length):
            pointers.append(pointer)
            assert pointer == source.address + source.tensor_offsets[name] + offset
            assert length == size
            assert ctypes.string_at(pointer, length) == bytes([17]) * size
            return Ok(None)

        accessor.read_to_buffer(rng_for(name, locals_[name], offset, size), take)
    assert pointers[1] - pointers[0] == source.tensor_offsets["bias"] + 256 - 4096
    assert len(accessor.requests) == 2
    assert accessor._meta_source is None
    assert source.report()["preparation_inside_save_call"] is False


@pytest.mark.parametrize("mode", ["faithful", "ring"])
def test_resident_source_requires_preparation(monkeypatch, mode):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", mode)
    planner = gdt_storage.GDTensorSavePlanner(simulated=True)
    with pytest.raises(ValueError, match="before starting the dcp.save timer"):
        planner.set_up_planner({})
    planner.prepare_save({})
    prepared = planner.meta_save_source
    planner.prepare_save({})
    assert planner.meta_save_source is prepared
    assert prepared.capacity == 0
    assert planner.meta_save_source_report["prefaulted"] is True
    assert planner.meta_save_source_report["mode"] == mode
    planner = gdt_storage.GDTensorSavePlanner(simulated=False)
    planner.prepare_save({})
    assert planner.meta_save_source is None
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    planner = gdt_storage.GDTensorSavePlanner(simulated=True)
    with pytest.raises(ValueError, match="require simulated D2H off"):
        planner.prepare_save({})


def test_fresh_save_retains_demand_zero_source(monkeypatch):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", "fresh")
    planner = gdt_storage.GDTensorSavePlanner(simulated=True)
    planner.prepare_save({})
    assert planner.meta_save_source is None
    assert planner.meta_save_source_report["mode"] == "demand-zero"


@pytest.mark.parametrize("capacity", [1024, 1027, 2**17])
def test_ring_source_reuses_prefaulted_mapping_for_large_ranges(monkeypatch, capacity):
    source = gdt_storage.MetaSaveSource(
        {"weight": 2**16, "bias": 2**14}, mode="ring", capacity=capacity
    )
    accessor, locals_ = make_accessor(monkeypatch, None)
    accessor.meta_save_source = source
    pointers, lengths = [], []
    expected_requests = []
    for name, offset, size in [("weight", 16, 600), ("bias", 128, 1000), ("weight", 0, 2**16)]:
        rng = rng_for(name, locals_[name], offset, size)
        expected_requests.append((name, rng.global_tensor_offset, size))
        lengths.clear()

        def take(pointer, length):
            assert source.address <= pointer < pointer + length <= source.address + source.capacity
            assert (pointer - source.address) % 4 == length % 4 == 0
            assert ctypes.string_at(pointer, length) == bytes(length)
            pointers.append(pointer)
            lengths.append(length)
            return Ok(None)

        accessor.read_to_buffer(rng, take)
        assert sum(lengths) == size
    assert accessor.requests == expected_requests
    assert accessor._meta_source is None
    assert source.capacity == min(capacity, source.local_shard_bytes)
    assert source.report()["prefaulted"] is True
    if capacity < 2**16:
        assert len(lengths) > 1
        assert len(set(pointers)) < len(pointers)


@pytest.mark.parametrize(
    "planner_type", [gdt_storage.GDTensorSavePlanner, gdt_storage.GDTensorLoadPlanner]
)
@pytest.mark.parametrize("configured", ["0", "7", "-1", "invalid"])
def test_ring_environment_rejects_invalid_capacity(monkeypatch, configured, planner_type):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", "ring")
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK_BYTES", configured)
    planner = planner_type(simulated=True)
    with pytest.raises(ValueError):
        (
            planner.prepare_save
            if planner_type is gdt_storage.GDTensorSavePlanner
            else planner.prepare_load
        )({})


@pytest.mark.parametrize(
    "planner_type", [gdt_storage.GDTensorSavePlanner, gdt_storage.GDTensorLoadPlanner]
)
def test_ring_environment_rejects_position_before_allocating(monkeypatch, planner_type):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", "ring")
    planner = planner_type(simulated=True, pattern="position")
    prepare = (
        planner.prepare_save
        if planner_type is gdt_storage.GDTensorSavePlanner
        else planner.prepare_load
    )
    with pytest.raises(ValueError, match="use faithful or fresh for position"):
        prepare({})


def test_direct_ring_accessor_rejects_position(monkeypatch):
    sink = gdt_storage.MetaLoadSink({}, mode="ring", capacity=64)
    with pytest.raises(ValueError, match="use faithful or fresh for position"):
        gdt_storage.ComputeAccessor({}, 0, meta_load_sink=sink, value_pattern="position")


def test_sink_environment_defaults_validation_and_preparation_guard(monkeypatch):
    planner = gdt_storage.GDTensorLoadPlanner(simulated=True)
    planner.prepare_load({})
    assert planner.meta_load_sink is None
    assert planner.meta_load_sink_report["mode"] == "fresh"
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", "typo")
    with pytest.raises(ValueError, match="Unknown Meta load sink mode"):
        gdt_storage.GDTensorLoadPlanner(simulated=True).prepare_load({})
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", "faithful")
    planner = gdt_storage.GDTensorLoadPlanner(simulated=True)
    with pytest.raises(ValueError, match="before starting the dcp.load timer"):
        planner.set_up_planner({})
    planner.prepare_load({})
    prepared = planner.meta_load_sink
    planner.prepare_load({})
    assert planner.meta_load_sink is prepared
    assert prepared.capacity == 0
    standard = gdt_storage.GDTensorLoadPlanner(simulated=False)
    standard.prepare_load({})
    assert standard.meta_load_sink is None


@pytest.mark.parametrize("format_name", ["dcp", "safetensors", "shared-safetensors"])
@pytest.mark.parametrize("pattern", ["zero", "position"])
def test_cpu_target_sinks_match_cpu_file_requests_and_validate_both_formats(
    tmp_path, process_group, monkeypatch, format_name, pattern
):
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))

    def state(device):
        tensors, rows = {}, []
        for name, shape, dtype in (
            ("weight", (17, 5), torch.float32),
            ("bias", (23,), torch.bfloat16),
        ):
            local = torch.empty(shape, dtype=dtype, device=device)
            tensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
            row = manifest_row(f"model.{name}", tensor, "model", [name], "cpu-sink-test")
            rows.append(row)
            if device == "cpu":
                fill_or_check(local, row["global_offsets"], row["global_shape"], name, pattern)
            tensors[name] = tensor
        return {"model": tensors}, rows

    writer_type, reader_type = {
        "dcp": (gdt_storage.GDTensorFileSystemWriter, gdt_storage.GDTensorFileSystemReader),
        "safetensors": (GDTensorHuggingFaceStorageWriter, GDTensorHuggingFaceStorageReader),
        "shared-safetensors": (
            SharedGDTensorHuggingFaceStorageWriter,
            SharedGDTensorHuggingFaceStorageReader,
        ),
    }[format_name]
    source, source_rows = state("cpu")
    path = tmp_path / "checkpoint"
    dcp.save(
        source,
        storage_writer=writer_type(path, sync_files=False),
        planner=gdt_storage.GDTensorSavePlanner(rows=source_rows, pattern=pattern),
    )
    sequences, payload_sequences = [], []
    for device, mode in [
        ("cpu", "fresh"),
        ("meta", "fresh"),
        ("meta", "faithful"),
        ("meta", "ring"),
    ]:
        if pattern == "position" and mode == "ring":
            continue
        monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", mode)
        # Force wrap and multipart handoff, including position checks.
        monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK_BYTES", "64")
        target, rows = state(device)
        options = {"rows": rows} if format_name != "dcp" else {}
        reader = reader_type(path, **options)
        reader.fs = RecordedFileSystem(reader.fs, IORecorder())
        planner = gdt_storage.GDTensorLoadPlanner(rows, simulated=device == "meta", pattern=pattern)
        planner.prepare_load(flatten_state_dict(target)[0])
        dcp.load(target, storage_reader=reader, planner=planner)
        planner.assert_complete()
        sequences.append(reader.requests)
        payload_sequences.append(reader.payload_requests)
        if mode in ("faithful", "ring"):
            assert planner.meta_load_sink.local_shard_bytes == sum(
                row["payload_bytes"] for row in rows
            )
    assert all(sequence == sequences[0] for sequence in sequences)
    assert all(sequence == payload_sequences[0] for sequence in payload_sequences)


@pytest.mark.parametrize("format_name", ["dcp", "safetensors", "shared-safetensors"])
@pytest.mark.parametrize(
    "mode,pattern", [("faithful", "zero"), ("faithful", "position"), ("ring", "zero")]
)
def test_cpu_target_source_matches_cpu_payloads_and_file_requests(
    tmp_path, process_group, monkeypatch, format_name, mode, pattern
):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", mode)
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK_BYTES", "64")
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    writer_type = {
        "dcp": gdt_storage.GDTensorFileSystemWriter,
        "safetensors": GDTensorHuggingFaceStorageWriter,
        "shared-safetensors": SharedGDTensorHuggingFaceStorageWriter,
    }[format_name]
    sequences, payloads = [], []
    for device in ["cpu", "meta"]:
        values, rows = {}, []
        for name, shape, dtype in [
            ("weight", (17, 5), torch.float32),
            ("bias", (23,), torch.bfloat16),
        ]:
            local = torch.empty(shape, dtype=dtype, device=device)
            tensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
            row = manifest_row(f"model.{name}", tensor, "model", [name], "cpu-source-test")
            if device == "cpu":
                fill_or_check(local, row["global_offsets"], row["global_shape"], name, pattern)
            values[name] = tensor
            rows.append(row)
        state = {"model": values}
        planner = gdt_storage.GDTensorSavePlanner(
            rows=rows, pattern=pattern, simulated=device == "meta"
        )
        planner.prepare_save(flatten_state_dict(state)[0])
        path = tmp_path / device
        writer = writer_type(path, sync_files=False)
        recorder = IORecorder()
        writer.fs = RecordedFileSystem(writer.fs, recorder)
        writer.capture_payload_events = True
        dcp.save(state, storage_writer=writer, planner=planner)
        sequences.append(writer.payload_requests)
        payloads.append({p.name: p.read_bytes() for p in path.iterdir() if p.name != ".metadata"})
        if device == "meta":
            assert planner.meta_save_source.local_shard_bytes == sum(
                row["payload_bytes"] for row in rows
            )
            assert planner.meta_save_source.capacity == (
                min(64, planner.meta_save_source.local_shard_bytes)
                if mode == "ring"
                else planner.meta_save_source.local_shard_bytes
            )
            assert all(event["op"] != "meta_generate" for event in writer.staging_events)
    assert sequences[0] == sequences[1]
    assert payloads[0] == payloads[1]
