"""Pageable whole-range and optional pinned staging with real CUDA storage."""

import ctypes
import json
import os
import resource
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from gdtensor.builder import Ok  # noqa: E402

from metasl import gdt_storage  # noqa: E402
from metasl.fsdp import materialize  # noqa: E402
from metasl.gdt_storage import (  # noqa: E402
    ComputeAccessor,
    GDTensorFileSystemReader,
    GDTensorFileSystemWriter,
    GDTensorLoadPlanner,
    GDTensorSavePlanner,
)
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.reference import checkpoint_mesh  # noqa: E402


@pytest.fixture
def local_mesh():
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("fake", rank=0, world_size=1)
    elif dist.get_world_size() != 1:
        pytest.skip("Local CUDA regressions require a one-rank process group")
    try:
        yield dist.device_mesh.init_device_mesh("cpu", (1,))
    finally:
        if owns_group:
            dist.destroy_process_group()


def distributed(local, mesh):
    if local.device.type != mesh.device_type:
        mesh = checkpoint_mesh(mesh, local.device)
    result = DTensor.from_local(
        local, mesh, [Shard(0)], shape=local.shape, stride=local.stride(), run_check=False
    )
    assert result.to_local().device == local.device
    assert result.to_local().data_ptr() == local.data_ptr()
    return result


def raw_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("start,count", [(3, 25), (0, 35)])
@pytest.mark.parametrize("staging_mode,staging_bytes", [(None, None), ("pinned", None), (None, 19)])
def test_cuda_staging_preserves_bytes_and_range_in_both_directions(
    local_mesh, dtype, start, count, staging_mode, staging_bytes
):
    torch.cuda.set_device(0)
    local = torch.arange(35).to(dtype).reshape(7, 5).cuda()
    accessor = ComputeAccessor(
        {"weight": distributed(local, local_mesh)},
        0,
        buffer_bytes=19,
        cuda_staging_bytes=staging_bytes,
        cuda_staging_mode=staging_mode,
    )
    itemsize = local.element_size()
    rng = SimpleNamespace(
        global_tensor_id="weight", global_tensor_offset=start * itemsize, size=count * itemsize
    )
    chunks, read_pointers, write_pointers = [], [], []
    block_size = rng.size if staging_bytes is None else staging_bytes // itemsize * itemsize

    def take(pointer, length):
        read_pointers.append(pointer)
        assert 0 < length <= block_size
        assert length % itemsize == 0
        host = torch.frombuffer((ctypes.c_ubyte * length).from_address(pointer), dtype=torch.uint8)
        assert host.is_pinned() == (accessor.cuda_staging_mode != "pageable")
        chunks.append(ctypes.string_at(pointer, length))
        return Ok(None)

    assert type(accessor.read_to_buffer(rng, take)) is Ok
    if accessor.cuda_staging_mode == "pageable":
        assert len(accessor._cuda_scratch) == rng.size
        assert accessor._cuda_buffers == []
    else:
        assert accessor._cuda_buffers[0].numel() == (
            rng.size if staging_bytes is None else min(local.nbytes, staging_bytes)
        )
    expected = raw_bytes(local.reshape(-1)[start : start + count].cpu())
    assert b"".join(chunks) == expected
    assert len(chunks) == (rng.size + block_size - 1) // block_size
    assert len(set(read_pointers)) == (1 if staging_bytes is None else 2)
    local.reshape(-1)[start : start + count].fill_(-1)
    before = local.clone()
    consumed = 0

    def give(pointer, length):
        nonlocal consumed
        write_pointers.append(pointer)
        assert 0 < length <= block_size
        host = torch.frombuffer((ctypes.c_ubyte * length).from_address(pointer), dtype=torch.uint8)
        assert host.is_pinned() == (accessor.cuda_staging_mode != "pageable")
        ctypes.memmove(pointer, expected[consumed : consumed + length], length)
        consumed += length
        return Ok(None)

    assert type(accessor.write_from_buffer(rng, give)) is Ok
    assert consumed == len(expected)
    assert len(write_pointers) == len(read_pointers)
    assert len(set(write_pointers)) == (1 if staging_bytes is None else 2)
    assert raw_bytes(local.reshape(-1)[start : start + count].cpu()) == expected
    assert torch.equal(local.reshape(-1)[:start], before.reshape(-1)[:start])
    assert torch.equal(local.reshape(-1)[start + count :], before.reshape(-1)[start + count :])
    assert accessor.requests == [("weight", start * itemsize, count * itemsize)] * 2


def resident_pages(pointer, size):
    page_size = os.sysconf("SC_PAGESIZE")
    pages = (size + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    if libc.mincore(pointer, size, vector):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return sum(value & 1 for value in vector), pages


@pytest.mark.skipif(
    not torch.cuda.is_available() or sys.platform != "linux",
    reason="requires real CUDA and Linux MADV_DONTNEED",
)
@pytest.mark.parametrize("write", [False, True])
def test_cuda_pageable_staging_faults_and_discards_each_range(local_mesh, write):
    local = torch.full((2**22,), 3, dtype=torch.uint8, device="cuda")
    accessor = ComputeAccessor({"weight": distributed(local, local_mesh)}, 0)
    assert accessor.cuda_staging_mode == "pageable"
    previous = None
    pointers = []
    for count in (2**20, 2**18, 2**22, 2**20):
        local.fill_(3)
        rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=count)
        calls = []

        def callback(pointer, length):
            calls.append((pointer, length))
            assert length == count
            resident, pages = resident_pages(pointer, length)
            assert resident == (0 if write else pages)
            if write:
                ctypes.memset(pointer, 7, length)
            else:
                assert ctypes.string_at(pointer, length) == b"\x03" * length
            assert resident_pages(pointer, length) == (pages, pages)
            return Ok(None)

        before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
        operation = accessor.write_from_buffer if write else accessor.read_to_buffer
        assert type(operation(rng, callback)) is Ok
        assert resource.getrusage(resource.RUSAGE_SELF).ru_minflt > before
        assert len(calls) == 1
        pointer, length = calls[0]
        pointers.append(pointer)
        assert resident_pages(pointer, len(accessor._cuda_scratch))[0] == 0
        assert raw_bytes(local[:count].cpu()) == bytes([7 if write else 3]) * count
        if previous is not None and previous is not accessor._cuda_scratch:
            assert previous.closed
        previous = accessor._cuda_scratch
    assert pointers[0] == pointers[1]
    assert pointers[2] == pointers[3]
    assert len(accessor._cuda_scratch) == 2**22
    assert accessor._cuda_buffers == []


@pytest.mark.skipif(
    not torch.cuda.is_available() or sys.platform != "linux",
    reason="requires real CUDA and Linux MADV_DONTNEED",
)
@pytest.mark.parametrize("write", [False, True])
@pytest.mark.parametrize("failure", ["exception", "result"])
def test_cuda_pageable_staging_discards_when_callback_fails(local_mesh, write, failure):
    local = torch.full((2**20,), 3, dtype=torch.uint8, device="cuda")
    accessor = ComputeAccessor({"weight": distributed(local, local_mesh)}, 0)
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=local.nbytes)

    def fail(pointer, length):
        ctypes.memset(pointer, 7, length)
        if failure == "exception":
            raise RuntimeError("injected callback error")
        return None

    operation = accessor.write_from_buffer if write else accessor.read_to_buffer
    with pytest.raises((RuntimeError, ValueError), match="injected|gdtensor operation failed"):
        operation(rng, fail)
    assert resident_pages(accessor._cuda_scratch_address, rng.size)[0] == 0
    assert raw_bytes(local.cpu()) == b"\x03" * rng.size


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("staging_bytes", [None, 16])
def test_cuda_staging_checks_callback_errors_before_copying(local_mesh, staging_bytes):
    local = torch.arange(8, dtype=torch.float32, device="cuda")
    accessor = ComputeAccessor(
        {"weight": distributed(local, local_mesh)}, 0, cuda_staging_bytes=staging_bytes
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=32)
    before = local.clone()
    with pytest.raises(ValueError, match="gdtensor operation failed"):
        accessor.write_from_buffer(rng, lambda pointer, length: None)
    assert torch.equal(local, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("staging_bytes", [None, 19])
def test_cuda_staging_uses_global_shard_offset_and_reuses_capacity(
    local_mesh, staging_bytes, monkeypatch
):
    # Retain real DTensor/CUDA storage and inject the two-rank geometry so a
    # nonzero rank can be checked without launching another GPU process.
    local = torch.arange(35, dtype=torch.float32, device="cuda").reshape(7, 5)
    state = {"weight": distributed(local, local_mesh)}
    monkeypatch.setattr(
        gdt_storage,
        "_compute_geometry",
        lambda tensor: gdt_storage.TensorGeometry((14, 5), tensor.dtype, 2),
    )
    accessor = ComputeAccessor(state, 1, cuda_staging_bytes=staging_bytes)
    pointers = []
    for start, count in ((3, 25), (4, 9), (3, 25)):
        rng = SimpleNamespace(
            global_tensor_id="weight", global_tensor_offset=(35 + start) * 4, size=count * 4
        )
        chunks, current_pointers = [], []

        def take(pointer, length, current_pointers=current_pointers, chunks=chunks):
            current_pointers.append(pointer)
            chunks.append(ctypes.string_at(pointer, length))
            return Ok(None)

        assert type(accessor.read_to_buffer(rng, take)) is Ok
        expected = raw_bytes(local.reshape(-1)[start : start + count].cpu())
        assert b"".join(chunks) == expected
        pointers.append(current_pointers[0])
        local.reshape(-1)[start : start + count].fill_(-1)
        before = local.clone()
        consumed = 0

        def give(pointer, length, expected=expected):
            nonlocal consumed
            ctypes.memmove(pointer, expected[consumed : consumed + length], length)
            consumed += length
            return Ok(None)

        assert type(accessor.write_from_buffer(rng, give)) is Ok
        assert consumed == rng.size
        assert raw_bytes(local.reshape(-1)[start : start + count].cpu()) == expected
        assert torch.equal(local.reshape(-1)[:start], before.reshape(-1)[:start])
        assert torch.equal(local.reshape(-1)[start + count :], before.reshape(-1)[start + count :])
    assert len(set(pointers)) == 1


@pytest.mark.parametrize("staging_bytes", [0, 1, 7, -1])
def test_cuda_staging_rejects_buffer_smaller_than_tensor_element(local_mesh, staging_bytes):
    state = {"weight": distributed(torch.zeros(3), local_mesh)}
    with pytest.raises(ValueError, match="staging|buffer|element"):
        ComputeAccessor(state, 0, cuda_staging_bytes=staging_bytes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("dtype", [torch.uint8, torch.bool, torch.float32])
@pytest.mark.parametrize("staging_bytes", [None, 8])
def test_cuda_staging_accepts_single_element_tensors(local_mesh, dtype, staging_bytes):
    local = torch.ones(1, dtype=dtype, device="cuda")
    accessor = ComputeAccessor(
        {"weight": distributed(local, local_mesh)}, 0, cuda_staging_bytes=staging_bytes
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=local.nbytes)
    expected, calls = raw_bytes(local.cpu()), []

    def take(pointer, length):
        calls.append(length)
        assert ctypes.string_at(pointer, length) == expected
        return Ok(None)

    assert type(accessor.read_to_buffer(rng, take)) is Ok
    local.zero_()

    def give(pointer, length):
        calls.append(length)
        ctypes.memmove(pointer, expected, length)
        return Ok(None)

    assert type(accessor.write_from_buffer(rng, give)) is Ok
    assert calls == [local.nbytes, local.nbytes]
    assert raw_bytes(local.cpu()) == expected


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("staging_bytes", [None, 65536])
def test_cuda_staging_obeys_nondefault_current_stream(local_mesh, staging_bytes):
    local = torch.empty(2**18, dtype=torch.float32, device="cuda")
    accessor = ComputeAccessor(
        {"weight": distributed(local, local_mesh)}, 0, cuda_staging_bytes=staging_bytes
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=local.nbytes)
    saved = raw_bytes(torch.full(local.shape, 3.5))
    restored = raw_bytes(torch.full(local.shape, -7.25))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    chunks, consumed = [], 0

    def take(pointer, length):
        chunks.append(ctypes.string_at(pointer, length))
        return Ok(None)

    def give(pointer, length):
        nonlocal consumed
        ctypes.memmove(pointer, restored[consumed : consumed + length], length)
        consumed += length
        return Ok(None)

    with torch.cuda.stream(stream):
        torch.cuda._sleep(10_000_000)
        local.fill_(3.5)
        assert type(accessor.read_to_buffer(rng, take)) is Ok
        assert b"".join(chunks) == saved
        assert type(accessor.write_from_buffer(rng, give)) is Ok
        assert stream.query()
    assert consumed == rng.size
    assert raw_bytes(local.cpu()) == restored


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("write", [False, True])
def test_cuda_bounded_staging_drains_pending_copies_when_callback_raises(local_mesh, write):
    local = torch.zeros(2**20, dtype=torch.uint8, device="cuda")
    accessor = ComputeAccessor(
        {"weight": distributed(local, local_mesh)}, 0, cuda_staging_bytes=2**18
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=local.nbytes)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    calls = 0

    def fail_second(pointer, length):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected asynchronous callback error")
        # Leave a copy pending on the current stream when callback 2 fails.
        torch.cuda._sleep(10_000_000)
        if write:
            ctypes.memset(pointer, 7, length)
        return Ok(None)

    with torch.cuda.stream(stream):
        operation = accessor.write_from_buffer if write else accessor.read_to_buffer
        with pytest.raises(RuntimeError, match="injected asynchronous callback error"):
            operation(rng, fail_second)
        assert calls == 2
        assert stream.query()
    actual = local.cpu()
    assert torch.count_nonzero(actual[2**18 :]).item() == 0
    assert torch.all(actual[: 2**18] == (7 if write else 0)).item()


def test_positioned_payload_events_capture_each_short_write_and_failure(tmp_path, monkeypatch):
    events = []
    original_pwrite = gdt_storage.os.pwrite
    with (tmp_path / "payload").open("w+b") as stream:
        session = gdt_storage._SaveSession(stream, [], False, payload_events=events)

        def short_write(fd, data, offset):
            return original_pwrite(fd, data[:3], offset)

        monkeypatch.setattr(gdt_storage.os, "pwrite", short_write)
        session.write("__0_0.distcp", 5, memoryview(b"abcdefgh"))
        stream.seek(5)
        assert stream.read() == b"abcdefgh"
        assert [(e["offset"], e["requested"], e["completed"]) for e in events] == [
            (5, 8, 3),
            (8, 5, 3),
            (11, 2, 2),
        ]
        assert all(e["op"] == "pwrite" and e["success"] for e in events)
        assert all(e["start_ns"] > 0 and e["duration_ns"] >= 0 for e in events)
        assert {e["file"] for e in events} == {"__0_0.distcp"}

        def fail_write(fd, data, offset):
            raise OSError("injected positioned write failure")

        monkeypatch.setattr(gdt_storage.os, "pwrite", fail_write)
        with pytest.raises(OSError, match="injected"):
            session.write("__0_0.distcp", 23, b"bad")
        assert events[-1]["offset"] == 23
        assert events[-1]["requested"] == 3 and events[-1]["completed"] == 0
        assert not events[-1]["success"]


def test_positioned_container_events_capture_short_writes_and_failure(tmp_path, monkeypatch):
    events = []
    original_pwrite = gdt_storage.os.pwrite
    with (tmp_path / "container").open("w+b") as stream:

        def short_write(fd, data, offset):
            return original_pwrite(fd, data[:3], offset)

        monkeypatch.setattr(gdt_storage.os, "pwrite", short_write)
        gdt_storage._container_pwrite(
            stream, 5, b"abcdefgh", write_events=events, file="__0_0.distcp"
        )
        stream.seek(5)
        assert stream.read() == b"abcdefgh"
        assert [(e["offset"], e["requested"], e["completed"]) for e in events] == [
            (5, 8, 3),
            (8, 5, 3),
            (11, 2, 2),
        ]
        assert all(e["op"] == "pwrite" and e["success"] for e in events)
        assert all(e["start_ns"] > 0 and e["duration_ns"] >= 0 for e in events)
        assert {e["file"] for e in events} == {"__0_0.distcp"}

        def fail_write(fd, data, offset):
            raise OSError("injected container patch failure")

        monkeypatch.setattr(gdt_storage.os, "pwrite", fail_write)
        with pytest.raises(OSError, match="injected"):
            gdt_storage._container_pwrite(
                stream, 23, b"bad", write_events=events, file="__0_0.distcp"
            )
        assert events[-1]["offset"] == 23
        assert events[-1]["requested"] == 3 and events[-1]["completed"] == 0
        assert not events[-1]["success"]


def save_local(
    path,
    state,
    recorder=None,
    *,
    sync_files=False,
    capture_lifecycle=False,
    capture_payload_events=True,
):
    writer = GDTensorFileSystemWriter(path, thread_count=1, sync_files=sync_files)
    if recorder is not None:
        writer.fs = RecordedFileSystem(writer.fs, recorder, capture_lifecycle=capture_lifecycle)
    writer.capture_payload_events = recorder is not None and capture_payload_events
    planner = GDTensorSavePlanner()
    writer.set_up_storage_writer(is_coordinator=True, rank=0)
    planner.set_up_planner(state, is_coordinator=True)
    local = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local])
    plan = planner.finish_plan(writer.prepare_global_plan(plans)[0])
    results = writer.write_data(plan, planner).wait()
    writer.finish(metadata, [results])
    return writer


def load_local(path, state):
    reader = GDTensorFileSystemReader(path)
    reader.fs = RecordedFileSystem(reader.fs, IORecorder())
    metadata = reader.read_metadata()
    reader.set_up_storage_reader(metadata, is_coordinator=True, rank=0)
    planner = GDTensorLoadPlanner()
    planner.set_up_planner(state, metadata, is_coordinator=True)
    local = reader.prepare_local_plan(planner.create_local_plan())
    plans = reader.prepare_global_plan(planner.create_global_plan([local]))
    reader.read_data(planner.finish_plan(plans[0]), planner).wait()
    planner.assert_complete()
    return reader


def test_writer_exposes_actual_payload_events_only_when_enabled(tmp_path, local_mesh):
    state = {"weight": distributed(torch.arange(100003, dtype=torch.float32), local_mesh)}
    traced = save_local(tmp_path / "traced", state, IORecorder())
    assert len(traced.payload_events) == len(traced.payload_requests)
    assert sum(e["completed"] for e in traced.payload_events) == 100003 * 4
    assert all(e["success"] for e in traced.payload_events)
    assert len(traced.container_patch_events) == 5
    assert sum(e["completed"] for e in traced.container_patch_events) == 56
    assert all(e["op"] == "pwrite" and e["success"] for e in traced.container_patch_events)
    plain = save_local(tmp_path / "plain", state)
    assert plain.payload_events == []
    assert plain.container_patch_events == []
    assert (tmp_path / "traced" / "__0_0.distcp").read_bytes() == (
        tmp_path / "plain" / "__0_0.distcp"
    ).read_bytes()


@pytest.mark.parametrize("device", ["cpu", "meta", "cuda"])
def test_save_payload_and_staging_share_recorder_clock_and_preserve_old_requests(
    tmp_path,
    local_mesh,
    device,
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires a real CUDA device")
    from metasl.trace_export import export_rank_trace

    state = {"weight": distributed(torch.zeros(100003, device=device), local_mesh)}
    recorder = IORecorder()
    traced = save_local(tmp_path / "traced", state, recorder, capture_lifecycle=True)
    trace = tmp_path / "rank.io.jsonl"
    recorder.summary(trace)
    events = recorder.files["__0_0.distcp"]["events"]
    payload = [e for e in events if e.get("io_layer") == "payload"]
    assert len(payload) == len(traced.payload_events)
    for relative, absolute in zip(payload, traced.payload_events, strict=True):
        assert relative["start_ns"] + recorder.origin_ns == absolute["start_ns"]
        assert relative["duration_ns"] == absolute["duration_ns"]
    stage = [e for e in events if e.get("io_layer") == "staging"]
    assert {"host_materialize", "host_copy"} <= {e["op"] for e in stage}
    assert {"cpu": "cpu_view", "meta": "meta_generate", "cuda": "d2h"}[device] in {
        e["op"] for e in stage
    }
    assert all(e["end_ns"] == e["start_ns"] + e["duration_ns"] for e in stage)
    assert sum(e["completed"] for e in stage if e["op"] == "host_copy") == 400012
    if device == "cuda":
        d2h = [e for e in stage if e["op"] == "d2h"]
        assert sum(e["completed"] for e in d2h) == 400012
        assert max(e["end_ns"] for e in d2h) <= min(e["start_ns"] for e in payload)

    # Disable only the new instrumentation; the original stream proxy remains.
    original_recorder = IORecorder()
    untraced = save_local(
        tmp_path / "old",
        state,
        original_recorder,
        capture_lifecycle=True,
        capture_payload_events=False,
    )
    old = original_recorder.files["__0_0.distcp"]["events"]
    fields = ("op", "offset", "requested", "completed", "whence")
    assert [tuple(e.get(k) for k in fields) for e in events if "io_layer" not in e] == [
        tuple(e.get(k) for k in fields) for e in old
    ]
    assert untraced.staging_events == []
    exported = export_rank_trace(
        trace,
        tmp_path / "export",
        rank=0,
        phase="save",
        origin_ns=recorder.origin_ns,
        payload_events=traced.payload_events + traced.container_patch_events,
    )
    rows = [json.loads(line) for line in open(exported["output_path"])]
    assert sum(e["op"] == "pwrite" for e in rows) == (
        len(traced.payload_events) + len(traced.container_patch_events)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("mode", ["pageable", "pinned", "double-buffer"])
@pytest.mark.parametrize("write", [False, True])
def test_d2h_tracing_preserves_staging_modes_without_global_sync(
    local_mesh, monkeypatch, mode, write
):
    local = torch.arange(16384, dtype=torch.float32, device="cuda")
    state = {"weight": distributed(local, local_mesh)}
    expected = raw_bytes(local.cpu())
    events, chunks = [], []
    accessor = ComputeAccessor(
        state,
        0,
        cuda_staging_mode=mode,
        cuda_staging_bytes=4096 if mode == "double-buffer" else None,
        staging_events=events,
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=local.nbytes)

    def forbidden_sync(*args, **kwargs):
        raise AssertionError("Tracing must not add a global CUDA synchronize")

    monkeypatch.setattr(torch.cuda, "synchronize", forbidden_sync)

    def take(pointer, length):
        chunks.append(ctypes.string_at(pointer, length))
        return Ok(None)

    if write:
        local.zero_()
        consumed = 0

        def give(pointer, length):
            nonlocal consumed
            ctypes.memmove(pointer, expected[consumed : consumed + length], length)
            consumed += length
            return Ok(None)

        assert type(accessor.write_from_buffer(rng, give)) is Ok
        assert raw_bytes(local.cpu()) == expected
    else:
        assert type(accessor.read_to_buffer(rng, take)) is Ok
        assert b"".join(chunks) == expected
    copies = [event for event in events if event["op"] == ("h2d" if write else "d2h")]
    assert len(copies) == (16 if mode == "double-buffer" else 1)
    assert sum(event["completed"] for event in copies) == local.nbytes
    assert all(event["duration_ns"] > 0 for event in copies)
    if mode == "double-buffer":
        assert all(event["cuda_duration_ns"] > 0 for event in copies)
        assert all(
            event["cuda_end_relative_ns"] > event["cuda_start_relative_ns"] for event in copies
        )
        waits = [e for e in events if e["op"] == "event_wait"]
        assert len(waits) == len(copies)
        reasons = {e["reason"] for e in waits}
        assert ("buffer_reuse" if write else "first_chunk") in reasons
    if mode != "pageable":
        alloc = [e for e in events if e["op"] == "pinned_alloc"]
        assert len(alloc) == 1
        assert alloc[0]["buffers"] == (2 if mode == "double-buffer" else 1)
    assert torch.cuda.current_stream().query()


def test_writer_records_data_and_native_metadata_sync_intervals(tmp_path, local_mesh, monkeypatch):
    from torch.distributed.checkpoint import FileSystemWriter

    state = {"weight": distributed(torch.arange(19, dtype=torch.float32), local_mesh)}
    recorder = IORecorder()
    captured = []
    traced_finish = GDTensorFileSystemWriter.finish

    def finish(writer, metadata, results):
        captured.append(metadata)
        return traced_finish(writer, metadata, results)

    monkeypatch.setattr(GDTensorFileSystemWriter, "finish", finish)
    writer = save_local(tmp_path, state, recorder, sync_files=True, capture_lifecycle=True)
    recorder.summary()
    for name in ("__0_0.distcp", ".metadata.tmp"):
        events = recorder.files[name]["events"]
        syncs = [event for event in events if event["op"] == "fsync"]
        assert len(syncs) == 1
        assert syncs[0]["request"] is False and syncs[0]["success"] == 1
        assert events[-1]["op"] == "close"
        assert events[-1]["start_ns"] >= syncs[0]["start_ns"] + syncs[0]["duration_ns"]
        assert any(event["op"] == "flush" for event in events)
    # The explicit metadata hook must serialize exactly what torch's original
    # finish writes for the same writer/save ID and completed results.
    writer.fs = writer.fs.fs
    original_metadata = (tmp_path / ".metadata").read_bytes()
    FileSystemWriter.finish(writer, captured[0], [writer.results])
    assert (tmp_path / ".metadata").read_bytes() == original_metadata


def test_size_ordered_writer_aligns_session_and_view_offsets(tmp_path, local_mesh, monkeypatch):
    expected = {"large": torch.arange(64, dtype=torch.float32), "small": torch.arange(3.0)}
    state = {name: distributed(value.clone(), local_mesh) for name, value in expected.items()}
    # Force the accelerator-dependent ordering on CPU tensors to exercise the
    # packed-offset mapping without requiring GPU storage or initialization.
    with monkeypatch.context() as order_patch:
        order_patch.setattr(torch.cuda, "is_available", lambda: True)
        save_local(tmp_path / "ordered", state, IORecorder())
    metadata = GDTensorFileSystemReader(tmp_path / "ordered").read_metadata()
    offsets = {index.fqn: info.offset for index, info in metadata.storage_data.items()}
    assert offsets["small"] < offsets["large"]
    for tensor in state.values():
        tensor.to_local().fill_(-1)
    load_local(tmp_path / "ordered", state)
    assert all(torch.equal(state[name].to_local(), value) for name, value in expected.items())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_real_cuda_gdtensor_files_match_cpu_and_restore_real_storage(tmp_path, local_mesh, dtype):
    torch.cuda.set_device(0)
    expected = torch.arange(100003).to(dtype).reshape(1, -1)
    cpu = {"weight": distributed(expected.clone(), local_mesh)}
    cuda = materialize(cpu, device="cuda:0")
    cuda["weight"].to_local().copy_(expected)
    assert cuda["weight"].to_local().is_cuda
    save_local(tmp_path / "cpu", cpu, IORecorder())
    save_local(tmp_path / "cuda", cuda, IORecorder())
    assert (tmp_path / "cpu" / "__0_0.distcp").read_bytes() == (
        tmp_path / "cuda" / "__0_0.distcp"
    ).read_bytes()
    for path in (tmp_path / "cpu", tmp_path / "cuda"):
        cuda["weight"].to_local().fill_(-1)
        load_local(path, cuda)
        assert cuda["weight"].to_local().is_cuda
        assert torch.equal(cuda["weight"].to_local().cpu(), expected)
    cpu["weight"].to_local().fill_(-1)
    load_local(tmp_path / "cuda", cpu)
    assert torch.equal(cpu["weight"].to_local(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a real CUDA device")
@pytest.mark.parametrize("mode", ["pageable", "pinned", "double-buffer"])
def test_load_publishes_payload_and_h2d_trace(tmp_path, local_mesh, monkeypatch, mode):
    monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_MODE", mode)
    if mode == "double-buffer":
        monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_BYTES", "4096")
    expected = torch.arange(100003, dtype=torch.float32)
    state = {"weight": distributed(expected.clone(), local_mesh)}
    save_local(tmp_path, state)
    destination = torch.zeros_like(expected, device="cuda")
    reader = load_local(tmp_path, {"weight": distributed(destination, local_mesh)})
    assert torch.equal(destination.cpu(), expected)
    assert sum(e["completed"] for e in reader.payload_events) == expected.nbytes
    assert sum(e["completed"] for e in reader.staging_events if e["op"] == "h2d") == expected.nbytes
    recorder = reader.fs.recorder
    raw = recorder.files["__0_0.distcp"]["events"]
    payload = [e for e in raw if e.get("io_layer") == "payload"]
    stages = [e for e in raw if e.get("io_layer") == "staging"]
    assert len(payload) == len(reader.payload_events)
    assert len(stages) == len(reader.staging_events)
    assert all(e["op"] == "read" and e["api"] == "buffered_stream_read" for e in payload)
    assert all(0 <= e["start_ns"] <= e["start_ns"] + e["duration_ns"] for e in raw)
