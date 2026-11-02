"""Compute range hand-off, anonymous-page residency, and optional validation."""

import ctypes
import json
import os
import resource
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from gdtensor.builder import Ok  # noqa: E402

from metasl import gdt_storage  # noqa: E402
from metasl.sharding import pattern_blocks  # noqa: E402

POPULATIONS = ["memset", "memset-flush", "populate-write", "non-temporal", "populate-write-evict"]


@pytest.fixture
def accessor_factory(monkeypatch):
    # Isolate the accessor contract while retaining real torch storage. A rank-1
    # shard ensures global offsets are converted to local offsets correctly.
    monkeypatch.setattr(
        gdt_storage,
        "_compute_geometry",
        lambda tensor: gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 2),
    )

    def make(local, **options):
        shard = SimpleNamespace(
            shape=(local.shape[0] * 2, *local.shape[1:]),
            dtype=local.dtype,
            to_local=lambda: local,
        )
        return gdt_storage.ComputeAccessor({"weight": shard}, 1, **options)

    return make


def requested_range(local, start=0, count=None):
    itemsize = local.element_size()
    return SimpleNamespace(
        global_tensor_id="weight",
        global_tensor_offset=(local.numel() + start) * itemsize,
        size=(local.numel() - start if count is None else count) * itemsize,
    )


def raw_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


def test_cuda_staging_configuration_uses_environment_and_explicit_override(
    accessor_factory, monkeypatch
):
    local = torch.zeros(3)
    monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_BYTES", str(8 * 2**20))
    accessor = accessor_factory(local)
    assert accessor.cuda_staging_bytes == 8 * 2**20
    assert accessor.cuda_staging_mode == "double-buffer"
    monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_BYTES", "invalid")
    assert accessor_factory(local, cuda_staging_bytes=19).cuda_staging_bytes == 19


def test_cuda_staging_mode_defaults_and_explicit_options(accessor_factory, monkeypatch):
    local = torch.zeros(3)
    monkeypatch.delenv("METASL_GDTENSOR_CUDA_STAGING_BYTES", raising=False)
    monkeypatch.delenv("METASL_GDTENSOR_CUDA_STAGING_MODE", raising=False)
    assert accessor_factory(local).cuda_staging_mode == "pageable"
    accessor = accessor_factory(local, cuda_staging_mode="double-buffer")
    assert accessor.cuda_staging_bytes == 8 * 2**20
    monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_MODE", "pinned")
    assert accessor_factory(local).cuda_staging_mode == "pinned"
    assert accessor_factory(local, cuda_staging_mode="pageable").cuda_staging_mode == "pageable"
    with pytest.raises(ValueError, match="Unknown CUDA staging mode"):
        accessor_factory(local, cuda_staging_mode="invalid")
    with pytest.raises(ValueError, match="requires double-buffer"):
        accessor_factory(local, cuda_staging_bytes=19)


@pytest.mark.parametrize("configured", ["0", "7", "-1", "invalid"])
def test_cuda_staging_environment_rejects_invalid_capacity(
    accessor_factory, monkeypatch, configured
):
    monkeypatch.setenv("METASL_GDTENSOR_CUDA_STAGING_BYTES", configured)
    with pytest.raises(ValueError):
        accessor_factory(torch.zeros(3))


@pytest.mark.parametrize("device", ["cpu", "meta"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
@pytest.mark.parametrize("start,count", [(0, 35), (3, 25), (34, 1)])
def test_default_compute_callbacks_deliver_one_complete_range(
    accessor_factory, device, dtype, start, count
):
    local = torch.arange(35).to(dtype).reshape(7, 5).to(device)
    accessor = accessor_factory(
        local, buffer_bytes=19, value_pattern="zero" if device == "meta" else None
    )
    rng = requested_range(local, start, count)
    expected = (
        bytes(rng.size) if device == "meta" else raw_bytes(local.reshape(-1)[start : start + count])
    )
    reads, writes = [], []

    def take(pointer, length):
        reads.append((pointer, length))
        assert length == rng.size
        assert ctypes.string_at(pointer, length) == expected
        if device == "cpu":
            assert pointer == local.data_ptr() + start * local.element_size()
        return Ok(None)

    assert type(accessor.read_to_buffer(rng, take)) is Ok
    if device == "cpu":
        local.reshape(-1)[start : start + count].fill_(-1)
        before = local.clone()

    def give(pointer, length):
        writes.append((pointer, length))
        assert length == rng.size
        ctypes.memmove(pointer, expected, length)
        return Ok(None)

    assert type(accessor.write_from_buffer(rng, give)) is Ok
    assert len(reads) == len(writes) == 1
    assert accessor.requests == [("weight", rng.global_tensor_offset, rng.size)] * 2
    if device == "cpu":
        assert raw_bytes(local.reshape(-1)[start : start + count]) == expected
        assert torch.equal(local.reshape(-1)[:start], before.reshape(-1)[:start])
        assert torch.equal(local.reshape(-1)[start + count :], before.reshape(-1)[start + count :])


def mapping_permissions(pointer):
    for line in Path("/proc/self/maps").read_text().splitlines():
        bounds, permissions, *_ = line.split()
        lo, hi = (int(bound, 16) for bound in bounds.split("-"))
        if lo <= pointer < hi:
            return permissions
    raise AssertionError(f"No mapping contains {pointer:#x}")


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux mapping permissions")
def test_meta_zero_source_reuses_largest_readonly_mapping(accessor_factory):
    local = torch.empty(2**20, dtype=torch.float32, device="meta")
    accessor = accessor_factory(local)
    pointers = []
    for count in (2**18, 2**16, 2**20, 2**18):
        rng = requested_range(local, 0, count)

        def take(pointer, length, rng=rng):
            pointers.append(pointer)
            assert length == rng.size
            assert mapping_permissions(pointer).startswith("r--")
            assert ctypes.string_at(pointer, length) == bytes(length)
            return Ok(None)

        assert type(accessor.read_to_buffer(rng, take)) is Ok
    assert pointers[0] == pointers[1]
    assert pointers[2] == pointers[3]


def resident_pages(pointer, size):
    page_size = os.sysconf("SC_PAGESIZE")
    start = pointer // page_size * page_size
    pages = (pointer - start + size + page_size - 1) // page_size
    vector = (ctypes.c_ubyte * pages)()
    libc = ctypes.CDLL(None, use_errno=True)
    libc.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte)]
    libc.mincore.restype = ctypes.c_int
    if libc.mincore(start, pointer - start + size, vector):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return sum(value & 1 for value in vector), pages


@pytest.mark.skipif(sys.platform != "linux", reason="uses Linux mapping permissions")
@pytest.mark.parametrize("mode", [None, "", "off"])
def test_meta_sim_d2h_disabled_preserves_readonly_source_and_events(
    accessor_factory, monkeypatch, mode
):
    if mode is None:
        monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H", raising=False)
    else:
        monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", mode)
    # A disabled simulation must not load or validate its optional profile.
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", "/nonexistent/profile.json")
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", "invalid-ignored-when-off")
    local = torch.empty(1024, device="meta")
    events = []
    accessor = accessor_factory(local, staging_events=events)

    def take(pointer, length):
        assert mapping_permissions(pointer).startswith("r--")
        assert ctypes.string_at(pointer, length) == bytes(length)
        return Ok(None)

    assert type(accessor.read_to_buffer(requested_range(local), take)) is Ok
    assert [event["op"] for event in events] == ["meta_generate", "host_materialize"]
    assert events[0]["demand_zero"] is True
    assert accessor._cuda_scratch is None
    assert accessor.meta_sim_d2h_profile is None


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux mincore and MADV_DONTNEED")
@pytest.mark.parametrize("population", POPULATIONS)
def test_meta_sim_d2h_touches_and_discards_pageable_ranges_with_real_timing(
    accessor_factory, monkeypatch, population
):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", population)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", raising=False)
    local = torch.empty(2**18, device="meta")
    events, pointers = [], []
    slot = SimpleNamespace(
        name="weight", file="__0_0.distcp", file_offset=8192, global_offset=local.numel() * 4
    )
    accessor = accessor_factory(local, staging_events=events, staging_slots=(slot,))
    assert accessor.meta_sim_d2h_profile == gdt_storage._meta_d2h_profile(None)
    for count in (2**18, 2**16, 2**18, 0):
        rng = requested_range(local, count=count)

        def take(pointer, length):
            if length:
                pointers.append(pointer)
                assert mapping_permissions(pointer).startswith("rw-")
                resident, pages = resident_pages(pointer, length)
                assert resident == pages  # Check before reading or touching the payload ourselves.
                assert ctypes.string_at(pointer, length) == bytes(length)
            return Ok(None)

        assert type(accessor.read_to_buffer(rng, take)) is Ok
        if rng.size:
            assert resident_pages(pointers[-1], rng.size)[0] == 0
    assert len(set(pointers)) == 1
    assert accessor._meta_source is None
    copies = [event for event in events if event["op"] == "d2h"]
    assert len(copies) == 3
    assert [event["requested"] for event in copies] == [2**20, 2**18, 2**20]
    for event in copies:
        assert event["timing"] == "simulated"
        assert event["population"] == population
        assert event["file"] == "__0_0.distcp"
        assert event["tensor"] == "weight"
        assert event["offset"] == 8192
        assert event["completed"] == event["requested"]
        assert event["success"] is True
        assert event["request"] is False
        assert event["io_layer"] == "staging"
        assert event["thread_id"] > 0
        assert event["duration_ns"] == event["end_ns"] - event["start_ns"]
        target_ns = round((0.00021 + event["requested"] / 1.6e9) * 1e9)
        assert target_ns <= event["duration_ns"] <= target_ns + 5_000_000
    assert [event["op"] for event in events] == ["d2h", "host_materialize"] * 3 + [
        "host_materialize"
    ]


@pytest.mark.parametrize(
    "latency,touch_ns,expects_sleep",
    [(0.004, 200_000, True), (0.0005, 100_000, False), (0.0005, 3_000_000, False)],
)
@pytest.mark.parametrize("population", POPULATIONS)
def test_meta_sim_d2h_waits_only_for_remainder(
    accessor_factory, monkeypatch, latency, touch_ns, expects_sleep, population
):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", population)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", raising=False)
    accessor = accessor_factory(torch.empty(1024, device="meta"))
    accessor.meta_sim_d2h_profile = {
        "a_seconds": latency,
        "bandwidth_bytes_per_s": 1e9,
    }
    now, sleeps, fills = [0], [], []

    def clock():
        result = now[0]
        now[0] += 100_000
        return result

    def fill(pointer, value, size):
        fills.append((pointer, value, size))
        now[0] += touch_ns

    def wait(seconds):
        sleeps.append(seconds)
        now[0] += round(seconds * 1e9)

    monkeypatch.setattr(gdt_storage, "perf_counter_ns", clock)
    monkeypatch.setattr(gdt_storage.ctypes, "memset", fill)
    if population != "memset":
        # The entire native population (including flush/fence) consumes the
        # same budget; do not add a second artificial delay after that work.
        accessor._meta_d2h_populate = lambda pointer, size: fill(pointer, 0, size)
    monkeypatch.setattr(gdt_storage, "sleep", wait)
    accessor._simulate_meta_d2h(12345, 1000)
    assert fills == [(12345, 0, 1000)]
    target_ns = round((latency + 1000 / 1e9) * 1e9)
    assert now[0] >= target_ns
    if expects_sleep:
        assert sleeps == [pytest.approx((target_ns - touch_ns - 100_000) / 1e9)]
    else:
        assert sleeps == []


def test_meta_sim_d2h_population_configuration(accessor_factory, monkeypatch):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", raising=False)
    assert accessor_factory(torch.empty(4, device="meta")).meta_sim_d2h_population == "non-temporal"
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", "typo")
    with pytest.raises(ValueError, match="Unknown Meta D2H population"):
        accessor_factory(torch.empty(4, device="meta"))


@pytest.mark.parametrize("size", [0, 1, 63, 64, 65, 4097])
@pytest.mark.parametrize("population", ["memset-flush", "non-temporal"])
def test_meta_d2h_native_fill_preserves_partial_line_guards(size, population):
    from metasl.meta_d2h_cache import population_function

    buffer = ctypes.create_string_buffer(b"x" * (size + 128))
    population_function(population)(ctypes.addressof(buffer) + 65, size)
    assert buffer.raw[:65] == b"x" * 65
    assert buffer.raw[65 : 65 + size] == bytes(size)
    assert buffer.raw[65 + size : -1] == b"x" * 63


def test_meta_d2h_populate_write_reports_kernel_failure():
    from metasl.meta_d2h_cache import populate_write

    with pytest.raises(OSError):
        populate_write()(1, 4096)  # Unaligned address; never silently fall back to hot stores.


@pytest.mark.parametrize(
    "simulated,mode,population,expected",
    [
        (True, "pageable", "memset-flush", True),
        (True, "off", "memset-flush", False),
        (False, "pageable", "memset-flush", False),
        (True, "pageable", "memset", False),
    ],
)
def test_meta_d2h_helper_prepared_before_dcp_call(
    monkeypatch, simulated, mode, population, expected
):
    from metasl import meta_d2h_cache

    calls = []
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", mode)
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", population)
    monkeypatch.setattr(meta_d2h_cache, "zero_and_flush", lambda: calls.append("prepared"))
    gdt_storage.GDTensorSavePlanner(simulated=simulated)
    assert calls == (["prepared"] if expected else [])


def test_meta_d2h_llc_eviction_uses_capacity_threshold(monkeypatch):
    from metasl import meta_d2h_cache

    calls = []
    monkeypatch.setattr(
        meta_d2h_cache, "populate_write", lambda: lambda p, n: calls.append(("populate", n))
    )
    monkeypatch.setattr(
        meta_d2h_cache, "_native_function", lambda name, *args: lambda p, n: calls.append((name, n))
    )
    monkeypatch.setattr(meta_d2h_cache, "Path", lambda _: SimpleNamespace(read_text=lambda: "128K"))
    populate = meta_d2h_cache.populate_write_evict.__wrapped__()
    populate(1, 4096)
    populate(1, 65536)
    assert calls == [
        ("populate", 4096),
        ("metasl_flush", 4096),
        ("populate", 65536),
        ("metasl_read_cachelines", 256 * 1024),
    ]


@pytest.mark.parametrize(
    "profile",
    [
        "invalid-json",
        [],
        {},
        {"a_seconds": 0},
        {"a_seconds": -1, "bandwidth_bytes_per_s": 1},
        {"a_seconds": float("nan"), "bandwidth_bytes_per_s": 1},
        {"a_seconds": True, "bandwidth_bytes_per_s": 1},
        {"a_seconds": "0", "bandwidth_bytes_per_s": 1},
        {"a_seconds": 0, "bandwidth_bytes_per_s": 0},
        {"a_seconds": 0, "bandwidth_bytes_per_s": -1},
        {"a_seconds": 0, "bandwidth_bytes_per_s": float("inf")},
    ],
)
def test_meta_sim_d2h_rejects_invalid_profile(accessor_factory, monkeypatch, tmp_path, profile):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    path = tmp_path / "profile.json"
    path.write_text(profile if isinstance(profile, str) else json.dumps(profile))
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", str(path))
    with pytest.raises(ValueError, match="Invalid Meta simulated D2H profile"):
        accessor_factory(torch.empty(1024, device="meta"))


def test_meta_sim_d2h_profile_configuration(accessor_factory, monkeypatch, tmp_path):
    local = torch.empty(1024, device="meta")
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "unknown")
    with pytest.raises(ValueError, match="Unknown Meta simulated D2H mode"):
        accessor_factory(local)
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    path = tmp_path / "profile.json"
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", str(path))
    with pytest.raises(ValueError, match="Invalid Meta simulated D2H profile"):
        accessor_factory(local)
    profile = {"a_seconds": 0, "bandwidth_bytes_per_s": 2e9, "fit_samples": 201}
    path.write_text(json.dumps(profile))
    assert accessor_factory(local).meta_sim_d2h_profile == {
        "a_seconds": 0,
        "bandwidth_bytes_per_s": 2e9,
    }


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux mincore and MADV_DONTNEED")
@pytest.mark.parametrize("failure", ["callback_exception", "callback_result"])
@pytest.mark.parametrize("population", POPULATIONS)
def test_meta_sim_d2h_discards_pages_after_failed_untraced_callback(
    accessor_factory, monkeypatch, failure, population
):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", population)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", raising=False)
    local = torch.empty(2**18, device="meta")
    accessor = accessor_factory(local)
    pointers = []

    def take(pointer, length):
        pointers.append(pointer)
        assert resident_pages(pointer, length)[0] > 0
        if failure == "callback_exception":
            raise RuntimeError("injected accessor callback error")
        return None

    match = "injected accessor callback error" if failure == "callback_exception" else "failed"
    error_type = RuntimeError if failure == "callback_exception" else ValueError
    with pytest.raises(error_type, match=match):
        accessor.read_to_buffer(requested_range(local), take)
    assert resident_pages(pointers[0], local.numel() * local.element_size())[0] == 0


def test_meta_sim_d2h_rejects_position_save_but_preserves_meta_load(accessor_factory, monkeypatch):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", raising=False)
    local = torch.empty(32, device="meta")
    accessor = accessor_factory(local, value_pattern="position")
    with pytest.raises(ValueError, match="supports only zero/default"):
        accessor.read_to_buffer(requested_range(local), lambda *_: Ok(None))
    events = []
    accessor = accessor_factory(local, value_pattern="zero", staging_events=events)

    def give(pointer, length):
        ctypes.memset(pointer, 0, length)
        return Ok(None)

    assert type(accessor.write_from_buffer(requested_range(local), give)) is Ok
    assert [event["op"] for event in events] == ["meta_generate", "host_sink", "h2d"]
    assert accessor._cuda_scratch is None


def test_meta_sim_load_never_prepares_native_helper(accessor_factory, monkeypatch):
    from metasl import meta_d2h_cache

    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_H2D_PROFILE", raising=False)

    def forbidden(*_):
        pytest.fail("Meta load must never prepare or load the D2H native helper")

    monkeypatch.setattr(meta_d2h_cache, "population_function", forbidden)
    monkeypatch.setattr(meta_d2h_cache, "_native_library", forbidden)
    local = torch.empty(2048, device="meta")
    events, pointers = [], []
    slot = SimpleNamespace(
        name="weight",
        file="model.safetensors",
        file_offset=4096,
        global_offset=local.numel() * local.element_size(),
    )
    accessor = accessor_factory(
        local, value_pattern="zero", staging_events=events, staging_slots=(slot,)
    )
    for count in (2048, 1024, 0):
        rng = requested_range(local, count=count)

        def give(pointer, length):
            if length:
                pointers.append(pointer)
                ctypes.memset(pointer, 0, length)
            return Ok(None)

        assert type(accessor.write_from_buffer(rng, give)) is Ok
        if count:
            assert resident_pages(pointers[-1], rng.size)[0] == 0
    assert pointers[0] == pointers[1]
    assert not accessor._meta_d2h_prepared
    assert accessor._meta_d2h_populate is None
    assert [event["op"] for event in events] == [
        "meta_generate",
        "host_sink",
        "h2d",
        "meta_generate",
        "host_sink",
        "h2d",
        "host_sink",
    ]
    for sink, copy in ((events[1], events[2]), (events[4], events[5])):
        assert sink["end_ns"] <= copy["start_ns"]
        assert copy["timing"] == "simulated"
        assert copy["requested"] == sink["requested"] == copy["completed"]
        assert copy["file"] == "model.safetensors"
        assert copy["offset"] == 4096
        assert copy["success"] and not copy["request"]


def test_meta_sim_h2d_uses_profile_and_waits_after_callback(
    accessor_factory, monkeypatch, tmp_path
):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    path = tmp_path / "h2d.json"
    path.write_text(json.dumps({"a_seconds": 0.004, "bandwidth_bytes_per_s": 1e9}))
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_H2D_PROFILE", str(path))
    local = torch.empty(1024, device="meta")
    accessor = accessor_factory(local)
    now, sleeps = [0], []

    def clock():
        result = now[0]
        now[0] += 100_000
        return result

    def wait(seconds):
        sleeps.append(seconds)
        now[0] += round(seconds * 1e9)

    monkeypatch.setattr(gdt_storage, "perf_counter_ns", clock)
    monkeypatch.setattr(gdt_storage, "sleep", wait)
    accessor._simulate_meta_h2d(1000)
    assert sleeps == [pytest.approx(0.003901)]
    assert now[0] >= 4_001_000


def test_meta_sim_h2d_does_not_run_after_failed_callback(accessor_factory, monkeypatch):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    local = torch.empty(1024, device="meta")
    events, pointers = [], []
    accessor = accessor_factory(local, staging_events=events)

    def give(pointer, length):
        pointers.append(pointer)
        ctypes.memset(pointer, 0, length)
        raise RuntimeError("load callback failure")

    with pytest.raises(RuntimeError, match="load callback failure"):
        accessor.write_from_buffer(requested_range(local), give)
    assert [event["op"] for event in events] == ["meta_generate", "host_sink"]
    assert resident_pages(pointers[0], local.numel() * local.element_size())[0] == 0


def test_direct_meta_save_prepares_once_before_range(accessor_factory, monkeypatch):
    from metasl import meta_d2h_cache

    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "pageable")
    calls = []

    def prepare(population):
        calls.append(population)
        return lambda pointer, size: ctypes.memset(pointer, 0, size)

    monkeypatch.setattr(meta_d2h_cache, "population_function", prepare)
    local = torch.empty(1024, device="meta")
    accessor = accessor_factory(local)
    assert calls == []
    accessor.prepare_save()
    accessor.prepare_save()
    accessor.read_to_buffer(requested_range(local), lambda *_: Ok(None))
    assert calls == ["non-temporal"]


def test_meta_native_helper_reuses_persistent_source_cache(monkeypatch, tmp_path):
    from metasl import meta_d2h_cache

    builds, loads = [], []

    def compile_library(command, **options):
        builds.append(command)
        assert options["check"]
        Path(command[-1]).write_bytes(b"test shared library")

    monkeypatch.setattr(meta_d2h_cache.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(meta_d2h_cache.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(meta_d2h_cache.subprocess, "run", compile_library)
    monkeypatch.setattr(meta_d2h_cache.ctypes, "CDLL", lambda path: loads.append(path))
    meta_d2h_cache._native_library.__wrapped__()
    meta_d2h_cache._native_library.__wrapped__()
    assert len(builds) == 1
    assert len(loads) == 2 and loads[0] == loads[1]
    assert Path(loads[0]).read_bytes() == b"test shared library"


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux mincore and MADV_DONTNEED")
def test_meta_sink_faults_and_discards_each_reused_range(accessor_factory):
    local = torch.empty(2**20, dtype=torch.float32, device="meta")
    accessor = accessor_factory(local, value_pattern="zero")
    pointers = []
    for count in (2**20, 2**18, 2**20):
        rng = requested_range(local, 0, count)

        def give(pointer, length, rng=rng):
            pointers.append(pointer)
            assert length == rng.size
            assert resident_pages(pointer, length)[0] == 0
            before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
            ctypes.memset(pointer, 0, length)
            assert resource.getrusage(resource.RUSAGE_SELF).ru_minflt > before
            resident, pages = resident_pages(pointer, length)
            assert resident == pages
            return Ok(None)

        assert type(accessor.write_from_buffer(rng, give)) is Ok
        assert resident_pages(pointers[-1], rng.size)[0] == 0
    assert len(set(pointers)) == 1


@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux mincore and MADV_DONTNEED")
@pytest.mark.parametrize("failure", ["callback_exception", "callback_result", "validation"])
def test_meta_sink_discards_pages_when_callback_or_validation_fails(accessor_factory, failure):
    local = torch.empty(2**18, dtype=torch.float32, device="meta")
    accessor = accessor_factory(local, value_pattern="zero")
    rng = requested_range(local)
    pointers = []

    def give(pointer, length):
        pointers.append(pointer)
        ctypes.memset(pointer, 1 if failure == "validation" else 0, length)
        if failure == "callback_exception":
            raise RuntimeError("injected accessor callback error")
        return None if failure == "callback_result" else Ok(None)

    match = {
        "callback_exception": "injected accessor callback error",
        "callback_result": "gdtensor operation failed",
        "validation": "nonzero checkpoint value",
    }[failure]
    with pytest.raises((RuntimeError, ValueError), match=match):
        accessor.write_from_buffer(rng, give)
    assert len(pointers) == 1
    assert resident_pages(pointers[0], rng.size)[0] == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16, torch.int64])
def test_meta_position_sink_checks_subrange_after_one_callback(accessor_factory, dtype):
    local = torch.empty((7, 5), dtype=dtype, device="meta")
    accessor = accessor_factory(
        local,
        buffer_bytes=19,
        value_pattern="position",
        alias_groups={"weight": ("canonical", "weight")},
    )
    rng = requested_range(local, 3, 25)
    expected_tensor = torch.cat(
        [
            block
            for _, block in pattern_blocks(
                (25,),
                (rng.global_tensor_offset // local.element_size(),),
                (70,),
                "canonical",
                dtype,
                block=7,
            )
        ]
    )
    expected = raw_bytes(expected_tensor)
    callbacks = []

    def give(pointer, length):
        callbacks.append(length)
        ctypes.memmove(pointer, expected, length)
        return Ok(None)

    assert type(accessor.write_from_buffer(rng, give)) is Ok
    assert callbacks == [rng.size]

    def corrupt(pointer, length):
        ctypes.memmove(pointer, bytes(length), length)
        return Ok(None)

    with pytest.raises(ValueError, match="numerical mismatch"):
        accessor.write_from_buffer(rng, corrupt)
