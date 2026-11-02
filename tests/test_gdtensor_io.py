"""Public gdtensor IO: exact DCP containers, CPU resharding and meta mappings."""

import hashlib
import io
import json
import math
import pickle
import sys
import tracemalloc
import zipfile
import zlib
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    TensorProperties,
    TensorStorageMetadata,
)
from torch.distributed.tensor import DTensor, Shard
from transformers import Qwen2Config

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from gdt_risk_probe import run_probe  # noqa: E402
from gdtensor.builder import Ok  # noqa: E402
from gdtensor.io import rank_load, rank_save  # noqa: E402
from gdtensor.session import StorageLoadSession, StorageSaveSession  # noqa: E402

from metasl import gdt_storage  # noqa: E402
from metasl.fsdp import apply_parallelism, describe_targets, materialize
from torch.distributed.checkpoint.state_dict import get_model_state_dict  # noqa: E402
from metasl.gdt_options import validate_backend  # noqa: E402
from metasl.gdt_storage import (  # noqa: E402
    GDTensorFileSystemReader,
    GDTensorFileSystemWriter,
    GDTensorLoadPlanner,
    GDTensorSavePlanner,
    begin_container,
    infer_shard_grid,
    patch_container,
    read_container,
)
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.layout import Layout, ParallelDims  # noqa: E402
from metasl.scratch import OptionalKeysLoadPlanner  # noqa: E402
from metasl.sharding import local_tensor  # noqa: E402
from metasl.skeleton import model_from_config, model_structure  # noqa: E402
from metasl.storage import AdmittedFileSystemReader, AdmittedFileSystemWriter  # noqa: E402
from metasl.run import build_parser  # noqa: E402

ROOT = Path(__file__).resolve().parent
DTYPES = (torch.float32, torch.bfloat16, torch.float16, torch.int64)


def random_tensor(shape, dtype, seed=734):
    generator = torch.Generator().manual_seed(seed)
    if dtype.is_floating_point:
        return torch.randn(shape, generator=generator).to(dtype)
    return torch.randint(-100000, 100000, shape, dtype=dtype, generator=generator)


def tensor_bytes(tensor):
    return tensor.reshape(-1).view(torch.uint8).numpy().tobytes()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (0, 4), (17,), (7, 5)])
def test_container_identity_for_random_values(dtype, shape):
    tensor = random_tensor(shape, dtype)
    expected = io.BytesIO()
    torch.save(tensor, expected)
    actual = io.BytesIO()
    actual.write(b"preceding item")
    container = begin_container(actual, shape, dtype)
    payload = tensor_bytes(tensor)
    actual.seek(container.payload_offset)
    actual.write(payload)
    patch_container(actual, container, zlib.crc32(payload), payload[:8])
    data = actual.getvalue()[container.offset : container.end]
    assert data == expected.getvalue()
    assert container.payload_size == tensor.numel() * tensor.element_size()
    assert (
        read_container(actual, container.offset, container.size, shape, dtype)
        == container.payload_offset
    )
    assert torch.equal(torch.load(io.BytesIO(data), weights_only=True), tensor)


def test_container_handles_zip_descriptor_signature_in_payload():
    tensor = torch.tensor(list(b"PK\x07\x08abc12345"), dtype=torch.uint8)
    expected = io.BytesIO()
    torch.save(tensor, expected)
    stream = io.BytesIO()
    container = begin_container(stream, tensor.shape, tensor.dtype)
    payload = tensor_bytes(tensor)
    stream.seek(container.payload_offset)
    stream.write(payload)
    patch_container(stream, container, zlib.crc32(payload), payload[:8])
    assert stream.getvalue() == expected.getvalue()


@pytest.mark.parametrize("payload_bytes", [2**32, 2**42])
def test_large_container_envelope_stays_sparse(payload_bytes):
    stream = gdt_storage._SparseEnvelope()
    shape = (payload_bytes // 4,)
    container = begin_container(stream, shape, torch.float32)
    assert container.payload_size == payload_bytes
    assert payload_bytes < container.size < payload_bytes + 4096
    assert sum(len(data) for _, data in stream.segments) < 4096
    assert (
        read_container(stream, container.offset, container.size, shape, torch.float32)
        == container.payload_offset
    )


def test_container_parser_does_not_read_payload():
    tensor = random_tensor((10000,), torch.float32)
    stream = io.BytesIO()
    torch.save(tensor, stream)
    description = begin_container(io.BytesIO(), tensor.shape, tensor.dtype)

    class HeaderOnlyStream(io.BytesIO):
        def __init__(self, data):
            super().__init__(data)
            self.bytes_read = 0

        def read(self, length=-1):
            start = self.tell()
            end = len(self.getbuffer()) if length < 0 else start + length
            assert end <= description.payload_offset or start >= (
                description.payload_offset + description.payload_size
            ), "container parser read tensor payload"
            data = super().read(length)
            self.bytes_read += len(data)
            return data

    metadata_stream = HeaderOnlyStream(stream.getvalue())
    assert (
        read_container(metadata_stream, 0, len(stream.getvalue()), tensor.shape, tensor.dtype)
        == description.payload_offset
    )
    assert metadata_stream.bytes_read < 4096


@pytest.mark.parametrize("damage", ["dtype", "shape", "stride", "offset", "storage", "compressed"])
def test_container_parser_rejects_unsupported_records(damage):
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    if damage == "stride":
        tensor = tensor.T
    elif damage == "offset":
        tensor = torch.arange(14, dtype=torch.float32)[2:].reshape(3, 4)
    elif damage == "storage":
        tensor = torch.arange(14, dtype=torch.float32)[:12].reshape(3, 4)
    stream = io.BytesIO()
    torch.save(tensor, stream)
    if damage == "compressed":
        compressed = io.BytesIO()
        with zipfile.ZipFile(stream) as source, zipfile.ZipFile(compressed, "w") as target:
            for info in source.infolist():
                target.writestr(
                    info.filename, source.read(info), compress_type=zipfile.ZIP_DEFLATED
                )
        stream = compressed
    dtype = torch.float16 if damage == "dtype" else tensor.dtype
    shape = (12,) if damage == "shape" else tensor.shape
    with pytest.raises(ValueError):
        read_container(stream, 0, len(stream.getvalue()), shape, dtype)


def test_public_payload_slot_mapping_and_unequal_worlds():
    result = run_probe()
    assert result["R1"]["save_requests"]
    assert [(r["saved"], r["loaded"]) for r in result["R2"]] == [(4, 3), (2, 4)]


def targets(rank, world, scope, simulated=False):
    config = Qwen2Config(**json.loads((ROOT / "tiny_qwen2.json").read_text()))
    with torch.device("meta"):
        model = model_from_config(config, torch.float32)
    structure = model_structure(model)
    layout = Layout(ParallelDims(dp_shard=world), config.num_hidden_layers, rank)
    layout.build_mesh()
    apply_parallelism(model, layout, "fp32")
    state = {"model": get_model_state_dict(model)}
    rows = describe_targets(state, structure)
    return (state if simulated else materialize(state)), rows


def expected_tensor(row, dtype):
    name = row["fqn"]
    if name.startswith("model."):
        name = "model." + row["alias_group"][0]
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    full = random_tensor(tuple(row["global_shape"]), dtype, seed)
    if not full.ndim:
        return full
    slices = tuple(
        slice(offset, offset + size)
        for offset, size in zip(row["global_offsets"], row["local_shape"], strict=True)
    )
    return full[slices]


def fill_values(state, rows, *, check=False, poison=False):
    flat = flatten_state_dict(state)[0]
    for row in rows:
        tensor = local_tensor(flat[row["fqn"]])
        expected = expected_tensor(row, tensor.dtype)
        if check:
            assert torch.equal(tensor, expected), row["fqn"]
        else:
            tensor.copy_(torch.full_like(expected, 77) if poison else expected)


def optional_keys(rank):
    return [f"dataloader.dp_rank_{rank}", f"rng.rank_{rank}"]


def writer(path, backend):
    cls = GDTensorFileSystemWriter if backend == "gdtensor" else AdmittedFileSystemWriter
    return cls(path, thread_count=1, sync_files=True, overwrite=False)


def compare_checkpoints(left, right):
    left_files = {p.name for p in left.glob("*.distcp")}
    assert left_files == {p.name for p in right.glob("*.distcp")}
    for name in left_files:
        assert (left / name).read_bytes() == (right / name).read_bytes(), name
    with (left / ".metadata").open("rb") as stream:
        left_md = pickle.load(stream)
    with (right / ".metadata").open("rb") as stream:
        right_md = pickle.load(stream)
    assert left_md.state_dict_metadata == right_md.state_dict_metadata
    assert left_md.storage_data == right_md.storage_data


def group_worker(rank, world, rendezvous, root, action, scope, simulated=False):
    torch.set_num_threads(1)
    torch.manual_seed(144)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).resolve().as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        root = Path(root)
        if action == "save":
            state, rows = targets(rank, world, scope, simulated)
            if not simulated:
                fill_values(state, rows)
            for backend in ("dcp", "gdtensor"):
                planner = (
                    GDTensorSavePlanner(simulated=simulated)
                    if backend == "gdtensor"
                    else dcp.DefaultSavePlanner()
                )
                dcp.save(state, storage_writer=writer(root / backend, backend), planner=planner)
            dist.barrier()
            if rank == 0:
                compare_checkpoints(root / "dcp", root / "gdtensor")
            return
        evidence = []
        for saved_by, loaded_by in (("dcp", "gdtensor"), ("gdtensor", "dcp")):
            state, rows = targets(rank, world, scope, simulated)
            if not simulated:
                fill_values(state, rows, poison=True)
            if scope == "training":
                state["lr_scheduler"]["last_epoch"] = -777
                state["dataloader"][f"dp_rank_{rank}"] = b"poison"
            if loaded_by == "gdtensor":
                reader = GDTensorFileSystemReader(root / saved_by, rows=rows)
                planner = GDTensorLoadPlanner(
                    rows,
                    simulated=simulated,
                    pattern="zero" if simulated else None,
                    optional_keys=optional_keys(rank),
                )
            else:
                reader = AdmittedFileSystemReader(root / saved_by, rows=rows)
                planner = OptionalKeysLoadPlanner()
            dcp.load(state, storage_reader=reader, planner=planner)
            if loaded_by == "gdtensor":
                planner.assert_complete()
                assert planner.completed_items == planner.expected_items
                payload = sum(request["length"] for request in reader.payload_requests)
                flat = flatten_state_dict(state)[0]
                expected = sum(
                    local_tensor(value).numel() * value.element_size()
                    for value in flat.values()
                    if isinstance(value, DTensor)
                )
                assert payload == expected
                evidence.append({"payload_bytes": payload, "requests": reader.payload_requests})
            if simulated:
                assert all(
                    local_tensor(value).is_meta
                    for value in flatten_state_dict(state)[0].values()
                    if isinstance(value, DTensor)
                )
            else:
                fill_values(state, rows, check=True)
            if scope == "training":
                assert state["lr_scheduler"]["last_epoch"] == 9
                if f"dataloader.dp_rank_{rank}" not in planner.skipped_keys:
                    assert state["dataloader"][f"dp_rank_{rank}"] == bytes(37)
        (root / f"load-{world}-{rank}.json").write_text(json.dumps(evidence))
    finally:
        dist.destroy_process_group()


def run_group(root, world, action, scope, simulated=False):
    rendezvous = root / f"rendezvous-{action}-{world}"
    mp.spawn(
        group_worker,
        args=(world, str(rendezvous), str(root), action, scope, simulated),
        nprocs=world,
        join=True,
    )


@pytest.mark.parametrize("scope", ["model"])
@pytest.mark.parametrize("saved,loaded", [(2, 3), (3, 2), (3, 4)])
def test_cpu_qwen_native_identity_and_cross_backend_resharding(tmp_path, scope, saved, loaded):
    run_group(tmp_path, saved, "save", scope)
    run_group(tmp_path, loaded, "load", scope)
    compare_checkpoints(tmp_path / "dcp", tmp_path / "gdtensor")
    assert all((tmp_path / f"load-{loaded}-{rank}.json").exists() for rank in range(loaded))




def tail_empty_worker(rank, world, rendezvous, root, action, scope):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=Path(rendezvous).resolve().as_uri(),
        rank=rank,
        world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        mesh = dist.device_mesh.init_device_mesh("cpu", (world,))
        full = torch.arange(10, dtype=torch.float32).reshape(5, 2)
        width = (5 + world - 1) // world
        begin, end = min(rank * width, 5), min((rank + 1) * width, 5)
        local = full[begin:end].clone()
        if action == "load":
            local.fill_(-999)
        tensor = DTensor.from_local(
            local, mesh, [Shard(0)], shape=full.shape, stride=full.stride(), run_check=False
        )
        state = (
            {"step": torch.tensor(9), "token": b"training bytes", "weight": tensor}
            if scope == "training"
            else {"weight": tensor}
        )
        path = Path(root) / "native"
        if action == "save":
            dcp.save(state, storage_writer=writer(path, "dcp"))
            return
        if scope == "training":
            state["step"].fill_(-1)
            state["token"] = b"poison"
        recorder = IORecorder()
        reader = GDTensorFileSystemReader(path)
        reader.fs = RecordedFileSystem(reader.fs, recorder)
        planner = GDTensorLoadPlanner()
        dcp.load(state, storage_reader=reader, planner=planner)
        planner.assert_complete()
        assert torch.equal(tensor.to_local(), full[begin:end])
        if scope == "training":
            assert state["step"].item() == 9
            assert state["token"] == b"training bytes"
        assert sum(r["length"] for r in reader.payload_requests) == local.nbytes
        trace = recorder.summary()
        for request in reader.payload_requests:
            events = recorder.files[request["file"]]["events"]
            assert any(
                e["op"] == "read"
                and e["offset"] == request["file_offset"]
                and e["completed"] == request["length"]
                for e in events
            )
        assert trace["files"]
        (Path(root) / f"tail-empty-load-{rank}.json").write_text(json.dumps(trace))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("scope", ["model"])
def test_native_tail_empty_shard_reshards_with_recorded_reads(tmp_path, scope):
    for action, world in (("save", 4), ("load", 3)):
        mp.spawn(
            tail_empty_worker,
            args=(world, str(tmp_path / f"rendezvous-{action}"), str(tmp_path), action, scope),
            nprocs=world,
            join=True,
        )
    with (tmp_path / "native" / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    assert sorted(
        (tuple(chunk.offsets), tuple(chunk.sizes))
        for chunk in metadata.state_dict_metadata["weight"].chunks
    ) == [((0, 0), (2, 2)), ((2, 0), (2, 2)), ((4, 0), (1, 2)), ((5, 0), (0, 2))]
    if scope == "training":
        native_files = {
            info.relative_path for index, info in metadata.storage_data.items()
            if index.fqn in ("step", "token")
        }
        tensor_files = {
            info.relative_path for index, info in metadata.storage_data.items()
            if index.fqn == "weight" and index.offset[0] < 5
        }
        assert native_files & tensor_files
    assert all((tmp_path / f"tail-empty-load-{rank}.json").exists() for rank in range(3))


def test_meta_mappings_keep_python_allocations_within_executor_range_bound(tmp_path, process_group):
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    allocations = []
    for size in (2**18, 2**23):
        tensor = DTensor.from_local(
            torch.empty(size, device="meta"),
            mesh,
            [Shard(0)],
            shape=torch.Size([size]),
            stride=(1,),
            run_check=False,
        )
        path = tmp_path / str(size)
        output = GDTensorFileSystemWriter(path, thread_count=1, sync_files=False)
        tracemalloc.start()
        dcp.save(
            {"weight": tensor}, storage_writer=output, planner=GDTensorSavePlanner(simulated=True)
        )
        _, save_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        reader = GDTensorFileSystemReader(path)
        planner = GDTensorLoadPlanner(simulated=True, pattern="zero")
        tracemalloc.start()
        dcp.load({"weight": tensor}, storage_reader=reader, planner=planner)
        _, load_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        planner.assert_complete()
        allocations.append((output.buffer_bytes, reader.buffer_bytes))
        payload = size * tensor.element_size()
        assert max(save_peak, load_peak) < 2.1 * payload + 2**21
        assert tensor.to_local().is_meta
    assert allocations[0] == allocations[1]
    assert all(0 < capacity <= 2**21 for capacity in allocations[0])


def test_whole_range_meta_callbacks_keep_public_storage_requests_bounded(monkeypatch):
    """Unit-test callbacks with injected geometry; real DTensor tests above retain gloo."""

    class MetaShard:
        def __init__(self, count):
            self.local = torch.empty(count, device="meta", dtype=torch.float32)
            self.shape, self.dtype = tuple(self.local.shape), self.local.dtype

        def element_size(self):
            return self.local.element_size()

        def to_local(self):
            return self.local

    class Session(StorageSaveSession, StorageLoadSession):
        def __init__(self, accessor):
            self.accessor = accessor

        def init_data_accessor(self):
            return Ok(self.accessor)

        def get_file_offset(self, pos):
            return Ok(0)

        def commit(self):
            return Ok(None)

        abort = commit
        finish = commit

    monkeypatch.setattr(
        gdt_storage,
        "_compute_geometry",
        lambda tensor: gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 1),
    )
    storage_capacities = []
    for payload_bytes in (2**20, 2**25):
        tensor = MetaShard(payload_bytes // 4)
        tensors = {"weight": tensor}
        compute = gdt_storage.ComputeAccessor(tensors, 0, value_pattern="zero")
        callback_lengths = {"save": [], "load": []}

        def track(method, operation, callback_lengths=callback_lengths):
            def call(rng, callback):
                def observed(pointer, length):
                    callback_lengths[operation].append(length)
                    return callback(pointer, length)

                return method(rng, observed)

            return call

        compute.read_to_buffer = track(compute.read_to_buffer, "save")
        compute.write_from_buffer = track(compute.write_from_buffer, "load")
        slot = gdt_storage.PayloadSlot("weight", 0, "unit.distcp", payload_bytes, 0)
        byte_counts = {"read": 0, "write": 0}

        def read(file, offset, length, byte_counts=byte_counts):
            assert length <= 65536
            byte_counts["read"] += length
            return bytes(length)

        def write(file, offset, data, byte_counts=byte_counts):
            assert len(data) <= 65536
            byte_counts["write"] += len(data)

        storage = gdt_storage.StorageAccessor([slot], read=read, write=write)
        assert storage._buffer is None
        session = Session(storage)
        begin = lambda view, rank, session=session: Ok(session)
        cv, sv = gdt_storage.make_views(
            tensors,
            {"weight": gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 1)},
            [slot],
            compute,
            begin_save=begin,
            begin_load=begin,
        )
        tracemalloc.start()
        assert type(rank_save(0, cv, sv)) is Ok
        assert type(rank_load(0, cv, sv)) is Ok
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        assert byte_counts == {"read": payload_bytes, "write": payload_bytes}
        assert callback_lengths == {"save": [payload_bytes], "load": [payload_bytes]}
        assert peak < 2.1 * payload_bytes + 2**21
        storage_capacities.append(len(storage.buffer))
    assert storage_capacities == [65536, 65536]


def test_resharded_load_reads_only_intersecting_payload(tmp_path):
    run_group(tmp_path, 2, "save", "model")
    run_group(tmp_path, 3, "load", "model")
    with (tmp_path / "dcp" / ".metadata").open("rb") as stream:
        metadata = pickle.load(stream)
    exact, native_full_chunks = 0, 0
    for rank in range(3):
        evidence = json.loads((tmp_path / f"load-3-{rank}.json").read_text())[0]
        exact += evidence["payload_bytes"]
        for md in metadata.state_dict_metadata.values():
            if not isinstance(md, TensorStorageMetadata):
                continue
            width = (md.size[0] + 2) // 3
            target_start, target_end = (
                min(rank * width, md.size[0]),
                min((rank + 1) * width, md.size[0]),
            )
            for chunk in md.chunks:
                if max(target_start, chunk.offsets[0]) < min(
                    target_end, chunk.offsets[0] + chunk.sizes[0]
                ):
                    native_full_chunks += (
                        math.prod(chunk.sizes)
                        * torch.empty((), dtype=md.properties.dtype).element_size()
                    )
    assert 0 < exact < native_full_chunks
    print(f"gdtensor intersecting payload={exact}, native full-chunk payload={native_full_chunks}")


def test_irregular_saved_chunk_grid_fails_closed():
    metadata = TensorStorageMetadata(
        TensorProperties(dtype=torch.float32),
        torch.Size([8, 4]),
        [
            ChunkStorageMetadata(torch.Size([0, 0]), torch.Size([3, 4])),
            ChunkStorageMetadata(torch.Size([3, 0]), torch.Size([5, 4])),
        ],
    )
    with pytest.raises(ValueError, match="ceil|grid|Shard"):
        infer_shard_grid("model.weight", metadata)


def backend_args(**overrides):
    values = dict(
        io_backend="gdtensor",
        action="save",
        parallel="",
        sharding="fsdp2",
        checkpoint_format="dcp",
        threads=1,
        dcp_file_layout="rank",
        implementation="simulated",
        standard_device="cpu",
        checkpoint=Path("checkpoint"),
        data_pattern="zero",
        world_size=2,
        ranks=2,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"parallel": "pp=2"},
        {"parallel": "tp=2"},
        {"parallel": "dp_replicate=2"},
        {"parallel": "cp=2"},
        {"sharding": "derived"},
        {"threads": 2},
        {"checkpoint_format": "safetensors", "checkpoint_state": "training"},
        {"dcp_file_layout": "item"},
        {"implementation": "standard", "standard_device": "mps"},
        {"action": "train"},
        {"checkpoint": "hdfs://cluster/checkpoint"},
    ],
)
def test_unsupported_gdtensor_options_fail_early(overrides):
    with pytest.raises(ValueError, match="gdtensor"):
        validate_backend(backend_args(**overrides))


def test_default_backend_and_supported_pure_fsdp_options():
    parser = build_parser()
    args = parser.parse_args(["--action", "save", "--bundle", "bundle", "--output", "output"])
    assert args.io_backend == "gdtensor"
    validate_backend(backend_args())
    validate_backend(backend_args(parallel="dp_shard=2"))
    validate_backend(backend_args(implementation="standard"))
    validate_backend(backend_args(implementation="standard", standard_device="cuda"))
    with pytest.raises(ValueError, match="only IO backend"):
        validate_backend(backend_args(io_backend="dcp"))


def test_sessions_copy_payload_through_public_io_and_commit_exact_containers(
    tmp_path
):
    from gdt_risk_probe import ComputeAccessor, compute_view, storage_view

    values = {
        "fp32": random_tensor((7, 5), torch.float32),
        "bf16_empty": torch.empty((0, 4), dtype=torch.bfloat16),
        "fp16": random_tensor((17,), torch.float16),
        "int64": random_tensor((5, 3), torch.int64),
        "descriptor_magic": torch.tensor(list(b"PK\x07\x08abcdefgh"), dtype=torch.uint8),
    }
    payloads = {name: bytearray(tensor_bytes(tensor)) for name, tensor in values.items()}
    shapes = {name: len(data) for name, data in payloads.items()}
    expected = io.BytesIO()
    for tensor in values.values():
        torch.save(tensor, expected)
    filename = "__0_0.distcp"
    slots = []
    with (tmp_path / filename).open("w+b") as stream:
        for name, tensor in values.items():
            container = begin_container(stream, tensor.shape, tensor.dtype)
            slots.append(
                gdt_storage.PayloadSlot(
                    name,
                    0,
                    filename,
                    tensor.nbytes,
                    0,
                    file_offset=container.payload_offset,
                    container=container,
                )
            )
        gdt_storage.assign_payload_offsets(slots, list(values))
        stream.flush()
        session = gdt_storage._SaveSession(stream, slots, sync_files=True)
        session.accessor.buffer_bytes = 3
        sv, _, _ = storage_view(shapes, 1, {filename: [0]})
        sv.set_begin_save(lambda view, rank: Ok(session))
        result = rank_save(0, compute_view(shapes, 1, ComputeAccessor(payloads)), sv)
        assert type(result) is Ok, result
        stream.seek(0)
        assert stream.read() == expected.getvalue()
        assert session.first_bytes[id(slots[-1])] == b"PK\x07\x08abcd"
    targets = {name: bytearray(len(data)) for name, data in payloads.items()}
    reader = GDTensorFileSystemReader(tmp_path)
    load_session = gdt_storage._LoadSession(reader, slots)
    load_session.accessor.buffer_bytes = 3
    sv.set_begin_load(lambda view, rank: Ok(load_session))
    result = rank_load(0, compute_view(shapes, 1, ComputeAccessor(targets)), sv)
    assert type(result) is Ok, result
    assert targets == payloads
    assert sum(r["length"] for r in reader.payload_requests) == sum(shapes.values())


def test_save_session_patches_completed_slots_once_after_payload_write(
    tmp_path
):
    # Deliberately leave the final envelope buffered: patch_container must flush
    # it before pwrite, without moving the stream's logical position.
    payload_events, patch_events = [], []
    with (tmp_path / "containers").open("w+b") as stream:
        slots = []
        expected = io.BytesIO()
        for name, size in (("a", 5), ("b", 4), ("empty", 0)):
            container = begin_container(stream, (size,), torch.uint8)
            slots.append(
                gdt_storage.PayloadSlot(
                    name,
                    0,
                    "containers",
                    size,
                    0,
                    file_offset=container.payload_offset,
                    container=container,
                )
            )
            torch.save(torch.arange(size, dtype=torch.uint8), expected)
        end = stream.tell()
        gdt_storage.assign_payload_offsets(slots, [s.name for s in slots])
        session = gdt_storage._SaveSession(
            stream,
            slots,
            False,
            payload_events=payload_events,
            container_patch_events=patch_events,
        )
        session.accessor.buffer_bytes = 3
        for index, slot in enumerate(slots[:2]):
            rng = SimpleNamespace(
                file_path=slot.file, contiguous_payload_offset=slot.payload_offset, size=slot.size
            )
            consumed = 0

            def callback(address, length):
                nonlocal consumed
                data = bytes(range(consumed, consumed + length))
                gdt_storage.ctypes.memmove(address, data, length)
                consumed += length
                return Ok(None)

            session.accessor.write_from_buffer(rng, callback)
            assert stream.tell() == end
            assert len(session.patched) == index + 1
            assert patch_events[-5]["start_ns"] >= (
                payload_events[-1]["start_ns"] + payload_events[-1]["duration_ns"]
            )
        # Empty slots are absent from StorageAccessor and require commit fallback.
        session.commit()
        assert len(session.patched) == 3
        # An empty ZIP payload has no data descriptor, hence four patches.
        assert len(patch_events) == 14
        session.commit()
        assert len(patch_events) == 14
        stream.seek(0)
        assert stream.read() == expected.getvalue()


def test_save_session_rejects_incomplete_payload(tmp_path):
    with (tmp_path / "container").open("w+b") as stream:
        container = begin_container(stream, (4,), torch.uint8)
        slot = gdt_storage.PayloadSlot("a", 0, "container", 4, 0, container=container)
        session = gdt_storage._SaveSession(stream, [slot], False)
        session.observe(slot, 0, b"ab")
        assert not session.patched
        with pytest.raises(ValueError, match="Incomplete"):
            session.commit()
        with pytest.raises(ValueError, match="Out-of-order"):
            session.observe(slot, 0, b"ab")
        with pytest.raises(ValueError, match="exceeds"):
            session.observe(slot, 2, b"abc")
