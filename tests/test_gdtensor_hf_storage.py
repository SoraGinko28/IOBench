"""Safetensors layout identity, native cross-loads and gdtensor payload traces."""

import hashlib
import json
import struct
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from torch.distributed.checkpoint.api import CheckpointException
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from metasl.gdt_hf_storage import (  # noqa: E402
    GDTensorHuggingFaceStorageReader,
    GDTensorHuggingFaceStorageWriter,
)
from metasl.gdt_options import validate_backend  # noqa: E402
from metasl.gdt_storage import GDTensorLoadPlanner, GDTensorSavePlanner  # noqa: E402
from metasl.hf_storage import (  # noqa: E402
    COMPLETION_MARKER,
    AdmittedHuggingFaceStorageReader,
    AdmittedHuggingFaceStorageWriter,
    safetensors_file_buckets,
    canonical_safetensors_header,
    validate_safetensors_checkpoint,
)
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.meta_d2h_profile import duration_seconds, load_profile  # noqa: E402
from metasl.sharding import fill_or_check, local_tensor, manifest_row  # noqa: E402


def model_state(mesh, *, device="cpu", pattern="position", bucketed=False):
    # Different element widths exercise safetensors' dtype/name payload order.
    specs = {
        "a_fp32": ((160 * 1024,) if bucketed else (7, 5), torch.float32),
        "b_bf16": ((320 * 1024,) if bucketed else (19,), torch.bfloat16),
        "c_fp16": ((11, 3), torch.float16),
        "d_i64": ((17,), torch.int64),
    }
    if bucketed:
        specs["oversized"] = ((300 * 1024,), torch.float32)
    state = {"model": {}}
    rows = []
    for name, (shape, dtype) in specs.items():
        local = torch.empty(shape, dtype=dtype, device=device)
        tensor = DTensor.from_local(
            local, mesh, [Shard(0)], shape=local.shape, stride=local.stride(), run_check=False
        )
        state["model"][name] = tensor
    state["model"]["scalar"] = torch.empty((), dtype=torch.int64, device=device)
    state["model"]["empty"] = torch.empty((0, 3), dtype=torch.float32, device=device)
    for name, tensor in state["model"].items():
        row = manifest_row(f"model.{name}", tensor, "model", [name], "hf-unit-test")
        rows.append(row)
        if device != "meta":
            fill_or_check(
                local_tensor(tensor), row["global_offsets"], row["global_shape"], name, pattern
            )
    return state, rows


def assert_values(state, rows, pattern):
    for row in rows:
        name = row["fqn"].removeprefix("model.")
        fill_or_check(
            local_tensor(state["model"][name]),
            row["global_offsets"],
            row["global_shape"],
            row["alias_group"][0],
            pattern,
            check=True,
        )


def checkpoint_bytes(path, *, canonical=False):
    files = {file.name: file.read_bytes() for file in sorted(path.glob("*.safetensors"))}
    if canonical:
        for name, data in files.items():
            size = struct.unpack("<Q", data[:8])[0]
            files[name] = canonical_safetensors_header(json.loads(data[8:8 + size])) + data[8 + size:]
    return files


def payload_extents(path):
    extents = []
    for file in sorted(path.glob("*.safetensors")):
        with file.open("rb") as stream:
            header_size = struct.unpack("<Q", stream.read(8))[0]
            header = json.loads(stream.read(header_size))
        for name, item in header.items():
            if name == "__metadata__":
                continue
            begin, end = item["data_offsets"]
            if end > begin:
                extents.append((file.name, 8 + header_size + begin, end - begin))
    return extents


def request_sequence(requests):
    return [
        (Path(request["file"]).name, request["file_offset"], request["length"])
        for request in requests
    ]


def assert_payload_ranges(sequence, extents):
    for file, start, size in sequence:
        assert size > 0
        assert any(
            file == saved_file and saved_start <= start and start + size <= saved_start + length
            for saved_file, saved_start, length in extents
        )
    by_file = {}
    for file, start, size in sequence:
        by_file.setdefault(file, []).append((start, start + size))
    for intervals in by_file.values():
        ordered = sorted(intervals)
        assert all(left[1] <= right[0] for left, right in zip(ordered, ordered[1:], strict=False))


@pytest.mark.parametrize("pattern", ["zero", "position"])
@pytest.mark.parametrize("bucketed", [False, True])
def test_cpu_meta_native_byte_identity_and_both_cross_loads(
    tmp_path, process_group, monkeypatch, pattern, bucketed
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H", raising=False)
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    original, rows = model_state(mesh, pattern=pattern, bucketed=bucketed)
    mapping, bucket_report = safetensors_file_buckets(rows, 1 if bucketed else None)
    native_path = tmp_path / "native"
    dcp.save(
        original,
        storage_writer=AdmittedHuggingFaceStorageWriter(
            native_path, sync_files=False, fqn_to_index_mapping=mapping
        ),
    )
    native_bytes = {}
    for name, data in checkpoint_bytes(native_path).items():
        header_size = struct.unpack("<Q", data[:8])[0]
        # The native payload is an independent compatibility oracle. Rust's
        # randomly ordered text-metadata map is canonicalized only in memory.
        header = canonical_safetensors_header(json.loads(data[8 : 8 + header_size]))
        native_bytes[name] = header + data[8 + header_size :]
    gdt_bytes = []
    native_manifest = validate_safetensors_checkpoint(native_path)
    assert native_bytes
    if bucketed:
        assert len(native_bytes) > 1
        assert bucket_report["largest_item_payload_bytes"] > 1024**2
    save_sequences = []
    load_sequences = []
    for device in ("cpu", "meta"):
        source, source_rows = model_state(
            mesh, device=device, pattern=pattern, bucketed=bucketed
        )
        path = tmp_path / device
        writer = GDTensorHuggingFaceStorageWriter(
            path, sync_files=False, fqn_to_index_mapping=mapping
        )
        writer.capture_payload_events = True
        save_planner = GDTensorSavePlanner(
            simulated=device == "meta", pattern=pattern, rows=source_rows
        )
        dcp.save(source, storage_writer=writer, planner=save_planner)
        actual = checkpoint_bytes(path)
        assert {name: hashlib.sha256(data).hexdigest() for name, data in actual.items()} == {
            name: hashlib.sha256(data).hexdigest() for name, data in native_bytes.items()
        }
        assert actual == native_bytes
        manifest = validate_safetensors_checkpoint(path)
        assert manifest["tensors"] == native_manifest["tensors"]
        assert manifest["world_size"] == native_manifest["world_size"]
        gdt_bytes.append(actual)
        assert not (path / ".metadata").exists()
        assert not (path / "model.safetensors.index.json").exists()
        assert not getattr(writer, "container_patch_events", [])
        sequence = request_sequence(writer.payload_requests)
        assert sequence
        # Only DTensor payloads pass through gdtensor; plain scalar/empty use Torch.
        expected_bytes = sum(
            local_tensor(tensor).nbytes
            for tensor in source["model"].values()
            if isinstance(tensor, DTensor)
        )
        assert sum(size for _, _, size in sequence) == expected_bytes
        assert_payload_ranges(sequence, payload_extents(path))
        save_sequences.append(sequence)

        # Native Torch HF load consumes the exact gdtensor-produced checkpoint.
        target, target_rows = model_state(mesh, pattern=pattern, bucketed=bucketed)
        for tensor in target["model"].values():
            local_tensor(tensor).fill_(-1)
        dcp.load(
            target,
            storage_reader=AdmittedHuggingFaceStorageReader(path, rows=target_rows),
        )
        assert_values(target, target_rows, pattern)

        # The gdtensor range reader consumes a checkpoint produced by native Torch.
        target, target_rows = model_state(
            mesh, device=device, pattern=pattern, bucketed=bucketed
        )
        if device == "cpu":
            for tensor in target["model"].values():
                local_tensor(tensor).fill_(-1)
        reader = GDTensorHuggingFaceStorageReader(native_path, rows=target_rows)
        recorder = IORecorder()
        reader.fs = RecordedFileSystem(reader.fs, recorder)
        planner = GDTensorLoadPlanner(
            target_rows, simulated=device == "meta", pattern=pattern
        )
        dcp.load(target, storage_reader=reader, planner=planner)
        planner.assert_complete()
        assert planner.completed_items == planner.expected_items
        if device == "cpu":
            assert_values(target, target_rows, pattern)
        else:
            assert all(local_tensor(tensor).is_meta for tensor in target["model"].values())
        sequence = request_sequence(reader.payload_requests)
        assert sequence
        assert sum(size for _, _, size in sequence) == expected_bytes
        assert_payload_ranges(sequence, payload_extents(native_path))
        assert recorder.summary()["files"]
        load_sequences.append(sequence)
    assert gdt_bytes[0] == gdt_bytes[1]
    assert save_sequences[0] == save_sequences[1]
    assert load_sequences[0] == load_sequences[1]




def test_meta_sim_default_profile_populates_and_times_each_safetensors_payload(
    tmp_path, process_group, monkeypatch
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_PROFILE", raising=False)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H_POPULATE", raising=False)
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    source, rows = model_state(mesh, device="meta", pattern="zero")
    writers = {}
    for mode in ("off", "pageable"):
        monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", mode)
        writer = GDTensorHuggingFaceStorageWriter(tmp_path / mode, sync_files=False)
        writer.capture_payload_events = True
        recorder = IORecorder()
        writer.fs = RecordedFileSystem(writer.fs, recorder)
        planner = GDTensorSavePlanner(simulated=True, pattern="zero", rows=rows)
        dcp.save(source, storage_writer=writer, planner=planner)
        writers[mode] = writer
    assert checkpoint_bytes(tmp_path / "off") == checkpoint_bytes(tmp_path / "pageable")
    assert validate_safetensors_checkpoint(tmp_path / "off") == validate_safetensors_checkpoint(
        tmp_path / "pageable"
    )
    assert request_sequence(writers["off"].payload_requests) == request_sequence(
        writers["pageable"].payload_requests
    )
    assert not [event for event in writers["off"].staging_events if event["op"] == "d2h"]
    events = [event for event in writers["pageable"].staging_events if event["op"] == "d2h"]
    expected = {
        f"model.{name}": local_tensor(tensor).nbytes
        for name, tensor in source["model"].items()
        if isinstance(tensor, DTensor) and local_tensor(tensor).nbytes
    }
    assert len(events) == len(expected)
    assert {event["tensor"]: event["requested"] for event in events} == expected
    profile = load_profile()
    for event in events:
        assert event["population"] == "non-temporal"
        assert event["timing"] == "simulated"
        assert event["completed"] == event["requested"]
        assert event["duration_ns"] == event["end_ns"] - event["start_ns"]
        assert event["duration_ns"] >= round(duration_seconds(profile, event["requested"]) * 1e9)
        assert event["success"] is True
        assert event["io_layer"] == "staging"
        assert event["request"] is False
        assert Path(event["file"]).suffix == ".safetensors"


def test_gdt_hf_rejects_non_model_state_before_completion(tmp_path):
    path = tmp_path / "checkpoint"
    writer = GDTensorHuggingFaceStorageWriter(path, sync_files=False)
    with pytest.raises(CheckpointException, match="model tensor entries only"):
        dcp.save(
            {"model": {"weight": torch.ones(3)}, "optimizer": {"step": torch.tensor(1)}},
            storage_writer=writer,
            no_dist=True,
        )
    assert not (path / COMPLETION_MARKER).exists()


def test_gdt_hf_records_payload_manifest_and_directory_fsync(tmp_path, process_group):
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    state, rows = model_state(mesh, pattern="zero")
    path = tmp_path / "checkpoint"
    recorder = IORecorder()
    writer = GDTensorHuggingFaceStorageWriter(path, sync_files=True)
    writer.fs = RecordedFileSystem(writer.fs, recorder, capture_lifecycle=True)
    dcp.save(state, storage_writer=writer, planner=GDTensorSavePlanner(rows=rows))
    manifest = validate_safetensors_checkpoint(path)
    fsyncs = [
        (file, event)
        for file, record in recorder.files.items()
        for event in record["events"]
        if event["op"] == "fsync"
    ]
    assert len(fsyncs) == len(manifest["files"]) + 2
    completion_events = [event for file, event in fsyncs if file == COMPLETION_MARKER]
    assert [event["target"] for event in completion_events] == [
        "completion_manifest", "checkpoint_directory"
    ]
    assert all(event["duration_ns"] >= 0 and event["success"] for _, event in fsyncs)
    assert all(event["request"] is False for _, event in fsyncs)


def distributed_hf_worker(rank, world, rendezvous, directory, action):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=Path(rendezvous).resolve().as_uri(), rank=rank, world_size=world,
        timeout=timedelta(seconds=60),
    )
    try:
        root = Path(directory)
        mesh = dist.device_mesh.init_device_mesh("cpu", (world,))
        shape = (3, 5)
        width = (shape[0] + world - 1) // world
        begin, end = min(rank * width, shape[0]), min((rank + 1) * width, shape[0])
        local = torch.empty((end - begin, shape[1]), dtype=torch.float32)
        tensor = DTensor.from_local(
            local, mesh, [Shard(0)], shape=torch.Size(shape), stride=(shape[1], 1),
            run_check=False,
        )
        state = {"model": {"weight": tensor}}
        row = manifest_row("model.weight", tensor, "model", ["weight"], "hf-unit-test")
        rows = [row]
        if action == "save":
            fill_or_check(local, row["global_offsets"], row["global_shape"], "weight", "position")
            for backend, cls in (
                ("native", AdmittedHuggingFaceStorageWriter),
                ("gdtensor", GDTensorHuggingFaceStorageWriter),
            ):
                dcp.save(
                    state, storage_writer=cls(root / backend, sync_files=False),
                    planner=GDTensorSavePlanner(pattern="position", rows=rows)
                    if backend == "gdtensor" else None,
                )
            dist.barrier()
            if rank == 0:
                assert checkpoint_bytes(root / "native", canonical=True) == checkpoint_bytes(root / "gdtensor")
                native_manifest = validate_safetensors_checkpoint(root / "native")
                gdt_manifest = validate_safetensors_checkpoint(root / "gdtensor")
                assert native_manifest["tensors"] == gdt_manifest["tensors"]
                assert native_manifest["world_size"] == gdt_manifest["world_size"]
            return
        for producer, reader_cls in (
            ("native", GDTensorHuggingFaceStorageReader),
            ("gdtensor", AdmittedHuggingFaceStorageReader),
        ):
            local.fill_(-999)
            reader = reader_cls(root / producer, rows=rows)
            planner = (
                GDTensorLoadPlanner(rows, pattern="position") if producer == "native" else None
            )
            dcp.load(state, storage_reader=reader, planner=planner)
            fill_or_check(
                local, row["global_offsets"], row["global_shape"], "weight", "position", check=True
            )
            if planner is not None:
                planner.assert_complete()
                sequence = request_sequence(reader.payload_requests)
                assert sum(size for _, _, size in sequence) == local.nbytes
                assert_payload_ranges(sequence, payload_extents(root / producer))
        (root / f"load-rank-{rank}.json").write_text(
            json.dumps(dict(rank=rank, local_shape=list(local.shape), payload_bytes=local.nbytes))
        )
    finally:
        dist.destroy_process_group()


def test_distributed_hf_cross_load_reshards_uneven_source_to_empty_tail_target(tmp_path):
    # Three rows split as (2, 1) when saving, then (1, 1, 1, 0) when loading.
    for action, world in (("save", 2), ("load", 4)):
        mp.spawn(
            distributed_hf_worker,
            args=(world, str(tmp_path / f"rendezvous-{action}"), str(tmp_path), action),
            nprocs=world, join=True,
        )
    evidence = [json.loads((tmp_path / f"load-rank-{rank}.json").read_text()) for rank in range(4)]
    assert [item["local_shape"] for item in evidence] == [[1, 5], [1, 5], [1, 5], [0, 5]]
    assert sum(item["payload_bytes"] for item in evidence) == 3 * 5 * 4


def backend_args(**overrides):
    values = dict(
        io_backend="gdtensor",
        checkpoint_format="safetensors",
        checkpoint_state="model",
        action="save",
        sharding="fsdp2",
        threads=1,
        dcp_file_layout="rank",
        standard_device="cpu",
        checkpoint=Path("checkpoint"),
        data_pattern="position",
        parallel="dp_shard=2",
        world_size=2,
    )
    return SimpleNamespace(**(values | overrides))


@pytest.mark.parametrize("action", ["save", "load"])
@pytest.mark.parametrize("pattern", ["zero", "position"])
def test_gdt_safetensors_options_accept_pure_fsdp_model_io(action, pattern):
    validate_backend(backend_args(action=action, data_pattern=pattern))


@pytest.mark.parametrize("action", ["save", "load"])
def test_gdt_hybrid_training_state_is_rejected_for_both_actions(action):
    with pytest.raises(ValueError, match="gdtensor.*model-only"):
        validate_backend(backend_args(action=action, checkpoint_state="training"))


@pytest.mark.parametrize(
    "overrides",
    [
        {"action": "train"},
        {"sharding": "derived"},
        {"threads": 2},
        {"dcp_file_layout": "item"},
        {"parallel": "tp=2"},
        {"parallel": "dp_replicate=2"},
        {"parallel": "dp_shard=3"},
        {"standard_device": "mps"},
        {"data_pattern": "random"},
        {"checkpoint": "hdfs://cluster/checkpoint"},
    ],
)
def test_gdt_safetensors_options_preserve_backend_restrictions(overrides):
    with pytest.raises(ValueError, match="gdtensor"):
        validate_backend(backend_args(**overrides))
