"""Native shared-file bytes, independent compute meshes, and HF split indices."""

import json
import struct
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
from safetensors.torch import load_file, save_file
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):  # noqa: UP036
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor")

from metasl.gdt_shared_hf_storage import (  # noqa: E402
    SharedGDTensorHuggingFaceStorageReader,
    SharedGDTensorHuggingFaceStorageWriter,
    shared_safetensors_layout,
)
from metasl.gdt_storage import GDTensorLoadPlanner, GDTensorSavePlanner  # noqa: E402
from metasl.hf_storage import validate_safetensors_checkpoint  # noqa: E402
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.sharding import fill_or_check, local_tensor, manifest_row  # noqa: E402

SPECS = {
    "model.layers.0.weight": ((7, 3), torch.float32),
    "lm_head.weight": ((3, 5), torch.bfloat16),
    "zero": ((0, 2), torch.float16),
    "scalar": ((), torch.int64),
}


def _state(mesh, device="cpu"):
    rank, world = dist.get_rank(), dist.get_world_size()
    values, rows = {}, []
    for name, (shape, dtype) in SPECS.items():
        if shape and shape[0]:
            width = (shape[0] + world - 1) // world
            begin, end = min(rank * width, shape[0]), min((rank + 1) * width, shape[0])
            local = torch.empty((end - begin, *shape[1:]), dtype=dtype, device=device)
            stride = tuple(torch.empty(shape, device="meta").stride())
            value = DTensor.from_local(
                local,
                mesh,
                [Shard(0)],
                shape=torch.Size(shape),
                stride=stride,
                run_check=False,
            )
        else:
            value = torch.empty(shape, dtype=dtype, device=device)
        row = manifest_row(f"model.{name}", value, "model", [name], "shared-unit")
        values[name] = value
        rows.append(row)
        if device != "meta":
            fill_or_check(local_tensor(value), row["global_offsets"], list(shape), name, "position")
    return {"model": values}, rows


def _reference():
    values = {}
    for name, (shape, dtype) in SPECS.items():
        value = torch.empty(shape, dtype=dtype)
        fill_or_check(value, [0] * len(shape), list(shape), name, "position")
        values[name] = value
    return values


def _assert_native(path, reference_dir):
    expected = _reference()
    reference_dir.mkdir()
    index_path = path / "model.safetensors.index.json"
    weight_map = (
        json.loads(index_path.read_text())["weight_map"]
        if index_path.exists()
        else {name: "model.safetensors" for name in expected}
    )
    assert set(weight_map) == set(expected)
    for file in sorted(set(weight_map.values())):
        bucket = {name: value for name, value in expected.items() if weight_map[name] == file}
        save_file(bucket, reference_dir / file)
        assert (path / file).read_bytes() == (reference_dir / file).read_bytes()
        loaded = load_file(path / file)
        assert all(torch.equal(loaded[name], value) for name, value in bucket.items())
        raw = (path / file).read_bytes()
        header = json.loads(raw[8 : 8 + struct.unpack("<Q", raw[:8])[0]])
        assert "__metadata__" not in header


@pytest.mark.parametrize("device,sink", [("cpu", "fresh"), ("meta", "fresh"), ("meta", "faithful")])
def test_shared_single_rank_native_bytes_and_meta_sinks(
    tmp_path, process_group, monkeypatch, device, sink
):
    monkeypatch.setenv("METASL_GDTENSOR_META_LOAD_SINK", sink)
    monkeypatch.delenv("METASL_GDTENSOR_META_SIM_D2H", raising=False)
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    state, rows = _state(mesh, device)
    path = tmp_path / "checkpoint"
    writer = SharedGDTensorHuggingFaceStorageWriter(path, sync_files=False)
    recorder = IORecorder()
    writer.fs = RecordedFileSystem(writer.fs, recorder, capture_lifecycle=True)
    writer.capture_payload_events = True
    save = GDTensorSavePlanner(simulated=device == "meta", pattern="position", rows=rows)
    save.prepare_save({f"model.{name}": value for name, value in state["model"].items()})
    dcp.save(state, storage_writer=writer, planner=save)
    assert writer.layout_report["files"][0]["path"] == "model.safetensors"
    assert validate_safetensors_checkpoint(path)["safetensors_layout"] == "shared"
    _assert_native(path, tmp_path / "reference")
    events = recorder.files["model.safetensors"]["events"]
    assert [event["stage"] for event in events if event["op"] == "barrier"] == ["publish", "ready"]
    assert any(event["op"] == "ftruncate" for event in events)
    target, target_rows = _state(mesh, device)
    if device == "cpu":
        for value in target["model"].values():
            local_tensor(value).zero_()
    reader = SharedGDTensorHuggingFaceStorageReader(path, rows=target_rows)
    reads = IORecorder()
    reader.fs = RecordedFileSystem(reader.fs, reads, capture_lifecycle=True)
    load = GDTensorLoadPlanner(target_rows, simulated=device == "meta", pattern="position")
    load.prepare_load({f"model.{name}": value for name, value in target["model"].items()})
    dcp.load(target, storage_reader=reader, planner=load)
    load.assert_complete()
    if device == "cpu":
        for row in target_rows:
            name = row["fqn"].removeprefix("model.")
            fill_or_check(
                local_tensor(target["model"][name]),
                row["global_offsets"],
                row["global_shape"],
                name,
                "position",
                check=True,
            )
    events = reads.files["model.safetensors"]["events"]
    assert [event["requested"] for event in events if event["op"] == "read"] == [
        8,
        writer.layout_report["files"][0]["header_bytes"] - 8,
    ]
    assert any(event["op"] == "pread" for event in events)
    assert sum(event["op"] == "close" for event in events) == 1


def test_shared_whole_tensor_split_index_is_native_and_validated(tmp_path, process_group):
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))
    state, rows = _state(mesh)
    path = tmp_path / "split"
    limit = 40 / 1024**3
    report = shared_safetensors_layout(rows, limit)
    assert [row["data_bytes"] for row in report["files"]] == [30, 84, 8]
    writer = SharedGDTensorHuggingFaceStorageWriter(path, max_file_gib=limit, sync_files=True)
    recorder = IORecorder()
    writer.fs = RecordedFileSystem(writer.fs, recorder, capture_lifecycle=True, capture_open=True)
    dcp.save(
        state, storage_writer=writer, planner=GDTensorSavePlanner(rows=rows, pattern="position")
    )
    _assert_native(path, tmp_path / "reference")
    index_path = path / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    assert index["metadata"]["total_size"] == 122
    assert sorted(set(index["weight_map"].values())) == [
        f"model-{rank:05d}-of-00003.safetensors" for rank in range(1, 4)
    ]
    events = recorder.summary()["files"][index_path.name]["calls"]
    assert events["open"] == events["write"] == events["fsync"] == events["close"] == 1
    validate_safetensors_checkpoint(path)
    index["weight_map"]["lm_head.weight"] = "../outside.safetensors"
    index_path.write_text(json.dumps(index))
    with pytest.raises(ValueError, match="index disagrees"):
        validate_safetensors_checkpoint(path)


def _distributed_worker(rank, world, rendezvous, directory, action):
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
        state, rows = _state(mesh)
        path = Path(directory) / "checkpoint"
        if action == "save":
            writer = SharedGDTensorHuggingFaceStorageWriter(path, sync_files=False)
            dcp.save(
                state,
                storage_writer=writer,
                planner=GDTensorSavePlanner(rows=rows, pattern="position"),
            )
            if rank == 0:
                _assert_native(path, Path(directory) / "reference")
        else:
            for value in state["model"].values():
                local_tensor(value).zero_()
            reader = SharedGDTensorHuggingFaceStorageReader(path, rows=rows)
            planner = GDTensorLoadPlanner(rows=rows, pattern="position")
            dcp.load(state, storage_reader=reader, planner=planner)
            planner.assert_complete()
            for row in rows:
                fill_or_check(
                    local_tensor(state["model"][_name := row["fqn"].removeprefix("model.")]),
                    row["global_offsets"],
                    row["global_shape"],
                    _name,
                    "position",
                    check=True,
                )
            (Path(directory) / f"load-{rank}.json").write_text(
                json.dumps(
                    {
                        name: list(local_tensor(value).shape)
                        for name, value in state["model"].items()
                    }
                )
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("save_world,load_world", [(2, 4), (4, 2)])
def test_shared_different_world_size_uneven_and_empty_target(tmp_path, save_world, load_world):
    for action, world in (("save", save_world), ("load", load_world)):
        mp.spawn(
            _distributed_worker,
            args=(world, str(tmp_path / f"store-{action}"), str(tmp_path), action),
            nprocs=world,
            join=True,
        )
    if load_world == 4:
        assert json.loads((tmp_path / "load-3.json").read_text())["lm_head.weight"] == [0, 5]


@pytest.mark.parametrize("limit", [0, -1, True, float("inf"), float("nan")])
def test_shared_rejects_invalid_file_limit(limit):
    with pytest.raises(ValueError, match="positive finite"):
        shared_safetensors_layout([], limit)
