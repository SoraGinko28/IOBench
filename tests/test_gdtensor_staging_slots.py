"""Destination staging labels survive per-file publication on multiple ranks."""

import ctypes
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.multiprocessing as mp
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
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.sharding import manifest_row  # noqa: E402


@pytest.mark.parametrize("rank", [0, 1])
def test_multi_slot_labels_use_local_request_interval_and_file_offset(monkeypatch, rank):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "off")
    local = torch.empty(16, device="meta")
    tensor = SimpleNamespace(shape=(32,), dtype=local.dtype, to_local=lambda: local)
    monkeypatch.setattr(
        gdt_storage, "_compute_geometry",
        lambda _: gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 2),
    )
    slots = [
        gdt_storage.PayloadSlot("weight", column, f"rank-{column}.distcp", 64, column * 64,
                                file_offset=1024 + column * 4096)
        for column in range(2)
    ]
    events = []
    accessor = gdt_storage.ComputeAccessor(
        {"weight": tensor}, rank, staging_events=events, staging_slots=slots
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=rank * 64 + 8, size=16)

    def give(pointer, length):
        ctypes.memset(pointer, 0, length)
        return Ok(None)

    accessor.write_from_buffer(rng, give)
    assert accessor.requests == [("weight", rank * 64 + 8, 16)]
    assert {event["op"] for event in events} == {"meta_generate", "host_sink"}
    assert all(event["file"] == slots[rank].file for event in events)
    assert all(event["offset"] == slots[rank].file_offset + 8 for event in events)


def test_resharded_callback_relative_chunks_use_saved_grid_not_compute_rank(monkeypatch):
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "off")
    local = torch.empty(16, device="meta")
    tensor = SimpleNamespace(shape=(16,), dtype=local.dtype, to_local=lambda: local)
    monkeypatch.setattr(
        gdt_storage, "_compute_geometry",
        lambda _: gdt_storage.TensorGeometry(tensor.shape, tensor.dtype, 1),
    )
    slots = [
        gdt_storage.PayloadSlot("weight", 0, "saved-0.distcp", 32, 0, file_offset=1024),
        gdt_storage.PayloadSlot("weight", 1, "saved-1.distcp", 32, 32, file_offset=2048),
    ]
    events = []
    accessor = gdt_storage.ComputeAccessor(
        {"weight": tensor}, 0, staging_events=events, staging_slots=slots
    )
    rng = SimpleNamespace(global_tensor_id="weight", global_tensor_offset=0, size=64)
    with accessor._stage(rng, "host_sink", 16, 8):
        pass
    with accessor._stage(rng, "host_sink", 16, 40):
        pass
    assert [(event["file"], event["offset"]) for event in events] == [
        ("saved-0.distcp", 1032), ("saved-1.distcp", 2056)
    ]
    assert accessor.requests == []  # Trace labeling introduces no data requests.


def test_load_session_publishes_all_sink_spans_without_opening_saved_files():
    events = []
    for name, offset in [("opened.distcp", 1032), ("other-rank.distcp", 2056)]:
        with gdt_storage._staging_event(events, "host_sink", name, offset, 16):
            pass
    origin = min(event["start_ns"] for event in events) - 100
    stream = SimpleNamespace(recorder=SimpleNamespace(origin_ns=origin), events=[])
    reader = SimpleNamespace(staging_events=events, abort=None, capture_payload_events=True)
    session = gdt_storage._LoadSession(reader, [])
    session.streams["opened.distcp"] = stream
    session.finish()
    session.finish()
    assert len(stream.events) == 2
    assert [(event["file"], event["offset"]) for event in stream.events] == [
        ("opened.distcp", 1032), ("other-rank.distcp", 2056)
    ]
    assert all(event["request"] is False for event in stream.events)
    assert [event["start_ns"] for event in stream.events] == [
        event["start_ns"] - origin for event in events
    ]
    assert list(session.streams) == ["opened.distcp"]


def _two_rank_staging_worker(rank, directory):
    torch.set_num_threads(1)
    os.environ["METASL_GDTENSOR_META_SIM_D2H"] = "off"
    os.environ["METASL_GDTENSOR_META_LOAD_SINK"] = "faithful"
    root = Path(directory)
    dist.init_process_group(
        "gloo", init_method=(root / "rendezvous").resolve().as_uri(), rank=rank,
        world_size=2, timeout=timedelta(seconds=60),
    )
    try:
        mesh = dist.device_mesh.init_device_mesh("cpu", (2,))

        def state(device):
            local = torch.zeros((5, 3), dtype=torch.float32, device=device)
            tensor = DTensor.from_local(
                local, mesh, [Shard(0)], shape=torch.Size((10, 3)), stride=(3, 1), run_check=False
            )
            row = manifest_row("model.weight", tensor, "model", ["weight"], "staging-slots-test")
            return {"model": {"weight": tensor}}, [row]

        evidence = []
        for format_name, writer_type, reader_type in [
            ("dcp", gdt_storage.GDTensorFileSystemWriter, gdt_storage.GDTensorFileSystemReader),
            ("safetensors", GDTensorHuggingFaceStorageWriter, GDTensorHuggingFaceStorageReader),
        ]:
            path = root / format_name
            source, rows = state("cpu")
            dcp.save(
                source, storage_writer=writer_type(path, sync_files=False),
                planner=gdt_storage.GDTensorSavePlanner(rows=rows),
            )
            sequences = []
            for device in ["cpu", "meta"]:
                target, rows = state(device)
                reader = reader_type(path, **({"rows": rows} if format_name == "safetensors" else {}))
                recorder = IORecorder()
                reader.fs = RecordedFileSystem(reader.fs, recorder)
                planner = gdt_storage.GDTensorLoadPlanner(rows, simulated=device == "meta")
                planner.prepare_load(flatten_state_dict(target)[0])
                dcp.load(target, storage_reader=reader, planner=planner)
                planner.assert_complete()
                recorder.summary()  # Unpack metadata events before inspecting published calls.
                recorded = [event for entry in recorder.files.values() for event in entry["events"]
                            if event["op"] == "host_sink"]
                assert len(recorded) == 1
                sink = recorded[0]
                requests = reader.payload_requests
                assert len(requests) == 1
                request = requests[0]
                assert sink["file"] == request["file"]
                assert sink["offset"] == request["file_offset"]
                assert sink["requested"] == request["length"] == 60
                assert len(reader.staging_events) == 3  # storage copy, materialize, final sink
                sequences.append(requests)
                evidence.append(dict(format=format_name, device=device, rank=rank,
                                     file=sink["file"], offset=sink["offset"], bytes=60))
            assert sequences[0] == sequences[1]
        (root / f"evidence-rank-{rank}.json").write_text(json.dumps(evidence))
    finally:
        dist.destroy_process_group()


def test_two_rank_both_format_sink_events_publish_for_cpu_and_meta(tmp_path):
    mp.spawn(_two_rank_staging_worker, args=(str(tmp_path),), nprocs=2, join=True)
    evidence = [json.loads((tmp_path / f"evidence-rank-{rank}.json").read_text()) for rank in range(2)]
    assert all(len(rank_events) == 4 for rank_events in evidence)
