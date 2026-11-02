"""Meta H2D placement and correctness through both supported containers."""

import sys

import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from metasl import meta_d2h_cache  # noqa: E402
from metasl.gdt_hf_storage import (  # noqa: E402
    GDTensorHuggingFaceStorageReader,
    GDTensorHuggingFaceStorageWriter,
)
from metasl.gdt_storage import (  # noqa: E402
    GDTensorFileSystemReader,
    GDTensorFileSystemWriter,
    GDTensorLoadPlanner,
    GDTensorSavePlanner,
)
from metasl.iostats import IORecorder, RecordedFileSystem  # noqa: E402
from metasl.sharding import fill_or_check, manifest_row  # noqa: E402


@pytest.mark.parametrize("format_name", ["dcp", "safetensors"])
@pytest.mark.parametrize("pattern", ["zero", "position"])
def test_meta_sim_h2d_both_formats_without_loading_d2h_helper(
    tmp_path, process_group, monkeypatch, format_name, pattern
):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", "off")
    mesh = dist.device_mesh.init_device_mesh("cpu", (1,))

    def state(device):
        tensors, rows = {}, []
        for name, shape, dtype in (
            ("weight", (17, 5), torch.float32),
            ("bias", (23,), torch.bfloat16),
        ):
            local = torch.empty(shape, dtype=dtype, device=device)
            tensor = DTensor.from_local(local, mesh, [Shard(0)], run_check=False)
            row = manifest_row(f"model.{name}", tensor, "model", [name], "h2d-unit-test")
            rows.append(row)
            if device == "cpu":
                fill_or_check(local, row["global_offsets"], row["global_shape"], name, pattern)
            tensors[name] = tensor
        return {"model": tensors}, rows

    source, rows = state("cpu")
    path = tmp_path / "checkpoint"
    writer_type, reader_type = (
        (GDTensorFileSystemWriter, GDTensorFileSystemReader)
        if format_name == "dcp"
        else (GDTensorHuggingFaceStorageWriter, GDTensorHuggingFaceStorageReader)
    )
    dcp.save(
        source,
        storage_writer=writer_type(path, sync_files=False),
        planner=GDTensorSavePlanner(rows=rows, pattern=pattern),
    )

    def forbidden(*_):
        pytest.fail("A load-only process must not prepare or load the D2H helper")

    monkeypatch.setattr(meta_d2h_cache, "population_function", forbidden)
    monkeypatch.setattr(meta_d2h_cache, "_native_library", forbidden)
    sequences = []
    for mode in ("off", "pageable"):
        monkeypatch.setenv("METASL_GDTENSOR_META_SIM_D2H", mode)
        target, target_rows = state("meta")
        options = {"rows": target_rows} if format_name == "safetensors" else {}
        reader = reader_type(path, **options)
        reader.fs = RecordedFileSystem(reader.fs, IORecorder())
        planner = GDTensorLoadPlanner(target_rows, simulated=True, pattern=pattern)
        dcp.load(target, storage_reader=reader, planner=planner)
        planner.assert_complete()
        assert planner.completed_items == planner.expected_items
        assert all(tensor.to_local().is_meta for tensor in target["model"].values())
        sequences.append(reader.payload_requests)
        events = reader.staging_events
        copies = [event for event in events if event["op"] == "h2d"]
        if mode == "off":
            assert copies == []
            continue
        expected = {
            f"model.{name}": tensor.to_local().nbytes for name, tensor in source["model"].items()
        }
        assert {event["tensor"]: event["requested"] for event in copies} == expected
        for copy in copies:
            sinks = [
                event
                for event in events
                if event["op"] == "host_sink" and event["tensor"] == copy["tensor"]
            ]
            assert len(sinks) == 1 and sinks[0]["end_ns"] <= copy["start_ns"]
            assert copy["timing"] == "simulated" and copy["success"]
            assert copy["requested"] == copy["completed"]
            assert not copy["request"] and copy["io_layer"] == "staging"
    assert sequences[0] == sequences[1]
