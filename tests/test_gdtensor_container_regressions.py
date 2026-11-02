"""Native container compatibility using local IO and a one-rank fake process group."""

import io
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.checkpoint import DefaultSavePlanner
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

from metasl.gdt_storage import (  # noqa: E402
    GDTensorFileSystemReader,
    GDTensorFileSystemWriter,
    GDTensorLoadPlanner,
    GDTensorSavePlanner,
    read_container,
)
from metasl.storage import AdmittedFileSystemWriter  # noqa: E402


@pytest.fixture
def local_mesh():
    """Avoid network collectives; reuse an existing one-rank group in a combined run."""
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("fake", rank=0, world_size=1)
    elif dist.get_world_size() != 1:
        pytest.skip("Local container regressions require a one-rank process group")
    try:
        yield dist.device_mesh.init_device_mesh("cpu", (1,))
    finally:
        if owns_group:
            dist.destroy_process_group()


def write_local_checkpoint(path, state, gdtensor):
    writer_class = GDTensorFileSystemWriter if gdtensor else AdmittedFileSystemWriter
    writer = writer_class(path, thread_count=1, sync_files=False)
    planner = GDTensorSavePlanner() if gdtensor else DefaultSavePlanner()
    writer.set_up_storage_writer(is_coordinator=True, rank=0)
    planner.set_up_planner(state, is_coordinator=True)
    local_plan = writer.prepare_local_plan(planner.create_local_plan())
    plans, metadata = planner.create_global_plan([local_plan])
    plan = planner.finish_plan(writer.prepare_global_plan(plans)[0])
    results = writer.write_data(plan, planner).wait()
    writer.finish(metadata, [results])
    return metadata


@pytest.mark.parametrize(
    "shape,stride",
    [((4, 1), (1, 100)), ((3, 1), (1, 0)), ((0, 4), (100, 1)), ((4, 0), (1, 100))],
)
def test_contiguous_strides_preserve_native_writer_bytes(tmp_path, local_mesh, shape, stride):
    local = torch.empty_strided(shape, stride, dtype=torch.float32)
    local.copy_(torch.arange(local.numel(), dtype=local.dtype).reshape(shape))
    tensor = DTensor.from_local(
        local, local_mesh, [Shard(0)], shape=shape, stride=stride, run_check=False
    )
    assert tensor.to_local().is_contiguous()
    state = {"weight": tensor, "step": torch.tensor(7), "bytes": b"native-byte-entry"}
    native = write_local_checkpoint(tmp_path / "native", state, gdtensor=False)
    gdtensor = write_local_checkpoint(tmp_path / "gdtensor", state, gdtensor=True)

    assert (tmp_path / "native" / "__0_0.distcp").read_bytes() == (
        tmp_path / "gdtensor" / "__0_0.distcp"
    ).read_bytes()
    assert native.state_dict_metadata == gdtensor.state_dict_metadata
    assert native.storage_data == gdtensor.storage_data

    destination = DTensor.from_local(
        torch.full(shape, -1.0),
        local_mesh,
        [Shard(0)],
        shape=shape,
        stride=torch.empty(shape).stride(),
        run_check=False,
    )
    reader = GDTensorFileSystemReader(tmp_path / "native")
    metadata = reader.read_metadata()
    reader.set_up_storage_reader(metadata, is_coordinator=True, rank=0)
    planner = GDTensorLoadPlanner()
    planner.set_up_planner({"weight": destination}, metadata, is_coordinator=True)
    local_plan = reader.prepare_local_plan(planner.create_local_plan())
    plans = reader.prepare_global_plan(planner.create_global_plan([local_plan]))
    reader.read_data(planner.finish_plan(plans[0]), planner).wait()
    planner.assert_complete()
    assert torch.equal(destination.to_local(), local)


@pytest.mark.parametrize(
    "shape,stride",
    [((4, 1), (1, 100)), ((3, 1), (1, 0)), ((0, 4), (100, 1)), ((4, 0), (1, 100))],
)
def test_native_container_parser_accepts_contiguous_strides(shape, stride):
    tensor = torch.empty_strided(shape, stride, dtype=torch.float32)
    stream = io.BytesIO()
    torch.save(tensor, stream)
    assert 0 < read_container(stream, 0, len(stream.getvalue()), shape, tensor.dtype)
