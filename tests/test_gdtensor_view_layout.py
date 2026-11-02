"""Compact public storage meshes preserve file slots and actual gdtensor IO."""

import dataclasses
import math
import sys

import pytest
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Shard

if sys.version_info < (3, 12):
    pytest.skip("Optional gdtensor backend requires Python 3.12", allow_module_level=True)
pytest.importorskip("gdtensor", reason="Optional gdtensor backend is not installed")

import gdtensor.builder as gb  # noqa: E402
from gdtensor.io import rank_load, rank_save  # noqa: E402
from gdtensor.session import StorageLoadSession, StorageSaveSession  # noqa: E402

from metasl.gdt_storage import (  # noqa: E402
    ComputeAccessor,
    PayloadSlot,
    StorageAccessor,
    TensorGeometry,
    assign_payload_offsets,
    make_views,
    shard_bounds,
)


@pytest.fixture
def local_mesh():
    owns_group = not dist.is_initialized()
    if owns_group:
        dist.init_process_group("fake", rank=0, world_size=1)
    elif dist.get_world_size() != 1:
        pytest.skip("Layout regressions require a one-rank compute process group")
    try:
        yield dist.device_mesh.init_device_mesh("cpu", (1,))
    finally:
        if owns_group:
            dist.destroy_process_group()


class MemorySession(StorageLoadSession, StorageSaveSession):
    def __init__(self, accessor, slots):
        self.accessor, self.slots = accessor, slots

    def init_data_accessor(self):
        return gb.Ok(self.accessor)

    def get_file_offset(self, position):
        return gb.Ok(next(
            slot.file_offset for slot in self.slots
            if slot.name == position.tensor_id and slot.column == position.rank
        ))

    def finish(self):
        return gb.Ok(None)

    def commit(self):
        return gb.Ok(None)

    def abort(self):
        return gb.Ok(None)


def state_and_slots(mesh, *, mixed):
    names = ["z", "a", "m", "b", "q", "c"] if mixed else [f"weight-{i}" for i in range(64)]
    shapes = [(5, 3), (2, 3), (0, 3), (7, 3), (1, 3), (9, 3)] if mixed else [(35, 2)] * len(names)
    columns = [4, 3, 1, 4, 2, 4] if mixed else [16] * len(names)
    tensors, expected, geometries, slots = {}, {}, {}, []
    for row, (name, shape, world) in enumerate(zip(names, shapes, columns, strict=True)):
        values = torch.arange(math.prod(shape), dtype=torch.float32) + 100 * row
        expected[name] = values.reshape(shape)
        local = torch.full(shape, -1.0)
        tensors[name] = DTensor.from_local(
            local, mesh, [Shard(0)], shape=shape, stride=local.stride(), run_check=False
        )
        geometries[name] = TensorGeometry(shape, torch.float32, world)
        for column in range(world):
            lo, hi = shard_bounds(shape, world, column)
            row_bytes = math.prod(shape[1:]) * 4
            file = f"file-{(row + column) % 3 if mixed else column}.distcp"
            slots.append(PayloadSlot(name, column, file, (hi - lo) * row_bytes, lo * row_bytes))
    assign_payload_offsets(slots, names)
    payloads = {}
    for slot in slots:
        slot.file_offset = slot.payload_offset
        data = expected[slot.name].reshape(-1).view(torch.uint8).numpy().tobytes()
        payloads.setdefault(slot.file, bytearray()).extend(
            data[slot.global_offset : slot.global_offset + slot.size]
        )
    return tensors, expected, geometries, slots, payloads


@pytest.mark.parametrize("mixed", [False, True])
def test_compact_meshes_preserve_public_file_mapping_and_payload_roundtrip(local_mesh, mixed):
    tensors, expected, geometries, slots, payloads = state_and_slots(local_mesh, mixed=mixed)
    reads, writes = [], []
    saved = {path: bytearray(len(data)) for path, data in payloads.items()}

    def read(path, offset, length):
        reads.append((path, offset, length))
        return bytes(payloads[path][offset : offset + length])

    def write(path, offset, data):
        writes.append((path, offset, len(data)))
        saved[path][offset : offset + len(data)] = data

    accessor = StorageAccessor(slots, read=read, write=write)
    session = MemorySession(accessor, slots)
    compute, storage = make_views(
        tensors, geometries, slots, ComputeAccessor(tensors, 0),
        begin_load=lambda view, rank: gb.Ok(session),
        begin_save=lambda view, rank: gb.Ok(session),
    )
    assert type(storage.check_constraints()) is gb.Ok
    layout = storage.physical_layout
    for slot in slots:
        result = gb.file_path_of(layout, gb.TensorComponentPos(slot.name, slot.column))
        assert type(result) is gb.Ok and result.value == slot.file
    if not mixed:
        for leader in layout.file_paths:
            group = layout.tensor_entries[leader]
            assert len(group.disjoint_storage_mesh_list) == 1
            assert gb.leader_tensor_component_pos_of(group) == leader
    assert type(rank_load(0, compute, storage)) is gb.Ok
    for name, tensor in tensors.items():
        assert torch.equal(tensor.to_local(), expected[name]), name
    assert type(rank_save(0, compute, storage)) is gb.Ok
    assert saved == payloads
    assert sorted(reads) == sorted(writes)
    assert sum(length for _, _, length in reads) == sum(len(data) for data in payloads.values())


@pytest.mark.parametrize("damage", ["offset", "duplicate"])
def test_compaction_keeps_slot_validation(local_mesh, damage):
    tensors, _, geometries, slots, _ = state_and_slots(local_mesh, mixed=True)
    if damage == "offset":
        slots[0] = dataclasses.replace(slots[0], global_offset=4)
    else:
        slots.append(dataclasses.replace(slots[0]))
    with pytest.raises(ValueError, match="Shard|Duplicate"):
        make_views(tensors, geometries, slots, ComputeAccessor(tensors, 0))
