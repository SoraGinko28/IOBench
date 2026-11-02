"""Public-API probes for the gdtensor payload-coordinate contract."""

from __future__ import annotations

import ctypes
import json
from dataclasses import dataclass

import gdtensor.builder as gb
from gdtensor.access import DTensorDataAccessor
from gdtensor.builder import Ok
from gdtensor.io import rank_load, rank_save
from gdtensor.session import StorageLoadSession, StorageSaveSession
from gdtensor.tag import ConstantOne, TensorCount, WorldSize


def checked(result):
    assert type(result) is Ok, result
    return result.value


def chunk_bounds(size, columns, rank):
    width = (size + columns - 1) // columns
    lo = min(rank * width, size)
    return lo, min(lo + width, size)


@dataclass
class Slot:
    tensor_id: str
    rank: int
    payload_offset: int
    global_offset: int
    size: int


class ComputeAccessor(DTensorDataAccessor):
    def __init__(self, values):
        self.values = values
        self.requests = []

    def read_to_buffer(self, rng, callback):
        data = self.values[rng.global_tensor_id]
        offset = rng.global_tensor_offset
        self.requests.append((rng.global_tensor_id, offset, rng.size))
        buffer = ctypes.create_string_buffer(bytes(data[offset : offset + rng.size]))
        return callback(ctypes.addressof(buffer), rng.size)

    def write_from_buffer(self, rng, callback):
        buffer = ctypes.create_string_buffer(rng.size)
        checked(callback(ctypes.addressof(buffer), rng.size))
        offset = rng.global_tensor_offset
        self.values[rng.global_tensor_id][offset : offset + rng.size] = buffer.raw
        self.requests.append((rng.global_tensor_id, offset, rng.size))
        return Ok(None)


class StorageAccessor(DTensorDataAccessor):
    def __init__(self, payloads, slots):
        self.payloads = payloads
        self.slots = slots
        self.requests = []

    def locate(self, rng):
        matches = [
            slot
            for slot in self.slots[rng.file_path]
            if slot.payload_offset <= rng.contiguous_payload_offset
            and rng.contiguous_payload_offset + rng.size <= slot.payload_offset + slot.size
        ]
        assert len(matches) == 1, (rng, matches)
        slot = matches[0]
        self.requests.append(
            (
                rng.file_path,
                rng.contiguous_payload_offset,
                rng.size,
                slot.tensor_id,
                slot.rank,
                slot.global_offset + rng.contiguous_payload_offset - slot.payload_offset,
            )
        )
        return self.payloads[rng.file_path], rng.contiguous_payload_offset

    def read_to_buffer(self, rng, callback):
        data, offset = self.locate(rng)
        buffer = ctypes.create_string_buffer(bytes(data[offset : offset + rng.size]))
        return callback(ctypes.addressof(buffer), rng.size)

    def write_from_buffer(self, rng, callback):
        data, offset = self.locate(rng)
        buffer = ctypes.create_string_buffer(rng.size)
        checked(callback(ctypes.addressof(buffer), rng.size))
        data[offset : offset + rng.size] = buffer.raw
        return Ok(None)


class ProbeSession(StorageSaveSession, StorageLoadSession):
    def __init__(self, accessor):
        self.accessor = accessor
        self.closed = False

    def get_file_offset(self, pos):
        for slots in self.accessor.slots.values():
            for slot in slots:
                if (slot.tensor_id, slot.rank) == (pos.tensor_id, pos.rank):
                    return Ok(slot.payload_offset + 100)
        raise AssertionError(pos)

    def init_data_accessor(self):
        return Ok(self.accessor)

    def commit(self):
        self.closed = True
        return Ok(None)

    def abort(self):
        self.closed = True
        return Ok(None)

    def finish(self):
        self.closed = True
        return Ok(None)


def compute_view(shapes, columns, accessor):
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([size], "U8", 1),
                    [gb.Shard(0)],
                    gb.ArrangedMesh(WorldSize(columns)),
                ),
            )
            for tid, size in shapes.items()
        ]
    )
    view = gb.ComputeDTensorsView(list(shapes), table, accessor)
    checked(view.check_constraints())
    return view


def storage_view(shapes, columns, file_columns):
    ids = list(shapes)
    entries, paths, slots, payloads = {}, {}, {}, {}
    table = gb.DTensorTable(
        [
            gb.NamedDTensor(
                tid,
                gb.NaiveDTensor(
                    gb.GlobalTensorMeta([size], "U8", 1),
                    [gb.Replicate(), gb.Shard(0)],
                    gb.SlotOuterProductMesh(
                        gb.ArrangedMesh(ConstantOne(1)),
                        gb.ArrangedMesh(WorldSize(columns)),
                    ),
                ),
            )
            for tid, size in shapes.items()
        ]
    )
    for path, ranks in file_columns.items():
        group = gb.DisjointStorageMeshList(
            [
                gb.SlotOuterProductMesh(
                    gb.ArrangedMesh(TensorCount(len(ids))),
                    gb.LiteralMesh(ranks, WorldSize(columns)),
                )
            ],
            ids,
        )
        paths[gb.TensorComponentPos(ids[0], min(ranks))] = path
        slots[path] = []
        payload_offset = 0
        for tid, size in shapes.items():
            for rank in sorted(ranks):
                entries[gb.TensorComponentPos(tid, rank)] = group
                lo, hi = chunk_bounds(size, columns, rank)
                if hi > lo:
                    slots[path].append(Slot(tid, rank, payload_offset, lo, hi - lo))
                payload_offset += hi - lo
        payloads[path] = bytearray([0xA5] * payload_offset)
    accessor = StorageAccessor(payloads, slots)
    sessions = []

    def begin(view, rank):
        session = ProbeSession(accessor)
        sessions.append(session)
        return Ok(session)

    view = gb.StorageDTensorsView(
        ids, table, accessor, gb.PhysicalLayoutTable("distcp", entries, paths), begin, begin
    )
    checked(view.check_constraints())
    return view, accessor, sessions


def run_probe():
    shapes = {"z_first": 17, "a_empty_slots": 2, "m_empty_tensor": 0, "b_last": 9}
    values = {
        tid: bytearray((i * 53 + j) % 256 for j in range(size))
        for i, (tid, size) in enumerate(shapes.items())
    }
    source = ComputeAccessor(values)
    cv = compute_view(shapes, 4, source)
    sv, storage, sessions = storage_view(shapes, 4, {"even.distcp": [0, 2], "odd.distcp": [1, 3]})
    for rank in range(4):
        checked(rank_save(rank, cv, sv))
    assert len(sessions) == 4 and all(session.closed for session in sessions)
    for path, slots in storage.slots.items():
        expected = b"".join(
            values[slot.tensor_id][slot.global_offset : slot.global_offset + slot.size]
            for slot in slots
        )
        assert storage.payloads[path] == expected
    save_requests = list(storage.requests)
    storage.requests.clear()
    for rank in range(4):
        target = ComputeAccessor({tid: bytearray([0xA5] * size) for tid, size in shapes.items()})
        checked(rank_load(rank, compute_view(shapes, 4, target), sv))
        for tid, size in shapes.items():
            lo, hi = chunk_bounds(size, 4, rank)
            assert target.values[tid][lo:hi] == values[tid][lo:hi]
    reshard = []
    for saved, loaded in [(4, 3), (2, 4)]:
        sv, storage, _ = storage_view(
            shapes, saved, {f"rank{rank}.distcp": [rank] for rank in range(saved)}
        )
        for path, slots in storage.slots.items():
            storage.payloads[path][:] = b"".join(
                values[slot.tensor_id][slot.global_offset : slot.global_offset + slot.size]
                for slot in slots
            )
        rank_requests = []
        for rank in range(loaded):
            storage.requests.clear()
            target = ComputeAccessor(
                {tid: bytearray([0xA5] * size) for tid, size in shapes.items()}
            )
            checked(rank_load(rank, compute_view(shapes, loaded, target), sv))
            for tid, size in shapes.items():
                lo, hi = chunk_bounds(size, loaded, rank)
                expected = bytearray([0xA5] * size)
                expected[lo:hi] = values[tid][lo:hi]
                assert target.values[tid] == expected
                actual = [
                    index
                    for name, offset, length in target.requests
                    if name == tid
                    for index in range(offset, offset + length)
                ]
                assert sorted(actual) == list(range(lo, hi))
            rank_requests.append(list(storage.requests))
        reshard.append({"saved": saved, "loaded": loaded, "requests": rank_requests})
    return {
        "R1": {
            "save_requests": save_requests,
            "slots": {
                p: [vars(s) for s in ss]
                for p, ss in storage_view(shapes, 4, {"even.distcp": [0, 2], "odd.distcp": [1, 3]})[
                    1
                ].slots.items()
            },
        },
        "R2": reshard,
    }


if __name__ == "__main__":
    print(json.dumps(run_probe(), indent=2))
