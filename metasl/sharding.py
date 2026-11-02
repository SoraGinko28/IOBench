"""Versioned Shard(0) descriptors. DCP alone plans source/target intersections."""

import copy
import dataclasses
import hashlib
import math

import torch
from torch.distributed.checkpoint.metadata import (
    ChunkStorageMetadata,
    Metadata,
    MetadataIndex,
    TensorStorageMetadata,
)
from torch.distributed.tensor import DTensor
from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

POLICY = "dtensor-shard0-v1"


def element_size(dtype):
    return torch._utils._element_size(dtype)


def alias_names(structure):
    result = {}
    for group in structure.get("aliases", []):
        names = sorted(x.split(":", 1)[1] for x in group)
        for name in names:
            result[name] = names
    return result




def manifest_row(fqn, tensor, kind, alias_group, policy_version):
    """One manifest row for a DTensor (or a plain tensor, treated as replicated).

    Local geometry is computed the way DCP's planner computes it, from the DTensor's
    own mesh and placements, so the row describes exactly the chunk that will be
    planned for this rank.
    """
    if isinstance(tensor, DTensor):
        mesh, placements = tensor.device_mesh, tensor.placements
        shape, offset = compute_local_shape_and_global_offset(tensor.shape, mesh, placements)
        mesh_dims = list(mesh.mesh_dim_names or ())
    else:
        placements, mesh_dims = (), []
        shape, offset = tuple(tensor.shape), (0,) * tensor.ndim
    return dict(
        fqn=fqn,
        global_shape=list(tensor.shape),
        dtype=str(tensor.dtype),
        global_stride=list(tensor.stride()),
        kind=kind,
        shard_axis=0 if tensor.ndim else None,
        placements=[str(p) for p in placements],
        mesh_dims=mesh_dims,
        global_offsets=list(offset),
        local_shape=list(shape),
        replicated=all(p.is_replicate() for p in placements),
        alias_group=list(alias_group),
        policy_version=policy_version,
        payload_bytes=math.prod(shape) * element_size(tensor.dtype),
    )


def local_tensor(t):
    return t.to_local() if isinstance(t, DTensor) else t


def check_partition(shape, boxes):
    """Prove disjoint rectangular coverage using containment, intersections and volume."""
    boxes = [(tuple(o), tuple(s)) for o, s in boxes]
    for o, s in boxes:
        if (
            len(o) != len(shape)
            or len(s) != len(shape)
            or any(a < 0 or b < 0 or a + b > g for a, b, g in zip(o, s, shape, strict=True))
        ):
            raise ValueError(("out of bounds", shape, o, s))
    boxes = [(o, s) for o, s in boxes if math.prod(s)]
    for i, (a, x) in enumerate(boxes):
        for b, y in boxes[i + 1 :]:
            if all(
                max(ai, bi) < min(ai + xi, bi + yi)
                for ai, bi, xi, yi in zip(a, b, x, y, strict=True)
            ):
                raise ValueError(("overlap", a, x, b, y))
    if sum(math.prod(s) for _, s in boxes) != math.prod(shape):
        raise ValueError(("coverage gap", shape, boxes))




def audit_metadata(metadata):
    """Saved chunks partition every tensor exactly; returns the storage mapping rows."""
    for md in metadata.state_dict_metadata.values():
        if isinstance(md, TensorStorageMetadata):
            check_partition(md.size, [(c.offsets, c.sizes) for c in md.chunks])
    return [
        dict(
            fqn=idx.fqn,
            offsets=list(idx.offset or ()),
            file=info.relative_path,
            file_offset=info.offset,
            file_bytes=info.length,
        )
        for idx, info in metadata.storage_data.items()
    ]


def box_overlaps(offsets, sizes, start, extent):
    """Positive overlap on every dimension, DCP's _check_shard_metadata_pair_overlap rule."""
    return all(
        max(o, s) < min(o + n, s + e)
        for o, n, s, e in zip(offsets, sizes, start, extent, strict=True)
    )


def prune_metadata(metadata, rows):
    """Metadata reduced to the chunks this rank's local shards can intersect.

    DCP's default load planner creates one ReadItem per (local chunk, saved chunk)
    pair with positive overlap on every dimension and never looks at the other
    saved chunks, so the local plan, the read requests and the file bytes are
    unchanged. Every key stays (missing-key and optional-key checks need them);
    only each tensor's chunk list and the matching storage_data entries shrink.
    Entries without a manifest row (bytes, plain CPU scalars, other stages'
    tensors) are kept whole. The result is a fresh object graph so the original,
    typically 100x larger, can be released entirely. Returns (metadata, counts).
    """
    boxes = {}
    for row in rows:
        if row["fqn"] in boxes:
            raise ValueError(("duplicate manifest row", row["fqn"]))
        boxes[row["fqn"]] = (tuple(row["global_offsets"]), tuple(row["local_shape"]))
    kept = {}
    state = {}
    for name, md in metadata.state_dict_metadata.items():
        if not isinstance(md, TensorStorageMetadata):
            state[name] = copy.copy(md)
            continue
        if name in boxes:
            start, extent = boxes[name]
            if len(start) != len(md.size):
                raise ValueError(("manifest rank mismatch", name, start, tuple(md.size)))
            chunks = [c for c in md.chunks if box_overlaps(c.offsets, c.sizes, start, extent)]
        else:
            chunks = md.chunks
        kept[name] = {tuple(c.offsets) for c in chunks}
        state[name] = TensorStorageMetadata(
            properties=dataclasses.replace(md.properties),
            size=torch.Size(md.size),
            chunks=[
                ChunkStorageMetadata(offsets=torch.Size(c.offsets), sizes=torch.Size(c.sizes))
                for c in chunks
            ],
        )
    storage = {}
    for index, info in metadata.storage_data.items():
        if index.fqn in kept and tuple(index.offset or ()) not in kept[index.fqn]:
            continue
        storage[
            MetadataIndex(index.fqn, None if index.offset is None else torch.Size(index.offset))
        ] = dataclasses.replace(info) if dataclasses.is_dataclass(info) else copy.copy(info)
    pruned = Metadata(
        state_dict_metadata=state,
        planner_data=copy.deepcopy(metadata.planner_data),
        storage_data=storage,
        storage_meta=copy.copy(metadata.storage_meta),
        version=metadata.version,
    )
    counts = dict(
        chunks_before=sum(
            len(md.chunks)
            for md in metadata.state_dict_metadata.values()
            if isinstance(md, TensorStorageMetadata)
        ),
        chunks_after=sum(len(c) for c in kept.values()),
        storage_entries_before=len(metadata.storage_data),
        storage_entries_after=len(storage),
        pruned_tensors=sum(name in boxes for name in kept),
        untouched_tensors=sum(name not in boxes for name in kept),
    )
    return pruned, counts


def read_record(item, metadata):
    """Describe one default-planner ReadItem in checkpoint and target coordinates."""
    md = metadata.state_dict_metadata[item.storage_index.fqn]
    chunk = next(c for c in md.chunks if tuple(c.offsets) == tuple(item.storage_index.offset))
    info = metadata.storage_data[item.storage_index]
    width = element_size(md.properties.dtype)
    return dict(
        fqn=item.dest_index.fqn,
        source_chunk_offsets=list(chunk.offsets),
        source_chunk_shape=list(chunk.sizes),
        source_offsets=list(item.storage_offsets),
        target_chunk_offsets=list(item.dest_index.offset),
        local_offsets=list(item.dest_offsets),
        global_offsets=[a + b for a, b in zip(chunk.offsets, item.storage_offsets, strict=True)],
        lengths=list(item.lengths),
        file=info.relative_path,
        file_offset=info.offset,
        file_bytes=info.length,
        target_bytes=math.prod(item.lengths) * width,
        source_chunk_bytes=math.prod(chunk.sizes) * width,
    )


def audit_local(rows, requests):
    """Read records stay inside their saved chunks and exactly tile each local shard."""
    names = {r["fqn"] for r in rows}
    if any(r["fqn"] not in names for r in requests):
        raise ValueError("Unexpected requested FQN")
    for r in requests:
        if any(
            o < 0 or n < 0 or o + n > s
            for o, n, s in zip(
                r["source_offsets"], r["lengths"], r["source_chunk_shape"], strict=True
            )
        ):
            raise ValueError("Read lies outside saved chunk")
        if r["global_offsets"] != [
            a + b for a, b in zip(r["source_chunk_offsets"], r["source_offsets"], strict=True)
        ]:
            raise ValueError("Incorrect source global offset")
    for row in rows:
        reqs = [r for r in requests if r["fqn"] == row["fqn"]]
        for r in reqs:
            if r["global_offsets"] != [
                a + b for a, b in zip(row["global_offsets"], r["local_offsets"], strict=True)
            ]:
                raise ValueError(("wrong global offset", r))
        check_partition(row["local_shape"], [(r["local_offsets"], r["lengths"]) for r in reqs])


def pattern_blocks(shape, offsets, global_shape, name, dtype, block=262144):
    """Bounded deterministic hash of FQN and global flat index; exact bf16 integers.

    Works for any rectangular block of a row-major tensor: each element's value is
    derived from its global flat index, so 2-D tensor-parallel chunks check too.
    Yields (local flat offset, values) for slabs of leading rows.
    """
    salt = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "little")
    shape, offsets, global_shape = tuple(shape), tuple(offsets), tuple(global_shape)

    def values(flat_index):
        x = ((flat_index ^ salt) * 1103515245 + 12345) & 0x7FFFFFFF
        x = (x ^ (x >> 11)) % 251 + 1
        if dtype == torch.bool:
            x = x % 2
        elif dtype == torch.int8:
            x = x % 127
        return x.to(dtype)

    if len(shape) == 0:
        yield 0, values(torch.zeros(1, dtype=torch.int64))
        return
    if math.prod(shape) == 0:
        return
    strides = [math.prod(global_shape[i + 1 :]) for i in range(len(global_shape))]
    inner = torch.zeros((), dtype=torch.int64)
    for dim in range(1, len(shape)):
        axis = (offsets[dim] + torch.arange(shape[dim], dtype=torch.int64)) * strides[dim]
        inner = inner.reshape(*inner.shape, 1) + axis
    inner = inner.reshape(-1)
    rows_per_slab = max(1, block // inner.numel())
    for r0 in range(0, shape[0], rows_per_slab):
        r1 = min(shape[0], r0 + rows_per_slab)
        rows = (offsets[0] + torch.arange(r0, r1, dtype=torch.int64)) * strides[0]
        yield r0 * inner.numel(), values((rows[:, None] + inner[None, :]).reshape(-1))


def fill_or_check(tensor, offsets, global_shape, name, mode, check=False):
    """Write (or verify) the zero/position pattern for a contiguous block of a tensor."""
    flat = tensor.reshape(-1)
    if mode == "zero":
        if check:
            if torch.count_nonzero(flat).item():
                raise ValueError("nonzero checkpoint value")
        else:
            flat.zero_()
        return
    for base, expected in pattern_blocks(tensor.shape, offsets, global_shape, name, tensor.dtype):
        actual = flat[base : base + expected.numel()]
        if check:
            observed = actual.to("cpu") if actual.device.type == "cuda" else actual
            if not torch.equal(observed, expected):
                raise ValueError(
                    f"numerical mismatch: {name}, global offset {offsets}, block {base}"
                )
        else:
            actual.copy_(expected)
