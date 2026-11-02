import pytest
import torch

from metasl.sharding import (
    alias_names,
    check_partition,
    fill_or_check,
    pattern_blocks,
)


def test_check_partition_accepts_exact_tiling():
    check_partition((4, 3), [((0, 0), (2, 3)), ((2, 0), (2, 3))])
    check_partition((4, 3), [((0, 0), (4, 3)), ((4, 0), (0, 3))])  # empty box ignored
    check_partition((), [((), ())])


@pytest.mark.parametrize(
    "boxes, reason",
    [
        ([((0, 0), (3, 3)), ((2, 0), (2, 3))], "overlap"),
        ([((0, 0), (1, 3)), ((2, 0), (2, 3))], "coverage gap"),
        ([((0, 0), (5, 3))], "out of bounds"),
        ([((0,), (4,))], "out of bounds"),
        ([((5, 0), (0, 3)), ((0, 0), (4, 3))], "out of bounds"),
    ],
)
def test_check_partition_rejects(boxes, reason):
    with pytest.raises(ValueError) as info:
        check_partition((4, 3), boxes)
    assert info.value.args[0][0] == reason


def test_pattern_blocks_are_deterministic_exact_and_name_dependent():
    a = torch.cat([x for _, x in pattern_blocks((6, 4), (2, 0), (10, 4), "w", torch.bfloat16, 5)])
    b = torch.cat([x for _, x in pattern_blocks((6, 4), (2, 0), (10, 4), "w", torch.bfloat16, 7)])
    c = torch.cat([x for _, x in pattern_blocks((6, 4), (2, 0), (10, 4), "v", torch.bfloat16, 5)])
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert a.shape == (24,)
    as_int = a.to(torch.int64)
    assert torch.equal(as_int.to(torch.bfloat16), a)
    assert int(as_int.min()) >= 1 and int(as_int.max()) <= 251
    # Row range [2:8) of a (10, 4) tensor is the flat range [8:32) of the whole.
    whole = torch.cat([x for _, x in pattern_blocks((10, 4), (0, 0), (10, 4), "w", torch.bfloat16)])
    assert torch.equal(a, whole[8:32])


def test_pattern_blocks_respects_dtype_ranges():
    for dtype, high in ((torch.bool, 1), (torch.int8, 126), (torch.int64, 251)):
        x = torch.cat([v for _, v in pattern_blocks((100,), (0,), (100,), "n", dtype)])
        assert x.dtype == dtype
        assert int(x.to(torch.int64).max()) <= high


def test_fill_or_check_roundtrip_and_mismatch():
    t = torch.empty(6, 4, dtype=torch.bfloat16)
    fill_or_check(t, (2, 0), (10, 4), "w", "position")
    fill_or_check(t, (2, 0), (10, 4), "w", "position", check=True)
    t[3, 1] += 1
    with pytest.raises(ValueError, match="numerical mismatch"):
        fill_or_check(t, (2, 0), (10, 4), "w", "position", check=True)
    fill_or_check(t, (2, 0), (10, 4), "w", "zero")
    fill_or_check(t, (2, 0), (10, 4), "w", "zero", check=True)
    t[0, 0] = 1
    with pytest.raises(ValueError, match="nonzero"):
        fill_or_check(t, (2, 0), (10, 4), "w", "zero", check=True)


def test_alias_names():
    structure = {"aliases": [["parameter:lm_head.weight", "parameter:embed.weight"]]}
    names = alias_names(structure)
    assert names == {
        "embed.weight": ["embed.weight", "lm_head.weight"],
        "lm_head.weight": ["embed.weight", "lm_head.weight"],
    }
    assert alias_names({}) == {}


def test_pattern_blocks_two_dimensional_block_matches_global_values():
    whole = torch.cat(
        [x for _, x in pattern_blocks((10, 6), (0, 0), (10, 6), "w", torch.bfloat16)]
    ).reshape(10, 6)
    block = torch.empty(4, 2, dtype=torch.bfloat16)
    for base, values in pattern_blocks((4, 2), (2, 3), (10, 6), "w", torch.bfloat16, block=3):
        block.reshape(-1)[base : base + values.numel()] = values
    assert torch.equal(block, whole[2:6, 3:5])
    fill_or_check(block, (2, 3), (10, 6), "w", "position", check=True)
    scalar = torch.empty((), dtype=torch.float32)
    fill_or_check(scalar, (), (), "s", "position")
    fill_or_check(scalar, (), (), "s", "position", check=True)










def _metadata():
    from torch.distributed.checkpoint.filesystem import _StorageInfo
    from torch.distributed.checkpoint.metadata import (
        BytesStorageMetadata,
        ChunkStorageMetadata,
        Metadata,
        MetadataIndex,
        TensorProperties,
        TensorStorageMetadata,
    )

    def tensor(size, boxes):
        return TensorStorageMetadata(
            TensorProperties(dtype=torch.bfloat16),
            torch.Size(size),
            [ChunkStorageMetadata(torch.Size(o), torch.Size(s)) for o, s in boxes],
        )

    # "a": uneven Shard(0) over 4 ranks incl. a short last chunk; "s": 0-dim; "b": no row.
    state = {
        "a": tensor(
            (10, 4), [((0, 0), (3, 4)), ((3, 0), (3, 4)), ((6, 0), (3, 4)), ((9, 0), (1, 4))]
        ),
        "s": tensor((), [((), ())]),
        "b": tensor((2, 2), [((0, 0), (1, 2)), ((1, 0), (1, 2))]),
        "extras.lr": BytesStorageMetadata(),
    }
    storage = {}
    for name, md in state.items():
        if isinstance(md, BytesStorageMetadata):
            storage[MetadataIndex(name)] = _StorageInfo("__0_0.distcp", 0, 10)
            continue
        for i, c in enumerate(md.chunks):
            storage[MetadataIndex(name, c.offsets)] = _StorageInfo(f"__{i}_0.distcp", 100 * i, 50)
    return Metadata(state, planner_data={"m": 1}, storage_data=storage, version="1")


def _row(fqn, offsets, shape):
    return dict(fqn=fqn, global_offsets=list(offsets), local_shape=list(shape))


@pytest.mark.parametrize(
    "box, kept",
    [
        (((3, 0), (3, 4)), [(3, 0)]),  # exactly one saved chunk
        (((2, 0), (4, 4)), [(0, 0), (3, 0)]),  # resharded: straddles two chunks
        (((9, 0), (1, 4)), [(9, 0)]),  # short last chunk
        (((10, 0), (0, 4)), []),  # empty shard: DCP plans no reads
        (((0, 0), (10, 4)), [(0, 0), (3, 0), (6, 0), (9, 0)]),  # replicated: everything
    ],
)
def test_prune_metadata_keeps_exactly_the_intersecting_chunks(box, kept):
    from metasl.sharding import prune_metadata

    metadata = _metadata()
    rows = [_row("a", *box), _row("s", (), ())]
    pruned, counts = prune_metadata(metadata, rows)
    assert [tuple(c.offsets) for c in pruned.state_dict_metadata["a"].chunks] == kept
    assert len(pruned.state_dict_metadata["s"].chunks) == 1
    # Keys, sizes, dtypes, bytes and unlisted tensors survive untouched.
    assert list(pruned.state_dict_metadata) == list(metadata.state_dict_metadata)
    assert pruned.state_dict_metadata["a"].size == torch.Size((10, 4))
    assert pruned.state_dict_metadata["a"].properties.dtype == torch.bfloat16
    assert len(pruned.state_dict_metadata["b"].chunks) == 2
    assert pruned.planner_data == {"m": 1} and pruned.version == "1"
    # storage_data shrinks consistently and still resolves every kept chunk.
    remaining = {(i.fqn, tuple(i.offset or ())) for i in pruned.storage_data}
    assert remaining == {("a", k) for k in kept} | {
        ("s", ()),
        ("b", (0, 0)),
        ("b", (1, 0)),
        ("extras.lr", ()),
    }
    for name, md in pruned.state_dict_metadata.items():
        for c in getattr(md, "chunks", []):
            info = pruned.storage_data[type(next(iter(pruned.storage_data)))(name, c.offsets)]
            assert info.relative_path.endswith(".distcp")
    assert counts == dict(
        chunks_before=7,
        chunks_after=len(kept) + 1 + 2,
        storage_entries_before=8,
        storage_entries_after=len(kept) + 1 + 2 + 1,
        pruned_tensors=2,
        untouched_tensors=1,
    )
    # The original is untouched and shares no objects with the pruned copy.
    assert len(metadata.state_dict_metadata["a"].chunks) == 4
    assert pruned.storage_data is not metadata.storage_data
    assert all(
        c is not o
        for c, o in zip(
            pruned.state_dict_metadata["b"].chunks,
            metadata.state_dict_metadata["b"].chunks,
            strict=True,
        )
    )


def test_prune_metadata_rejects_bad_rows():
    from metasl.sharding import prune_metadata

    with pytest.raises(ValueError, match="duplicate"):
        prune_metadata(_metadata(), [_row("a", (0, 0), (3, 4)), _row("a", (3, 0), (3, 4))])
    with pytest.raises(ValueError, match="rank mismatch"):
        prune_metadata(_metadata(), [_row("a", (0,), (3,))])
    # Rows for keys absent from the checkpoint are ignored: the planner reports them.
    pruned, _ = prune_metadata(_metadata(), [_row("missing", (0,), (3,))])
    assert len(pruned.state_dict_metadata["a"].chunks) == 4
