"""Small independent checks for the checkpoint identity audit helpers."""

import base64
import hashlib
import io
import pickle

import pytest

from scripts.dcp_identity_helpers import byte_diffs, digest_stream, field_diffs
from scripts.validate_dcp_identity import compare, file_table


def test_streaming_hash_and_bounded_ranges():
    data = b"abc123" * 1500000
    assert digest_stream(io.BytesIO(data)) == hashlib.blake2b(data, digest_size=32).hexdigest()
    stream = io.BytesIO(b"prefix" + data + b"tail")
    stream.seek(6)
    assert digest_stream(stream, len(data)) == digest_stream(io.BytesIO(data))
    assert stream.read() == b"tail"
    with pytest.raises(EOFError):
        digest_stream(io.BytesIO(b"short"), 6)


def test_exact_diff_ranges_and_metadata_fields():
    assert byte_diffs(b"abcXYfG", b"abc12fH!", 100) == [
        dict(start=103, end=105, left_hex=b"XY".hex(), right_hex=b"12".hex()),
        dict(start=106, end=108, left_hex=b"G".hex(), right_hex=b"H!".hex()),
    ]
    assert byte_diffs(b"same", b"same") == []
    assert field_diffs(
        {"storage_meta": {"save_id": "a", "path": "same"}},
        {"storage_meta": {"save_id": "b", "path": "same"}},
    ) == [dict(field="storage_meta.save_id", left="a", right="b")]


def test_uuid_normalization_preserves_real_metadata_changes():
    def checkpoint(save_id, marker="baseline"):
        metadata = dict(storage_meta=dict(save_id=save_id), marker=marker)
        raw = pickle.dumps(metadata)
        return dict(
            metadata=metadata,
            files={
                ".metadata": dict(
                    size=len(raw),
                    blake2b_256=digest_stream(io.BytesIO(raw)),
                    envelopes=[dict(offset=0, data=base64.b64encode(raw).decode())],
                    payloads=[],
                    members=[],
                )
            },
        )

    a, b, g = checkpoint("1" * 36), checkpoint("2" * 36), checkpoint("3" * 36, "changed!")
    control = compare(a, b, mask_save_id=True)
    changed = compare(a, g, mask_save_id=True)
    assert control["normalized_identical"]
    assert not changed["normalized_identical"]
    table = file_table(
        dict(
            checkpoints=dict(native_a=a, native_b=b, gdtensor=g),
            masked_fields=["storage_meta.save_id"],
            comparisons=dict(native_a_vs_native_b=control, native_a_vs_gdtensor=changed),
        )
    )
    copies = table[".metadata"]
    assert (
        copies["native_a"]["normalized_blake2b_256"]
        == (copies["native_b"]["normalized_blake2b_256"])
    )
    assert (
        copies["native_a"]["normalized_blake2b_256"]
        != (copies["gdtensor"]["normalized_blake2b_256"])
    )
