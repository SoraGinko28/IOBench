"""Bounded native-file comparison and coverage of the shared correctness runner."""

import hashlib
import json
import struct

import pytest

from scripts.validate_shared_safetensors import (
    compare_files,
    gate_plan,
    parse_args,
    read_standard_header,
)


def test_streamed_byte_comparison_hashes_every_byte(tmp_path):
    data = bytes(range(256)) * 9
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    actual.write_bytes(data)
    reference.write_bytes(data)
    report = compare_files(actual, reference, block_bytes=17)
    assert report == dict(
        byte_identical=True,
        bytes=len(data),
        actual_sha256=hashlib.sha256(data).hexdigest(),
        reference_sha256=hashlib.sha256(data).hexdigest(),
    )


@pytest.mark.parametrize("offset", [0, 16, 17, 255, 511])
def test_streamed_byte_comparison_detects_corruption_across_blocks(tmp_path, offset):
    data = bytearray(bytes(range(256)) * 2)
    reference, actual = tmp_path / "reference", tmp_path / "actual"
    reference.write_bytes(data)
    data[offset] ^= 1
    actual.write_bytes(data)
    with pytest.raises(ValueError, match=f"offset {offset}:"):
        compare_files(actual, reference, block_bytes=17)


def test_streamed_byte_comparison_rejects_truncation(tmp_path):
    actual, reference = tmp_path / "actual", tmp_path / "reference"
    actual.write_bytes(b"ab")
    reference.write_bytes(b"abc")
    with pytest.raises(ValueError, match="size differs"):
        compare_files(actual, reference)


def test_standard_header_rejects_hidden_dcp_metadata(tmp_path):
    path = tmp_path / "model.safetensors"
    raw = json.dumps({"__metadata__": {"DCP_SHARDING_INFO": "{}"}}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw)
    with pytest.raises(ValueError, match="DCP_SHARDING_INFO"):
        read_standard_header(path)


@pytest.mark.parametrize("data", [b"short", struct.pack("<Q", 100_000_001)])
def test_standard_header_rejects_invalid_prefix(tmp_path, data):
    path = tmp_path / "model.safetensors"
    path.write_bytes(data)
    with pytest.raises(ValueError, match="safetensors"):
        read_standard_header(path)


def test_cross_rank_gate_plan_covers_both_directions_and_meta_sinks():
    plan = gate_plan([1, 4, 8])
    assert {(row["save_ranks"], row["load_ranks"], row["mode"]) for row in plan} == {
        (saved, loaded, mode)
        for saved, loaded in ((4, 8), (8, 4))
        for mode in ("cpu", "meta-fresh", "meta-faithful")
    }
    assert not gate_plan([1])


def test_correctness_cli_defaults_require_both_models_and_all_rank_counts(tmp_path):
    args = parse_args(["--output", str(tmp_path / "gates")])
    assert args.models == ["1b", "3b"]
    assert args.ranks == [1, 4, 8]
    assert not args.skip_meta_save


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--output", "out", "--ranks", "4", "4"],
        ["--output", "out", "--bundle", "bundle"],
        ["--worker", "reference"],
    ],
)
def test_correctness_cli_rejects_incomplete_or_ambiguous_requests(argv):
    with pytest.raises(SystemExit):
        parse_args(argv)
