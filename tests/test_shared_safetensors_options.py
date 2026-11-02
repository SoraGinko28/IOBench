"""Shared safetensors CLI scope and whole-tensor size-limit validation."""

import sys

import pytest

from metasl.run import parse_args
from scripts.run_gdtensor_trace_matrix import phase_command


def command_args(tmp_path, **options):
    return phase_command(
        sys.executable,
        tmp_path / "bundle",
        tmp_path / "checkpoint",
        tmp_path / "output",
        4,
        "save",
        "cpu",
        checkpoint_format="safetensors",
        **options,
    )[3:]


def test_safetensors_default_preserves_per_rank_layout(tmp_path):
    args = parse_args(command_args(tmp_path))
    assert args.safetensors_layout == "per-rank"
    assert args.safetensors_max_file_gib is None


@pytest.mark.parametrize("limit", [None, 5, 0.0001])
def test_shared_command_passes_layout_and_global_size_limit_to_real_parser(tmp_path, limit):
    args = parse_args(
        command_args(
            tmp_path, safetensors_layout="shared", safetensors_max_file_gib=limit, writer_limit=4
        )
    )
    assert args.safetensors_layout == "shared"
    assert args.safetensors_max_file_gib == limit
    assert args.writer_limit == 4


def test_shared_accepts_existing_mib_size_unit(tmp_path):
    args = parse_args(
        command_args(tmp_path, safetensors_layout="shared", safetensors_max_file_mib=5120)
    )
    assert args.safetensors_layout == "shared"
    assert args.safetensors_max_file_mib == 5120
    assert args.safetensors_max_file_gib is None


@pytest.mark.parametrize("limit", ["0", "-1", "nan", "inf", "-inf"])
def test_shared_rejects_nonfinite_and_nonpositive_size_limits(tmp_path, limit):
    argv = command_args(tmp_path, safetensors_layout="shared")
    with pytest.raises(SystemExit):
        parse_args(argv + [f"--safetensors-max-file-gib={limit}"])


def test_gib_requires_shared_layout(tmp_path):
    with pytest.raises(SystemExit):
        parse_args(command_args(tmp_path) + ["--safetensors-max-file-gib", "5"])


def test_shared_size_units_are_mutually_exclusive(tmp_path):
    argv = command_args(tmp_path, safetensors_layout="shared", safetensors_max_file_mib=5120)
    with pytest.raises(SystemExit):
        parse_args(argv + ["--safetensors-max-file-gib", "5"])


def test_shared_layout_rejects_dcp_format(tmp_path):
    argv = phase_command(
        sys.executable,
        tmp_path / "bundle",
        tmp_path / "checkpoint",
        tmp_path / "output",
        4,
        "save",
        "cpu",
        safetensors_layout="shared",
    )[3:]
    with pytest.raises(SystemExit):
        parse_args(argv)
