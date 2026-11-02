"""Pure FSDP coordinates and torch-free launcher parsing."""

import subprocess
import sys

import pytest

from metasl.layout import Layout, ParallelDims, parse_parallel


def test_fsdp_dims_and_coordinates():
    dims = parse_parallel("dp_shard=4")
    assert dims.world_size == 4 and dims.mesh_shape == (1, 1, 4, 1)
    assert [dims.coordinate(rank) for rank in range(4)] == [
        (0, 0, rank, 0) for rank in range(4)
    ]
    assert [dims.dp_rank(rank) for rank in range(4)] == list(range(4))
    layout = Layout(dims, 22, 3)
    assert layout.stage == 0 and layout.dp_rank == 3
    assert layout.param_dims == ["dp_shard"] and not layout.untie
    with pytest.raises(ValueError):
        ParallelDims(dp_shard=0)


def test_layout_module_imports_without_torch(tmp_path):
    code = (
        "import sys; from metasl.run import parse_args; "
        "parse_args(['--action', 'save', '--bundle', 'b', "
        f"'--output', {str(tmp_path / 'out')!r}, '--ranks', '4', "
        "'--parallel', 'dp_shard=4']); "
        "print('torch' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False", out.stdout + out.stderr
