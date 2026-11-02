"""FSDP geometry and Meta load validation used by the IO bench."""
import shutil
from pathlib import Path

import pytest
import torch
from torch.distributed.checkpoint.state_dict import get_model_state_dict
from torch.distributed.tensor import DTensor

from metasl.fsdp import apply_parallelism, materialize
from metasl.layout import Layout, ParallelDims
from metasl.scratch import MetaLoadPlanner
from metasl.skeleton import build_model, prepare_bundle

ROOT = Path(__file__).resolve().parent


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    root = tmp_path_factory.mktemp("tiny")
    (root / "config").mkdir()
    shutil.copy(ROOT / "tiny_qwen2.json", root / "config" / "config.json")
    prepare_bundle(str(root / "config"), "fp32", root / "bundle", local_files_only=True)
    return root / "bundle"



def test_materialize_keeps_geometry_and_model_meta(bundle, process_group):
    model, manifest = build_model(bundle, "meta")
    layout = Layout(ParallelDims(dp_shard=1), len(model.model.layers), 0)
    layout.build_mesh()
    apply_parallelism(model, layout, "fp32")
    state = get_model_state_dict(model)
    cpu = materialize(state)
    for name, tensor in state.items():
        other = cpu[name]
        assert isinstance(other, DTensor) and not other.is_meta
        assert other.to_local().device.type == "cpu"
        assert other.placements == tensor.placements and other.shape == tensor.shape
        assert other.to_local().shape == tensor.to_local().shape
    assert all(p.is_meta for p in model.parameters())



def test_meta_load_planner_rejects_resident_targets():
    planner = MetaLoadPlanner()
    planner.set_up_planner({"w": torch.empty(2, device="meta")})
    with pytest.raises(TypeError, match="meta tensor skeleton"):
        planner.set_up_planner({"w": torch.empty(2)})

