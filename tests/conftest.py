import pytest
import torch
import torch.distributed as dist


@pytest.fixture(scope="session")
def process_group(tmp_path_factory):
    """Single-rank gloo group; DTensor meshes and DCP collectives need one."""
    store = tmp_path_factory.mktemp("rendezvous") / "store"
    dist.init_process_group("gloo", init_method=store.resolve().as_uri(), rank=0, world_size=1)
    torch.set_num_threads(1)
    yield
    dist.destroy_process_group()
