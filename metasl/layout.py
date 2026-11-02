"""Pure FSDP layout; preserve the torch mesh used by the IO trace baseline."""

from dataclasses import dataclass

MESH_DIMS = ("pp", "dp_replicate", "dp_shard", "tp")

@dataclass(frozen=True)
class ParallelDims:
    pp: int = 1
    dp_replicate: int = 1
    dp_shard: int = 1
    cp: int = 1
    tp: int = 1

    def __post_init__(self):
        for name in ("pp", "dp_replicate", "dp_shard", "cp", "tp"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")

    @property
    def world_size(self):
        return self.pp * self.dp_replicate * self.dp_shard * self.cp * self.tp

    @property
    def mesh_shape(self):
        """Parameter layout mesh; cp is folded into dp_shard as torchtitan's dp_shard_cp."""
        return (self.pp, self.dp_replicate, self.dp_shard * self.cp, self.tp)

    def coordinate(self, rank):
        """Rank -> (pp, dp_replicate, dp_shard_cp, tp) index in row-major mesh order."""
        coords = []
        for size in reversed(self.mesh_shape):
            coords.append(rank % size)
            rank //= size
        return tuple(reversed(coords))

    def dp_rank(self, rank):
        """Data-parallel position (dataloader key): shared by the tp, pp and cp ranks of one shard.

        The folded dp_shard_cp coordinate is dp_shard_index * cp + cp_index, so cp ranks
        collapse onto one data shard as in torchtitan.
        """
        _, replicate, shard_cp, _ = self.coordinate(rank)
        return replicate * self.dp_shard + shard_cp // self.cp

    def stage(self, rank):
        return self.coordinate(rank)[0]


def parse_parallel(text):
    """'pp=1,dp_replicate=2,dp_shard=4,tp=2' -> ParallelDims."""
    values = {}
    for part in filter(None, text.split(",")):
        key, value = part.split("=")
        values[key.strip()] = int(value)
    return ParallelDims(**values)


def job_layout(args, num_layers):
    """Layout of this rank from --parallel (default pure FSDP over the world); torch-free."""
    parallel = getattr(args, "parallel", "") or ""
    dims = parse_parallel(parallel) if parallel else ParallelDims(dp_shard=args.world_size)
    if dims.world_size != args.world_size:
        raise ValueError(
            f"parallel dims {vars(dims)} do not multiply to world size {args.world_size}"
        )
    if any(getattr(dims, name) != 1 for name in ("pp", "tp", "cp", "dp_replicate")):
        raise ValueError("Only pure FSDP dp_shard parallelism is supported")
    return Layout(dims, num_layers, args.rank)


class Layout:
    """The original torch mesh dimension order, restricted to pure FSDP."""

    def __init__(self, dims, num_layers, rank):
        self.dims, self.num_layers, self.rank = dims, num_layers, rank
        self.stages = [(0, num_layers)]
        self.stage = dims.stage(rank)
        self.dp_rank = dims.dp_rank(rank)
        self.untie = dims.tp > 1
        self.dp_dims = (["dp_replicate"] if dims.dp_replicate > 1 else []) + ["dp_shard"]
        self.param_dims = self.dp_dims + (["tp"] if dims.tp > 1 else [])
        self.world_mesh = self.mesh = None

    def build_mesh(self):
        from torch.distributed.device_mesh import init_device_mesh

        self.world_mesh = init_device_mesh("cpu", self.dims.mesh_shape, mesh_dim_names=MESH_DIMS)
        names = tuple(self.param_dims)
        self.mesh = self.world_mesh[names] if len(names) > 1 else self.world_mesh[names[0]]
        return self.mesh

    def describe(self):
        return dict(
            parallel=vars(self.dims),
            rank=self.rank,
            stage=self.stage,
            dp_rank=self.dp_rank,
            stages=self.stages,
            param_mesh_dims=self.param_dims,
            untie=self.untie,
        )
