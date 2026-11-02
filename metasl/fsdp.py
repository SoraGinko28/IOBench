"""FSDP2 geometry on Meta, with optional resident CPU/CUDA local shards."""

import torch
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor import DTensor
from metasl.sharding import alias_names, manifest_row
from metasl.skeleton import (
    DTYPES,
    decoder_layers,
)

POLICY = "fsdp2-v1"


def apply_parallelism(model, layout, compute_dtype="bf16"):
    """Shard each decoder layer; keep embeddings/norm/head in the root group.

    The root group preserves tied embedding identity.
    """
    if layout.world_mesh is None:
        raise ValueError("Layout mesh must be built before applying parallelism")
    names = tuple(layout.dp_dims)
    dp_mesh = layout.world_mesh[names] if len(names) > 1 else layout.world_mesh[names[0]]
    policy = MixedPrecisionPolicy(param_dtype=DTYPES[compute_dtype], reduce_dtype=torch.float32)
    for layer in decoder_layers(model):
        if not isinstance(layer, torch.nn.Identity):
            fully_shard(layer, mesh=dp_mesh, mp_policy=policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=policy)
    return model


def describe_targets(state_dict, structure):
    """Describe model tensor geometry through the same DTensor API as DCP.

    Kinds and tied aliases come from the prepared structure manifest.
    """
    aliases = alias_names(structure)
    kinds = {x["name"]: x["kind"] for x in structure["tensors"]}
    flat, _ = flatten_state_dict(state_dict)
    rows = []
    for fqn, tensor in flat.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        top, _, name = fqn.partition(".")
        if top == "model":
            kind, group = kinds.get(name, top), aliases.get(name, [name])
        else:
            kind, group = top, [name]
        rows.append(manifest_row(fqn, tensor, kind, group, POLICY))
        if isinstance(tensor, DTensor):
            local = tensor.to_local()
            if list(local.shape) != rows[-1]["local_shape"]:
                raise ValueError(f"{fqn}: local shard {tuple(local.shape)} != planned chunk")
    return rows


def materialize(state_dict, device="cpu"):
    """Same DTensors with real (uninitialised) local storage, for the standard control path.

    Each rank allocates only its own shard; the model stays meta. CUDA meshes
    reuse the layout's Gloo groups for checkpoint metadata only. No CUDA
    collectives, forward pass or optimizer update are performed here.
    """
    from metasl.reference import checkpoint_mesh

    device = torch.device(device)
    meshes = {}

    def target_mesh(mesh):
        if mesh.device_type == device.type:
            return mesh
        if id(mesh) not in meshes:
            meshes[id(mesh)] = checkpoint_mesh(mesh, device)
        return meshes[id(mesh)]

    def convert(value):
        if isinstance(value, DTensor):
            local = torch.empty_like(value.to_local(), device=device)
            result = DTensor.from_local(
                local,
                target_mesh(value.device_mesh),
                value.placements,
                run_check=False,
                shape=value.shape,
                stride=value.stride(),
            )
            if result.to_local().device.type != device.type:
                raise RuntimeError("DTensor materialization changed the requested storage device")
            return result
        if isinstance(value, torch.Tensor):
            return torch.empty_like(value, device=device)
        if isinstance(value, dict):
            return {k: convert(v) for k, v in value.items()}
        return value

    return convert(state_dict)
