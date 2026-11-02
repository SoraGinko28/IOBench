"""Select and verify real checkpoint reference storage, independently of the control PG.

The CPU path stays torch-free until storage is inspected. CUDA is opt-in, maps
one worker to each visible device unless sharing is explicit, and never falls
back to CPU. This is a checkpoint fixture; a CUDA device mesh does not imply
GPU training was run.
"""

from collections import Counter


def standard_device(args):
    device = getattr(args, "standard_device", "cpu")
    if device not in ("cpu", "cuda"):
        raise ValueError("standard_device must be cpu or cuda")
    return device


def _cuda_mapping(args, count):
    world = getattr(args, "world_size", 1)
    rank = getattr(args, "rank", 0)
    if type(world) is not int or world < 1 or type(rank) is not int or not 0 <= rank < world:
        raise ValueError("CUDA reference requires a valid rank and world_size")
    text = getattr(args, "cuda_devices", "") or ""
    if not isinstance(text, str):
        raise ValueError("cuda_devices must be comma-separated visible device ordinals")
    if text.strip():
        try:
            mapping = [int(part.strip()) for part in text.split(",")]
        except ValueError as error:
            raise ValueError("cuda_devices must contain integer visible ordinals") from error
        if len(mapping) != world:
            raise ValueError("cuda_devices must contain exactly one device per worker")
    else:
        mapping = list(range(world))
    if not getattr(args, "cuda_share_devices", False) and len(set(mapping)) != len(mapping):
        raise ValueError("CUDA reference workers require distinct devices; duplicate mapping")
    if any(index < 0 or index >= count for index in mapping):
        raise ValueError(
            f"CUDA mapping {mapping} exceeds the {count} visible GPUs; "
            "every worker must map to an available visible GPU"
        )
    return mapping, rank


def cuda_mapping_report(mapping, visible_count, *, allow_sharing=False):
    """Describe mapped visible devices; None entries represent unfinished workers."""
    counts = Counter(index for index in mapping if index is not None)
    return {
        "cuda_device_mapping": list(mapping),
        "cuda_mapping_complete": all(index is not None for index in mapping),
        "cuda_share_devices_allowed": allow_sharing,
        "cuda_devices_shared": any(count > 1 for count in counts.values()),
        "cuda_physical_gpu_count": len(counts),
        "cuda_visible_device_count": visible_count,
        "cuda_ranks_per_device": {str(index): counts[index] for index in sorted(counts)},
    }


def initialize_reference(args):
    """Select this worker's device before building meshes or allocating storage."""
    if getattr(args, "implementation", "scratch") != "standard":
        return "meta"
    if standard_device(args) == "cpu":
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA reference requested but CUDA is unavailable; no CPU fallback")
    mapping, rank = _cuda_mapping(args, torch.cuda.device_count())
    index = mapping[rank]
    torch.cuda.set_device(index)
    device = f"cuda:{index}"
    args._reference_device = device
    return device


def synchronize_reference(args):
    """Make reference GPU work complete at an explicitly recorded phase boundary."""
    if getattr(args, "implementation", "scratch") == "standard" and standard_device(args) == "cuda":
        import torch

        torch.cuda.synchronize(getattr(args, "_reference_device", None))


def checkpoint_mesh(mesh, device):
    """Reuse layout groups under the storage device, for checkpoint operations only."""
    import torch
    from torch.distributed.device_mesh import DeviceMesh

    kind = torch.device(device).type
    if kind == "meta" or mesh.device_type == kind:
        return mesh
    if kind not in ("cpu", "cuda"):
        raise ValueError(f"Unsupported checkpoint mesh device: {kind}")
    groups = [mesh.get_group(dim) for dim in range(mesh.ndim)]
    return DeviceMesh.from_group(
        groups[0] if len(groups) == 1 else groups,
        kind,
        mesh=mesh.mesh,
        mesh_dim_names=mesh.mesh_dim_names,
    )


def reference_report(args, state_dict):
    """Inspect actual local storage and report CUDA memory separately from CPU PSS."""
    import torch
    from torch.distributed.checkpoint._nested_dict import flatten_state_dict
    from torch.distributed.tensor import DTensor

    requested = standard_device(args)
    standard = getattr(args, "implementation", "scratch") == "standard"
    counts = Counter()
    payloads = Counter()
    for value in flatten_state_dict(state_dict)[0].values():
        if not isinstance(value, torch.Tensor):
            continue
        tensor = value.to_local() if isinstance(value, DTensor) else value
        if (
            not standard
            and not tensor.is_meta
            and (tensor.device.type != "cpu" or tensor.ndim != 0)
        ):
            raise RuntimeError(
                "Scratch reference requires meta tensors and CPU scalar counters; "
                f"found {tensor.device} storage with shape {tuple(tensor.shape)}"
            )
        counts[str(tensor.device)] += 1
        payloads[str(tensor.device)] += tensor.nbytes
    if standard:
        bad = [device for device in counts if torch.device(device).type != requested]
        if bad:
            raise RuntimeError(f"Reference storage requested on {requested}, found {bad}")
        if requested == "cuda" and len(counts) != 1:
            raise RuntimeError("CUDA reference requires nonempty storage on one local GPU")
    state_device = (
        next(iter(counts)) if standard and len(counts) == 1 else (requested if standard else "meta")
    )
    report = {
        "state_device": state_device,
        "standard_device": requested,
        "cpu_state_resident": standard and requested == "cpu" and bool(counts),
        "cuda_state_resident": standard and requested == "cuda" and bool(counts),
        "local_tensor_devices": dict(sorted(counts.items())),
        "local_tensor_payload_bytes_by_device": dict(sorted(payloads.items())),
    }
    if standard and requested == "cuda":
        device = torch.device(state_device)
        expected = getattr(args, "_reference_device", state_device)
        if device != torch.device(expected):
            raise RuntimeError(f"CUDA reference storage is on {device}, expected {expected}")
        properties = torch.cuda.get_device_properties(device)
        visible_count = torch.cuda.device_count()
        mapping, _ = _cuda_mapping(args, visible_count)
        report.update(
            cuda_mapping_report(
                mapping,
                visible_count,
                allow_sharing=getattr(args, "cuda_share_devices", False),
            )
        )
        report.update(
            cuda_device_index=device.index,
            cuda_device_name=properties.name,
            cuda_compute_capability=[properties.major, properties.minor],
            cuda_total_memory_bytes=properties.total_memory,
            cuda_memory_allocated_bytes=torch.cuda.memory_allocated(device),
            cuda_memory_reserved_bytes=torch.cuda.memory_reserved(device),
            cuda_peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
            cuda_peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            cuda_runtime_version=torch.version.cuda,
            cuda_compiled_architectures=torch.cuda.get_arch_list(),
            cuda_checkpoint_only=True,
        )
    return report
