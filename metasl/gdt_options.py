"""Validate the supported single-node gdtensor checkpoint configuration."""

import sys

from metasl.layout import parse_parallel


def validate_backend(args, runtime=None):
    backend = getattr(args, "io_backend", "gdtensor")
    if backend != "gdtensor":
        raise ValueError(f"gdtensor is the only IO backend: {backend}")
    if sys.version_info < (3, 12):
        raise ValueError("gdtensor IO requires Python 3.12 or newer")
    capacity = getattr(args, "gdtensor_cuda_staging_bytes", None)
    if capacity is not None and capacity < 8:
        raise ValueError("CUDA staging buffer must hold at least one tensor element")
    mode = getattr(args, "gdtensor_cuda_staging_mode", None)
    if capacity is not None and mode in ("pageable", "pinned"):
        raise ValueError("CUDA staging bytes requires double-buffer mode")
    if getattr(args, "action", None) not in ("save", "load"):
        raise ValueError("gdtensor IO supports standalone save/load")
    format_name = getattr(args, "checkpoint_format", "dcp")
    if format_name not in ("dcp", "safetensors"):
        raise ValueError("gdtensor IO supports DCP or safetensors")
    if getattr(args, "checkpoint_state", "model") != "model":
        raise ValueError("gdtensor IO requires model-only state")
    if getattr(args, "sharding", "fsdp2") != "fsdp2":
        raise ValueError("gdtensor IO requires --sharding fsdp2")
    if getattr(args, "threads", 1) != 1:
        raise ValueError("gdtensor IO requires --threads 1")
    if getattr(args, "dcp_file_layout", "rank") != "rank":
        raise ValueError("gdtensor IO requires a single file per rank")
    if getattr(args, "standard_device", "cpu") not in ("cpu", "cuda"):
        raise ValueError("gdtensor IO supports CPU or CUDA reference storage")
    if getattr(args, "data_pattern", "zero") not in ("zero", "position"):
        raise ValueError(
            "gdtensor CLI supports zero/position patterns; random audits are unsupported"
        )
    if str(getattr(args, "checkpoint", "")).startswith("hdfs:"):
        raise ValueError("gdtensor IO requires local files; HDFS is unsupported")
    if runtime and runtime.get("filesystem") is not None:
        raise ValueError("gdtensor IO requires local files; HDFS is unsupported")
    parallel = getattr(args, "parallel", "")
    if parallel:
        dims = parse_parallel(parallel)
        if any(getattr(dims, name) != 1 for name in ("pp", "tp", "dp_replicate", "cp")):
            raise ValueError("gdtensor IO requires pure FSDP: pp=tp=dp_replicate=cp=1")
        world = getattr(args, "world_size", getattr(args, "ranks", dims.world_size))
        if dims.world_size != world:
            raise ValueError("gdtensor IO parallel dimensions must match the world size")
