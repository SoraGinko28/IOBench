"""One rank's FSDP gdtensor save/load; the global model stays on Meta."""

import json
import math
import os
import tempfile
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from time import perf_counter, perf_counter_ns

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint._nested_dict import flatten_state_dict
from torch.distributed.checkpoint.metadata import TensorStorageMetadata
from torch.distributed.tensor import DTensor

from metasl.checkpoint_state import checkpoint_state, content_audit, dcp_file_layout
from metasl.iostats import DATA_SUFFIXES, IORecorder, RecordedFileSystem, measure_call
from metasl.layout import job_layout
from metasl.procmem import memory
from metasl.reference import initialize_reference, reference_report, standard_device
from metasl.sharding import POLICY, audit_metadata, element_size, fill_or_check, local_tensor
from metasl.skeleton import build_model, decoder_layers, require_torch
from metasl.storage import AbortOnReaderFailure, local_metadata_broadcast

def measure_dcp(fn, *, rank=None, action=None, checkpoint=None, world_size=None, cuda_device=None):
    if cuda_device is not None:
        torch.cuda.synchronize(cuda_device)
    if dist.is_initialized():
        dist.barrier()  # Complete construction on every rank before the phase timer.
    gpu_before = None
    if cuda_device is not None:
        torch.cuda.reset_peak_memory_stats(cuda_device)
        gpu_before = dict(
            allocated_bytes=torch.cuda.memory_allocated(cuda_device),
            reserved_bytes=torch.cuda.memory_reserved(cuda_device),
        )
    start, ended = perf_counter(), None
    start_ns, ended_ns = perf_counter_ns(), None
    dcp_started, dcp_ended, dcp_started_ns, dcp_ended_ns = None, None, None, None

    def call():
        nonlocal ended, ended_ns, dcp_started, dcp_ended, dcp_started_ns, dcp_ended_ns
        dcp_started = perf_counter()
        dcp_started_ns = perf_counter_ns()
        result = fn()
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        dcp_ended_ns = perf_counter_ns()
        dcp_ended = perf_counter()
        return result

    result, stats = measure_call(call)
    stats["phase_started"] = start
    stats["phase_ended"] = ended if ended is not None else perf_counter()
    stats["dcp_started"] = dcp_started
    stats["dcp_ended"] = dcp_ended
    stats["dcp_wall_seconds"] = dcp_ended - dcp_started
    stats["phase_started_ns"] = start_ns
    stats["phase_ended_ns"] = ended_ns if ended_ns is not None else perf_counter_ns()
    stats["dcp_started_ns"] = dcp_started_ns
    stats["dcp_ended_ns"] = dcp_ended_ns
    if cuda_device is not None:
        stats["cuda_synchronized"] = True
        stats["cuda_timing_scope"] = (
            "synchronize before call and before return timestamp; includes default DCP "
            "device/host transfers and any enabled in-call content audits"
        )
        stats["cuda_memory"] = dict(
            device=str(cuda_device),
            before=gpu_before,
            allocated_bytes=torch.cuda.memory_allocated(cuda_device),
            reserved_bytes=torch.cuda.memory_reserved(cuda_device),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(cuda_device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(cuda_device),
        )
    return result, stats


def _cuda_reference_device(args):
    if getattr(args, "implementation", "scratch") == "standard" and standard_device(args) == "cuda":
        return torch.device("cuda", torch.cuda.current_device())
    return None


def _checkpoint_file_observation(path, component_names):
    """Stat published local files once; file sizes are global checkpoint inventory."""
    path = Path(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint inventory root is not a directory: {path}")
    files = []
    components = {name: {"data_file_bytes": 0, "data_file_count": 0} for name in component_names}
    for file in sorted(path.rglob("*")):
        if file.is_symlink():
            raise ValueError(f"Checkpoint inventory cannot follow a symlink: {file}")
        if not file.is_file():
            continue
        relative = file.relative_to(path)
        component = (
            component_names[0]
            if len(component_names) == 1
            else relative.parts[0]
            if relative.parts[0] in components
            else None
        )
        size = file.stat().st_size
        data = file.name.endswith(DATA_SUFFIXES)
        files.append(
            {"path": relative.as_posix(), "size_bytes": size, "component": component, "data": data}
        )
        if data and component is not None:
            components[component]["data_file_bytes"] += size
            components[component]["data_file_count"] += 1
    return {
        "data_file_bytes": sum(file["size_bytes"] for file in files if file["data"]),
        "data_file_count": sum(file["data"] for file in files),
        "checkpoint_file_bytes": sum(file["size_bytes"] for file in files),
        "data_files": [file for file in files if file["data"]],
        "data_file_bytes_scope": "checkpoint",
        "components": components,
        "inventory_observation": "rank0 stat inventory after publication; no file payload read",
    }


def _published_observation(
    args,
    runtime,
    path,
    component_names,
    *,
    commit_fn=None,
    commit_end_ns=None,
    component_commit_ends_ns=None,
):
    """Broadcast one coordinator publication boundary and one global inventory."""
    payload = None
    if args.rank == 0:
        try:
            if commit_fn is not None:
                commit_fn()
                commit_end_ns = perf_counter_ns()
            if runtime.get("filesystem") is None:
                try:
                    inventory = _checkpoint_file_observation(path, component_names)
                except (OSError, ValueError) as error:
                    inventory = {"inventory_observation": f"unavailable: {error}"}
            else:
                inventory = {"inventory_observation": "nonlocal filesystem inventory not observed"}
            payload = {
                "commit_end_ns": commit_end_ns,
                "component_commit_ends_ns": component_commit_ends_ns or {},
                "inventory": inventory,
            }
        except BaseException as error:
            payload = {"publication_error": f"{type(error).__name__}: {error}"}
    if dist.is_initialized():
        shared = [payload]
        dist.broadcast_object_list(shared, src=0)
        payload = shared[0]
    if payload is None:
        raise RuntimeError("Checkpoint coordinator observation was not received")
    if "publication_error" in payload:
        raise RuntimeError(f"Checkpoint publication failed: {payload['publication_error']}")
    return payload


def _timeline_observation(
    args, action, stats, recorder, publication, *, component=None, runtime=None
):
    runtime = runtime or {}
    deadline_ns = runtime.get("deadline_ns")
    if (
        deadline_ns is None
        and runtime.get("job_start_ns") is not None
        and runtime.get("observation_seconds") is not None
    ):
        deadline_ns = runtime["job_start_ns"] + round(runtime["observation_seconds"] * 1e9)
    observed = recorder.observation(deadline_ns) if recorder is not None else {}
    unknown = {
        f"app_{category}_{kind}_{metric}": None
        for category in ("data", "metadata")
        for kind in ("read", "write")
        for metric in ("bytes", "calls")
    }
    unknown.update(
        {
            f"app_{category}_{kind}_bytes_{period}": None
            for category in ("data", "metadata")
            for kind in ("read", "write")
            for period in ("within_T", "after_T")
        }
    )
    inventory = publication["inventory"]
    if component is not None:
        sizes = inventory.get("components", {}).get(component, {})
        known_files = inventory.get("data_files")
        files = (
            [file for file in known_files if file["component"] == component]
            if known_files is not None
            else None
        )
    else:
        sizes, files = inventory, inventory.get("data_files")
    timeline = {
        **unknown,
        "observation_schema_version": 1,
        "clock": "monotonic_perf_counter_ns",
        "rank": args.rank,
        "action": action,
        "component": component,
        "call_begin_ns": stats["dcp_started_ns"],
        "call_end_ns": stats["dcp_ended_ns"],
        "commit_end_ns": (
            publication.get("component_commit_ends_ns", {}).get(component)
            if component is not None
            else publication["commit_end_ns"]
        )
        if action == "save"
        else None,
        "durable_end_ns": None,
        "device_write_bytes": None,
        "data_file_bytes": sizes.get("data_file_bytes"),
        "data_file_count": sizes.get("data_file_count"),
        "data_file_bytes_scope": "checkpoint" if component is None else "checkpoint_component",
        "checkpoint_file_bytes": inventory.get("checkpoint_file_bytes")
        if component is None
        else None,
        "data_files": files,
        "inventory_observation": inventory.get("inventory_observation"),
        "recorder_origin_ns": None,
        "first_data_write_start_ns": None,
        "last_data_write_end_ns": None,
        "first_data_read_start_ns": None,
        "last_data_read_end_ns": None,
        "max_simultaneous_data_calls": None,
        "crossing_write_calls": None,
        "app_io_layer": "Python buffered-file API; successful returned bytes, not syscalls",
        "app_io_coverage": "recorded_streams_only" if recorder is not None else "not_recorded",
        "metadata_write_observation": (
            "recorder-observed subset; excludes direct completion-manifest writes, "
            "logs and all unproxied streams"
        ),
        **observed,
    }
    safetensors_read = (
        action == "load" and getattr(args, "checkpoint_format", "dcp") == "safetensors"
    )
    gdtensor_read = safetensors_read and getattr(args, "io_backend", "dcp") == "gdtensor"
    if gdtensor_read and recorder is not None:
        positioned = [
            (event, entry.get("origin_ns", recorder.origin_ns))
            for entry in recorder.files.values()
            for event in entry["events"]
            if event["op"] == "pread" and event.get("io_layer") == "payload"
        ]
        timeline["app_data_read_bytes"] = sum(e["completed"] for e, _ in positioned)
        timeline["app_data_read_calls"] = len(positioned)
        timeline["first_data_read_start_ns"] = min(
            (origin + e["start_ns"] for e, origin in positioned), default=None
        )
        timeline["last_data_read_end_ns"] = max(
            (origin + e["start_ns"] + e["duration_ns"] for e, origin in positioned),
            default=None,
        )
        if deadline_ns is not None:
            for period in ("within_T", "after_T"):
                timeline[f"app_data_read_bytes_{period}"] = sum(
                    e["completed"] for e, origin in positioned
                    if (origin + e["start_ns"] + e["duration_ns"] <= deadline_ns)
                    == (period == "within_T")
                )
        timeline["read_observation"] = "Measured positioned DTensor payload reads"
        timeline["app_io_layer"] = "Measured positioned DTensor payload API and buffered streams"
    for kind in ("read", "write"):
        values = [timeline[f"app_{category}_{kind}_bytes"] for category in ("data", "metadata")]
        timeline[f"app_{kind}_bytes"] = sum(values) if all(v is not None for v in values) else None
    return timeline


def fill_local(state_dict, rows, pattern, check):
    flat = flatten_state_dict(state_dict)[0]
    for row in rows:
        fill_or_check(
            local_tensor(flat[row["fqn"]]),
            row["global_offsets"],
            row["global_shape"],
            row["alias_group"][0],
            pattern,
            check=check,
        )


def _content_audit_report(args):
    mode = content_audit(args)
    if mode != "none":
        return {"content_audit": mode}
    return {
        "content_audit": "none",
        "tensor_content_hash_enabled": False,
        "byte_content_hash_enabled": False,
        "full_value_scan_enabled": False,
        "numerical_values_verified": False,
        "byte_values_verified": False,
        "call_timing_includes_weight_hash": False,
        "call_timing_includes_training_state_hash": False,
        "content_validation_scope": (
            "schema, header and metadata bindings plus actual reader/copy completion; "
            "tensor content hashes and numerical scans disabled"
        ),
    }


def checkpoint_format(args, runtime, state_dict):
    """Validate the model-only gdtensor checkpoint contract."""
    from metasl.gdt_options import validate_backend

    validate_backend(args, runtime)
    content_audit(args)
    dcp_file_layout(args)
    checkpoint_state(args)
    format_name = getattr(args, "checkpoint_format", "dcp")
    if format_name not in ("dcp", "safetensors"):
        raise ValueError(f"Unknown checkpoint format: {format_name}")
    if set(state_dict) != {"model"} or any(
        not isinstance(tensor, torch.Tensor)
        for tensor in flatten_state_dict(state_dict)[0].values()
    ):
        raise ValueError("Checkpoint state must contain model tensors only")
    return format_name


def _prepare_save_phase(args, runtime, state_dict, rows, pool, path, recorder, status_path, phase):
    """One dcp.save of the nested `state_dict`; returns (stats, report fields)."""
    format_name = checkpoint_format(args, runtime, state_dict)
    if (path / ".metadata").exists():
        raise ValueError("Refuse checkpoint overwrite")
    from metasl.gdt_storage import GDTensorFileSystemWriter, GDTensorSavePlanner

    standard = args.implementation == "standard"
    if standard:
        fill_local(state_dict, rows, args.data_pattern, check=False)
    planner = GDTensorSavePlanner(simulated=not standard, pattern=args.data_pattern, rows=rows)
    planner.prepare_save(flatten_state_dict(state_dict)[0])
    default_writer = GDTensorFileSystemWriter
    writer_options = {}
    if format_name == "dcp":
        writer_options["single_file_per_rank"] = dcp_file_layout(args) == "rank"
    bucket_report = None
    if format_name == "safetensors":
        if getattr(args, "safetensors_layout", "per-rank") == "shared":
            from metasl.gdt_shared_hf_storage import SharedGDTensorHuggingFaceStorageWriter

            default_writer = SharedGDTensorHuggingFaceStorageWriter
            maximum = getattr(args, "safetensors_max_file_gib", None)
            if maximum is None and getattr(args, "safetensors_max_file_mib", None) is not None:
                maximum = args.safetensors_max_file_mib / 1024
            writer_options["max_file_gib"] = maximum
        else:
            from metasl.hf_storage import safetensors_file_buckets
            from metasl.gdt_hf_storage import GDTensorHuggingFaceStorageWriter

            default_writer = GDTensorHuggingFaceStorageWriter
            mapping, bucket_report = safetensors_file_buckets(
                rows, getattr(args, "safetensors_max_file_mib", None)
            )
            if mapping is not None:
                writer_options["fqn_to_index_mapping"] = mapping
    writer_factory = runtime.get("writer_factory", default_writer)
    writer = writer_factory(
        path,
        client=runtime.get("client"),
        abort=runtime.get("abort"),
        status_path=status_path,
        thread_count=args.threads,
        sync_files=True,
        overwrite=False,
        **writer_options,
    )
    writer.capture_payload_events = recorder is not None
    if recorder:
        writer.fs = RecordedFileSystem(
            writer.fs, recorder, capture_lifecycle=True,
            capture_open=getattr(args, "safetensors_layout", "per-rank") == "shared",
        )
    phase("planning")
    broadcast = {}
    metadata_context = (
        local_metadata_broadcast(writer, broadcast) if format_name == "dcp" else nullcontext()
    )

    def call():
        with metadata_context:
            return dcp.save(state_dict, storage_writer=writer, planner=planner)

    def finish(metadata):
        # Only the coordinator (or a single process) holds the global Metadata; other
        # ranks hold a view of their own entries, so global figures are reported by rank 0.
        full = broadcast.get("full", True)
        extents = audit_metadata(metadata) if args.rank == 0 and full else []
        tasks, byte_tasks = [], []
        for i in planner.plan.items:
            info = metadata.storage_data[i.index]
            data = i.tensor_data
            if data is None:
                byte_tasks.append(
                    dict(
                        fqn=i.index.fqn,
                        file=info.relative_path,
                        file_offset=info.offset,
                        file_bytes=info.length,
                    )
                )
                continue
            tasks.append(
                dict(
                    fqn=i.index.fqn,
                    offsets=list(data.chunk.offsets),
                    file=info.relative_path,
                    file_offset=info.offset,
                    file_bytes=info.length,
                    shape=list(data.chunk.sizes),
                    dtype=str(data.properties.dtype),
                    bytes=math.prod(data.chunk.sizes) * element_size(data.properties.dtype),
                )
            )
        max_chunk = (
            max(
                (
                    math.prod(c.sizes) * element_size(md.properties.dtype)
                    for md in metadata.state_dict_metadata.values()
                    if isinstance(md, TensorStorageMetadata)
                    for c in md.chunks
                ),
                default=0,
            )
            if full
            else None
        )
        extra = dict(
            checkpoint_format=format_name,
            save_layout="sharded",
            chunk_count=len(metadata.storage_data) if full else None,
            assigned_items=len(tasks),
            writer_limit=args.writer_limit,
            writer_threads=args.threads,
            writer_timing=writer.timing,
            meta_save_source=planner.meta_save_source_report,
            completed_write_indices=writer.completed_write_indices,
            write_tasks=tasks,
            byte_tasks=byte_tasks,
            byte_item_count=len(byte_tasks),
            byte_item_bytes=sum(t["file_bytes"] for t in byte_tasks),
            storage_mapping=extents,
            target_payload_bytes=sum(t["bytes"] for t in tasks),
            file_count=len({info.relative_path for info in metadata.storage_data.values()})
            if full
            else None,
            max_chunk_payload_bytes=max_chunk,
            metadata=broadcast or None,
        )
        extra.update(_content_audit_report(args))
        extra.update(
            io_backend="gdtensor", payload_requests=writer.payload_requests,
            payload_events=writer.payload_events,
            container_patch_events=writer.container_patch_events,
            gdtensor_payload_write_bytes=sum(r["length"] for r in writer.payload_requests),
            gdtensor_buffer_bytes=writer.buffer_bytes,
        )
        if format_name == "dcp":
            extra.update(
                dcp_file_layout=dcp_file_layout(args),
                dcp_single_file_per_rank=writer_options["single_file_per_rank"],
            )
        if format_name == "safetensors":
            shared = getattr(args, "safetensors_layout", "per-rank") == "shared"
            layout_report = getattr(writer, "layout_report", {}) if shared else bucket_report
            extra["serialization_memory_scope"] = (
                "Header-only serialization; DTensor staging is bounded by the largest range"
            )
            file_payloads = {}
            for task in tasks:
                file_payloads[task["file"]] = file_payloads.get(task["file"], 0) + task["bytes"]
            local_files = []
            for name, payload in sorted(file_payloads.items()):
                file = Path(name)
                if not file.is_absolute():
                    file = path / file
                size = file.stat().st_size
                local_files.append(
                    dict(
                        path=file.name,
                        size_bytes=size,
                        data_bytes=payload,
                        header_bytes=(next(
                            row["header_bytes"] for row in layout_report["files"]
                            if row["path"] == file.name
                        ) if shared else size - payload),
                    )
                )
            extra["safetensors_max_file_mib"] = getattr(args, "safetensors_max_file_mib", None)
            extra["safetensors_max_file_gib"] = getattr(args, "safetensors_max_file_gib", None)
            extra["safetensors_layout"] = getattr(args, "safetensors_layout", "per-rank")
            extra["safetensors_file_bucketing"] = {
                **layout_report,
                "actual_local_files": local_files,
                "actual_local_file_count": len(local_files),
                "actual_max_local_file_bytes": max(
                    (file["size_bytes"] for file in local_files), default=0
                ),
                "actual_max_local_file_data_bytes": max(
                    (file["data_bytes"] for file in local_files), default=0
                ),
            }
        return extra

    return call, finish


def _prepare_load_phase(
    args,
    runtime,
    state_dict,
    rows,
    pool,
    path,
    recorder,
    status_path,
    phase,
    optional_keys=(),
):
    """One dcp.load into the nested `state_dict`; returns (stats, report fields)."""
    standard = args.implementation == "standard"
    format_name = checkpoint_format(args, runtime, state_dict)
    target_reset = None
    if standard and content_audit(args) == "none":
        phase("load_target_reset")
        flat = flatten_state_dict(state_dict)[0]
        reset_started = perf_counter_ns()
        before = memory(os.getpid())
        for row in rows:
            tensor = local_tensor(flat[row["fqn"]])
            # The sentinel makes the resident destination visibly different before IO.
            tensor.fill_(float("nan") if tensor.is_floating_point() else -1)
        cuda_device = _cuda_reference_device(args)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
        target_reset = {
            "start_ns": reset_started,
            "end_ns": perf_counter_ns(),
            "memory_before": before,
            "memory_after": memory(os.getpid()),
            "strategy": "resident target reset to NaN sentinel before actual checkpoint load",
            "inside_load_call": False,
        }
    client, abort = runtime.get("client"), runtime.get("abort")
    from metasl.gdt_storage import GDTensorFileSystemReader, GDTensorLoadPlanner

    class ModelReader(GDTensorFileSystemReader):
        def read_metadata(self, *args, **kwargs):
            metadata = super().read_metadata(*args, **kwargs)
            if any(not name.startswith("model.") for name in metadata.state_dict_metadata):
                raise ValueError("Checkpoint state scope mismatch: expected model-only DCP")
            return metadata

    default_reader = ModelReader
    if format_name == "safetensors":
        if getattr(args, "safetensors_layout", "per-rank") == "shared":
            from metasl.gdt_shared_hf_storage import SharedGDTensorHuggingFaceStorageReader

            default_reader = SharedGDTensorHuggingFaceStorageReader
        else:
            from metasl.gdt_hf_storage import GDTensorHuggingFaceStorageReader

            default_reader = GDTensorHuggingFaceStorageReader
    reader_factory = runtime.get("reader_factory", default_reader)
    reader = reader_factory(
        path,
        client=client,
        abort=abort,
        status_path=status_path,
        rows=rows,
        gate=runtime.get("metadata_gate"),
    )
    if abort is not None:
        reader.fs = AbortOnReaderFailure(reader.fs, abort)
    if recorder:
        reader.fs = RecordedFileSystem(
            reader.fs, recorder,
            capture_lifecycle=getattr(args, "safetensors_layout", "per-rank") == "shared",
            capture_open=getattr(args, "safetensors_layout", "per-rank") == "shared",
        )
    value_pattern = args.data_pattern if args.validate_values else None
    planner = GDTensorLoadPlanner(
        rows, simulated=not standard, pattern=value_pattern, optional_keys=optional_keys,
        abort=abort, progress=runtime.get("progress"),
    )
    planner.prepare_load(flatten_state_dict(state_dict)[0])
    phase("planning")

    def call():
        return dcp.load(state_dict, storage_reader=reader, planner=planner)

    def finish(result):
        planner.assert_complete()
        if standard and args.validate_values:
            fill_local(state_dict, rows, args.data_pattern, check=True)
        requests = reader.requests
        extra = dict(
            checkpoint_format=format_name,
            requests=requests,
            expected_items=len(requests),
            completed_items=len(requests),
            completed_elements=sum(math.prod(r["lengths"]) for r in requests),
            target_payload_bytes=sum(r["target_bytes"] for r in requests),
            source_chunk_bytes=sum(r["source_chunk_bytes"] for r in requests),
            referenced_file_count=len({r["file"] for r in requests}),
            referenced_chunk_count=len(
                {(r["fqn"], tuple(r["source_chunk_offsets"])) for r in requests}
            ),
            byte_item_count=reader.byte_items,
            cpu_scalar_items=sum(
                isinstance(t, torch.Tensor) and not t.is_meta and t.ndim == 0
                for t in flatten_state_dict(state_dict)[0].values()
            )
            if not standard
            else 0,
            skipped_keys=planner.skipped_keys,
            complete_load_verified=True,
            numerical_values_verified=args.validate_values,
            default_cpu_planner=standard and standard_device(args) == "cpu",
            default_cuda_planner=standard and standard_device(args) == "cuda",
            private_scratch=not standard,
            scratch_owner_pid=os.getpid(),
            scratch_capacity_bytes=planner.meta_load_sink_report["resident_buffer_bytes"],
            discard_calls=0,
            scratch_page_policy="resident" if planner.meta_load_sink is not None else "discard",
            all_meta=not standard,
            copy_events=getattr(planner, "copy_events", []),
            reader_limit=args.reader_limit,
            reader_timing=dict(reader.timing),
            meta_load_sink=planner.meta_load_sink_report,
            metadata=reader.timing.get("metadata"),
        )
        if client:
            extra["queue_timing"] = reader.timing
        extra.pop("source_chunk_bytes")
        dtensors = {name for name, value in flatten_state_dict(state_dict)[0].items()
                    if isinstance(value, DTensor)}
        extra.update(
            io_backend="gdtensor",
            payload_requests=reader.payload_requests,
            payload_events=reader.payload_events,
            gdtensor_payload_read_bytes=sum(r["length"] for r in reader.payload_requests),
            native_full_chunk_payload_bytes=sum(
                r["source_chunk_bytes"] for r in requests if r["fqn"] in dtensors
            ),
            gdtensor_buffer_bytes=reader.buffer_bytes,
            coverage_crosscheck=True,
            default_cpu_planner=False,
            default_cuda_planner=False,
            private_scratch=False,
        )
        extra.update(_content_audit_report(args))
        if format_name == "safetensors":
            extra["safetensors_layout"] = getattr(args, "safetensors_layout", "per-rank")
        if format_name == "dcp":
            extra.update(
                dcp_file_layout=dcp_file_layout(args),
                dcp_single_file_per_rank=dcp_file_layout(args) == "rank",
            )
        if target_reset is not None and content_audit(args) == "none":
            extra["load_target_reset"] = target_reset
        return extra

    return call, finish


def _prepare_call_cache(args, path, action):
    """Warm only after resident targets exist; probe each rank just before IO."""
    if action != "load" or not getattr(args, "warm_cache", False):
        return None
    from metasl.cache_control import checkpoint_residency
    from metasl.iostats import checkpoint_files, warm_checkpoint

    if dist.is_initialized():
        dist.barrier()
    warmup = None
    if args.rank == 0:
        warmup = warm_checkpoint(path)
        metadata_bytes = 0
        for file in checkpoint_files(path):
            if not file.name.endswith(DATA_SUFFIXES):
                with file.open("rb") as stream:
                    while block := stream.read(1024 * 1024):
                        metadata_bytes += len(block)
        warmup["metadata_bytes"] = metadata_bytes
    if dist.is_initialized():
        dist.barrier()
    probe_start_ns = perf_counter_ns()
    residency = checkpoint_residency(path)
    return dict(
        warmup=warmup, residency=residency, probe_start_ns=probe_start_ns,
        probe_end_ns=perf_counter_ns(), after_resident_buffer_preparation=True,
    )


def _run_checkpoint_phase(
    args,
    runtime,
    state_dict,
    rows,
    pool,
    path,
    recorder,
    status_path,
    phase,
    action,
    optional_keys=(),
):
    phase_begin_ns = perf_counter_ns()
    scope = checkpoint_state(args)
    prepare = _prepare_save_phase if action == "save" else _prepare_load_phase
    options = {"optional_keys": optional_keys} if action == "load" else {}
    call, finish = prepare(
        args, runtime, state_dict, rows, pool, path, recorder, status_path, phase, **options
    )
    # The safety sample covers fully materialized buffers and is acknowledged by
    # every rank before its checkpoint clock starts. Actual admission requests
    # stay inside DCP, preserving their order and permit scheduling.
    client = runtime.get("client")
    admission_preparation = None
    if client is not None:
        admission_preparation = client.prepare()
    cache_before_call = _prepare_call_cache(args, path, action)
    result, stats = measure_dcp(
        call,
        rank=args.rank,
        action=action,
        checkpoint=path,
        world_size=getattr(args, "world_size", 1),
        cuda_device=_cuda_reference_device(args),
    )
    extra = finish(result)
    if admission_preparation is not None:
        extra["admission_preparation"] = admission_preparation
    if cache_before_call is not None:
        extra["cache_before_call"] = cache_before_call
    extra.update(reference_report(args, state_dict))
    extra["checkpoint_state"] = scope
    extra.update(_content_audit_report(args))
    component = "model" if extra.get("checkpoint_format") == "safetensors" else "checkpoint"
    publication = _published_observation(
        args,
        runtime,
        path,
        [component],
        commit_end_ns=stats["dcp_ended_ns"] if action == "save" else None,
        component_commit_ends_ns={component: stats["dcp_ended_ns"]} if action == "save" else None,
    )
    extra["timeline"] = _timeline_observation(
        args, action, stats, recorder, publication, runtime=runtime
    )
    extra["timeline"]["commit_semantics"] = (
        "DCP API returned after global .metadata publication; durable directory fsync not observed"
        if action == "save" and component == "checkpoint"
        else (
            "safetensors DCP API returned after completion manifest publication; "
            "durability not observed"
        )
        if action == "save"
        else "recovery call end; no new checkpoint publication"
    )
    extra["timeline"]["components"] = {
        component: _timeline_observation(
            args, action, stats, recorder, publication, component=component, runtime=runtime
        )
    }
    extra["timeline"]["components"][component]["commit_semantics"] = extra["timeline"][
        "commit_semantics"
    ]
    extra["data_file_bytes"] = extra["timeline"]["data_file_bytes"]
    extra["data_file_count"] = extra["timeline"]["data_file_count"]
    extra["data_file_bytes_scope"] = "checkpoint"
    _finish_timeline(extra["timeline"], action, phase_begin_ns)
    return stats, extra


def _finish_timeline(timeline, action, phase_begin_ns):
    timeline["phase_begin_ns"] = phase_begin_ns
    timeline["phase_end_ns"] = perf_counter_ns()
    timeline[f"{action}_begin_ns"] = phase_begin_ns
    timeline[f"{action}_end_ns"] = timeline["phase_end_ns"]
    timeline["phase_end_semantics"] = (
        "save phase returned after publication, audits and inventory"
        if action == "save"
        else "recovery phase returned after validation and inventory"
    )


def save_phase(args, runtime, state_dict, rows, pool, path, recorder, status_path, phase):
    return _run_checkpoint_phase(
        args, runtime, state_dict, rows, pool, path, recorder, status_path, phase, "save"
    )


def load_phase(
    args, runtime, state_dict, rows, pool, path, recorder, status_path, phase, optional_keys=()
):
    return _run_checkpoint_phase(
        args,
        runtime,
        state_dict,
        rows,
        pool,
        path,
        recorder,
        status_path,
        phase,
        "load",
        optional_keys,
    )


def join_group(args):
    """Initialise gloo from the rank's file rendezvous unless a group already exists (tests)."""
    if dist.is_initialized():
        return False
    rendezvous = args.rendezvous
    if rendezvous is None:
        rendezvous = Path(tempfile.mkdtemp(prefix="metasl-rendezvous-")) / "store"
    dist.init_process_group(
        "gloo",
        init_method=rendezvous.resolve().as_uri(),
        rank=args.rank,
        world_size=args.world_size,
        timeout=timedelta(seconds=max(1800, args.timeout)),
    )
    return True


def run_phase(args, runtime):
    """Run one rank; `runtime` may carry client/abort/progress from the supervisor."""
    from metasl.gdt_options import validate_backend

    validate_backend(args, runtime)
    require_torch()
    content_audit(args)
    dcp_file_layout(args)
    start = perf_counter()
    output = args.output
    status_path = output / f"rank-{args.rank:03d}.status" if output else None

    def phase(name):
        if status_path is not None:
            status_path.write_text(name)

    phase("gloo_init")
    reference_device = initialize_reference(args)
    owns_group = join_group(args)
    init_seconds = perf_counter() - start
    try:
        phase("model_build")
        model, manifest = build_model(args.bundle, "meta")
        standard = args.implementation == "standard"
        targets_started = perf_counter()
        checkpoint_state(args)
        from torch.distributed.checkpoint.state_dict import get_model_state_dict
        from metasl.fsdp import apply_parallelism, describe_targets, materialize

        layout = job_layout(args, len(decoder_layers(model)))
        layout.build_mesh()
        apply_parallelism(model, layout, getattr(args, "compute_dtype", "bf16"))
        state = get_model_state_dict(model)
        if standard:
            state = materialize(state, device=reference_device)
        state_dict = {"model": state}
        rows = describe_targets(state_dict, manifest["structure"])
        targets_seconds = perf_counter() - targets_started
        reference_report(args, state_dict)  # Reject an accidental CUDA-to-CPU conversion before IO.
        if output:
            (output / f"rank-{args.rank:03d}.manifest.json").write_text(json.dumps(rows, indent=2))
        baseline = memory(os.getpid())
        path = args.checkpoint or args.bundle / "checkpoint"
        pool = None
        recorder = IORecorder() if args.trace_io else None
        run = save_phase if args.action == "save" else load_phase
        stats, extra = run(
            args, runtime, state_dict, rows, pool, path, recorder, status_path, phase
        )
        if recorder:
            extra["io"] = recorder.summary(
                output / f"rank-{args.rank:03d}.io.jsonl" if output else None
            )
            if extra.get("io_backend") == "gdtensor" and args.action == "save":
                extra["io"]["write_observation"] = (
                    "Stream events cover envelopes/native items; positioned payload writes "
                    "are recorded separately in payload_requests"
                )
            if (
                extra.get("checkpoint_format", "dcp") == "safetensors"
                and args.action == "load"
                and extra.get("io_backend") == "gdtensor"
            ):
                payload = [e for e in extra.get("payload_events", []) if e["op"] == "pread"]
                extra["io"]["data_read_bytes"] = sum(e["completed"] for e in payload)
                extra["io"]["data_read_calls"] = len(payload)
                extra["io"]["read_observation"] = "Measured positioned DTensor payload reads"
        # FSDP rewrites classes and parameters; the global model must stay Meta.
        assert all(p.is_meta for p in model.parameters())
        assert all(t.is_meta for t in model.state_dict().values())
        # Keep early ranks resident through both DCP's final collective and validation.
        phase("final_sync")
        dist.barrier()
        phase("validated")
        return dict(
            action=args.action,
            distribution="sharded",
            policy_version=rows[0]["policy_version"] if rows else POLICY,
            sharding=getattr(args, "sharding", "fsdp2"),
            compute_dtype=getattr(args, "compute_dtype", "bf16"),
            layout=layout.describe() if layout else None,
            targets_seconds=targets_seconds,
            rank=args.rank,
            world_size=args.world_size,
            pid=os.getpid(),
            backend="gloo",
            process_start_ticks=Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
            init_seconds=init_seconds,
            structure_sha256=manifest["structure_sha256"],
            empty_tensor_count=sum(r["payload_bytes"] == 0 for r in rows),
            replicated_tensor_count=sum(r["replicated"] for r in rows),
            local_manifest=rows,
            logical_bytes=sum(t.nbytes for t in model.state_dict().values()),
            baseline_memory=baseline,
            stats=stats,
            pattern=args.data_pattern,
            **extra,
        )
    finally:
        if owns_group:
            dist.destroy_process_group()
