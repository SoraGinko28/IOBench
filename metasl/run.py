"""Launch a single-node FSDP gdtensor save/load phase with bounded IO concurrency."""

import argparse
import json
import math
import multiprocessing as mp
import multiprocessing.forkserver as forkserver
import multiprocessing.resource_tracker as resource_tracker
import os
import re
import time
import traceback
from pathlib import Path

from metasl.admission import PermitQueue
from metasl.iostats import advise_evict_checkpoint, checkpoint_files
from metasl.procmem import available_memory, memory, process_counters, system_memory

GIB = 1024**3


def worker(args, rank, abort, connection=None, progress=None, gate=None):
    startup = dict(worker_entry_ns=time.perf_counter_ns())
    log = (args.output / f"rank-{rank:03d}.log").open("w", buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    status_path = args.output / f"rank-{rank:03d}.status"
    status_path.write_text("import")
    try:
        startup["import_begin_ns"] = time.perf_counter_ns()
        import torch

        torch.set_num_threads(1)
        from metasl.admission import PermitClient
        from metasl.phase import run_phase

        startup["import_end_ns"] = time.perf_counter_ns()
        args.rank = rank
        args.world_size = args.ranks
        args.rendezvous = args.output / "rendezvous"
        status_path.write_text(args.action)
        runtime = dict(
            abort=abort,
            progress=progress,
            metadata_gate=gate,
        )
        runtime["worker_startup_timing"] = startup
        if connection is not None:
            runtime["client"] = PermitClient(connection, abort, args.timeout)
        startup["run_entry_ns"] = time.perf_counter_ns()
        result = run_phase(args, runtime)
        result["worker_startup_timing"] = startup
        (args.output / f"rank-{rank:03d}.json").write_text(json.dumps(result, indent=2) + "\n")
        status_path.write_text("done")
    except BaseException:
        abort.set()
        status_path.write_text("failed")
        traceback.print_exc()
        raise


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--action", choices=("save", "load"), required=True)
    p.add_argument("--io-backend", choices=("gdtensor",), default="gdtensor")
    p.add_argument("--checkpoint-format", choices=("dcp", "safetensors"), default="dcp")
    p.add_argument("--checkpoint-state", choices=("model",), default="model")
    p.add_argument("--dcp-file-layout", choices=("rank",), default="rank")
    p.add_argument("--safetensors-layout", choices=("per-rank", "shared"), default="per-rank")
    p.add_argument("--safetensors-max-file-mib", type=int)
    p.add_argument("--safetensors-max-file-gib", type=float)
    p.add_argument("--bundle", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--ranks", type=int, default=1)
    p.add_argument("--implementation", choices=("simulated", "standard"), default="simulated")
    p.add_argument("--standard-device", choices=("cpu", "cuda"), default="cpu")
    p.add_argument("--cuda-devices", help="Comma-separated visible GPU ordinals, one per rank")
    p.add_argument("--cuda-share-devices", action="store_true")
    p.add_argument("--sharding", choices=("fsdp2",), default="fsdp2")
    p.add_argument("--parallel", default="", help="Pure FSDP dimensions, e.g. dp_shard=4")
    p.add_argument("--compute-dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--data-pattern", choices=("zero", "position"), default="zero")
    p.add_argument("--validate-values", action="store_true")
    p.add_argument("--content-audit", choices=("auto", "none"), default="auto")
    p.add_argument("--threads", type=int, choices=(1,), default=1)
    p.add_argument("--writer-limit", type=int, default=0)
    p.add_argument("--reader-limit", type=int, default=0)
    p.add_argument("--metadata-limit", type=int, default=4)
    p.add_argument("--trace-io", action="store_true")
    p.add_argument("--warm-cache", action="store_true")
    p.add_argument("--evict-cache", action="store_true")
    p.add_argument("--measure-cache-residency", action="store_true")
    p.add_argument("--start-method", choices=("spawn", "forkserver"), default="forkserver")
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("--min-available-gib", type=float, default=3)
    p.add_argument("--max-pss-gib", type=float, default=26)
    p.add_argument("--io-soft-pss-gib", type=float)
    p.add_argument("--sample-seconds", type=float, default=0.5)
    return p


def io_soft_memory_limits(args):
    soft = getattr(args, "io_soft_pss_gib", None)
    soft = args.max_pss_gib - 2 if soft is None else soft
    if (
        not math.isfinite(args.max_pss_gib)
        or args.max_pss_gib <= 0
        or not math.isfinite(soft)
        or not 0 < soft <= args.max_pss_gib
        or not math.isfinite(args.min_available_gib)
        or args.min_available_gib <= 0
    ):
        raise ValueError("IO soft PSS must be finite, positive and at most the hard PSS limit")
    return dict(
        soft_pss_bytes=soft * GIB,
        soft_available_bytes=(args.min_available_gib + 2) * GIB,
    )


def preflight_checkpoint(args):
    """Check local HF targets before output creation without importing torch."""
    if args.checkpoint_format != "safetensors":
        return
    checkpoint = args.checkpoint or args.bundle / "checkpoint"
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:/", str(checkpoint)):
        raise ValueError("safetensors requires a local filesystem checkpoint path")
    if args.action == "save":
        if checkpoint.exists() and (not checkpoint.is_dir() or any(checkpoint.iterdir())):
            raise ValueError("safetensors save requires a new or empty checkpoint directory")
        return
    marker = checkpoint / ".metasl-complete.json"
    try:
        completed = json.loads(marker.read_text())
    except (OSError, ValueError) as exc:
        raise ValueError("safetensors load requires a valid .metasl-complete.json") from exc
    if (
        not isinstance(completed, dict)
        or type(completed.get("schema_version")) is not int
        or completed.get("schema_version") != 1
        or completed.get("checkpoint_format") != "safetensors"
        or completed.get("checkpoint_state", "model") != "model"
    ):
        raise ValueError("safetensors checkpoint completion marker has the wrong schema or format")


def parse_args(argv=None):
    p = build_parser()
    args = p.parse_args(argv)
    from metasl.gdt_options import validate_backend
    from metasl.checkpoint_state import content_audit

    try:
        validate_backend(args)
        io_soft_memory_limits(args)
        content_audit(args)
    except (ValueError, TypeError) as exc:
        p.error(str(exc))
    if args.ranks < 1 or min(args.writer_limit, args.reader_limit, args.metadata_limit) < 0:
        p.error("ranks must be positive and IO limits nonnegative")
    if args.timeout <= 0 or not math.isfinite(args.sample_seconds) or args.sample_seconds <= 0:
        p.error("timeout and sample-seconds must be finite and positive")
    cuda = args.implementation == "standard" and args.standard_device == "cuda"
    if (args.cuda_share_devices or args.cuda_devices or args.standard_device == "cuda") and not cuda:
        p.error("CUDA options require --implementation standard --standard-device cuda")
    if args.cuda_devices:
        try:
            devices = [int(item) for item in args.cuda_devices.split(",")]
            if (len(devices) != args.ranks or min(devices) < 0
                    or (not args.cuda_share_devices and len(set(devices)) != len(devices))):
                raise ValueError
        except ValueError:
            p.error("--cuda-devices requires one nonnegative ordinal per rank; repeated ordinals require --cuda-share-devices")
    if args.safetensors_max_file_mib is not None and (
        args.checkpoint_format != "safetensors" or args.safetensors_max_file_mib <= 0
    ):
        p.error("--safetensors-max-file-mib requires safetensors and a positive size")
    if args.safetensors_layout == "shared" and args.checkpoint_format != "safetensors":
        p.error("--safetensors-layout shared requires safetensors")
    if args.safetensors_max_file_gib is not None and (
        args.checkpoint_format != "safetensors" or args.safetensors_layout != "shared"
        or not math.isfinite(args.safetensors_max_file_gib)
        or args.safetensors_max_file_gib <= 0
    ):
        p.error("--safetensors-max-file-gib requires shared safetensors and a finite positive size")
    if args.safetensors_max_file_gib is not None and args.safetensors_max_file_mib is not None:
        p.error("choose one safetensors maximum file size unit")
    if (args.warm_cache or args.evict_cache) and args.action != "load":
        p.error("--warm-cache and --evict-cache apply to load")
    if args.warm_cache and args.evict_cache:
        p.error("--warm-cache excludes --evict-cache")
    if args.measure_cache_residency and not (args.warm_cache or args.evict_cache):
        p.error("--measure-cache-residency requires --warm-cache or --evict-cache")
    if args.output.exists():
        p.error("Output must be a new directory (isolates rendezvous and results)")
    try:
        preflight_checkpoint(args)
    except ValueError as exc:
        p.error(str(exc))
    return args


def metadata_summary(records):
    """Group view of the per-rank .metadata read, gate wait and pruning."""
    records = [r for r in records if r]
    if not records:
        return None
    summary = dict(
        ranks=len(records),
        gate=records[0]["gate"],
        pruned=records[0]["pruned"],
        max_wait_seconds=max(r["read_start"] - r["queue_enter"] for r in records),
        max_read_seconds=max(r["read_end"] - r["read_start"] for r in records),
        span_seconds=max(r.get("prune_end", r["read_end"]) for r in records)
        - min(r["queue_enter"] for r in records),
    )
    if summary["pruned"]:
        summary.update(
            max_prune_seconds=max(r["prune_end"] - r["read_end"] for r in records),
            chunks_before=records[0]["chunks_before"],
            max_chunks_after=max(r["chunks_after"] for r in records),
            sum_chunks_after=sum(r["chunks_after"] for r in records),
            max_pss_after_prune_bytes=max(r["pss_after_prune"] for r in records),
            max_pss_unpickle_bytes=max(
                r["pss_before_prune"] - r["pss_before_read"]
                for r in records
                if "pss_before_read" in r
            ),
            max_pss_freed_bytes=max(r["pss_before_prune"] - r["pss_after_prune"] for r in records),
        )
    return summary


def broadcast_summary(records):
    """Group view of DCP's end-of-save Metadata broadcast (rank 0 keeps the global object)."""
    records = [r for r in records if r and r.get("collective")]
    if not records:
        return None
    return dict(
        ranks=len(records),
        payload_bytes=records[0]["payload_bytes"],
        full_ranks=sum(r["full"] for r in records),
        unpickled_ranks=sum(bool(r.get("unpickled")) for r in records),
        max_broadcast_seconds=max(r["broadcast_end"] - r["broadcast_start"] for r in records),
        span_seconds=max(r["broadcast_end"] for r in records)
        - min(r["broadcast_start"] for r in records),
        max_pss_after_bytes=max((r.get("pss_after", 0) for r in records), default=0),
    )


def sample_group(processes, last_counters, output, queue, progress, started):
    ticket = queue.begin_sample() if queue else None
    sample_start_ns = ticket["sample_start_ns"] if ticket else time.perf_counter_ns()
    live = [proc for proc in processes if proc.is_alive()]
    values = [memory(proc.pid) for proc in live]
    for proc in live:
        counts = process_counters(proc.pid)
        if counts is not None:
            last_counters[proc.pid] = counts
    controller = memory(os.getpid())
    helper_pids = [resource_tracker._resource_tracker._pid, forkserver._forkserver._forkserver_pid]
    helpers = [memory(pid) for pid in helper_pids if pid]
    helper_pss = sum(v["pss"] for v in helpers)
    phases = {}
    for f in output.glob("*.status"):
        phase = f.read_text()
        phases[phase] = phases.get(phase, 0) + 1
    everyone = values + helpers + [controller]
    sample = dict(
        seconds=time.perf_counter() - started,
        live=len(live),
        phases=phases,
        worker_pss_bytes=sum(v["pss"] for v in values),
        worker_rss_bytes=sum(v["rss"] for v in values),
        controller_pss_bytes=controller["pss"],
        helper_pss_bytes=helper_pss,
        available_bytes=available_memory(),
        worker_counters={
            key: sum(value[key] for value in last_counters.values())
            for key in ("cpu_seconds", "read_bytes", "write_bytes", "rchar", "wchar")
        },
        workers={str(proc.pid): value for proc, value in zip(live, values, strict=True)},
        system_memory=system_memory(),
        total_private_bytes=sum(v.get("private_bytes", 0) for v in everyone),
        total_shmem_pss_bytes=sum(v.get("pss_shmem", 0) for v in everyone),
        total_anon_pss_bytes=sum(v.get("pss_anon", 0) for v in everyone),
        total_file_pss_bytes=sum(v.get("pss_file", 0) for v in everyone),
    )
    sample["total_pss_bytes"] = sample["worker_pss_bytes"] + controller["pss"] + helper_pss
    sample["sample_start_ns"] = sample_start_ns
    sample["sample_end_ns"] = time.perf_counter_ns()
    if queue:
        completed = [value.value for value in progress]
        sample["admission_sample_ticket"] = ticket
        sample.update(
            queue.snapshot(), rank_completed_bytes=completed, completed_logical_bytes=sum(completed)
        )
    return sample, live


def advance_sampling_deadline(deadline_ns, now_ns, interval_ns):
    """An early request sample does not postpone the periodic protection deadline."""
    if now_ns < deadline_ns:
        return deadline_ns
    return deadline_ns + ((now_ns - deadline_ns) // interval_ns + 1) * interval_ns


def wait_for_sample(queue, periodic_deadline_ns):
    remaining = max(0, periodic_deadline_ns - time.perf_counter_ns()) / 1_000_000_000
    if queue:
        queue.sample_needed.wait(remaining)
    else:
        time.sleep(remaining)


def read_rank_reports(output):
    """Keep recoverable timelines when a killed worker left a truncated rank report."""
    reports, errors = [], []
    for path in sorted(output.glob("rank-[0-9][0-9][0-9].json")):
        try:
            report = json.loads(path.read_text())
            rank = int(path.stem.rsplit("-", 1)[-1])
            if not isinstance(report, dict) or type(report.get("rank")) is not int:
                raise ValueError("Rank report must be a mapping with an integer rank")
            if report["rank"] != rank:
                raise ValueError("Rank report filename and rank differ")
            reports.append(report)
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f"Invalid rank report {path.name}: {exc}")
    return reports, errors


def main(argv=None):
    job_start_ns = time.perf_counter_ns()
    args = parse_args(argv)
    supervisor_timing = dict(job_start_ns=job_start_ns, arguments_parsed_ns=time.perf_counter_ns())
    args.output.mkdir(parents=True)
    checkpoint = args.checkpoint or args.bundle / "checkpoint"
    cache_warmup = None
    cache = None
    if args.evict_cache:
        cache = advise_evict_checkpoint(checkpoint)
    if args.warm_cache:
        # Workers first allocate/fault their source and sink buffers, then warm
        # the checkpoint and record residency immediately before the timed call.
        cache = dict(warmup_deferred_to_workers=True)
    if args.evict_cache and args.measure_cache_residency:
        from metasl.cache_control import checkpoint_residency

        key = "residency_after_eviction" if args.evict_cache else "residency_after_warmup"
        cache[key] = checkpoint_residency(checkpoint)
    if cache is not None:
        (args.output / "cache.json").write_text(json.dumps(cache, indent=2) + "\n")
    for name in (
        "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("TORCH_GLOO_LAZY_INIT", "1")
    if args.start_method == "forkserver":
        # Import-only preload; models and process groups are constructed in workers.
        mp.set_forkserver_preload(["metasl.preload"])
    context = mp.get_context(args.start_method)
    supervisor_timing["context_ready_ns"] = time.perf_counter_ns()
    supervisor_timing["launch_begin_ns"] = time.perf_counter_ns()
    supervisor_timing["rank_launches"] = []
    processes = []
    errors = []
    samples = []
    last_counters = {}
    started = time.perf_counter()
    queue = None
    abort = context.Event()
    limit = args.reader_limit if args.action == "load" else args.writer_limit
    try:
        pipes = [context.Pipe() for _ in range(args.ranks)] if limit else []
        progress = [context.Value("Q", 0) for _ in range(args.ranks)] if limit else []
        gate = context.BoundedSemaphore(args.metadata_limit) if args.metadata_limit else None
        for rank in range(args.ranks):
            launch_begin_ns = time.perf_counter_ns()
            proc = context.Process(
                target=worker,
                args=(
                    args,
                    rank,
                    abort,
                    pipes[rank][1] if pipes else None,
                    progress[rank] if progress else None,
                    gate,
                ),
            )
            proc.start()
            supervisor_timing["rank_launches"].append(
                dict(rank=rank, begin_ns=launch_begin_ns, end_ns=time.perf_counter_ns())
            )
            processes.append(proc)
            if available_memory() < args.min_available_gib * GIB:
                raise RuntimeError("Stopped during launch at host MemAvailable safety threshold")
        supervisor_timing["launch_end_ns"] = time.perf_counter_ns()
        if limit:
            for _parent, child in pipes:
                child.close()
            queue = PermitQueue(
                [pair[0] for pair in pipes],
                abort,
                limit,
                args.output,
                "reader" if args.action == "load" else "writer",
                **io_soft_memory_limits(args),
                max_sample_age_seconds=args.sample_seconds,
            )
            queue.thread.start()
        with (args.output / "memory.jsonl").open("w", buffering=1) as out:
            interval_ns = round(args.sample_seconds * 1_000_000_000)
            next_periodic_ns = time.perf_counter_ns()
            while True:
                sampling_started_ns = time.perf_counter_ns()
                sample_reason = (
                    "periodic" if time.perf_counter_ns() >= next_periodic_ns else "admission_baseline"
                )
                sample, live = sample_group(
                    processes, last_counters, args.output, queue, progress, started
                )
                samples.append(sample)
                sample["sample_reason"] = sample_reason
                sample["periodic_deadline_ns"] = next_periodic_ns
                try:
                    if queue:
                        queue.check_memory(sample)
                    if queue and queue.error:
                        raise RuntimeError(queue.error)
                    if queue and abort.is_set():
                        raise RuntimeError("Group aborted; terminating all ranks")
                    failures = [
                        (i, proc.exitcode)
                        for i, proc in enumerate(processes)
                        if proc.exitcode not in (None, 0)
                    ]
                    if failures:
                        raise RuntimeError(f"Worker failures: {failures}; inspect rank logs")
                    if not live:
                        break
                    if sample["available_bytes"] < args.min_available_gib * GIB:
                        raise RuntimeError("Stopped at host MemAvailable safety threshold")
                    if sample["total_pss_bytes"] > args.max_pss_gib * GIB:
                        raise RuntimeError("Stopped at total PSS limit")
                    if time.perf_counter() - started > args.timeout:
                        raise TimeoutError("Supervisor wall time limit")
                    if queue:
                        # No grant may precede the full supervisor safety/failure checks above.
                        sample["supervisor_safety_checked_ns"] = time.perf_counter_ns()
                        queue.update(sample, sample.get("completed_logical_bytes", 0))
                        sample.update(queue.snapshot())
                finally:
                    out.write(json.dumps(sample) + "\n")
                next_periodic_ns = advance_sampling_deadline(
                    next_periodic_ns,
                    sample.get("sample_start_ns", sampling_started_ns),
                    interval_ns,
                )
                wait_for_sample(queue, next_periodic_ns)
    except BaseException as exc:
        errors.append(str(exc))
    finally:
        if errors:
            abort.set()
        if queue:
            queue.close()
        for proc in processes:
            if proc.is_alive():
                proc.terminate()
        for proc in processes:
            proc.join(timeout=5)
            if proc.is_alive():
                proc.kill()
                proc.join()
    reports, report_errors = read_rank_reports(args.output)
    errors.extend(report_errors)
    summary = dict(
        action=args.action,
        io_backend=getattr(args, "io_backend", "dcp"),
        checkpoint_format=args.checkpoint_format,
        checkpoint_state=args.checkpoint_state,
        ranks=args.ranks,
        distribution="sharded",
        measurement_kind="complete_load" if args.action == "load" else args.action,
        start_method=args.start_method,
        configuration={
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if not key.startswith("_")
        },
        completed=len(reports),
        errors=errors,
        wall_seconds=time.perf_counter() - started,
        supervisor_started_perf=started,
        job_start_ns=job_start_ns,
        job_start_source="launcher_before_launch",
        supervisor_timing=supervisor_timing,
        job_wall_seconds=(time.perf_counter_ns() - job_start_ns) / 1_000_000_000,
        cache_warmup=cache_warmup,
        peak_sampled_total_pss_bytes=max((s["total_pss_bytes"] for s in samples), default=0),
        peak_sampled_worker_rss_bytes=max((s["worker_rss_bytes"] for s in samples), default=0),
        sample_seconds=args.sample_seconds,
        note="PSS includes supervisor and multiprocessing helpers; samples are not exact peaks",
        cache_policy=(
            "Workers warm data after buffer allocation; verify pre-call residency and read_bytes"
            if args.warm_cache
            else "File-level eviction attempted; cold cache not guaranteed"
            if args.evict_cache
            else "No page cache eviction; cached file access is allowed"
        ),
        reports=reports,
        report_diagnostics=report_errors,
        standard_device=getattr(args, "standard_device", "cpu"),
        reference_device=getattr(args, "standard_device", "cpu"),
        state_devices=[r.get("state_device") for r in reports],
    )
    cuda_reports = [report for report in reports if report.get("cuda_state_resident")]
    if cuda_reports:
        from metasl.reference import cuda_mapping_report

        indices = {report["rank"]: report["cuda_device_index"] for report in cuda_reports}
        visible_counts = [
            report["cuda_visible_device_count"] for report in cuda_reports
            if report.get("cuda_visible_device_count") is not None
        ]
        summary.update(
            cuda_mapping_report(
                [indices.get(rank) for rank in range(args.ranks)],
                max(visible_counts) if visible_counts else None,
                allow_sharing=getattr(args, "cuda_share_devices", False),
            )
        )
    if reports:
        span = max(r["stats"]["phase_ended"] for r in reports) - min(
            r["stats"]["phase_started"] for r in reports
        )
        payload = sum(r["target_payload_bytes"] for r in reports)
        files = checkpoint_files(checkpoint)
        data_suffixes = (".safetensors",) if args.checkpoint_format == "safetensors" else (".distcp",)
        summary.update(
            dcp_seconds=span,
            dcp_max_rank_seconds=(
                max(r["stats"]["dcp_wall_seconds"] for r in reports)
                if all("dcp_wall_seconds" in r["stats"] for r in reports)
                else None
            ),
            aggregate_logical_bytes=payload,
            aggregate_logical_gib_s=payload / span / GIB,
            checkpoint_bytes=sum(f.stat().st_size for f in files),
            checkpoint_files=len(files),
            data_file_count=sum(f.suffix in data_suffixes for f in files),
            data_file_bytes=sum(f.stat().st_size for f in files if f.suffix in data_suffixes),
            aggregate_process_read_bytes=sum(
                r["stats"]["proc_io_delta"]["read_bytes"] for r in reports
            ),
            aggregate_process_write_bytes=sum(
                r["stats"]["proc_io_delta"]["write_bytes"] for r in reports
            ),
            sum_worker_process_peak_rss_bytes=sum(
                r["stats"]["process_high_water_rss_bytes"] for r in reports
            ),
            aggregate_source_chunk_bytes=None,
            metadata=(broadcast_summary if args.action == "save" else metadata_summary)(
                [r.get("metadata") for r in reports]
            ),
            aggregate_file_api_read_bytes=(
                sum(r["io"].get("data_read_bytes", 0) for r in reports)
                if all("io" in r for r in reports) else None
            ),
            file_api_read_observation=(
                "Measured positioned DTensor payload reads; requires --trace-io"
                if args.checkpoint_format == "safetensors"
                else "Python buffered-file API; requires --trace-io"
            ),
            note_bandwidth="Logical payload, not block-device bandwidth; "
            "process IO counters also include runtime/metadata",
        )
        if summary["io_backend"] == "gdtensor":
            fields = (
                ("gdtensor_payload_read_bytes", "native_full_chunk_payload_bytes")
                if args.action == "load"
                else ("gdtensor_payload_write_bytes",)
            )
            summary.update(
                {
                    f"aggregate_{field}": (
                        sum(report[field] for report in reports)
                        if all(field in report for report in reports) else None
                    )
                    for field in fields
                }
            )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "reports"}), flush=True)
    if errors or len(reports) != args.ranks:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
