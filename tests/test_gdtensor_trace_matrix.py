"""Matrix admission, evidence checks, and cleanup never need full model payloads."""

import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_gdtensor_trace_matrix as matrix


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def phase_fixture(
    tmp_path, *, action="load", device="meta", output=None,
    data_pattern="position", validate_values=True, checkpoint_format="dcp",
):
    output = output or tmp_path / "phase"
    expected_device = "cuda:0" if device == "cuda" else device
    report = dict(
        rank=0,
        world_size=1,
        action=action,
        io_backend="gdtensor",
        state_device=expected_device,
        cpu_state_resident=device == "cpu",
        cuda_state_resident=device == "cuda",
        local_tensor_devices={expected_device: 1},
        complete_load_verified=True,
        numerical_values_verified=action == "load" and validate_values,
        coverage_crosscheck=True,
        requests=[dict(lengths=[2, 3])],
        completed_items=1,
        expected_items=1,
        completed_elements=6,
        stats=dict(cuda_synchronized=True),
        io=dict(origin_ns=100),
        assigned_items=1,
        completed_write_indices=["item"],
    )
    configuration = dict(
        io_backend="gdtensor",
        ranks=1,
        checkpoint_state="model",
        checkpoint_format=checkpoint_format,
        sharding="fsdp2",
        compute_dtype="bf16",
        data_pattern=data_pattern,
        validate_values=action == "load" and validate_values,
        threads=1,
    )
    write_json(
        output / "summary.json",
        dict(completed=1, errors=[], action=action, configuration=configuration),
    )
    write_json(output / "rank-000.json", report)
    (output / "rank-000.io.jsonl").write_text("{}\n")
    return output, report


def test_phase_commands_use_only_gdtensor_and_explicit_cuda_rank_sharing():
    for device in matrix.DEVICES:
        command = matrix.phase_command(
            "/venv/bin/python", "bundle", "checkpoint", "out", 16, "load", device
        )
        assert command[:3] == ["/venv/bin/python", "-m", "metasl.run"]
        for key, value in (
            ("--io-backend", "gdtensor"),
            ("--parallel", "dp_shard=16"),
            ("--max-pss-gib", "26"),
            ("--io-soft-pss-gib", "26"),
            ("--min-available-gib", "3"),
            ("--data-pattern", "position"),
        ):
            assert command[command.index(key) + 1] == value
        assert "--trace-io" in command and "--validate-values" in command
        if device == "cuda":
            assert "--cuda-share-devices" in command
            assert command[command.index("--cuda-devices") + 1] == ",".join(["0"] * 16)
    for device in matrix.DEVICES:
        save = matrix.phase_command("python", "b", "c", "o", 16, "save", device)
        assert "--trace-io" in save
        assert "--validate-values" not in save
        assert save[save.index("--implementation") + 1] == (
            "simulated" if device == "meta" else "standard"
        )
        assert save[save.index("--writer-limit") + 1] == "1"
        if device == "cuda":
            assert "--cuda-share-devices" in save
            assert save[save.index("--cuda-devices") + 1] == ",".join(["0"] * 16)


@pytest.mark.parametrize("action", ["save", "load"])
@pytest.mark.parametrize("device", matrix.DEVICES)
def test_zero_timing_commands_disable_value_validation_on_every_device(action, device):
    command = matrix.phase_command(
        "python", "bundle", "checkpoint", "out", 1, action, device,
        data_pattern="zero", validate_values=False,
    )
    assert command[command.index("--data-pattern") + 1] == "zero"
    assert "--validate-values" not in command
    assert "--trace-io" in command


def test_matrix_cli_keeps_position_validation_defaults_and_accepts_zero_timing():
    defaults = matrix.parse_args(["--output", "out"])
    assert defaults.data_pattern == "position"
    assert defaults.validate_values is True
    assert defaults.cache_control is False
    assert defaults.checkpoint_format == "dcp"
    assert defaults.safetensors_max_file_mib is None
    assert defaults.writer_limit == 1
    timing = matrix.parse_args([
        "--output", "out", "--data-pattern", "zero", "--no-validate-values",
    ])
    assert timing.data_pattern == "zero"
    assert timing.validate_values is False
    explicit = matrix.parse_args(["--output", "out", "--validate-values"])
    assert explicit.validate_values is True
    assert matrix.parse_args(["--output", "out", "--cache-control"]).cache_control is True
    with pytest.raises(SystemExit):
        matrix.parse_args(["--output", "out", "--data-pattern", "random"])


@pytest.mark.parametrize("writer_limit", [0, 1, 8])
def test_writer_limit_is_configurable_for_saves_and_leaves_loads_unlimited(writer_limit):
    for action in ("save", "load"):
        command = matrix.phase_command("python", "b", "c", "o", 8, action,
                                       writer_limit=writer_limit)
        option = "--writer-limit" if action == "save" else "--reader-limit"
        assert command[command.index(option) + 1] == str(writer_limit if action == "save" else 0)
        assert ("--writer-limit" in command) is (action == "save")
    assert matrix.parse_args(["--output", "out", "--writer-limit", str(writer_limit)]).writer_limit == writer_limit


def test_negative_writer_limit_is_rejected_by_cli_and_command_builder():
    with pytest.raises(SystemExit):
        matrix.parse_args(["--output", "out", "--writer-limit", "-1"])
    with pytest.raises(ValueError, match="nonnegative"):
        matrix.phase_command("python", "b", "c", "o", 8, "save", writer_limit=-1)


def test_matrix_cli_accepts_safetensors_and_validates_file_buckets():
    args = matrix.parse_args([
        "--output", "out", "--checkpoint-format", "safetensors",
        "--safetensors-max-file-mib", "64",
    ])
    assert args.checkpoint_format == "safetensors"
    assert args.safetensors_max_file_mib == 64
    for options in (
        ["--safetensors-max-file-mib", "64"],
        ["--checkpoint-format", "safetensors", "--safetensors-max-file-mib", "0"],
        ["--checkpoint-format", "hybrid"],
    ):
        with pytest.raises(SystemExit):
            matrix.parse_args(["--output", "out", *options])


@pytest.mark.parametrize("action", ["save", "load"])
@pytest.mark.parametrize("device", matrix.DEVICES)
def test_safetensors_phase_commands_retain_tracing_and_bucket_layout(action, device):
    command = matrix.phase_command(
        "python", "bundle", "checkpoint", "out", 1, action, device,
        checkpoint_format="safetensors", safetensors_max_file_mib=64,
    )
    assert command[command.index("--checkpoint-format") + 1] == "safetensors"
    assert command[command.index("--safetensors-max-file-mib") + 1] == "64"
    assert "--trace-io" in command




@pytest.mark.skipif(sys.version_info < (3, 12), reason="gdtensor CLI requires Python 3.12")
@pytest.mark.parametrize(
    "action,device", [(action, device) for action in ("save", "load") for device in matrix.DEVICES]
)
@pytest.mark.parametrize("data_pattern,validate_values", [("position", True), ("zero", False)])
def test_phase_commands_pass_the_real_cli_parser(
    tmp_path, action, device, data_pattern, validate_values,
):
    from metasl.run import parse_args

    command = matrix.phase_command(
        sys.executable, tmp_path / "bundle", tmp_path / "checkpoint", tmp_path / "output",
        4, action, device, data_pattern=data_pattern, validate_values=validate_values,
    )
    args = parse_args(command[3:])
    assert args.action == action and args.io_backend == "gdtensor"
    assert args.writer_limit == (1 if action == "save" else 0)
    assert args.reader_limit == 0
    assert args.metadata_limit == 4
    assert args.data_pattern == data_pattern
    assert args.validate_values is (action == "load" and validate_values)


@pytest.mark.parametrize("device", matrix.DEVICES)
@pytest.mark.parametrize("cache_control", [False, True])
def test_cache_control_uses_existing_load_eviction_and_requests_measurement(device, cache_control):
    from metasl.run import parse_args

    for action in ("save", "load"):
        command = matrix.phase_command(
            sys.executable, "bundle", "checkpoint", "output", 1, action, device,
            cache_control=cache_control,
        )
        args = parse_args(command[3:])
        expected = action == "load" and cache_control
        assert args.evict_cache is expected
        assert args.measure_cache_residency is expected


def test_residency_measurement_requires_eviction():
    from metasl.run import parse_args

    command = matrix.phase_command(sys.executable, "bundle", "checkpoint", "output", 1, "load")
    with pytest.raises(SystemExit):
        parse_args(command[3:] + ["--measure-cache-residency"])


def test_unique_parameter_count_deduplicates_tied_aliases():
    manifest = dict(
        structure=dict(
            aliases=[["parameter:embed.weight", "parameter:head.weight"]],
            tensors=[
                dict(kind="parameter", name=name, shape=[4, 3])
                for name in ("embed.weight", "head.weight", "projection.weight")
            ]
            + [dict(kind="buffer", name="buffer", shape=[100])],
        )
    )
    facts = matrix.parameter_facts(manifest)
    assert facts["unique_parameters"] == 24
    assert facts["parameter_state_entry_elements"] == 36
    assert facts["unique_parameter_bf16_bytes"] == 48


@pytest.mark.parametrize("device", matrix.DEVICES)
def test_phase_requires_actual_target_and_complete_values_and_coverage(tmp_path, device):
    output, report = phase_fixture(tmp_path, device=device)
    _, reports = matrix.validate_phase(output, action="load", device=device, ranks=1)
    assert reports == [report]
    report["numerical_values_verified"] = False
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="numerical_values_verified"):
        matrix.validate_phase(output, action="load", device=device, ranks=1)


def test_phase_validation_requires_requested_checkpoint_format(tmp_path):
    output, _ = phase_fixture(tmp_path, checkpoint_format="safetensors")
    matrix.validate_phase(
        output, action="load", device="meta", ranks=1, checkpoint_format="safetensors",
    )
    with pytest.raises(RuntimeError, match="unexpected phase configuration"):
        matrix.validate_phase(output, action="load", device="meta", ranks=1)


@pytest.mark.parametrize("device", matrix.DEVICES)
def test_zero_load_without_values_keeps_completeness_and_coverage_checks(tmp_path, device):
    output, report = phase_fixture(
        tmp_path, device=device, data_pattern="zero", validate_values=False,
    )
    options = dict(
        action="load", device=device, ranks=1, data_pattern="zero", validate_values=False,
    )
    _, reports = matrix.validate_phase(output, **options)
    assert reports == [report]
    assert report["numerical_values_verified"] is False
    for field in ("complete_load_verified", "coverage_crosscheck"):
        report[field] = False
        write_json(output / "rank-000.json", report)
        with pytest.raises(RuntimeError, match=field):
            matrix.validate_phase(output, **options)
        report[field] = True
    report["completed_elements"] -= 1
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="coverage"):
        matrix.validate_phase(output, **options)
    report["completed_elements"] += 1
    report["numerical_values_verified"] = True
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="numerical_values_verified"):
        matrix.validate_phase(output, **options)


@pytest.mark.parametrize("action", ["save", "load"])
@pytest.mark.parametrize("field,value", [("data_pattern", "position"), ("validate_values", True)])
def test_phase_rejects_mismatched_timing_configuration(tmp_path, action, field, value):
    output, _ = phase_fixture(
        tmp_path, action=action, data_pattern="zero", validate_values=False,
    )
    summary = matrix.load(output / "summary.json")
    summary["configuration"][field] = value
    write_json(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="configuration"):
        matrix.validate_phase(
            output, action=action, device="meta", ranks=1,
            data_pattern="zero", validate_values=False,
        )


def test_accidental_cpu_fallback_and_missing_rank_fail_closed(tmp_path):
    output, report = phase_fixture(tmp_path, device="cuda")
    report["state_device"] = "cpu"
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="actual state device"):
        matrix.validate_phase(output, action="load", device="cuda", ranks=1)
    report["state_device"] = "cuda:0"
    report["rank"] = 1
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="rank reports"):
        matrix.validate_phase(output, action="load", device="cuda", ranks=1)


@pytest.mark.parametrize("device", matrix.DEVICES)
def test_save_requires_measured_trace_and_complete_write_coverage(tmp_path, device):
    output, report = phase_fixture(tmp_path, action="save", device=device)
    _, reports = matrix.validate_phase(output, action="save", device=device, ranks=1)
    assert reports == [report]
    report["completed_write_indices"] = []
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="write items"):
        matrix.validate_phase(output, action="save", device=device, ranks=1)
    report["completed_write_indices"] = ["item"]
    write_json(output / "rank-000.json", report)
    (output / "rank-000.io.jsonl").unlink()
    with pytest.raises(RuntimeError, match="trace"):
        matrix.validate_phase(output, action="save", device=device, ranks=1)


def test_cuda_save_requires_synchronized_timing(tmp_path):
    output, report = phase_fixture(tmp_path, action="save", device="cuda")
    report["stats"]["cuda_synchronized"] = False
    write_json(output / "rank-000.json", report)
    with pytest.raises(RuntimeError, match="synchronized"):
        matrix.validate_phase(output, action="save", device="cuda", ranks=1)


def checkpoint_fixture(tmp_path, *, checkpoint_format="dcp"):
    case = tmp_path / "case"
    checkpoint = case / "checkpoint"
    checkpoint.mkdir(parents=True)
    metadata, suffix = (
        (".metadata", ".distcp") if checkpoint_format == "dcp"
        else (".metasl-complete.json", ".safetensors")
    )
    (checkpoint / metadata).write_bytes(b"metadata")
    (checkpoint / f"__0_0{suffix}").write_bytes(b"payload0")
    (checkpoint / f"__1_0{suffix}").write_bytes(b"payload1")
    inventory = matrix.checkpoint_inventory(checkpoint)
    write_json(
        case / "checkpoint-owner.json", dict(registration_id="owner", checkpoint=str(checkpoint))
    )
    return case, checkpoint, inventory


def test_safetensors_inventory_cleanup_and_data_hashes(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path, checkpoint_format="safetensors")
    changed = json.loads(json.dumps(inventory))
    changed["entries"][1]["sha256"] = "changed"
    comparison = matrix.compare_checkpoint_inventories(dict(meta=inventory, cuda=changed))
    assert comparison["status"] == comparison["data_byte_identity"] == "different"
    result = matrix.cleanup_checkpoint(case, "owner", inventory)
    assert result["deleted_bytes"] == 16
    assert (checkpoint / ".metasl-complete.json").is_file()
    assert not list(checkpoint.glob("*.safetensors"))
    matrix.verify_checkpoint(inventory, allow_deleted=True)
    assert matrix.cleanup_checkpoint(case, "owner", inventory)["state"] == "complete"


def test_safetensors_inventory_requires_completion_manifest(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path, checkpoint_format="safetensors")
    (checkpoint / ".metasl-complete.json").unlink()
    with pytest.raises(RuntimeError, match="metasl-complete"):
        matrix.checkpoint_inventory(checkpoint)


def test_cleanup_removes_only_owned_payloads_and_keeps_metadata_sha_inventory(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path)
    unrelated = tmp_path / "unrelated.distcp"
    unrelated.write_bytes(b"other")
    result = matrix.cleanup_checkpoint(case, "owner", inventory)
    assert result["state"] == "complete"
    assert result["deleted_bytes"] == 16
    assert (checkpoint / ".metadata").read_bytes() == b"metadata"
    assert unrelated.read_bytes() == b"other"
    assert not list(checkpoint.glob("*.distcp"))
    assert all(len(row["sha256"]) == 64 for row in inventory["entries"])
    assert matrix.cleanup_checkpoint(case, "owner", inventory)["state"] == "complete"


def test_cleanup_rejects_wrong_owner_or_changed_checkpoint(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path)
    with pytest.raises(RuntimeError, match="ownership"):
        matrix.cleanup_checkpoint(case, "different-owner", inventory)
    (checkpoint / "__0_0.distcp").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="changed"):
        matrix.cleanup_checkpoint(case, "owner", inventory)
    assert (checkpoint / "__0_0.distcp").exists()
    assert (checkpoint / "__1_0.distcp").exists()


def test_cleanup_can_finish_a_journaled_interruption(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path)
    write_json(
        case / "checkpoint-deletion.json", dict(checkpoint=str(checkpoint), state="deleting")
    )
    (checkpoint / "__0_0.distcp").unlink()
    result = matrix.cleanup_checkpoint(case, "owner", inventory)
    assert result["state"] == "complete"
    assert not (checkpoint / "__1_0.distcp").exists()


def test_cleanup_device_journals_do_not_authorize_another_checkpoint(tmp_path):
    case, checkpoint, inventory = checkpoint_fixture(tmp_path)
    for device in ("cpu", "cuda"):
        write_json(
            case / f"checkpoint-{device}-owner.json",
            dict(registration_id="owner", checkpoint=str(checkpoint)),
        )
    result = matrix.cleanup_checkpoint(case, "owner", inventory, device="cpu")
    assert result["state"] == "complete"
    assert (case / "checkpoint-cpu-deletion.json").is_file()
    assert not (case / "checkpoint-cuda-deletion.json").exists()
    with pytest.raises(RuntimeError, match="missing"):
        matrix.cleanup_checkpoint(case, "owner", inventory, device="cuda")


def inventory_fixture():
    return {
        device: dict(
            entries=[
                dict(path=".metadata", bytes=4, sha256="metadata-" + device),
                dict(path="__0_0.distcp", bytes=8, sha256="same-payload"),
            ]
        )
        for device in matrix.DEVICES
    }


def test_checkpoint_comparison_preserves_metadata_hash_differences():
    result = matrix.compare_checkpoint_inventories(inventory_fixture())
    assert result["status"] == "identical"
    assert result["names_and_sizes_status"] == "identical"
    assert result["data_byte_identity"] == "identical"
    assert result["metadata_byte_identity"] == "different"
    assert len(result["comparisons"]) == 3


@pytest.mark.parametrize("field,value", [("path", "__1_0.distcp"), ("bytes", 9), ("sha256", "different")])
def test_checkpoint_comparison_detects_layout_and_data_byte_changes(field, value):
    inventories = inventory_fixture()
    inventories["cuda"]["entries"][1][field] = value
    result = matrix.compare_checkpoint_inventories(inventories)
    assert result["status"] == "different"
    assert any(row["first_difference"] is not None for row in result["comparisons"])
    if field == "sha256":
        assert result["names_and_sizes_status"] == "identical"
        assert result["data_byte_identity"] == "different"
    else:
        assert result["names_and_sizes_status"] == "different"


def test_capacity_guard_accounts_for_all_three_save_payloads(tmp_path, monkeypatch):
    facts = dict(estimated_checkpoint_tensor_bytes=matrix.GIB)
    one_payload = math.ceil(matrix.GIB * 1.1)
    free = one_payload * 3 + matrix.GIB - 1
    monkeypatch.setattr(matrix.shutil, "disk_usage", lambda path: SimpleNamespace(free=free))
    matrix.guard_checkpoint_capacity(tmp_path, facts, remaining_checkpoints=1)
    with pytest.raises(RuntimeError, match="[Ii]nsufficient disk"):
        matrix.guard_checkpoint_capacity(tmp_path, facts)
    free += 1
    matrix.guard_checkpoint_capacity(tmp_path, facts)


@pytest.mark.parametrize("suffix", [".distcp", ".safetensors"])
def test_save_metrics_count_positioned_payload_calls_and_keep_memory_evidence(tmp_path, suffix):
    trace = tmp_path / "export.jsonl"
    trace.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                dict(path="__0_0" + suffix, op="write", completed=4),
                dict(path="__0_0" + suffix, op="pwrite", completed=8),
                dict(path=".metadata.tmp", op="write", completed=3),
                dict(path="__0_0" + suffix, op="seek", completed=99),
            )
        )
    )
    report = dict(
        state_device="cuda:0",
        stats=dict(
            seconds=2.5,
            process_high_water_rss_bytes=40,
            cuda_memory=dict(peak_allocated_bytes=50, peak_reserved_bytes=60),
        ),
        io=dict(data_write_calls=1, data_write_bytes=4),
        gdtensor_payload_write_bytes=8,
        payload_requests=[dict(length=8)],
    )
    metrics = matrix.phase_metrics(
        dict(peak_sampled_total_pss_bytes=20, peak_sampled_worker_rss_bytes=30),
        [report],
        [dict(output_path=str(trace))],
        action="save",
        ranks=1,
        output=tmp_path,
    )
    assert metrics["write_calls"] == 3
    assert metrics["write_bytes"] == 15
    assert metrics["data_write_calls"] == 2
    assert metrics["data_write_bytes"] == 12
    assert metrics["metadata_write_bytes"] == 3
    assert metrics["payload_pwrite_calls"] == 1
    assert metrics["max_rank_dcp_seconds"] == 2.5
    assert metrics["peak_sampled_total_pss_bytes"] == 20
    assert metrics["max_rank_process_high_water_rss_bytes"] == 40
    assert metrics["max_rank_cuda_peak_allocated_bytes"] == 50
    assert metrics["max_rank_cuda_peak_reserved_bytes"] == 60


@pytest.mark.parametrize("verified", [True, False, None])
def test_load_metrics_report_actual_validation_evidence_from_every_rank(tmp_path, verified):
    reports = []
    for rank in range(2):
        _, report = phase_fixture(tmp_path, output=tmp_path / f"rank-{rank}")
        report["rank"] = rank
        report["stats"]["seconds"] = 1.0
        report["gdtensor_payload_read_bytes"] = 12
        report["payload_requests"] = [dict(length=12)]
        reports.append(report)
    if verified is None:
        reports[1].pop("numerical_values_verified")
    else:
        reports[1]["numerical_values_verified"] = verified
    metrics = matrix.phase_metrics(
        {}, reports, [], action="load", ranks=2, output=tmp_path,
    )
    assert metrics["numerical_values_verified"] is (verified is True)
    assert metrics["complete_load_verified"] is True
    assert metrics["coverage_crosscheck"] is True


def registration_fixture(tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    write_json(configs / "tiny/config.json", dict(model_type="qwen2"))
    source = tmp_path / "runtime.py"
    source.write_text("source = 1\n")
    environment = dict(
        gdtensor_package=str(tmp_path),
        actual_gpu_mode="shared device 0",
        cuda_devices=[dict(index=0)],
    )
    monkeypatch.setattr(matrix, "environment_evidence", lambda python: environment)
    monkeypatch.setattr(
        matrix,
        "source_inventory",
        lambda package: {"runtime.py": dict(source=str(source), sha256=matrix.sha256(source))},
    )
    args = SimpleNamespace(
        output=tmp_path / "out",
        python=Path(sys.executable),
        configs=configs,
        models=["tiny"],
        ranks=[1],
        keep_checkpoints=False,
        data_pattern="position",
        validate_values=True,
        resume=False,
    )
    return args, source


def test_resume_checks_source_config_and_archived_copies(tmp_path, monkeypatch):
    args, source = registration_fixture(tmp_path, monkeypatch)
    matrix.Matrix(args)
    args.resume = True
    matrix.Matrix(args)
    source.write_text("source = 2\n")
    with pytest.raises(RuntimeError, match="sources differs"):
        matrix.Matrix(args)
    source.write_text("source = 1\n")
    (args.output / "input-configs/tiny/config.json").write_text("{}")
    with pytest.raises(RuntimeError, match="Archived model configuration"):
        matrix.Matrix(args)


@pytest.mark.parametrize("data_pattern,validate_values", [("position", True), ("zero", False)])
def test_registration_records_data_pattern_and_validation_options(
    tmp_path, monkeypatch, data_pattern, validate_values,
):
    args, _ = registration_fixture(tmp_path, monkeypatch)
    args.data_pattern = data_pattern
    args.validate_values = validate_values
    experiment = matrix.Matrix(args)
    assert experiment.registration["options"]["data_pattern"] == data_pattern
    assert experiment.registration["options"]["validate_values"] is validate_values
    assert data_pattern in experiment.registration["fixture"]


@pytest.mark.parametrize(
    "option,value", [
        ("data_pattern", "zero"), ("validate_values", False), ("cache_control", True),
        ("checkpoint_format", "safetensors"), ("safetensors_max_file_mib", 64),
        ("writer_limit", 0),
    ]
)
def test_resume_rejects_changed_data_pattern_or_validation(tmp_path, monkeypatch, option, value):
    args, _ = registration_fixture(tmp_path, monkeypatch)
    matrix.Matrix(args)
    args.resume = True
    setattr(args, option, value)
    with pytest.raises(RuntimeError, match="Resume options differs"):
        matrix.Matrix(args)


def test_virtualenv_interpreter_symlink_is_preserved(tmp_path, monkeypatch):
    args, _ = registration_fixture(tmp_path, monkeypatch)
    executable = tmp_path / "venv/bin/python"
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    args.python = executable
    experiment = matrix.Matrix(args)
    assert str(experiment.python) == str(executable)
    assert experiment.python != executable.resolve()


def test_failed_stage_cannot_retry_after_source_change(tmp_path, monkeypatch):
    args, source = registration_fixture(tmp_path, monkeypatch)
    experiment = matrix.Matrix(args)
    experiment.progress["stages"]["phase"] = dict(status="failed", attempt=1)
    source.write_text("source = 2\n")
    with pytest.raises(RuntimeError, match="Runtime source changed"):
        experiment.stage("phase", tmp_path / "retry", lambda output: pytest.fail("restarted"))
    assert experiment.progress["stages"]["phase"]["attempt"] == 1
    assert not (tmp_path / "retry-attempt02").exists()


def test_completed_stage_is_verified_instead_of_blindly_skipped(tmp_path):
    evidence = tmp_path / "complete.json"
    evidence.write_text("proof")
    experiment = matrix.Matrix.__new__(matrix.Matrix)
    experiment.progress = dict(
        stages={
            "phase": dict(
                status="complete", output=str(tmp_path), artifacts=matrix.artifacts([evidence])
            )
        }
    )
    experiment.verify_sources = lambda: None
    experiment.update = lambda message: None
    assert experiment.stage("phase", tmp_path, lambda output: pytest.fail("restarted")) == tmp_path
    evidence.write_text("changed")
    with pytest.raises(RuntimeError, match="Completed evidence changed"):
        experiment.stage("phase", tmp_path, lambda output: pytest.fail("restarted"))






@pytest.mark.parametrize(
    "different_payload,interrupt_cleanup", [(False, False), (True, False), (False, True)]
)
@pytest.mark.parametrize("data_pattern,validate_values", [("position", True), ("zero", False)])
@pytest.mark.parametrize("cache_control", [False, True])
@pytest.mark.parametrize("checkpoint_format", ["dcp", "safetensors"])
def test_case_uses_independent_saves_common_load_and_compares_before_cleanup(
    tmp_path, monkeypatch, different_payload, interrupt_cleanup, data_pattern, validate_values,
    cache_control, checkpoint_format,
):
    experiment = matrix.Matrix.__new__(matrix.Matrix)
    experiment.output = tmp_path / "run"
    experiment.python = Path(sys.executable)
    experiment.args = SimpleNamespace(
        keep_checkpoints=False, data_pattern=data_pattern, validate_values=validate_values,
        cache_control=cache_control,
        checkpoint_format=checkpoint_format,
        safetensors_max_file_mib=64 if checkpoint_format == "safetensors" else None,
    )
    experiment.registration = dict(registration_id="owner")
    experiment.environment = dict(cuda_devices=[dict(index=0)])
    experiment.progress = dict(stages={}, cases={"tiny-r01": dict(status="pending")})
    experiment.update = lambda message: None
    calls = []
    cache_events = []
    suffix = ".distcp" if checkpoint_format == "dcp" else ".safetensors"
    marker = ".metadata" if checkpoint_format == "dcp" else ".metasl-complete.json"

    def residency(checkpoint):
        cache_events.append(("residency", checkpoint.name))
        return dict(empty=not checkpoint.exists(), files=[], near_zero_data_residency=True)

    def evict(checkpoint):
        cache_events.append(("evict", checkpoint.name))
        return dict(method="POSIX_FADV_DONTNEED")

    original_inventory = matrix.checkpoint_inventory

    def inventory(checkpoint):
        cache_events.append(("inventory", checkpoint.name))
        return original_inventory(checkpoint)

    monkeypatch.setattr(matrix, "checkpoint_residency", residency)
    monkeypatch.setattr(matrix, "advise_evict_checkpoint", evict)
    monkeypatch.setattr(matrix, "checkpoint_inventory", inventory)

    def fake_stage(key, output, build_command):
        command = build_command(output)
        action = command[command.index("--action") + 1]
        device = key.rsplit("-", 1)[1]
        checkpoint = Path(command[command.index("--checkpoint") + 1])
        assert command[command.index("--data-pattern") + 1] == data_pattern
        assert command[command.index("--checkpoint-format") + 1] == checkpoint_format
        assert ("--safetensors-max-file-mib" in command) is (checkpoint_format == "safetensors")
        assert ("--validate-values" in command) is (action == "load" and validate_values)
        assert ("--evict-cache" in command) is (action == "load" and cache_control)
        assert ("--measure-cache-residency" in command) is (action == "load" and cache_control)
        calls.append((action, device, checkpoint))
        cache_events.append((action, checkpoint.name))
        if action == "save":
            checkpoint.mkdir(parents=True)
            (checkpoint / marker).write_bytes(b"metadata")
            (checkpoint / ("__0_0" + suffix)).write_bytes(
                b"different" if different_payload and device == "cuda" else b"payload!!"
            )
        output, report = phase_fixture(
            tmp_path, action=action, device=device, output=output,
            data_pattern=data_pattern, validate_values=validate_values,
            checkpoint_format=checkpoint_format,
        )
        if action == "load" and cache_control:
            write_json(output / "cache.json", dict(residency_after_eviction=residency(checkpoint)))
        report["stats"]["seconds"] = 1.0
        report["payload_requests"] = [dict(length=4 if action == "save" else 8)]
        report[f"gdtensor_payload_{'write' if action == 'save' else 'read'}_bytes"] = (
            4 if action == "save" else 8
        )
        event = dict(
            file=str(checkpoint / ("__0_0" + suffix)),
            op="write" if action == "save" else "read",
            offset=0,
            requested=4 if action == "save" else 8,
            completed=4 if action == "save" else 8,
            start_ns=1,
            duration_ns=1,
        )
        if action == "save":
            report["payload_events"] = [
                dict(event, op="pwrite", offset=4, start_ns=110)
            ]
        write_json(output / "rank-000.json", report)
        (output / "rank-000.io.jsonl").write_text(json.dumps(event) + "\n")
        experiment.progress["stages"][key] = dict(
            status="complete", output=str(output), artifacts=matrix.artifacts(output.iterdir())
        )
        return output

    experiment.stage = fake_stage
    original_cleanup = matrix.cleanup_checkpoint
    deleted = []

    def checked_cleanup(case_dir, registration_id, inventory, *, device=None):
        for filename in ("compare.json", "compare-save.json", "compare-load.json", "compare-checkpoints.json"):
            assert (case_dir / filename).is_file()
        case = experiment.progress["cases"]["tiny-r01"]
        assert case["comparison_artifacts"]
        matrix.verify_artifacts(case["comparison_artifacts"])
        deleted.append(device)
        result = original_cleanup(case_dir, registration_id, inventory, device=device)
        if interrupt_cleanup and len(deleted) == 1:
            raise RuntimeError("interrupted cleanup")
        return result

    monkeypatch.setattr(matrix, "cleanup_checkpoint", checked_cleanup)
    facts = dict(estimated_checkpoint_tensor_bytes=9)
    if interrupt_cleanup:
        with pytest.raises(RuntimeError, match="interrupted cleanup"):
            experiment.run_case("tiny", 1, tmp_path / "bundle", facts)
        experiment.stage = lambda *args: pytest.fail("cleanup recovery restarted a stage")
    experiment.run_case("tiny", 1, tmp_path / "bundle", facts)
    saves = [checkpoint for action, device, checkpoint in calls if action == "save"]
    loads = [checkpoint for action, device, checkpoint in calls if action == "load"]
    assert len(saves) == 3 and len(set(saves)) == 3
    assert loads == [saves[0]] * 3
    if cache_control:
        expected_cache_events = []
        for checkpoint in saves:
            expected_cache_events += [
                (event, checkpoint.name)
                for event in ("residency", "save", "residency", "inventory", "evict", "residency")
            ]
        assert cache_events[:18] == expected_cache_events
        for device in matrix.DEVICES:
            evidence = matrix.load(
                experiment.output / "cases/tiny-r01/cache-control" / f"save-{device}.json"
            )
            assert evidence["before_save"]["empty"] is True
            assert evidence["after_save"]["empty"] is False
            assert evidence["after_eviction"]["near_zero_data_residency"] is True
    case = experiment.progress["cases"]["tiny-r01"]
    comparison = matrix.load(case["compare"])
    assert comparison["save"]["status"] == comparison["load"]["status"] == "identical"
    assert all(
        metrics["numerical_values_verified"] is validate_values
        and metrics["complete_load_verified"] is True
        and metrics["coverage_crosscheck"] is True
        for metrics in case["metrics"]["load"].values()
    )
    assert case["status"] == ("different" if different_payload else "identical")
    assert deleted == (
        [] if different_payload else ["meta", *matrix.DEVICES] if interrupt_cleanup else list(matrix.DEVICES)
    )
    for checkpoint in saves:
        assert (checkpoint / marker).is_file()
        assert (checkpoint / ("__0_0" + suffix)).exists() is different_payload
    experiment.stage = lambda *args: pytest.fail("completed case restarted")
    experiment.run_case("tiny", 1, tmp_path / "bundle", facts)


def test_different_case_does_not_prevent_remaining_cases(tmp_path):
    experiment = matrix.Matrix.__new__(matrix.Matrix)
    experiment.args = SimpleNamespace(models=["tiny"], ranks=[1, 4])
    experiment.progress = dict(
        cases={"tiny-r01": dict(status="pending"), "tiny-r04": dict(status="pending")}
    )
    experiment.prepare = lambda model: (tmp_path, {})
    calls = []

    def run_case(model, ranks, bundle, facts):
        calls.append(ranks)
        experiment.progress["cases"][f"tiny-r{ranks:02d}"]["status"] = (
            "different" if ranks == 1 else "identical"
        )

    experiment.run_case = run_case
    experiment.update = lambda message: None
    assert experiment.run() is False
    assert calls == [1, 4]
    assert experiment.progress["status"] == "complete_with_differences"
