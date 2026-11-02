"""Measurement runs must never reuse reports or remove foreign checkpoints."""

import json
from types import SimpleNamespace

import pytest

from scripts import run_gdt_safetensors as runner


def arguments(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    return SimpleNamespace(
        output=tmp_path / "validation" / "new-run",
        checkpoint_root=tmp_path / "external-checkpoints",
        ranks=4,
        checkpoint_format="dcp",
        safetensors_max_file_mib=None,
        models=["1b"],
        repeats=1,
        configs=["meta"],
    )


def test_occupied_external_checkpoint_root_is_rejected_without_deletion(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    foreign = args.checkpoint_root / "measure-1b-r1-meta----"
    foreign.mkdir(parents=True)
    payload = foreign / "foreign.distcp"
    payload.write_bytes(b"another run's checkpoint")

    with pytest.raises(ValueError, match="new or empty"):
        runner.Runner(args)

    assert payload.read_bytes() == b"another run's checkpoint"
    assert not args.output.exists()
    assert not (args.checkpoint_root / ".metasl-owner.json").exists()


def test_reused_output_is_rejected_before_stale_rank_reports_can_be_skipped(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    args.output.mkdir(parents=True)
    completed = args.output / "completed.json"
    completed.write_text(json.dumps({"ranks": 1, "rows": ["old single-rank result"]}))

    with pytest.raises(ValueError, match="do not resume"):
        runner.Runner(args)

    assert json.loads(completed.read_text())["ranks"] == 1
    assert not args.checkpoint_root.exists()
    assert not (args.output / "progress.json").exists()


def test_empty_external_root_is_claimed_and_cleaned_after_success(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    args.checkpoint_root.mkdir()
    measurement = runner.Runner(args)
    owner = json.loads(measurement.owner_path.read_text())
    assert owner["output"] == str(args.output)
    monkeypatch.setattr(measurement, "measurements", lambda: None)

    measurement.run()

    assert not args.checkpoint_root.exists()
    assert json.loads((args.output / "cleanup.json").read_text())["checkpoints_removed"]
    assert json.loads((args.output / "checkpoint-ownership.json").read_text()) == owner


def test_checkpoint_name_collision_is_preserved_even_after_root_claim(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    measurement = runner.Runner(args)
    foreign = args.checkpoint_root / "measure-1b-r1-meta----"
    foreign.mkdir()
    payload = foreign / "foreign.distcp"
    payload.write_bytes(b"foreign")

    with pytest.raises(FileExistsError):
        measurement.run()

    assert payload.read_bytes() == b"foreign"
    assert not json.loads((args.output / "cleanup.json").read_text())["checkpoints_removed"]


def test_measurement_cli_defaults_and_save_only_writer_options():
    defaults = runner.parse_args([])
    assert defaults.writer_limit == 1 and not defaults.save_only
    assert defaults.meta_load_sink is None and defaults.meta_load_sink_mib == 64
    args = runner.parse_args(["--save-only", "--writer-limit", "0"])
    assert args.save_only and args.writer_limit == 0
    with pytest.raises(SystemExit):
        runner.parse_args(["--writer-limit", "-1"])


@pytest.mark.parametrize("sink", ["fresh", "faithful", "ring"])
def test_measurement_cli_keeps_explicit_sink_and_capacity(sink):
    args = runner.parse_args(["--meta-load-sink", sink, "--meta-load-sink-mib", "256"])
    assert args.meta_load_sink == sink and args.meta_load_sink_mib == 256
    with pytest.raises(SystemExit):
        runner.parse_args(["--meta-load-sink-mib", "0"])


@pytest.mark.parametrize("failure_at", [None, 3])
@pytest.mark.parametrize("checkpoint_format", ["dcp", "safetensors"])
def test_save_only_alternates_five_pairs_and_cleans_checkpoints(
        tmp_path, monkeypatch, failure_at, checkpoint_format):
    args = arguments(tmp_path, monkeypatch)
    args.save_only = True
    args.writer_limit = 4
    args.requested_writer_limit = 0
    args.repeats = 5
    args.configs = ["meta", "cpu"]
    args.checkpoint_format = checkpoint_format
    measurement = runner.Runner(args)
    calls = []

    def phase(case, checkpoint, model, config, action):
        assert action == "save" and checkpoint.is_dir()
        case.mkdir(parents=True, exist_ok=True)
        calls.append((case.name, config, checkpoint))
        metadata = ".metadata" if checkpoint_format == "dcp" else ".metasl-complete.json"
        (checkpoint / metadata).write_text("{}")
        (checkpoint / "payload").write_bytes(b"payload")
        if len(calls) == failure_at:
            raise RuntimeError("injected save failure")
        return [dict(config=config, phase=action, rank=rank) for rank in range(args.ranks)]

    monkeypatch.setattr(measurement, "phase", phase)
    if failure_at:
        with pytest.raises(RuntimeError, match="injected save"):
            measurement.run()
    else:
        measurement.run()
    expected = [(f"r{repeat}-{config}", config)
                for repeat in range(1, 6)
                for config in (["meta", "cpu"] if repeat % 2 else ["cpu", "meta"])]
    assert [(case, config) for case, config, _ in calls] == expected[:failure_at]
    assert all(not checkpoint.exists() for _, _, checkpoint in calls)
    assert not args.checkpoint_root.exists()
    progress = json.loads((args.output / "progress.json").read_text())
    assert progress["writer_limit"] == 4 and progress["requested_writer_limit"] == 0
    assert progress["save_only"] is True
    assert progress["status"] == ("failed" if failure_at else "stage-complete")
    accepted = json.loads((args.output / "measurement-results.json").read_text())
    assert len(accepted) == ((failure_at - 1) if failure_at else 10) * args.ranks
    assert all(row["phase"] == "save" for row in accepted)
    assert json.loads((args.output / "cleanup.json").read_text())["checkpoints_removed"]


def test_negative_writer_limit_is_rejected_before_claiming_output(tmp_path, monkeypatch):
    args = arguments(tmp_path, monkeypatch)
    args.writer_limit = -1
    with pytest.raises(ValueError, match="nonnegative"):
        runner.Runner(args)
    assert not args.output.exists() and not args.checkpoint_root.exists()


def warm_report(rank=0, *, residency=100.0, block_reads=0):
    cache = {} if residency is None else dict(residency=dict(
        data_resident_percent=residency,
        fully_warm_data_residency=residency >= 99,
        warm_threshold_percent=99,
    ))
    return dict(rank=rank, cache_before_call=cache, io=dict(origin_ns=100),
                stats=dict(dcp_wall_seconds=1.0,
                           proc_io_delta=dict(read_bytes=block_reads)))


@pytest.mark.parametrize("residency,block_reads,valid,reasons", [
    (98.999, 0, False, ["checkpoint residency below 99% or missing"]),
    (99.0, 0, True, []),
    (100.0, 0, True, []),
    (100.0, 1, False, ["timed block-read bytes > 0"]),
    (98.0, 4096, False, ["checkpoint residency below 99% or missing",
                        "timed block-read bytes > 0"]),
    (None, 0, False, ["checkpoint residency below 99% or missing"]),
])
def test_warm_validity_requires_99_percent_residency_and_zero_block_reads(
        residency, block_reads, valid, reasons):
    result = runner.warm_validity(warm_report(residency=residency, block_reads=block_reads))
    assert result == dict(valid=valid, reasons=reasons,
                          residency_percent=residency, block_read_bytes=block_reads)


@pytest.mark.parametrize("block_reads,valid", [(4096, True), (65536, True), (65537, False)])
def test_shared_round_warm_tolerance_records_stray_reads(block_reads, valid):
    result = runner.warm_validity(warm_report(block_reads=block_reads), 65536)
    assert result["valid"] is valid
    assert result["block_read_bytes"] == block_reads


def test_shared_layout_requires_one_file_without_a_size_limit(tmp_path):
    (tmp_path / ".metasl-complete.json").write_text("{}")
    (tmp_path / "model.safetensors").touch()
    assert runner.verify_layout(tmp_path, None, "safetensors", 8, "shared")["file_count"] == 1
    with pytest.raises(RuntimeError, match="Expected 8"):
        runner.verify_layout(tmp_path, None, "safetensors", 8)


def warm_phase_fixture(tmp_path, monkeypatch, reports_by_attempt):
    """Retain real phase artifacts, replacing only execution and trace export."""
    measurement = runner.Runner.__new__(runner.Runner)
    measurement.args = arguments(tmp_path, monkeypatch)
    measurement.args.data_pattern = "zero"
    measurement.args.validate_values = False
    measurement.args.ranks = 2
    measurement.progress = {}
    measurement.update = lambda **fields: measurement.progress.update(fields)
    measurement.bundle = lambda model: tmp_path / "bundle"
    calls = []

    monkeypatch.setattr(runner, "phase_command", lambda *args, **kwargs:
                        ["fake-worker", "--output", str(args[3])])
    monkeypatch.setattr(runner, "runtime_environment", lambda: {})
    monkeypatch.setattr(runner, "verify_layout", lambda *args: dict(file_count=2))
    monkeypatch.setattr(runner, "export_rank_trace", lambda trace, output, **kwargs:
                        dict(output_path=str(output / f"rank-{kwargs['rank']:03d}.jsonl")))

    def launch(command, **kwargs):
        output = tmp_path / command[command.index("--output") + 1]
        assert "--warm-cache" in command and "--measure-cache-residency" in command
        output.mkdir(parents=True)
        (output / "worker-evidence.json").write_text(json.dumps(dict(attempt=len(calls) + 1)))
        calls.append(output)
        return SimpleNamespace(returncode=0)

    def reports(output, **kwargs):
        return {}, reports_by_attempt[len(calls) - 1]

    monkeypatch.setattr(runner.subprocess, "run", launch)
    monkeypatch.setattr(runner, "validate_phase", reports)
    return measurement, calls


@pytest.mark.parametrize("config,pattern,selection,expected", [
    ("meta", "zero", "missing", "ring"),
    ("meta", "zero", None, "ring"),
    ("meta", "position", None, "fresh"),
    ("meta-sim", "zero", None, "fresh"),
    ("meta-sim", "position", None, "fresh"),
    ("meta", "zero", "fresh", "fresh"),
    ("meta", "zero", "faithful", "faithful"),
    ("meta", "zero", "ring", "ring"),
    ("meta", "position", "ring", "ring"),
    ("meta-sim", "zero", "faithful", "faithful"),
    ("cpu", "zero", None, None),
    ("gpu", "zero", None, None),
])
def test_phase_selects_bounded_cpu_default_and_preserves_other_modes(
        tmp_path, monkeypatch, config, pattern, selection, expected):
    measurement, _ = warm_phase_fixture(tmp_path, monkeypatch, [[warm_report(0), warm_report(1)]])
    measurement.args.data_pattern = pattern
    if selection != "missing":
        measurement.args.meta_load_sink = selection
    # Ambient settings must not override automatic selection or introduce a
    # CPU-target resident buffer into the GPU transfer simulation.
    monkeypatch.setattr(runner, "runtime_environment", lambda: {
        "METASL_GDTENSOR_META_LOAD_SINK": "faithful",
        "METASL_GDTENSOR_META_LOAD_SINK_BYTES": str(1024**3),
        "METASL_GDTENSOR_META_SIM_D2H": "pageable",
    })
    case = tmp_path / "case"

    measurement.phase(case, tmp_path / "checkpoint", "1b", config, "warm-load")

    environment = json.loads((case / "warm-load-environment.json").read_text())
    if expected is None:
        assert "METASL_GDTENSOR_META_LOAD_SINK" not in environment
        assert "METASL_GDTENSOR_META_LOAD_SINK_BYTES" not in environment
    else:
        assert environment["METASL_GDTENSOR_META_LOAD_SINK"] == expected
        assert environment["METASL_GDTENSOR_META_LOAD_SINK_BYTES"] == str(64 * 1024**2)
    if config == "meta-sim":
        assert environment["METASL_GDTENSOR_META_SIM_D2H"] == "pageable"
        assert environment["METASL_GDTENSOR_META_SIM_D2H_POPULATE"] == "non-temporal"
    else:
        assert "METASL_GDTENSOR_META_SIM_D2H" not in environment


def test_explicit_faithful_reference_arguments_keep_the_full_shard_mode(tmp_path, monkeypatch):
    from scripts.run_multirank_meta_cpu import arguments as reference_arguments

    measurement, _ = warm_phase_fixture(tmp_path, monkeypatch, [[warm_report(0), warm_report(1)]])
    measurement.args = reference_arguments(tmp_path / "reference", "3b", 8, "dcp", "faithful")
    case = tmp_path / "case"

    measurement.phase(case, tmp_path / "checkpoint", "3b", "meta", "warm-load")

    environment = json.loads((case / "warm-load-environment.json").read_text())
    assert environment["METASL_GDTENSOR_META_LOAD_SINK"] == "faithful"
    assert environment["METASL_GDTENSOR_META_LOAD_SINK_BYTES"] == str(256 * 1024**2)


def test_invalid_warm_redoes_once_and_retains_both_attempts(tmp_path, monkeypatch):
    measurement, calls = warm_phase_fixture(tmp_path, monkeypatch, [
        [warm_report(0), warm_report(1, residency=98.0)],
        [warm_report(0, residency=99.0), warm_report(1)],
    ])
    case = tmp_path / "case"
    rows = measurement.phase(case, tmp_path / "checkpoint", "1b", "meta", "warm-load")

    assert [path.name for path in calls] == ["warm-load", "warm-load-redo1"]
    assert all(row["warm_attempt"] == 2 and row["warm_validity"]["valid"] for row in rows)
    assert all(row["output"] == str(case / "warm-load-redo1") for row in rows)
    assert json.loads((case / "warm-load/worker-evidence.json").read_text()) == dict(attempt=1)
    assert json.loads((case / "warm-load-redo1/worker-evidence.json").read_text()) == dict(attempt=2)
    first = json.loads((case / "warm-load-result.json").read_text())
    assert first[1]["warm_validity"]["valid"] is False
    assert first[1]["output"] == str(case / "warm-load")
    record = json.loads((case / "warm-load-attempts.json").read_text())
    assert record["accepted_attempt"] == 2
    assert [(attempt["attempt"], attempt["valid"]) for attempt in record["attempts"]] == [
        (1, False), (2, True),
    ]
    invalid = measurement.progress["invalid_warm_attempts"]
    assert len(invalid) == 1 and invalid[0]["attempt"] == 1
    assert invalid[0]["ranks"][1]["reasons"] == ["checkpoint residency below 99% or missing"]


def test_valid_warm_is_accepted_without_redo(tmp_path, monkeypatch):
    measurement, calls = warm_phase_fixture(tmp_path, monkeypatch, [
        [warm_report(0), warm_report(1, residency=99.0)],
    ])
    case = tmp_path / "case"
    rows = measurement.phase(case, tmp_path / "checkpoint", "1b", "cpu", "warm-load")

    assert len(calls) == 1 and all(row["warm_attempt"] == 1 for row in rows)
    assert "invalid_warm_attempts" not in measurement.progress
    assert not (case / "warm-load-redo1").exists()
    record = json.loads((case / "warm-load-attempts.json").read_text())
    assert record["accepted_attempt"] == 1 and len(record["attempts"]) == 1


@pytest.mark.parametrize("writer_limit", [0, 8])
def test_measurement_phase_forwards_the_selected_writer_limit(tmp_path, monkeypatch, writer_limit):
    measurement, _ = warm_phase_fixture(tmp_path, monkeypatch, [[warm_report(0), warm_report(1)]])
    measurement.args.writer_limit = writer_limit
    build = runner.phase_command
    selected = []

    def command(*args, **kwargs):
        selected.append(kwargs["writer_limit"])
        return build(*args, **kwargs)

    monkeypatch.setattr(runner, "phase_command", command)
    measurement.phase(tmp_path / "case", tmp_path / "checkpoint", "1b", "cpu", "warm-load")
    assert selected == [writer_limit]


def test_second_invalid_warm_fails_without_a_third_attempt(tmp_path, monkeypatch):
    measurement, calls = warm_phase_fixture(tmp_path, monkeypatch, [
        [warm_report(0, residency=98.0), warm_report(1)],
        [warm_report(0), warm_report(1, block_reads=4096)],
    ])
    case = tmp_path / "case"
    with pytest.raises(RuntimeError, match="after the one permitted redo"):
        measurement.phase(case, tmp_path / "checkpoint", "3b", "meta", "warm-load")

    assert [path.name for path in calls] == ["warm-load", "warm-load-redo1"]
    record = json.loads((case / "warm-load-attempts.json").read_text())
    assert record["accepted_attempt"] is None
    assert len(record["attempts"]) == 2
    assert all(attempt["valid"] is False for attempt in record["attempts"])
    assert [attempt["attempt"] for attempt in measurement.progress["invalid_warm_attempts"]] == [1, 2]
    assert (case / "warm-load-result.json").exists()
    assert (case / "warm-load-redo1-result.json").exists()
