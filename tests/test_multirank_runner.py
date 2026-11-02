"""Focused orchestration tests; no processes, checkpoints or measurements run."""

import json
import sys
from types import SimpleNamespace

import pytest

from scripts import run_multirank_meta_cpu as runner


def test_paired_summary_preserves_repeat_pairs_instead_of_subtracting_medians():
    result = runner.paired_summary([11., 8., 30.], [10., 10., 20.])
    assert result["paired_delta_s"] == 1.
    assert result["paired_delta_percent"] == 10.
    assert result["median_s"] == 11.
    with pytest.raises(ValueError):
        runner.paired_summary([1., 2.], [1.])


def test_request_gate_checks_each_rank_and_repeat_and_rejects_one_mismatch(monkeypatch):
    rows = [dict(config=config, repeat=repeat, rank=rank,
                 export=dict(output_path=(config, repeat, rank)))
            for config in ("meta-ring", "cpu")
            for repeat in range(1, 4) for rank in range(4)]
    compared = []
    monkeypatch.setattr(runner, "prepare_rank_trace", lambda path: path)

    def compare(meta, cpu):
        assert meta[1:] == cpu[1:]
        compared.append(meta[1:])
        # Real compare_prepared_rank_traces includes rank. Keep that field here
        # to catch duplicate-key construction in warm_request_identity.
        return dict(rank=meta[2],
                    status="different" if meta[1:] == (3, 3) else "identical")

    monkeypatch.setattr(runner, "compare_prepared_rank_traces", compare)
    result = runner.warm_request_identity(list(reversed(rows)), "meta-ring")
    assert not result["requests_identical"]
    assert [row["rank"] for row in result["request_comparisons"]] == list(range(4)) * 3
    assert compared == [(repeat, rank) for repeat in range(1, 4) for rank in range(4)]
    with pytest.raises(StopIteration):
        runner.warm_request_identity(rows[:-1], "meta-ring")


def pilot_fixture(tmp_path, monkeypatch, fail_at=None):
    pilot = runner.PilotRunner.__new__(runner.PilotRunner)
    pilot.args = runner.arguments(tmp_path / "pilot", "1b", 4, "dcp", "faithful")
    pilot.root = pilot.args.output
    pilot.ckroot = tmp_path / "checkpoints"
    pilot.ckroot.mkdir()
    pilot.progress = dict(completed=[])
    pilot.verify_ownership = lambda: None
    pilot.update = lambda: None
    calls = []
    cached = []

    def warm(checkpoint):
        assert checkpoint.is_dir()
        cached.append(True)
        return dict(target_met=True)

    def phase(case, checkpoint, model, config, action):
        assert checkpoint.is_dir()
        if action == "save":
            calls.append(("setup", "cpu"))
            return []
        assert cached.pop(), "Warm cache preparation must precede every timed load"
        label = "cpu" if config == "cpu" else "meta-" + pilot.args.meta_load_sink
        calls.append((case.name, label))
        if len(calls) == fail_at:
            raise RuntimeError("injected phase failure")
        return [dict(rank=rank, dcp_seconds=1. + rank * .01)
                for rank in range(pilot.args.ranks)]

    pilot.phase = phase
    monkeypatch.setattr(runner, "warm_load_checkpoint", warm)
    monkeypatch.setattr(runner, "warm_request_identity",
                        lambda rows, config: dict(requests_identical=True))
    return pilot, calls


def test_pilot_rotates_all_three_targets_and_uses_max_rank_per_repeat(tmp_path, monkeypatch):
    pilot, calls = pilot_fixture(tmp_path, monkeypatch)
    pilot.measurements()
    assert [label for _, label in calls] == [
        "cpu", "meta-faithful", "meta-ring", "cpu",
        "meta-ring", "cpu", "meta-faithful",
        "cpu", "meta-faithful", "meta-ring",
    ]
    result = json.loads((pilot.root / "pilot-result.json").read_text())
    assert result["cpu_samples_s"] == [1.03, 1.03, 1.03]
    assert result["faithful"]["samples_s"] == [1.03, 1.03, 1.03]
    assert len(json.loads((pilot.root / "measurement-results.json").read_text())) == 36
    assert not (pilot.ckroot / "pilot-shared-checkpoint").exists()


def test_pilot_preserves_finished_results_and_cleans_checkpoint_on_failure(tmp_path, monkeypatch):
    pilot, _ = pilot_fixture(tmp_path, monkeypatch, fail_at=3)
    with pytest.raises(RuntimeError, match="injected"):
        pilot.measurements()
    rows = json.loads((pilot.root / "measurement-results.json").read_text())
    assert len(rows) == 4
    assert {row["config"] for row in rows} == {"meta-faithful"}
    assert not (pilot.ckroot / "pilot-shared-checkpoint").exists()
    assert not (pilot.root / "pilot-result.json").exists()


@pytest.mark.parametrize("pilot_problem,expected_sink", [
    (None, "ring"), ("slow", "faithful"),
    ("requests", "faithful"), ("failure", "faithful"),
])
def test_main_runs_every_matrix_cell_in_model_order_and_records_failures(
        tmp_path, monkeypatch, pilot_problem, expected_sink):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    output = tmp_path / "validation" / "new-run"
    monkeypatch.setattr(sys, "argv", ["run_multirank_meta_cpu", "--output", str(output)])
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["/fake/gdtensor"]))
    monkeypatch.setattr(runner, "inventory", lambda: {})
    monkeypatch.setattr(runner.subprocess, "run", lambda *a, **kw: SimpleNamespace(stdout="idle"))
    calls = []

    class FakeRunner:
        def __init__(self, args):
            self.args = args
            self.root = args.output
            self.progress = {}

        def update(self, **fields):
            self.progress.update(fields)

        def run(self):
            calls.append((self.args.models[0], self.args.ranks,
                          self.args.checkpoint_format, self.args.meta_load_sink))
            assert self.args.repeats == 3
            assert self.args.configs == ["meta", "cpu"]
            assert self.args.data_pattern == "zero"
            if len(calls) == 2:
                raise RuntimeError("injected matrix resource failure")

    class FakePilot(FakeRunner):
        def run(self):
            if pilot_problem == "failure" and self.args.models == ["3b"]:
                raise RuntimeError("injected pilot failure")
            result = dict(model=self.args.models[0],
                          checkpoint_format=self.args.checkpoint_format,
                          ring=dict(requests_identical=pilot_problem != "requests",
                                    paired_delta_percent=3.001 if pilot_problem == "slow" else 3.0))
            runner.dump(self.root / "pilot-result.json", result)

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    monkeypatch.setattr(runner, "PilotRunner", FakePilot)
    runner.main()
    assert calls == [(model, ranks, fmt, expected_sink)
                     for model in ("1b", "3b") for ranks in (4, 8, 16)
                     for fmt in ("dcp", "safetensors")]
    progress = json.loads((output / "progress.json").read_text())
    matrix_errors = [error for error in progress["failed"] if error["stage"] == "matrix"]
    assert len(matrix_errors) == 1
    assert matrix_errors[0]["cell"] == "1b-4r-safetensors"
    assert len([cell for cell in progress["completed"] if cell.startswith("matrix-")]) == 11
    assert progress["status"] == "measurements-complete"
    assert (output / "sink-selection.json").is_file()


@pytest.mark.parametrize("failed_cell", [None, "1b-4r-safetensors"])
def test_fix1_runs_only_eight_cells_with_five_repeats_and_no_pilot(
        tmp_path, monkeypatch, failed_cell):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["/fake/gdtensor"]))
    monkeypatch.setattr(runner, "inventory", lambda: {})
    output = tmp_path / "validation" / "fix1"
    calls = []

    class FakeRunner:
        def __init__(self, args):
            self.args = args
            self.progress = {}

        def update(self, **fields):
            self.progress.update(fields)

        def run(self):
            calls.append((self.args.models[0], self.args.ranks,
                          self.args.checkpoint_format, self.args.meta_load_sink))
            assert self.args.repeats == 5
            assert self.args.configs == ["meta", "cpu"]
            assert self.args.data_pattern == "zero" and not self.args.validate_values
            assert self.args.output.parent == output / "matrix"
            self.update(current="fake phase", completed=[])
            progress = json.loads((output / "progress.json").read_text())
            assert progress["phase_progress"]["current"] == "fake phase"
            if self.args.output.name == failed_cell:
                raise RuntimeError("injected fix1 failure")

    class ForbiddenPilot:
        def __init__(self, *args, **kwargs):
            pytest.fail("The fix1 gated matrix must not run a sink pilot")

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    monkeypatch.setattr(runner, "PilotRunner", ForbiddenPilot)
    runner.run_fix1(output)

    assert calls == [(model, ranks, fmt, "faithful")
                     for model in ("1b", "3b") for ranks in (4, 8)
                     for fmt in ("dcp", "safetensors")]
    progress = json.loads((output / "progress.json").read_text())
    assert progress["matrix_cells"] == 8 and progress["repeats"] == 5
    assert progress["selected_sink"] == "faithful"
    assert progress["selected_source"] == "resident-local-shards"
    assert progress["status"] == "measurements-complete" and progress["current"] is None
    assert len(progress["completed"]) == (7 if failed_cell else 8)
    assert progress["failed"] == ([dict(stage="matrix", cell=failed_cell,
                                       error="injected fix1 failure")] if failed_cell else [])
    selection = json.loads((output / "sink-selection.json").read_text())
    assert selection["pilot_run"] is False and selection["ring_retained"] is True
    assert selection["selected_source"] == "resident-local-shards"
    assert not (output / "sink-pilot").exists()


def test_fix1_rejects_existing_measurement_output(tmp_path):
    output = tmp_path / "fix1"
    output.mkdir()
    evidence = output / "progress.json"
    evidence.write_text('{"status":"existing"}')
    with pytest.raises(ValueError, match="fresh fix1"):
        runner.run_fix1(output)
    assert evidence.read_text() == '{"status":"existing"}'


@pytest.mark.parametrize("writer_limit", [0, 1, 8])
@pytest.mark.parametrize("failed_cell", [None, "1b-4r-safetensors"])
def test_concurrent_save_runs_eight_save_only_cells_and_retains_memory_admission(
        tmp_path, monkeypatch, writer_limit, failed_cell):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["/fake/gdtensor"]))
    monkeypatch.setattr(runner, "inventory", lambda: {})
    output = tmp_path / "validation" / "concurrent"
    calls = []

    class FakeRunner:
        def __init__(self, args):
            self.args = args
            self.progress = {}

        def update(self, **fields):
            self.progress.update(fields)

        def run(self):
            args = self.args
            calls.append((args.models[0], args.ranks, args.checkpoint_format))
            assert args.repeats == 5 and args.save_only
            assert args.configs == ["meta", "cpu"]
            assert args.meta_load_sink == "faithful"
            assert args.writer_limit == (writer_limit or args.ranks)
            assert args.requested_writer_limit == writer_limit
            self.update(current="fake save", completed=[])
            progress = json.loads((output / "progress.json").read_text())
            assert progress["phase_progress"]["current"] == "fake save"
            assert progress["effective_writer_limit"] == args.writer_limit
            if args.output.name == failed_cell:
                raise RuntimeError("injected concurrent failure")

    monkeypatch.setattr(runner, "Runner", FakeRunner)
    runner.run_concurrent_save(output, writer_limit=writer_limit)
    assert calls == [(model, ranks, fmt) for model in ("1b", "3b") for ranks in (4, 8)
                     for fmt in ("dcp", "safetensors")]
    progress = json.loads((output / "progress.json").read_text())
    assert progress["status"] == "measurements-complete" and progress["current"] is None
    assert progress["phases"] == ["save"] and progress["matrix_cells"] == 8
    assert progress["requested_writer_limit"] == writer_limit
    assert progress["effective_writer_limits"] == {"4": writer_limit or 4, "8": writer_limit or 8}
    assert len(progress["completed"]) == (7 if failed_cell else 8)
    assert progress["failed"] == ([dict(stage="matrix", cell=failed_cell,
                                       error="injected concurrent failure")] if failed_cell else [])
    environment = json.loads((output / "environment.json").read_text())
    assert environment["effective_writer_limits"] == progress["effective_writer_limits"]
    assert environment["writer_limit_policy"] == progress["writer_limit_policy"]
    assert not (output / "sink-pilot").exists()


def test_concurrent_save_preserves_existing_evidence_and_records_interruption(tmp_path, monkeypatch):
    output = tmp_path / "validation" / "concurrent"
    output.mkdir(parents=True)
    evidence = output / "progress.json"
    evidence.write_text('{"status":"existing"}')
    with pytest.raises(ValueError, match="fresh concurrent-save"):
        runner.run_concurrent_save(output, writer_limit=0)
    assert evidence.read_text() == '{"status":"existing"}'
    with pytest.raises(ValueError, match="nonnegative"):
        runner.run_concurrent_save(output, writer_limit=-1)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["/fake/gdtensor"]))
    monkeypatch.setattr(runner, "inventory", lambda: {})

    class InterruptedRunner:
        def __init__(self, args):
            pass

        def run(self):
            raise KeyboardInterrupt("injected interruption")

    monkeypatch.setattr(runner, "Runner", InterruptedRunner)
    interrupted = tmp_path / "validation" / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        runner.run_concurrent_save(interrupted, writer_limit=0)
    progress = json.loads((interrupted / "progress.json").read_text())
    assert progress["status"] == "failed" and progress["current"] == "1b-4r-dcp"
    assert progress["error"] == "KeyboardInterrupt: injected interruption"


def test_concurrent_cli_selects_fresh_default_and_validates_writer_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    calls = []
    monkeypatch.setattr(runner, "run_concurrent_save", lambda output, **kwargs:
                        calls.append((output, kwargs["writer_limit"])))
    monkeypatch.setattr(sys, "argv", ["runner", "--concurrent-save", "--writer-limit", "0"])
    runner.main()
    assert calls == [(tmp_path / "validation/multirank-concurrent-save-20261030", 0)]
    monkeypatch.setattr(sys, "argv", ["runner", "--concurrent-save", "--writer-limit", "-1"])
    with pytest.raises(SystemExit):
        runner.main()
    monkeypatch.setattr(sys, "argv", ["runner", "--concurrent-save", "--fix1"])
    with pytest.raises(SystemExit):
        runner.main()
