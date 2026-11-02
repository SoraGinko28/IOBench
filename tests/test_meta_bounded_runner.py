"""Bounded-buffer matrix orchestration with no workers or measurement IO."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_meta_bounded_mem as runner


@pytest.fixture
def orchestration(tmp_path, monkeypatch):
    calls, events, analyzed, inventories = [], [], [], []
    output = tmp_path / "validation" / "bounded"
    options = SimpleNamespace(stage="sweep", buffer_mib=None, output=str(output))
    selection = {"chosen_size_mib": 64}
    real_read_text = Path.read_text

    def read_text(path, *args, **kwargs):
        if path == Path("/proc/meminfo"):
            return "MemTotal: 33554432 kB\n"
        return real_read_text(path, *args, **kwargs)

    def inventory():
        inventories.append(True)
        return {"metasl/gdt_storage.py": "fake-source-hash"}

    class Guard:
        def __init__(self, root, opts, update):
            assert root == output and opts is options
            self.device = (8, 1)

        def wait(self, context):
            events.append(("guard", dict(context)))

    class BaseRunner:
        def __init__(self, args):
            self.args, self.root, self.progress = args, args.output, {}

        def update(self, **fields):
            self.progress.update(fields)

        def _phase_once(self, case, checkpoint, model, config, action, *, attempt=1):
            events.append(
                (
                    "phase",
                    dict(
                        cell=self.root.name,
                        stage=options.stage,
                        buffer_mib=self.args.meta_load_sink_mib,
                        model=model,
                        config=config,
                        action=action,
                        attempt=attempt,
                        output=str(case),
                    ),
                )
            )
            return (checkpoint, action, attempt)

        def run(self):
            calls.append(self.args)
            self.root.mkdir(parents=True)
            self.update(current="mock phase", completed=[])
            recorded = json.loads((output / "progress.json").read_text())
            assert recorded["phase_progress"]["current"] == "mock phase"
            for config in self.args.configs:
                for action, attempt in (
                    ("save", 1),
                    ("cold-load", 1),
                    ("warm-load", 1),
                    ("warm-load", 2),
                ):
                    case = self.root / f"{config}-{action}-{attempt}"
                    checkpoint = self.root / "unused-checkpoint"
                    result = self._phase_once(
                        case, checkpoint, self.args.models[0], config, action, attempt=attempt
                    )
                    assert result == (checkpoint, action, attempt)
            (self.root / "measurement-results.json").write_text("[]")
            (self.root / "progress.json").write_text('{"status":"stage-complete"}')

    monkeypatch.setattr(runner, "Runner", BaseRunner)
    monkeypatch.setattr(runner, "IdleGuard", Guard)
    monkeypatch.setattr(runner, "inventory", inventory)
    monkeypatch.setattr(runner.platform, "platform", lambda: "mock platform")
    monkeypatch.setattr(Path, "read_text", read_text)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="mock torch"))
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["/mock/gdtensor"]))
    monkeypatch.setitem(
        sys.modules,
        "scripts.analyze_meta_bounded_mem",
        SimpleNamespace(
            analyze_cell=lambda path: analyzed.append(path),
            summarize_sweep=lambda root: dict(selection),
        ),
    )
    return SimpleNamespace(
        output=output,
        options=options,
        calls=calls,
        events=events,
        analyzed=analyzed,
        inventories=inventories,
        selection=selection,
    )


def assert_cpu_target_options(args):
    assert args.configs == ["meta", "cpu"]
    assert args.meta_load_sink == "ring"
    assert args.writer_limit == args.ranks and args.requested_writer_limit == 0
    assert args.warm_block_read_tolerance_bytes == 64 * 1024
    assert args.data_pattern == "zero" and not args.validate_values
    assert not args.save_only


def test_sweep_runs_twelve_shared_cells_with_three_pairs_at_each_size(orchestration):
    harness = orchestration
    progress = runner.run_stage(harness.output, harness.options)

    assert [
        (args.meta_load_sink_mib, args.models[0], args.ranks, args.repeats)
        for args in harness.calls
    ] == [
        (size, model, ranks, 3)
        for size in (64, 256, 1024)
        for model in ("1b", "3b")
        for ranks in (4, 8)
    ]
    for args in harness.calls:
        assert_cpu_target_options(args)
        assert args.checkpoint_format == "safetensors"
        assert args.safetensors_layout == "shared"
        assert args.output == (
            harness.output
            / "sweep"
            / f"{args.meta_load_sink_mib}mib"
            / f"{args.models[0]}-{args.ranks}r-safetensors"
        )
    assert len(harness.analyzed) == len(progress["completed"]) == 12
    assert harness.analyzed == [args.output for args in harness.calls]
    assert len(harness.inventories) == 2
    assert progress["status"] == "sweep-complete"
    assert progress["stage"] == "awaiting-default-change"
    assert progress["chosen_size_mib"] == 64 and not progress["failed"]
    environment = json.loads((harness.output / "sweep-environment.json").read_text())
    assert environment["meminfo"] == "MemTotal: 33554432 kB\n"
    assert environment["torch"] == "mock torch"


def test_confirmation_runs_four_shared_cells_and_three_billion_eight_rank_dcp(orchestration):
    harness = orchestration
    harness.options.stage = "confirm"
    harness.options.buffer_mib = 256
    harness.output.mkdir(parents=True)
    (harness.output / "sweep-summary.json").write_text('{"chosen_size_mib":256}')

    progress = runner.run_stage(harness.output, harness.options)

    assert [
        (args.models[0], args.ranks, args.checkpoint_format, args.repeats)
        for args in harness.calls
    ] == [
        ("1b", 4, "safetensors", 5),
        ("1b", 8, "safetensors", 5),
        ("3b", 4, "safetensors", 5),
        ("3b", 8, "safetensors", 5),
        ("3b", 8, "dcp", 3),
    ]
    for args in harness.calls:
        assert_cpu_target_options(args)
        assert args.meta_load_sink_mib == 256
        assert args.output.parent == harness.output / "confirm"
        assert args.safetensors_layout == (
            "shared" if args.checkpoint_format == "safetensors" else "per-rank"
        )
    assert len(harness.analyzed) == len(progress["completed"]) == 5
    assert progress["status"] == "confirmation-complete"
    assert progress["chosen_size_mib"] == 256 and not progress["failed"]


def test_confirmation_refuses_when_no_buffer_size_passed(orchestration):
    harness = orchestration
    harness.options.stage = "confirm"
    harness.output.mkdir(parents=True)
    (harness.output / "sweep-summary.json").write_text('{"chosen_size_mib":null}')

    with pytest.raises(ValueError, match="No buffer size passed the sweep"):
        runner.run_stage(harness.output, harness.options)

    assert not harness.calls and not harness.events and not harness.analyzed
    assert not (harness.output / "confirm").exists()


def test_idle_guard_immediately_precedes_every_phase_and_warm_retry(orchestration, monkeypatch):
    harness = orchestration
    monkeypatch.setattr(runner, "SIZES_MIB", (64,))
    monkeypatch.setattr(runner, "CELLS", (("1b", 4),))

    runner.run_stage(harness.output, harness.options)

    assert len(harness.events) == 16
    guarded = []
    for index in range(0, len(harness.events), 2):
        guard, phase = harness.events[index:index + 2]
        assert guard[0] == "guard" and phase[0] == "phase"
        assert guard[1] == phase[1]
        assert guard[1]["buffer_mib"] == 64
        assert guard[1]["stage"] == "sweep"
        guarded.append((guard[1]["config"], guard[1]["action"], guard[1]["attempt"]))
    assert guarded == [
        (config, action, attempt)
        for config in ("meta", "cpu")
        for action, attempt in (
            ("save", 1), ("cold-load", 1), ("warm-load", 1), ("warm-load", 2)
        )
    ]
