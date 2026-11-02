"""Small orchestration and analysis checks, without worker processes or payloads."""

import csv
import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from scripts import analyze_shared_safetensors as analysis
from scripts import run_shared_safetensors as runner


@pytest.mark.parametrize("argv,expected", [
    (["python", "-m", "metasl.run", "--action", "save"], True),
    (["/venv/bin/python3", "-m", "metasl.run"], True),
    (["bash", "-lc", "python -m metasl.run"], False),
    (["pgrep", "-f", "metasl.run"], False),
    (["python", "scripts/run_shared_safetensors.py"], False),
    (["python", "scripts/fsbench.py"], True),
    (["/usr/bin/fsbench", "read"], True),
    ([], False),
])
def test_detect_actual_measurement_programs(argv, expected):
    assert runner.measurement_process(argv) is expected


def test_idle_delta_uses_elapsed_disk_time_and_cpu_ticks():
    before = dict(monotonic=2, cpu_total=100, cpu_idle=70, disk_busy_ms=20)
    after = dict(monotonic=4, cpu_total=300, cpu_idle=250, disk_busy_ms=40, available_bytes=123)
    result = runner.idle_delta(before, after)
    assert result["cpu_busy_percent"] == pytest.approx(10)
    assert result["disk_busy_percent"] == 1
    assert result["available_bytes"] == 123


def test_matrix_reuses_runner_limits_pairs_phases_and_warm_redo_guard(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner, "inventory", lambda: {"source": "hash"})
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(__version__="test"))
    monkeypatch.setitem(sys.modules, "gdtensor", SimpleNamespace(__path__=["test"]))
    calls, guards = [], []

    class Guard:
        def __init__(self, output, options, update):
            self.device = (1, 2)

        def wait(self, context):
            guards.append(context)

    class BaseRunner:
        def __init__(self, args):
            self.args, self.root, self.progress = args, args.output, {}

        def update(self, **fields):
            self.progress.update(fields)

        def _phase_once(self, case, checkpoint, model, config, action, *, attempt=1):
            assert guards[-1]["action"] == action
            assert guards[-1]["attempt"] == attempt

        def run(self):
            calls.append(self.args)
            for action, attempt in (("save", 1), ("cold-load", 1), ("warm-load", 1), ("warm-load", 2)):
                self._phase_once(self.root, self.root, self.args.models[0], "meta", action, attempt=attempt)

    monkeypatch.setattr(runner, "Runner", BaseRunner)
    monkeypatch.setattr(runner, "IdleGuard", Guard)
    output = tmp_path / "validation" / "new"
    (output / "correctness").mkdir(parents=True)
    (output / "correctness" / "progress.json").write_text('{"status":"complete"}')
    result = runner.run_matrix(output, SimpleNamespace(output=str(output)))
    assert len(calls) == 6 and len(guards) == 24
    assert [(a.models[0], a.ranks, a.repeats, a.safetensors_max_file_gib) for a in calls] == [
        ("1b", 4, 5, None), ("1b", 8, 5, None), ("3b", 4, 5, None), ("3b", 8, 5, None),
        ("3b", 4, 1, 5), ("3b", 8, 1, 5)]
    for args in calls:
        assert args.safetensors_layout == "shared"
        assert args.writer_limit == args.ranks and args.requested_writer_limit == 0
        assert args.warm_block_read_tolerance_bytes == 65536
        assert args.configs == ["meta", "cpu"] and args.meta_load_sink == "faithful"
        assert not args.save_only and args.data_pattern == "zero"
    assert result["correctness_before_measurements"]["status"] == "complete"
    assert result["status"] == "measurements-complete" and not result["failed"]


def test_profile_preserves_hist_header_and_file_rank_counts(tmp_path):
    raw = tmp_path / "rank-000.io.jsonl"
    events = [
        dict(file="model.safetensors", op="write", requested=512, completed=512, offset=0),
        dict(file="model.safetensors", op="pwrite", requested=65536, completed=65536, offset=512, io_layer="payload"),
        dict(file="model.safetensors", op="fsync", requested=0, completed=0, offset=0),
        dict(file="model.safetensors", op="host_sink", completed=65536, io_layer="staging"),
    ]
    raw.write_text("".join(json.dumps(e) + "\n" for e in events))
    case = dict(row=dict(report=str(tmp_path / "rank-000.json"), model="1b", world_size=2,
                         phase="save", config="cpu", repeat=1, rank=0),
                provenance=dict(trace_sha256=hashlib.sha256(raw.read_bytes()).hexdigest()))
    profile = analysis.profile_case(case)
    assert profile["payload_request_size_histogram"] == {"65536": 1}
    assert profile["files"]["model.safetensors"]["header_bytes"] == 512
    peer = dict(profile, rank=1)
    combined = analysis.aggregate_profiles([profile, peer])[0]
    assert combined["complete"]
    assert combined["payload_file_count"] == 1
    assert combined["files"]["model.safetensors"]["writer_ranks"] == [0, 1]
    assert combined["files"]["model.safetensors"]["reader_ranks"] == []
    assert combined["payload_request_size_histogram"] == {"65536": 2}
    assert combined["fsyncs"] == 2
    assert combined["observed_payload_file_opens"] is None
    assert combined["distinct_rank_file_pairs"] == 2
    assert combined["header_requests"] == 2
    raw.write_text(raw.read_text() + json.dumps(events[0]) + "\n")
    with pytest.raises(ValueError, match="Raw trace changed"):
        analysis.profile_case(case)


@pytest.mark.parametrize("block_reads,residency,valid", [
    (0, 100., True), (4096, 100., True), (65536, 99., True),
    (65537, 100., False), (0, 98.99, False),
])
def test_warm_rule_records_reads_and_accepts_exactly_64kib(monkeypatch, block_reads, residency, valid):
    rank = dict(report="r.json", rank=0, read_bytes=block_reads, residency_percent=residency,
                cache_before_call=dict(after_resident_buffer_preparation=True, probe_end_ns=99,
                                       residency=dict(fully_warm_data_residency=residency >= 99)))
    sample = dict(rank_checks=[rank], valid=False)
    monkeypatch.setattr(analysis.core, "warm_cache_io_audit", lambda cases: dict(samples=[sample]))
    result = analysis.warm_audit([dict(row=dict(report="r.json"), stats=dict(dcp_started_ns=100))])
    assert result["all_valid"] is valid
    assert result["samples"][0]["rank_checks"][0]["read_bytes"] == block_reads


def test_critical_csv_keeps_phase_and_disjoint_partition(tmp_path):
    partition = dict(pre_payload=.1, admission_wait=0, payload_io=.2, materialize=0, sink=.1,
                     post_read_wait=0, uncovered=.1, fsync=0, other_staging_or_io=0)
    record = dict(repeat=1, config="cpu", rank=3, roles=["max_wall"], total_s=.5, report="r.json",
                  breakdown=dict(disjoint_partition_s=partition, partition_sum_s=.5, partition_matches_wall=True))
    failure = dict(phase="cold-load", failed_gates_strict=["wall"], failed_gates_floor=[],
                   critical_rank_breakdowns=[record])
    path = tmp_path / "critical.csv"
    analysis.critical_csv([dict(model="1b", ranks=4, failures=[failure])], path)
    row = list(csv.DictReader(path.open()))[0]
    assert row["phase"] == "cold-load" and row["rank"] == "3"
    assert float(row["partition_sum_s"]) == .5
