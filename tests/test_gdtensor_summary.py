"""Backend byte metrics through the real supervisor summary publication path."""

import json
import threading
from types import SimpleNamespace

import pytest

from metasl import run as supervisor


@pytest.fixture
def publish_summary(tmp_path, monkeypatch):
    """Replace worker execution; retain report-file reads and summary generation."""

    def run(reports, *, backend="gdtensor", action="load", scope="model", format_name="dcp"):
        args = supervisor.build_parser().parse_args(
            [
                "--action", action,
                "--io-backend", backend,
                "--checkpoint-state", scope,
                "--checkpoint-format", format_name,
                "--bundle", str(tmp_path / "bundle"),
                "--checkpoint", str(tmp_path / "checkpoint"),
                "--output", str(tmp_path / "output"),
                "--ranks", str(len(reports)),
                "--start-method", "spawn",
                "--metadata-limit", "0",
            ]
        )

        class Process:
            exitcode = 0

            def __init__(self, *, target, args):
                self.args = args

            def start(self):
                worker_args, rank = self.args[:2]
                (worker_args.output / f"rank-{rank:03d}.json").write_text(
                    json.dumps(reports[rank])
                )

            def is_alive(self):
                return False

            def join(self, timeout=None):
                pass

        context = SimpleNamespace(Event=threading.Event, Process=Process)
        monkeypatch.setattr(supervisor, "parse_args", lambda _: args)
        monkeypatch.setattr(supervisor.mp, "get_context", lambda _: context)
        monkeypatch.setattr(supervisor, "available_memory", lambda: 100 * supervisor.GIB)
        monkeypatch.setattr(
            supervisor,
            "sample_group",
            lambda *args: (
                {
                    "total_pss_bytes": 0,
                    "worker_rss_bytes": 0,
                    "available_bytes": 100 * supervisor.GIB,
                },
                [],
            ),
        )
        supervisor.main([])
        return json.loads((args.output / "summary.json").read_text())

    return run


def rank_report(rank, **fields):
    return {
        "rank": rank,
        "world_size": 2,
        "stats": {
            "phase_started": 10.0 + rank,
            "phase_ended": 12.0 + rank,
            "dcp_wall_seconds": 2.0,
            "proc_io_delta": {"read_bytes": 0, "write_bytes": 0},
            "process_high_water_rss_bytes": 1024,
        },
        "target_payload_bytes": 12,
        **fields,
    }


@pytest.mark.parametrize("scope", ["model"])
def test_load_summary_distinguishes_intersections_from_native_chunks(publish_summary, scope):
    reports = [
        rank_report(0, gdtensor_payload_read_bytes=12, native_full_chunk_payload_bytes=24),
        rank_report(1, gdtensor_payload_read_bytes=8, native_full_chunk_payload_bytes=20),
    ]
    summary = publish_summary(reports, scope=scope)
    assert summary["completed"] == 2 and not summary["errors"]
    assert summary["io_backend"] == "gdtensor"
    assert summary["checkpoint_state"] == scope
    assert summary["aggregate_gdtensor_payload_read_bytes"] == 20
    assert summary["aggregate_native_full_chunk_payload_bytes"] == 44
    assert summary["aggregate_source_chunk_bytes"] is None
    assert "aggregate_gdtensor_payload_write_bytes" not in summary


def test_save_summary_reports_writes_without_inventing_read_metrics(publish_summary):
    summary = publish_summary(
        [
            rank_report(0, gdtensor_payload_write_bytes=12),
            rank_report(1, gdtensor_payload_write_bytes=8),
        ],
        action="save",
    )
    assert summary["aggregate_gdtensor_payload_write_bytes"] == 20
    assert summary["aggregate_source_chunk_bytes"] is None
    assert "aggregate_gdtensor_payload_read_bytes" not in summary
    assert "aggregate_native_full_chunk_payload_bytes" not in summary


def test_missing_rank_metric_stays_unknown_instead_of_undercounting(publish_summary):
    summary = publish_summary(
        [
            rank_report(0, gdtensor_payload_read_bytes=12, native_full_chunk_payload_bytes=24),
            rank_report(1, native_full_chunk_payload_bytes=20),
        ]
    )
    assert summary["aggregate_gdtensor_payload_read_bytes"] is None
    assert summary["aggregate_native_full_chunk_payload_bytes"] == 44


@pytest.mark.parametrize("backend", ["gdtensor"])
def test_safetensors_summary_retains_measured_gdtensor_reads(publish_summary, backend):
    reports = [
        rank_report(0, io={"data_read_bytes": 12}),
        rank_report(1, io={"data_read_bytes": 8}),
    ]
    summary = publish_summary(reports, backend=backend, format_name="safetensors")
    if backend == "gdtensor":
        assert summary["aggregate_file_api_read_bytes"] == 20
        assert "positioned" in summary["file_api_read_observation"]
    else:
        assert summary["aggregate_file_api_read_bytes"] is None
        assert "bypass" in summary["file_api_read_observation"]


