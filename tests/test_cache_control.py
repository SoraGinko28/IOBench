"""Cache probes must measure residency without reading the checkpoint payload."""

import json
from types import SimpleNamespace

import pytest

from metasl import cache_control


def test_empty_target_records_memory_without_running_fincore(tmp_path, monkeypatch):
    monkeypatch.setattr(cache_control.subprocess, "run", lambda *a, **k: pytest.fail("empty probe"))
    result = cache_control.checkpoint_residency(tmp_path / "absent")
    assert result["empty"] is True
    assert result["directory_exists"] is False
    assert result["files"] == []
    assert result["memory_bytes"]["MemAvailable"] > 0
    assert result["near_zero_data_residency"] is None


@pytest.mark.parametrize("resident_pages,percent,cold", [(0, 0, True), (1, 100, False)])
def test_fincore_uses_pages_for_partial_files_and_preserves_residency(
    tmp_path, monkeypatch, resident_pages, percent, cold,
):
    file = tmp_path / "__0_0.distcp"
    file.write_bytes(b"payload")

    def probe(command, **kwargs):
        assert command == [
            "fincore", "--json", "--bytes", "--output", "PAGES,SIZE,RES,FILE", str(file),
        ]
        assert kwargs["check"] is True
        return SimpleNamespace(stdout=json.dumps(dict(fincore=[dict(
            file=str(file), size=7, pages=resident_pages, res=resident_pages * 4096,
        )])))

    monkeypatch.setattr(cache_control.subprocess, "run", probe)
    monkeypatch.setattr(cache_control.os, "sysconf", lambda name: 4096)
    result = cache_control.checkpoint_residency(tmp_path)
    assert result["near_zero_data_residency"] is cold
    assert result["files"][0]["resident_percent"] == percent
    assert result["files"][0]["resident_pages"] == resident_pages
    assert "host-side" in result["limitation"]


def test_fincore_missing_file_is_an_error_instead_of_a_cold_claim(tmp_path, monkeypatch):
    (tmp_path / "__0_0.distcp").write_bytes(b"payload")
    monkeypatch.setattr(
        cache_control.subprocess, "run",
        lambda *a, **k: SimpleNamespace(stdout='{"fincore": []}'),
    )
    with pytest.raises(RuntimeError, match="every checkpoint file"):
        cache_control.checkpoint_residency(tmp_path)
