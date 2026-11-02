import io
import json
from contextlib import contextmanager

import pytest

from metasl.iostats import IORecorder, RecordedFileSystem, RecordedStream, fsync_stream, union_size


def test_union_size():
    assert union_size([(0, 4), (2, 6), (10, 12)]) == 8
    assert union_size([]) == 0


def test_recorded_stream_tracks_offsets_and_summarizes():
    recorder = IORecorder()
    stream = RecordedStream(io.BytesIO(), recorder, "__0_0.distcp", "wb")
    stream.write(b"abc")
    stream.write(b"defg")
    stream.seek(0)
    stream.read(2)
    stream.read(2)
    stream.seek(1)
    stream.read(1)
    ops = [(e["op"], e["offset"], e["completed"]) for e in stream.events]
    assert ops == [
        ("write", 0, 3),
        ("write", 3, 4),
        ("seek", 7, 0),
        ("read", 0, 2),
        ("read", 2, 2),
        ("seek", 4, 1),
        ("read", 1, 1),
    ]
    summary = recorder.summary()
    data = summary["files"]["__0_0.distcp"]
    assert data["write"]["bytes"] == 7 and data["write"]["unique_bytes"] == 7
    assert data["read"]["bytes"] == 5 and data["read"]["unique_bytes"] == 4
    assert data["read"]["offset_transitions"] == {"contiguous": 1, "backward_or_overlap": 1}
    assert data["seek_directions"] == {"backward": 2}
    assert summary["data_write_bytes"] == 7 and summary["data_read_bytes"] == 5
    assert summary["data_backward_seeks"] == 2
    again = IORecorder()
    other = RecordedStream(io.BytesIO(), again, "__0_0.distcp", "wb")
    other.write(b"abc")
    other.write(b"defg")
    other.seek(0)
    other.read(2)
    other.read(2)
    other.seek(1)
    other.read(1)
    assert again.summary()["data_pattern_sha256"] == summary["data_pattern_sha256"]


class Failing(io.BytesIO):
    def read(self, size=-1):
        raise OSError("boom")


def test_failed_call_leaves_no_event_and_no_active_count():
    recorder = IORecorder()
    stream = RecordedStream(Failing(b"abc"), recorder, "__0_0.distcp", "rb")
    with pytest.raises(OSError):
        stream.read(2)
    assert recorder.active == 0 and recorder.max_active == 1
    assert stream.events == []
    assert recorder.summary()["max_simultaneous_data_calls"] == 1


def test_metadata_file_excluded_from_data_signature():
    def run(metadata_ops):
        recorder = IORecorder()
        data = RecordedStream(io.BytesIO(), recorder, "__0_0.distcp", "wb")
        data.write(b"payload")
        meta = RecordedStream(io.BytesIO(), recorder, ".metadata", "wb")
        for chunk in metadata_ops:
            meta.write(chunk)
        return recorder.summary()

    a, b = run([b"x"]), run([b"xyz", b"more"])
    assert a["data_pattern_sha256"] == b["data_pattern_sha256"]
    assert a["metadata_write_bytes"] == 1 and b["metadata_write_bytes"] == 7
    assert a["data_write_bytes"] == b["data_write_bytes"] == 7


def test_read_to_eof_and_readinto_accounting():
    recorder = IORecorder()
    stream = RecordedStream(io.BytesIO(b"0123456789"), recorder, "__0_0.distcp", "rb")
    assert stream.read() == b"0123456789"
    buffer = bytearray(4)
    stream.seek(2)
    assert stream.readinto(buffer) == 4 and bytes(buffer) == b"2345"
    ops = [(e["op"], e["offset"], e["requested"], e["completed"]) for e in stream.events]
    assert ops == [("read", 0, -1, 10), ("seek", 10, 2, 2), ("readinto", 2, 4, 4)]
    assert recorder.summary()["files"]["__0_0.distcp"]["read"]["bytes"] == 14


class LocalFileSystem:
    def __init__(self, fail_close=False):
        self.fail_close = fail_close

    @contextmanager
    def create_stream(self, path, mode):
        with path.open(mode) as stream:
            yield stream
        if self.fail_close:
            raise OSError("close failed")


@pytest.mark.parametrize("name", [".metadata", ".metadata.tmp", "state.bin"])
def test_metadata_close_releases_events_and_preserves_summary_and_trace(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"first line\nsecond line\n")
    recorder = IORecorder()
    filesystem = RecordedFileSystem(LocalFileSystem(), recorder)
    before_trace, after_trace = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    with filesystem.create_stream(path, "rb") as stream:
        assert stream.read(3) == b"fir"
        assert stream.readline() == b"st line\n"
        stream.seek(0)
        buffer = bytearray(5)
        assert stream.readinto(buffer) == 5
        assert stream.read() == b" line\nsecond line\n"
        original_events = stream.events
        expected_summary = recorder.summary(before_trace)
        expected_trace = before_trace.read_text()
    assert original_events == []
    assert isinstance(recorder.files[name]["events"], bytes)
    assert recorder.summary(after_trace) == expected_summary
    assert after_trace.read_text() == expected_trace
    assert recorder.summary() == expected_summary
    restored_events = recorder.files[name]["events"]
    assert restored_events is not original_events
    assert [{"file": name, **event} for event in restored_events] == [
        json.loads(line) for line in expected_trace.splitlines()
    ]


@pytest.mark.parametrize("failure", ["body", "close"])
def test_metadata_exceptional_close_preserves_partial_evidence(tmp_path, failure):
    path = tmp_path / ".metadata"
    path.write_bytes(b"checkpoint metadata")
    recorder = IORecorder()
    filesystem = RecordedFileSystem(LocalFileSystem(failure == "close"), recorder)
    before_trace, after_trace = tmp_path / "before.jsonl", tmp_path / "after.jsonl"
    with pytest.raises(OSError, match="failed"):
        with filesystem.create_stream(path, "rb") as stream:
            assert stream.read(10) == b"checkpoint"
            original_events = stream.events
            expected_summary = recorder.summary(before_trace)
            expected_trace = before_trace.read_text()
            if failure == "body":
                raise OSError("metadata decoding failed")
    assert original_events == []
    assert isinstance(recorder.files[path.name]["events"], bytes)
    assert recorder.summary(after_trace) == expected_summary
    assert after_trace.read_text() == expected_trace


def test_data_stream_close_keeps_existing_events(tmp_path):
    path = tmp_path / "__0_0.distcp"
    recorder = IORecorder()
    filesystem = RecordedFileSystem(LocalFileSystem(), recorder)
    with filesystem.create_stream(path, "wb") as stream:
        stream.write(b"data")
        original_events = stream.events
    assert recorder.files[path.name]["events"] is original_events
    assert len(original_events) == 1
    assert recorder.summary()["data_write_bytes"] == 4


@pytest.mark.parametrize("name", ["__0_0.distcp", ".metadata.tmp"])
def test_opted_in_lifecycle_events_preserve_bytes_and_request_signature(tmp_path, name):
    path = tmp_path / name
    recorder = IORecorder()
    filesystem = RecordedFileSystem(LocalFileSystem(), recorder, capture_lifecycle=True)
    with filesystem.create_stream(path, "wb") as stream:
        stream.write(b"checkpoint")
        before = recorder.summary()
        stream.flush()
        fsync_stream(stream)
    summary = recorder.summary()
    events = recorder.files[name]["events"]
    assert [event["op"] for event in events] == ["write", "flush", "fsync", "close"]
    assert path.read_bytes() == b"checkpoint"
    assert summary["files"][name]["pattern_sha256"] == before["files"][name]["pattern_sha256"]
    assert summary["data_pattern_sha256"] == before["data_pattern_sha256"]
    assert recorder.in_flight == recorder.active == 0
    for event in events[1:]:
        assert event["request"] is False and event["success"] == 1
        assert event["offset"] == event["requested"] == event["completed"] == 0
        assert event["start_ns"] >= 0 and event["duration_ns"] >= 0


def test_shared_file_open_records_actual_enter_interval(tmp_path):
    import time

    @contextmanager
    def delayed_open(path, mode):
        time.sleep(0.01)
        with path.open(mode) as stream:
            yield stream

    filesystem = LocalFileSystem()
    filesystem.create_stream = delayed_open
    recorder = IORecorder()
    recorded = RecordedFileSystem(filesystem, recorder, capture_lifecycle=True, capture_open=True)
    with recorded.create_stream(tmp_path / "model.safetensors", "wb") as stream:
        stream.write(b"header")
    opened = recorder.files["model.safetensors"]["events"][0]
    assert opened["op"] == "open" and opened["mode"] == "wb"
    assert opened["duration_ns"] >= 10_000_000
    assert opened["request"] is False and opened["completed"] == 0


def test_failed_lifecycle_call_is_recorded_without_losing_context_cleanup(tmp_path, monkeypatch):
    def fail_fsync(descriptor):
        raise OSError("injected fsync failure")

    monkeypatch.setattr("metasl.iostats.os.fsync", fail_fsync)
    recorder = IORecorder()
    filesystem = RecordedFileSystem(LocalFileSystem(), recorder, capture_lifecycle=True)
    with pytest.raises(OSError, match="injected fsync failure"):
        with filesystem.create_stream(tmp_path / "__0_0.distcp", "wb") as stream:
            stream.write(b"payload")
            fsync_stream(stream)
    assert stream.stream.closed
    assert [(event["op"], event.get("success", 1)) for event in stream.events] == [
        ("write", 1),
        ("fsync", 0),
        ("close", 1),
    ]
    assert recorder.in_flight == recorder.active == 0


def test_underlying_context_exit_failure_is_timed_as_failed_close(tmp_path):
    recorder = IORecorder()
    filesystem = RecordedFileSystem(
        LocalFileSystem(fail_close=True), recorder, capture_lifecycle=True
    )
    with pytest.raises(OSError, match="close failed"):
        with filesystem.create_stream(tmp_path / "__0_0.distcp", "wb") as stream:
            stream.write(b"payload")
    assert stream.events[-1]["op"] == "close"
    assert stream.events[-1]["success"] == 0
    assert recorder.in_flight == recorder.active == 0
