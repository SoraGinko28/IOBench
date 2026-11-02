"""DCP comparison clocks exclude process sampling overhead."""
from metasl import phase


def test_exact_dcp_clock_excludes_process_sampling(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(phase.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(phase, "perf_counter", lambda: clock[0])

    def measured(call):
        clock[0] += 10
        result = call()
        clock[0] += 20
        return result, {"seconds": 32}

    monkeypatch.setattr(phase, "measure_call", measured)

    def work():
        clock[0] += 2
        return "loaded"

    result, stats = phase.measure_dcp(work, action="load", world_size=1)
    assert result == "loaded"
    assert stats["dcp_wall_seconds"] == 2
    assert stats["dcp_ended"] - stats["dcp_started"] == 2


def test_warm_cache_runs_after_preparation_and_probes_every_rank(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from metasl import cache_control, iostats

    events = []
    metadata = tmp_path / '.metadata'
    metadata.write_bytes(b'metadata')
    monkeypatch.setattr(phase.dist, 'is_initialized', lambda: True)
    monkeypatch.setattr(phase.dist, 'barrier', lambda: events.append('barrier'))
    monkeypatch.setattr(iostats, 'warm_checkpoint',
                        lambda path: events.append('warm') or dict(data_bytes=4096))
    monkeypatch.setattr(iostats, 'checkpoint_files', lambda path: [metadata])
    monkeypatch.setattr(cache_control, 'checkpoint_residency',
                        lambda path: events.append('probe') or dict(data_resident_percent=100))
    args = SimpleNamespace(rank=0, warm_cache=True)
    result = phase._prepare_call_cache(args, tmp_path, 'load')
    assert events == ['barrier', 'warm', 'barrier', 'probe']
    assert result['warmup']['metadata_bytes'] == 8
    assert result['after_resident_buffer_preparation']
    assert result['probe_start_ns'] <= result['probe_end_ns']
    events.clear()
    args.rank = 1
    result = phase._prepare_call_cache(args, tmp_path, 'load')
    assert events == ['barrier', 'barrier', 'probe']
    assert result['warmup'] is None
    assert result['residency']['data_resident_percent'] == 100
    events.clear()
    assert phase._prepare_call_cache(args, tmp_path, 'save') is None
    args.warm_cache = False
    assert phase._prepare_call_cache(args, tmp_path, 'load') is None
    assert not events
