"""CUDA API times must end after device work, while CPU calls never touch CUDA."""

from metasl import phase


def test_cuda_call_synchronizes_before_and_after_work_and_reports_memory(monkeypatch):
    order = []
    monkeypatch.setattr(phase.dist, "is_initialized", lambda: False)
    monkeypatch.setattr(phase.torch.cuda, "synchronize", lambda device: order.append("sync"))
    monkeypatch.setattr(phase.torch.cuda, "reset_peak_memory_stats", lambda device: None)
    for name, value in (
        ("memory_allocated", 12),
        ("memory_reserved", 24),
        ("max_memory_allocated", 32),
        ("max_memory_reserved", 48),
    ):
        monkeypatch.setattr(phase.torch.cuda, name, lambda device, value=value: value)

    def work():
        order.append("work")
        return 7

    value, stats = phase.measure_dcp(work, cuda_device="cuda:0")
    assert value == 7
    assert order == ["sync", "work", "sync"]
    assert stats["cuda_synchronized"] is True
    assert stats["cuda_memory"]["peak_allocated_bytes"] == 32
    assert stats["dcp_wall_seconds"] >= 0


def test_cpu_call_does_not_initialize_cuda(monkeypatch):
    monkeypatch.setattr(phase.dist, "is_initialized", lambda: False)

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU timing touched CUDA")

    monkeypatch.setattr(phase.torch.cuda, "synchronize", forbidden)
    monkeypatch.setattr(phase.torch.cuda, "reset_peak_memory_stats", forbidden)
    value, stats = phase.measure_dcp(lambda: 9)
    assert value == 9
    assert "cuda_synchronized" not in stats
