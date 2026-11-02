import json
import multiprocessing as mp
import threading
import time

import pytest

from metasl.admission import GIB, PermitClient, PermitQueue

SAMPLE = {"total_pss_bytes": 1 * GIB, "available_bytes": 64 * GIB}


def fresh_sample(queue, **overrides):
    ticket = queue.begin_sample()
    return dict(
        SAMPLE,
        admission_sample_ticket=ticket,
        sample_start_ns=ticket["sample_start_ns"],
        sample_end_ns=time.perf_counter_ns(),
        **overrides,
    )


def wait_for(predicate, timeout=5):
    deadline = time.perf_counter() + timeout
    while not predicate():
        if time.perf_counter() > deadline:
            raise TimeoutError
        time.sleep(0.005)


def make_queue(tmp_path, ranks, limit, role="writer"):
    pipes = [mp.Pipe() for _ in range(ranks)]
    abort = threading.Event()
    queue = PermitQueue([p[0] for p in pipes], abort, limit, tmp_path, role)
    clients = [PermitClient(p[1], abort, timeout=5) for p in pipes]
    queue.thread.start()
    return queue, clients, abort


def test_fifo_permits_and_empty_plans(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 3, limit=1)
    tokens = {}

    def acquire(rank, empty=False):
        tokens[rank] = clients[rank].acquire(empty=empty)

    threads = [
        threading.Thread(target=acquire, args=(0,)),
        threading.Thread(target=acquire, args=(1,)),
        threading.Thread(target=acquire, args=(2, True)),
    ]
    threads[0].start()
    wait_for(lambda: 0 in queue.seen)
    threads[1].start()
    wait_for(lambda: 1 in queue.seen)
    threads[2].start()
    threads[2].join(2)
    assert tokens[2] is None  # empty plans never wait for the baseline
    time.sleep(0.05)
    assert 0 not in tokens  # no grants before the baseline sample
    queue.update(fresh_sample(queue))
    threads[0].join(2)
    assert tokens[0] == 0
    time.sleep(0.05)
    assert 1 not in tokens
    clients[0].release(0, time.perf_counter())
    threads[1].join(2)
    assert tokens[1] == 1
    clients[1].release(1, time.perf_counter())
    wait_for(lambda: not queue.active)
    assert not abort.is_set() and queue.error is None
    queue.close()
    events = [
        line.split('"event": "')[1].split('"')[0]
        for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()
    ]
    assert [v for v in events if v not in {"sample_requested", "sample_begin"}] == [
        "enqueue",
        "enqueue",
        "empty",
        "baseline",
        "grant",
        "release",
        "grant",
        "release",
    ]


def test_duplicate_request_aborts_group(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1, role="reader")
    clients[0].connection.send(("request", 0.0))
    clients[0].connection.send(("request", 0.0))
    wait_for(abort.is_set)
    assert "Duplicate reader request" in queue.error
    queue.close()


def test_wrong_release_token_aborts_group(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    tokens = []
    thread = threading.Thread(target=lambda: tokens.append(clients[0].acquire()))
    thread.start()
    wait_for(lambda: 0 in queue.seen)
    queue.update(fresh_sample(queue))
    thread.join(2)
    assert tokens == [0]
    clients[0].release(7, 0.0)
    wait_for(abort.is_set)
    assert "Invalid writer release" in queue.error
    queue.close()


def test_soft_memory_threshold_aborts(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    queue.update({"total_pss_bytes": 30 * GIB, "available_bytes": 64 * GIB})
    assert abort.is_set() and "soft memory threshold" in queue.error
    with pytest.raises(RuntimeError, match="aborted"):
        clients[0].acquire()
    queue.close()


def test_limit_must_be_positive(tmp_path):
    with pytest.raises(ValueError):
        PermitQueue([], threading.Event(), 0, tmp_path, "writer")


def test_rank_may_request_again_after_release(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    tokens = []
    thread = threading.Thread(target=lambda: tokens.append(clients[0].acquire()))
    thread.start()
    wait_for(lambda: 0 in queue.seen)
    queue.update(fresh_sample(queue))
    thread.join(2)
    clients[0].release(0, 0.0)
    wait_for(lambda: not queue.active)
    assert clients[0].acquire() == 0  # second round, granted directly
    clients[0].release(0, 0.0)
    wait_for(lambda: not queue.active)
    assert queue.rounds[0] == 2 and not abort.is_set()
    queue.close()


def test_rank_that_never_requested_may_exit(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 2, limit=1)
    clients[1].connection.close()  # rank 1 finished without ever needing a permit
    time.sleep(0.1)
    assert not abort.is_set() and queue.error is None and queue.connections[1] is None
    tokens = []
    thread = threading.Thread(target=lambda: tokens.append(clients[0].acquire()))
    thread.start()
    wait_for(lambda: 0 in queue.seen)
    queue.update(fresh_sample(queue))
    thread.join(2)
    assert tokens == [0]
    clients[0].connection.close()  # rank 0 exits while holding the permit: that is a failure
    wait_for(abort.is_set)
    assert "holding or awaiting" in queue.error
    queue.close()


def test_first_requests_wake_only_after_all_live_ranks_report(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 2, limit=1)
    clients[0].connection.send(("request", {"request_send_ns": time.perf_counter_ns()}))
    wait_for(lambda: 0 in queue.seen)
    assert not queue.sample_needed.is_set()
    clients[1].connection.send(("empty", {"request_send_ns": time.perf_counter_ns()}))
    wait_for(queue.sample_needed.is_set)
    assert queue.all_request_ready_ns is not None and queue.baseline is None
    ticket = queue.begin_sample()
    assert not queue.sample_needed.is_set()
    assert ticket["required_ranks"] == [0, 1]
    assert ticket["sample_start_ns"] >= ticket["all_request_ready_ns"]
    queue.close()


def test_sample_before_request_cannot_publish_baseline(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    old = fresh_sample(queue)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    assert queue.update(old) is False
    assert queue.baseline is None and queue.sample_needed.is_set()
    assert queue.update(fresh_sample(queue)) is True
    assert queue.baseline == SAMPLE["total_pss_bytes"]
    queue.close()


def test_untimed_sample_cannot_publish_baseline(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    assert queue.update(SAMPLE) is False
    assert queue.baseline is None and queue.sample_needed.is_set()
    queue.close()


def test_prebaseline_empty_then_request_invalidates_inflight_sample(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    assert clients[0].acquire(empty=True) is None
    wait_for(queue.sample_needed.is_set)
    old = fresh_sample(queue)
    old_generation = old["admission_sample_ticket"]["request_generation"]
    clients[0].connection.send(("request", 0.0))
    wait_for(lambda: queue.request_generation > old_generation)
    assert queue.sample_needed.is_set()
    assert queue.update(old) is False
    assert queue.baseline is None and queue.sample_needed.is_set()
    assert queue.update(fresh_sample(queue)) is True
    queue.close()


def test_live_set_change_invalidates_sample_and_wakes_without_false_failure(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 2, limit=1)
    assert clients[0].acquire(empty=True) is None
    assert clients[1].acquire(empty=True) is None
    wait_for(queue.sample_needed.is_set)
    old = fresh_sample(queue)
    clients[1].connection.close()
    wait_for(lambda: queue.connections[1] is None)
    assert queue.sample_needed.is_set() and not abort.is_set()
    assert queue.update(old) is False
    assert queue.update(fresh_sample(queue)) is True
    queue.close()


def test_unrequested_exit_completes_required_set_and_wakes_sampler(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 2, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(lambda: 0 in queue.seen)
    clients[1].connection.close()
    wait_for(queue.sample_needed.is_set)
    assert queue._live_ranks() == [0] and not abort.is_set()
    assert queue.update(fresh_sample(queue)) is True
    queue.close()


@pytest.mark.parametrize("invalid", ["too_old", "starts_before_ready", "future_end"])
def test_stale_or_invalid_sample_clock_is_not_accepted(tmp_path, invalid):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    sample = fresh_sample(queue)
    if invalid == "too_old":
        sample["sample_start_ns"] = 1
        sample["sample_end_ns"] = 2
    elif invalid == "starts_before_ready":
        sample["sample_start_ns"] = queue.all_request_ready_ns - 1
    else:
        sample["sample_end_ns"] = time.perf_counter_ns() + 10_000_000_000
    sample["admission_sample_ticket"]["sample_start_ns"] = sample["sample_start_ns"]
    assert queue.update(sample) is False and queue.baseline is None
    assert queue.sample_needed.is_set()
    queue.close()


def test_safety_check_without_publish_never_grants(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    sample = fresh_sample(queue)
    assert queue.check_memory(sample) is True
    assert queue.baseline is None and not queue.active
    queue.close()


def test_matching_generation_sample_can_expire_before_publication(tmp_path, monkeypatch):
    queue, clients, _ = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    old = fresh_sample(queue)
    monkeypatch.setattr(
        "metasl.admission.perf_counter_ns",
        lambda: old["sample_end_ns"] + queue.max_sample_age_ns + 1,
    )
    assert queue.update(old) is False
    assert queue.baseline is None and queue.sample_needed.is_set()
    queue.close()


@pytest.mark.parametrize(
    "field,value", [("total_pss_bytes", 20 * GIB), ("available_bytes", 7 * GIB - 1)]
)
def test_fresh_unsafe_sample_aborts_and_wakes_both_waiters(tmp_path, field, value):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    sample = fresh_sample(queue)
    sample[field] = value
    assert queue.update(sample) is False
    assert abort.is_set() and queue.baseline is None
    assert queue.sample_needed.is_set() and queue.work_available.is_set()
    queue.close()


def test_client_grant_ns_chain_and_existing_token_release_contract(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    assert clients[0].last_acquire_timing is None
    tokens = []
    thread = threading.Thread(target=lambda: tokens.append(clients[0].acquire()))
    thread.start()
    wait_for(queue.sample_needed.is_set)
    queue.update(fresh_sample(queue))
    thread.join(2)
    assert tokens == [0]
    t = clients[0].last_acquire_timing
    assert (
        t["request_send_ns"]
        <= t["server_receive_ns"]
        <= t["server_enqueue_ns"]
        <= t["all_request_ready_ns"]
        <= t["sample_start_ns"]
        <= t["sample_end_ns"]
        <= t["baseline_ready_ns"]
        <= t["grant_ns"]
        <= t["client_receive_ns"]
    )
    assert t["client_request_index"] == t["queue_round"] == 1
    assert not t["empty"]
    clients[0].release(0, time.perf_counter())
    wait_for(lambda: not queue.active)
    assert clients[0].acquire(empty=True) is None
    assert clients[0].last_acquire_timing["empty"]
    assert clients[0].last_acquire_timing["client_request_index"] == 2
    assert not abort.is_set()
    queue.close()
    events = [
        json.loads(line) for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()
    ]
    assert all(type(v["time_ns"]) is int for v in events)
    grant = next(v for v in events if v["event"] == "grant")
    assert grant["sample_request_generation"] == 1 and grant["rank"] == 0


def test_close_wakes_sampler_and_queue_without_unintended_abort(tmp_path):
    queue, _, abort = make_queue(tmp_path, 1, limit=1)
    queue.close()
    assert queue.sample_needed.is_set() and queue.work_available.is_set()
    assert not abort.is_set()


def test_stop_blocks_sample_publication_and_direct_grants(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("request", 0.0))
    wait_for(queue.sample_needed.is_set)
    sample = fresh_sample(queue)
    with queue.lock:
        queue.stop.set()
        assert queue.update(sample) is False
        assert queue._grant(0, 0) is False
    queue.close()
    events = [
        json.loads(line) for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()
    ]
    assert not any(v["event"] == "grant" for v in events)
    assert not abort.is_set()


def test_preparation_samples_before_requests_and_preserves_actual_fifo(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 2, limit=1)
    prepared = {}
    preparation_threads = [
        threading.Thread(target=lambda rank=rank: prepared.update({rank: clients[rank].prepare()}))
        for rank in range(2)
    ]
    preparation_threads[0].start()
    wait_for(lambda: 0 in queue.prepared)
    assert not queue.sample_needed.is_set()
    preparation_threads[1].start()
    wait_for(queue.sample_needed.is_set)
    assert not prepared and queue.baseline is None
    assert queue.request_generation == 0 and not queue.seen and not queue.pending
    sample = fresh_sample(queue)
    assert queue.update(sample) is True
    for thread in preparation_threads:
        thread.join(2)
    assert set(prepared) == {0, 1}
    assert not queue.active and not queue.pending
    assert all(client.request_index == 0 for client in clients)
    for timing in prepared.values():
        assert (
            timing["all_prepare_ready_ns"]
            <= timing["sample_start_ns"]
            <= timing["sample_end_ns"]
            <= timing["baseline_ready_ns"]
            <= timing["client_prepared_ns"]
        )
        assert timing["admission_baseline_before_timer"] is True

    tokens = {}
    # Real request order deliberately differs from preparation order.
    first = threading.Thread(target=lambda: tokens.update({1: clients[1].acquire()}))
    second = threading.Thread(target=lambda: tokens.update({0: clients[0].acquire()}))
    first.start()
    wait_for(lambda: 1 in queue.seen)
    assert not tokens and not queue.active  # still require every rank's first request
    second.start()
    first.join(2)
    assert tokens == {1: 1}
    clients[1].release(1, time.perf_counter())
    second.join(2)
    assert tokens == {1: 1, 0: 0}
    clients[0].release(0, time.perf_counter())
    wait_for(lambda: not queue.active)
    assert all(client.last_acquire_timing["client_request_index"] == 1 for client in clients)
    assert all(
        client.last_acquire_timing["sample_end_ns"]
        <= client.last_acquire_timing["request_send_ns"]
        for client in clients
    )
    assert not abort.is_set()
    queue.close()
    events = [json.loads(line) for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()]
    assert [event["rank"] for event in events if event["event"] == "grant"] == [1, 0]


def test_preparation_change_invalidates_inflight_sample(tmp_path):
    queue, clients, _ = make_queue(tmp_path, 2, limit=1)
    clients[0].connection.send(("prepare", None))
    wait_for(lambda: 0 in queue.prepared)
    early = fresh_sample(queue)
    clients[1].connection.send(("prepare", None))
    wait_for(queue.sample_needed.is_set)
    assert queue.update(early) is False
    assert queue.baseline is None and queue.sample_needed.is_set()
    assert queue.update(fresh_sample(queue)) is True
    assert queue.baseline_sample["sample_request_generation"] == 0
    queue.close()


def test_unsafe_preparation_never_acknowledges_or_grants(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("prepare", None))
    wait_for(queue.sample_needed.is_set)
    assert queue.update(fresh_sample(queue, available_bytes=0)) is False
    assert abort.is_set() and queue.baseline is None
    assert not clients[0].connection.poll() and not queue.active
    queue.close()


def test_preparation_rejects_io_before_safe_baseline(tmp_path):
    queue, clients, abort = make_queue(tmp_path, 1, limit=1)
    clients[0].connection.send(("prepare", None))
    wait_for(queue.sample_needed.is_set)
    clients[0].connection.send(("request", {}))
    wait_for(abort.is_set)
    assert "before admission preparation completed" in queue.error
    assert not queue.active and queue.baseline is None
    queue.close()


@pytest.mark.parametrize("ranks", [4, 8])
def test_world_size_limit_grants_every_rank_before_any_release_after_safe_preparation(tmp_path, ranks):
    queue, clients, abort = make_queue(tmp_path, ranks, limit=ranks)
    try:
        for client in clients:
            client.connection.send(("prepare", None))
        wait_for(queue.sample_needed.is_set)
        assert len(queue.prepared) == ranks and not queue.active and not queue.seen
        assert all(not client.connection.poll() for client in clients)
        sample = fresh_sample(queue)
        assert queue.update(sample) is True
        for client in clients:
            assert client.connection.poll(2)
            kind, timing = client.connection.recv()
            assert kind == "prepared" and timing["admission_baseline_before_timer"] is True
            assert timing["sample_end_ns"] <= timing["baseline_ready_ns"]

        for rank in range(ranks - 1):
            clients[rank].connection.send(("request", dict(
                request_send_ns=time.perf_counter_ns(), client_request_index=1)))
        wait_for(lambda: len(queue.seen) == ranks - 1)
        assert not queue.active  # the last live rank has not reported its real request
        clients[-1].connection.send(("request", dict(
            request_send_ns=time.perf_counter_ns(), client_request_index=1)))
        wait_for(lambda: len(queue.active) == ranks)
        assert queue.active == set(range(ranks)) and not abort.is_set()
        for rank, client in enumerate(clients):
            assert client.connection.poll(2)
            kind, grant = client.connection.recv()
            assert kind == "grant" and grant["token"] == rank
            timing = grant["timing"]
            assert timing["sample_end_ns"] <= timing["request_send_ns"] <= timing["grant_ns"]
        # All ranks hold a permit together; no release was needed to admit any rank.
        for rank, client in enumerate(clients):
            client.release(rank, time.perf_counter())
        wait_for(lambda: not queue.active)
        assert queue.error is None
    finally:
        queue.close()
    events = [json.loads(line) for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()]
    grants = [i for i, row in enumerate(events) if row["event"] == "grant"]
    releases = [i for i, row in enumerate(events) if row["event"] == "release"]
    assert len(grants) == len(releases) == ranks and max(grants) < min(releases)


@pytest.mark.parametrize("ranks", [4, 8])
@pytest.mark.parametrize("unsafe", [dict(available_bytes=0), dict(total_pss_bytes=30 * GIB)])
def test_world_size_limit_aborts_unsafe_preparation_without_acknowledgment_or_grants(
        tmp_path, ranks, unsafe):
    queue, clients, abort = make_queue(tmp_path, ranks, limit=ranks)
    try:
        for client in clients:
            client.connection.send(("prepare", None))
        wait_for(queue.sample_needed.is_set)
        assert queue.update(fresh_sample(queue, **unsafe)) is False
        assert abort.is_set() and queue.baseline is None and not queue.active
        assert all(not client.connection.poll() for client in clients)
        assert "soft memory threshold" in queue.error
    finally:
        queue.close()
    events = [json.loads(line) for line in (tmp_path / "writer-queue.jsonl").read_text().splitlines()]
    assert not any(row["event"] == "grant" for row in events)
