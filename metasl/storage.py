"""Default FileSystemWriter/Reader with permit admission around the local data call.

All serialization and file IO stay in torch's classes; only the local
write_data/read_data call is admitted, observed and timed.
"""

import collections
import ctypes
import gc
import io
import os
import pickle
import pickletools
from contextlib import contextmanager
from time import perf_counter, perf_counter_ns

import torch
import torch.distributed as dist
from torch.distributed import distributed_c10d as c10d
from torch.distributed.checkpoint import FileSystemReader, FileSystemWriter, state_dict_saver
from torch.distributed.checkpoint.metadata import Metadata
from torch.distributed.checkpoint.planner import LoadItemType

from metasl.procmem import memory
from metasl.sharding import audit_local, prune_metadata, read_record


def index_key(index):
    return index.fqn, tuple(index.offset or ())


class AbortOnReaderFailure:
    """Instance filesystem wrapper: propagate errors before DCP's final collective.

    The underlying reader and every file operation remain unchanged. Exceptions
    raised anywhere in its file context (including torch.load/copy_) notify ranks
    waiting for a reader permit, so all ranks can reach DCP's error collective.
    """

    def __init__(self, fs, abort):
        self.fs, self.abort = fs, abort

    def __getattr__(self, name):
        return getattr(self.fs, name)

    @contextmanager
    def create_stream(self, path, mode):
        try:
            with self.fs.create_stream(path, mode) as stream:
                yield stream
        except BaseException:
            # Metadata errors belong to DCP's earlier plan/error collective;
            # only a data-file failure can strand peers waiting for a permit.
            if str(path).endswith(".distcp"):
                self.abort.set()
            raise


class AdmittedFileSystemWriter(FileSystemWriter):
    """Planning is already global when DCP calls write_data.

    Waiting for the returned Future here does not wait for DCP's later
    all-reduce or metadata commit.
    """

    def __init__(self, path, *, client=None, abort=None, status_path=None, **kwargs):
        super().__init__(path, **kwargs)
        self.client, self.abort, self.status_path = client, abort, status_path
        self.timing = {}
        self.completed_write_indices = []
        self.results = None

    def phase(self, name):
        if self.status_path is not None:
            self.status_path.write_text(name)

    def write_data(self, plan, planner):
        self.phase("writer_queue" if self.client else "writing")
        self.timing["queue_enter"] = perf_counter()
        self.timing["queue_enter_ns"] = perf_counter_ns()
        token = None
        try:
            if self.client:
                token = self.client.acquire(empty=not plan.items)
                self.timing["admission"] = dict(
                    getattr(self.client, "last_acquire_timing", None) or {}
                )
            self.timing["write_enter"] = perf_counter()
            self.timing["write_enter_ns"] = perf_counter_ns()
            self.timing["permit"] = token
            self.timing["empty_plan"] = not bool(plan.items)
            self.phase("writing")
            future = super().write_data(plan, planner)
            results = future.wait()
            self.timing["future_completed"] = perf_counter()
            self.timing["future_completed_ns"] = perf_counter_ns()
            # A default writer background thread can fail without forwarding its
            # exception through the Future. Never accept partial WriteResults.
            expected = collections.Counter(index_key(i.index) for i in plan.items)
            actual = collections.Counter(index_key(r.index) for r in results)
            if actual != expected:
                raise RuntimeError("Incomplete or duplicated local WriteResults")
            self.results = list(results)
            self.completed_write_indices = [
                dict(fqn=r.index.fqn, offsets=list(r.index.offset or ())) for r in results
            ]
            if self.client and token is not None:
                self.client.release(token, self.timing["future_completed"])
            self.timing["release_sent"] = perf_counter()
            self.timing["release_sent_ns"] = perf_counter_ns()
            self.phase("final_sync")
            return future
        except BaseException:
            # Tracebacks can still own tensors; abort instead of reusing a permit.
            if self.abort is not None:
                self.abort.set()
            if self.client:
                self.client.fail()
            raise


METADATA_CLASS = ("torch.distributed.checkpoint.metadata", "Metadata")
STRING_OPS = frozenset(
    (
        "SHORT_BINUNICODE",
        "BINUNICODE",
        "BINUNICODE8",
        "UNICODE",
        "STRING",
        "BINSTRING",
        "SHORT_BINSTRING",
    )
)


def top_level_class(buffer):
    """(module, name) of the first class a pickle stream references, without unpickling."""
    strings = []
    for op, arg, _ in pickletools.genops(io.BytesIO(buffer)):
        if op.name == "GLOBAL":
            return tuple(arg.split(" ", 1))
        if op.name == "STACK_GLOBAL":
            return tuple(strings[-2:])
        if op.name in STRING_OPS:
            strings.append(arg)
    return None


def local_metadata(results):
    """The Metadata view a non-coordinator rank needs after a save: its own entries.

    DCP broadcasts rank 0's global Metadata to every rank at the end of a save
    and the caller only uses it to look up where its own items landed; those
    _StorageInfo records are exactly this rank's WriteResults.
    """
    return Metadata(
        state_dict_metadata={},
        planner_data=None,
        storage_data={r.index: r.storage_data for r in results},
        storage_meta=None,
        version=None,
    )


@contextmanager
def local_metadata_broadcast(writer, stats):
    """Run dcp.save with DCP's final Metadata broadcast received but not unpickled.

    Wraps DCP's ``_DistWrapper.broadcast_object`` for the duration of the call.
    Rank 0 pickles and broadcasts the object exactly as ``broadcast_object_list``
    does (one sizes tensor, one payload tensor), so the collective traffic and
    the ``.metadata`` file are unchanged. Every other rank receives the payload,
    reads only its top-level class name, and returns ``local_metadata`` built
    from its own WriteResults instead of unpickling a global object that grows
    with the rank count (85 MB + 2.75 KB per chunk on every rank). Anything
    that is not a Metadata (DCP's CheckpointException) is unpickled as before.
    """
    original = state_dict_saver._DistWrapper

    class LocalMetadataWrapper(original):
        def broadcast_object(self, object):
            if not self.use_dist:
                stats.update(full=True, collective=False)
                return super().broadcast_object(object)
            device = c10d._get_object_coll_device(self.group)
            src = self.global_coordinator_rank
            stats["broadcast_start"] = perf_counter()
            if self.is_coordinator:
                payload, sizes = c10d._object_to_tensor(object, device, self.group)
            else:
                sizes = torch.empty(1, dtype=torch.long, device=device)
            dist.broadcast(sizes, src=src, group=self.group)
            size = int(sizes.item())
            if not self.is_coordinator:
                payload = torch.empty(size, dtype=torch.uint8, device=device)
            dist.broadcast(payload, src=src, group=self.group)
            stats.update(
                broadcast_end=perf_counter(),
                payload_bytes=size,
                full=self.is_coordinator,
                collective=True,
            )
            if self.is_coordinator:
                return object
            kind = top_level_class(payload.numpy().data)
            if kind != METADATA_CLASS or writer.results is None:
                stats["unpickled"] = True
                return c10d._tensor_to_object(payload, size, self.group)
            stats["unpickled"] = False
            view = local_metadata(writer.results)
            stats.update(local_entries=len(view.storage_data), pss_after=memory(os.getpid())["pss"])
            return view

    state_dict_saver._DistWrapper = LocalMetadataWrapper
    try:
        yield stats
    finally:
        state_dict_saver._DistWrapper = original


def release_heap():
    """Return freed arenas to the OS so the pruned Metadata's saving shows in PSS."""
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def rebuild(blob):
    """Unpickle `blob` into fresh arenas once every other Python object is released.

    The pruned copy is built while the full Metadata is alive, so its small
    objects land in the same 1 MiB pymalloc arenas; freeing the original then
    leaves most arenas pinned by a few survivors (measured: 118 MB of 130 MB
    retained for a 71k-chunk file). Callers pickle the survivors, drop every
    reference (including their own), and call this to trim and rebuild.
    """
    release_heap()
    obj = pickle.loads(blob)
    del blob
    return obj


class AdmittedFileSystemReader(FileSystemReader):
    """Observe the default reader's plan, then admit its complete local read_data.

    read_metadata keeps torch's file read and unpickling, then hands DCP a copy
    holding only the chunks this rank's local shards can intersect (see
    ``prune_metadata``). ``gate`` bounds how many ranks unpickle at once: the
    full object is 100x the pruned one, so on a host running every rank the
    peak is set by ``gate`` rather than by the rank count.
    """

    def __init__(self, path, *, client=None, abort=None, status_path=None, rows=None, gate=None):
        super().__init__(path)
        self.client, self.abort, self.status_path, self.rows = client, abort, status_path, rows
        self.gate = gate
        self.timing = {}
        self.requests = []
        self.byte_items = 0

    def phase(self, name):
        if self.status_path is not None:
            self.status_path.write_text(name)

    def acquire_gate(self):
        if self.gate is None:
            return
        while True:
            if self.abort is not None and self.abort.is_set():
                raise RuntimeError("Metadata gate aborted by peer or supervisor")
            if self.gate.acquire(timeout=0.1):
                return

    def read_metadata(self, *args, **kwargs):
        stats = dict(gate=self.gate is not None, pruned=self.rows is not None)
        self.phase("metadata_queue" if self.gate is not None else "metadata")
        stats["queue_enter"] = perf_counter()
        self.acquire_gate()
        stats["read_start"] = perf_counter()
        stats["pss_before_read"] = memory(os.getpid())["pss"]
        try:
            self.phase("metadata")
            # Unchanged torch code path: same file open, same reads, same unpickling.
            metadata = super().read_metadata(*args, **kwargs)
            stats["read_end"] = perf_counter()
            if self.rows is None:
                return metadata
            stats["pss_before_prune"] = memory(os.getpid())["pss"]
            pruned, counts = prune_metadata(metadata, self.rows)
            stats.update(counts)
            blob = pickle.dumps(pruned, protocol=pickle.HIGHEST_PROTOCOL)
            del metadata, pruned  # no reference may survive before rebuild() trims
            metadata = rebuild(blob)
            del blob
            stats["prune_end"] = perf_counter()
            stats["pss_after_prune"] = memory(os.getpid())["pss"]
            return metadata
        finally:
            if self.gate is not None:
                self.gate.release()
            self.timing["metadata"] = stats

    # Called right after planner.create_local_plan and before the plan collective.
    def prepare_local_plan(self, plan):
        self.timing["local_plan"] = perf_counter()
        self.timing["pss_local_plan"] = memory(os.getpid())["pss"]
        return super().prepare_local_plan(plan)

    # state_dict_loader.py invokes this AFTER planning and BEFORE the final collective.
    def read_data(self, plan, planner):
        self.timing["pss_reader_enter"] = memory(os.getpid())["pss"]
        self.requests = [
            read_record(i, planner.metadata) for i in plan.items if i.type != LoadItemType.BYTE_IO
        ]
        self.byte_items = sum(i.type == LoadItemType.BYTE_IO for i in plan.items)
        if self.rows is not None:
            audit_local(self.rows, self.requests)
        self.phase("reader_queue" if self.client else "reading")
        self.timing["queue_enter"] = perf_counter()
        self.timing["queue_enter_ns"] = perf_counter_ns()
        token = None
        try:
            if self.client:
                token = self.client.acquire(empty=not plan.items)
                self.timing["admission"] = dict(
                    getattr(self.client, "last_acquire_timing", None) or {}
                )
            self.timing["reader_enter"] = perf_counter()
            self.timing["reader_enter_ns"] = perf_counter_ns()
            self.timing["permit"] = token
            planner.reader_enter_time = self.timing["reader_enter"]
            self.phase("reading")
            future = super().read_data(plan, planner)
            future.wait()
            # Default reader's stack, including the final source/target, is gone here.
            self.timing["reader_return"] = perf_counter()
            self.timing["reader_return_ns"] = perf_counter_ns()
            if self.client and token is not None:
                self.client.release(token, self.timing["reader_return"])
            self.timing["release_sent"] = perf_counter()
            self.timing["release_sent_ns"] = perf_counter_ns()
            self.phase("final_sync")
            return future
        except BaseException:
            if self.abort is not None:
                self.abort.set()
            if self.client:
                self.client.fail()
            raise
