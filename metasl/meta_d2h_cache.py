"""Optional cache-cold zero population for the Meta pageable D2H simulator.

The tiny x86 helper is cached by source hash and loaded once per process. Save
planners prepare it before the timed checkpoint call; load paths never need it.
No native code or compiler is used by memset or plain populate-write.
"""

import ctypes
import functools
import hashlib
import mmap
import os
import platform
import subprocess
import tempfile
from pathlib import Path

DEFAULT_POPULATION = "non-temporal"
POPULATIONS = ("memset", "memset-flush", "populate-write", "non-temporal", "populate-write-evict")


@functools.lru_cache(maxsize=1)
def _native_library():
    if platform.machine() != "x86_64":
        raise ValueError("Meta D2H native cache population requires x86_64")
    source = Path(__file__).with_name("meta_d2h_cache.c")
    flags = ("-O2", "-shared", "-fPIC")
    abi = (platform.machine(), ctypes.sizeof(ctypes.c_void_p), flags)
    key = hashlib.sha256(source.read_bytes() + repr(abi).encode()).hexdigest()
    directory = Path(tempfile.gettempdir()) / f"metasl-d2h-cache-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    library = directory / f"{key}.so"
    # Multiple ranks may prepare concurrently. A lock plus atomic publication
    # prevents either duplicate builds or loading a partially written library.
    import fcntl

    with (directory / f"{key}.lock").open("a+b") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not library.is_file():
            with tempfile.TemporaryDirectory(prefix="build-", dir=directory) as build_directory:
                built = Path(build_directory) / "cache.so"
                try:
                    subprocess.run(
                        ["cc", *flags, str(source), "-o", str(built)],
                        check=True,
                        capture_output=True,
                        text=True,
                    )
                except (OSError, subprocess.CalledProcessError) as error:
                    raise RuntimeError(
                        "Meta D2H native cache population needs a working C compiler (cc)"
                    ) from error
                os.replace(built, library)
    return ctypes.CDLL(str(library))


def _native_function(name, restype=None):
    function = getattr(_native_library(), name)
    function.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    function.restype = restype
    return function


@functools.lru_cache(maxsize=1)
def zero_and_flush():
    """Return zero fill followed by cache-line flushes and a fence."""
    return _native_function("metasl_zero_and_flush")


@functools.lru_cache(maxsize=1)
def zero_non_temporal():
    """Streaming SSE2 stores, fenced before the callback can consume the data."""
    return _native_function("metasl_zero_non_temporal")


@functools.lru_cache(maxsize=1)
def populate_write_evict():
    """Kernel prefault, then evict cache state within the same D2H budget.

    Near-LLC-sized ranges dominate the pwrite discrepancy. For ranges >= half
    the LLC, stream over twice its capacity. Smaller ranges only flush their
    source lines: a full eviction would exceed their short D2H time budget.
    """
    fill = populate_write()
    flush = _native_function("metasl_flush")
    read = _native_function("metasl_read_cachelines", ctypes.c_uint64)
    cache_size = Path("/sys/devices/system/cpu/cpu0/cache/index3/size").read_text().strip()
    if not cache_size.endswith("K"):
        raise ValueError(f"Unsupported LLC size format: {cache_size}")
    llc_size = int(cache_size[:-1]) * 1024
    size = 2 * llc_size
    buffer = mmap.mmap(-1, size, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    pointer = ctypes.addressof(ctypes.c_char.from_buffer(buffer))
    ctypes.memset(pointer, 0, size)  # Resident private pages, prepared before DCP timing.

    def populate(address, length):
        fill(address, length)
        if length >= llc_size // 2:
            read(pointer, len(buffer))  # Closure retains ownership of the eviction buffer.
        else:
            flush(address, length)

    return populate


def population_function(population):
    """Prepare only the requested strategy; callers do this before timing DCP."""
    validate_population(population)
    if population == "memset":
        return None
    factories = {
        "memset-flush": zero_and_flush,
        "populate-write": populate_write,
        "non-temporal": zero_non_temporal,
        "populate-write-evict": populate_write_evict,
    }
    return factories[population]()


def validate_population(population):
    """Validate configuration without compiling or loading any native helper."""
    if population not in POPULATIONS:
        raise ValueError(f"Unknown Meta D2H population: {population}")
    return population


@functools.lru_cache(maxsize=1)
def populate_write():
    """Prefault private writable pages with Linux MADV_POPULATE_WRITE (5.14+).

    The caller owns a fresh/MADV_DONTNEED private anonymous mapping, so populated
    pages are zero. Unlike memset this must never be used to clear existing data.
    """
    madvise = ctypes.CDLL(None, use_errno=True).madvise
    madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    madvise.restype = ctypes.c_int

    def populate(address, size):
        if madvise(address, size, 23):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))

    return populate
