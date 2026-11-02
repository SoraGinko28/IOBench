"""Process and host memory/IO counters read from /proc; no torch dependency."""

import os
from pathlib import Path


def process_exited(pid):
    """/proc IO permissions can disappear before a worker is reaped."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        # Linux include/linux/sched.h: PF_EXITING=0x4. An exiting task can
        # lose /proc IO access while still reporting R, before becoming Z.
        return fields[0] in ("Z", "X", "x") or bool(int(fields[6]) & 0x4)
    except (FileNotFoundError, ProcessLookupError):
        return True


def memory(pid):
    try:
        fields = {}
        for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines():
            parts = line.split()
            if parts[0] in (
                "Pss:",
                "Rss:",
                "Pss_Anon:",
                "Pss_File:",
                "Pss_Shmem:",
                "Private_Clean:",
                "Private_Dirty:",
            ):
                fields[parts[0][:-1].lower()] = int(parts[1]) * 1024
        fields["private_bytes"] = fields.get("private_clean", 0) + fields.get("private_dirty", 0)
        return fields
    except (FileNotFoundError, ProcessLookupError):
        return {"pss": 0, "rss": 0}
    except PermissionError:
        if process_exited(pid):
            return {"pss": 0, "rss": 0}
        raise


def available_memory():
    return next(
        int(line.split()[1]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        if line.startswith("MemAvailable:")
    )


def system_memory():
    fields = {
        k.rstrip(":"): int(v) * 1024
        for k, v, *_ in (line.split() for line in Path("/proc/meminfo").read_text().splitlines())
    }
    return {
        k: fields[k]
        for k in (
            "MemAvailable",
            "Cached",
            "Buffers",
            "Shmem",
            "Dirty",
            "Writeback",
            "SReclaimable",
            "MemTotal",
        )
    }


def process_counters(pid):
    """Monotonic process counters for sampled curves (includes startup/runtime)."""
    try:
        io = {
            k: int(v)
            for k, v in (
                line.split(":") for line in Path(f"/proc/{pid}/io").read_text().splitlines()
            )
        }
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return dict(
            cpu_seconds=(int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK"),
            read_bytes=io["read_bytes"],
            write_bytes=io["write_bytes"],
            rchar=io["rchar"],
            wchar=io["wchar"],
        )
    except (FileNotFoundError, ProcessLookupError):
        return None
    except PermissionError:
        if process_exited(pid):
            return None
        raise
