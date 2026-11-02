"""Observe guest page residency without reading checkpoint contents or dropping caches."""

import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from metasl.iostats import DATA_SUFFIXES, checkpoint_files


def memory_evidence():
    fields = {"MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached", "Dirty", "Writeback"}
    return {
        key: int(value.split()[0]) * 1024
        for line in Path("/proc/meminfo").read_text().splitlines()
        for key, value in [line.split(":", 1)]
        if key in fields
    }


def checkpoint_residency(checkpoint):
    """fincore uses mincore, so this probe does not fault file data into RAM."""
    checkpoint = Path(checkpoint)
    files = checkpoint_files(checkpoint)
    command = ["fincore", "--json", "--bytes", "--output", "PAGES,SIZE,RES,FILE"]
    rows = []
    if files:
        command += [str(file.resolve()) for file in files]
        result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=60)
        raw = json.loads(result.stdout)["fincore"]
        if len(raw) != len(files):
            raise RuntimeError("fincore did not report every checkpoint file")
        page_size = os.sysconf("SC_PAGE_SIZE")
        for file, row in zip(files, raw, strict=True):
            if Path(row["file"]).resolve() != file.resolve():
                raise RuntimeError("fincore file order differs from checkpoint inventory")
            size = int(row["size"])
            pages = (size + page_size - 1) // page_size
            resident = int(row["pages"])
            rows.append(dict(
                path=file.relative_to(checkpoint).as_posix(), bytes=size,
                pages=pages, resident_pages=resident, resident_bytes=int(row["res"]),
                resident_percent=100 * resident / pages if pages else 0.0,
                data_file=file.name.endswith(DATA_SUFFIXES),
            ))
    data = [row for row in rows if row["data_file"]]
    data_pages = sum(row["pages"] for row in data)
    data_resident_pages = sum(row["resident_pages"] for row in data)
    return dict(
        captured_at=datetime.now(UTC).isoformat(), method="fincore/mincore",
        checkpoint=str(checkpoint.resolve()), directory_exists=checkpoint.exists(),
        empty=not files, files=rows, command=command if files else None,
        near_zero_data_residency=(
            all(row["resident_percent"] <= 0.1 for row in data) if data else None
        ),
        near_zero_threshold_percent=0.1, memory_bytes=memory_evidence(),
        data_resident_percent=100 * data_resident_pages / data_pages if data_pages else None,
        fully_warm_data_residency=(
            all(row["resident_percent"] >= 99 for row in data) if data_pages else False
        ),
        warm_threshold_percent=99,
        limitation="Guest page cache only; VM host-side caching is neither controlled nor observed",
    )
