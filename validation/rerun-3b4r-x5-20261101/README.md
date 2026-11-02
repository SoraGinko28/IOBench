# 3B · 4 ranks · shared safetensors · save, 5 reruns (Meta ring 64 MiB vs CPU)

Driver: `rerun_3b4r_x5.py` (save only, paired/interleaved). Plot: `rerun-x5-3b-4r-save.png`.

| Round | Save total CPU / Meta (s) | Writes end CPU / Meta (s) | Stalled pwrite per rank CPU / Meta (s) |
|---|---|---|---|
| 1 | 38.65 / 38.84 | 17.8 / 13.2 | 8.3 / 2.9 |
| 2 | 38.61 / 39.32 | 18.8 / 13.7 | 10.1 / 4.8 |
| 3 | 38.71 / 39.35 | 17.3 / 14.2 | 8.6 / 4.0 |
| 4 | 38.63 / 39.31 | 18.4 / 13.7 | 9.7 / 4.6 |
| 5 | 38.65 / 39.27 | 17.0 / 14.4 | 8.7 / 4.4 |

Per-rank request sequences are identical across all 10 runs. Meta finishes writing ~4 s earlier because
its smaller memory footprint raises Linux's dirty-page threshold; see `../meta-early-write-end-20261102/`.
Raw traces are not committed.
