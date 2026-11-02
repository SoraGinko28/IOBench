# IOBench: checkpoint IO benchmark with Meta-device ranks

IOBench runs FSDP2 ranks on PyTorch's `meta` device and saves/loads checkpoints through the
[gdtensor](https://github.com/cklxx/gdtensor) IO backend. Meta ranks hold no real weights, yet issue
the same file IO requests as real CPU/GPU ranks, so one host can simulate many ranks to stress storage.

Key results (1B/3B, local disk):
- **Same checkpoint**: DCP checkpoints written through gdtensor are byte-identical to native
  `torch.distributed.checkpoint` (1/4/8 ranks; only torch's random `save_id` differs).
  See [`validation/dcp-gdtensor-identity-20261030/`](validation/dcp-gdtensor-identity-20261030/).
- **Same IO trace**: Meta and CPU issue identical per-rank request sequences; total save/load time
  within 3% (4/8 ranks). Meta needs only a fixed 64 MiB host buffer per rank.
  See [`validation/meta-bounded-mem-20261030/`](validation/meta-bounded-mem-20261030/).

## Layout

- `metasl/` — FSDP skeleton, gdtensor storage adapters, IO tracing, Meta D2H/H2D simulation.
- `scripts/` — runners, analysis and plotting.
- `configs/{1b,3b,7b}/` — offline model configs (no weights). `tests/` — regression tests.
- `docs/` — design notes. `validation/` — reports and plots from past runs (raw traces not included).

## Setup

Python 3.12+, PyTorch 2.14.0, transformers 5.17.0, safetensors 0.8.0, and a local gdtensor checkout:

```bash
pip install -r requirements.txt            # or requirements-cuda-v100.txt for GPU
pip install -e ../gdtensor
```

## Run

Save, cold load and warm load for each device config, then clean up the checkpoint:

```bash
python scripts/run_gdt_safetensors.py --output validation/NEW-RUN \
  --models 1b --ranks 4 --repeats 1 --checkpoint-format safetensors \
  --configs meta cpu
```

- `--checkpoint-format dcp|safetensors`; `--configs meta meta-sim cpu gpu`; `--ranks 1|4|8|16`.
- `--safetensors-layout shared`: all ranks write their shards directly into one standard HF
  `model.safetensors` (no per-rank files, no consolidation). Add `--safetensors-max-file-gib N` to split.
- `--meta-load-sink ring` (default, 64 MiB/rank, set with `--meta-load-sink-mib`) or `faithful`
  (full-shard buffers, same memory as CPU).
- `--checkpoint-root DIR` writes checkpoints elsewhere (e.g. an NFS mount); `--save-only` skips loads.
- `--output` must be a new directory.

Analyze and plot any run directory:

```bash
python scripts/analyze_gdtensor_traces.py validation/NEW-RUN --output validation/NEW-ANALYSIS
python scripts/plot_gdtensor_traces.py validation/NEW-ANALYSIS/analysis.json --output validation/NEW-PLOTS
```

Check that gdtensor writes the same DCP checkpoint as native torch (random weights, BLAKE2b-256):

```bash
python scripts/validate_dcp_identity.py --output validation/NEW-IDENTITY --models 1b 3b --ranks 1 4 8
```

## Notes

- Meta `meta-sim` simulates pageable D2H on save and H2D on load (`METASL_GDTENSOR_META_SIM_D2H=pageable`);
  see [`docs/meta-d2h-simulation.md`](docs/meta-d2h-simulation.md) and [`docs/meta-h2d-simulation.md`](docs/meta-h2d-simulation.md).
- On large saves, Meta can finish its `pwrite` phase earlier than CPU: holding less memory gives it a
  higher Linux dirty-page limit. Total time is unchanged; see
  [`validation/meta-early-write-end-20261102/`](validation/meta-early-write-end-20261102/).
- Design: [`docs/gdtensor-design.md`](docs/gdtensor-design.md).
