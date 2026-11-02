# Core runtime changes

The exact pre-removal subprocess flags are recorded in [core-cli-options.md](core-cli-options.md).

## Removed internal paths

- Moved the standalone worker launcher into `metasl.run` and removed `supervisor.py`. Removed train/resume/retention/wall-clock schedules, interleaved plans, training-state/AdamW/RNG/byte audits, hybrid model+state format, random initialization/audits, native IO selection, derived placements, tensor/pipeline parallel runtime and external block-capture window hooks.
- Retained the original pure-FSDP mesh dimension ordering `(pp=1, dp_replicate=1, dp_shard=N, tp=1)` and Meta model construction; CPU/CUDA still materialize only local shards. The launch process remains torch-free before forkserver setup.
- `phase.py` now selects only gdtensor planners and the DCP/safetensors writers. Kept call timing, CUDA synchronization, metadata admission/pruning/broadcast, actual-read coverage checks, value validation, trace capture/export, publication/file observations and memory guards.
- Deleted newly unreachable modules: `checkpoint_retention`, `continuous_io`, `gdt_diagnostics`, `hybrid_state_io`, `interleaved`, `io_window`, `random_weights`, `timeline`, `train`, `training_audit`, `training_state`.
- Removed native scratch pools/save/load/audit planners; kept only strict load skeletons imported by gdtensor. Removed global generalized-layout audit and native derived target builder; kept metadata, extent and local coverage audits used by the backend.
- Removed deferred training trace finalization/snapshot/component merging from iostats; kept the metadata event packing needed for heap release. Removed native safetensors header stream proxy (gdtensor already emits canonical headers).

## Runner changes

- Matrix runner: removed continuous IO sampler and its flags; uses the consolidated analysis/plot entry points in source inventory; default offline configs are `configs/`.
- Safetensors runner: now a common DCP/safetensors serial measurement runner, one payload per rank, preserving save/cold/warm phases. Removed native-vs-gdtensor identity gates. Added `--ranks`/`--nproc`, per-rank trace export/validation, `--bundle`/`--bundles` overrides, offline missing-bundle preparation, `--model-configs` and `--checkpoint-root` (including NFS targets). Default 1B/config list includes Meta, Meta+sim, CPU and GPU.
- Kept optional safetensors splitting because file bucketing is small and shared; default stays one file per rank. Existing `phase_command` signature and core subprocess switches stay compatible. Removed old one-rank 7B CPU skip because multi-rank shards may fit; normal memory guards enforce actual residency.

## Deliberately retained

- `admission.py`, worker PSS/time guards and metadata gate: both core runners actually set writer/reader/metadata limits; dropping these would alter multi-rank save ordering and memory behavior.
- `storage.py` and HuggingFace shared reader/writer bases: gdtensor inherits their admission, completion, metadata validation and manifest logic. Native backend is no longer selectable.
- `prepare.py` and `preload.py`: initial import scan missed subprocess/forkserver reachability. The matrix invokes preparation; forkserver preloading keeps multi-rank imports inexpensive.
- Profile/cache modules and trace export remain for Meta D2H/H2D simulation and acceptance comparisons.

## Line counts for large touched entry points

| File | Before | After |
|---|---:|---:|
| metasl/phase.py | 1662 | 868 |
| metasl/supervisor.py | 1285 | 0 |
| metasl/scratch.py | 745 | 56 |
| scripts/run_gdtensor_trace_matrix.py | 1255 | 1179 |
| scripts/run_gdt_safetensors.py | 399 | 248 |
| metasl/run.py | 7 | 591 |

## Validation performed by core agent

- Import every kept `metasl/*.py` module using `/tmp/metasl-gdt-venv/bin/python`: passed.
- `--help` for metasl.run, metasl.prepare and both core runner scripts: passed; captured in `core-help.txt`.
- Construct shared save phase command with four ranks and parse it through the narrowed CLI: passed.
- Read-only review retained the strict model-only DCP metadata scope check in the lean loader; removed training schema machinery.
- Parent coordinates the full pytest suite and separate acceptance agent runs all 1B behavior and four-rank gates. No checkpoints created by this agent.

## Kept runtime modules

`__init__.py`
`admission.py`
`cache_control.py`
`checkpoint_state.py`
`fsdp.py`
`gdt_hf_storage.py`
`gdt_options.py`
`gdt_storage.py`
`hf_storage.py`
`iostats.py`
`layout.py`
`meta_d2h_cache.py`
`meta_d2h_profile.py`
`meta_h2d_profile.py`
`phase.py`
`preload.py`
`prepare.py`
`procmem.py`
`reference.py`
`run.py`
`scratch.py`
`sharding.py`
`skeleton.py`
`storage.py`
`trace_export.py`

Read-only review: the shared FSDP mesh and gdtensor planner/writer ordering remain unchanged for pure dp_shard=1/4/8/16; both runners pass the requested world size and export/validate each rank. No hard-coded single-rank check remains in the measurement runner. The native safetensors oracle test normalizes header metadata only in memory; native files remain independent cross-load controls.
