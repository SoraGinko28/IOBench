Identical except the native DCP nondeterministic `StorageMeta.save_id`.

Real CPU BF16 tensors; the runner's layer/root FSDP2 Shard(0) wrapping and local configs (Qwen/Qwen2.5-3B, TinyLlama/TinyLlama-1.1B-Chat-v1.0). Each FQN's canonical tied-alias name seeds a CPU Generator with the little-endian BLAKE2b-64 digest; full `torch.randn` tensors are generated directly in model dtype, then sliced into rank shards. Full contiguous logical bytes are gathered to rank 0 and hashed with BLAKE2b-256.

Native baseline: plain `dcp.save` and `FileSystemWriter(path)` with all defaults: `thread_count=1`, `single_file_per_rank=True`, `sync_files=True`, `per_thread_copy_ahead=10000000`, `cache_staged_state_dict=False`, `overwrite=True`, TORCH_SAVE serialization, with no stream extensions. These mirror the backend's documented one thread/file per rank and native planning/ZIP layout. Both writers use their respective default save planners.

Native A, native B, and gdtensor reuse the exact same path sequentially. Each checkpoint is loaded and validated, streamed through BLAKE2b-256, and immediately removed. Only small ZIP envelopes, metadata, member hashes, and tensor hashes remain. No timings are measured. Every checkpoint is loaded using native `dcp.load`; native A is additionally loaded using the gdtensor reader. Destinations are poisoned with NaNs before each load.

| Model | Ranks | Files | Raw A/B | Raw A/G | Raw B/G | Normalized | Logical hashes |
|---|---:|---:|---:|---:|---:|---|---|
| 1b | 1 | 2 | 1/2 | 1/2 | 1/2 | True | True |
| 1b | 4 | 5 | 4/5 | 4/5 | 4/5 | True | True |
| 1b | 8 | 9 | 8/9 | 8/9 | 8/9 | True | True |
| 3b | 1 | 2 | 1/2 | 1/2 | 1/2 | True | True |
| 3b | 4 | 5 | 4/5 | 4/5 | 4/5 | True | True |
| 3b | 8 | 9 | 8/9 | 8/9 | 8/9 | True | True |

Full logical tensor hashes also match across all tested rank counts for each model.

The native A/B control differs only in `.metadata` → `storage_meta.save_id` (a fresh UUID). The entire 36-byte UUID value is masked only after that cell's control demonstrates nondeterminism. Exact differing byte runs (half-open file offsets and hex values), all metadata fields, and field-by-field diffs are in each cell's `evidence.json`. All `.distcp` ZIP bytes, including tensor payloads, CRCs, member names, headers and `.data/serialization_id`, are compared without normalization. No additional backend differences were found.

Each cell's `file_table` stores size, raw BLAKE2b-256, normalized BLAKE2b-256, and exact UUID mask spans for every file in all three writes. ZIP tensor-storage members have independent hashes; equal hashes prove excluded payload regions contain no differing byte ranges. Retained non-payload bytes allow exact envelope diffs after deletion. Persisted metadata is unpickled afresh after loading, independently of the reader's in-memory load id. Torch's installed `distributed/checkpoint/filesystem.py` generates `save_id` using `uuid.uuid4()`; the backend inherits this behavior.

Scope: model-state CPU BF16 checkpoints on torch 2.14.0+cpu with these configs and pure FSDP layout (1b: 201 tensors, 3b: 435 tensors). All 24 load checks match the generated full logical tensor hashes. No backend-caused byte differences were found.

Reusable validator: `scripts/validate_dcp_identity.py` (use `--output` with an empty directory; `--resume` continues this audit's interrupted output; `--tiny --models 1b --ranks 2` runs the small integration probe). Helper tests: `python -m pytest tests/test_dcp_identity_helpers.py`. All checkpoint directories were removed; prior validation directories and the gdtensor library were untouched.
