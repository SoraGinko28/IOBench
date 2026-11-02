# gdtensor checkpoint IO bench

The bench prepares a transformer on Meta, applies PyTorch FSDP2, and keeps the
resulting DTensor geometry. Meta uses bounded anonymous scratch for each range;
CPU and CUDA materialize each rank's local shard. No training or forward pass is
performed. Only model state is checkpointed.

`metasl/gdt_storage.py` uses PyTorch DCP planning, rank filenames and metadata.
gdtensor performs DTensor payload movement through public `rank_save` and
`rank_load`. Sparse torch.save envelopes preserve the native tensor container;
CRC and serialization-ID fields are patched immediately after each payload.
Native PyTorch handling remains for small plain tensors and BYTE_IO metadata
inside these containers.

`metasl/gdt_hf_storage.py` uses the same compute and storage accessors with
safetensors extents and deterministic headers. The default is one safetensors
file per rank. A completion manifest makes incomplete saves detectable. Shared
storage base classes retain PyTorch planning, metadata broadcasts, admission and
publication; payload backends exposed by the CLI are always gdtensor.

All ranks use a Gloo process group for checkpoint coordination. CUDA checkpoint
targets reuse that group's metadata communication and copy local tensors through
pageable host staging. Explicit GPU sharing permits 4/8/16 checkpoint ranks on a
single GPU. Memory guards and writer/reader admission limit concurrent work.

Each phase writes rank JSON reports, a summary and exported traces. Recorded
layers distinguish file API requests, gdtensor payload requests, container patches
and staging transfers. Request comparisons retain operation, offset and length
per file and rank; staging durations are not request identities.

Cold loads evict checkpoint pages using file advice; warm loads read all data and
metadata and inspect residency before loading. Local disk and an already-mounted
NFS directory such as `/mnt/nfs-ckpt` use the same file backend. Nothing configures
NFS or clears global kernel caches.

Reported save totals and plots remove the union of fsync intervals. Load totals
retain the full checkpoint API duration. Per-rank durations and phase wall time
are separate measurements: transfer and IO durations can overlap across ranks.

See [D2H simulation](meta-d2h-simulation.md),
[H2D simulation](meta-h2d-simulation.md), and the [run examples](../README.md).
