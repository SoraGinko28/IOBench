# Meta pageable H2D simulation

`METASL_GDTENSOR_META_SIM_D2H=pageable` now selects the pageable GPU timing
model for both directions: save uses D2H, load uses H2D. The existing name and
CLI flag remain compatible. A second enable switch would allow accidentally
simulating only half of the intended pageable GPU workload. This mode models
single-rank pageable staging on the measured V100 host; it does not model pinned
or double-buffered transfers.

## Placement and memory lifecycle

For pageable GPU load, `ComputeAccessor._call_cuda(write=True)` creates a CPU
view of a reusable private anonymous mapping. gdtensor first completes the
range's payload reads (`pread` for safetensors, buffered `read` for DCP) and joins
the read bytes before invoking this
compute destination. The accessor's callback copies those bytes into its host
sink. Only then does
`target.copy_(staged, non_blocking=False)` execute, under one host-clock `h2d`
event. The blocking copy completes before `MADV_DONTNEED` discards the staging
pages and before the next range starts. There is no extra device-wide synchronize.
For save the blocking D2H precedes the callback's writes. Unbounded pinned
staging uses the same ordering, retaining its pinned pages; bounded double
buffering instead enqueues nonblocking copies and waits on per-slot CUDA events
before reuse or final return. It can overlap copies with subsequent host-sink
chunks; gdtensor has already read the whole range, and drains the copies before
the next range's payload reads. The pageable profile does not apply to those modes.

Meta load already writes every destination staging byte through the callback,
then discards the mapping after the range. It therefore pays the writable-page
first-touch costs on every range, as does pageable GPU. The added simulated H2D
sits immediately after this callback, before verification and discard. CUDA reads
already resident source pages; there are no new demand-zero source faults to
imitate at that point. Because the range is immediately discarded and all its
bytes will be rewritten before reuse, an additional fake source scan would not
preserve useful per-range source state. The simulation waits for the fitted
blocking copy duration without an extra scan. Effects on disk readahead and
writeback while that time elapses remain real. This is a timing model, not an
emulation of CUDA's internal bounce buffer, DMA or CPU cache transactions.

Every nonempty Meta load range records `op="h2d", timing="simulated"` with the
same tensor/range identity as its real GPU counterpart. The common compute
accessor supplies this behavior to both DCP and safetensors readers. Disabled
simulation, CPU and real GPU copies retain their existing execution paths.

## Profile

The built-in affine profile is fitted to 7,056 ranges from the retained 1B/3B/7B,
cold and warm, three-repeat single-file safetensors GPU traces and earlier local
1B pageable DCP traces. It uses 126.083 microseconds plus bytes divided by
4.581911e9 bytes/s. Whole repeats across all models, phases and formats are left
out for validation. Fresh DCP acceptance measurements are independent of fitting.
Fit coefficients, per-range
median/p95 absolute errors, large-size relative errors and input provenance are
in `validation/meta-h2d-sim-20261029/profile/`. No new acceptance measurements are
used to tune the default. `METASL_GDTENSOR_META_SIM_H2D_PROFILE=PATH` can override
the profile JSON without changing the enable switch. Size knots are supported
for future hardware but are unnecessary for this fit.
The observed range sizes span 512 bytes through 1,089,994,752 bytes. Accuracy
beyond that interval, on other hosts, or under other concurrency is unvalidated.

## Save-only native helper

Accessor construction and load never prepare or load the NT population helper.
`GDTensorSavePlanner` prepares the requested save strategy before the measured
checkpoint API. The helper is cached persistently in a user-private temporary
cache, keyed by its C source, platform ABI and compiler flags, then loaded once
per process. Direct accessor callers can call `prepare_save()` before starting
their timer; the save path retains lazy preparation for compatibility. No helper
compilation belongs to an `h2d` or `d2h` event. A cold direct save caller that
omits preparation can still pay initialization before its first D2H event.

Previous measured save calls already used planner preparation before their
API timer. Removing load initialization fixes their load-only pre-payload gap;
it does not justify subtracting that compilation cost from previous save times.
The new save measurements are an independent regression check, not a paired
causal estimate against the historical run.

## Reproduction

Run the targeted tests before measurements. GPU and multiprocessing phases
require escalated execution on this host. Never run measurements concurrently
with tests or another measurement runner. Use a fresh output directory:

```bash
CUDA_PY=/tmp/metasl-gdt-cuda-venv/bin/python
CPU_PY=/tmp/metasl-gdt-venv/bin/python
$CUDA_PY scripts/run_gdt_safetensors.py --stage measurements \
  --output validation/NEW-RUN --models 1b 3b 7b --repeats 3
$CUDA_PY scripts/run_gdt_safetensors.py --stage measurements \
  --output validation/NEW-RUN/dcp --checkpoint-format dcp \
  --models 1b --configs meta-sim gpu --repeats 3
$CPU_PY scripts/analyze_gdtensor_traces.py validation/NEW-RUN \
  --output validation/NEW-ANALYSIS
$CPU_PY scripts/plot_gdtensor_traces.py validation/NEW-ANALYSIS/analysis.json \
  --output validation/NEW-PLOTS
```

Reported save durations and save plot axes exclude the measured fsync interval
union; raw checkpoint timings retain it. Cold and warm
loads include the full checkpoint API call. Payload requests are compared exactly
across all configurations and repeats, including simulated loads. Checkpoints
are removed after each complete case and again checked at runner exit.

Refit from real GPU load traces with `scripts/fit_pageable_h2d.py --help`.
The [retained report](../validation/meta-h2d-sim-20261029/report-zh.md)
contains the single-rank baseline measurements and fitting provenance.
