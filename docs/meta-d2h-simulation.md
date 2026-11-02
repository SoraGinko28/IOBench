# Meta pageable D2H simulation

This optional gdtensor save mode models one blocking pageable GPU D2H transfer
for each nonempty Meta zero/default range. It is disabled by default:

```bash
METASL_GDTENSOR_META_SIM_D2H=pageable
```

`METASL_GDTENSOR_META_SIM_D2H_PROFILE` selects a JSON object with `a_seconds`,
`bandwidth_bytes_per_s`, and optional `knots`. The built-in V100 profile preserves
0.00021 seconds + bytes / 1.6e9 through 593.5 MiB, then interpolates calibrated
duration knots through 2 GiB. This default applies only when pageable Meta D2H
simulation is enabled. Select the previous single-line model explicitly with:

```bash
METASL_GDTENSOR_META_SIM_D2H_PROFILE=legacy
```

Existing affine JSON files remain supported. Optional `knots` is a list of
`{"bytes": integer, "seconds": number}` objects with strictly increasing positive
sizes and durations. The first knot must meet the affine branch continuously.
Sizes through that knot use the affine formula; larger sizes interpolate duration
linearly in bytes. Beyond the last knot the final segment is extrapolated, without
an accuracy guarantee. The built-in profile is specific to the measured V100 and
host; other hardware can select a custom profile or `legacy`.

`METASL_GDTENSOR_META_SIM_D2H_POPULATE` selects how the writable private anonymous
staging range is populated:

| Value | Behavior |
|---|---|
| `non-temporal` (default when simulation is enabled) | SSE2 streaming zero stores followed by SFENCE. |
| `memset` | Previous implementation: zero fill with `ctypes.memset`. |
| `memset-flush` | Zero fill, flush every intersecting 64-byte CPU cache line, then fence. |
| `populate-write` | Linux `MADV_POPULATE_WRITE` on fresh/discarded anonymous pages. Unsupported kernels raise an error. |
| `populate-write-evict` | Kernel population; source-line flush below half LLC size, otherwise read twice the LLC capacity from a prepared resident buffer. |

The non-temporal, flush and eviction variants require x86_64 and a working `cc`.
Their small C helper is
cached in a user-private temporary cache keyed by source and platform ABI, and
loaded once per process on save-only paths. Cache-line flushing selects
CLFLUSHOPT when supported, otherwise
CLFLUSH. The normal save planner prepares the helper and any eviction buffer
before the DCP API timer.
Direct accessor users call `prepare_save()` before starting their save timer;
construction and load never build or load the helper. There is no
compiler use when simulation is off, or `memset` / plain `populate-write` is selected.

Each range emits exactly one staging event with `op="d2h"`,
`timing="simulated"`, and `population` identifying the selected method.
Population, cache flushing, and its fence all consume the same
selected profile's duration budget. Only the remainder is
waited; overruns are recorded without adding another delay. The callback is
unchanged, and the entire staging mapping is discarded with `MADV_DONTNEED`
after each range, including callback failures.

The disabled save path retains the existing read-only demand-zero mapping and
does not inspect the population/profile settings. CPU and CUDA data paths retain
their existing behavior. The same enable switch now adds fitted pageable H2D
timing to Meta loads; see [Meta H2D simulation](meta-h2d-simulation.md).
Position-pattern Meta saves are
unsupported when simulation is enabled.

## Refit and evidence

Refit the current knot profile from real GPU save traces:

```bash
python scripts/fit_pageable_d2h_knots.py --help
```

Use the unified `scripts/analyze_gdtensor_traces.py` and
`scripts/plot_gdtensor_traces.py` for request comparisons and timing plots.
Save totals and axes exclude the measured fsync interval union. Container CRC
and serialization-ID patches always run immediately when a payload completes;
empty payloads use the commit fallback.

The retained reports describe the measured V100 host and single-rank results:

- [Early container patches](../validation/gdt-early-container-patch-20261028/report-zh.txt).
- [Repeated D2H checks](../validation/gdt-meta-d2h-repeats-20261028/analysis/report-zh.md).
- [3B/7B scaling](../validation/gdt-meta-sim-scale-20261028/report-zh.md).

The calibrated profile is hardware-specific. Multi-rank request equivalence does
not establish multi-rank timing accuracy; concurrent ranges can change host and
GPU transfer contention.
