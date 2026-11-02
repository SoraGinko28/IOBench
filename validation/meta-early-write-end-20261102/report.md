Meta finishes its buffered writes earlier mainly because its smaller anonymous-memory footprint leaves a larger Linux dirty-page budget. The external ballast intervention reproduces roughly four seconds of the delay without changing Meta's source buffers or callbacks.

Nine successful saves were measured: 3B, four ranks, one shared safetensors file, 64 MiB ring per Meta rank, three rounds in CPU/Meta/ballast, Meta/ballast/CPU, and ballast/CPU/Meta order. Each save wrote 6,794,207,232 payload bytes (6.33 GiB), plus a 49,856-byte header. All 104,388 positioned-write requests, including their order, offsets, lengths, and successful completion, matched by rank across all nine runs.

The ballast was a separate process with a private anonymous mapping of 6,493,814,784 bytes (6.05 GiB), the page-rounded difference between the previous five-run median CPU and Meta peak PSS. Every page was write-touched before launching the save; readiness timestamps precede every save timer. The process remained alive until the save returned. It did no timed work. Meta's ring capacity stayed unchanged.

Times below are actual wall time, not CPU time. **Write end** is the last payload pwrite completion relative to the earliest rank's timed save-call start; **total** ends at the latest rank's save-call return and includes the existing file fsync operations. The separate “write span” column starts at the first payload pwrite. Stalled time sums the complete duration of pwrite calls exceeding 20 ms, takes the median across ranks, and is not a sum across concurrent ranks. Dirty counters and thresholds are host-wide, converted using 4,096-byte pages.

| Run | Write end, s | Write span, s | Total, s | Stalled pwrite/rank, s | Median dirty threshold, GiB | Peak dirty, GiB | Dirty at write end, GiB |
|---|---:|---:|---:|---:|---:|---:|---:|
| r1 CPU | 17.743 | 16.430 | 37.766 | 7.185 | 4.125 | 3.708 | 3.478 |
| r1 Meta | 12.703 | 11.274 | 38.461 | 3.171 | 5.327 | 4.616 | 4.477 |
| r1 Meta + ballast | 16.353 | 14.986 | 37.982 | 6.828 | 4.112 | 3.945 | 3.695 |
| r2 CPU | 17.536 | 16.207 | 37.858 | 7.502 | 4.093 | 3.645 | 3.471 |
| r2 Meta | 12.323 | 10.896 | 38.520 | 3.262 | 5.317 | 4.613 | 4.555 |
| r2 Meta + ballast | 16.612 | 15.220 | 37.965 | 7.450 | 4.101 | 3.947 | 3.625 |
| r3 CPU | 17.377 | 16.018 | 37.930 | 8.558 | 4.089 | 3.589 | 3.484 |
| r3 Meta | 12.558 | 11.085 | 38.511 | 3.369 | 5.298 | 4.593 | 4.523 |
| r3 Meta + ballast | 17.304 | 15.915 | 37.935 | 8.499 | 4.093 | 3.586 | 3.531 |

| Median metric | CPU | Meta | Meta + ballast |
|---|---:|---:|---:|
| Write end from save start, s | 17.536 | 12.558 | 16.612 |
| Total save including fsync, s | 37.858 | 38.511 | 37.965 |
| Stalled pwrite/rank, s | 7.502 | 3.262 | 7.450 |
| Dirty threshold during writes, GiB | 4.093 | 5.317 | 4.101 |
| Background dirty threshold, GiB | 2.044 | 2.655 | 2.048 |
| System anonymous pages, GiB | 9.180 | 3.078 | 9.142 |
| Free + active/inactive file pages, GiB | 20.724 | 26.841 | 20.762 |
| First freerun-ceiling crossing from save start, s | 3.513 | 4.441 | 3.647 |
| Dirty + writeback at last pwrite, GiB | 3.554 | 4.538 | 3.719 |
| Host disk bytes written by last pwrite, GiB | 2.794 | 1.797 | 2.615 |
| Time from last pwrite to save return, s | 20.323 | 25.952 | 21.352 |

These runs reproduce a 4.98 s difference between median write ends, somewhat larger than the earlier approximately 4 s observation. Ballast moves the median Meta write end 4.05 s toward CPU. Paired shifts are 3.65, 4.29, and 4.75 s, restoring 72%, 82%, and 98% of each round's gap. Residual CPU-minus-ballast delays are 1.39, 0.92, and 0.07 s. Thus the intervention strongly identifies the dominant cause, while it does not reproduce CPU timing exactly in every round. The 0.2% threshold difference between CPU and ballast is much smaller than their remaining timing difference. Total-save medians differ by only 1.72% between Meta and CPU, and 0.28% between ballast and CPU.

| Hypothesis | Verdict | Evidence |
|---|---|---|
| H1: anonymous memory changes dirty-page throttling | **Supported; dominant cause** | Removing 6.10 GiB of system anonymous pages raises the dirty threshold by 1.223 GiB. Adding external ballast restores the threshold to 4.101 versus CPU's 4.093 GiB, restores stalled time to 7.450 versus 7.502 s/rank, and restores roughly four seconds of write time. |
| H2: a hot ring makes source copies substantially faster | **Rejected as the seconds-scale cause** | Source materialization takes 0.497/0.513/0.486 s per rank for CPU/Meta/ballast; storage host-copy callbacks take 0.356/0.354/0.347 s. Meta is slightly slower at source materialization. Median gaps between pwrites are 23.69/22.71/22.83 microseconds; p95 gaps are 54.33/52.09/51.33 microseconds. Total gaps are 1.228/1.295/1.250 s per rank. These differences cannot supply four seconds. |
| H3: journal commits independently explain the CPU/Meta gap | **Rejected as the dominant explanation; periodic modulation supported** | The mounted filesystem reports data=ordered and commit=5. Commit increments are roughly five seconds apart in all conditions. Five common stalls lasting at least 100 ms coincide within 0–50 ms with observed commit increments: three in CPU, two with ballast, none in Meta. Their total common overlap is only 0.480 s across all three CPU runs and 0.739 s across all ballast runs. Memory matching restores the longer write phase without changing journal settings. |
| H4: different I/O work, or shared-inode amplification | **Different work rejected; locking is a plausible amplifier** | Identical per-rank request signatures and payload bytes rule out extra CPU requests. Median four-rank simultaneous stall overlap is 6.223/2.529/5.252 s for CPU/Meta/ballast. Shared-inode serialization is consistent with these synchronized waits; its specific kernel wait sites were not observable. |

The threshold arithmetic independently agrees with H1: 20% × 6.10 GiB = 1.22 GiB. On Linux 5.15 the global dirtyable base is free pages minus kernel reserves, plus active and inactive file pages. Anonymous tensor storage reduces that base. The global freerun ceiling is halfway between the background and dirty thresholds, so throttling can start below nr_dirty_threshold; the measured dirty peaks were about 89%, 87%, and 96% of that threshold. [Linux 5.15 writeback source](https://github.com/torvalds/linux/blob/v5.15/mm/page-writeback.c).

The onset inferred from Dirty + Writeback crossing that ceiling is approximately 3.51 s for CPU, 4.44 s for Meta, and 3.65 s with ballast, relative to save start. A separate throughput heuristic—the first 500 ms window below 35 MB per 50 ms, after two seconds of payload writing—starts at 2.25/3.05/2.40 s relative to first pwrite. The first individual slow syscall is less reliable: CPU r1 has an isolated >20 ms call at 1.42 s of payload time, before sustained throttling. These are counter/throughput inferences, not traced balance_dirty_pages entries.

Meta leaves approximately 0.98 GiB more Dirty + Writeback at its last pwrite and has written approximately 1.00 GiB less to the device by that point. The subsequent drain takes 25.95 rather than 20.32 s. About 0.98 GiB / 0.175 GiB/s = 5.6 s of additional drain accounts for the shift, allowing for the approximately 0.65 s total-save difference and different first-write offsets. Disk-counter deltas over the complete saves are approximately 6.333 GiB in every condition. Earlier pwrite completion therefore measures additional buffering headroom, while total saves remain limited by the same storage throughput.

Journal counter phases are not fixed relative to first pwrite: the first observed in-write increment ranges from 0.15 to 4.73 s. The long common stalls above are consistent with journal-related interruptions, but journal counters alone cannot identify each blocked syscall. ext4's ordered mode and five-second transaction age are documented in the [kernel ext4 guide](https://docs.kernel.org/admin-guide/ext4.html). Linux 5.15 takes the inode lock around generic_perform_write for buffered ext4 writes, providing a mechanism by which one writer's delay can hold up peers writing the same file. This is an inference from [ext4's buffered-write source](https://github.com/torvalds/linux/blob/v5.15/fs/ext4/file.c), not a measured lock-stack attribution. All sampled wchan values were hidden as 0. The residual CPU/ballast timing difference remains unassigned; differing commit phases and writeback-controller history are candidates, not established causes.

For scaling, use a fluid approximation, not a prediction of individual journal stalls. Let R be RAM, A_i anonymous memory, K other non-dirtyable/reserved memory, S checkpoint bytes, W the unthrottled application write rate, b the sustained device rate, and beta the effective fraction of the dirty threshold available as backlog:

```text
D_i ≈ 0.20 × (R − A_i − K)
C_i ≈ beta × D_i                  (observed end-of-write Dirty/D ≈ 0.84–0.90)
t_write,i ≈ max(S/W, (S − C_i)/b)
gap ≈ max(S/W, (S − C_CPU)/b) − max(S/W, (S − C_Meta)/b)
t_total ≈ S/b + setup/publication overhead
```

When both paths throttle and their memory difference is fixed, the leading gap is approximately beta × 0.20 × (A_CPU − A_Meta) / b; RAM cancels from that difference. Increasing RAM raises both budgets and eventually removes the gap when both saves fit their unthrottled buffering window. The relative memory-induced threshold difference also shrinks as RAM increases. Smaller checkpoints can stay below the throttling window, consistent with the earlier 2.5 GB 1B result. If checkpoint size grows while CPU retains its resident tensors and Meta keeps its bounded ring, A_CPU − A_Meta grows approximately with model bytes, so the gap can grow approximately linearly until memory pressure or different I/O regimes invalidate the approximation. Clean unrelated cache generally replaces free pages in the dirtyable sum, so a cache-drop experiment was not necessary to explain this result. The [kernel VM documentation](https://kernel.org/doc/html/latest/admin-guide/sysctl/vm.html) describes the free/reclaimable-memory basis of the ratio settings.

Users should compare complete saves including fsync and report the resident-memory footprint and dirty budget alongside pwrite timing. Keep the 64 MiB ring. For an experiment isolating source-copy throughput, compare callback timings or use a small checkpoint below both throttling windows. A separately provisioned benchmark could match absolute dirty-byte budgets rather than consuming extra Meta memory; no system settings were changed here. No production fix was implemented.

The sampler was a separate process, started before every save and stopped after save completion and trace export. It records absolute perf_counter_ns and realtime_ns, all eight requested vmstat fields, meminfo Dirty/Writeback, and extra journal/disk counters. Nominal sampling was 50 ms: median 50.000 ms, worst per-run p95 50.023 ms; scheduling or logging delays occasionally produced gaps up to 422.6 ms. Median counter-read time was 0.65–0.67 ms. These gaps limit precise journal-event timing but do not obscure the GiB-scale threshold change or seconds-scale intervention result.

All jobs ran serially, checked host-visible pgrep for foreign metasl.run jobs before launch, and waited for two low-dirty baseline observations. No cache dropping, sysctl writes, remounting, git operations, production-source edits, or broad tests were performed. The initial sandbox launch failed before any worker started; its logs are retained in failed-sandbox-launch and excluded from measurements. The successful harness used elevated execution solely to permit process launching and local distributed sockets. All owned checkpoints and ballast processes were cleaned up.

The three comparison figures use real elapsed seconds aligned to first payload pwrite, with the full save interval retained. Throughput is successful bytes assigned to their pwrite completion's 50 ms bin, in decimal MB. Dirty and dirty-threshold curves use the same absolute clock. There are no fsync markers or time compression. Nine individual run figures are also provided in plots/; each figure is available as PNG and SVG.

- [Round 1 comparison](plots/round-1.png)
- [Round 2 comparison](plots/round-2.png)
- [Round 3 comparison](plots/round-3.png)
- [All run metrics](run-metrics.csv), [all-rank stall events](all-rank-stalls.csv), [complete analysis](analysis.json), and [plot data](plot-series.json)
- [Harness and sampler](driver.py), [analysis script](analyze.py), [protocol and prior PSS inputs](protocol.json), [checkpoint cleanup](cleanup.json), and [progress](progress.json)

Each measurements/3b/rN-condition directory contains vmstat.jsonl, diagnostic.json, analysis.json, the exact command/environment, completion inventory, and all four rank reports and traces.
