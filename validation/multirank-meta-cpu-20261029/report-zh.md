# 多 rank Meta / CPU IO 对齐

**总体结论：4/8 ranks 的时间/形状验收未通过；完整/payload 请求全部通过。** 24 个 wall gate 中 2 个失败；96 个 shape quantile gate 中 17 个失败，分布于 8/24 个阶段。16 ranks 仅报告时间，不参与这些门限计数。

所有 wall 秒数均取每次运行各 rank 的 dcp_wall_seconds 最大值。配对差取三次 Meta−CPU 的中位数，百分比分母为对应三次 CPU 的中位数。时间门限 ±3%；shape 使用 50 ms 聚合 payload bins 插值的 10/50/90/100% 到达时间，门限 max(3% × CPU 中位数, 50 ms)。

完整与 payload 请求均逐 rank、逐阶段、逐 repeat 精确比较。16 ranks 在 8 核机器上过度订阅：请求相同仍为硬门限，时间/shape 仅报告。

save 图与 shape 从所有 rank 的最早 dcp_started_ns 起步，扣除所有 rank 的 fsync 区间并集；wall 门限含 fsync。save excl. local fsync（各 rank 扣各自 fsync 后再取最大值）另行报告，不能等同于全局压缩时间。

**分项遥测限制：1200/1344 个 load rank case 缺少完整 host_sink spans。** 既有多 rank compute staging 使用最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，DCP/safetensors 均会丢失部分 rank 的 sink spans。表内 sink_s 记 unknown（CSV 留空），sink_observed_s 仅为可观测下界；other 和 reader_active_uncovered 含未观测的真实目标写入，不能称其为纯 Python 开销。逐 rank 同时报告 sink 下/上界：上界为 observed sink 加 reader 活跃区间内未被已知数据读写/staging 覆盖的残余；残余还含 callback join/control，因此它是保守上界而非估计值。other 的真实区间也相应给出上下界。完整请求、payload 字节、wall、shape 与计时前 faithful buffer 准备证据仍完整；原始测量未改写。后续源码修复只影响新 trace，不能补回历史 spans。

## CPU-target sink

默认仍使用 fresh sink；CPU-target 采用计时前分配并预触页的 rank 本地持久缓冲区，faithful 覆盖本 rank 全部 shard，bounded 复用固定大小 ring。选择及两模型 pilot 证据见 [sink-selection.json](sink-selection.json)。

矩阵采用 **faithful**；ring 大小 256 MiB。下表为 4 ranks warm load；差值同样取配对中位数。

|模型|格式|CPU 秒|Faithful 秒 / 差值|Ring 秒 / 差值|请求 faithful / ring|
|---|---|---:|---|---|---|
|1b|dcp|2.033|2.039 / +0.62%|1.947 / -6.55%|identical / identical|
|1b|safetensors|1.835|1.897 / +4.15%|1.807 / -1.53%|identical / identical|
|3b|dcp|6.648|6.497 / +0.52%|6.326 / -5.88%|identical / identical|
|3b|safetensors|6.067|6.085 / +0.95%|6.006 / -2.00%|identical / identical|

## Warm 缓存方法与实际磁盘 IO

预热协议在启动 launcher 前用 mincore 检查 100% 驻留；它不保证 worker 启动、目标预触页及实际调用期间始终驻留。保留所有 warm 标签与原始样本；调用内 proc_io.read_bytes>0 的样本为 1/72，涉及 1/672 个 rank case，总计 4,096 bytes 实际块读取。详见 [warm-cache-io-audit.json](warm-cache-io-audit.json)。该计数覆盖整次调用，也可能来自 metadata；少量非零不能直接证明 payload 缓存失效，且它与 payload 请求字节不同。

|模型 / ranks / 格式|配置|Repeat|调用内块读取 bytes|Wall 秒|
|---|---|---:|---:|---:|
|3b / 16 / safetensors|cpu|1|4,096|47.633|

## Gate 表

单元格为 Meta / CPU 中位秒数、配对差 %；✅/❌ 为该项实测门限结果，16r 的时间结果不用于验收。shape 列按 save/cold/warm 顺序。

|模型|ranks|格式|请求 full / payload|Save 含 fsync|Cold load|Warm load|Shape 10/50/90/100%|
|---|---:|---|---|---|---|---|---|
|1b|4|dcp|identical / identical|✅ 14.118 / 14.608 (-2.31%)|✅ 12.009 / 12.003 (-0.32%)|❌ 1.985 / 2.079 (-4.94%)|✅/✅/❌|
|1b|4|safetensors|identical / identical|✅ 11.838 / 11.907 (-0.58%)|✅ 12.130 / 12.149 (-0.38%)|✅ 1.946 / 1.912 (+1.79%)|❌/✅/✅|
|1b|8|dcp|identical / identical|✅ 14.292 / 14.448 (-1.31%)|✅ 13.647 / 13.706 (-0.39%)|✅ 3.864 / 3.853 (-0.08%)|✅/✅/✅|
|1b|8|safetensors|identical / identical|✅ 12.222 / 12.305 (+0.94%)|✅ 14.152 / 14.170 (-0.12%)|✅ 3.830 / 3.864 (-0.99%)|❌/✅/✅|
|1b|16（过订阅）|dcp|identical / identical|✅ 20.783 / 20.718 (-0.35%)|❌ 21.518 / 21.121 (+3.67%)|✅ 12.678 / 12.745 (-0.52%)|✅/❌/✅|
|1b|16（过订阅）|safetensors|identical / identical|✅ 14.270 / 14.692 (-2.60%)|✅ 21.767 / 21.631 (+0.12%)|✅ 13.955 / 13.809 (+0.41%)|✅/✅/✅|
|3b|4|dcp|identical / identical|✅ 50.357 / 50.780 (-0.53%)|✅ 40.202 / 40.305 (-0.29%)|❌ 6.604 / 6.656 (-5.64%)|❌/✅/❌|
|3b|4|safetensors|identical / identical|✅ 46.065 / 46.205 (-0.22%)|✅ 39.870 / 39.842 (+0.11%)|✅ 6.236 / 6.328 (-1.47%)|❌/✅/✅|
|3b|8|dcp|identical / identical|✅ 58.683 / 58.417 (-0.02%)|✅ 47.306 / 46.844 (+1.06%)|✅ 13.624 / 13.432 (+1.33%)|✅/❌/✅|
|3b|8|safetensors|identical / identical|✅ 47.539 / 47.774 (-0.49%)|✅ 47.126 / 46.758 (+0.48%)|✅ 13.535 / 13.494 (+0.38%)|❌/✅/✅|
|3b|16（过订阅）|dcp|identical / identical|✅ 72.677 / 72.601 (-0.58%)|✅ 73.057 / 73.764 (-1.36%)|✅ 45.558 / 45.500 (+0.13%)|❌/✅/✅|
|3b|16（过订阅）|safetensors|identical / identical|✅ 57.294 / 57.674 (-0.66%)|✅ 76.442 / 76.445 (+0.34%)|✅ 47.822 / 47.665 (+0.40%)|❌/✅/✅|

## Shape 与 save 扣 fsync 明细

shape 单元格为 Meta / CPU 秒数（配对差毫秒）。量化精度受 50 ms bins 的均匀插值限制。

|模型 / ranks / 格式|阶段|P10|P50|P90|P100|
|---|---|---|---|---|---|
|1b / 4 / dcp|save|1.171/1.208 (-37.8 ms)|3.250/3.350 (-100.0 ms)|6.033/6.179 (-145.7 ms)|6.400/6.550 (-150.0 ms)|
|1b / 4 / dcp|cold-load|1.534/1.521 (-4.3 ms)|6.161/6.149 (-39.7 ms)|10.807/10.799 (-47.2 ms)|12.000/12.000 (-50.0 ms)|
|1b / 4 / dcp|warm-load|1.288/1.329 (-28.3 ms)|1.588/1.632 (-35.9 ms)|1.847/1.919 (-109.2 ms)|1.950/2.050 (-100.0 ms)|
|1b / 4 / safetensors|save|0.695/0.731 (-58.2 ms)|2.050/2.100 (-100.0 ms)|3.687/3.771 (-147.0 ms)|3.950/4.050 (-150.0 ms)|
|1b / 4 / safetensors|cold-load|1.326/1.341 (+3.3 ms)|6.216/6.267 (-51.0 ms)|10.915/10.939 (-46.5 ms)|12.100/12.100 (-50.0 ms)|
|1b / 4 / safetensors|warm-load|1.228/1.260 (-42.9 ms)|1.484/1.509 (-39.0 ms)|1.745/1.757 (-8.7 ms)|1.900/1.900 (+0.0 ms)|
|1b / 8 / dcp|save|1.362/1.406 (-44.5 ms)|5.300/5.400 (-150.0 ms)|9.946/10.122 (-176.4 ms)|10.300/10.450 (-150.0 ms)|
|1b / 8 / dcp|cold-load|3.313/3.348 (-47.2 ms)|7.837/7.898 (-47.7 ms)|12.463/12.527 (-42.2 ms)|13.648/13.700 (-50.0 ms)|
|1b / 8 / dcp|warm-load|3.157/3.123 (+41.7 ms)|3.446/3.419 (+20.8 ms)|3.693/3.701 (-4.6 ms)|3.850/3.850 (+0.0 ms)|
|1b / 8 / safetensors|save|0.880/0.971 (-93.9 ms)|3.350/3.500 (-150.0 ms)|6.312/6.473 (-160.9 ms)|6.600/6.750 (-150.0 ms)|
|1b / 8 / safetensors|cold-load|3.433/3.426 (+10.4 ms)|8.304/8.325 (-7.8 ms)|12.957/12.984 (-12.4 ms)|14.150/14.150 (+0.0 ms)|
|1b / 8 / safetensors|warm-load|3.260/3.249 (-4.9 ms)|3.487/3.484 (-32.2 ms)|3.699/3.708 (-37.2 ms)|3.800/3.850 (-46.6 ms)|
|1b / 16 / dcp|save|2.620/2.632 (-6.9 ms)|9.800/9.900 (-150.0 ms)|18.068/18.039 (-79.2 ms)|19.400/19.400 (-150.0 ms)|
|1b / 16 / dcp|cold-load|11.094/11.050 (+545.1 ms)|14.898/14.789 (+552.9 ms)|20.105/19.954 (+764.1 ms)|21.500/21.100 (+798.8 ms)|
|1b / 16 / dcp|warm-load|9.958/9.965 (-7.2 ms)|11.978/11.931 (+143.2 ms)|12.525/12.494 (+31.2 ms)|12.650/12.745 (-94.8 ms)|
|1b / 16 / safetensors|save|1.798/1.867 (-55.6 ms)|6.500/6.650 (-150.0 ms)|11.802/12.079 (-233.4 ms)|12.650/13.000 (-350.0 ms)|
|1b / 16 / safetensors|cold-load|12.151/11.790 (+55.2 ms)|15.934/15.794 (+28.3 ms)|20.582/20.441 (+25.9 ms)|21.750/21.600 (+8.4 ms)|
|1b / 16 / safetensors|warm-load|11.394/11.384 (+9.3 ms)|13.207/13.133 (+57.2 ms)|13.736/13.681 (+56.0 ms)|13.950/13.800 (+50.0 ms)|
|3b / 4 / dcp|save|2.936/3.054 (-153.3 ms)|8.750/9.100 (-300.0 ms)|16.661/16.614 (-256.7 ms)|18.050/17.700 (-400.0 ms)|
|3b / 4 / dcp|cold-load|6.405/6.534 (-148.3 ms)|21.709/22.029 (-320.8 ms)|36.387/36.512 (-146.4 ms)|40.000/40.150 (-150.0 ms)|
|3b / 4 / dcp|warm-load|4.030/4.204 (-232.1 ms)|4.981/5.235 (-275.4 ms)|5.915/6.134 (-273.4 ms)|6.450/6.500 (-400.0 ms)|
|3b / 4 / safetensors|save|2.015/2.132 (-117.0 ms)|6.500/6.500 (+0.0 ms)|11.741/11.908 (-166.9 ms)|12.600/12.800 (-250.0 ms)|
|3b / 4 / safetensors|cold-load|6.212/6.223 (-36.9 ms)|21.557/21.446 (+346.9 ms)|36.072/36.038 (+58.8 ms)|39.700/39.650 (+50.0 ms)|
|3b / 4 / safetensors|warm-load|3.963/3.990 (-27.3 ms)|4.924/4.893 (-14.0 ms)|5.748/5.721 (-45.8 ms)|6.050/6.150 (-100.0 ms)|
|3b / 8 / dcp|save|3.567/3.689 (-110.3 ms)|14.950/15.100 (-200.0 ms)|28.300/28.467 (-32.8 ms)|29.450/29.700 (-50.0 ms)|
|3b / 8 / dcp|cold-load|14.342/13.545 (+815.7 ms)|29.238/28.785 (+492.5 ms)|43.594/43.135 (+488.0 ms)|47.200/46.750 (+500.0 ms)|
|3b / 8 / dcp|warm-load|11.306/11.100 (+227.8 ms)|12.217/12.070 (+139.5 ms)|13.086/12.924 (+161.9 ms)|13.550/13.300 (+200.0 ms)|
|3b / 8 / safetensors|save|2.286/2.410 (-123.3 ms)|9.250/9.450 (-200.0 ms)|17.393/17.673 (-221.9 ms)|18.150/18.500 (-250.0 ms)|
|3b / 8 / safetensors|cold-load|13.560/13.178 (+268.6 ms)|29.020/28.646 (+245.2 ms)|43.413/43.040 (+240.9 ms)|47.000/46.650 (+250.0 ms)|
|3b / 8 / safetensors|warm-load|11.512/11.509 (+3.5 ms)|12.337/12.289 (+27.1 ms)|13.072/13.043 (+35.2 ms)|13.400/13.400 (+50.0 ms)|
|3b / 16 / dcp|save|6.677/6.881 (-209.3 ms)|26.200/26.500 (-300.0 ms)|48.063/48.517 (-486.8 ms)|51.700/52.150 (-450.0 ms)|
|3b / 16 / dcp|cold-load|40.779/41.774 (-994.9 ms)|55.104/55.816 (-1007.1 ms)|69.393/70.111 (-1005.9 ms)|73.000/73.700 (-1000.0 ms)|
|3b / 16 / dcp|warm-load|40.850/41.029 (-462.0 ms)|43.564/43.794 (-230.2 ms)|44.790/44.989 (-189.6 ms)|45.500/45.450 (+50.0 ms)|
|3b / 16 / safetensors|save|4.781/4.870 (-156.8 ms)|18.250/18.250 (-200.0 ms)|33.283/33.346 (-313.2 ms)|35.750/35.950 (-350.0 ms)|
|3b / 16 / safetensors|cold-load|44.221/43.920 (+335.4 ms)|58.393/58.354 (+261.9 ms)|72.772/72.756 (+267.9 ms)|76.400/76.350 (+250.0 ms)|
|3b / 16 / safetensors|warm-load|43.380/43.650 (-270.2 ms)|46.143/45.934 (+208.9 ms)|47.307/47.218 (+89.9 ms)|47.750/47.600 (+150.0 ms)|

|模型 / ranks / 格式|Save excl. local fsync：Meta / CPU|Global fsync union removed：Meta / CPU|全局压缩时间：Meta / CPU|
|---|---|---|---|
|1b / 4 / dcp|12.196 / 12.688 s|7.736 / 8.018 s|6.419 / 6.582 s|
|1b / 4 / safetensors|9.906 / 9.983 s|7.912 / 7.784 s|4.041 / 4.137 s|
|1b / 8 / dcp|13.827 / 13.988 s|3.962 / 3.921 s|10.358 / 10.527 s|
|1b / 8 / safetensors|11.745 / 11.813 s|5.503 / 5.324 s|6.830 / 6.981 s|
|1b / 16 / dcp|20.713 / 20.646 s|1.275 / 1.217 s|19.508 / 19.504 s|
|1b / 16 / safetensors|14.201 / 14.617 s|1.135 / 1.178 s|13.154 / 13.506 s|
|3b / 4 / dcp|42.487 / 42.755 s|33.030 / 33.029 s|18.191 / 17.835 s|
|3b / 4 / safetensors|38.050 / 38.191 s|33.215 / 33.087 s|12.850 / 13.094 s|
|3b / 8 / dcp|55.177 / 54.898 s|29.113 / 28.784 s|29.570 / 29.849 s|
|3b / 8 / safetensors|44.031 / 44.266 s|28.851 / 28.704 s|18.688 / 19.012 s|
|3b / 16 / dcp|71.422 / 71.340 s|20.511 / 20.214 s|51.930 / 52.394 s|
|3b / 16 / safetensors|56.031 / 56.417 s|20.426 / 20.452 s|36.973 / 37.160 s|

## Save gap 证据

runner 使用 writer_limit=1，写入许可覆盖 write_data 以及其 fsync。一个 rank 的 checkpoint wall 包含等待其他 rank 的写入与 fsync；只扣本 rank 的 fsync，会把其他 rank 的 fsync 仍保留在本 rank 的 other 中。因此本地 save excl. fsync 很长而 pwrite 很短可以同时成立，是实测准入/collective 等待。

逐 rank 的 queue_enter_ns/write_enter_ns/future_completed_ns 与 fsync 原始时间见 analysis.json；[rank-breakdown.csv](rank-breakdown.csv) 分列准入等待、writer 活跃区间和完成后的等待。pwrite/read/fsync/sink 为各自区间并集；other 是其总并集的补集，host_copy 等 staging 和 Python 调用开销会计入 other，分项若有重叠不作盲目相加。

load 进一步保留 metadata gate 等待、metadata 读取/反序列化/校验、剪枝重建、planning/collective、reader 准入/活跃/返回后区间。CSV 保留 mandatory 的 pwrite/read/fsync/sink/other 与 staging_union。恢复后的 compact analysis 仅保留 CSV 的 host_copy、host_materialize、observed_sink 可观测下界；其他 staging op 的完整细分未重新扫描，次数/字节/排除主分项后的独立区间缺失时记 unknown。原始 JSONL 全部保留，定点诊断另从 raw trace 提供 host_read_copy、host_read_materialize 等分项；不声称 compact JSON 已包含每个 op 的完整分解。嵌套 staging spans 不相加推导 wall；uninstrumented 排除了已测 payload/数据读写/fsync/staging，仍可能含已记录 seek/flush/close 等辅助调用；它不能解释为纯未插桩或纯 Python 时间。

额外诊断文件：[save-gap.json](save-gap.json)。

Warm 与 pilot 的定点诊断：[warm-gap-diagnosis.json](warm-gap-diagnosis.json)、[pilot-gap-diagnosis.md](pilot-gap-diagnosis.md)、[pilot-gap-diagnosis.json](pilot-gap-diagnosis.json)、[save-shape-diagnosis.md](save-shape-diagnosis.md)、[save-shape-diagnosis.json](save-shape-diagnosis.json)、[cold-shape-diagnosis.md](cold-shape-diagnosis.md)、[cold-shape-diagnosis.json](cold-shape-diagnosis.json)、[oversubscribed-gap-diagnosis.md](oversubscribed-gap-diagnosis.md)、[postmeasure-staging-fix-tests.json](postmeasure-staging-fix-tests.json)、[postmeasure-staging-fix-audit.json](postmeasure-staging-fix-audit.json)、[execution-audit-zh.md](execution-audit-zh.md)、[final-artifact-audit.json](final-artifact-audit.json)、[plot-file-audit.json](plot-file-audit.json)、[plot-layout-audit.json](plot-layout-audit.json)、[analysis-recovery.json](analysis-recovery.json)。pilot 保留 1B safetensors faithful +4.151% 的真实偏差和 3B DCP faithful 首次 16.257 s 样本；后者额外约 9.3 s 的 payload read 与实际磁盘读取相符，未删样本或重跑替换。

## 未通过 shape 的专门诊断

五个 gated save 形状失败均在扣除全 rank fsync 后仍成立。首次 writer 的 memory-safety 准入有意要求 fresh PSS/baseline sample；CPU resident 模型与 Meta zero source 的采样和 source bytes 物化成本不同，是实测路径差异，未证明算术 bug。不能把这些 payload 到达偏差归因于写后 collective。

|单元|P10 首次准入等待 Δ ms|其中 baseline sample Δ ms|
|---|---:|---:|
|1b-4r-safetensors|-32.254|-28.706|
|1b-8r-safetensors|-60.981|-10.014|
|3b-4r-dcp|-98.549|-81.011|
|3b-4r-safetensors|-91.109|-70.438|
|3b-8r-safetensors|-81.315|-85.850|

1B/4r SafeTensors P100 的实际中位配对：host_materialize -141.358 ms、pwrite +46.284 ms；净 shape -150.000 ms。该 span 是 source 读取/bytes 分配/ctypes callback 的整体，不等同于磁盘 IO 或 load sink。所有精确配对、累计区间和 source 行号见 save-shape-diagnosis。

3B/8r DCP cold P10：中位配对延迟 +815.747 ms（+6.023%），但 Meta 首 payload 更早 290.264 ms；差异在 payload 开始后扩大。到 P10 的 read API rank-sum 多 6.727 s，其中 ≥50 ms 调用多 3.671 s/43 次；三次整调用块读取字节差均为 0，同一配对整调用 CPU 时间差 −9.858 ms。由此定位到早段并发 read/interleaving；read API 包含内核/设备等待与调度，不能唯一拆开。P50/P90/P100 与 wall 均通过。

P10 延迟在 payload 已开始之后扩大，而不是 first-payload 前置延迟；同一 P10 中位配对中早段 read API 耗时与 rank 进度分布变化可观测。后续 P50/P90/P100 及 wall 仍在门限内。read API 含内核缓存/磁盘等待和调度，现有 trace 不能唯一拆成设备服务或调度时间；sink 历史 spans 的缺口也不允许将所有间隙归为 pure memcpy 或 Python。

16r 仅报告项：1B DCP cold 的中位 wall 配对差 +774.934 ms；关键 rank read API +762.512 ms、onset +2.415 ms、source staging +2.276 ms。字节相同，证据与并发读调用延迟/调度变化一致；16 进程/8 核是已知条件，不能唯一证明超订阅本身是原因。

4 ranks DCP warm 的 wall 未通过项，以下使用实际决定配对中位差的同一 repeat（毫秒），不是分别取分项中位数后相加：

|模型|Repeat|进入 reader 前 Δ|Reader 活跃 Δ|返回后等待 Δ|Sink+join 包络 Δ|
|---|---:|---:|---:|---:|---:|
|1b|3|-30.209|+122.740|-195.211|+20.208|
|3b|2|-44.084|-183.192|-148.023|+24.650|

包络由整段 tensor 的最后一次 source materialize 到下一次 read 的实际时间界定，并验证包含所有已发布 rank3 sink；它包括目标写入及 callback join/control，不是纯 memcpy。两项中位配对的 Meta sink+join 包络均更长，因此不能把更短 wall 解释为 Meta 获得了更短目标写入包络。1B 主要差值位于返回后的全局完成等待；3B 同时包含 read API 与完成等待的变化。

## 未通过项及归因

- 验收未通过：1b / 4r / dcp / warm-load，wall, shape-p90, shape-p100。
  Trace 证据：18/24 个 rank case 的 host_sink 字节覆盖不足；sink_s 记为 unknown，observed sink 下界和未覆盖字节保留。other/reader_active_uncovered 含这些未观测的真实 sink 执行。 每次各配置最慢 rank 的配对分项差最大位于 other_s（-0.0055 s）；关键 rank 的身份保留在 analysis.json。 reader/metadata 分解中最大配对变化为 after_metadata_to_reader_queue_s -0.0210 s。 shape P10/P50/P90/P100 配对差 -28.3/-35.9/-109.2/-100.0 ms；首 payload -12.6 ms，P10–P90 跨度 -52.1 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 -24.8 ms；进入→首 payload +13.6 ms。 决定 wall 配对中位差的 r3：进入 reader 前 -0.0302 s、reader 活跃 +0.1227 s、reader 返回后 -0.1952 s；这三个同一配对区间相加为 -0.1027 s。 同一配对的最晚 reader 返回 rank Meta/CPU 3/1；全 rank reader 返回偏斜 0.1252/0.3011 s。
  基于这些区间的归因：现有多 rank trace 的 compute staging 以最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，导致部分 rank 的 host_sink span 丢失（DCP/safetensors 均受影响）；因此不能把该部分 other 唯一归因于 Python/gdtensor 调度。请求、payload bytes、wall 和 shape 门限不受这一分项遥测缺口影响。 偏差发生在 metadata 返回后、read_data 准入前的 planning/collective 路径。 shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。 wall 中位差的 r3 主要来自本 rank read_data 返回后的完成等待，而非该 rank 自身 read/sink 更慢；全 rank 最晚 reader 返回者另行记录，避免把任意一个 wall 最大 rank 误当作执行瓶颈。
- 验收未通过：1b / 4r / safetensors / save，shape-p10, shape-p50, shape-p90, shape-p100。
  Trace 证据：P10 中位配对 r2：shape -58.212 ms；首次 writer 准入等待 -32.254 ms（其中 baseline sample -28.706 ms）；累计 host_materialize -15.593 ms、pwrite +1.917 ms。 P100 中位配对 r1：shape -150.000 ms；首次 writer 准入等待 -18.830 ms（其中 baseline sample -51.048 ms）；累计 host_materialize -141.358 ms、pwrite +46.284 ms。
  基于这些区间的归因：该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。
- 验收未通过：1b / 8r / safetensors / save，shape-p10, shape-p50。
  Trace 证据：P10 中位配对 r2：shape -93.896 ms；首次 writer 准入等待 -60.981 ms（其中 baseline sample -10.014 ms）；累计 host_materialize -17.372 ms、pwrite -11.520 ms。 P50 中位配对 r1：shape -150.000 ms；首次 writer 准入等待 -93.039 ms（其中 baseline sample -11.325 ms）；累计 host_materialize -64.229 ms、pwrite +23.699 ms。
  基于这些区间的归因：该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。
- 16r 仅报告：1b / 16r / dcp / cold-load，wall, shape-p10, shape-p50, shape-p90, shape-p100。
  Trace 证据：90/96 个 rank case 的 host_sink 字节覆盖不足；sink_s 记为 unknown，observed sink 下界和未覆盖字节保留。other/reader_active_uncovered 含这些未观测的真实 sink 执行。 每次各配置最慢 rank 的配对分项差最大位于 read_s（+1.1894 s）；关键 rank 的身份保留在 analysis.json。 shape P10/P50/P90/P100 配对差 +545.1/+552.9/+764.1/+798.8 ms；首 payload +30.8 ms，P10–P90 跨度 +104.2 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 +22.3 ms；进入→首 payload +64.7 ms。 决定 wall 配对中位差的 r2：进入 reader 前 -0.0062 s、reader 活跃 +2.0200 s、reader 返回后 -1.2388 s；这三个同一配对区间相加为 +0.7749 s。 同一配对的最晚 reader 返回 rank Meta/CPU 13/12；全 rank reader 返回偏斜 11.6814/11.1068 s。
  基于这些区间的归因：现有多 rank trace 的 compute staging 以最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，导致部分 rank 的 host_sink span 丢失（DCP/safetensors 均受影响）；因此不能把该部分 other 唯一归因于 Python/gdtensor 调度。请求、payload bytes、wall 和 shape 门限不受这一分项遥测缺口影响。 偏差直接落在实际 payload read 调用区间；这些是文件 API 耗时，包含内核/缓存等待，不能单凭它推断块设备的服务时间。 shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。
- 验收未通过：3b / 4r / dcp / save，shape-p10, shape-p50。
  Trace 证据：P10 中位配对 r2：shape -153.261 ms；首次 writer 准入等待 -98.549 ms（其中 baseline sample -81.011 ms）；累计 host_materialize -41.558 ms、pwrite +26.003 ms。 P50 中位配对 r1：shape -300.000 ms；首次 writer 准入等待 -90.304 ms（其中 baseline sample -85.757 ms）；累计 host_materialize -131.764 ms、pwrite +3.292 ms。
  基于这些区间的归因：该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。
- 验收未通过：3b / 4r / dcp / warm-load，wall, shape-p10, shape-p50, shape-p90, shape-p100。
  Trace 证据：18/24 个 rank case 的 host_sink 字节覆盖不足；sink_s 记为 unknown，observed sink 下界和未覆盖字节保留。other/reader_active_uncovered 含这些未观测的真实 sink 执行。 每次各配置最慢 rank 的配对分项差最大位于 other_s（-0.2505 s）；关键 rank 的身份保留在 analysis.json。 reader/metadata 分解中最大配对变化为 reader_active_uncovered_s -0.1871 s。 shape P10/P50/P90/P100 配对差 -232.1/-275.4/-273.4/-400.0 ms；首 payload -190.6 ms，P10–P90 跨度 -41.3 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 -38.1 ms；进入→首 payload -120.7 ms。 决定 wall 配对中位差的 r2：进入 reader 前 -0.0441 s、reader 活跃 -0.1832 s、reader 返回后 -0.1480 s；这三个同一配对区间相加为 -0.3753 s。 同一配对的最晚 reader 返回 rank Meta/CPU 3/0；全 rank reader 返回偏斜 0.0808/0.2719 s。
  基于这些区间的归因：现有多 rank trace 的 compute staging 以最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，导致部分 rank 的 host_sink span 丢失（DCP/safetensors 均受影响）；因此不能把该部分 other 唯一归因于 Python/gdtensor 调度。请求、payload bytes、wall 和 shape 门限不受这一分项遥测缺口影响。 偏差发生在 read_data 内、已测 IO/staging spans 之外；该区间可能包含未观测的 sink 与 accessor/Python/gdtensor 调度，现有 trace 无法再唯一细分。
- 验收未通过：3b / 4r / safetensors / save，shape-p10。
  Trace 证据：P10 中位配对 r3：shape -116.955 ms；首次 writer 准入等待 -91.109 ms（其中 baseline sample -70.438 ms）；累计 host_materialize -42.521 ms、pwrite +43.297 ms。
  基于这些区间的归因：该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。
- 验收未通过：3b / 8r / dcp / cold-load，shape-p10。
  Trace 证据：42/48 个 rank case 的 host_sink 字节覆盖不足；sink_s 记为 unknown，observed sink 下界和未覆盖字节保留。other/reader_active_uncovered 含这些未观测的真实 sink 执行。 每次各配置最慢 rank 的配对分项差最大位于 read_s（+1.1773 s）；关键 rank 的身份保留在 analysis.json。 shape P10/P50/P90/P100 配对差 +815.7/+492.5/+488.0/+500.0 ms；首 payload -290.3 ms，P10–P90 跨度 -327.7 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 +58.4 ms；进入→首 payload -348.7 ms。 决定 wall 配对中位差的 r1：进入 reader 前 +0.0205 s、reader 活跃 +1.6848 s、reader 返回后 -1.2103 s；这三个同一配对区间相加为 +0.4951 s。 同一配对的最晚 reader 返回 rank Meta/CPU 7/0；全 rank reader 返回偏斜 6.1836/3.4681 s。
  基于这些区间的归因：现有多 rank trace 的 compute staging 以最后一个 slot 的文件名标记，发布时按本 rank 已打开文件过滤，导致部分 rank 的 host_sink span 丢失（DCP/safetensors 均受影响）；因此不能把该部分 other 唯一归因于 Python/gdtensor 调度。请求、payload bytes、wall 和 shape 门限不受这一分项遥测缺口影响。 偏差直接落在实际 payload read 调用区间；这些是文件 API 耗时，包含内核/缓存等待，不能单凭它推断块设备的服务时间。 shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。
- 验收未通过：3b / 8r / safetensors / save，shape-p10。
  Trace 证据：P10 中位配对 r3：shape -123.287 ms；首次 writer 准入等待 -81.315 ms（其中 baseline sample -85.850 ms）；累计 host_materialize -50.131 ms、pwrite +48.669 ms。
  基于这些区间的归因：该形状已扣除全 rank fsync 并集，不能归因于 fsync 或 payload 完成后的 collective。定点 trace 将偏差定位到计时内有意设置的首次准入 fresh PSS/baseline 采样，以及 source 端 bytes 物化跨度；两者与 CPU resident source 和 Meta untouched zero source 的实际宿主内存路径有关，不是请求字节不同，也不是 wall 算术错误。
- 16r 仅报告：3b / 16r / dcp / save，shape-p10。
  Trace 证据：每次各配置最慢 rank 的配对分项差最大位于 other_s（-0.4136 s）；关键 rank 的身份保留在 analysis.json。 全 rank fsync 并集 Meta/CPU 20.511/20.214 s，配对差 +0.2967 s；去 fsync 全局时间差 -0.4644 s。 writer 等待分解中最大配对变化为 writer_admission_wait_s +22.4872 s。 关键 rank staging 最大变化 host_materialize：Meta/CPU 0.0723/0.0983 s，配对差 -0.0265 s。 shape P10/P50/P90/P100 配对差 -209.3/-300.0/-486.8/-450.0 ms；首 payload -157.3 ms，P10–P90 跨度 -277.5 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 -129.1 ms；进入→首 payload -27.3 ms。
  基于这些区间的归因：含 fsync 的 wall 差与实测同步刷盘区间变化相符；单 writer 许可覆盖 fsync，因此该时间也传播为其他 rank 的准入/结束等待。 变化被定位到单 writer 的实测准入等待；不应把本地 other 当作未记录的 pwrite。 shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。
- 16r 仅报告：3b / 16r / safetensors / save，shape-p10。
  Trace 证据：每次各配置最慢 rank 的配对分项差最大位于 other_s（-0.3907 s）；关键 rank 的身份保留在 analysis.json。 全 rank fsync 并集 Meta/CPU 20.426/20.452 s，配对差 -0.0260 s；去 fsync 全局时间差 -0.4328 s。 writer 等待分解中最大配对变化为 after_writer_completed_s -3.5702 s。 关键 rank staging 最大变化 host_materialize：Meta/CPU 0.0741/0.0951 s，配对差 -0.0210 s。 shape P10/P50/P90/P100 配对差 -156.8/-200.0/-313.2/-350.0 ms；首 payload -162.6 ms，P10–P90 跨度 -145.6 ms。 最早 payload 所属 rank 的开始分解：phase→reader/writer 进入配对差 -127.6 ms；进入→首 payload +15.7 ms。
  基于这些区间的归因：变化被定位到本 rank 写入完成后的 collective/metadata 发布路径；不应把本地 other 当作未记录的 pwrite。 shape 偏移在 payload 进行过程中扩大，需结合 read/pwrite 与 staging 分项解释；不能仅由聚合曲线认定磁盘带宽改变。

## 输出与验证范围

分析包含 2016 个 rank case。完整/载荷 trace exports、每次全 rank 聚合 bins、逐 repeat 配对值、gate 明细与 provenance SHA256 均保留在 [analysis.json](analysis.json) 和 exports/；逐 rank 分解见 rank-breakdown.csv；图见 plots/。

绘图每个 cell/phase 选择配对平均时间最接近三次中位数的同一 repeat；Meta/CPU 各自使用该 repeat。save 均无 fsync 轴区间。冷/热缓存证据与清理结果保留在 runner 的原始目录。读取已有 validation 仅用于背景核对，未改写旧目录。
