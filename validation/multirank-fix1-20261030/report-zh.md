# 多 rank Meta / CPU 对齐：修复第 1 轮

请求逐 rank/phase/repeat 完整及 payload 序列全部相同。严格：wall 22/24（失败1、unknown 1），shape 90/96（失败2、unknown 4）；100ms floor：wall 23/24（失败0、unknown 1），shape 92/96（失败0、unknown 4）。

目标每格5次配对、交错运行，主要顺序1B→3B；其中1B/8r/Safe warm 缺少第5个有效配对，保留unknown，执行顺序例外见下文。完整 gate 的 wall 为各 rank checkpoint 调用最大时长，save 含 fsync。Δ=median(Meta−CPU)，百分比分母为5次 CPU 中位秒；严格±3%，floor=max(3%×CPU中位秒,100ms)。shape 用50ms聚合bins的P10/P50/P90/P100，save 删除全 rank fsync 区间并集。16 ranks 本轮跳过；ring 保留，未作为门禁或重测。

## 修复

- `metasl/admission.py`、`metasl/run.py`、`metasl/phase.py`：baseline 内存安全采样移至 checkpoint 计时前，保留安全准入和请求顺序。
- `metasl/gdt_storage.py`、`metasl/gdt_hf_storage.py`、`metasl/phase.py`：`--meta-load-sink faithful` 同时选择 CPU-target Meta resident source 与 faithful sink；source 按本地 shard 大小在计时前分配并触页，默认 GPU 模拟路径保留。resident source/sink 在 warmup 前准备。
- `metasl/gdt_storage.py`：staging 按操作的全局区间选择 saved slot，修复 host_sink 标注与发布覆盖，标注选择也适用于 reshard；定向单测证据见本轮测试记录。
- `scripts/run_gdt_safetensors.py`、`scripts/run_multirank_meta_cpu.py`：每 rank 在计时前检查 checkpoint mincore residency；低于99%或计时内 block-read>0 为无效，保留首次证据且最多重做一次。

## 8 格前后对比

旧轮3次、新轮目标5次，旧数据只读，按本轮严格及100ms floor重新判定。Δ% 按 Save/Cold/Warm；通过数 wall=3、shape=12；unknown(n=4)不展示不完整的5次统计。

|单元|旧 wall Δ% S/C/W|新 wall Δ% S/C/W|Wall严格 旧→新|Wall floor 旧→新|Shape严格 旧→新|Shape floor 旧→新|
|---|---|---|---|---|---|---|
|1B/4r/DCP|-2.31/-0.32/-4.94|-0.15/-1.58/+1.81|2/3→3/3|2/3→3/3|9/12→12/12|10/12→12/12|
|1B/4r/Safe|-0.58/-0.38/+1.79|+0.08/-1.27/-3.01|3/3→2/3|3/3→3/3|7/12→10/12|9/12→12/12|
|1B/8r/DCP|-1.31/-0.39/-0.08|-0.44/-2.01/-0.32|3/3→3/3|3/3→3/3|11/12→12/12|12/12→12/12|
|1B/8r/Safe|+0.94/-0.12/-0.99|-1.21/-0.15/unknown(n=4)|3/3→2/3 (?1)|3/3→2/3 (?1)|10/12→8/12 (?4)|11/12→8/12 (?4)|
|3B/4r/DCP|-0.53/-0.29/-5.64|-1.14/-0.03/-0.49|2/3→3/3|2/3→3/3|6/12→12/12|6/12→12/12|
|3B/4r/Safe|-0.22/+0.11/-1.47|-0.41/-0.07/+1.42|3/3→3/3|3/3→3/3|11/12→12/12|11/12→12/12|
|3B/8r/DCP|-0.02/+1.06/+1.33|-0.15/+0.06/-0.31|3/3→3/3|3/3→3/3|11/12→12/12|11/12→12/12|
|3B/8r/Safe|-0.49/+0.48/+0.38|-0.22/-0.05/-0.64|3/3→3/3|3/3→3/3|11/12→12/12|11/12→12/12|

逐 phase、逐 quantile 的前后秒差和两种判定见 [gate-table.md](gate-table.md)；全部配对样本、请求签名与原始/export SHA256 见 [analysis.json](analysis.json)。

## Warm residency

接受样本 79/79 有效（目标80），共 472 个 rank-call；计时前最低 residency 100.0%，计时内 block-read 合计 0 bytes。保留 7 次无效尝试、共执行 6 次规定内重做，未做第三次尝试；详细检查见 [warm-cache-io-audit.json](warm-cache-io-audit.json) 与 [warm-attempt-inventory.json](warm-attempt-inventory.json)。
所有 7 次无效尝试另行逐 rank 比较完整/payload 请求（共 52 个 rank-call），全部相同；sink spans 完整。无效尝试最低 residency 97.731%，单次最大块读取 454742016 bytes。这部分只补充请求硬门禁，从未加入时间统计或绘图，独立 exports 及 SHA256 见 [invalid-warm-request-audit.json](invalid-warm-request-audit.json)。


## 剩余失败与关键 rank

- 1B/8r/Safe / warm-load：wall及4个shape分位点在严格/floor均为unknown（有效配对n=4）。原因：第4次CPU warm两次尝试各有4096 bytes块读取，缺少第5个有效配对；两次调用前所有rank均为100%residency，未做第3次重测。
- 1B/4r/Safe / warm-load：严格 wall,shape-p50,shape-p90；floor 通过。wall中位配对r1的关键rank Meta/CPU=1/2，同向互斥分区合计 -208.2 ms（最大为剩余前置路径 -135.5 ms），反向分区合计 +150.5 ms 部分抵消，合成wall -57.6 ms；shape首payload +1.4 ms、P10–P90跨度 -26.1 ms，证据与payload进行中的聚合进度变化相符；这些区间已定位变化，未做唯一函数或设备服务时间归因。

[critical-rank-breakdown.csv](critical-rank-breakdown.csv) 列出所有失败阶段各 repeat 的 max-wall、最晚 reader/writer、首 payload rank，以及 pre-payload、admission wait、payload IO、materialize、完整 sink、post-read wait、uncovered、fsync、other staging/IO，单位秒。CSV各列互斥，每行可相加重构该rank wall；优先顺序 admission→payload→materialize→sink→fsync→post-read wait→pre-payload→other staging/IO→uncovered，因此pre-payload列为剔除已归类操作后的剩余前置时间。原始重叠union另存analysis.json的raw_union_s，仅用于诊断；各分项的跨repeat中位数不相加推导wall，未覆盖区间不冒充具体函数或磁盘服务时间。

本轮 load sink spans 字节覆盖缺失 0 个 rank case。完整性：{"expected_cells": 8, "observed_cells": 8, "expected_rank_cases": 1440, "observed_rank_cases": 1432, "complete": false, "accepted_warm_valid": true}。

## 执行与验证

测量前209项定向测试及F lint通过；测后18项分析回归通过（含4项新测试），见 [analysis-tests.log](analysis-tests.log)。63项源码文件hash前后完全一致（含测试），26项当前runtime文件核对一致，28份实际测量代码快照一致，见 [runtime-fix-audit.json](runtime-fix-audit.json)、[measured-runtime-provenance.json](measured-runtime-provenance.json)。8格checkpoint清理由独立审核确认；详见 [execution-audit.json](execution-audit.json)。

顺序偏离记录：1B/8r/Safe 的无效warm重做耗尽后，原runner中止该格并先启动3B/r1-Meta save及cold；待正在执行的child退出，暂停串行父进程，补齐此前未运行的1B r4-meta/r5-meta/r5-cpu，再恢复3B。补齐只收集未执行case；除已记录的6次无效warm重做外，未重放已计时阶段，见 [orchestration-pause.json](orchestration-pause.json)、[orchestration-resume.json](orchestration-resume.json)。

规则偏离记录：最初读取brief时曾并行执行只读git status，当时尚未获知no-git要求；之后无git操作。旧validation只读，未调整NFS/系统设置。

## 图与产物

[plots/summary.png](plots/summary.png) 展示完整配对差百分比，省略缺失CPU warm配对的汇总点；对应吞吐图标注4 valid pairs / gate incomplete。每格/阶段的 Meta 对 CPU 聚合吞吐图位于 plots/。所有 save 图及汇总 save 点使用删除全 rank fsync 并集的时间轴；wall 门禁仍包含fsync。PNG/PDF均保留；布局审核见plot-layout-audit.json。绘图复用完整分析结果，在既有CPUvenv完成，未重读原始trace；环境记录见plot-environment.json。
