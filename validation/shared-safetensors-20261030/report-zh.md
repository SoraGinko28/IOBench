# 共享标准 safetensors：多 rank Meta / CPU

采用一个标准完整 tensor header，每个 rank 通过 gdtensor 并发写入自身 FSDP 行区间；header/文件大小由 rank 0 建立，barrier 后各 rank 写入互不重叠的 payload。默认 model.safetensors；按完整 tensor 拆文件时使用 HF index。没有每 rank 文件或合并读写回合。运行时细节及 setup/header/barrier 证据保留在原始 IO trace。

4个主单元、每格5次Meta/CPU配对并交错，save/cold/warm。完整及payload请求硬门禁全部通过。严格wall 11/12（fail 1，unknown 0）、shape 43/48（fail 5，unknown 0）；100ms floor wall 12/12（fail 0，unknown 0）、shape 47/48（fail 1，unknown 0）。

wall为各rank checkpoint调用最大值（save含fsync）。Δ=median(逐repeat Meta−CPU)，百分比分母为CPU中位秒；严格±3%，floor为±max(3%×CPU中位秒,100ms)。shape使用50ms聚合byte bins插值P10/P50/P90/P100。所有save shape及图均删除全rank fsync区间并集，同时保留与其他rank fsync重叠的payload字节。

位置pattern正确性：状态complete，通过20/20项；cross-rank-load 12、meta-native-byte-equality 2、native-byte-equality-and-load-file 6。证据见[correctness/progress.json](correctness/progress.json)及[correctness/results.json](correctness/results.json)。

变更：metasl/gdt_shared_hf_storage.py实现共享layout，hf_storage.py及run.py/phase.py接入；共享实现复用未改动的gdt_hf_storage.py会话；iostats.py/trace_export.py保留实际open、setup、barrier辅助证据；scripts/run_gdt_safetensors.py保留旧per-rank默认，新增shared和warm 64KiB容差；新run_shared_safetensors.py、validate_shared_safetensors.py、analyze_shared_safetensors.py负责编排、正确性及分析。

|单元|阶段|Wall Meta/CPU s|Wall Δ%|Wall严格/floor|Shape Δ% P10/P50/P90/P100|Shape严格/floor|
|---|---|---:|---:|---|---|---|
|1B/4r|save|12.897/12.944|-0.20|pass/pass|-1.17/+0.71/+0.51/-2.63|4/4 → 4/4|
|1B/4r|cold-load|11.224/11.192|+0.18|pass/pass|-1.05/+0.06/+0.44/+0.45|4/4 → 4/4|
|1B/4r|warm-load|1.262/1.205|+4.44|fail/pass|+3.42/+4.13/+3.46/+4.35|0/4 → 4/4|
|1B/8r|save|13.283/13.279|+0.51|pass/pass|+1.00/+1.23/+1.20/+2.33|4/4 → 4/4|
|1B/8r|cold-load|11.473/11.442|+0.28|pass/pass|+0.07/+0.54/+0.35/+0.08|4/4 → 4/4|
|1B/8r|warm-load|1.402/1.396|+0.20|pass/pass|+0.50/-0.16/+0.14/+0.22|4/4 → 4/4|
|3B/4r|save|38.662/38.870|+0.14|pass/pass|+0.20/+1.58/+0.51/-0.30|4/4 → 4/4|
|3B/4r|cold-load|37.427/37.419|+0.11|pass/pass|+1.20/+0.20/+0.18/+0.13|4/4 → 4/4|
|3B/4r|warm-load|3.720/3.745|-1.60|pass/pass|+0.96/+0.94/+0.68/-1.41|4/4 → 4/4|
|3B/8r|save|39.854/39.838|+0.00|pass/pass|-0.58/-0.19/-0.78/+8.71|3/4 → 3/4|
|3B/8r|cold-load|38.100/38.162|-0.10|pass/pass|-0.56/-1.27/-0.05/-0.13|4/4 → 4/4|
|3B/8r|warm-load|4.213/4.234|-0.66|pass/pass|-1.13/-1.00/+0.04/-1.20|4/4 → 4/4|


门禁逐秒、毫秒差和容限见[gate-table.csv](gate-table.csv)。warm保持≥99%计时前residency；每rank计时内≤64KiB块读取有效但仍逐字节记录，最多重做一次。本轮有效warm 40/40，最低residency 100.0%，块读取合计 4096 bytes。

共享与per-rank使用两轮既有只读基线：并发save对比multirank-concurrent-save，cold/warm及串行save背景对比fix1。跨轮比值是描述性中位数比较，没有把旧新数据视作时间配对。

|基线|单元/阶段|Wall per-rank→shared s（Meta / CPU）|共享变化% M/C|文件数 per-rank→shared|fsync次数 per-rank→shared（M/C）|
|---|---|---|---:|---|---|
|concurrent-save|1B/4r/save|12.422→12.897 / 12.500→12.944|+3.8/+3.6|4→1|6→6/6→6|
|concurrent-save|1B/8r/save|12.642→13.283 / 12.482→13.279|+5.1/+6.4|8→1|10→10/10→10|
|concurrent-save|3B/4r/save|38.649→38.662 / 38.677→38.870|+0.0/+0.5|4→1|6→6/6→6|
|concurrent-save|3B/8r/save|40.082→39.854 / 40.132→39.838|-0.6/-0.7|8→1|10→10/10→10|
|fix1|1B/4r/cold-load|12.230→11.224 / 12.387→11.192|-8.2/-9.6|4→1|0→0/0→0|
|fix1|1B/4r/save|11.848→12.897 / 11.786→12.944|+8.9/+9.8|4→1|6→6/6→6|
|fix1|1B/4r/warm-load|1.835→1.262 / 1.914→1.205|-31.2/-37.1|4→1|0→0/0→0|
|fix1|1B/8r/cold-load|14.175→11.473 / 14.142→11.442|-19.1/-19.1|8→1|0→0/0→0|
|fix1|1B/8r/save|12.003→13.283 / 11.974→13.279|+10.7/+10.9|8→1|10→10/10→10|
|fix1|1B/8r/warm-load|unknown(n=4)→1.402 / unknown(n=4)→1.396|unknown/unknown|8→1|0→0/0.0→0|
|fix1|3B/4r/cold-load|39.793→37.427 / 39.845→37.419|-5.9/-6.1|4→1|0→0/0→0|
|fix1|3B/4r/save|46.834→38.662 / 47.289→38.870|-17.4/-17.8|4→1|6→6/6→6|
|fix1|3B/4r/warm-load|6.175→3.720 / 6.194→3.745|-39.8/-39.5|4→1|0→0/0→0|
|fix1|3B/8r/cold-load|47.140→38.100 / 47.187→38.162|-19.2/-19.1|8→1|0→0/0→0|
|fix1|3B/8r/save|46.917→39.854 / 47.352→39.838|-15.1/-15.9|8→1|10→10/10→10|
|fix1|3B/8r/warm-load|13.667→4.213 / 13.754→4.234|-69.2/-69.2|8→1|0→0/0→0|

请求size histogram、每文件reader/writer rank数、header读写字节与请求、文件/handle关系及原始hash见[layout-comparison.json](layout-comparison.json)，简表见[layout-comparison.csv](layout-comparison.csv)。新共享路径的observed opens同样是下界：rank0 finish为校验header再次打开文件，这些open/read未插桩；header计数仅指已记录请求。历史路径没有open调用插桩，exact opens记unknown；另列实际触达的(rank,file)对数作为handle使用下界。旧per-rank cold/warm的header解析也未进入原始IO trace，因此旧header读取次数记未观测。

fix1基线1B/8r safetensors warm只有4个有效配对，比较wall、差百分比及加速比保留unknown；原始4对中位数仅作为证据存入layout-comparison.json，不与本轮5对结果构成完整比较。

生产加载关联：共享路径先读取8字节header长度和JSON header，再对目标tensor范围pread；实际沿用64KiB块，与生产所述4MiB pread粒度不同，tensor/shard尾部还会更小。文件组织和header→range-read顺序接近标准HF生产路径；吞吐不能直接外推到4MiB请求配置。

剩余失败（每项一句原因，详细互斥分区见[critical-rank-breakdown.csv](critical-rank-breakdown.csv)）：

- 1B/4r/warm-load，strict wall,shape-p10,shape-p50,shape-p90,shape-p100；floor 无：wall中位配对r2关键rank Meta/CPU=1/0，pre_payload +102.9ms，互斥分区合计 +53.5ms；shape-p10 +19.1ms/shape-p50 +33.7ms/shape-p90 +37.4ms/shape-p100 +50.0ms；首payload +22.3ms、P10–P90跨度 +7.3ms；定位为聚合payload进度变化；API spans含内核等待和调度，未唯一归因到设备服务时间。
- 3B/8r/save，strict shape-p100；floor shape-p100：shape-p100 +1350.0ms；首payload +58.5ms、P10–P90跨度 -102.5ms；P100中位配对r1：末payload相对各自本次最早checkpoint的完成Δ-420.6ms，减去此前fsync并集删除量Δ-1776.9ms，得到压缩末payloadΔ+1356.3ms（该配对P100分bin差+1350.0ms）；尾部shape差与跨rank pwrite/fsync重叠及压缩边界差相符；API spans含内核等待和调度，未唯一归因到设备服务时间。

每个失败保留逐repeat max-wall、latest-reader/writer、first/last-payload关键rank及互斥wall分区。分区可相加重构该rank wall；跨repeat分项中位数不可相加。IO API时长含内核等待和调度，不唯一归因于磁盘服务时间。

3B按5GiB整tensor拆文件的1-repeat smoke保留在[split-smoke-retry/](split-smoke-retry/)与[split-smoke-analysis.json](split-smoke-analysis.json)，不混入5-repeat主门禁。最初split-smoke的index文本写入失败及修复/重新执行记录见progress.json与[progress-first-pass.json](progress-first-pass.json)；主矩阵单文件路径不受该index分支修复影响。

图见[plots/summary.png](plots/summary.png)及plots/中的12个Meta/CPU聚合吞吐图（PNG/PDF）。代表repeat由同一配对平均压缩时长最接近5次中位数选取；save去fsync。summary的save点同样用去fsync全局span，cold/warm点用wall。布局检查见[plot-layout-audit.json](plot-layout-audit.json)。

每phase启动前foreign metasl.run/fsbench、CPU/本地盘空闲检查见[idle-guards.jsonl](idle-guards.jsonl)，进度见[progress.json](progress.json)，每格cleanup.json记录checkpoint清理；无旧validation写入、git操作、NFS或系统设置变更。


实现边界及顺序：支持模型张量和一维 FSDP `Shard(0)`，包含不均匀分片及空本地 shard。各 rank 独立计算全局 shape、native dtype/name 顺序及相同 layout hash；rank 0 对最终文件执行 `open(w+b) → header write → flush → ftruncate`，随后 DCP 的 global-plan scatter 发布计划，其他 rank 才进入 `r+b` 和直接 range write。没有额外的 payload 读回或合并。trace 中 setup/barrier span 包含计划发布、控制流程及准入，不能当作独立 `dist.barrier` 等待时间。加载保留同一个已读取 header 的描述符进行 payload pread。自有 reader 继续要求 `.metasl-complete.json` 完成标记；标准 safetensors/Transformers loader 可直接读取标准文件。CUDA accessor 路径保留，本轮没有实测 GPU。

位置 pattern 的 20 项门禁具体为：1B/3B × 1/4/8 ranks 的 6 项 native 整文件字节及 `load_file` 值检查（1B 三项同时检查 Transformers CPU 全部值）；两模型各自 save4→load8、save8→load4 × CPU/Meta fresh/Meta faithful 的 12 项跨 rank 加载；1B/8 ranks 的 Meta fresh/faithful 两项保存字节检查。零值计时数据没有用于替代正确性验证。

文件及 IO 几何的数值简表如下，计数为一次完整调用的所有 rank 之和；histogram 同时适用于 save 和 load，Meta/CPU、所有 repeat 及旧新布局的 payload 几何一致。

|单元|payload 文件数 per-rank→shared|payload 请求字节×次数，两布局相同|每文件 writer / reader rank 数 per-rank→shared|已观测 save header 字节 per-rank→shared|save fsync 次数 per-rank→shared|
|---|---|---|---|---|---|
|1B/4r|4→1|1024×180；65536×33568|1/1→4/4|159984→23064|6→6|
|1B/8r|8→1|512×360；65536×33568|1/1→8/8|319304→23064|10→10|
|3B/4r|4→1|128×288；1024×436；65536×103664|1/1→4/4|344744→49856|6→6|
|3B/8r|8→1|64×576；512×872；65536×103664|1/1→8/8|686656→49856|10→10|

fsync 总数包含两个完成标记/目录同步；数据文件本身为 R 次，每 rank 同步一个共享文件描述符。新布局已观测的数据文件 open 为 R 次，是下界；rank 0 finish 中另有未插桩的 header 校验 reopen。旧布局 exact open 未观测，不能把 rank/file 对数当作系统调用次数。共享 header 仅 rank 0 写一次；每个 load rank 读取 8 字节长度及其余 JSON，因此 4/8 ranks 的已观测 header read 分别为 8/16 次；1B 每 rank 共 23064 字节，3B 每 rank 共 49856 字节。数值复核见 [io-report-review.json](io-report-review.json)。

主矩阵完成 120 次调用、720 个 rank case；两个 5 GiB split smoke 各完成一组 Meta/CPU × 三阶段，共 12 次调用、72 个 rank case。总计 132 次成功调用、792 个 rank case，44 次成功 save 的准入及实际 pwrite 并发峰值均达到 world size。最初两个 split smoke 的 Meta save 在 index 写入阶段因 text/bytes 类型错误失败，未计入有效时间样本；仅修复多文件 index 的二进制序列化分支后，4/8 ranks smoke 均在新目录重跑成功。两次原失败及其日志保持原样，修复 diff、SHA 和影响范围见 [split-index-fix.md](split-index-fix.md)。主矩阵单文件分支不执行这段 index 代码；主矩阵与修复后的 smoke 各自区间内 73 个源码文件 hash 均保持不变，两个区间使用的版本明确分开。

全部 264 个有效 warm rank case 的计时前驻留率为 100%，无需 warm redo。唯一非零块读取为 1B/4r、repeat 2、Meta、rank 3 的 4096 字节，仍逐字节记录，并按本轮 ≤64 KiB 规则保留。44 个 cold 准备记录均为 0% 驻留。157 次 CLI 调用（含正确性、集成检查及原失败）时间区间互不重叠；每阶段的 host PID、CPU、磁盘和可用内存守卫证据保持在日志中。

3B/8r save 的 P100 尾部还由原始 trace 独立复算：中位配对 repeat 1 的 CPU/Meta 最后 payload rank 均为 4；相对各自最早 checkpoint 的末 payload 时间差为 −420.589 ms，之前被删除的 fsync 并集长度差为 −1776.936 ms，压缩末端差因此为 +1356.347 ms，50 ms 分 bin 后得到门禁的 +1350 ms。save 曲线是保留 payload 字节的压缩 API 进度，相交 fsync 区间被删除后可能形成尾峰，不能解释为物理盘瞬时吞吐或设备服务时间。

定向验证日志：[storage-tests-final.log](storage-tests-final.log)（13 passed，含不均匀/空 shard、2↔4 ranks 及 native bytes）、[final-helper-tests.log](final-helper-tests.log)（30 passed）、[split-index-regression-test.log](split-index-regression-test.log)（1 passed）、[recovery-analysis-tests.log](recovery-analysis-tests.log)（17 passed）、[runner-tests.log](runner-tests.log) 及 [lint.log](lint.log)。测试没有与计时测量并行执行。完整改动清单见 [changed-python-files.json](changed-python-files.json)：其中 `analyze_concurrent_save.py` 和对应旧测试相对于上轮快照的差异在本任务前已存在，不属于本次修改；README、共享路径和对应定向测试为本次改动。

最终独立复核见 [independent-audit.json](independent-audit.json)：从 rank JSON 独立重算 12 个 wall gate，从 120 条聚合 bins 独立插值重算 48 个 shape gate，并核对原始 trace hash、header/setup 顺序、配对顺序、文件清单和 warm 规则。所有 checkpoint payload 已清理；[preservation-cleanup-audit.json](preservation-cleanup-audit.json) 确认旧 validation 的 49238 个文件未变更、未新增或删除。未修改 gdtensor 库，未执行 git 操作，未变更 NFS 或系统设置。
