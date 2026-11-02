# 4-rank DCP warm 门禁差距诊断

仅检查每个模型 Meta/CPU × 3 repeats 中由 `warm-load-result.json` 选出的最长 DCP 调用 rank，共 12 份 rank report/raw trace；没有重跑测量。

## 1B：-4.9397%（配对中位差 -0.102680 s）

|Repeat|Meta rank / wall s|CPU rank / wall s|Δ wall ms|Δ 读前 ms|Δ reader active ms|Δ 读后等待 ms|
|---|---:|---:|---:|---:|---:|---:|
|1|2 / 1.984984|2 / 1.984789|+0.195|-18.819|-18.707|+37.721|
|2|0 / 1.943020|0 / 2.078683|-135.663|-20.786|-124.259|+9.382|
|3|2 / 2.073321|3 / 2.176001|-102.680|-30.209|+122.740|-195.211|

## 3B：-5.6387%（配对中位差 -0.375299 s）

|Repeat|Meta rank / wall s|CPU rank / wall s|Δ wall ms|Δ 读前 ms|Δ reader active ms|Δ 读后等待 ms|
|---|---:|---:|---:|---:|---:|---:|
|1|0 / 6.612778|2 / 6.519586|+93.192|-7.156|-111.073|+211.421|
|2|1 / 6.280458|2 / 6.655757|-375.299|-44.084|-183.192|-148.023|
|3|3 / 6.603694|1 / 7.783771|-1180.077|-73.048|-1106.147|-0.882|

1B 4-rank DCP warm 的配对中位差 -102.680 ms 取自 repeat 3，而不是两个配置各自中位时间之差。该对 Meta critical rank 2 的本地 reader_active 反而慢 122.740 ms，但读完后等待 DCP 最终 all_gather 少 195.211 ms，读前规划等少 30.209 ms，合成总差 -102.680 ms。因此这项门禁失败不能归因于 Meta sink 复制更快。
3B 的配对中位差 -375.299 ms 取自 repeat 2：Meta 读前少 44.084 ms、reader_active 少 183.192 ms、读后等待少 148.023 ms。本地 payload read 少 124.823 ms，两个 storage-source staging 操作合计少 8.953 ms；包含目标 sink 的 join/control envelope 反而多 24.650 ms。读前、读后等待和已测量 payload 路径解释主要差距，剩余具体函数归属受 trace 覆盖限制。
3B repeat 3 的更大 -1.180077 s 差中，至少 942.524 ms 的配置差发生于所有已记录操作与目标 sink 上界 envelope 之外，不能把这部分归咎于未预触页的目标 sink。该对 CPU critical 进程多用约 1.005 CPU s、额外 908 次非自愿切换，证明是实际更长的本地执行/调度时间，而现有 trace 不能进一步定位到具体 Python/gdtensor 函数。

## 明确的 trace 缺口

DCP host_sink 发布存在明确测量缺口：大多数 critical rank 的 sink 必须标为 unknown，并保留上界；不能用 0 填充。没有重新测量，没有将门禁失败改为通过，也没有将残差无证据地称为 OS noise。

来源：`metasl/gdt_storage.py:831` 将同一 tensor 的全部 rank slot 压成最后一个；`:846` 给 compute staging 标上最后一个文件；`:2065` 只发布当前 rank 实际打开文件的事件。因此 rank 0/1/2 的 `host_sink` 丢失，rank 3 保留。未发布数组没有写入 rank report，无法在进程退出后精确找回。`host_read_copy` 是 storage buffer 复制，不能充当目标 sink。

依据执行顺序 `_read → join → _write`，可用“该 tensor 最后一次 source materialize 结束至下一次 payload read 开始”包住最终 sink。1B 各 trace 恰好 201 组、3B 各 435 组；所有已发布 sink 都落在这些区间内。它包含 join/control，只是上界。

|模型 / Repeat / Config|Critical rank|精确 host_sink s|缺失时上界 s|排除全部已记录事件及 sink envelope 后 reader active s|
|---|---:|---:|---:|---:|
|1b / 1 / meta|2|unknown|0.304670|1.163902|
|1b / 1 / cpu|2|unknown|0.307589|1.176458|
|1b / 2 / meta|0|unknown|0.297208|1.176992|
|1b / 2 / cpu|0|unknown|0.301525|1.273108|
|1b / 3 / meta|2|unknown|0.307964|1.194929|
|1b / 3 / cpu|3|0.109481|—|1.125288|
|3b / 1 / meta|0|unknown|0.998180|3.844655|
|3b / 1 / cpu|2|unknown|1.061159|3.968855|
|3b / 2 / meta|1|unknown|1.029625|3.991586|
|3b / 2 / cpu|2|unknown|1.004975|4.064765|
|3b / 3 / meta|3|0.368092|—|4.193898|
|3b / 3 / cpu|1|unknown|1.094175|5.136423|

两种目标都在 load 计时前驻留；Meta 使用单个预触页 mmap，CPU 使用预填充值的本地 tensor。相同字节工作集不保证包括源 staging、分配/拼接、元数据规划和最终 rank 同步在内的完整 DCP 调用相同时间。所有选中 trace 的 payload proc read_bytes 为 0；不能把这些热读差距解释成磁盘字节量不同。

剩余区间确实位于 reader 内、payload/staging 系统调用之外。代码中包含 range 检查、构造 view/计划、Python 回调和 bytes 分配/拼接，但现有 trace 没有逐函数 CPU profile；无法把残差唯一归给这些函数中的某一个。相同 rank/phase 的 request 序列验证仍由完整矩阵分析负责，墙钟门禁继续保留失败。

完整操作 union、事件数、resource minflt/major fault/context switch/CPU、每对差值及源文件位置见 `warm-gap-diagnosis.json`。
