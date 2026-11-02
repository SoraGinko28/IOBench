# 3B / 8 ranks / DCP cold-load P10 诊断

门限 P10 配对中位差 +0.815747 s（+6.023%），容差 0.406341 s；P50/P90/P100 与 wall 仍通过。决定 P10 配对中位数的是 repeat 1。原始请求完整且未重跑。

|Repeat|P10 Meta / CPU s|First payload Meta / CPU s|首 payload 至 P10 s|Prefix read rank-sum Meta / CPU s|≥50ms read rank-sum Meta / CPU s|块读取 Meta / CPU bytes|
|---|---|---|---|---|---|---|
|1|14.360440 / 13.544693|10.645482 / 10.935747|3.714958 / 2.608946|23.782367 / 17.055146|3.670667 / 0.000000|6800392192 / 6800392192|
|2|13.885509 / 14.357619|10.441386 / 11.110245|3.444123 / 3.247374|20.473264 / 23.590788|5.964349 / 8.194349|6800392192 / 6800392192|
|3|14.342170 / 13.350038|10.835685 / 10.739406|3.506484 / 2.610632|23.925757 / 16.962344|5.553342 / 0.000000|6800392192 / 6800392192|

同一 repeat 1：Meta 的首 payload 提前 0.290264 s，但首 payload 至 P10 增加 1.106011 s，合成 P10 晚 0.815747 s。早段 clipped read API rank-sum 增加 6.727221 s，其中单次至少 50 ms 的 read rank-sum 增加 3.670667 s。这些是前段实际文件 read 调用及跨 rank 进度变化的证据，不能归因于仅仅开始较晚。全阶段实际块读取差 0 bytes。

P10 延迟在 payload 已开始之后扩大，而不是 first-payload 前置延迟；同一 P10 中位配对中早段 read API 耗时与 rank 进度分布变化可观测。后续 P50/P90/P100 及 wall 仍在门限内。read API 含内核缓存/磁盘等待和调度，现有 trace 不能唯一拆成设备服务或调度时间；sink 历史 spans 的缺口也不允许将所有间隙归为 pure memcpy 或 Python。

前缀以各自的 50 ms bins 插值 P10 为截止时间；读取字节按每个调用内均匀分配，是与曲线一致的近似。rank-sum 不等于 wall，不能把它与 wall 差直接相加。逐 rank 进度和三次原始配对数值见 [cold-shape-diagnosis.json](cold-shape-diagnosis.json)。
