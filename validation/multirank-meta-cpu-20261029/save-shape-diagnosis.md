# Save payload shape 超门限诊断

检查 5 个 4/8-rank gated save shape 失败阶段。每个分位点选择实际形成配对中位差的 repeat；累计分项都先在同一主机时钟上扣除所有 rank 的 fsync 并集，再截取到该分位点。只读取截至所需分位点已开始 payload 的 rank，共 30 份 raw trace；没有重测。

主要差距来自首次 writer 许可前的内存安全采样/准入，以及 CPU/Meta 实际 source materialization 成本。fsync 时间已经从下表全部移除；不能用 fsync 更短解释这些 shape 偏移。

|单元|P / 选中 repeat|Shape Δ ms|累计 pwrite Δ ms|累计 host_materialize Δ ms|其余 staging Δ ms|其余压缩时间 Δ ms|
|---|---|---:|---:|---:|---:|---:|
|1b-4r-safetensors|P10 / r2|-58.212|+1.917|-15.593|-1.423|-43.113|
|1b-4r-safetensors|P50 / r1|-100.000|+11.371|-68.394|-11.684|-31.294|
|1b-4r-safetensors|P90 / r1|-147.013|+40.054|-132.653|-15.195|-39.218|
|1b-4r-safetensors|P100 / r1|-150.000|+46.284|-141.358|-13.752|-41.174|
|1b-8r-safetensors|P10 / r2|-93.896|-11.520|-17.372|-6.555|-58.448|
|1b-8r-safetensors|P50 / r1|-150.000|+23.699|-64.229|-14.769|-94.701|
|3b-4r-dcp|P10 / r2|-153.261|+26.003|-41.558|-5.332|-132.374|
|3b-4r-dcp|P50 / r1|-300.000|+3.292|-131.764|-3.365|-168.163|
|3b-4r-safetensors|P10 / r3|-116.955|+43.297|-42.521|-2.664|-115.068|
|3b-8r-safetensors|P10 / r3|-123.287|+48.669|-50.131|+1.733|-123.559|

各行分项为同一配对、同一截点的区间并集，四项相加重构该行 shape 差；不是把不同配对中位数相加。`其余压缩时间`含 planning、准入/控制器等待及未记录的 callback/join/control；不能据此指定某个 Python 函数。

## 第一段偏移：实际准入安全策略

代码 `metasl/admission.py:175` 要求全部 rank 请求后取得新鲜 PSS/available-memory 样本；`:307` 仅在 baseline 建立后授予 FIFO permit。`metasl/run.py:239` 的控制器采样遍历 worker、controller/helper 的内存、计数器和状态，`metasl/procmem.py:18` 读取 `smaps_rollup`。worker 在已开始计时的 `write_data` 内等候许可，因此这段真实等待进入墙钟和 payload onset。

|单元 / P10选中 repeat|Meta / CPU baseline sample ms|准入等待 Δ ms|进入 writer→首 payload Δ ms|首 payload Δ ms|
|---|---:|---:|---:|---:|
|1b-4r-safetensors / r2|37.435 / 66.141|-32.254|-19.127|-45.970|
|1b-8r-safetensors / r2|58.793 / 68.807|-60.981|+7.333|-50.181|
|3b-4r-dcp / r2|38.024 / 119.035|-98.549|+5.504|-112.111|
|3b-4r-safetensors / r3|37.175 / 107.613|-91.109|-22.848|-116.225|
|3b-8r-safetensors / r3|57.317 / 143.166|-81.315|-52.188|-130.535|

这是有意实现的内存安全准入，含控制器观察开销；未发现墙钟算术错误。CPU 有实际驻留 tensor，控制器采样总耗时更长；采样包含 `smaps_rollup` 等多项操作，现有证据只精确计到整体 sample，不能把每毫秒都归给页表统计。取消或挪动采样会改变被测算法，需要新测量，不能追溯修改现有结果。

## 第二段偏移：CPU/Meta 源字节物化工作不同

`metasl/gdt_storage.py:1226` 的 `host_materialize` 实际包住 gdtensor `take()`，其中 `ctypes.string_at` 创建 source bytes。CPU 从真实 resident tensor 读取；Meta 默认 zero source 从未写入的只读匿名映射供给 demand-zero 页（`:910`、`:1017`），不是为每个 tensor 填充同等独立物理源页。相同 IO 序列因此不保证相同 source materialization 成本。

1B/4r safetensors 的 P100（repeat 1）最直接：累计 host_materialize Meta/CPU 为 **303.142 / 444.500 ms**，Meta 少 **141.358 ms**；累计 pwrite 却为 **1326.747 / 1280.463 ms**，Meta 多 **46.284 ms**。host_copy、view/generate 和其余压缩时间补足 −150 ms 总偏移。因此该项不支持“Meta 磁盘写得更快”的解释。

3B/4r DCP 的 P50（repeat 1）同样体现累计 source 差：host_materialize −131.764 ms，其他 staging −3.365 ms，加上 planning/准入/控制等残余 −168.163 ms，pwrite +3.292 ms，合成 −300 ms。3B/4r、8r safetensors 的 P10 首 payload 已分别提前 116.225 / 130.535 ms；读入曲线开始后的 pwrite 更慢，source materialization 更快，二者部分抵消，最终 P10 仍提前 116.955 / 123.287 ms。

50 ms bins 的量化影响也保留：1B/8r safetensors P50 的 binned 差 −150 ms，按实际 request interval 字节累计得到 −119.761 ms；约 −30.239 ms 来自 bin 插值差。两种诊断都超过该门限 105 ms，原始 gated 结果继续失败。其他点的精确 request 到达时刻与插值偏移见 JSON。

## 16 ranks：仅报告的 onset 证据

|单元 / P10选中 repeat|P10 Δ ms|首 payload Δ ms|准入等待 Δ ms|baseline sample Δ ms|
|---|---:|---:|---:|---:|
|3b-16r-dcp / r2|-209.306|-157.320|-122.835|-77.832|
|3b-16r-safetensors / r3|-156.780|-111.904|-151.058|-65.100|

以上 16r 只读 cache，无新增 raw 解析；均为 8 核机器 oversubscribed，仅报告，不作为 4/8r 门禁。完整各对分位点、原始样本、累计 staging 操作、首次 writer 时间分解与 source 位置见 [save-shape-diagnosis.json](save-shape-diagnosis.json)。门禁失败没有因诊断而改判，未调整源模拟参数、采样策略或重跑测量。
