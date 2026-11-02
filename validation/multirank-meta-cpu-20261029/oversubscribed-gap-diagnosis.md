# 1B 16-rank DCP cold-load diagnosis

16 processes / 8 cores；仅报告，未作为 4/8-rank 时间 gate。无重跑，原数据不变。请求序列逐 rank、逐 repeat 均一致。

| Repeat | Meta wall (s) | CPU wall (s) | Delta (s) |
|---:|---:|---:|---:|
| 1 | 20.811099 | 21.408466 | -0.597368 |
| 2 | 21.896117 | 21.121183 | +0.774934 |
| 3 | 21.518312 | 20.420674 | +1.097638 |

paired median 为 repeat 2：**+0.774934 s，+3.668989%**。两种配置各自的三次 wall 中位数为 Meta 21.518312 s、CPU 21.121183 s；gate 按 paired delta 计算。

| Repeat-2 measured span | Meta critical rank 13 (s) | CPU critical rank 12 (s) | Delta (s) |
|---|---:|---:|---:|
| pre_payload_s | 12.323326 | 12.320910 | +0.002415 |
| read_s | 9.298129 | 8.535617 | +0.762512 |
| staging_union_s | 0.047575 | 0.045299 | +0.002276 |
| nonpayload_io_s | 0.161943 | 0.157673 | +0.004271 |
| uninstrumented_s | 12.371576 | 12.365239 | +0.006337 |
| reader_active_s | 21.229614 | 20.446222 | +0.783392 |
| after_reader_return_s | 0.010929 | 0.009204 | +0.001725 |

max-wall rank 分别为 Meta 7 / CPU 2；实际最晚 reader 为 Meta 13 / CPU 12，不能将早完成 reader 的 collective 等待算成读耗时。reader 进入约在 0.65 s，首个 payload 约在 12.32 s；共同约 11.67 s 的 reader 内前置区间包含 DCP 容器/头部检查、规划/视图构造与调度，现有 span 不能再细分。关键 rank 的首 payload 差仅 +2.415 ms，但 read union 差 **+762.512 ms**，与 max-wall +774.934 ms 同量级。

| Aggregate threshold | Meta repeat 2 (s) | CPU repeat 2 (s) | Paired median delta (s) | Status |
|---|---:|---:|---:|---|
| 10% | 11.614628 | 11.069494 | +0.545134 | fail |
| 50% | 15.463088 | 14.910230 | +0.552858 | fail |
| 90% | 20.720755 | 19.956609 | +0.764146 | fail |
| 100% | 21.898821 | 21.100000 | +0.798821 | fail |

repeat 2 最早 payload 的 aggregate onset 为 Meta 9.122026 s、CPU 9.091183 s，仅差 +30.843 ms；10/50/90/100% 的偏移逐步达到 +545/+553/+764/+799 ms，不能全部解释为启动整体平移。

两种配置每次都发生相同的 **2,205,810,688 bytes** 计时内块读取，cold fincore 快照均接近零。repeat 2 的 aggregate CPU 为 93.592288/92.798872 s（仅 +0.855%）；voluntary context switches 为 376358/351328，involuntary 为 236537/220187。major faults 均为 0，minor faults 为 214690/215220，未支持 Meta 在计时内额外目标缺页。rank 15 保留下来的 sink 为 0.026870/0.026630 s；关键 rank 的旧 sink 事件已丢失，仍应写 unavailable。

**推断**：主要差异随并发 cold read 的调用服务/调度时长变化，16 进程对 8 核的过量并发及 IO 交织与更高上下文切换一致。不能由此证明全部由 CPU 抢占导致：read call 同时包含实际存储服务、内核等待和失调度；没有 block latency、runqueue 或逐调用 CPU profiler，无法分离。保留 wall 和全部 shape 数值失败，明确 marked oversubscribed，不隐藏或重跑选择样本。

JSON 包含六次完整轻量 per-rank 计数、关键 rank span、cold 快照、两个关键 raw trace 的 SHA256 校验和全部原始分析 quantile/tolerance。没有重复全矩阵 raw 扫描。

两个实际关键 rank 的 CPU 为 Meta 5.725275 / CPU 5.748455 s；voluntary switches 26006/19472，involuntary 12475/12862。raw 核验得 storage host_read_copy 0.020715/0.019694 s、host_read_materialize 0.026859/0.025605 s，增量约 2.276 ms。全 worker CPU 除以 max wall 平均仅约 4.3–4.4 核，因此“16 进程 / 8 核”不等于全阶段持续占满 8 核；只能把调度/并发 IO 作为有证据的解释方向，不能归因成已证明的单一 CPU 抢占原因。
