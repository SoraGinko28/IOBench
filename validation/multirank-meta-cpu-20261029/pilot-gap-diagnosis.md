# Pilot warm-load gap diagnosis

仅复核 1B safetensors 与 3B DCP 的 faithful/CPU 三次 warm load；不重跑，不改原始证据。选择 reader_return 最晚的实际 payload 关键 rank，同时记录 max-wall rank，避免把末尾 collective 等待误判为该 rank 的读时长。

| Cell | Repeat | Config | Critical rank | Max wall (s) | Payload read (s) | Storage copy/materialize (s) | Sink (s) | Uncovered (s) | Block read (bytes) |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1b safetensors | 1 | meta-faithful | 2 | 1.976894 | 0.225900 | 0.106205 | unobserved | 1.644757 | 0 |
| 1b safetensors | 1 | cpu | 1 | 1.834935 | 0.193680 | 0.092935 | unobserved | 1.548204 | 0 |
| 1b safetensors | 2 | meta-faithful | 3 | 1.897170 | 0.170603 | 0.081661 | 0.110094 | 1.534101 | 0 |
| 1b safetensors | 2 | cpu | 2 | 1.904626 | 0.191086 | 0.117282 | unobserved | 1.596169 | 0 |
| 1b safetensors | 3 | meta-faithful | 2 | 1.827904 | 0.182811 | 0.107095 | unobserved | 1.537880 | 0 |
| 1b safetensors | 3 | cpu | 1 | 1.751733 | 0.178157 | 0.085398 | unobserved | 1.487250 | 0 |
| 3b dcp | 1 | meta-faithful | 3 | 16.257211 | 9.924624 | 0.367249 | 0.332729 | 5.443299 | 1640558592 |
| 3b dcp | 1 | cpu | 0 | 6.716464 | 0.739837 | 0.301983 | unobserved | 5.599846 | 0 |
| 3b dcp | 2 | meta-faithful | 1 | 6.346137 | 0.718526 | 0.282208 | unobserved | 5.283602 | 0 |
| 3b dcp | 2 | cpu | 0 | 6.648332 | 0.767099 | 0.317058 | unobserved | 5.491380 | 0 |
| 3b dcp | 3 | meta-faithful | 0 | 6.497327 | 0.753227 | 0.301374 | unobserved | 5.377689 | 0 |
| 3b dcp | 3 | cpu | 3 | 6.462709 | 0.661185 | 0.298712 | 0.369075 | 5.047412 | 0 |

1B safetensors 的 faithful/CPU 样本分别为 1.976894/1.834935、1.897170/1.904626、1.827904/1.751733 s。paired delta 为 +0.141958、−0.007456、+0.076171 s，中位数 +0.076171 s / CPU 中位数 1.834935 s = **+4.151146%**，保留失败。六个关键 rank 的 measured read_bytes=0、major faults=0；中位 pair（repeat 3）Meta minor faults 32,467，CPU 39,757，未支持额外目标缺页解释。关键 rank 的 pread 增加 4.654 ms、storage copy/materialize 增加 21.697 ms、uncovered 增加 50.629 ms，resource CPU 增加 73.609 ms。变化主要落在 CPU 执行/复制及未覆盖时段；由于末端 sink 事件丢失，不能进一步可靠区分最终复制、Python/分配器与调度。

3B DCP 首次 faithful 16.257211 s 对 CPU 6.716464 s 的关键证据是**实际块读取**。Meta rank 3 的 payload read 9.924624 s（全部 read 10.051133 s），CPU 关键 rank 0 的 payload read 0.739837 s（全部 read 0.754628 s）。Meta rank 3 在计时内读取 1,640,558,592 个块 IO 字节，全 rank 合计 2,452,008,960；CPU 为 0。Meta sink 0.332729 s，后续 CPU rank 3 的 sink 0.369075 s，量级相当。真正 payload 关键 rank 3 在 16.253576 s 返回；max-wall rank 0 早在 7.4891 s 返回 reader 后等待其他 rank，不能按 rank 0 的 read 时长解释全局 wall。

warmup 后、worker 启动前 fincore 的确为 100%。当时 MemFree 5.652 GB、MemAvailable 29.794 GB、Cached 23.915 GB；四个 faithful buffer 合计 6.794 GB，后续还有读缓冲和进程内存。采样 worker RSS 峰值 9.656 GB，Cached 最低 20.490 GB，但 MemAvailable 仍高于 21.365 GB。这支持 checkpoint 页面在 warmup 快照后没有完整保留，不能据此宣称内存不足。缺少调用开始时/调用中的 checkpoint mincore，无法判定是在 worker 启动、预触页还是调用中回收；process read_bytes 也不是宿主机物理块 trace。后两次 faithful 为 6.346137/6.497327 s 且 measured block read=0，未删除首个异常样本。

旧 trace 存在确认的观测缺失：ComputeAccessor 用 tensor name 把多个保存列的 staging slot 压成最后一个（rank 3）；_stage 由此误标其他 rank 的 file/offset，而 session.finish 仅发布实际打开文件的事件。DCP 与 safetensors 都会丢失非末列 host_sink 事件。表中的 unobserved 不等于零，实际复制耗时仍包含在 uncovered 内。该缺失无法从保留文件恢复；后续仅修复观测标记，不改变原测量、请求或复制。

所有时长使用 dcp_started_ns 到 dcp_ended_ns；operation 区间按相位裁剪并取并集，uncovered 是 wall 减去已观测区间并集，包含未记录的 sink、Python/规划、字节拼接、trace 开销、调度与等待，不能全部称为 Python CPU 时间。JSON 附完整关键 rank 计数、内存/缓存快照、最大事件、轻量全 rank 计数及源文件 SHA256。

测量全部结束后已修复 staging slot 的 file/offset 选择，仅更改 trace 标记与发布覆盖；22 个定向测试通过（原 18 + 新 4，含双 rank、双格式 CPU/Meta 发布回归），F lint 通过。见 `postmeasure-staging-fix-tests.json` 与 `postmeasure-staging-fix-audit.json`。测量前后 runtime hash 原本一致，该修复另存新 hash，不替换测量来源，也不修补旧缺失事件。
