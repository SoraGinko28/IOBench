# 并发 save 准入与内存审核

原始 `metasl.run --writer-limit 0` 关闭 PermitQueue：不会建立 admission pipe、计时前 baseline 或 soft memory gate；仍有 supervisor 定期检查 PSS >26 GiB / MemAvailable <3 GiB。`PermitQueue` 本身拒绝 limit=0（`metasl/run.py:365–402,434–437`；`metasl/admission.py:107–108`）。

本轮专用入口将请求值 0 解析成各格 world size（4/8），然后传给底层 writer。所有 rank 可以同时持 permit，同时保留上一轮修复的计时前安全采样和失败传播。矩阵显式配置 soft PSS=26 GiB、soft MemAvailable=5 GiB；hard PSS=26 GiB、hard MemAvailable=3 GiB。计时前 baseline 在 resident source 准备后建立（`metasl/phase.py:345,699–705`）。

CPU-target Meta source 为本地 shard 大小、写触页的 resident buffer，准备不进入 save 计时（`metasl/gdt_storage.py:660–700,743–750,1523–1570`）。队列不持有 tensor 或共享 payload buffer；gdtensor 每 rank 顺序复制一个 range，额外活跃内存保守按每 rank 最大 range 的两倍，加64 KiB storage staging估算。

只读抽取 fix1 第1对 DCP：1B/3B 总 payload 为2.049/6.328 GiB；所有 rank 最大 range 之和为0.122/0.580 GiB（4/8 rank相同），保守临时 buffer 合计约0.244/1.159 GiB。旧样本 PSS 峰值范围为3.047–3.288/7.474–7.792 GiB；这些是并发前参考，不是新测量结果。并发实测的 PSS 峰值、最低可用内存、480次rank baseline及80次完整准入由 `runtime-audit.json` 给出。

这些保护是定期采样后终止，不是分配预留或精确峰值上限。全部 rank 可同时持 permit 与实际 pwrite 重叠是两个观测，审核分别记录；pwrite重叠不代表物理块设备并发。

实测审核完成：`runtime-audit.json` 的3140项断言全部通过，80/80次save、480/480次rank调用完整。80次调用的permit同时活跃峰值全部达到world size；实际pwrite调用在4-rank格峰值均为4，8-rank格峰值为7–8。全轮采样PSS最高8.052 GiB、MemAvailable最低19.802 GiB，均来自3B/8r/DCP第5对Meta；soft/hard保护均未触发。80份checkpoint均已清理，精确交错顺序及测量job无重叠通过。

计时前resident source、safe baseline及真实IO请求时间顺序全部通过；65份运行源码测量前后hash一致。43919份既有validation文件的大小和mtime均与独立基线一致；fix1的analysis/report/progress/gate-table四份基线文件SHA256均一致。
