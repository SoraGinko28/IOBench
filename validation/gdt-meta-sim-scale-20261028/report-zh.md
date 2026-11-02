Meta D2H 模拟残差与模型规模验证。3B、7B 本地三次交错重复的四项中位数验收均通过，但不能据此认为逐大 range 的 D2H 外推准确：7B 的 1039.5 MiB range 存在约 62 ms 的模拟低估，完整 save 的正残差主要来自 container patch。模拟源码和 profile 均未改变。

**范围与复现条件。** 全部新 save 串行运行于本地 `/data00`，rank=1、zero、bf16、gdtensor、writer=1、64 KiB buffer，公共 flags 与 1B local follow-up 相同。每次两次连续 idle 检查后 sync + POSIX_FADV_DONTNEED，结束即删除 checkpoint；测量期间没有测试或分析计算并发。GPU 通过 escalated execution。保留原验证使用的舍入 profile：a=0.21 ms、BW=1.60 GB/s；未换用旧拟合文件的未舍入数值。

**Part A 的配对和会计。** 只读 1B local follow-up 与 NFS population matrix 的 NT/GPU traces；按同 repeat、file offset、size、tensor 配对，每 save 201 ranges。每个 range 窗口从 D2H 开始到下一 D2H 开始，最后一个到 preflush gap 结束。host_materialize/string_at、host_copy、pwrite 以区间并集优先互斥计时；other 是剩余时间，包含未细分的回调、释放和下一个 range 的规划，属于会计归属而非因果识别。前缀和 gap 外时间单独保留；所有 fsync 并集均扣除。每次总残差闭合误差小于 1e-8 s。

以下为三次配对差的**均值**，正数表示 NT 较慢；ms 是该大小全部 ranges 每 save 的累计差，ns/B 为累计差除以该组总字节。

| range MiB | 每 save 个数 | local pwrite ms | NFS pwrite ms | local ns/B | NFS ns/B | local 固定公式−真实 D2H ms/range | NFS 同项 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.00390625 | 45 | +0.008 | -0.042 | +0.04158 | -0.22820 | +0.167 | +0.168 |
| 1 | 44 | +0.691 | +0.426 | +0.01497 | +0.00923 | +0.017 | +0.017 |
| 8 | 44 | +8.298 | -0.678 | +0.02248 | -0.00184 | -0.330 | -0.316 |
| 22 | 66 | +27.599 | +8.549 | +0.01813 | +0.00562 | -0.607 | -0.591 |
| 125 | 2 | -3.594 | -0.781 | -0.01371 | -0.00298 | -2.029 | -2.895 |

local 的主要正 pwrite 残差集中在 8/22 MiB，NFS 集中在 22 MiB，未集中到大于 LLC 35.8 MiB 的 125 MiB 组，点估计不支持统一比例或稳定逐 range 常数；三次数据不足以统计排除微小按字节项。按每配置三次中位数再相减，local 五组分别为 +0.034/+0.587/+7.496/+26.353/+0.705 ms，NFS 为 −0.042/+0.098/−1.850/+10.058/−0.237 ms。分组中位数不能相加还原总量中位数。

原验收的 +1.43%/+1.27% 是两配置 pwrite 总量中位数之差（local +17.616 ms、NFS +13.138 ms）；逐配对差均值则为 +33.001/+7.474 ms。local 第一轮的 D2H 和 string_at 波动使均值与中位数明显不同。固定 D2H 公式在 1B 内随大小逐渐低估真实 D2H；实际模拟事件还包含等待/调度超预算，详见 CSV 的 sim_minus_target_d2h，不能把公式误差和模拟墙钟误差混用。125 MiB 仅占 2/201 ranges，却贡献约 59.45% 的中心化大小方差，因此斜率并非简单由小中 range 数量主导。

**1B 每大小的其它分量。** 下表同样为每 save 三次配对差的均值，单位 ms；payload pwrite 在上一表中，全部分量相加等于窗口总差。

| 文件系统 | range MiB | D2H | string_at | host_copy | other | 窗口总差 |
|---|---:|---:|---:|---:|---:|---:|
| local | 0.00390625 | +7.578 | -0.115 | -0.004 | -1.588 | +5.879 |
| local | 1 | +0.833 | -0.367 | +0.039 | -2.462 | -1.266 |
| local | 8 | -3.203 | -0.019 | -0.176 | +0.658 | +5.557 |
| local | 22 | +4.789 | +1.739 | +3.647 | +26.082 | +63.857 |
| local | 125 | -2.815 | +13.932 | +0.420 | -7.803 | +0.140 |
| nfs | 0.00390625 | +7.639 | -0.107 | -0.034 | -1.755 | +5.701 |
| nfs | 1 | +0.826 | -0.325 | -0.041 | -1.629 | -0.743 |
| nfs | 8 | -12.197 | +0.940 | +0.199 | -6.246 | -17.983 |
| nfs | 22 | -27.565 | +1.745 | +1.100 | -3.261 | -19.432 |
| nfs | 125 | -5.099 | +5.542 | -0.269 | -4.355 | -4.961 |

c0 表示每个完整 range 的常数项，不能直接解释为每个 pwrite syscall 的固定成本。当前 64 KiB buffer 下，除尾块和小 range 外，pwrite 次数几乎与 bytes 成比例，因此单独的 per-pwrite 常数与按字节项无法从这组 traces 辨识。

完整残差的加法会计采用均值，单位 ms；gap 外已经细分 container_patch、metadata IO、data close 和未追踪余量。

| 项目 | 1B local | 1B NFS | 3B local | 7B local |
|---|---:|---:|---:|---:|
| d2h | +7.182 | -36.396 | -40.629 | -228.458 |
| string_at | +15.170 | +7.795 | +264.002 | +1.568 |
| host_copy | +3.927 | +0.955 | -36.485 | -29.180 |
| pwrite | +33.001 | +7.474 | +149.621 | +115.813 |
| other | +14.888 | -17.246 | -40.914 | -91.564 |
| gap 内前缀 | -3.118 | -0.419 | -13.763 | +1.516 |
| container patch | -0.068 | +7.972 | +0.102 | +1135.530 |
| metadata IO | -0.011 | -0.006 | -0.001 | -0.013 |
| data close | -0.014 | -0.024 | -0.002 | +0.028 |
| gap 外未追踪余量 | +1.997 | +6.927 | -16.935 | -9.193 |
| **save 去 fsync 总残差** | +72.954 | -22.968 | +264.995 | +896.046 |

**线性模型。** 对每个 repeat 的逐 range 差做 OLS，Δ = c0·ranges + c1·bytes。表中 c0 为 µs/range，c1 为 ns/B；括号是三个 repeat 估计的 Student t(2) 95% 区间，保留二者协方差做预测。区间条件于该固定 range 分布，不把 603 个 ranges 当成 603 次独立实验，也不是新模型的预测区间。完整 save 的外推另假设 prefix/gap 外差维持 1B 常数；它们无法从固定 N、bytes 的 1B 总量单独辨识。

| 文件系统与残差 | c0 µs/range 与 95% CI | c1 ns/B 与 95% CI |
|---|---:|---:|
| local payload pwrite | 192.425 [-1543.714, 1928.564] | -0.00258 [-0.14926, 0.14410] |
| local D2H | 117.599 [-456.135, 691.334] | -0.00748 [-0.05825, 0.04329] |
| local range 窗口总计 | 205.531 [-4137.205, 4548.267] | 0.01493 [-0.25131, 0.28118] |
| nfs payload pwrite | 35.415 [-89.524, 160.354] | 0.00016 [-0.01564, 0.01597] |
| nfs D2H | 46.048 [-90.761, 182.857] | -0.02075 [-0.03683, -0.00467] |
| nfs range 窗口总计 | 3.379 [-93.860, 100.618] | -0.01732 [-0.03238, -0.00225] |

**Part B 的可行性。** 主机 RAM 31.10 GiB、初查 available 约 28 GiB、无 swap，本地空闲磁盘初查约 309 GiB；V100 32 GiB。以前只有 1B bundle；本次从 trace-matrix 已有的 3B/7B 本地 config 使用 metasl.prepare 离线生成 meta bundle，未下载权重。3B checkpoint tensor bytes 为 6.328 GiB，最大 range 593.5 MiB；7B 为 14.185 GiB，最大 1039.5 MiB。3B 加 CPU 参照；7B CPU 的模型+完整 page cache+3 GiB reserve 合计 31.37 GiB，已超过主机 RAM，尚未计 Python/staging，故跳过。7B GPU save 观察到 GPU 使用约 15 GiB；采样的 MemAvailable 最低值为 3B 20.856 GiB、7B 25.428 GiB。详见 feasibility.json 和 independent-audit.json。

| 四项验收 | 3B | 7B | 判据 |
|---|---:|---:|---|
| payload 序列 | 完全一致，103,845 次 / 6,794,207,232 B，通过 | 完全一致，232,541 次 / 15,231,233,024 B，通过 | 顺序/offset/requested/completed 完全相同 |
| 逐 range 配对 D2H 绝对误差中位数 | 0.172673 ms，通过 | 0.197751 ms，通过 | ≤0.5 ms |
| save 去 fsync 中位数差 | +0.100662 s / +0.490%，通过 | +0.540221 s / +0.835%，通过 | ±3% 或 ±0.1 s |
| payload pwrite 中位数差 | +0.004703 s / +0.094%，通过 | +0.188627 s / +0.593%，通过 | ±3% |

| 模型 | NT save 去 fsync s | GPU s | CPU s | NT pwrite s | GPU pwrite s |
|---|---:|---:|---:|---:|---:|
| 3b | 20.624164 | 20.523502 | 22.717630 | 5.030369 | 5.025666 |
| 7b | 65.266672 | 64.726451 | 跳过 | 32.009825 | 31.821198 |

**1B 模型的预测与实测。** 以下都是配对差均值，避免把用于回归的均值和验收中位数混在一起；括号是 1B 三个 repeat 预测估计的 t 区间，仅表示系数不确定性。

| 模型 | range 总残差预测 / 实测 s | save 总残差预测 / 实测 s | pwrite 预测 / 实测 s |
|---|---|---|---|
| 3b | +0.190870 [+0.100, +0.282] / +0.295595 | +0.189656 [+0.082, +0.298] / +0.264995 | +0.066176 [-0.181, +0.313] / +0.149621 |
| 7b | +0.297137 [-2.286, +2.881] / -0.231821 | +0.295923 [-2.262, +2.854] / +0.896046 | +0.025935 [-1.621, +1.673] / +0.115813 |

3B save 均值残差的点预测误差为 0.075 s，存在首轮波动；7B 的 range 残差连符号也未预测对，总 save 差大部分来自模型之外的 container patch。很宽的区间包含实测，不等于外推准确。

**超过 1B 最大 range 的 D2H。** 每个数为所有同大小、同 repeat 配对样本统计；公式误差使用固定 profile 预算，模拟误差使用实际记录的事件墙钟。

| 模型与 range MiB | 每 save 个数 | 公式−真实均值 ms/range | 模拟−真实均值 ms/range | 配对绝对误差中位数 ms |
|---|---:|---:|---:|---:|
| 3b 593.5 | 2 | +0.455 | +0.535 | 1.968 |
| 7b 129.5 | 84 | -0.436 | -0.362 | 0.871 |
| 7b 1039.5 | 2 | -68.496 | -61.777 | 64.634 |

1B 内偏差随大小增长并不能线性延续到所有大 range：593.5 MiB 的预算均值接近真实，但 1039.5 MiB 明显低估。7B 两个最大 range 的真实 D2H 均值约 750 ms，预算约 681 ms，低估约 9.1%；模拟本身有约 6.7 ms/range 超预算，仍低估约 61.8 ms。全体 range 的中位数标准会遮蔽这类少量大 tensor 误差。

**7B 尾部。** NT/GPU 的 container_patch 总时长分别为 r1 1.241/0.366 s、r2 0.862/0.699 s、r3 2.470/0.102 s；r3 NT 的 339 个 40-byte serialization-ID patches 中 51 个至少 1 ms，40-byte patch 累计 2.317 s，最长约 305 ms。它们属于 container_patch，不属于 payload，也没有被 fsync 剔除。r3 save 差 +2.220 s（+3.459%），该单次不满足 save 容差；本次只能声称中位数通过。NT payload/D2H/host/other 的 range 窗口均值合计反而快 0.232 s，container patch 均值慢 1.136 s，解释了正总残差。3B 第一轮则主要为 string_at +0.750 s 和 payload pwrite +0.396 s，与 other 等项部分抵消。精确各次闭合见 range-analysis.json，7B patch 请求见 patch-diagnosis-7b.json。不能把本地 patch 尾部归因于 NFS RPC；这些 traces 也不足以断言 NT 引起更高尾部概率。

7B pwrite 约 32 s、3B 约 5 s，绝对耗时不按字节线性增长；该计时也可能包含文件系统回压。主机 dirty_ratio/background_ratio 为 20/10。脏页/写回机制是与观察一致的解释，但没有块层延迟追踪来证明具体内核等待原因；“去 fsync”仅删除显式 fsync 区间。

**只提议修复。** 1 GiB D2H 明确需要额外标定，不能把 1B profile 当成逐大 range 精确模型。可保留小中 range 的旧 profile，采更多 ≥1 GiB 样本并加入独立大范围带宽分段。离线评估中，单一非负 affine 大范围拟合会使 129.5/593.5 MiB 更差，故不建议直接替换。采用按大小的标定节点、按整次 repeat 留一验证，1039.5 MiB 的目标绝对误差中位数可从 64.71 降到 28.54 ms，p95 从 115.03 降到 50.32 ms；只有六个此大小样本，同大小在训练和验证中都出现，不能承诺未测大小或 ≤0.5 ms。它不修复 container patch 尾部，也不等于新的端到端 save 保证。profile-proposal.json 保留原始和两种候选的误差；所有新 runs 都使用旧 profile，未应用任何修复。

**图与风险。** 两图都从横轴和总时长移除全部 fsync 并集，固定 50 ms bins；代表重复按去 fsync DCP 时长中位数分别选择：3B NT r3 / GPU r2 / CPU r1，7B NT r1 / GPU r1。PNG 已目视检查，PDF 和 plot-evidence.json 同时保留：

- [3B save 去 fsync](3b/save-no-fsync-local.png) / [PDF](3b/save-no-fsync-local.pdf)。
- [7B save 去 fsync](7b/save-no-fsync-local.png) / [PDF](7b/save-no-fsync-local.pdf)。

7B 单 rank 的中位数接近 GPU，但包含 D2H 低估与 patch 正尾部的抵消；三次结果不足以保证尾部概率。多 rank 未测，PCIe/主存带宽、LLC、page cache 和文件系统排队会共享，不能按 rank 数线性推广。较大模型没有运行 NFS 矩阵。

**文件与清理。** 改动 scripts/run_gdt_save_gap.py（可指定 bundle、跳过 CPU、本地矩阵不访问 NFS eviction root）、scripts/analyze_meta_sim_pwrite.py（可变 range 数、CPU 可选、图中 payload 总字节动态化）、新增 scripts/analyze_meta_sim_scale.py、tests/test_meta_sim_pwrite_analysis.py、docs/meta-d2h-simulation.md，以及本验证目录的证据、audit.py 和 assess_profile.py。模拟的五个源码文件与先前快照逐字节比较不变。三个绘图/配对回归测试通过，Ruff 通过；独立流式审计核验 15 次原始请求/计时、profile、公共 flags、代表重复、分箱字节和清理。

共 15 次新 save（3B 9 次、7B 6 次），15 个 checkpoint 均删除，两个本地 checkpoint root 均不存在；GPU 最终 utilization/memory 为 0%/0 MiB，helper build 目录无残留。没有 Git 操作，没有修改以前的 run 目录或 NFS 配置。仅新的分析输出并发，发生于全部 save 结束后。完整证据见 independent-audit.json、cleanup-verification.json、source-final.json；早期分析草稿目录已清理，仅保留最终分析。

复现主分析：

```bash
/tmp/metasl-gdt-venv/bin/python scripts/analyze_meta_sim_scale.py RUN NEW_ANALYSIS --prediction validation/gdt-meta-sim-scale-20261028/part-a-local-final/range-analysis.json
/tmp/metasl-gdt-venv/bin/python scripts/analyze_meta_sim_pwrite.py RUN NEW_PLOTS --chosen meta-sim-nt
```
新矩阵命令见 local-3b 与 local-7b 各 case 的 command.json；matrix runner 参数见 runner 日志与 source-final 的源码。
