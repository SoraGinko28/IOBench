# 多 rank Meta / CPU：并发 save

完成 1B/3B × 4/8 ranks × DCP/safetensors，每格5次配对、逐对交错，仅测save：80次调用、480个rank case，无失败、重测或顺序偏离。240个rank配对的完整及payload请求序列全部相同（硬门禁通过）。严格：wall 8/8（失败0，unknown 0），shape 26/32（失败6，unknown 0）；100ms floor：wall 8/8（失败0，unknown 0），shape 28/32（失败4，unknown 0）。

新增 `--writer-limit`，默认1，支持非负值；load仍为`--reader-limit 0`。`run_gdt_safetensors.py --save-only`跳过加载，`run_multirank_meta_cpu.py --concurrent-save --writer-limit 0`运行本轮矩阵。底层原始0会关闭PermitQueue及其baseline/soft检查；本轮专用入口将请求0解析为world size（4/8），保留计时前baseline及soft/hard内存保护。请求值与实际值记入progress及environment。沿用fix1的计时外准入采样、CPU-target resident local-shard source、faithful配置和sink trace修复。

80次准入峰值均达到world size；77次pwrite峰值达到world size，其余3次为8-rank中的7。采样PSS最高8.052 GiB，MemAvailable最低19.802 GiB，满足soft PSS<26 GiB/available≥5 GiB及hard 26/3 GiB保护；采样是fail-stop检查，不是精确峰值或内存预留。详见[准入说明](admission-review.md)和[3140项运行审核](runtime-audit.json)。

门禁Δ=median(五次配对 Meta−CPU)，百分比分母为五次CPU中位秒数。wall取各rank checkpoint调用最大时长，包含fsync。shape取50ms聚合byte bins插值的P10/P50/P90/P100，先删除全rank fsync区间并集；strict为±3%，floor为±max(3%×CPU中位秒,100ms)。下表shape Δ%按P10/P50/P90/P100顺序，shape通过数按严格/floor列出。

|单元|Wall Meta/CPU s|Wall 配对Δ%|Wall严格/floor|Shape 配对Δ%|Shape严格/floor|
|---|---:|---:|---|---|---|
|1B/4r/DCP|13.137/13.100|+0.58|通过/通过|+1.15/+0.66/+0.48/+0.00|4/4 → 4/4|
|1B/4r/Safe|12.422/12.500|-0.56|通过/通过|+0.34/-0.79/-0.23/+0.00|4/4 → 4/4|
|1B/8r/DCP|13.292/13.294|+0.62|通过/通过|+0.01/+0.26/+1.99/+2.04|4/4 → 4/4|
|1B/8r/Safe|12.642/12.482|+1.83|通过/通过|-0.97/+0.19/+3.01/+3.23|2/4 → 4/4|
|3B/4r/DCP|40.147/40.225|-0.30|通过/通过|-2.33/-2.07/-0.66/-0.81|4/4 → 4/4|
|3B/4r/Safe|38.649/38.677|-0.01|通过/通过|-1.87/+0.64/-0.33/+0.00|4/4 → 4/4|
|3B/8r/DCP|42.380/42.389|-0.05|通过/通过|+0.67/-1.48/+10.60/+10.62|2/4 → 2/4|
|3B/8r/Safe|40.082/40.132|-0.11|通过/通过|-0.93/-0.60/+8.67/+8.66|2/4 → 2/4|

全部40项门禁的秒数、毫秒差及容限见[gate-table.md](gate-table.md)。[独立审核](independent-gate-audit.json)从原始rank JSON重算wall，并从80条聚合bins独立重算shape；没有调用原分析的配对或CDF函数。

与fix1串行writer-limit=1相比，含fsync wall的加速比（串行中位秒/并发中位秒，>1为更快）为0.943–1.335×。3B各格为1.171–1.335×，wall缩短14.6–25.1%；1B DCP为1.057–1.097×；1B safetensors为0.943–0.959×，wall增加4.2–6.1%。两轮各自5次Meta/CPU配对，跨轮比值是描述性比较，没有将两轮样本当成时间配对。以下每格M/C分别代表Meta/CPU，P100已扣全局fsync并集。

|单元|Wall 串行→并发 s（M / C）|Wall加速 M/C|P100 串行→并发 s（M / C）|P100加速 M/C|
|---|---|---:|---|---:|
|1B/4r/DCP|14.374→13.137 / 14.366→13.100|1.094×/1.097×|6.350→2.300 / 6.400→2.300|2.761×/2.783×|
|1B/4r/Safe|11.848→12.422 / 11.786→12.500|0.954×/0.943×|3.900→1.400 / 3.900→1.400|2.786×/2.786×|
|1B/8r/DCP|14.047→13.292 / 14.109→13.294|1.057×/1.061×|10.150→2.500 / 10.150→2.450|4.060×/4.143×|
|1B/8r/Safe|12.003→12.642 / 11.974→12.482|0.949×/0.959×|6.300→1.600 / 6.200→1.550|3.937×/4.000×|
|3B/4r/DCP|51.630→40.147 / 51.793→40.225|1.286×/1.288×|18.900→18.250 / 18.750→18.550|1.036×/1.011×|
|3B/4r/Safe|46.834→38.649 / 47.289→38.677|1.212×/1.223×|13.700→16.750 / 13.800→16.600|0.818×/0.831×|
|3B/8r/DCP|56.321→42.380 / 56.599→42.389|1.329×/1.335×|27.550→12.300 / 27.700→11.300|2.240×/2.451×|
|3B/8r/Safe|46.917→40.082 / 47.352→40.132|1.171×/1.180×|18.300→14.500 / 18.500→12.700|1.262×/1.457×|

串行save严格/floor均为wall 8/8、shape 32/32；并发的完整分位数前后秒值、加速比及Meta−CPU对齐变化百分点见[serial-shape-comparison.csv](serial-shape-comparison.csv)，wall/压缩span/逐rank区间汇总见[serial-comparison.md](serial-comparison.md)和[serial-comparison.json](serial-comparison.json)。

下表为每配置五次中位数：pwrite/fsync最大重叠rank数以M/C列出；peer重叠占比=各rank与任意其他rank同类API区间相交时长之和/各rank该API时长之和。串行两类峰值均为1、peer占比均为0，pwrite与fsync跨rank交集为0。并发统计保留实际区间交集，不能解释为块设备并发。

|单元|并发pwrite峰值 M/C|并发fsync峰值 M/C|pwrite peer重叠% M/C|fsync peer重叠% M/C|pwrite∩fsync并集 s M/C|
|---|---:|---:|---:|---:|---:|
|1B/4r/DCP|4/4|4/4|73.9/74.9|99.8/99.8|0.082/0.046|
|1B/4r/Safe|4/4|4/4|84.9/86.3|99.8/99.8|0.069/0.072|
|1B/8r/DCP|8/8|8/8|85.0/82.1|99.9/99.6|0.364/0.578|
|1B/8r/Safe|8/8|8/8|95.6/95.1|99.8/99.9|0.236/0.258|
|3B/4r/DCP|4/4|4/4|96.6/97.5|98.7/99.1|2.085/1.926|
|3B/4r/Safe|4/4|4/4|98.3/98.1|98.6/97.9|2.350/2.490|
|3B/8r/DCP|8/8|8/8|98.4/98.0|96.8/98.0|12.363/13.752|
|3B/8r/Safe|8/8|8/8|99.4/98.5|98.9/98.5|7.853/9.617|

所有960个逐rank/round样本（两轮各480个）的pwrite、fsync、同类peer重叠和交叉重叠保存在[per-rank-overlap.csv](per-rank-overlap.csv)。精确绝对半开区间位于overlap-intervals/，串行原始trace SHA256逐份核对；准入与实际pwrite重叠分别记录。

剩余失败（按单元合并同源门禁，严格及floor分别列出）：

- 1B/8r/Safe：严格 shape-p90,shape-p100；floor 无。P100中位配对r4末payload关键rank Meta/CPU=5/6。原因：shape-p90 +46.2ms/shape-p100 +50.0ms；首payload -16.4ms、P10–P90跨度 +32.0ms；P100中位配对r4：末payload相对各自本次最早checkpoint的完成Δ+31.7ms，减去此前fsync并集删除量Δ-10.8ms，得到压缩末payloadΔ+42.5ms（该配对P100分bin差+50.0ms）；尾部shape差与跨rank pwrite/fsync重叠及压缩边界差相符；API spans含内核等待和调度，未唯一归因到设备服务时间。
- 3B/8r/DCP：严格 shape-p90,shape-p100；floor shape-p90,shape-p100。P100中位配对r4末payload关键rank Meta/CPU=5/5。原因：shape-p90 +1195.9ms/shape-p100 +1200.0ms；首payload +69.2ms、P10–P90跨度 +1126.7ms；P100中位配对r4：末payload相对各自本次最早checkpoint的完成Δ-655.8ms，减去此前fsync并集删除量Δ-1860.0ms，得到压缩末payloadΔ+1204.3ms（该配对P100分bin差+1200.0ms）；尾部shape差与跨rank pwrite/fsync重叠及压缩边界差相符；API spans含内核等待和调度，未唯一归因到设备服务时间。
- 3B/8r/Safe：严格 shape-p90,shape-p100；floor shape-p90,shape-p100。P100中位配对r5末payload关键rank Meta/CPU=0/3。原因：shape-p90 +1099.0ms/shape-p100 +1100.0ms；首payload +70.1ms、P10–P90跨度 +1143.5ms；P100中位配对r5：末payload相对各自本次最早checkpoint的完成Δ-41.9ms，减去此前fsync并集删除量Δ-1117.1ms，得到压缩末payloadΔ+1075.3ms（该配对P100分bin差+1100.0ms）；尾部shape差与跨rank pwrite/fsync重叠及压缩边界差相符；API spans含内核等待和调度，未唯一归因到设备服务时间。

[critical-rank-breakdown.csv](critical-rank-breakdown.csv)保留每个失败单元全部repeat的max-wall、最晚writer、首payload及末payload rank，以及pre-payload/admission/payload/materialize/sink/post-wait/uncovered/fsync/other互斥分区；每行可重构该rank wall。shape失败另保留onset、P10–P90跨度及各分位配对证据。未把API span直接归因于磁盘服务时间；各分项跨repeat中位数不能相加重构wall。

所有图从共享时间轴删除全rank fsync并集。若一个rank的payload与另一个rank的fsync重叠，该段byte映射到压缩轴上的瞬间，保留总byte；因此曲线是压缩时间下的聚合API进度，不是物理设备带宽。每格图选取配对平均压缩时长最接近五次中位数的同一个repeat，Meta/CPU保持该配对；两轮图各自使用本轮的代表repeat。

[串行 vs 并发总图](plots/serial-vs-concurrent-save.png)（[PDF](plots/serial-vs-concurrent-save.pdf)）。8格Meta/CPU吞吐图：

- 1B/4r/DCP：[PNG](plots/1b-4r-dcp-save.png) / [PDF](plots/1b-4r-dcp-save.pdf)
- 1B/4r/Safe：[PNG](plots/1b-4r-safetensors-save.png) / [PDF](plots/1b-4r-safetensors-save.pdf)
- 1B/8r/DCP：[PNG](plots/1b-8r-dcp-save.png) / [PDF](plots/1b-8r-dcp-save.pdf)
- 1B/8r/Safe：[PNG](plots/1b-8r-safetensors-save.png) / [PDF](plots/1b-8r-safetensors-save.pdf)
- 3B/4r/DCP：[PNG](plots/3b-4r-dcp-save.png) / [PDF](plots/3b-4r-dcp-save.pdf)
- 3B/4r/Safe：[PNG](plots/3b-4r-safetensors-save.png) / [PDF](plots/3b-4r-safetensors-save.pdf)
- 3B/8r/DCP：[PNG](plots/3b-8r-dcp-save.png) / [PDF](plots/3b-8r-dcp-save.pdf)
- 3B/8r/Safe：[PNG](plots/3b-8r-safetensors-save.png) / [PDF](plots/3b-8r-safetensors-save.pdf)

定向测试181项runner/admission及8项分析测试通过，F lint通过；绘图只读缓存bins，9图PNG/PDF，布局审核全部通过。测量前空闲窗口CPU忙碌3.4%、本地ext4磁盘忙碌0.45%；测量用时54.6分钟，期间没有测试或分析，80个进程组无时间重叠。65份测量源码hash前后一致；43919份旧validation文件大小/mtime及4份fix1基线SHA256未变。全程无git操作、NFS或系统设置变更，80份checkpoint均已清理。证据：[progress.json](progress.json)、[环境](environment.json)、[运行审核](runtime-audit.json)、[布局审核](plot-layout-audit.json)、[分析及provenance](analysis.json)。
