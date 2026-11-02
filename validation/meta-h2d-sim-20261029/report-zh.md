# Meta pageable H2D 模拟验证

已完成 99 个 safetensors 阶段及 18 个 1B DCP 阶段，全部原始测量保留；全部验收门禁通过。

实验为单 rank、本地盘、BF16、zero、每 rank 单文件，按配置交错重复三轮；7B CPU 按内存限制跳过。GPU 为 Tesla V100-SXM2-32GB，使用 pageable staging。save 时间及所有 save 图扣除 checkpoint 内实际 fsync 区间并集；cold/warm load 为完整 checkpoint 调用。输出目录为 [/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029)。

大模型 zero 测量使用 `--content-audit none`，检查完整 load 覆盖，不增加全量数值扫描；两种格式的 tiny zero/position 测试提供数值正确性检查。

## GPU H2D 位置与模拟设计

pageable GPU load 每个 range 先完成 payload 读取（safetensors 为 `pread`，DCP 为 buffered `read`）并拼接读入字节，随后调用 compute destination：callback 把字节复制到匿名映射的 host sink，然后执行 `target.copy_(staged, non_blocking=False)`。H2D 使用 host clock 计时，copy 返回后再 `MADV_DONTNEED` 丢弃 staging 页，之后才开始下一 range；没有额外的 device-wide synchronize。save 的阻塞 D2H 则发生在 callback 写文件之前。无界 pinned staging 的顺序相同但保留 pinned 页；有界 double buffer 使用非阻塞 copy 和每槽位 CUDA event 等待，可将 H2D 与后续 host_sink chunk 重叠，但整段 payload 读取仍在 compute destination 前完成，进入下一 range 前须等待全部 CUDA event。当前 pageable profile 不覆盖这些模式。

Meta 原有 callback 已写入全部 host sink 字节，再按 range 丢弃页，因而已承担与 pageable GPU 对应的可写页 first-touch 成本。新增 H2D 模拟精确放在 callback 后、值验证及页丢弃之前，记录相同 file/offset/bytes/tensor 身份的 `op=h2d, timing=simulated`。真实 GPU H2D 此时读取已驻留的源页，不需要再制造 demand-zero fault；该映射随后立即丢弃，复用前全部重写，因此模拟只等待拟合的阻塞 copy 时长，不额外扫描源内存。这是实测主机的时间模型，不模拟 CUDA bounce buffer、DMA 或 CPU cache 事务。

继续使用 `METASL_GDTENSOR_META_SIM_D2H=pageable`，一个设置同时启用 save D2H 和 load H2D，避免两条路径的开关状态分裂。`METASL_GDTENSOR_META_SIM_H2D_PROFILE=PATH` 可替换 profile，不改变启用条件。模拟位于公共 compute accessor，因此 DCP 与 safetensors 共用实现。设计及复现命令见 [meta-h2d-simulation.md](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/docs/meta-h2d-simulation.md)。

## Profile 与独立重复验证

冻结的 affine profile 为：

```text
H2D 秒 = 0.0001260831091564399 + bytes / 4581911166.565587
```

即固定开销 126.083 μs、带宽 4.581911 GB/s。拟合使用历史 1B/3B/7B safetensors 与 1B pageable DCP 的 cold/warm load，共 24 条 GPU trace、7,056 个 range、14 种尺寸，范围 512 B 至 1,089,994,752 B。按完整 repeat 留一，跨模型、格式、冷热阶段同时排除同轮数据；本轮验收数据未参与调参。affine 已通过门禁，无需尺寸节点，接口保留 knots 支持。

|历史评估|绝对误差中位数 ms|绝对误差 p95 ms|593.5 MiB 平均绝对相对误差|1039.5 MiB 平均绝对相对误差|
|---|---:|---:|---:|---:|
|拟合样本|0.0721|0.2655|0.7944%|1.5431%|
|leave-repeat-out|0.0717|0.2720|0.8503%|1.6221%|

历史分阶段/模型的大 range leave-repeat-out 最差值为 safetensors 7B warm load 的 1039.5 MiB range，平均绝对相对误差 2.3594%。证据：[profile.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/profile/profile.json)、[cross-validation.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/profile/cross-validation.json)。

## NT helper 编译修复

accessor 构造和 load 路径均不再 prepare 或加载 NT helper。`GDTensorSavePlanner` 保留原有计时前调用 `population_function` 的预热；native `.so` 使用用户私有临时目录持久缓存，key 包括 C 源码 hash、平台 ABI 和编译参数，每个进程只加载一次。直接 accessor 使用者可在自己的计时前调用新增的 `prepare_save()`；兼容旧调用的 save lazy fallback 仍可在未预热时支付首次初始化成本，但编译不计入 H2D/D2H event。

本轮预热命中缓存耗时 0.000586 s，明确在 checkpoint timer 外，见 [helper-preparation.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/helper-preparation.json)。之前 save 已在 API timer 前进行 planner 准备，因此旧 save 数字保持原样，不能从旧 save 时间扣除约 0.48 s；本轮 save 与 GPU 的配对结果承担回归检查，历史前后时间仅作未配对参考。

<!-- MEASURED_RESULTS_BEGIN -->
## 实测门禁

总体验收：全部通过。

表内各配置为三轮独立中位数；验收列先按相同 repeat 计算 `100 × (Meta+sim − GPU) / GPU`，再取三轮中位数。例如 1B save 的两个独立中位数为 4.5454/4.6888 s，验收使用配对中位差 −2.5103%，不使用两个独立中位数相减后的百分比。save 均已去 fsync，单位 s。

|格式 / 模型|阶段|Meta|Meta+sim|GPU|CPU|配对差 s|配对差 %|±3%|
|---|---|---:|---:|---:|---:|---:|---:|---|
|safetensors 1B|save|2.893|4.545|4.689|3.339|-0.118|-2.510|通过|
|safetensors 1B|cold-load|11.484|11.351|11.352|11.365|+0.010|+0.090|通过|
|safetensors 1B|warm-load|3.628|4.220|4.307|2.697|-0.113|-2.613|通过|
|safetensors 3B|save|15.430|18.117|18.400|19.148|-0.238|-1.296|通过|
|safetensors 3B|cold-load|38.004|38.192|38.194|38.142|-0.012|-0.032|通过|
|safetensors 3B|warm-load|12.089|13.637|13.879|9.811|-0.100|-0.722|通过|
|safetensors 7B|save|59.369|62.546|63.338|—|-0.792|-1.250|通过|
|safetensors 7B|cold-load|85.244|85.628|85.630|—|-0.032|-0.038|通过|
|safetensors 7B|warm-load|26.067|29.495|28.745|—|+0.777|+2.702|通过|
|dcp 1B|save|—|6.004|6.151|—|-0.132|-2.151|仅参考|
|dcp 1B|cold-load|—|11.473|11.484|—|-0.011|-0.095|通过|
|dcp 1B|warm-load|—|4.859|4.834|—|+0.036|+0.750|通过|

DCP save 是 load fixture，时间仅供参考，不纳入 ±3% 耗时验收；safetensors save 与两种格式的 cold/warm load 均须通过对应耗时门禁。

## 新测量的逐 range H2D 误差

按 phase/repeat 和精确 file/offset/bytes 身份匹配实际模拟事件与 GPU 事件。绝对误差单位 ms；median 门限 ≤0.5 ms，大 range（≥512 MiB）平均绝对相对误差门限 ≤3%，同时检查 pooled 与每种大尺寸。

|格式 / 模型|range 数|绝对误差 median|绝对误差 p95|大 range 平均绝对相对误差|门禁|
|---|---:|---:|---:|---:|---|
|safetensors 1B|1206|0.0557|0.1453|无大 range|通过|
|safetensors 3B|2610|0.0743|0.1773|1.117%|通过|
|safetensors 7B|2034|0.0745|0.3645|1.173%|通过|
|dcp 1B|1206|0.0628|0.1698|无大 range|通过|

|格式 / 模型|range MiB|样本数|平均绝对相对误差|≤3%|
|---|---:|---:|---:|---|
|safetensors 3B|593.5|12|1.117%|通过|
|safetensors 7B|1039.5|12|1.173%|通过|

## 请求序列门禁

比较范围为 recorder 观测的完整请求与 payload API 子序列，覆盖所有配置及三轮重复，包括 Meta+sim 的 cold/warm load。请求身份不比较时间戳/耗时；staging 和 fsync/flush/close 排除。时钟仅用于验证串行调用顺序。直接 header guard 与 manifest 发布不在该请求范围内，因此本结论不宣称底层物理 I/O 全等。

|格式 / 模型|阶段|参与 trace 数|完整请求|payload 子序列|
|---|---|---:|---|---|
|safetensors 1B|save|12|identical|identical|
|safetensors 1B|cold-load|12|identical|identical|
|safetensors 1B|warm-load|12|identical|identical|
|safetensors 3B|save|12|identical|identical|
|safetensors 3B|cold-load|12|identical|identical|
|safetensors 3B|warm-load|12|identical|identical|
|safetensors 7B|save|9|identical|identical|
|safetensors 7B|cold-load|9|identical|identical|
|safetensors 7B|warm-load|9|identical|identical|
|dcp 1B|save|6|identical|identical|
|dcp 1B|cold-load|6|identical|identical|
|dcp 1B|warm-load|6|identical|identical|

## 首个 payload 前耗时与历史 save 对照

下表为 checkpoint 调用开始至首个 payload pread 开始，单位 s；括号外为三轮中位数，原始样本按 r1/r2/r3 列出。配对差为 Meta+sim − Meta。

|模型|阶段|Meta 中位数|Meta+sim 中位数|配对差中位数|Meta 三轮|Meta+sim 三轮|
|---|---|---:|---:|---:|---|---|
|1B|cold-load|0.359|0.352|-0.008|0.350 / 0.359 / 0.359|0.354 / 0.345 / 0.352|
|1B|warm-load|0.354|0.352|+0.004|0.337 / 0.355 / 0.354|0.351 / 0.358 / 0.352|
|3B|cold-load|1.019|1.023|+0.010|1.027 / 1.013 / 1.019|1.001 / 1.023 / 1.043|
|3B|warm-load|1.029|1.018|+0.008|1.029 / 1.029 / 0.993|1.040 / 1.018 / 1.001|
|7B|cold-load|0.724|0.719|-0.001|0.724 / 0.714 / 0.726|0.723 / 0.719 / 0.718|
|7B|warm-load|0.723|0.728|+0.005|0.737 / 0.699 / 0.723|0.730 / 0.726 / 0.728|

这些结果与 load 路径不再准备/加载 NT helper 的目标测试共同验证：Meta+sim 不再承担此前约 0.48 s 的额外初始化。下列历史 save 对照未配对，不能把变化单独归因于 helper 缓存或编译修复，原始历史时间未扣除 0.48 s。

|模型|历史 Meta+sim save 去 fsync|当前|变化 %|
|---|---:|---:|---:|
|1B|4.530|4.545|+0.33|
|3B|18.385|18.117|-1.46|
|7B|62.633|62.546|-0.14|

## 图与完整数据

各图使用短英文标题，图上方图例标注配置及该代表轮总秒数，无副标题、页脚或阴影。load 比较 Meta+sim / GPU / CPU（7B 省略 CPU）；save 图完全剔除 fsync 时间。代表轮为各配置最接近三轮去 fsync 中位数的一轮，50 ms bin 按事件重叠时间分配 payload bytes，字节总量守恒。

safetensors 1B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-warm-load.png)。
safetensors 3B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-warm-load.png)。
safetensors 7B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-warm-load.png)。
dcp 1B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-warm-load.png)。

[完整分析 JSON](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/analysis.json)、[分析报告](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/report-zh.md)。
<!-- MEASURED_RESULTS_END -->

## 测量保留、恢复与检查

原始慢样本完整保留：3B CPU 第 1 轮 warm load 约 23.72 s、7B Meta 第 1 轮 warm load 约 60.11 s、7B GPU 第 1 轮 warm load 约 33.64 s。没有删除样本、挑选替代值或为结果重新测量；没有仅凭总耗时把它们归因于 pread。

DCP 第一个 case 的三个计时阶段已完成，随后后处理因错误假设存在 safetensors 的 `.metasl-complete.json` 而失败。runner 已改为归档 DCP `.metadata`，从原始报告恢复该 case，未重测、未替换任何时间样本。首个 case 的检查点已清理，未保留 `.metadata` 二进制副本；恢复依据为既有阶段报告、trace、缓存和文件尺寸记录；计时 runtime 保持不变，见 [runner-fix.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/runner-fix.json)。所有 117 个阶段期间 runtime 源码保持冻结；supervisor 的测后变更仅更新帮助文字，说明旧开关同时适用于 save D2H/load H2D，见 [measurement-audit.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/measurement-audit.json)、[help-update.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/help-update.json)。

首次离线分析在处理 DCP save 时因导出未保留 `io_layer` 而把 1,005 条 container-patch pwrite 混入 payload 子序列，覆盖检查失败。修复后按 file/op/offset/requested/completed/绝对 start_ns 精确匹配原始 payload layer；原始完整请求导出仍保留 patch，未修改任何测量证据。离线比较同时改为每条 trace 只验证、解析一次并缓存精确 signature tuple，由公共比较函数复用，保留时钟、失败、来源 hash 和首次差异检查；每个完成模型立即持久化，`--resume` 要求输入清单 hash、源码 hash、设置、模型及格式一致。第二次离线分析发现 DCP load 的 payload 使用 `read`/`api=buffered_stream_read`，原 safetensors 专用 `pread` 过滤器选不到事件；独立扫描全部 18 条 DCP trace：save/cold/warm 各 6 条、各 201,678 个 payload 调用，分别为 pwrite/read/read；短读、EOF 和失败 payload 均为 0。修复仅使分析按格式选择正确操作；三个已验证的 safetensors 缓存先归档，再经源码差异、输入/设置及 comparator 一致性检查迁移，entry hash 完全不变，9 个 safetensors 图的 JSON hash 亦与首次分析相同。证据：[analysis-dcp-read-fix.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis-dcp-read-fix.json)。原始测量及 safetensors 结论均未改变。这些均为测量结束后的只读分析修复，与计时 runtime 冻结相互独立，见 [analysis-correction-checks.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis-correction-checks.json)。

目标测试：runtime 113 项、profile 17 项、修复后分析 14 项、trace export 37 项通过，共 181 个不同测试。runtime 首次有四个格式测试受到 sandbox Gloo 本地地址限制，使用 `GLOO_SOCKET_IFNAME=lo` 在允许本地 socket 的执行环境中重试通过；测试与测量未并行。代码 Ruff 检查通过。证据：[runtime-tests.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/runtime-tests.json)、[profile-tests.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/profile-tests.json)、[analysis-tests.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis-tests.json)。

safetensors 和 DCP checkpoint 目录均已清空，见 [cleanup.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/cleanup.json)、[DCP cleanup.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/dcp/cleanup.json)。历史结果目录、NFS 配置和系统设置未修改，未执行 git 操作。

## 文件变更

|文件|变更|
|---|---|
|[metasl/gdt_storage.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/metasl/gdt_storage.py)|公共 Meta load H2D 模拟、事件记录、save helper 显式准备|
|[metasl/meta_d2h_cache.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/metasl/meta_d2h_cache.py)|源码 hash/ABI/参数键控的持久 native helper 缓存|
|[metasl/meta_h2d_profile.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/metasl/meta_h2d_profile.py)|新增默认 H2D profile 及评估接口|
|[metasl/trace_export.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/metasl/trace_export.py)|测量后新增一次验证/解析的 prepared trace 比较接口，保留原有精确比较语义|
|[metasl/supervisor.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/metasl/supervisor.py)|测量完成后仅更新模拟开关帮助文字|
|[scripts/run_gdt_safetensors.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/scripts/run_gdt_safetensors.py)|复用串行测量 runner，支持格式/配置选择、layout 检查、等长路径和 DCP metadata 归档|
|[scripts/fit_pageable_h2d.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/scripts/fit_pageable_h2d.py)|新增历史实测拟合、leave-repeat-out 和来源证据|
|[scripts/analyze_meta_h2d.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/scripts/analyze_meta_h2d.py)|新增配对门禁、全请求序列比较、逐 range 误差及绘图|
|[tests/test_gdtensor_accessor.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/tests/test_gdtensor_accessor.py)|扩展模拟顺序、事件、异常清理和 helper 缓存测试|
|[tests/test_meta_h2d_load.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/tests/test_meta_h2d_load.py)|新增两种格式、zero/position load、请求身份测试|
|[tests/test_meta_h2d_profile.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/tests/test_meta_h2d_profile.py)|新增 profile 拟合、参数、节点及评估测试|
|[tests/test_meta_h2d_analysis.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/tests/test_meta_h2d_analysis.py)|新增配对百分比、H2D range 匹配及模拟 load 序列覆盖测试|
|[tests/test_trace_export.py](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/tests/test_trace_export.py)|新增 prepared 比较等价及无需重新打开来源文件的测试|
|[docs/meta-h2d-simulation.md](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/docs/meta-h2d-simulation.md)|新增机制、适用范围、缓存及复现说明|
|[docs/meta-d2h-simulation.md](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/docs/meta-d2h-simulation.md)|同步双向模拟与 NT helper 生命周期说明|

最终核对：[图表与字节守恒](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/final-plot-checks.json)、[源码与 Ruff](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/final-source-checks.json)、[测量与清理](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/final-measurement-checks.json)、[14 项分析测试](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis-correction-checks.json)、[37 项请求导出测试](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/prepared-comparison-tests.json)、[progress.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/progress.json)、[completion.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/completion.json)。
