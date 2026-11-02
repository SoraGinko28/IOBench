# 仓库精简记录

范围：单机 FSDP2、Meta/CPU/CUDA、gdtensor DCP/safetensors、save/cold/warm load。未执行任何 Git 操作，未修改 NFS 或系统设置；原有四个 validation 目录仅被读取。

## 数量

|目录|文件：前 → 后|源码/文档行：前 → 后|
|---|---:|---:|
|metasl/|63 → 26|24,863 → 7,532|
|scripts/|71 → 6|41,203 → 2,292|
|tests/|90 → 26|25,926 → 6,074|
|docs/|93 → 3|12,555 → 234|

行数统计 Python/C/Markdown/JSON/TXT，不含 SVG/PNG 等图形资产；文件数包含这些资产，忽略 __pycache__。

## 删除与保留

- 删除不可达模块及脚本、Megatron、20-sample/E1–E6 外层、旧 IO matrix、longrun/HDFS、训练态/AdamW/RNG、hybrid、interleaved/continuous IO、retention、旧文档与无用 configs/vendor/integration_provenance/PLAN。
- 收窄保留模块：metasl.run 仅启动 model save/load；phase/fsdp/skeleton/scratch/checkpoint_state 删除 native 后端分支、训练态和派生 TP/PP；删除旧 commit-time container patch switch 和诊断 hook，只保留 early patch；删除 deferred/component trace 合并。
- 先记录了核心 runner 实际传参，再删除旧选项，见 [core-cli-options.md](core-cli-options.md)。
- 保留 admission/内存与 metadata 门控，因为多 rank runner 确实使用；保留共享 PyTorch 存储基类的 planning/metadata/manifest 和 plain model buffer 处理，因为 gdtensor 继承它们；native 后端不可通过 CLI 选择。
- prepare/preload 是静态扫描遗漏的子进程入口；恢复 forkserver GC freeze。保留轻量 safetensors 分桶选项、position 校验、低层 CUDA staging/Meta population 控制，避免改变当前 accessor 的兼容和缓存行为。
- 依赖均仍为核心所需；numpy 是 Tensor.numpy 和 gdtensor 的运行依赖，matplotlib 用于绘图。统一 requirements，pyproject 明确 Python 3.12、torch/gdtensor 和核心依赖。
- README 已改为简短中文；docs 仅保留当前集成、D2H、H2D 三篇。configs 仅保留当前 1B/3B/7B 离线模型配置。

## 统一入口

```bash
python scripts/analyze_gdtensor_traces.py RUN [OTHER_RUN] --output ANALYSIS
python scripts/plot_gdtensor_traces.py ANALYSIS --output PLOTS
python scripts/fit_pageable_d2h_knots.py --input GPU_RUN [OTHER_GPU_RUN] --output NEW_D2H
python scripts/fit_pageable_h2d.py --input GPU_RUN [OTHER_GPU_RUN] --output NEW_H2D
```

分析逐 rank 比较完整及 payload 请求序列，并报告 save（扣除 fsync 并集）、D2H/H2D、payload pwrite/read、fsync 和 cold/warm load。绘图沿用 clean 样式，图例显示配置与总秒数，无副标题/页脚/阴影，save 时间轴排除 fsync。

更多实现说明：[core-changes.md](core-changes.md)、[analysis-changes.md](analysis-changes.md)。

## 验收

|Gate|结果|
|---|---|
|CPU pytest|475 passed，58 skipped|
|CUDA pytest|69 passed|
|全部模块导入 / 入口 --help|25 个模块、6 个核心脚本及 run/prepare/验收入口全通过|
|删除模块引用 / Ruff F|通过|
|1B 单 rank、双格式、四配置 save/cold/warm|24/24 与原始基线完整请求序列 identical|
|1B 四 rank、双格式 Meta/CPU|6 个阶段组、24/24 逐 rank 比较 identical；全部 save/load 完成|
|冷/热缓存与格式布局|全部通过|
|检查点及测试临时文件清理|通过，详见 final-cleanup.json|

机器结果：[gates.json](gates.json)、[request-comparisons.json](request-comparisons.json)、[progress.json](progress.json)。

DCP 初次验收发现仅路径序列化造成的 metadata 长度差：baseline 路径为 128 字节、11 个 PosixPath components，首次测量少一层目录。验收路径修正为完全相同的长度和层数后重新测量，最终包含 metadata 的完整序列精确一致。旧测量隔离在 path-artifact/，未参与最终分析；运行时代码和比较规则未为此改变。


## Meta+sim 与 GPU 耗时

以下为单 rank、zero pattern、各 1 repeat。save 扣除 fsync 并集；load 使用完整 API 耗时。所有配置使用相同的请求序列。多 rank 本次验证请求和完成性，不声称模拟时序已针对并发校准。

|格式|阶段|Meta+sim 秒|GPU 秒|差值 秒|差值 %|
|---|---|---:|---:|---:|---:|
|dcp|cold-load|11.5504|11.4295|+0.1209|+1.06%|
|dcp|save|5.6733|5.5446|+0.1286|+2.32%|
|dcp|warm-load|4.5650|4.4785|+0.0864|+1.93%|
|safetensors|cold-load|11.5149|11.4314|+0.0835|+0.73%|
|safetensors|save|4.5498|4.6196|-0.0698|-1.51%|
|safetensors|warm-load|4.1726|4.1516|+0.0210|+0.50%|

与保留 baseline 中同配置、同阶段的 18 个可比样本相比，耗时变化为 -21.84% 至 +2.53%，量级一致；DCP baseline 未包含普通 Meta/CPU，因此这两个配置只比较原始请求序列。单 repeat 不能用于估计分布。

新分析：[analysis/report.md](analysis/report.md)、[analysis/analysis.json](analysis/analysis.json)；验收细节见 acceptance-report.md。全部 12 组 full/payload 序列一致，包含 72 个 rank case。

全部验收结论和逐阶段秒数见 [acceptance-report.md](acceptance-report.md)、[timing-comparison.json](timing-comparison.json)。已生成 30 张新图（6 张单 rank、24 张多 rank），并目视检查单 rank safetensors save 与四 rank DCP save；图形证据见 [fresh-analysis-checks.json](fresh-analysis-checks.json)。
