# 分析、绘图与 profile 拟合精简

保留两个批量运行器；分析统一为 `scripts/analyze_gdtensor_traces.py`，绘图统一为 `scripts/plot_gdtensor_traces.py`。分析入口递归发现报告和原始 IO JSONL，兼容 measurement-results/results 清单及无清单的双格式、多 rank 目录。不读取 checkpoint payload。每个模型、格式、world size、pattern、位置、阶段分别比较，每个 rank 单独核对完整请求与 payload 请求；缺 rank 或不可验证的顺序保留 unknown。fsync/flush/close 不参与请求身份；DCP header patch 不混入 payload 比较。

统计完整 checkpoint 调用的 total、save 去 fsync、D2H、H2D、payload pwrite/read、fsync，另给 Meta+sim 与 GPU 的逐 repeat、逐 rank 配对差。时间取调用区间并集，避免重叠重复计时。save 的 50 ms 绘图轴扣除 fsync 并集，load 保留完整调用时间；分桶保持 payload bytes 总量，若 payload 与 fsync 重叠则报错而非失真压缩。

```bash
PY=/tmp/metasl-gdt-venv/bin/python
$PY scripts/analyze_gdtensor_traces.py RUN [BASELINE_RUN] --output RUN_ANALYSIS
$PY scripts/plot_gdtensor_traces.py RUN_ANALYSIS --output RUN_PLOTS
# 可用 --models 1b --repeats 1 限定分析范围；绘图支持 --models / --phases / --ranks。
$PY scripts/fit_pageable_d2h_knots.py --input GPU_RUN [GPU_RUN_2] --output NEW_D2H_FIT
$PY scripts/fit_pageable_h2d.py --input GPU_RUN [GPU_RUN_2] --output NEW_H2D_FIT
```

恰好保留一个 D2H 拟合脚本 `fit_pageable_d2h_knots.py` 和一个 H2D 拟合脚本 `fit_pageable_h2d.py`。D2H 延续当前 size knots / weighted isotonic 方法，保持 593.5 MiB 以下 affine 分支，改为直接接收实测 GPU trace，不再依赖已删除的额外 calibration runner；H2D 延续当前 affine 方法。两者共享实测事件选择逻辑，支持两种格式和所有 rank，只使用成功、完整、blocking_copy_host_clock 的 pageable 范围；按整个 repeat 留出，模型/格式/rank 不泄漏到训练集。拟合结果只写新 artifact，不自动修改内置 profile。GPU 本身不参与拟合。

删除：`analyze_gdt_safetensors.py`、`analyze_meta_h2d.py`、`analyze_meta_d2h_sim.py`、`analyze_meta_sim_repeats.py`、`analyze_meta_sim_pwrite.py`、`plot_gdtensor_trace_consistency.py`、`plot_gdtensor_trace_timeline.py`、`plot_gdtensor_diskstats.py`、旧 affine D2H `fit_d2h_profile.py`、独立校准 runner `calibrate_pageable_d2h.py`。其余不可达实验分析脚本由主清理任务删除。原五个专用分析/绘图测试删除，关键时间、分桶、请求/patch 区分、rank 完整性、逐 repeat 配对与 D2H 留出验证合入 `tests/test_gdtensor_trace_analysis.py`。

验证：

- 新分析测试与原 H2D profile 测试：24 passed。
- 四个保留/新增分析、绘图、拟合脚本 `--help` 均通过，均可独立导入。
- 使用只读保留基线的 1B、repeat 1：18 个 case、6 组；DCP 与 safetensors 的 save/cold-load/warm-load，完整与 payload 请求序列全部 identical。输出在 `baseline-analysis/`，复用了验收代理运行的统一分析结果。
- 绘图入口生成 `baseline-plots/` 的 6 张 PNG；已目视检查 safetensors save：短英文标题、图上方配置与总秒数图例，无副标题/页脚/阴影，save 横轴去 fsync。
- D2H 实测 trace 拟合 3528 个范围，输出 `refit-d2h-smoke/`；H2D 7056 个范围，输出 `refit-h2d-smoke/`，H2D 留出 median absolute error 0.0755 ms、大范围 mean absolute relative error 1.16–1.20%。两者均成功完成实测输入与留出验证工作流；未更改默认参数。
- 本子任务没有改动 `metasl/`，没有写四个保留 validation 目录，没有生成 checkpoint。

最终新跑数据：显式输入 single-rank 与 multi-rank 两个目录，统一分析完成 36 个 phase output / 72 个 rank case / 12 组；完整与 payload 序列全部 identical、rank 完整。生成 30 张图（单 rank 6 张、多 rank 逐 rank 24 张），已目视检查单 rank safetensors save 与四 rank DCP save 的 rank 0 图。确切命令与检查见 `fresh-analysis-checks.json`。
