1B 本地盘、zero、每配置 1 次。单-rank Meta / Meta+sim / CPU / GPU 与 4-rank Meta / CPU，DCP 和 safetensors 共 12 配置、36 phase、72 rank case，全部完成。

| Gate | 结果 |
|---|---|
| 单-rank 对原始基线完整请求序列 | 24/24 identical，包含 metadata |
| 4-rank Meta/CPU 逐 rank 序列 | 24/24 identical（6 个 phase group） |
| 新数据统一分析 | 12/12 full + payload group identical |
| 冷/热缓存、每 rank 一文件 | 各 12/12 配置通过 |
| 新图表 | 30 张（单-rank 6，多-rank 24），样式目视检查通过 |
| Checkpoint 清理 | 根目录与 payload 文件均不存在 |

直接匹配的 18 项基线耗时比为 0.782–1.025×；原始 DCP 仅保留 Meta+sim/GPU，Meta/CPU 的 DCP 请求使用 Meta+sim 基线比较。保存耗时均扣除 fsync；下表 Δ=(Meta+sim/GPU−1)×100%。

| 格式 | Phase | Meta+sim (s) | GPU (s) | Δ |
|---|---|---:|---:|---:|
| safetensors | save | 4.5498 | 4.6196 | -1.51% |
| safetensors | cold-load | 11.5149 | 11.4314 | +0.73% |
| safetensors | warm-load | 4.1726 | 4.1516 | +0.50% |
| dcp | save | 5.6733 | 5.5446 | +2.32% |
| dcp | cold-load | 11.5504 | 11.4295 | +1.06% |
| dcp | warm-load | 4.5650 | 4.4785 | +1.93% |

[机器结果](acceptance-results.json) · [完整 timing breakdown](timing-comparison.json) · [分析](analysis/analysis.json) · [图表](plots/) · [命令记录](acceptance-commands.json)

初次 DCP 路径分量少一层，造成最终 metadata write 少 2 字节；已保持总长 128 字节并匹配 11 个 PosixPath 分量，清理后重跑全部 DCP。最终比较保留原始请求字段；诊断记录见 [path correction](checkpoint-path-correction.json)。

复跑：`/tmp/metasl-gdt-cuda-venv/bin/python validation/repo-slim-20261029/run_acceptance.py --stage all --resume`，包含 GPU，须 escalated execution。
