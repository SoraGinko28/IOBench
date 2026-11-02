# Meta H2D 本地验证

单 rank、BF16、zero、每 rank 单文件、三次交错重复。时间单位 s；save 扣除 checkpoint 内实际 fsync 区间并集，load 使用完整 checkpoint 调用。配对百分比先计算每轮 (Meta+sim − GPU) / GPU，再取中位数。

|格式 / 模型|阶段|Meta|Meta+sim|GPU|CPU|配对差 s|配对差 %|±3%|
|---|---|---:|---:|---:|---:|---:|---:|---|
|safetensors 1B|save|2.893|4.545|4.689|3.339|-0.118|-2.51|通过|
|safetensors 1B|cold-load|11.484|11.351|11.352|11.365|+0.010|+0.09|通过|
|safetensors 1B|warm-load|3.628|4.220|4.307|2.697|-0.113|-2.61|通过|
|safetensors 3B|save|15.430|18.117|18.400|19.148|-0.238|-1.30|通过|
|safetensors 3B|cold-load|38.004|38.192|38.194|38.142|-0.012|-0.03|通过|
|safetensors 3B|warm-load|12.089|13.637|13.879|9.811|-0.100|-0.72|通过|
|safetensors 7B|save|59.369|62.546|63.338|—|-0.792|-1.25|通过|
|safetensors 7B|cold-load|85.244|85.628|85.630|—|-0.032|-0.04|通过|
|safetensors 7B|warm-load|26.067|29.495|28.745|—|+0.777|+2.70|通过|
|dcp 1B|save|—|6.004|6.151|—|-0.132|-2.15|仅参考|
|dcp 1B|cold-load|—|11.473|11.484|—|-0.011|-0.09|通过|
|dcp 1B|warm-load|—|4.859|4.834|—|+0.036|+0.75|通过|

DCP save 用于生成 load 的 checkpoint，时间仅供参考；DCP 的耗时验收针对 cold/warm load，safetensors save 仍须通过 ±3% 门禁。

H2D 按 phase / repeat / file / offset / bytes 精确匹配实际模拟与 GPU 事件；下表为所有 cold/warm range，误差单位 ms。大 range 为 ≥512 MiB。

|格式 / 模型|range 数|绝对误差中位数|绝对误差 p95|大 range 平均绝对相对误差|完整 / payload 请求序列|
|---|---:|---:|---:|---:|---|
|safetensors 1B|1206|0.056|0.145|无 ≥512 MiB range|全部相同|
|safetensors 3B|2610|0.074|0.177|1.12%|全部相同|
|safetensors 7B|2034|0.074|0.365|1.17%|全部相同|
|dcp 1B|1206|0.063|0.170|无 ≥512 MiB range|全部相同|

|格式 / 模型|range MiB|样本数|平均绝对相对误差|≤3%|
|---|---:|---:|---:|---|
|safetensors 3B|593.5|12|1.12%|通过|
|safetensors 7B|1039.5|12|1.17%|通过|

## 首个 payload 前耗时

从 checkpoint 调用开始到首个 payload pread 开始；单位 s。

|模型|阶段|Meta|Meta+sim|配对差中位数|
|---|---|---:|---:|---:|
|1B|cold-load|0.359|0.352|-0.008|
|1B|warm-load|0.354|0.352|+0.004|
|3B|cold-load|1.019|1.023|+0.010|
|3B|warm-load|1.029|1.018|+0.008|
|7B|cold-load|0.724|0.719|-0.001|
|7B|warm-load|0.723|0.728|+0.005|

## 保存历史参考

与上一轮单文件实验未配对，包含主机/磁盘噪声，不能仅归因于 helper 预编译。

|模型|Meta+sim 原 save 去 fsync|当前|变化 %|
|---|---:|---:|---:|
|1B|4.530|4.545|+0.33|
|3B|18.385|18.117|-1.46|
|7B|62.633|62.546|-0.14|

Profile 拟合及 leave-repeat-out：[cross-validation.json](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/profile/cross-validation.json)。

## 图

safetensors 1B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/1b-warm-load.png)。
safetensors 3B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/3b-warm-load.png)。
safetensors 7B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/7b-warm-load.png)。
dcp 1B：[save](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-save.png)、[cold-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-cold-load.png)、[warm-load](/data00/home/wutingjun/AIStore/Meta-Device-DCP-Save-Load/validation/meta-h2d-sim-20261029/analysis/dcp/1b-warm-load.png)。

7B CPU 按内存限制跳过。图选择各配置最接近三轮去 fsync 中位数的一轮，50 ms 等宽 bin 按事件重叠时间分摊 payload bytes，字节总量守恒。load 图含 Meta+sim / GPU / CPU（7B 无 CPU），save 图完全排除 fsync 时间。所有配置及重复都参与完整和 payload 请求序列精确比较；staging 与 fsync/flush/close 不属于请求身份。
