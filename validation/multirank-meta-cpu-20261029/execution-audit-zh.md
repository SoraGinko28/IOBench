# 执行与证据补充核对

主矩阵完整执行：12 个 model × ranks × format 单元，3 repeats × 2 configs × 3 phases，共 **216 次阶段调用、2016 个 rank case**，无跳过、无矩阵执行错误。72 个配置完成标记齐全；Meta/CPU 在每个 repeat 内交替，主矩阵先完成 1B 的全部 ranks，再执行 3B。16 ranks 应在结果中标为 8 核机器上的 oversubscribed，不能与 4/8 ranks 的门禁要求混为一项。

缓存与清理均已核对：主矩阵 **72 次 cold + 72 次 warm** 验证全部通过，cold 数据页驻留率最大 0%，warm 最小 100%；每阶段一文件/rank 的 layout 共检查 216 次。全部 12 个矩阵 checkpoint 根目录已经删除；另有 4 个 pilot 的 checkpoint 清理和 36 次 warm 验证通过。cold 使用逐文件 `POSIX_FADV_DONTNEED` 并检查 guest page cache，没有修改系统缓存设置；VM host cache 不在观测范围。详见 [final-runner-audit.json](final-runner-audit.json)。

全部 4 个 pilot 的 36 条 warm rank 记录分别完整保留。最初仅汇总步骤因比较结果已带 `rank`、再次传同名关键字而失败；测量过程没有失败。进程退出后，修复脚本校验完整行集合及 export SHA，再重做实际逐 rank 请求比较与配对统计。原始 4 条错误、原始 progress 和选择记录保存在 [pilot-summary-correction.json](pilot-summary-correction.json)，仅移除了经验证的同一汇总错误；未重新测量。重建推荐仍为 **faithful**，与实际矩阵配置一致：ring 在 1B/3B DCP pilot 分别为 −6.55%/−5.88%，超出 ±3%。

定向验证按各次真实执行记录报告，不能把重复运行相加：runner/所有权测试 **12 passed、0 skipped**；分析测试 **14 passed**（含绘图失败保留结果及缓存必须显式复用的回归）；测量后 staging 修复的定向回归 **22 passed**，其中包含此前重复通过的 18 项 sink/fresh 回归和 4 项新增测试。相关 F lint 均通过；这些不是全库测试结论。最终测试均在测量进程退出后运行；没有重跑矩阵。命令与结果见 [final-runner-tests.json](final-runner-tests.json)、[analysis-tests.json](analysis-tests.json)、[postmeasure-staging-fix-tests.json](postmeasure-staging-fix-tests.json)。

测量期间的运行时代码一致：独立比较 [source-before.json](source-before.json) 与 [source-after-measurements.json](source-after-measurements.json)，`metasl/*.py` 与实际阶段驱动共 26 个文件的 SHA256 全部相同。之后仅 `metasl/gdt_storage.py` 有已记录的运行时源码变更：修复 compute staging 归属错误文件、从而被发布过滤丢弃的问题；该修复发生于全部测量结束后，不修改历史 raw、墙钟、请求序列或已有矩阵。[postmeasure-staging-fix-audit.json](postmeasure-staging-fix-audit.json) 保存测量版与修复版 SHA 和 tiny 两 rank CPU 集成证据。

历史测量的 sink 缺口仍然存在：1344 个 load rank case 中，仅 144 个有完整 `host_sink` spans，1200 个必须保留 unknown 与保守上界，不能因源码已修好而填成 0 或推定精确耗时。warm 门禁残差只定位到有证据的读取、规划、同步等待及未覆盖区间；没有逐函数 profile 的部分继续标明限制，见 [warm-gap-diagnosis.md](warm-gap-diagnosis.md)。此项影响 per-rank sink 精确拆分，不改变已保存的实际 IO 请求或 phase 墙钟。

改动清单共 12 个源码/测试/文档文件，另外的诊断及测量产物均在本次新 validation 目录：

- 运行时：`metasl/gdt_storage.py`、`metasl/gdt_hf_storage.py`、`metasl/phase.py`。
- 驱动与分析：`scripts/run_gdt_safetensors.py`、`scripts/run_multirank_meta_cpu.py`、`scripts/analyze_multirank_meta_cpu.py`、`scripts/repair_multirank_pilot_summaries.py`。
- 测试：`tests/test_meta_cpu_target_sink.py`、`tests/test_multirank_runner.py`、`tests/test_multirank_alignment.py`、`tests/test_gdtensor_staging_slots.py`。
- 文档：`README.md`。

Git 规则有一项应披露的初始偏差：主代理在读 brief 之前执行过一次只读 `git status`；读取 brief 后没有再执行 Git 命令，也没有 Git 修改操作。[final-source-audit.json](final-source-audit.json) 和 [changed-files.json](changed-files.json) 已记录此项。旧 validation 目录作为只读证据来源；本次执行没有 NFS、GPU 测量或系统参数修改。

最终分析恢复完整：2016 rank cases、0 errors；完整与 payload 请求各 1008 个逐 rank 配对比较全部相同，全部 216 个聚合时间线字节守恒、全局 fsync 压缩及配对门限算术经独立复核。36 张阶段图加 1 张汇总图均有 PNG/PDF，共 74 文件，格式与数量检查通过。首次绘图因测量环境缺少 matplotlib 退出；恢复使用原 CSV、带原始 payload 校验和的 rank 报告及 fsync 原始行，未重新测量。恢复细节见 [analysis-recovery.json](analysis-recovery.json)，最终核对见 [final-artifact-audit.json](final-artifact-audit.json)。矩阵执行完整不代表全部时间/shape 门限通过，具体失败保留在主报告及诊断中。

最终源码哈希、全部改动 Python 文件的 F lint 和 22 + 12 + 14 = 48 项定向测试记录汇总于 [final-source-validation.json](final-source-validation.json)。
