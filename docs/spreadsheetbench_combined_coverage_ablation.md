# Combined core 覆盖率公平对照：5% vs 10%

> 实验日期：2026-08-16（UTC）。本报告只比较同一 coverage-control
> preservation 集合下的 Combined 5% 与 Combined 10%。旧的旗舰
> `Combined 5% + shared KL` 使用另一组 preservation 位置，不能作为这个消融的唯一对照。

## 1. 问题与预注册比较

Task-specific oracle 曾观察到把 Combined core 从 5% 增至 10% 后，同题自由生成由
28/61 提升到35/61。本实验检验这一现象能否迁移到**一个跨任务共享的 length-8 soft
prefix**。唯一主要处理变量是 Combined core 覆盖率；两个冻结检查点无论 Val40 高低都
各执行一次 Test280。

## 2. 公平协议

| 项目 | Combined 5% control | Combined 10% treatment |
|---|---:|---:|
| 成功教师轨迹 | 61 | 61 |
| 非 EOS core token | 2,777 | 5,522 |
| 含固定 EOS 的监督 token | 2,838 | 5,583 |
| preservation token | 2,777 | 2,777 |
| prefix | length 8，16,384 参数 | 相同 |
| 训练 | seed 1，1 epoch，batch 1 / accumulation 2，31 step | 相同 |
| 学习率 / preservation 权重 | $10^{-3}$ / 1.0 | 相同 |
| Val40 | 4096 token，batch 8 | 相同 |
| Test280 | greedy，8192 token，batch 2 | 相同 |

Combined ranking 逐轨迹嵌套，因此 5% core 是 10% core 的子集。共同 preservation
位置按轨迹从 Combined Top-20% core **之外**固定采样，故与 5%、10% core 均不相交；
两组的任务顺序、初始化、模型快照和 preservation 索引逐题完全一致。训练只使用61条
成功轨迹；Val40 不参与反向传播；Test280 在训练和冻结检查点选择完成后才访问。

配置：

- `configs/spreadsheetbench/combined_core05_coverage_shared_preserve.yaml`
- `configs/spreadsheetbench/combined_core10_coverage_shared_preserve.yaml`

冻结检查点 SHA-256：

- 5%：`afe1a5c276c96dc74619ecb47d2644968893573430b5fdb3097b925359690308`
- 10%：`8480c337b05f9367904db597bef2a2df4d1aca56f8611b09d7e783cbdae4d7ea`

## 3. 训练与 Val40

| Core | Selected CE | Preserve KL | 总 loss | Val40 | Cell | Sheet | 执行通过 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5% | 2.7375 | 0.005777 | 2.7433 | 12/40 (30.0%) | 9/29 | 3/11 | 33/40 |
| 10% | 2.0225 | 0.007800 | 2.0303 | 12/40 (30.0%) | 9/29 | 3/11 | 31/40 |

两组各有2个独占成功，精确双侧配对检验 $p=1.0$。10% 的 selected CE 数值更低，但
两个 CE 是在不同 token 集合上取平均，不能据此断言10%拟合更好。相反，10% 的
preservation KL 比5%高约35.0%，说明更大的 core 监督对未选位置造成了更强扰动。

## 4. 冻结 Test280

| 方法 | 成功 | Wilson 95% CI | Cell | Sheet | 执行失败 | 可执行但语义失败 |
|---|---:|---:|---:|---:|---:|---:|
| Combined 5% coverage-control | **79/280 (28.21%)** | [23.27%, 33.75%] | 47/193 | **32/87** | **64** | 137 |
| Combined 10% coverage-control | 74/280 (26.43%) | [21.61%, 31.89%] | 47/193 | 27/87 | 73 | **133** |

逐题比较中两者共同成功49题；5%独占30题，10%独占25题，净差为5题，精确双侧
McNemar/二项检验 $p=0.590053$。因此当前单 seed 结果既不支持“10%优于5%”，也不足以
证明5%稳定优于10%。净下降全部来自 sheet-level（10%新增6题、丢失11题）；cell-level
新增和丢失均为19题。

10% 相对5%多9个执行失败。失败状态转移显示，28题从5%的语义失败变为10%的执行失败，
而只有19题反向转移；更大覆盖率没有改善共享 prefix 的代码结构稳定性。

## 5. 生成长度与效率

| Core | 平均 token | 中位 token | P95 token | 触及 8192 | 生成时间 |
|---:|---:|---:|---:|---:|---:|
| 5% | **493.2** | 389.0 | 1,131.7 | **1** | 2:06:48 |
| 10% | 551.6 | **363.5** | **980.4** | 4 | 2:30:45 |

10%的中位数和P95更短，但出现更多极端非终止长尾，导致平均长度和总生成时间更高。
这与执行失败增加一致，说明 teacher-forced core 覆盖不能单独保证自由生成终止稳定性。

## 6. 为什么不能直接与旧 Combined 5% 归因于覆盖率

旧旗舰 Combined 5% 为101/280（36.07%），显著高于本次 coverage-control 5%的
79/280；逐题为旧版本独占43题、新版本独占21题，$p=0.008147$。两者的 core 完全相同，
区别是 preservation 位置：两组各2,777个位置但只重叠155个，全局 Jaccard 为2.87%。

旧 preservation 中有120个位置落入 Combined Top-10%，425个落入 Top-20%。若直接将
它用于10%训练，会在120个位置同时施加 core CE 和 preservation KL，造成目标冲突。
所以 coverage-only 对照必须改用 Top-20% 之外的共同 preserve；代价是这组 preserve
明显弱于旧采样。该结果揭示了一个此前未充分控制的因素：**preservation 位置选择对
共享 soft prompt 的自由生成性能非常敏感**。

## 7. 结论

1. Task-specific oracle 中10%的收益没有迁移到共享 length-8 prefix。
2. 在严格相同 preservation 下，10%为74/280，低于5%的79/280，配对差异不显著。
3. 覆盖率增加使 preservation KL、执行失败、触顶数和生成时间恶化；共享容量下更可能
   出现跨任务梯度冲突和关键 token 权重稀释。
4. 旧 Combined 5%的101/280主要说明 preservation 采样本身很重要，不能用它与新10%
   的74/280直接估计覆盖率效应。
5. 结果仅有一个训练 seed。下一步若继续研究覆盖率，应固定这套共同 preserve，至少补
   seed 2/3；更值得优先研究的是按信息量选择或加权 preservation，而不是继续扩大 core。

## 8. 结果文件

- 5% Val40：`outputs/SpreadsheetBench_combined_core05_coverage_shared_preserve_len8_seed1`
- 10% Val40：`outputs/SpreadsheetBench_combined_core10_coverage_shared_preserve_len8_seed1`
- 5% Test280：`outputs/SpreadsheetBench_combined_core05_coverage_shared_preserve_len8_seed1_test280_softskill_matched`
- 10% Test280：`outputs/SpreadsheetBench_combined_core10_coverage_shared_preserve_len8_seed1_test280_softskill_matched`
- 配对统计：`outputs/SpreadsheetBench_combined_coverage_shared_comparison.json`
