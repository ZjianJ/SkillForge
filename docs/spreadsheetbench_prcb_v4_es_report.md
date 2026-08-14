# SpreadsheetBench PRCB-v4-ES 实验报告

## 结论

PRCB-v4-ES 将七个滑动阶段的实际 optimizer step 从固定 v4 的 224 步降至
104 步（减少 53.6%），但 val40 自由生成执行成功率从固定 v4 的
16/40（40.0%）降至 14/40（35.0%）。逐题比较为 ES 新增成功 3 题、丢失
5 题，McNemar 精确双侧检验 p=0.7266，当前单次实验不能证明两者有显著差异，
也不能声称 ES 优于固定 32-step v4。

teacher-forced monitor 在所有阶段都选到了低于阶段起点的 checkpoint，且没有
触发 preservation 安全回滚；然而这些改进没有完全转化成自由生成成功率。这说明
该 monitor 可用于节省训练量和阻止明显退化，但暂时不适合作为 agentic execution
质量的充分代理指标。

## 实验协议

- 成功轨迹总数：61。
- 固定梯度训练集：49 条。
- 固定 teacher-forced monitor：12 条；不参与反向传播。
- 滑动顺序：01 -> 12 -> 23 -> 34 -> 45 -> 56 -> 67。
- 每阶段最多 32 optimizer step，每 2 步计算一次固定 monitor。
- monitor loss：CE + 0.5 SkillKL + 0.5 margin + preserve。
- min_steps=4，patience=3，min_relative_improvement=0.2%。
- 保存 monitor composite loss 最低的安全 pair，而不是最后一步。
- preservation KL 超过阶段起点 10% 时停止并回滚。
- 动态定位始终只在 61 条成功轨迹的 gold 上文上计算。
- val40 只在最终 checkpoint 冻结后评测一次；test280 未访问。

固定划分记录见 `outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/trajectory_split.json`。

## 分阶段 early-stop 结果

| 阶段 | Pair | 实际/上限 step | 保存 step | 停止原因 | Monitor loss 起点 -> 最佳 | Preservation KL 起点 -> 最佳 |
|---:|:---:|---:|---:|:---:|---:|---:|
| 1 | 0-1 | 18/32 | 16 | patience | 2.62429 -> 2.53501 | 0.07128 -> 0.06689 |
| 2 | 1-2 | 8/32 | 6 | patience | 2.36214 -> 2.33409 | 0.06940 -> 0.06825 |
| 3 | 2-3 | 10/32 | 4 | patience | 2.41351 -> 2.36191 | 0.07019 -> 0.06676 |
| 4 | 3-4 | 16/32 | 10 | patience | 2.40117 -> 2.37007 | 0.06930 -> 0.06735 |
| 5 | 4-5 | 26/32 | 20 | patience | 2.40866 -> 2.35323 | 0.06943 -> 0.06618 |
| 6 | 5-6 | 10/32 | 4 | patience | 2.39938 -> 2.35126 | 0.06893 -> 0.06580 |
| 7 | 6-7 | 16/32 | 10 | patience | 2.38686 -> 2.35686 | 0.06778 -> 0.06712 |
| 合计 | - | 104/224 | - | 7 x patience | - | - |

实际轨迹呈现次数为 208（每个 optimizer step 累积两条训练轨迹）。所有阶段的
最佳 preservation KL 都低于阶段起点，因此本次正式实验没有触发 10% 安全阈值。

## 最终定位统计

| 方法 | Locator mass | Top-5% mass capture | Eligible token | Decisive token |
|:---|---:|---:|---:|---:|
| 固定 32-step PRCB-v4 | 59,874.85 | 63.04% | 24,986 | 717 |
| PRCB-v4-ES | 58,696.26 | 63.35% | 24,788 | 728 |
| ES - 固定 v4 | -1,178.58 (-1.97%) | +0.31 pp | -198 | +11 |

ES 的最终 locator mass 更低且 capture 略高，说明剩余 Combined mass 更集中；
但 decisive token 数略增，而且自由生成结果更差。因此 locator 指标与执行成功率仍
不是单调对应关系。

## val40 自由生成与逐题得失

| 方法 | 成功数 | 成功率 |
|:---|---:|---:|
| PRCB-v4-ES | 14/40 | 35.0% |
| 固定 32-step PRCB-v4 | 16/40 | 40.0% |
| Combined 5% + shared preserve | 16/40 | 40.0% |

相对固定 32-step v4：

- ES 独有成功（3）：39046、402-43、9726。
- 固定 v4 独有成功（5）：2768、45635、48620、59196、9569。
- 两者都成功：11；两者都失败：21。
- McNemar 精确双侧 p=0.7265625。

相对 Combined：ES 独有成功 5、Combined 独有成功 7，p=0.7744141。上述差异均
不显著，但点估计不支持 ES 提升成功率。

## 可复现产物

- 最终 checkpoint：`outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/prcb_v4_es_prefix.pt`
- SHA-256：`ccf199937ed5eef09d08c56049e51ed7c7cd347a7ea6825dca42627acfa59426`
- 总结：`outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/summary.json`
- 最终定位：`outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/final_locator/locator_statistics.json`
- val40 逐题结果：`outputs/SpreadsheetBench_prcb_v4_es_overlap1_len8_seed1/eval/final/valid_seen/results.jsonl`

## 解释与下一步

本次结果支持“按固定 teacher-forced monitor early-stop 能减少冗余更新”，但不支持
“monitor loss 最低点就是自由生成最优点”。关键错位可能来自 exposure bias：monitor
始终沿 gold 上文评估，而自由生成一旦早期 token 偏离，后续状态分布就不再属于
monitor 所覆盖的分布。下一步应保持 val40 不参与选择，先在独立成功轨迹 monitor
内部加入短程 rollout-stability 指标，或至少比较不同 checkpoint 的 gold-prefix
扰动鲁棒性，再用新的冻结规则做一次新的 val40 验证。
