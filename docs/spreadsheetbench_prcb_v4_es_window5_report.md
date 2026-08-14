# SpreadsheetBench PRCB-v4-ES 长度 5 重叠窗口实验

> **已被更正实验取代。** 本文件记录的是遗漏 `34567` 的三窗口实验，源于对
> 用户窗口序列的误读，不代表完整 stride-1、window-size=5 日程。完整四窗口结果
> 见 `docs/spreadsheetbench_prcb_v4_es_window5_all4_report.md`。

## 结论

按 `01234 -> 12345 -> 23456` 更新后，val40 自由生成执行成功率为
10/40（25.0%），低于长度 2 PRCB-v4-ES 的 14/40（35.0%），也低于固定
32-step PRCB-v4 的 16/40（40.0%）。本次结果不支持将更新窗口直接扩大到五个
soft-prefix 向量。

该实验严格实现了指定的三个窗口，但不是纯粹的单变量“窗口宽度”对照：长度 5
版本只有 3 次动态重定位，长度 2 版本有 7 次；此外 prefix 向量 7 在指定日程中
始终冻结。结论应表述为“三个长度 5 窗口的所提日程效果更差”，不能推广成所有
长度 5 日程一定更差。

## 协议

- 初始 checkpoint、61 条成功轨迹、49/12 固定 train/monitor 划分与长度 2 ES 相同。
- 窗口：`[0,1,2,3,4] -> [1,2,3,4,5] -> [2,3,4,5,6]`。
- 每阶段最多 32 optimizer step，每 2 步评估固定 monitor。
- monitor loss：CE + 0.5 SkillKL + 0.5 margin + preserve。
- min_steps=4，patience=3，最小相对改进 0.2%，preservation 上限为阶段起点的 110%。
- locator 只沿成功轨迹 gold 上文计算；val40 仅在 checkpoint 冻结后运行一次。
- test280 未访问。

## 训练结果

| 阶段 | 窗口 | 实际/上限 step | 保存 step | Monitor loss 起点 -> 最佳 | Preservation KL 起点 -> 最佳 |
|---:|:---:|---:|---:|---:|---:|
| 1 | 01234 | 8/32 | 2 | 2.62429 -> 2.53582 | 0.07128 -> 0.06607 |
| 2 | 12345 | 18/32 | 12 | 2.35861 -> 2.29406 | 0.06813 -> 0.06050 |
| 3 | 23456 | 12/32 | 6 | 2.36692 -> 2.32402 | 0.06349 -> 0.06123 |
| 合计 | - | 38/96 | - | - | - |

三个阶段均由 patience 停止，没有触发 preservation guard。实际轨迹呈现次数为 76。

## val40 对比

| 方法 | 成功数 | 成功率 | 实际 optimizer step |
|:---|---:|---:|---:|
| 长度 5 ES（三窗口） | 10/40 | 25.0% | 38 |
| 长度 2 ES（七窗口） | 14/40 | 35.0% | 104 |
| 固定 32-step v4 | 16/40 | 40.0% | 224 |

相对长度 2 ES：长度 5 独有成功 2 题（53383、9569），长度 2 独有成功 6 题
（12864、402-43、46897、50768、8942、9726），McNemar 精确双侧 p=0.2890625。

相对固定 v4：长度 5 独有成功 2 题（39046、53383），固定 v4 独有成功 8 题
（12864、2768、45635、46897、48620、50768、59196、8942），p=0.109375。

差异在单次 40 题样本上未达到 0.05 显著性，但两个逐题比较的点估计均明显不利于
长度 5 日程。

## 最终 locator

| 方法 | Locator mass | Top-5% mass capture | Eligible token | Decisive token |
|:---|---:|---:|---:|---:|
| 固定 v4 | 59,874.85 | 63.04% | 24,986 | 717 |
| 长度 2 ES | 58,696.26 | 63.35% | 24,788 | 728 |
| 长度 5 ES | 59,458.90 | 63.00% | 25,100 | 736 |

长度 5 相比长度 2 留下更多 locator mass、更多 eligible/decisive token，且质量集中度
略低。这与其较低的自由生成成功率方向一致。

## 原因分析

1. 一个阶段同时移动五个向量，teacher-forced monitor 可以很快下降，但更新自由度更高，
   更容易形成只适用于 gold 上文的联合补偿。
2. 中间向量 2、3、4 被连续更新三次，向量 0、6 只更新一次，向量 7 完全不更新，
   优化强度明显不均衡。
3. 三阶段只有三次动态重定位。长度 2 版本在七个阶段前重新确认残余位置，能更频繁地
   修正 locator；长度 5 版本可能在每个大窗口内部过度优化已经开始过时的位置。
4. 更大的窗口增强 prefix 参数之间的共同适配，却没有扩大 teacher-forced monitor 对
   自由生成偏离状态的覆盖，因而 exposure bias 仍然存在，甚至可能被放大。

## 产物

- checkpoint：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_len8_seed1/prcb_v4_es_prefix.pt`
- SHA-256：`5590081d6873e34078559bc11f02cc7f07683d6a45c283d94c5124497ee28add`
- summary：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_len8_seed1/summary.json`
- final locator：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_len8_seed1/final_locator/locator_statistics.json`
- val40：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_len8_seed1/eval/final/valid_seen/results.jsonl`

更严谨的下一步是补做四窗口 `01234 -> 12345 -> 23456 -> 34567`，使所有八个向量
至少更新一次；但仍需明确它与长度 2 版本的重定位次数不同。若要做真正的窗口宽度
消融，应固定总 optimizer step、动态重定位次数和各向量累计更新机会。
