# SpreadsheetBench PRCB-v4-ES 完整长度 5 滑动窗口实验

## 更正说明与结论

长度 8 prefix、窗口宽度 5、stride 1 的完整日程是：

`01234 -> 12345 -> 23456 -> 34567`

此前三窗口实验遗漏了 `34567`，其 10/40 结果不能代表用户提出的方法。本报告记录
修正后的完整四窗口实验。四窗口 val40 为 13/40（32.5%）：相比不完整三窗口恢复
3 题，但仍略低于长度 2 ES 的 14/40（35.0%）和固定 v4 的 16/40（40.0%）。

相对长度 2 ES 的逐题差异为新增 3、丢失 4，McNemar 精确双侧 p=1.0；因此当前
40 题结果只能认为两种窗口宽度基本持平，不能证明长度 5 更优或更差。

## 协议

- 61 条成功轨迹固定划分为 49 条梯度训练、12 条 teacher-forced monitor。
- 四个窗口：01234、12345、23456、34567。
- 每阶段最多 32 optimizer step；每 2 步评估固定 monitor。
- min_steps=4、patience=3、最小相对改进 0.2%。
- monitor loss：CE + 0.5 SkillKL + 0.5 margin + preserve。
- preservation KL 超过阶段起点 10% 时停止并回滚。
- locator 仅在成功轨迹 gold 上文计算；val40 仅在最终 checkpoint 冻结后运行。
- test280 未访问。

前三阶段直接复用了三窗口实验中计算路径完全相同的产物。复制前后 checkpoint
SHA-256 逐一核对一致；第 4 阶段和最终 val40 均重新运行。

## 分阶段结果

| 阶段 | 窗口 | 实际/上限 step | 保存 step | Monitor loss 起点 -> 最佳 | Preservation KL 起点 -> 最佳 |
|---:|:---:|---:|---:|---:|---:|
| 1 | 01234 | 8/32 | 2 | 2.62429 -> 2.53582 | 0.07128 -> 0.06607 |
| 2 | 12345 | 18/32 | 12 | 2.35861 -> 2.29406 | 0.06813 -> 0.06050 |
| 3 | 23456 | 12/32 | 6 | 2.36692 -> 2.32402 | 0.06349 -> 0.06123 |
| 4 | 34567 | 8/32 | 2 | 2.36279 -> 2.33355 | 0.06326 -> 0.05930 |
| 合计 | - | 46/128 | - | - | - |

四阶段均由 patience 停止，没有触发 preservation guard；实际轨迹呈现次数为 92。

## val40 对比

| 方法 | 成功数 | 成功率 | 实际 optimizer step |
|:---|---:|---:|---:|
| 长度 5 ES，完整四窗口 | 13/40 | 32.5% | 46 |
| 长度 5 ES，不完整三窗口 | 10/40 | 25.0% | 38 |
| 长度 2 ES，完整七窗口 | 14/40 | 35.0% | 104 |
| 固定 32-step v4 | 16/40 | 40.0% | 224 |

逐题比较：

- 四窗口相对三窗口：新增成功 6、丢失 3，净增 3，p=0.5078125。
- 四窗口相对长度 2 ES：新增成功 3、丢失 4，净减 1，p=1.0。
- 四窗口相对固定 v4：新增成功 2、丢失 5，净减 3，p=0.453125。

四窗口相对长度 2 ES 的独有成功为 2768、45635、55049；长度 2 ES 独有成功为
12864、39046、402-43、46897。两者共同成功 10 题。

## 最终 locator

| 方法 | Locator mass | Top-5% mass capture | Eligible token | Decisive token |
|:---|---:|---:|---:|---:|
| 长度 5，不完整三窗口 | 59,458.90 | 63.00% | 25,100 | 736 |
| 长度 5，完整四窗口 | 58,918.85 | 63.88% | 24,930 | 743 |
| 长度 2 ES | 58,696.26 | 63.35% | 24,788 | 728 |
| 固定 v4 | 59,874.85 | 63.04% | 24,986 | 717 |

末端窗口将 locator mass 降低 540.04，并使 mass capture 增加 0.88 个百分点；这与
成功率从 25.0% 恢复到 32.5% 方向一致。但 decisive token 增至 743，且最终成功率
仍未超过长度 2。

## 解释

1. 补齐 `34567` 很重要：它首次更新向量 7，并再次联合调整 3--6，修复了三窗口
   日程的末端覆盖缺口。
2. 长度 5 用 46 步达到与长度 2 的 104 步接近的成功率，显示大窗口有更高的单步
   参数覆盖效率。
3. 但大窗口同时移动五个向量，更容易在 gold 上文上形成联合补偿；monitor 和
   locator 的改善仍不能保证自由生成同步改善。
4. 该比较仍未固定总 optimizer step：它回答“各自 early-stop 后哪种完整日程更好”，
   而不是纯粹的窗口宽度因果效应。

## 产物

- checkpoint：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_all4_len8_seed1/prcb_v4_es_prefix.pt`
- SHA-256：`24650889260410e57c99b9170cae73d3509607df756ad9f911cda529a048019f`
- summary：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_all4_len8_seed1/summary.json`
- final locator：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_all4_len8_seed1/final_locator/locator_statistics.json`
- val40：`outputs/SpreadsheetBench_prcb_v4_es_window5_stride1_all4_len8_seed1/eval/final/valid_seen/results.jsonl`
