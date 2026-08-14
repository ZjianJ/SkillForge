# SoftSkill 复现与压缩机制实验总览

> 结果快照：2026-08-14（UTC）。本报告只汇总工作区中可由输出文件核验的实验。
> 不同数据划分和生成协议的结果分表展示，不能直接混合排序。

## 1. 实验范围与统一口径

- 冻结模型：SpreadsheetBench 系列均为本地
  `Qwen3.6-35B-A3B`，只训练 soft prefix；SearchQA 使用
  `Qwen3.5-4B`。
- SpreadsheetBench 数据划分：train80、val40、test280；其中生成教师成功轨迹的
  训练任务为 61 个。
- 共享 soft prefix：所有任务共用一个长度 8 prefix，可在 val40 或 test280 上评测。
- task-specific oracle：61 个训练任务各有独立长度 8 prefix，只回答同题轨迹拟合能否
  转化成同题自由生成，不能作为泛化结果。
- `hard`/`soft` 在 SpreadsheetBench 中是执行评测字段，二者相同；在 SearchQA 中则是
  两种评分口径，不代表 Hard Prompt 与 Soft Prompt 两种条件。
- Val40 主系列通常使用 `max_new_tokens=4096, generation_batch_size=8`；严格匹配的
  Test280 使用 `8192, batch_size=2`。Positive 5% 的早期 Test280 例外，因此只作为诊断。

## 2. 原始 SoftSkill 复现状态

| 任务 | 模型/Prefix | 评测集 | 结果 | 状态 |
|---|---|---:|---:|---|
| SearchQA | Qwen3.5-4B，length 32 | val64 | 最佳 hard 76.56%，soft 82.45% | 完成 |
| SearchQA | 同上 | test1400 | hard 77.14%，soft 84.33% | 完成 |
| SpreadsheetBench paper-style SoftSkill | Qwen3.6-35B-A3B，length 8 | val16 | 最佳 4/16（25.0%） | 完成；不能与后续 val40 直接比较 |
| SpreadsheetBench paper-style SoftSkill | 同上 | test280 | 85/280（30.36%） | 完成，后续正式 Test280 基线 |
| ALFWorld | 仓库有适配器、配置、Skill 与路径 manifest | -- | 无本地实验输出 | **尚未复现** |

SearchQA 的三个 epoch loss 为 0.3103、0.2286、0.1951，最佳验证出现在 epoch 3。
SpreadsheetBench paper-style 训练 loss 从 0.3910 降至 0.3471，但 val16 从 25.0%
降到 18.75%，再次显示轨迹 CE 与自由生成成功率不单调。

## 3. SpreadsheetBench 教师轨迹

使用完整文本 Skill + GPT-5.5 在 train80 上重新生成轨迹，每题一次 rollout：

| 项目 | 数量 |
|---|---:|
| 总训练任务 | 80 |
| 成功轨迹 | 61（76.25%） |
| Cell-level 成功 | 42/53 |
| Sheet-level 成功 | 19/27 |
| 可定位的目标 token | 54,929 |

这些是本项目重新生成的教师轨迹，不是论文作者发布的 SpreadsheetBench 原始轨迹缓存。
所有定位、Selective、PRCB 和 task-specific 实验均复用这 61 条成功轨迹。

## 4. 第一阶段：token 定位统计

### 4.1 Positive-gain 集中度

| Top token 比例 | 全局正增益捕获 | 平均逐轨迹捕获 | 随机捕获 |
|---:|---:|---:|---:|
| 1% | 42.11% | 41.97% | 0.99% |
| 2% | 59.72% | 59.13% | 1.99% |
| 5% | 84.05% | 82.99% | 4.99% |
| 10% | 96.12% | 95.42% | 9.97% |
| 20% | 99.76% | 99.62% | 19.84% |

所有 61 条轨迹都满足预注册标准 `C(10%)>=0.60` 且 `C(20%)>=0.80`，因此
“文本 Skill 的影响在少量 token 上集中”这一定位假设得到支持。

### 4.2 Positive 与 Combined 定位器

Combined 定义为正 target-token gain 与完整词表 JS 差异的乘积。

| Top-5% 定位器 | 正增益捕获 | 精确 JS 捕获 | Combined 质量捕获 |
|---|---:|---:|---:|
| Positive | 81.91% | 44.69% | 95.90% |
| Combined | 76.23% | 50.61% | 97.56% |

- 两组位置平均 Jaccard 为 0.6974，说明全分布信息带来了非冗余选择。
- Skill/No-Skill Top-64 平均覆盖词表概率质量 99.859%/99.829%。
- 统计只证明定位信号更完整，不直接证明自由生成成功率更高。

## 5. 第二阶段：共享 prefix Selective Distillation（Val40）

以下主表使用安全数据路径。CE token 数包含每条轨迹固定监督的 EOS。

| 方法 | CE token/覆盖 | Preserve token | Epoch | Val40 |
|---|---:|---:|---:|---:|
| Full-trajectory CE | 54,990 / 100% | 0 | 3 | 11/40（27.5%） |
| Random 5% core | 2,838 / 5.06% | 0 | 3 | 15/40（37.5%） |
| Positive 5% core | 2,838 / 5.06% | 0 | 2/3 | 13/40（32.5%） |
| Random 5% + independent KL | 2,838 | 2,777 | 1 | 15/40（37.5%） |
| Positive 5% + independent KL | 2,838 | 2,777 | 1（最佳） | **18/40（45.0%）** |
| Positive 5% L2/R8 window + KL | 19,424 / 35.25% | 2,777 | 1 | 12/40（30.0%） |
| Positive 5% + shared KL | 2,838 | 2,777 | 1 | 10/40（25.0%） |
| Combined 5% + shared KL | 2,838 | 2,777 | 1 | **16/40（40.0%）** |

解释：

1. 全序列 CE 明显弱于选择性训练，说明大量低信息 token 会稀释有限 prefix 的能力。
2. 独立 preservation 采样下 Positive 达到 45%，但不是严格定位器对照。
3. 严格共享 preservation 时，Combined 比 Positive 多 6 题（40% 对 25%），但
   逐题精确检验 `p=0.146`，方向有利但样本不足以确认显著性。
4. L2/R8 扩窗把平均覆盖扩大到 35.25%，成功率反而降至 30%，不支持固定人工扩窗。
5. Positive + KL 在 epoch 1 为 45%，继续训练到 epoch 2/3 降至25%/17.5%；loss
   下降并不意味着自由生成更稳定。

### 5.1 被取代的早期诊断

| 运行 | 最佳 Val40 | 说明 |
|---|---:|---|
| Initial Full CE, batch2 | 4/40（10%） | 暴露 generation batch 依赖 |
| Initial Full CE, batch8 | 12/40（30%） | batch 诊断 |
| Fixed Full CE | 14/40（35%） | 后被安全路径重跑取代 |
| Fixed Random 5% | 16/40（40%） | 后被安全路径重跑取代 |
| Fixed Positive 5% | 15/40（37.5%） | 只完成2/3 epoch |

## 6. Test280 正式/诊断结果

| 方法 | 协议 | 成功 | 相对 SoftSkill | 配对 p（vs SoftSkill） |
|---|---|---:|---:|---:|
| Original SoftSkill | matched 8192/b2 | 85/280（30.36%） | -- | -- |
| Positive 5% + KL | **unmatched** 4096/b8 | 91/280（32.50%） | +6题 | 不作正式比较 |
| Combined 5% + shared KL | matched 8192/b2 | **101/280（36.07%）** | **+16题/+5.71pp** | **0.0479** |
| PRCB-v5 | matched 8192/b2 | 96/280（34.29%） | +11题/+3.93pp | 0.1690 |
| Plain Qwen，无 Skill/Prefix | matched 8192/b2 | 97/280（34.64%） | +12题/+4.29pp | 0.1263 |

关键比较：

- Combined 是当前共享 prefix 的最佳 Test280 结果，也是唯一相对原始 SoftSkill
  达到名义 `p<0.05` 的版本。
- Combined 与 Plain 的逐题差异为 Combined-only 40、Plain-only 36，`p=0.731`；
  因此它没有统计上优于裸 Qwen。
- PRCB-v5 低于 Combined 5题，并与 Plain 基本相同（96 对97，配对 `p=1.0`）。
- Combined 的提升集中在 cell-level：SoftSkill 45/193，Combined 61/193；
  sheet-level 均为40/87。

## 7. PRCB/Boosting 系列（Val40）

所有正式 PRCB 版本都从同一个 Combined 5% + shared KL checkpoint（16/40）开始，
定位只沿成功轨迹 gold 上文进行，Val40 不参与训练和 early stop。

| 方法 | 核心改动 | 实际 optimizer step | Val40 | Exec fail | 结论 |
|---|---|---:|---:|---:|---|
| Combined warm start | 初始 checkpoint | -- | 16/40（40%） | 9 | 基线 |
| PRCB-v1 | 梯度探针选两个 prefix row | 32 | 14/40（35%） | 9 | 失败 |
| PRCB-v2 tail-to-head | 67→45→23→01 | 32 | 11/40（27.5%） | 10 | 最差方向 |
| PRCB-v2 head-to-tail | 01→23→45→67 | 32 | 15/40（37.5%） | 11 | 好于 tail，但未超基线 |
| PRCB-v3 | gold 轨迹 margin locator | 32 | 16/40（40%） | 8 | 与基线持平 |
| PRCB-v4 fixed overlap1 | 01→12→…→67，总步数匹配 | **32** | 16/40（40%） | 9 | overlap 无收益 |
| PRCB-v4-ES window2 | 49/12 monitor early stop | 104 | 14/40（35%） | 12 | monitor 改善未转化 |
| PRCB-v4-ES window5，漏末窗 | 01234→12345→23456 | 38 | 10/40（25%） | 11 | 设计不完整，已废弃 |
| PRCB-v4-ES window5，完整 | 再加入34567 | 46 | 13/40（32.5%） | 6 | 恢复3题，仍未提升 |
| PRCB-v5 | replay retention + alpha line search | 74 | **18/40（45%）** | 6 | Val 最佳，Test 未保持 |
| PRCB-v6 full learner student | 独立 logit learner + 回蒸馏 | 28 | 16/40（40%） | 9 | 只接受1个 learner |
| PRCB-v6 full learner ensemble | 不回蒸馏直接集成 | 28 | 16/40（40%） | 8 | 不蒸馏也未提升 |
| PRCB-v6 low-rank r2 student | 25.1% 参数 learner | 18 | 16/40（40%） | 9 | 与 full learner 相同成功率 |

### 7.1 PRCB 的内部信号

- v3 的 locator mass 从 60,974.14 降到 59,852.90，仅下降1.84%，Val 与 Combined
  都是16/40。
- v4 overlap0/1 的最终 locator mass 和 Val 均近似相同；滑动重叠没有消除 prefix
  row 之间的因果干扰。
- v4-ES 的 locator mass 比固定 v4 低1.97%，但 Val 从16降到14，证明 monitor
  loss 与自由生成成功率不单调。
- **历史 v4-ES 报告中的“104步比固定 v4 的224步少53.6%”只是在比较每阶段32步的
  理论上限。实际固定 v4 使用 `[5,4,5,4,5,4,5]`，总计32步；因此 ES 实际是
  104步，对固定 v4 为3.25倍计算量。**
- v5 在 Val40 达到18/40，但 Test280 为96/280，低于 Combined 的101/280，说明
  单次 Val40 的 +2 题没有稳定泛化。
- v6 full learner 第一阶段 `alpha=0.5`，全局目标相对改善2.80%；第二阶段
  `alpha=0` 被拒绝。Stage2 与 Stage1 core 高度重叠，说明第一个 learner 没有消除
  主要残差。
- v6 的 student distillation 在 monitor 上没有超过未训练的初始 student；未蒸馏
  ensemble 与 student 都是16/40。因此失败不主要来自“蒸馏回单一 prefix”。
- low-rank r2 只训练4,112/16,384参数，第一阶段改善1.13%，仍在第二阶段失去
  functional edge；减少参数没有解决 boosting 失败。

## 8. Task-specific oracle

该实验为每个成功训练任务单独训练一个长度8 prefix，然后在同一个任务上自由生成。

| 条件 | 61题成功 | Teacher-forced core-KL closure | 状态 |
|---|---:|---:|---|
| 完整文本 Skill | 32/61（52.46%） | -- | 已评测 |
| Plain Qwen | 26/61（42.62%） | -- | 已评测 |
| Task-specific Combined 5%，旧 shared-preserve | 26/61（42.62%） | 63.49% | 完成 |
| Task-specific Combined 5%，覆盖率消融共同 preserve | **28/61（45.90%）** | 62.18% | 完成 |
| Task-specific Combined 10%，共同 preserve | **35/61（57.38%）** | 56.86% | 完成 |
| Task-specific Combined 20%，共同 preserve | **33/61（54.10%）** | 50.74% | 完成 |

旧 5% 实验中所有61题的 core KL 都改善，但自由生成只成功26题；closure 与成功的
Spearman 相关约为0.194且不显著。逐题诊断显示35个 Soft 失败中，13个是执行失败，
22个是可执行但语义错误。共同-preserve 5% 新实验提升到28题，执行失败为15题。
扩大到10%后成功数增至35题：相对5%新增11题、丢失4题，净增7题（+11.48pp），
配对精确检验 `p=0.1185`。执行失败从15题降至8题，而可执行但语义错误保持18题。
这为“5%覆盖偏小、额外core主要补足代码结构稳定性”提供了明确的方向性证据，但
单种子61题尚未达到统计显著。继续扩大到20%后成功数回落到33题：相对10%新增5题、
丢失7题，净减2题，配对精确检验 `p=0.7744`。执行失败仅由8降至7，可执行但语义错误
则由18增至21；覆盖更多 token 的结构收益已趋于饱和，并开始出现语义目标稀释。

覆盖率消融三组的唯一处理变量是 Combined core 比例：

| Core | Core token | 固定 preserve token | Combined 质量捕获 |
|---:|---:|---:|---:|
| 5% | 2,777 | 2,777 | 97.56% |
| 10% | 5,522 | 2,777 | 99.73% |
| 20% | 11,007 | 2,777 | 99.997% |

三档结果表明覆盖收益不是单调的：5%到10%提高7题，10%到20%反而减少2题。10%是
本次固定 seed、固定训练预算下的观察最优折中点，但仍需要多种子复验才能确认为稳定结论。

## 9. 当前可靠结论

1. **定位假设成立。** Skill 引起的正增益高度集中，全分布 Combined 信号相对
   target-token gain 提供了非冗余信息。
2. **共享 prefix 的最佳方法仍是 Combined 5% + shared KL。** 它在严格匹配
   Test280 上为101/280，优于原始 SoftSkill的85/280。
3. **尚不能证明优于裸模型。** Plain Qwen 为97/280，与Combined逐题差异不显著。
4. **固定人工扩窗无效。** 5% L2/R8 覆盖35.25%却只得到12/40。
5. **当前 boosting 定义未成功。** v1--v6 没有在 Test280 超过 Combined；降低
   teacher-forced residual、monitor loss 或增加阶段数量均不能稳定提升自由生成。
6. **主要矛盾是 objective/state mismatch。** gold-prefix 上局部分布拟合可以显著
   改善，但自由生成遇到早期偏离、API/变量一致性、完整代码结构和执行语义后，效果
   不能可靠保持。
7. **task-specific 10%已超过其5%版本和同题完整文本 Skill的聚合成功数。** 结果为
   35/61，对5%的28/61净增7题；但这是同题oracle、单种子结果，不能作为泛化优势。
8. **覆盖率是当前瓶颈之一，但扩大并非单调有效。** 10%的主要收益表现为执行失败
   减少近一半；20%回落到33/61且语义错误增加，显示更大覆盖开始产生梯度稀释或
   相互冲突的局部教师目标。

## 10. 审计性与未完成产物

- `SpreadsheetBench_*_smoke`：仅验证代码链路，不进入效果比较。
- `SpreadsheetBench_selective_stage2_len8_seed1*` 的 initial/b8/fixed：用于定位
  batch依赖和输入安全问题，已由 safe/shared runs 取代。
- `SpreadsheetBench_combined_...incomplete_20260805_72of279`：中断的 Test280
  产物，正式结果来自完整101/280目录。
- `SpreadsheetBench_prcb_v6_diagnostic_alpha1_checkpoint_rejected`：验证 alpha=1
  不满足接收条件。
- `SpreadsheetBench_prcb_v6_diagnostic_overlapping_core_history`：验证阶段 core
  高重叠和 history 机制，不作为最终效果行。
- 覆盖率5%/10%/20%三组均已完成；状态与逐题比较以第8节为准。

## 11. 主要结果文件

- Selective/定位 LaTeX 表：`docs/spreadsheetbench_selective_results_tables.tex`
- 61题逐题诊断：`docs/spreadsheetbench_task_specific_per_task_analysis.md`
- Stage1 定位：`outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/summary.json`
- Combined Val40：`outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared/combined_top0.05_core_shared_preserve/summary.json`
- Combined Test280：`outputs/SpreadsheetBench_combined_top0.05_shared_preserve_len8_epoch1_test280_softskill_matched/summary.json`
- Plain Test280：`outputs/SpreadsheetBench_qwen36_noskill_noprefix_test280_softskill_matched/summary.json`
- PRCB-v5：`outputs/SpreadsheetBench_prcb_v5_retention_alpha_len8_seed1/summary.json`
- PRCB-v6 ensemble：`outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1/eval/ensemble/valid_seen/ensemble_eval_summary.json`
- Task-specific：`outputs/SpreadsheetBench_task_specific_selective_skillkl_len8_seed1/summary.json`
- 覆盖率5%：`outputs/SpreadsheetBench_task_specific_combined_core05_len8_seed1_coverage_ablation/summary.json`
- 覆盖率10%：`outputs/SpreadsheetBench_task_specific_combined_core10_len8_seed1_coverage_ablation/summary.json`
- 覆盖率20%：`outputs/SpreadsheetBench_task_specific_combined_core20_len8_seed1_coverage_ablation/summary.json`
