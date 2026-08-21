# SkillForge 实验结果总览

> 结果快照：2026-08-21（UTC）。本报告只汇总工作区中可由输出文件核验的实验。
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
| Combined 5% + coverage-control KL | 2,838 | 2,777 | 1 | 12/40（30.0%） |
| Combined 10% + coverage-control KL | 5,583 | 2,777 | 1 | 12/40（30.0%） |
| Combined 10% full-vocabulary Skill-KL | 5,522 | 2,777 | 1 | 15/40（37.5%） |
| SE-KD-Prefix（official-adapted） | 11,019 / 20.04% | 0 | 1 | **24/40（60.0%）** |
| OPCD-Prefix（official-adapted） | 59,820（on-policy） | 0 | 1 | 21/40（52.5%） |

解释：

1. 全序列 CE 明显弱于选择性训练，说明大量低信息 token 会稀释有限 prefix 的能力。
2. 独立 preservation 采样下 Positive 达到 45%，但不是严格定位器对照。
3. 严格共享 preservation 时，Combined 比 Positive 多 6 题（40% 对 25%），但
   逐题精确检验 `p=0.146`，方向有利但样本不足以确认显著性。
4. L2/R8 扩窗把平均覆盖扩大到 35.25%，成功率反而降至 30%，不支持固定人工扩窗。
5. Positive + KL 在 epoch 1 为 45%，继续训练到 epoch 2/3 降至25%/17.5%；loss
   下降并不意味着自由生成更稳定。
6. coverage-control 使用逐题相同且位于 Combined Top-20% 之外的2,777个 preserve；
   5%和10%在Val40同为12/40，未观察到扩大覆盖率的净收益。
7. 在同一 Combined Top-10% core 上把 one-hot CE 换成全词表 forward KL，Val40 由
   12/40 升到15/40（净增3题，`p=0.5489`）；Test280 为103/280，相对 CE 版仅 +2题。
8. SE-KD/OPCD 不含 preservation 项，其 Val40 与上方各行不是单变量对照；它们的
   公平协议见第 6.2 节。

### 5.1 被取代的早期诊断

| 运行 | 最佳 Val40 | 说明 |
|---|---:|---|
| Initial Full CE, batch2 | 4/40（10%） | 暴露 generation batch 依赖 |
| Initial Full CE, batch8 | 12/40（30%） | batch 诊断 |
| Fixed Full CE | 14/40（35%） | 后被安全路径重跑取代 |
| Fixed Random 5% | 16/40（40%） | 后被安全路径重跑取代 |
| Fixed Positive 5% | 15/40（37.5%） | 只完成2/3 epoch |

## 6. Test280 正式/诊断结果

本表有两条参照线：**地板**是 Plain Qwen（97/280），**天花板**是 Hard-Skill 教师
本体（118/280）。所有共享 prefix 方法都声称把该教师压缩进 8 个 soft token，因此
只有同时给出两条线，主表才能被解释。

| 方法 | 协议 | 成功 | 相对 SoftSkill | 配对 p（vs SoftSkill） |
|---|---|---:|---:|---:|
| **Hard-Skill 教师（文本 Skill，无 prefix）** | matched 8192/b2 | **118/280（42.14%）** | **+33题/+11.79pp** | **0.000142** |
| **SE-KD-Prefix (Official-Adapted)** | matched 8192/b2 | **113/280（40.36%）** | **+28题/+10.00pp** | **0.000497** |
| **OPCD-Prefix (Official-Adapted)** | matched 8192/b2 | **107/280（38.21%）** | **+22题/+7.86pp** | **0.004562** |
| Combined 10% full-vocabulary Skill-KL | matched 8192/b2 | 103/280（36.79%） | +18题/+6.43pp | **0.0300** |
| Combined 5% + shared KL | matched 8192/b2 | 101/280（36.07%） | +16题/+5.71pp | **0.0479** |
| Plain Qwen，无 Skill/Prefix | matched 8192/b2 | 97/280（34.64%） | +12题/+4.29pp | 0.1263 |
| PRCB-v5 | matched 8192/b2 | 96/280（34.29%） | +11题/+3.93pp | 0.1690 |
| Original SoftSkill | matched 8192/b2 | 85/280（30.36%） | -- | -- |
| Combined 5% coverage-control | matched 8192/b2 | 79/280（28.21%） | -6题/-2.14pp | 0.5190 |
| Combined 10% coverage-control | matched 8192/b2 | 74/280（26.43%） | -11题/-3.93pp | 0.2145 |
| Positive 5% + KL | **unmatched** 4096/b8 | 91/280（32.50%） | +6题 | 不作正式比较 |

任务 `42930` 在数据集中 `n_cases=0`，在全部运行中都于生成前判失败，属数据集固有
属性，不影响逐题配对。

### 6.1 回收率

Skill 在 Test280 上的可回收空间为 118−97=**21 题**。

| 方法 | 相对 Plain | 回收率 | 相对教师 | 配对 p（vs 教师） |
|---|---:|---:|---:|---:|
| SE-KD-Prefix | +16 | **76.2%** | −5 | 0.6147 |
| OPCD-Prefix | +10 | 47.6% | −11 | 0.2351 |
| Combined 10% full-vocab Skill-KL | +6 | 28.6% | −15 | 0.0769 |
| Combined 5% + shared KL | +4 | 19.0% | −17 | 0.0533 |
| Original SoftSkill | −12 | 负 | −33 | 0.000142 |

关键比较：

- **Hard-Skill 教师是七组最高（118/280），所有 prefix 方法都在其之下。** 它也是
  主表上唯一相对 Plain 达到名义显著的条目（+21题，`p=0.0238`），因此"文本 Skill
  在未见任务上确实有效"这一压缩前提成立，不是 train61 的局部现象。
- **SE-KD-Prefix 是唯一与教师统计无法区分的方法**（−5题，`p=0.6147`），用 8 个
  soft token 替代 2,773 token 的文本 Skill（347× 上下文压缩、16,384 个 BF16 参数）
  回收 76.2% 的增益。
- SE-KD/OPCD 相对 Combined 分别净增12/6题，但配对 `p=0.1818/0.5446`；相对 Plain
  的 `p=0.0519/0.1839`，尚不能证明优于这两个更强基线。
- Combined 与 Plain 的逐题差异为 Combined-only 40、Plain-only 36，`p=0.731`；
  因此它没有统计上优于裸 Qwen。Combined 与 full-vocabulary Skill-KL 的差异为
  33对31，`p=0.9007`，两者实质等同。
- Combined 10% 把 core 目标由 one-hot CE 换成全词表 forward KL 后为103/280，相对
  Original SoftSkill 达到 `p=0.0300`，但相对 Combined 仅 +2题。详见
  [全词表 Skill-KL 对照](spreadsheetbench_combined_full_vocab_skill_kl.md)。
- PRCB-v5 低于 Combined 5题，并与 Plain 基本相同（96 对97，配对 `p=1.0`）。
- **Skill 的收益几乎全部在 cell-level。** 教师相对 Plain 的 +21 题中，cell-level
  占 +19（76/193 对 57/193），sheet-level 只占 +2（42/87 对 40/87）。七组的
  sheet-level 全部落在 40–44/87 的窄带内，SE-KD 的 44/87 甚至高于教师。因此
  sheet-level 的瓶颈是教师本身不具备该能力，而非蒸馏不到位；该类型上不存在可
  回收空间。完整协议、提示审计与逐题统计见
  [Hard-Skill 教师基线](spreadsheetbench_hard_skill_teacher_baseline.md)。
- coverage-only 对照为5% 79/280、10% 74/280；逐题30对25，`p=0.5901`。10%多9个
  执行失败并有4题触及8192（5%为1题），不支持共享prefix中扩大到10%。旧Combined
  的101/280使用不同preserve，不能与新10%直接作覆盖率归因。详见
  [覆盖率公平对照](spreadsheetbench_combined_coverage_ablation.md)。

### 6.2 Official-Adapted SE-KD/OPCD 公平协议

两者固定同一 Qwen3.6-35B-A3B、length-8 prefix（16,384参数）、文本 Skill 前8 token
初始化、train61、seed1、batch1/accumulation2 和31个 optimizer step。SE-KD 使用官方
逐序列 entropy Top-20% 与 full-vocabulary forward KL；OPCD 使用 student on-policy
state、Top-256 non-renormalized reverse KL。Val40 分别为24/40与21/40。

Test280 分解为 SE-KD 69/193 cell、44/87 sheet，OPCD 65/193 cell、42/87 sheet。
SE-KD 平均生成1237.8 token且9题触及8192，OPCD为867.4 token且4题触顶；生成 wall
time 分别5:55:18与4:05:31。OPCD 训练需5,372.2s，是SE-KD 374.6s的14.34倍。完整
设置、逐题检验和官方 commit 见
[官方目标适配报告](spreadsheetbench_official_prefix_baselines.md)。

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
| Task-specific SE-KD | **32/61（52.46%）** | core 41.86%；全序列39.12% | 完成 |

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

## 9. Entropy、动态定位与定位加权（Val40）

### 9.1 静态 Entropy/EAC

| 定位器 | 主要变化 | Val40 |
|---|---|---:|
| Random Top-5% | 随机对照 | 13/40（32.5%） |
| Entropy Top-5% | 只按裸模型不确定度 | 14/40（35.0%） |
| JS Top-5% | 只按全分布差异 | 14/40（35.0%） |
| Legacy Combined | `G * JS` | 16/40（40.0%） |
| Additive Skill | `0.5G + 0.5JS` | **17/40（42.5%）** |
| EAC lambda=.25 | Additive 乘 entropy 调制 | 11/40（27.5%） |
| EAC lambda=.5 | 更强 entropy 调制 | 9/40（22.5%） |
| EAC lambda=1 | 最大 entropy 调制 | 16/40（40.0%） |

EAC 的全序列审计显示 entropy 与 JS 排名高度冗余（Spearman 约0.93）。从 Additive
到 EAC-1，Top-5% 集合仍有93.47%重叠；entropy 主要重排原本已经高 Skill-effect 的
位置，没有提供足够独立的新信息。Positive Gain 的该批重跑为0/40且有39个执行失败，
属于异常退化运行，不取代第5节的安全 Positive 结果。

### 9.2 动态重定位与连续权重

| 方法 | 主要变化 | Steps | 最终 TF 诊断 | Val40 |
|---|---|---:|---|---:|
| Dynamic Combined v1 | 每阶段重定位；preserve +10% guard | 32 | 首轮后 preservation 恶化并停止 | 15/40（37.5%） |
| Dynamic Additive | `0.5G+0.5JS`；取消 guard | 128 | 全局 KL −29.12%；preserve 3.36x | 16/40（40.0%） |
| Locator-weighted KL | Top10%内按 locator 连续加权 | 128 | 最终 KL +4.10%、preserve +18.37% vs 等权 | 13/40（32.5%） |
| Dynamic G+C | 用 strongest-competitor suppression 替代 JS | 128 | preserve +29.64% vs G+JS | 11/40（27.5%） |

定位加权的 nominal support 是5,522 token，但初始 ESS 仅2,198（39.8%）；locator 与
KL 残余相关0.729，导致相同高残余位置被 KL 本身和权重再次放大。G+C 虽把 competitor
mass 捕获从13.83%提高到36.48%，full-KL mass 却从80.90%降到53.21%，说明竞争项可作
辅助信号，但不能替代全分布覆盖。

## 10. 四信号自适应定位（Val40）

四信号为 `G`、`JS`、最强竞争 margin `C` 和 resolved uncertainty `R`。所有 full
版本均用 Train49/Monitor12、Top10%、128 step、full-vocabulary Skill-KL 和固定
preservation；Val40 只在 checkpoint 冻结后各评一次，Test280 未访问。

| 方法 | 定位/调权 | KL residual ratio | Preservation KL | Val40 |
|---|---|---:|---:|---:|
| A0 | 固定 `0.5G+0.5JS` | 0.8371 | **0.006641** | **19/40（47.5%）** |
| A1 | 固定 `0.4G+0.4JS+0.1C+0.1R` | 0.7798 | 0.007461 | **19/40（47.5%）** |
| A2 | 自适应四信号加性 | 0.7925 | 0.007825 | 14/40（35.0%） |
| A3 | 自适应 effect + 乘性 R | **0.6528** | 0.009699 | 15/40（37.5%） |

JS 与 full-vocabulary KL 的 Spearman 为0.9985；`R` 与 `G` 也较强相关（0.7008）。
A3 在 teacher-forced residual 和 monitor 上最好，却没有得到最佳执行率。A0/A1 的
19/40说明固定多信号定位有方向性收益，但没有超过 SE-KD 24/40，也未达到进入 Test280
的预注册条件。

## 11. Future-Impact Locator v2

FIL-v2 用精确 forward-mode JVP 估计“在位置 t 的局部训练方向会怎样改变独立 outer
轨迹上的未来 KL”，试图把定位目标从当前 Skill effect 改成更新后的 future impact。

| 指标 | 结果 |
|---|---:|
| Locator 轨迹/token | 39 / 38,327 |
| FIL 与 Combined Top10 Jaccard | 0.2462 |
| 重复稳定性 Spearman/Top Jaccard | 1.0 / 1.0 |
| 数值稳定性 gate | 通过 |
| FIL Top10 outer loss change | +0.007647（恶化） |
| FIL preservation change | +52.69% |
| Cal12：No-update / FIL / Combined / Random / Bottom | 5 / 6 / 4 / **7** / 5 |
| 预注册 group gate | **失败** |

FIL 排名不是 Combined 的重述，但它没有获得可靠 functional edge；Random10 的 outer
loss 改善0.001393且Cal12为7/12，均优于FIL。按 gate 规则实验在内部 calibration 停止，
没有访问Val40/Test280。

## 12. 任务表示与 task-conditioned prefix

### 12.1 Stage A：task-specific prompt manifold

- 61个 task-specific prefix，每个形状 `8 x 2048`；effective rank 56.18、participation
  ratio 52.09。
- PCA-8/16/32/48累计解释23.92%/40.35%/66.41%/87.18%方差，否定简单低维线性流形。
- task-specific residual RMS 为 shared prompt 的75.22%。
- 五折 TF-IDF 到 prompt residual 的 `R2=0.000086`、mean cosine 0.0172，文本词袋
  不能生成有效坐标。

### 12.2 Stage B：任务表示筛选

没有表示通过 `prompt ridge R2>=0.01 且 mean cosine>=0.05` 的坐标预测 gate。相对较好
的行为代理是 `qwen_task_spec_last`（closure `R2=0.1140`、Spearman 0.4512）和
`qwen_instruction_last`（success AUC 0.5703），但其 prompt coordinate R2 仍近似0。

### 12.3 Stage C：held-out transfer pilot（未完成）

固定 fold1 的49 train/12 held-out Train61 任务，目前只完成三个条件：shared 3/12、
mean49 **7/12**、nearest-neighbor task-spec 4/12。mean49 相对 shared 为4增0减，但
`top3_task_spec`、`ridge_task_spec`、`nn_instruction` 尚未完成，因此不能将单 fold 的
7/12作为方法最终结论。Stage A--C 均未访问Val40/Test280。

## 13. 当前可靠结论

1. **定位假设成立。** Skill 引起的正增益高度集中，全分布 Combined 信号相对
   target-token gain 提供了非冗余信息。
2. **压缩前提成立。** Hard-Skill 教师在从未参与训练的 Test280 上为118/280，相对
   裸 Qwen +21题、`p=0.0238`，是主表上唯一相对 Plain 达到名义显著的条目。此前
   只有 train61 的32/61 对26/61，无法排除 Skill 只对其被撰写的那批任务有效。
3. **教师是当前上界，所有 prefix 方法都在其之下。** 排序为 Hard-Skill 118 >
   SE-KD 113 > OPCD 107 > full-vocab Skill-KL 103 > Combined 101 > Plain 97 >
   Original SoftSkill 85。
4. **当前最高共享 prefix 是 SE-KD-Prefix，且是唯一与教师统计无异者。** 它为
   113/280，相对教师 −5题、`p=0.6147`，用 8 个 soft token 替代2,773 token 的文本
   Skill（347× 上下文压缩）回收76.2%的增益。Combined 与 full-vocab Skill-KL 相对
   教师的 `p=0.0533/0.0769` 接近显著，其差距不是噪声。
5. **尚不能证明官方目标适配优于 Combined 或裸模型。** 相应四个逐题比较均未达到
   名义 `p<0.05`；SE-KD vs Plain 的 `p=0.0519` 最接近阈值。
6. **sheet-level 不存在可回收空间。** 教师在该类型上只比裸模型多2题（42/87 对
   40/87），七组全部落在40–44/87；SE-KD 的44/87 甚至高于教师。该类型的瓶颈是
   教师本身不具备能力，改进蒸馏目标无法产生收益。Skill 的21题增量中19题来自
   cell-level。
7. **固定人工扩窗无效。** 5% L2/R8 覆盖35.25%却只得到12/40。
8. **当前 boosting 定义未成功。** v1--v6 没有在 Test280 超过 Combined；降低
   teacher-forced residual、monitor loss 或增加阶段数量均不能稳定提升自由生成。
9. **主要矛盾是 objective/state mismatch。** gold-prefix 上局部分布拟合可以显著
   改善，但自由生成遇到早期偏离、API/变量一致性、完整代码结构和执行语义后，效果
   不能可靠保持。OPCD 是唯一把状态分布换成学生 on-policy 的方法，其执行失败为
   五组共享 prefix 中最低（31/280），但语义失败仍有142题。
10. **task-specific 10%已超过其5%版本和同题完整文本 Skill的聚合成功数。** 结果为
    35/61，对5%的28/61净增7题；但这是同题oracle、单种子结果，不能作为泛化优势。
11. **覆盖率是当前瓶颈之一，但扩大并非单调有效。** 10%的主要收益表现为执行失败
    减少近一半；20%回落到33/61且语义错误增加，显示更大覆盖开始产生梯度稀释或
    相互冲突的局部教师目标。
12. **最大的未测量项是训练种子方差。** 全部结果为 seed 1；Test280 的 Wilson CI
    只覆盖任务采样不确定性。113/107/103/101 之间的名次差异落在单种子噪声可解释
    的范围内，回收率同样是单种子点估计。
13. **entropy 与自适应调权没有解决迁移问题。** A3 把 TF residual ratio 降到0.6528，
    但Val40只有15/40；固定A0/A1反而为19/40。
14. **FIL-v2 的新定位确实不同，但可靠性 gate 失败。** 当前一阶 future-impact 代理
    不能可靠预测真实局部更新收益，因此没有进入Val/Test。
15. **task-specific prompt 不是简单低维流形。** 简单任务文本或Qwen表示不能直接恢复
    16,384维 prompt；内部 fold 的 mean prompt 结果值得续跑，但尚未形成泛化证据。

## 14. 审计性与未完成产物

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
- Task-conditioned Stage C 只完成 shared、mean49、nn_task_spec 三个fold1条件；不能
  把当前7/12作为完整实验。
- FIL-v2 只完成内部 numerical/group gate；按预注册规则未进入Val40/Test280。
- 任务 `42930` 在数据集中 `n_cases=0`，在全部七次 Test280 中都于生成阶段之前判为
  失败（response 为空）。因此每次运行实际生成279题、结果280行。这是数据集固有
  属性且七组完全一致，不构成运行间差异。
- Hard-Skill 教师基线的 `prompt_audit.json` 记录279条实际发出提示的 SHA-256，用于
  独立证明 Skill 确实进入了每条提示；`evaluate_spreadsheet_prefix` 自身保存的
  `target_system_prompt.txt` 是未注入的 clean 版本，不能用作该证据。

## 15. 主要结果文件

- Selective/定位 LaTeX 表：`docs/spreadsheetbench_selective_results_tables.tex`
- 61题逐题诊断：`docs/spreadsheetbench_task_specific_per_task_analysis.md`
- Stage1 定位：`outputs/SpreadsheetBench_selective_stage1_qwen36_gpt55/summary.json`
- Combined Val40：`outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared/combined_top0.05_core_shared_preserve/summary.json`
- Combined Test280：`outputs/SpreadsheetBench_combined_top0.05_shared_preserve_len8_epoch1_test280_softskill_matched/summary.json`
- Plain Test280：`outputs/SpreadsheetBench_qwen36_noskill_noprefix_test280_softskill_matched/summary.json`
- **Hard-Skill 教师 Test280**：`outputs/SpreadsheetBench_qwen36_hard_skill_test280_softskill_matched/summary.json`
- 全词表 Skill-KL Test280：`outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1_test280_softskill_matched/summary.json`
- SE-KD Test280：`outputs/SpreadsheetBench_sekd_prefix_official_top20_len8_seed1_test280/summary.json`
- OPCD Test280：`outputs/SpreadsheetBench_opcd_prefix_official_top256_len8_seed1_test280/summary.json`
- 七方配对比较：`outputs/SpreadsheetBench_hard_skill_seven_way_comparison.json`
- PRCB-v5：`outputs/SpreadsheetBench_prcb_v5_retention_alpha_len8_seed1/summary.json`
- PRCB-v6 ensemble：`outputs/SpreadsheetBench_prcb_v6_functional_len8_seed1/eval/ensemble/valid_seen/ensemble_eval_summary.json`
- Task-specific：`outputs/SpreadsheetBench_task_specific_selective_skillkl_len8_seed1/summary.json`
- 覆盖率5%：`outputs/SpreadsheetBench_task_specific_combined_core05_len8_seed1_coverage_ablation/summary.json`
- 覆盖率10%：`outputs/SpreadsheetBench_task_specific_combined_core10_len8_seed1_coverage_ablation/summary.json`
- 覆盖率20%：`outputs/SpreadsheetBench_task_specific_combined_core20_len8_seed1_coverage_ablation/summary.json`
- Task-specific SE-KD：`outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/summary.json`
- 四信号报告：`docs/spreadsheetbench_meta_adaptive_four_signal_locator.md`
- FIL-v2：`outputs/SpreadsheetBench_fil_v2_jvp_chunked_direct_gate_fp32_len8_seed1/summary.json`
- Task manifold Stage A：`outputs/SpreadsheetBench_task_prompt_manifold_stage_a/summary.json`
- Task representation Stage B：`outputs/SpreadsheetBench_task_representation_stage_b/summary.json`
