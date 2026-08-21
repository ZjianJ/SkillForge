# SpreadsheetBench Official-Adapted Prefix Baselines

> 实验日期：2026-08-15 至 2026-08-16（UTC）。本报告中的 SE-KD-Prefix 与
> OPCD-Prefix 是把官方蒸馏目标适配为只训练 soft prefix 的公平对照，不能表述为原论文
> 模型规模和数据设置的直接复现。

## 1. 目的

本实验回答：在相同冻结 Qwen、相同长度 8 soft prefix、相同训练任务和相同执行评测下，
现有选择性知识蒸馏与 on-policy 蒸馏方法能否比本项目的 SoftSkill/Combined 基线更有效地
把完整文本 Skill 压缩到 soft prefix。

## 2. 官方源码与适配边界

| 方法 | 官方仓库 | 固定 commit | 原样复用的核心机制 | SpreadsheetBench 适配 |
|---|---|---|---|---|
| SE-KD-Prefix | [SE-KD3x](https://github.com/almogtavor/SE-KD3x) | `08b276383a31fe5c07eb6685f9c4557b78e42880` | 学生熵、逐序列 `ceil(Top-20%)` 选择、full-vocabulary forward KL | 教师为同一 Qwen + hard Skill；只更新 length-8 prefix |
| OPCD-Prefix | [LMOps/OPCD](https://github.com/microsoft/LMOps/tree/main/opcd) | `4f2a9deb5f08e459fd44c2e4792344d78ca89fc3` | student Top-256 support、非重归一化 reverse KL、student on-policy state | 学生自由生成 SpreadsheetBench 代码；教师在同一生成上文加 hard Skill 评分；只更新 length-8 prefix |

第三方仓库不提交到本项目 Git 历史，通过以下命令按 commit 下载和核验：

```bash
bash scripts/setup_official_distillation_baselines.sh
```

适配代码直接加载固定 checkout 中的官方函数。SE-KD 的 selector 和 KL 函数按模块加载；
OPCD 的 `kl_penalty` 从固定官方源文件 AST 中编译，避免为了一个损失函数引入 Ray/VeRL
运行时。适配没有改变 KL 方向、位置/词类选择规则或 on/off-policy 状态分布。

## 3. 公平性协议

| 维度 | 统一设置 |
|---|---|
| 冻结基础模型 | 本地 `Qwen3.6-35B-A3B` 同一 snapshot |
| Trainable parameters | 8 x 2048 = 16,384 个 BF16 prefix 参数 |
| 初始化 | 同一 `gpt5.5_skill.md` 前 8 个 token embedding |
| 训练数据 | 同一 61 条 GPT-5.5 成功训练任务/轨迹 |
| 训练顺序 | seed 1 的同一 shuffle 顺序 |
| 训练预算 | 1 epoch，batch 1，accumulation 2，31 optimizer step |
| Optimizer | AdamW，learning rate `1e-3` |
| Val40 | 40 题，`max_new_tokens=4096`，generation batch 8 |
| Test280 | 280 题，`max_new_tokens=8192`，generation batch 2 |
| Checkpoint selection | 一轮训练后的唯一 checkpoint；Val40 不参与选 epoch/调参 |
| Test policy | 两种预先固定方法都测试；不根据 Val40 只选择胜者 |

SE-KD 与 OPCD 的计算量不强行相等：二者看到相同训练任务且参数更新次数相同，但
SE-KD 只在 gold 轨迹的高熵 20% token 上反向传播，OPCD 必须生成学生轨迹并在全部
on-policy token 上优化。实际 token 数与 wall time因此单独报告。

## 4. 训练与 Val40 结果

| 方法 | 训练状态 | 蒸馏 token | Step | Train wall time | 内部指标 | Val40 |
|---|---|---:|---:|---:|---:|---:|
| SE-KD-Prefix | gold 成功轨迹 Top-20% entropy | 11,019 / 54,990（20.04%） | 31 | 374.6s | forward KL 0.2030 | **24/40（60.0%）** |
| OPCD-Prefix | student on-policy 全位置 | 59,820 | 31 | 5,372.2s | reverse KL 0.0441 | 21/40（52.5%） |
| Combined 5% + shared KL | gold 成功轨迹 Combined core | 2,838 core + 2,777 preserve | 31 | -- | -- | 16/40（40.0%） |

OPCD 的 Top-256 学生支持平均覆盖 99.9888% 概率质量。61 条 on-policy 响应平均
980.7 token，只有 1 条达到 4096-token 上限；因此该实验中 Top-256 截断和大量 runaway
generation 都不是主要计算误差来源。

### 4.1 逐题配对与错误类型

| 比较 | 前者独占成功 | 后者独占成功 | 精确双侧配对 p |
|---|---:|---:|---:|
| SE-KD vs OPCD | 8 | 5 | 0.5811 |
| SE-KD vs Combined | 12 | 4 | 0.0768 |
| OPCD vs Combined | 8 | 3 | 0.2266 |

| 方法 | Cell-level | Sheet-level | 执行失败 | 可执行但语义失败 |
|---|---:|---:|---:|---:|
| SE-KD-Prefix | 20/29 | 4/11 | 4 | 12 |
| OPCD-Prefix | 16/29 | 5/11 | 3 | 16 |
| Combined | 11/29 | 5/11 | 9 | 15 |

两个官方适配方法都减少执行失败，收益主要来自 cell-level；当前单次 Val40 的配对差异
均未达到 0.05 显著性阈值，不能据此宣称稳定优于 Combined。

## 5. 冻结检查点

| 方法 | 路径 | SHA-256 |
|---|---|---|
| SE-KD-Prefix | `outputs/SpreadsheetBench_sekd_prefix_official_top20_len8_seed1/best_prefix.pt` | `5e5f3850cf104dbd75b2d96c35b7696edef5890023b8cf91f45c0b2a57d0b9a2` |
| OPCD-Prefix | `outputs/SpreadsheetBench_opcd_prefix_official_top256_len8_seed1/best_prefix.pt` | `279ffcd73337c998b55f126afa045048a65c3dc13cbfc7d58cb17bc4daa1ec37` |

两个 checkpoint 都只包含 `(8, 2048)` 的 BF16 `prefix_embeddings` 和 prefix length，
不包含也不更新基础模型权重。

## 6. 运行命令

### 6.1 训练与 Val40

```bash
MODEL=/absolute/path/to/Qwen3.6-35B-A3B

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_official_prefix_baseline.py \
  --config configs/spreadsheetbench/sekd_prefix_official.yaml \
  --out_root outputs/SpreadsheetBench_sekd_prefix_official_top20_len8_seed1 \
  --model_name "$MODEL"

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_official_prefix_baseline.py \
  --config configs/spreadsheetbench/opcd_prefix_official.yaml \
  --out_root outputs/SpreadsheetBench_opcd_prefix_official_top256_len8_seed1 \
  --model_name "$MODEL"
```

### 6.2 冻结 Test280

对每个方法把 `CHECKPOINT` 与 `OUT` 替换为对应路径：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_soft_prefix.py \
  --config configs/spreadsheetbench/selective_distillation_stage2.yaml \
  --out_root "$OUT" \
  --model_name "$MODEL" \
  --cfg-options \
    train.num_epochs=0 \
    evaluation.sel_env_num=0 \
    evaluation.test_env_num=0 \
    evaluation.eval_test=true \
    env.checkpoint_eval_val=false \
    env.generation_batch_size=2 \
    soft_prefix.max_new_tokens=8192 \
    soft_prefix.eval_init_prefix=false \
    soft_prefix.eval_plain_baseline=false \
    soft_prefix.checkpoint_path="$CHECKPOINT"
```

评测完成后，用同一脚本验证任务集合并计算逐题配对统计：

```bash
python scripts/analyze_spreadsheetbench_prefix_comparison.py \
  Original=PATH_TO_ORIGINAL_RESULTS_JSONL \
  Combined=PATH_TO_COMBINED_RESULTS_JSONL \
  Plain=PATH_TO_PLAIN_RESULTS_JSONL \
  SEKD=PATH_TO_SEKD_RESULTS_JSONL \
  OPCD=PATH_TO_OPCD_RESULTS_JSONL
```

该脚本在比较前强制检查 280 行、任务 ID 唯一、各方法任务集合完全相同，并使用不一致
任务上的精确双侧二项检验（paired exact test），避免把同一批任务误当成独立样本。

## 7. Test280

两个预先固定的方法都在相同 `8192/batch2` 协议上完成一次 Test280；测试期间没有根据
中间结果改 checkpoint、超参数或任务子集。五组结果均通过 280 行、任务 ID 唯一、任务
集合一致和 `ok/hard/soft` 逐题一致性检查。

### 7.1 总体与任务类型

| 方法 | 成功 | Wilson 95% CI | Cell-level | Sheet-level | 执行失败 | 可执行但语义失败 |
|---|---:|---:|---:|---:|---:|---:|
| Original SoftSkill | 85/280（30.36%） | [25.27%, 35.98%] | 45/193 | 40/87 | 42 | 153 |
| Plain Qwen | 97/280（34.64%） | [29.31%, 40.39%] | 57/193 | 40/87 | 39 | 144 |
| Combined 5% + shared KL | 101/280（36.07%） | [30.67%, 41.85%] | 61/193 | 40/87 | 53 | 126 |
| **SE-KD-Prefix** | **113/280（40.36%）** | [34.78%, 46.20%] | 69/193 | **44/87** | 43 | **124** |
| **OPCD-Prefix** | **107/280（38.21%）** | [32.72%, 44.03%] | 65/193 | 42/87 | **31** | 142 |
| *Hard-Skill 教师（上界参照）* | *118/280（42.14%）* | *[36.50%, 47.99%]* | *76/193* | *42/87* | *33* | *129* |

SE-KD-Prefix 是本次单 seed 中**训练所得 prefix** 的最高结果。相对 Original
SoftSkill，它增加 28 题（+10.00 percentage points）；OPCD-Prefix 增加 22 题
（+7.86 points）。但相对更强且协议匹配的 Combined，增量分别只有 12 题和 6 题，
置信区间也高度重叠。

补充 Hard-Skill 教师基线后，两种方法的定位可以更精确地表述。以 Plain Qwen 为地板、
教师为天花板，文本 Skill 在 Test280 上的可回收空间是 21 题：

| 方法 | 相对 Plain | 回收率 | 相对教师 | 配对 p（vs 教师） |
|---|---:|---:|---:|---:|
| SE-KD-Prefix | +16 | **76.2%** | −5 | 0.614655 |
| OPCD-Prefix | +10 | 47.6% | −11 | 0.235098 |
| Combined 5% + shared KL | +4 | 19.0% | −17 | 0.053289 |

**SE-KD-Prefix 是唯一在统计上与教师无法区分的方法**，用 8 个 soft token（16,384 个
BF16 参数）替代 2,773 token 的文本 Skill，实现 347× 上下文压缩并回收 76.2% 的增益。
Combined 相对教师的 `p=0.0533` 接近显著，说明其与教师的差距不是噪声。

值得注意的是 sheet-level：教师为 42/87，而 SE-KD 为 44/87，五组全部落在 40–44/87 的
窄带内。教师本身相对 Plain 在该类型上只多 2 题，因此 sheet-level 不存在可回收空间，
两种官方目标在该类型上的表现已达到教师水平。SE-KD 的全部优势来自 cell-level，而
它在该类型上仍低于教师（69/193 对 76/193）。教师基线的完整协议见
[Hard-Skill 教师基线](spreadsheetbench_hard_skill_teacher_baseline.md)。

SE-KD 同时提升 cell 和 sheet；OPCD 的执行失败只有31题，是五组最低，但可执行后仍有
142题语义失败。因此 OPCD 的 on-policy 训练明显改善代码可执行性，却没有把全部改善
转化为 SpreadsheetBench 任务成功。

### 7.2 逐题配对检验

| 比较（前者 vs 后者） | 前者独占成功 | 后者独占成功 | 前者净增 | 精确双侧配对 p |
|---|---:|---:|---:|---:|
| SE-KD vs Original | 45 | 17 | +28 | **0.000497** |
| OPCD vs Original | 39 | 17 | +22 | **0.004562** |
| SE-KD vs Combined | 40 | 28 | +12 | 0.181810 |
| OPCD vs Combined | 37 | 31 | +6 | 0.544612 |
| SE-KD vs Plain | 38 | 22 | +16 | 0.051894 |
| OPCD vs Plain | 28 | 18 | +10 | 0.183925 |
| SE-KD vs OPCD | 32 | 26 | +6 | 0.511842 |

按未校正的名义 `p<0.05`，两种方法都优于 Original SoftSkill；它们与 Combined、Plain
以及彼此的差异均未达到该阈值。尤其 SE-KD vs Plain 的 `p=0.0519` 很接近但仍不能写成
显著。这里报告全部预定比较，不根据 p 值删除不利结果，也不作多重比较校正。

### 7.3 生成长度与 wall-clock

| 方法 | 平均 token | 中位 token | P95 token | 触及 8192 | Test280 生成时间 |
|---|---:|---:|---:|---:|---:|
| Original SoftSkill | 729.1 | 492.5 | 1,896.3 | 3 | 日志未保留 |
| Combined | 754.4 | 507.5 | 1,815.8 | 5 | 3:01:52 |
| Plain Qwen | 792.9 | 610.5 | 1,948.3 | 2 | 3:55:11 |
| SE-KD-Prefix | 1,237.8 | 780.0 | 3,497.6 | 9 | 5:55:18 |
| OPCD-Prefix | 867.4 | 591.0 | 2,271.6 | 4 | 4:05:31 |

SE-KD 的最高成功率伴随明显更长的响应和最多的截断；OPCD 的推理长度更接近现有基线，
但其训练需要 student rollout、teacher 评分和反向传播，训练 wall time 是 5,372.2s，约为
SE-KD 374.6s 的14.34倍。成功率和计算/终止稳定性因此必须分别报告。

## 8. 结论与限制

1. 在完全相同的冻结模型、prefix 参数量、train61、更新次数和 Test280 协议下，
   SE-KD-Prefix 为113/280，OPCD-Prefix 为107/280，均超过本项目 Original SoftSkill。
2. 两者尚不能证明优于 Combined 或裸 Qwen；目前最稳妥的结论是官方蒸馏目标改善了
   full-trajectory SoftSkill 基线，而不是已经建立了相对所有强基线的统计优势。
2b. 相对 Hard-Skill 教师上界，SE-KD-Prefix 回收 76.2% 的 Skill 增益且与教师统计
   无法区分（−5题，`p=0.6147`），OPCD-Prefix 回收 47.6%。这是比"优于 Original
   SoftSkill"更强也更可解释的表述，因为它有明确的分母。sheet-level 上不存在可
   回收空间，两者的全部差距都在 cell-level。
3. 结果只使用 seed 1。Test280 提供任务采样不确定性，但没有覆盖训练初始化方差；若要
   形成论文主结论，应至少再训练2个 seed，并在每个 seed 上冻结测试。
4. 这是把官方目标适配到 SpreadsheetBench soft-prefix 参数化的公平对照，不是两篇原
   工作在其原模型、数据和训练预算上的数值复现。
