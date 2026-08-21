# SpreadsheetBench Task-Specific SE-KD-Prefix

## 目的

比较逐任务训练时，SE-KD 的动态学生熵定位是否优于固定的 Combined Skill-effect
定位。该实验是同题 oracle/capacity 诊断，不是未见任务泛化实验；Val40 和 Test280
均未访问。

## 设置

- 冻结模型：Qwen3.6-35B-A3B，固定 snapshot
  `995ad96eacd98c81ed38be0c5b274b04031597b0`。
- 数据：61 条成功的 SpreadsheetBench 训练轨迹。
- 每个任务训练一个独立的 length-8 prefix。
- 初始化：完整 Hard Skill 前 8 个 token embedding。
- 每题 AdamW 32 steps，学习率 `1e-3`，seed 1。
- SE-KD 每一步按当前学生 entropy 动态选择 `ceil(20%)` gold 位置。
- 选择后使用冻结 Qwen + Hard Skill 教师的 full-vocabulary forward KL。
- 按官方方法不添加 preservation、prefix delta regularizer 或 gradient clipping。
- 训练后使用 `soft prefix + clean task` 贪心自由生成，执行 SpreadsheetBench checker。

官方 SE-KD 源码固定在 `almogtavor/SE-KD3x` commit
`08b276383a31fe5c07eb6685f9c4557b78e42880`。实现按官方 selective-lm-head 路径，
每一步只运行一次学生 Transformer；冻结 Hard-Skill 教师 hidden states 每题缓存一次。
该优化不改变选择、损失或梯度。

## 结果

| 方法 | 定位/训练支持 | 同题自由生成成功 | 执行失败 | 语义失败 | Mean core-10 probe closure |
|---|---|---:|---:|---:|---:|
| Combined 5% | 固定 Combined 5% + preserve | 28/61 (45.90%) | 15 | 18 | 62.18% |
| Combined 10% | 固定 Combined 10% + preserve | **35/61 (57.38%)** | 8 | 18 | **56.86%** |
| Combined 20% | 固定 Combined 20% + preserve | 33/61 (54.10%) | **7** | 21 | 50.74% |
| SE-KD 20% | 动态学生 entropy 20%，无 preserve | 32/61 (52.46%) | 9 | 20 | 41.86% |

SE-KD 的全轨迹 Hard-Skill probe closure 为 39.12%，61 题全部为正；训练位置上的
full-vocabulary KL 均值由 0.2436 降至 0.1391，平均相对下降 42.04%。这些下降没有使
自由生成超过固定 Combined 10%。

## 逐题配对

| 对照 | SE-KD only | 对照 only | Both | Neither | Exact McNemar p |
|---|---:|---:|---:|---:|---:|
| Combined 5% | 9 | 5 | 23 | 24 | 0.4240 |
| Combined 10% | 6 | 9 | 26 | 20 | 0.6072 |
| Combined 20% | 6 | 7 | 26 | 22 | 1.0000 |

相对当前最好的 Combined 10%，SE-KD 新增成功任务为：`108-24`、`1818`、`472-15`、
`48745`、`55060`、`66-24`；同时丢失：`141-20`、`31011`、`31915`、`32255`、
`408-39`、`42354`、`48080`、`54085`、`55965`。

## 诊断

- SE-KD 成功任务的全轨迹 closure 均值为 42.74%，失败任务为 35.12%；点二列相关
  `r=0.290, p=0.0236`。全局分布拟合与执行成功存在弱相关，但远非充分条件。
- 固定 Combined-10 probe closure 在成功/失败任务上分别为 44.84%/38.57%，相关性
  不显著 (`r=0.201, p=0.1204`)。
- SE-KD 自身训练 KL 的相对下降在成功/失败任务上为 44.64%/39.17%，相关性不显著
  (`r=0.184, p=0.1566`)。
- 高 entropy 表示模型不确定，并不表示 Hard Skill 显著改变该位置。动态更新学生状态
  会改变不确定性，但不会自动把定位转化为 Skill-effect 定位。
- SE-KD 没有 preservation，可能在学习高 entropy token 时同时扰动基础模型原有能力。

因此，在本次官方方法口径下，没有证据表明动态 entropy 定位优于固定 Combined；
聚合结果反而低 3 题，但配对差异不显著。由于两种方法的覆盖率、KL 形式和 preservation
也不同，这是一项完整方法比较，不是严格的 locator-only 因果消融。后续若要只判断
定位器，应固定相同的 10% 覆盖率、full-vocabulary Skill KL、preservation、初始化和
optimizer，仅替换 Combined 与动态 entropy mask。

## 产物

- `outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/summary.json`
- `outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/training_results.jsonl`
- `outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/evaluation_results.jsonl`
- `outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/run.log`
- 每题 checkpoint：`outputs/SpreadsheetBench_task_specific_sekd_len8_seed1/training/<task>/final_prefix.pt`
