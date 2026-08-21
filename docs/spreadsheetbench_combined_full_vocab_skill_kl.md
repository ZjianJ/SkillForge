# Combined Top-10%：one-hot CE 与全词表 Skill-KL 对照

## 1. 实验问题

本实验只检验一个变量：在固定的 Combined Top-10% core 位置上，把 GPT-5.5
成功轨迹 token 的 one-hot CE，替换为冻结 Qwen+完整文本 Skill 教师与
Qwen+soft-prefix 学生之间的完整词表 forward KL。轨迹仍只提供共同的
teacher-forced gold 上文，不再把 GPT-5.5 的单个 token 当作 core 蒸馏标签。

## 2. 严格控制协议

| 项目 | Combined10 CE control | Full-vocabulary Skill-KL treatment |
|---|---|---|
| 冻结基础模型 | Qwen3.6-35B-A3B | 相同 |
| 共享 soft prefix | length 8，16,384 参数 | 相同 |
| 初始化 | 完整文本 Skill 前 8 token embedding | 相同 |
| 训练数据 | 同一 61 条 GPT-5.5 成功轨迹 | 相同 |
| 定位器 | 固定 Combined Top-10% | 相同 |
| core 非 EOS 位置 | 5,522 | 相同 |
| EOS 位置 | 61 个 one-hot CE | 相同 |
| preservation | 同一 2,777 个位置、权重 1.0 | 相同 |
| optimizer | AdamW，LR 1e-3 | 相同 |
| 数据顺序 | seed 1，batch 1，accumulation 2 | 相同 |
| optimizer step | 31 | 相同 |
| 唯一处理变量 | core one-hot CE | core full-vocabulary forward KL |

这里“取消 Top-64”只针对 **Hard-Skill core 蒸馏目标**：每个 core 位置都在完整
词表上计算教师与学生分布。preservation 不是 Skill 教师拟合目标，仍沿用 control
中的 no-Skill Top-64+residual-bucket KL，以免同时改变第二个实验变量。

## 3. 目标函数

对轨迹中被 Combined 选中的位置集合 \(C\)，完整文本 Skill 教师分布为
\(q_t=\operatorname{softmax}(z_t^{\text{hard}})\)，soft-prefix 学生分布为
\(p_t=\operatorname{softmax}(z_t^{\text{soft}})\)。core 损失改为

\[
L_{\text{Skill-KL}}
=\frac{1}{|C|}\sum_{t\in C}
\sum_{v\in\mathcal V}q_t(v)\log\frac{q_t(v)}{p_t(v)}.
\]

实现对词表维分块计算 log-sum-exp 和 KL，减少临时显存，但没有截断、Top-k 或
residual bucket，因此数值上是完整词表的精确 forward KL。总目标保持为

\[
L=L_{\text{core}}+L_{\text{EOS-CE}}+L_{\text{preserve}}.
\]

其中 EOS CE 与 preservation 的定义和权重均未改变。

## 4. 实现与复现

- 配置：`configs/spreadsheetbench/combined_core10_full_vocab_skillkl_shared_preserve.yaml`
- 训练器：`scripts/train_spreadsheetbench_combined_full_vocab_kl.py`
- 损失实现：`skillopt/softprefix/distillation_losses.py`
- 单元测试：`tests/test_full_vocab_distillation.py`

```bash
cd /home/u6ow/zijian.u6ow/softskill

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_combined_full_vocab_kl.py \
  --config \
    configs/spreadsheetbench/combined_core10_full_vocab_skillkl_shared_preserve.yaml \
  --out_root \
    outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1
```

冻结检查点：

```text
outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1/best_prefix.pt
SHA-256 c5e4bf281ff0df69ee193551ae8298fd53a9100c126fcba1477d586a5f906831
```

## 5. 已完成结果

### 5.1 训练

| 指标 | 结果 |
|---|---:|
| 成功轨迹 | 61 |
| optimizer step | 31 |
| full-vocabulary Skill-KL token | 5,522 |
| EOS CE token | 61 |
| preservation token | 2,777 |
| mean full-vocabulary Skill KL | 0.300246 |
| mean preservation KL | 0.004422 |
| mean total objective | 0.301389 |
| 训练 wall time | 604 s |

### 5.2 Val40

| 方法 | 成功率 | Cell | Sheet | 执行失败 | 语义失败 |
|---|---:|---:|---:|---:|---:|
| Combined10 one-hot CE | 12/40 (30.0%) | 9/29 | 3/11 | 9 | 19 |
| **Combined10 full-vocabulary Skill-KL** | **15/40 (37.5%)** | **11/29** | **4/11** | **7** | **18** |

逐题配对为：两者都成功 8 题、KL 独赢 7 题、CE 独赢 4 题、两者都失败 21 题；
净增 3 题，精确双侧符号检验 \(p=0.548828\)。方向为正，但 Val40 尚不足以证明稳定
提升，因此结论以冻结检查点的一次 Test280 评测为准。

### 5.3 冻结 Test280

| 方法 | 成功率 | Cell | Sheet | 执行失败 | 语义失败 |
|---|---:|---:|---:|---:|---:|
| Combined10 one-hot CE | 74/280 (26.43%) | 47/193 | 27/87 | 73 | 133 |
| **Combined10 full-vocabulary Skill-KL** | **103/280 (36.79%)** | **63/193** | **40/87** | **43** | 134 |

两者共同成功56题、全词表KL独赢47题、CE独赢18题、共同失败159题；净增29题、
10.36个百分点，精确双侧配对检验 \(p=0.000422\)。提升同时出现在 cell-level
（+16题）和 sheet-level（+13题），并把执行失败从73题降到43题。因此在固定定位、
preservation、初始化和训练预算下，完整 Hard-Skill 分布是明显优于 GPT轨迹 one-hot
token 的 core 教师。

### 5.4 与其他共享-prefix方法的关系

| 方法 | Test280 | 相对本方法独赢/独输 | 配对 \(p\) |
|---|---:|---:|---:|
| 本方法：Combined10 full-vocabulary Skill-KL | 103/280 (36.79%) | -- | -- |
| 旧 Combined5 + shared preservation | 101/280 (36.07%) | 33/31 | 0.900653 |
| SE-KD-Prefix | 113/280 (40.36%) | 22/32 | 0.220328 |
| OPCD-Prefix | 107/280 (38.21%) | 25/29 | 0.683489 |
| 裸 Qwen | 97/280 (34.64%) | 30/24 | 0.496617 |

本方法显著修复了 coverage-control Combined 的目标错配，但尚不能声称优于旧
Combined、SE-KD、OPCD 或裸模型：上述四个差异在当前单 seed 下均不显著。它以与
SE-KD/OPCD 接近的结果说明，**在选定状态上拟合完整教师分布**比拟合轨迹单个 token
更符合“压缩文本 Skill”的目标；余下瓶颈仍是共享 length-8 容量、定位/权重分配以及
teacher-forced 状态到自由生成状态的迁移。

结果目录：

```text
outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1_test280_softskill_matched
```

### 5.3 Test280（冻结检查点，matched 8192 / batch2）

| 方法 | 成功 | Wilson 95% CI | Cell | Sheet | 执行失败 | 语义失败 |
|---|---:|---:|---:|---:|---:|---:|
| Combined 5% + shared KL（CE core） | 101/280（36.07%） | [30.67%, 41.85%] | 61/193 | 40/87 | 53 | 126 |
| **Combined 10% full-vocabulary Skill-KL** | **103/280（36.79%）** | [31.35%, 42.58%] | 63/193 | 40/87 | 43 | 134 |

逐题配对检验：

| 比较（前者 vs 后者） | 前者独占 | 后者独占 | 净差 | 精确双侧配对 p |
|---|---:|---:|---:|---:|
| full-vocab KL vs Original SoftSkill | 40 | 22 | +18 | **0.030016** |
| full-vocab KL vs Plain Qwen | 30 | 24 | +6 | 0.496617 |
| full-vocab KL vs Combined（CE core） | 33 | 31 | +2 | 0.900653 |
| SE-KD-Prefix vs full-vocab KL | 32 | 22 | +10 | 0.220328 |
| Hard-Skill 教师 vs full-vocab KL | 39 | 24 | +15 | 0.076926 |

结论：把 core 目标由 one-hot CE 换成完整词表 forward KL，相对 Original SoftSkill
达到名义 `p<0.05`，执行失败由 53 降到 43；但相对同 preserve 的 Combined CE 版仅
净增 2 题（`p=0.9007`），Val40 上 +3 题的方向性优势没有在 Test280 放大。

以 Plain Qwen（97/280）为地板、Hard-Skill 教师（118/280）为天花板，本方法回收
28.6% 的 Skill 增益，低于 SE-KD-Prefix 的 76.2% 和 OPCD-Prefix 的 47.6%；相对教师
的 `p=0.0769` 接近显著。因此**在固定 gold 轨迹状态分布下，仅把 core 目标从
one-hot 换成全词表并不足以缩小与教师的差距**——SE-KD 同样使用全词表 forward KL，
其优势来自逐序列学生熵 Top-20% 的在线选择而非 KL 的词表宽度。
