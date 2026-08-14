# 实验复现手册

本文件记录当前分支已经执行或正在执行的实验。结果快照为 2026-08-14 UTC；所有数字
均来自本地输出文件，综合结论见 `docs/experiment_results_overview.md`，论文式 LaTeX 表见
`docs/spreadsheetbench_selective_results_tables.tex`。

## 1. 统一环境

### 1.1 安装

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,softprefix,qwen]"
python scripts/data/prepare_searchqa.py
python scripts/data/prepare_spreadsheetbench.py
```

SpreadsheetBench 实验默认从 Hugging Face 加载 `Qwen/Qwen3.6-35B-A3B`。已有本地快照时：

```bash
export SPREADSHEETBENCH_MODEL=/absolute/path/to/Qwen3.6-35B-A3B/snapshot
export CUDA_VISIBLE_DEVICES=0
```

所有 SpreadsheetBench 压缩实验冻结 Qwen，只优化 soft prefix。数据固定为 train80、
val40、test280；教师生成在 train80 中产生 61 条执行成功轨迹。共享实验使用一个长度 8
prefix；task-specific oracle 为 61 个任务分别训练长度 8 prefix。

### 1.2 GPT-5.5 教师配置

只有重新生成教师轨迹时需要 GPT API。定位、训练和本地自由生成可以复用缓存，不需要
再次调用 API。

```bash
mkdir -p configs/local
cp configs/spreadsheetbench/teacher_rollout_gpt55.example.yaml \
  configs/local/spreadsheetbench_paper_gpt55.local.yaml
# 只编辑被 .gitignore 排除的 local 文件，填入 endpoint 与 key。
```

## 2. 原始 SoftSkill 复现

### 2.1 SearchQA

目的：核验论文的冻结模型、文本初始化、length-32 soft prefix 和验证集选 checkpoint 链路。

设置：`Qwen/Qwen3.5-4B`，prefix length 32，3 epochs；最终只在冻结 checkpoint 上测试。

```bash
bash scripts/reproduce_searchqa_gh200.sh
```

结果：val64 最佳 hard/soft 为 76.56%/82.45%；test1400 为
77.14%/84.33%。三个 epoch 的训练 loss 为 0.3103、0.2286、0.1951。

### 2.2 SpreadsheetBench paper-style SoftSkill

目的：按原工作方式，用完整成功轨迹 next-token CE 训练一个共享 length-8 prefix，作为
后续选择性蒸馏的基线。

设置：冻结 `Qwen3.6-35B-A3B`；GPT-5.5 + 完整文本 Skill 生成轨迹；seed 1；prefix 8；
3 epochs；有效 batch size 2；学习率 1e-3；全轨迹 next-token CE；文本 Skill 的前8个
token embedding 初始化 prefix。

```bash
# 会先生成/复用教师轨迹，再训练 paper-style SoftSkill。
bash scripts/reproduce_spreadsheetbench_paper.sh

# 已有 rollout 缓存时只训练。
bash scripts/train_spreadsheetbench_paper_from_cache.sh
```

结果：val16 最佳 4/16（25.0%）；匹配协议 test280 为 85/280（30.36%）。训练 loss
由 0.3910 降到 0.3471，但后续 epoch 的自由生成验证下降。

### 2.3 ALFWorld

目的：复现论文 agent task。当前只完成环境适配器、配置、Skill 与 split manifest，尚无
本地结果，因此不能声称已经复现。

```bash
pip install -e "[alfworld]"
alfworld-download
export ALFWORLD_DATA=/path/to/alfworld/data
python scripts/data/prepare_alfworld.py --data_root "$ALFWORLD_DATA"
```

## 3. SpreadsheetBench 教师轨迹

目的：得到“完整文本 Skill + 任务 -> 成功代码”的监督轨迹，供 Qwen teacher-forcing
定位与 soft-prefix 训练使用。

设置：train80 每题一次 GPT-5.5 rollout，执行器判分；成功轨迹才进入后续实验。

```bash
python -u scripts/collect_spreadsheetbench_teacher_rollouts.py \
  --config configs/local/spreadsheetbench_paper_gpt55.local.yaml \
  --split_dir data/spreadsheetbench_split \
  --out_root outputs/SpreadsheetBench_teacher_gpt55_collection
```

结果：61/80 成功（76.25%），其中 cell-level 42/53、sheet-level 19/27；可定位
目标 token 共 54,929。它们是本项目重新生成的缓存，不是论文作者发布的原始轨迹。

## 4. 阶段一：自动 token 定位

### 4.1 Positive-gain 定位

目的：沿同一成功轨迹的 gold 上文，分别计算 Qwen 在有 Skill 和无 Skill 时的下一 token
分布，判断 Skill 影响是否集中在少量位置。两遍前向只用于定位，不在这一步训练 prefix。

定义：对 gold token `y_t`，正增益为
`max(log q_skill(y_t) - log p_plain(y_t), 0)`。

```bash
bash scripts/run_spreadsheetbench_selective_stage1.sh
```

结果：Top 1/2/5/10/20% token 捕获全局正增益
42.11/59.72/84.05/96.12/99.76%。61 条轨迹全部通过预注册集中度标准，因此支持
继续进行 selective soft-prompt 训练。

### 4.2 全分布 Combined 定位

目的：检验仅观察 gold-token 概率是否遗漏 Skill 对其他候选 token 的重排。

设置：保存 Skill/No-Skill Top-64 分布，计算 JS 差异；Combined 分数是正 target-token
gain 与全分布 JS 信号的组合。Top-64 平均覆盖概率质量超过 99.8%。

```bash
python scripts/analyze_spreadsheetbench_full_distribution.py
python scripts/prepare_spreadsheetbench_selective_stage2.py
```

结果：Top-5% Positive 捕获 81.91% 正增益、44.69% JS；Combined 捕获
76.23% 正增益、50.61% JS、97.56% Combined 质量。两者平均 Jaccard 0.6974，说明
全分布信息改变了定位，而不是重复同一排序。

## 5. 阶段二：共享 Selective Soft Prompt

统一设置：冻结 Qwen；一个共享 prefix length 8；seed 1；学习率 1e-3；成功轨迹
teacher-forcing；val40 选择 checkpoint；测试集不参与训练或定位。preservation loss 在未选
位置匹配无 Skill Qwen 的 Top-64 分布，抑制非目标行为漂移。

共享 Selective 的实际目标为：

```text
L = CE(gold token，仅 selected core 与固定 EOS)
    + lambda_preserve * KL(no-Skill reference || soft-prefix distribution，preserve 位置)
```

主 preservation 实验取 `lambda_preserve=1`。因此这里的 core 项是 gold-token CE；第7节
task-specific oracle 才直接用完整文本 Skill 的 Top-64 分布作为 core KL 教师。

### 5.1 Full CE、Random、Positive 与窗口对照

目的：验证选择性 CE 是否优于全序列 CE、重要性定位是否优于同覆盖率随机位置，以及
人工固定扩窗是否改善代码框架完整性。

```bash
bash scripts/run_spreadsheetbench_selective_stage2.sh
bash scripts/run_spreadsheetbench_random_core_preservation.sh
bash scripts/run_spreadsheetbench_positive_preservation.sh
bash scripts/run_spreadsheetbench_positive_window_preservation.sh
```

主要结果（val40）：Full CE 11/40；Random 5% 15/40；Positive 5% 13/40；
Random 5% + KL 15/40；Positive 5% + KL 18/40；Positive L2/R8 扩窗 + KL 12/40。
扩窗覆盖 35.25% token 仍变差，不支持人工固定窗口。

### 5.2 Positive 与 Combined 的严格共享-preserve 对照

目的：固定相同的 preserve token，只改变定位器，公平检验 Top-64 全分布信息。

```bash
bash scripts/run_spreadsheetbench_full_distribution_locator_validation.sh
```

设置：Top-5% core 2,838 个含 EOS 的 CE token；2,777 个共同 preserve token；
1 epoch；prefix 8；val40。结果：Positive 10/40，Combined 16/40；配对检验
`p=0.146`，方向有利但 val40 样本不足。

### 5.3 冻结 checkpoint 的 test280

目的：在不继续调参的情况下检验共享 prefix 泛化。正式对比统一使用
`max_new_tokens=8192`、`generation_batch_size=2`。

```bash
CHECKPOINT=outputs/SpreadsheetBench_full_distribution_locator_len8_seed1_shared/combined_top0.05_core_shared_preserve/best_prefix.pt
OUT=outputs/SpreadsheetBench_combined_top0.05_shared_preserve_len8_epoch1_test280
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_soft_prefix.py \
  --config configs/spreadsheetbench/selective_distillation_stage2.yaml \
  --out_root "$OUT" --cfg-options \
    train.num_epochs=0 evaluation.sel_env_num=0 evaluation.test_env_num=0 \
    evaluation.eval_test=true env.checkpoint_eval_val=false \
    env.generation_batch_size=2 soft_prefix.max_new_tokens=8192 \
    soft_prefix.checkpoint_path="$CHECKPOINT"
```

结果：Original SoftSkill 85/280（30.36%）；Combined 101/280（36.07%），相对
SoftSkill +16 题、配对 `p=0.0479`。裸 Qwen 是 97/280（34.64%）；Combined 与裸模型
差异不显著（`p=0.731`），因此不能声称新方法优于裸模型。

裸模型命令：

```bash
OUT=outputs/SpreadsheetBench_qwen36_noskill_noprefix_test280
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_soft_prefix.py \
  --config configs/spreadsheetbench/selective_distillation_stage2.yaml \
  --out_root "$OUT" --cfg-options \
    train.num_epochs=0 evaluation.sel_env_num=0 evaluation.test_env_num=0 \
    evaluation.eval_test=true env.checkpoint_eval_val=false \
    env.generation_batch_size=2 soft_prefix.max_new_tokens=8192 \
    soft_prefix.eval_init_prefix=false soft_prefix.eval_plain_baseline=true \
    soft_prefix.checkpoint_path=""
```

## 6. PRCB：受 boosting 启发的逐阶段实验

共同目的：让不同 prefix 子空间或独立 weak learner 逐阶段拟合仍未学会的 teacher-forced
残差。共同约束：定位只使用 61 条成功轨迹 gold 上文；val40 不反向传播；基础 Qwen
冻结；初始化 checkpoint 为 Combined 5% + shared KL（16/40）。

| 版本 | 设置与目的 | 实际 step | Val40 | Test280 |
|---|---|---:|---:|---:|
| v1 | 梯度探针选两个 prefix row，检验局部更新 | 32 | 14/40 | -- |
| v2 tail | 不用梯度选 row，按 67→45→23→01 | 32 | 11/40 | -- |
| v2 head | 改为 01→23→45→67 | 32 | 15/40 | -- |
| v3 | 每阶段在 gold 轨迹重算 residual/margin locator | 32 | 16/40 | -- |
| v4 | 重叠 pair 01→12→…→67，总步数匹配 | 32 | 16/40 | -- |
| v4-ES | 49 train/12 monitor，每2步评估并回滚 | 104 | 14/40 | -- |
| v4-ES w5 | 01234→12345→23456→34567 | 46 | 13/40 | -- |
| v5 | replay retention + alpha line search | 74 | **18/40** | 96/280 |
| v6 | 独立 length-8 logit learner + line search + 回蒸馏 | 28 | 16/40 | -- |
| v6 ensemble | 不回蒸馏，直接组合已接收 learner | 28 | 16/40 | -- |
| v6 low-rank | rank-2 learner，仅25.1%参数 | 18 | 16/40 | -- |

正式运行入口：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v1.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v2.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v2_head_to_tail.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v3.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v4.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v4_es.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v4_es_window5.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v5.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v6.py
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v6_lowrank.py
python -u scripts/evaluate_spreadsheetbench_prcb_v6_ensemble.py
```

结论：v5 的 val40 +2 题没有在 test280 保持；v6 第一阶段获得 functional edge，第二
阶段系数为 0，且未蒸馏 ensemble 也没有提升。当前失败的主要原因不是 student 回蒸馏，
而是 teacher-forced residual/monitor 目标与自由生成任务成功率不一致，且 prefix learner
经注意力传播后并不构成真正独立的弱学习器函数空间。

v4-ES 的固定设置是 49 条训练轨迹和12条只监控、不反传的轨迹；每2个 optimizer step
评估一次，`min_steps=4`、`patience=3`、最小相对改善0.2%、每阶段最多32步。monitor 为
`CE + 0.5 SkillKL + 0.5 margin + preserve`；preservation 相对阶段起点恶化超过10%时
回滚。v6 每个 weak learner 与原 prefix 同为 length 8，历史 learner 全冻结，在
`alpha={0,0.125,0.25,0.5,1}` 上按全局 Skill-KL line search；`alpha=0` 时丢弃该阶段。

## 7. Task-specific oracle 与覆盖率消融

目的：排除跨任务共享容量限制，为每条成功轨迹单独训练一个 prefix，再在同一训练任务
上自由生成。它只衡量 teacher-forced 分布拟合能否闭环到同题生成，不能代表 val/test
泛化，也没有访问 test280。

设置：61 个 prefix；length 8；文本 Skill 前8个 embedding 初始化；32 optimizer steps；
Combined Top-64 selective Skill-KL；固定 preserve 2,777 token；seed 1。

```bash
# 一次生成 5%、10%、20% manifest，并逐个训练/自由生成。
GPU_IDS=0 bash scripts/run_spreadsheetbench_coverage_ablation.sh
```

| Core | 成功/61 | Core-KL closure | 状态 |
|---:|---:|---:|---|
| 5% | 28（45.90%） | 62.18% | 完成 |
| 10% | 35（57.38%） | 56.86% | 完成 |
| 20% | 33（54.10%） | 50.74% | 完成 |

参考条件：Plain Qwen 26/61；完整文本 Skill 32/61。10% 相对5%新增11题、丢失4题，
净增7题，执行失败从15降至8，而可执行但语义错误保持18；单种子配对检验
`p=0.1185`。20%相对10%新增5题、丢失7题，净减2题（`p=0.7744`）；执行失败只由
8降至7，但语义错误由18增至21。结果支持5%覆盖对代码结构偏窄，同时表明继续扩大到
20%开始引入稀释或目标冲突；10%是本次单种子消融的观察最优点，而非已确认的全局最优。

`core-KL closure` 表示 soft prefix 消除了多少“裸模型到完整 Skill 教师”的 core-token
KL 差距：1 为完全复现教师分布，0 为没有缩小。它是 teacher-forced 指标，不等价于
自由生成成功率。

## 8. 结果口径与污染控制

- `val40` 用于开发与 checkpoint 选择；大量版本都看过 val40，因此不能把其中最高值当
  成无偏测试结论。
- `test280` 只能在方法冻结后评一次；若看过结果后继续修改算法，再次评测就会形成适应性
  测试污染。
- task-specific 61题来自 train80 成功轨迹，是 oracle/闭环诊断，不是独立测试集。
- 不同 `max_new_tokens`、generation batch、执行器版本或输入工作簿不能直接比较。
- smoke、未完成目录和早期不安全输入路径不进入主结果表。

## 9. 输出定位

Git 不保存 `outputs/`，复现实验后主要查看：

- `summary.json`：聚合设置与指标；
- `run.log`/`train.log`：完整终端日志；
- `best_prefix.pt`：验证集选择的共享 prefix；
- `eval/**/results.jsonl`：逐题自由生成与执行结果；
- `docs/experiment_results_overview.md`：本次工作区的统一审计表。

若要公开这些大型产物，参见 `docs/GITHUB_BACKUP.md`，不要直接取消 `.gitignore`。
