# 实验复现手册

本文件记录当前分支已经执行或正在执行的实验。结果快照为 2026-08-17 UTC；所有数字
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

### 5.3 Combined 5%/10% coverage-only 对照

目的：检验 task-specific oracle 中扩大覆盖率的收益能否迁移到共享 prefix。5%与10%
使用逐题完全相同的2,777个 preserve，且这些位置固定采样于 Combined Top-20% core
之外；只有 core 覆盖率变化。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_soft_prefix.py \
  --config configs/spreadsheetbench/combined_core05_coverage_shared_preserve.yaml \
  --out_root outputs/SpreadsheetBench_combined_core05_coverage_shared_preserve_len8_seed1

CUDA_VISIBLE_DEVICES=0 python -u scripts/train_soft_prefix.py \
  --config configs/spreadsheetbench/combined_core10_coverage_shared_preserve.yaml \
  --out_root outputs/SpreadsheetBench_combined_core10_coverage_shared_preserve_len8_seed1
```

结果：Val40均为12/40。冻结 Test280（8192/batch2）为5% 79/280（28.21%）、10%
74/280（26.43%）；5%独占30题、10%独占25题，精确配对 `p=0.5901`。10%执行失败
73题，5%为64题；触及8192上限分别4题和1题。当前结果不支持共享prefix中扩大到10%。
旧旗舰Combined 5%的101/280使用另一套preserve，不能与新10%直接作覆盖率归因。完整
统计见 `docs/spreadsheetbench_combined_coverage_ablation.md`。

### 5.4 冻结 checkpoint 的 test280

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
差异不显著（`p=0.731`），因此不能声称新方法优于裸模型。作为上界参照，Hard-Skill
教师本体为 118/280（见 5.7 节），Combined 相对教师为 −17 题、`p=0.0533`。

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

### 5.5 Official-Adapted SE-KD-Prefix 与 OPCD-Prefix

目的：在完全相同的冻结 Qwen、length-8 prefix、文本 Skill 初始化、train61、seed1 和
31次 optimizer update 下，把现有工作的原生蒸馏目标适配到 SpreadsheetBench，作为
Combined 的官方代码来源对照。

- SE-KD-Prefix：官方逐序列 student-entropy Top-20% selector，选中11,019/54,990
  token，使用 full-vocabulary forward KL；训练374.6s，Val40 24/40，Test280
  113/280（40.36%）。
- OPCD-Prefix：student on-policy rollout、student Top-256 support、官方非重归一化
  reverse KL；训练59,820 token、5,372.2s，Val40 21/40，Test280 107/280（38.21%）。
- 两种方法相对 Original SoftSkill 的逐题精确双侧 `p=0.000497/0.004562`；相对
  Combined 的 `p=0.181810/0.544612`，相对 Plain 的 `p=0.051894/0.183925`。

官方仓库固定 commit、训练/测试命令、完整错误分解和效率统计见
[SpreadsheetBench Official-Adapted Prefix Baselines](spreadsheetbench_official_prefix_baselines.md)。

### 5.6 Combined 10% 全词表 Skill-KL

目的：只改一个变量——把固定 Combined Top-10% core 上的 GPT-5.5 one-hot CE 换成
冻结 Qwen+文本 Skill 教师与 soft-prefix 学生之间的完整词表 forward KL。轨迹仍只
提供 teacher-forced gold 上文，preservation 与 EOS CE 完全不变。

```bash
CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_combined_full_vocab_kl.py \
  --config \
    configs/spreadsheetbench/combined_core10_full_vocab_skillkl_shared_preserve.yaml \
  --out_root \
    outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1
```

结果：31 step、5,522 个 KL token、mean Skill-KL 0.300246、训练604s。Val40 由 CE 版的
12/40 升至15/40（净增3题，`p=0.5489`）；冻结 Test280 为103/280（36.79%），相对
Original SoftSkill `p=0.0300`，相对 Combined CE 版仅 +2题（`p=0.9007`）。完整设置见
[全词表 Skill-KL 对照](spreadsheetbench_combined_full_vocab_skill_kl.md)。

### 5.7 Hard-Skill 教师基线

目的：测量**被压缩的对象本身**。此前主表只有地板（Plain Qwen）没有天花板，无法
计算回收率，也无法区分"蒸馏没做好"与"教师本来就不会"。本实验不训练、不加载、不
安装任何 prefix，只把完整文本 Skill 写入 system prompt。

```bash
MODEL=/absolute/path/to/Qwen3.6-35B-A3B

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/evaluate_spreadsheetbench_hard_skill_baseline.py \
  --model-path "$MODEL" \
  --out-root outputs/SpreadsheetBench_qwen36_hard_skill_test280_softskill_matched
```

脚本默认值即匹配协议（8192 / batch2 / greedy / 单次生成 / prompt_start），不应覆盖；
`--limit` 仅用于 smoke。输出目录已有 `results.jsonl` 时脚本拒绝运行。

结果：**118/280（42.14%）**，是七组最高。相对 Plain +21题（`p=0.0238`），是主表上
唯一相对裸模型达到名义显著的条目。Cell-level 76/193、sheet-level 42/87、执行失败33。
以 Plain 为地板、教师为天花板，可回收空间为21题，SE-KD 回收76.2%且与教师统计无异
（`p=0.6147`）。完整协议、注入等价性证明与提示审计见
[Hard-Skill 教师基线](spreadsheetbench_hard_skill_teacher_baseline.md)。

七方逐题比较：

```bash
python scripts/analyze_spreadsheetbench_prefix_comparison.py \
  Original=outputs/SoftSkill_spreadsheetbench_paper_len8_seed1_single_gpu/eval/checkpoint/valid_unseen/results.jsonl \
  Plain=outputs/SpreadsheetBench_qwen36_noskill_noprefix_test280_softskill_matched/eval/plain/valid_unseen/results.jsonl \
  HardSkill=outputs/SpreadsheetBench_qwen36_hard_skill_test280_softskill_matched/eval/hard_skill/valid_unseen/results.jsonl \
  Combined=outputs/SpreadsheetBench_combined_top0.05_shared_preserve_len8_epoch1_test280_softskill_matched/eval/checkpoint/valid_unseen/results.jsonl \
  FullVocabKL=outputs/SpreadsheetBench_combined_core10_full_vocab_skillkl_shared_preserve_len8_seed1_test280_softskill_matched/eval/checkpoint/valid_unseen/results.jsonl \
  SEKD=outputs/SpreadsheetBench_sekd_prefix_official_top20_len8_seed1_test280/eval/checkpoint/valid_unseen/results.jsonl \
  OPCD=outputs/SpreadsheetBench_opcd_prefix_official_top256_len8_seed1_test280/eval/checkpoint/valid_unseen/results.jsonl \
  --json-out outputs/SpreadsheetBench_hard_skill_seven_way_comparison.json
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

## 8. Entropy/EAC 与动态定位

目的：检验裸模型不确定度、动态 Skill residual、竞争 token margin 和连续定位权重，能否
比静态 Combined 更准确地找到值得训练的位置。所有实验均使用同一批61条成功轨迹，未
达到内部 gate 的候选不进入 Test280。

```bash
# 静态 Random/Entropy/JS/Additive/EAC Top-5% 对照。
bash scripts/run_spreadsheetbench_entropy_localization.sh

# 动态 Combined；配置可替换为 additive、weighted 或 G+C 版本。
CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_dynamic_combined.py \
  --config configs/spreadsheetbench/dynamic_combined_full_vocab_skillkl.yaml \
  --out_root outputs/SpreadsheetBench_dynamic_combined_v1_full_vocab_len8_seed1
```

核心结果：静态 Additive 17/40；Legacy Combined 16/40；EAC `.25/.5/1.0` 分别
11/9/16；Dynamic Combined 15/40；Dynamic Additive 16/40；locator-weighted KL
13/40；G+C 11/40。动态 Additive 虽把全局 full-vocabulary KL 降低29.12%，preservation
KL 却升到初始值3.36倍，执行率没有超过静态 Additive。

对应报告：

- [Entropy定位](spreadsheetbench_entropy_localization.md)
- [EAC全序列统计](spreadsheetbench_eac_full_sequence_weight_analysis.md)
- [Dynamic Combined](spreadsheetbench_dynamic_combined_v1.md)
- [Dynamic Additive](spreadsheetbench_dynamic_additive_skill_no_guard.md)
- [连续定位加权KL](spreadsheetbench_locator_weighted_kl_experiment.md)
- [G+C竞争项定位](spreadsheetbench_dynamic_gain_competitor_locator.md)

## 9. 四信号自适应定位

四信号为 `G/JS/C/R`。先用相同32-step预算筛选五种固定公式，再用4阶段、每阶段32步的
固定预算比较 A0--A3。Monitor12 不反向传播；Val40只在训练完成后评测一次。

```bash
python scripts/analyze_spreadsheetbench_meta_locator_signals.py \
  --locator-root \
    outputs/SpreadsheetBench_meta_locator_a0_fixed_g_js_full128_seed1/locators/round_00 \
  --trajectory-manifest \
    outputs/SpreadsheetBench_selective_stage2_manifests/combined_top0.10_core_coverage_ablation.jsonl \
  --out-root outputs/SpreadsheetBench_meta_locator_signal_audit_seed1

python scripts/analyze_spreadsheetbench_meta_locator_val40.py
```

结果：A0固定G+JS与A1固定四信号均为19/40；A2自适应加性14/40；A3自适应乘性15/40。
A3 的 TF residual ratio 最低（0.6528），但没有转化为执行优势。详细配置与逐题对照见
[四信号报告](spreadsheetbench_meta_adaptive_four_signal_locator.md)。

## 10. Future-Impact Locator v2

FIL-v2 用 forward-mode JVP 估计候选 token 更新对独立 outer trajectory KL 的未来影响。
脚本固定 Source39/Outer10/Cal12，内部 gate 失败时禁止访问Val40/Test280。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_fil_v2.py \
  --config configs/spreadsheetbench/future_impact_locator_v2.yaml \
  --out_root outputs/SpreadsheetBench_fil_v2_len8_seed1
```

数值 gate 通过，但方法 gate 失败：FIL Top10 更新使 outer loss 增加0.007647、preservation
恶化52.69%；Cal12为6/12，低于Random10的7/12。因此本版本按预注册规则停止。

## 11. 任务表示与 task-conditioned prefix

Stage A分析61个task-specific prompt的流形；Stage B用schema与Qwen任务表示做五折坐标/
行为代理预测；Stage C在fold1的49/12拆分上测试held-out prompt transfer。

```bash
python scripts/analyze_spreadsheetbench_task_prompt_manifold.py
CUDA_VISIBLE_DEVICES=0 python scripts/analyze_spreadsheetbench_task_representations.py
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate_spreadsheetbench_task_prompt_transfer.py
```

Stage A effective rank为56.18，PCA-16只解释40.35%，TF-IDF residual R2近似0。Stage B
没有表示通过raw-coordinate gate。Stage C目前只完成shared 3/12、mean49 7/12、
nn_task_spec 4/12；另外三个条件未完成，且没有访问Val40/Test280。

## 12. 结果口径与污染控制

- `val40` 用于开发与 checkpoint 选择；大量版本都看过 val40，因此不能把其中最高值当
  成无偏测试结论。
- `test280` 只能在方法冻结后评一次；若看过结果后继续修改算法，再次评测就会形成适应性
  测试污染。
- task-specific 61题来自 train80 成功轨迹，是 oracle/闭环诊断，不是独立测试集。
- 不同 `max_new_tokens`、generation batch、执行器版本或输入工作簿不能直接比较。
- smoke、未完成目录和早期不安全输入路径不进入主结果表。
- Test280 有两条参照线：地板是 Plain Qwen（97/280），天花板是 Hard-Skill 教师
  （118/280）。任何"压缩"类结论都应给出相对这两条线的位置，而不只是相对
  Original SoftSkill 的增量。
- 任务 `42930` 在数据集中 `n_cases=0`，在全部七次 Test280 中都于生成前判失败，
  实际生成279题、结果280行。七组一致，不影响逐题配对。
- 配置中的 `runtime.max_turns=30` 在 `local_hf` 后端下从不生效：
  `_build_vllm_eval_generator` 返回 `None`，`use_repair` 恒为 False。既有全部
  Test280 均为单次生成，新增评测必须保持单次。

## 13. 输出定位

Git 不保存 `outputs/`，复现实验后主要查看：

- `summary.json`：聚合设置与指标；
- `run.log`/`train.log`：完整终端日志；
- `best_prefix.pt`：验证集选择的共享 prefix；
- `eval/**/results.jsonl`：逐题自由生成与执行结果；
- `docs/experiment_results_overview.md`：本次工作区的统一审计表。

若要公开这些大型产物，参见 `docs/GITHUB_BACKUP.md`，不要直接取消 `.gitignore`。
