# SkillForge

**面向 Agent 成功轨迹的选择性 Soft Prompt 蒸馏实验框架。**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](pyproject.toml)
[![Upstream](https://img.shields.io/badge/Upstream-SoftSkill-blue)](https://github.com/xijia-tao/SoftSkill)

SkillForge 研究如何把自然语言 Skill 对冻结语言模型产生的关键行为变化，压缩到短长度
soft prompt 中。项目沿用 SoftSkill 的冻结模型、soft prefix、成功轨迹、Agent 任务和
验证集选择框架，重点探索以下问题：

- Skill 的影响是否集中在成功轨迹的少量 token 位置；
- 选择性蒸馏是否优于全轨迹 next-token prediction；
- 全词表分布信息能否改善关键 token 定位；
- 如何在学习关键位置的同时保持未选位置的裸模型行为；
- teacher-forced 分布拟合能否转化为 Agent 自由生成和执行成功；
- 受 boosting 启发的逐阶段 residual learner 是否能改善 soft prompt。

当前主实验任务是 SpreadsheetBench，冻结模型为 `Qwen/Qwen3.6-35B-A3B`，共享
soft prefix 长度为 8。SearchQA 用于核验原始训练链路，模型为 `Qwen/Qwen3.5-4B`，
prefix 长度为 32。

> 本仓库不是 SoftSkill 官方仓库。SoftSkill 只作为上游代码与实验基线；SkillForge 的
> 新增模块、实验结果和结论均属于本仓库的研究扩展。项目成员以本仓库 GitHub
> contributors 为准，不继承上游论文作者名单。

## 方法概览

### Selective Soft Prompt Distillation

对每条执行成功的教师轨迹，在相同 gold 上文下分别计算：

- `q_skill`：完整文本 Skill + 任务条件下的 Qwen 下一 token 分布；
- `p_plain`：无 Skill + 任务条件下的 Qwen 下一 token 分布。

Positive-gain 定位器使用 gold token 的概率增益：

```text
positive_gain(t) = max(log q_skill(y_t) - log p_plain(y_t), 0)
```

Combined 定位器再结合 Skill/No-Skill 完整输出分布的 JS 差异，使定位不只依赖当前
gold token 的概率。实际使用 Top-64 加 residual bucket 近似完整词表分布；平均覆盖概率
质量超过 99.8%。

共享 selective prefix 的主要目标为：

```text
L = CE(gold token，仅 selected core 与固定 EOS)
    + lambda_preserve * KL(no-Skill reference || soft-prefix，preserve 位置)
```

其中 preservation loss 用来限制 soft prompt 在未选位置破坏裸模型原有分布。

### PRCB

PRCB 是受 boosting 启发的逐阶段实验系列，包括 prefix row 分块更新、滑动重叠窗口、
teacher-forced monitor early stop、replay retention、alpha line search、独立 logit learner、
ensemble 回蒸馏和 low-rank learner。当前 v1--v6 没有在正式 test280 上稳定超过 Combined
共享 prefix，相关失败结果同样完整保留。

### Task-specific oracle

为61条成功训练轨迹分别训练独立长度8 prefix，再在同一个训练任务上让 Qwen 自由生成。
该实验用于检查 teacher-forced 分布拟合能否闭环到同题执行，不代表验证集或测试集泛化。

## 项目目录

```text
SkillForge/
├── configs/
│   ├── _base_/                    # 公共基础配置
│   ├── searchqa/                  # SearchQA 配置
│   ├── spreadsheetbench/          # 复现、Selective 与评测配置
│   └── local/                     # 本地 API 配置，已被 Git 忽略
├── ckpt/                          # 文本 Skill；大模型 checkpoint 不进入 Git
├── data/
│   ├── *_id_split/                # 可公开的数据划分 manifest
│   └── README.md                  # 数据来源和准备说明
├── skillopt/
│   ├── envs/                      # SearchQA、SpreadsheetBench、ALFWorld 等环境
│   └── softprefix/                # soft prefix、Selective loss、PRCB、ensemble
├── scripts/
│   ├── data/                      # 数据集准备脚本
│   ├── train/                     # 原始训练与评测入口
│   ├── analyze_*                  # token 与全分布定位分析
│   ├── run_spreadsheetbench_*     # 主要实验启动脚本
│   └── train_spreadsheetbench_*   # PRCB 与 task-specific 训练器
├── tests/                         # 单元与回归测试
├── docs/
│   ├── EXPERIMENTS.md             # 每项实验的设置、目的和命令
│   ├── experiment_results_overview.md
│   ├── spreadsheetbench_selective_results_tables.tex
│   └── GITHUB_BACKUP.md           # GitHub 与大型制品管理
├── outputs/                       # 本地实验输出，已被 Git 忽略
├── rollouts/                      # 本地教师轨迹，已被 Git 忽略
└── README.md
```

## 统一实验设置

| 项目 | SearchQA | SpreadsheetBench |
|---|---|---|
| 冻结模型 | Qwen3.5-4B | Qwen3.6-35B-A3B |
| Prefix length | 32 | 8 |
| 数据划分 | val64、test1400 | train80、val40、test280 |
| 教师监督 | QA answer | GPT-5.5 执行成功代码轨迹 |
| 成功教师轨迹 | 按原数据链路 | 61/80 |
| 共享实验 | 单一 prefix | 单一 prefix 跨任务 |
| Task-specific | 未使用 | 61个任务分别训练独立 prefix |
| 主随机种子 | 1 | 1 |

SpreadsheetBench 的定位、Selective、PRCB 与 task-specific 实验均复用同一批61条成功
轨迹。val40 用于开发和 checkpoint 选择；test280 只用于冻结方法后的正式评测；
task-specific 61题来自训练集，不能当作测试结果。

## 当前全部实验结果

结果快照：2026-08-14 UTC。不同表格使用不同评测集或协议，不应跨表直接排序。

### 1. 原始链路复现

| 任务 | 设置 | 评测集 | 结果 | 状态 |
|---|---|---:|---:|---|
| SearchQA | Qwen3.5-4B，prefix 32 | val64 | hard 76.56%，soft 82.45% | 完成 |
| SearchQA | 同上 | test1400 | hard 77.14%，soft 84.33% | 完成 |
| SpreadsheetBench full-trajectory CE | Qwen3.6-35B-A3B，prefix 8 | val16 | 4/16（25.00%） | 完成 |
| SpreadsheetBench full-trajectory CE | 同上，matched 8192/b2 | test280 | 85/280（30.36%） | 完成 |
| ALFWorld | 已有环境、配置和 split manifest | -- | 尚无本地结果 | 未复现 |

SearchQA 三个 epoch loss 为 0.3103、0.2286、0.1951。SpreadsheetBench paper-style
loss 从 0.3910 降到 0.3471，但自由生成验证没有随 loss 单调提高。

> SearchQA 结果中的 `hard`/`soft` 是该评测器的两种评分口径，不代表 Hard Prompt 与
> Soft Prompt 两种输入条件。

### 2. GPT-5.5 教师轨迹

| 指标 | 结果 |
|---|---:|
| train80 总任务 | 80 |
| 执行成功轨迹 | 61（76.25%） |
| Cell-level 成功 | 42/53 |
| Sheet-level 成功 | 19/27 |
| 可定位目标 token | 54,929 |

这些轨迹由本项目使用完整文本 Skill + GPT-5.5 重新生成，并非上游论文发布的原始缓存。

### 3. Positive-gain 集中度

| Top token 比例 | 全局正增益捕获 | 平均逐轨迹捕获 | 随机期望 |
|---:|---:|---:|---:|
| 1% | 42.11% | 41.97% | 0.99% |
| 2% | 59.72% | 59.13% | 1.99% |
| 5% | 84.05% | 82.99% | 4.99% |
| 10% | 96.12% | 95.42% | 9.97% |
| 20% | 99.76% | 99.62% | 19.84% |

61条轨迹全部满足预注册标准 `C(10%)>=0.60` 且 `C(20%)>=0.80`，支持“Skill 影响
集中在少量 token 位置”的定位假设。

### 4. Positive 与 Combined 定位统计

| Top-5% 定位器 | 正增益捕获 | 精确 JS 捕获 | Combined 质量捕获 |
|---|---:|---:|---:|
| Positive | 81.91% | 44.69% | 95.90% |
| Combined | 76.23% | 50.61% | 97.56% |

两组位置平均 Jaccard 为0.6974。Combined 牺牲少量 gold-token 正增益覆盖，换取更高的
全分布差异和 Combined 质量覆盖。

### 5. 共享 Selective Distillation：val40

| 方法 | Core CE token/覆盖 | Preserve token | Epoch | Val40 |
|---|---:|---:|---:|---:|
| Full-trajectory CE | 54,990 / 100% | 0 | 3 | 11/40（27.50%） |
| Random 5% core | 2,838 / 5.06% | 0 | 3 | 15/40（37.50%） |
| Positive 5% core | 2,838 / 5.06% | 0 | 2/3 | 13/40（32.50%） |
| Random 5% + independent KL | 2,838 | 2,777 | 1 | 15/40（37.50%） |
| Positive 5% + independent KL | 2,838 | 2,777 | 1 | **18/40（45.00%）** |
| Positive 5% L2/R8 window + KL | 19,424 / 35.25% | 2,777 | 1 | 12/40（30.00%） |
| Positive 5% + shared KL | 2,838 | 2,777 | 1 | 10/40（25.00%） |
| Combined 5% + shared KL | 2,838 | 2,777 | 1 | **16/40（40.00%）** |

Positive independent-KL 的18/40不是严格定位器对照，因为其 preserve 位置与其他实验不
完全相同。在严格共享 preserve 下，Combined 为16/40，Positive 为10/40；配对检验
`p=0.146`。固定 L2/R8 扩窗覆盖35.25% token 仍下降到12/40，不支持人工固定扩窗。

### 6. 被后续安全路径取代的早期诊断

| 运行 | Val40 | 说明 |
|---|---:|---|
| Initial Full CE，generation batch 2 | 4/40（10.00%） | 暴露 generation batch 依赖 |
| Initial Full CE，generation batch 8 | 12/40（30.00%） | batch 诊断 |
| Fixed Full CE | 14/40（35.00%） | 后被 safe run 取代 |
| Fixed Random 5% | 16/40（40.00%） | 后被 safe run 取代 |
| Fixed Positive 5% | 15/40（37.50%） | 只完成2/3 epoch |

这些数字保留用于审计，不进入最终方法比较。

### 7. 冻结 checkpoint：test280

| 方法 | 生成协议 | 成功率 | 相对原始 SoftSkill | 配对 p（vs SoftSkill） |
|---|---|---:|---:|---:|
| Original SoftSkill | matched 8192/b2 | 85/280（30.36%） | -- | -- |
| Positive 5% + KL | unmatched 4096/b8 | 91/280（32.50%） | +6题 | 不作正式比较 |
| Combined 5% + shared KL | matched 8192/b2 | **101/280（36.07%）** | **+16题/+5.71pp** | **0.0479** |
| PRCB-v5 | matched 8192/b2 | 96/280（34.29%） | +11题/+3.93pp | 0.1690 |
| Plain Qwen，无 Skill/Prefix | matched 8192/b2 | 97/280（34.64%） | +12题/+4.29pp | 0.1263 |

Combined 是当前最佳共享 prefix test280 结果，并显著超过本项目复现的原始 SoftSkill
基线；但 Combined-only/Plain-only 为40/36，配对 `p=0.731`，尚不能证明优于裸 Qwen。

### 8. PRCB 系列：val40

所有版本从同一 Combined 5% + shared KL checkpoint（16/40）开始。定位只沿成功轨迹
gold 上文进行，val40 不参与反向传播或 early stop。

| 方法 | 核心设置 | Optimizer step | Val40 | Exec fail |
|---|---|---:|---:|---:|
| Combined warm start | 初始共享 checkpoint | -- | 16/40（40.00%） | 9 |
| PRCB-v1 | 梯度探针选两个 prefix row | 32 | 14/40（35.00%） | 9 |
| PRCB-v2 tail-to-head | 67→45→23→01 | 32 | 11/40（27.50%） | 10 |
| PRCB-v2 head-to-tail | 01→23→45→67 | 32 | 15/40（37.50%） | 11 |
| PRCB-v3 | gold 轨迹 margin locator | 32 | 16/40（40.00%） | 8 |
| PRCB-v4 overlap1 | 01→12→…→67，总步数匹配 | 32 | 16/40（40.00%） | 9 |
| PRCB-v4-ES window2 | 49/12 monitor early stop | 104 | 14/40（35.00%） | 12 |
| PRCB-v4-ES window5，漏末窗 | 01234→12345→23456 | 38 | 10/40（25.00%） | 11 |
| PRCB-v4-ES window5，完整 | 再加入34567 | 46 | 13/40（32.50%） | 6 |
| PRCB-v5 | replay retention + alpha line search | 74 | **18/40（45.00%）** | 6 |
| PRCB-v6 student | 独立 length-8 learner + 回蒸馏 | 28 | 16/40（40.00%） | 9 |
| PRCB-v6 ensemble | 不回蒸馏，直接组合 learner | 28 | 16/40（40.00%） | 8 |
| PRCB-v6 low-rank r2 | 训练25.1% learner 参数 | 18 | 16/40（40.00%） | 9 |

PRCB-v5 的 val40 最高值没有在 test280 保持：96/280，低于 Combined 的101/280。
v6 第二阶段 line search 得到 `alpha=0`，未蒸馏 ensemble 也没有改善，说明问题不主要
来自最后的 student 回蒸馏。

### 9. Task-specific oracle 与覆盖率消融

| 条件 | 同题自由生成成功 | Core Skill-KL closure | 状态 |
|---|---:|---:|---|
| Plain Qwen | 26/61（42.62%） | -- | 完成 |
| 完整文本 Skill | 32/61（52.46%） | -- | 完成 |
| Combined 5%，旧 shared-preserve | 26/61（42.62%） | 63.49% | 完成 |
| Combined 5%，共同 preserve | 28/61（45.90%） | 62.18% | 完成 |
| Combined 10%，共同 preserve | **35/61（57.38%）** | 56.86% | 完成 |
| Combined 20%，共同 preserve | 33/61（54.10%） | 50.74% | 完成 |

逐题变化：

| 比较 | 新增成功 | 丢失成功 | 净变化 | 配对 p | Exec fail变化 | 语义失败变化 |
|---|---:|---:|---:|---:|---:|---:|
| 5% → 10% | 11 | 4 | +7 | 0.1185 | 15→8 | 18→18 |
| 10% → 20% | 5 | 7 | -2 | 0.7744 | 8→7 | 18→21 |
| 5% → 20% | 8 | 3 | +5 | 0.2266 | 15→7 | 18→21 |

5%到10%的主要收益是代码执行失败减少；继续扩大到20%后结构收益趋于饱和，语义错误
增加。10%是当前固定 seed、固定32步预算下的观察最优点，仍需多种子复验。

`Core Skill-KL closure` 表示 soft prefix 消除了多少“裸模型到完整文本 Skill 教师”的
core-token KL 差距。它是 teacher-forced 指标，不等价于自由生成成功率；本实验中覆盖率
增加时 closure 下降，但自由生成先升后降，进一步说明两种指标不能互相替代。

## 当前结论

| 问题 | 当前证据 |
|---|---|
| Skill 影响是否稀疏集中 | 是；Top-5%捕获84.05%正增益，Top-10%捕获96.12% |
| Selective 是否优于 Full CE | val40 上多数组合优于 Full CE 11/40 |
| 全分布信息是否有用 | Combined 改变定位并在共享 preserve 对照中16/40 vs Positive 10/40 |
| 固定人工扩窗是否有效 | 否；覆盖35.25%时只有12/40 |
| Combined 是否优于原始 SoftSkill | test280：101/280 vs 85/280，配对 p=0.0479 |
| Combined 是否优于裸 Qwen | 尚不能；101/280 vs 97/280，配对 p=0.731 |
| 当前 PRCB boosting 是否有效 | 未形成稳定 test 增益 |
| 增大 core 覆盖是否单调有效 | 否；28→35→33，10%为当前观察最优 |

主要瓶颈是 teacher-forced gold-state 分布拟合与自由生成 Agent 执行成功率之间的目标和
状态不匹配。soft prompt 可以降低局部 KL 或 CE，却仍可能在自由生成中出现早期偏离、
变量/API 不一致、代码框架不完整或工作簿语义错误。

## 安装

```bash
git clone https://github.com/ZjianJ/SkillForge.git
cd SkillForge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,softprefix,qwen]"
```

本地模型是默认路径，不要求部署远程 Qwen API。没有本地快照时使用 Hugging Face model ID；
已有持久化模型时设置：

```bash
export SPREADSHEETBENCH_MODEL=/absolute/path/to/Qwen3.6-35B-A3B/snapshot
export HF_HOME=/absolute/path/to/huggingface/cache
export CUDA_VISIBLE_DEVICES=0
```

## 准备数据

```bash
python scripts/data/prepare_searchqa.py
python scripts/data/prepare_spreadsheetbench.py
```

仓库只保存公开 split manifest，不保存完整 SpreadsheetBench 工作簿。其他数据集准备方式
见 [data/README.md](data/README.md)。

## GPT-5.5 教师配置

只有重新生成教师轨迹时需要 GPT API；复用缓存进行定位、训练和本地评测不需要再次调用。

```bash
mkdir -p configs/local
cp configs/spreadsheetbench/teacher_rollout_gpt55.example.yaml \
  configs/local/spreadsheetbench_paper_gpt55.local.yaml
```

在 `configs/local/spreadsheetbench_paper_gpt55.local.yaml` 填入 endpoint 和 API key。
`configs/local/` 已被 `.gitignore` 排除，禁止把真实 key 写入公开配置。

重新生成轨迹并运行 paper-style 基线：

```bash
bash scripts/reproduce_spreadsheetbench_paper.sh
```

已有轨迹缓存时：

```bash
bash scripts/train_spreadsheetbench_paper_from_cache.sh
```

## 运行主要实验

### SearchQA 复现

```bash
bash scripts/reproduce_searchqa_gh200.sh
```

### 阶段一：token 定位

```bash
bash scripts/run_spreadsheetbench_selective_stage1.sh
python scripts/analyze_spreadsheetbench_full_distribution.py
python scripts/prepare_spreadsheetbench_selective_stage2.py
```

### 阶段二：Selective 与 Combined

```bash
bash scripts/run_spreadsheetbench_selective_stage2.sh
bash scripts/run_spreadsheetbench_random_core_preservation.sh
bash scripts/run_spreadsheetbench_positive_preservation.sh
bash scripts/run_spreadsheetbench_positive_window_preservation.sh
bash scripts/run_spreadsheetbench_full_distribution_locator_validation.sh
```

### PRCB v1--v6

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

### Task-specific 5%/10%/20% coverage

```bash
GPU_IDS=0 bash scripts/run_spreadsheetbench_coverage_ablation.sh
```

完整超参数、正式 test280 命令、early-stop 设置和输出目录见
[实验复现手册](docs/EXPERIMENTS.md)。更细的逐题诊断见
[task-specific 分析](docs/spreadsheetbench_task_specific_per_task_analysis.md)。

## 输出与 GitHub 管理

以下内容默认不会进入 Git：

- `outputs/`、`rollouts/`、`logs/`；
- `.env`、`configs/local/`；
- `*.pt`、`*.safetensors` 和模型缓存；
- 完整数据集与 SpreadsheetBench 工作簿。

代码、配置和报告使用普通 Git；大型 checkpoint 与逐题输出应通过 Git LFS、GitHub Release
或对象存储发布，并附 SHA-256 manifest。详见 [GitHub 备份指南](docs/GITHUB_BACKUP.md)。

## 上游依赖与致谢

SkillForge 基于以下开源工作构建：

- [SoftSkill](https://github.com/xijia-tao/SoftSkill)：冻结模型、soft prefix、成功轨迹与
  验证集选择框架；对应工作为 [SoftSkill: Behavioral Compression for Contextual Adaptation](https://arxiv.org/abs/2606.20333)。
- [Microsoft SkillOpt](https://github.com/microsoft/SkillOpt)：Agent Skill 优化与环境框架。

这些上游项目的作者不是 SkillForge 的自动作者或成员。需要引用其方法时，请使用各自仓库
或论文提供的正式引用信息。

## License

本仓库保留上游代码中的 MIT License 与版权声明，见 [LICENSE](LICENSE)。
