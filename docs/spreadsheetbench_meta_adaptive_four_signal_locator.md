# SpreadsheetBench 四信号自适应定位实验

## 1. 研究问题与冻结协议

本实验检验在动态 Combined 定位中加入四个 token 信号并自适应学习权重，能否把
teacher-forced 分布拟合改善转化为 SpreadsheetBench 自由生成执行能力。四个信号为：

- `G`：Hard-Skill 相对 no-Skill 对 gold token 的正向概率增益；
- `JS`：Hard-Skill 与当前 soft-prefix 全词表分布的 Jensen--Shannon 差异；
- `C`：Hard-Skill 是否改变 gold token 相对最强竞争 token 的 margin；
- `R`：学生比 Hard-Skill 教师更不确定的 resolved uncertainty，
  `[H(student)-H(teacher)]_+` 乘教师置信度。

所有训练使用 Qwen3.6-35B-A3B、共享长度 8 prefix、61 条成功轨迹、Top-10% token、
full-vocabulary forward Skill-KL 和固定 preservation loss。61 条轨迹固定分成 Train49 与
Monitor12；Monitor12 不反向传播，Val40 不用于训练、调权或 early stop，Test280 未访问。

## 2. 实验顺序

1. 在 61 条轨迹的 52,152 个 eligible token 上统计四信号相关性及 Top-10% 覆盖。
2. 用相同初始化和 32 optimizer step 筛选五个固定定位器，只看 Monitor12。
3. 固定预算 4 个阶段、每阶段 32 step，训练 A0、A1、A2、A3。
4. 冻结各自 checkpoint 后，各自在同一个 Val40 上单次自由生成；协议为
   `max_new_tokens=4096`、batch size 8、greedy、single shot。

## 3. 信号审计

### 3.1 Pooled Spearman 相关性

| 信号对 | Spearman rho |
|---|---:|
| G--JS | 0.3555 |
| G--C | 0.6265 |
| G--R | 0.7008 |
| JS--C | 0.0056 |
| JS--R | 0.3675 |
| JS--full KL | **0.9985** |
| C--full KL | -0.0237 |
| R--full KL | 0.3350 |

JS 几乎就是 full-vocabulary KL 的排序代理。C 提供高度不同的排名，而 R 与 G 已有较强
冗余。

### 3.2 初始 Top-10% 定位

| 定位器 | 权重/形式 | 与 F0 Jaccard | 捕获 G | 捕获 JS | 捕获 C | 捕获 R | 捕获 full KL |
|---|---|---:|---:|---:|---:|---:|---:|
| F0 | 0.5G+0.5JS | 1.0000 | 89.31% | 80.88% | 14.44% | 56.76% | **83.01%** |
| F1 | 0.45G+0.45JS+0.10C | 0.7753 | 87.70% | 78.37% | 25.00% | 56.73% | 80.28% |
| F2 | (0.5G+0.5JS)(1+0.25R) | 0.9818 | 89.31% | 80.87% | 14.54% | 57.26% | 82.95% |
| F3 | 0.4G+0.4JS+0.1C+0.1R | 0.7277 | 86.56% | 78.47% | 24.95% | **61.92%** | 79.79% |
| F4 | (0.45G+0.45JS+0.1C)(1+0.25R) | 0.7749 | 87.65% | 78.43% | 24.94% | 57.56% | 80.18% |

单独把 R 作为乘性项几乎不改变 F0 排名（Jaccard 0.9818）；C 和加性 R 才会实质改变
所选 token。

## 4. 固定 32-step 筛选

Monitor objective 为全序列 `Skill-KL + preservation + 0.1 * gold-NLL`。

| 方法 | Monitor loss | Monitor Skill-KL | Gold NLL | Monitor preserve | Train full-KL residual ratio |
|---|---:|---:|---:|---:|---:|
| F0 G+JS | 0.078696 | 0.041779 | 0.349255 | **0.001991** | 0.9378 |
| F1 G+JS+C | 0.080000 | 0.042640 | 0.350140 | 0.002347 | 0.9280 |
| F2 G+JS x R | 0.079487 | 0.041891 | **0.345460** | 0.003049 | 0.9220 |
| F3 four-additive | **0.078437** | **0.040345** | 0.348274 | 0.003264 | **0.9002** |
| F4 four-multiplicative | 0.082699 | 0.043780 | 0.355585 | 0.003360 | 0.9636 |

F3 的 Monitor total 与 Skill-KL 最好，因此与 F0 一同进入 full-128；但 F3 的
preservation 已明显更差。

## 5. Full-128 teacher-forced 结果

| 方法 | 定位/调权 | Steps | 最终 KL residual ratio | 最佳 Monitor | 最终 preservation KL | 训练耗时 |
|---|---|---:|---:|---:|---:|---:|
| A0 | 固定 0.5G+0.5JS | 128 | 0.8371 | 0.077206 | **0.006641** | 3163.7 s |
| A1 | 固定 0.4G+0.4JS+0.1C+0.1R | 128 | 0.7798 | 0.074580 | 0.007461 | **2961.9 s** |
| A2 | 自适应四信号加性 | 128 | 0.7925 | 0.075217 | 0.007825 | 4156.7 s |
| A3 | 自适应 effect + 乘性 R | 128 | **0.6528** | **0.069773** | 0.009699 | 3784.0 s |

A2 最终权重为 `[G,JS,C,R]=[0.4065,0.4256,0.0846,0.0833]`。A3 最终为
`[G,JS,C,d]=[0.4609,0.4658,0.0733,0.2440]`。自适应过程总体降低 C；A2 也降低 R，
A3 的 R 调制强度先升后降。部分轮次四个专家与 Monitor 梯度同时为负，说明相对调权
无法保证存在共同下降方向。

## 6. 冻结 Val40 自由生成

| 方法 | Cell | Sheet | 总成功率 | 执行失败 | 语义失败 | 平均响应字符数 |
|---|---:|---:|---:|---:|---:|---:|
| A0 fixed G+JS | 14/29 | **5/11** | **19/40 (47.5%)** | **3** | 18 | 3129.8 |
| A1 fixed four-additive | **15/29** | 4/11 | **19/40 (47.5%)** | **3** | 18 | 3006.7 |
| A2 adaptive additive | 11/29 | 3/11 | 14/40 (35.0%) | 12 | **14** | 3666.6 |
| A3 adaptive multiplicative | 11/29 | 4/11 | 15/40 (37.5%) | 6 | 19 | 3152.6 |
| Combined10 full-vocab KL | 11/29 | 4/11 | 15/40 (37.5%) | 7 | 18 | 2802.0 |
| OPCD | 16/29 | **5/11** | 21/40 (52.5%) | **3** | 16 | 2676.9 |
| SE-KD | **20/29** | 4/11 | **24/40 (60.0%)** | 4 | **12** | 3703.0 |
| Hard Skill | 17/29 | 4/11 | 21/40 (52.5%) | 4 | 15 | 3236.3 |

### 6.1 精确逐题配对

表中“胜/负”均以左侧 A0 为参照。

| 对比 | 对方胜 | 对方负 | Exact paired p |
|---|---:|---:|---:|
| A1 vs A0 | 5 | 5 | 1.0000 |
| A2 vs A0 | 5 | 10 | 0.3018 |
| A3 vs A0 | 3 | 7 | 0.3438 |
| Combined10 vs A0 | 3 | 7 | 0.3438 |
| OPCD vs A0 | 5 | 3 | 0.7266 |
| SE-KD vs A0 | 9 | 4 | 0.2668 |
| Hard Skill vs A0 | 8 | 6 | 0.7905 |

A0 与 A1 虽同为 19/40，但并非相同 19 题，而是 5 胜 5 负。A2 的退化主要来自执行
失败增加到 12 题。所有 Val40 配对在 40 题规模下均未达到 0.05 显著性阈值。

## 7. 结论

1. **当前 entropy 辅助信号没有改善自由生成执行能力。** 最强 entropy 方案 A3 的
   teacher-forced residual ratio 从 A0 的 0.8371 降至 0.6528，但 Val40 从 19/40 降至
   15/40。
2. **固定简单定位器最稳。** A0 与 A1 都是 19/40，优于原 Combined10 的 15/40；但仍
   低于 OPCD 的 21/40 与 SE-KD 的 24/40。
3. **teacher-forced 指标与执行迁移明显错位。** A3 拥有最低 KL 和 Monitor loss，却不是
   最佳生成模型。只优化成功轨迹上的局部分布仍可能把学生推离自由生成会访问的状态。
4. **preservation 是更可信的风险信号。** 四个 full 方法中，最终 preservation 越大，
   Val40 总体越差；样本只有四个，不能作显著性结论，但方向与失败结构一致。
5. **不进入 Test280。** 本轮没有证据表明 A1/A2/A3 优于 A0，故不应为这些候选消耗新的
   Test280 查询。下一步应先改善 teacher-forced 到 on-policy 的迁移目标，而不是继续搜索
   `G/JS/C/R` 的静态线性权重。

## 8. 产物

- 信号审计：`outputs/SpreadsheetBench_meta_locator_signal_audit_seed1/summary.json`
- A0--A3 训练：`outputs/SpreadsheetBench_meta_locator_a{0,1,2,3}_*_full128_seed1/`
- A0--A3 Val40：相应训练目录名加 `_val40/`
- 汇总脚本：`scripts/analyze_spreadsheetbench_meta_locator_val40.py`
- 所有训练 summary 和 Val40 summary 均记录 `test_split_accessed: false`。
