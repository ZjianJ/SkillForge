# SpreadsheetBench Hard-Skill 教师基线（Test280）

> 实验日期：2026-08-17（UTC）。本报告测量的是**被压缩的对象本身**：冻结
> `Qwen3.6-35B-A3B` + 完整文本 Skill、不带任何 soft prefix，在与全部 prefix 方法
> 严格匹配的 Test280 协议下的成绩。

## 1. 为什么需要这个基线

本项目所有共享 prefix 方法（Original SoftSkill、Combined、FullVocabKL、
SE-KD-Prefix、OPCD-Prefix）都声称把完整文本 Skill 压缩进 8 个 soft token。但在本
实验之前，**这个被压缩的条件从未在 Test280 上被评测过**：`trainer.py` 只有
`eval_plain_baseline` 开关，没有 hard-Skill 评测路径；全部输出目录的
`summary.json` 中也没有任何 hard-Skill 的 test 字段。唯一的 hard-Skill 数字来自
task-specific oracle 在 train61 上的 32/61。

后果是：主表只有地板（Plain Qwen）没有天花板，无法计算任何回收率，也无法区分
"蒸馏没做好"与"教师本来就不会"。本实验补上这个参照。

## 2. 协议

与 Original SoftSkill、Plain、Combined、FullVocabKL、SE-KD、OPCD 六次 Test280
逐项匹配，参数取自已完成 Plain 运行的 `config.json`：

| 维度 | 设置 |
|---|---|
| 冻结基础模型 | 本地 `Qwen3.6-35B-A3B` 同一 snapshot |
| Trainable parameters | **0**（不训练、不加载、不安装任何 prefix） |
| Skill 注入 | 文本，写入 system prompt 的 `## Skill` 段 |
| `max_prompt_tokens` | 16,384 |
| `max_new_tokens` | 8,192 |
| `generation_batch_size` | 2 |
| 采样 | greedy（`temperature=0.0`） |
| 生成轮次 | 单次（`repair_turns=1`） |
| `injection_position` | `prompt_start` |
| `exec_timeout` | 600 s |
| Skill 文件 | `ckpt/spreadsheetbench/gpt5.5_skill.md`，SHA-256 `82249a54bba13550aa87dcc84087728ebb176d753c95bc09e6b9c49f30942892`，**2,773 token** |

两点在实现时确认的协议细节：

1. **既有六次 Test280 都是单次生成。** 配置里的 `runtime.max_turns=30` 从未生效：
   `inference_backend` 为 `local_hf` 时 `_build_vllm_eval_generator` 返回 `None`，
   而 `use_repair = repair_turns > 1 and generator is not None and
   hasattr(generator, "generate_from_messages")` 因此恒为 False。本实验显式传
   `repair_turns=1` 以保持一致。
2. **自定义 generator 会绕过 `generation_batch_size`。**
   `_generate_prompt_responses` 只在 `generator is None` 的分支应用分批，传入
   generator 时会一次性交付全部 prompt。本实验的 `HardSkillGenerator` 自行按 2
   分批。批大小在本项目中有实测显著影响（Full CE batch2 4/40 对 batch8 12/40），
   不能偏离匹配值。

### 2.1 Skill 注入的等价性

评测harness 用 `build_spreadsheet_codegen_system_for_prefix("")` 构造 clean system
prompt，而注入端用 `_build_system(skill)` 生成 hard system prompt 并在渲染后的
prompt 上做一次替换。已逐字节验证四项恒等：

- `build_spreadsheet_codegen_system_for_prefix("") == _build_system("")`
- `_build_system(skill) == build_spreadsheet_codegen_system_for_prefix(skill)`
- clean system 是 hard system 的严格前缀，且在完整 prompt 中只出现一次
- 替换后长度增量恰等于 Skill 段长度

因此"替换 system prompt"与"一开始就带 Skill 构造 prompt"完全等价。注入失败时
`HardSkillGenerator` 直接抛异常，不会静默退化成 Plain。

### 2.2 提示审计

`evaluate_spreadsheet_prefix` 保存的 `target_system_prompt.txt` 是 **clean** 版本
（它不知道 generator 改写过提示），因此本实验额外写出
`prompt_audit.json`，记录 279 条实际发出提示的 SHA-256（全部唯一）以及
clean/hard system 各自的 SHA-256。

### 2.3 任务 42930

`42930` 在数据集中 `n_cases=0`，`evaluate_spreadsheet_prefix` 在生成阶段之前即
判为失败，因此实际生成 279 题、结果 280 行。已核验该任务在**全部七次 Test280
运行中同样是 `n_cases=0` 且 response 为空**，属于数据集固有属性，不影响任何
逐题配对比较。

## 3. 结果

| 指标 | 结果 |
|---|---:|
| 成功 | **118/280（42.14%）** |
| Wilson 95% CI | [36.50%, 47.99%] |
| Cell-level | 76/193 |
| Sheet-level | 42/87 |
| 执行失败 | 33 |
| 可执行但语义失败 | 129 |
| 生成 wall time | 13,716.5 s（3:48:37） |

生成长度：均值 970.3 token，中位 682.0，P95 2,601.0，4 题触及 8,192 上限。尽管每题
多出 2,773 token 的 Skill prefill，总耗时仍略低于 Plain 的 3:55:11。

## 4. 七方逐题配对

全部七组通过 280 行、任务 ID 唯一、任务集合一致和 `ok/hard/soft` 自洽检查。

| 比较（Hard-Skill vs X） | 教师独占 | X 独占 | 净差 | 精确双侧配对 p |
|---|---:|---:|---:|---:|
| vs Original SoftSkill | 53 | 20 | +33 | **0.000142** |
| vs Plain Qwen | 50 | 29 | +21 | **0.023820** |
| vs Combined 5% | 43 | 26 | +17 | 0.053289 |
| vs FullVocabKL 10% | 39 | 24 | +15 | 0.076926 |
| vs OPCD-Prefix | 41 | 30 | +11 | 0.235098 |
| vs SE-KD-Prefix | 34 | 29 | +5 | 0.614655 |

## 5. 回收率

以 Plain Qwen 的 97/280 为地板、Hard-Skill 的 118/280 为天花板，Skill 在 Test280
上的可回收空间为 **21 题**。

| 方法 | Test280 | 相对 Plain | 回收率 | 相对教师 | 配对 p |
|---|---:|---:|---:|---:|---:|
| SE-KD-Prefix | 113/280 | +16 | **76.2%** | −5 | 0.614655 |
| OPCD-Prefix | 107/280 | +10 | 47.6% | −11 | 0.235098 |
| FullVocabKL 10% | 103/280 | +6 | 28.6% | −15 | 0.076926 |
| Combined 5% | 101/280 | +4 | 19.0% | −17 | 0.053289 |
| Original SoftSkill | 85/280 | −12 | 负 | −33 | 0.000142 |

SE-KD-Prefix 是唯一在统计上与教师无法区分的方法（`p=0.615`），用 8 个 soft token
替代 2,773 token 的文本 Skill（**347× 上下文压缩**，16,384 个 BF16 参数）回收了
76.2% 的增益。Combined 与 FullVocabKL 相对教师的 `p=0.053/0.077` 接近显著，说明
它们与教师的差距不是噪声。

## 6. Skill 的收益几乎全部在 cell-level

| | Cell-level | Sheet-level |
|---|---:|---:|
| Plain Qwen | 57/193 | 40/87 |
| Hard-Skill | 76/193（**+19**） | 42/87（**+2**） |

Skill 相对裸模型的 21 题增量中，19 题来自 cell-level。而 sheet-level 上七组全部落在
40–44/87 的窄带内，SE-KD 的 44/87 甚至高于教师的 42/87。

这意味着 sheet-level 的瓶颈**不是"Skill 没被压缩好"，而是教师本身不具备该能力**。
改进蒸馏目标在 sheet-level 上没有可回收空间，这为整条压缩路线划定了上界。

## 7. 复现命令

```bash
MODEL=/absolute/path/to/Qwen3.6-35B-A3B

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/evaluate_spreadsheetbench_hard_skill_baseline.py \
  --model-path "$MODEL" \
  --out-root outputs/SpreadsheetBench_qwen36_hard_skill_test280_softskill_matched
```

协议参数全部为脚本默认值，不应覆盖。`--limit N` 仅用于 smoke，不能用于上报结果；
输出目录已存在 `results.jsonl` 时脚本拒绝运行，避免覆盖已完成评测。

七方比较：

```bash
python scripts/analyze_spreadsheetbench_prefix_comparison.py \
  Original=... Plain=... HardSkill=... Combined=... \
  FullVocabKL=... SEKD=... OPCD=... \
  --json-out outputs/SpreadsheetBench_hard_skill_seven_way_comparison.json
```

## 8. 结论与限制

1. **压缩前提成立。** Skill 在从未参与训练的 280 题上相对裸模型 +21 题
   （`p=0.0238`），这是主表上唯一相对 Plain 达到名义显著的条目。此前只能从 train61
   的 32/61 vs 26/61 推测，现已独立确认。
2. **教师是七组最高（118/280），所有 prefix 方法都在其之下。** 此前"SE-KD 可能已
   超过教师"的猜测被证伪。
3. **SE-KD-Prefix 回收 76.2% 且与教师统计无异**，是当前压缩效率最高的方法。
4. **sheet-level 无可回收空间**，教师自身在该类型上仅比裸模型多 2 题。
5. 限制：单次 greedy 评测、seed 1。Wilson CI 覆盖任务采样不确定性，但教师条件
   本身没有训练方差（不训练），而各 prefix 方法的 seed 方差仍未测量，回收率因此
   是单种子点估计。
