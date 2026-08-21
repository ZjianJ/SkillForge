# Dynamic Combined v1：随 soft prompt 状态重定位

## 1. 目标

静态 Combined 只计算一次 Hard-Skill 与 No-Skill 的差异，训练期间不会知道当前
soft prompt 已经学会了哪些位置。Dynamic Combined v1 在每个局部收敛阶段结束后，
沿同一批成功轨迹 gold 上文重新计算 Hard-Skill 与**当前 soft prompt**的完整词表
残差，重新分配下一阶段的 Top-10% core。

## 2. 动态定位

先用原始缓存限定文本 Skill 真正有益的位置：

\[
b_t=\mathbf 1[\log q_t(y_t)-\log p_t^{\rm plain}(y_t)>0].
\]

第 \(s\) 次重定位时，用当前 soft prompt 分布 \(p_t^{(s)}\) 计算：

\[
r_t^{(s)}=b_t\,\max(\log q_t(y_t)-\log p_t^{(s)}(y_t),0)
\,JS(q_t,p_t^{(s)}).
\]

每条轨迹独立选择 \(r_t^{(s)}\) 的 Top-10%。固定 preservation 位置从候选中排除，
避免同一位置同时拟合 Hard-Skill 和保持 No-Skill。当前版本不使用人工扩窗。

## 3. 每阶段训练与局部收敛

core 使用完整词表 forward KL，EOS 使用 one-hot CE，preservation 沿用固定
Top-64+residual-bucket KL。每次重定位后重置 AdamW 状态，避免旧 core 的动量污染新
阶段。

固定12条成功轨迹构成 teacher-forced monitor panel；monitor 前向永不反向传播，也不
涉及 Val40/Test280。为了与静态 Combined10 保持相同的61条训练支持，这12题仍可在正常
训练 batch 中出现；因此这里监测的是固定训练状态上的优化停滞，不把它解释成泛化指标。
每4个 optimizer step 在该阶段固定的 core 上计算同一 monitor 目标。参数为：

- `min_steps_per_stage=8`
- `max_steps_per_stage=32`
- `monitor_interval_steps=4`
- `monitor_patience=3`
- `min_relative_monitor_improvement=0.2%`

连续3次没有达到0.2%相对改善时，回滚到该阶段 monitor 最低的 prefix，然后重新定位。
若一直改善，则32步安全上限后重定位。

## 4. 全局停止与回滚

每次新定位都在全部61条成功轨迹上计算可比较的全局动态残余质量
\(M_s=\sum_t r_t^{(s)}\)。满足任一条件即停止：

1. \(M_s/M_0\le 10\%\)；
2. 原始 Skill-beneficial 位置的平均完整词表 KL 不超过0.02；
3. 连续两轮全局 residual 的相对改善小于0.2%；
4. preservation KL 相对初始定位恶化超过10%；
5. 完成4次重定位；
6. 已无正动态残差。

最终加载所有通过 preservation guard 的定位轮次中全局 residual mass 最低的 prefix，
而不是无条件使用最后一轮。

## 5. 审计产物

每轮输出：

```text
locators/round_XX/summary.json
locators/round_XX/manifest.jsonl
locators/round_XX/arrays/<task>.npz
stage_XX_best_prefix.pt
```

逐题数组包含 residual gain、完整词表 KL、完整词表 JS、动态 Combined 分数和新 core；
汇总记录 residual mass、Top-10%质量捕获、与上一轮 core 的 Jaccard、preservation KL、
实际 optimizer step 与停止原因。

## 6. 运行

```bash
cd /home/u6ow/zijian.u6ow/softskill

OUT=outputs/SpreadsheetBench_dynamic_combined_v1_full_vocab_len8_seed1
MODEL=/home/u6ow/zijian.u6ow/model_cache/huggingface/models--Qwen--Qwen3.6-35B-A3B/snapshots/995ad96eacd98c81ed38be0c5b274b04031597b0

CUDA_VISIBLE_DEVICES=0 python -u \
  scripts/train_spreadsheetbench_dynamic_combined.py \
  --config configs/spreadsheetbench/dynamic_combined_full_vocab_skillkl.yaml \
  --out_root "$OUT" \
  --model_name "$MODEL" \
  2>&1 | tee "${OUT}_driver.log"
```

训练期间 `evaluation.eval_test=false` 且 `test_env_num=0`。应先根据 Val40 和动态残差
诊断决定是否冻结检查点；正式 Test280 仍只允许测试一次。
