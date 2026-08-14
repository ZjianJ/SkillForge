# PRCB-v5：低计算成本版本

PRCB-v5 保留 v4-ES 的成功轨迹定位、49/12 固定划分、长度 2 的重叠窗口和
teacher-forced early stopping。测试集不参与定位、训练或 early stopping。

## 新增机制

在阶段 (s) 开始定位时，已有的一次 student 前向同时缓存当前 soft prefix 在
上一阶段 replay gold token 上的 Top-64 概率和一个剩余词表质量桶。训练损失为

\[
L=L_{CE}+L_{preserve}+0.5L_{SkillKL}+0.5L_{margin}
  +L_{retain},
\]

其中

\[
L_{retain}=KL(p_{s,\mathrm{start}}^{64+R}\|p_{s,\mathrm{current}}^{64+R}).
\]

它约束新窗口不要破坏上一阶段已经学习的位置。第一个阶段没有 replay token，
因此该项自动为零。

阶段内 early stopping 仍只监控固定的 `alpha=0.25` 候选。阶段结束后仅对最佳
raw update 做一次低成本线搜索：`{0, 0.125, 0.25, 0.5}`。其中 `0` 直接复用
阶段起点，`0.25` 复用已有 monitor 结果，只有 `0.125` 和 `0.5` 产生额外前向。
候选必须同时满足 preservation guard 和 replay retention KL 不超过 `0.02`；
若没有安全候选优于 `alpha=0`，整阶段拒绝并回滚。

## 正式运行

```bash
cd /path/to/softskill
mkdir -p outputs/SpreadsheetBench_prcb_v5_retention_alpha_len8_seed1
CUDA_VISIBLE_DEVICES=0 python -u scripts/train_spreadsheetbench_prcb_v5.py \
  2>&1 | tee outputs/SpreadsheetBench_prcb_v5_retention_alpha_len8_seed1/run.log
```

每阶段的 `selected_alpha`、`stage_rejected`、`retention_top64_kl_loss` 和完整
`alpha_line_search` 会写入 `round_XX/summary.json`。最终只评估 40 题验证集。
