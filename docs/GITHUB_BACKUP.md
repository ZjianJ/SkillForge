# GitHub 备份与版本管理

本仓库采用“代码与小型元数据进入 Git，大型实验产物进入制品存储”的策略。普通 Git
提交包含源码、配置、测试、文档和数据划分 manifest；不会包含 API key、模型权重、完整
数据集、教师 rollout、训练 checkpoint 或生成结果。

## 1. 提交边界

应提交：

- `skillopt/`、`scripts/`、`configs/` 中的公开代码与配置；
- `tests/`；
- `README.md`、`REPRODUCTION_LOCAL.md` 和 `docs/` 中的实验报告；
- `data/*_id_split/`、`data/alfworld_path_split/` 等小型划分 manifest；
- `ckpt/**/gpt5.5_skill.md` 等允许公开的文本 Skill。

不应提交：

- `.env`、`configs/local/`、`*.local.yaml`；
- `outputs/`、`rollouts/`、`logs/`；
- `*.pt`、`*.safetensors`、模型缓存；
- SpreadsheetBench 工作簿和其他受许可约束的原始数据。

上述规则已经写入 `.gitignore`。API 配置的公开模板是
`configs/spreadsheetbench/teacher_rollout_gpt55.example.yaml`。

## 2. 首次备份前检查

在项目根目录运行：

```bash
git status --short --branch
git diff --check
git status --ignored --short | less

# 检查大文件；普通 Git 中原则上不保留模型或输出文件。
find . -type f -not -path './.git/*' -size +50M -print

# 检查常见密钥形态。命中公开占位符时人工确认即可。
rg -n 'sk-[A-Za-z0-9_-]{16,}|api[_-]?key\s*[:=]' \
  --glob '!configs/local/**' --glob '!.env' --glob '!outputs/**' \
  --glob '!rollouts/**' .
```

不要使用 `git add -f` 强行加入被忽略的目录。

## 3. 建立自己的 GitHub 远端

当前仓库可能仍把论文作者仓库配置为 `origin`。建议保留它作为 `upstream`，再把自己的
备份仓库设为 `origin`：

```bash
git remote -v
git remote rename origin upstream
git remote add origin git@github.com:<YOUR_ACCOUNT>/<YOUR_REPOSITORY>.git
```

若 `origin` 已经是自己的仓库，则跳过重命名，只核验 URL：

```bash
git remote set-url origin git@github.com:<YOUR_ACCOUNT>/<YOUR_REPOSITORY>.git
```

审核并提交：

```bash
git add .
git diff --cached --stat
git diff --cached --check
git status --short
git commit -m "Add SpreadsheetBench selective-distillation experiments"
git push -u origin main
```

本项目未自动执行最后的 `commit` 和 `push`：它们会改变 Git 历史和外部仓库，应在确认
目标仓库地址与可公开内容后由维护者执行。

## 4. 大型产物如何备份

`outputs/` 当前包含可复核实验结果，但不适合直接塞进 Git 历史。建议选择以下一种：

1. checkpoint 放 Git LFS，输出 JSON/日志压缩后放 GitHub Release；
2. 全量输出放对象存储或机构持久盘，Git 中只提交 manifest 与 SHA-256；
3. 只公开论文表格所依赖的 `summary.json`，把逐题生成和模型权重作为单独制品。

生成可审计清单：

```bash
find outputs -type f -print0 | sort -z | xargs -0 sha256sum \
  > outputs_manifest.sha256
du -sh outputs rollouts data model_cache 2>/dev/null
```

`outputs_manifest.sha256` 可能暴露本地文件名，提交前应人工检查。恢复大型产物后可用：

```bash
sha256sum -c outputs_manifest.sha256
```

## 5. 日常分支建议

- `main`：始终保持可运行、文档与结果口径一致；
- `experiment/<name>`：每个新压缩机制独立分支；
- 实验完成后先更新 `docs/EXPERIMENTS.md`，再合并代码；
- 对正式 test280 只评测冻结 checkpoint，并在报告中记录 checkpoint SHA-256；
- 不根据 test280 反复选择方法或超参数。
