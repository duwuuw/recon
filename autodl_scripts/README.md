# RAICOM AutoDL 脚本

上传到 AutoDL 后，在项目根目录使用本文件夹中的脚本。**默认使用 GPU**；仅调试时设 `CPU=1`。

建议目录结构：

```text
/root/raicom/          # git clone 或上传代码
  data/raw/dataset/    # 可选：仓库内数据集（cloudy/rainy/snowy/sunny）
  scripts/
  src/
  autodl_scripts/
```

数据路径优先级：

1. 环境变量 `DATA_ROOT` 或 `RAICOM_DATA_ROOT`
2. 仓库内 `data/raw/dataset/`（若存在）
3. 默认 `/root/autodl-tmp/data/`

训练输出默认：`/root/autodl-tmp/raicom_runs/`

权重汇总（本地）：`checkpoints/`（可用 `CHECKPOINT_DIR` 覆盖）

## 1. 安装环境（GPU PyTorch）

```bash
cd /root/raicom/autodl_scripts
bash install_env.sh
source autodl_env.sh
```

默认行为：

- conda 环境 `raicom`，Python 3.10
- PyTorch **CUDA 12.4**（`cu124` index）
- `pip install -e ".[train]"`
- 生成 `autodl_env.sh`（含 HF 镜像 + GPU 自检）

AutoDL 若已是 cu128 镜像，可指定：

```bash
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 bash install_env.sh
```

镜像已装好 PyTorch 时：

```bash
SKIP_TORCH=1 bash install_env.sh
```

GDN 需要额外依赖：

```bash
INSTALL_GDN=1 bash install_env.sh
```

安装结束会打印 `[autodl] GPU OK: cuda:0 (...)`；若失败请检查实例是否选了 GPU 卡型。

## 2. 单模型训练

```bash
source autodl_env.sh
bash train_one.sh
```

指定脚本：

```bash
SCRIPT_NAME=train_mobilenetv4.py bash train_one.sh
SCRIPT_NAME=train_fastvit_s24.py bash train_one.sh
SCRIPT_NAME=train_mambaout_kobe.py bash train_one.sh
```

timm preset：

```bash
MODEL_PRESET=convnextv2_tiny bash train_one.sh
```

常用参数：

```bash
SCRIPT_NAME=train_fastvit_s24.py \
DATA_ROOT=/root/autodl-tmp/data \
RUN_NAME=fastvit_s24_baseline \
BATCH_SIZE=32 \
HEAD_EPOCHS=84 \
FINETUNE_EPOCHS=16 \
bash train_one.sh
```

仅拼接命令、不训练：

```bash
DRY_RUN=1 SCRIPT_NAME=train_fastvit_s24.py bash train_one.sh
```

强制 CPU（不推荐）：

```bash
CPU=1 bash train_one.sh
```

## 3. 批量训练 15 个核心单模型

**默认顺序**：MobileNet → FastViT → MambaOut → 其余 → GDN

```bash
source autodl_env.sh
bash train_all_single_models.sh
```

指定子集：

```bash
bash train_all_single_models.sh mobilenetv4 fastvit_s24 mambaout_kobe convnext11
```

跑全部 timm preset（旧行为）：

```bash
ALL_PRESETS=1 bash train_all_single_models.sh
```

统一超参：

```bash
BATCH_SIZE=24 HEAD_EPOCHS=84 FINETUNE_EPOCHS=16 \
bash train_all_single_models.sh mobilenetv4 fastvit_s24
```

批量结束后自动运行 `scripts/summarize_f1.py` 汇总 test macro-F1。

## 4. 集成训练

```bash
source autodl_env.sh
bash train_ensemble.sh
```

常用参数：

```bash
ENSEMBLE=balanced \
CUDA_DEVICE=0 \
HEAD_EPOCHS=84 \
FINETUNE_EPOCHS=16 \
bash train_ensemble.sh
```

## 5. GDN

```bash
INSTALL_GDN=1 bash install_env.sh
source autodl_env.sh
bash train_gdn.sh
```

## 6. tmux 后台

```bash
TARGET=train_all_single_models.sh SESSION_NAME=all_models bash launch_tmux.sh
tmux attach -t all_models
```

## 7. 日志与权重

每次运行在 `OUTPUT_DIR` 下保存：

- `logs/command.txt` — 实际命令
- `logs/train.log` — 训练日志
- `*.pth` — best val acc / best val F1 / last

单模型脚本还会写入项目 `checkpoints/<name>.pth` 等（与本地 Windows 流程一致）。
