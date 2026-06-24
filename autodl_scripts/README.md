# RAICOM AutoDL 脚本

这个文件夹是给 `D:\dl\raicom` 项目上传到 AutoDL 后使用的训练脚本。建议放置位置：

```text
/root/raicom/
  scripts/
  src/
  requirements.txt
  pyproject.toml
  autodl_scripts/
```

数据集使用 ImageFolder 格式，默认路径：

```text
/root/autodl-tmp/data/
  class_a/
  class_b/
  ...
```

训练输出默认写到：

```text
/root/autodl-tmp/raicom_runs/
```

## 1. 安装环境

```bash
cd /root/raicom/autodl_scripts
bash install_env.sh
```

默认行为：

- 创建 conda 环境 `raicom`
- Python `3.10`
- 安装 CUDA 12.1 版 PyTorch
- 安装 `requirements.txt`
- 执行 `pip install -e ".[train]"`
- 生成 `autodl_env.sh`

常用覆盖：

```bash
ENV_NAME=raicom PYTHON_VERSION=3.10 bash install_env.sh
TORCH_INDEX_URL=https://download.pytorch.org/whl/cu118 bash install_env.sh
SKIP_TORCH=1 bash install_env.sh
INSTALL_GDN=1 bash install_env.sh
```

如果 AutoDL 镜像已经装好合适的 PyTorch，用 `SKIP_TORCH=1` 可以省时间。

## 2. 单模型训练

默认训练 `train_convnext11.py`：

```bash
bash train_one.sh
```

指定模型脚本：

```bash
SCRIPT_NAME=train_fastvit_s24.py bash train_one.sh
SCRIPT_NAME=train_mambaout_kobe.py bash train_one.sh
SCRIPT_NAME=train_mobilenetv4_hyper.py bash train_one.sh
```

使用 30M 以内 timm preset：

```bash
MODEL_PRESET=convnextv2_tiny bash train_one.sh
MODEL_PRESET=mobilenetv4_conv_medium bash train_one.sh
MODEL_PRESET=fastvit_sa24 bash train_one.sh
```

查看全部 preset：

```bash
python ../scripts/train_timm_preset.py --list
```

带常用参数：

```bash
SCRIPT_NAME=train_fastvit_s24.py \
DATA_ROOT=/root/autodl-tmp/data \
RUN_NAME=fastvit_s24_baseline \
BATCH_SIZE=32 \
IMAGE_SIZE=224 \
HEAD_EPOCHS=84 \
FINETUNE_EPOCHS=16 \
HEAD_LR=5e-4 \
FINETUNE_LR=2e-7 \
EARLY_STOP=8 \
NUM_WORKERS=4 \
SEED=2023 \
bash train_one.sh
```

只测试命令拼接、不真正开始训练：

```bash
DRY_RUN=1 SCRIPT_NAME=train_fastvit_s24.py bash train_one.sh
```

## 3. 批量训练指定模型

不带参数时，顺序跑所有 30M 以内 timm preset：

```bash
bash train_all_single_models.sh
```

分工训练时，在命令行后面直接写要跑的模型名：

```bash
bash train_all_single_models.sh convnextv2_tiny mobilenetv4_conv_medium fastvit_sa24
```

也支持完整脚本名：

```bash
bash train_all_single_models.sh train_convnext11.py train_fastvit_s24.py
```

可以统一设置训练参数：

```bash
BATCH_SIZE=24 HEAD_EPOCHS=30 FINETUNE_EPOCHS=10 \
bash train_all_single_models.sh convnextv2_tiny mambaout_tiny repvit_m2_3
```

批量脚本也支持 dry run：

```bash
DRY_RUN=1 bash train_all_single_models.sh convnextv2_tiny fastvit_sa24 repvit_m2_3
```

可用模型名：

```bash
bash train_all_single_models.sh --help
python ../scripts/train_timm_preset.py --list
```

## 4. 小参数强模型集成 + XGBoost

```bash
bash train_ensemble.sh
```

只测试集成训练命令：

```bash
DRY_RUN=1 bash train_ensemble.sh
```

常用参数：

```bash
DATA_ROOT=/root/autodl-tmp/data \
RUN_NAME=ensemble_baseline \
ENSEMBLE=balanced \
HEAD_EPOCHS=84 \
FINETUNE_EPOCHS=16 \
HEAD_LR=5e-4 \
FINETUNE_LR=2e-7 \
CUDA_DEVICE=0 \
EARLY_STOP=8 \
bash train_ensemble.sh
```

内置集成方案：

- `ENSEMBLE=balanced`：默认，`convnextv2_nano + mobilenetv4_hybrid_medium + fastvit_sa24 + mambaout_kobe + tiny_vit_11m_224`
- `ENSEMBLE=lite`：更快，适合显存小或快速试榜
- `ENSEMBLE=strong`：额外加入 `caformer_s18`
- `ENSEMBLE=legacy`：旧 notebook 风格四模型 baseline

Kaggle 风格常用开关：

```bash
TTA_HFLIP=1 bash train_ensemble.sh
REUSE_CHECKPOINTS=1 bash train_ensemble.sh
MODELS="convnextv2_pico fastvit_s12 repvit_m1_1 tiny_vit_5m_224" bash train_ensemble.sh
NO_PRETRAINED=1 CPU=1 DRY_RUN=1 bash train_ensemble.sh
```

## 5. GDN

GDN 依赖 `flash-linear-attention`，建议先单独安装：

```bash
INSTALL_GDN=1 bash install_env.sh
bash train_gdn.sh
```

如果安装失败，先跳过 GDN，优先跑 timm 单模型。

## 6. tmux 后台训练

单模型后台跑：

```bash
SCRIPT_NAME=train_convnext11.py TARGET=train_one.sh SESSION_NAME=convnext bash launch_tmux.sh
tmux attach -t convnext
```

指定多个模型后台跑：

```bash
TARGET=train_all_single_models.sh \
TARGET_ARGS="convnext11 fastvit_s24 repvit_m2" \
SESSION_NAME=part_a \
bash launch_tmux.sh

tmux attach -t part_a
```

全模型后台跑：

```bash
TARGET=train_all_single_models.sh SESSION_NAME=all_models bash launch_tmux.sh
tmux attach -t all_models
```

集成后台跑：

```bash
TARGET=train_ensemble.sh SESSION_NAME=ensemble bash launch_tmux.sh
tmux attach -t ensemble
```

## 7. 日志和输出

每次训练目录中会保存：

- `logs/command.txt`: 本次真实执行命令
- `logs/train.log`: 完整训练日志
- `.pth`: 最佳权重
- `.png`: 训练曲线
- 集成训练还会保存 `ensemble_meta_xgboost.pkl` 与 `ensemble_report.json`

默认输出示例：

```text
/root/autodl-tmp/raicom_runs/train_fastvit_s24_20260614_120000/
  logs/
    command.txt
    train.log
  fastvit_s24.pth
  fastvit_s24.png
```
