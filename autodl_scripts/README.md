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

带常用参数：

```bash
SCRIPT_NAME=train_fastvit_s24.py \
DATA_ROOT=/root/autodl-tmp/data \
RUN_NAME=fastvit_s24_baseline \
BATCH_SIZE=32 \
HEAD_EPOCHS=80 \
FINETUNE_EPOCHS=20 \
HEAD_LR=5e-4 \
FINETUNE_LR=1e-6 \
NUM_WORKERS=4 \
SEED=2023 \
bash train_one.sh
```

只测试命令拼接、不真正开始训练：

```bash
DRY_RUN=1 SCRIPT_NAME=train_fastvit_s24.py bash train_one.sh
```

## 3. 批量训练指定模型

不带参数时，顺序跑所有单模型：

```bash
bash train_all_single_models.sh
```

分工训练时，在命令行后面直接写要跑的模型名：

```bash
bash train_all_single_models.sh convnext11 fastvit_s24 repvit_m2
```

也支持完整脚本名：

```bash
bash train_all_single_models.sh train_convnext11.py train_fastvit_s24.py
```

可以统一设置训练参数：

```bash
BATCH_SIZE=24 HEAD_EPOCHS=30 FINETUNE_EPOCHS=10 \
bash train_all_single_models.sh convnext11 mambaout_kobe mobilenetv4_hyper
```

批量脚本也支持 dry run：

```bash
DRY_RUN=1 bash train_all_single_models.sh convnext11 fastvit_s24 repvit_m2
```

可用模型名：

```text
convnext11
efficientnet
vit11
resnet18
mambaout_kobe
mambaout_small_rw
mobilenetv4
mobilenetv4_hyper
fastvit_s24
fastvit_sa36
fasternet
repvit
repvit_m2
```

## 4. 四模型集成 + XGBoost

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
BATCH_SIZE=28 \
HEAD_EPOCHS=80 \
FINETUNE_EPOCHS=20 \
CUDA_DEVICE=0 \
EARLY_STOP=0 \
bash train_ensemble.sh
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
- 集成训练还会保存 `ensemble_meta_xgboost.pkl`

默认输出示例：

```text
/root/autodl-tmp/raicom_runs/train_fastvit_s24_20260614_120000/
  logs/
    command.txt
    train.log
  fastvit_s24.pth
  fastvit_s24.png
```
