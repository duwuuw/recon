# 训练与测试指南

本文档说明：**如何训练仓库里每一个模型**、**权重保存在哪**、**如何单独测试并查看 macro-F1**。  
所有路径均相对于**仓库根目录**（含 `scripts/`、`data/` 的那一层）。

数据准备见 [data/README.md](../data/README.md)。

---

## 0. 一次性准备（必须用 GPU 环境）

RTX 5060 等显卡需要 **PyTorch 2.x + CUDA 12.8**。**不要用 base 自带 Python**（会 silently 回退 CPU 或报 CUDA 错）。

在仓库根目录打开 **Anaconda Prompt / PowerShell**：

```powershell
cd 路径\raicom

# 1. 激活已配好的 GPU 环境（二选一）
conda activate fm
# conda activate xianyu

# 或一键脚本（默认 fm，换 xianyu 见下）
. .\scripts\use_gpu_env.ps1
# $env:RAICOM_CONDA_ENV = "xianyu"; . .\scripts\use_gpu_env.ps1

# 2. 安装本仓库（每个环境只需一次）
pip install -r requirements.txt
pip install -e .

# 3. 确认 GPU 可用（必须看到 GPU 名称）
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO GPU')"

# 4. 确认数据（4 类，约 4999 张）
Get-ChildItem data\raw\dataset -Directory | ForEach-Object {
    "$($_.Name): $((Get-ChildItem $_.FullName -File).Count)"
}

# 5. 国内下载预训练权重（推荐）
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

| Conda 环境 | 路径 | 说明 |
|------------|------|------|
| `fm` | `D:\conda_envs\fm` | **默认推荐**，torch 2.11+cu128 |
| `xianyu` | `D:\conda_envs\xianyu` | 同上，可互换 |

训练开始时应打印：

```
Device: cuda:0 (NVIDIA GeForce RTX 5060 Laptop GPU)
```

若看到 `Device: cpu` 或 CUDA 警告，说明**环境没激活对**，先 `conda activate fm` 再跑。

---

## 1. 数据怎么划分

脚本自动读取 `data/raw/dataset/`，按**每个类别内**随机划分（seed=42）：

| 子集 | 比例 | 约张数 |
|------|------|--------|
| 训练 train | 80% | 3999 |
| 验证 val | 10% | 500 |
| 测试 test | 10% | 500 |

- **训练**时看 train loss/acc，用 **val** 选最佳权重  
- **测试**只在训练结束后跑一次（或你用 eval 脚本单独跑 test）  
- **macro-F1**：4 个类别 F1 的算术平均，训练日志里会打印 `Val F1` 和最终 `macro-F1`

---

## 2. 训练通用命令格式

每个 timm 单模型脚本用法相同（**先 `conda activate fm`**）：

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"

python scripts/train_<模型名>.py --data-root data/raw/dataset
```

默认使用 **GPU**，无需额外参数。脚本会自动选 `cuda:0`。

常用可选参数：

| 参数 | 说明 |
|------|------|
| `--data-root data/raw/dataset` | 数据路径（默认已是这个，可省略） |
| `--batch-size 16` | 显存不够改小（8GB 卡可试 16 或 8） |
| `--cpu` | **仅调试**；正常训练不要用 |
| `--output-dir checkpoints` | 权重输出目录（默认 `checkpoints/`） |

### 2.1 训练时会自动保存哪些权重

默认写入 `checkpoints/`，**无需手动保存**：

| 文件示例 | 何时写入 |
|----------|----------|
| `convnext.pth` | 验证集 **acc** 创新高 |
| `convnext_best_f1.pth` | 验证集 **macro-F1** 创新高 |
| `convnext_last.pth` | 全部 epoch 跑完后的最后一轮 |
| `convnext.png` | 训练曲线图 |

训练过程中看到类似输出即表示已保存：

```
-> 保存最佳权重 checkpoints\convnext.pth (val_acc=0.xxxx)
-> 保存最佳 F1 权重 checkpoints\convnext_best_f1.pth (val macro-F1=0.xxxx)
```

### 2.2 训练结束时会自动打印 F1

每个单模型脚本跑完后会输出：

```
最终测试集 Loss: x.xxxx | 准确率: x.xxxx | macro-F1: x.xxxx
```

后面还有**按类别的 classification report**（precision/recall/f1）。

### 2.3 默认训练计划

- **阶段 1**：84 epoch，只训练分类头，lr=5e-4  
- **阶段 2**：16 epoch，全网络微调，lr=2e-7，带早停  

合计约 100 epoch。在 **GPU（fm/xianyu）** 上单模型通常需 **数十分钟～数小时**；CPU 会慢一个数量级以上，请勿用 CPU 跑全量。

---

## 3. 逐个训练：15 个单模型 + GDN

**必须每个都跑一遍**（下面表格每一行一条命令）。  
建议在仓库根目录**按顺序**执行，或直接用 [第 5 节](#5-一键训练全部单模型) 批量脚本。

| # | 脚本 | 输出权重 | timm 骨干 |
|---|------|----------|-----------|
| 1 | `scripts/train_mobilenetv4.py` | `checkpoints/mobilenetv4_hybrid_medium.pth` | mobilenetv4_hybrid_medium |
| 2 | `scripts/train_mobilenetv4_hyper.py` | `checkpoints/mobilenetv4_hybrid_large.pth` | mobilenetv4_hybrid_large |
| 3 | `scripts/train_fastvit_s24.py` | `checkpoints/fastvit_s24.pth` | fastvit_sa24 |
| 4 | `scripts/train_fastvit_sa36.py` | `checkpoints/fastvit_sa36.apple_dist_in1k.pth` | fastvit_sa36 |
| 5 | `scripts/train_mambaout_kobe.py` | `checkpoints/mambaout_kobe.pth` | mambaout_kobe |
| 6 | `scripts/train_mambaout_small_rw.py` | `checkpoints/mambaout_small_rw.pth` | mambaout_small_rw |
| 7 | `scripts/train_convnext11.py` | `checkpoints/convnext.pth` | convnextv2_nano |
| 8 | `scripts/train_dinov2_small.py` | `checkpoints/dinov2_small.pth` | vit_small_patch14_dinov2 |
| 9 | `scripts/train_efficientnet.py` | `checkpoints/efficientnet.pth` | tf_efficientnetv2_s |
| 10 | `scripts/train_fasternet.py` | `checkpoints/fasternet_t2.pth` | fasternet_t2 |
| 11 | `scripts/train_repvit.py` | `checkpoints/repvit_m1_5.pth` | repvit_m1_5 |
| 12 | `scripts/train_repvit_m2.py` | `checkpoints/repvit_m2_3.pth` | repvit_m2_3 |
| 13 | `scripts/train_resnet18.py` | `checkpoints/resnet50.pth` | ecaresnet50d |
| 14 | `scripts/train_vit11.py` | `checkpoints/vit.pth` | vit_base_patch16_rope_224 |
| 15 | `scripts/train_gdn.py` | `checkpoints/gdn_best.pth` | 自定义 GDN（非 timm） |

### 3.1 复制粘贴：逐个训练（GPU）

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"
$data = "data/raw/dataset"

python scripts/train_mobilenetv4.py       --data-root $data
python scripts/train_mobilenetv4_hyper.py --data-root $data
python scripts/train_fastvit_s24.py       --data-root $data
python scripts/train_fastvit_sa36.py      --data-root $data
python scripts/train_mambaout_kobe.py     --data-root $data
python scripts/train_mambaout_small_rw.py --data-root $data
python scripts/train_convnext11.py        --data-root $data
python scripts/train_dinov2_small.py      --data-root $data
python scripts/train_efficientnet.py      --data-root $data
python scripts/train_fasternet.py         --data-root $data
python scripts/train_repvit.py           --data-root $data
python scripts/train_repvit_m2.py        --data-root $data
python scripts/train_resnet18.py         --data-root $data
python scripts/train_vit11.py            --data-root $data
python scripts/train_gdn.py              --data-root $data
```

> **GDN 说明**：依赖 `flash-linear-attention`，Windows 上可能装不上；失败可先在 Linux/WSL 训练，或暂时跳过，其余 14 个 timm 模型不受影响。

---

## 4. 集成模型训练（可选，在单模型之后）

单模型都训完后，再跑集成（内部会再训多个骨干 + 软投票 + XGB meta）：

```powershell
python scripts/train_ensemble_four_models_meta.py --data-root data/raw/dataset
```

等价入口：

```powershell
python scripts/train_small_strong_ensemble.py --data-root data/raw/dataset
```

输出（均在 `checkpoints/`）：

| 文件 | 说明 |
|------|------|
| `ensemble_ckpt_<骨干名>.pth` | 各骨干最佳权重 |
| `ensemble_report.json` | 验证/测试 acc、F1、各骨干单模型 F1 |
| `ensemble_meta_xgboost.pkl` | meta 分类器 |
| `ensemble_soft_vote_val_macro_f1_per_epoch.png` | 可选 F1 曲线 |

终端末尾会打印类似：

```
--- Test set ---
Soft vote          Acc=0.xxxx  Macro-F1=0.xxxx
Hard vote          Acc=0.xxxx  Macro-F1=0.xxxx
Weighted soft vote Acc=0.xxxx  Macro-F1=0.xxxx
Meta (XGB)         Acc=0.xxxx  Macro-F1=0.xxxx
```

---

## 5. 一键训练全部单模型（GPU）

```powershell
# 默认 conda 环境 fm
.\scripts\run_all_single_models.ps1

# 使用 xianyu 环境
$env:RAICOM_CONDA_ENV = "xianyu"
.\scripts\run_all_single_models.ps1
```

脚本会先检测 CUDA，再按顺序训练 15 个模型，最后 `summarize_f1.py`。

---

## 6. 单独测试已有权重（计算 F1）

训练已完成、只想对某个 `.pth` 再算一遍 val/test F1 时用。

### 6.1 评估单个 timm 模型

```powershell
conda activate fm

python scripts/eval_checkpoint.py `
  --checkpoint checkpoints/convnext.pth `
  --timm-model convnextv2_nano `
  --data-root data/raw/dataset
```

`--timm-model` 必须与训练脚本里的一致（见第 3 节表格）。

只看测试集：

```powershell
conda activate fm

python scripts/eval_checkpoint.py `
  --checkpoint checkpoints/convnext.pth `
  --timm-model convnextv2_nano `
  --data-root data/raw/dataset `
  --split test
```

### 6.2 评估 GDN

```powershell
conda activate fm

python scripts/eval_checkpoint.py `
  --checkpoint checkpoints/gdn_best.pth `
  --gdn `
  --data-root data/raw/dataset
```

### 6.3 评估 best-F1 权重

若更关心 F1 最优那次保存：

```powershell
conda activate fm

python scripts/eval_checkpoint.py `
  --checkpoint checkpoints/convnext_best_f1.pth `
  --timm-model convnextv2_nano `
  --data-root data/raw/dataset
```

---

## 7. 汇总全部模型的 Test F1

15 个单模型都训完后，一张表对比谁最好：

```powershell
conda activate fm
python scripts/summarize_f1.py --data-root data/raw/dataset
```

输出示例：

```
Model                  Test Acc    Test F1  Checkpoint
------------------------------------------------------------------------
convnext11               0.xxxx     0.xxxx  convnext.pth
...
gdn                           —          —  (未训练: gdn_best.pth)
```

同时生成 JSON：`checkpoints/f1_summary.json`（方便写报告）。

未训练的模型会显示 `(未训练: xxx.pth)`，训完再跑一次即可更新。

---

## 8. 推荐完整流程（ checklist ）

按顺序打勾：

- [ ] **GPU 环境**：`conda activate fm`（或 xianyu），CUDA 检查通过
- [ ] **预下载权重**：`$env:HF_ENDPOINT="https://hf-mirror.com"` 后训练第一个模型
- [ ] **训练 15 个单模型**：第 3 节逐条或 `run_all_single_models.ps1`
- [ ] **汇总 F1**：`python scripts/summarize_f1.py`
- [ ] **（可选）训练集成**：`train_ensemble_four_models_meta.py`
- [ ] **（可选）单独复测**：`eval_checkpoint.py`

---

## 9. 常见问题

### Q1：怎么知道某个模型训完了？

对应 `checkpoints/` 下出现 `.pth` 文件，且终端有 `最终测试集 ... macro-F1`。

### Q2：训练中断了怎么办？

重新跑同一条训练命令即可；会从预训练权重重新开始（暂不支持断点续训）。  
已保存的 `convnext.pth` 等不会被覆盖，除非 val acc 再次超过历史最佳。

### Q3：显存不够

```powershell
conda activate fm
python scripts/train_convnext11.py --batch-size 8
```

### Q4：终端显示 Device: cpu

未激活 GPU 环境。请：

```powershell
conda activate fm   # 或 xianyu
python -c "import torch; print(torch.cuda.is_available())"  # 必须 True
```

### Q5：F1 和 acc 差很多？

类别不平衡时（如 sunny 1966 vs rainy 446）很常见；**以 macro-F1 为主**更公平，可看 `*_best_f1.pth`。

### Q6：集成要单独测吗？

集成脚本训练结束已打印 test F1；细节见 `checkpoints/ensemble_report.json`。

---

## 10. 命令速查

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 训练一个
python scripts/train_convnext11.py --data-root data/raw/dataset

# 训练全部单模型（GPU）
.\scripts\run_all_single_models.ps1

# 测一个权重
python scripts/eval_checkpoint.py --checkpoint checkpoints/convnext.pth --timm-model convnextv2_nano --data-root data/raw/dataset

# 汇总全部 Test F1
python scripts/summarize_f1.py --data-root data/raw/dataset

# 训练集成
python scripts/train_ensemble_four_models_meta.py --data-root data/raw/dataset
```

---

## 11. 相关文件

| 路径 | 作用 |
|------|------|
| `scripts/train_*.py` | 各模型训练入口 |
| `scripts/eval_checkpoint.py` | 单独评估权重 + F1 |
| `scripts/summarize_f1.py` | 汇总全部模型 Test F1 |
| `scripts/run_all_single_models.ps1` | GPU 批量训练 15 个单模型 |
| `scripts/use_gpu_env.ps1` | 激活 fm/xianyu 并检查 GPU |
| `checkpoints/` | 权重与曲线（训练后生成，不提交 git） |
| `src/raicom/classifier.py` | 训练主流程与自动存盘 |
| `src/raicom/data.py` | 80/10/10 划分 |
| `data/README.md` | 数据集导入与路径说明 |
