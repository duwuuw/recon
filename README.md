# raicom

基于 PyTorch 与 [timm](https://github.com/huggingface/pytorch-image-models) 的天气图像分类实验仓库。支持单模型训练、两阶段微调、小参数强模型集成、weighted soft vote、TTA 与 XGBoost 元分类器。

AutoDL / Linux 训练脚本在 `autodl_scripts/` 目录。

---

## 目录结构

```
raicom/
├── scripts/              # 训练入口（推荐从这里启动）
│   ├── train_*.py        # 各 timm 骨干的单模型训练脚本
│   ├── train_gdn.py      # GDN 自定义模型
│   ├── train_ensemble_four_models_meta.py
│   ├── train_small_strong_ensemble.py
│   └── check_model_size.py
├── autodl_scripts/       # AutoDL / Linux 批量训练脚本
├── src/raicom/           # 共享训练库（pip install -e . 后可直接 import）
│   ├── classifier.py     # timm 单模型完整训练流程
│   ├── two_phase.py      # 两阶段微调（冻结骨干 → 全网络）
│   ├── timm_factory.py   # 统一 create_model（drop_rate=0.1）
│   ├── data.py           # 数据集划分与 DataLoader
│   ├── training.py       # 训练 / 验证循环
│   ├── checkpoints.py    # 最佳权重保存
│   ├── ensemble.py       # 多骨干集成训练
│   └── models/gdn.py     # GDN 模型定义
├── notebooks/            # 原始 Jupyter 实验笔记本（参考用）
├── checkpoints/          # 训练输出（权重与曲线，运行后自动生成）
├── data/raw/dataset/     # 默认数据集位置（ImageFolder）
├── requirements.txt
└── pyproject.toml
```

---

## 环境配置

**要求：Python 3.10+，NVIDIA GPU，conda 环境 `fm` 或 `xianyu`（已配 torch 2.11+cu128）**

```powershell
conda activate fm
# 或 conda activate xianyu

cd 路径\raicom
pip install -r requirements.txt
pip install -e .

# 确认 GPU
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

一键检查 GPU 环境：`. .\scripts\use_gpu_env.ps1`

训练与测试完整说明：[docs/TRAINING_AND_EVAL.md](docs/TRAINING_AND_EVAL.md)

GDN 可选依赖 `flash-linear-attention` 在 Windows 上较难安装，建议 Linux/WSL。

---

## 数据准备

仓库已包含 **天气识别竞赛 train 集**（4 类，4999 张），路径 `data/raw/dataset/`：

| 类别 | 文件夹 | 张数 |
|------|--------|------|
| 多云 | cloudy | 2184 |
| 雨天 | rainy | 446 |
| 雪天 | snowy | 403 |
| 晴天 | sunny | 1966 |

所有训练脚本分类头固定为 **4 类**（`src/raicom/constants.py` 中 `NUM_CLASSES = 4`）。

**从 zip 重新导入**（会覆盖 `data/raw/dataset/`，详见 [data/README.md](data/README.md)）：

```bash
python scripts/import_weather_zip.py --zip data/raw/天气识别.zip
```

**训练全部模型、测试、汇总 F1**：见 [docs/TRAINING_AND_EVAL.md](docs/TRAINING_AND_EVAL.md)。

数据集需为 **ImageFolder** 格式：根目录下每个子文件夹对应一个类别。

```
dataset/
├── 类别A/
│   ├── img001.jpg
│   └── ...
├── 类别B/
│   └── ...
└── ...
```

指定数据路径（二选一）：

**方式 A**：放到默认路径

```
data/raw/dataset/
```

**方式 B**：设置环境变量

Windows PowerShell：

```powershell
$env:RAICOM_DATA_ROOT = "D:\path\to\your\dataset"
```

Windows CMD：

```cmd
set RAICOM_DATA_ROOT=D:\path\to\your\dataset
```

Linux / macOS：

```bash
export RAICOM_DATA_ROOT=/path/to/your/dataset
```

---

## 启动训练

在仓库根目录 `d:\dl\raicom` 下执行。

### 单模型训练

```powershell
python scripts/train_fastvit_s24.py
python scripts/train_convnext11.py
python scripts/train_mambaout_kobe.py
python scripts/train_resnet18.py
# ... scripts/ 下其他 train_*.py 同理
```

**当前可用脚本：**


| 脚本                           | 用途 |
| ---------------------------- | ---- |
| `train_timm_preset.py <preset>` | 30M 参数以内 timm SOTA 小/中模型预设库，覆盖 ConvNeXtV2、MobileNetV4 纯 CNN / Hybrid、FastViT、EfficientViT、MaxViT、CoAtNet、EfficientFormerV2、EdgeNeXt、FasterNet、RepViT、MambaOut、TinyViT、EVA02、DINOv2、DeiT3、MobileViTv2、MobileOne、CAFormer、ConvFormer、SwiftFormer 等 |
| `train_convnext11.py`        | 历史独立脚本：convnextv2_nano |
| `train_efficientnet.py`      | 历史独立脚本：tf_efficientnetv2_s |
| `train_dinov2_small.py`      | 历史独立脚本：vit_small_patch14_dinov2 |
| `train_vit11.py`             | 历史独立脚本：vit_base_patch16_rope_224 |
| `train_resnet18.py`          | 历史独立脚本：ecaresnet50d |
| 其他 `train_*.py`            | 旧实验入口仍保留，推荐新模型优先用 preset |

查看全部 preset：

```powershell
python scripts/train_timm_preset.py --list
```

训练示例：

```powershell
python scripts/train_timm_preset.py convnextv2_tiny
python scripts/train_timm_preset.py mobilenetv4_conv_medium
python scripts/train_timm_preset.py mobilenetv4_hybrid_medium
python scripts/train_timm_preset.py fastvit_sa24
python scripts/train_timm_preset.py dinov2_small_reg4
```


### 常用命令行参数(懒得调超参数的话就按照单模型训练那里的命令来就行了)

```powershell
python scripts/train_fastvit_s24.py --help

python scripts/train_fastvit_s24.py --data-root D:\path\to\dataset
python scripts/train_fastvit_s24.py --output-dir checkpoints
python scripts/train_fastvit_s24.py --batch-size 16
python scripts/train_fastvit_s24.py --image-size 224
python scripts/train_fastvit_s24.py --head-epochs 84 --finetune-epochs 16
python scripts/train_fastvit_s24.py --head-lr 5e-4 --finetune-lr 2e-7
python scripts/train_fastvit_s24.py --early-stop 8 --early-stop-min-delta 1e-4
```

### 小参数强模型集成 + XGBoost

默认 `balanced` 方案会训练 5 个参数量较小但互补性强的 timm 骨干：

| 方案 | 骨干 |
| ---- | ---- |
| `balanced` | `convnextv2_nano`、`mobilenetv4_hybrid_medium`、`fastvit_sa24`、`mambaout_kobe`、`tiny_vit_11m_224` |
| `lite` | 更快的小模型组合，适合快速试榜或显存较小的机器 |
| `strong` | 在 `balanced` 基础上额外加入 `caformer_s18` |
| `legacy` | 保留旧 notebook 风格四模型 baseline |

推荐入口：

```powershell
python scripts/train_small_strong_ensemble.py --ensemble balanced
```

常用 Kaggle 风格开关：

```powershell
# 启用水平翻转 TTA
python scripts/train_small_strong_ensemble.py --ensemble balanced --tta-hflip

# 复用已有 ensemble_ckpt_*.pth，直接重新收集概率、搜索权重和训练 meta learner
python scripts/train_small_strong_ensemble.py --ensemble balanced --reuse-checkpoints

# 手动指定 timm preset 或完整 timm 模型名
python scripts/train_small_strong_ensemble.py --models convnextv2_pico fastvit_s12 repvit_m1_1 tiny_vit_5m_224

# 离线调试入口，不下载预训练权重、不使用 GPU
python scripts/train_small_strong_ensemble.py --models convnextv2_atto --head-epochs 0 --finetune-epochs 0 --no-pretrained --cpu
```

### GDN 自定义模型
这个模型依赖flash-linear-attention，如果你没有安装这个运行会很慢，甚至报错
```powershell
python scripts/train_gdn.py
```

### 查看模型参数量（不训练）

```powershell
python scripts/check_model_size.py --model convnextv2_nano
```

### AutoDL / Linux 脚本

```bash
cd /root/raicom/autodl_scripts
bash install_env.sh
bash train_ensemble.sh
```

常用覆盖：

```bash
ENSEMBLE=balanced TTA_HFLIP=1 bash train_ensemble.sh
ENSEMBLE=lite RUN_NAME=ensemble_lite bash train_ensemble.sh
REUSE_CHECKPOINTS=1 bash train_ensemble.sh
MODELS="convnextv2_pico fastvit_s12 repvit_m1_1 tiny_vit_5m_224" bash train_ensemble.sh
```

---

## 训练策略

所有训练脚本默认采用 **两阶段微调**（配置在 `src/raicom/two_phase.py`）：


| 阶段   | Epoch  | 可训练部分 | 学习率  | 调度器                          |
| ---- | ------ | ----- | ---- | ---------------------------- |
| 阶段 1 | 1–84   | 仅分类头  | 5e-4 | CosineAnnealing，最低 lr = 5e-7 |
| 阶段 2 | 85–100 | 全网络   | 2e-7 | CosineAnnealing，最低 lr = 2e-10 |


其他默认设置：

- **MixUp** alpha = 0.205（GDN 不使用 MixUp）
- **drop_rate** = 0.1（通过 `src/raicom/timm_factory.py` 统一传入 timm）
- **优化器** AdamW，weight_decay = 2.5e-4（各脚本可单独覆盖）
- **早停** 阶段 2 默认启用，patience = 8，min_delta = 1e-4；`--early-stop 0` 可关闭
- **数据划分** 训练 / 验证 / 测试 = 8 : 1 : 1（分层采样，seed=42）
- **数据增强** 训练集 RandomHorizontalFlip + RandomRotation(10)；验证 / 测试集无随机增强
- **集成推理** 支持普通 soft vote、验证集随机搜索 weighted soft vote、hard vote、XGBoost meta learner；`--tta-hflip` 可启用水平翻转 TTA

---

## 输出文件

默认写入 `checkpoints/` 目录（可用 `--output-dir` 修改）：


| 内容      | 示例路径                                    |
| ------- | --------------------------------------- |
| 最佳权重    | `checkpoints/fastvit_s24.pth`           |
| 训练曲线    | `checkpoints/fastvit_s24.png`           |
| 集成各骨干权重 | `checkpoints/ensemble_ckpt_*.pth`       |
| 集成元分类器  | `checkpoints/ensemble_meta_xgboost.pkl` |
| 集成报告    | `checkpoints/ensemble_report.json`      |


验证集准确率每轮提升时自动覆盖保存最佳权重；训练结束后加载最佳权重在测试集上评估。

---

## 新增模型

1. timm SOTA 小/中模型优先加到 `src/raicom/timm_presets.py`，然后用 `scripts/train_timm_preset.py <preset>` 训练；AutoDL 批量脚本也会使用同一批 preset 名称。
2. 若只想快速复制独立脚本，也可以复制任意 `scripts/train_*.py`，改三处即可：(timm库里面模型要自己去查，假如名字不对会报错)

```python
ClassifierTrainConfig(
    timm_model="your_timm_model_name",   # timm 模型名
    checkpoint_name="your_model.pth",      # 权重文件名
    curves_name="your_model.png",          # 曲线图文件名
    batch_size=32,                         # 按需调整
    image_size=224,                        # MaxViT 等固定 256 输入模型要改成 256
)
```

3. 两阶段训练、drop_rate、最佳权重保存等逻辑自动生效，无需改 `src/raicom/`。
4. 若要修改全局默认（84+16 epoch、学习率、dropout 等），改 `src/raicom/two_phase.py` 或 `src/raicom/timm_factory.py`。
5. 加入集成：优先直接用 `python scripts/train_small_strong_ensemble.py --models preset_a preset_b ...`；若要长期保留为内置方案，编辑 `scripts/train_ensemble_four_models_meta.py` 中的 `ENSEMBLE_PRESETS`。

---

## 常见问题

**报错「数据目录不存在」**

检查 `RAICOM_DATA_ROOT` 或 `--data-root` 是否指向 ImageFolder 根目录（含各类别子文件夹）。

**GPU 未使用**

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

应输出 `True`。若为 `False`，需重装对应 CUDA 版本的 PyTorch。

**Windows 上 DataLoader 卡住**

脚本默认 `num_workers=0`，一般无需修改。集成脚本在 Windows 上同样使用 0。

**timm 模型下载慢**

集成脚本已设置 Hugging Face 镜像；也可手动配置 `HF_ENDPOINT=https://hf-mirror.com`。

**本机只想调试流程，不想下载权重**

```powershell
python scripts/train_small_strong_ensemble.py --models convnextv2_atto --head-epochs 0 --finetune-epochs 0 --no-pretrained --cpu
```

---

## 协作与许可证

协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。许可证为 MIT，见 [LICENSE](LICENSE)。



--------------------------------------------------------------------------------------
## 特别鸣谢：

cursor：是他完成了整个仓库的重构，以及大部分脚本的编写

gemini: 是他帮我完成timm库模型的查阅和计算参数量与精准度

codex/chatgpt:作为主力帮我完成我的其他项目，让我空出时间打理睿抗

豆包：告诉我timm库的存在
