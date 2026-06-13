# raicom

基于 PyTorch 与 [timm](https://github.com/huggingface/pytorch-image-models) 的天气图像分类实验仓库。支持单模型训练、两阶段微调、四模型集成 + XGBoost 元分类器。

ssh的脚本稍后奉上

---

## 目录结构

```
raicom/
├── scripts/              # 训练入口（推荐从这里启动）
│   ├── train_*.py        # 各 timm 骨干的单模型训练脚本
│   ├── train_gdn.py      # GDN 自定义模型
│   ├── train_ensemble_four_models_meta.py
│   └── check_model_size.py
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

**要求：Python 3.10+**

1. 安装 PyTorch（按 [pytorch.org](https://pytorch.org/) 选择 CUDA 版本）
2. 在仓库根目录安装依赖：

```bash
pip install -e ".[train]"
```

或分步：

```bash
pip install -r requirements.txt
pip install -e .
```

1. GDN 模型额外依赖（可选）：(我也不知道这个是否适合做分类任务)

此外，flash-linear-attention这个包在windows非常难用，得在wsl或ubuntu下使用，不建议安装

```bash
pip install flash-linear-attention
```

---

## 数据准备

仓库已包含 **4 类 × 50 张** 样本（约 13MB），路径 `data/raw/dataset/`：
目前官方没有公布数据集所以我就选了50张先上传来测试，便于各位先快速跑通代码
| 类别 | 张数 |
|------|------|
| rain | 50 |
| snow | 50 |
| fogsmog | 50 |
| hail | 50 |

所有训练脚本分类头固定为 **4 类**（`src/raicom/constants.py` 中 `NUM_CLASSES = 4`）。



```bash
python scripts/build_sample_dataset.py
```

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


| 脚本                           | timm 模型                      |
| ---------------------------- | ---------------------------- |
| `train_convnext11.py`        | convnextv2_nano              |
| `train_efficientnet.py`      | tf_efficientnetv2_s          |
| `train_vit11.py`             | vit_base_patch16_rope_224    |
| `train_resnet18.py`          | ecaresnet50d                 |
| `train_mambaout_kobe.py`     | mambaout_kobe                |
| `train_mambaout_small_rw.py` | mambaout_small_rw            |
| `train_mobilenetv4.py`       | mobilenetv4_hybrid_medium    |
| `train_mobilenetv4_hyper.py` | mobilenetv4_hybrid_large     |
| `train_fastvit_s24.py`       | fastvit_sa24.apple_dist_in1k |
| `train_fastvit_sa36.py`      | fastvit_sa36.apple_dist_in1k |
| `train_fasternet.py`         | fasternet_t2                 |
| `train_repvit.py`            | repvit_m1_5                  |
| `train_repvit_m2.py`         | repvit_m2_3                  |


### 常用命令行参数(懒得调超参数的话就按照单模型训练那里的命令来就行了)

```powershell
python scripts/train_fastvit_s24.py --help

python scripts/train_fastvit_s24.py --data-root D:\path\to\dataset
python scripts/train_fastvit_s24.py --output-dir checkpoints
python scripts/train_fastvit_s24.py --batch-size 16
python scripts/train_fastvit_s24.py --head-epochs 80 --finetune-epochs 20
python scripts/train_fastvit_s24.py --head-lr 5e-4 --finetune-lr 1e-6
```

### 四模型集成 + XGBoost(已废弃)

```powershell
python scripts/train_ensemble_four_models_meta.py
```

### GDN 自定义模型

```powershell
python scripts/train_gdn.py
```

### 查看模型参数量（不训练）

```powershell
python scripts/check_model_size.py --model convnextv2_nano
```

---

## 训练策略

所有 timm 单模型脚本默认采用 **两阶段微调**（配置在 `src/raicom/two_phase.py`）：


| 阶段   | Epoch  | 可训练部分 | 学习率  | 调度器                          |
| ---- | ------ | ----- | ---- | ---------------------------- |
| 阶段 1 | 1–80   | 仅分类头  | 5e-4 | CosineAnnealing，最低 lr = 5e-7 |
| 阶段 2 | 81–100 | 全网络   | 1e-6 | CosineAnnealing，最低 lr = 1e-9 |


其他默认设置：

- **MixUp** alpha = 0.205（GDN 不使用 MixUp）
- **drop_rate** = 0.1（通过 `src/raicom/timm_factory.py` 统一传入 timm）
- **优化器** AdamW，weight_decay = 2.5e-4（各脚本可单独覆盖）
- **数据划分** 训练 / 验证 / 测试 = 8 : 1 : 1（分层采样，seed=42）
- **数据增强** 训练集 RandomHorizontalFlip + RandomRotation(10)；验证 / 测试集无随机增强

---

## 输出文件

默认写入 `checkpoints/` 目录（可用 `--output-dir` 修改）：


| 内容      | 示例路径                                    |
| ------- | --------------------------------------- |
| 最佳权重    | `checkpoints/fastvit_s24.pth`           |
| 训练曲线    | `checkpoints/fastvit_s24.png`           |
| 集成各骨干权重 | `checkpoints/ensemble_ckpt_*.pth`       |
| 集成元分类器  | `checkpoints/ensemble_meta_xgboost.pkl` |


验证集准确率每轮提升时自动覆盖保存最佳权重；训练结束后加载最佳权重在测试集上评估。

---

## 新增模型

1. 复制任意 `scripts/train_*.py`，改三处即可：(timm库里面模型要自己去查，假如名字不对会报错)

```python
ClassifierTrainConfig(
    timm_model="your_timm_model_name",   # timm 模型名
    checkpoint_name="your_model.pth",      # 权重文件名
    curves_name="your_model.png",          # 曲线图文件名
    batch_size=32,                         # 按需调整
)
```

1. 两阶段训练、drop_rate、最佳权重保存等逻辑自动生效，无需改 `src/raicom/`。
2. 若要修改全局默认（80+20 epoch、学习率、dropout 等），改 `src/raicom/two_phase.py` 或 `src/raicom/timm_factory.py`。
3. 加入四模型集成：编辑 `scripts/train_ensemble_four_models_meta.py` 中的 `BACKBONE_SPECS`。

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

---

## 协作与许可证

协作约定见 [CONTRIBUTING.md](CONTRIBUTING.md)。许可证为 MIT，见 [LICENSE](LICENSE)。
