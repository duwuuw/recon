# 数据集说明与操作指南

本文档面向**未使用 AI 辅助**的队友：按顺序逐步操作即可。  
文中所有路径均相对于**仓库根目录**（即包含 `scripts/`、`data/`、`src/` 的那一层，下文简称「根目录」）。

---

## 0. 开始之前：确认你在正确目录

打开终端（Windows 推荐 **PowerShell** 或 **Anaconda Prompt**），进入仓库根目录：

```powershell
cd 你的路径\raicom
```

**检查是否进对位置**（以下命令应能列出文件，而不是报错）：

```powershell
dir scripts
dir data\raw\dataset
```

若 `data\raw\dataset` 不存在或为空，请先完成 [第二节：导入数据集](#二第一次使用导入数据集)。

---

## 1. 数据是什么、放在哪里

### 1.1 默认数据路径

```
data/raw/dataset/
```

训练脚本**默认**读取上述路径，一般**不需要**再传 `--data-root`。

### 1.2 当前数据概况

| 中文 | 文件夹名 | 约张数 |
|------|----------|--------|
| 多云 | `cloudy` | 2184 |
| 雨天 | `rainy` | 446 |
| 雪天 | `snowy` | 403 |
| 晴天 | `sunny` | 1966 |

合计 **4999 张**，**4 个类别**。  
类别数必须与代码里 `src/raicom/constants.py` 的 `NUM_CLASSES = 4` 一致（**不要改**除非全队一起改代码）。

### 1.3 正确目录结构（ImageFolder）

根目录下每个**子文件夹 = 一个类别**，文件夹里直接放图片：

```
data/raw/dataset/
├── cloudy/
│   ├── cloudy_00001.jpg
│   ├── cloudy_00002.jpg
│   └── ...
├── rainy/
│   └── ...
├── snowy/
│   └── ...
└── sunny/
    └── ...
```

**错误示例（不要这样）：**

```
data/raw/dataset/train/cloudy/...   ← 多了一层 train，脚本会找不到类
data/raw/dataset/cloudy/cloudy/...  ← 多嵌套一层
```

### 1.4 手动检查数据是否齐全（复制粘贴即可）

**PowerShell：**

```powershell
Get-ChildItem data\raw\dataset -Directory | ForEach-Object {
    $n = (Get-ChildItem $_.FullName -File).Count
    Write-Output "$($_.Name): $n"
}
```

**预期输出应包含 4 行，且合计约 4999：**

```
cloudy: 2184
rainy: 446
snowy: 403
sunny: 1966
```

若类别名不是这四个，或只有 1～2 个文件夹，说明数据未导入完整，回到第二节重新导入。

---

## 2. 第一次使用：导入数据集

适用于：刚克隆仓库、`data/raw/dataset` 为空、或需要**用官方 zip 覆盖**现有数据。

### 2.1 准备 zip 文件

竞赛包一般名为 `天气识别.zip`，内部结构类似：

```
天气识别.zip
└── （解压后某文件夹）/
    ├── train.zip       ← 图片在这个文件里
    ├── train.py
    └── ...
```

`train.zip` 内部才是：

```
train/
├── cloudy/
├── rainy/
├── snowy/
└── sunny/
```

**推荐做法（便于全队统一相对路径）：**

1. 把 `天气识别.zip` 复制到仓库内：`data/raw/天气识别.zip`
2. 后续导入命令统一写 `--zip data/raw/天气识别.zip`

也可以把 zip 放在任意位置，但 `--zip` 后面写**相对于当前终端所在目录**的路径；**请始终在根目录下执行命令**，避免搞混。

### 2.2 安装 Python 依赖（只需做一次）

在**根目录**执行：

```powershell
pip install -r requirements.txt
pip install -e .
```

若使用 conda，先激活你的环境再执行上面两行。

验证安装：

```powershell
python -c "import torch; import timm; print('ok', torch.__version__)"
```

应打印 `ok` 和 PyTorch 版本号，无报错。

### 2.3 执行导入（会覆盖旧数据）

在**根目录**执行：

```powershell
python scripts/import_weather_zip.py --zip data/raw/天气识别.zip
```

若 zip 不在 `data/raw/`，把路径换成你的相对路径，例如：

```powershell
python scripts/import_weather_zip.py --zip ../downloads/天气识别.zip
```

**仅查看 zip 里有多少张、不写入磁盘：**

```powershell
python scripts/import_weather_zip.py --zip data/raw/天气识别.zip --dry-run
```

**导入成功时**，终端末尾类似：

```
  cloudy: 2184
  rainy: 446
  snowy: 403
  sunny: 1966
done: 4 classes, 4999 images -> data\raw\dataset
```

导入完成后，再跑 [1.4 节](#14-手动检查数据是否齐全复制粘贴即可) 的检查命令确认。

### 2.4 导入脚本参数说明

| 参数 | 含义 | 默认值 |
|------|------|--------|
| `--zip` | `天气识别.zip`、`train.zip` 或已解压文件夹 | （必填） |
| `--dest` | 输出 ImageFolder 根目录 | `data/raw/dataset` |
| `--dry-run` | 只统计，不删除、不写入 | 关闭 |

> **注意：** 不加 `--dry-run` 时，会先**删除** `data/raw/dataset` 再写入，不要在该目录里存放其他文件。

---

## 3. 环境：一次性准备清单

按顺序打勾：

- [ ] 已进入仓库**根目录**（`dir scripts` 有输出）
- [ ] **已激活 GPU 环境**：`conda activate fm` 或 `conda activate xianyu`
- [ ] CUDA 可用：`python -c "import torch; print(torch.cuda.is_available())"` → **True**
- [ ] 已 `pip install -e .`（在该 conda 环境内）
- [ ] `data/raw/dataset` 下 4 个类别文件夹齐全（见 1.4 节）

**预下载 timm 预训练权重**（联网时，在 fm/xianyu 环境中）：

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "import timm; timm.create_model('convnextv2_nano', pretrained=True, num_classes=4)"
```

---

## 4. 开始训练（最常用流程）

> **完整版**：15 个单模型逐条命令、批量训练、单独测试、F1 汇总见 [docs/TRAINING_AND_EVAL.md](../docs/TRAINING_AND_EVAL.md)。

以下命令均在**根目录**执行。

### 4.1 单模型训练（GPU）

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"

python scripts/train_convnext11.py
```

默认 **GPU 训练**，日志应出现 `Device: cuda:0 (...)`。

**训练成功的标志：**

1. 开头打印 `类别数 4` 和 `['cloudy', 'rainy', 'snowy', 'sunny']`（顺序可能按字母排）
2. 打印 `train/val/test:` 约 `3999 / 500 / 500`
3. 每个 epoch 有 `Val Acc`、`Val F1`
4. 结束时有 `最终测试集` 和 `macro-F1`
5. 生成文件：
   - `checkpoints/convnext.pth`（最佳权重）
   - `checkpoints/convnext.png`（训练曲线）

### 4.2 其他单模型脚本（用法相同）

| 脚本 | 说明 |
|------|------|
| `scripts/train_convnext11.py` | ConvNeXt V2 Nano |
| `scripts/train_mobilenetv4.py` | MobileNetV4 |
| `scripts/train_efficientnet.py` | EfficientNet V2 |
| `scripts/train_resnet18.py` | ResNet 系 |
| `scripts/train_vit11.py` | ViT |
| `scripts/train_timm_preset.py` | 通用 preset，需额外参数 |

查看 preset 列表：

```powershell
python scripts/train_timm_preset.py --list
```

训练某个 preset（示例）：

```powershell
python scripts/train_timm_preset.py convnextv2_nano
```

### 4.3 集成训练（多模型，耗时更长）

```powershell
python scripts/train_ensemble_four_models_meta.py
```

输出目录默认 `checkpoints/`，含各骨干权重与 `ensemble_report.json`。

### 4.4 仅微调分类头 10 轮（本地快速实验）

```powershell
python head_only_local/train_convnext11.py
```

权重默认在 `head_only_local/checkpoints/`。说明见 `head_only_local/README.md`。

### 4.5 常用可选参数

```powershell
# 显式指定数据（一般不必）
python scripts/train_convnext11.py --data-root data/raw/dataset

# 改 batch size（显存不足时改小，如 16 或 8）
python scripts/train_convnext11.py --batch-size 16

# 指定权重输出目录
python scripts/train_convnext11.py --output-dir checkpoints/my_run

# 强制 CPU（仅调试，正常勿用）
# python scripts/train_convnext11.py --cpu
```

---

## 5. 指定数据路径的三种方式

优先级从高到低：

| 方式 | 做法 | 适用场景 |
|------|------|----------|
| 命令行 | `--data-root data/raw/dataset` | 临时换一份数据 |
| 环境变量 | 见下表 | 整个终端会话都用同一路径 |
| 默认 | 放好 `data/raw/dataset` | **推荐，最省事** |

**环境变量（当前终端有效）：**

PowerShell：

```powershell
$env:RAICOM_DATA_ROOT = "data/raw/dataset"
python scripts/train_convnext11.py
```

CMD：

```cmd
set RAICOM_DATA_ROOT=data/raw/dataset
python scripts/train_convnext11.py
```

Linux / macOS：

```bash
export RAICOM_DATA_ROOT=data/raw/dataset
python scripts/train_convnext11.py
```

路径可以是相对路径（相对**启动命令时的工作目录**，因此请始终在根目录执行）。

---

## 6. 数据如何被划分

逻辑在 `src/raicom/data.py`，**固定随机种子**，全队结果可复现：

| 子集 | 比例 | 当前约张数 |
|------|------|------------|
| 训练 | 80% | 3999 |
| 验证 | 10% | 500 |
| 测试 | 10% | 500 |

验证集用于训练过程中选最佳权重；测试集只在最后评估一次。

**自己验证划分（可选）：**

```powershell
python -c "import sys; sys.path.insert(0,'src'); from raicom.data import build_imagefolder_loaders; tl,vl,te,nc,n=build_imagefolder_loaders('data/raw/dataset',32); print(nc,n); print(len(tl.dataset),len(vl.dataset),len(te.dataset))"
```

预期：`4 ['cloudy', 'rainy', 'snowy', 'sunny']` 和 `3999 500 500`。

---

## 7. 生成小数据集（调试专用，可选）

完整集训练较慢时，可抽样做 smoke test：

```powershell
python scripts/build_sample_dataset.py --source data/raw/dataset --dest data/raw/dataset_sample --per-class 50
```

然后用小集训练：

```powershell
python scripts/train_convnext11.py --data-root data/raw/dataset_sample
```

`data/raw/dataset_sample` 不会自动提交 git，仅本地使用。

---

## 8. 常见问题与处理

### 8.1 `数据目录不存在`

**原因：** `data/raw/dataset` 不存在或路径写错。

**处理：**

1. 执行 [第二节](#二第一次使用导入数据集) 导入  
2. 或 `--data-root` 指向真实存在的 ImageFolder 根目录  

---

### 8.2 `数据集有 X 类，需要 4 类`

**原因：** 文件夹数量不是 4，或类名文件夹嵌套错了。

**处理：**

1. 重新检查 [1.3 节](#13-正确目录结构imagefolder) 结构  
2. 重新导入：`python scripts/import_weather_zip.py --zip data/raw/天气识别.zip`  
3. **不要**只改 `NUM_CLASSES` 而不改数据和模型  

---

### 8.3 终端显示 Device: cpu

**原因：** 用了 base Python，未激活 fm/xianyu。

**处理：**

```powershell
conda activate fm
python -c "import torch; print(torch.cuda.is_available())"
python scripts/train_convnext11.py
```

---

### 8.4 预训练权重下载失败 / 超时

**原因：** 无法访问 huggingface.co。

**处理（PowerShell，同一终端内先设再训练）：**

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
python head_only_local/prefetch_weights.py
python scripts/train_convnext11.py
```

---

### 8.5 训练很慢

确认日志里是 **`Device: cuda:0`**，不是 cpu。全量 100 epoch 在 GPU 上仍需要较长时间。

---

### 8.6 `ModuleNotFoundError: No module named 'raicom'`

**原因：** 未安装本仓库包。

**处理（根目录）：**

```powershell
pip install -e .
```

---

## 9. 队友协作建议

1. **统一在根目录执行命令**，路径都写相对的，例如 `data/raw/dataset`，不要每人写不同的绝对盘符。  
2. **zip 统一放** `data/raw/天气识别.zip`（该文件较大，通常不提交 git；每人本地放一份）。  
3. **改数据或类别**务必全队同步，并改 `src/raicom/constants.py`。  
4. **权重与曲线**在 `checkpoints/`，已在 `.gitignore`；要共享模型请用网盘 / 对象存储，不要直接 push 大文件。  
5. 遇到报错：先复制**完整终端报错**给队友，并说明执行的是哪一条命令。

---

## 10. 快速命令速查（全部在根目录）

```powershell
# 检查数据
Get-ChildItem data\raw\dataset -Directory | ForEach-Object { "$($_.Name): $((Get-ChildItem $_.FullName -File).Count)" }

# 导入数据（覆盖）
python scripts/import_weather_zip.py --zip data/raw/天气识别.zip

# 检查划分
python -c "import sys; sys.path.insert(0,'src'); from raicom.data import build_imagefolder_loaders; tl,vl,te,nc,n=build_imagefolder_loaders('data/raw/dataset',32); print(nc,n,len(tl.dataset),len(vl.dataset),len(te.dataset))"

```powershell
conda activate fm
$env:HF_ENDPOINT = "https://hf-mirror.com"

# 训练
python scripts/train_convnext11.py

# 批量训练全部单模型（GPU）
.\scripts\run_all_single_models.ps1
```

---

## 11. 相关文件索引

| 文件 | 作用 |
|------|------|
| `data/raw/dataset/` | 默认训练图片 |
| `data/raw/README.md` | 数据目录简要说明 |
| `scripts/import_weather_zip.py` | 从 zip 导入 |
| `scripts/build_sample_dataset.py` | 抽样小数据集 |
| `src/raicom/constants.py` | 类别数 `NUM_CLASSES` |
| `src/raicom/data.py` | 划分与 DataLoader |
| `src/raicom/paths.py` | 默认数据路径逻辑 |
| `scripts/train_*.py` | 各模型训练入口 |
| `checkpoints/` | 训练输出（运行后生成） |

更完整的模型与训练说明见仓库根目录 [README.md](../README.md)。  
**逐个模型训练、测试、汇总 F1** 见 [docs/TRAINING_AND_EVAL.md](../docs/TRAINING_AND_EVAL.md)。
