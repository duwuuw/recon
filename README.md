# raicom

基于 PyTorch 与 [timm](https://github.com/huggingface/pytorch-image-models) 的天气图像分类实验：单模型训练，以及四模型集成 + Stacking 元分类器（XGBoost）。

## 目录说明

| 路径 | 说明 |
|------|------|
| `notebooks/` | Jupyter 笔记本（训练、集成、小检查脚本等） |
| `src/raicom/` | 小型 Python 包（如 `paths` 用于统一数据集根路径） |
| `data/raw/` | 将 ImageFolder 格式数据放在 `dataset/` 下（详见 `data/README.md`） |
| `outputs/checkpoints/` | 保存的模型权重（比赛用途，可纳入版本库） |
| `outputs/figures/` | 曲线图与导出图（同上） |

## 环境配置

1. 建议使用 **Python 3.10+**。
2. 若需指定 CUDA 等构建，请先到 [pytorch.org](https://pytorch.org/) 安装对应 PyTorch。
3. 在仓库根目录执行：

```bash
pip install -e ".[train]"
```

或分步：

```bash
pip install -r requirements.txt
pip install -e .
```

4. 将数据集放到 `data/raw/dataset/`（ImageFolder：每个类别一个子文件夹），**或**设置环境变量指向你的数据根目录：

**Windows（cmd）：**

```bash
set RAICOM_DATA_ROOT=D:\path\to\your\dataset
```

**Linux / macOS：**

```bash
export RAICOM_DATA_ROOT=/path/to/your/dataset
```

5. 打开 `notebooks/` 下的笔记本即可；未执行 `pip install -e .` 时，部分笔记本会把仓库中的 `src` 加入 `sys.path` 以加载 `raicom.paths`。

## 许可证

MIT，见仓库根目录 `LICENSE`。
