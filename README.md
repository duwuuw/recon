# raicom

基于 PyTorch 与 [timm](https://github.com/huggingface/pytorch-image-models) 的天气图像分类实验：单模型训练，以及四模型集成 + Stacking 元分类器（XGBoost）。
这个仓库的代码全是cursor写的

四模型集成那里还没有写好，这是后续我即将完善的多模型投票+机器学习模型决策的思路

---

## 各文件夹与文件是做什么的

### 顶层目录

| 路径 | 作用 |
|------|------|
| `notebooks/` | 所有实验用 Jupyter 笔记本：单骨干训练、四模型集成元学习、临时检查（如 `check.ipynb`）等。重构后笔记本**只**放在这里，不再使用旧的 `module/` 目录名。 |
| `src/raicom/` | 可安装的 Python 包源码。目前主要是 `paths.py`：根据环境变量 `RAICOM_DATA_ROOT` 或仓库内常见目录，解析 **ImageFolder 数据集根路径**，方便多人协作时各机器路径不同。执行 `pip install -e .` 后可在任意位置 `from raicom.paths import default_data_root`。 |
| `data/` | 与**原始数据**相关的说明与占位。`data/raw/` 用来放本地大图集（默认不提交二进制数据，见 `.gitignore`）；更细的约定见 `data/README.md`。 |
| `outputs/` | **建议**存放要提交或备份的**权重**与**图**：`outputs/checkpoints/`、`outputs/figures/`。仓库已允许将这些文件纳入 Git（比赛提交用）。若目录里只有 `.gitkeep`，表示尚未放入实际文件。 |
| `.git/` | Git 版本历史（由 Git 管理，一般不用手改）。 |
| `.gitignore` | 规定哪些文件不上传（如虚拟环境、缓存、本地大数据集路径等）。 |
| `.env.example` | 环境变量示例；可复制为 `.env` 在本地使用（`.env` 本身被忽略，避免泄露路径或密钥）。 |
| `pyproject.toml` | 现代 Python 项目配置：包名 `raicom`、可选依赖组 `[train]` 等。 |
| `requirements.txt` | 用 `pip` 一键安装训练相关依赖的列表（与 `pyproject` 中的可选依赖对应）。 |
| `LICENSE` | 开源许可证文本（当前为 MIT）。 |
| `CONTRIBUTING.md` | 给协作者看的简单约定（分支、PR、数据与密钥等）。 |
| `README.md` | 本说明文件。 |

### `notebooks/` 里大致有什么

| 文件（示例） | 大致用途 |
|--------------|----------|
| `ensemble_four_models_meta.ipynb` | 四模型集成 + 元分类器（XGBoost）、软投票与相关曲线。 |
| `convnext11.ipynb`、`mambaout_kobe.ipynb`、`resnet18.ipynb` 等 | 各 timm 骨干的单模型训练与验证流程。 |
| `check.ipynb` | 环境或小功能检查。 |
| `gdn.ipynb` | 其他实验草稿。 |

具体保存路径以笔记本里的 `torch.save` / `plt.savefig` 为准（见下文「权重与图」）。

### `src/raicom/` 里有什么

| 文件 | 作用 |
|------|------|
| `__init__.py` | 包标识与版本号。 |
| `paths.py` | `repo_root()`、`default_data_root()`：解析仓库根目录与数据集根目录。 |

本地执行 `pip install -e .` 后，可能生成 `src/raicom.egg-info/`（已在 `.gitignore` 中忽略，不必提交）。

---

## 权重和图：我之前那些文件你放到哪里了？我就这样与cursor吵架

**没有**把你的旧权重、旧图「搬家」到 `outputs/` 或其它新路径里；重构只做了一件事：把 **`module/` 下的 `.ipynb` 移到 `notebooks/`**，并删除空的 **`module/` 目录**。

因此分两种情况说明：

1. **笔记本里写的是相对文件名**（例如 `efficientnet.pth`、`vit.png`、`convnext.pth`）  
   这些文件会写在 **Jupyter 当时的「当前工作目录」** 下，不一定是 `notebooks/`。常见是：从仓库根目录启动 Jupyter 时文件在 **`d:\dl\raicom\`** 根下；若从 `notebooks/` 启动，则可能在 **`d:\dl\raicom\notebooks\`**。  
   **当前仓库里的 `notebooks/` 下只有 `.ipynb`，没有替你自动生成或迁移任何 `.pth` / `.png`。**

2. **若你曾把 `.pth`、`.png` 放在已删除的 `module/` 里**  
   当时用脚本删除了整个 `module/` 文件夹，**其中若含有非笔记本文件，有可能一并被删掉**。请到 **Windows 回收站** 或自己的 **网盘 / 备份** 里查找是否还能恢复；我无法从仓库里还原未提交过的二进制文件。

**之后建议**：在笔记本里把保存路径改成仓库内的统一目录，例如：

- 权重 → `outputs/checkpoints/某模型名.pth`
- 图 → `outputs/figures/某图名.png`

便于比赛打包与 Git 提交。需要的话可以再改一版笔记本里的保存路径常量（可单独说一声）。

---

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

4. 将数据集放到 `data/raw/dataset/`（ImageFolder：每个类别一个子文件夹），**或**设置环境变量：

**Windows（cmd）：**

```bash
set RAICOM_DATA_ROOT=D:\path\to\your\dataset
```

**Linux / macOS：**

```bash
export RAICOM_DATA_ROOT=/path/to/your/dataset
```

5. 打开 `notebooks/` 下的笔记本即可；未执行 `pip install -e .` 时，部分笔记本会把仓库中的 `src` 加入 `sys.path` 以加载 `raicom.paths`。

---

## 协作与许可证

协作流程见 `CONTRIBUTING.md`。许可证为 MIT，见 `LICENSE`。
