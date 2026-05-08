# artifacts

本目录用于集中存放**训练产生的权重、曲线图**等（建议与 `notebooks/` 里保存路径一致）。

## 从回收站恢复的内容（`recovered_from_recycle_bin/`）

已用仓库内脚本扫描 `D:\$RECYCLE.BIN` 中「删除前路径在 `D:\dl\raicom\` 下」的项目，并**复制**到 `artifacts/recovered_from_recycle_bin/`，**不会清空回收站**。

- **`module/`**：在回收站里找到的、曾经位于 `D:\dl\raicom\module\` 的笔记本（如 flow matching 相关 `.ipynb`）。  
- **其它子目录**（如 `rl/`、`flow_mamba/`）：若你曾删过整个目录且进了回收站，也会被一并复制过来，体积可能很大；不需要可自行删除本机副本。

### 关于天气分类的 `.pth` / 训练曲线 `.png`

当时用 PowerShell 的 `Remove-Item -Recurse` 删除 `module/` 时，**这类文件默认不会进入回收站**，因此无法从回收站找回。若你曾在**仓库根目录**或**其它路径**保存过（Jupyter 当前工作目录），仍可能留在本机别处，需要自己在资源管理器中按文件名搜索（如 `efficientnet.pth`、`ensemble_ckpt_*.pth`）。

### 再次扫描 / 恢复

在仓库根目录执行：

```bash
python scripts/scan_recycle_bin.py
```

若回收站 SID 文件夹与脚本中不一致，请编辑 `scripts/scan_recycle_bin.py` 里的 `RECYCLE_SID`（在 `D:\$RECYCLE.BIN\` 下查看以 `S-1-5-` 开头的子文件夹名）。
