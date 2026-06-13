# 数据集（仓库内样本）

本目录为 **4 类 × 50 张** 的子集，供克隆仓库后直接训练：

| 类别 | 说明 |
|------|------|
| rain | 雨 |
| snow | 雪 |
| fogsmog | 雾/霾 |
| hail | 冰雹 |

完整 11 类数据请放在本地 `weather_dataset/dataset/`，或设置 `RAICOM_DATA_ROOT`。

重新生成样本：

```bash
python scripts/build_sample_dataset.py
```

默认从 `weather_dataset/dataset` 抽样，输出到 `data/raw/dataset/`。
