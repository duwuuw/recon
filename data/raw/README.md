# data/raw/dataset

完整 **天气识别** 训练集（4 类，4999 张），ImageFolder 格式。

| 类别 | 张数 |
|------|------|
| cloudy | 2184 |
| rainy | 446 |
| snowy | 403 |
| sunny | 1966 |

详细导入与训练说明见上级目录：[../README.md](../README.md)

重新从 zip 导入：

```bash
python scripts/import_weather_zip.py --zip "路径/天气识别.zip"
```
