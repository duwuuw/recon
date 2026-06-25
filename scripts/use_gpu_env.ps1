# GPU 训练环境（PowerShell）
# 在仓库根目录执行:  . .\scripts\use_gpu_env.ps1
#
# 默认激活 fm；改用 xianyu:  $env:RAICOM_CONDA_ENV = "xianyu"; . .\scripts\use_gpu_env.ps1

$ErrorActionPreference = "Stop"

$envName = if ($env:RAICOM_CONDA_ENV) { $env:RAICOM_CONDA_ENV } else { "fm" }
$allowed = @("fm", "xianyu")
if ($envName -notin $allowed) {
    Write-Warning "RAICOM_CONDA_ENV=$envName 不在推荐列表 fm/xianyu，仍将尝试激活"
}

Write-Host "== 激活 conda 环境: $envName ==" -ForegroundColor Cyan
conda activate $envName

$env:HF_ENDPOINT = "https://hf-mirror.com"

python -c @"
import torch
print('torch', torch.__version__)
print('cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    x = torch.zeros(1, device='cuda:0')
    print('gpu', torch.cuda.get_device_name(0))
"@

Write-Host "`n已就绪。示例: python scripts/train_convnext11.py --data-root data/raw/dataset" -ForegroundColor Green
