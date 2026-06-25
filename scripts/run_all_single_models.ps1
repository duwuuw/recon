# 依次在 GPU 上训练全部单模型（PowerShell）
# 用法（仓库根目录）:
#   .\scripts\run_all_single_models.ps1
# 换环境:
#   $env:RAICOM_CONDA_ENV = "xianyu"; .\scripts\run_all_single_models.ps1

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$condaEnv = if ($env:RAICOM_CONDA_ENV) { $env:RAICOM_CONDA_ENV } else { "fm" }

if (-not (Test-Path "data\raw\dataset\cloudy")) {
    Write-Error "缺少 data/raw/dataset，请先按 data/README.md 准备数据"
}

$env:HF_ENDPOINT = "https://hf-mirror.com"
$data = "data/raw/dataset"

function Invoke-GpuPython {
    param([string[]]$Args)
    conda run -n $condaEnv --no-capture-output python @Args
}

Write-Host "== GPU 批量训练 | conda env: $condaEnv ==" -ForegroundColor Cyan
conda run -n $condaEnv python -c "import torch; assert torch.cuda.is_available(), 'CUDA 不可用，请检查 fm/xianyu 环境'; print('GPU OK:', torch.cuda.get_device_name(0))"

$scripts = @(
    # 1. MobileNet
    "scripts/train_mobilenetv4.py",
    "scripts/train_mobilenetv4_hyper.py",
    # 2. FastViT
    "scripts/train_fastvit_s24.py",
    "scripts/train_fastvit_sa36.py",
    # 3. MambaOut
    "scripts/train_mambaout_kobe.py",
    "scripts/train_mambaout_small_rw.py",
    # 其余
    "scripts/train_convnext11.py",
    "scripts/train_dinov2_small.py",
    "scripts/train_efficientnet.py",
    "scripts/train_fasternet.py",
    "scripts/train_repvit.py",
    "scripts/train_repvit_m2.py",
    "scripts/train_resnet18.py",
    "scripts/train_vit11.py",
    "scripts/train_gdn.py"
)

$i = 0
foreach ($s in $scripts) {
    $i++
    Write-Host "`n[$i/$($scripts.Count)] $s" -ForegroundColor Yellow
    Invoke-GpuPython @($s, "--data-root", $data)
    if ($LASTEXITCODE -ne 0) {
        Write-Error "训练失败: $s"
    }
}

Write-Host "`n== 汇总 Test F1 ==" -ForegroundColor Cyan
Invoke-GpuPython @("scripts/summarize_f1.py", "--data-root", $data)
