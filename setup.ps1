<# 
Setup script for the EPP Streamlit app on Windows.
Creates venv, installs PyTorch CUDA, installs requirements, and verifies GPU.
#>

$ErrorActionPreference = "Stop"

Write-Host "== EPP Streamlit App Setup ==" -ForegroundColor Cyan

# Ensure we're running from the project directory
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

if (-not (Test-Path ".venv")) {
  Write-Host "Creating venv (.venv) with Python 3.11..."
  py -3.11 -m venv .venv
} else {
  Write-Host "Found existing .venv"
}

Write-Host "Activating venv..."
. ".\.venv\Scripts\Activate.ps1"

Write-Host "Checking Python version..."
$pyVersion = python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
if ($pyVersion -ne "3.11") {
  Write-Host "ERROR: This project requires Python 3.11. Detected $pyVersion." -ForegroundColor Red
  Write-Host "Please install Python 3.11 from python.org and recreate the venv." -ForegroundColor Yellow
  Exit 1
}

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing PyTorch with CUDA (cu121)..."
try {
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
} catch {
  Write-Host "cu121 failed, trying cu124..." -ForegroundColor Yellow
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
}

Write-Host "Installing project requirements..."
pip install -r requirements.txt

Write-Host "Verifying CUDA availability..."
python -c "import torch; print('cuda_available:', torch.cuda.is_available()); print('gpu:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

Write-Host ""
Write-Host "Setup complete." -ForegroundColor Green
Write-Host "To run the app: .\\.venv\\Scripts\\Activate.ps1; streamlit run app.py"
