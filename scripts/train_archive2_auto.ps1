<#
Train YOLOv8 using dataset in archive2_auto.
Creates a data yaml automatically if not provided.

Usage:
  .\scripts\train_archive2_auto.ps1
  .\scripts\train_archive2_auto.ps1 -Epochs 80 -ImgSize 640 -Model yolov8s.pt -Name epp_v1
#>

param(
  [string]$DatasetDir = "archive2_auto",
  [string]$DataYaml = "data_archive2_auto.yaml",
  [string]$Model = "yolov8n.pt",
  [int]$Epochs = 50,
  [int]$ImgSize = 640,
  [int]$Batch = 16,
  [int]$Workers = 4,
  [string]$Device = "0",
  [string]$Project = "runs/train",
  [string]$Name = "archive2_auto_exp",
  [int]$NumClasses = 4,
  [string]$ClassNames = "person,helmet,safety vest,no safety vest"
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (-not (Test-Path ".venv\\Scripts\\python.exe")) {
  Write-Host "ERROR: .venv not found. Run setup.ps1 first." -ForegroundColor Red
  exit 1
}

$py = ".\\.venv\\Scripts\\python.exe"

if (-not (Test-Path $DatasetDir)) {
  Write-Host "ERROR: Dataset folder not found: $DatasetDir" -ForegroundColor Red
  exit 1
}

if (-not (Test-Path $DataYaml)) {
  Write-Host "Creating $DataYaml from $DatasetDir..."

  $trainImages = Join-Path $DatasetDir "train\\images"
  $valImages = Join-Path $DatasetDir "val\\images"
  $imagesRoot = Join-Path $DatasetDir "images"

  if ((Test-Path $trainImages) -and (Test-Path $valImages)) {
    $trainPath = $trainImages
    $valPath = $valImages
  } elseif (Test-Path $imagesRoot) {
    $trainPath = $imagesRoot
    if (Test-Path $valImages) {
      $valPath = $valImages
    } else {
      $valPath = $imagesRoot
    }
  } else {
    $trainPath = $DatasetDir
    $valPath = $DatasetDir
  }

  $names = $ClassNames.Split(",") | ForEach-Object { $_.Trim() } | Where-Object { $_ -ne "" }
  $namesQuoted = $names | ForEach-Object { '"' + $_ + '"' }

  @(
    "train: $trainPath",
    "val: $valPath",
    "",
    "nc: $NumClasses",
    "",
    ("names: [" + ($namesQuoted -join ", ") + "]")
  ) | Set-Content -Path $DataYaml -Encoding utf8

  Write-Host "Wrote $DataYaml"
}

Write-Host "Starting training..."
& $py "train_yolo.py" --data $DataYaml --epochs $Epochs --model $Model --imgsz $ImgSize --batch $Batch --workers $Workers --device $Device --project $Project --name $Name
