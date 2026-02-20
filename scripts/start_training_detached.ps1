param(
  [switch]$Detached,
  [switch]$Resume = $true,
  [string]$RunName = "archive2_auto_exp",
  [string]$Project = "runs/train",
  [string]$FromCheckpoint = "",
  [string]$DataYaml = "data_archive2_auto.yaml",
  [string]$Model = "yolov8n.pt",
  [int]$Epochs = 50,
  [int]$ImgSize = 640,
  [int]$Batch = 16,
  [int]$Workers = 4,
  [string]$Device = "0"
)

$ErrorActionPreference = "Stop"

# Primera ejecución: abrir otra ventana de PowerShell y correr este mismo script en modo Detached
if (-not $Detached) {
  $argsList = @(
    "-NoExit",
    "-ExecutionPolicy", "Bypass",
    "-File", "`"$PSCommandPath`"",
    "-Detached",
    "-RunName", "`"$RunName`"",
    "-Project", "`"$Project`"",
    "-DataYaml", "`"$DataYaml`"",
    "-Model", "`"$Model`"",
    "-Epochs", "$Epochs",
    "-ImgSize", "$ImgSize",
    "-Batch", "$Batch",
    "-Workers", "$Workers",
    "-Device", "`"$Device`""
  )

  if ($Resume) {
    $argsList += "-Resume"
  }

  if ($FromCheckpoint -ne "") {
    $argsList += @("-FromCheckpoint", "`"$FromCheckpoint`"")
  }

  Start-Process -FilePath "powershell.exe" -ArgumentList $argsList | Out-Null
  Write-Host "Ventana independiente iniciada. El entrenamiento seguirá aunque cierres VS Code." -ForegroundColor Green
  exit 0
}

$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

# Forzar UTF-8 para evitar caracteres corruptos en barra de progreso (tqdm/YOLO)
chcp 65001 | Out-Null
[Console]::InputEncoding = [System.Text.Encoding]::UTF8
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

# Detectar entorno virtual usado en este proyecto
$pythonCandidates = @(
  ".venv-1\Scripts\python.exe",
  ".venv\Scripts\python.exe"
)

$py = $null
foreach ($candidate in $pythonCandidates) {
  if (Test-Path $candidate) {
    $py = $candidate
    break
  }
}

if (-not $py) {
  Write-Host "ERROR: No se encontró python del venv (.venv-1 o .venv)." -ForegroundColor Red
  exit 1
}

if (-not (Test-Path $DataYaml)) {
  Write-Host "ERROR: No existe el archivo de dataset: $DataYaml" -ForegroundColor Red
  exit 1
}

if (-not (Test-Path "train_yolo.py")) {
  Write-Host "ERROR: No existe train_yolo.py en el root del proyecto." -ForegroundColor Red
  exit 1
}

$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logPath = "training_log_$timestamp.txt"

$checkpoint = ""
if ($FromCheckpoint -ne "") {
  $checkpoint = $FromCheckpoint
} elseif ($Resume) {
  $checkpoint = Join-Path $Project "$RunName\weights\last.pt"
}

Write-Host "=" * 80
Write-Host "Python: $py"
Write-Host "Log: $logPath"
Write-Host "Proyecto: $Project"
Write-Host "RunName: $RunName"
Write-Host "=" * 80

if ($Resume -and $checkpoint -ne "" -and (Test-Path $checkpoint)) {
  Write-Host "Reanudando desde checkpoint: $checkpoint" -ForegroundColor Cyan

  $code = @"
from ultralytics import YOLO
model = YOLO(r'''$checkpoint''')
model.train(resume=True, device='$Device')
"@

  & $py -c $code 2>&1 | Tee-Object -FilePath $logPath
  $exitCode = $LASTEXITCODE
} else {
  if ($Resume) {
    Write-Host "No se encontró checkpoint en: $checkpoint" -ForegroundColor Yellow
    Write-Host "Iniciando entrenamiento desde cero..." -ForegroundColor Yellow
  }

  & $py "train_yolo.py" --data $DataYaml --model $Model --epochs $Epochs --imgsz $ImgSize --batch $Batch --workers $Workers --device $Device --project $Project --name $RunName 2>&1 | Tee-Object -FilePath $logPath
  $exitCode = $LASTEXITCODE
}

Write-Host "=" * 80
if ($exitCode -eq 0) {
  Write-Host "Entrenamiento finalizado correctamente." -ForegroundColor Green
} else {
  Write-Host "Entrenamiento finalizó con error. ExitCode=$exitCode" -ForegroundColor Red
}
Write-Host "Revisa el log en: $logPath"
Write-Host "=" * 80

exit $exitCode
