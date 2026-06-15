$filter = 'streamlit|app.py'
$procs = Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match $filter) }
if ($procs) {
  $procs | Select-Object ProcessId,CommandLine | Format-List
  foreach ($p in $procs) {
    try {
      Stop-Process -Id $p.ProcessId -Force -ErrorAction Stop
      Write-Output "Stopped PID $($p.ProcessId)"
    } catch {
      Write-Warning "Failed to stop PID $($p.ProcessId): $_"
    }
  }
} else {
  Write-Output 'No running Streamlit/app.py processes found'
}
Start-Sleep -Seconds 1
# Activate venv and start streamlit detached
$venvActivate = Join-Path -Path (Get-Location) -ChildPath '.venv\Scripts\Activate.ps1'
if (Test-Path $venvActivate) {
  & $venvActivate
} else {
  Write-Warning "Activate script not found at $venvActivate"
}
$pythonExe = Join-Path -Path (Get-Location) -ChildPath '.venv\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { Write-Error "Python executable not found at $pythonExe"; exit 1 }
$started = Start-Process -FilePath $pythonExe -ArgumentList '-m streamlit run app.py --server.headless true --server.port 8501' -NoNewWindow -WindowStyle Hidden -PassThru
if ($started) {
  Write-Output "Started Streamlit (detached) on port 8501 with PID $($started.Id)"
} else {
  Write-Warning 'Start-Process did not return a process object; Streamlit may not have started.'
}