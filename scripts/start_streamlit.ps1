$pythonExe = Join-Path -Path (Get-Location) -ChildPath '.venv311\Scripts\python.exe'
if (-not (Test-Path $pythonExe)) { Write-Error "Python executable not found at $pythonExe"; exit 1 }
$proc = Start-Process -FilePath $pythonExe -ArgumentList '-m streamlit run app.py --server.headless true --server.port 8501' -WindowStyle Hidden -PassThru
if ($proc) { Write-Output "Started Streamlit PID $($proc.Id)" } else { Write-Warning 'Start-Process did not return a process object.' }
Start-Sleep -Seconds 2
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -eq 8501 } | Select-Object LocalAddress,LocalPort,OwningProcess | Format-Table -AutoSize
