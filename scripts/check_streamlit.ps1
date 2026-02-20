# List streamlit/app.py processes
Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -and ($_.CommandLine -match 'streamlit' -or $_.CommandLine -match 'app.py') } | Select-Object ProcessId,CommandLine | Format-List

# Show listeners on ports 8501-8510
Get-NetTCPConnection -State Listen | Where-Object { $_.LocalPort -ge 8501 -and $_.LocalPort -le 8510 } | Select-Object LocalAddress,LocalPort,OwningProcess | Format-Table -AutoSize

# Show python processes from venv311
Get-Process -Name python -ErrorAction SilentlyContinue | Where-Object { $_.Path -and ($_.Path -like '*\\.venv311\\*') } | Select-Object Id,Path | Format-List
