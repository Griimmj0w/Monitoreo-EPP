# Check existence of best.pt
if (Test-Path .\best.pt) {
  Write-Output 'BEST_PT: EXISTS'
  Get-Item .\best.pt | Select-Object FullName,Length | Format-List
} else {
  Write-Output 'BEST_PT: NOT FOUND'
}

# Try fetching the Streamlit root page
try {
  $r = Invoke-WebRequest -Uri 'http://localhost:8501' -UseBasicParsing -TimeoutSec 10
  Write-Output 'HTTP: OK'
  Write-Output ("StatusCode: $($r.StatusCode)")
  $body = $r.Content
  if ($body) {
    Write-Output '--- Body (first 500 chars) ---'
    $body.Substring(0,[Math]::Min(500,$body.Length)) | Out-String | Write-Output
  } else {
    Write-Output 'Body empty'
  }
} catch {
  Write-Output 'HTTP: FAIL'
  Write-Output $_.Exception.Message
}
