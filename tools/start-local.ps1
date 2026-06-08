$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $PSScriptRoot
$Node = "C:\Program Files\nodejs\node.exe"
$Python = "C:\Program Files\Python312\python.exe"

$ports = @(5173, 8010)
foreach ($port in $ports) {
  $listeners = netstat -ano | Select-String ":$port\s"
  foreach ($line in $listeners) {
    $parts = ($line.ToString() -split "\s+") | Where-Object { $_ }
    if ($parts.Length -ge 5 -and $parts[3] -eq "LISTENING") {
      Stop-Process -Id ([int]$parts[4]) -Force -ErrorAction SilentlyContinue
    }
  }
}

$env:PYTHONPATH = Join-Path $Root "backend"

$apiOut = Join-Path $Root "backend-8010.out.log"
$apiErr = Join-Path $Root "backend-8010.err.log"
$webOut = Join-Path $Root "frontend-5173.out.log"
$webErr = Join-Path $Root "frontend-5173.err.log"

$api = Start-Process -FilePath $Python `
  -ArgumentList @("-m", "uvicorn", "rsl_gesture.api.app:app", "--host", "127.0.0.1", "--port", "8010", "--app-dir", "backend") `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput $apiOut `
  -RedirectStandardError $apiErr `
  -PassThru

$web = Start-Process -FilePath $Node `
  -ArgumentList @("tools/static-server.mjs", "frontend/dist", "127.0.0.1", "5173") `
  -WorkingDirectory $Root `
  -WindowStyle Hidden `
  -RedirectStandardOutput $webOut `
  -RedirectStandardError $webErr `
  -PassThru

Start-Sleep -Seconds 2

Write-Output "backend_pid=$($api.Id)"
Write-Output "frontend_pid=$($web.Id)"
Write-Output "frontend=http://127.0.0.1:5173"
Write-Output "backend=http://127.0.0.1:8010"
