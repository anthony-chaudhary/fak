# ci.ps1 — the `make ci` equivalent on Windows (unit 12): build + vet + test.
# Exit non-zero on any failure so it is a single mechanical witness.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

Write-Host "== go build =="
go build ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== go vet =="
go vet ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== go test =="
go test ./...
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "== claims lint =="
& (Join-Path $PSScriptRoot "claims-lint.ps1")
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "CI OK"
exit 0
