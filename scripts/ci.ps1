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

# index-sync (#511): the curated INDEX.md / llms.txt must not drift from the tree.
# Resolve a python interpreter once (Windows ships `python`/`py`, not `python3`).
$py = if (Get-Command python3 -ErrorAction SilentlyContinue) { "python3" }
      elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" }
      else { $null }
if ($null -ne $py) {
    Write-Host "== index-sync =="
    Set-Location (Join-Path $PSScriptRoot "..")
    & $py tools/check_index_sync.py --audit-tree
    if ($LASTEXITCODE -ne 0) { exit 1 }
    & $py tools/gen_llms_full.py --check
    if ($LASTEXITCODE -ne 0) { exit 1 }
} else {
    Write-Host "== index-sync (warn): python not found; gate skipped =="
}

Write-Host "CI OK"
exit 0
