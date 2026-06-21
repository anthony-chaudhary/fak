# fetch-model.ps1 — Windows/PowerShell sibling of fetch-model.sh. Produces the real
# model weights the in-kernel engine loads from FAK_MODEL_DIR. See ../GETTING-STARTED.md §4b.
#
# Usage:
#   ./scripts/fetch-model.ps1            # export the default SmolLM2-135M-Instruct
#   ./scripts/fetch-model.ps1 -Check     # preflight only: report python + the plan
#
# Knobs (env): FAK_EXPORT_MODEL, FAK_MODEL_DIR, FAK_EXPORT_VENV, PYTHON.
param([switch]$Check)
$ErrorActionPreference = "Stop"

$Model = if ($env:FAK_EXPORT_MODEL) { $env:FAK_EXPORT_MODEL } else { "HuggingFaceTB/SmolLM2-135M-Instruct" }

# Resolve the fak module root (this script lives in <fak>\scripts\).
$Root     = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ExportPy = Join-Path $Root "internal\model\export_oracle.py"
$Req      = Join-Path $Root "scripts\requirements-export.txt"

function Get-DefaultName {
    $n = ($Model -split "/")[-1]
    $n = $n.ToLower()
    if ($n -eq "smollm2-135m-instruct") { return "smollm2-135m" }
    return $n
}
$Out  = if ($env:FAK_MODEL_DIR)   { $env:FAK_MODEL_DIR }   else { Join-Path $Root ("internal\model\.cache\" + (Get-DefaultName)) }
# Default the venv under the already-git-ignored model cache so it is never committed.
$Venv = if ($env:FAK_EXPORT_VENV) { $env:FAK_EXPORT_VENV } else { Join-Path $Root "internal\model\.cache\export-venv" }

# Pick a python.
$Py = if ($env:PYTHON) { $env:PYTHON } else { "python" }
if (-not (Get-Command $Py -ErrorAction SilentlyContinue)) {
    Write-Error "fetch-model: need python (set PYTHON=...)"; exit 1
}
if (-not (Test-Path $ExportPy)) {
    Write-Error "fetch-model: cannot find $ExportPy - run from inside the fak/ tree"; exit 1
}

function Assert-LastExit {
    param([string]$What)
    if ($LASTEXITCODE -ne 0) {
        throw "$What failed with exit code $LASTEXITCODE"
    }
}

if ($Check) {
    Write-Host "fetch-model preflight"
    Write-Host ("  python : " + (& $Py --version 2>&1) + "  (" + (Get-Command $Py).Source + ")")
    Write-Host "  model  : $Model"
    Write-Host "  out    : $Out"
    Write-Host "  venv   : $Venv"
    Write-Host "  export : $ExportPy"
    Write-Host "(run without -Check to create the venv, download, and export)"
    exit 0
}

# venv + deps (idempotent).
if (-not (Test-Path $Venv)) {
    Write-Host "fetch-model: creating venv at $Venv"
    & $Py -m venv $Venv
    Assert-LastExit "python -m venv"
}
$VenvPy = Join-Path $Venv "Scripts\python.exe"
if (-not (Test-Path $VenvPy)) { $VenvPy = Join-Path $Venv "bin/python" }  # WSL-style venv
if (-not (Test-Path $VenvPy)) { Write-Error "fetch-model: venv at $Venv has no python"; exit 1 }

Write-Host "fetch-model: installing export deps (CPU torch + transformers + numpy)..."
& $VenvPy -m pip install --quiet --upgrade pip
Assert-LastExit "pip install --upgrade pip"
& $VenvPy -m pip install --quiet -r $Req
Assert-LastExit "pip install -r requirements-export.txt"

New-Item -ItemType Directory -Force -Path $Out | Out-Null

# export_oracle.py pins HF offline via os.environ.setdefault; pre-set to 0 so the first
# download is allowed. The HF cache makes later runs offline-safe.
$env:HF_HUB_OFFLINE = "0"
$env:TRANSFORMERS_OFFLINE = "0"

Write-Host "fetch-model: exporting $Model -> $Out"
& $VenvPy $ExportPy --model $Model --out $Out
Assert-LastExit "export_oracle.py"

foreach ($file in @("config.json", "manifest.json", "weights.f32")) {
    $path = Join-Path $Out $file
    if (-not (Test-Path $path)) {
        throw "fetch-model: export did not produce $path"
    }
    if ((Get-Item $path).Length -le 0) {
        throw "fetch-model: export produced empty $path"
    }
}

Write-Host ""
Write-Host "fetch-model: done. To serve the real weights:"
Write-Host ("  `$env:FAK_MODEL_DIR = `"" + $Out + "`"")
Write-Host ("  ./fak serve --engine inkernel --model " + (Get-DefaultName))
