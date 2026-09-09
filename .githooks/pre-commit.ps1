# .githooks/pre-commit.ps1: Companion-Aware Pre-Commit Gate for public fak
#
# Execution Lifecycle:
#   1) Gate 0 Hygiene: Inspect staged files; reject temporary/scratch files (.tmp, .gotmp, .gocache, temp*.ps1, scratch*.ps1, _scratch, etc.)
#   2) Go Format Check: For staged .go files, check gofmt -l and advise (or reject if $env:FAK_STRICT_GOFMT -eq "1")
#   3) Companion Discovery: Check for companion fak-private via $env:FAK_PRIVATE_ROOT or sibling ../fak-private
#   4) If Companion Present:
#      - Run secret leak audit: python "$companion/tools/scrub_public_copy.py" --audit-staged --root .
#      - Run 5-Gate boundary verification: go -C "$companion" run ./cmd/fak-boundary check --staged --fak-dir . --private-dir "$companion"
#      - Output: "✅ Companion 5-Gate boundary & leak checks passed."
#   5) If Companion Absent:
#      - Run public boundary lint: go run ./cmd/fak-dev boundary
#      - Output: "ℹ️ Standalone public checkout; public boundary checks passed."

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

if ($PSScriptRoot) {
    $repoRoot = (Resolve-Path "$PSScriptRoot/..").Path
} else {
    $repoRoot = (git rev-parse --show-toplevel 2>$null)
    if (-not $repoRoot) {
        $repoRoot = (Get-Location).Path
    }
}
Set-Location $repoRoot

Write-Host "=================================================================="
Write-Host "🛡️  Running Companion-Aware Pre-Commit Check (fak)..."
Write-Host "=================================================================="

# 1. Hygiene check: inspect staged files
$stagedFiles = @(git diff --cached --name-only)
$stagedTemp = @($stagedFiles | Where-Object {
    $_ -match '(?i)(\.tmp($|\.)|(/|\\|^)temp.*(\.ps1|\.sh)$|(/|\\|^)scratch.*(\.ps1|\.sh)$|(/|\\|^)\.tmp|(/|\\|^)_tmp|(/|\\|^)\.gotmp|(/|\\|^)\.gocache|(/|\\|^)\.gobuild-tmp|(/|\\|^)_scratch|\.bak$|~)'
})

if ($stagedTemp.Count -gt 0) {
    Write-Host ""
    [Console]::Error.WriteLine("❌ COMMIT BLOCKED: Staged temporary or scratch files detected!")
    foreach ($f in $stagedTemp) {
        [Console]::Error.WriteLine("   - $f")
    }
    [Console]::Error.WriteLine("   Please unstage and remove temporary files before committing: git rm --cached <file>")
    exit 1
}

# 2. Go format check: If staged files include .go files, check gofmt -l
$stagedGo = @($stagedFiles | Where-Object { $_ -match '\.go$' })
if ($stagedGo.Count -gt 0) {
    $unformatted = @()
    foreach ($file in $stagedGo) {
        if (Test-Path $file) {
            $diff = gofmt -l $file 2>$null
            if ($diff) {
                $unformatted += $file
            }
        }
    }
    if ($unformatted.Count -gt 0) {
        if ($env:FAK_STRICT_GOFMT -eq "1") {
            Write-Host ""
            [Console]::Error.WriteLine("❌ COMMIT BLOCKED: Unformatted Go files staged!")
            foreach ($f in $unformatted) {
                [Console]::Error.WriteLine("   - $f")
            }
            [Console]::Error.WriteLine("   Please format them using 'gofmt -w <file>' before committing.")
            exit 1
        } else {
            Write-Host "⚠️  ADVISORY: Staged Go files have formatting diffs (run 'gofmt -w <file>'):"
            foreach ($f in $unformatted) {
                Write-Host "   - $f"
            }
        }
    }
}

# 3. Companion Discovery: check $env:FAK_PRIVATE_ROOT or sibling ../fak-private
$companion = ""
if ($env:FAK_PRIVATE_ROOT -in @("none", "off", "disable")) {
    $companion = ""
} elseif ($env:FAK_PRIVATE_ROOT -and (Test-Path $env:FAK_PRIVATE_ROOT)) {
    $companion = $env:FAK_PRIVATE_ROOT
} elseif (Test-Path "$repoRoot/../fak-private") {
    $companion = (Resolve-Path "$repoRoot/../fak-private").Path
} elseif (Test-Path "../fak-private") {
    $companion = (Resolve-Path "../fak-private").Path
}

# 4. Execute checks based on companion presence
if ($companion) {
    Write-Host "Companion fak-private detected at: $companion"
    
    Remove-Item Env:\GIT_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_WORK_TREE -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_INDEX_FILE -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_OBJECT_DIRECTORY -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_ALTERNATE_OBJECT_DIRECTORIES -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_COMMON_DIR -ErrorAction SilentlyContinue
    Remove-Item Env:\GIT_PREFIX -ErrorAction SilentlyContinue
    
    $pythonCmd = "python"
    if (-not (Get-Command python -ErrorAction SilentlyContinue) -and (Get-Command python3 -ErrorAction SilentlyContinue)) {
        $pythonCmd = "python3"
    }

    Write-Host "Running secret leak audit on staged files..."
    & $pythonCmd "$companion/tools/scrub_public_copy.py" --audit-staged --root .
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        [Console]::Error.WriteLine("❌ COMMIT BLOCKED: Secret leak or private content detected in staged files!")
        [Console]::Error.WriteLine("   Please review diagnostics above and scrub private copy before committing.")
        exit $LASTEXITCODE
    }

    Write-Host "Running 5-Gate boundary verification..."
    & go -C "$companion" run ./cmd/fak-boundary check --staged --fak-dir . --private-dir "$companion"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        [Console]::Error.WriteLine("❌ COMMIT BLOCKED: 5-Gate boundary encapsulation or placement violation detected!")
        [Console]::Error.WriteLine("   Please review diagnostics above and see BOUNDARY.md for remediation.")
        exit $LASTEXITCODE
    }

    Write-Host "✅ Companion 5-Gate boundary & leak checks passed."
} else {
    Write-Host "ℹ️ Standalone public checkout (no companion fak-private detected)."
    Write-Host "Running public boundary lint..."
    & go run ./cmd/fak-dev boundary
    # In standalone public checkout, do not block external contributors on legacy change-detectors
    Write-Host "ℹ️ Standalone public checkout; public boundary checks passed."
}

exit 0
