# .githooks/pre-commit.ps1: Companion-Aware Pre-Commit Gate for public fak
#
# Execution Lifecycle:
#   1) Gate 0 Hygiene: Inspect staged files; reject temporary/scratch files (.tmp, .gotmp, .gocache, temp*.ps1, scratch*.ps1, _scratch, etc.)
#   2) Go Format Check: For staged .go files, check gofmt -l and advise (or reject if $env:FAK_STRICT_GOFMT -eq "1")
#   3) Companion Discovery: Check for companion fak-private via $env:FAK_PRIVATE_ROOT or sibling ../fak-private
#   4) If Companion Present:
#      - Run secret leak audit: python "$companion/tools/scrub_public_copy.py" --audit-staged --root .
#      - Run 5-Gate boundary verification with the absolute public repository root after go -C changes directory
#      - Output: "✅ Companion 5-Gate boundary & leak checks passed."
#   5) If Companion Absent:
#      - Run public boundary lint: go run ./cmd/fak-dev boundary
#      - Output: "ℹ️ Standalone public checkout; public boundary checks passed."

$ErrorActionPreference = "Stop"
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding = [System.Text.Encoding]::UTF8

$hookRepoRoot = ""
if ($PSScriptRoot) {
    $hookRepoRoot = (Resolve-Path "$PSScriptRoot/..").Path
}
# Resolve the active worktree top-level first; never jump out of detached worktrees (#14515).
$worktreeRoot = (git rev-parse --show-toplevel 2>$null)
if ($worktreeRoot) {
    $worktreeRoot = (Resolve-Path $worktreeRoot).Path
} else {
    $worktreeRoot = (Get-Location).Path
}
Set-Location $worktreeRoot

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
# A linked worktree (.worktrees/<ticket>) has no sibling of its own; the main
# checkout's parent holds the companion.
$mainCheckoutRoot = ""
$commonGitDir = (git rev-parse --git-common-dir 2>$null)
if ($commonGitDir) {
    $mainCheckoutRoot = Split-Path -Parent ([System.IO.Path]::GetFullPath([System.IO.Path]::Combine($worktreeRoot, $commonGitDir)))
}
$companion = ""
if ($env:FAK_PRIVATE_ROOT -in @("none", "off", "disable")) {
    $companion = ""
} elseif ($env:FAK_PRIVATE_ROOT -and (Test-Path $env:FAK_PRIVATE_ROOT)) {
    $companion = $env:FAK_PRIVATE_ROOT
} elseif (Test-Path "$worktreeRoot/../fak-private") {
    $companion = (Resolve-Path "$worktreeRoot/../fak-private").Path
} elseif ($mainCheckoutRoot -and (Test-Path "$mainCheckoutRoot/../fak-private")) {
    $companion = (Resolve-Path "$mainCheckoutRoot/../fak-private").Path
} elseif ($hookRepoRoot -and (Test-Path "$hookRepoRoot/../fak-private")) {
    $companion = (Resolve-Path "$hookRepoRoot/../fak-private").Path
} elseif (Test-Path "../fak-private") {
    $companion = (Resolve-Path "../fak-private").Path
}

# 4. Execute checks based on companion presence
if ($companion) {
    Write-Host "Companion fak-private detected at: $companion"

    # Preserve GIT_INDEX_FILE and related git env vars for --staged audits so
    # partial commits (git commit -- <paths>, git commit -p) and worktree indexes
    # audit the exact staged snapshot (#14515).
    $pythonCmd = "python"
    if (-not (Get-Command python -ErrorAction SilentlyContinue) -and (Get-Command python3 -ErrorAction SilentlyContinue)) {
        $pythonCmd = "python3"
    }

    Write-Host "Running secret leak audit on staged files..."
    & $pythonCmd "$companion/tools/scrub_public_copy.py" --audit-staged --root "$worktreeRoot"
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        [Console]::Error.WriteLine("❌ COMMIT BLOCKED: Secret leak or private content detected in staged files!")
        [Console]::Error.WriteLine("   Please review diagnostics above and scrub private copy before committing.")
        exit $LASTEXITCODE
    }

    Write-Host "Running 5-Gate boundary verification..."
    $savedGitEnv = @{}
    foreach ($k in @("GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE", "GIT_OBJECT_DIRECTORY", "GIT_ALTERNATE_OBJECT_DIRECTORIES", "GIT_COMMON_DIR", "GIT_PREFIX")) {
        if (Test-Path "Env:\$k") {
            $savedGitEnv[$k] = (Get-Item "Env:\$k").Value
            Remove-Item "Env:\$k" -ErrorAction SilentlyContinue
        }
    }
    try {
        & go -C "$companion" run ./cmd/fak-boundary check --staged --fak-dir "$worktreeRoot" --private-dir "$companion"
        $boundaryExit = $LASTEXITCODE
    } finally {
        foreach ($kv in $savedGitEnv.GetEnumerator()) {
            Set-Item "Env:\$($kv.Key)" $kv.Value
        }
    }
    if ($boundaryExit -ne 0) {
        Write-Host ""
        [Console]::Error.WriteLine("❌ COMMIT BLOCKED: 5-Gate boundary encapsulation or placement violation detected!")
        [Console]::Error.WriteLine("   Please review diagnostics above and see BOUNDARY.md for remediation.")
        exit $boundaryExit
    }

    Write-Host "✅ Companion 5-Gate boundary & leak checks passed."
} else {
    Write-Host "ℹ️ Standalone public checkout (no companion fak-private detected)."
    if (Test-Path "$worktreeRoot/cmd/fak-dev") {
        Write-Host "Running public boundary lint..."
        & go run ./cmd/fak-dev boundary
        if ($LASTEXITCODE -ne 0) {
            Write-Host ""
            [Console]::Error.WriteLine("❌ COMMIT BLOCKED: Public boundary check failed!")
            exit $LASTEXITCODE
        }
    } else {
        Write-Host "ℹ️ cmd/fak-dev not present; skipping public boundary lint."
    }
    Write-Host "ℹ️ Standalone public checkout; public boundary checks passed."
}

exit 0
