# fak/test.ps1 — Windows entry point for the canonical test runner.
#
# Running `go test ./...` natively on this host is blocked intermittently by a
# Windows Application Control policy (it refuses to exec the unsigned per-package
# test .exe files from %TEMP%). This wrapper runs the suite inside WSL instead,
# where Linux ELF test binaries aren't subject to that policy. See fak/test.sh
# for the full rationale.
#
# Usage:
#   .\fak\test.ps1                       # whole suite (./...)
#   .\fak\test.ps1 ./internal/ctxmmu/    # one package
#   .\fak\test.ps1 -count=1 ./...        # force a clean run
#
# Pass-through: every argument is forwarded verbatim to `go test` inside WSL.
#
# Distro selection: honor FAK_WSL_DISTRO if set; else prefer 'Ubuntu-24.04' when
# it is actually installed; else fall back to WSL's *default* distro (omit -d).
# Hardcoding 'Ubuntu-24.04' was a footgun — a node whose distro is just 'Ubuntu'
# (and with FAK_WSL_DISTRO unset) hit `WSL_E_DISTRO_NOT_FOUND` and never ran.
[CmdletBinding()]
param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $Rest)

$ErrorActionPreference = 'Stop'

$distro = $env:FAK_WSL_DISTRO
if (-not $distro) {
    # `wsl -l -q` is UTF-16 with stray NULs when captured; strip them per line.
    $installed = (& wsl.exe --list --quiet) |
        ForEach-Object { ($_ -replace "`0", '').Trim() } |
        Where-Object { $_ }
    if ($installed -contains 'Ubuntu-24.04') { $distro = 'Ubuntu-24.04' }
    # else: leave $distro empty -> use the WSL default distro below.
}

# Translate this script's Windows dir to a WSL /mnt path in PowerShell itself.
# We do NOT shell out to `wslpath`: PowerShell strips backslashes when passing
# a `C:\...` argument to wsl.exe, so wslpath sees a mangled path. Assumes the
# default /mnt/<drive> automount (the WSL default).
$drive = $PSScriptRoot.Substring(0, 1).ToLower()
$tail  = $PSScriptRoot.Substring(2) -replace '\\', '/'
$wslDir = "/mnt/$drive$tail"

# Forward selected FAK_* opt-ins into WSL. Env vars do NOT cross into WSL unless
# named in WSLENV, so add them there when set. Usage from PowerShell:
#   $env:FAK_FAST=1; .\fak\test.ps1 ./internal/ctxmmu/
function Add-WSLEnvVar([string]$Entry) {
    $parts = @()
    if ($env:WSLENV) {
        $parts = @($env:WSLENV -split ':' | Where-Object { $_ })
    }
    if ($parts -notcontains $Entry) {
        $env:WSLENV = (@($parts) + $Entry) -join ':'
    }
}

if ($env:FAK_FAST) {
    Add-WSLEnvVar 'FAK_FAST/u'
}
if ($env:FAK_ORACLE_DIRS) {
    # Keep this as a plain value: oracle directories are normally repo-relative
    # `.cache/...` paths, and the Go helper accepts Windows-style ';' lists.
    Add-WSLEnvVar 'FAK_ORACLE_DIRS/u'
}
if ($env:FAK_ORACLE_REQUIRED_FAMILIES) {
    Add-WSLEnvVar 'FAK_ORACLE_REQUIRED_FAMILIES/u'
}

$wslArgs = @()
if ($distro) { $wslArgs += @('-d', $distro) }
& wsl.exe @wslArgs bash "$wslDir/test.sh" @Rest
exit $LASTEXITCODE
