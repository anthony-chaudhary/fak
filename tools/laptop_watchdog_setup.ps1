# Laptop (user) DOS Supervisor Watchdog Setup
#
# This script registers the DOS supervisor watchdog as a Windows Scheduled Task
# for continuous fleet operation on the laptop. The Python watchdog (dos_supervisor_watchdog.py)
# provides dry-run-by-default canary launches with workspace safety checks.
#
# Usage:
#   .\tools\laptop_watchdog_setup.ps1 [-Target 2] [-Enable]
#
# Parameters:
#   -Target  : Number of concurrent workers (default: 2 for laptop)
#   -Enable  : Actually enable the task (default: dry-run/report only)
#
# The watchdog runs every 5 minutes and invokes dos_supervisor_watchdog.py
# with the target worker count. Use --live with the watchdog directly for
# an actual worker dispatch.

[CmdletBinding()]
param(
    [int]$Target = 2,
    [switch]$Enable
)

$ErrorActionPreference = 'Stop'

# Paths — resolve the repo root from this script's own location (tools/ lives at
# the repo root) so the watchdog setup works from any clone, not just one
# operator's machine path.
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$WatchdogScript = "python"
$WatchdogArgs = Join-Path $RepoRoot "tools\dos_supervisor_watchdog.py"
$TaskName = "FleetDOSWatchdog"

# Verify repo exists
if (-not (Test-Path $RepoRoot)) {
    Write-Error "Repository not found at $RepoRoot"
    exit 1
}

# Verify watchdog script exists
if (-not (Test-Path $WatchdogArgs)) {
    Write-Error "Watchdog script not found at $WatchdogArgs"
    exit 1
}

Write-Host "=== Fleet Supervisor Watchdog Setup (Laptop) ===" -ForegroundColor Cyan
Write-Host "Repository: $RepoRoot"
Write-Host "Target workers: $Target"
Write-Host "Task name: $TaskName"
Write-Host ""

# Check if task already exists
$ExistingTask = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
if ($ExistingTask) {
    Write-Host "Task '$TaskName' already exists." -ForegroundColor Yellow
    Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:$false"
    Write-Host ""
}

# Build the action
$Action = New-ScheduledTaskAction `
    -Execute $WatchdogScript `
    -Argument "`"$WatchdogArgs`" --target $Target --json" `
    -WorkingDirectory $RepoRoot

# Build the trigger (every 5 minutes)
$Trigger = New-ScheduledTaskTrigger -Once -At (Get-Date) -RepetitionInterval (New-TimeSpan -Minutes 5)

# Principal (run as current user)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Highest

# Settings
$Settings = New-ScheduledTaskSettingsSet `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -StartWhenAvailable `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 1)

Write-Host "Task configuration:" -ForegroundColor Cyan
Write-Host "  Trigger  : Every 5 minutes"
Write-Host "  Action   : python $WatchdogArgs --target $Target --json"
Write-Host "  Principal: $($Principal.UserId) (interactive)"
Write-Host "  Settings : Allow on battery, restart on failure"
Write-Host ""

if (-not $Enable) {
    Write-Host "DRY-RUN MODE: Pass -Enable to actually create the task." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "To create manually:"
    Write-Host "  Register-ScheduledTask -TaskName '$TaskName' -Action `$Action -Trigger `$Trigger -Principal `$Principal -Settings `$Settings"
    Write-Host ""
    Write-Host "To test the watchdog immediately:"
    Write-Host "  & `$WatchdogScript -Target $Target"

    # Test the watchdog manually
    Write-Host ""
    Write-Host "Testing watchdog (dry-run)..." -ForegroundColor Cyan
    & $WatchdogScript $WatchdogArgs --target $Target --json | ConvertFrom-Json | Select-Object ok, action, reason

    exit 0
}

# Register the task
Write-Host "Creating scheduled task..." -ForegroundColor Cyan
try {
    Register-ScheduledTask -TaskName $TaskName -Action $Action -Trigger $Trigger -Principal $Principal -Settings $Settings -ErrorAction Stop | Out-Null
    Write-Host "Task '$TaskName' created successfully." -ForegroundColor Green
    Write-Host ""
    Write-Host "To view: Get-ScheduledTask -TaskName '$TaskName' | Select-Object *"
    Write-Host "To run: Start-ScheduledTask -TaskName '$TaskName'"
    Write-Host "To remove: Unregister-ScheduledTask -TaskName '$TaskName' -Confirm:`$false"
} catch {
    Write-Error "Failed to create task: $_"
    exit 1
}
