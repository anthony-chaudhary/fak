<#
register_proc_resource_guard.ps1 - install/remove the Scheduled Task that runs the
process-resource guard on a standing interval, so the host has a durable watch for
the runaway classes the level
watchdogs miss: a process whose thread/handle/working-set count has gone
pathological, an orphaned ephemeral helper (a dos_mcp.server outliving its session),
and -- the reason this companion to the find/grep reaper exists -- a SINGLE-THREADED
process pinning one core to 100% (a stuck spin loop / wedged terminal), which has a
normal thread count and so trips nothing else.

Report-only by default (logs to tools/_watchdog/proc_guard.log; never kills). Pass
-Enact to flip it to reaping: it then reaps flagged NON-protected runaways, orphaned
helpers, and orphaned/idle launcher shells (bash/sh/pwsh with a dead owner or no live
children -- always scanned so report-only installs log kill candidates for the soak),
and a CPU pin only after it has held the threshold across every sample window
(default 4 samples x 2s = 6s sustained), so a legitimate compile/test burst is never
killed. It never touches an OS-critical process (System/csrss/lsass/...) or its own
tree. See docs/perf-runaway-guard.md.

Registration prefers the native `fak process-guard` implementation because its
Windows relation scan uses Toolhelp/process APIs instead of `Get-CimInstance
Win32_Process`, avoiding the WMI-overload class this guard must not worsen. It
falls back to the legacy Python tool only when no fak binary is available.

Registration prefers an S4U task (windowless, session 0, survives logoff) when run
from an elevated shell, and otherwise falls back automatically to an Interactive
task (runs as this user while logged on, no admin needed) launched via pythonw.exe
so it never flashes a console window.

  .\register_proc_resource_guard.ps1                 # install REPORT-ONLY (safe default)
  .\register_proc_resource_guard.ps1 -Enact          # install ENACTING (reaps runaways/orphans/CPU pins)
  .\register_proc_resource_guard.ps1 -Action status
  .\register_proc_resource_guard.ps1 -Action remove
#>
[CmdletBinding()]
param(
  [ValidateSet('install','remove','status')] [string]$Action = 'install',
  [switch]$Enact,
  [int]$EveryMin = 10,
  # CPU-pin reaping bar (per-core %, sustained across $CpuSamples windows of $CpuWindow s).
  # 90%/core over 4x2s = 6s safely excludes legit bursty compute -- see the doc.
  [int]$MaxCpuPct = 90,
  [double]$CpuWindow = 2.0,
  [int]$CpuSamples = 4,
  [string]$TaskName = 'FleetProcResourceGuard',
  # Empty by default and resolved in the body: $PSScriptRoot is EMPTY when read in a
  # param-block default under Windows PowerShell 5.1 launched via -File (and in some
  # Scheduled-Task contexts), which crashed the old `Join-Path $PSScriptRoot ...`
  # default. Resolve it below with a 3-tier fallback -- the trap host-maintenance docs.
  [string]$Guard = '',
  [string]$FakExe = '',
  [string]$RepoRoot = ''
)
$ErrorActionPreference = 'Stop'

# --- Robust script root (works on PS 5.1 -File, dot-source, and pwsh 7) ---
$ScriptRoot = $PSScriptRoot
if (-not $ScriptRoot) { $ScriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path }
if (-not $ScriptRoot) { $ScriptRoot = (Get-Location).Path }
if (-not $Guard)    { $Guard    = Join-Path $ScriptRoot 'proc_resource_guard.py' }
if (-not $RepoRoot) { $RepoRoot = Split-Path -Parent $ScriptRoot }
if (-not $FakExe) {
  $cmd = Get-Command fak.exe -ErrorAction SilentlyContinue
  if ($cmd) { $FakExe = $cmd.Source }
  else {
    $repoFak = Join-Path $RepoRoot 'fak.exe'
    if (Test-Path $repoFak) { $FakExe = $repoFak }
  }
}

if ($Action -eq 'status') {
  $t = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
  if (-not $t) { Write-Output "NOT INSTALLED ($TaskName)"; return }
  $i = Get-ScheduledTaskInfo -TaskName $TaskName
  $a = ($t.Actions | Select-Object -First 1).Arguments
  $modeStr = if ($a -match '--enact') { 'ENACT (reaps)' } else { 'REPORT-ONLY' }
  $logon = ($t.Principal.LogonType)
  Write-Output "State=$($t.State) mode=$modeStr logon=$logon LastRun=$($i.LastRunTime) LastResult=$($i.LastTaskResult) NextRun=$($i.NextRunTime)"
  return
}
if ($Action -eq 'remove') {
  schtasks /Delete /TN $TaskName /F 2>$null | Out-Null
  Write-Output "removed $TaskName"; return
}

# Resolve a Python launcher: prefer `python`, fall back to the `py` launcher; and the
# windowless sibling (pythonw / pyw) used by the no-elevation Interactive fallback.
$py = (Get-Command python.exe -ErrorAction SilentlyContinue).Source
if (-not $py) { $py = (Get-Command py.exe -ErrorAction SilentlyContinue).Source }
if (-not $py) { $py = 'python' }
$pyw = ''
if ($py -match '(?i)python\.exe$') { $c = $py -replace '(?i)python\.exe$','pythonw.exe'; if (Test-Path $c) { $pyw = $c } }
elseif ($py -match '(?i)\\py\.exe$') { $c = $py -replace '(?i)py\.exe$','pyw.exe'; if (Test-Path $c) { $pyw = $c } }

# The guard scans every live process; it flags thread/handle runaways, reaps orphaned
# dos_mcp.server helpers, reaps orphaned/idle launcher shells (bash/sh/pwsh with a dead
# owner or zero live children -- the Git-for-Windows sh.exe hook-child sprawl the
# find/grep reaper never sees), and (the single-threaded-core-pin witness) a process
# holding >$MaxCpuPct% of one core sustained across $CpuSamples windows. --reap-idle-shells
# is always passed so report-only installs still LOG idle/orphan-shell kill candidates to
# the jsonl (the dry-run soak before flipping), while --enact -- only added with -Enact --
# is what actually reaps any of them.
$enactArg = if ($Enact) { ' --enact' } else { '' }
$useNative = $FakExe -and (Test-Path $FakExe)
if ($useNative) {
  $guardExe = $FakExe
  $guardArgs = "process-guard report --reap-orphans --reap-idle-shells --max-cpu-pct $MaxCpuPct --cpu-window $CpuWindow --cpu-samples $CpuSamples$enactArg"
} else {
  $guardExe = ''
  $guardArgs = "`"$Guard`" --reap-orphans --reap-idle-shells --max-cpu-pct $MaxCpuPct --cpu-window $CpuWindow --cpu-samples $CpuSamples$enactArg"
}

$trigger  = New-ScheduledTaskTrigger -Once -At (Get-Date).AddMinutes(1) `
              -RepetitionInterval (New-TimeSpan -Minutes $EveryMin) `
              -RepetitionDuration (New-TimeSpan -Days 3650)
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
              -StartWhenAvailable -MultipleInstances IgnoreNew -ExecutionTimeLimit (New-TimeSpan -Minutes 5)

function Register-Guard([string]$exe, $principal) {
  $taskAction = New-ScheduledTaskAction -Execute $exe -Argument $guardArgs -WorkingDirectory $RepoRoot
  Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
    -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
}

# Prefer S4U (windowless, session 0, survives logoff) -- needs an elevated shell.
# On access-denied, fall back to an Interactive task (this user, while logged on, no
# admin) launched via pythonw.exe so it stays windowless.
$modeNote = ''
$taskExe = if ($useNative) { $guardExe } else { $py }
try {
  $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
  Register-Guard $taskExe $principal
  $modeNote = 'S4U (windowless, session 0, survives logoff)'
} catch {
  $s4uErr = $_.Exception.Message
  $exe = if ($useNative) { $guardExe } elseif ($pyw) { $pyw } else { $py }
  try {
    $principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType S4U -RunLevel Limited
    $headlessArgs = "--headless `"$exe`" $guardArgs"
    $taskAction = New-ScheduledTaskAction -Execute 'conhost.exe' -Argument $headlessArgs -WorkingDirectory $RepoRoot
    Register-ScheduledTask -TaskName $TaskName -Action $taskAction -Trigger $trigger `
      -Principal $principal -Settings $settings -Force -ErrorAction Stop | Out-Null
    $wl = if ($useNative) { 'native fak via conhost --headless' } elseif ($pyw) { 'pythonw via conhost --headless' } else { 'python via conhost --headless' }
    $modeNote = "Interactive as $env:USERNAME (runs while logged on; $wl). S4U skipped: $s4uErr"
  } catch {
    throw "register failed. S4U: $s4uErr ; Interactive: $($_.Exception.Message). Try an elevated shell for the S4U daemon."
  }
}

$mode = if ($Enact) { "ENACT (reaps runaways/orphans/idle+orphan shells + CPU pins >$MaxCpuPct%/core sustained ${CpuSamples}x${CpuWindow}s)" } else { 'REPORT-ONLY (logs kill candidates, incl. idle/orphan shells; kills nothing)' }
Write-Output "installed $TaskName - every $EveryMin min, $mode"
if ($useNative) { Write-Output "  guard: native fak process-guard ($FakExe)" }
else { Write-Output "  guard: legacy Python fallback ($Guard)" }
Write-Output "  principal: $modeNote"
Write-Output "  log: tools/_watchdog/proc_guard.log (one line per scan)"
Write-Output "flip to enacting later:  .\tools\register_proc_resource_guard.ps1 -Enact"
Write-Output "remove:                  .\tools\register_proc_resource_guard.ps1 -Action remove"
