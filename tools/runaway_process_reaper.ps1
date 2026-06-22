<#
runaway_process_reaper.ps1 -- kill runaway Git-Bash search processes before they
eat the machine.

Why this exists: on this Windows box, `find /` under Git Bash descends into
/proc/registry* (the whole Windows Registry, mounted as a dir tree, x3 views) and
/mnt/c (the whole C: drive, which contains self-referential junction loops like
C:\ProgramData\Application Data -> C:\ProgramData). MSYS find's cycle detection
fails on reparse points, so it recurses forever, holding an open directory handle
at every level. On 2026-06-21 two orphaned `find /` processes held 16.9M of the
system's 17.1M handles (98.8%) after ~4h each. The harness's command timeout had
killed their parent shells but not the find.exe grandchildren, so they ran on
detached. This reaper closes that gap.

A legitimate find/grep finishes in seconds with a few hundred handles. The
defaults here (200k handles / 10 CPU-min / orphaned >30min) cannot false-positive
on real usage -- they only fire on the runaway signature.

  .\runaway_process_reaper.ps1                 # DRY-RUN (safe default): log what it WOULD kill
  .\runaway_process_reaper.ps1 -Live           # actually kill
  .\runaway_process_reaper.ps1 -Live -HandleMax 100000
#>
[CmdletBinding()]
param(
  [switch]$Live,
  # Process image names treated as bounded search tools that must never run away.
  [string[]]$Names = @('find.exe','grep.exe','findstr.exe','egrep.exe','fgrep.exe'),
  # Any matching process at/above this handle count is a runaway, full stop.
  [int]$HandleMax = 200000,
  # ...or at/above this much accumulated CPU time (minutes).
  [int]$CpuMaxMin = 10,
  # ...or orphaned (parent gone) AND older than this many minutes (catches it early,
  # before handles balloon -- an orphaned search that outlives its shell is stuck).
  [int]$OrphanAgeMin = 30,
  # Don't kill more than this many per tick (defense against a logic bug).
  [int]$MaxPerTick = 25,
  [string]$LogDir = ''
)
$ErrorActionPreference = 'Stop'

$stateRoot = if ($env:FLEET_STATE_DIR) { $env:FLEET_STATE_DIR }
  elseif ($env:LOCALAPPDATA) { Join-Path $env:LOCALAPPDATA 'Fleet' }
  else { Join-Path ([System.IO.Path]::GetTempPath()) 'Fleet' }
if (-not $LogDir) { $LogDir = Join-Path $stateRoot 'watchdog' }
if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir -Force | Out-Null }
$log = Join-Path $LogDir 'runaway_reaper.log'
$notify = Join-Path $PSScriptRoot 'notify.ps1'

function Note($m) {
  $line = "{0}  {1}" -f ([DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')), $m
  Add-Content -Path $log -Value $line; Write-Output $line
}
function Toast($title, $msg, $key) {
  if (-not (Test-Path $notify)) { return }
  try {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $notify `
      -Title $title -Message $msg -Level 'warn' -LogDir $LogDir -Key $key -MinIntervalMinutes 60
  } catch {}
}

$mode = if ($Live) { 'LIVE' } else { 'DRY-RUN' }

# Snapshot of live PIDs so we can detect orphans (parent no longer present).
# NB: not $live -- that collides with the case-insensitive -Live switch param.
$livePids = @{}
Get-Process -ErrorAction SilentlyContinue | ForEach-Object { $livePids[$_.Id] = $true }

$nameFilter = ($Names | ForEach-Object { "Name='$_'" }) -join ' OR '
$cands = @(Get-CimInstance Win32_Process -Filter $nameFilter -ErrorAction SilentlyContinue)
Note ("TICK $mode candidates={0} thresholds: handles>=$HandleMax cpu>=${CpuMaxMin}m orphan>${OrphanAgeMin}m cap=$MaxPerTick" -f $cands.Count)

$now = Get-Date
$killed = 0
foreach ($c in $cands) {
  $p = Get-Process -Id $c.ProcessId -ErrorAction SilentlyContinue
  if (-not $p) { continue }
  $handles = $p.Handles
  $cpuMin  = if ($p.CPU) { [math]::Round($p.CPU/60,1) } else { 0 }
  $ageMin  = if ($c.CreationDate) { [math]::Round(($now - $c.CreationDate).TotalMinutes,1) } else { 0 }
  $orphan  = -not $livePids.ContainsKey([int]$c.ParentProcessId)

  $reasons = @()
  if ($handles -ge $HandleMax) { $reasons += "handles=$handles" }
  if ($cpuMin  -ge $CpuMaxMin) { $reasons += "cpu=${cpuMin}m" }
  if ($orphan -and $ageMin -ge $OrphanAgeMin) { $reasons += "orphan+age=${ageMin}m" }
  if (-not $reasons.Count) { continue }

  $why = $reasons -join ' '
  $tag = "$($c.Name) pid=$($c.ProcessId) $why"
  if ($killed -ge $MaxPerTick) { Note "  per-tick cap reached ($MaxPerTick); leaving $tag"; continue }

  if (-not $Live) { Note "  WOULD REAP $tag :: $($c.CommandLine)"; continue }
  try {
    Stop-Process -Id $c.ProcessId -Force -ErrorAction Stop
    $killed++
    Note "  REAPED $tag :: $($c.CommandLine)"
    Toast "Killed runaway $($c.Name)" "$why (pid $($c.ProcessId))" "reap:$($c.ProcessId)"
  } catch {
    Note "  FAILED to reap pid=$($c.ProcessId): $($_.Exception.Message)"
  }
}

Note "  done: reaped=$killed"
exit 0
