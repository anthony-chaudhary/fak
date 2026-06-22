<#
runaway_process_reaper.ps1 -- kill runaway Git-Bash search processes before they
eat the machine, and log enough provenance to find out what spawned them.

Why this exists: on this Windows box, `find /` under Git Bash descends into
/proc/registry* (the whole Windows Registry, mounted as a dir tree, x3 views) and
/mnt/c (the whole C: drive, which contains self-referential junction loops like
C:\ProgramData\Application Data -> C:\ProgramData). MSYS find's cycle detection
fails on reparse points, so it recurses forever, holding an open directory handle
at every level. On 2026-06-21 two orphaned `find /` processes held 16.9M of the
system's 17.1M handles (98.8%) after ~4h each. The harness's command timeout had
killed their parent shells but not the find.exe grandchildren, so they ran on
detached. This reaper closes that gap.

Provenance: a runaway orphans FAST (the harness kills the parent shell at its
~2-min command timeout), so by the time the kill threshold trips, the live parent
is usually gone. To still answer "what spawned this", the reaper:
  * walks the ancestry chain (find <= bash <= claude ...) while ancestors are alive,
  * has a WATCH tier (low bar, log-only, deduped per process) that records that
    chain EARLY -- often before the process orphans,
  * always preserves the durable signals (full command line, PPID, start time)
    even after orphaning, and
  * writes a structured JSONL record per event for later aggregation.
Logs land in:  %LOCALAPPDATA%\Fleet\watchdog\runaway_reaper.log   (human)
               %LOCALAPPDATA%\Fleet\watchdog\runaway_reaper.jsonl (structured)

A legitimate find/grep finishes in seconds with a few hundred handles. The kill
defaults (200k handles / 10 CPU-min / orphaned >30min) cannot false-positive on
real usage; the WATCH bar (5k handles / 3 CPU-min) only logs, never kills.

  .\runaway_process_reaper.ps1                 # DRY-RUN (safe default): log what it WOULD kill
  .\runaway_process_reaper.ps1 -Live           # actually kill
  .\runaway_process_reaper.ps1 -Live -HandleMax 100000
#>
[CmdletBinding()]
param(
  [switch]$Live,
  # Process image names treated as bounded search tools that must never run away.
  [string[]]$Names = @('find.exe','grep.exe','findstr.exe','egrep.exe','fgrep.exe'),
  # KILL thresholds -- any one trips a reap:
  [int]$HandleMax = 200000,    # at/above this handle count == runaway, full stop.
  [int]$CpuMaxMin = 10,        # ...or this much accumulated CPU time (minutes).
  [int]$OrphanAgeMin = 30,     # ...or orphaned (parent gone) AND older than this (minutes).
  # WATCH thresholds -- log provenance (no kill) so we catch the spawner early:
  [int]$WatchHandles = 5000,   # a legit find/grep never reaches this.
  [int]$WatchCpuMin = 3,
  # How deep to walk the parent chain when recording provenance.
  [int]$AncestryDepth = 8,
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
$log    = Join-Path $LogDir 'runaway_reaper.log'
$jsonl  = Join-Path $LogDir 'runaway_reaper.jsonl'
$seenPath = Join-Path $LogDir 'runaway_seen.json'
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

$nowIso = [DateTimeOffset]::UtcNow.ToString('yyyy-MM-ddTHH:mm:ssZ')
function Iso($dt) { if ($dt) { ([DateTimeOffset]$dt).ToUniversalTime().ToString('yyyy-MM-ddTHH:mm:ssZ') } else { $null } }

$mode = if ($Live) { 'LIVE' } else { 'DRY-RUN' }
$now = Get-Date

# One enumeration of EVERY process -> the table we walk for ancestry + orphan detection.
$allProc = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue)
$byId = @{}
foreach ($pr in $allProc) { $byId[[int]$pr.ProcessId] = $pr }

function Cpu-Min($pr) {
  # UserModeTime + KernelModeTime are in 100-ns units; /6e8 -> minutes.
  if ($pr.UserModeTime -ne $null) { [math]::Round((([double]$pr.UserModeTime + [double]$pr.KernelModeTime)/6e8),1) } else { 0 }
}
function Get-Ancestry([int]$startPpid) {
  # Walk up the parent chain through still-alive ancestors. An orphan returns an
  # empty chain (the parent already exited) -- which is itself the finding.
  $chain = @(); $cur = $startPpid; $depth = 0
  while ($cur -and $byId.ContainsKey($cur) -and $depth -lt $AncestryDepth) {
    $a = $byId[$cur]
    $chain += [ordered]@{
      pid = [int]$a.ProcessId; ppid = [int]$a.ParentProcessId; name = $a.Name
      start = (Iso $a.CreationDate); cmd = $a.CommandLine
    }
    if ([int]$a.ParentProcessId -eq $cur) { break }   # guard against a self-parent loop
    $cur = [int]$a.ParentProcessId; $depth++
  }
  return $chain
}

# Dedup ledger so a long-lived WATCH suspect is logged once, not every tick.
# Keyed by "pid:startTicks" (stable identity). Reaps are always logged.
$seen = @{}
if (Test-Path $seenPath) {
  try { (Get-Content $seenPath -Raw | ConvertFrom-Json).PSObject.Properties | ForEach-Object { $seen[$_.Name] = $_.Value } } catch {}
}

$cands = @($allProc | Where-Object { $Names -contains $_.Name })
Note ("TICK $mode candidates={0} kill[h>=$HandleMax cpu>=${CpuMaxMin}m orphan>${OrphanAgeMin}m] watch[h>=$WatchHandles cpu>=${WatchCpuMin}m] cap=$MaxPerTick" -f $cands.Count)

$killed = 0
$livePidKeys = @{}
foreach ($c in $cands) {
  $cpid    = [int]$c.ProcessId
  $handles = [int]$c.HandleCount
  $cpuMin  = Cpu-Min $c
  $ageMin  = if ($c.CreationDate) { [math]::Round(($now - $c.CreationDate).TotalMinutes,1) } else { 0 }
  $ppid    = [int]$c.ParentProcessId
  $orphan  = -not $byId.ContainsKey($ppid)

  $reapReasons = @()
  if ($handles -ge $HandleMax) { $reapReasons += "handles=$handles" }
  if ($cpuMin  -ge $CpuMaxMin) { $reapReasons += "cpu=${cpuMin}m" }
  if ($orphan -and $ageMin -ge $OrphanAgeMin) { $reapReasons += "orphan+age=${ageMin}m" }

  $watch = ($handles -ge $WatchHandles) -or ($cpuMin -ge $WatchCpuMin)
  if (-not $reapReasons.Count -and -not $watch) { continue }

  # Provenance: ancestry chain (alive ancestors) + a compact summary line.
  $chain = Get-Ancestry $ppid
  $chainStr = if ($chain.Count) {
    ($chain | ForEach-Object { "$($_.name)(pid $($_.pid))" }) -join ' <= '
  } elseif ($orphan) { "LOST (parent pid $ppid already exited -- orphaned)" } else { "(none)" }

  $action = if ($reapReasons.Count) { if ($Live) { 'REAP' } else { 'WOULD-REAP' } } else { 'WATCH' }
  $reasonStr = if ($reapReasons.Count) { $reapReasons -join ' ' } else { "watch h=$handles cpu=${cpuMin}m" }
  $key = "${cpid}:" + $(if ($c.CreationDate) { ([DateTimeOffset]$c.CreationDate).UtcTicks } else { 0 })
  $livePidKeys[$key] = $nowIso

  # WATCH is deduped (log once per process); REAP/WOULD-REAP always logged.
  $skipLog = ($action -eq 'WATCH') -and $seen.ContainsKey($key)
  if (-not $skipLog) {
    Note "  $action $($c.Name) pid=$cpid ppid=$ppid age=${ageMin}m $reasonStr <= $chainStr"
    Note "       cmd: $($c.CommandLine)"
    $rec = [ordered]@{
      ts = $nowIso; action = $action; name = $c.Name; pid = $cpid; ppid = $ppid
      orphan = $orphan; handles = $handles; cpu_min = $cpuMin; age_min = $ageMin
      start = (Iso $c.CreationDate); reasons = $reapReasons; cmd = $c.CommandLine; ancestry = $chain
    } | ConvertTo-Json -Compress -Depth 8
    Add-Content -Path $jsonl -Value $rec
  }
  $seen[$key] = $nowIso

  if (-not $reapReasons.Count) { continue }     # WATCH only -> never kills
  if ($killed -ge $MaxPerTick) { Note "  per-tick cap reached ($MaxPerTick); leaving pid=$cpid"; continue }
  if (-not $Live) { continue }                  # DRY-RUN -> already logged WOULD-REAP
  try {
    Stop-Process -Id $cpid -Force -ErrorAction Stop
    $killed++
    Note "  KILLED pid=$cpid"
    Toast "Killed runaway $($c.Name)" "$reasonStr (pid $cpid)" "reap:$cpid"
  } catch {
    Note "  FAILED to reap pid=${cpid}: $($_.Exception.Message)"
  }
}

# Prune the dedup ledger to suspects still alive this tick, then persist.
$pruned = @{}
foreach ($k in $seen.Keys) { if ($livePidKeys.ContainsKey($k)) { $pruned[$k] = $seen[$k] } }
($pruned | ConvertTo-Json) | Set-Content -Path $seenPath -Encoding UTF8

Note "  done: reaped=$killed"
exit 0
