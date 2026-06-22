<#
runaway_process_scan.ps1 - READ-ONLY audit of processes that look like runaways or
leaks. Never kills anything; just reports. Use it for a manual machine health pass.

Surfaces, in order:
  1. system-wide handle total + the top handle holders (a handle leak shows here),
  2. Git-Bash search tools (find/grep/findstr) ranked by handles -- the known footgun,
  3. top CPU-time accumulators, and any orphaned (parent-gone) processes among them.

  .\runaway_process_scan.ps1
  .\runaway_process_scan.ps1 -Top 20
#>
[CmdletBinding()]
param(
  [int]$Top = 12,
  [string[]]$SearchNames = @('find.exe','grep.exe','findstr.exe','egrep.exe','fgrep.exe'),
  # Flag a search tool as suspicious at/above this many handles (legit ones use hundreds).
  [int]$SuspectHandles = 50000
)
$ErrorActionPreference = 'Stop'

$all = Get-Process -ErrorAction SilentlyContinue
$totalHandles = ($all | Measure-Object Handles -Sum).Sum
Write-Output "=== System-wide handles: $totalHandles ==="
if ($totalHandles -gt 1000000) { Write-Output "  WARNING: >1M handles system-wide -- a leak is likely. See the top holders below." }

Write-Output ""
Write-Output "=== Top $Top handle holders ==="
$all | Sort-Object Handles -Descending | Select-Object -First $Top `
  Name, Id, Handles, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}}, @{N='CPU_min';E={[math]::Round($_.CPU/60,1)}} |
  Format-Table -AutoSize | Out-String | Write-Output

# Git-Bash search tools: the known runaway shape. Join CIM (parent/start) with handles.
$live = @{}; $all | ForEach-Object { $live[$_.Id] = $true }
$nameFilter = ($SearchNames | ForEach-Object { "Name='$_'" }) -join ' OR '
$cands = @(Get-CimInstance Win32_Process -Filter $nameFilter -ErrorAction SilentlyContinue)
Write-Output "=== Git-Bash search tools (find/grep): $($cands.Count) running ==="
if ($cands.Count) {
  $now = Get-Date
  $rows = foreach ($c in $cands) {
    $p = Get-Process -Id $c.ProcessId -ErrorAction SilentlyContinue
    [PSCustomObject]@{
      Name    = $c.Name
      PID     = $c.ProcessId
      Handles = if ($p) { $p.Handles } else { 0 }
      CPU_min = if ($p -and $p.CPU) { [math]::Round($p.CPU/60,1) } else { 0 }
      Age_min = if ($c.CreationDate) { [math]::Round(($now - $c.CreationDate).TotalMinutes,1) } else { 0 }
      Orphan  = -not $live.ContainsKey([int]$c.ParentProcessId)
      Suspect = ($p -and $p.Handles -ge $SuspectHandles)
      CmdLine = $c.CommandLine
    }
  }
  $rows | Sort-Object Handles -Descending |
    Format-Table Name, PID, Handles, CPU_min, Age_min, Orphan, Suspect -AutoSize | Out-String | Write-Output
  $hot = @($rows | Where-Object { $_.Suspect -or $_.Orphan })
  if ($hot.Count) {
    Write-Output "  SUSPECTS (high handles or orphaned) -- consider runaway_process_reaper.ps1 -Live:"
    foreach ($h in $hot) { Write-Output "    pid=$($h.PID) handles=$($h.Handles) :: $($h.CmdLine)" }
  } else {
    Write-Output "  none look runaway."
  }
}

Write-Output ""
Write-Output "=== Top $Top CPU-time accumulators ==="
$all | Sort-Object CPU -Descending | Select-Object -First $Top `
  Name, Id, @{N='CPU_min';E={[math]::Round($_.CPU/60,1)}}, Handles, @{N='WS_MB';E={[math]::Round($_.WorkingSet64/1MB,1)}} |
  Format-Table -AutoSize | Out-String | Write-Output
