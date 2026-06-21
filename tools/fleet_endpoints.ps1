<#
fleet_endpoints.ps1 -- health/status for OPTIONAL private benchmark endpoints.

Reads an ignored local endpoint registry, decides each
endpoint's state from HARD signals only -- a `tailscale ping` reply and a TCP
connect -- never from prose, and prints a one-glance card. State is written to
tools/_registry/ENDPOINTS.txt so the latest picture is always on disk.

An endpoint that is disabled, offline, or has no live serve port is reported as
such and SKIPPED by the benchmark resolver; it is never an error. This tool is
read-only: it pings and opens TCP connections, it changes nothing.

  .\fleet_endpoints.ps1            # ping + classify enabled endpoints, print card
  .\fleet_endpoints.ps1 -Probe     # also TCP-probe the serve + ssh ports
  .\fleet_endpoints.ps1 -All       # include disabled (opt-in-pending) endpoints
  .\fleet_endpoints.ps1 -Quiet     # just refresh ENDPOINTS.txt (for a watchdog tick)
  .\fleet_endpoints.ps1 -Json      # emit machine-readable state to stdout
  .\fleet_endpoints.ps1 -Registry tools\fleet_endpoints.local.json
#>
[CmdletBinding()]
param(
  [string]$FleetDir = 'C:\work\fleet',
  [string]$Registry = '',
  [switch]$Probe,
  [switch]$All,
  [switch]$Quiet,
  [switch]$Json
)
$ErrorActionPreference = 'SilentlyContinue'

$regPath = $null
if ($Registry) {
  $regPath = if ([System.IO.Path]::IsPathRooted($Registry)) { $Registry } else { Join-Path $FleetDir $Registry }
} else {
  foreach ($candidate in @(
      'tools\fleet_endpoints.local.json',
      'tools\fleet_endpoints.json',
      'tools\fleet_endpoints.example.json'
    )) {
    $path = Join-Path $FleetDir $candidate
    if (Test-Path $path) { $regPath = $path; break }
  }
}
if (-not (Test-Path $regPath)) { Write-Error "registry not found: $regPath"; exit 2 }
$reg = Get-Content $regPath -Raw | ConvertFrom-Json

# Locate the tailscale CLI (PATH first, then the default install location).
$ts = (Get-Command tailscale -ErrorAction SilentlyContinue).Source
if (-not $ts) { $ts = 'C:\Program Files\Tailscale\tailscale.exe' }
$haveTs = Test-Path $ts

function Test-Tcp([string]$ipOrHost, [int]$port, [int]$timeoutMs = 2500) {
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $c.BeginConnect($ipOrHost, $port, $null, $null)
    if ($iar.AsyncWaitHandle.WaitOne($timeoutMs) -and $c.Connected) { return $true }
    return $false
  } catch { return $false } finally { $c.Close() }
}

# tailscale ping -> ('online', <ms>) on a pong, ('offline', $null) otherwise.
function Get-PingState([string]$hostName) {
  if (-not $haveTs) { return @{ state = 'unknown'; ms = $null } }
  $out = & $ts ping --c 1 --timeout 3s $hostName 2>&1 | Out-String
  if ($out -match 'pong .*in\s+(\d+)ms') { return @{ state = 'online'; ms = [int]$Matches[1] } }
  if ($out -match 'pong')                { return @{ state = 'online'; ms = $null } }
  return @{ state = 'offline'; ms = $null }
}

$rows = New-Object System.Collections.Generic.List[object]
foreach ($e in $reg.endpoints) {
  if (-not $e.enabled -and -not $All) {
    $rows.Add([pscustomobject]@{ name=$e.name; host=$e.tailnet_host; os=$e.os; state='DISABLED';
      detail='opt-in-pending (enabled=false)'; rtt=$null; serve=$null; ssh=$null }); continue
  }
  $p = Get-PingState $e.tailnet_host
  $serve = $null; $sshOpen = $null
  if ($Probe -and $p.state -eq 'online') {
    $serve   = Test-Tcp $e.tailscale_ip ([int]$e.serve_port)
    $sshOpen = Test-Tcp $e.tailscale_ip ([int]$e.ssh.port)
  }
  # Classify on hard signals, most-actionable last.
  $state = 'OFFLINE'; $detail = 'no tailscale pong'
  if ($p.state -eq 'unknown') { $state='UNKNOWN'; $detail='tailscale CLI not found' }
  elseif ($p.state -eq 'online') {
    if (-not $Probe)      { $state='ONLINE';       $detail='reachable (run -Probe for serve/ssh)' }
    elseif ($serve)       { $state='READY';        $detail="serve :$($e.serve_port) live" }
    else                  { $state='ONLINE';       $detail="reachable; serve :$($e.serve_port) not up" }
  }
  $rows.Add([pscustomobject]@{ name=$e.name; host=$e.tailnet_host; os=$e.os; state=$state;
    detail=$detail; rtt=$p.ms; serve=$serve; ssh=$sshOpen })
}

if ($Json) { $rows | ConvertTo-Json -Depth 4; exit 0 }

$now = (Get-Date).ToUniversalTime().ToString('yyyy-MM-dd HH:mm')
$L = New-Object System.Collections.Generic.List[string]
$L.Add("==================== FLEET BENCHMARK ENDPOINTS @ ${now}Z ====================")
$L.Add("driver: $($reg.driver.name) ($($reg.driver.tailscale_ip))   tailnet: $($reg.tailnet)")
if (-not $haveTs) { $L.Add("WARNING: tailscale CLI not found at $ts -- states are UNKNOWN") }
$L.Add("")
$L.Add(("{0,-16} {1,-26} {2,-8} {3,-9} {4,-6} {5}" -f 'NAME','HOST','OS','STATE','RTT','DETAIL'))
foreach ($r in $rows) {
  $rtt = if ($null -ne $r.rtt) { "$($r.rtt)ms" } else { '-' }
  $L.Add(("{0,-16} {1,-26} {2,-8} {3,-9} {4,-6} {5}" -f $r.name,$r.host,$r.os,$r.state,$rtt,$r.detail))
}
$ready = @($rows | Where-Object { $_.state -eq 'READY' }).Count
$online = @($rows | Where-Object { $_.state -eq 'ONLINE' }).Count
$L.Add("")
$L.Add("summary: $ready READY (serve live), $online ONLINE (no serve yet), of $($rows.Count) endpoint(s)")
if (-not $Probe) { $L.Add("hint: re-run with -Probe to test the serve/ssh ports") }
$L.Add("=============================================================================")

$card = $L -join "`n"
$regDir = Join-Path $FleetDir 'tools\_registry'
if (-not (Test-Path $regDir)) { New-Item -ItemType Directory -Path $regDir -Force | Out-Null }
$card | Set-Content -Path (Join-Path $regDir 'ENDPOINTS.txt') -Encoding UTF8
if (-not $Quiet) { $card }
