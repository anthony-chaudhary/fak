<#
relogin_seat.ps1 -- the human backstop for a genuinely dead seat credential.

fak now self-heals a stale headless OAuth token by triggering Claude Code's own refresh
(cmd/fak/guard_child.go: the FAK_GUARD_AUTO_REFRESH branch spawns `claude -p` against the
seat's config dir so Claude Code rotates its own .credentials.json). That covers the common
case: the access token expired but the refresh token is still good.

This script is for the case auto-refresh CANNOT fix: the refresh token itself is dead (weeks
idle, revoked, org change), so only an interactive /login re-establishes the seat. It resolves
a seat name to its CLAUDE_CONFIG_DIR from `fak accounts status --json`, REFUSES to touch a seat
that is already healthy (status=ready / can_serve=true), and otherwise opens `claude /login`
for exactly that config dir.

  .\relogin_seat.ps1 -Seat july1-netra           # dry-run: report what it WOULD do
  .\relogin_seat.ps1 -Seat july1-netra -Live     # actually open /login for that seat

Exit codes: 0 = healthy seat, nothing to do (or dry-run report); 0 = launched on -Live;
2 = seat not found; 3 = resolution/tooling error.
#>
[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$Seat,
  # Actually open the interactive /login. Default is dry-run (report only), matching the
  # fleet's other launchers so a bare invocation is always side-effect-free.
  [switch]$Live,
  # Override the claude binary; else the fleet convention (FLEET_CLAUDE_EXE, then PATH, then
  # ~/.local/bin/claude.exe) -- the same order rwClaudeExe / accounts.ClaudeExe resolve.
  [string]$ClaudeExe = '',
  # Override the fak binary used to read account status.
  [string]$FakExe = ''
)
$ErrorActionPreference = 'Stop'

function Note($m) { Write-Output ("relogin_seat: {0}" -f $m) }

# Resolve fak: explicit flag, then FAK_EXE, then a built binary at the repo root, then PATH.
if (-not $FakExe) {
  $repoRoot = Split-Path -Parent $PSScriptRoot
  $fakCandidates = @($env:FAK_EXE, (Join-Path $repoRoot 'fak.exe'), (Join-Path $repoRoot 'fak'), 'fak.exe', 'fak') |
    Where-Object { $_ }
  foreach ($c in $fakCandidates) {
    if ([System.IO.Path]::IsPathRooted($c) -or $c.Contains('\') -or $c.Contains('/')) {
      if (Test-Path $c) { $FakExe = (Resolve-Path $c).Path; break }
    } else {
      $cmd = Get-Command $c -ErrorAction SilentlyContinue
      if ($cmd) { $FakExe = $cmd.Source; break }
    }
  }
}
if (-not $FakExe) { Note 'could not locate the fak binary (set -FakExe or FAK_EXE)'; exit 3 }

# Resolve claude: explicit flag, then the fleet convention.
if (-not $ClaudeExe) {
  if ($env:FLEET_CLAUDE_EXE) {
    $ClaudeExe = $env:FLEET_CLAUDE_EXE
  } else {
    $cmd = Get-Command claude -ErrorAction SilentlyContinue
    if ($cmd) { $ClaudeExe = $cmd.Source } else { $ClaudeExe = Join-Path $env:USERPROFILE '.local\bin\claude.exe' }
  }
}

# Read the account status surface (the single source of truth for seat -> dir + login state).
try {
  $raw = & $FakExe accounts status --json 2>$null
  $doc = ($raw | Out-String) | ConvertFrom-Json
} catch {
  Note "could not read `fak accounts status --json`: $($_.Exception.Message)"
  exit 3
}

$row = @($doc.seats) | Where-Object { $_.name -eq $Seat } | Select-Object -First 1
if (-not $row) {
  Note "seat '$Seat' not found in accounts status"
  Note ("known seats: " + ((@($doc.seats) | ForEach-Object { $_.name }) -join ', '))
  exit 2
}

$dir = "$($row.dir)"
$status = "$($row.status)"
$canServe = [bool]$row.can_serve

Note "seat=$Seat status=$status can_serve=$canServe dir=$dir"

# NEVER re-login a healthy seat: a ready/can_serve seat has live credentials, and opening
# /login would only risk landing on the wrong browser profile and smearing the identity.
if ($status -eq 'ready' -or $canServe) {
  Note 'already ready -- refusing to re-login a healthy seat (nothing to do)'
  exit 0
}

# A tombstoned/rehomed seat is not re-logged-in in place; it is served through its rehome
# target. Surface the next action fak already computed rather than opening /login on a dead dir.
if ($status -eq 'tombstoned' -or $status -eq 'missing_dir') {
  Note "seat status '$status' is not fixed by /login here."
  if ($row.next_action) { Note "next action: $($row.next_action)" }
  exit 0
}

if (-not (Test-Path $dir)) {
  Note "config dir does not exist: $dir"
  if ($row.next_action) { Note "next action: $($row.next_action)" }
  exit 3
}

if (-not $Live) {
  Note "WOULD run: CLAUDE_CONFIG_DIR=$dir `"$ClaudeExe`" /login   (re-run with -Live to do it)"
  if ($row.next_action) { Note "fak's next action: $($row.next_action)" }
  exit 0
}

# Live: open the interactive /login for exactly this seat's config dir.
Note "opening /login for $Seat (CLAUDE_CONFIG_DIR=$dir)"
$env:CLAUDE_CONFIG_DIR = $dir
& $ClaudeExe /login
exit 0
