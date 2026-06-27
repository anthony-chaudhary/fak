# connect-fak-node.ps1 — connect this PowerShell session to a remote fak gateway.
#
# Exports ANTHROPIC_BASE_URL and ANTHROPIC_API_KEY into the current shell so that
# `claude`, `fak guard -- claude`, and any tool that reads those env vars will route
# through the always-on fak kernel on the target Mac node.
#
# Usage:
#   # In the current shell (env vars live only for this session):
#   . scripts\connect-fak-node.ps1 -GatewayHost 100.x.y.z -GatewayKey sk-fak-...
#
#   # Or with named values written to $PROFILE for persistence:
#   . scripts\connect-fak-node.ps1 -GatewayHost 100.x.y.z -GatewayKey sk-fak-... -Persist
#
#   # Disconnect (restore original ANTHROPIC_BASE_URL / clear the key):
#   . scripts\connect-fak-node.ps1 -Disconnect
#
#   # Smoke-test the gateway (curl /healthz):
#   . scripts\connect-fak-node.ps1 -GatewayHost 100.x.y.z -GatewayKey sk-fak-... -Probe
#
# The gateway is the fak serve instance started by tools/install-mac-node.sh on the
# Mac node. Get GatewayHost from: tailscale ip -4  (run on the Mac).
# Get GatewayKey from the output of: ./tools/install-mac-node.sh --bind-all
#
# TIP: after sourcing (dot-sourcing), run `claude` — every tool call it proposes goes
# through the kernel on the Mac before hitting the real Anthropic API.
[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string] $GatewayHost,

    [Parameter(Position = 1)]
    [string] $GatewayKey,

    [ValidateRange(1, 65535)]
    [int]    $GatewayPort = 8080,

    [switch] $Probe,
    [switch] $Persist,
    [switch] $Disconnect
)

$ErrorActionPreference = 'Stop'

function Write-Info  { Write-Host "[connect-fak-node] $args" -ForegroundColor Cyan }
function Write-Warn  { Write-Host "[connect-fak-node] WARNING: $args" -ForegroundColor Yellow }
function Write-Fatal { Write-Host "[connect-fak-node] ERROR: $args" -ForegroundColor Red; return }

# --- Disconnect: clear the fak gateway env, restore saved original if any ---
if ($Disconnect) {
    $saved = [System.Environment]::GetEnvironmentVariable('_FAK_ORIG_ANTHROPIC_BASE_URL', 'Process')
    if ($null -ne $saved) {
        $env:ANTHROPIC_BASE_URL = $saved
        [System.Environment]::SetEnvironmentVariable('_FAK_ORIG_ANTHROPIC_BASE_URL', $null, 'Process')
    } else {
        Remove-Item Env:\ANTHROPIC_BASE_URL -ErrorAction SilentlyContinue
    }
    Remove-Item Env:\ANTHROPIC_API_KEY -ErrorAction SilentlyContinue
    Write-Info "disconnected — ANTHROPIC_BASE_URL restored, API key cleared"
    return
}

# --- Validate args ---
if (-not $GatewayHost) {
    Write-Fatal "GatewayHost is required. Usage: . scripts\connect-fak-node.ps1 -GatewayHost <tailscale-ip> -GatewayKey <bearer-key>"
    return
}
if (-not $GatewayKey) {
    Write-Fatal "GatewayKey is required. Get it from the output of: ./tools/install-mac-node.sh --bind-all"
    return
}

$baseUrl = "http://${GatewayHost}:${GatewayPort}"
$healthzUrl = "${baseUrl}/v1/messages"   # fak serve exposes /healthz at the gateway root
$healthzCheck = "${baseUrl}/healthz"

# --- Probe: curl /healthz to verify reachability ---
if ($Probe) {
    Write-Info "probing $healthzCheck ..."
    try {
        $resp = Invoke-RestMethod -Uri $healthzCheck -Method GET `
            -Headers @{ Authorization = "Bearer $GatewayKey" } `
            -TimeoutSec 5
        Write-Info "gateway healthy: $($resp | ConvertTo-Json -Compress)"
    } catch {
        Write-Warn "gateway not reachable at $healthzCheck"
        Write-Warn "  Is the Mac node running?  Check: ./tools/install-mac-node.sh --status (on the Mac)"
        Write-Warn "  Is Tailscale connected?   tailscale status"
        Write-Warn "  Is the port exposed?      The node must have been installed with --bind-all"
        return
    }
}

# --- Save existing ANTHROPIC_BASE_URL so -Disconnect can restore it ---
if ($env:ANTHROPIC_BASE_URL -and $env:ANTHROPIC_BASE_URL -ne $baseUrl) {
    [System.Environment]::SetEnvironmentVariable('_FAK_ORIG_ANTHROPIC_BASE_URL', $env:ANTHROPIC_BASE_URL, 'Process')
}

# --- Set env vars in the current session ---
$env:ANTHROPIC_BASE_URL = $baseUrl
$env:ANTHROPIC_API_KEY  = $GatewayKey

Write-Info "connected to fak gateway at $baseUrl"
Write-Info "  ANTHROPIC_BASE_URL = $env:ANTHROPIC_BASE_URL"
Write-Info "  ANTHROPIC_API_KEY  = $($GatewayKey.Substring(0, [Math]::Min(8, $GatewayKey.Length)))... (truncated)"
Write-Info ""
Write-Info "Run:  claude    — every tool call now crosses the fak kernel on the Mac."
Write-Info "Done: . scripts\connect-fak-node.ps1 -Disconnect"

# --- Persist: write to $PROFILE so every new PowerShell session auto-connects ---
if ($Persist) {
    $block = @"

# fak gateway (written by connect-fak-node.ps1 — remove this block to disconnect)
`$env:ANTHROPIC_BASE_URL = "$baseUrl"
`$env:ANTHROPIC_API_KEY  = "$GatewayKey"
# end fak gateway
"@
    $profileDir = Split-Path $PROFILE -Parent
    if (-not (Test-Path $profileDir)) { New-Item -ItemType Directory -Force $profileDir | Out-Null }
    if (-not (Test-Path $PROFILE))    { New-Item -ItemType File -Force $PROFILE | Out-Null }

    $content = Get-Content $PROFILE -Raw -ErrorAction SilentlyContinue
    if ($content -notmatch '# fak gateway') {
        Add-Content -Path $PROFILE -Value $block
        Write-Info "written to `$PROFILE ($PROFILE) — all new sessions will auto-connect"
        Write-Warn "Remove the '# fak gateway' block from `$PROFILE to disconnect permanently"
    } else {
        Write-Warn "`$PROFILE already contains a fak gateway block — edit it manually to update"
    }
}
