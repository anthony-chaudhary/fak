# Laptop (user) dispatch worker configuration
#
# This script configures the laptop node for DOS dispatch workers with deconfliction
# against the desktop node. Both nodes can run workers simultaneously without collision
# because the DOS arbiter enforces file-tree disjointness per lane.
#
# Setup:
# 1. Run this script to set environment variables for the session
# 2. Or set these permanently in System Environment Variables
# 3. Use different TARGET values per node to distribute load
#
# Deconfliction strategies:
# - Same lanes, different nodes: OK (arbiter serializes by lane)
# - Different lanes: ideal (parallel work, no arbitration needed)
# - Lane filtering via env: FLEET_LANE_ALLOWLIST (not yet implemented)
#
# Laptop (user) defaults:
# - Target: 2 workers (laptop is for testing/dev, desktop is production)
# - Backend: claude (default, can switch to opencode via FLEET_WORKER_BACKEND)
# - Workspace: auto-detected (repo root)

param(
    [int]$Target = 2,
    [string]$Backend = "claude",
    [switch]$Permanent
)

$ErrorActionPreference = 'Stop'

# Validate backend
if ($Backend -notin @('claude', 'opencode')) {
    Write-Error "Backend must be 'claude' or 'opencode'"
    exit 1
}

# Set session environment variables
$env:FLEET_WORKER_BACKEND = $Backend
$env:FAK_SUPERVISOR_TARGET = $Target

Write-Host "Laptop dispatch config (session):"
Write-Host "  FLEET_WORKER_BACKEND = $env:FLEET_WORKER_BACKEND"
Write-Host "  FAK_SUPERVISOR_TARGET = $env:FAK_SUPERVISOR_TARGET"
Write-Host ""
Write-Host "To make permanent, re-run with -Permanent"
Write-Host "To test dry-run: tools\.bin\dispatchworker --dry-run --backend $Backend --lane tools"

if ($Permanent) {
    # Set permanent environment variables (requires admin)
    [Environment]::SetEnvironmentVariable('FLEET_WORKER_BACKEND', $Backend, 'User')
    [Environment]::SetEnvironmentVariable('FAK_SUPERVISOR_TARGET', $Target, 'User')
    Write-Host "Set permanent environment variables (User scope)"
}

# Test the configuration via the interpreter-free Go launcher (cmd/dispatchworker;
# `make build` drops it at tools/.bin/dispatchworker, the dos.toml [supervise]
# worker_launch_template). Avoids the bare-'python' ENOENT on a python3-only node
# (#22) — the live launch path no longer shells a bare 'python'.
Write-Host ""
Write-Host "Testing tools\.bin\dispatchworker..."
tools\.bin\dispatchworker --dry-run --backend $Backend --lane tools --json | ConvertFrom-Json | Select-Object ok, lane, backend, workspace
