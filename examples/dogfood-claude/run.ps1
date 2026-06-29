<#
.SYNOPSIS
  Adoption-shaped wrapper for the dogfood-claude launcher (Windows twin of run.sh).

.DESCRIPTION
  This is a THIN wrapper: it prints "what to look for" hints, then invokes the shipped
  launcher (scripts\dogfood-claude.ps1) with your flags passed straight through. All the
  real work — build fak, serve a local model (the in-tree transformers shim by default,
  no ollama), front it with the kernel as a native Anthropic /v1/messages server, wire and
  launch the real Claude Code CLI, tear down on exit — lives in that one script. We do NOT
  reimplement any of it here (see README.md).

    .\examples\dogfood-claude\run.ps1 --smoke            # curl the wire end-to-end (no model), then exit
    .\examples\dogfood-claude\run.ps1 --probe "say pong" # ONE headless live Claude Code turn, then exit
    .\examples\dogfood-claude\run.ps1                    # interactive Claude Code on the local model

  Every flag the launcher accepts (--smoke / --probe / --print-env / --list-accounts /
  --install / -Kernel) and every FAK_DOGFOOD_* env knob passes through unchanged. The full
  reference is ..\..\DOGFOOD-CLAUDE.md.
#>

$ErrorActionPreference = 'Stop'

$here    = Split-Path -Parent $MyInvocation.MyCommand.Path
$fakDir  = (Resolve-Path (Join-Path $here '..\..')).Path     # examples\dogfood-claude -> fak\
$launcher = Join-Path $fakDir 'scripts\dogfood-claude.ps1'

function Log([string]$msg) { Write-Host "[dogfood-example] $msg" -ForegroundColor Cyan }

if (-not (Test-Path $launcher)) {
    Write-Error "launcher not found: $launcher"
    exit 1
}

Log "wrapping the shipped launcher: $launcher"
Log "what to look for once a turn runs:"
Log "  - the fak serve log - each /v1/messages turn + the per-call verdict on every tool"
Log "    Claude proposes: ALLOW, POLICY_BLOCK / DEFAULT_DENY / SELF_MODIFY refusals, a"
Log "    TRANSFORM (redacted arg), or a quarantine event."
Log "  - try a deny live: ask Claude to run 'rm -rf /tmp/x', 'sudo ...', or 'git push' -"
Log "    the kernel refuses it before the shell sees it, while 'ls'/'cat' run."
Log "  - capability floor: examples\dogfood-claude-policy.json (see README.md)."
Log "handing off to the launcher (flags passed through) ..."

# Forward every argument verbatim to the shipped launcher.
& $launcher @args
exit $LASTEXITCODE
