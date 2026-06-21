<#
.SYNOPSIS
  Launch a headless `/goal` worker fully DETACHED from the launching shell,
  under a SWITCHER-CHOSEN account at the right model tier.

.WHY
  A `/goal` worker run inline (`claude -p ...` as a child of a tool-call shell, or
  via dispatch_worker.py's blocking subprocess.run) dies the moment that parent shell
  is reaped -- which is exactly why the first resolve-tickets run was cut off mid-loop
  with no `end_turn`. This launcher uses Start-Process to spawn claude as an INDEPENDENT
  process (its own process tree, not a child of this shell), redirects its output to a
  dated log, records the PID, and returns immediately. The worker then survives this
  session ending.

  ACCOUNT SWITCHER + TIER ROUTING (the dispatch integration): historically this script
  grabbed the ambient `claude` and launched under whatever account happened to be the
  default -- no CLAUDE_CONFIG_DIR, no availability check, no tier. A throttled or
  auth-blocked default account silently failed the dispatch. It now resolves an account
  through the SAME switcher front door every other consumer uses
  (`fleet_accounts.py resolve` -- one call returns config_dir + oauth_token + tier),
  pins `CLAUDE_CONFIG_DIR` to it, and picks the model tier by WORK KIND:

    -WorkKind engineering   -> tier 1 (max-quality frontier; the DEFAULT, unchanged)
    -WorkKind gardening     -> tier 2 (GLM/light) for maintenance/cleanup loops
    -Tier t1|t2|t3|auto     -> explicit tier override (work-kind wins if both given)

  Engineering is the default, so a plain `launch_goal_detached.ps1` keeps the old
  max-quality behavior; only an explicit gardening/maintenance dispatch drops to tier 2.
  Gardening is non-strict: if no tier-2 account is free it up-shifts to tier 1 rather
  than stalling. If NO account is available at all, the launch FAILS loudly (the whole
  point of the switcher) instead of silently running on a blocked ambient account.

.NOTES
  - The goal condition is read from the launch POINTER file (kept <4000 chars for the
    /goal cap); the worker reads the full spec from disk itself.
  - bypassPermissions is required for an unattended worker (it edits files, runs git).
  - This does NOT modify the tree or commit; it only starts a process. Stop it with
    `Stop-Process -Id <pid>` (the PID is printed and written to the .pid file).
#>
[CmdletBinding()]
param(
  [string]$PointerFile = ".claude/goal-prompts/resolve-tickets-witnessed.md",
  [string]$Workspace   = "C:\work\fleet",
  [string]$LogDir      = "C:\work\fleet\.goal-runs",
  # Work kind drives the tier: engineering (default) -> tier1, gardening -> tier2.
  # Leave empty to fall back to -Tier. See fleet_accounts.GARDENING/ENGINEERING_WORK_KINDS.
  [ValidateSet('engineering','eng','dev','feature','implementation',
               'gardening','garden','maintenance','maint','cleanup','chore','triage','')]
  [string]$WorkKind    = 'engineering',
  # Explicit tier override (only consulted when -WorkKind is empty).
  [ValidateSet('auto','t1','t2','t3','1','2','3')]
  [string]$Tier        = 'auto',
  # Pin a specific account by tag/basename instead of routing (rare; for debugging).
  [string]$Account     = '',
  # Let an engineering/tier-1 dispatch fall back to tier 2 when no tier-1 account is
  # free, rather than refusing. Off by default so engineering stays max-quality.
  [switch]$AllowTierFallback
)

$ErrorActionPreference = "Stop"
Set-Location $Workspace

if (-not (Test-Path $PointerFile)) { throw "pointer file not found: $PointerFile" }
$body = Get-Content -Raw $PointerFile
$cond = "/goal $body"
if ($cond.Length -gt 4000) { throw "goal condition is $($cond.Length) chars (>4000 cap) -- shrink the pointer" }

$claude = (Get-Command claude).Source

# --- Resolve the account + tier through the switcher (the dispatch integration) -------
# ONE call to the switcher's canonical front door (`fleet_accounts.py resolve`): pin OR
# tier/work-kind route, plus the account's oauth token, in a single flat record. Scoped
# to claude (this launches Claude Code, not opencode). Capture JSON to a temp file to
# dodge PS native-exe stdout quirks, then parse. On a refusal we FAIL -- never silently
# run ambient.
$py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'python3' }
$tmpOut = Join-Path ([System.IO.Path]::GetTempPath()) ("goal-route-{0}.json" -f ([Guid]::NewGuid().ToString('N')))

$resolveArgs = @((Join-Path $Workspace 'tools\fleet_accounts.py'), 'resolve', '--product', 'claude')
if ($Account)          { $resolveArgs += @('--account', $Account) }
elseif ($WorkKind)     { $resolveArgs += @('--work-kind', $WorkKind) }
else                   { $resolveArgs += @('--tier', $Tier) }
if ($AllowTierFallback) { $resolveArgs += '--allow-tier-fallback' }
& $py @resolveArgs > $tmpOut 2>$null
$resolveRc = $LASTEXITCODE
$r = $null
if (Test-Path $tmpOut) { try { $r = Get-Content -Raw $tmpOut | ConvertFrom-Json } catch { $r = $null }; Remove-Item $tmpOut -ErrorAction SilentlyContinue }
if (-not $r) { throw "account resolve produced no JSON (python=$py, rc=$resolveRc) -- cannot dispatch" }
if (-not $r.ok) {
  $reason = if ($r.reason) { $r.reason } else { 'no available account' }
  throw "account switcher refused dispatch: $reason -- fix the account (re-login / wait for reset) or pass -AllowTierFallback."
}
$acct      = $r
$configDir = $r.config_dir
$tierSel   = $r.selected_tier
$fellBack  = [bool]$r.fallback_used
if (-not $configDir) { throw "resolved account $($r.account) has no config dir" }

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Path $LogDir | Out-Null }
$stamp  = Get-Date -Format "yyyyMMdd-HHmmss"
$tag    = [IO.Path]::GetFileNameWithoutExtension($PointerFile)
$logOut = Join-Path $LogDir "$tag-$stamp.out.log"
$logErr = Join-Path $LogDir "$tag-$stamp.err.log"
$pidF   = Join-Path $LogDir "$tag-$stamp.pid"
$inF    = Join-Path $LogDir "$tag-$stamp.in.txt"

# Feed the goal via STDIN, never as a CLI argument. The condition text contains
# backtick-wrapped commands and `--flags` (e.g. `dos commit-audit --json`); passing it
# through Start-Process -ArgumentList lets CommandLineToArgvW re-split it and claude's own
# arg parser then chokes on a stray `--json` ("unknown option"). `claude -p` with no prompt
# arg reads the prompt from stdin, which is parse-safe. Write the prompt to a UTF-8 file
# (no BOM) and redirect it in.
[IO.File]::WriteAllText($inF, $cond, [Text.UTF8Encoding]::new($false))

# Pin the chosen account for the detached worker. CLAUDE_CONFIG_DIR is inherited by
# the Start-Process child (it copies the parent env), so the worker runs under the
# switcher-selected account, not the ambient default.
$env:CLAUDE_CONFIG_DIR = $configDir

# Serve on the credential the account actually authenticates with: the resolver already
# applied the switcher's single token rule (prefer the dir's long-lived .oauth-token over
# the interactive .credentials.json, which EXPIRES) and handed back oauth_token. Observed
# 2026-06-21: gem7/gem8 serve via their setup token while their interactive creds report
# "Not logged in" — so a launcher that pins only CLAUDE_CONFIG_DIR false-fails the worker
# on turn 1. Drop any ambient token when the account has none, so a sibling account's
# token never bleeds into this worker.
if ($r.oauth_token) { $env:CLAUDE_CODE_OAUTH_TOKEN = "$($r.oauth_token)" }
else { Remove-Item Env:CLAUDE_CODE_OAUTH_TOKEN -ErrorAction SilentlyContinue }

# Start-Process => detached child in its own process tree; -Redirect* keep the streams.
$p = Start-Process -FilePath $claude `
  -ArgumentList @("-p", "--permission-mode", "bypassPermissions") `
  -WorkingDirectory $Workspace `
  -RedirectStandardInput  $inF `
  -RedirectStandardOutput $logOut `
  -RedirectStandardError  $logErr `
  -WindowStyle Hidden `
  -PassThru

$p.Id | Out-File -FilePath $pidF -Encoding ascii

[pscustomobject]@{
  pid         = $p.Id
  account     = $acct.account
  account_tag = $acct.tag
  config_dir  = $configDir
  work_kind   = $WorkKind
  tier        = "t$tierSel"
  tier_fallback = $fellBack
  model       = $acct.model
  cond_chars  = $cond.Length
  out_log     = $logOut
  err_log     = $logErr
  pid_file    = $pidF
} | Format-List
if ($fellBack) {
  "note: requested tier had no free account; up-shifted to t$tierSel (work preserved)."
}
"DETACHED -- worker survives this session, pinned to account '$($acct.tag)' (t$tierSel). Stop with: Stop-Process -Id $($p.Id)"
