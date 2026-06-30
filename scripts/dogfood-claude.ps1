<#
.SYNOPSIS
  dogfood-claude.ps1 - ONE command to use fak as a product on Windows: spin up a
  local model, put the fak kernel in front of it as a NATIVE Anthropic Messages
  server, and point the real Claude Code CLI at it. Every tool call Claude proposes
  is adjudicated by the kernel (dropped / grammar-repaired) before Claude sees it.

    Claude Code  --/v1/messages-->  fak serve (the kernel)  --/v1/chat/...-->  local model
       (harness)  <----- SSE ------  adjudicates every tool   <-------------    (transformers shim)

  This is the Windows-native twin of scripts/dogfood-claude.sh. Differences that
  make it work on a CPU-only Windows host out of the box:
    * Backend defaults to the in-tree transformers `shim` (no ollama dependency).
    * Model defaults to SmolLM2-135M-Instruct - ~85x faster than Qwen-1.5B on CPU
      (a real Claude Code turn lands in seconds, not minutes). Point FAK_DOGFOOD_MODEL
      at a larger tool-capable model for real work (and raise FAK_PLANNER_TIMEOUT_S).
    * Windows paths, PowerShell process management, and port auto-bump (so it does
      not collide with another session already on :8080).

.PARAMETER (positional)
  (none)            interactive Claude Code on the local model
  -Kernel           OPT-IN: use fak's OWN in-kernel pure-Go forward (--gguf), no Python
                    shim / no proxy engine. Alias for FAK_DOGFOOD_BACKEND=gguf; composes
                    with --probe/--smoke/--print-env (e.g. `-Kernel --probe "..."`).
  --probe "<text>"  ONE headless live turn (witnessable proof), then exit
  --smoke           curl the wire end-to-end (no model needed), then exit
  --print-env       print the env lines for your own `claude` invocation
  --list-accounts   show the account switcher's roster, then exit
  --install         copy `fak.exe` + `fak-dogfood.cmd` / `claude-glm-gcp.cmd` shims onto PATH, then exit
  --help            this help

.NOTES
  Knobs (env):
    FAK_DOGFOOD_PORT       fak serve port                 (default 8080, auto-bumped if busy)
    FAK_DOGFOOD_SHIM_PORT  transformers shim port         (default 8190, auto-bumped if busy)
    FAK_DOGFOOD_MODEL      served model id                (default SmolLM2-135M for shim; qwen2.5-coder:7b for ollama; qwen2.5-7b-q8 for gguf; empty for anthropic)
    FAK_DOGFOOD_CTX        ollama context window          (default 32768; baked via a derived num_ctx model so the ~25K Claude Code prompt is not truncated; 0 disables)
    FAK_DOGFOOD_PRESET     glm-gcp                        (auto from the invoked name claude-glm-gcp)
                             glm-gcp = front GLM-5.2 served on the GCP node (scripts/gcp-glm-serve.sh)
                             via the openai backend. Set FAK_GLM_GCP_BASE_URL to its /v1 (a Tailscale
                             host, or a localhost SSH/IAP tunnel; default http://127.0.0.1:8200/v1).
    FAK_GLM_GCP_BASE_URL   glm-gcp preset's GLM-5.2 /v1 base URL   (default http://127.0.0.1:8200/v1)
    FAK_DOGFOOD_BASE_URL   openai backend upstream /v1 base URL    (overrides the preset's URL)
    FAK_DOGFOOD_BACKEND    shim | ollama | openai | gguf | anthropic   (default shim)
                             openai = a remote OpenAI-compatible /v1 (e.g. GLM-5.2 on GCP); fak
                             proxies straight to it. Needs FAK_DOGFOOD_BASE_URL (or the preset's URL).
                             ollama = a coding-capable local model (default qwen2.5-coder:7b)
                             auto-pulled and served with a 32K context. The usable local
                             path: Claude Code -> fak (adjudicates) -> ollama. Needs the
                             ollama CLI on PATH or the AMD AI-Bundle install.
                             gguf = fak's OWN in-kernel pure-Go forward (the -Kernel alias):
                             Claude Code -> fak serve --gguf (fak's decode, NO Python, NO
                             proxy) -> back. The path that proves agentic work on fak's own
                             kernel. Loads FAK_DOGFOOD_GGUF; requires a tokenizer-bearing
                             GGUF (Qwen2.5 GGUFs embed one); CPU prefill is slow so the
                             timeouts auto-raise to 900s. Asserts /healthz planner=inkernel.
                             anthropic = front the REAL Claude API (api.anthropic.com):
                             Claude Code -> fak (adjudicates) -> real Claude. Your own
                             key + real model tiers flow through; cache_control survives
                             byte-for-byte. Override the upstream with FAK_DOGFOOD_BASE_URL.
    FAK_DOGFOOD_GGUF       gguf backend: local .gguf to load  (default <home>\.cache\fak-models\gguf\Qwen2.5-7B-Instruct-Q8_0.gguf)
    FAK_DOGFOOD_ACCOUNT    account tag for the switcher    (default: isolated .claude-faklocal)
    FAK_DOGFOOD_BINDIR     --install target dir on PATH    (default: <home>\bin)
    FAK_PLANNER_TIMEOUT_S  upstream model round-trip cap   (default 60; raise for big CPU models)
    FAK_PYTHON             python executable               (default: python)

  It cannot damage your normal `claude`: every wiring env var is set only for the
  child `claude` this script spawns (PowerShell child processes inherit the
  process env, not your shell profile), and CLAUDE_CONFIG_DIR points at an isolated
  .claude-faklocal account, never your default ~/.claude.
#>
$ErrorActionPreference = 'Stop'

# ---- parse mode ------------------------------------------------------------
# -Kernel / --kernel is an alias for FAK_DOGFOOD_BACKEND=gguf (fak's own in-kernel
# forward). Pre-scan and strip it so it composes with --probe/--smoke/--print-env, e.g.
# `dogfood-claude.ps1 -Kernel --probe "..."`. Leaves the default (shim) untouched when absent.
$argv = @()
foreach ($a in $args) {
  if ($a -eq '-Kernel' -or $a -eq '--kernel') { $env:FAK_DOGFOOD_BACKEND = 'gguf' }
  else { $argv += $a }
}
$args = $argv

$Mode = 'run'
$ProbePrompt = 'Reply with exactly the word: pong'
$RunArgs = @()
if ($args.Count -ge 1) {
  switch ($args[0]) {
    '--probe'         { $Mode = 'probe'; if ($args.Count -ge 2) { $ProbePrompt = [string]$args[1] } }
    '--smoke'         { $Mode = 'smoke' }
    '--print-env'     { $Mode = 'print-env' }
    '--list-accounts' { $Mode = 'list-accounts' }
    '--install'       { $Mode = 'install' }
    '--help'          { $Mode = 'help' }
    '-h'              { $Mode = 'help' }
    default           { $RunArgs = $args }   # interactive: pass everything through to claude
  }
}

# ---- locate the repo (this script lives in fak/scripts/) -------------------
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$FakDir = (Resolve-Path (Join-Path $ScriptDir '..')).Path
# The Go module is the repository root (AGENTS.md). The kernel binary and the account
# switcher live under the repo's OWN tools/ dir — tools/ is a CHILD of $FakDir, not a
# sibling — so $Root == $FakDir. (A previous version set $Root to $FakDir\.. — one level
# ABOVE the repo — so the build silently wrote fak.exe into, and read account-switcher files
# from, an unrelated SIBLING tools\ dir outside the repo, clobbering whatever lived there.)
$Root = $FakDir

# ---- knobs -----------------------------------------------------------------
$Port      = if ($env:FAK_DOGFOOD_PORT)      { [int]$env:FAK_DOGFOOD_PORT }      else { 8080 }
$ShimPort  = if ($env:FAK_DOGFOOD_SHIM_PORT) { [int]$env:FAK_DOGFOOD_SHIM_PORT } else { 8190 }
# ---- preset (FAK_DOGFOOD_PRESET) -------------------------------------------
# A preset is a named bundle of (backend, base URL, model) defaults, selected by env or
# by the installed launcher name via the .cmd shim. claude-glm-gcp => the glm-gcp preset:
# point fak's openai backend at GLM-5.2 served on the GCP node (scripts/gcp-glm-serve.sh).
# An explicit FAK_DOGFOOD_BACKEND / _MODEL / _BASE_URL still overrides the preset below.
$Preset           = $env:FAK_DOGFOOD_PRESET
$PresetBackend    = ''
$PresetBaseUrl    = ''
$PresetModel      = ''
$PresetApiKeyEnv  = ''   # env var holding the upstream bearer token (authenticated remotes)
if ($Preset) {
  switch ($Preset) {
    'glm-gcp' {
      $PresetBackend = 'openai'
      $PresetBaseUrl = if ($env:FAK_GLM_GCP_BASE_URL) { $env:FAK_GLM_GCP_BASE_URL } else { 'http://127.0.0.1:8200/v1' }
      $PresetModel   = if ($env:FAK_GLM_GCP_MODEL)    { $env:FAK_GLM_GCP_MODEL }    else { 'glm-5.2' }
    }
    'mac' {
      # Point fak at the always-on Mac node (node-macos-a) running fak serve over Tailscale.
      # FAK_MAC_GATEWAY and FAK_GATEWAY_KEY are the canonical env vars for the Mac node;
      # FAK_MAC_MODEL names the currently-served model (set on the node and re-exported here).
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_MAC_GATEWAY)  { $env:FAK_MAC_GATEWAY }  else { Die 'FAK_DOGFOOD_PRESET=mac requires FAK_MAC_GATEWAY=http://<tailscale-ip>:8080' }
      $PresetModel     = if ($env:FAK_MAC_MODEL)    { $env:FAK_MAC_MODEL }    else { 'lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M' }
      $PresetApiKeyEnv = 'FAK_GATEWAY_KEY'
    }
    default { Die "unknown FAK_DOGFOOD_PRESET=$Preset (want glm-gcp|mac)" }
  }
}

$Backend   = if ($env:FAK_DOGFOOD_BACKEND)   { $env:FAK_DOGFOOD_BACKEND }   elseif ($PresetBackend) { $PresetBackend } else { 'shim' }
# The 'gguf' (kernel) backend is the OPT-IN sibling: it runs `fak serve --gguf` — fak's
# OWN pure-Go in-kernel forward, NO Python shim, NO proxy engine, NO --base-url. This is
# the path that proves Claude Code doing agentic work against fak's own kernel. It is OFF
# by default (default stays 'shim'); set FAK_DOGFOOD_BACKEND=gguf or pass -Kernel.
$KernelBackend = ($Backend -eq 'gguf')
# The 'openai' backend fronts a REMOTE OpenAI-compatible /v1 (e.g. GLM-5.2 on the GCP
# node) — fak proxies straight to it, no local model. The base URL comes from the preset
# or FAK_DOGFOOD_BASE_URL; the model is resolved from /models when not pinned.
$OpenaiBackend    = ($Backend -eq 'openai')
$OpenaiBaseUrl    = if ($env:FAK_DOGFOOD_BASE_URL)      { $env:FAK_DOGFOOD_BASE_URL }      else { $PresetBaseUrl }
$OpenaiApiKeyEnv  = if ($env:FAK_DOGFOOD_API_KEY_ENV)   { $env:FAK_DOGFOOD_API_KEY_ENV }   elseif ($PresetApiKeyEnv) { $PresetApiKeyEnv } else { '' }
# The 'anthropic' upstream fronts the REAL Claude API — Claude Code keeps its own
# real model tiers (claude-opus-4-8, etc.), so the single-model override is OFF and
# the default 'model' is empty. Local backends still map every tier onto one model.
$AnthropicUpstream = ($Backend -eq 'anthropic')
$DefaultModel = if ($AnthropicUpstream) { '' } elseif ($KernelBackend) { 'qwen2.5-7b-q8' } elseif ($OpenaiBackend) { $PresetModel } elseif ($Backend -eq 'ollama') { 'qwen2.5-coder:7b' } else { 'HuggingFaceTB/SmolLM2-135M-Instruct' }
$Model     = if ($env:FAK_DOGFOOD_MODEL)     { $env:FAK_DOGFOOD_MODEL }     else { $DefaultModel }
$Account   = if ($env:FAK_DOGFOOD_ACCOUNT)   { $env:FAK_DOGFOOD_ACCOUNT }   else { 'faklocal' }
$UserHome  = if ($env:FLEET_USER_HOME)       { $env:FLEET_USER_HOME }       else { $env:USERPROFILE }
# Kernel backend: the local GGUF the in-kernel forward loads, and the bearer secret the
# kernel requires (must equal ANTHROPIC_API_KEY so Claude Code's x-api-key authenticates).
$Gguf      = if ($env:FAK_DOGFOOD_GGUF)       { $env:FAK_DOGFOOD_GGUF }       else { Join-Path $UserHome '.cache\fak-models\gguf\Qwen2.5-7B-Instruct-Q8_0.gguf' }
$KeyEnv    = 'FAK_DOGFOOD_KEY'
$Python    = if ($env:FAK_PYTHON)            { $env:FAK_PYTHON }            else { 'python' }
$Bin       = Join-Path $Root 'tools\.bin\fak.exe'
# The native account switcher (`fak fleet-accounts`) globs FLEET_USER_HOME/.claude*.
$env:FLEET_USER_HOME = $UserHome

function Log  { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Cyan }
function Warn { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Yellow }
function Die  { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Red; exit 1 }

# Build the kernel binary into $out, resilient to a transiently-broken shared trunk.
# This is a live multi-session tree: a peer can have an uncommitted, half-written edit
# in the working tree that doesn't compile yet (e.g. a new strconv use before the import
# lands). A naive `go build` of the working tree would then dead-end every adopter on
# someone else's WIP. So: try the working tree first (the normal, fast path); if that
# fails, fall back to building the LAST COMMITTED trunk (HEAD) — which is by definition
# the peer-clean shared state — from a throwaway `git archive` checkout. The fallback is
# an honest, current binary (the committed trunk), never a stale prebuilt artifact. Set
# FAK_DOGFOOD_NO_HEAD_FALLBACK=1 to refuse the fallback and fail hard instead (CI/strict).
function Build-FakBinary {
  param([Parameter(Mandatory)] [string]$out)
  New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
  Push-Location $FakDir
  try {
    & go build -o $out ./cmd/fak 2>&1 | ForEach-Object { Write-Host $_ }
    if ($LASTEXITCODE -eq 0) { return }
  } finally { Pop-Location }

  if ($env:FAK_DOGFOOD_NO_HEAD_FALLBACK -in @('1','true','yes')) {
    Die "go build failed (working tree) and FAK_DOGFOOD_NO_HEAD_FALLBACK is set"
  }
  Warn "working-tree build failed - a peer's uncommitted edit likely doesn't compile yet."
  Warn "falling back to the last committed trunk (HEAD) so dogfood still works."
  $head = Join-Path ([System.IO.Path]::GetTempPath()) ("fak-dogfood-head-" + [System.Guid]::NewGuid().ToString('N'))
  New-Item -ItemType Directory -Force $head | Out-Null
  $tarball = "$head.tar"
  # Extract with Windows' own System32\tar.exe (bsdtar) — it handles drive-letter paths
  # natively; the Git/MSYS tar on PATH reads a `C:` path as a remote host and dies. Fall
  # back to whatever `tar` is on PATH only if the system one is somehow absent.
  $systar = Join-Path $env:SystemRoot 'System32\tar.exe'
  if (-not (Test-Path $systar)) { $systar = 'tar' }
  try {
    Push-Location $FakDir
    # `git archive -o <file>` writes the tar itself (no fragile native-to-native pipe).
    try { & git archive --format=tar -o $tarball HEAD; if ($LASTEXITCODE -ne 0) { Die "git archive HEAD failed - cannot build the committed trunk" } }
    finally { Pop-Location }
    & $systar -x -f $tarball -C $head; if ($LASTEXITCODE -ne 0) { Die "could not extract the committed-trunk archive" }
    Push-Location $head
    try {
      & go build -o $out ./cmd/fak 2>&1 | ForEach-Object { Write-Host $_ }
      if ($LASTEXITCODE -ne 0) { Die "even the committed trunk (HEAD) failed to build - this is a real break, not a peer's WIP" }
    } finally { Pop-Location }
    Log "built fak from the committed trunk (HEAD); your working-tree edit was skipped."
  } finally {
    Remove-Item -Recurse -Force $head -ErrorAction SilentlyContinue
    Remove-Item -Force $tarball -ErrorAction SilentlyContinue
  }
}

function Ensure-FakBinary {
  if (-not $Bin.ToLowerInvariant().StartsWith($FakDir.ToLowerInvariant())) {
    Die "refusing to build outside the repo: $Bin (expected under $FakDir)"
  }
  if (-not (Test-Path $Bin)) {
    Log "building fak -> $Bin"
    Build-FakBinary -out $Bin
  }
}

function Test-PortFree { param([int]$p)
  -not (Get-NetTCPConnection -LocalPort $p -State Listen -ErrorAction SilentlyContinue)
}
function Get-UsablePort { param([int]$p)
  for ($i = 0; $i -lt 50; $i++) { if (Test-PortFree ($p + $i)) { return ($p + $i) } }
  Die "no free port near $p"
}
function Wait-Url { param([string]$url, [int]$timeoutSec = 120)
  for ($i = 0; $i -lt ($timeoutSec * 2); $i++) {
    try { Invoke-WebRequest -Uri $url -TimeoutSec 2 -UseBasicParsing | Out-Null; return $true } catch { Start-Sleep -Milliseconds 500 }
  }
  return $false
}

# ---- openai backend: discover a remote OpenAI-compatible /v1 ----------------
# The glm-gcp preset (and any FAK_DOGFOOD_BACKEND=openai) fronts a REMOTE /v1 — GLM-5.2
# on the GCP node, reached over Tailscale or a localhost tunnel. These two helpers mirror
# the bash twin's normalize_openai_base_url / first_openai_model_from_models: confirm the
# endpoint answers /models (so we never wire a dead upstream) and pick a served model id.
function Get-JsonOrNull { param([string]$url, [hashtable]$headers = @{})
  try { return (Invoke-WebRequest -Uri $url -Headers $headers -TimeoutSec 5 -UseBasicParsing).Content } catch { return $null }
}
function Resolve-OpenAiBaseUrl { param([string]$raw, [hashtable]$authHeaders = @{})
  $raw = ([string]$raw).TrimEnd('/')
  if (-not $raw) { return $null }
  # First try /healthz (auth-free fak probe, works even when /models requires a bearer).
  $baseForHealthz = if ($raw -match '/v1$') { $raw -replace '/v1$','' } else { $raw }
  if (Get-JsonOrNull "$baseForHealthz/healthz") {
    # fak serve confirmed up; determine the canonical /v1 base.
    if ($raw -match '/v1$') { return $raw }
    return "$raw/v1"
  }
  # Fallback: try /v1/models or /models with optional auth header.
  if ($raw -match '/v1$') {
    if (Get-JsonOrNull "$raw/models" $authHeaders) { return $raw }
  } else {
    if (Get-JsonOrNull "$raw/v1/models" $authHeaders) { return "$raw/v1" }
  }
  if (Get-JsonOrNull "$raw/models" $authHeaders) { return $raw }
  return $null
}
function Get-FirstOpenAiModel { param([string]$url, [hashtable]$headers = @{})
  $body = Get-JsonOrNull $url $headers
  if (-not $body) { return $null }
  try {
    $doc = $body | ConvertFrom-Json
    $rows = if ($doc.data) { $doc.data } elseif ($doc.models) { $doc.models } else { @() }
    foreach ($row in $rows) {
      $id = if ($row.id) { $row.id } elseif ($row.name) { $row.name } elseif ($row.model) { $row.model } else { $null }
      if ($id) { return [string]$id }
    }
  } catch { }
  return $null
}

# ---- help / list-accounts: no stack needed ---------------------------------
if ($Mode -eq 'help') {
  Get-Help $MyInvocation.MyCommand.Path -Detailed
  exit 0
}
if ($Mode -eq 'list-accounts') {
  Ensure-FakBinary
  & $Bin fleet-accounts list
  exit $LASTEXITCODE
}
if ($Mode -eq 'install') {
  # Windows twin of the bash `--install`: put launchers on PATH so you can run the
  # dogfood + the repo CLI from any directory. Windows symlinks need elevation/dev-mode,
  # so (per the install decision) we COPY the built fak.exe and write a .cmd SHIM for the
  # dogfood launcher (a launcher can't be "copied" like an exe; the shim always runs the
  # in-tree script). Idempotent: re-running refreshes the fak.exe copy (it goes stale
  # until then) and rewrites the shim.
  $BinDir = if ($env:FAK_DOGFOOD_BINDIR) { $env:FAK_DOGFOOD_BINDIR } else { Join-Path $UserHome 'bin' }
  if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Force $BinDir | Out-Null }
  if (-not (Test-Path $BinDir)) { Die "could not create bin dir: $BinDir (set FAK_DOGFOOD_BINDIR)" }

  # Build + copy the repo CLI as fak.exe.
  Log "building fak -> $Bin"
  Build-FakBinary -out $Bin
  Copy-Item -Force $Bin (Join-Path $BinDir 'fak.exe')

  # Write the fak-dogfood.cmd shim -> this script, by absolute path.
  $self = $MyInvocation.MyCommand.Path
  $shim = Join-Path $BinDir 'fak-dogfood.cmd'
  $shimBody = "@powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" %*`r`n"
  [System.IO.File]::WriteAllText($shim, $shimBody, (New-Object System.Text.ASCIIEncoding))

  # Write the claude-glm-gcp.cmd preset shim: same script, with FAK_DOGFOOD_PRESET=glm-gcp
  # pinned for the child only (a .cmd's `set` is local to its own cmd.exe instance).
  $glmShim = Join-Path $BinDir 'claude-glm-gcp.cmd'
  $glmShimBody = "@set FAK_DOGFOOD_PRESET=glm-gcp`r`n@powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" %*`r`n"
  [System.IO.File]::WriteAllText($glmShim, $glmShimBody, (New-Object System.Text.ASCIIEncoding))

  # Write claude-mac.cmd: same script, preset=mac (fak → Mac fak serve → Qwen3.6-27B).
  # Uses FAK_MAC_GATEWAY (required; set to http://<tailscale-ip>:8080), FAK_GATEWAY_KEY, and FAK_MAC_MODEL
  # to route Claude Code through the always-on Mac node without a subscription.
  $macShim = Join-Path $BinDir 'claude-mac.cmd'
  $macShimBody = "@set FAK_DOGFOOD_PRESET=mac`r`n@powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" %*`r`n"
  [System.IO.File]::WriteAllText($macShim, $macShimBody, (New-Object System.Text.ASCIIEncoding))

  Log "installed: $(Join-Path $BinDir 'fak.exe')  (copied; re-run --install to refresh)"
  Log "installed: $shim  -> $self"
  Log "installed: $glmShim  -> $self (preset glm-gcp)"
  Log "installed: $macShim  -> $self (preset mac: fak -> Mac fak serve -> Qwen3.6-27B)"
  $onPath = (($env:PATH -split ';') | ForEach-Object { $_.TrimEnd('\') }) -contains $BinDir.TrimEnd('\')
  if ($onPath) {
    Log "ready - run ``fak-dogfood --smoke`` or ``fak serve --help`` from anywhere"
  } else {
    Log "add to PATH (current user), then reopen your shell:"
    Log "  setx PATH `"`$env:PATH;$BinDir`""
  }
  exit 0
}

$script:children = @()
function Stop-Children {
  foreach ($p in $script:children) {
    if ($p -and -not $p.HasExited) { try { Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue } catch {} }
  }
}

try {
  # ---- build the kernel binary ---------------------------------------------
  # Durability guard: never build outside the repo. If $Bin ever resolves above the
  # module root again, refuse rather than polluting (or clobbering) an external dir.
  if (-not $Bin.ToLowerInvariant().StartsWith($FakDir.ToLowerInvariant())) { Die "refusing to build outside the repo: $Bin (expected under $FakDir)" }
  Log "building fak -> $Bin"
  Build-FakBinary -out $Bin

  # ---- bring up the local model backend (skipped for --smoke) --------------
  # The 'anthropic' upstream needs no local model: fak proxies straight to the real
  # Claude API. Set the upstream URL unconditionally (even for --smoke, so the wire
  # smoke exercises the same provider path) and skip the shim/ollama bring-up.
  $BaseUrl = ''
  $Provider = 'openai'
  if ($AnthropicUpstream) {
    $Provider = 'anthropic'
    $BaseUrl  = if ($env:FAK_DOGFOOD_BASE_URL) { $env:FAK_DOGFOOD_BASE_URL } else { 'https://api.anthropic.com' }
    Log "upstream = REAL Claude API ($BaseUrl) - fak adjudicates every tool call on your live turns"
  }
  if ($KernelBackend) {
    # fak's OWN in-kernel forward: no shim, no proxy, no --base-url. `fak serve --gguf`
    # loads the GGUF resident and serves /v1/messages with fak's pure-Go decode. Verify the
    # weights exist before we try to bind (the eager load happens before the listener does).
    if (-not (Test-Path $Gguf)) { Die "gguf not found: $Gguf  (set FAK_DOGFOOD_GGUF to a local .gguf, e.g. a Qwen2.5 Q8)" }
    Log "kernel backend: fak's OWN in-kernel forward over $Gguf (no shim, no proxy)"
  }
  if ($Mode -ne 'smoke' -and -not $AnthropicUpstream -and -not $KernelBackend) {
    if ($Backend -eq 'shim') {
      if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) { Die "python ('$Python') not found - install Python 3 or set FAK_PYTHON" }
      $ShimPort = Get-UsablePort $ShimPort
      Log "starting transformers shim ($Model) on :$ShimPort"
      $shimOut = Join-Path $env:TEMP 'fak-dogfood-shim.out.log'
      $shimErr = Join-Path $env:TEMP 'fak-dogfood-shim.err.log'
      $shim = Start-Process -FilePath $Python `
        -ArgumentList @((Join-Path $FakDir 'experiments\agent-live\local_shim.py'), '--model', $Model, '--port', $ShimPort) `
        -PassThru -WindowStyle Hidden -RedirectStandardOutput $shimOut -RedirectStandardError $shimErr
      $script:children += $shim
      if (-not (Wait-Url "http://127.0.0.1:$ShimPort/v1/models" 180)) {
        Get-Content $shimErr -Tail 20 -ErrorAction SilentlyContinue | Write-Host
        Die "shim did not come up on :$ShimPort"
      }
      $BaseUrl = "http://127.0.0.1:$ShimPort/v1"
    }
    elseif ($Backend -eq 'ollama') {
      $oll = if ($env:OLLAMA_HOST) { $env:OLLAMA_HOST } else { '127.0.0.1:11434' }
      if (-not (Wait-Url "http://$oll/api/tags" 3)) { Die "ollama not reachable at $oll (start 'ollama serve' or use FAK_DOGFOOD_BACKEND=shim)" }
      $BaseUrl = "http://$oll/v1"

      # Resolve the ollama CLI (PATH first, then the AMD AI-Bundle install) so we can
      # auto-pull the model and bake a large context. With neither, fall back to just
      # serving whatever is already loaded (the API path still works).
      $ollamaExe = (Get-Command ollama -ErrorAction SilentlyContinue).Source
      if (-not $ollamaExe) {
        $amd = Join-Path $env:LOCALAPPDATA 'AMD\AI_Bundle\Ollama\ollama.exe'
        if (Test-Path $amd) { $ollamaExe = $amd }
      }

      if ($ollamaExe) {
        # ollama's CLI writes progress to stderr; under this script's $ErrorActionPreference
        # = 'Stop', merging native stderr into the pipeline (2>&1) is treated as a terminating
        # error. ollamaCli runs each call with all streams sent to a log file (*> file), so
        # stderr never trips Stop, and tolerates a non-zero exit (a derive/pull hiccup must
        # degrade to "serve the base model", not kill the launcher).
        $ollLog = Join-Path $env:TEMP 'fak-dogfood-ollama-cli.log'
        function ollamaCli { param([string[]]$cliArgs) try { & $ollamaExe @cliArgs *> $ollLog } catch {} ; Get-Content $ollLog -Raw -ErrorAction SilentlyContinue }

        # Auto-pull the chosen model if it isn't installed (one-time download). The default
        # is qwen2.5-coder:7b (set in $DefaultModel) — a coding-capable model that actually
        # drives Claude Code, unlike the 135M CPU shim default.
        $baseModel = $Model -replace '-fakctx\d+$',''
        if (-not ((ollamaCli @('list')) -match [regex]::Escape($baseModel))) {
          Log "pulling ollama model $baseModel (one-time download)"
          ollamaCli @('pull', $baseModel) | Out-Null
        }

        # Claude Code sends a ~25K-token agent prompt; ollama defaults to a 4K context and
        # SILENTLY TRUNCATES it, which breaks the turn. OLLAMA_CONTEXT_LENGTH and a per-request
        # num_ctx on the OpenAI /v1 endpoint are both ignored, so bake the context into a
        # derived model via a Modelfile (PARAMETER num_ctx) — the only endpoint-independent
        # way. Skip when the operator already pinned a large-ctx tag or set FAK_DOGFOOD_CTX=0.
        $ctx = if ($env:FAK_DOGFOOD_CTX) { [int]$env:FAK_DOGFOOD_CTX } else { 32768 }
        if ($ctx -gt 0 -and $Model -notmatch '-fakctx\d+$') {
          $ctxModel = "$baseModel-fakctx$ctx"
          if (-not ((ollamaCli @('list')) -match [regex]::Escape($ctxModel))) {
            $mf = Join-Path $env:TEMP "fak-$($baseModel -replace '[:/]','_')-ctx.Modelfile"
            "FROM $baseModel`nPARAMETER num_ctx $ctx" | Set-Content -Path $mf -Encoding ASCII
            Log "deriving $ctxModel (num_ctx=$ctx) so the large Claude Code prompt is not truncated"
            ollamaCli @('create', $ctxModel, '-f', $mf) | Out-Null
          }
          $Model = $ctxModel
        }
      } else {
        Log "ollama CLI not found (PATH or AMD AI-Bundle) - serving the already-loaded model; set FAK_DOGFOOD_CTX or pre-create a num_ctx model if the prompt is truncated."
      }
    }
    elseif ($Backend -eq 'openai') {
      # Remote OpenAI-compatible upstream (e.g. GLM-5.2 on the GCP node, or the Mac fak serve).
      # Validate it is reachable and resolve the served model; fak proxies straight to it (no
      # local model to start). For authenticated endpoints (FAK_DOGFOOD_API_KEY_ENV / preset key),
      # pass the bearer in probes — /healthz is checked first (auth-free) as the canonical
      # fak-serve liveness signal, then /v1/models with the key as fallback.
      if (-not $OpenaiBaseUrl) { Die "FAK_DOGFOOD_BACKEND=openai needs a base URL - set FAK_DOGFOOD_BASE_URL (or FAK_MAC_GATEWAY / FAK_GLM_GCP_BASE_URL for the presets)" }
      $openaiKey = if ($OpenaiApiKeyEnv) { [System.Environment]::GetEnvironmentVariable($OpenaiApiKeyEnv) } else { '' }
      $openaiAuthHeaders = if ($openaiKey) { @{ 'x-api-key' = $openaiKey } } else { @{} }
      $BaseUrl = Resolve-OpenAiBaseUrl $OpenaiBaseUrl $openaiAuthHeaders
      if (-not $BaseUrl) { Die "OpenAI-compatible endpoint not reachable at $OpenaiBaseUrl.`n       Check that the remote fak serve is up (curl $OpenaiBaseUrl/healthz) and the Tailscale / tunnel path is open." }
      if (-not $Model) { $Model = Get-FirstOpenAiModel "$BaseUrl/models" $openaiAuthHeaders }
      if (-not $Model) { Die "could not resolve a model from $BaseUrl/models; set FAK_DOGFOOD_MODEL" }
      Log "using OpenAI-compatible backend $BaseUrl (model: $Model)"
    }
    else { Die "unknown FAK_DOGFOOD_BACKEND=$Backend (want shim|ollama|openai)" }
  }

  # ---- start fak serve (the kernel) in front of the model ------------------
  # Claude Code's prompt is large; even SmolLM2-135M can take >60s/turn on a CPU
  # box, which would trip the planner's 60s and the gateway's 90s WriteTimeout and
  # cut the turn off with a 502. Default both generous (300s) unless the operator
  # already set them. fak serve inherits these from this process env via Start-Process.
  # The in-kernel 7B Q8 CPU forward is much slower than the SmolLM shim — a real Claude
  # Code turn (a multi-thousand-token tool prompt prefilled on CPU) can take minutes — so
  # the kernel arm raises the floor higher (900s) to avoid a 502 mid-turn.
  # The remote openai backend (GLM-5.2 on GCP) is a big model with a long prefill — give
  # its turns the same generous 900s floor as the slow in-kernel CPU forward.
  $TimeoutFloor = if ($KernelBackend -or $OpenaiBackend) { '900' } else { '300' }
  if (-not $env:FAK_PLANNER_TIMEOUT_S)    { $env:FAK_PLANNER_TIMEOUT_S = $TimeoutFloor }
  if (-not $env:FAK_HTTP_WRITE_TIMEOUT_S) { $env:FAK_HTTP_WRITE_TIMEOUT_S = $TimeoutFloor }
  # Claude Code's OWN client request timeout must outlast a slow CPU turn, or the harness
  # aborts the request before fak's forward finishes prefilling the multi-thousand-token
  # prompt — fatal exactly on the kernel (gguf) arm, whose pure-Go CPU forward is the
  # slowest backend. Mirror the bash twin (scripts/dogfood-claude.sh): derive API_TIMEOUT_MS
  # from the planner timeout (seconds -> ms) unless the operator already pinned it. The
  # gateway also emits SSE pings during a slow generation, so the raised ceiling, not an
  # idle disconnect, is what governs.
  if (-not $env:API_TIMEOUT_MS -and [int]$env:FAK_PLANNER_TIMEOUT_S -gt 0) {
    $env:API_TIMEOUT_MS = [string]([int]$env:FAK_PLANNER_TIMEOUT_S * 1000)
  }
  $Port = Get-UsablePort $Port
  $serveArgs = @('serve', '--addr', "127.0.0.1:$Port", '--provider', $Provider)
  if ($Model)   { $serveArgs += @('--model', $Model) }
  if ($BaseUrl) { $serveArgs += @('--base-url', $BaseUrl) }
  if ($KernelBackend) {
    # The required-key value MUST equal ANTHROPIC_API_KEY (Claude Code sends it as
    # x-api-key, which the gateway authenticates against this secret). Set it in THIS
    # process env BEFORE Start-Process so `fak serve` inherits it (an unset/empty
    # required-key env makes serve exit 2). Pin both to one source below in the wiring block.
    $kernelKey = if ($env:ANTHROPIC_API_KEY) { $env:ANTHROPIC_API_KEY } else { 'fak-local-dogfood' }
    Set-Item -Path "Env:$KeyEnv" -Value $kernelKey
    $serveArgs += @('--gguf', $Gguf, '--require-key-env', $KeyEnv)
  }
  if ($OpenaiApiKeyEnv -and $OpenaiBackend) {
    # Authenticated OpenAI-compatible upstream (e.g. the Mac fak serve with --require-key-env).
    # Pass the env var name through to fak serve so it can set the Authorization / x-api-key
    # header on every upstream call; the value itself stays in the env and is never logged.
    $serveArgs += @('--api-key-env', $OpenaiApiKeyEnv)
  }
  if ($env:FAK_DOGFOOD_POLICY) { $serveArgs += @('--policy', $env:FAK_DOGFOOD_POLICY) }
  Log "starting kernel: fak $($serveArgs -join ' ')"
  $serveOut = Join-Path $env:TEMP 'fak-dogfood-serve.out.log'
  $serveErr = Join-Path $env:TEMP 'fak-dogfood-serve.err.log'
  $serve = Start-Process -FilePath $Bin -ArgumentList $serveArgs -PassThru -WindowStyle Hidden `
    -RedirectStandardOutput $serveOut -RedirectStandardError $serveErr
  $script:children += $serve
  # The kernel arm loads the GGUF eagerly BEFORE the listener binds, so /healthz is
  # unreachable until the (slow, CPU) load completes — give it far longer than the shim.
  $HealthTimeout = if ($KernelBackend) { 600 } else { 30 }
  if (-not (Wait-Url "http://127.0.0.1:$Port/healthz" $HealthTimeout)) {
    Get-Content $serveErr -Tail 20 -ErrorAction SilentlyContinue | Write-Host
    Die "fak serve did not become healthy on :$Port"
  }
  $hzBody = (Invoke-WebRequest "http://127.0.0.1:$Port/healthz" -UseBasicParsing).Content
  Log "kernel healthy: $hzBody"
  if ($KernelBackend) {
    # A GGUF lacking an embedded BPE tokenizer would SILENTLY drop to the offline
    # MockPlanner (scripted text), not fak's forward. Turn that into a hard failure: the
    # whole point of this backend is the in-kernel forward, so refuse anything else.
    $planner = ($hzBody | ConvertFrom-Json).planner
    if ($planner -ne 'inkernel') {
      Die "kernel backend expected planner=inkernel but /healthz reports planner='$planner'; the GGUF may lack an embedded tokenizer or fell back to the mock planner"
    }
    Log "verified: planner=inkernel; fak in-kernel forward is serving the wire"
  }

  # ---- resolve the account dir through the native switcher -----------------
  # ONE call to the switcher's canonical front door: resolve --faklocal-ok pins the
  # named tag, or synthesizes the isolated .claude-faklocal dogfood account for the
  # faklocal default, returning the config dir in a single flat record.
  function Resolve-AccountDir { param([string]$tag)
    Ensure-FakBinary
    $r = (& $Bin fleet-accounts resolve --faklocal-ok --account $tag | ConvertFrom-Json)
    if (-not $r -or -not $r.ok) {
      $why = if ($r -and $r.reason) { $r.reason } else { 'resolve failed' }
      Die "account tag '$tag' not resolved: $why - run with --list-accounts"
    }
    return $r.config_dir
  }
  $AccountDir = Resolve-AccountDir $Account

  # ---- wire the Claude Code harness to the kernel --------------------------
  # Claude Code appends /v1/messages itself, so the base URL must NOT include /v1.
  $env:ANTHROPIC_BASE_URL = "http://127.0.0.1:$Port"
  $env:CLAUDE_CONFIG_DIR  = $AccountDir
  if ($AnthropicUpstream) {
    # Real Claude API upstream: fak is a TRANSPARENT hop. Leave the model tiers and
    # the API key ALONE — Claude Code uses its real models (claude-opus-4-8, ...) and
    # its own credential, which fak forwards verbatim to api.anthropic.com (the inbound
    # x-api-key is passed through; cache_control survives byte-for-byte). Do NOT pin a
    # placeholder key or remap tiers — that would defeat the point.
    Log "Claude Code wired (REAL Claude API through fak):"
    Log "  ANTHROPIC_BASE_URL = $($env:ANTHROPIC_BASE_URL)   (native /v1/messages on the kernel)"
    Log "  CLAUDE_CONFIG_DIR  = $($env:CLAUDE_CONFIG_DIR)    (account: $Account)"
    Log "  upstream           = $BaseUrl   (real models + your own key flow through)"
  } else {
    $env:ANTHROPIC_API_KEY  = if ($env:ANTHROPIC_API_KEY) { $env:ANTHROPIC_API_KEY } else { 'fak-local-dogfood' }
    # Map every Claude Code model tier onto our single local model.
    $env:ANTHROPIC_MODEL = $Model
    $env:ANTHROPIC_DEFAULT_OPUS_MODEL = $Model
    $env:ANTHROPIC_DEFAULT_SONNET_MODEL = $Model
    $env:ANTHROPIC_DEFAULT_HAIKU_MODEL = $Model
    $env:ANTHROPIC_SMALL_FAST_MODEL = $Model
    Log "Claude Code wired:"
    Log "  ANTHROPIC_BASE_URL = $($env:ANTHROPIC_BASE_URL)   (native /v1/messages on the kernel)"
    Log "  CLAUDE_CONFIG_DIR  = $($env:CLAUDE_CONFIG_DIR)    (account: $Account)"
    Log "  model (all tiers)  = $Model"
    if ($KernelBackend) {
      Log "  forward            = fak's OWN in-kernel pure-Go decode over $Gguf  (NO Python, NO proxy)"
    }
  }
  if (-not $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC) { $env:CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC = '1' }

  switch ($Mode) {
    'smoke' {
      # Use Invoke-WebRequest, NOT curl.exe: when PowerShell passes a JSON `-d`
      # argument to a native exe it strips the inner double-quotes, so curl would
      # send unquoted keys and the kernel rejects the body. Native PS HTTP avoids it.
      # Send the bearer as x-api-key (Claude Code's header) so the smoke also passes when
      # the backend requires auth (the gguf/kernel arm does; the shim default does not).
      $smokeHdr = @{ 'x-api-key' = $env:ANTHROPIC_API_KEY }
      $wire = {
        param([string]$body)
        try { (Invoke-WebRequest -Uri "$($env:ANTHROPIC_BASE_URL)/v1/messages" -Method Post -ContentType 'application/json' -Headers $smokeHdr -Body $body -UseBasicParsing).Content }
        catch {
          # PowerShell 7's error record carries the response body directly; older builds
          # expose a .Response stream. Try the modern path first so a non-2xx (e.g. a 401
          # surfaces a readable body instead of crashing on a missing GetResponseStream.
          if ($_.ErrorDetails -and $_.ErrorDetails.Message) { $_.ErrorDetails.Message }
          else { "[wire error] $($_.Exception.Message)" }
        }
      }
      Log "wire smoke - POST /v1/messages (buffered):"
      $b = '{"model":"claude-smoke","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}'
      (& $wire $b) | ForEach-Object { "    $_" }
      Log "wire smoke - POST /v1/messages (stream:true) event names:"
      $bs = '{"model":"claude-smoke","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}'
      ((& $wire $bs) -split "`n") | Select-String '^event:' | ForEach-Object { "    $($_.Line.Trim())" }
      Log "smoke ok"
    }
    'print-env' {
      @"
`$env:ANTHROPIC_BASE_URL="$($env:ANTHROPIC_BASE_URL)"
`$env:ANTHROPIC_API_KEY="$($env:ANTHROPIC_API_KEY)"
`$env:CLAUDE_CONFIG_DIR="$($env:CLAUDE_CONFIG_DIR)"
`$env:ANTHROPIC_MODEL="$Model"
`$env:ANTHROPIC_DEFAULT_OPUS_MODEL="$Model"
`$env:ANTHROPIC_DEFAULT_SONNET_MODEL="$Model"
`$env:ANTHROPIC_DEFAULT_HAIKU_MODEL="$Model"
`$env:ANTHROPIC_SMALL_FAST_MODEL="$Model"
"@ | Write-Host
    }
    'probe' {
      $out = Join-Path $FakDir 'experiments\agent-live\dogfood-claude-probe-win.json'
      Log "one live turn through Claude Code (headless): `"$ProbePrompt`""
      $perr = Join-Path $env:TEMP 'fak-dogfood-claude.err.log'
      # Relax $ErrorActionPreference around the call: in PS 5.1, `2>` redirecting a
      # native command's stderr under 'Stop' turns claude's harmless "no stdin in 3s"
      # warning into a TERMINATING NativeCommandError that aborts the probe. Pipe
      # empty stdin ($null) so claude gets immediate EOF instead of waiting 3s.
      $prevEAP = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      try {
        $null | & claude -p $ProbePrompt --output-format json --dangerously-skip-permissions 1>$out 2>$perr
        $rc = $LASTEXITCODE
      } finally { $ErrorActionPreference = $prevEAP }
      if ($rc -ne 0) { Get-Content $perr -Tail 20 -ErrorAction SilentlyContinue | Write-Host; Die "claude probe exited $rc" }
      # PS 5.1's `1>` redirect writes UTF-16; normalize the committed witness to UTF-8.
      if (Test-Path $out) {
        $raw = Get-Content $out -Raw
        [System.IO.File]::WriteAllText($out, $raw, (New-Object System.Text.UTF8Encoding($false)))
      }
      Log "transcript -> $out"
      try {
        $d = Get-Content $out -Raw | ConvertFrom-Json
        $res = if ($d.result) { $d.result } else { $d }
        Log ("result: " + ([string]$res).Substring(0, [math]::Min(400, ([string]$res).Length)))
        Log ("subtype=$($d.subtype)  is_error=$($d.is_error)  turns=$($d.num_turns)  model=$(@($d.modelUsage.PSObject.Properties.Name) -join ',')")
      } catch { Log "(probe JSON written; parse skipped)" }
      Log "live turn ok - Claude Code completed a turn against the local kernel-fronted model"
    }
    'run' {
      Log "launching interactive Claude Code (Ctrl-C to stop; kernel shuts down on exit)"
      & claude --dangerously-skip-permissions @RunArgs
    }
  }
}
finally {
  Stop-Children
}
