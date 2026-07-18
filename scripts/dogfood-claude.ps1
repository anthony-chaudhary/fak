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
  --install         copy `fak.exe` + write a .cmd shim for each GRADUATED launcher onto PATH, then exit (graduation rule, #3034)
  --install-all     same, but also write shims for the opt-in (not-yet-graduated) external-provider launchers
  --graduation      print each launcher's graduation verdict (graduated => installed by --install; opt-in => stays a wrapper), then exit
  --help            this help

.NOTES
  Knobs (env):
    FAK_DOGFOOD_PORT       fak serve port                 (default 8080, auto-bumped if busy)
    FAK_DOGFOOD_SHIM_PORT  transformers shim port         (default 8190, auto-bumped if busy)
    FAK_DOGFOOD_MODEL      served model id                (default SmolLM2-135M for shim; qwen2.5-coder:7b for ollama; qwen2.5-7b-q8 for gguf; empty for anthropic)
    FAK_DOGFOOD_CTX        ollama context window          (default 32768; baked via a derived num_ctx model so the ~25K Claude Code prompt is not truncated; 0 disables)
    FAK_DOGFOOD_PRESET     qwen36-local | glm-gcp | glm-zai | gemini-gcp | groq-qwen36 | groq-compound | nim-kimi | nim-deepseek-v4-pro | mac    (auto from the installed preset shims)
                             qwen36-local = front a local Qwen3.6 OpenAI-compatible server at
                             http://127.0.0.1:8131/v1 with the LM Studio Q4_K_M model id
                             glm-gcp = front GLM-5.2 served on the GCP node (scripts/gcp-glm-serve.sh)
                             via the openai backend. Set FAK_GLM_GCP_BASE_URL to its /v1 (a Tailscale
                             host, or a localhost SSH/IAP tunnel; default http://127.0.0.1:8200/v1).
                             glm-zai = front the HOSTED Z.ai coding-plan GLM-5.2 (api.z.ai/api/coding/paas/v4)
                             via the openai backend; no VM, just set ZAI_API_KEY.
                             gemini-gcp = front GCP Vertex AI Gemini 3.5 Flash (its OpenAI-compat
                             endpoint) via the openai backend. Set FAK_GEMINI_GCP_PROJECT (or
                             GCP_PROJECT) + FAK_GEMINI_GCP_KEY (a GCP access token); no VM needed.
                             groq-qwen36 = front Groq's OpenAI-compatible Qwen3.6-27B endpoint via
                             the openai backend. Set FAK_GROQ_API_KEY first.
                             groq-compound = front Groq's lower-quality Compound endpoint via
                             the openai backend. Set FAK_GROQ_API_KEY first.
                             nim-kimi = front NVIDIA NIM's OpenAI-compatible Kimi K2.6 endpoint via
                             the openai backend. Set NVIDIA_API_KEY first.
                             nim-deepseek-v4-pro = front NVIDIA NIM's OpenAI-compatible DeepSeek V4 Pro
                             endpoint via the openai backend. Set NVIDIA_API_KEY first.
    FAK_GLM_GCP_BASE_URL   glm-gcp preset's GLM-5.2 /v1 base URL   (default http://127.0.0.1:8200/v1)
    ZAI_API_KEY            glm-zai preset's Z.ai coding-plan bearer (Authorization: Bearer)
    FAK_ZAI_BASE_URL       glm-zai preset's Z.ai coding-plan root  (default https://api.z.ai/api/coding/paas/v4)
    FAK_ZAI_MODEL          glm-zai preset's upstream model id      (default zai-coding-plan/glm-5.2)
    FAK_ZAI_API_KEY_ENV    env var holding the Z.ai bearer         (default ZAI_API_KEY)
    FAK_GEMINI_GCP_PROJECT / FAK_GEMINI_GCP_LOCATION  gemini-gcp preset's GCP project + region (location
                             default global) — builds the Vertex OpenAI-compat base when
                             FAK_GEMINI_GCP_BASE_URL is unset; project falls back to GCP_PROJECT
    FAK_GEMINI_GCP_MODEL   gemini-gcp preset's upstream model id  (default google/gemini-3.5-flash)
    FAK_GEMINI_GCP_KEY     bearer token for Vertex — a GCP access token, e.g. `gcloud auth print-access-token`
    FAK_GROQ_API_KEY       Groq API key for the groq-qwen36 preset
    FAK_GROQ_BASE_URL      Groq OpenAI-compatible base (default https://api.groq.com/openai/v1)
    FAK_GROQ_MODEL         Groq Qwen model slug (default qwen/qwen3.6-27b)
    FAK_DOGFOOD_PROVIDER_MAX_TOKENS optional upstream max_tokens cap (groq-compound default 8192)
    FAK_NIM_KIMI_API_KEY_ENV env var holding the NVIDIA NIM bearer (default NVIDIA_API_KEY)
    FAK_NIM_KIMI_BASE_URL  NVIDIA NIM OpenAI-compatible base (default https://integrate.api.nvidia.com/v1)
    FAK_NIM_KIMI_MODEL     NVIDIA NIM Kimi model slug (default moonshotai/kimi-k2.6)
    FAK_NIM_DEEPSEEK_API_KEY_ENV env var holding the NVIDIA NIM bearer (default NVIDIA_API_KEY)
    FAK_NIM_DEEPSEEK_BASE_URL NVIDIA NIM OpenAI-compatible base (default https://integrate.api.nvidia.com/v1)
    FAK_NIM_DEEPSEEK_MODEL  NVIDIA NIM DeepSeek model slug (default deepseek-ai/deepseek-v4-pro)
    FAK_MAC_GATEWAY        mac preset's fak serve gateway base URL, e.g. http://<macbook-ip>:8080
    FAK_MAC_MODEL          mac preset's served model id (default lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M)
    FAK_GATEWAY_KEY        bearer used by the mac preset when the gateway requires --require-key-env
    FAK_MAC_SSH_HOST       mac preset ssh host for fetching ~/.fak-gateway-key when FAK_GATEWAY_KEY is empty
    FAK_MAC_SSH_KEY        optional ssh identity for the mac preset key fetch
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
    FAK_DOGFOOD_PROBE_OUT  --probe transcript path         (default experiments\agent-live\dogfood-claude-probe-win.json)
    FAK_DOGFOOD_PROBE_ERR  --probe Claude stderr path      (default %TEMP%\fak-dogfood-claude.err.log, or <PROBE_OUT>.err.log when PROBE_OUT is set)
    FAK_DOGFOOD_PROBE_TOOLS --probe Claude Code tools      (default empty = disabled; set default or Read,Write,Edit,Bash for tool probes)
    FAK_DOGFOOD_PROBE_ALLOWED_TOOLS optional --probe allow-list for Claude Code tools
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
$InstallAll = [bool]$env:FAK_DOGFOOD_INSTALL_ALL
if ($args.Count -ge 1) {
  switch ($args[0]) {
    '--probe'         { $Mode = 'probe'; if ($args.Count -ge 2) { $ProbePrompt = [string]$args[1] } }
    '--smoke'         { $Mode = 'smoke' }
    '--print-env'     { $Mode = 'print-env' }
    '--list-accounts' { $Mode = 'list-accounts' }
    '--install'       { $Mode = 'install' }
    '--install-all'   { $Mode = 'install'; $InstallAll = $true }
    '--graduation'    { $Mode = 'graduation' }
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
$PresetExtraBody  = ''
$PresetOpenAIToolMessagesAsText = ''
$PresetProviderMaxTokens = ''
if ($Preset) {
  switch ($Preset) {
    'qwen36' {
      $Preset = 'qwen36-local'
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = 'http://127.0.0.1:8131/v1'
      $PresetModel     = 'lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M'
      $PresetExtraBody = '{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'
    }
    'qwen36-local' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = 'http://127.0.0.1:8131/v1'
      $PresetModel     = 'lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M'
      $PresetExtraBody = '{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'
    }
    'glm-gcp' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_GLM_GCP_BASE_URL) { $env:FAK_GLM_GCP_BASE_URL } else { 'http://127.0.0.1:8200/v1' }
      $PresetModel     = if ($env:FAK_GLM_GCP_MODEL)    { $env:FAK_GLM_GCP_MODEL }    else { 'glm-5.2' }
      $PresetExtraBody = '{"chat_template_kwargs":{"enable_thinking":false}}'
    }
    'glm-zai' {
      # HOSTED Z.ai coding-plan GLM-5.2 via the openai backend — the claude-glm-gcp wire pointed
      # at Z.ai's MANAGED endpoint (no VM, just a ZAI_API_KEY). The base is Z.ai's OpenAI-compatible
      # coding-plan root (fak appends /chat/completions); the served id is the provider-scoped
      # zai-coding-plan/glm-5.2, forwarded verbatim. Mirrors tools/claude_agent_chat.py --glm.
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_ZAI_BASE_URL) { $env:FAK_ZAI_BASE_URL } else { 'https://api.z.ai/api/coding/paas/v4' }
      $PresetModel     = if ($env:FAK_ZAI_MODEL)    { $env:FAK_ZAI_MODEL }    else { 'zai-coding-plan/glm-5.2' }
      $PresetApiKeyEnv = if ($env:FAK_ZAI_API_KEY_ENV) { $env:FAK_ZAI_API_KEY_ENV } else { 'ZAI_API_KEY' }
    }
    'gemini-gcp' {
      # Gemini 3.5 Flash served by GCP Vertex AI (its OpenAI-compat endpoint) fronted by fak's
      # openai backend — the claude-glm-gcp wire pointed at a Google-MANAGED model, so there is
      # no VM to stand up, only GCP creds. The bearer is a short-lived GCP access token
      # (`gcloud auth print-access-token`) in FAK_GEMINI_GCP_KEY, read as Authorization: Bearer
      # like the mac preset's gateway key. The upstream id is google/gemini-3.5-flash (a Vertex
      # tier-2 lightweight seat). fak's openai backend appends /chat/completions to the base, so
      # the base is the Vertex .../endpoints/openapi.
      $PresetBackend   = 'openai'
      if ($env:FAK_GEMINI_GCP_BASE_URL) {
        $PresetBaseUrl = $env:FAK_GEMINI_GCP_BASE_URL
      } else {
        $gemProj = if ($env:FAK_GEMINI_GCP_PROJECT) { $env:FAK_GEMINI_GCP_PROJECT } elseif ($env:GCP_PROJECT) { $env:GCP_PROJECT } else { Die 'FAK_DOGFOOD_PRESET=gemini-gcp needs FAK_GEMINI_GCP_BASE_URL, or a project via FAK_GEMINI_GCP_PROJECT / GCP_PROJECT (the Vertex OpenAI-compat base is built from project+location)' }
        $gemLoc  = if ($env:FAK_GEMINI_GCP_LOCATION) { $env:FAK_GEMINI_GCP_LOCATION } else { 'global' }
        $gemHost = if ($gemLoc -eq 'global') { 'aiplatform.googleapis.com' } else { "${gemLoc}-aiplatform.googleapis.com" }
        $PresetBaseUrl = "https://${gemHost}/v1beta1/projects/${gemProj}/locations/${gemLoc}/endpoints/openapi"
      }
      $PresetModel     = if ($env:FAK_GEMINI_GCP_MODEL) { $env:FAK_GEMINI_GCP_MODEL } else { 'google/gemini-3.5-flash' }
      $PresetApiKeyEnv = 'FAK_GEMINI_GCP_KEY'
      # Vertex accepts Gemini's first tool call through the OpenAI-compatible route, but
      # rejects the follow-up tool-result role unless fak serializes tool messages as text.
      $PresetOpenAIToolMessagesAsText = '1'
    }
    'groq' {
      $Preset = 'groq-qwen36'
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_GROQ_BASE_URL) { $env:FAK_GROQ_BASE_URL } else { 'https://api.groq.com/openai/v1' }
      $PresetModel     = if ($env:FAK_GROQ_MODEL)    { $env:FAK_GROQ_MODEL }    else { 'qwen/qwen3.6-27b' }
      $PresetApiKeyEnv = 'FAK_GROQ_API_KEY'
    }
    'groq-qwen36' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_GROQ_BASE_URL) { $env:FAK_GROQ_BASE_URL } else { 'https://api.groq.com/openai/v1' }
      $PresetModel     = if ($env:FAK_GROQ_MODEL)    { $env:FAK_GROQ_MODEL }    else { 'qwen/qwen3.6-27b' }
      $PresetApiKeyEnv = 'FAK_GROQ_API_KEY'
    }
    'groq-compound' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_GROQ_BASE_URL) { $env:FAK_GROQ_BASE_URL } else { 'https://api.groq.com/openai/v1' }
      $PresetModel     = if ($env:FAK_GROQ_MODEL)    { $env:FAK_GROQ_MODEL }    else { 'groq/compound' }
      $PresetApiKeyEnv = 'FAK_GROQ_API_KEY'
      $PresetProviderMaxTokens = '8192'
    }
    'mac' {
      # Point fak at the always-on Mac node (node-macos-a) running fak serve over Tailscale.
      # FAK_MAC_GATEWAY and FAK_GATEWAY_KEY are the canonical env vars for the Mac node;
      # FAK_MAC_MODEL names the currently-served model (set on the node and re-exported here).
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_MAC_GATEWAY)  { $env:FAK_MAC_GATEWAY }  else { Die 'FAK_DOGFOOD_PRESET=mac requires FAK_MAC_GATEWAY=http://<tailscale-ip>:8080' }
      $PresetModel     = if ($env:FAK_MAC_MODEL)    { $env:FAK_MAC_MODEL }    else { 'lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M' }
      $PresetApiKeyEnv = 'FAK_GATEWAY_KEY'
      $PresetOpenAIToolMessagesAsText = '1'
    }
    'nim-kimi' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_NIM_KIMI_BASE_URL) { $env:FAK_NIM_KIMI_BASE_URL } else { 'https://integrate.api.nvidia.com/v1' }
      $PresetModel     = if ($env:FAK_NIM_KIMI_MODEL)    { $env:FAK_NIM_KIMI_MODEL }    else { 'moonshotai/kimi-k2.6' }
      $PresetApiKeyEnv = if ($env:FAK_NIM_KIMI_API_KEY_ENV) { $env:FAK_NIM_KIMI_API_KEY_ENV } else { 'NVIDIA_API_KEY' }
    }
    'nim-deepseek-v4-pro' {
      $PresetBackend   = 'openai'
      $PresetBaseUrl   = if ($env:FAK_NIM_DEEPSEEK_BASE_URL) { $env:FAK_NIM_DEEPSEEK_BASE_URL } else { 'https://integrate.api.nvidia.com/v1' }
      $PresetModel     = if ($env:FAK_NIM_DEEPSEEK_MODEL)    { $env:FAK_NIM_DEEPSEEK_MODEL }    else { 'deepseek-ai/deepseek-v4-pro' }
      $PresetApiKeyEnv = if ($env:FAK_NIM_DEEPSEEK_API_KEY_ENV) { $env:FAK_NIM_DEEPSEEK_API_KEY_ENV } else { 'NVIDIA_API_KEY' }
    }
    default { Die "unknown FAK_DOGFOOD_PRESET=$Preset (want qwen36-local|glm-gcp|glm-zai|gemini-gcp|groq-qwen36|groq-compound|nim-kimi|nim-deepseek-v4-pro|mac)" }
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
$OpenAIToolMessagesAsText = if ($env:FAK_DOGFOOD_OPENAI_TOOL_MESSAGES_AS_TEXT) { $env:FAK_DOGFOOD_OPENAI_TOOL_MESSAGES_AS_TEXT } elseif ($PresetOpenAIToolMessagesAsText) { $PresetOpenAIToolMessagesAsText } else { '' }
$ProviderMaxTokens = if ($env:FAK_DOGFOOD_PROVIDER_MAX_TOKENS) { $env:FAK_DOGFOOD_PROVIDER_MAX_TOKENS } elseif ($PresetProviderMaxTokens) { $PresetProviderMaxTokens } else { '' }
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
$Policy    = if ($env:FAK_DOGFOOD_POLICY)    { $env:FAK_DOGFOOD_POLICY }    else { Join-Path $FakDir 'examples\dogfood-claude-policy.json' }
# The native account switcher (`fak fleet-accounts`) globs FLEET_USER_HOME/.claude*.
$env:FLEET_USER_HOME = $UserHome

function Log  { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Cyan }
function Warn { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Yellow }
function Die  { param([string]$m) Write-Host "[dogfood] $m" -ForegroundColor Red; exit 1 }

function Ensure-MacGatewayKey {
  param([string]$EnvName)
  if ($Preset -ne 'mac') { return }
  if (-not $EnvName) { return }
  if ([System.Environment]::GetEnvironmentVariable($EnvName)) { return }
  if (-not $env:FAK_MAC_SSH_HOST) { return }

  $sshKey = if ($env:FAK_MAC_SSH_KEY) { $env:FAK_MAC_SSH_KEY } else { Join-Path $UserHome '.ssh\id_ed25519_prod_to_laptop' }
  $sshArgs = @('-n', '-o', 'BatchMode=yes', '-o', 'NumberOfPasswordPrompts=0', '-o', 'StrictHostKeyChecking=accept-new', '-o', 'ConnectTimeout=5', '-o', 'ConnectionAttempts=1')
  if ($sshKey) { $sshArgs += @('-i', $sshKey) }
  $sshArgs += @($env:FAK_MAC_SSH_HOST, 'cat ~/.fak-gateway-key')

  Log "fetching Mac gateway bearer from configured Mac SSH host"
  $tmpOut = [System.IO.Path]::GetTempFileName()
  $tmpErr = [System.IO.Path]::GetTempFileName()
  try {
    $proc = Start-Process -FilePath 'ssh' -ArgumentList $sshArgs -PassThru -WindowStyle Hidden -RedirectStandardOutput $tmpOut -RedirectStandardError $tmpErr
    if (-not $proc.WaitForExit(10000)) {
      Stop-Process -Id $proc.Id -Force -ErrorAction SilentlyContinue
      Die "fetch gateway key over ssh timed out after 10s; set $EnvName directly, or set FAK_MAC_SSH_HOST / FAK_MAC_SSH_KEY for the Mac preset"
    }
    $proc.Refresh()
    $secretOut = Get-Content -LiteralPath $tmpOut -Raw -ErrorAction SilentlyContinue
    $errOut = Get-Content -LiteralPath $tmpErr -Raw -ErrorAction SilentlyContinue
    $secret = ([string]$secretOut).Trim()
    if ($proc.ExitCode -ne 0 -and -not $secret) {
      $detail = if ($null -ne $errOut) { ([string]$errOut).Trim() } else { '' }
      if (-not $detail) { $detail = "ssh exited $($proc.ExitCode) without stderr" }
      Die "fetch gateway key over ssh failed: $detail`nset $EnvName directly, or set FAK_MAC_SSH_HOST / FAK_MAC_SSH_KEY for the Mac preset"
    }
  } finally {
    Remove-Item -LiteralPath $tmpOut,$tmpErr -Force -ErrorAction SilentlyContinue
  }
  if (-not $secret) { Die "fetch gateway key over ssh: empty key" }
  Set-Item -Path "Env:$EnvName" -Value $secret
}

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
  $headCommit = (& git -C $FakDir rev-parse HEAD 2>$null).Trim()
  if ($LASTEXITCODE -ne 0 -or $headCommit -notmatch '^[0-9a-fA-F]{40}$') { Die "could not resolve HEAD for committed-trunk build provenance" }
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
      & go build -ldflags "-X github.com/anthony-chaudhary/fak/internal/appversion.BuildCommit=$headCommit" -o $out ./cmd/fak 2>&1 | ForEach-Object { Write-Host $_ }
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
function Ensure-TimeoutFloor { param([string]$name, [int]$floor)
  $raw = [System.Environment]::GetEnvironmentVariable($name)
  $parsed = 0
  if (-not [int]::TryParse($raw, [ref]$parsed) -or $parsed -lt $floor) {
    Set-Item -Path "Env:$name" -Value ([string]$floor)
  }
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

# ---- external-provider graduation rule (#3034) -----------------------------
# Windows twin of the manifest in scripts/dogfood-claude.sh. `--install` used to
# write a .cmd shim for EVERY preset unconditionally — provider-specific enthusiasm.
# Instead, a launcher is installed only if it has GRADUATED against a GENERIC,
# evidence-based bar (documented key env or keyless-local; a currently-available
# route — live route-health defers to #3035; minimum successful probes recorded
# in-repo; at least one tool-use/coding witness; known rate/expiry caveats). The two
# keyless local presets clear the bar today (committed probe transcripts + the
# bounded-wait regression, always-reachable local route); the external cloud/self-
# hosted presets stay OPT-IN — runnable via FAK_DOGFOOD_PRESET or --install-all, but
# never default-installed. This manifest is the SINGLE source of truth the install
# gate and --graduation both read.
function Get-GraduationManifest {
  @(
    [pscustomobject]@{ Launcher='fak-dogfood';          Preset='';                    Graduated=$true;  KeyEnv='-';                 Caveat='generic local shim/ollama; witnessed by committed probe + bounded-wait regression' }
    [pscustomobject]@{ Launcher='fak-qwen36-claude';    Preset='qwen36-local';        Graduated=$true;  KeyEnv='-';                 Caveat='local Qwen3.6 (127.0.0.1:8131); needs the local server up' }
    [pscustomobject]@{ Launcher='claude-glm-gcp';       Preset='glm-gcp';             Graduated=$false; KeyEnv='-';                 Caveat='self-hosted GCP GLM-5.2; route stood up per-operator, availability not yet classified (#3035)' }
    [pscustomobject]@{ Launcher='claude-glm-zai';       Preset='glm-zai';             Graduated=$false; KeyEnv='ZAI_API_KEY';       Caveat='hosted Z.ai coding-plan GLM-5.2; managed endpoint, key/route per-operator, not classified (#3035)' }
    [pscustomobject]@{ Launcher='claude-gemini-gcp';    Preset='gemini-gcp';          Graduated=$false; KeyEnv='FAK_GEMINI_GCP_KEY'; Caveat='Vertex bearer is a short-lived GCP token; route not yet classified (#3035)' }
    [pscustomobject]@{ Launcher='claude-groq-qwen36';   Preset='groq-qwen36';         Graduated=$false; KeyEnv='FAK_GROQ_API_KEY';   Caveat='Groq tier; rate-limited, route not yet classified (#3035)' }
    [pscustomobject]@{ Launcher='claude-groq-compound'; Preset='groq-compound';       Graduated=$false; KeyEnv='FAK_GROQ_API_KEY';   Caveat='Groq lower-tier compound; rate/token-capped, route not yet classified (#3035)' }
    [pscustomobject]@{ Launcher='claude-nim-kimi';      Preset='nim-kimi';            Graduated=$false; KeyEnv='NVIDIA_API_KEY';     Caveat='NVIDIA NIM Kimi K2.6 preview; small in-repo sample, key/route may expire (#3035)' }
    [pscustomobject]@{ Launcher='claude-nim-deepseek';  Preset='nim-deepseek-v4-pro'; Graduated=$false; KeyEnv='NVIDIA_API_KEY';     Caveat='NVIDIA NIM DeepSeek V4 Pro preview; small sample, key/route may expire (#3035)' }
    [pscustomobject]@{ Launcher='claude-mac';           Preset='mac';                 Graduated=$false; KeyEnv='FAK_GATEWAY_KEY';    Caveat='needs a reachable Mac fak serve gateway (FAK_MAC_GATEWAY); route per-operator, not classified (#3035)' }
  )
}
# The one decision --install and its dry-run both consume: graduated by default, or
# every launcher when $InstallAll (--install-all / FAK_DOGFOOD_INSTALL_ALL) is set.
function Get-GraduatedLaunchers {
  Get-GraduationManifest | Where-Object { $InstallAll -or $_.Graduated } | ForEach-Object { $_.Launcher }
}

if ($Mode -eq 'graduation') {
  Log "external-provider graduation (rule #3034): 'graduated' launchers are installed by --install; 'opt-in' stay wrappers"
  foreach ($m in Get-GraduationManifest) {
    $verdict = if ($m.Graduated) { 'graduated' } else { 'opt-in' }
    $preset  = if ($m.Preset) { $m.Preset } else { '-' }
    Write-Host ('{0,-10} {1,-22} preset={2,-20} key={3,-16} {4}' -f $verdict, $m.Launcher, $preset, $m.KeyEnv, $m.Caveat)
  }
  exit 0
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
  # so (per the install decision) we COPY the built fak.exe and write a .cmd SHIM for each
  # launcher (a launcher can't be "copied" like an exe; the shim always runs the in-tree
  # script). The graduation gate (#3034) means only launchers Get-GraduatedLaunchers
  # returns get a shim; opt-in ones are reported with their one-command probe recipe.
  # Idempotent: re-running refreshes the fak.exe copy and rewrites the graduated shims.
  $BinDir = if ($env:FAK_DOGFOOD_BINDIR) { $env:FAK_DOGFOOD_BINDIR } else { Join-Path $UserHome 'bin' }
  $self = $MyInvocation.MyCommand.Path
  $installSet = @(Get-GraduatedLaunchers)

  # Dry-run: report exactly what --install would create / skip, then exit BEFORE any
  # build or shim write. The offline seam the regression test drives on the .sh twin.
  if ($env:FAK_DOGFOOD_INSTALL_DRYRUN) {
    foreach ($m in Get-GraduationManifest) {
      if ($installSet -contains $m.Launcher) {
        Write-Host ("would-install: {0}" -f $m.Launcher)
      } else {
        $keyhint = if ($m.KeyEnv -eq '-') { '' } else { "$($m.KeyEnv)=... " }
        Write-Host ("opt-in-skip: {0} (run: FAK_DOGFOOD_PRESET={1} {2}{3} --probe ""say pong"")" -f $m.Launcher, $m.Preset, $keyhint, $self)
      }
    }
    exit 0
  }

  if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Force $BinDir | Out-Null }
  if (-not (Test-Path $BinDir)) { Die "could not create bin dir: $BinDir (set FAK_DOGFOOD_BINDIR)" }

  # Build + install the repo CLI as fak.exe, LOCK-SAFELY. A plain `Copy-Item -Force` over
  # an already-running fak.exe fails on Windows (the PE is locked while mapped), which
  # stranded the fresh build as a dangling `.new` and left the on-PATH copy stale until a
  # hand-run swap. Windows PERMITS *renaming* a running image (only overwrite/delete is
  # blocked), so swap by rename: stage the fresh build beside the target, rename the
  # current exe aside under a timestamped name (never collides with a still-mapped prior
  # backup), move the fresh one into place, then best-effort reclaim the parked old copy
  # (a live process may still map it; it is reclaimed on the next --install). This is what
  # makes `--install` a dependable one-command local-bin refresh that always completes.
  Log "building fak -> $Bin"
  Build-FakBinary -out $Bin
  $dest = Join-Path $BinDir 'fak.exe'
  $new  = "$dest.new"
  Copy-Item -Force $Bin $new
  # Fail-closed provenance read-back (#5211): before swapping the on-PATH exe, run BOTH the
  # freshly built binary and the staged .new copy and confirm each reports a `build:` revision
  # equal to the repo HEAD's 12-char prefix. A stale/corrupt staged copy or a future provenance
  # regression fails HARD here rather than installing an unverifiable executable.
  $wantRev = (& git -C $FakDir rev-parse --short=12 HEAD 2>$null).Trim()
  if ($LASTEXITCODE -ne 0 -or $wantRev -notmatch '^[0-9a-fA-F]{12}$') { Die "provenance read-back: could not resolve HEAD revision" }
  foreach ($cand in @($Bin, $new)) {
    $out = (& $cand version 2>$null | Out-String)
    $m = [regex]::Match($out, 'build:\s*([0-9a-fA-F]{12})')
    if (-not $m.Success) { Die "provenance read-back FAILED: $cand did not report a build: revision" }
    if ($m.Groups[1].Value -ne $wantRev) { Die "provenance read-back FAILED: $cand reports build $($m.Groups[1].Value) != HEAD $wantRev - refusing to install an unverifiable candidate" }
  }
  Log "provenance read-back OK: built + staged candidates report build:$wantRev (pre-swap)"
  if (Test-Path $dest) {
    $old = "$dest.old-$(Get-Date -Format yyyyMMddHHmmss)"
    Move-Item -Force $dest $old
    Move-Item -Force $new  $dest
    Remove-Item -Force $old -ErrorAction SilentlyContinue
  } else {
    Move-Item -Force $new $dest
  }
  Log "installed: $dest  (lock-safe swap; refreshes even while fak.exe is running)"

  # Write a .cmd shim per graduated launcher (a preset shim pins FAK_DOGFOOD_PRESET for
  # its own cmd.exe instance only; the generic fak-dogfood shim pins nothing). Opt-in
  # launchers are reported with how to run and how to force-install them.
  foreach ($m in Get-GraduationManifest) {
    $shimPath = Join-Path $BinDir ($m.Launcher + '.cmd')
    if ($installSet -contains $m.Launcher) {
      if ($m.Preset) {
        $shimBody = "@set FAK_DOGFOOD_PRESET=$($m.Preset)`r`n@powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" %*`r`n"
      } else {
        $shimBody = "@powershell -NoProfile -ExecutionPolicy Bypass -File `"$self`" %*`r`n"
      }
      [System.IO.File]::WriteAllText($shimPath, $shimBody, (New-Object System.Text.ASCIIEncoding))
      $presetLabel = if ($m.Preset) { $m.Preset } else { '-' }
      Log "installed: $shimPath  -> $self (preset $presetLabel)"
    } else {
      $keyhint = if ($m.KeyEnv -eq '-') { '' } else { "$($m.KeyEnv)=... " }
      Log ("opt-in (not installed): {0} - try: `$env:FAK_DOGFOOD_PRESET='{1}'; {2}{3} --probe 'say pong'  [{4}]; force-install with --install-all" -f $m.Launcher, $m.Preset, $keyhint, $self, $m.Caveat)
    }
  }

  $onPath = (($env:PATH -split ';') | ForEach-Object { $_.TrimEnd('\') }) -contains $BinDir.TrimEnd('\')
  if ($onPath) {
    Log "ready - run ``fak-dogfood --smoke``, ``fak-qwen36-claude --probe``, or ``fak serve --help`` from anywhere (see --graduation for the opt-in launchers)"
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
      Ensure-MacGatewayKey $OpenaiApiKeyEnv
      $openaiKey = if ($OpenaiApiKeyEnv) { [System.Environment]::GetEnvironmentVariable($OpenaiApiKeyEnv) } else { '' }
      $openaiAuthHeaders = if ($openaiKey) { @{ 'x-api-key' = $openaiKey; 'Authorization' = "Bearer $openaiKey" } } else { @{} }
      $BaseUrl = Resolve-OpenAiBaseUrl $OpenaiBaseUrl $openaiAuthHeaders
      $modelsProbeSkipped = $false
      if (-not $BaseUrl -and $Preset -eq 'gemini-gcp' -and $Model) {
        # Vertex's OpenAI-compatible Gemini endpoint exposes chat/completions but may not
        # expose /models. With a pinned model, let fak serve perform the real request.
        $BaseUrl = ([string]$OpenaiBaseUrl).TrimEnd('/')
        $modelsProbeSkipped = $true
      }
      if (-not $BaseUrl) { Die "OpenAI-compatible endpoint not reachable at $OpenaiBaseUrl.`n       Check that the remote fak serve is up (curl $OpenaiBaseUrl/healthz) and the Tailscale / tunnel path is open." }
      if (-not $Model) { $Model = Get-FirstOpenAiModel "$BaseUrl/models" $openaiAuthHeaders }
      if (-not $Model) { Die "could not resolve a model from $BaseUrl/models; set FAK_DOGFOOD_MODEL" }
      if ($modelsProbeSkipped) {
        Log "using OpenAI-compatible backend $BaseUrl (model: $Model; /models probe skipped)"
      } else {
        Log "using OpenAI-compatible backend $BaseUrl (model: $Model)"
      }
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
  $TimeoutFloor = if ($KernelBackend -or $OpenaiBackend) { 900 } else { 300 }
  Ensure-TimeoutFloor 'FAK_PLANNER_TIMEOUT_S' $TimeoutFloor
  Ensure-TimeoutFloor 'FAK_HTTP_WRITE_TIMEOUT_S' ([int]$env:FAK_PLANNER_TIMEOUT_S)
  # Claude Code's OWN client request timeout must outlast a slow CPU turn, or the harness
  # aborts the request before fak's forward finishes prefilling the multi-thousand-token
  # prompt — fatal exactly on the kernel (gguf) arm, whose pure-Go CPU forward is the
  # slowest backend. Mirror the bash twin (scripts/dogfood-claude.sh): derive API_TIMEOUT_MS
  # from the planner timeout (seconds -> ms) unless the operator already pinned it. The
  # gateway also emits SSE pings during a slow generation, so the raised ceiling, not an
  # idle disconnect, is what governs.
  $apiTimeoutFloor = [int64]$env:FAK_PLANNER_TIMEOUT_S * 1000
  $apiTimeout = 0L
  if (-not [int64]::TryParse($env:API_TIMEOUT_MS, [ref]$apiTimeout) -or $apiTimeout -lt $apiTimeoutFloor) {
    $env:API_TIMEOUT_MS = [string]$apiTimeoutFloor
  }
  if ($PresetExtraBody -and -not $env:FAK_PROVIDER_EXTRA_BODY_JSON) {
    $env:FAK_PROVIDER_EXTRA_BODY_JSON = $PresetExtraBody
  }
  if ($OpenAIToolMessagesAsText -and -not $env:FAK_OPENAI_TOOL_MESSAGES_AS_TEXT) {
    $env:FAK_OPENAI_TOOL_MESSAGES_AS_TEXT = $OpenAIToolMessagesAsText
  }
  if ($ProviderMaxTokens -and -not $env:FAK_PROVIDER_MAX_TOKENS) {
    $env:FAK_PROVIDER_MAX_TOKENS = $ProviderMaxTokens
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
  if ($Mode -ne 'smoke' -and $Policy -and $Policy -ne 'none') {
    if (-not (Test-Path $Policy)) { Die "policy manifest not found: $Policy (set FAK_DOGFOOD_POLICY=none to disable)" }
    $serveArgs += @('--policy', $Policy)
  }
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
      $out = if ($env:FAK_DOGFOOD_PROBE_OUT) { $env:FAK_DOGFOOD_PROBE_OUT } else { Join-Path $FakDir 'experiments\agent-live\dogfood-claude-probe-win.json' }
      Log "one live turn through Claude Code (headless): `"$ProbePrompt`""
      $perr = if ($env:FAK_DOGFOOD_PROBE_ERR) { $env:FAK_DOGFOOD_PROBE_ERR } elseif ($env:FAK_DOGFOOD_PROBE_OUT) { "$out.err.log" } else { Join-Path $env:TEMP 'fak-dogfood-claude.err.log' }
      New-Item -ItemType Directory -Force (Split-Path -Parent $out) | Out-Null
      New-Item -ItemType Directory -Force (Split-Path -Parent $perr) | Out-Null
      # Relax $ErrorActionPreference around the call: in PS 5.1, `2>` redirecting a
      # native command's stderr under 'Stop' turns claude's harmless "no stdin in 3s"
      # warning into a TERMINATING NativeCommandError that aborts the probe. Pipe
      # empty stdin ($null) so claude gets immediate EOF instead of waiting 3s.
      $prevEAP = $ErrorActionPreference
      $ErrorActionPreference = 'Continue'
      try {
        $probeTools = if ($env:FAK_DOGFOOD_PROBE_TOOLS) { $env:FAK_DOGFOOD_PROBE_TOOLS } else { '' }
        $claudeProbeArgs = @(
          '--bare',
          '--system-prompt', 'You are a terse assistant. Follow the user instruction exactly.',
          '-p', $ProbePrompt,
          '--output-format', 'json',
          '--dangerously-skip-permissions',
          '--safe-mode',
          '--tools', $probeTools,
          '--disable-slash-commands',
          '--no-session-persistence'
        )
        if ($env:FAK_DOGFOOD_PROBE_ALLOWED_TOOLS) {
          $claudeProbeArgs += @('--allowedTools', $env:FAK_DOGFOOD_PROBE_ALLOWED_TOOLS)
        }
        $null | & claude @claudeProbeArgs 1>$out 2>$perr
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
