#!/usr/bin/env bash
# dogfood-claude.sh — ONE command to use fak as a product: spin up a local model,
# put the fak kernel in front of it as a NATIVE Anthropic Messages server, and point
# the real Claude Code CLI at it. Every tool call Claude proposes is adjudicated by
# the kernel (dropped / grammar-repaired) before Claude ever sees it.
#
#   ┌─────────────┐   /v1/messages    ┌──────────────────────┐   /v1/chat/...   ┌────────────┐
#   │ Claude Code │ ────────────────▶ │ fak serve (the kernel)│ ───────────────▶ │ local model│
#   │  (harness)  │ ◀──── SSE ─────── │  adjudicates tools     │ ◀────────────── │  (ollama)  │
#   └─────────────┘                    └──────────────────────┘                  └────────────┘
#
# Usage:
#   ./scripts/dogfood-claude.sh                 # interactive Claude Code on the local model
#   ./scripts/dogfood-claude.sh --kernel        # OPT-IN: fak's OWN in-kernel pure-Go --gguf forward (no shim/proxy); composes, e.g. --kernel --probe "hi"
#   ./scripts/dogfood-claude.sh --probe "hi"    # ONE headless live turn (witnessable proof), then exit
#   ./scripts/dogfood-claude.sh --smoke         # curl the wire (no model intelligence needed), then exit
#   ./scripts/dogfood-claude.sh --print-env     # print the export lines for your own `claude` invocation
#   ./scripts/dogfood-claude.sh --list-accounts # show the account switcher's roster, then exit
#   ./scripts/dogfood-claude.sh --install       # symlink `fak-dogfood`, `fak-qwen36-claude`, and `fak` onto PATH
#   fak-qwen36-claude --probe "hi"              # installed Qwen3.6 local preset
#
# Knobs (env):
#   FAK_DOGFOOD_PRESET   qwen36-local preset    (auto when invoked as fak-qwen36-claude)
#   FAK_DOGFOOD_PORT     fak serve port              (default 8080)
#   FAK_DOGFOOD_MODEL    served model id             (default: first ollama model, else qwen2.5:1.5b)
#   FAK_DOGFOOD_BACKEND  ollama | shim | openai | gguf | anthropic
#                          (default: auto - ollama if installed, else the in-tree python3
#                           shim; with neither, an actionable hint, never a dead end. Pin a
#                           value to force it and get an honest die if its deps are missing.)
#                          gguf = fak's OWN in-kernel pure-Go --gguf forward (the --kernel
#                          alias): Claude Code -> fak serve --gguf (NO shim, NO proxy) -> back.
#                          Loads FAK_DOGFOOD_GGUF; needs a tokenizer-bearing GGUF (Qwen2.5
#                          GGUFs embed one); CPU prefill is slow so the timeouts auto-raise to
#                          900s. Asserts /healthz planner=inkernel.
#                          anthropic = front the REAL Claude API (api.anthropic.com):
#                          Claude Code -> fak (adjudicates) -> real Claude. Your own
#                          key + real model tiers flow through; cache_control survives
#                          byte-for-byte. Override the upstream with FAK_DOGFOOD_BASE_URL.
#   FAK_DOGFOOD_GGUF     gguf backend: local .gguf to load (default ~/.cache/fak-models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf)
#   FAK_DOGFOOD_BASE_URL OpenAI-compatible upstream  (required for BACKEND=openai, e.g. http://127.0.0.1:8131/v1;
#                                                     optional override for BACKEND=anthropic)
#   FAK_DOGFOOD_API_KEY_ENV optional upstream API key env var for BACKEND=openai
#   FAK_DOGFOOD_TIMEOUT_S planner/write timeout      (default 300s, or 900s for BACKEND=openai|gguf)
#   FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON extra JSON merged into upstream provider requests
#   FAK_DOGFOOD_CLAUDE_DEBUG Claude Code debug filter (default api; 0/off/none disables)
#   FAK_DOGFOOD_CLAUDE_DEBUG_FILE optional Claude Code --debug-file path
#   FAK_DOGFOOD_ACCOUNT  account tag for the switcher (default: an isolated .claude-faklocal dogfood account)
#   FAK_DOGFOOD_POLICY   capability-floor manifest to enforce (default: built-in)
#   FAK_DOGFOOD_NO_ATTACH set=1 to force a fresh kernel (default: attach if one is up)
#   OLLAMA_HOST          ollama base                 (default 127.0.0.1:11434)
set -euo pipefail

# --- locate the repo (this script lives in fak/scripts/) ----------------------
# Resolve symlinks so the launcher works when invoked through a symlink on PATH
# (e.g. ~/.local/bin/fak-dogfood -> .../fak/scripts/dogfood-claude.sh). Without
# this, $SCRIPT_DIR would be the PATH dir and the repo would not be found.
INVOKED_NAME="$(basename "${BASH_SOURCE[0]}")"
SELF="${BASH_SOURCE[0]}"
while [ -L "$SELF" ]; do
  link="$(readlink "$SELF")"
  case "$link" in
    /*) SELF="$link" ;;
    *)  SELF="$(cd "$(dirname "$SELF")" && pwd)/$link" ;;
  esac
done
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
FAK_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
# The Go module is the repository root (AGENTS.md). The kernel binary and the account
# switcher live under the repo's OWN tools/ dir — tools/ is a CHILD of $FAK_DIR, not a
# sibling — so ROOT == FAK_DIR. (A previous version set ROOT to $FAK_DIR/.. — one level
# ABOVE the repo — so the build silently wrote the binary into, and read
# fleet_accounts.py from, an unrelated SIBLING tools/ dir outside the repo.)
ROOT="$FAK_DIR"
cd "$FAK_DIR"

log()  { printf '\033[36m[dogfood]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[dogfood] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[dogfood] %s\033[0m\n' "$*" >&2; exit 1; }

# --- -Kernel / --kernel alias: FAK_DOGFOOD_BACKEND=gguf (fak's OWN in-kernel forward) ---
# The Windows twin (dogfood-claude.ps1) accepts -Kernel as an alias for the gguf backend;
# mirror it here so the flag composes with --probe/--smoke/--print-env, e.g.
# `dogfood-claude.sh --kernel --probe "..."`. Pre-scan and strip it from the positional
# args, forcing the gguf backend (a pinned, explicit choice). Absent, the default
# (shim/ollama) path is untouched. The empty-array expansion guard keeps `set -u` happy
# under the bash 3.2 the macOS twin may run.
_kargv=()
for _ka in "$@"; do
  case "$_ka" in
    -Kernel|--kernel) export FAK_DOGFOOD_BACKEND=gguf ;;
    *) _kargv+=("$_ka") ;;
  esac
done
set -- "${_kargv[@]+"${_kargv[@]}"}"

# Build the kernel binary into $1, resilient to a transiently-broken shared trunk.
# This is a live multi-session tree: a peer can have an uncommitted, half-written edit
# in the working tree that doesn't compile yet (e.g. a new strconv use before the import
# lands). A naive `go build` would then dead-end every adopter on someone else's WIP. So:
# try the working tree first (the normal, fast path); on failure fall back to building the
# LAST COMMITTED trunk (HEAD) — the peer-clean shared state — from a throwaway `git archive`
# checkout. The fallback is an honest, current binary (the committed trunk), never a stale
# prebuilt. Set FAK_DOGFOOD_NO_HEAD_FALLBACK=1 to refuse the fallback and fail hard (CI/strict).
build_fak() {
  local out="$1"
  mkdir -p "$(dirname "$out")"
  if ( cd "$FAK_DIR" && go build -o "$out" ./cmd/fak ); then return 0; fi
  case "${FAK_DOGFOOD_NO_HEAD_FALLBACK:-}" in
    1|true|yes) die "go build failed (working tree) and FAK_DOGFOOD_NO_HEAD_FALLBACK is set" ;;
  esac
  warn "working-tree build failed - a peer's uncommitted edit likely doesn't compile yet."
  warn "falling back to the last committed trunk (HEAD) so dogfood still works."
  local head; head="$(mktemp -d "${TMPDIR:-/tmp}/fak-dogfood-head.XXXXXX")"
  # shellcheck disable=SC2064
  trap "rm -rf '$head'" RETURN
  ( cd "$FAK_DIR" && git archive HEAD ) | tar -x -C "$head" || die "git archive HEAD failed - cannot build the committed trunk"
  ( cd "$head" && go build -o "$out" ./cmd/fak ) || die "even the committed trunk (HEAD) failed to build - this is a real break, not a peer's WIP"
  log "built fak from the committed trunk (HEAD); your working-tree edit was skipped."
}

PRESET="${FAK_DOGFOOD_PRESET:-}"
if [ -z "$PRESET" ] && [ "$INVOKED_NAME" = "fak-qwen36-claude" ]; then
  PRESET="qwen36-local"
fi

DEFAULT_BACKEND="ollama"
DEFAULT_OPENAI_BASE_URL=""
DEFAULT_MODEL=""
DEFAULT_PROVIDER_EXTRA_BODY=""
case "$PRESET" in
  "") ;;
  qwen36|qwen36-local)
    PRESET="qwen36-local"
    DEFAULT_BACKEND="openai"
    DEFAULT_OPENAI_BASE_URL="http://127.0.0.1:8131/v1"
    DEFAULT_MODEL="lmstudio-community/Qwen3.6-27B-GGUF:Q4_K_M"
    DEFAULT_PROVIDER_EXTRA_BODY='{"top_k":20,"chat_template_kwargs":{"preserve_thinking":true}}'
    ;;
  *) die "unknown FAK_DOGFOOD_PRESET=$PRESET (want qwen36-local)" ;;
esac

PORT="${FAK_DOGFOOD_PORT:-8080}"
# BACKEND_PINNED records whether the operator EXPLICITLY chose a backend. When they did,
# we honor it and fail loud if its deps are missing (explicit intent deserves an honest
# error). When they did NOT (the common first-run case), the default backend is a
# preference, not a demand — so a missing dep should cascade to the next available local
# path rather than dead-end the adopter (see pick_auto_backend, called at backend bring-up).
if [ -n "${FAK_DOGFOOD_BACKEND:-}" ]; then BACKEND_PINNED=1; else BACKEND_PINNED=""; fi
BACKEND="${FAK_DOGFOOD_BACKEND:-$DEFAULT_BACKEND}"
# The 'gguf' (kernel) backend is the OPT-IN sibling (the --kernel alias): it runs
# `fak serve --gguf` — fak's OWN pure-Go in-kernel forward, NO shim, NO proxy engine,
# NO --base-url. The path that proves Claude Code doing agentic work on fak's own kernel.
# OFF by default; set FAK_DOGFOOD_BACKEND=gguf or pass --kernel. CPU prefill is slow, so
# the timeouts/healthz-wait auto-raise below and it asserts /healthz planner=inkernel.
if [ "$BACKEND" = "gguf" ]; then KERNEL_BACKEND=1; else KERNEL_BACKEND=""; fi
# Kernel-backend knobs: the local GGUF the in-kernel forward loads (Qwen2.5 GGUFs embed
# the BPE tokenizer the forward needs), and the env var name holding the bearer secret the
# kernel requires (pinned to ANTHROPIC_API_KEY below so Claude Code's x-api-key authenticates).
GGUF="${FAK_DOGFOOD_GGUF:-$HOME/.cache/fak-models/gguf/Qwen2.5-7B-Instruct-Q8_0.gguf}"
KEY_ENV="FAK_DOGFOOD_KEY"
# Map every Claude Code model tier onto one served id for the kernel arm (default below).
if [ -n "$KERNEL_BACKEND" ] && [ -z "$DEFAULT_MODEL" ]; then DEFAULT_MODEL="qwen2.5-7b-q8"; fi
OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:11434}"
SHIM_PORT="${FAK_DOGFOOD_SHIM_PORT:-8099}"
OPENAI_BASE_URL="${FAK_DOGFOOD_BASE_URL:-$DEFAULT_OPENAI_BASE_URL}"
UPSTREAM_API_KEY_ENV="${FAK_DOGFOOD_API_KEY_ENV:-}"
PROVIDER_EXTRA_BODY="${FAK_DOGFOOD_PROVIDER_EXTRA_BODY_JSON:-${FAK_DOGFOOD_EXTRA_BODY_JSON:-$DEFAULT_PROVIDER_EXTRA_BODY}}"
CLAUDE_DEBUG="${FAK_DOGFOOD_CLAUDE_DEBUG:-api}"
CLAUDE_DEBUG_FILE="${FAK_DOGFOOD_CLAUDE_DEBUG_FILE:-}"
BIN="$ROOT/tools/.bin/fak"
# Durability guard: never build outside the repo (see ROOT note above). If BIN ever
# resolves above the module root again, refuse rather than polluting an external dir.
case "$BIN" in "$FAK_DIR"/*) : ;; *) die "refusing to build outside the repo: $BIN (expected under $FAK_DIR)" ;; esac
# The account switcher (tools/fleet_accounts.py) globs FLEET_USER_HOME/.claude*; on
# this Mac the accounts live under $HOME, so point it there (the default is a
# Windows path). This is the alignment seam with the fleet account switcher.
export FLEET_USER_HOME="${FLEET_USER_HOME:-$HOME}"

CLAUDE_DEBUG_ARGS=()
case "$(printf '%s' "$CLAUDE_DEBUG" | tr 'A-Z' 'a-z')" in
  ""|0|false|off|none|no) ;;
  1|true|on|yes) CLAUDE_DEBUG_ARGS+=(--debug) ;;
  *) CLAUDE_DEBUG_ARGS+=(--debug "$CLAUDE_DEBUG") ;;
esac
[ -n "$CLAUDE_DEBUG_FILE" ] && CLAUDE_DEBUG_ARGS+=(--debug-file "$CLAUDE_DEBUG_FILE")

MODE="run"; PROBE_PROMPT="Reply with exactly the word: pong"
case "${1:-}" in
  --probe)         MODE="probe"; [ $# -ge 2 ] && PROBE_PROMPT="$2" ;;
  --smoke)         MODE="smoke" ;;
  --print-env)     MODE="print-env" ;;
  --list-accounts) MODE="list-accounts" ;;
  --install)       MODE="install" ;;
  --help|-h)       sed -n '2,50p' "$0"; exit 0 ;;
esac

# --- --install: put `fak-dogfood`, `fak-qwen36-claude`, and `fak` on PATH -----
# Idempotent. Picks the first writable PATH dir among ~/.local/bin, /opt/homebrew/bin,
# /usr/local/bin (or $FAK_DOGFOOD_BINDIR), symlinks this script there as generic
# `fak-dogfood` and preset `fak-qwen36-claude`, and builds+symlinks the repo CLI as
# `fak` so `fak serve ...` works from any cwd.
if [ "$MODE" = "install" ]; then
  name="fak-dogfood"
  qwen_name="fak-qwen36-claude"
  if [ -n "${FAK_DOGFOOD_BINDIR:-}" ]; then
    cands="$FAK_DOGFOOD_BINDIR"
  else
    cands="$HOME/.local/bin /opt/homebrew/bin /usr/local/bin"
  fi
  bindir=""
  for d in $cands; do
    if [ -d "$d" ] && [ -w "$d" ]; then bindir="$d"; break; fi
    if [ ! -d "$d" ] && mkdir -p "$d" 2>/dev/null; then bindir="$d"; break; fi
  done
  [ -n "$bindir" ] || die "no writable bin dir found in: $cands (set FAK_DOGFOOD_BINDIR)"
  # SCRIPT_DIR is already absolute (resolved above); make the link target absolute
  # so it doesn't dangle when invoked via a relative path like ./scripts/...
  target="$SCRIPT_DIR/$(basename "$SELF")"
  ln -sf "$target" "$bindir/$name"
  ln -sf "$target" "$bindir/$qwen_name"
  log "building fak -> $BIN"
  build_fak "$BIN"
  ln -sf "$BIN" "$bindir/fak"
  log "installed: $bindir/$name -> $target"
  log "installed: $bindir/$qwen_name -> $target"
  log "installed: $bindir/fak -> $BIN"
  case ":$PATH:" in
    *":$bindir:"*) log "ready — run \`fak serve --help\`, \`$name --probe\`, \`$qwen_name --probe\`, or \`$qwen_name\` from anywhere" ;;
    *)             log "NOTE: $bindir is not on PATH — add it: export PATH=\"$bindir:\$PATH\"" ;;
  esac
  exit 0
fi

# --- account switcher: pick the .claude config dir Claude Code runs under -----
# Reuse tools/fleet_accounts.py as the single source of truth for "what is an
# account". Default to a dedicated, isolated dogfood account so live-model
# experiments never pollute a real worker account's session history — and so it
# shows up in `fleet_accounts.py list` as a first-class switchable account.
resolve_account_dir() {
  local tag="${FAK_DOGFOOD_ACCOUNT:-faklocal}"
  # ONE call to the switcher's canonical front door: `resolve --faklocal-ok` pins the
  # named tag (or synthesizes the isolated .claude-faklocal dogfood account for the
  # 'faklocal' default), printing the config dir from a single flat record.
  local dir
  dir="$(python3 "$ROOT/tools/fleet_accounts.py" resolve --faklocal-ok --account "$tag" 2>/dev/null \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['config_dir'] if d.get('ok') else '')" 2>/dev/null || true)"
  [ -n "$dir" ] || die "account tag '$tag' not resolved — run: $0 --list-accounts"
  printf '%s' "$dir"
}

if [ "$MODE" = "list-accounts" ]; then
  python3 "$ROOT/tools/fleet_accounts.py" list
  exit 0
fi

# --- attach to an already-running kernel (the default) ------------------------
# If a dogfood kernel is already healthy on $PORT, ATTACH to it instead of
# building + starting a second one: adopt its model from /healthz, wire Claude
# Code straight to it, and leave it running on exit (we didn't start it, so we
# don't stop it). --smoke always wants its own offline-mock kernel, so it never
# attaches. Set FAK_DOGFOOD_NO_ATTACH=1 to force a fresh kernel (and fail loud
# if the port is busy, as the guard below always did).
ATTACHED=""; POLICY=""
# The kernel (gguf) backend never attaches: it must start its OWN `fak serve --gguf` so the
# planner=inkernel assertion below actually proves fak's forward is on the wire (an
# already-running kernel on the port could be a shim/proxy one). Mirror of the .ps1, which
# always starts fresh. The default (shim/ollama) attach behavior is unchanged.
if [ "$MODE" != "smoke" ] && [ -z "$KERNEL_BACKEND" ] && [ -z "${FAK_DOGFOOD_NO_ATTACH:-}" ] \
   && curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  # Tolerate optional whitespace around the JSON colon (the kernel emits compact
  # json.Marshal output, but a pretty-printer / proxy in between may add spaces).
  RUNNING_MODEL="$(curl -s "http://127.0.0.1:$PORT/healthz" 2>/dev/null \
    | sed -n 's/.*"model"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  # An operator-pinned model that disagrees with the live kernel is a real
  # conflict — fail loud rather than silently serving the wrong model.
  if [ -n "${FAK_DOGFOOD_MODEL:-}" ] && [ "$FAK_DOGFOOD_MODEL" != "$RUNNING_MODEL" ]; then
    die "port $PORT already serving model '$RUNNING_MODEL' but FAK_DOGFOOD_MODEL='$FAK_DOGFOOD_MODEL'.
         Attach to it (unset FAK_DOGFOOD_MODEL) or run a fresh kernel elsewhere:
         FAK_DOGFOOD_PORT=8090 FAK_DOGFOOD_NO_ATTACH=1 $0"
  fi
  ATTACHED=1
  MODEL="${RUNNING_MODEL:-mock}"
  log "attaching to running kernel on :$PORT (model: $MODEL) — leaving it up on exit"
fi

# --- build the kernel binary --------------------------------------------------
if [ -z "$ATTACHED" ]; then
log "building fak -> $BIN"
build_fak "$BIN"
fi

# --- bring up the local model backend ----------------------------------------
# Keep an attach-adopted MODEL (set above) intact; otherwise honor FAK_DOGFOOD_MODEL.
BASE_URL=""; MODEL="${MODEL:-${FAK_DOGFOOD_MODEL:-$DEFAULT_MODEL}}"; SHIM_PID=""
start_ollama_backend() {
  command -v ollama >/dev/null || die "FAK_DOGFOOD_BACKEND=ollama is pinned but ollama is not installed.
       Install it (https://ollama.com), or unset FAK_DOGFOOD_BACKEND to auto-pick a backend,
       or set FAK_DOGFOOD_BACKEND=shim (python3) / =anthropic (real Claude API)."
  if ! curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; then
    log "starting 'ollama serve'"
    nohup ollama serve >/tmp/fak-ollama.log 2>&1 &
    until curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null 2>&1; do sleep 1; done
  fi
  # Model selection (when FAK_DOGFOOD_MODEL is unset): prefer the LARGEST installed
  # model by on-disk size, not whatever ollama happens to list first — a bigger model
  # is the better default for agentic tool use. If the box has only tiny models (or
  # none), fall back to a capable default and pull it once.
  if [ -z "$MODEL" ]; then
    MODEL="$(largest_installed_model)"
    if [ -z "$MODEL" ] || model_is_small "$MODEL"; then
      [ -n "$MODEL" ] && log "only small model(s) installed (largest: $MODEL) — upgrading to $FALLBACK_MODEL"
      MODEL="$FALLBACK_MODEL"
    fi
  fi
  if ! ollama list 2>/dev/null | awk '{print $1}' | grep -qx "$MODEL"; then
    log "pulling model $MODEL (one-time download)"
    ollama pull "$MODEL"
  fi
  BASE_URL="http://$OLLAMA_HOST/v1"
}

# The capable default to use / pull when auto-selection finds only tiny models.
FALLBACK_MODEL="${FAK_DOGFOOD_FALLBACK_MODEL:-qwen2.5-coder:7b}"

# largest_installed_model prints the NAME of the biggest installed ollama model by
# on-disk size (normalizing the SIZE column's KB/MB/GB/TB unit to MB), or nothing.
largest_installed_model() {
  ollama list 2>/dev/null | awk '
    NR>1 && $1!="" {
      mb=$3
      if ($4=="GB") mb=$3*1024
      else if ($4=="TB") mb=$3*1024*1024
      else if ($4=="KB") mb=$3/1024
      if (mb>max) { max=mb; name=$1 }
    }
    END { if (name!="") print name }'
}

# model_is_small reports (exit 0) when a model id looks ≤ ~3B params. Tiny models
# cannot reliably drive Claude Code's large multi-tool agentic protocol — they
# intermittently emit malformed/raw tool calls. The kernel now lifts text-form
# <tool_call>s back onto the adjudication path, but the turns are still flaky.
model_is_small() {
  case "$(printf '%s' "$1" | tr 'A-Z' 'a-z')" in
    *:0.5b|*:1b|*:1.5b|*:2b|*:3b|*-0.5b-*|*-1.5b-*|*-3b-*) return 0 ;;
    *) return 1 ;;
  esac
}

warn_if_small_model() {
  if model_is_small "$1"; then
    log "WARNING: '$1' is a small model — agentic tool use will be flaky."
    log "         For a usable dogfood set e.g. FAK_DOGFOOD_MODEL=$FALLBACK_MODEL"
  fi
}
start_shim_backend() {
  command -v python3 >/dev/null || die "python3 required for the shim backend"
  [ -n "$MODEL" ] || MODEL="Qwen/Qwen2.5-1.5B-Instruct"
  log "starting local_shim.py ($MODEL) on :$SHIM_PORT"
  python3 "$FAK_DIR/experiments/agent-live/local_shim.py" --model "$MODEL" --port "$SHIM_PORT" &
  SHIM_PID=$!
  until curl -sf "http://127.0.0.1:$SHIM_PORT/v1/models" >/dev/null 2>&1; do sleep 1; done
  BASE_URL="http://127.0.0.1:$SHIM_PORT/v1"
}

curl_openai_json() {
  local url="$1"
  if [ -n "$UPSTREAM_API_KEY_ENV" ]; then
    local key="${!UPSTREAM_API_KEY_ENV:-}"
    if [ -n "$key" ]; then
      curl -sf -H "Authorization: Bearer $key" "$url"
      return
    fi
  fi
  curl -sf "$url"
}

normalize_openai_base_url() {
  local raw="${1%/}"
  [ -n "$raw" ] || return 1
  case "$raw" in
    */v1)
      if curl_openai_json "$raw/models" >/dev/null 2>&1; then
        printf '%s' "$raw"
        return 0
      fi
      ;;
    *)
      if curl_openai_json "$raw/v1/models" >/dev/null 2>&1; then
        printf '%s' "$raw/v1"
        return 0
      fi
      ;;
  esac
  if curl_openai_json "$raw/models" >/dev/null 2>&1; then
    printf '%s' "$raw"
    return 0
  fi
  return 1
}

first_openai_model_from_models() {
  python3 -c '
import json, sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(0)
rows = doc.get("data") or doc.get("models") or []
for row in rows:
    if not isinstance(row, dict):
        continue
    model = row.get("id") or row.get("name") or row.get("model")
    if model:
        print(model)
        break
'
}

start_openai_backend() {
  command -v python3 >/dev/null || die "python3 required for BACKEND=openai model discovery"
  [ -n "$OPENAI_BASE_URL" ] || die "FAK_DOGFOOD_BASE_URL is required when FAK_DOGFOOD_BACKEND=openai"
  BASE_URL="$(normalize_openai_base_url "$OPENAI_BASE_URL")" || die "OpenAI-compatible endpoint is not reachable at $OPENAI_BASE_URL (/models or /v1/models)"
  if [ -z "$MODEL" ]; then
    MODEL="$(curl_openai_json "$BASE_URL/models" | first_openai_model_from_models || true)"
  fi
  [ -n "$MODEL" ] || die "could not resolve model from $BASE_URL/models; set FAK_DOGFOOD_MODEL"
  log "using OpenAI-compatible backend $BASE_URL (model: $MODEL)"
}

# The provider wire fak serve speaks UPSTREAM. The 'anthropic' backend fronts the
# REAL Claude API: Claude Code -> fak (adjudicates) -> api.anthropic.com. fak forwards
# the inbound request bytes verbatim (cache_control survives) and the client's own
# key, so no local model and no model-tier remap — Claude Code uses its real models.
PROVIDER="openai"
if [ "$BACKEND" = "anthropic" ]; then
  PROVIDER="anthropic"
  BASE_URL="${FAK_DOGFOOD_BASE_URL:-https://api.anthropic.com}"
  MODEL=""   # real model tiers flow through; fak does not pin one
  log "upstream = REAL Claude API ($BASE_URL) — fak adjudicates every tool call on your live turns"
fi
# The 'gguf' (kernel) backend has no shim/ollama/proxy to start: `fak serve --gguf` loads
# the GGUF resident and serves /v1/messages with fak's OWN pure-Go decode (PROVIDER stays
# openai; no --base-url). Verify the weights exist before we try to bind (the eager load
# happens before the listener does); the actual bring-up is the --gguf serve invocation below.
if [ -n "$KERNEL_BACKEND" ]; then
  [ -f "$GGUF" ] || die "gguf not found: $GGUF  (set FAK_DOGFOOD_GGUF to a local .gguf, e.g. a Qwen2.5 Q8)"
  log "kernel backend: fak's OWN in-kernel forward over $GGUF (no shim, no proxy)"
fi

# --smoke proves the WIRE with zero model dependency: front the offline mock planner.
# When attached, the live kernel already owns its backend — don't start our own.
# The 'anthropic' backend has no local model to start (the upstream is the real API).
# pick_auto_backend cascades to the first LOCALLY-AVAILABLE backend when the operator did
# NOT pin one, setting BACKEND directly (no command substitution, so a dead-end can `die`
# the whole script, not just a subshell). Order: ollama (best agentic UX if installed) ->
# shim (in-tree, needs only python3). It never downgrades a PINNED choice — that path keeps
# its own honest die. If neither is available it emits an actionable hint listing every way
# forward instead of a bare "ollama not found" dead end.
pick_auto_backend() {
  if command -v ollama >/dev/null 2>&1; then BACKEND=ollama; return; fi
  if command -v python3 >/dev/null 2>&1; then
    log "ollama not found - falling back to the in-tree transformers shim (python3)."
    BACKEND=shim; return
  fi
  die "no local model backend available. Pick one:
       - install ollama (https://ollama.com)        then re-run            [best agentic UX]
       - install python3                            then re-run            [in-tree shim]
       - FAK_DOGFOOD_BACKEND=anthropic ANTHROPIC_API_KEY=... $0   [front the real Claude API]
       Or prove just the wire with no model:  $0 --smoke"
}

if [ -z "$ATTACHED" ] && [ "$MODE" != "smoke" ] && [ "$BACKEND" != "anthropic" ] && [ "$BACKEND" != "gguf" ]; then
  # An unpinned default is a preference, not a demand: resolve it to what's actually here.
  if [ -z "$BACKEND_PINNED" ] && [ "$BACKEND" = "ollama" ]; then pick_auto_backend; fi
  case "$BACKEND" in
    ollama) start_ollama_backend ;;
    shim)   start_shim_backend ;;
    openai) start_openai_backend ;;
    *) die "unknown FAK_DOGFOOD_BACKEND=$BACKEND (want ollama|shim|openai|anthropic)" ;;
  esac
  [ -n "$MODEL" ] || die "no model resolved"
  warn_if_small_model "$MODEL"
fi

# Everything from here to the health-confirm is the "start our own kernel" path;
# when ATTACHED it's all skipped (the live kernel is already healthy and adopted).
if [ -z "$ATTACHED" ]; then

# --- guard the port BEFORE launching (else we silently attach to a stale kernel) -
# We only reach this guard when NOT attaching (no kernel answered at attach time,
# or FAK_DOGFOOD_NO_ATTACH forced a fresh one). If something is on $PORT now it's
# a stale/foreign kernel — fail loud with the pid holding the port rather than
# wiring Claude Code to the WRONG kernel (possibly a different model/base-url).
if curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; then
  holder="$(lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -1)"
  die "port $PORT already serving (pid ${holder:-?}) — a previous dogfood kernel is still up.
       Stop it (kill ${holder:-<pid>}) or run on another port: FAK_DOGFOOD_PORT=8090 $0"
fi

# --- capability floor (policy) ------------------------------------------------
# Default to a sensible floor that ALLOWS the standard Claude Code tool set so
# interactive sessions actually work — with no policy the kernel default-denies
# EVERY tool and the harness can do nothing. The kernel still adjudicates every
# call: the default manifest denies destructive shell commands by argument value
# (rm -rf, sudo, git push, fork bomb, curl|sh, mkfs/dd) and write-protects the
# kernel/.git/secret paths. Override with FAK_DOGFOOD_POLICY=<path|none>.
POLICY="${FAK_DOGFOOD_POLICY:-$FAK_DIR/examples/dogfood-claude-policy.json}"

# --- start fak serve (the kernel) in front of the model -----------------------
if [ -z "${FAK_PLANNER_TIMEOUT_S:-}" ]; then
  # The in-kernel 7B Q8 CPU forward is much slower than the shim/ollama path — a real Claude
  # Code turn (a multi-thousand-token tool prompt prefilled on CPU) can take minutes — so the
  # kernel arm raises the floor to 900s (matching BACKEND=openai) to avoid a 502 mid-turn.
  # FAK_HTTP_WRITE_TIMEOUT_S and API_TIMEOUT_MS derive from it below.
  if [ "$BACKEND" = "openai" ] || [ -n "$KERNEL_BACKEND" ]; then
    export FAK_PLANNER_TIMEOUT_S="${FAK_DOGFOOD_TIMEOUT_S:-900}"
  else
    export FAK_PLANNER_TIMEOUT_S="${FAK_DOGFOOD_TIMEOUT_S:-300}"
  fi
fi
if [ -z "${FAK_HTTP_WRITE_TIMEOUT_S:-}" ]; then
  export FAK_HTTP_WRITE_TIMEOUT_S="${FAK_DOGFOOD_TIMEOUT_S:-$FAK_PLANNER_TIMEOUT_S}"
fi
if [ -n "$PROVIDER_EXTRA_BODY" ] && [ -z "${FAK_PROVIDER_EXTRA_BODY_JSON:-}" ]; then
  export FAK_PROVIDER_EXTRA_BODY_JSON="$PROVIDER_EXTRA_BODY"
fi

# --engine is the registered engine fak_syscall dispatches an ALLOWED tool call to
# (orthogonal to --model, which is the upstream chat planner). Default to the fused
# in-kernel model so a self-dispatched call exercises the kernel's own decode path;
# override with FAK_DOGFOOD_ENGINE (e.g. mock, cassette).
SERVE_ARGS=(serve --addr "127.0.0.1:$PORT" --provider "$PROVIDER" --engine "${FAK_DOGFOOD_ENGINE:-inkernel}")
# The anthropic upstream carries no fixed model (Claude Code sends real model ids
# per request); local/openai backends serve one model, with mock as the smoke floor.
if [ "$PROVIDER" != "anthropic" ]; then SERVE_ARGS+=(--model "${MODEL:-mock}"); fi
[ -n "$BASE_URL" ] && SERVE_ARGS+=(--base-url "$BASE_URL")
[ -n "$UPSTREAM_API_KEY_ENV" ] && SERVE_ARGS+=(--api-key-env "$UPSTREAM_API_KEY_ENV")
if [ -n "$KERNEL_BACKEND" ]; then
  # fak's OWN in-kernel forward: load the GGUF, NO --base-url (no proxy/shim upstream).
  # The required-key value MUST equal ANTHROPIC_API_KEY (Claude Code sends it as x-api-key,
  # which the gateway authenticates against this secret). Export it BEFORE launching so
  # `fak serve` inherits it (an unset/empty required-key env makes serve exit 2); the wiring
  # block below pins ANTHROPIC_API_KEY to the same source so the two always agree.
  export "$KEY_ENV=${ANTHROPIC_API_KEY:-fak-local-dogfood}"
  SERVE_ARGS+=(--gguf "$GGUF" --require-key-env "$KEY_ENV")
fi
# --smoke fronts the offline mock planner to prove the WIRE; leave its tool
# unadjudicated so the smoke output still shows a tool_use block.
if [ "$MODE" != "smoke" ] && [ -n "$POLICY" ] && [ "$POLICY" != "none" ]; then
  [ -f "$POLICY" ] || die "policy manifest not found: $POLICY (set FAK_DOGFOOD_POLICY=none to disable)"
  SERVE_ARGS+=(--policy "$POLICY")
fi
log "starting kernel: fak ${SERVE_ARGS[*]}"
"$BIN" "${SERVE_ARGS[@]}" >/tmp/fak-serve.log 2>&1 &
SERVE_PID=$!

cleanup() {
  [ -n "${SERVE_PID:-}" ] && kill "$SERVE_PID" 2>/dev/null || true
  [ -n "${SHIM_PID:-}" ]  && kill "$SHIM_PID"  2>/dev/null || true
}
trap cleanup EXIT INT TERM

until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1; do
  kill -0 "$SERVE_PID" 2>/dev/null || { cat /tmp/fak-serve.log >&2; die "fak serve died on startup"; }
  sleep 0.3
done
HEALTH="$(curl -s http://127.0.0.1:$PORT/healthz)"
log "kernel healthy: $HEALTH"
# Confirm the healthy kernel is OURS (right model), not some other process that
# happened to win the port between the guard and now.
case "$HEALTH" in
  *"\"model\":\"${MODEL:-mock}\""*) : ;;
  *) die "kernel on :$PORT reports an unexpected model (wanted ${MODEL:-mock}): $HEALTH" ;;
esac
if [ -n "$KERNEL_BACKEND" ]; then
  # A GGUF lacking an embedded BPE tokenizer would SILENTLY drop to the offline MockPlanner
  # (scripted text), not fak's forward. The whole point of this backend is the in-kernel
  # forward, so make anything but planner=inkernel a hard failure. Tolerate optional
  # whitespace around the JSON colon (the kernel emits compact json.Marshal, but a
  # pretty-printer/proxy in between may add spaces) — same parse as RUNNING_MODEL above.
  PLANNER="$(printf '%s' "$HEALTH" | sed -n 's/.*"planner"[[:space:]]*:[[:space:]]*"\([^"]*\)".*/\1/p')"
  [ "$PLANNER" = "inkernel" ] || die "kernel backend expected planner=inkernel but /healthz reports planner='$PLANNER'; the GGUF may lack an embedded tokenizer or fell back to the mock planner"
  log "verified: planner=inkernel; fak in-kernel forward is serving the wire"
fi

fi  # end "start our own kernel" path (skipped when ATTACHED)

# --- wire the Claude Code harness to the kernel -------------------------------
ACCOUNT_DIR="$(resolve_account_dir)"
# Claude Code appends /v1/messages itself, so the base URL must NOT include /v1.
export ANTHROPIC_BASE_URL="http://127.0.0.1:$PORT"
export CLAUDE_CONFIG_DIR="$ACCOUNT_DIR"
if [ "$BACKEND" = "anthropic" ]; then
  # Real Claude API upstream: fak is a TRANSPARENT hop. Leave the model tiers and the
  # API key ALONE — Claude Code uses its real models (claude-opus-4-8, …) and its own
  # credential, which fak forwards verbatim to api.anthropic.com (inbound x-api-key
  # passed through; cache_control survives). Pinning a placeholder key or remapping
  # tiers would defeat the point.
  :
else
  export ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY:-fak-local-dogfood}"   # non-empty; the loopback gateway ignores it
  # Map every Claude Code model tier onto our single local model so the main loop,
  # the background "small fast" calls, and the /model picker all hit the kernel.
  export ANTHROPIC_MODEL="$MODEL"
  export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
  export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
  export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
  export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
fi
export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="${CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC:-1}"
if [ -z "${API_TIMEOUT_MS:-}" ] && [ -n "${FAK_PLANNER_TIMEOUT_S:-}" ] && [ "$FAK_PLANNER_TIMEOUT_S" -gt 0 ] 2>/dev/null; then
  export API_TIMEOUT_MS="$((FAK_PLANNER_TIMEOUT_S * 1000))"
fi

print_env() {
  cat <<EOF
export ANTHROPIC_BASE_URL="$ANTHROPIC_BASE_URL"
export ANTHROPIC_API_KEY="$ANTHROPIC_API_KEY"
export CLAUDE_CONFIG_DIR="$CLAUDE_CONFIG_DIR"
export API_TIMEOUT_MS="${API_TIMEOUT_MS:-}"
export ANTHROPIC_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_OPUS_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_SONNET_MODEL="$MODEL"
export ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL"
export ANTHROPIC_SMALL_FAST_MODEL="$MODEL"
EOF
}

log "Claude Code wired:"
log "  ANTHROPIC_BASE_URL = $ANTHROPIC_BASE_URL   (native /v1/messages on the kernel)"
log "  CLAUDE_CONFIG_DIR  = $CLAUDE_CONFIG_DIR    (account: ${FAK_DOGFOOD_ACCOUNT:-faklocal})"
log "  model (all tiers)  = ${MODEL:-mock}"
[ -n "$KERNEL_BACKEND" ] && log "  forward            = fak's OWN in-kernel pure-Go decode over $GGUF  (NO Python, NO proxy)"
[ -n "$PRESET" ] && log "  preset             = $PRESET"
if [ ${#CLAUDE_DEBUG_ARGS[@]} -gt 0 ]; then
  log "  Claude debug       = ${CLAUDE_DEBUG_ARGS[*]}"
else
  log "  Claude debug       = off"
fi
if [ "$MODE" != "smoke" ] && [ -n "$POLICY" ] && [ "$POLICY" != "none" ]; then
  log "  policy             = $POLICY"
  log "  try a deny demo: ask Claude to run \`rm -rf /tmp/x\`, \`sudo ...\`, or \`git push\` —"
  log "  the kernel refuses it (POLICY_BLOCK) before the shell ever sees it, while \`ls\`/\`cat\` run."
fi

case "$MODE" in
  smoke)
    log "wire smoke — POST /v1/messages (buffered):"
    curl -s -X POST "$ANTHROPIC_BASE_URL/v1/messages" -H 'content-type: application/json' \
      -d '{"model":"claude-smoke","max_tokens":64,"messages":[{"role":"user","content":"hi"}]}' | sed 's/^/    /'
    echo
    log "wire smoke — POST /v1/messages (stream:true) event names:"
    curl -s -N -X POST "$ANTHROPIC_BASE_URL/v1/messages" -H 'content-type: application/json' \
      -d '{"model":"claude-smoke","max_tokens":64,"stream":true,"messages":[{"role":"user","content":"hi"}]}' \
      | grep '^event:' | sed 's/^/    /'
    log "smoke ok"
    ;;
  print-env)
    print_env
    ;;
  probe)
    OUT="$FAK_DIR/experiments/agent-live/dogfood-claude-probe.json"
    log "one live turn through Claude Code (headless): \"$PROBE_PROMPT\""
    log "Claude stderr/debug -> /tmp/fak-claude.log"
    set +e
    claude "${CLAUDE_DEBUG_ARGS[@]}" -p "$PROBE_PROMPT" --output-format json --dangerously-skip-permissions >"$OUT" 2>/tmp/fak-claude.log
    rc=$?
    set -e
    if [ $rc -ne 0 ]; then cat /tmp/fak-claude.log >&2; die "claude probe exited $rc"; fi
    log "transcript -> $OUT"
    python3 - "$OUT" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
res=d.get("result") if isinstance(d,dict) else d
print("    result:", (str(res)[:400] if res else d))
PY
    log "live turn ok — Claude Code completed a turn against the local kernel-fronted model"
    ;;
  run)
    if [ -n "$ATTACHED" ]; then
      log "launching interactive Claude Code (Ctrl-C to stop; attached kernel stays up)"
    else
      log "launching interactive Claude Code (Ctrl-C to stop; kernel shuts down on exit)"
    fi
    shift || true
    exec claude "${CLAUDE_DEBUG_ARGS[@]}" --dangerously-skip-permissions "$@"
    ;;
esac
