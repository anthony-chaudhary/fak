#!/usr/bin/env bash
# dogfood-opencode-glm.sh — front the opencode/GLM dispatch lane with the kernel
# (issue #730). The claude lane is guarded by default; the opencode lane fronts a
# LOCAL GLM server, so `fak guard --provider openai` would misroute it to the public
# OpenAI API unless `FLEET_DOGFOOD_GUARD_BASEURL` names that local upstream. This
# launcher discovers the GLM /v1 base URL, exports that env, and execs the dispatch
# worker so `guard_wrap` fronts opencode with
# `fak guard --provider openai --base-url <glm> -- opencode …`.
#
#   ┌──────────┐   /v1/chat/...   ┌──────────────────────┐   /v1/chat/...   ┌───────────┐
#   │ opencode │ ───────────────▶ │ fak guard (the kernel)│ ───────────────▶ │ GLM server│
#   │  worker  │ ◀──── SSE ─────── │  adjudicates tools     │ ◀────────────── │  (/v1)    │
#   └──────────┘                    └──────────────────────┘                  └───────────┘
#
# Usage:
#   ./scripts/dogfood-opencode-glm.sh --lane glm                 # dry-run: print the guarded argv
#   ./scripts/dogfood-opencode-glm.sh --lane glm --live          # spawn ONE guarded opencode worker
#   ./scripts/dogfood-opencode-glm.sh --lane glm --print-env     # print the export line, then exit
#
# Knobs (env):
#   FAK_GLM_BASE_URL   the GLM OpenAI-compatible base URL (e.g. http://127.0.0.1:8001/v1).
#                      REQUIRED on a real node — the live endpoint is per-node and
#                      hardware-gated, so this launcher refuses to guess a wrong upstream.
#                      Candidate ports are probed only as a discovery convenience.
#   FAK_GLM_PORTS      space-separated localhost ports to probe for a GLM /v1 (default:
#                      "8001 8000 8080 11434") when FAK_GLM_BASE_URL is unset.
#   FAK_BIN            an explicit fak binary (else tools/.bin/fak is built/used).
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
while [ -L "$SELF" ]; do
  link="$(readlink "$SELF")"
  case "$link" in /*) SELF="$link" ;; *) SELF="$(cd "$(dirname "$SELF")" && pwd)/$link" ;; esac
done
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT"

log()  { printf '\033[36m[dogfood-glm]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[dogfood-glm] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[dogfood-glm] %s\033[0m\n' "$*" >&2; exit 1; }

LANE=""; MODE="dry-run"
while [ $# -gt 0 ]; do
  case "$1" in
    --lane)      LANE="${2:-}"; shift 2 ;;
    --live)      MODE="live"; shift ;;
    --dry-run)   MODE="dry-run"; shift ;;
    --print-env) MODE="print-env"; shift ;;
    --help|-h)   sed -n '2,30p' "$0"; exit 0 ;;
    *) die "unknown arg: $1 (see --help)" ;;
  esac
done
[ -n "$LANE" ] || die "a --lane is required (e.g. --lane glm)"

# --- discover the GLM base URL ------------------------------------------------
# Precedence: explicit FAK_GLM_BASE_URL (the production path on a real node) > a
# reachability probe of candidate localhost ports (a convenience, never a guess we
# act on silently). A node operator sets FAK_GLM_BASE_URL once; the probe just helps
# them find it the first time.
discover_glm_base_url() {
  local explicit="${FAK_GLM_BASE_URL:-}"
  if [ -n "$explicit" ]; then
    local raw="${explicit%/}"
    case "$raw" in */v1) : ;; *) raw="$raw/v1" ;; esac
    printf '%s' "$raw"
    return 0
  fi
  local ports="${FAK_GLM_PORTS:-8001 8000 8080 11434}"
  for p in $ports; do
    local cand="http://127.0.0.1:$p/v1"
    if curl -sf "$cand/models" >/dev/null 2>&1; then
      warn "discovered a reachable GLM /v1 at $cand (probe) — set FAK_GLM_BASE_URL to pin it"
      printf '%s' "$cand"
      return 0
    fi
  done
  return 1
}

if ! GLM_BASE="$(discover_glm_base_url)"; then
  die "no GLM /v1 base URL found.
       Set FAK_GLM_BASE_URL to the live GLM server (per-node, hardware-gated), e.g.:
         FAK_GLM_BASE_URL=http://127.0.0.1:8001/v1 $0 --lane $LANE --live
       (the actual endpoint flip on a GLM-serving node is deferred — see
        docs/fak/opencode-glm-guard.md for the discovery + activation steps)."
fi
export FLEET_DOGFOOD_GUARD_BASEURL="$GLM_BASE"
log "FLEET_DOGFOOD_GUARD_BASEURL=$GLM_BASE  (opencode will be guarded on the OpenAI wire)"

if [ "$MODE" = "print-env" ]; then
  printf 'export FLEET_DOGFOOD_GUARD_BASEURL=%q\n' "$GLM_BASE"
  exit 0
fi

# --- build/locate the kernel binary so guard-wrapping engages -----------------
# resolve_fak_bin looks at FAK_BIN -> tools/.bin/fak -> PATH; build the in-tree one
# when none is set so the worker is actually guarded (it fails OPEN otherwise).
if [ -z "${FAK_BIN:-}" ]; then
  BIN="$ROOT/tools/.bin/fak"
  [ "$(uname -s 2>/dev/null)" != "Linux" ] && [ -f "$BIN.exe" ] && BIN="$BIN.exe"
  if [ ! -x "$BIN" ]; then
    log "building fak -> $ROOT/tools/.bin/fak"
    ( cd "$ROOT" && go build -o "$ROOT/tools/.bin/fak" ./cmd/fak ) || die "go build ./cmd/fak failed"
    BIN="$ROOT/tools/.bin/fak"
  fi
  export FAK_BIN="$BIN"
fi

# --- run the dispatch worker on the GLM lane ----------------------------------
# Dry-run by default (prints the guarded argv so an operator can witness guarded:true
# before any live spawn); --live spawns one guarded opencode worker.
DRY=(--dry-run --json)
[ "$MODE" = "live" ] && DRY=()
log "dispatch_worker --lane $LANE --backend opencode ${DRY[*]:-(live)}"
exec python3 "$ROOT/tools/dispatch_worker.py" --lane "$LANE" --backend opencode "${DRY[@]}"
