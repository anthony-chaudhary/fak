#!/usr/bin/env bash
# gcp-glm-demo.sh — the ONE-COMMAND GCP-H100 GLM-5.2 kernel demo, end to end.
#
# This is the orchestrator the goal "demoable working fak kernel GLM-5.2 on GCP
# H100(s)" names: it chains the four steps that a live demo needs into a single
# reviewable command, defaulting to the 8x H100 tier (a3-high-h100, 640 GB —
# GLM-5.2 is 343-487 GB so it needs the 8-GPU shape, never 1x80 GB):
#
#   1. PROVISION + SERVE — stand GLM-5.2 up through the PURE FAK KERNEL on a GCP
#      H100 node. Delegated verbatim to scripts/gcp-glm-serve.sh (the canonical
#      bring-up); this script never re-implements the gcloud/serve rendering.
#   2. PROBE — two `claude-glm-gcp --probe` turns that SHARE a system+tools prefix,
#      so turn 2 hits the prefix fak already holds resident.
#   3. CACHE-VALUE — scrape fak's OWN RadixAttention KV-prefix reuse off the serve
#      /metrics surface (fak_gateway_kv_prefix_reused_tokens_total). reuse > 0 on
#      turn 2..N is the WITNESSED demo datum (#1010): the prefill fak did NOT redo.
#   4. TEARDOWN — delete the VM (always, on a trap) so the demo leaves zero cost.
#
#       ┌────────────────┐  /v1   ┌────────────────────────┐  /v1   ┌──────────────────┐
#       │ claude-glm-gcp │ ─────▶ │ fak serve (the kernel) │ ─────▶ │ GLM-5.2 on 8x     │
#       │  (Claude Code) │ ◀───── │ glm_moe_dsa, adjud.    │ ◀───── │ H100 (a3-high)    │
#       └────────────────┘        └────────────────────────┘        └──────────────────┘
#                 turn 2 reuses turn 1's system+tools prefix  ── fak_gateway_kv_prefix_reused_tokens_total > 0
#
# PLAN BY DEFAULT. With no creds this prints every step — the gcloud create (via the
# serve script), the two probes, the cache-value scrape, and the teardown — and exits
# 0, so the whole demo is reviewable before a dollar is spent. `--apply` runs it and
# needs `gcloud` + GCP_PROJECT (and HF_TOKEN for the gated checkpoint).
#
# Usage:
#   ./scripts/gcp-glm-demo.sh                                 # PLAN: the full demo, no creds
#   GCP_PROJECT=my-proj ./scripts/gcp-glm-demo.sh --apply     # run it on 8x H100, then tear down
#   GCP_TIER=a3-ultra-h200 ./scripts/gcp-glm-demo.sh          # plan it on H200 instead
#   KEEP=1 GCP_PROJECT=my-proj ./scripts/gcp-glm-demo.sh --apply  # skip teardown (debug the node)
#
# Knobs (env) — the serve knobs (SERVE, GLM_GGUF_*, HF_TOKEN, ...) pass straight through
# to scripts/gcp-glm-serve.sh; this script adds only the demo-specific ones:
#   GCP_TIER          gcp_accel.py tier slug          (default a3-high-h100 — the 8x H100 demo tier)
#   VM_NAME           instance name                   (default fak-glm-demo)
#   PROBE_PROMPT      the headless probe turn          (default "say pong")
#   PROBE_TURNS       how many cache-warming turns     (default 2 — turn 2 is where reuse must bite)
#   LOCAL_TUNNEL_PORT local port the tunnel binds      (default 8200 — the preset default)
#   KEEP              1 = skip teardown (debug)         (default empty = always tear down)
set -euo pipefail

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# The demo tier is the 8x H100 shape by default — the goal's "GCP H100(s)". GLM-5.2 at
# UD-Q4_K_M is ~466 GB, so it only fits across the 640 GB of a3-high-h100's 8 GPUs.
GCP_TIER="${GCP_TIER:-a3-high-h100}"
VM_NAME="${VM_NAME:-fak-glm-demo}"
# The demo is the PURE FAK KERNEL demo: force SERVE=fak even on sm_90+ H100, where the
# serve script would otherwise default to stock SGLang. This is load-bearing twice over —
# it's the whole point of "demoable working fak kernel", AND the cache-value witness
# (fak_gateway_kv_prefix_reused_tokens_total) only exists when fak itself serves the model.
SERVE="${SERVE:-fak}"
GLM_PORT="${GLM_PORT:-8000}"
LOCAL_TUNNEL_PORT="${LOCAL_TUNNEL_PORT:-8200}"
PROBE_PROMPT="${PROBE_PROMPT:-say pong}"
PROBE_TURNS="${PROBE_TURNS:-2}"
KEEP="${KEEP:-}"

MODE="plan"
case "${1:-}" in
  --apply)    MODE="apply" ;;
  --plan|"")  MODE="plan" ;;
  --help|-h)  sed -n '2,52p' "$0"; exit 0 ;;
  *) echo "unknown arg: $1 (see --help)" >&2; exit 2 ;;
esac

log()  { printf '\033[36m[gcp-glm-demo]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[33m[gcp-glm-demo] %s\033[0m\n' "$*" >&2; }
die()  { printf '\033[31m[gcp-glm-demo] %s\033[0m\n' "$*" >&2; exit 1; }

# --- resolve the tier from the single registry (tools/gcp_accel.py) so the demo's own
# summary + teardown command name the same zone/shape the serve script provisions. ---
command -v python >/dev/null 2>&1 && PY=python || PY=python3
TIER_SHELL="$("$PY" "$ROOT/tools/gcp_accel.py" --emit-shell "$GCP_TIER")" \
  || die "unknown GCP_TIER='$GCP_TIER' — run: $PY tools/gcp_accel.py  (to list the ladder)"
eval "$TIER_SHELL"   # defines GLM_MACHINE_TYPE, GLM_DEFAULT_ZONE, GLM_GPU_COUNT, ...
GCP_ZONE="${GCP_ZONE:-${GLM_DEFAULT_ZONE}}"

# --- step 1 render: delegate the provision+serve plan to the canonical bring-up. ------
# We never re-implement the gcloud/serve rendering; we COMPOSE it, then add the demo's
# probe + cache-value + teardown steps around it.
render_serve_plan() {
  GCP_TIER="$GCP_TIER" VM_NAME="$VM_NAME" GLM_PORT="$GLM_PORT" SERVE="$SERVE" \
  LOCAL_TUNNEL_PORT="$LOCAL_TUNNEL_PORT" GCP_ZONE="$GCP_ZONE" \
    bash "$ROOT/scripts/gcp-glm-serve.sh" --plan
}

# --- step 2+3 render: the cache-value probe — two turns that share a prefix, then the
# scrape of fak's OWN realized KV-prefix reuse off the serve /metrics surface. ---------
print_cache_value_steps() {
  cat <<DEMO
# === DEMO step 2 — drive ${PROBE_TURNS} headless turns through the kernel ===
# Each turn carries the SAME system+tools prefix, so turn 2..N reuse the KV prefix fak
# already holds resident. (Run scripts/dogfood-claude.sh --install once for the preset.)
export FAK_GLM_GCP_BASE_URL=http://127.0.0.1:${LOCAL_TUNNEL_PORT}/v1
for turn in \$(seq 1 ${PROBE_TURNS}); do
  claude-glm-gcp --probe "${PROBE_PROMPT}"   # one witnessable headless turn
done

# === DEMO step 3 — WITNESS the cache value (the #1010 lever) ===
# fak's RadixAttention prefix cache eliminates the repeated system+tools+repo prefill on
# turns 2..N. Scrape the counter fak itself authored (WITNESSED, not relayed):
curl -s http://127.0.0.1:${LOCAL_TUNNEL_PORT}/metrics \\
  | grep '^fak_gateway_kv_prefix_reused_tokens_total'
# DEMO PASSES when fak_gateway_kv_prefix_reused_tokens_total > 0 — the prefill fak did NOT redo.
DEMO
}

# --- step 4 render: teardown so the demo is self-contained and leaves zero cost. ------
print_teardown_steps() {
  if [ -n "$KEEP" ]; then
    echo "# === DEMO step 4 — teardown SKIPPED (KEEP=1); delete it yourself when done: ==="
  else
    echo "# === DEMO step 4 — teardown (always, so the demo leaves zero residual cost) ==="
  fi
  printf 'gcloud compute instances delete %q --zone %q' "$VM_NAME" "$GCP_ZONE"
  echo "${GCP_PROJECT:+ --project ${GCP_PROJECT}} --quiet"
}

if [ "$MODE" = "plan" ]; then
  log "PLAN (no apply). One-command GLM-5.2 demo on tier '$GCP_TIER' ($GLM_MACHINE_TYPE, ${GLM_GPU_COUNT}x ${GLM_GPU_LABEL}, sm_${GLM_COMPUTE_CAP}) as '$VM_NAME' in $GCP_ZONE."
  echo "# ============================================================================"
  echo "# DEMO step 1 — PROVISION + SERVE (delegated to scripts/gcp-glm-serve.sh):"
  echo "# ============================================================================"
  render_serve_plan
  echo
  print_cache_value_steps
  echo
  print_teardown_steps
  echo
  log "to run it for real: GCP_PROJECT=<id> HF_TOKEN=<hf> $0 --apply"
  exit 0
fi

# --- apply --------------------------------------------------------------------
command -v gcloud >/dev/null || die "gcloud not found — install the Cloud SDK"
[ -n "${GCP_PROJECT:-}" ] || die "GCP_PROJECT is required for --apply"

# Teardown ALWAYS runs (unless KEEP=1): a crashed demo must not leave an 8x H100 node
# burning. The trap fires on any exit after the VM is created.
teardown() {
  [ -n "$KEEP" ] && { warn "KEEP=1 — leaving $VM_NAME up; delete it with: gcloud compute instances delete $VM_NAME --zone $GCP_ZONE --quiet"; return 0; }
  log "tearing down $VM_NAME (zero residual cost)"
  gcloud compute instances delete "$VM_NAME" --zone "$GCP_ZONE" ${GCP_PROJECT:+--project "$GCP_PROJECT"} --quiet || true
}
trap teardown EXIT

log "DEMO step 1 — provision + serve GLM-5.2 on $VM_NAME via the pure fak kernel"
GCP_TIER="$GCP_TIER" VM_NAME="$VM_NAME" GLM_PORT="$GLM_PORT" SERVE="$SERVE" \
LOCAL_TUNNEL_PORT="$LOCAL_TUNNEL_PORT" GCP_ZONE="$GCP_ZONE" GCP_PROJECT="$GCP_PROJECT" \
  bash "$ROOT/scripts/gcp-glm-serve.sh" --apply

log "DEMO node created. The GLM-5.2 load is multi-hundred-GB (~40min); the live probe +"
log "cache-value scrape + teardown is the operator's next step once the serve is healthy:"
echo
print_cache_value_steps
echo
log "then this script's EXIT trap tears the node down (KEEP=1 to keep it)."
