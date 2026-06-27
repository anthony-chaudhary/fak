#!/usr/bin/env bash
# ornith_fak_serve.sh — durable stage + build + serve of an Ornith 1.0 checkpoint through
# fak's OWN in-kernel forward, PER SIZE (9B laptop CPU/Metal → 35B/397B-MoE A100), so it can
# be used DIRECTLY from Claude Code here as the usable-fallback coding model (#1034, epic #1026).
#
#   this machine (claude) --/v1/messages--> fak serve (in-kernel, OWN forward) --> Ornith 1.0
#           laptop:  9B  q4_k  (CPU / Metal)              A100 fleet:  35B / 397B-MoE  (--backend cuda)
#
# This is the Ornith sibling of tools/qwen36_a100_fak_serve.sh — same spine, but PER SIZE:
# it dispatches on $ORNITH_SIZE to pick the backend and the MoE flags each size needs. The
# epic's decisive finding (#1026, primary-sourced from each live config.json) is that EVERY
# Ornith checkpoint is a Qwen3.5 architecture (model_type qwen3_5 / qwen3_5_moe) — fak is
# purpose-built for this backbone (internal/model/qwen35.go), so this serves it, not ports it.
#
#   | Size | model_type    | shape (from config.json)                                  | tier             |
#   |------|---------------|-----------------------------------------------------------|------------------|
#   | 9b   | qwen3_5       | dense, hidden 4096, 32L (24 linear/8 full), 16Q/4KV       | laptop CPU/Metal |
#   | 35b  | qwen3_5_moe   | hidden 2048, 40L, 16Q/2KV, 256 experts top-8, ~3B act/tok | one A100 (cuda)  |
#   | 397b | qwen3_5_moe   | hidden 4096, 60L, 32Q/2KV, 512 experts top-10             | FP8 / TP=8 — DEFER |
#
# ── HONEST SCOPE — read before trusting any serve this stands up ──────────────────────────────
# This asserts NO throughput number and makes NO live-serve claim by itself. It builds the fak
# binary, fetches the checkpoint, stands the endpoint up, and health-checks a REAL chat turn AND
# a REAL tool-using turn (<tool_call> lifted to a tool_use block). A live Ornith serve turn — and
# its per-size decode tok/s (feeds the decode-parity playbook #977) — is HARDWARE-gated until this
# runs on a real laptop (9B) / A100 (35B), AND is DEPENDENCY-gated on the model-support chain:
#
#   #1029 (registry aliases ornith:9b/35b/397b)  → run-by-name; until it lands pass an explicit
#                                                   GGUF via ORNITH_REPO/ORNITH_FILE_GLOB.
#   #1027 (MoE expert-key fix — num_experts /     → WITHOUT this a 35B/397B checkpoint silently
#          shared_expert_intermediate_size)         loads as DENSE. DO NOT trust a 35B/397B serve
#                                                   for parity until #1027 is WITNESSED live.
#   #1030 (Qwen3.5 tokenizer + chat_template)     → correct prompt framing / eos.
#   #1028 (<think> reasoning-parser)              → strip the qwen3 reasoning so it does not leak
#                                                   into Claude Code context.
#   #1031 / #1032 (per-size HF oracle parity,     → cosine ≥0.9999 before any output is trusted.
#          qwen3_5_moe routing parity)
#
# Until those land this script is the reusable APPARATUS (the one-command serve a credentialed
# operator runs once the chain is green), not a witness. Point it at a real Ornith GGUF only when
# the size you are serving has its parity rung witnessed; otherwise it stands up but the output is
# not parity-trustworthy. This is the same posture tools/qwen36_a100_fak_serve.sh holds for #934.
#
# Usage (RUN ON THE TARGET HOST, detached so a disconnect does not orphan the load):
#   # 9B on a laptop, CPU reference forward:
#   ORNITH_SIZE=9b ORNITH_REPO=<hf-repo> bash tools/ornith_fak_serve.sh
#   # 9B on Apple-Silicon Metal (needs a -tags fakmetal build + a Q8 GGUF):
#   ORNITH_SIZE=9b BACKEND=metal ORNITH_REPO=<hf-repo> bash tools/ornith_fak_serve.sh
#   # 35B MoE on one A100 (cuda; add --cpu-offload-experts if it does not fit one GPU):
#   FAK_GATEWAY_KEY=sk-fak-... ORNITH_SIZE=35b CPU_OFFLOAD_EXPERTS=1 \
#     systemd-run --unit=ornithserve --collect bash tools/ornith_fak_serve.sh
# then poll:  cat "$ORNITH_DIR/PHASE"   and on ORNITH_FAK_SERVE_READY connect from the client
#             (scripts/connect-fak-node.{ps1,sh}). The GCP one-command deploy is the follow-on
#             scripts/gcp-ornith-serve.sh (sibling of scripts/gcp-qwen-serve.sh).
set -uo pipefail

# ── per-size configuration ────────────────────────────────────────────────────────────────────
ORNITH_SIZE="${ORNITH_SIZE:-9b}"
ORNITH_DIR="${ORNITH_DIR:-/opt/fak-ornith-serve}"
PORT="${PORT:-8080}"
ADDR="${ADDR:-0.0.0.0:${PORT}}"
# Context budget: caps the planned KV cache. Ornith is native 262144-ctx (no rope_scaling) — the
# in-kernel default would plan the FULL context → a 100s-of-GiB KV → FitTooBig. 32K holds a full
# Claude Code agent prompt; raise it only if you have the VRAM/RAM headroom.
CTX="${CTX:-32768}"
GO_VERSION="${GO_VERSION:-1.26.4}"
FAK_BIN="${FAK_BIN:-/usr/local/bin/fak}"
CUDA_GRAPH="${CUDA_GRAPH:-0}"               # 1 => --cuda-graph (#483); KEEP 0 (crashes the in-kernel
                                            # serve on lazy KV prealloc, #932) until KV is pre-allocated.
CPU_OFFLOAD_EXPERTS="${CPU_OFFLOAD_EXPERTS:-0}"  # 1 => --cpu-offload-experts (MoE experts on host RAM)

# Defaults that vary by size. BACKEND auto-selects per size; override it explicitly (cpu|metal|cuda).
case "$ORNITH_SIZE" in
  9b)
    BACKEND="${BACKEND:-cpu}"                                   # laptop: cpu reference, or metal, or cuda
    MODEL_ID="${MODEL_ID:-ornith:9b}"
    # ~19 GB bf16; q4_k_m is far smaller and fits a laptop. Metal wants a Q8 GGUF (GPU-resident Q8).
    ORNITH_FILE_GLOB="${ORNITH_FILE_GLOB:-*[Qq]4_[Kk]_[Mm]*.gguf}"
    ;;
  35b)
    BACKEND="${BACKEND:-cuda}"                                  # MoE: A100, experts loaded as MoE (#1027)
    MODEL_ID="${MODEL_ID:-ornith:35b}"
    ORNITH_FILE_GLOB="${ORNITH_FILE_GLOB:-*[Qq]4_[Kk]_[Mm]*.gguf}"
    ;;
  397b)
    BACKEND="${BACKEND:-cuda}"
    MODEL_ID="${MODEL_ID:-ornith:397b}"
    ORNITH_FILE_GLOB="${ORNITH_FILE_GLOB:-*[Qq]4_[Kk]_[Mm]*.gguf}"
    ;;
  *)
    echo "ornith_fak_serve: unknown ORNITH_SIZE='$ORNITH_SIZE' (want 9b|35b|397b)" >&2; exit 2 ;;
esac
# REPO is required (no default) — Ornith run-by-name (ornith:9b) needs #1029's registry alias to
# resolve, so until that lands the operator MUST name the HF GGUF repo explicitly.
ORNITH_REPO="${ORNITH_REPO:-}"

PHASE="$ORNITH_DIR/PHASE"
LOG="$ORNITH_DIR/fak_native_serve.log"
mkdir -p "$ORNITH_DIR"
ph(){ echo "$(date -u +%H:%M:%S) $*" | tee -a "$LOG"; echo "$*" > "$PHASE"; }

ph "ORNITH_SIZE=$ORNITH_SIZE BACKEND=$BACKEND MODEL_ID=$MODEL_ID (epic #1026, serve #1034)"

# ── 397B: DEFERRED — document the requirement and stop (no fabricated serve) ─────────────────────
if [ "$ORNITH_SIZE" = "397b" ]; then
  ph "DEFERRED_397B"
  cat >&2 <<'DEFER'
ornith_fak_serve: 397B (qwen3_5_moe, 512 experts top-10) is DEFERRED behind the FP8 path.
  Requirement: either the 397B-FP8 compressed-tensors path (#1032 / epic #F) on the device,
  or bf16 TP=8 on 8x80GB. fak's in-kernel forward serves it through the SAME qwen3_5_moe family
  as 35B once #1027 (MoE expert keys) + #1032 (routing parity + FP8 decision) land. Serve 9B/35B
  first; revisit 397B when the FP8 rung is witnessed.
DEFER
  exit 0
fi

# ── preflight: REPO must be named (until #1029 run-by-name lands) ────────────────────────────────
if [ -z "$ORNITH_REPO" ]; then
  ph "NO_REPO"
  cat >&2 <<'NOREPO'
ornith_fak_serve: set ORNITH_REPO to the Hugging Face GGUF repo for the Ornith size you are serving,
e.g. ORNITH_REPO=<org>/Ornith-1.0-9B-GGUF. Run-by-name (--gguf ornith:9b) needs the registry alias
from #1029; until that lands the repo must be named explicitly. Optionally narrow the shard with
ORNITH_FILE_GLOB (default a q4_k_m glob).
NOREPO
  exit 3
fi

export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
mkdir -p "$GOCACHE" "$GOPATH"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export FAK_CUDA_ARCH="${FAK_CUDA_ARCH:-sm_80}"
export PATH="/usr/local/go/bin:${CUDA_HOME}/bin:$PATH"

SELF="${BASH_SOURCE[0]}"
SCRIPT_DIR="$(cd "$(dirname "$SELF")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── 1. Ensure Go (the cuda build recipe expects it at /usr/local/go/bin) ─────────────────────────
if ! command -v go >/dev/null 2>&1; then
  ph "INSTALL_GO ${GO_VERSION}"
  if curl -fsSL "https://go.dev/dl/go${GO_VERSION}.linux-amd64.tar.gz" -o /tmp/go.tgz >>"$LOG" 2>&1 \
     && rm -rf /usr/local/go && tar -C /usr/local -xzf /tmp/go.tgz >>"$LOG" 2>&1; then :; else
    ph "GO_INSTALL_FAIL"; exit 11
  fi
fi
ph "GO $(go version 2>/dev/null || echo missing)"

# ── 2. Download the GGUF shard(s) (resumable; the HF CLI skips an already-complete file) ─────────
ph "DOWNLOAD_START repo=$ORNITH_REPO glob=$ORNITH_FILE_GLOB dir=$ORNITH_DIR"
if command -v hf >/dev/null 2>&1; then
  hf download "$ORNITH_REPO" --include "$ORNITH_FILE_GLOB" --local-dir "$ORNITH_DIR" >>"$LOG" 2>&1; DL_RC=$?
elif command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli download "$ORNITH_REPO" --include "$ORNITH_FILE_GLOB" --local-dir "$ORNITH_DIR" >>"$LOG" 2>&1; DL_RC=$?
else
  ph "NO_HF_CLI install huggingface_hub first"; exit 10
fi
# shellcheck disable=SC2086
GGUF=$(ls "$ORNITH_DIR"/$ORNITH_FILE_GLOB 2>/dev/null | sort | head -1)
ph "DOWNLOAD_DONE rc=$DL_RC gguf=$GGUF"
[ "${DL_RC:-1}" -eq 0 ] && [ -n "$GGUF" ] && [ -f "$GGUF" ] || { ph "DOWNLOAD_FAIL"; exit 20; }

# ── 3. Build the fak binary the chosen backend needs ─────────────────────────────────────────────
#   cuda  : libfakcuda (sm_XX) + cmd/fak via the canonical recipe.
#   metal : -tags fakmetal (Apple-Silicon GPU forward, #67).
#   cpu   : the plain CPU reference forward — no tags.
case "$BACKEND" in
  cuda)
    if [ ! -x "$FAK_BIN" ] || [ "${REBUILD_FAK:-0}" = "1" ]; then
      ph "BUILD_FAK_CUDA arch=$FAK_CUDA_ARCH out=$FAK_BIN"
      if ( cd "$ROOT" && bash internal/compute/build_cuda.sh binary ./cmd/fak "$FAK_BIN" ) >>"$LOG" 2>&1; then :; else
        ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
      fi
    fi
    export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${CUDA_HOME}/lib:${LD_LIBRARY_PATH:-}"
    ;;
  metal)
    FAK_BIN="${FAK_BIN_METAL:-$ORNITH_DIR/fak}"
    ph "BUILD_FAK_METAL out=$FAK_BIN (-tags fakmetal; needs an Apple-Silicon Metal device)"
    if ( cd "$ROOT" && go build -tags fakmetal -o "$FAK_BIN" ./cmd/fak ) >>"$LOG" 2>&1; then :; else
      ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
    fi
    ;;
  cpu)
    FAK_BIN="${FAK_BIN_CPU:-$ORNITH_DIR/fak}"
    ph "BUILD_FAK_CPU out=$FAK_BIN (CPU reference forward)"
    if ( cd "$ROOT" && go build -o "$FAK_BIN" ./cmd/fak ) >>"$LOG" 2>&1; then :; else
      ph "BUILD_FAK_FAIL"; tail -40 "$LOG" >&2 || true; exit 30
    fi
    ;;
  *)
    ph "UNKNOWN_BACKEND $BACKEND"; echo "ornith_fak_serve: BACKEND must be cpu|metal|cuda" >&2; exit 2 ;;
esac
[ -x "$FAK_BIN" ] || { ph "BUILD_FAK_FAIL"; exit 30; }
ph "FAK_BIN_READY $FAK_BIN"

# ── 4. Serve through the PURE FAK KERNEL ─────────────────────────────────────────────────────────
# The embedded GGUF tokenizer makes /v1/chat/completions AND /v1/messages serve real in-kernel
# chat; the eager load binds the listener only AFTER the weights are resident, so /v1/models
# answering means it is loaded. The InKernelPlanner lifts <tool_call> → structured tool_use; the
# <think> reasoning strip is #1028 (not yet on this path — until it lands, reasoning may leak).
SERVE_ARGS=(serve --addr "$ADDR" --gguf "$GGUF" --model "$MODEL_ID" --context-budget-tokens "$CTX")
case "$BACKEND" in
  cuda)
    SERVE_ARGS+=(--backend cuda)
    [ "$CUDA_GRAPH" = "1" ] && SERVE_ARGS+=(--cuda-graph)
    # MoE (35B/397B): keep the expert GEMMs on host RAM if they dwarf VRAM. Requires #1027 so the
    # experts load as MoE in the first place — without it the checkpoint loads DENSE.
    [ "$CPU_OFFLOAD_EXPERTS" = "1" ] && SERVE_ARGS+=(--cpu-offload-experts)
    ;;
  metal)
    SERVE_ARGS+=(--metal) ;;   # Apple-Silicon GPU forward; mutually exclusive with --backend
  cpu)
    : ;;                        # empty backend = the CPU reference path
esac
# Inbound bearer auth: a network-facing gateway must require a key. connect-fak-node uses it.
if [ -n "${FAK_GATEWAY_KEY:-}" ]; then
  export FAK_GATEWAY_KEY
  SERVE_ARGS+=(--require-key-env FAK_GATEWAY_KEY)
else
  ph "WARN no FAK_GATEWAY_KEY set — serving with NO inbound auth (loopback/tunnel only!)"
fi
# FAK_Q4K=1 holds the q4_k matmul tensors raw on device (the resident-Q4_K decode lever) on cuda;
# inert on cpu/metal where the dequant path is used.
ph "LAUNCH FAK_Q4K=1 fak ${SERVE_ARGS[*]} (size=$ORNITH_SIZE backend=$BACKEND)"
FAK_Q4K=1 "$FAK_BIN" "${SERVE_ARGS[@]}" > "$ORNITH_DIR/server.log" 2>&1 &
SRV=$!
ph "SERVER_PID=$SRV"

# ── 5. Health-check: a crashed load fails fast; a REAL chat answer AND a REAL tool-using turn ───
#       must pass before READY. 90 x 20s = 30 min, ample for a single-file load.
AUTH_HDR=()
[ -n "${FAK_GATEWAY_KEY:-}" ] && AUTH_HDR=(-H "Authorization: Bearer ${FAK_GATEWAY_KEY}")
for _ in $(seq 1 90); do
  if ! kill -0 "$SRV" 2>/dev/null; then ph "SERVER_EXITED_EARLY"; tail -40 "$ORNITH_DIR/server.log" >>"$LOG" 2>&1; exit 40; fi
  if curl -sf -m 5 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/healthz" >/dev/null 2>&1 \
     || curl -sf -m 5 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
    # (a) a real chat completion
    smoke=$(curl -s -m 180 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word: ok\"}],\"max_tokens\":16}")
    echo "SMOKE_CHAT: $smoke" >>"$LOG"
    if ! printf '%s' "$smoke" | grep -q '"content"' || printf '%s' "$smoke" | grep -q '"error"'; then
      ph "SMOKE_CHAT_FAIL"; exit 41
    fi
    # (b) a real tool-using turn — assert <tool_call> is lifted to a structured tool call, the
    #     acceptance lever for an agentic Claude Code turn (#1034). A model may legitimately not
    #     call the tool; this asserts the serve ACCEPTS a tools request and answers without error.
    toolsmoke=$(curl -s -m 180 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get weather for a city\",\"parameters\":{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}},\"required\":[\"city\"]}}}],\"max_tokens\":64}")
    echo "SMOKE_TOOLS: $toolsmoke" >>"$LOG"
    if printf '%s' "$toolsmoke" | grep -q '"error"'; then
      ph "SMOKE_TOOLS_FAIL"; exit 43
    fi
    # The raw <tool_call> tag must NOT leak through unlifted — if it does, the InKernelPlanner did
    # not parse it (a real defect for an agent turn). A clean tool_calls structure or a plain
    # answer both pass; a leaked literal tag does not.
    if printf '%s' "$toolsmoke" | grep -q '<tool_call>'; then
      ph "SMOKE_TOOLS_UNLIFTED tool_call tag leaked unparsed — see $LOG"; exit 44
    fi
    # (c) per-size decode tok/s sample — feeds #977. Recorded ONLY from a real timed generation on
    #     this host, so it is WITNESSED-on-run, never a literal baked into the script.
    t0=$(date +%s.%N 2>/dev/null || date +%s)
    rate=$(curl -s -m 240 "${AUTH_HDR[@]}" "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL_ID\",\"messages\":[{\"role\":\"user\",\"content\":\"Count from 1 to 100, comma-separated.\"}],\"max_tokens\":256}")
    t1=$(date +%s.%N 2>/dev/null || date +%s)
    ntok=$(printf '%s' "$rate" | grep -o '"completion_tokens":[0-9]*' | grep -o '[0-9]*' | head -1)
    dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.3f", (b-a>0)?b-a:1}')
    if [ -n "$ntok" ] && [ "$ntok" -gt 0 ] 2>/dev/null; then
      tps=$(awk -v n="$ntok" -v d="$dt" 'BEGIN{printf "%.2f", n/d}')
      echo "{\"size\":\"$ORNITH_SIZE\",\"backend\":\"$BACKEND\",\"model\":\"$MODEL_ID\",\"completion_tokens\":$ntok,\"seconds\":$dt,\"decode_tok_per_s\":$tps,\"label\":\"WITNESSED\"}" \
        > "$ORNITH_DIR/decode_tok_s.json"
      ph "DECODE_TOK_S size=$ORNITH_SIZE backend=$BACKEND tok/s=$tps (WITNESSED — $ORNITH_DIR/decode_tok_s.json)"
    fi
    ph "ORNITH_FAK_SERVE_READY port=$PORT size=$ORNITH_SIZE backend=$BACKEND model=$MODEL_ID"
    exit 0
  fi
  sleep 20
done
ph "HEALTH_TIMEOUT"; tail -20 "$ORNITH_DIR/server.log" >>"$LOG" 2>&1; exit 42
