#!/usr/bin/env bash
# livecodebench_smoke_gpu_server.sh — run the LiveCodeBench generation arm against a model
# that is ALREADY SERVING on this box over an OpenAI-compatible endpoint, and write a compact
# RESULT file plus a .done(rc) sentinel so the whole run can be driven DETACHED over the
# control bridge: ship this script, launch it, pull the files back.
#
# This is the client-side counterpart of tools/glm52_load_witness.sh. That one stands the
# model UP and measures the load; this one assumes the serve is already healthy and exercises
# the benchmark verb (cmd/livecodebench) against it. The runner is a pure-Go HTTP client, so
# nothing here needs CUDA, -tags cuda, or a GPU.
#
# HONESTY FENCE: `livecodebench raw` GRADES NOTHING. It emits generations plus a shape check;
# the emitted report carries result_claim_allowed=false. Publishing a pass@1 still requires the
# official lcb_runner evaluator (docs/RUNBOOK step 3 in the livecodebench-eval workspace).
#
# Env (all defaulted):
#   LCB_ENDPOINT     OpenAI-compatible base URL ending in /v1 (default http://127.0.0.1:8000/v1)
#   LCB_MODEL        model id sent on each request (default: first id advertised by $LCB_ENDPOINT/models)
#   LCB_SUITE        normalized suite JSON (default: the committed 3-problem release_v2 sample —
#                    offline, so a smoke never depends on the HuggingFace datasets-server)
#   LCB_N            samples per problem (default 1; upstream lcb_runner default is 10)
#   LCB_TEMPERATURE  sampling temperature (default 0 — deterministic smoke; upstream default 0.2)
#   LCB_MAX_TOKENS   max_tokens per request (default 2048 — a REASONING model spends this
#                    budget on its private chain first, and a budget that runs out mid-think
#                    returns finish_reason=length with an EMPTY content field. Witnessed on
#                    GLM-5.2: at 512 one sample came back 0 chars with 512/512 tokens all in
#                    reasoning_content. Do not read that empty as a broken serve.)
#   LCB_CONCURRENCY  in-flight requests (default 1 — a host-offloaded 753B decode is serial-slow)
#   LCB_REQ_TIMEOUT  per-request HTTP timeout (default 900s)
#   LCB_OUTDIR       artifact directory (default /tmp/fakgpu/lcb-smoke)
#   LCB_OUT          RESULT file (default $LCB_OUTDIR/RESULT.txt); $LCB_OUT.done holds the rc
#
# Exit / rc codes (also written to $LCB_OUT.done):
#   0  smoke passed          90 no repo root        91 go build failed
#   92 endpoint unreachable  93 no model id         94 runner failed   95 empty generations
set -uo pipefail
export PATH="/usr/local/go/bin:$PATH"
export HOME="${HOME:-/root}" GOCACHE="${GOCACHE:-/tmp/gocache}" GOPATH="${GOPATH:-/tmp/gopath}"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"

LCB_ENDPOINT="${LCB_ENDPOINT:-http://127.0.0.1:8000/v1}"
LCB_MODEL="${LCB_MODEL:-}"
LCB_N="${LCB_N:-1}"
LCB_TEMPERATURE="${LCB_TEMPERATURE:-0}"
LCB_MAX_TOKENS="${LCB_MAX_TOKENS:-2048}"
LCB_CONCURRENCY="${LCB_CONCURRENCY:-1}"
LCB_REQ_TIMEOUT="${LCB_REQ_TIMEOUT:-900s}"
LCB_OUTDIR="${LCB_OUTDIR:-/tmp/fakgpu/lcb-smoke}"
LCB_OUT="${LCB_OUT:-$LCB_OUTDIR/RESULT.txt}"
DONE="${LCB_OUT}.done"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$LCB_OUTDIR" "$GOCACHE" "$GOPATH"
rm -f "$DONE"
: > "$LCB_OUT"
log(){ echo "$*" | tee -a "$LCB_OUT"; }
fail(){ log "$1"; echo "$2" > "$DONE"; exit "$2"; }

LCB_SUITE="${LCB_SUITE:-$ROOT/internal/livecodebench/testdata/suite_release_v2_sample.json}"
REPORT="$LCB_OUTDIR/raw-report.json"

cd "$ROOT" || fail "NO_REPO_ROOT" 90
log "LCB_SMOKE_START $(date -u +%Y-%m-%dT%H:%M:%SZ) host=$(hostname)"
log "HEAD=$(git rev-parse --short HEAD 2>/dev/null) $(git log -1 --format=%s 2>/dev/null | cut -c1-64)"
log "LCB_ENDPOINT=$LCB_ENDPOINT LCB_SUITE=$(basename "$LCB_SUITE") LCB_N=$LCB_N LCB_TEMPERATURE=$LCB_TEMPERATURE LCB_MAX_TOKENS=$LCB_MAX_TOKENS LCB_CONCURRENCY=$LCB_CONCURRENCY"

# 1. the serve must already be up — this script never starts one.
MODELS_JSON="$(curl -sf -m 30 "${LCB_ENDPOINT%/}/models" 2>/dev/null)" \
  || fail "ENDPOINT_UNREACHABLE $LCB_ENDPOINT — this script does not start a serve" 92
log "ENDPOINT_OK models=$(printf '%s' "$MODELS_JSON" | tr -d '\n' | cut -c1-240)"
if [ -z "$LCB_MODEL" ]; then
  LCB_MODEL="$(printf '%s' "$MODELS_JSON" | grep -oE '"id"[[:space:]]*:[[:space:]]*"[^"]+"' | head -1 | sed 's/.*"\([^"]*\)"$/\1/')"
fi
[ -n "$LCB_MODEL" ] || fail "NO_MODEL_ID — set LCB_MODEL=<id>" 93
log "LCB_MODEL=$LCB_MODEL"

# 2. build the benchmark verb from THIS checkout, so the run witnesses the checked-out code.
t=$(date +%s)
if go build -o "$LCB_OUTDIR/livecodebench" ./cmd/livecodebench >"$LCB_OUTDIR/build.log" 2>&1; then
  log "BUILD_OK $(( $(date +%s) - t ))s"
else
  log "BUILD_FAIL"; tail -25 "$LCB_OUTDIR/build.log" | tee -a "$LCB_OUT"; echo 91 > "$DONE"; exit 91
fi

# 3. generate. `raw` is the UNADJUDICATED arm: straight to the serving endpoint, no gateway
#    in the path — the right arm for smoking a bare model serve.
log "== livecodebench raw =="
t=$(date +%s)
"$LCB_OUTDIR/livecodebench" raw \
  --suite "$LCB_SUITE" --model "$LCB_MODEL" --endpoint "$LCB_ENDPOINT" \
  -n "$LCB_N" --temperature "$LCB_TEMPERATURE" --max-tokens "$LCB_MAX_TOKENS" \
  --concurrency "$LCB_CONCURRENCY" --timeout "$LCB_REQ_TIMEOUT" \
  --out "$REPORT" >"$LCB_OUTDIR/raw.log" 2>&1
RC=$?
GEN_S=$(( $(date +%s) - t ))
tail -6 "$LCB_OUTDIR/raw.log" | tee -a "$LCB_OUT"
[ "$RC" = 0 ] || fail "RAW_FAIL rc=$RC gen_s=$GEN_S" 94
log "RAW_OK gen_s=$GEN_S report=$REPORT"

# 4. summarize the generations. A report whose completions are all empty is a FAILED smoke
#    even though the runner exited 0 — an endpoint answering 200 with nothing is the exact
#    false-green this check exists to catch.
SUMMARY="$(python3 - "$REPORT" "$LCB_MAX_TOKENS" <<'PY' 2>/dev/null
import json,sys
r=json.load(open(sys.argv[1]))
ps=r.get("problems") or []
comps=[c for p in ps for c in (p.get("completions") or [])]
nonempty=[c for c in comps if c and c.strip()]
coded=[c for c in nonempty if "```" in c or "def " in c or "class " in c or "import " in c]
u=r.get("usage") or {}
print("PROBLEMS=%d SAMPLES=%d NONEMPTY=%d WITH_CODE=%d" % (len(ps),len(comps),len(nonempty),len(coded)))
print("TOKENS prompt=%s completion=%s cached=%s retries=%s" % (
    u.get("prompt_tokens"),u.get("completion_tokens"),u.get("cached_prompt_tokens"),u.get("retries")))
for p in ps:
    c=(p.get("completions") or [""])[0] or ""
    print("  %-16s chars=%-6d head=%s" % (p.get("question_id"),len(c)," ".join(c.split())[:90]))
if len(nonempty) < len(comps):
    # An empty completion from a reasoning model is almost always a spent think budget
    # (finish_reason=length, every token in reasoning_content), not a dead endpoint.
    print("EMPTY_HINT %d/%d sample(s) empty — on a reasoning model this is usually the think"
          " budget running out; re-run with a larger LCB_MAX_TOKENS (currently %s) before"
          " suspecting the serve" % (len(comps)-len(nonempty),len(comps),sys.argv[2]))
print("NONEMPTY_COUNT=%d" % len(nonempty))
PY
)"
if [ -n "$SUMMARY" ]; then
  printf '%s\n' "$SUMMARY" | tee -a "$LCB_OUT"
  NONEMPTY="$(printf '%s' "$SUMMARY" | grep -oE 'NONEMPTY_COUNT=[0-9]+' | cut -d= -f2)"
else
  # python3 absent: fall back to a byte-level check so the gate still has teeth.
  log "SUMMARY_UNAVAILABLE (no python3) — falling back to a byte check"
  NONEMPTY=$(grep -c '"completions"' "$REPORT" 2>/dev/null)
  log "REPORT_BYTES=$(wc -c < "$REPORT")"
fi
[ "${NONEMPTY:-0}" -gt 0 ] || fail "SMOKE_FAIL every completion was empty" 95

log "SMOKE_PASS nonempty=$NONEMPTY gen_s=$GEN_S"
log "ARTIFACTS $REPORT $LCB_OUT"
log "CLAIM result_claim_allowed=false — generation smoke only; pass@1 needs the official lcb_runner evaluator"
log "LCB_SMOKE_DONE rc=0"
echo 0 > "$DONE"
