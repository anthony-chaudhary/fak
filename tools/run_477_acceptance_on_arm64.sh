#!/usr/bin/env bash
# run_477_acceptance_on_arm64.sh — the on-arm64 acceptance gate for issue #477
# (the amortized-FP-reduction Q8_0 decode kernel: qdot8amortNEON in internal/model/quant_arm64_amort.s,
# selected via FAK_QKERNEL=amort, with FEAT_I8MM detection in quant_arm64.go staged for the SMMLA
# follow-up tier). The kernel keeps each 32-wide block's int32 SDOT dot as a 4-lane vector,
# converts the whole vector with one SCVTF.4S, FMLAs it into four lane-parallel float accumulators,
# and pays the horizontal float reduce ONCE at the end — the llama.cpp/ggml deferred reduction that
# M3-LLAMACPP-RESULTS.md sec. 4 names as the decode lever. It forfeits bit-identity to scalar, so it
# has its OWN gate (argmax-exact + cosine), not TestQdot8NEONMatchesScalar.
#
# WHAT NODE THIS RUNS ON
#   An arm64 host with FEAT_DotProd (SDOT) — every Apple Silicon Mac (M1+) and ARMv8.4+ Linux.
#   The headline target is an i8mm-capable part (Apple M2/M3, ARMv8.6+) where the SMMLA follow-up
#   will pay off; the amortized kernel itself runs on any SDOT part. It CANNOT run on the win32/x86
#   host that authored this code: that host cannot execute arm64, so the kernel's argmax/cosine RUN
#   and the tok/s measure are the explicit residual handed off here. On the authoring host the code
#   was confirmed to: build green by default (`go build ./...`), CROSS-COMPILE for arm64
#   (`GOOS=darwin/linux GOARCH=arm64 go build ./internal/model/`), and TEST-COMPILE for arm64
#   (`GOOS=darwin GOARCH=arm64 go test -c ./internal/model/`). Only the device execution below
#   needs real arm64 hardware.
#
# WHAT IT PROVES
#   1. The dedicated correctness gate, run under FAK_QKERNEL=amort:
#        TestQdot8AmortArgmaxAndCosine  (internal/model/quant_arm64_amort_test.go)
#      which, over the Qwen2.5-1.5B decode GEMV shapes + a vocab-sized argmax target, asserts the
#      amortized kernel's logit vector is ARGMAX-EXACT vs the scalar reference (greedy token
#      preserved) and COSINE ≥ 0.9999. A SKIP (no FEAT_DotProd) is NOT a pass.
#   2. The bit-identity anchor is UNTOUCHED and still green on the default path:
#        TestQdot8NEONMatchesScalar  (internal/model/quant_arm64_test.go)
#   3. The single-core kernel A/B (MAC/ns): qdot8asm vs unroll4 vs amort, via
#        BenchmarkDotKernelAmort
#      so the amortized kernel's decode-dot throughput delta over the default is recorded.
#   4. (Optional) the full-model decode tok/s A/B — default vs FAK_QKERNEL=amort — via modelbench,
#      when a Qwen2.5-1.5B HF snapshot or GGUF file is supplied, to re-measure against the 28.9 tok/s baseline in
#      docs/benchmarks/M3-LLAMACPP-RESULTS.md sec. 3 (the acceptance-criteria uplift line).
#
# USAGE
#   bash tools/run_477_acceptance_on_arm64.sh
#   env knobs:
#     FAK_QWEN_SNAPSHOT=/path/to/qwen2.5-1.5b-instruct   # enables the optional full-model tok/s A/B
#     FAK_QWEN_GGUF=/path/to/qwen2.5-1.5b-instruct-q8_0.gguf
#                                                         # same, using a GGUF Q8 artifact
#     FAK_WORKERS=6                                       # cores for the full-model run (modelbench)
#     FAK_BENCHTIME=2s                                    # per-kernel benchmark time (default 2s)
#     FAK_477_OUT=experiments/benchmark/runs/by-machine/node-macos-a/<run>/score.json
#                                                         # write a machine-readable scorecard
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG='./internal/model/'
GATE='TestQdot8AmortArgmaxAndCosine'
ANCHOR='TestQdot8NEONMatchesScalar'
BENCH='^BenchmarkDotKernelAmort$'
BENCHTIME="${FAK_BENCHTIME:-2s}"
OUT_JSON="${FAK_477_OUT:-}"
TMP_DIR="$(mktemp -d -t fak477out.XXXXXX)"
cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

echo "== #477 amortized-FP-reduction arm64 decode kernel acceptance =="
echo "[477] repo root : $MOD_DIR"
echo "[477] gate      : $GATE (FAK_QKERNEL=amort, $PKG)"
echo "[477] anchor    : $ANCHOR (default path, must stay green)"
echo "[477] benchmark : BenchmarkDotKernelAmort (asm vs unroll4 vs amort, MAC/ns)"
if [ -n "$OUT_JSON" ]; then
  echo "[477] scorecard : $OUT_JSON"
fi

# ---- arch / feature sanity --------------------------------------------------------
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
  arm64|aarch64) ;;
  *) echo "[477] FAIL: host arch is '$ARCH', not arm64 — this gate must run on Apple Silicon / arm64." >&2
     echo "[477] (The kernel was authored + cross-compiled on x86; this is the arm64 RUN residual.)" >&2
     exit 3 ;;
esac

cd "$MOD_DIR"
command -v go >/dev/null 2>&1 || { echo "[477] FAIL: no 'go' on PATH." >&2; exit 3; }

# ---- (1) the dedicated correctness gate under FAK_QKERNEL=amort --------------------
LOG="$(mktemp -t fak477.XXXXXX.log)"
echo
echo "== (1) correctness gate: argmax-exact + cosine (FAK_QKERNEL=amort) =="
set +e
FAK_QKERNEL=amort go test -count=1 -v -run "^${GATE}$" "$PKG" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if grep -aqE "no FEAT_DotProd|asimddp\) not available|--- SKIP" "$LOG"; then
  echo "[477] INCONCLUSIVE: the gate SKIPPED — no FEAT_DotProd on this arm64 part." >&2
  echo "[477] A skip is not a pass. Run on an SDOT-capable arm64 host (any Apple Silicon)." >&2
  rm -f "$LOG"
  exit 4
fi
if [ "$rc" -ne 0 ]; then
  echo "[477] FAIL: amortized kernel gate did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[477] PASS: amortized decode kernel is argmax-exact + cosine ≥ 0.9999 vs the scalar reference."

# ---- (2) the bit-identity anchor is untouched and still green ---------------------
echo
echo "== (2) bit-identity anchor still green (default path) =="
set +e
go test -count=1 -v -run "^${ANCHOR}$" "$PKG" 2>&1 | tee -a "$LOG"
arc=${PIPESTATUS[0]}
set -e
if [ "$arc" -ne 0 ]; then
  echo "[477] FAIL: the bit-identity anchor $ANCHOR regressed (rc=$arc) — qdot8asm path must be intact." >&2
  rm -f "$LOG"
  exit "$arc"
fi
echo "[477] PASS: $ANCHOR still bit-identical (the default decode path is unchanged)."

# ---- (3) single-core kernel A/B (MAC/ns) ------------------------------------------
echo
echo "== (3) single-core decode-dot throughput: asm vs unroll4 vs amort =="
BLOG="$(mktemp -t fak477bench.XXXXXX.log)"
set +e
go test -count=1 -run '^$' -bench "$BENCH" -benchtime="$BENCHTIME" "$PKG" 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e
echo
echo "[477] MAC/ns by shape/kernel (higher is faster):"
awk '/BenchmarkDotKernelAmort/{
  for(i=1;i<=NF;i++) if($i=="MAC/ns"){ printf "   %-40s %s MAC/ns\n", $1, $(i-1) }
}' "$BLOG" || true
if [ "$brc" -ne 0 ]; then
  echo "[477] NOTE: benchmark rc=$brc — MAC/ns table may be partial." >&2
fi

# ---- (4) optional full-model decode tok/s A/B (default vs amort) ------------------
SNAP="${FAK_QWEN_SNAPSHOT:-}"
GGUF="${FAK_QWEN_GGUF:-}"
MODEL_DEFAULT_JSON=""
MODEL_AMORT_JSON=""
MODEL_DEFAULT_TOK=""
MODEL_AMORT_TOK=""
MODEL_RATIO=""
MODEL_SOURCE_KIND=""
MODEL_SOURCE_LABEL=""
MODEL_ARGS=()
if [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
  MODEL_SOURCE_KIND="hf_snapshot"
  MODEL_SOURCE_LABEL="HF snapshot"
  MODEL_ARGS=(-hf "$SNAP" -lean)
elif [ -n "$GGUF" ] && [ -f "$GGUF" ]; then
  MODEL_SOURCE_KIND="gguf"
  MODEL_SOURCE_LABEL="GGUF"
  MODEL_ARGS=(-gguf "$GGUF" -lean)
fi
if [ -n "$MODEL_SOURCE_KIND" ]; then
  echo
  echo "== (4) full-model decode tok/s A/B: default vs FAK_QKERNEL=amort (Qwen2.5-1.5B, -lean) =="
  echo "[477] source: $MODEL_SOURCE_LABEL   (baseline to beat: 28.9 tok/s — M3-LLAMACPP-RESULTS.md sec. 3)"
  for tier in default amort; do
    MLOG="$(mktemp -t fak477m.$tier.XXXXXX.log)"
    MJSON="$TMP_DIR/modelbench-$tier.json"
    if [ "$tier" = default ]; then KENV=""; else KENV="amort"; fi
    set +e
    FAK_QKERNEL="$KENV" go run ./cmd/modelbench "${MODEL_ARGS[@]}" -decode-reps 6 -prefill-reps 1 -out "$MJSON" >"$MLOG" 2>&1
    set -e
    # modelbench emits JSON to $MJSON; pull decode.tok_per_sec for a quick read.
    dec="$(awk '
      /"decode": \{/ { in_decode=1; next }
      in_decode && /"tok_per_sec"/ {
        line=$0
        sub(/.*"tok_per_sec":[[:space:]]*/, "", line)
        sub(/,.*/, "", line)
        print line
        exit
      }
      in_decode && /^  \}/ { in_decode=0 }
    ' "$MJSON" 2>/dev/null || true)"
    if [ "$tier" = default ]; then
      MODEL_DEFAULT_JSON="$MJSON"
      MODEL_DEFAULT_TOK="$dec"
    else
      MODEL_AMORT_JSON="$MJSON"
      MODEL_AMORT_TOK="$dec"
    fi
    echo "[477] tier=$tier decode: ${dec:-<see $MLOG>}"
  done
  if [ -n "$MODEL_DEFAULT_TOK" ] && [ -n "$MODEL_AMORT_TOK" ]; then
    MODEL_RATIO="$(awk -v a="$MODEL_AMORT_TOK" -v b="$MODEL_DEFAULT_TOK" 'BEGIN { if (b > 0) printf "%.6f", a / b }')"
    echo "[477] full-model ratio: amort/default=${MODEL_RATIO}x"
  fi
  echo "[477] Record the amort decode tok/s vs 28.9 in docs/benchmarks/M3-LLAMACPP-RESULTS.md sec. 3"
  echo "[477] (acceptance-criteria 'uplift over 28.9 tok/s' line)."
else
  echo
  echo "== (4) full-model decode tok/s A/B: SKIPPED (set FAK_QWEN_SNAPSHOT=<qwen2.5-1.5b dir> or FAK_QWEN_GGUF=<qwen2.5-1.5b q8 gguf> to enable) =="
  echo "[477] kernel-level MAC/ns above is the direct amort-vs-default decode-dot witness; the"
  echo "[477] full-model tok/s headline needs the Qwen snapshot or GGUF. Repro is M3-LLAMACPP-RESULTS.md sec. 6."
fi

if [ -n "$OUT_JSON" ]; then
  mkdir -p "$(dirname "$OUT_JSON")"
  TS="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  GIT_REV="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
  GIT_BRANCH="$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"
  GO_VERSION="$(go version 2>/dev/null | sed 's/"/'\''/g' || echo unknown)"
  HOST="$(hostname 2>/dev/null || echo unknown)"
  {
    printf '{\n'
    printf '  "schema": "fak.arm64-qkernel-score.v1",\n'
    printf '  "issue": 477,\n'
    printf '  "generated_at": "%s",\n' "$TS"
    printf '  "host": "%s",\n' "$HOST"
    printf '  "arch": "%s",\n' "$ARCH"
    printf '  "git": {"rev": "%s", "branch": "%s"},\n' "$GIT_REV" "$GIT_BRANCH"
    printf '  "go": "%s",\n' "$GO_VERSION"
    printf '  "bench_time": "%s",\n' "$BENCHTIME"
    printf '  "gates": {\n'
    printf '    "amort_argmax_cosine": "pass",\n'
    printf '    "default_bit_identity_anchor": "pass",\n'
    printf '    "benchmark_return_code": %s\n' "$brc"
    printf '  },\n'
    printf '  "kernel_bench": [\n'
    awk '
      /BenchmarkDotKernelAmort/ && /MAC\/ns/ {
        name=$1
        sub(/-[0-9]+$/, "", name)
        n=split(name, parts, "/")
        shape=(n >= 2 ? parts[2] : "")
        kernel=(n >= 3 ? parts[3] : "")
        mac=""
        for (i=1; i<=NF; i++) if ($i == "MAC/ns") mac=$(i-1)
        if (shape != "" && kernel != "" && mac != "") {
          if (count++ > 0) printf ",\n"
          printf "    {\"shape\": \"%s\", \"kernel\": \"%s\", \"mac_per_ns\": %s}", shape, kernel, mac
        }
      }
      END { if (count > 0) printf "\n" }
    ' "$BLOG"
    printf '  ],\n'
    printf '  "kernel_ratios": [\n'
    awk '
      /BenchmarkDotKernelAmort/ && /MAC\/ns/ {
        name=$1
        sub(/-[0-9]+$/, "", name)
        n=split(name, parts, "/")
        shape=(n >= 2 ? parts[2] : "")
        kernel=(n >= 3 ? parts[3] : "")
        mac=""
        for (i=1; i<=NF; i++) if ($i == "MAC/ns") mac=$(i-1)
        if (shape != "" && kernel != "" && mac != "") {
          shapes[shape]=1
          val[shape SUBSEP kernel]=mac
        }
      }
      END {
        for (shape in shapes) {
          unroll=val[shape SUBSEP "unroll4"]
          amort=val[shape SUBSEP "amort"]
          asm=val[shape SUBSEP "asm"]
          if (unroll > 0 && amort > 0) {
            if (count++ > 0) printf ",\n"
            printf "    {\"shape\": \"%s\", \"amort_over_unroll4\": %.6f", shape, amort / unroll
            if (asm > 0) printf ", \"amort_over_asm\": %.6f", amort / asm
            printf "}"
          }
        }
        if (count > 0) printf "\n"
      }
    ' "$BLOG"
    printf '  ],\n'
    printf '  "full_model": {\n'
    if [ -n "$MODEL_SOURCE_KIND" ]; then
      printf '    "enabled": true,\n'
      printf '    "model": "qwen2.5-1.5b",\n'
      printf '    "source_kind": "%s",\n' "$MODEL_SOURCE_KIND"
      printf '    "default_decode_tok_per_sec": %s,\n' "${MODEL_DEFAULT_TOK:-null}"
      printf '    "amort_decode_tok_per_sec": %s,\n' "${MODEL_AMORT_TOK:-null}"
      printf '    "amort_over_default": %s\n' "${MODEL_RATIO:-null}"
    else
      printf '    "enabled": false,\n'
      printf '    "reason": "set FAK_QWEN_SNAPSHOT or FAK_QWEN_GGUF to score full-model tok/s"\n'
    fi
    printf '  }\n'
    printf '}\n'
  } > "$OUT_JSON"
  echo "[477] wrote scorecard: $OUT_JSON"
fi

rm -f "$LOG" "$BLOG"
echo
echo "[477] DONE: amortized decode kernel gate PASSED on arm64 (argmax-exact + cosine; anchor intact)."
exit 0
