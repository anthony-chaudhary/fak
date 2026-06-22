#!/usr/bin/env bash
# run_476_acceptance_on_arm64.sh — the on-arm64 acceptance gate for issue #476
# (NEON activation quantization: quantizeVecQ8 / quantizeBatchPanelInto behind FEAT_DotProd).
#
# WHAT NODE THIS RUNS ON
#   Any arm64 host whose CPU has FEAT_DotProd (asimddp):
#     • a darwin/arm64 Mac — every Apple Silicon part (M1+) has FEAT_DotProd, so the NEON
#       quantizer is always live there; OR
#     • a linux/arm64 box with asimddp (ARMv8.4+; check `grep asimddp /proc/cpuinfo`).
#   It CANNOT run on the win32/x86 dev host that produced this code: that host is amd64, so
#   the arm64 NEON kernel (quant_quantize_arm64.s) is not even compiled into its binary and
#   quantizeRowQ8 dispatches the AVX-512/scalar path instead. The Go of the NEON quantizer
#   already CROSS-COMPILES on the dev host (`GOOS=darwin GOARCH=arm64 go build ./internal/model/`
#   and the linux/arm64 form are green); only the NEON equivalence RUN + the scalar-vs-NEON
#   micro-bench below need real arm64 silicon. That RUN is the explicit residual handed off here.
#
# WHAT IT PROVES
#   1. Equivalence (bit-match): the dispatched quantizeRowQ8 — the NEON kernel on a FEAT_DotProd
#      part — produces BIT-IDENTICAL codes and scale bits to the scalar reference quantizeRowQ8scalar,
#      over zero blocks, denormals, all-negative blocks, large dynamic range, and exact round-half
#      boundaries. Tests:
#        TestQuantizeRowAsmMatchesScalar                         (quant_quantize_test.go)
#        TestQuantizeVecQ8MatchesScalar                          (the decode-facing q8Vec)
#        TestQuantizeVecQ8IntoReuseMatchesScalarAndClearsZeroBlocks
#      Because these compare the *dispatched* quantizer to scalar, a host WITHOUT FEAT_DotProd would
#      pass them trivially (scalar==scalar); this script surfaces that case as INCONCLUSIVE for the
#      NEON claim (see below), so a green here means NEON==scalar on real silicon.
#   2. Speedup: BenchmarkQuantizeRowScalarVsNEON times quantizeRowQ8scalar vs quantizeRowAsmNEON in
#      one process at the real decode widths (576, 1536) and this script prints the scalar/NEON
#      ns/op ratio — the activation-quant term #476 set out to size.
#   Exits 0 only if the equivalence tests PASS *and* the NEON kernel was actually exercised
#   (FEAT_DotProd present); non-zero on any build/test failure, and INCONCLUSIVE (exit 4) if the
#   node lacks FEAT_DotProd so the NEON path never ran.
#
# USAGE
#   bash tools/run_476_acceptance_on_arm64.sh
#   env knobs:
#     FAK_BENCHTIME=200ms   per-sub-benchmark wall time (default 200ms)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG='./internal/model/'
EQUIV_RE='^(TestQuantizeRowAsmMatchesScalar|TestQuantizeVecQ8MatchesScalar|TestQuantizeVecQ8IntoReuseMatchesScalarAndClearsZeroBlocks)$'
BENCH_RE='^BenchmarkQuantizeRowScalarVsNEON$'
BENCHTIME="${FAK_BENCHTIME:-200ms}"

echo "== #476 on-arm64 NEON activation-quant acceptance =="
echo "[476] repo root : $MOD_DIR"
echo "[476] package   : $PKG"

# ---- this is an arm64-only gate: the NEON kernel does not exist on other arches ---
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
  arm64|aarch64) echo "[476] arch      : $ARCH (arm64 — NEON quantizer compiled in)";;
  *)
    echo "[476] FAIL: host arch is '$ARCH', not arm64 — the NEON quantizer (quant_quantize_arm64.s)" >&2
    echo "[476] is not built here and quantizeRowQ8 dispatches a different kernel. Run on arm64." >&2
    exit 3
    ;;
esac

if ! command -v go >/dev/null 2>&1; then
  echo "[476] FAIL: no 'go' on PATH." >&2
  exit 3
fi
echo "[476] go        : $(go version)"

# ---- 1) equivalence (bit-match) ---------------------------------------------------
echo
echo "[476] (1/2) equivalence: NEON quantizeRowQ8 == scalar, bit-for-bit ..."
set +e
( cd "$MOD_DIR" && go test -count=1 -v -run "$EQUIV_RE" "$PKG" )
equiv_rc=$?
set -e
if [ "$equiv_rc" -ne 0 ]; then
  echo "[476] FAIL: the equivalence test did not pass (go test rc=$equiv_rc)." >&2
  echo "[476] The NEON quantizer is NOT bit-identical to the scalar reference on this node." >&2
  exit "$equiv_rc"
fi
echo "[476] equivalence PASS."

# ---- 2) scalar-vs-NEON micro-bench ------------------------------------------------
echo
echo "[476] (2/2) micro-bench: scalar vs NEON (-benchtime=$BENCHTIME) ..."
LOG="$(mktemp -t fak476.XXXXXX.log)"
set +e
( cd "$MOD_DIR" && go test -run '^$' -bench "$BENCH_RE" -benchmem -benchtime="$BENCHTIME" "$PKG" ) 2>&1 | tee "$LOG"
bench_rc=${PIPESTATUS[0]}
set -e
if [ "$bench_rc" -ne 0 ]; then
  echo "[476] FAIL: the micro-bench did not build/run (go test rc=$bench_rc)." >&2
  rm -f "$LOG"
  exit "$bench_rc"
fi

# Did the NEON sub-benchmark actually run, or skip for lack of FEAT_DotProd?
if grep -aqE "FEAT_DotProd .* not available|--- SKIP: BenchmarkQuantizeRowScalarVsNEON/neon" "$LOG"; then
  echo
  echo "[476] INCONCLUSIVE: this arm64 node lacks FEAT_DotProd (asimddp) — the NEON quantizer was" >&2
  echo "[476] never dispatched, so the equivalence test above compared scalar-to-scalar and the" >&2
  echo "[476] speedup is unmeasured. Run on a FEAT_DotProd part (any Apple Silicon M1+)." >&2
  rm -f "$LOG"
  exit 4
fi

# ns/op for a sub-benchmark name pattern (the field just before "ns/op").
ns_of() { awk -v pat="$1" '$0 ~ pat { for (i=1;i<=NF;i++) if ($i=="ns/op") { print $(i-1); exit } }' "$2"; }

echo
echo "== scalar-vs-NEON activation-quant delta =="
for w in 576 1536; do
  s_ns="$(ns_of "/scalar/width${w}" "$LOG")"
  n_ns="$(ns_of "/neon/width${w}" "$LOG")"
  if [ -n "$s_ns" ] && [ -n "$n_ns" ]; then
    ratio="$(awk -v s="$s_ns" -v n="$n_ns" 'BEGIN{ if (n>0) printf "%.2f", s/n; else print "n/a" }')"
    echo "[476] width ${w}: scalar ${s_ns} ns/op  vs  NEON ${n_ns} ns/op  =>  ${ratio}x"
  else
    echo "[476] width ${w}: could not parse ns/op (scalar='${s_ns}' neon='${n_ns}') — see bench output above."
  fi
done

rm -f "$LOG"
echo
echo "[476] PASS: NEON quantizer is bit-identical to scalar AND was exercised on FEAT_DotProd silicon."
echo "[476] (The ratios above are this node's measured scalar-vs-NEON activation-quant speedup.)"
exit 0
