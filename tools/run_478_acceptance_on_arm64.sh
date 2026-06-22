#!/usr/bin/env bash
# run_478_acceptance_on_arm64.sh — the on-arm64 acceptance gate for issue #478
# (the arm64 NEON register-blocked Q8_0 GEMM tile: qgemm8tileNEON in
# internal/model/quant_arm64_tile.s, a 4-row × 4-token int32-SDOT register block — each 32-wide
# weight block streamed ONCE and SDOT'd against all 4 tokens, the float reduction deferred to the
# end of the K loop — dispatched by qGemm8TileInto in internal/model/quant_arm64_gemm.go with the
# out%4 row-remainder and P%4 token-remainder swept through the scalar reference qgemm8cell(...,4),
# the arm64 twin of amd64's qgemm8tile512). It is the register-blocked prefill GEMM tile
# M3-LLAMACPP-RESULTS.md sec. 4 names as "the next prefill lever, and the bigger one".
#
# REDUCTION ORDER / GATE POSTURE
#   The tile preserves the deferred-reduction order of qgemm8cell(...,4) exactly (SDOT → exact
#   SCVTF → broadcast-scale FMLA matching math.FMA, then the same (a0+a2)+(a1+a3) pairwise lane
#   fold), so it is BIT-IDENTICAL to the scalar reference — the stronger branch of #478's gate.
#   No argmax/cosine relaxation is needed, and no existing prefill bit-identity test is loosened:
#   the tile is pinned by exact-bits equality, and the SHIPPED default prefill path
#   (qGemm8Into, the 2×4 sweep) is pinned by the SAME test it always was.
#
# WHAT NODE THIS RUNS ON
#   An arm64 host with FEAT_DotProd (SDOT) — every Apple Silicon Mac (M1+) and ARMv8.4+ Linux.
#   The headline target is Apple M3 Pro (issue #478's measured node). It CANNOT run on the
#   win32/x86 host that authored this code: that host cannot execute arm64, so the tile's
#   correctness-gate RUN and the prefill tok/s measure are the explicit residual handed off here.
#   On the authoring host the code was confirmed to: build green by default
#   (`go build ./internal/model/`), CROSS-COMPILE for arm64
#   (`GOOS=darwin/linux GOARCH=arm64 go build ./internal/model/`), TEST-COMPILE for arm64
#   (`GOOS=darwin GOARCH=arm64 go test -c ./internal/model/`), and pass asmdecl vet
#   (`GOOS=darwin GOARCH=arm64 go vet ./internal/model/`). Only the device execution below needs
#   real arm64 hardware.
#
# WHAT IT PROVES
#   1. The tile correctness gate (bit-identity vs the deferred-reduction reference), all in
#      internal/model on darwin/arm64:
#        TestQGemm8TileNEONMatchesCell    — the isolated 4×4 qgemm8tileNEON tile, every cell
#                                           Float32bits-identical to qgemm8cell(...,4).
#        TestQGemm8IntoMatchesScalarNEON  — the FULL dispatchers (qGemm8TileInto's tile + row/token
#                                           remainders, AND the shipped-default qGemm8Into) both
#                                           bit-identical to the lane-4 scalar reference qGemm8scalar
#                                           at non-tile-aligned shapes. This is the prefill
#                                           bit-identity test #478 must NOT relax — it stays exact.
#        TestQMatRows4NEONMatchesCell     — the decode-GEMV anchor, still bit-identical (untouched).
#      A SKIP (no FEAT_DotProd) is NOT a pass.
#   2. The per-cell-vs-tile prefill A/B (MAC/ns, higher = faster) at Qwen2.5-1.5B projection shapes:
#        BenchmarkPrefillTileVsCell478    — shipped default (qGemm8Into, 2×4 sweep) vs the 4×4 tile
#                                           (qGemm8TileInto → qgemm8tileNEON), full-dispatch.
#        BenchmarkGemmKernelSingleCore    — per-cell unroll4 vs row4 (1×4), single-core kernel A/B.
#        BenchmarkGemmTile2x4SingleCore   — row4 (1×4) vs tile2x4 (2×4), single-core kernel A/B.
#      On Apple Silicon (M3 Pro) the shipped default has MEASURED faster than the 4×4 tile (which is
#      why the tile is opt-in via FAK_ARM_TILE=1, see qGemm8Into's doc comment): a faithful read is
#      "default ≥ tile4x4 on M3". On non-Apple arm64 the wider tile's amortization may pay off.
#   3. (Optional) the full-model prefill tok/s A/B — default vs FAK_ARM_TILE=1 — via modelbench,
#      when a Qwen2.5-1.5B HF snapshot is supplied, to re-measure against the 55.5 tok/s prefill
#      (pp256) baseline in docs/benchmarks/M3-LLAMACPP-RESULTS.md sec. 3 (the acceptance-criteria
#      uplift line).
#
# USAGE
#   bash tools/run_478_acceptance_on_arm64.sh
#   env knobs:
#     FAK_QWEN_SNAPSHOT=/path/to/qwen2.5-1.5b-instruct   # enables the optional full-model tok/s A/B
#     FAK_WORKERS=6                                       # cores for the full-model run (modelbench)
#     FAK_BENCHTIME=2s                                    # per-bench benchmark time (default 2s)
set -euo pipefail

# ---- locate the module root (this script lives in <root>/tools) -------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MOD_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PKG='./internal/model/'
GATE='TestQGemm8Tile478Gate|TestQGemm8TileNEONMatchesCell|TestQGemm8Tile4x2NEONMatchesCell|TestQGemm8Tile4x2IntoMatchesScalar|TestQGemm8IntoMatchesScalarNEON|TestQMatRows4NEONMatchesCell'
BENCH='^(BenchmarkPrefillTileVsCell478|BenchmarkGemmKernelSingleCore|BenchmarkGemmTile2x4SingleCore)$'
BENCHTIME="${FAK_BENCHTIME:-2s}"

echo "== #478 arm64 NEON register-blocked Q8_0 GEMM tile acceptance =="
echo "[478] repo root : $MOD_DIR"
echo "[478] kernel    : qgemm8tileNEON (internal/model/quant_arm64_tile.s), dispatched by"
echo "[478]             qGemm8TileInto (internal/model/quant_arm64_gemm.go); opt-in FAK_ARM_TILE=1"
echo "[478] gate      : bit-identity vs qgemm8cell(...,4) / qGemm8scalar ($PKG)"
echo "[478] benchmark : default (2×4 sweep) vs 4×4 tile, MAC/ns"

# ---- arch / feature sanity --------------------------------------------------------
ARCH="$(uname -m 2>/dev/null || echo unknown)"
case "$ARCH" in
  arm64|aarch64) ;;
  *) echo "[478] FAIL: host arch is '$ARCH', not arm64 — this gate must run on Apple Silicon / arm64." >&2
     echo "[478] (The tile was authored + cross-compiled + asmdecl-vetted on x86; this is the arm64 RUN residual.)" >&2
     exit 3 ;;
esac

cd "$MOD_DIR"
command -v go >/dev/null 2>&1 || { echo "[478] FAIL: no 'go' on PATH." >&2; exit 3; }

# ---- (1) the tile correctness gate (bit-identity) ---------------------------------
LOG="$(mktemp -t fak478.XXXXXX.log)"
echo
echo "== (1) tile correctness gate: bit-identity vs the lane-4 scalar reference =="
set +e
go test -count=1 -v -run "^(${GATE})$" "$PKG" 2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

if grep -aqE "no FEAT_DotProd|asimddp\) not available|--- SKIP" "$LOG"; then
  echo "[478] INCONCLUSIVE: a gate SKIPPED — no FEAT_DotProd on this arm64 part." >&2
  echo "[478] A skip is not a pass. Run on an SDOT-capable arm64 host (any Apple Silicon)." >&2
  rm -f "$LOG"
  exit 4
fi
if [ "$rc" -ne 0 ]; then
  echo "[478] FAIL: tile correctness gate did not pass (go test rc=$rc)." >&2
  rm -f "$LOG"
  exit "$rc"
fi
echo "[478] PASS: qgemm8tileNEON + qGemm8TileInto are bit-identical to qgemm8cell(...,4) /"
echo "[478]       qGemm8scalar, and the shipped-default prefill path is unchanged."

# ---- (2) per-cell-vs-tile prefill A/B (MAC/ns) ------------------------------------
echo
echo "== (2) per-cell-vs-tile prefill throughput: default (2×4 sweep) vs 4×4 tile =="
echo "== (single-core kernel A/Bs included; pass -cpu 1 reads as raw kernel, default = parFor agg) =="
BLOG="$(mktemp -t fak478bench.XXXXXX.log)"
set +e
go test -count=1 -run '^$' -bench "$BENCH" -benchtime="$BENCHTIME" "$PKG" 2>&1 | tee "$BLOG"
brc=${PIPESTATUS[0]}
set -e
echo
echo "[478] MAC/ns by shape/kernel (higher is faster; 'default' vs 'tile4x4' is the #478 lever):"
awk '/MAC\/ns/ && /Benchmark/{
  for(i=1;i<=NF;i++) if($i=="MAC/ns"){ printf "   %-44s %s MAC/ns\n", $1, $(i-1) }
}' "$BLOG" || true
if [ "$brc" -ne 0 ]; then
  echo "[478] NOTE: benchmark rc=$brc — MAC/ns table may be partial." >&2
fi
echo "[478] On M3 Pro the shipped default is expected to match/beat the 4×4 tile (tile is opt-in,"
echo "[478] FAK_ARM_TILE=1). Record the default-vs-tile4x4 delta in docs/benchmarks/M3-LLAMACPP-RESULTS.md sec. 4."

# ---- (3) optional full-model prefill tok/s A/B (default vs FAK_ARM_TILE=1) ---------
SNAP="${FAK_QWEN_SNAPSHOT:-}"
if [ -n "$SNAP" ] && [ -d "$SNAP" ]; then
  echo
  echo "== (3) full-model prefill tok/s A/B: default vs FAK_ARM_TILE=1 (Qwen2.5-1.5B Q8_0, pp256) =="
  echo "[478] snapshot: $SNAP   (baseline to beat: 55.5 tok/s — M3-LLAMACPP-RESULTS.md sec. 3)"
  WK="${FAK_WORKERS:-}"
  for mode in default tile; do
    MLOG="$(mktemp -t fak478m.$mode.XXXXXX.log)"
    if [ "$mode" = default ]; then TENV=""; else TENV="1"; fi
    set +e
    FAK_ARM_TILE="$TENV" FAK_WORKERS="$WK" \
      go run ./cmd/modelbench -hf "$SNAP" -lean -prefill-sizes 256 -prefill-reps 5 -decode-reps 1 >"$MLOG" 2>&1
    set -e
    # modelbench emits JSON; pull a prefill tok/s-ish field for a quick read (full JSON in $MLOG).
    pre="$(grep -aoE '"prefill[^,}]*tok[^,}]*"[: ]*[0-9.]+' "$MLOG" | head -1 || true)"
    echo "[478] mode=$mode (FAK_ARM_TILE='${TENV}') prefill: ${pre:-<see $MLOG>}"
  done
  echo "[478] Record the prefill tok/s for both modes vs 55.5 in docs/benchmarks/M3-LLAMACPP-RESULTS.md sec. 3/4"
  echo "[478] (acceptance-criteria 'real uplift over 55.5 tok/s' line)."
else
  echo
  echo "== (3) full-model prefill tok/s A/B: SKIPPED (set FAK_QWEN_SNAPSHOT=<qwen2.5-1.5b dir> to enable) =="
  echo "[478] kernel-level MAC/ns above is the direct default-vs-tile prefill witness; the full-model"
  echo "[478] tok/s headline needs the Qwen snapshot. Repro is M3-LLAMACPP-RESULTS.md sec. 6."
fi

rm -f "$LOG" "$BLOG"
echo
echo "[478] DONE: arm64 GEMM tile gate PASSED on arm64 (bit-identical to the scalar reference; default path intact)."
exit 0
