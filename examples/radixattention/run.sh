#!/bin/bash
# run.sh — one command to run the fak RadixAttention adoption walkthrough (#322).
#
# It drives the REAL benchmark harness, `cmd/radixbench` (internal/radixkv — fak's
# rebuild of SGLang's RadixAttention over the kernel-owned KV cache), over the four
# bundled sample workloads in sample-workload/ — one JSON per SGLang shape
# (few-shot / multi-turn-chat / tree-of-thought / agents) — and prints, per shape:
#
#   - the CACHE HIT RATE (the fraction of prompt tokens reused, SGLang's own
#     hardware-/model-independent headline axis); and
#   - the cross-subtree win: how much more the radix TREE reuses than declaring the
#     one global shared prefix (fak's pre-radix path) — the thing the tree adds.
#
# It then runs the harness's POLICY-EVICTION witness (verdict-driven span eviction,
# the capability an opportunistic LRU radix cache structurally cannot offer).
#
# The bundled workloads are deliberately SMALL and self-contained so the example needs
# no model and no network: their token ids are < 256 so the default synthetic model can
# embed them and the live wall-clock arm runs. The hit-rate accounting is exact integer
# token bookkeeping — a function of the token structure + the matching algorithm only —
# so it is identical on any model. The published 77.2–88.2% band (docs/benchmarks/
# RADIXATTENTION-RESULTS.md) comes from the FULL synthetic SGLang shapes (run radixbench
# with NO -workload flag); these small fixtures illustrate the same reuse SHAPE at a
# size a reader can verify by hand.
#
#   ./run.sh                       # run the bench over the four bundled workloads
#   ./run.sh --out report.json     # also write the full JSON report
#   GO=/path/to/go ./run.sh        # use a specific Go toolchain
#
# Requires the Go toolchain (the bench is a Go program: cmd/radixbench). If Go is not
# installed, the script prints the exact command + the published numbers and exits 0 —
# the walkthrough still stands; see README.md "Run it without a Go toolchain".
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/radixattention -> fak/
WL="$HERE/sample-workload"
WORKLOADS="$WL/few-shot.json,$WL/multi-turn-chat.json,$WL/tree-of-thought.json,$WL/agents.json"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

# Pass any extra args straight through to radixbench (e.g. --out report.json, --only few-shot).
EXTRA=("$@")

GO_BIN="${GO:-go}"
if ! command -v "$GO_BIN" >/dev/null 2>&1; then
  log "Go toolchain not found — the bench (cmd/radixbench) is a Go program and is not in the fak binary."
  log "Install Go from https://go.dev/dl, or read the published numbers in this walkthrough:"
  echo
  echo "  To run it yourself once Go is installed:"
  echo "    cd $FAK_DIR"
  echo "    go run ./cmd/radixbench -workload \\"
  echo "      examples/radixattention/sample-workload/few-shot.json,\\"
  echo "      examples/radixattention/sample-workload/multi-turn-chat.json,\\"
  echo "      examples/radixattention/sample-workload/tree-of-thought.json,\\"
  echo "      examples/radixattention/sample-workload/agents.json"
  echo
  echo "  The published full-shape numbers (docs/benchmarks/RADIXATTENTION-RESULTS.md):"
  echo "    few-shot 88.2%  multi-turn-chat 79.5%  tree-of-thought 77.2%  agents 86.7%"
  echo "    all inside SGLang's verified 50-99% hit-rate band; reuse-through-a-split"
  echo "    proven bit-identical to recompute (max|Δ|=0)."
  echo
  log "walkthrough printed (live witness UNVERIFIED here — no Go toolchain). See EXAMPLE-OUTPUT.md for a captured run."
  exit 0
fi

log "running cmd/radixbench over the four bundled workloads (sample-workload/*.json)"
log "axis = cache hit rate (SGLang's hardware-/model-independent headline metric)"
echo
( cd "$FAK_DIR" && "$GO_BIN" run ./cmd/radixbench -workload "$WORKLOADS" "${EXTRA[@]}" )
