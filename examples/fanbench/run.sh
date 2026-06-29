#!/bin/bash
# run.sh — one command to run the fak fanbench adoption walkthrough (#331).
#
# It drives the REAL benchmark harness, `cmd/fanbench` (engine internal/turnbench/fanout.go),
# which sweeps the ONE-MASTER-GOAL -> N-SUBAGENT fan-out (the orchestrator-worker topology)
# and reports, per fan-out width N:
#
#   - cross_uplift = shared - isolated : the MEASURED cross-agent tool-result dedup the
#     interleaved fan-out gets that the same sub-agents run APART cannot (a real k.Syscall
#     path-swap, the same ablation discipline as fleetbench); and
#   - prefix_tokens_saved = (N-1)*P : the EXACT shared-prefix KV-reuse geometry the kernel
#     never redoes (NewBatchFromPrefix prefills the master-goal prefix once + clones it).
#   - tax_clawed_back, parallel_speedup : the MODELED cost-model half, reported APART.
#
# The default sweep is the research profile at N=1,8,64,512 with a 2048-token shared prefix.
# N=1 gives EXACTLY 0 cross_uplift (a lone worker has no sibling) — the anti-inflation
# control. `--profile no-share` is the partner control: 0 uplift at every N.
#
#   ./run.sh                              # sweep N=1,8,64,512 (research, P=2048)
#   ./run.sh --scale --grid canonical     # the D-001 acceptance ladder (N=1/100/500/1000)
#   ./run.sh --profile no-share           # the anti-inflation control: 0 uplift at every N
#   ./run.sh --prefixes big --model-config examples/fanbench/sample-model-config.json
#   GO=/path/to/go ./run.sh               # use a specific Go toolchain
#
# Any extra args are passed straight through to fanbench (so the flags above compose with
# the defaults; an explicit --agents / --profile / --grid overrides this script's default).
#
# Requires the Go toolchain (the bench is a Go program: cmd/fanbench — it is NOT in the fak
# binary; `fak fanbench` is an unknown verb). If Go is not installed, the script prints the
# exact command + the published numbers and exits 0 — the walkthrough still stands; see
# README.md "Run it without a Go toolchain".
set -e
set -u
set -o pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
FAK_DIR="$(cd "$HERE/../.." && pwd)"           # examples/fanbench -> fak/
OUT="$HERE/fanout.json"
CSV="$HERE/fanout.csv"
log(){ printf '\033[36m[demo]\033[0m %s\n' "$*" >&2; }

# This script's defaults. An explicit --agents/--profile/--grid in "$@" overrides them
# (fanbench's flag parser takes the last value), so the extra args compose cleanly.
DEFAULTS=(--profile research --agents 1,8,64,512 --prefix 2048)
EXTRA=("$@")

GO_BIN="${GO:-go}"
if ! command -v "$GO_BIN" >/dev/null 2>&1; then
  log "Go toolchain not found — the bench (cmd/fanbench) is a Go program and is NOT in the fak binary."
  log "Install Go from https://go.dev/dl, or read the published numbers in this walkthrough:"
  echo
  echo "  To run it yourself once Go is installed:"
  echo "    cd $FAK_DIR"
  echo "    go run ./cmd/fanbench --profile research --agents 1,8,64,512 --prefix 2048 \\"
  echo "      --out examples/fanbench/fanout.json --csv examples/fanbench/fanout.csv"
  echo
  echo "  The published research-goal cross_uplift curve (docs/benchmarks/FANOUT-BENCH-RESULTS.md §1):"
  echo "    N=1   cross_uplift  0    (the anti-inflation control — a lone worker has no sibling)"
  echo "    N=8   cross_uplift +6"
  echo "    N=64  cross_uplift +58   prefix_tokens_saved=(N-1)*P=129,024"
  echo "    N=512 cross_uplift +501  prefix_tokens_saved=(N-1)*P=1,046,528"
  echo "    tax_clawed_back (MODELED) saturates ~61.7%; this is reuse-vs-no-reuse, NOT a"
  echo "    head-to-head win over a tuned shared-prefix engine (SGLang/RadixAttention/vLLM-APC)."
  echo
  log "walkthrough printed (live witness UNVERIFIED here — no Go toolchain). See EXAMPLE-OUTPUT.md for a captured run."
  exit 0
fi

log "running cmd/fanbench: one master goal -> N sub-agents, swept N=1,8,64,512 (research profile, P=2048)"
log "headline = cross_uplift (MEASURED shared-vs-isolated dedup) + (N-1)*P prefix reuse; cost model reported apart"
echo
( cd "$FAK_DIR" && "$GO_BIN" run ./cmd/fanbench "${DEFAULTS[@]}" --out "$OUT" --csv "$CSV" "${EXTRA[@]}" )
echo
log "wrote $OUT and $CSV — one row per (prefix, N, sub-turns) cell"
