#!/usr/bin/env bash
# eraser_sweep.sh — measure the FINER ERASER on the real kernel and roll the headline
# columns into one reproducible CSV (experiments/fleet/eraser/eraser-summary.csv). It
# regenerates every number cited in fak/FLEET-ERASER-RESULTS.md:
#   Axis 1  write-rate crossover per eraser (global/namespace/resource) at T=30, A=50.
#   Axis 2  pool sensitivity of the resource eraser at long sessions (T=120, A=24) with
#           genuinely-distinct routes, where a >8 catalog actually pays off (the ~1/pool
#           "100x" lever; the T=30 axis saturates at the coupon-collector limit).
#
# Run from the repo root:  wsl bash ./tools/eraser_sweep.sh
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../fak"
export GOTOOLCHAIN="${GOTOOLCHAIN:-auto}"

OUT=experiments/fleet/eraser
mkdir -p "$OUT"
BIN="$(mktemp -d)/fleetbench"
echo "eraser_sweep: building fleetbench (go=$(go version | awk '{print $3}'))"
go build -o "$BIN" ./cmd/fleetbench

# --- Axis 1: write-rate crossover per eraser, at the canonical T=30, A=50 cell. ---
WR="0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.05 0.1 0.15 0.2 0.3 0.5 0.7"
TRIALS=32
echo "eraser_sweep: write-rate crossover (T=30 A=50 trials=$TRIALS)"
for g in global namespace resource; do
  for w in $WR; do
    "$BIN" --profile read-heavy --granularity "$g" --write-rate "$w" \
      --turns 30 --agents 50 --trials "$TRIALS" \
      --out "$OUT/cross-$g-w$w.json" --csv "$OUT/cross-$g-w$w.csv" 2>/dev/null
  done
done

# --- Axis 2: pool sensitivity at LONG sessions (distinct routes past 8). ---
POOLS="8 16 32 64 128"
WFIX=0.02
echo "eraser_sweep: pool sensitivity (write-rate=$WFIX T=120 A=24 trials=16)"
for g in global resource; do
  for p in $POOLS; do
    "$BIN" --profile read-heavy --granularity "$g" --write-rate "$WFIX" --shared-pool "$p" \
      --turns 120 --agents 24 --trials 16 \
      --out "$OUT/poolDist-$g-p$p.json" --csv "$OUT/poolDist-$g-p$p.csv" 2>/dev/null
  done
done

# --- Roll up the headline columns into one tidy CSV (eraser, axis, x, cross_uplift). ---
SUM="$OUT/eraser-summary.csv"
echo "eraser,axis,x,turns,agents,shared_saved_p50,isolated_saved_p50,cross_uplift_p50,cross_uplift_mean" > "$SUM"
row() { # file eraser axis x
  tail -n +2 "$1" | while IFS=, read -r turns agents calls ss is cu spa ssm cum rest; do
    echo "$2,$3,$4,$turns,$agents,$ss,$is,$cu,$cum" >> "$SUM"
  done
}
for g in global namespace resource; do for w in $WR; do row "$OUT/cross-$g-w$w.csv" "$g" "write_rate" "$w"; done; done
for g in global resource; do for p in $POOLS; do row "$OUT/poolDist-$g-p$p.csv" "$g" "pool" "$p"; done; done

echo "eraser_sweep: done -> $SUM"
