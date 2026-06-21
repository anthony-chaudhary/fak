#!/usr/bin/env bash
# Quick write-rate crossover scan for the fleet turn-tax sweep. Builds the binary
# and prints the cross-uplift at one (T,A) cell across a write-rate ladder, so we
# can see where cross-agent dedup stops paying for the global-world-bump it causes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../fak"
export GOTOOLCHAIN=auto
go build -o /tmp/fb ./cmd/fleetbench
echo "write_rate  shared  isolated  cross  (T=30 A=50 trials=24)"
for w in 0.0 0.01 0.02 0.05 0.10 0.20 0.30; do
  /tmp/fb --profile read-heavy --write-rate "$w" --turns 30 --agents 50 \
    --trials 24 --out "/tmp/w_$w.json" --csv "/tmp/w_$w.csv" >/dev/null 2>&1
  row=$(tail -1 "/tmp/w_$w.csv")
  shared=$(echo "$row" | cut -d, -f4)
  isolated=$(echo "$row" | cut -d, -f5)
  cross=$(echo "$row" | cut -d, -f6)
  printf "%-10s  %-6s  %-8s  %s\n" "$w" "$shared" "$isolated" "$cross"
done
