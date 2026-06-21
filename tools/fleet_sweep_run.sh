#!/usr/bin/env bash
# fleet_sweep_run.sh — drive the full 2-D turn-tax fleet sweep (turns × agents)
# plus its companion axes, writing every artifact under fak/experiments/fleet/.
#
# Each invocation of the `fleetbench` binary is single-threaded and uses its OWN
# process-global vDSO world, so the runs are mutually isolated and safe to run
# concurrently. This box has 16 cores; we launch the three big 50×50 surfaces in
# the background and fan the cheap axis scans out with a concurrency cap, then wait.
#
# Determinism: every run is seeded, so re-running reproduces the identical surface.
#
# Usage:  wsl bash ./tools/fleet_sweep_run.sh            # full run (~1h wall-clock)
#         TRIALS=32 wsl bash ./tools/fleet_sweep_run.sh  # faster, looser order stats
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../fak"
export GOTOOLCHAIN=auto

OUT="experiments/fleet"
mkdir -p "$OUT"
BIN="/tmp/fleetbench.$$"
TRIALS="${TRIALS:-64}"          # headline surface trial count
CTRIALS="${CTRIALS:-32}"        # companion-surface trial count
ATRIALS="${ATRIALS:-48}"        # axis-scan trial count

echo "fleet_sweep_run: building fleetbench ..."
go build -o "$BIN" ./cmd/fleetbench
echo "fleet_sweep_run: built. TRIALS=$TRIALS CTRIALS=$CTRIALS ATRIALS=$ATRIALS  out=$OUT"
date -u +"start: %Y-%m-%dT%H:%M:%SZ"

# ----- the three big 50×50 surfaces, in the background -------------------------
# 1) HEADLINE: read-only fleet — the clean, positive, saturating turns×agents law.
"$BIN" --profile read-heavy --turn-max 50 --agent-max 50 --grid full \
  --trials "$TRIALS" --out "$OUT/readfleet-50x50.json" --csv "$OUT/readfleet-50x50.csv" \
  >"$OUT/readfleet-50x50.log" 2>&1 &
PID_HEAD=$!

# 2) CONTROL: no-share, no-write — cross uplift MUST be ~0 across the whole surface.
"$BIN" --profile no-share --turn-max 50 --agent-max 50 --grid full \
  --trials "$CTRIALS" --out "$OUT/noshare-50x50.json" --csv "$OUT/noshare-50x50.csv" \
  >"$OUT/noshare-50x50.log" 2>&1 &
PID_CTRL=$!

# 3) WRITE-HEAVY: the negative surface — global-world-bump erodes the shared cache.
"$BIN" --profile write-heavy --turn-max 50 --agent-max 50 --grid full \
  --trials "$CTRIALS" --out "$OUT/writeheavy-50x50.json" --csv "$OUT/writeheavy-50x50.csv" \
  >"$OUT/writeheavy-50x50.log" 2>&1 &
PID_WH=$!

# ----- axis scans (cheap), fanned out with a concurrency cap -------------------
run_capped() { # run a command, capping background jobs to $1
  local cap=$1; shift
  while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do wait -n; done
  "$@" &
}

# WRITE-RATE axis: cross-uplift vs write_rate over the agent grid (T=30). One run
# per write rate; finely sampled near the crossover (~0.5–1%).
for w in 0.0 0.0025 0.005 0.0075 0.01 0.015 0.02 0.03 0.05 0.10 0.20 0.30; do
  run_capped 10 "$BIN" --profile read-heavy --write-rate "$w" \
    --turns 30 --agent-max 50 --grid full --trials "$ATRIALS" \
    --out "$OUT/writeaxis-w$w.json" --csv "$OUT/writeaxis-w$w.csv"
done

# SHARED-POOL axis: how the catalog size moves the cross-agent saturation knee
# (read-only fleet, T=30, agents 1..50). One run per pool size.
for pool in 1 2 4 8 16 32 64 128; do
  run_capped 10 "$BIN" --profile read-heavy --shared-pool "$pool" \
    --turns 30 --agent-max 50 --grid full --trials "$ATRIALS" \
    --out "$OUT/poolaxis-p$pool.json" --csv "$OUT/poolaxis-p$pool.csv"
done

echo "fleet_sweep_run: axis scans launched; waiting on all jobs ..."
wait
date -u +"done: %Y-%m-%dT%H:%M:%SZ"
echo "fleet_sweep_run: ALL DONE. artifacts in $OUT/"
ls -la "$OUT"/*.csv
rm -f "$BIN"
