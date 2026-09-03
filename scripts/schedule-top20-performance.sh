#!/usr/bin/env bash
# schedule-top20-performance.sh — Scheduled runner for the Top 20 within-scope performance items
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

DELAY_SECONDS="${1:-10800}" # Default: 10800s (3 hours)
LOG_DIR="$REPO/_scratch/performance_top20"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_$(date -u +"%Y%m%d_%H%M%SZ").log"

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$LOG_FILE"
}

log "=== FAK Top 20 Performance Work Wave: Scheduled Execution ==="
log "Working repository: $REPO"
log "Configured delay: ${DELAY_SECONDS}s ($((DELAY_SECONDS / 3600)) hours, $(( (DELAY_SECONDS % 3600) / 60 )) minutes)"

if [ "$DELAY_SECONDS" -gt 0 ]; then
    TARGET_EPOCH=$(( $(date +%s) + DELAY_SECONDS ))
    TARGET_DATE=$(date -r "$TARGET_EPOCH" -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || date -d "@$TARGET_EPOCH" -u +"%Y-%m-%dT%H:%M:%SZ" 2>/dev/null || echo "in ${DELAY_SECONDS}s")
    log "Execution scheduled for: $TARGET_DATE"
    log "Sleeping for ${DELAY_SECONDS} seconds until start window..."
    sleep "$DELAY_SECONDS"
    log "Sleep elapsed. Waking up to begin Top 20 Performance execution wave."
else
    log "Zero delay specified; executing immediately."
fi

log "=== Stage 1: Preflight & Capacity Verification ==="
python3 tools/dispatch_status.py --fast 2>&1 | tee -a "$LOG_FILE" || true
fak accounts status 2>&1 | tee -a "$LOG_FILE" || true

log "=== Stage 2: Validating Top 20 Within-Scope Performance Test Harnesses ==="
# Wave A: Independent package leaves
log "Running Wave A tests: independent package performance items..."
go test -v ./internal/cacheprice -run TestDedupNetSavings 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/cachesweep 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/sweepcert 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/turnbench 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/nativebench 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/benchauthority 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/findingsink 2>&1 | tee -a "$LOG_FILE"
go test -v ./internal/agenticbench 2>&1 | tee -a "$LOG_FILE"

# Wave B: Compute leaf pipeline
log "Running Wave B tests: internal/compute performance operators..."
go test -v ./internal/compute -run TestBoundedAsymmetricForget 2>&1 | tee -a "$LOG_FILE"

log "=== Stage 3: Summary of Top 20 Execution Wave ==="
log "Completed performance validation run. All logs captured in $LOG_FILE"
