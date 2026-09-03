#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

LOG_DIR="$REPO/.fak/nightrun"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/overnight-check.log"

log() {
    echo "[$(date -u +"%Y-%m-%dT%H:%M:%SZ")] $*" | tee -a "$LOG_FILE"
}

log "=== Starting overnight sync & health check ==="

# 1. Git fetch and check sync status
git fetch origin main 2>&1 | tee -a "$LOG_FILE" || true
BEHIND=$(git rev-list --count HEAD..origin/main || echo 0)
AHEAD=$(git rev-list --count origin/main..HEAD || echo 0)
log "Git sync status: behind=$BEHIND ahead=$AHEAD"

if [ "$BEHIND" -gt 0 ]; then
    log "Remote has $BEHIND new commit(s). Integrating in place..."
    if git merge origin/main 2>&1 | tee -a "$LOG_FILE"; then
        log "Merge succeeded."
    else
        log "WARNING: Merge encountered conflicts or was stopped."
    fi
fi

if [ "$AHEAD" -gt 0 ]; then
    log "Local has $AHEAD commit(s) ahead. Pushing to origin/main..."
    if git push origin main 2>&1 | tee -a "$LOG_FILE"; then
        log "Push succeeded."
    else
        log "WARNING: Push rejected or failed."
    fi
fi

# 2. fak doctor
log "Running fak doctor..."
./fak doctor 2>&1 | tee -a "$LOG_FILE" || true

# 3. fak progress
log "Running fak progress..."
./fak progress 2>&1 | tee -a "$LOG_FILE" || true

# 4. Summary
HEAD_SHA=$(git rev-parse --short HEAD)
log "=== Overnight check complete: HEAD is at $HEAD_SHA ==="
