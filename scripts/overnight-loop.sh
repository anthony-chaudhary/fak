#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO"

INTERVAL="${INTERVAL:-3600}"

echo "Starting overnight check loop (interval: ${INTERVAL}s)..."
while true; do
    "$REPO/scripts/overnight-check.sh" || true
    echo "Check tick finished at $(date -u +"%Y-%m-%dT%H:%M:%SZ"). Sleeping ${INTERVAL}s until next check..."
    sleep "$INTERVAL"
done
