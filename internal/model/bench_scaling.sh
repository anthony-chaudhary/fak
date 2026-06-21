#!/usr/bin/env bash
# Measure fak decode + prefill latency vs worker count in ONE environment (so the
# parallelism speedup is confound-free: same binary, same machine, same runtime).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."   # -> fak/
export GOTOOLCHAIN=auto
go build -o /tmp/mbench ./cmd/modelbench
echo "workers,decode_ms,prefill16_ms,prefill64_ms,prefill256_ms"
for W in 1 2 4 8 16 32; do
  out=$(FAK_WORKERS=$W /tmp/mbench -prefill-reps 2 -decode-reps 3 2>&1)
  dec=$(echo "$out" | grep -aoE 'decode: [0-9.]+' | grep -aoE '[0-9.]+')
  p16=$(echo "$out" | grep -aoE 'prefill P=16: [0-9.]+' | grep -aoE '[0-9.]+$')
  p64=$(echo "$out" | grep -aoE 'prefill P=64: [0-9.]+' | grep -aoE '[0-9.]+$')
  p256=$(echo "$out" | grep -aoE 'prefill P=256: [0-9.]+' | grep -aoE '[0-9.]+$')
  echo "$W,$dec,$p16,$p64,$p256"
done
