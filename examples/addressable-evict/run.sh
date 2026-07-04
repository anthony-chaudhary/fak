#!/usr/bin/env bash
# addressable-evict: evict ONE poisoned span from a kept run and prove the
# post-eviction KV cache is BIT-FOR-BIT identical to a run that never saw the span
# (max|Δ| = 0), with a non-vacuous poison control (poison-vs-never > 0). No key, no
# model download, no GPU, no network — the witness runs on a synthetic model through
# the REAL quarantine gate. See README.md for what each line means.
set -euo pipefail

cd "$(dirname "$0")/../.."   # repo root: the Go module is the repository root

if ! command -v go >/dev/null 2>&1; then
  echo "addressable-evict: 'go' not found on PATH — install Go and rerun." >&2
  echo "  (Windows: run this from WSL/Git Bash, or call the go commands below directly.)" >&2
  exit 2
fi

echo "== 1/2 write-time evict == never-saw (the headline: max|Δ| = 0) =="
echo "   internal/kvmmu.TestWriteTimeEvictEqualsNeverSaw"
echo "   A poisoned tool result is ADMITTED through the real ctxmmu gate; the gate"
echo "   returns Quarantine, the bridge evicts that span write-time, and the query is"
echo "   prefilled AFTER eviction. evict-vs-never must be 0; poison-vs-never must be > 0."
go test ./internal/kvmmu -run 'TestWriteTimeEvictEqualsNeverSaw$' -count=1 -v 2>&1 \
  | grep -E 'max\|Δ\||PASS|FAIL|ok '
echo

echo "== 2/2 paged (non-contiguous) evict == contiguous evict, bit-for-bit =="
echo "   internal/model.TestPagedEvictBitIdenticalToContiguous"
echo "   The same middle span is evicted on a churned, non-contiguous page pool and on"
echo "   a contiguous cache; every layer's K/Kraw/V and the next-token logits are"
echo "   asserted float32-bits-equal (max|Δ| = 0 by construction)."
go test ./internal/model -run 'TestPagedEvictBitIdenticalToContiguous$' -count=1 -v 2>&1 \
  | grep -E 'PASS|FAIL|ok '
echo

echo "Both witnesses ran offline on a synthetic model. The load-bearing line is"
echo "'max|Δ| evict-vs-never = 0.000e+00' with 'poison-vs-never = 3.257e-01' (> 0):"
echo "removing the span leaves the cache bit-identical to never having seen it, and the"
echo "control proves the poison really did perturb the distribution (the test is not vacuous)."
echo
echo "For the same property against REAL Hugging Face weights (token-for-token vs HF's"
echo "never-saw run), see README.md — that rung needs the gitignored ~538MB oracle export."
