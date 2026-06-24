# Ultra-long-context live wall-clock anchor — #524 — turnkey runbook + named blocker

**Status:** host-tractable scaffold already SHIPPED; the live ≥100k wall-clock acceptance is
**deferred to a GPU bench node** (same honesty pattern as
[`internal/compute/PREFILL-B001-NOTES.md`](../../internal/compute/PREFILL-B001-NOTES.md)).
**Blocked:** the only host that can run a 100k prefill in tractable time (node-macos-a, M3 Pro,
llama.cpp Metal) was **offline** when this was authored. **No 100k wall-clock number was run,
estimated, or fabricated.**

Part of #519. The exact, contention-free **work floor** for this regime is shipped (`c28e861`,
`internal/turnbench/longcontext.go` + `cmd/longctxbench`, artifact
[`ultra-long-context-floor-20260622.json`](ultra-long-context-floor-20260622.json)). What remains
is the live wall-clock validation — the analogue of `cmd/sessionbench -validate` at small scale.

## Why this is bench-node/GPU gated (measured, not asserted)

Running arms B/C live at P=100k means a real ~100k-token prefill through the forward pass.

| engine | prefill rate | a single 100k prefill | arm B (C=5 → 5×100k) | validate arm A (naive re-prefill) |
|---|---|---|---|---|
| **fak pure-Go, this Windows box** | **~120 t/s @256 falling to ~29 t/s @835** (O(L²) attention; measured `go run ./cmd/sessionbench -synthetic qwen25-7b`) | tens of min → hours (rate keeps falling) | intractable | intractable (O(T²)) |
| **fak pure-Go, M3 Pro** | ~16 t/s (QWEN25-7B-RESULTS card) | ~hours | intractable | intractable |
| **llama.cpp Metal, M3 Pro (node-macos-a)** | **392 t/s** (measured, `macbook-m3pro-7b-batched-ctx.log`) | **~4–5 min** | ~20–25 min | computed from the prefill curve |

The highest **committed live** context to date is per-agent PP=8192 / aggregate N_KV=42240 (B=5),
in `macbook-m3pro-7b-batched-ctx.log` — short of the ≥100k regime. Only the llama.cpp Metal lane
on a resident GPU reaches 100k in bench-tractable time, and the `node-macos-a` bench node was
offline at authoring (its Tailscale peer showed `offline, last seen ~now`). The GPU server bridge is
not bench-ready (no llama.cpp/model). So the live anchor cannot be witnessed on any host reachable
from this box today.

## Turnkey procedure — next agent, on node-macos-a when it is online

The runner already exists (`tools/bench_node.sh`, read-only w.r.t. the committed tree → results
land in gitignored scratch; promoting to a committed path needs a redaction pass — no host names /
CPU brands in the committed artifact).

1. **Confirm reachable:** `tools/bench_node.sh node-macos-a wait` (auto-backoff until online).
2. **Run arms B and C live at the ultra-long shape**, arm A computed from the measured prefill
   curve — at C=1 and C=5, P≈100000, T=10, D=200, R=500, the exact shape the floor uses
   (`ultra-long-context-floor-20260622.json` cells 2 and 3). Drive the resident 7B Q8 GGUF with
   `llama-batched-bench` at a ≥100k prompt-processing length to anchor `prefillCost(L)` at the
   real regime (NOT extrapolated from the ≤8k card — flat-rate extrapolation under-counts the
   O(L²) attention; `tools/fleet_10min_projection.py` is fenced to the terse regime for exactly
   this reason and must not be used at P=100k).
3. **Assemble the artifact** in the floor's JSON schema with arms B/C `live: true` and arm A
   `live: false` (computed), plus a `live_validate` block confirming
   `anchored_computed_over_live ≈ 1.0` at a small scale (as `sessionbench -validate` does).
4. **Redact + commit** the JSON as `experiments/session/ultra-long-context-live-YYYYMMDD.json`
   next to the floor artifact; record lineage (version + UTC + commit + sanitized machine).
5. **Promote the claim:** add a BENCHMARK-AUTHORITY row for the live anchor and flip CLAIMS
   line ~141 from `[SIMULATED]` to `[SHIPPED]` **only if** the measured B/C and the A/C trend are
   consistent with the floor's prediction within the regime rules, citing the new artifact as the
   witness. If they diverge, keep `[SIMULATED]` and record the divergence — do not force a pass.

## Acceptance gate (unchanged from #524)

A committed **live** artifact at ≥100k context whose measured ratios are consistent with the floor;
CLAIMS `[SIMULATED]` wall-clock line upgraded to `[SHIPPED]` with that artifact as witness. The
floor's WORK numbers stand independently of this gate.
