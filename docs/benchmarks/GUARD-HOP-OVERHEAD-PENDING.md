# Guard-hop overhead + prompt-cache preservation — BENCHMARK-AUTHORITY row (PENDING/PROJECTED)

**Issue:** [#734](https://github.com/anthony-chaudhary/fak/issues/734)
**Harness:** `tools/guard_hop_bench.py` (+ hermetic `tools/guard_hop_bench_test.py`)
**Status:** harness + row STRUCTURE landed; the live MEASURED wall-clock is **PENDING** (hardware-gated).

> **Why this lives here, not yet in `BENCHMARK-AUTHORITY.md`.** The authority file holds
> only numbers traced to a committed artifact. This row's overhead arm is a **PROJECTION**
> from an existing committed artifact, and its prompt-cache arm is **PENDING** a live
> provider run — neither is a fresh measured wall-clock. Per the authority's own honesty
> rules (never present a projected/pending number as "measured"), the row sits in this
> companion until the MEASURED run lands, then it folds in as a normal row.

## What it measures

When a worker is fronted with `fak guard` (the dogfood default), every tool call crosses
the kernel before reaching the provider. A serving stack's value proposition is that this
safety hop (a) adds negligible latency and (b) does **not** break the provider prompt-cache.
This row quantifies both.

| Arm | Metric | Status | Value | Basis / blocker |
|---|---|---|---|---|
| **guard-hop overhead** | added latency per turn / per session | **PROJECTED** | per-turn **2.90–19.42 µs** (8 calls/turn); per-session **0.14–0.97 ms** (50 turns) | closed-form from the committed pure-kernel decide-latency rows: 362 ns Decide (floor) → 2.427 µs in-process adjudication (ceil), commit `bcad56e`, `experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json` |
| **guard-hop overhead** | gateway p50 − direct p50 (wall-clock) | **PENDING** | — | needs a live `fak serve` gateway + a direct mock upstream on the same box: `python tools/guard_hop_bench.py measure --gateway-url … --direct-url …` |
| **prompt-cache preservation** | provider `cache_read` tokens, guarded vs direct | **PENDING** | — | needs a live provider that reports cache tokens (hardware/credential gated); the structural byte-for-byte `cache_control` forwarding is already exercised by `internal/gateway` tests |

The two bounds on the PROJECTED arm are deliberate: the guard process is long-lived (one
`fak guard` fronts the whole session), so the marginal cost of the hop is the **in-process**
adjudication per tool call, not the ~6.9 ms spawned-`fak hook` boundary the in-process path
replaces. Floor = the cheapest ALLOW (`Decide`, 362 ns); ceil = the full in-process syscall
p50 (2.427 µs). Even the ceil is **sub-millisecond per 50-turn session** — the projection's
whole point: the safety hop is in the noise next to a single provider round-trip.

## The honesty gate

`tools/guard_hop_bench.py --check <row.json>` refuses a dishonest row: every metric arm must
carry a `status ∈ {MEASURED, PROJECTED, PENDING}`; a PENDING arm may carry **no** number; a
MEASURED arm must carry a single-box disclosure; a PROJECTED arm must name its committed
basis (commit + artifact). This is the structural guarantee the harness cannot fabricate a
"measured" guard-hop number it never measured — the gate is tested in
`tools/guard_hop_bench_test.py`.

## Reproduce

```bash
# PROJECTED + PENDING row (no hardware):
python tools/guard_hop_bench.py describe --json

# MEASURED overhead (requires a live gateway + a direct mock on this box):
#   term 1:  ./scripts/dogfood-claude.sh --smoke      # an offline-mock fak serve on :8080
#   term 2:  (a direct mock upstream on :8099)
python tools/guard_hop_bench.py measure \
  --gateway-url http://127.0.0.1:8080 --direct-url http://127.0.0.1:8099
```

## Deferred → MEASURED

To turn the PENDING arms into a MEASURED row foldable into `BENCHMARK-AUTHORITY.md`:

1. Stand up a `fak serve` gateway + a direct mock upstream on one box; run `measure`.
2. Run a guarded-vs-direct A/B against a cache-reporting provider and compare
   `provider_cache_read` tokens per turn (the token-decomposition harness in
   `tools/cross_agent_ablate.py` already decomposes the four token classes).
3. Commit the JSON artifact, then add the row to `BENCHMARK-AUTHORITY.md` with the
   MEASURED status, single-box disclosure, and the artifact path — and tombstone this
   PENDING doc.
