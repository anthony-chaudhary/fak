# Self-Tax Trend — fak's own overhead and net effect, tracked over time

> **What this is.** The living trend companion to the **self-tax** row in
> [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md). It tracks fak's *own*
> mediation overhead — and the cases where that mediation makes work **faster** —
> as a dated series, net-true-labeled, every point tracing to a committed artifact
> and a reproduce command like every other authority row.
>
> **Epic:** L5/T12 of [#1147](https://github.com/anthony-chaudhary/fak/issues/1147)
> ([#1169](https://github.com/anthony-chaudhary/fak/issues/1169)). Design note (survey,
> maturity ladder, full DoD, per-ticket specs):
> [`docs/notes/self-tax-performance-assurance-tracking-1147.md`](../notes/self-tax-performance-assurance-tracking-1147.md).
> **Anchor standard:** [`net-true-value`](../standards/net-true-value.md) Question #2
> (net-of-introduced-cost).

**Snapshot date:** 2026-06-28
**Status:** Living document — append a row when a new self-tax point lands.

## Why a self-tax trend exists

fak inserts itself into the hot path of every tool call, result, turn, and commit.
Each insertion has a cost — latency, tokens, wall-clock — and sometimes it *removes*
cost (reuse it would otherwise redo). The security floor already proves a bad call
can't get through. This trend is the dual read-out: it proves a good call doesn't get
slower than its budget — and **names it when fak made the work faster**.

The honest shape is two-directional. fak's tax is **workload-dependent**: a measurable
*cost* on a tool-light task (nothing to reuse, only mediation to pay for), a measurable
*saving* on a reuse-heavy task (prefill not redone, net of the tokens mediation adds). A
single average would hide both; this trend reports the signed points and lets the
workload speak.

> A budget is an **envelope with a stated scope**, not a promise of zero cost — a gate
> that costs 8% and saves 40% is a net win, and this plane says so rather than redding on
> the 8% alone (epic non-goal (c)).

## The trend (signed: `+` = cost added, `−` = cost removed)

Every row's number is already a committed entry in the Authority; this doc is the *fold*
over them that makes the self-tax direction legible as one series. Provenance is labeled
per the standard: **WITNESSED** (fak measured it itself), **OBSERVED** (provider-relayed),
**MODELED** (computed from a stated model).

| First committed | Surface (lifecycle stage) | Self-tax point | Sign | Provenance | Artifact | Reproduce |
|---|---|---|---|---|---|---|
| 2026-06-26 | adjudication read-path floor (every tool call) | **~0.55 ns/op · 0 allocs · FLAT N=1→1000 drivers** | `+` (floor) | WITNESSED | `internal/abi/registry_scaling_test.go` | `go test ./internal/abi -bench BenchmarkRegistryReadScaling -benchmem` |
| 2026-06-20 | pure-kernel decide — canonical allow (Submit) | **362 ns/op** allow (560–605 ns w/ ArgPredicates) | `+` | WITNESSED | `experiments/mac-m3pro-kernel-20260620/kernel-latency-mac-m3pro-20260620.json` | `go test -bench` per the M3 Pro Authority row |
| epic #607 (Regime A) | vDSO self-ablation, tau2-airline smoke (frozen trace) | **tokens 937→417 = −520 tok** (engine_calls 12→5, vdso_hits 0→7) | `−` (net saving) | WITNESSED | `experiments/ablate/tau2-smoke-vdso-ablation.json` (doc: `docs/benchmarks/ABLATE-RESULTS.md`) | `go run ./cmd/fak ablate --trace testdata/tau2/tau2-smoke.json --sweep vdso` |
| epic #607 (Regime B, #623) | cross-agent bare `claude` vs `fak guard -- claude` (pong, tool-light) | **total-ingested 1.56× = +28,986 tok** (5/5 ALLOW, 0 deny) | `+` (net cost) | WITNESSED (kernel counters) | `experiments/ablate/cross-agent-pong-opus.json` (doc: `docs/benchmarks/ABLATE-RESULTS.md`) | `python tools/cross_agent_ablate.py report --reps <reps.json>` |
| 2026-06-28 | net-true **improvement** detector (decode reuse) | **net_tokens = local-reuse ⊕ provider-reuse − mediation-tax** → positive on a reuse-favorable trace | `−` (net saving) | WITNESSED ⊕ OBSERVED − MODELED | design note §9 (`…self-tax-performance-assurance-tracking-1147.md`) | `curl -s localhost:PORT/metrics \| grep -E 'kv_prefix_reused_tokens_total\|inference_cached_prompt_tokens_total'` (§9.4) |

### How to read a signed point

- **`+` (cost):** mediation fak adds. The two latency floors (0.55 ns read, 362 ns decide)
  are the per-call drop-in cost; the cross-agent `+28,986 tok` is what guarding a tool-light
  task ingests on top of the bare agent. These are the numbers a budget breach is defined
  against (the per-rung envelope is T2 of the epic).
- **`−` (saving):** cost fak *removes*. The vDSO ablation skips 520 tokens of redundant
  engine work on a frozen trace; the improvement detector folds realized KV-prefix reuse
  (WITNESSED, in-kernel) ⊕ provider `cache_read` (OBSERVED, disjoint plane — never
  double-counted) minus the tokens mediation re-emits (MODELED), and nets positive whenever
  reuse exceeds the tax.

The series is the witness for the epic's net-true claim: fak is **not silently degrading
performance** (the `+` floors are small and flat), and it **detects when it improves**
performance (the `−` rows), each net of its own cost.

## Honesty fences

- **This is fak's *mediation* overhead, not raw-inference parity.** fak-vs-llama.cpp
  forward-pass parity is the separate axis tracked by
  [`track-b-performance-parity #306`](../notes/track-b-performance-parity-tracking-306.md) —
  cross-linked, never blended into this number.
- **Curated fold today, auto-emitted series tomorrow.** Each row here is an
  independently-committed point measurement. The *always-on* version — a frozen-workload
  fak-on-vs-fak-off CI gate with change-point + persistence + SPRT (T6/T7/T8) and a single
  `fak perf` read-out + net-true `/metrics` family (T11) — is the named follow-on in the
  epic, not built here. Until it lands, this trend is a hand-folded read of committed
  artifacts, not a single auto-regenerated JSON. Stated plainly rather than implied.
- **Single-box wall-clocks stay single-box.** The latency floors are machine-stamped
  (Ryzen 9 9950X for the read floor; M3 Pro for decide); the deterministic counter rows
  (vDSO ablation token deltas, the reuse split) reproduce byte-identically and are
  hardware-independent. The cross-agent row is one tool-light task on one host — an honest
  worst case for tax (nothing to reuse), not a fleet SLA.
- **Improvement is double-count-guarded by construction.** Local KV-prefix reuse and
  provider `cache_read` arrive on structurally disjoint `internal/cachemeta` planes, so
  their sum is realized reuse with the provider-vs-local split intact — a token is local *or*
  provider, never both (design note §9.1).

## Appending a point (keeping it living)

1. Land the new self-tax measurement as a committed artifact (a JSON under `experiments/…`
   or a bench/test witness) with its own reproduce command — the same discipline as any
   Authority row.
2. Add one signed row above: first-committed date, surface, the number, its sign,
   its WITNESSED/OBSERVED/MODELED label, the artifact path, and the reproduce command.
3. Bump the **Snapshot date**, and update the self-tax row in
   [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) if the headline snapshot changed.

## Verification

- Every artifact path above is a tracked file; every number is the committed value already
  carried in [`BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md) (the read-path floor,
  pure-kernel decide, vDSO ablation, cross-agent ablation rows) or pinned in the epic design
  note §9 (the improvement detector). This doc adds no new number — it folds existing
  witnessed numbers into the self-tax direction.
- Cross-agent ratio re-derived from the artifact: bare `total_input_tokens.mean = 52,077.8`,
  fak `81,063.4` → `1.557×`, Δ `+28,986` tok (`experiments/ablate/cross-agent-pong-opus.json`).
- vDSO ablation counters re-derived from the artifact: `engine_calls 12→5`, `vdso_hits 0→7`
  (`experiments/ablate/tau2-smoke-vdso-ablation.json`).
