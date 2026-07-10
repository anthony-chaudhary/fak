---
title: "tair-kvcache borrow study: one cache-budget-sweep gap filed, the rest present-on-witness"
description: "A deep-read of alibaba/tair-kvcache @ 6896aae1 (Apache-2.0) scanned for techniques fak lacks. Net result: one genuine high-leverage borrow filed (#3952 — single-pass budget→reuse miss-ratio-curve with infinite-cache ceiling + 99% knee); the calibration/MAPE, gen-fenced-reclaim, and single-flight candidates dropped on witness as already present."
---

# tair-kvcache borrow study: one cache-budget-sweep gap filed, the rest present-on-witness

*2026-07-10. Study / borrow-scan note (witnessed against the tree, not self-reported).*

## What was studied

[`alibaba/tair-kvcache`](https://github.com/alibaba/tair-kvcache) @ `6896aae1e1d6cdb3a4c473febeecabc7d4dec5b3` (**Apache-2.0**) — Alibaba Cloud's distributed KVCache manager (`kvcm`) + GPU-free inference simulator (`hisim`). Acquired to a scratch clone; SHA pinned. Method: fan-out deep-read over the load-bearing subsystems (optimizer/tradeoff sweep, hisim latency predictor, LeafAwareLRU / RandomLRU eviction, sharded-lock meta, TTL refresh, tiered storage, instance-group quota), extract one technique per candidate, license-gate each (default posture: **borrow the technique, reimplement in Go — never vendor** the differently-scoped upstream), then **witness each candidate against fak's actual tree** and classify PRESENT / PARTIAL / ABSENT before filing anything.

The discipline that shaped the outcome: a candidate is only worth an issue if it is *both* genuinely absent (or partial) *and* high-leverage. The witness step exists precisely to kill plausible-looking borrows that fak already embodies — filing a redundant issue is the failure mode, not the goal.

## Net result: one issue filed

**[#3952](https://github.com/anthony-chaudhary/fak/issues/3952) — single-pass cache-budget→reuse sweep with infinite-cache ceiling + 99% knee.** Replay a recorded prefix-access trace across N budget points **plus one unbounded pass**; report the reuse-vs-budget curve, the infinite-cache theoretical ceiling, and the smallest budget reaching 99% of it (the ROI knee) — so `--compact-history-budget` and the radixkv LRU budget can be sized from evidence instead of intuition.

- Upstream technique: `optimizer/analysis/script/run/tradeoff.py:49-108,269-338` + `manager/optimizer_runner.cc:43-152`. The load-bearing trick is the infinite-capacity warmup pass (`warmup_pass_with_metrics(..., -1, ...)`) as the theoretical hit ceiling, with the finite sweep early-stopping at 99% of it.
- fak status: **PARTIAL→ABSENT**. The feeders exist — `internal/radixkv` (the replay engine, LRU-leaf eviction), `cacheobs` (realized reuse ratio + histogram, but at *one* operating point), `internal/cachevalueledger` (a replayable realized-reuse trace) — but nothing sweeps the budget to produce the *curve* + *ceiling* + *knee*. `turnbench` replays security policy, not cache budget; `radixbench` is fixed-policy. Proposed borrow shape: a small `internal/cachesweep` leaf driving the existing radixkv engine.

## Dropped on witness (the redundancy the witness step caught)

- **Calibration scale-factor + MAPE-vs-measured gate (hisim `time_predictor`)** — the candidate ledger rated this STRONG/ABSENT, but that was *before* witnessing the tree. fak already carries the "bind MODELED cost to MEASURED reality, gate on drift" discipline in **three** independent places: `internal/gateway/calibration_corpus_drift_test.go` (an executable CI gate that recomputes corpus percentiles and *fails* when a hand-calibrated const drifts out of its documented band), `internal/gateway/resume_projection.go` (OBSERVED−PROJECTED cost & cold-write-share **residual gauges** + an explicit "persistent drift is the signal to refit" line), and `internal/vcachecal` (a warmth estimator with a full prediction-error report — false-warm/false-cold rates *reported*, not assumed zero). The genuinely-absent remainder is hisim's **XGBoost residual-ratio** model — opaque ML machinery fak's deterministic/pure/auditable ethos would deliberately reject. **DROP** (present ×3 + a technique fak wouldn't adopt).
- **Gen-fenced stale-lease reclaim** — `internal/leaseref/fence.go` is already a textbook monotonic fencing-token: re-read-live-lease-before-write (`Fence`), CAS old-value, TRANSITION-with-strict-generation-bump on reap, halt-and-reacquire on `STALE_LEASE`. This *is* the Kleppmann/etcd/Chubby paused-then-resumed-holder fix tair's meta describes, and more rigorous. **DROP — PRESENT.**
- **Single-flight / call-avoidance** — `internal/callavoid/avoid.go` is a full memoization break-even (`ProveMemo`) + amplification-accounting gate; the avoid *decision* is deeply modeled (validate/capture cost, mutation rate, stale-miss as a strict loss). The bare RUNNING-placeholder mechanic is a low-leverage detail its framework would host. **DROP.**
- **LeafAwareLRU / tiered-storage-demote / instance-group quota** — PRESENT (radixkv LRU-leaf eviction; `cachemeta` demote-not-evict; headroom/fleetaccounts quota). **RandomLRU** (sampled/approx LRU) is situational and low-leverage against radixkv's exact O(1) LRU. **DROP.**

## Deferred (real but low-leverage)

- **Fixed-window vs sliding-window TTL** (don't refresh the anchor on read; separate anchor-time from last-access) — `optimizer_runner.cc:207-221,257-263`. `cachemeta` already separates the anchor (`AdmittedAt`) from `LastAccess`, so this is a small refinement (an explicit no-refresh-on-read knob), rated low-med. Not filed; recorded here so the trail is complete.

## Honest fences

Only #3952 was filed — one small, scoped issue, not a monolith. Nothing from tair was vendored; the borrow is a Go reimplementation of a technique. The scratch clone + candidate ledger are working artifacts, not committed. Every PRESENT/DROP verdict above is backed by a named file in this tree, not by the upstream's plausibility.
