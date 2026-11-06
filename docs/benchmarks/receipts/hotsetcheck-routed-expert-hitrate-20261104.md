# Routed-expert cache hit rate on the learner-wired residency seam (P2-4)

Date: 2026-11-04
Receipt: `docs/benchmarks/receipts/hotsetcheck-routed-expert-hitrate-20261104.json`
Witness: `go run ./cmd/hotsetcheck --json` (and the human form below it)
Label: [SW-VERIFIED] (synthetic fixtures, in-memory counting backing; no weights, no kernels, no clock, no hardware)

## Named quantity

The routed-expert cache hit rate — the decisive unmeasured quantity of the MoE
expert-cache program (fak-private#cluster-bringup-rtx6000-b300-readiness-20261104 P2-4;
the moe-expert-cache-benchmarks umbrella names it the single highest-leverage
component). This receipt measures it through the NEW learner→residency seam:
`model.ExpertHotSetAdvisor` (feeds `ExpertHotSetLearner`'s committed hot-set and
displacement band into `ctxmmu.ExpertResidencyPolicy`) versus the plain-LRU baseline
arm, over identical event streams.

## The loop measured

- **Warm start**: the learning phase (first 25% of tokens, observation-only) feeds the
  learner; its committed hot-seed is loaded once from the counting backing via
  `WarmStartFromAdvisor()`.
- **Steady state**: every decode touch is observed into the learner before it is
  decided; cold-miss admission that would evict a resident is gated by the learner's
  hysteresis band (`Admit(chHeat, vHeat)`); inside the band the touch is BYPASSED
  (served from backing, not made resident — `decode_bypasses`). `Learn()` re-commits
  at every turn boundary; steady state actuates through the gate alone, so the learned
  pins protect the ring instead of churning it.
- **Transient ring**: one deterministic prefill sweep (64 experts) every 20 measured
  tokens exercises the separate prefill ring, which can only churn itself.

## Fixture results (moment-in-time; no extrapolation to hardware)

Shape pinned at V4.1 (40 layers × 384 experts / top-6), 200 tokens, seed 1,
1 MiB per (layer, expert), hot ring 512 slots, transient ring 128 slots,
36,512 total touches (36,000 decode + 512 prefill).

| fixture | arm | hit_rate | hot_evictions | demand_misses | bypasses |
|---|---|---|---|---|---|
| zipf(s=1.2), hot-range 96 | baseline-lru | 0.6081 | 13,376 | 14,309 | 0 |
| zipf(s=1.2), hot-range 96 | learner-wired | **0.7032** | **157** | 10,835 | 10,257 |
| uniform | baseline-lru | 0.0376 | 34,359 | 35,140 | 0 |
| uniform | learner-wired | 0.0398 | 262 | 35,060 | 34,530 |
| zipf(s=1.2), hot-range 48, hot ring 2048 slots (near-fit) | baseline-lru | 0.9364 | 0 | 2,323 | 0 |
| zipf(s=1.2), hot-range 48, hot ring 2048 slots (near-fit) | learner-wired | **0.9800** | 0 | 730 | 0 |

Reading: where the working set concentrates (zipf), the learner-wired arm raises the
hit rate **+9.5pp** (default budget) / **+4.4pp** (near-fit) while cutting hot-set
evictions ~85× (13,376 → 157) — the incumbents stay pinned and lukewarm challengers
are bypassed instead of thrashing the ring. Under uniform routing there is no
structure to exploit: both arms sit near the cold-stream floor (~3.8%), and the
learner still protects the hot set (34,359 → 262 hot evictions) with only +0.2pp.
Pin stability on the default fixture: 22 `Learn()` swaps over 2 turns against a 512-unit
committed set (modest drift, not oscillation); re-learning unchanged heat returns
zero swaps (witnessed by `TestExpertHotSetLearnerHysteresisPreventsOscillation` and
`TestAdvisorAdmitMirrorsLearnBand`). Transient-ring behavior: sweeps churn only the
transient ring (303 evictions, end occupancy 118/128) and never the hot set; the
advisory `SizeTransientRing` sizing over the observed breadth recommends 128 MiB for
these fixtures (matches the configured cap).

## Reproduction

```bash
go run ./cmd/hotsetcheck                                  # default zipf fixture, human form
go run ./cmd/hotsetcheck --json                           # machine-readable receipt (fak.hotsetcheck/v1)
go run ./cmd/hotsetcheck --zipf 0                         # uniform contrast
go run ./cmd/hotsetcheck --zipf 1.2 --hot-set-size 48 --hot-bytes 2147483648   # near-fit regime
go test ./cmd/hotsetcheck -count=1                        # driver witnesses (determinism, skew ordering, jitter bypass, trace-file)
```

## Boundaries

No kernel, weight, or throughput claim: every counter derives from the policy's own
accounting over a counting fake backing; throughput/latency fields are absent from
`fak.hotsetcheck/v1` by construction. These numbers are [SW-VERIFIED] fixture
measurements at one moment in time — they bound the search space for the physical
rung, they do not predict it. The `[HW-WITNESSED]` rung remains owned by the
moe-expert-cache-benchmarks suite's Halo witness. Serve-path hookup of the adapter
(one-line seam, documented in `internal/model/expert_hot_set_advisor.go`) is the
follow-up; this receipt measures the loop fully unit-tested behind the seam.
