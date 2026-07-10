# Concept study — speculative decoding (article) vs fak's real seams — 2026-07-10

A `/study-repo` pass: read a speculative-decoding thought-leadership article and its
primary sources, then ablated every technique it raises against fak's *actual* seams.
Headline: fak's spec-decode surface is **already saturated with honest self-tracking** —
the industry scorecard, open issues #3078/#3197, the GLM-5.2 MTP levers, and **three
prior study notes** ([DFlash](CONCEPT-STUDY-DFLASH-2026-07-10.md),
[Dynamic-SD scout](study-vllm-dynamic-sd-borrow-2026-07-10.md),
[DSD supplement](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md)) already mined the
drafter gap, the load-adaptive-K axis, and the `speculate.go` `SuccessProb` gate. So most
axes are **earned dismissals**. Terminal outcome: **1 filed (#4202), 1 folded (#4201→#809),
0 net new tracking on already-covered seams.**

## Sources (pinned)

| Source | Ref | Pin |
|---|---|---|
| Article (spec-decode survey/opinion) | Leviathan 2211.17192 · Chen 2302.01318 · Medusa 2401.10774 · EAGLE-1/2/3 | read 2026-07-10 |
| EAGLE reference impl | `SafeAILab/EAGLE` | `cb7e0841` |
| vLLM v1 spec_decode + rejection sampler | `vllm-project/vllm` | `26ff616b` |
| fak (this repo) | seams below | `78464d0a` |

## Candidate table (the full ablation)

| Borrow (article axis) | Source anchor @sha | fak seam | On-axis | Verdict |
|---|---|---|---|---|
| **Stochastic (T>0) losslessness** — rejection-sampling accept + residual ("recovered") resample preserves the target distribution when sampling | Chen 2302.01318 Thm 1; vLLM `rejection_sampler.py:394,663,608` @`26ff616b` | `internal/polymodel/polymodel.go:450` `AcceptGreedy` is **argmax-only**; sampler exists (`internal/agent/inkernel_sampling.go:113`) but no stochastic accept rule; scorecard losslessness row = "greedy half `parity`, stochastic half **unbuilt**" | **PARTIAL** (greedy parity; sampled half absent) | **FILED #4202** (inspire) — no prior note or issue touches T>0 accept |
| **Empirical α feedback** — measure the *observed* accept rate on real traffic; gate on measured, not declared, prob | vLLM `rejection_sampler.py:394` acceptance accounting @`26ff616b` | `internal/abi/speculate.go:71` `SuccessProb` is a **static** field; `:132` `Predict` gates on it; `:263` `Resolve`/`Commit`/`Squash` compute the true outcome but **no `Observe` feeds it back** | **ABSENT** (outcome computed then discarded) | **FOLD → #809** (closed #4201). The [DSD note](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md) already homes this exact gate to #809 as record-don't-file; the empirical-**measurement** axis (distinct from its live-**load** axis) is recorded on #809, not re-filed |
| **Prompt-lookup / n-gram (model-free) drafter** | vLLM `ngram_proposer.py:12` `NgramProposer` @`26ff616b` | `polymodel.go:546` `PickDrafter` selects a *model* only | tracked | **DISMISS → #3078** (open: "…prompt-lookup, measure acceptance + tok/s") |
| **Acceptance-rate + tok/s measurement** | Leviathan 2211.17192 §speedup | `EffectiveTokensPerVerify` (`polymodel.go:602`) closed-form only | tracked | **DISMISS → #3078** (+ closed bench rung #535) |
| **Trained self-spec drafter** (Medusa / EAGLE feature layer / EAGLE-3 / block-diffusion) | EAGLE `cnets.py:479` (`top_k`, `threshold` dyn-tree) @`cb7e0841` | `PickDrafter` reuses *idle co-resident* models, zero training; GLM-5.2 C2 MTP substrate landed no consumer | tracked | **DISMISS → #3197** + [DFlash note](CONCEPT-STUDY-DFLASH-2026-07-10.md) + GLM-5.2 C2 lever |
| **Load-adaptive K / never-regress-under-load** (batch_size→K, K→0 fallback) | vLLM DSD PR #32374 @`4ef4492e` | control plane already occupies this (min-folds + graded setpoints #4036/#4038/#3368/#4069) | out-of-scope for non-batched spec path | **DIVERGENT** — [Dynamic-SD scout](study-vllm-dynamic-sd-borrow-2026-07-10.md) settled it, 0 leaves |
| **Wall-clock / ITL / batch-scaling speedup numbers** | article throughput framing | CPU-synthetic engine, `FAK_POLYMODEL` default-off | out-of-scope **by construction** | **DIVERGENT** — scorecard: "OUT OF SCOPE for a reuse kernel" |

## Why #4202 is the one on-thesis, genuinely-untracked borrow

Losslessness is the **single `parity` row** in the scorecard — the one property fak
actively competes on (everything else is `no-claim`/out-of-scope). The three prior
spec-decode notes all target the **drafter** (draft source, block-diffusion, adaptive-K)
or the **control gate** (SuccessProb, load) — **none touches the *accept rule* at T>0.**
fak's accept core is argmax-only (`AcceptGreedy`); the sampler and per-position logits
already exist, so distribution-preserving rejection sampling is a clean, GPU-free
correctness slice that upgrades the losslessness row from "greedy half at parity,
stochastic half unbuilt" to full parity. No drafter, no GPU, no throughput claim.

## Why #4201 was folded (not kept as a standalone issue)

I filed #4201 (empirical-α measurement) before reading the DSD supplement note, which
had already examined the *same* `speculate.go` `SuccessProb` gate, homed it to epic
**#809**, and deliberately recorded-don't-filed it as "low urgency (opt-in)." My axis
(measure α from `Commit`/`Squash` outcomes) is distinct from the DSD note's axis (gate on
live system load), but it is the **same seam under the same epic**, and the repo's
established handling is to track it on #809, not to spawn issues. So #4201 was closed and
the empirical-α design recorded as a comment on #809 alongside the DSD load-gate finding.

## Already-recorded, NOT re-claimed
The `internal/spec` phantom (`SpeculativeGreedy`/`VerifyTree`, `go test ./internal/spec`)
cited across CLAIMS.md / the scorecard / `verify.go` but absent from the live tree (parked
at `.dos/_dos_park/_iso_build/internal/spec/`; real core is `internal/polymodel` +
`internal/model`) is **already documented** by the [DFlash](CONCEPT-STUDY-DFLASH-2026-07-10.md)
and [DSD](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md) notes. Flagged again only as
context for #4202; no separate claim.

## Companions
- #4202 (filed) — stochastic T>0 rejection-sampling losslessness.
- #809 (epic) — homes the folded empirical-α axis; comment links the record.
- #3078 / #3197 — tracked drafter + acceptance-measurement axes this pass dismissed against.
- Prior notes: DFlash (drafter/block-diffusion), Dynamic-SD scout + DSD supplement (adaptive-K, SuccessProb gate).
- Scorecard `docs/industry-scorecard/decoding.md`; GLM-5.2 self-spec levers (C2/C4/C5).
