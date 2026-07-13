---
title: "SparDA borrow study: 0 new issues — near-total convergence; SparDA is the *train-and-trust* pole of the same decouple-select-and-prefetch pattern fak already runs as *predict-as-hint-then-verify*"
description: "A deep four-way fan-out read of NVlabs/SparDA @ c8e235ba (Apache-2.0 root, MIT NOSA subtree) — a CUDA/PyTorch sparse-attention system for long-context LLM inference whose headline is a lightweight per-layer Forecast head that predicts layer l+1's KV blocks so the runtime prefetches them CPU→GPU while layer l still computes. Net: 0 issues filed. Every load-bearing shape — one-hop speculative prefetch, decoupled top-k selection, forced-admission floor, diff-based resident-set reconcile + hot-slot pin, miss=cheap-fault-not-correctness-loss, EV/dense fallback, guarded-resume identity check — resolves to PRESENT-on-axis in fak's ctxplan planner / loopmgr speculation governor / radixkv prewarm / dos_arbitrate / benchckpt resume, usually in near-identical language. The one genuinely distinct SparDA stance (train the cheap predictor to imitate an expensive oracle and TRUST it on the hot path, no runtime check) is a DIVERGENT for fak by design: DOS gates every irreversible consume on git-evidence verification, which is the correct pole for a side-effecting substrate. Nothing vendored; every verdict witnessed against a named path:line in this tree."
---

# SparDA borrow study: 0 new issues — the convergence is the finding

*2026-07-13. Study / borrow-scan note (witnessed against the tree on `main`, not self-reported).*

## What was studied

[`NVlabs/SparDA`](https://github.com/NVlabs/SparDA) @ `c8e235bacc3ead8dd19b4ff7a37dce8f09f05e9d`
(**Apache-2.0** root `LICENSE`; the vendored **`NOSA/` subtree is MIT** — THUNLP; arxiv
`2606.04511`). "SparDA: Sparse Decoupled Attention for Efficient Long-Context LLM
Inference" — a CUDA/PyTorch **training + eval codebase** (253 `.cu`, ~26K lines of Python/shell)
whose one idea is: add a lightweight per-layer **Forecast** projection alongside Q/K/V; the
Forecast from layer `l` predicts the KV blocks layer `l+1` will attend to, so the runtime
**prefetches them CPU→GPU while layer `l` still computes**. This decouples block *selection*
from the attention *query*, gives one layer of lookahead for prefetch, and reduces selection
overhead by sharing one Forecast head per GQA group. Reported: <0.5% added params, backbone
frozen (only the Forecast/indexer trains), up to 1.25× prefill / 1.7× decode speedup over
sparse offload.

**Method.** Acquired to a scratch clone (never committable); SHA pinned as the falsifiable
anchor. A **four-way fan-out deep read** over the load-bearing subsystems —
(1) the `NOSA/nosi/` inference runtime (the prefetch pipeline), (2) the `eval/` sweep
orchestration + `collect_results.py` aggregator, (3) the `training/` indexer trainer +
resume, (4) the `models/` Forecast decoupling core — each reader instructed to look **past
the CUDA/ML math for mechanism shapes on fak's axis** (selection / scheduling / prefetch /
verification / dedup / refusal / resume) and cite `path:line`. Each candidate was then
**ablated to its one axis** and **witnessed at that axis grain against fak's actual tree**
before any filing decision. Default license posture: **borrow the technique, reimplement in
Go — never vendor.**

**Completeness critic.** Opened: `NOSA/nosi/nosi/*`, `models/{nosa,minicpm}/*`, `training/*`,
`eval/*`, `NOSA/benchmarks/*/run.py` + `Efficiency/bench.sh` + `HELMET/run.py`. Deliberately
**not** opened, with justification: `infllmv2_cuda_impl/csrc/**` + the 253 `.cu`/`.cuh` kernels
(pure GPU sparse-attention math on CUTLASS — all four readers independently classified the
Triton/CUDA copy kernels, RoPE, and pinned-DMA plumbing as GPU-memory-hierarchy-specific; only
the *control shape* around them transfers, and that shape is what the runtime readers captured);
`docker/` (build infra); and the vendored third-party benchmark *internals* of
LongBench/HELMET/RULER (upstream projects, each own-licensed — the orchestration wrappers were
read). Nothing material to a fak borrow was left unopened.

## Net result: 0 issues filed — and that is the correct outcome

SparDA and fak have **independently built the same pattern**, and fak's version is the
side-effect-safe one. This is the same discipline the
[`deepspec` study](deepspec-borrow-study-2026-07-11.md) applied (0 filed) and the
[`tair-kvcache` study](tair-kvcache-borrow-study-2026-07-10.md) named: *"filing a redundant
issue is the failure mode, not the goal."* A candidate earns an issue only if it is genuinely
**absent on its axis**, high-leverage, and untracked. None here clear that bar. This repo sits
squarely inside the long-context-inference / KV-offload / speculative-prefetch family fak has
already scouted a dozen times (see Companions) — the convergence is not laziness, it is the
witnessed result.

## The intellectual core (the one thing worth carrying forward)

Two of the four readers **independently** reached the same thesis, and it is the reason the
study is worth a note even at 0 issues:

> SparDA's hot path **trusts** the cheap Forecast and never re-verifies the pick — it is
> *trained* to imitate the expensive oracle (a per-layer KL distillation,
> `models/nosa/modeling_llama_nosa.py:1959-1977`) and then believed at decode, with only a
> coarse dense fallback (`NOSA/nosi/nosi/nosa_llama.py:2163-2170`) and an *offline* waste
> ledger (`cache_engine.py:128-166`). This is safe **only because a mis-selected KV block is
> idempotent and side-effect-free** — a wrong guess costs a little attention mass, never a
> crash.

fak occupies the **opposite, correct pole for a substrate whose consumes have side effects.**
The exact same "cheap predictor emits a selection set the expensive stage consumes" shape is
`ctxplan.Forecast` + `TrajectoryAuthor` — but fak's forecast is explicitly *a hint, never a
commit*: `internal/ctxplan/forecast.go:10-20` — *"a forecast is cheap and revisable... a MISS
costs one demand-page fault... never a lost fact, so a bad forecast degrades efficiency, not
correctness"* — and any irreversible action is gated by hard arbitration
(`dos_arbitrate`) + git-evidence `dos_verify` at the moment a worker actually commits, plus the
EV/default-deny speculation governor (`internal/loopmgr/speculation.go:76-138`: effect-free
proof first, then slack, then positive EV). **Borrow SparDA's prefetch-as-hint; keep DOS's
verify-at-consume.** SparDA is the clean external confirmation of *why* fak refuses to let a
forecaster be load-bearing for correctness.

## Convergence map — every load-bearing SparDA shape, witnessed PRESENT-on-axis

Source paths are SparDA-relative @ `c8e235ba`; fak paths are on `main` 2026-07-13.

| # | SparDA shape (source `path:line`) | Axis | fak analogue (`path:line`) | Verdict |
|---|---|---|---|---|
| 1 | Cheap side-head emits only a top-k id set the expensive stage consumes (`models/nosa/modeling_llama_nosa.py:676-682,1151-1245`; `nosa_llama.py:144-161`) | selection = a pure function returning ids, decoupled from execution | `ctxplan.Forecast`/`TrajectoryAuthor` emits `Intents` → `Index.Probe` returns bounded candidate set (`internal/ctxplan/forecast.go`, `index.go`, `author.go`); kernel-side DSA index selection (`docs/notes/GLM52-DSA-INDEX-SELECTION-ON-PURE-KERNEL-2026-06-23.md`) | **PRESENT** |
| 2 | One-hop pipeline register: layer `l`'s forecast drives `l+1`'s selection, prefetch overlaps `l`'s compute (`modeling_llama_nosa.py:2496,2535,2558`; `nosa_llama.py:2874-2909`) | speculative prefetch, one step ahead, event-gated consume | `Forecast.Horizon` (N-turn lookahead) + `radixkv` "background-prefetch a probabilistically predicted next prefix" (`internal/radixkv/prewarm.go:16`); comm/compute overlap arm (`cmd/fanbench/overlap_test.go:164`) | **PRESENT** |
| 3 | Fixed-budget top-k + forced-admission floor: always keep `init`+`local` blocks (`nosa_llama.py:1868-1870`; `infllmv2_llama.py:1139-1141`) | never evict the active working set | `Forecast.Pins` (system prompt + active goal + first/last user turn) charged against budget **first** (`internal/agent/ctxplan_seam.go:199-208`; `forecast.go:34-38`) | **PRESENT** |
| 4 | Diff-based resident-set reconcile: keep intersection, fetch only the delta, **pin the slot being written** (`flash_cache_engine/diff_offload_kernel.cu:13-191`; `cache_engine.py:182-199`) | incremental reconcile + don't-revoke-the-hot-slot | `kvmmu.Context.ApplyPlan` evicts residency to the planned view (`internal/kvmmu`); `dos_arbitrate` exclusive holder is never revoked mid-write (lease disjointness) | **PRESENT** (hot-slot pin = arbiter's exclusive lease; incremental-vs-rebuild is low-leverage — fak's store is hot, a miss is a cheap demand-fault not a slow transfer) |
| 5 | Trust the forecaster on the hot path; account for waste offline (`nosa_llama.py:1292`; `cache_engine.py:128-166`) | measure the predictor's hit-rate, don't assume it | `cmd/ctxplanbench` measures the real forecast miss-rate (31.7% miss, **100% served, 0 lost**) + `Forecast.Learn` revises from the witnessed Outcome | **PRESENT** (the *measurement*); the *trust* stance itself → **DIVERGENT**, see below |
| 6 | Dense fallback below a length threshold; skip selection+prefetch (`nosa_llama.py:2163-2170,2657-2667`) | exact fallback when the fast path's premise fails | ctxplan seam **OFF by default** → append+compact unchanged (`ctxplan_seam.go:39-51`); `TestReplayLooseBudgetHoldsEverything`; speculation governor refuses → "run the normal turn path" | **PRESENT** |
| 7 | Head-of-chain self-selects; **fail-fast refuse** when the predictor head is absent (`modeling_llama_nosa.py:679-682,1360`; `nosa_llama.py:2144-2159`) | self-predict at the head; refuse structurally on a missing predictor | `heuristicForecast` falls back to priors+pins with no intents (`ctxplan_seam.go:218-231`); seam is fail-safe (disabled ⇒ inert, never a broken turn) | **PRESENT** |
| 8 | Guarded resume: bind the run to an immutable config + hardware/identity fingerprint, **hard-refuse on drift** (`eval/run_efficiency.sh:561-591,411-430`; `training/train_indexer.py:1354-1368`) | a resume must prove it is the same run/env | `benchckpt` **`ErrFingerprintMismatch`** — a resume whose fingerprint differs **refuses (exit 2) rather than silently mixing** (`cmd/modelbench/main.go:882`; `internal/benchckpt/benchckpt_test.go:91`, #2382); `leaseref.Fence` refuses a paused-then-resumed stale holder (#906-C1); `resume/identitymap.go` UUID↔trace join; portable session image records "ran under model A, resumed under model B on host X" | **PRESENT** (fak **enforces** the identity pointer SparDA only **stores** — `train_indexer.py:364` saves `base_model_path`, never re-checks it on resume) |
| 9 | Derive the accumulation knob from `target ÷ (batch × GPUs)`, floor-clamp, recompute-to-verify (`training/run_train.sh:229-232`; `train_indexer.py:957-961`) | derive concurrency/budget from a target + available capacity | `loopmgr.Governor` derives `MaxConcurrent`; fleet `--seat-ramp-delta`/`--max-workers` derive admits from seat/host min | **PRESENT** |
| 10 | Idempotent sweep: keyed completion ledger, skip terminal cells, rebuild-from-disk aggregation, latest-by-provenance dedup, suspect-duplicate flag (`eval/run_efficiency.sh:395-700`; `collect_results.py:569-623,1442-1603`) | resume/aggregate/dedup by *re-scanning evidence*, not trusting a scheduler's memory | fak dispatch resume = in-flight de-dup + cooldown ledger (`internal/dispatchtick`, `internal/accounts/cooldown.go`); `dos_verify` re-derives shipped-vs-claim from **git**, not self-report; ledgers accumulate raw num/den components | **PRESENT** (richer; and the eval harness is single-host trust-by-artifact — the *opposite* trust model) |

## Dropped as DIVERGENT — earned by ablation, with the tradeoff + their user world stated

Per the skill's rule, each is a place fak is *different* on the axis and chose the other way
**on purpose** — not a coarse "we already have that."

- **KL-teacher distillation: train a cheap student to imitate the expensive oracle's decision
  distribution, then trust it** (`models/nosa/modeling_llama_nosa.py:795-800,1959-1977`;
  `training/train_indexer.py:4-10,795-874`). **Their world:** an inference-latency research
  team serving a *frozen* model under a hard decode-latency budget where there is no time to
  verify a pick and a wrong block is a *soft* quality hit — so they buy safety **offline with
  training** and never pay for a runtime check. **fak's tradeoff (still holds):** a distilled
  *approximate* verifier/arbiter is exactly what a trust substrate must not ship — DOS's whole
  value is that `dos_verify`/`dos_arbitrate`/the refusal vocabulary are **deterministic and
  auditable** kernels, and a wrong lease/verify is a *hard* collision, not a soft miss. fak
  keeps the *measurement* half (predictor hit-rate, `ctxplanbench`) but rejects the trust-it
  half. The KL *mechanism* is cleanly factored (oracle as a swappable non-aliasing side-module)
  — a good pattern to remember **if** fak ever wants a cheap *hint* graded against an oracle,
  never a cheap *decision* replacing one.

- **"Exactly one hop" lookahead over a fixed linear stack of equal-cost stages**
  (`modeling_llama_nosa.py:2496-2558`). **Their world:** transformer layers run in a strict
  chain where fetch-latency ≈ one layer of compute, so depth-1 is provably enough. **fak's
  tradeoff:** fak's dispatch is a **DAG/pool of tickets, not a chain** — "the next stage" is not
  known, so "one hop ahead" would have to become "speculative fan-out over N candidate next-
  tickets," a real redesign with weak succession-predictability, not a copy. Low expected
  value; fak's `radixkv` already does the *probabilistic* single-prefix prewarm where the
  succession *is* predictable (a tool-latency window).

## Design reminders (note-only — not issues; none clear the ship-alone bar)

- **R1 — the predict-as-hint / verify-at-consume split is fak's design thesis, and SparDA is
  the external validation of the opposite pole.** Recorded here (the intellectual core above),
  not filed — it affirms existing design, it does not add work.
- **R2 — "amortize one predictor at the coarsest granularity of shared need"**
  (`modeling_llama_nosa.py:676-677`, one Forecast head per GQA group). A scheduler heuristic to
  keep in mind *if* a future lane ever shares one pre-warmed context/lease-check across a
  cluster of same-lane workers. Today fak's forecast is already measured-cheap (~68K candidate
  scores over 851 turns) and shared context across same-lane workers is already prompt-cache
  reuse (75.1% realized) — so amortizing an already-cheap predictor is low-leverage. Note-only.
- **R3 — output-fingerprint suspect-duplicate detection** (`collect_results.py:1442-1508`: flag
  two "different" configs whose numbers are improbably identical as `suspect_sparse_alias` with
  a reason, rather than presenting them as clean). This is the most DOS-aligned aggregator idea
  (distrust that two different runs are actually different); fak's dedup is **identity**-based
  (in-flight), not **output-fingerprint**-based. Recorded as a stance worth remembering, **not
  filed**: it is a speculative threshold-heuristic, low-leverage, and would risk fak's own
  "detection-without-enforcement" anti-pattern. If a real "two agents secretly did the same
  work / an aliased commit" problem is ever *witnessed* in the fleet, this is the shape to reach
  for — witness first, then file.

## Honest fences

- **0 issues is the deliverable, not an under-delivery.** Every candidate resolved to
  PRESENT-on-axis (fak code read on the seam and confirmed) or a DIVERGENT whose tradeoff **and**
  their user world are named. No borrow was killed at the coarse capability level.
- **Nothing vendored.** SparDA is Apache-2.0 (NOSA subtree MIT) — both permissive, so an
  INTEGRATE would have been *license-clean*, but there was nothing to integrate: the borrowable
  content is idea-level shapes already present in fak's Go, and the copyable bytes are GPU
  kernels off fak's axis. Default INSPIRE posture held; sources are cited regardless.
- **The witness is lexical + a snapshot.** `fak_feature_query`'s ranker is substring-shaped
  (false-ABSENT risk) and true only as of 2026-07-13; the PRESENT verdicts were cross-checked
  with raw `Grep` for the load-bearing symbols (`ErrFingerprintMismatch`, `TrajectoryAuthor`,
  `radixkv` prewarm, `ApplyPlan`) and read at the fak seam, not inferred from a rank.
- **Their "worldview" is a reconstruction.** The "inference-latency research team serving a
  frozen model" reading is inferred from the code + README non-goals + the KL-teacher/frozen-
  backbone defaults, not their testimony — kept falsifiable by citing the defaults, and never
  hardened into a dismissal (the DIVERGENTs rest on fak's tradeoff, not a guessed motive).
- **The scratch clone + the four fan-out reader outputs are working artifacts, not committed.**

## Companions

- Hands off to [`field-borrow`](../../.claude/skills/field-borrow/SKILL.md) — the per-capability
  witness+file this study feeds (no capability survived to hand over).
- Same-family prior studies (all reached PRESENT/DIVERGENT or few-to-zero files in this exact
  KV-offload / speculative-prefetch / long-context lane):
  [`deepspec`](deepspec-borrow-study-2026-07-11.md) (0 filed),
  [`CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE`](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md),
  [`CONCEPT-STUDY-MOONCAKE`](CONCEPT-STUDY-MOONCAKE-2026-07-10.md),
  [`CONCEPT-STUDY-KVCACHE-FACTORY`](CONCEPT-STUDY-KVCACHE-FACTORY-2026-07-10.md).
- The InfiniGen entry SparDA descends from is already cataloged in fak's own
  [`docs/awesome-token-efficiency.md:322`](../awesome-token-efficiency.md) — *"InfiniGen: full
  KV in CPU; speculatively prefetch next-layer critical tokens."*
