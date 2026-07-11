---
title: "study-repo: vLLM delta + 8-axis convergence witness → fak (2026-07-10)"
description: "Re-clone of vLLM at 26ff616. Verified findings: (1) the 4-commit delta over the M2 pass is off fak's transferable axis (AMD MoE kernels + a deepseek_v4/amd model + one config guard), with the one on-axis bit being the DSpark sub-block-length correctness contract; (2) an 8-reader + critic witness workflow found fak CONVERGED on vLLM's CI / liveness / quorum / observability-producer axes (all PRESENT/PARTIAL-adjacent); (3) a SECOND, deeper adversarial re-witness of the decode-constraint / spec-sample / metrics-consumer control-plane (workflow wf_2bed7fc7-41f, 67 agents, refute-every-gap) found 3 code-verified on-axis SURVIVORS the first table never covered — 2 latent/downstream design-inputs (recorded), 1 candidate metrics leaf (per-request queued→prefill→decode→inference timeline). So 'saturated' is scoped to the axes actually witnessed, not universal. Includes the verified `gh issue list --search` fabrication hazard + reliable alternative."
---

# study-repo: vLLM delta + 8-axis convergence witness → fak (2026-07-10)

> **STATUS: MOSTLY CONVERGED — the "saturated" claim is now SCOPED, not universal.** Two passes ran.
> **Pass A** (workflow `wf_e9081795-807`: 8 subsystem readers + 1 completeness critic, 880 020 subagent
> tokens, anchors spot-checked at SHA `26ff616`) witnessed fak's **CI / minimal-checks / liveness / quorum /
> observability-producer** axes all PRESENT or PARTIAL-adjacent — the convergence table below stands for
> *those* axes. **Pass B** (workflow `wf_2bed7fc7-41f`: 67 agents over the **decode-constraint / spec-sample /
> metrics-consumer** control-plane, every claimed gap sent to adversarial refutation) drilled a *different*
> candidate set and found **3 genuine on-axis SURVIVORS** that Pass A never covered — each re-verified below
> against fak `internal/` code, not against `gh issue list --search` (which this session proved fabricates
> number↔title, see hazard table). So the earlier revision's blanket "the borrow surface is saturated / 0
> leaves" was an **over-broad headline**: correct for Pass A's axes, wrong as a universal claim. The honest
> verdict is **broad convergence + 3 residual gaps**, of which **2 are latent/downstream design-inputs
> (recorded, not filed)** and **1 is a candidate metrics leaf** — see [§ deeper re-witness](#verified--deeper-re-witness-decode--sample--metrics-control-plane-3-survivors).

## Tooling hazard (verified — the reusable lesson)

`gh issue list --repo anthony-chaudhary/fak --search "<query>"` returned issue objects whose `number` did
**not** match the `title` on direct lookup:

| search claimed | `gh issue view N` ground truth |
|---|---|
| `#470` = "[2607.05147v1] DSpark: Confidence-Scheduled Speculative Decoding" | **#470 = CLOSED "harden text-embedded tool-call parsing"** |
| `#591` = "tree-structured verification for native MTP" | **#591 = CLOSED "feat(brand): naming-consistency guard"** |
| `#622` = "Add Medusa speculator" | **#622 = CLOSED "feat(ablate): env-gated feature arms"** |
| `#734` = "grammar-constrained decode slower…" | **#734 = CLOSED "measure kernel-in-the-loop guard-hop overhead"** |
| `#175` = "Structured-output enforcement for local backends" | **#175 = CLOSED "Fig 36 bar value has no numerator"** |
| `#4791`, `#38069`, `#6771`, `#48197` (various spec-decode titles) | **do not resolve — no such issues** |

**Reliable alternatives (use these):** `gh search issues "<terms>" --repo anthony-chaudhary/fak --json
number,title,state` (GraphQL search — returned real, verifiable issues such as **#3020**, **#4199**), and
`gh issue view N` for direct confirmation. Treat any number from `gh issue list --search` as unverified
until `gh issue view` corroborates it. Every ticket number in this note was confirmed with `gh issue view`.

## VERIFIED — the clone is the M2 tree; the delta is inert for fak

Fresh shallow clone pinned at **`26ff616`** ("Register Qwen/Qwen3.5-4B example model", #48276). GitHub
compare `08dfd68...26ff616` = **4 commits, 10 files, +611/−69** — the same tree the KV-cache-value M2 pass
([`CONCEPT-STUDY-VLLM-M2-2026-07-10.md`](CONCEPT-STUDY-VLLM-M2-2026-07-10.md)) already read, four commits
earlier. Every changed file is **off fak's transferable axis** (fak is a prompt / tool-page / session /
cache-value / decode-gating control plane that "never touches a KV tensor" — kernels/weights are out):

| file | +/− | axis |
|---|---|---|
| `csrc/libtorch_stable/moe/moe_align_sum_kernels.cu` | +47−48 | CUDA MoE kernel |
| `vllm/models/deepseek_v4/amd/dspark.py` (new) | +499 | model weights (DSpark draft head, AMD) |
| `vllm/models/deepseek_v4/amd/model.py` | +45−5 | model weights |
| `vllm/models/deepseek_v4/amd/rocm.py` | +8−2 | ROCm glue |
| `vllm/config/speculative.py` | +25 | **config contract (the one on-axis bit — below)** |
| `cmake/external_projects/flashmla.cmake` | +1−1 | build |
| `csrc/.../test_moe_align_block_size.py`, `tests/models/registry.py`, `tests/models/test_registry.py` | test | — |

## VERIFIED — the one on-axis contract in the delta (read from disk)

`vllm/config/speculative.py:945–968@26ff616` rejects `num_speculative_tokens < dspark_block_size` because a
semi-autoregressive **block** drafter fed a sub-block speculative length "yields **incorrect (garbled)
output** rather than merely lower acceptance."

Transferable lesson, one level up from the kernel: **for a block / semi-AR drafter, a draft length below
the block size is a correctness bug, not a perf knob.** The right home is fak's block/semi-AR spec-decode
work — candidate targets **#3020** ("evaluate V4 MTP/speculative decoding route") and **#4199**
("mlx-dspark CapController borrow"), both verified OPEN. It would land as an admission guard
(`draft_len >= block_size`) plus a `malformed-precondition` class in the accept/reject ledger (**#4102**,
verified OPEN), distinct from `low-acceptance`. **Recorded, not filed** — it is a pre-implementation design
input for an epic fak already owns, not a ship-alone leaf, and fak's spec substrate is not yet far enough
along to consume it.

## VERIFIED — the convergence witness (8 readers + critic, 880k tokens @26ff616)

The workflow deep-read eight fak-relevant vLLM subsystems, ablated each to a domain-general axis, and
hypothesised a fak home; the critic spot-checked every anchor against the tree (all matched). I then
witnessed the strongest, least-obviously-present candidates against fak code. **Every axis witnessed PRESENT
or PARTIAL-present-adjacent** — the readers themselves consistently wrote "the analogue lives in [existing
fak subsystem]". The pattern is not "already ticketed"; it is **fak already implements the axis**, often by a
stronger mechanism.

| vLLM axis (source `path:line@26ff616`) | witness — fak home (verified) | disp. |
|---|---|---|
| Flaky checker → advisory/non-gating, tracked for repair (`.buildkite/test-amd.yaml:13`; `docs/contributing/ci/failures.md`) | **PRESENT** — `cmd/fak/knownbad.go` (flaky-vs-shared ledger, "flaky not shared" retraction) + `internal/adjudicator/advisory.go` ("refusal citing an advisory reason → admit-and-log") + `internal/abi` `ScreenQuarantine`/`VerdictDefer` | drop |
| Run only checks whose declared input region intersects the change; blast-radius escape hatch (`.buildkite/test_areas/*.yaml source_file_dependencies`; `ci_config.yaml run_all_patterns`) | **PRESENT (stronger)** — `cmd/fak/affected.go` + `internal/affectedtests`: tests only changed packages + transitive test-importers via the Go import graph (computed, not hand-declared) | drop |
| Cheap-propose / expensive-verify; batch-verify a range; truncate at first rejection; rollback by rejected count; distribution-preserving recovered token (`vllm/v1/spec_decode/*`, `vllm/v1/sample/rejection_sampler.py`) | **PRESENT** — `dos_review` (CLEARED-vs-RESIDUAL rev-range band = truncate-at-first-rejection), `dos_verify`/`dos_commit_audit` (git evidence overrides self-report = recovered token), `guard_carryforward`/resume (rewind to last verified commit). Distribution-preservation has **no fak analogue** (token-distribution, not git-evidence) — correctly droppable | drop |
| Content-addressed + parent-chained state identity; longest-prefix hit with first-miss short-circuit; hash→block dedup with refcount co-residency; residency event stream for external routers (`vllm/v1/core/{kv_cache_utils,block_pool}.py`, `kv_events.py`) | **PRESENT / OWNED** — the KV-cache-value M2 axis, already studied + filed (#3893/#3894/#3897 + enrichments). Content-addressing is fak's home turf (git SHAs, commit-audit hashes). Residency-stream → fleetpane snapshot + `STALE_SNAPSHOT` (59e309f3c) | drop (M2-owned) |
| Cross-worker quorum: only trust a fact witnessed by all/N reporters; HWM bounded-lossy buffer; coarse resync + idempotent consumption (`kv_events.py:118-196`, `config/kv_events.py`) | **PRESENT** — dos "do not believe a single self-report" is the quorum shape; fleetpane prefers a bounded droppable snapshot + STALE flag over blocking dispatch | drop |
| Deterministic selection: frozen hashable config key + memoized resolution; priority ladder; per-candidate reason lists; mutual-exclusion-on-ambiguity; order-independent override table (`vllm/v1/attention/selector.py`, `platforms/cuda.py`, `platforms/__init__.py`) | **PRESENT** — `dos_arbitrate` (mutual exclusion on a contended lane), `dos_refuse_reasons` (per-candidate reason tokens, not booleans), lane-autopick ladder (rank-ordered). "Fold routing inputs into one frozen key + memoize" is a marginal determinism refinement | drop |
| AI-contribution dedup / fail-closed on busywork; accountability trailer; protected-domains read-guide-or-refuse; instruction-file token budget (`AGENTS.md`, `docs/contributing/*`) | **PRESENT** — dispatch preflight dedup (`dispatch_tick_preflight.go`), `(fak <leaf>)` ship-trailer + `dos_commit_audit`, `core_locks.toml` (hard-locked core), skill-score + guard carryforward budget | drop |
| Sentinel/OS-level liveness (not self-report); fail-stop vs reassign; published per-worker load for external router placement; elastic scale-up/down with drain-before-remove (`vllm/v1/executor/multiproc_executor.py`, `engine/coordinator.py`, `core_client.py`) | **PRESENT / present-adjacent** — guard distrusts self-report + verifies from git; `fak_resume_history`; dispatch redispatch (loosely-coupled agents → can reassign, the property the vLLM contrast names); fleetpane published load + STALE_SNAPSHOT | drop |
| Death-pipe reverse liveness: subordinate detects supervisor death via an auto-closing channel and tears itself down before orphaning a resource (`multiproc_executor.py:675-712, 784-807`) | **PARTIAL, present-adjacent + recorded** — no death-pipe in fak; but `dos_arbitrate` reclaims via lease-staleness (last-touch), and the exact failure is already in memory `[[guard-env-leak-detached-launches]]`. Proactive lease-release-on-parent-death would be a *latency* refinement over existing staleness reclaim | recorded, not filed |

**Critic's TOP-missed trunk subsystems** (no reader opened them; flagged for a *future* deep-read, not
filed — each is subsystem-scale, not a ship-alone leaf, and maps to an existing fak home):
`vllm/v1/core/sched/scheduler.py` admission+preemption trunk (≈ fak dispatch admission + `class_budgets` +
cap-park/rotate); `vllm/v1/structured_output/backend_types.py` `StructuredOutputGrammar`
(accept/validate-prefix/rollback/`fill_bitmask` ≈ dos closed refusal vocab + `dos_review` CLEARED/RESIDUAL);
`vllm/tool_parsers/abstract_tool_parser.py` `ToolParserManager` (≈ guard MCP mediation `guard_mcp.go`);
`vllm/v1/metrics/{stats,loggers}.py` (producer/schema half of fleetpane observability).

## VERIFIED — deeper re-witness (decode / sample / metrics control-plane): 3 survivors

Pass B (`wf_2bed7fc7-41f`) re-scouted the axes Pass A's readers did **not** open — the decode-constraint
mask, the spec-sample acceptance ledger, and the metrics-consumer timeline — and sent every claimed gap to
adversarial refutation (majority-refute ⇒ killed). Of 50 candidates: **25 witnessed PRESENT, 12 already
TRACKED, 7 refuted, 3 survived.** All three anchors below were **re-verified by hand against fak's tree**
(not via issue search), and each survivor's disposition is chosen by the same scout-loop discipline Pass A
used — *record a latent/downstream design-input, file only a small ship-alone on-axis leaf.*

| # | survivor (vLLM axis) | fak ground-truth anchor (verified this pass) | why it's REAL, not present | disposition |
|---|---|---|---|---|
| S1 | Reasoning-phase mask suspension: suspend the grammar/byte mask while the model is inside a `<think>` span, resume at the reasoning-end boundary | `internal/model/constraint.go:203` `StepMask.MaskLogits` keys on `step := len(history)` from step 0; `:271` `GuidedByteMask.MaskLogits` rebuilds the prefix from **all** history via `decodePrefix` — **neither has any reasoning-phase gate** | The mask has no notion of a reasoning boundary. **But** `:234` states the mask is **"NOT WIRED LIVE … dormant until `FAK_NATIVE_GUIDED_DECODE=1`"** (default-off; corroborated across `constraint_test.go`, `grammar/compile.go:122`, 3 notes). So the gap is **latent** — it only bites the moment the mask is wired into the live sampler, which the code itself defers to "a later slice" | **Recorded, not filed.** Add "suspend on reasoning span" to the later-slice wiring contract for the native-guided-decode epic (#26 / #2596-family). Pre-implementation design-input for an epic fak owns; no ship-now surface |
| S2 | Synthetic acceptance-ledger replay: drive a rejection-sampler test from a per-position **conditional** rate curve (cᵢ = pᵢ/pᵢ₋₁) instead of running the model | `internal/turnbench/stochastic.go:49` `RateProfile` = four **independent** per-class Bernoulli draws against an eligible base call (`AliasRate`/`DupRate`/`PureRate`/`StaticRate`); grep `synthetic_mode\|unconditional_to_conditional\|conditionalRate` in `internal/` = **0** | fak simulates via real-kernel Bernoulli draws, not a replayed per-position conditional-acceptance ledger. Genuinely un-borrowed **but** testing-grade, and **downstream of per-position acceptance work fak hasn't built** (tracked via the DeepSpec per-position line, `[[deepspec-borrow-study-2026-07-11]]`) | **Recorded, not filed.** One-line note on the existing per-position-acceptance tracked item; it can't ship before the mechanism it replays exists |
| S3 | Per-request latency **event timeline**: QUEUED→SCHEDULED→first→last ledger yielding a **queued-wait** interval + a first-schedule-wins **preemption fold**, not just a prefill/decode split | `internal/gateway/metrics_observe.go:573` `observeInferenceTimed` is *handed* `dur, ttft` and only splits prefill=ttft / decode=dur−ttft; grep `scheduled_ts\|queued_time\|first_schedule\|queuedWait\|QueueWait` in `internal/` = **0**; reliable `gh search issues` finds no on-point ticket (nearest: #3391 token-hit-rate, #4008 TTL-witness — both other metrics) | No queued-wait component and no event ledger, though fak **does** queue at admission (dispatch-tick / seat-park) so a queue-wait is measurable. On fak's exact metrics axis; sibling to the active gateway-metrics enrichment line (#3391) | **FILED #4261** — the one small, ship-alone, on-axis, unowned lead this pass produced. Lands as a `queued`/`prefill`/`decode`/`inference` interval decomposition on `gatewayMetrics` with a first-schedule-wins fold; sibling to #3391 |

**Why these survived where Pass A saw only convergence:** Pass A's readers mapped CI/liveness/quorum/observability-*producer* axes, all of which fak implements. Pass B drilled the *consumer/decode* interior — the grammar mask's phase-awareness, the acceptance ledger's replay mode, the metrics timeline's granularity — where fak's mechanisms are present but **coarser** than vLLM's. None is a "fak is missing a subsystem" gap; all three are **refinements at the edge of a mechanism fak already has**, which is why 2 of 3 are design-inputs rather than leaves.

## Why the outcome is still disciplined (broad convergence + one candidate lead)

Scout-loop rule: one lead per pass, file SMALL + ship-alone + on-axis, and **a witnessed "we already have
it" is a good result**. vLLM is fak's most-studied repo (M2, spec-decode, LMCache, Mooncake, SGLang, EPLB,
ktransformers, kvcache-factory). Filing from this firehose would be the "storm the backlog" anti-pattern.
Across both passes the tally is exactly what the discipline wants: the Pass-A axes are PRESENT or a
marginal-PARTIAL over an existing mechanism, and Pass B's 3 survivors reduce to **2 recorded design-inputs
(S1, S2 — latent / downstream of unbuilt work) + 1 ship-alone leaf filed as #4261 (S3)** — one lead, not a
backlog storm. The primary registerable finding remains the convergence: **fak's substrate has
independently arrived at vLLM's orchestration axes** — dos (refusal-vocab / advisory-downgrade / quorum /
verify-audit), affected (minimal-checks), knownbad (flaky ledger), fleetpane (load + STALE), guard
(carryforward / resume), lane leases (arbitrate) — with the honest caveat that fak's decode/sample/metrics
*interior* is coarser than vLLM's at three edges (S1–S3), not saturated.

## Honest limits

- License gate unchanged from the M2 pass: Apache-2.0 ↔ Apache-2.0; any candidate is Python→Go ⇒ `inspire`,
  no bytes vendored.
- "PRESENT" here means the *axis* is implemented in fak by grep-witnessed code, not that fak's mechanism is
  bit-for-bit vLLM's — several fak homes (Go import-graph affected-tests, git-evidence verify) are stronger
  and structurally different.
- The critic's four trunk subsystems were witnessed only at the *mapping* level (each has an obvious fak
  home), not by a line-level read of fak's counterpart; a future pass could deep-read one if a concrete gap
  is suspected. Ticket state is a snapshot — re-confirm with `gh issue view`.

## Companions

- Same tree, KV-cache-value seam (filed #3893/#3894/#3897): [`CONCEPT-STUDY-VLLM-M2-2026-07-10.md`](CONCEPT-STUDY-VLLM-M2-2026-07-10.md).
- Broad adapter map: [`VLLM-OPTIMIZATION-REUSE-TICKET-MAP-2026-06-30.md`](VLLM-OPTIMIZATION-REUSE-TICKET-MAP-2026-06-30.md).
- Dynamic-SD / DSpark studied (independently reached "0 leaves / present-by-other-means"):
  [vLLM Dynamic-SD borrow](study-vllm-dynamic-sd-borrow-2026-07-10.md) ·
  [DSD dynamic spec-decode](CONCEPT-STUDY-DSD-DYNAMIC-SPEC-DECODE-2026-07-10.md) ·
  [DeepSpec borrow](deepspec-borrow-study-2026-07-11.md) · [DFLASH](CONCEPT-STUDY-DFLASH-2026-07-10.md).
- Known-failure memory this pass's one PARTIAL maps to: `[[guard-env-leak-detached-launches]]`.
