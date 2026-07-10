---
title: "study-repo: vLLM delta + 8-axis convergence witness → fak (2026-07-10)"
description: "Re-clone of vLLM at 26ff616. Two verified findings: (1) the 4-commit delta over the M2 pass is off fak's transferable axis (AMD MoE kernels + a deepseek_v4/amd model + one config guard), with the one on-axis bit being the DSpark sub-block-length correctness contract; (2) an exhaustive 8-reader + critic witness workflow (880k tokens, SHA 26ff616) found fak's control plane has independently CONVERGED on vLLM's orchestration axes — every candidate witnessed PRESENT or PARTIAL-present-adjacent, so 0 new leaves filed. Includes the verified `gh issue list --search` fabrication hazard + reliable alternative."
---

# study-repo: vLLM delta + 8-axis convergence witness → fak (2026-07-10)

> **STATUS: CLOSED, 0 leaves filed — an honest "we already have it".** The exhaustive verified witness
> that an earlier revision was waiting on has landed (workflow `wf_e9081795-807`: 8 subsystem readers + 1
> completeness critic, 880 020 subagent tokens, all anchors spot-checked against the tree at SHA
> `26ff616`). Its verdict, cross-checked against fak `internal/` + `cmd/fak/` code below, is that **fak's
> control plane has independently converged on vLLM's orchestration patterns** — the borrow surface is
> saturated *by fak's own substrate*, not merely "already ticketed". The prior revision's saturation claim
> (which had rested on `gh issue list --search` output this session proved fabricated) is now re-established
> from code witnesses, and no new issues are filed this pass.

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

## Why 0 leaves is the honest, disciplined outcome

Scout-loop rule: one lead per pass, file SMALL + ship-alone + on-axis, and **a witnessed "we already have
it" is a good result**. vLLM is fak's most-studied repo (M2, spec-decode, LMCache, Mooncake, SGLang, EPLB,
ktransformers, kvcache-factory). Filing from this firehose would be the "storm the backlog" anti-pattern.
Every witnessed candidate is PRESENT or a marginal-PARTIAL over an existing mechanism; the one PARTIAL
(death-pipe lease-release) is already recorded in memory and present-adjacent via lease-staleness. The
registerable finding is the convergence itself: **fak's substrate has independently arrived at vLLM's
orchestration axes** — dos (refusal-vocab / advisory-downgrade / quorum / verify-audit), affected
(minimal-checks), knownbad (flaky ledger), fleetpane (load + STALE), guard (carryforward / resume), lane
leases (arbitrate).

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
