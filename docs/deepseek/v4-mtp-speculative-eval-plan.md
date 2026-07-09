---
title: "DeepSeek V4 MTP / speculative-decoding evaluation plan"
description: "A benchmark PLAN for measuring DeepSeek V4 with and without Multi-Token-Prediction / speculative decoding on a TUNED external-engine route (vLLM/SGLang), not a native MTP head. Extends the existing deepseekbench Row schema with a speculative=off|mtp|dspark axis and paired speed+quality columns (TTFT/TPOT/output tok-s, accepted-token ratio, exact-output acceptance, task-score parity), and defines how fak forwards engine-specific speculative config through the ExtraBody escape hatch without polluting provider-neutral APIs. Gates every speedup claim on score parity, not tokens/sec. No number here is measured."
---

# DeepSeek V4 MTP / speculative-decoding evaluation plan

> **Status: evaluation PLAN, not a benchmark.** This document carries **no
> measured throughput, latency, or task-score number**. Every quantitative
> expectation is labelled **MODELED**; everything that needs a GPU / tuned
> serving engine is labelled **host-gated**. The deliverable is a *row schema +
> parity-gate rule*, wired onto real fak seams, that a later host-gated run can
> fill honestly.

Resolves [#3020](https://github.com/anthony-chaudhary/fak/issues/3020). Parent
epic: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006). Sibling to
the other `docs/deepseek/*.md` / `docs/benchmarks/DEEPSEEK-V4-*.md` plan notes —
in particular the self-host wire-readiness runbook
[`../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md)
(#3013) whose OpenAI-compatible route this plan measures on, and the attention /
KV / MoE seam notes under `docs/notes/DEEPSEEK-V4-*-2026-07-08.md`.

## Why external-engine-first, and why MTP at all

DeepSeek V4 retains **Multi-Token Prediction (MTP)**: V4 Pro carries an MTP head
of **depth 1** (one draft token per step), which serving engines use as the
draft model for self-speculative decoding
([arXiv 2606.19348v1](https://arxiv.org/abs/2606.19348)). Two engines expose this
today:

- **vLLM** documents MTP speculative decoding plus a three-tier reasoning control
  for V4-Pro ([recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro)).
- **SGLang** tracks **DSpark** speculative decoding for V4 (a tree/multi-branch
  draft generalization), alongside its existing EAGLE MTP path
  ([docs.sglang.ai](https://docs.sglang.ai)).

Per the repo's prior-art / default-spine rule (see the self-host runbook), the
support path is to **front a tuned engine and measure the delta MTP buys there
first**. A native in-kernel MTP/NextN draft head is **explicitly out of scope**
until an external-engine benchmark shows a real, parity-verified target worth the
native weight-loading work. That gate is the whole point of this plan: MTP is a
*speed* optimization that must not change outputs, so a speedup is only real if
task score is unchanged.

fak already carries the DeepSeek-lineage geometry a future native head would need
(cited below), so this plan reuses that machinery for *labeling and comparability*
and does not invent a parallel stack.

## Grounding: real fak seams this plan maps onto

Before proposing any new column, these are the **actual** seams read in the tree
(file:line where load-bearing). New work is named distinctly in
[§ Missing seams](#missing-seams-named-distinctly).

### The benchmark row + honesty gate (primary seam to extend)

`internal/deepseekbench/deepseekbench.go` already defines the exact artifact the
issue asks to extend:

- **`Row` struct** (`deepseekbench.go:33-69`) — provenance-first: `Measurement`
  (`"dry-run-fixture"|"live"`) and `SpeedProvenance`
  (`"fixture-placeholder-not-measured"|"provider-observed"`) are read *before* any
  latency field. Route identity carries `EngineRoute` (`"hosted-api"|"vllm"|"sglang"`)
  and `Hosting`. Speed columns are `TTFTMillis` / `TPOTMillis` / `E2EMillis` /
  `OutputToksPerSec` (`:51-54`). Comparability keys are `PromptShapeKey` and
  `QualityParity` (`"unknown"|"verified"|"differs"`, `:67-68`).
- **`RequiredFields()`** (`:74-84`) is a *locked* JSON key set; a field-lock test
  (`TestRequiredFields`) marshals a row and asserts exactly these keys — adding a
  field without updating the list fails on purpose. This is where the new
  speculative axis + columns must be registered.
- **Locked axis vocabularies** (`:86-91`): `ContextBuckets`, `OutputTargets`,
  `ReasoningModes` (`non-thinking|high|max`) — the last already matches vLLM's
  three-tier reasoning control.
- **`PromptShape(...)`** (`:132-135`) is the comparability key builder; two rows
  are comparable only when it matches.
- **`CompareSpeedup(subject, baseline)`** (`:258-277`) is the honesty gate: it
  prints a speedup **only** when both rows are `Measurement=="live"`, share a
  `PromptShapeKey`, and both carry `QualityParity=="verified"`; otherwise it
  returns `"[NOT COMPARABLE: …]"` and `printed=false`.
- **`LiveGate(hasKey, spend)`** (`:279-291`) refuses a live run without a key and
  an explicit `--spend`, before any network call.
- **`MeasureStreamed(client, baseURL, key, model)`** (`:293+`) issues one
  streaming `/chat/completions` and times TTFT/TPOT/E2E and reads the final usage
  block. It is the exact insertion point for reading engine-exposed acceptance
  counters, and its `client` is injectable so an httptest SSE server can witness
  parsing with no key.

### Config forwarding without polluting provider-neutral APIs (the ExtraBody seam)

fak **already has** the escape hatch the issue demands — a per-request provider
extra body that rides alongside, but never overwrites, the provider-neutral
request:

- `internal/agent/adapters.go:68-72` — `ExtraBody json.RawMessage` on the request;
  empty ⇒ omitted, so an unset request is byte-for-byte the pre-seam wire.
  `marshalWithExtraBody` (`:353`) merges it into the outbound JSON.
- `internal/agent/chat.go:478` (`type HTTPPlanner`), `:511` (`ExtraBody`), `:625`
  (`SetExtraBodyJSON`) — the planner-level carrier and setter.
- `ParseExtraBodyJSON` **rejects core-field overrides** (`adapters_test.go:344`,
  `TestProviderExtraBodyRejectsCoreOverrides`): the escape hatch structurally
  cannot smuggle `model`/`messages`/`stream` etc. This is exactly the
  "don't pollute the neutral API" contract — engine-specific spec config goes here
  and *only* here.
- `internal/gateway/debug.go:109-114` + `debugProviderExtraBody` (`:485-502`)
  surface **keys only, never values** of the extra body on `/debug/vars`
  (`ProviderExtraBodySet` / `ProviderExtraBodyKeys`), so an operator can verify a
  speculative posture without leaking raw config. `TestDebugVarsExposeProviderExtraBodyKeysOnly`
  pins it.

The launch-side half already exists too: `tools/glm52_sglang_vllm_serve.sh:189-195`
starts SGLang with EAGLE MTP flags
(`--speculative-algorithm EAGLE --speculative-num-steps 5 --speculative-eagle-topk 1
--speculative-num-draft-tokens 6`), gated by a `SPECULATIVE=1` env (`:58`). So the
**speculative=off** arm is `SPECULATIVE=0` at engine launch, and the **mtp/dspark**
arms are launch flags — *not* wire fields.

### Accepted-token accounting semantics (polymodel)

fak's own speculative-decoding accept kernels define the precise meaning of
"accepted-token ratio", which this plan reuses as the *definition* even though the
external engine computes it:

- `internal/polymodel/polymodel.go:390-395` — `SpecResult{Accepted, Advance,
  KeepKV, EvictKV}`; `AcceptGreedy(draft, targetArgmax)` (`:410-427`) accepts the
  longest matching prefix, `Advance = Accepted + 1` correction/bonus.
- `SpecTree` / `TreeResult` / `AcceptTree` (`:448-495`) generalize to a **tree**
  draft and note that "a LINEAR chain reduces exactly to AcceptGreedy". This maps
  cleanly to the two engine arms: **mtp** (depth-1 linear chain → AcceptGreedy
  shape) vs **dspark** (multi-branch tree → AcceptTree shape).

Definition used throughout this plan: **accepted-token ratio =
accepted-draft-tokens / proposed-draft-tokens** over a run (equivalently SGLang's
mean `spec_accept_length` normalized by draft length). It is a *provider-observed*
efficiency number, never fak-authored.

### DeepSeek V4 model geometry, reasoning, and routing (reused for labeling)

- `internal/ggufload/gguf_config.go:191-196` — the loader already subtracts
  `nextn_predict_layers` (the MTP/NextN draft block) from `NumLayers`; the block
  contract is pinned by `gguf_qwen35_nextn_test.go` (block_count *includes* the
  trailing NextN/MTP draft block; the text forward never runs it). This is the
  exact tensor-level fact a native MTP head would build on — and the reason a
  native head is deferred, not free.
- `internal/ggufload/gguf_config.go:236-256, 381-449, 513-523` — `glm_moe_dsa`
  arch detection + MoE/MLA/DSA-indexer config parse (GLM-DSA attention *is*
  DeepSeek MLA + a learned indexer). `internal/model/glm_dsa.go:10-30` wires the
  MLA q_a/q_b, kv_a/kv_b projections and DSA top-k attention. Reused here only to
  justify the `model_id`/route labels; no native path is exercised.
- `internal/agent/reasoning.go:5-34` — `splitReasoning` preserves the
  `<think>…</think>` block into `Message.ReasoningContent` (the in-kernel twin of
  vLLM `--reasoning-parser`). This is what lets the harness score the **answer**
  separately from reasoning tokens across the three reasoning tiers, and why
  `ReasoningTokens` is its own row counter (`deepseekbench.go:59`).
  Conformance/usage/reasoning witnesses live in
  `internal/agent/deepseek_conformance_test.go`, `deepseek_usage_test.go`,
  `deepseek_reasoning_test.go`.
- `internal/modelroute/account.go:72-81, 661-677` — the V4 model ids
  (`deepseek-v4-pro` / `deepseek-v4-flash`) and routes
  (`deepseek` OpenAI-compatible, `deepseek-anthropic`, plus the generic
  `openai-compatible` self-host route). These populate `ModelID` / `ProviderRoute`
  / `EngineRoute`.

## The extended row schema (the deliverable)

The issue's acceptance is a **row schema with the speculative axis + speed+quality
columns + the parity-gate rule**. Concretely, extend `deepseekbench.Row` and its
locked `RequiredFields()` with the following. Field names are proposed JSON keys.

### New axis

| JSON key | domain | meaning |
|---|---|---|
| `speculative` | `off` \| `mtp` \| `dspark` | which draft path the serving engine ran. `off` = engine launched with speculation disabled; `mtp` = depth-1 self-speculative (EAGLE/MTP linear chain); `dspark` = SGLang DSpark tree draft. Sourced from the engine launch posture, **not** the neutral wire. |

`off` is the mandatory **within-shape baseline** every `mtp`/`dspark` row is
compared against (same `model_id`, `context_bucket`, `output_target`,
`reasoning_mode`, `stream`).

### New speed columns (paired — every speed number gets a provenance)

| JSON key | type | meaning |
|---|---|---|
| `accepted_token_ratio` | float, `0..1` | accepted draft tokens / proposed draft tokens (§ polymodel definition). `0` and a `accept_ratio_source="not-exposed"` when the engine omits it — **never synthesized**. |
| `accept_ratio_source` | string | `"provider-observed"` \| `"not-exposed"`. The honesty twin of `accepted_token_ratio`, mirroring the existing `cache_attribution` field. |

TTFT/TPOT/E2E/`output_toks_per_s` already exist and are reused unchanged; MTP
should move TPOT and `output_toks_per_s`, not TTFT.

### New quality columns (the parity evidence)

| JSON key | type | meaning |
|---|---|---|
| `exact_output_acceptance` | float, `0..1` | for **deterministic** (greedy / `temperature=0`) tasks: fraction of the smoke set whose `mtp`/`dspark` output is **byte-identical** to the `off` output at the same shape. `1.0` is the correctness bar MTP must clear. |
| `task_score` | float | score on the coding/reasoning smoke set for **this** row. |
| `task_score_baseline` | float | the paired `off`-arm `task_score` at the same shape, copied in for a self-contained parity check. |
| `task_score_delta` | float | `task_score - task_score_baseline`; the parity gate keys on `abs(delta) <= epsilon`. |

`QualityParity` (existing) is *derived* from these: `"verified"` iff
`exact_output_acceptance == 1.0` for deterministic tasks **and**
`abs(task_score_delta) <= epsilon` for scored tasks; `"differs"` if either fails;
`"unknown"` on a dry-run fixture.

### Extended `RequiredFields()` lock

Append to the locked list (`deepseekbench.go:74-84`):
`speculative`, `accepted_token_ratio`, `accept_ratio_source`,
`exact_output_acceptance`, `task_score`, `task_score_baseline`,
`task_score_delta`. The field-lock test then enforces they can never silently
drift.

## The parity gate rule (speedup gated on score, not tokens/sec)

Extend `CompareSpeedup` (`deepseekbench.go:258-277`) so a speculative speedup is
printed **only** when, in addition to the existing three conditions
(both `live`, shared `PromptShapeKey`, both `QualityParity=="verified"`):

1. `subject.speculative != "off"` and `baseline.speculative == "off"` — a
   speedup is only meaningful against the same model's non-speculative arm.
2. `subject.exact_output_acceptance == 1.0` for the deterministic subset **and**
   `abs(subject.task_score_delta) <= epsilon` for the scored subset. This is the
   load-bearing addition: **a higher `output_toks_per_s` with a task-score
   regression prints `[NOT COMPARABLE: speculative changed outputs — parity
   failed]`, never a speedup.**

Any other case returns a `[NOT COMPARABLE: …]` line and `printed=false`, exactly
as the existing gate does for dry-run / shape-mismatch / unverified parity. The
gate refuses; it does not warn-and-print.

`epsilon` is a declared constant with a documented rationale (proposed **MODELED**
default: `0` for exact-match deterministic tasks, and a small task-suite-specific
tolerance for scored tasks that must be *justified in the row's notes*, never
silently widened to admit a regression).

## How fak forwards the speculative config (the wiring)

Two planes, matching how the engines actually take the config — and neither
touches a provider-neutral field:

1. **Engine-launch plane (off vs mtp/dspark selection).** The arm is chosen when
   the engine starts, exactly as `tools/glm52_sglang_vllm_serve.sh:189-195`
   already does for EAGLE. `speculative=off` ⇒ `SPECULATIVE=0`; `mtp` ⇒ the
   EAGLE/MTP flags; `dspark` ⇒ SGLang's DSpark flags. fak records which arm ran in
   the `speculative` column from the serve posture, not by inspecting the wire.
2. **Per-request plane (any spec knob the engine reads from the request body).**
   Where an engine accepts request-level speculative controls, they ride the
   **existing** `ExtraBody` escape hatch (`agent/adapters.go:68-72`,
   `chat.go:511/625`), which `ParseExtraBodyJSON` already forbids from overriding
   core fields (`adapters_test.go:344`). No new field is added to the
   OpenAI/Anthropic request types; the provider-neutral surface is unchanged, and
   `/debug/vars` shows the posture **keys-only** via the existing
   `ProviderExtraBodyKeys` (`gateway/debug.go:109-114`).

This is the whole "without polluting provider-neutral APIs" requirement: the
neutral request type gains **zero** speculative fields; the config lives at engine
launch and in the vetted `ExtraBody` bag.

## Measurement protocol (host-gated)

All numbers below are **host-gated**; this section defines *how*, not *what*.

1. **Matrix.** For each `model_id ∈ {v4-pro, v4-flash}` × `speculative ∈
   {off, mtp, dspark}` × the existing `context_bucket` × `output_target` ×
   `reasoning_mode` × `stream` axes (skipping contradictory
   `non-thinking + long-reasoning` cells, as `DryRunRows` already does), emit one
   row. `off` is measured **first** per shape so its `task_score` seeds
   `task_score_baseline` for the `mtp`/`dspark` rows at that shape.
2. **Speed.** Reuse `MeasureStreamed` (`deepseekbench.go:293+`) for TTFT/TPOT/E2E.
   Read `accepted_token_ratio` from the engine's usage/metrics surface when
   exposed (SGLang `spec_accept_length`, vLLM spec metrics); set
   `accept_ratio_source="not-exposed"` and leave the ratio `0` otherwise — a
   recorded gap, never a fabricated number (same discipline as the runbook's
   usage-counter rung).
3. **Quality — deterministic.** At `temperature=0`, run the deterministic smoke
   set on the `off` and speculative arms and set `exact_output_acceptance` to the
   byte-identical fraction. MTP/EAGLE is *lossless by construction* (the target
   verifies every draft token), so **`exact_output_acceptance < 1.0` is a bug
   signal** — an engine/config defect or a non-greedy path — and must block the
   speedup, not be averaged away.
4. **Quality — scored.** Run a small coding/reasoning smoke set (host-gated
   grader), record `task_score`, copy the paired baseline, compute the delta.
5. **Compare.** Feed each `(speculative, off)` shape pair to the extended
   `CompareSpeedup`. Only a parity-verified pair yields an `OBSERVED provider
   speed: … ×` line; everything else is `[NOT COMPARABLE: …]`.

### Dry-run fixture (no key, no network, CI-green)

Extend `DryRunRows` (`deepseekbench.go:160+`) to emit the `speculative` axis with
`Measurement="dry-run-fixture"`, `SpeedProvenance="fixture-placeholder-not-measured"`,
`accept_ratio_source="not-exposed"`, and `QualityParity="unknown"`. Because the
gate refuses any non-`live` pair, the dry-run matrix **correctly prints no
speedup** — the fixture proves the schema and the refusal, not a result. No Go
fixture is written by this PLAN doc itself; the fixture extension is the
follow-on implementation ticket.

## MODELED expectations (labelled, not measured)

**MODELED, host-gated — do not cite as evidence.** Coarse priors only, to shape
the matrix, not to claim an outcome:

- MTP depth-1 typically accepts ~1 extra token/step on structured/deterministic
  text and less on high-entropy prose, so the **MODELED** `accepted_token_ratio`
  band is wide and shape-dependent; the `off` vs `mtp` TPOT gap is the real
  target, unknown until measured.
- DSpark's tree draft **MODELED**-plausibly raises acceptance over linear MTP at a
  compute cost, so its win is workload-dependent and must clear the *same* parity
  gate.
- The only honest pre-measurement claim: **the schema, the config-forwarding
  seam, and the parity gate are wired; no speedup is asserted.**

## Missing seams (named distinctly)

Each is a distinct follow-on implementation ticket; none is claimed done here.

1. **`speculative` axis on `deepseekbench.Row` + `RequiredFields()` lock** — the
   enum column and its vocabulary constant.
2. **Speed columns `accepted_token_ratio` + `accept_ratio_source`** — struct
   fields + the `MeasureStreamed` reader for SGLang/vLLM acceptance metrics.
3. **Quality columns `exact_output_acceptance`, `task_score`,
   `task_score_baseline`, `task_score_delta`** — plus the `QualityParity`
   derivation from them.
4. **`CompareSpeedup` parity-gate extension** — the `speculative != off` +
   exact-acceptance + task-score-delta conditions and the new refusal string.
5. **Deterministic exact-output harness** — greedy `off`-vs-speculative
   byte-identity comparator over the smoke set.
6. **Coding/reasoning smoke set + grader** — the `task_score` source (host-gated).
7. **Engine spec-config builder** — a helper that emits the launch flags per arm
   and, for request-level knobs, a vetted `ExtraBody` blob (reusing
   `ParseExtraBodyJSON`'s core-field guard); plus surfacing the `speculative`
   posture in `/debug/vars` keys.
8. **Acceptance-metric ingestion adapters** — parse SGLang `spec_accept_length` /
   vLLM spec metrics into `accepted_token_ratio` (host-gated, engine-version
   dependent).

Explicitly **NOT** in scope (per the issue): a native in-kernel MTP/NextN draft
head. The `nextn_predict_layers` loader handling (`gguf_config.go:191-196`) and the
GLM-DSA MLA path (`model/glm_dsa.go`) are the seams it *would* build on, deferred
until an external-engine benchmark shows a parity-verified target.

## Acceptance mapping

Bullet-by-bullet against the issue's acceptance / requirements:

- **"Benchmark PLAN for V4 with and without MTP/speculative decoding on a TUNED
  ENGINE route (not native)."** → this doc; measurement runs on the
  vLLM/SGLang OpenAI-compatible route
  (`../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`); native head is
  deferred in [§ Missing seams](#missing-seams-named-distinctly).
- **"V4 retains MTP; V4 Pro uses MTP depth 1."** →
  [§ Why external-engine-first](#why-external-engine-first-and-why-mtp-at-all),
  arXiv 2606.19348v1; grounded on the loader's `nextn_predict_layers` handling.
- **"SGLang tracks DSpark; vLLM recipe describes MTP + three-tier reasoning."** →
  same section + [§ Sources](#sources-researched-july-2026); three reasoning tiers
  map to the existing `ReasoningModes` (`non-thinking|high|max`).
- **"Measure SPEED AND QUALITY together (TTFT/TPOT/output tok-s, accepted-token
  ratio when exposed, exact output acceptance for deterministic tasks, task-score
  parity)."** → [§ Extended row schema](#the-extended-row-schema-the-deliverable)
  speed + quality columns; accepted ratio carries an explicit `not-exposed`
  provenance.
- **"Define how fak forwards engine-specific speculative config WITHOUT polluting
  provider-neutral APIs."** →
  [§ How fak forwards the speculative config](#how-fak-forwards-the-speculative-config-the-wiring):
  engine-launch plane + the existing `ExtraBody` escape hatch that
  `ParseExtraBodyJSON` forbids from overriding core fields; zero new neutral-wire
  fields.
- **"Benchmark rows must include speculative=off|mtp|dspark."** → the
  `speculative` axis added to `Row` and `RequiredFields()`.
- **"GATE any speedup claim on score parity, not tokens/sec."** →
  [§ The parity gate rule](#the-parity-gate-rule-speedup-gated-on-score-not-tokenssec):
  `CompareSpeedup` refuses unless exact-output acceptance and task-score delta
  pass; a faster-but-worse arm prints `[NOT COMPARABLE]`.
- **"Out of scope: no native MTP head until an external benchmark shows a real
  target."** → stated in [§ Missing seams](#missing-seams-named-distinctly).
- **"Acceptance: benchmark row schema with the speculative axis + speed+quality
  columns + parity gate rule."** → the three schema/gate sections above are that
  artifact.
- **House rules** → no measured number; MODELED and host-gated labels applied;
  external facts sourced below; epic #3006 and sibling notes cross-linked;
  YAML frontmatter present.

## Sources (researched July 2026)

- DeepSeek V4 MTP (depth-1) architecture — arXiv 2606.19348v1:
  <https://arxiv.org/abs/2606.19348>
- vLLM DeepSeek-V4-Pro recipe (MTP speculative decoding + three-tier reasoning):
  <https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro>
- SGLang DSpark / EAGLE speculative decoding docs: <https://docs.sglang.ai>
- DeepSeek V4 model card / vendor announcement (DSA, sparse attention):
  <https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro> ·
  <https://api-docs.deepseek.com/news/news260424>

## Cross-links

- Parent epic: [#3006](https://github.com/anthony-chaudhary/fak/issues/3006).
- Self-host wire-readiness route this plan measures on:
  [`../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md`](../benchmarks/DEEPSEEK-V4-SELFHOST-BASELINE-RUNBOOK.md)
  (#3013).
- Sibling seam notes: `docs/notes/DEEPSEEK-V4-ATTENTION-SEAM-MAP-2026-07-08.md`,
  `DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md`,
  `DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md`,
  `DEEPSEEK-V4-FP4-QUANT-PLAN-2026-07-08.md`.
- Benchmark authority ledger (the only place a number becomes authoritative):
  [`../../BENCHMARK-AUTHORITY.md`](../../BENCHMARK-AUTHORITY.md).
- The bench harness this schema extends: `internal/deepseekbench/deepseekbench.go`.
