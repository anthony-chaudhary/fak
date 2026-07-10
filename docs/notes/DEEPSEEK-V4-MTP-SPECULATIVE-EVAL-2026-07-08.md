---
title: "DeepSeek V4 MTP / speculative-decoding route — evaluation plan"
description: "Benchmark-plan only — no native MTP head lands here; any speedup claim is gated on measured score parity, and speed is never credited from tokens/sec alone."
---

# DeepSeek V4 MTP / speculative-decoding route — evaluation plan

**2026-07-08.** Issue **#3020**, parent epic **#3006** (DeepSeek-V4 performance track).
**Benchmark-plan + evaluation only** — no native Multi-Token-Prediction (MTP) head is
built here, and no speedup is credited without a paired score-parity witness. Current-state
claims are witnessed against the exact `path:line` cited (read 2026-07-08 on `main`).
Companion to the V4 scorecard harness (`internal/deepseekbench`, #3014) and the KV / MoE
kernel notes (#3017 `DEEPSEEK-V4-KV-LAYOUT-PLAN-2026-07-08.md`, #3018
`DEEPSEEK-V4-MOE-DISPATCH-BASELINE-2026-07-08.md`).

## Thesis — MTP/speculative decoding is a speed lever that can hide a quality regression

DeepSeek V4 retains the **Multi-Token Prediction** strategy (V4 Pro at **MTP depth 1**), and
tuned serving stacks already track **DSpark speculative decoding** for V4. Speculative
decoding proposes several tokens from a cheap draft path and *verifies* them against the full
model, keeping only the accepted prefix — so it is only ever a *speed* change if the accepted
output is identical to non-speculative decoding. The failure mode is silent: an engine can
report higher tokens/sec while a mis-tuned draft lowers the accepted-token ratio and drifts
task score. So the honest first step is a **benchmark plan that measures speed and quality
together**, and a report contract that **refuses a speedup claim when the parity fields are
missing** — before any native MTP head is written.

## The V4 MTP facts that drive the plan (from the issue grounding)

- V4 retains the **Multi-Token Prediction** strategy; **V4 Pro uses MTP depth 1**:
  https://arxiv.org/html/2606.19348v1
- SGLang's V4 roadmap tracks **DSpark speculative decoding** for DeepSeek V4:
  https://github.com/sgl-project/sglang/issues/23602
- The vLLM V4-Pro recipe describes **MTP speculative decoding** and three-tier reasoning:
  https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Pro

## Seam map — V4 MTP/speculative requirement → fak seam (`path:line`) or proposed

| Requirement | Nearest fak seam (verified `path:line`) | Fit / gap |
|---|---|---|
| **Speed+quality scorecard row** | `internal/deepseekbench/deepseekbench.go:33` `Row`, `:74` `RequiredFields` (already lists `quality_parity`, `:68`) | **Strong fit** — the V4 scorecard already carries the paired speed *and* `quality_parity` fields per row. |
| **Speedup gated on parity** | `internal/deepseekbench/deepseekbench.go:263` `CompareSpeedup` — refuses (`printed=false`) when `QualityParity != "verified"` for **both** rows (`:269`) | **Strong fit** — the "refuse speedup if parity unverified" rule the issue asks for is **already enforced** by the comparator; it just has no speculative axis yet. |
| **`speculative=off\|mtp\|dspark` axis** | `internal/deepseekbench/deepseekbench.go:87` locked axis vocabularies (`ContextBuckets`/`OutputTargets`/`ReasoningModes`) | **Gap.** No `Speculative` axis on `Row` or in the locked vocabularies; this note names it as the one schema addition. |
| **MTP / NextN head at load** | `internal/ggufload/gguf_config.go:189`–`:196` (`nextn_predict_layers` — Qwen3.5/3.6 trailing NextN/MTP draft blocks); `internal/model/config.go:737,:777` (MTP-head tensor-skip axis) | **Partial precedent** — the loader already recognizes and can *skip* a trailing NextN/MTP draft block; V4 Pro's depth-1 MTP is the same shape (`nextn_predict_layers=1`). No native MTP *decode* path exists — fence it. |
| **Live opt-in guard** | `internal/deepseekbench/deepseekbench.go:283` `LiveGate` (key + `--spend` required before any network call) | **Fit** — the speculative benchmark reuses the same opt-in so default CI never touches the network. |
| **Streamed TTFT/TPOT/E2E + accepted-token read** | `internal/deepseekbench/deepseekbench.go:300` `MeasureStreamed` | **Fit for TTFT/TPOT/E2E**; the **accepted-token / acceptance-ratio** field must be read from the engine usage block when exposed — a `Row` addition alongside the speculative axis. |
| **Provider-neutral engine config forwarding** | scorecard `engine_route` field (`RequiredFields:74`) | **Partial** — `engine_route` names the tuned engine; the *speculative config* (draft depth, DSpark params) must ride an engine-scoped side-channel, **not** the provider-neutral API surface. |

## Benchmark plan (the witness)

A benchmark plan for V4 **with and without** MTP/speculative decoding on a tuned engine
route (vLLM/SGLang), measuring speed and quality **in the same run**:

1. **Axis addition** — extend the locked scorecard vocabularies with
   `speculative = off | mtp | dspark` (or the engine's equivalent), and add an
   `accepted_token_ratio` field to `Row` (populated from the engine usage block when exposed,
   `"unknown"` otherwise).
2. **Speed metrics** — TTFT / TPOT / output tok/s (already measured by `MeasureStreamed`).
3. **Quality metrics, paired per row** —
   - **exact output acceptance** for deterministic tasks (bit-exact vs `speculative=off`),
   - **task score parity** for a coding/reasoning smoke set,
   - the engine's **accepted-token ratio** when exposed.
4. **The gate** — a speedup is credited **only** when `quality_parity == "verified"` for both
   the speculative and the `off` baseline row; `CompareSpeedup` (`:263`) already returns
   `printed=false` otherwise. A row missing the parity/quality fields **cannot** print a
   speedup — the report refuses, it does not warn-and-continue.

## The report contract (row schema)

Each benchmark row is keyed by the existing axes **plus** `speculative`, and carries paired
speed and quality fields:

```
model_id · provider_route · engine_route · hosting
context_bucket · output_target · reasoning_mode · stream · speculative        ← speculative NEW
ttft_ms · tpot_ms · e2e_ms · output_toks_per_s · accepted_token_ratio          ← accepted_token_ratio NEW
quality_parity ∈ {unknown, verified, differs}                                  ← the gate input
```

A `speculative=mtp` row with `quality_parity=unknown` is a **valid measured row** but is
**not comparable** — it can be recorded, never used to claim a speedup.

## Honest fences (what is NOT decided or built)

- **No native MTP decode head** — the loader can *skip* a NextN/MTP draft block
  (`gguf_config.go:196`); it cannot *run* one. Native MTP is the deferred follow-on, gated on
  this evaluation showing a real external-engine target + metric.
- **No `Speculative` axis / `accepted_token_ratio` field exists yet** — named here as the one
  schema addition the witness needs.
- **No speedup is credited from tokens/sec alone** — parity is a hard gate, already enforced
  by `CompareSpeedup`.
- **Speculative config is engine-scoped** — it must not leak into fak's provider-neutral API.
- **Acceptance gate is open** — the issue asks which harness this benchmark extends and the
  report path. Recommendation: extend `internal/deepseekbench` (it already owns the locked
  scorecard + parity gate) rather than a new package — flagged for operator input, not
  silently chosen.

## Next rungs

1. Add the `speculative` axis + `accepted_token_ratio` field to the `internal/deepseekbench`
   `Row` and locked vocabularies (dry-run rows first, no network).
2. Land a deterministic **exact-output-acceptance** check (`speculative=mtp` bit-exact vs
   `off`) as a pure fixture.
3. Wire the engine-scoped speculative config side-channel for the tuned vLLM/SGLang route.
4. Only then file the **native MTP head** follow-on — and make it name the engine acceptance
   metric and target threshold (per the issue's done-condition), not just "add MTP".
