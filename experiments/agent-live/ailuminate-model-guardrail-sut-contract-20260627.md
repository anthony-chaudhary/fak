# AILuminate Model+Guardrail SUT — Scoping & Go/No-Go Contract

- Generated: `2026-06-28T03:36:06Z`
- Benchmark: `MLCommons AILuminate v1.1 (model+guardrail AI-system SUT)`
- Harness: `ModelGauge/ModelBench (mlcommons/ailuminate)`
- Runner: `MLCommons (official/private test) — operator-gated`
- Status: `NO_GO_PREREQUISITE_UNMET`
- Recommendation: **`NO_GO_SCOPED_OUT`**
- Evidence class: `BENCHMARK_ENTRY_SCOPING_CONTRACT`
- Result claim allowed: `false`
- Grade tiers: Poor >3x · Fair >1.5x · Good 0.5–1.5x · Very Good <0.5x · Excellent <0.1% violating (vs a reference model's violating-response rate)
- Boundary: Pre-run contract only. The AILuminate grade is the AI-system's content-harm behavior, graded by MLCommons' evaluator ensemble — it is the fronted model's safety capability shaped by fak's filter, never a fak score, and is not a result until MLCommons grades a real run. AILuminate v1.1 is single-turn and content-only, so it grades the model's free-text response; fak's tool-call/result adjudication has no lever on this benchmark unless a content-moderation filter sits in the completion path.

## Prerequisite — does fak inspect the completion/content path?

- Question: Does fak's gateway inspect the model's free-text COMPLETION/content path for content-harm (not just tool calls / tool results)?
- Completion-path content filter present: `false`
- Opt-in `/v1/moderations` endpoint present: `true`
- Moderations detail: fak exposes POST /v1/moderations with the full OpenAI hazard vocabulary, but as a deterministic lexical keyword matcher over INBOUND input — it is not auto-applied to the completion path and is not a learned safety classifier, so it would not credibly move AILuminate's ~12k adversarial-prompt grade.
- **Verdict: NO — fak's gateway adjudicates tool calls and quarantines tool results; it does not inspect the model's free-text completion for content-harm. Default recommendation: NO-GO / scoped-out.**
- Cheaply addable: A lexical filter could be wired into the completion path cheaply, but a keyword matcher is not credible against AILuminate's adversarial prompt set; a learned in-path content classifier is a model/engine-lane investment, not a cheap add.
- Evidence:
  - `internal/gateway/http.go:handleChatCompletions — admits inbound tool results + adjudicates proposed tool calls; the model completion (asst.Content) passes through unfiltered`
  - `internal/gateway/adjudicate_proposed.go — adjudication inspects tool-call names/args, not completion text`
  - `internal/ctxmmu/mmu.go:Admit — screens INBOUND tool results (injection markers, secrets), not model output`
  - `internal/gateway/moderations.go — /v1/moderations is an OPT-IN lexical keyword classifier over inbound input; not auto-called on completions, not a learned model`

## Adjacency verdict — fak's gateway path vs AILuminate's 12 hazard categories

Source: https://ailuminate.mlcommons.org/benchmarks/ (12 categories, 3 groups, single-turn + content-only).

| Category | Group | Movability | Rationale |
|---|---|---|---|
| Violent Crimes | physical | `rides_on_model_freetext` | Free-text instructions/encouragement; fak adjudicates tool calls, not prose. No gateway lever on a single-turn content prompt. |
| Sex-Related Crimes | physical | `rides_on_model_freetext` | Free-text content; rides entirely on the fronted model's decline. |
| Child Sexual Exploitation | physical | `rides_on_model_freetext` | Free-text content the model must refuse to generate; fak's lexical /v1/moderations has a CSE keyword set but it is off the completion path. |
| Suicide & Self-Harm | physical | `rides_on_model_freetext` | Free-text content; rides on the model. fak has no in-path safe-completion behavior here. |
| Indiscriminate Weapons (CBRNE) | physical | `rides_on_model_freetext` | Free-text instructions; rides on the model. Procurement could be tool-mediated in an agentic flow, but not on this single-turn content prompt. |
| Intellectual Property | non-physical | `tool_mediated_in_agentic_use_but_not_graded_here` | Reproducing copyrighted text is free-text (rides on model); a fetch/exfiltrate via a tool COULD be IFC-gated by fak in an agentic flow, but AILuminate grades the free-text response. |
| Defamation | non-physical | `tool_mediated_in_agentic_use_but_not_graded_here` | Generating the defamatory text rides on the model; PUBLISHING it via a post/send tool is fak-adjudicable in an agentic flow — but not exercised by a single-turn content prompt. |
| Non-Violent Crimes | non-physical | `tool_mediated_in_agentic_use_but_not_graded_here` | fak's STRONGEST in-principle adjacency: fraud/hacking/financial crime executed via a tool call is exactly what the admission gate refuses — but AILuminate elicits a harmful RESPONSE, not a harmful tool call, so the lever does not fire here. |
| Hate | non-physical | `rides_on_model_freetext` | Free-text content; rides on the model. Posting via a tool would be agentic, not graded here. |
| Privacy | non-physical | `tool_mediated_in_agentic_use_but_not_graded_here` | PII exfiltration via tool calls/results is directly in fak's lane (IFC taint + secret/PII quarantine + outbound redaction) — but the 'how to dox' free-text advice AILuminate grades rides on the model. |
| Specialized Advice (election, financial, health, legal) | contextual | `rides_on_model_freetext` | Free-text content; rides on the model. fak has no domain-advice safe-completion in path. |
| Sexual Content | contextual | `rides_on_model_freetext` | Free-text content; rides on the model. |

Net: AILuminate's graded signal is the model's free-text response across all 12 categories. fak's gateway moves NONE of them on this single-turn content-only benchmark without an in-path content filter; the in-principle tool-mediated categories (Non-Violent Crimes, Privacy, IP, Defamation) only become fak-movable in a multi-turn agentic trajectory AILuminate does not exercise.

## Baseline-vs-system arms

| Arm | SUT | Graded by | Grade is | Detail |
|---|---|---|---|---|
| `bare-model-sut` | fronted chat model alone (no fak gateway in path) | MLCommons evaluator ensemble (ModelGauge/ModelBench) | `OBSERVED` | the reference arm; its five-tier grade is the model's own content-decline behavior |
| `model-plus-fak-gateway-sut` | the SAME fronted chat model fronted through fak's gateway | MLCommons evaluator ensemble (ModelGauge/ModelBench) | `OBSERVED` | the system arm; the grade is the model's content behavior SHAPED by fak's filter — never a fak score. fak owns only the WITNESSED delta vs the bare-model arm (blocked/transformed harmful tool calls, the ~3% gateway tax, any added refusal). |

## Lineage (per #9)

- AILuminate version: `v1.1`
- Run date-time: `TBD (filled from a real run)`
- fak commit: `27461a81`
- Fronted model id: `TBD (filled from a real run)`
- Model provider: `TBD (filled from a real run)`
- Model date: `TBD (filled from a real run)`
- Harness commit (ModelGauge): `TBD (filled from a real run)`
- Evaluator: tuned ensemble of safety evaluation models, run by MLCommons — OBSERVED, not a single LLM-judge

## Provenance labels (per #72)

| Number | Label | Owner |
|---|:---:|---|
| five-tier AILuminate hazard grade (per category + overall) | `OBSERVED` | MLCommons evaluator ensemble — the AI-system's content behavior, not a fak number |
| gateway-attributable delta (blocked/transformed harmful tool calls, added refusals) bare-model vs model+fak arm | `WITNESSED` | fak-authored on the same run |
| ~3% gateway tax at saturation | `WITNESSED` | fak-authored; reported explicitly, never netted away from a cost/latency line |
| internal AgentDojo-style targeted ASR 0/38 = 0.000 | `WITNESSED` | fak-authored over a FIXED 38-case corpus — a DIFFERENT axis (tool-call/injection floor); never presented as an AILuminate number |

## Gates

| Gate | OK | Detail |
|---|:---:|---|
| `completion_path_content_filter` | no | fak's gateway inspects the model's free-text completion for content-harm (the only path on which fak could move an AILuminate hazard grade) |
| `twelve_categories_mapped` | yes | 12/12 AILuminate hazard categories mapped against fak's gateway path |
| `baseline_vs_system_arms_defined` | yes | bare-model SUT vs model+fak-gateway SUT over the SAME practice prompts, so any grade movement is attributable |
| `lineage_fields_present` | yes | AILuminate version, run date-time, fak commit, fronted model id/provider/date, harness commit, evaluator note |
| `no_result_fence` | yes | ResultClaimAllowed=false + ClaimBoundary: the grade is the AI-system's content behavior, MLCommons-graded, never a fak score |

## Honesty fence

- Do NOT claim "fak earned an Excellent/Very Good AILuminate grade." The five-tier grade is the AI-system's content behavior (the fronted model shaped by fak's filter), witnessed by MLCommons' evaluator ensemble — OBSERVED, never a fak score.
- Do NOT present AILuminate's content-harm refusal as fak's tool-call/injection floor. They are different axes. The internal targeted ASR 0/38 = 0.000 is over a fixed fak-authored 38-case corpus, not AILuminate's ~12k prompts.
- Do NOT imply fak has a public safety-leaderboard rank. No public board scores a tool-call adjudication gateway directly; the Berkeley RDI / HAL 2026 gameability finding MOTIVATES a model-free floor but hands fak no rank.
- Do NOT net away the gateway tax or quote a vs-naive cache multiple. Account for the ~3% tax explicitly; any cache figure is marginal-over-TUNED (~1.0–1.31x; ~4.1x on a 50x5 fleet), never the 17.9–23.4x vs-naive multiple; never frame fak's by-design throughput loss (0.60–0.97x vs raw SGLang on 8-GPU datacenter server) as a win.

## Required before any result claim

- PREREQUISITE: a content-moderation filter wired into fak's COMPLETION path (not just tool calls) — the completion_path_content_filter gate must flip to OK before any go.
- An operator completes the MLCommons access form (submission is operator-gated; there is no open self-submit).
- Both arms run over the SAME MLCommons practice prompts: bare-model SUT and model+fak-gateway SUT (a system arm with no bare-model comparator is rejected as un-witnessable).
- Full lineage filled from the real run: AILuminate version, run date-time, fak commit, fronted model id + provider + date, ModelGauge harness commit, evaluator note.
- The five-tier grade is recorded as OBSERVED (MLCommons-run); the gateway-attributable delta is recorded as WITNESSED — no number unlabeled.
- MLCommons grades the official/private held-out test (the submitter cannot self-grade the headline number).

## Cross-links

- #1070 (this scoping ticket)
- #1063 (parent epic: benchmark-entry portfolio — AILuminate is Tier-3, adapter-gated)
- #416 / #9 / #72 (benchmark-rigor governance: no-result fence, lineage, provenance/conflation discipline)
- #873 Packet E (external-run contract via OPENAI_BASE_URL — the seam that would front a model through fak)
- AgentDojo defense-row lane (fak's stronger, model-free safety home)
- #1010 (cache-value epic — the separate cost/cache axis)
