---
title: "Borrow study: routing / signals / gateway — plano (archgw) vs fak's witnessed frontier (2026-07-13)"
date: 2026-07-13
kind: borrow-study
source: "katanemo/archgw (plano) @ d2127b83ffaee67b230e0bc1858cef8c676e0615 (release 0.4.27, 2026-07-09)"
license: "Apache-2.0 (integration-with-attribution permitted; Rust→Go means most transfers are design, not code)"
---

# Borrow study: plano (archgw) — routing, signals, provider-handling, guard, budget

Deep-read of **plano / archgw** (`katanemo/archgw`, the AI-native Envoy + `brightstaff`
data-plane) against fak's already-shipped frontier on the five axes it overlaps: model
routing, behavioral signals, provider translation, guard/refusal, budget/cost.

- **Pinned:** `d2127b83ffaee67b230e0bc1858cef8c676e0615` — release 0.4.27, 2026-07-09.
- **License:** Apache-2.0 (`LICENSE` at repo root; no per-crate BSL/Elastic headers). Integration
  with attribution is permissible. In practice plano is Rust/Envoy + a Python CLI and fak is
  Go+Rust, so every candidate below transfers as **design**, not copied code.
- **Method:** 5-way reader fan-out (router/orchestrator, signals+state, hermesllm streaming,
  guardrails+common, config+CLI+obs), then per-candidate ablate → license-gate → **witness on-axis
  against fak** → file only the net-new.

## Headline

This is a **convergent-frontier** study. plano and fak independently built routing, behavioral
signals, provider handling, guard, and budget — and on nearly every axis fak has **already shipped
the mechanism** or made a **deliberate opposite choice**. Witnessing plano's mechanisms against
fak's tree turned up exactly **one** net-new, wanted, on-axis gap: a **live latency signal + an
explicit `prefer` policy knob** in routing — which is precisely the unbuilt "latency" third of the
already-open **#600**. Everything else routes to an existing issue or documents as a validated
divergence.

Following the house pattern (cf. `BORROW-TRAJECTORY-QUERY-AGENTLENS-LAMINAR-2026-07-09.md`, which
filed the net-new and routed the rest): **1 new issue filed, 1 consolidating comment on #600, the
remainder documented here** as route-to-existing / divergence / already-shipped.

---

## The witnessed matrix

### Axis 1 — Model routing (`internal/modelroute`, epic #595)

**plano mechanism.** One small "Plano-Orchestrator" classifier drives *both* LLM routing and agent
orchestration (`brightstaff/router/orchestrator_model_v1.rs`). The routing path
(`OrchestratorService::determine_route`) ranks a route's candidate models via
**`ModelMetricsService::rank_models`** (`router/model_metrics.rs`) on a **`SelectionPolicy.prefer`**
of `{Cheapest | Fastest | None}` — Cheapest = ascending cost (models.dev + DigitalOcean catalogs),
Fastest = ascending **live p50 latency (Prometheus, `model_name` label)**, None = config order;
unmetriced models sort last. Returns a **ranked models list** (first = primary, rest = 429/5xx
fallbacks) and **pins the choice per session** (`session_cache`, `tenant:session`, TTL 600s,
`pinned: true`). Bounded classifier call: 16-turn cap, `len/4` token estimate, middle-trim of
oversized user messages, `temperature 0.01`, lenient JSON repair; `"none"` sentinel = keep the
client's model.

**fak witness.** `internal/modelroute` is the routing spine: `Route(Subject)→Decision` per-aspect +
ensembles, `ScoutRoute` (classify-first via a `Classifier` *interface*, not a hardcoded model),
`fak routebench` (offline cost/latency/quality lens), live dispatch bound only at
`cmd/fak/commit_review.go` (`ReviewDiffWithScout`, #1185). `audit_route.go` already **tie-breaks
same-preference candidates by cost** (`InputMicrosPerMillionTokens`; `cost-orders-same-preference`
test) and ranks provider health. `cost.go` is a **deliberately rough** lens
(`FrontierAnchor{In:3,Out:15}`, tier-keyed `DefaultPrices`, "a lens not a bill", overridable via
`fak route --prices`, unpriced→frontier fallback).

**Verdicts.**
- **Live latency axis + `prefer` knob → NET-NEW, FILED.** fak ranks on cost + health but has **no
  live latency axis** anywhere in routing (latency is measured *offline* in routebench, never fed
  back). The gateway **already records per-turn `elapsed`** (`internal/gateway/debug_stats.go:267`);
  the gap is aggregating it per-model and adding a symmetric latency axis + explicit
  `prefer={Cheapest|Fastest}` policy. This is the concrete, buildable first slice of #600's
  "latency" third. **Filed as a child of #600.** plano's `rank_models`/`SelectionPolicy` is the
  prior-art mechanism.
- **Full cost/latency/quality feedback loop → ROUTE TO #600 (do not re-file).** #600
  ("telemetry to learned routing (cost/latency/quality feedback, per-aspect)") already owns this
  scope. plano's `ModelMetricsService` is concrete prior-art — left as a consolidating comment.
- **Session-sticky route pinning → DIVERGENCE (not filed).** plano pins because its router is a
  **nondeterministic classifier** that could route the same session differently turn-to-turn. fak's
  `Route(Subject)` is **deterministic**, so a session already routes stably — pinning is moot on the
  policy path. It would only matter for the *nondeterministic scout* path, which is not broadly live
  yet (only #1185). Revisit if/when scout routing goes fleet-wide.
- **One-classifier-two-consumers (routing == agent selection) → DIVERGENCE.** Elegant, but fak
  dispatches agents by **lane/lease (structural)**, not by a classifier, and deliberately keeps a
  model call *out* of the routing hot path (policy-first, scout optional). Documented, not adopted.
- **Ranked-list model fallback on 429 → DIVERGENCE (cache economics).** fak fails over **accounts**
  (same model) on an org-wall and **backs off toward reset** on usage/overage 429s
  (`upstream_remedy`), because switching *models* mid-session **breaks the prompt cache** fak
  optimizes heavily. plano has no equivalent cache to protect, so next-model fallback is cheap for
  it and costly for fak. Documented.

### Axis 2 — Behavioral signals (`internal/trajhook`, `internal/antipattern`)

**plano mechanism.** A passive, **model-free**, side-effect-only signal analyzer
(`brightstaff/signals/*`) run in `on_complete` after bytes are streamed, emitting OTel span
attributes (and a 🚩 marker on the span operation name when a report is "concerning"). Three-layer
taxonomy — interaction (misalignment / stagnation / disengagement / satisfaction), execution
(failure: invalid-args/tool-not-found/auth-misuse/state-error/bad-query, keyed to the preceding
`function_call`; loops), environment (exhaustion: rate-limit/api-error/timeout/network/
context-overflow) — over a **deterministic 3-layer text-match cascade** (exact phrase → char-trigram
Jaccard → token cosine, `signals/text_processing.rs`).

**fak witness.** Heavily occupied space:
- **#2365 (CLOSED/shipped)** — behavioral lens over `session_audit.py`: **tool error rates, timeout
  kills, sleep-polls, edit churn, repeat failures**. Covers plano's execution-failure / loops /
  stagnation signals.
- **#3442 (CLOSED/shipped)** — **plain-English (LLM-evaluated) behavioral signal evaluators**
  (borrowed from Laminar). Covers the NL-defined signal path.
- **#59 (OPEN)** — "Conversation quality evaluation + learning loop." Owns the conversation-quality
  space plano's interaction signals target.
- `internal/trajhook` — hand-coded Go scorers (`duplicate_query`, `cost_outlier`, `high_deny_rate`)
  + `ContextHealth`. `internal/antipattern` — **code/commit** anti-patterns (`REDUNDANT_REWORK`,
  `UNWIRED_PKG`, `ORPHAN_FUNC`, checker-gaming), *not* conversation/tool signals.

**Verdicts.**
- **Execution-failure / loops / stagnation signals → ALREADY-SHIPPED (#2365).** No re-file.
- **NL / conversation-quality signals → ROUTE TO #3442 (closed) / #59 (open).** No re-file.
- **Deterministic 3-layer text-match cascade → DOCUMENTED TECHNIQUE (not filed).** fak has
  `internal/simhash` (locality hashing) but not a phrase-level exact→trigram-Jaccard→cosine cascade.
  It is a genuine model-free NL-matching primitive and a natural *deterministic floor* under
  #3442's LLM evaluators (mirrors fak's heuristic-screener-under-model-screener idiom) — but it has
  **no named consumer** in fak today, and filing infra without a consumer is premature. Recorded as
  available for when a consumer (e.g. a no-egress signal floor) is scoped.
- **🚩-in-span-operation-name → DOCUMENTED (minor).** Hoisting a behavioral flag into the span
  *name* (visible in any trace viewer without a query) is a cheap observability nicety; too minor to
  file.
- **Note — do NOT copy plano's signal bugs:** lowercase-offset snippet desync for non-ASCII
  (`signals/execution/failure.rs:219` slices original bytes with lowercased-match offsets),
  `NoTls` against Supabase (`state/postgresql.rs:19`), full-conversation debug logging
  (`state/mod.rs:95`).

### Axis 3 — Provider translation (`internal/gateway`, hermesllm)

**plano mechanism.** hermesllm is a full **N-provider hub-and-spoke translation** matrix
(`crates/hermesllm`): stateless 1:1 **transform** (upstream event → client event) separated from a
stateful **buffer** that owns protocol-envelope well-formedness (`AnthropicMessagesStreamBuffer`
guarantees `message_stop` never without `message_start`, synthesizes a minimal envelope for
empty/errored upstream, merges Bedrock's split stop/usage deltas). `SseChunkProcessor` reassembles
JSON split across TCP chunks. **Fail-open on a bad event, fail-closed on the protocol.**

**fak witness.** fak's gateway is **Anthropic byte-preserving passthrough** — cache-exact, it
*forwards* the provider's own envelope and never synthesizes Anthropic lifecycle events
(`gateway.go`, `dual_planner.go`). The only synthesis path is `completions.go`'s OpenAI-compat
legacy SSE (`writeCompletionStream`), whose envelope (choices/delta/finish_reason + `[DONE]`) is
far more forgiving than Anthropic's strict contract.

**Verdicts.**
- **Transform/buffer separation + N×N translation → DIVERGENCE (by design).** fak deliberately
  chose passthrough-for-cache over general N-provider translation. The transform/buffer split is a
  good reference *if* fak ever builds N×N translation — it isn't.
- **`message_stop`-without-`message_start` envelope robustness → NOT-APPLICABLE.** fak never
  synthesizes Anthropic envelopes (it forwards them), so it is structurally immune to the exact
  Claude-Code bug this buffer fixes. Recorded as a reference for the completions.go OpenAI-compat
  path only (where the envelope is looser and the risk is low).

### Axis 4 — Guard / refusal (`fak manage`, closed refusal vocabulary)

**plano mechanism.** `PromptGuards` / `GuardType::Jailbreak` / `on_exception` /
`forward_to_error_target` are **defined and parsed** in `common/configuration.rs` with DTOs in
`common/api/prompt_guard.rs` — but a cross-crate grep finds **no active dispatch site** in the WASM
filters; jailbreak/toxicity/hallucination are **delegated to the external model server** via the
`ArchFC` callout. Guardrails are also modeled as a **composable, config-ordered filter chain**
(`AgentFilterChain`/`ResolvedFilterChain`, MCP + raw-HTTP transports).

**fak witness.** `fak manage` is a **closed 12-reason refusal vocabulary**, deny-as-value
(RETRYABLE/WAIT/ESCALATE/TERMINAL disposition), default-deny, no-flag-bypass, structural
provenance-based — **inline, synchronous, no judge model**.

**Verdicts.**
- **Model-delegated guard → DIVERGENCE that VALIDATES fak.** plano's guard-plumbing-without-inline-
  enforcement is exactly the failure mode fak's *structural inline floor* is built to avoid.
- **Composable filter chain → DIVERGENCE.** fak deliberately rejects a config-ordered, bypassable
  filter chain in favor of a non-bypassable structural floor. Documented as a design contrast; no
  action.

### Axis 5 — Budget / cost / rate-limit (`fak spend`, `upstream_remedy`, obs)

**plano mechanism.** (a) **Proactive token-bucket egress rate limiting** (`common/ratelimit.rs`,
`governor`; provider→{header→keyed limiter}). (b) **Two-sided price/name alias expansion**
(`obs/pricing.py`): emitted `claude-haiku-4-5-20251001` never matches catalog `anthropic-claude-
haiku-4.5`, so it expands *both* sides (date-strip, `provider/`-strip, dash↔dot, family↔version
reorder) and treats a 0-rate as **unknown, not $0**. (c) **In-process OTLP sink + ring buffer +
Rich TUI** (`obs/collector.py`, `obs/render.py`): a self-contained live cost/latency console over
its own spans, no Prometheus/Grafana, with bind-failure remediation UX and trace reassembly by
`parentSpanId`.

**fak witness.** `fak spend` (cross-account rollup + provenance), `upstream_remedy` (429 classifier,
account failover, `SwitchModel`, unified-ratelimit headers), nightrun JSONL corpora, `eveimport`
(OTel import). Rate handling is **header-reactive** (reads the provider's *actual*
`anthropic-ratelimit-*` / `x-ratelimit-*`), not a local guessed bucket.

**Verdicts.**
- **Proactive token bucket → DIVERGENCE (fak's is arguably better).** fak reacts to the provider's
  *real* limit headers rather than guessing a local rate; a blind local bucket would be less
  accurate. Documented, not adopted.
- **Two-sided price/name alias expansion → DEFERRED TECHNIQUE (trade-off flagged).** The technique
  is genuinely useful (model-name normalization is a real fak pain), but adopting a per-model price
  catalog fights `cost.go`'s **deliberate blend-independence** ("rough proportional ladder, zero
  price table to maintain"). If ever wanted, apply it only to the *override / unpriced-fallback*
  path (resolve more models to real prices when the user supplies them) while keeping the rough
  proportional default — noted in the #600 comment, not filed.
- **In-process live cost/latency TUI → DEFERRED CANDIDATE.** fak has post-hoc JSONL + `fak spend`
  but no *live* console over the running gateway's spans. Genuinely absent and operator-useful, but
  somewhat against fak's grain (fak prefers witnessable/replayable git corpora over ephemeral live
  views). Flagged as a possible follow-on; not filed pending a clear home.

### Off-axis (config/CLI tooling)

plano's config compiler (`validate → migrate-and-warn → infer → render`), zero-config synthesis with
env-driven auth degradation, and template/demo **drift-gate** are strong techniques — but they land
on dev-tooling, not the five axes, and fak already owns drift-gating (`fak index freshness`).
Documented for reference; no action.

---

## Outcome

| Candidate | Axis | Verdict |
|---|---|---|
| Live latency signal + `prefer={Cheapest\|Fastest}` policy | routing | **FILED** (child of #600) |
| Full cost/latency/quality feedback loop | routing | ROUTE → **#600** (comment; prior-art) |
| Session-sticky route pinning | routing | DIVERGENCE (fak is deterministic) |
| One-classifier-two-consumers | routing | DIVERGENCE (fak dispatches structurally) |
| Ranked-list model fallback on 429 | routing/budget | DIVERGENCE (cache economics) |
| Execution-failure / loops / stagnation signals | signals | ALREADY-SHIPPED (#2365) |
| NL / conversation-quality signals | signals | ROUTE → #3442 (closed) / #59 (open) |
| Deterministic 3-layer text-match cascade | signals | DOCUMENTED technique (no consumer yet) |
| Transform/buffer N×N stream translation | gateway | DIVERGENCE (fak = passthrough) |
| Envelope well-formedness (`message_stop` guard) | gateway | NOT-APPLICABLE (fak forwards, never synthesizes) |
| Model-delegated guard / composable filter chain | guard | DIVERGENCE that VALIDATES fak |
| Proactive token-bucket rate limit | budget | DIVERGENCE (fak = header-reactive) |
| Two-sided price/name alias expansion | budget | DEFERRED technique (blend-independence trade-off) |
| In-process live cost/latency TUI | budget/obs | DEFERRED candidate |
| Config compiler / zero-config / drift-gate | off-axis | Documented (fak owns freshness) |

**Filed:** `feat(modelroute): live per-model latency signal + Fastest selection policy — borrowed
from plano/archgw@d2127b8` (child of #600 / epic #595).
**Routed:** consolidating prior-art comment on #600.
