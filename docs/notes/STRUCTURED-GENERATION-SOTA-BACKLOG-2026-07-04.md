---
title: "Structured generation SOTA, fak fit, and backlog split"
description: "Current structured-output / constrained-decoding landscape, why the phrase feels quieter, and the fak-specific integration plus minimal-native-spine backlog."
date: 2026-07-04
status: research + backlog map
---

# Structured generation SOTA, fak fit, and backlog split

## Short answer

Structured generation did not fall out of use. It got absorbed into three louder
surfaces:

- **Provider-native structured outputs**: OpenAI `response_format` /
  Responses `text.format`, strict function calling, Anthropic/Gemini/xAI native
  output modes.
- **Serving-engine guided decoding**: vLLM/SGLang/TensorRT-LLM using XGrammar,
  llguidance, Outlines, or similar token-mask engines.
- **Agent/tool ecosystems**: LangChain, Pydantic AI, Instructor, BAML, LlamaIndex,
  MCP, and tool calling use schemas as product plumbing rather than as a research
  headline.

For fak, the winning stance is not "be another structured-output library." It is:

> Let popular libraries and engines produce schema-valid candidates; fak remains
> the boundary that adjudicates whether a valid candidate is allowed to have the
> requested effect.

That means three workstreams:

1. **Ride integrations**: preserve structured-output fields from OpenAI-compatible
   clients and popular libraries to vLLM/SGLang/OpenAI-compatible upstreams, then
   adjudicate tool calls exactly as today.
2. **Minimal native spine**: own the small compiler that turns fak tool schemas into
   `model.LogitMask` for the in-kernel engine, proving the seam without trying to
   out-engineer XGrammar.
3. **Evidence discipline**: measure structural validity, semantic/value accuracy,
   policy safety, and latency separately. A valid JSON object can still be wrong or
   forbidden.

## Current SOTA readout

### Engine layer

- **XGrammar-2** is the current agentic SOTA reference. The May 2026 MLC post
  positions it as an agent-focused upgrade with Structural Tag, cross-grammar
  caching, repetition-state compression, batching, speculative-decoding support,
  Python/C++/Rust/JS surfaces, and integrations in SGLang, vLLM, TensorRT-LLM, and
  MLC-LLM. It claims 100% schema accuracy in its tool-calling evals and up to 80x
  compilation speedup over XGrammar on large tool sets.
- **XGrammar** remains the common engine-default shape: JSON Schema / regex /
  custom CFG lowered to per-token masks, with broad platform and language support.
  The repo says it is the default structured-generation backend in major inference
  engines, including vLLM, SGLang, TensorRT-LLM, and MLC-LLM.
- **vLLM** exposes structured outputs with `structured_outputs` and supports
  xgrammar/guidance/outlines-style backends. Its docs warn that older guided fields
  were removed in v0.12.0, which matters for fak compatibility tests: the gateway
  should preserve OpenAI-compatible `response_format` and a small allow-list of
  engine-native fields, but recipes should prefer the current API names.
- **SGLang** supports JSON Schema, regex, and EBNF constraints with XGrammar as the
  default backend, plus Outlines and llguidance alternatives.
- **llguidance** is the fast lazy-mask contender: its README reports roughly 50 us
  mask compute on a 128k-token tokenizer for typical JSON-schema-derived grammars,
  broad integrations, and no significant startup cost.
- **Outlines** remains a high-level constrained-generation library with guaranteed
  schema compliance, regex/CFG support, and compilation-once semantics.

### Provider and app-library layer

- **OpenAI Structured Outputs** now has two first-class forms: function calling for
  model-to-system actions and `json_schema` response formats for structured user
  responses. In the Responses API the carrier is `text.format`; in Chat
  Completions it is `response_format`.
- **LangChain** treats structured output as an agent response strategy. Its
  `ProviderStrategy` uses model-provider native structured output where available;
  `ToolStrategy` turns the schema into tool calling.
- **Pydantic AI** names three output modes: Tool Output, Native Output, and
  Prompted Output. Only the first two can map to an actual provider/engine
  enforcement path; Prompted Output is parse/validate/retry plumbing.
- **Instructor** is still a major validation/retry layer for typed extraction. Its
  core value is provider portability, Pydantic validation, automatic retries, and
  streaming, not necessarily token-level constrained decoding.
- **BAML** frames structured output as a typed DSL and generated client workflow.
  This is valuable for fak as a producer of schemas and traces, not as a runtime
  decoder to reimplement.

### Benchmarks and cautionary findings

- **JSONSchemaBench** is the right structural-coverage benchmark: 10K real-world
  JSON schemas, six constrained decoding frameworks, and axes for efficiency,
  coverage, and output quality. It makes "supports JSON Schema" a measured claim
  instead of a checklist.
- **ExtractBench** is the caution: complex extraction can get structurally valid
  while becoming less useful. In its 2026 study, structured-output mode dropped
  overall validity from 51% to 37%, and GPT-5 credit-agreement pass rate from
  86.9% to 70.0%, because schema restrictions and long document grounding interact.
- **CodeSpear / CodeShield** is the security caution: grammar-constrained decoding
  can become an attack surface for code generation. A schema/grammar is not a
  security policy.

## Popularity signals

GitHub repo snapshots collected on 2026-07-04:

| Repo | Stars | Why it matters to fak |
|---|---:|---|
| `langchain-ai/langchain` | 140,880 | structured output is now an agent-framework feature |
| `vllm-project/vllm` | 85,306 | primary OpenAI-compatible serving target |
| `run-llama/llama_index` | 50,632 | document/extraction workflows want typed outputs |
| `sgl-project/sglang` | 29,883 | primary local guided-decoding serving target |
| `guidance-ai/guidance` | 21,532 | grammar DSL + structured local generation lineage |
| `pydantic/pydantic-ai` | 18,196 | typed Python agent outputs |
| `dottxt-ai/outlines` | 14,365 | structured generation library |
| `567-labs/instructor` | 13,378 | validation/retry layer with many providers |
| `BoundaryML/baml` | 8,508 | typed DSL/codegen approach |
| `noamgat/lm-format-enforcer` | 2,025 | older/portable mask enforcement |
| `mlc-ai/xgrammar` | 1,774 | engine-level structured-generation backend |
| `guidance-ai/llguidance` | 804 | fast backend used by engines |

Noisy GitHub issue-search counts also show the phrase shift:

| Query | 2024 | 2025 | 2026 to Jul 4 |
|---|---:|---:|---:|
| `"structured generation" "LLM"` | 426 | 4,013 | 7,286 |
| `"structured output" "LLM"` | 3,878 | 16,589 | 49,388 |
| `"constrained decoding" "LLM"` | 86 | 262 | 1,206 |
| `"tool calling" "LLM"` | 4,420 | 14,607 | 34,761 |
| `"agent" "MCP"` | 1,542 | 452,794 | 1,054,184 |

Treat these as directional only. The useful conclusion is not exact volume; it is
that "structured generation" is now mostly discussed as structured output, tool
calling, MCP, provider-native output, and engine guided decoding.

## Why it feels quieter

1. **The syntax part became table stakes.** Engines and APIs advertise "Structured
   Outputs" or "tool calling" instead of "structured generation."
2. **The interesting failures moved up-stack.** The hard problems are now semantic
   correctness, large/nested schemas, reasoning-channel coexistence, streaming,
   speculative decoding, and security under attacker-chosen grammars.
3. **Frameworks hid it behind types.** A LangChain/Pydantic/Instructor/BAML user
   often sees a Python type, Pydantic model, Zod schema, or BAML function, not a
   grammar engine.
4. **Agent discussion displaced decoder discussion.** MCP, tool registries,
   reasoning models, and agent harnesses are where schema enforcement shows up.
5. **"100% valid JSON" stopped being the buying criterion.** Benchmarks now show
   that valid output can still be inaccurate, rejected by provider schema subsets,
   slow, or unsafe.

## fak map

### Already shipped

- `internal/gateway/wire.go`, `http.go`, and `stream_proxy.go` parse and forward
  `response_format`, `logit_bias`, and selected guided-decoding fields to the ride
  engine.
- `internal/gateway/structured_output_passthrough_test.go` proves those fields
  cross the upstream wire and that generated tool calls still enter fak
  adjudication.
- `internal/model/constraint.go` provides the native sink: `LogitBias`,
  `LogitMask`, `AllowedSetMask`, `StepMask`, and `GenerateConstrained`.
- `internal/model/constraint_test.go` proves bit-exact-off, active logit bias,
  flagged schema mask, and batch constrained decode behavior.
- `internal/agent/guided_decode_adjudication_test.go` proves a native masked
  decode can emit a tool-call envelope that enters `grammar.Rung` unchanged.

### Remaining gap

The missing piece is the compiler:

> `grammar.Grammar` / tool JSON Schema + tokenizer -> `model.LogitMask`

That compiler must live above `internal/model` because `internal/model` cannot
import `internal/grammar`. It should reuse existing schema normalization/dedup
patterns where possible and feed the model seam rather than modify the model
package directly.

### Minimal native spine

The first own-spine should deliberately be small:

- Input: one or two tool schemas, `tool_choice=required`, required scalar fields.
- Intermediate: canonical `oneOf` tool-call JSON Schema.
- Output: a `StepMask` / trie-backed `LogitMask` built with a tokenizer interface.
- Witness: synthetic CPU model + byte tokenizer emits
  `{"name":"lookup","arguments":{"q":"sf"}}`; the emitted arguments enter
  `grammar.Rung` unchanged; unconstrained decode remains bit-exact through
  existing tests.

Non-goal: matching XGrammar-2 performance or full JSON Schema coverage in the
first spine. fak should integrate XGrammar/vLLM/SGLang for production ride mode
and own only the smallest native proof needed to keep the kernel honest.

## Integration posture

| Library / surface | fak integration posture |
|---|---|
| OpenAI SDKs | set `base_url` to fak; preserve `response_format`, `text.format`, tools, `strict`, `logit_bias`; gate tool calls |
| vLLM / SGLang | ride-mode upstreams; preserve current structured-output fields; measure gateway tax separately |
| LangChain | recipe for `ProviderStrategy` and `ToolStrategy` through fak; prove tool calls still adjudicate |
| Pydantic AI | recipe that distinguishes Tool Output, Native Output, and Prompted Output; only Tool/Native are enforcement-grade |
| Instructor | recipe for validation/retry through fak base URL; fak should not claim Instructor retries are constrained decoding |
| BAML | recipe/codegen note for OpenAI-compatible fak endpoint; useful as typed schema producer |
| Outlines / Guidance / llguidance / XGrammar | prior art and optional engine/library dependencies; do not fork internals into fak's hot path |

## Backlog split

1. [#2596](https://github.com/anthony-chaudhary/fak/issues/2596) Native
   compiler spine: tool schemas to `LogitMask`, with byte-tokenizer witness.
2. [#2597](https://github.com/anthony-chaudhary/fak/issues/2597) Integration
   cookbook: OpenAI SDK, LangChain, Pydantic AI, Instructor, BAML, vLLM, and
   SGLang recipes through fak.
3. [#2598](https://github.com/anthony-chaudhary/fak/issues/2598) Benchmark and
   risk readout: JSONSchemaBench/ExtractBench/CodeSpear mapped to fak claims, so
   we never equate "schema-valid" with "correct" or "safe."
4. [#2599](https://github.com/anthony-chaudhary/fak/issues/2599) Positioning
   note: explain that structured generation became infrastructure and state
   fak's differentiator as effect adjudication after valid generation.

## #26 base-item close-out

[#26](https://github.com/anthony-chaudhary/fak/issues/26) is the parent
base-item ("structured/guided decoding feeding tool-call gating"). Its
base-item acceptance is satisfied and independently re-witnessed on
2026-07-04; the residual is carried forward as a distinct tracked child, so
#26 closes as a base item rather than staying open behind a non-goal.

Acceptance checklist bound to evidence:

| #26 acceptance item | State | Witness |
|---|---|---|
| Constrained generation on at least one track (Track A ride is the min bar) | shipped | [#907](https://github.com/anthony-chaudhary/fak/issues/907) (CLOSED) |
| `SampleParams` carries structured-decode fields additively; call sites compile | shipped | `internal/agent/chat.go` `ResponseFormat`/`LogitBias` + `WithResponseFormat`/`WithLogitBias` opts |
| Gateway parses OpenAI `response_format`/`logit_bias` and forwards them | shipped | `TestChatProxyForwardsStructuredOutputFieldsToRideEngine`, `...OmitsStructuredOutputFieldsWhenAbsent` (PASS) |
| Ride path enforces + integrates with the whole-turn gate | shipped | `internal/gateway/structured_output_passthrough_test.go` |
| Native path: logit-bias/grammar mask at the StepBatch step, unconstrained bit-exact | shipped | [#929](https://github.com/anthony-chaudhary/fak/issues/929) (CLOSED); `internal/model/constraint.go`; `constraint_test.go` (8 tests PASS) |
| Test: constrained tool-call workload admitted/transformed as a policy decision | shipped | `TestNativeMaskedDecodeEntersGrammarAdjudication` (PASS) |
| Reconciliation note posted to #348 | superseded | see invalidating assumption below |

**Promotion evidence (gen/now, base-item shipped):** #907 and #929 both
CLOSED; `CLAIMS.md` records `[SHIPPED] Native decode-time constraint hook
(#929, the in-kernel half of #907/#26)`; the three witness suites
(`internal/gateway`, `internal/model`, `internal/agent`) are green.

**Retirement-by-completion:** the one remaining fak-owned gap — the
`grammar.Grammar` / tool JSON Schema + tokenizer -> `model.LogitMask`
compiler — is a #26 non-goal ("a full high-performance grammar compiler ...
is out of scope; a correct minimal logit-mask / JSON-schema constraint is the
bar", which #929 met). It is tracked in
[#2596](https://github.com/anthony-chaudhary/fak/issues/2596) (OPEN by
design), so #26 carries no untracked debt.

**Invalidating assumption:** the acceptance item "Reconciliation note posted
to #348" assumes #348 is a live GitHub issue. It is not — `gh issue view 348`
resolves to nothing; #348 is an internal-tracker sibling number from the
migration, not a fak GitHub issue. That acceptance item is unreachable as
literally written and is superseded by this note, `CLAIMS.md`, and the owner's
2026-07-04 reconciliation comment on #26.

**Classification (recommended, repair blocked):** #26 is unclassified
(no `gen/*` label; milestone "Ship releases automatically on a green trunk").
The evidence classifies it as `gen/now` — a shipped base serving item. The
label/milestone repair is blocked by the preview-confirm gate (`gh issue edit`
is refused); an operator with issue-edit rights should add `generation` +
`gen/now` and bind the `Generation G0 - Now / Immediate` milestone.

## Sources

- XGrammar-2 blog: <https://blog.mlc.ai/2026/05/04/xgrammar-2-fast-customizable-structured-generation>
- XGrammar docs/repo: <https://xgrammar.mlc.ai/docs/> and <https://github.com/mlc-ai/xgrammar>
- vLLM structured outputs: <https://docs.vllm.ai/en/latest/features/structured_outputs/>
- SGLang structured outputs: <https://docs.sglang.io/docs/advanced_features/structured_outputs>
- llguidance repo: <https://github.com/guidance-ai/llguidance>
- Outlines docs: <https://dottxt-ai.github.io/outlines/latest/>
- OpenAI Structured Outputs: <https://developers.openai.com/api/docs/guides/structured-outputs>
- LangChain structured output: <https://docs.langchain.com/oss/python/langchain/structured-output>
- Pydantic AI output modes: <https://pydantic.dev/docs/ai/core-concepts/output/>
- Instructor docs: <https://python.useinstructor.com/>
- BAML docs: <https://docs.boundaryml.com/home>
- JSONSchemaBench: <https://arxiv.org/abs/2501.10868>
- ExtractBench: <https://arxiv.org/html/2602.12247v1>
- CodeSpear / CodeShield: <https://arxiv.org/abs/2606.11817>
