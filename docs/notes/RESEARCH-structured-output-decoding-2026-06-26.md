---
title: "Structured-output decoding for schema-valid tool-call candidates"
description: "Research for #907: a current guided-decoding SOTA readout, a field map from every structured-output surface (OpenAI response_format, vLLM/SGLang guided fields, Anthropic/MCP tool schema, fak Grammar) onto one canonical internal constraint shape, what fak forwards in ride mode vs owns natively, and how constrained candidates still reconcile with whole-turn adjudication."
date: 2026-06-26
issue: 907
status: research + ride-mode passthrough wired (internal/gateway) + native sampler sink shipped (internal/model, #929); the schema→token compiler is the remaining follow-on
---

# Structured-output decoding for schema-valid tool-call candidates (#907)

A constrained decoder makes the model's output *well-formed before it exists*:
the engine masks each step's logits so the only tokens it can emit keep the
output on a valid JSON-schema / regex / grammar path. For tool calls this means
the `{"name": …, "arguments": …}` shape is correct by construction instead of
parsed-and-repaired after the fact.

This is **upstream of fak's gate, not a replacement for it.** Structured outputs
reduce the malformed-candidate rate the gate has to cope with; they say nothing
about whether the call is *allowed*. A perfectly-formed `delete_account` call is
still denied by structure. So the design rule is: forward the constraint to
whoever can enforce it, then adjudicate the result unchanged. This note is the
guided-decoding readout, the field map onto one internal constraint shape, and
the split between what fak forwards (ride mode) and what it must own (native).

It is the structured-output companion to
[`RESEARCH-grammar-constrained-tool-call-decoding-2026-06-22.md`](RESEARCH-grammar-constrained-tool-call-decoding-2026-06-22.md)
(#469), which framed the upstream-constraint vs downstream-repair trade. That note
argued *whether* to constrain; this one names the *carrier* and *shape* and wires
the ride-mode passthrough.

## Guided-decoding SOTA readout (current engine docs)

| Engine / library | Mechanism | What it constrains | Doc |
|---|---|---|---|
| **vLLM** structured outputs | `guided_json`, `guided_regex`, `guided_grammar`, `guided_choice`, and the OpenAI-compatible `response_format` (`json_object` / `json_schema`) | JSON Schema, regex, EBNF/CFG, fixed choice set | <https://docs.vllm.ai/en/latest/features/structured_outputs.html> · OpenAI surface: <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html> |
| **SGLang** structured outputs | `response_format` (`json_schema`), `regex`, `ebnf` (per-request); backend selectable (xgrammar / outlines / llguidance) | JSON Schema, regex, EBNF | <https://docs.sglang.ai/advanced_features/structured_outputs.html> |
| **XGrammar** | compressed context-free-grammar engine; the constraint engine *inside* vLLM/SGLang | CFG / JSON Schema lowered to a per-step token mask, low per-token overhead | <https://xgrammar.mlc.ai/docs/> |
| **llguidance** | constrained-decoding library (Guidance grammars); an alternate backend for the engines above | regex / CFG / JSON Schema, fast lazy mask computation | <https://github.com/guidance-ai/llguidance> |
| **Outlines** | structured-generation library; FSM-indexed logit masks | regex, JSON Schema (via regex/CFG), Pydantic models | <https://dottxt-ai.github.io/outlines/latest/> |
| **llama.cpp / ollama** | `--grammar` (GBNF), `--json-schema`; ollama `format` (`"json"` or a JSON-Schema object) | GBNF grammar, JSON Schema | (covered in #469) |

The common shape across all of them: **the engine accepts a constraint
description (almost always reducible to JSON Schema), compiles it to a per-step
token mask, and applies the mask to the logits before sampling.** vLLM and SGLang
both accept the OpenAI `response_format` object directly, which is why that one
carrier is enough to drive ride-mode structured outputs without an engine-specific
field for the common case; the engine-native `guided_*` / `regex` / `ebnf` fields
are the escape hatch for regex/EBNF constraints the OpenAI shape can't express.

## The canonical internal constraint shape

Every surface fak touches can be normalized to **one JSON-Schema document** — the
lingua franca every engine accepts — sourced from fak's own per-tool descriptor:

- **fak's descriptor of record is `internal/grammar` (`grammar.Grammar` /
  `grammar.Param`, plus the `Aliases` synonym map).** It is already derived from a
  tool's MCP/JSON-Schema `input_schema`, content-addressed, and deduped fleet-wide.
- **The canonical constraint is a JSON-Schema `oneOf` union over the inbound
  tools**: each branch pins `name` to a `const` and sets `arguments` to that tool's
  `input_schema` (the design sketch in #469). This is the single shape that
  (a) every ride engine accepts via `response_format` / `guided_json` /
  `json_schema`, and (b) a native sampler can compile to a logit mask.
- **JSON Schema is the floor; regex/EBNF are lowerings.** A backend that only
  speaks GBNF (raw llama.cpp) gets the same `oneOf` mechanically compiled to GBNF;
  nothing upstream forks per engine.

So: **`grammar.Grammar` → `oneOf` JSON Schema → carried on the wire by the
`response_format` object (ride mode) or compiled to a logit mask (native mode).**

### Field map: every structured-output surface → the internal shape

| Source surface | Field / carrier | Constraint it expresses | Where it lands in fak |
|---|---|---|---|
| OpenAI Chat `response_format` | `{type: "json_object" \| "json_schema", json_schema:{schema}}` | JSON object / JSON Schema | `gateway.ChatRequest.ResponseFormat` → `agent.SampleParams.ResponseFormat` → OpenAI wire `response_format` **verbatim** (#907, now wired) |
| OpenAI `logit_bias` | `{token_id: bias}` (−100..100) | per-token mask | `gateway.ChatRequest.LogitBias` → `agent.SampleParams.LogitBias` → wire `logit_bias` verbatim (#907) |
| OpenAI **Responses** API | `text.format` (`json_schema`) | JSON Schema | `internal/agent/adapters.go responsesText()` maps `response_format` → `text.format` |
| vLLM guided fields | `guided_json` / `guided_regex` / `guided_grammar` / `guided_choice`, or `response_format` | JSON Schema / regex / EBNF / choice | the `response_format` path forwards verbatim; `guided_*` ride the provider `ExtraBody` |
| SGLang structured fields | `response_format` (`json_schema`), `regex`, `ebnf` | JSON Schema / regex / EBNF | `response_format` forwards verbatim; `regex`/`ebnf` ride `ExtraBody` |
| Anthropic tools | `tools[].input_schema` (+ `tool_choice`) | JSON Schema (per-tool args) | `internal/gateway/anthropic_server.go` forwards `input_schema` verbatim; downstream → `grammar.Grammar` |
| MCP tool schema | `tool.inputSchema` | JSON Schema (per-tool args) | normalized into `grammar.Grammar` / `Param` |
| **fak Grammar descriptor** | `grammar.Grammar` / `grammar.Param` (+ `Aliases`) | per-tool arg shape, content-addressed | **the canonical internal shape** the rest map onto |

## Ride mode: what fak forwards (now wired)

The issue's first question — *which constraints are pass-through-only in ride
mode?* — is now answered in code, not prose. The OpenAI `/v1/chat/completions`
proxy forwards the client's `response_format` and `logit_bias` to the ride engine
**verbatim**, so vLLM/SGLang enforce the JSON-schema/grammar during generation and
fak adjudicates the resulting tool candidate afterward:

- `internal/gateway/wire.go` — `ChatRequest` now parses `response_format` /
  `logit_bias` (previously dropped as "unknown OpenAI fields").
- `internal/gateway/http.go` (`handleChatCompletions`) and
  `internal/gateway/stream_proxy.go` (`streamChatLive`) — both pass
  `agent.WithResponseFormat` / `agent.WithLogitBias` to the planner, so buffered
  and streamed turns are constrained identically.
- `internal/agent/adapters.go` already marshals both onto the OpenAI/vLLM/SGLang
  wire (`response_format` / `logit_bias`, `omitempty`); the gateway was the only
  missing hop.

Proof: `internal/gateway/structured_output_passthrough_test.go` stands an
OpenAI-compatible ride engine behind the gateway, sends a `json_schema`
`response_format` + a `logit_bias` mask, and asserts (1) both crossed the upstream
wire byte-equivalent, and (2) the constrained generation's tool candidate still
entered the gate (deny dropped, allow kept, both adjudicated). A second test pins
**bit-exact drop-in**: a request with neither field produces an upstream body with
neither key present, so a non-structured client is never silently constrained.

This is forward-only by design: fak never *simulates* a constraint a backend can't
apply, and the gate stays the source of policy truth. The honest residual is that
the engine-native `guided_*` / `regex` / `ebnf` fields ride `ExtraBody` rather than
a first-class gateway field — fine for the common `response_format` case, a named
gap for regex/EBNF-only constraints.

## Native mode: what fak must own (issue #929)

On the in-kernel reference engine (`internal/model`) there is no upstream to
forward to — the decode is fak's own greedy/argmax over post-head logits. To make
the *same* `response_format` carrier enforceable there, fak owns the sampler hook.
That shipped as **#929** (the issue's "implementation issue for the first minimal
sampler constraint") in `internal/model/constraint.go`, smallest-first:

1. a **logit-bias mask** at the sampler boundary — `DecodeConstraint.Bias` applies
   the per-request `SampleParams.LogitBias` carrier (clamped to ±100) before argmax;
   this is the in-kernel sink the carrier was already threading to, then
2. a **JSON-schema / grammar logit mask** behind the `FAK_NATIVE_GUIDED_DECODE`
   feature flag (default OFF) — the injected `LogitMask` seam (`AllowedSetMask` /
   `StepMask`) a higher layer fills with the compiled `oneOf`-of-tools shape above;
   `internal/model` stays tier-1 and never imports `internal/grammar`.

The load-bearing acceptance criterion, **bit-exact-off**, holds: with no constraint
set, `GenerateConstrained` is token-identical (`max|Δ| = 0`) to today's greedy
`Session.Generate` — the hook is a proven no-op on the unconstrained path (witness:
`internal/model/constraint_test.go`). The remaining follow-on is the tokenizer-aware
compiler that lowers a `grammar.Grammar` `oneOf`-of-tools schema (deduped once per
fleet via `grammar.Rung`) to those per-step token masks — the `[STUB]` tracked in
CLAIMS.md; #929 shipped the sampler sink + the concrete mask primitives, not the
schema→token compiler.

## How constrained candidates reconcile with whole-turn adjudication

Structured outputs change *well-formedness*, never *permission*. The reconcile
rule, unchanged by this work:

- **Buffered turn** — the whole tool-call set is adjudicated after generation;
  deny drops, transform redacts, allow keeps. A constrained candidate is simply a
  candidate that parses on the first try.
- **Streamed turn** — `streamChatLive` + the lift-guard HOLD every proposed call
  off-wire until the whole turn is buffered and adjudicated; no un-adjudicated call
  is ever streamed, constrained or not. Structured outputs make the held call
  well-formed; they do not let it skip the gate.
- **`tool_choice` interaction** (from #469): `required`/`any` is the clean case to
  constrain to the `oneOf` union; `auto` needs a top-level `{text | tool_call}`
  union or stays unconstrained to preserve the model's decline option; over-
  constraining `auto` trades a malformed-call error for a worse wrong-action error.

## House rule: schema shape for an agent's own find → propose → verify pass

The guided decoding above is the *engine* making one candidate well-formed. The
dual discipline lives a layer up, in how an agent **designs its own
StructuredOutput schema** when it runs a find → propose → verify pass (e.g. a
`Workflow` fan-out). The rule, learned by hitting the wall live:

> **find/propose/verify → flat schema + separate stages, never one nested schema.**

The failure mode is a single heavy schema — `claims[] + proposedIssues[] +
adversarial`, each item carrying its own proposed issue and its own adversarial
verdict. One call cannot populate all three concerns well: the model spends its
output on the envelope and the result either fails validation or degrades into
shallow findings, stub proposals, and rubber-stamp verdicts. The recovery that
works is to **ground inline** — flatten to shallow top-level fields and split the
concerns across separate calls / pipeline stages:

- **Schema depth ≤ 2.** A finding is `{title, file, evidence}` — flat, with no
  sub-objects the same call must also populate.
- **One concern per call.** Finding, proposing, and verifying are *different*
  calls, not *fields* of one schema.
- **Verify is adversarial and separate** — a `{isReal, reason}` verdict call per
  finding, run as its own `parallel()` / `pipeline()` stage.
- **Prefer `pipeline()` over a barrier'd mega-schema.** Item A can verify while
  item B is still being found; a single deep schema serializes all of it into one
  fragile call.

It is the same upstream-constrain-then-adjudicate split as the gateway path, turned
on the agent's own output: keep each call's shape simple enough to satisfy by
construction, then reconcile the pieces in a later stage instead of demanding one
call get everything right at once. The code sibling is **#1339** (split the heavy
`claims[]+proposedIssues[]+adversarial` StructuredOutput); this note is the **#1340**
house-rule home.

## Cross-links

- **#469** — upstream-constraint vs downstream-repair research; the `oneOf`
  compiler sketch and the `tool_choice` analysis this note builds on.
- **#560** — the `response_format` / `logit_bias` carrier seam in
  `internal/agent` that #907 connected to the gateway.
- **#649** — grammar / message-datatype work.
- **#338 / #313** — grammar / repair demos.
- **#929** — the native sampler implementation issue, shipped (logit-bias first,
  schema mask flagged behind `FAK_NATIVE_GUIDED_DECODE`, bit-exact-off).
- **#1340 / #1339** — the find/propose/verify schema house rule (the section
  above) and its code sibling: flatten + stage the heavy nested StructuredOutput
  rather than nesting `claims[]+proposedIssues[]+adversarial` into one call.

## References (code)

- `internal/gateway/wire.go` — `ChatRequest.ResponseFormat` / `LogitBias`.
- `internal/gateway/http.go`, `internal/gateway/stream_proxy.go` — the two
  forwarding sites (buffered + streamed).
- `internal/gateway/structured_output_passthrough_test.go` — the pass-through +
  bit-exact-off witnesses.
- `internal/agent/adapters.go`, `internal/agent/chat.go`,
  `internal/agent/stream.go` — the `response_format` / `logit_bias` carrier onto
  the OpenAI/vLLM/SGLang wire.
- `internal/grammar/grammar.go` — the canonical per-tool descriptor.
- `internal/model` — the in-kernel greedy decode boundary (#929's home);
  `constraint.go` is the shipped sampler sink (`DecodeConstraint` / `LogitMask` /
  `GenerateConstrained`), `constraint_test.go` the bit-exact-off + active-bias +
  flagged-mask witnesses.
