---
title: "Grammar-constrained tool-call decoding: upstream prevention vs downstream repair"
description: "Research for #469: where constrained decoding can force weak models to emit valid tool calls, how to compile an Anthropic input_schema into a per-turn constraint, and whether it is worth wiring through the fak gateway or stays a backend concern."
date: 2026-06-22
issue: 469
status: research (no code phase)
---

# Grammar-constrained tool-call decoding (#469)

A small model under a large multi-tool prompt sometimes emits a tool call as
free text instead of through the structured channel, or gets the argument shape
wrong. fak handles that today by reading whatever arrives and fixing it. The
question this note answers is whether we should instead stop the bad output from
being generated in the first place, and if so, where that is actually possible.

Two levers, on opposite sides of the decode step:

- **Downstream repair** — let the model generate freely, then parse and repair
  what it produced. Provider-agnostic, already shipped.
- **Upstream constraint** — mask the decoder's logits each step so the only
  tokens it can emit keep the output on a valid-tool-call path. Backend-specific,
  not yet wired.

Both are worth having. They fail in different places, so running both is defense
in depth rather than a choice between them.

## What fak does today (downstream, shipped)

Two rungs, both provider-agnostic and always on:

- `internal/grammar/grammar.go` — the grammar rung. It derives a tool's argument
  shape from the MCP/JSON-Schema `input_schema` (`grammar.Grammar` / `Param`),
  content-addresses and dedupes it fleet-wide, and repairs a malformed-but-fixable
  call in-syscall: positional args zipped into named params, and a synonym key
  renamed to its canonical param (the `Aliases` map). A well-formed call defers; an
  unrepairable one denies with `MISROUTE`; a tool with no grammar fails open.
- `internal/agent/toolcall_fallback.go` — the dialect lift. When a model writes the
  call into the content string instead of the structured field, six extractors
  recover it: Hermes/Qwen `<tool_call>`, `<function_call>`, Llama-3.1
  `<|python_tag|>`, Mistral `[TOOL_CALLS]`, fenced ```json```, and bare objects. Each
  is conservative and skips a nameless block rather than fabricate a call.

These cover the case where the model already emitted something. Neither prevents
the malformed output from being produced, which is the cost upstream constraint
would remove.

## Where upstream constraint is feasible, per backend

Constrained decoding needs the engine to accept a grammar or schema and apply it
to the token logits. That is an engine capability, so the matrix is about which
serving path fak talks to, not about fak itself.

| Serving path | Constraint mechanism | Tool-call grammar enforceable? | Notes |
|---|---|---|---|
| llama.cpp | `--grammar` (GBNF), `--json-schema` | Yes | Native GBNF; the engine under ollama |
| ollama | `format` (`"json"` or a JSON Schema object) | Yes (schema-level) | `format:"json"` forces valid JSON; a schema object forces the shape. Delegates to llama.cpp grammars |
| vLLM | `guided_json` / `guided_grammar` (outlines, lm-format-enforcer, xgrammar) | Yes | Per-request guided decoding |
| SGLang | regex / JSON-schema constraints | Yes | Per-request |
| Hosted Anthropic / OpenAI APIs | `tool_choice` only | No grammar lever | The model's logits are not exposed. `tool_choice: required` forces *a* call but not *well-formedness*. Downstream repair is the only floor here |

The split that matters: every local engine we run can enforce a tool-call
grammar; the hosted provider APIs cannot. So upstream constraint is a capability
that exists for some backends and is structurally impossible for others. That is
the reason downstream repair has to stay on regardless.

## The wiring seam already exists

The serving path already carries the two fields a constraint would ride on, as
verbatim passthroughs:

- `internal/agent/chat.go:286-293` — `ResponseFormat` (the OpenAI
  `response_format` carrier) and a `logit_bias` mask, documented as the `#560`
  guided-decode seam. `WithResponseFormat` (`chat.go:382-390`) sets it per request,
  and it rides through to the backend at `chat.go:638`. `tool_choice` is in the
  passthrough allow-list at `chat.go:553`.
- `internal/agent/anthropic_server.go:80,119` — a tool's `input_schema` is passed
  through verbatim into the backend's tool definition.

So the gateway can already hand a backend a `response_format`/guided-json object.
What is missing is the part that *builds* that object from the inbound tools, and
the policy for when to synthesize it instead of only forwarding whatever the
client sent.

## Schema → constraint compiler (design sketch)

The goal is to compile the inbound `tools[].input_schema` set into one per-turn
constraint that forces a structured call of the form
`{"name": <one of the tool names>, "arguments": <that tool's schema>}`.

A backend-neutral intermediate keeps this from forking per engine:

1. Build a JSON Schema `oneOf` over the tools. Each branch pins `name` to a const
   and sets `arguments` to that tool's `input_schema`. This is the lingua franca
   that ollama `format`, vLLM `guided_json`, and SGLang all accept directly.
2. For GBNF-only paths (raw llama.cpp), compile the same `oneOf` to GBNF. A JSON
   Schema → GBNF step is mechanical and already exists in the llama.cpp tree as a
   reference.
3. Content-address and dedupe the compiled constraint the way
   `grammar.Rung` already deduplicates grammars, so a tool set compiled once is
   reused across the fleet instead of recompiled per turn.

Reusing `grammar.Param` for the argument shape keeps one source of truth: the same
schema feeds both the downstream repair grammar and the upstream constraint.

## tool_choice maps onto how much freedom the decoder keeps

`tool_choice` decides whether the model may decline to call a tool, which decides
whether a hard constraint is even correct:

- `required` / `any` — the model must call a tool. Constrain to the `oneOf` union
  of tool calls. This is the clean case: the output is forced to a valid call.
- `auto` — the model may answer in prose or call a tool. A hard call-constraint
  would be wrong here because it removes the decline option. The useful form is a
  top-level union, `{ assistant_text | tool_call }`: the model freely chooses the
  branch, and once it commits to the tool branch the arguments are forced
  well-formed. Where a backend cannot express that union, leave `auto` turns
  unconstrained and rely on downstream repair.
- `none` — no constraint.

The `auto` case is where over-constraint does damage: forcing a call when the
model should have declined trades a malformed-call error for a wrong-action error,
which is worse. The measurement below should watch for exactly that.

## Measurement plan

Reuse the #53 dialect corpus and run on CPU-resident small models through ollama,
which exercises the real llama.cpp constraint path:

- **Models**: qwen2.5:1.5b and qwen2.5:3b.
- **Arms**: unconstrained (today), constrained-`required`, constrained-`auto`
  (top-level union).
- **Metrics**:
  - malformed-tool-call rate, read from the existing counters: the
    `grammar.Rung` deny count and the number of calls recovered by
    `toolcall_fallback` (a recovered call means the structured channel failed).
  - task success on the corpus tasks.
  - decline accuracy on `auto` turns where the correct answer is prose, to catch
    the over-constraint regression.
- **Expectation**: malformed rate drops toward zero on the constrained arms;
  the signal to watch is whether `auto` task success or decline accuracy regresses.

## Recommendation

Wire upstream constraint as a backend-local capability behind the existing `#560`
`response_format` seam, gated on two conditions: the backend advertises a guided
mechanism (ollama / vLLM / SGLang / llama.cpp), and `tool_choice` is `required` or
`any` (or `auto` where a top-level union is expressible). Synthesize the
`oneOf`-of-tools constraint from the inbound schemas at that point and forward it.

Keep downstream repair on unconditionally. It is the only lever on hosted
Anthropic/OpenAI turns, and it is the safety net for `auto` turns that ship
unconstrained. Constrained decoding removes the malformed-output class where the
engine can enforce it; downstream repair catches whatever a backend cannot
constrain, everywhere.

This stays a backend concern in the sense that fak never simulates a constraint a
backend cannot apply. It becomes a gateway concern only in the one narrow place
that is genuinely shared: compiling the tools into a constraint object, which the
gateway is already positioned to do because the tools and the `response_format`
carrier both pass through it.

## Open questions for the implementation slice

- Confirm the installed ollama version accepts a full JSON-Schema `oneOf` in
  `format`, not just `"json"`. If only `"json"`, the first slice forces valid JSON
  without forcing the call shape, and the shape still leans on the grammar rung.
- Measure per-turn compile cost for vLLM `guided_grammar` (xgrammar/outlines)
  against the dedupe cache hit rate, since a cold compile per distinct tool set
  could dominate a short turn.
- First code slice, smallest useful: a `tools → oneOf JSON Schema` synthesizer
  plus the measurement harness above on qwen2.5:1.5b. That decides, with numbers,
  whether the `auto`-union case earns its complexity or whether `required`/`any`
  is the whole worthwhile surface.

## References

- Issue #469 (this research), #53 (downstream text-tool-call seed), #560
  (guided-decode / `response_format` seam).
- `internal/grammar/grammar.go` — downstream grammar repair rung.
- `internal/agent/toolcall_fallback.go` — six-dialect text→structured lift.
- `internal/agent/chat.go:286-293,382-390,553,638` — `response_format` /
  `logit_bias` / `tool_choice` passthrough.
- `internal/agent/anthropic_server.go:80,119` — `input_schema` passthrough.
