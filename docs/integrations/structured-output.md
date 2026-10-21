---
title: "Structured output through fak: OpenAI SDK, vLLM/SGLang, LangChain, Pydantic AI, Instructor, BAML, LlamaIndex"
description: "An integration cookbook for the popular structured-output surfaces. Point each library's base URL at fak; the library (or the ride engine) makes candidates well-formed, and fak adjudicates whether a well-formed candidate is allowed to have its effect. Names each library's exact integration posture, what fak can honestly prove, and why prompt-only validation is not token-level constrained decoding."
---

# Structured output through fak

Structured generation did not go away — it got absorbed into three louder surfaces:
provider-native structured outputs (OpenAI `response_format` / Responses `text.format`,
strict function calling), serving-engine guided decoding (vLLM/SGLang via XGrammar,
llguidance, Outlines), and app/framework schemas (LangChain, Pydantic AI, Instructor,
BAML, LlamaIndex). fak does not compete with any of them. It sits on the seam they all
share and keeps one narrow claim:

> **The library or engine makes a candidate well-formed. fak adjudicates whether a
> well-formed candidate is allowed to have the requested effect.** A schema-valid tool
> call is still a *proposed* effect — it crosses the capability floor before it runs, and
> policy can deny it.

You adopt this the same way you adopt fak for anything else: start `fak serve` in front of
whatever serves your tokens and repoint one base URL. No library fork, no adapter per
framework — every surface below speaks the OpenAI-compatible wire fak already exposes.
Background and sources: [`docs/notes/STRUCTURED-GENERATION-SOTA-BACKLOG-2026-07-04.md`](../notes/STRUCTURED-GENERATION-SOTA-BACKLOG-2026-07-04.md).

---

## The one honest fence: valid ≠ enforced ≠ allowed

Three different things get called "structured output," and this doc keeps them separate
because fak's claim is only true for one of them:

1. **Token-level constrained decoding** — the engine (vLLM/SGLang/XGrammar/llguidance/the
   fak native path) masks logits at each step so the tokens *cannot* leave the grammar.
   This is enforcement at generation time.
2. **Prompt-only parse / validate / retry** — the model is *asked* for a shape (a JSON
   instruction, a Pydantic model, an Instructor schema), the raw text is parsed, and on a
   validation failure the client retries. Nothing constrained the tokens; the guarantee is
   "it eventually parsed," not "it could not have been malformed."
3. **Effect adjudication** — fak's job. Whether a candidate is well-formed by (1) or (2),
   the tool call it carries is a proposed effect, and fak decides ALLOW / DENY / TRANSFORM
   / QUARANTINE against a reviewable capability floor.

**Prompt-only parsing/validation is NOT the same as token-level constrained decoding.**
Instructor retries, Pydantic AI *Prompted Output*, and LlamaIndex extraction are category
(2): useful, but they do not enforce the grammar during decoding, and fak never reports
them as if they did. fak forwards the real constraint carriers to a category-(1) engine
when the client sends them, and it adjudicates the effect in every case.

A valid JSON object can still be wrong, forbidden, or unsafe. `ExtractBench` found
structured-output mode dropping overall validity from 51% to 37% on hard extraction; a
grammar is not a security policy (`CodeShield`). fak measures structural validity,
semantic accuracy, and policy safety as *separate* axes — see
[#2598](https://github.com/anthony-chaudhary/fak/issues/2598).

---

## Posture table: each library, its integration, what fak proves

| Library / surface | fak integration posture | What fak can honestly prove |
|---|---|---|
| **OpenAI SDK** (Python/JS) | `base_url` → fak; `response_format`, Responses `text.format`, `tools`, `strict`, and `logit_bias` are forwarded to the ride engine **verbatim** | The constraint carriers cross the upstream wire byte-equivalent, and every generated tool call enters adjudication before any survivor is forwarded — test-backed below. fak does **not** perform the decode. |
| **vLLM / SGLang** (ride upstream) | Ride-mode engine behind fak; current structured-output field names forwarded; the **engine** enforces the grammar at decode | fak forwards the constraint and adjudicates the result; token-level enforcement is the engine's. Gateway tax is measured separately, not folded into the engine's numbers. |
| **LangChain** (`ProviderStrategy` / `ToolStrategy`) | Same OpenAI `base_url`. `ProviderStrategy` uses provider-native structured output; `ToolStrategy` lowers the schema to tool calling | Tool calls produced by **either** strategy still adjudicate. fak is agnostic to which strategy the app chose. |
| **Pydantic AI** (Tool / Native / Prompted Output) | `base_url` → fak. **Tool Output** and **Native Output** map to a real provider/engine enforcement path; **Prompted Output** is parse/validate/retry | Only **Tool** and **Native** are enforcement-grade. **Prompted Output is not constrained decoding** and fak does not claim it is. |
| **Instructor** | `base_url` → fak; Instructor's Pydantic validation + automatic retries stay **client-side** | fak gates the calls that cross it. Instructor's retries are **not** constrained decoding — they are re-asks, and fak reports them as such. |
| **BAML** | Typed DSL / codegen producing schemas and a generated client; target fak's OpenAI-compatible endpoint | fak is the **endpoint and gate**; BAML is the schema/trace **producer**. fak does not re-implement BAML's decoder. |
| **LlamaIndex** | `base_url` → fak; extraction/parsing run **in the client** | fak is the **endpoint/gate, not the parser**. The extraction program is LlamaIndex's; the effect boundary is fak's. |
| Outlines / Guidance / llguidance / XGrammar | Prior art and optional engine/library dependencies | Used *by* the ride engine; fak does not fork their internals into its hot path. |

---

## Recipes

Start the gate once, in front of whatever serves your tokens (a cloud API, or a local
vLLM/SGLang/Ollama). Every recipe below then differs only in the client:

```bash
# fronts any OpenAI-compatible upstream; omit --base-url for the offline mock planner
fak serve --addr 127.0.0.1:8080 \
  --provider openai \
  --base-url http://localhost:8000/v1 \
  --model your-model \
  --policy examples/customer-support-readonly-policy.json
```

### Native JSON-schema responses

On a native model route, an explicit `response_format` can constrain each token to
a JSON object. For example:

```json
{
  "type": "json_schema",
  "json_schema": {
    "name": "answer",
    "strict": true,
    "schema": {
      "type": "object",
      "properties": {"answer": {"type": "string", "enum": ["yes", "no"]}},
      "required": ["answer"],
      "additionalProperties": false
    }
  }
}
```

The native subset supports required and optional scalar properties: strings,
booleans, integers and numbers. A response may declare at most 16 properties,
of which at most four may be optional, and its schema may contain at most 64 KiB
(65,536 bytes). Optional properties may be omitted or emitted once; required
properties are always emitted. Emitted keys use canonical lexicographic order.
Strings use printable ASCII without escapes; string `enum` and `const` values
must fit that surface. Native response schemas also accept integer `enum` and
`const` constraints with 1–32 raw members. Members must be signed 64-bit JSON
integers; they are emitted in canonical decimal form, with `-0` normalized to
`0` and duplicates removed. Fractions, exponents, overflow, `null`, mixed-type
integer sets and constrained number values are rejected before inference.
Unconstrained numbers use canonical JSON syntax without exponents. A property may
also be an array whose `items` is one of those scalar schemas or a fixed-shape
object, with optional `minItems` and `maxItems` between 0 and 64; arrays are
emitted as `[a,b]` with no whitespace. Every member of an object item, at every
level inside it, must be required, and an array inside an item may hold only
scalars, so every item carries the same keys in canonical order, for example
`{"rows":[{"kind":"a","ok":true},{"kind":"b","ok":false}]}`. Optional item
members, arrays of objects inside an item, arrays of arrays, tuple `items`,
`uniqueItems` and other array keywords are rejected, and tool schemas remain
array-free. A property may also be a nested object with the same property rules,
up to three object levels including the top; an object item sits one level below
the object that holds its array. The 16-property and four-optional limits count
every property at every level, including item members; an omitted optional
object omits its members.
Unknown constraint keywords, `additionalProperties:true`, union keywords, regex
and EBNF are also rejected before inference. `strict` may be true
or omitted. Native `json_object` is unsupported; upstream engine passthrough
retains its own supported surface.

The mask composes with temperature, top-k, top-p, logit bias and repetition
penalties. Buffered and per-token streaming preserve the generated JSON bytes,
including strings resembling reasoning delimiters or tool calls. EOS is admitted
only after the object is complete. A token limit, cancellation or explicit stop
can still interrupt a response; callers must check the finish reason before
treating a partial response as a complete object.

Reasoning may precede the object. When the prompt opens a reasoning span (a
thinking-mode Qwen3.5-family template) or the request sets a reasoning effort or
thinking budget, decoding is unconstrained inside the span and the schema mask
starts only after the model emits the atomic `</think>` token. Text that merely
spells `</think>` with ordinary tokens does not end the span. Without a seeded
span, the first token may be the atomic `<think>` token or the start of the
object. EOS stays unavailable until the object is complete, so a response cannot
end inside reasoning. The thinking budget forces `</think>` when it is spent;
once the object starts, the budget no longer observes tokens, so think tags
inside JSON strings are ordinary content. The reasoning before `</think>` is
returned as reasoning content, never as the response; per-token streams carry
only the object bytes. Both delimiters must be single in-vocabulary non-stop
tokens; otherwise the request is refused. A span still open at the token limit
returns empty content with finish reason `length`. This is adapted from
DeepInfer's structured-output trigger/bound matcher (Apache-2.0); no upstream
source is copied.

Structured requests currently use target-only decoding. The bounded optional
property support applies only to native JSON-schema responses; tool-schema
compilation remains required-only and accepts only its existing string enum
surface. Requests combining this mode with tools, DFlash2 or an unmodified-logit
inference receipt are refused.
Plain text and omitted formats retain the ordinary route. Unsupported native
combinations return HTTP 400 with code
`native_structured_output_unsupported`; inability to select a valid token returns
HTTP 500 with code `native_structured_decode_failed`.

### OpenAI SDK — preserve `response_format`, `text.format`, `strict`, `logit_bias`

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local")

resp = client.chat.completions.create(
    model="your-model",
    messages=[{"role": "user", "content": "extract the order id"}],
    # Structured-output constraint carriers pass through fak to the ride engine verbatim:
    response_format={
        "type": "json_schema",
        "json_schema": {"name": "order", "strict": True,
                        "schema": {"type": "object",
                                   "properties": {"order_id": {"type": "string"}},
                                   "required": ["order_id"]}},
    },
    logit_bias={"50256": -100},
    tools=[...],           # tool schemas also forwarded; the calls they produce adjudicate
)
```

On the **Responses** API the carrier is `text.format` instead of `response_format`; fak
forwards it the same way. Any tool call the constrained generation emits still enters the
capability floor before it is returned to your code.

### vLLM / SGLang — ride mode, current field names, one deprecation note

Front the engine and let it do the token-level enforcement:

```bash
fak serve --provider openai --base-url http://127.0.0.1:8000/v1 --model your-model
```

Use the **current** structured-output field names: `response_format` (OpenAI-compatible)
or the engine's `structured_outputs` object. fak is a transparent proxy — it forwards
whatever constraint field the client sends and does not rewrite it — but recipes should
prefer the current names.

> **Deprecated-field warning.** vLLM **removed** the older top-level guided fields
> (`guided_json`, `guided_grammar`, `guided_choice`, `guided_regex`) in **v0.12.0** in
> favor of `structured_outputs` / `response_format`. A client still sending the old
> fields will have them forwarded unchanged by fak, but a current vLLM will ignore them.
> Prefer `response_format` / `structured_outputs`. SGLang accepts JSON Schema, regex, and
> EBNF with XGrammar as the default backend.

The engine enforces the grammar at decode; fak forwards the constraint and adjudicates the
resulting tool calls. Measure the gateway tax as its own number — do not fold it into the
engine's tokens/sec.

### LangChain — `ProviderStrategy` and `ToolStrategy`

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local", model="your-model")

# ProviderStrategy: provider-native structured output; ToolStrategy: schema -> tool calls.
structured = llm.with_structured_output(YourPydanticModel)   # either strategy, same base URL
```

Whichever strategy LangChain picks, the tool calls it produces cross fak's floor. fak does
not need to know which strategy you chose.

### Pydantic AI — mark which modes are enforcement-grade

```python
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel

model = OpenAIModel("your-model", base_url="http://127.0.0.1:8080/v1", api_key="fak-local")
agent = Agent(model, output_type=YourModel)   # Tool Output / Native Output are enforcement-grade
```

- **Tool Output** and **Native Output** map to a real provider/engine enforcement path —
  enforcement-grade.
- **Prompted Output** injects the schema into the prompt and parses the reply. It is
  parse/validate/retry, **not** constrained decoding. Use it when the model/engine offers
  no native mode, and do not treat its output as grammar-enforced.

### Instructor — validation/retry through fak (not constrained decoding)

```python
import instructor
from openai import OpenAI

client = instructor.from_openai(OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="fak-local"))
user = client.chat.completions.create(
    model="your-model", response_model=YourModel,   # Pydantic validation + retries are client-side
    messages=[{"role": "user", "content": "..."}],
)
```

Instructor's value is provider-portable Pydantic validation, automatic retries, and
streaming. Those retries are **re-asks on a validation failure, not token-level constrained
decoding** — fak gates each attempt that crosses it and never reports an Instructor retry
as a constrained decode.

### BAML — typed schema/codegen producer targeting fak's endpoint

Point BAML's generated OpenAI client at fak's `base_url`. BAML is the typed producer of
schemas and traces; fak is the OpenAI-compatible endpoint and the effect gate. fak does
not re-implement BAML's decoder — the two compose along the wire.

### LlamaIndex — extraction where fak is the endpoint/gate, not the parser

```python
from llama_index.llms.openai import OpenAI as LlamaOpenAI

llm = LlamaOpenAI(model="your-model", api_base="http://127.0.0.1:8080/v1", api_key="fak-local")
# LlamaIndex runs the extraction program; fak is the endpoint and the tool-call gate.
```

The extraction/parsing logic stays in LlamaIndex. fak is the endpoint and the boundary
that adjudicates any tool call the flow proposes — it is not the parser.

---

## Runnable / test-backed examples

The issue's acceptance asks for one OpenAI-SDK/LangChain route and one
Pydantic/Instructor/BAML typed-output route, plus a tool-call example proving a schema-valid
candidate still enters adjudication and can be denied. All three are witnessed here.

### Example 1 (OpenAI SDK / ride route) — test-backed, GPU-free

`internal/gateway/structured_output_passthrough_test.go` is the witness for the OpenAI-SDK
route. It sends a request carrying a `json_schema` `response_format` (with `strict:true`), a
`logit_bias` mask, and `guided_grammar`/`guided_choice`, and asserts against the bytes that
actually crossed the upstream wire:

- **`TestChatProxyForwardsStructuredOutputFieldsToRideEngine`** — the constraint carriers
  reach the ride engine **verbatim**, and the two tool calls the (mock) constrained
  generation emits **both enter fak adjudication**; the denied one (`deny_write`) is dropped
  and only the allowed one (`allow_read`) survives.
- **`TestChatProxyOmitsStructuredOutputFieldsWhenAbsent`** — a client that sends no
  structured-output fields produces an upstream body with **neither** key, so a
  non-structured client is never silently constrained.

Run it (no model, no key, no GPU):

```bash
go test ./internal/gateway/ -run 'TestChatProxyForwardsStructuredOutputFieldsToRideEngine|TestChatProxyOmitsStructuredOutputFieldsWhenAbsent' -count=1
# ok  github.com/anthony-chaudhary/fak/internal/gateway
```

### Example 2 (Instructor / Pydantic typed-output route) — same wire, same witness

Instructor, Pydantic AI (Tool/Native Output), and BAML all speak the **same** OpenAI Chat
Completions wire as Example 1 — they set `base_url` to fak and send `tools` /
`response_format`. The passthrough test above therefore backs this route too: the typed
client's schema and tool calls cross fak byte-equivalent and adjudicate. The Instructor and
Pydantic AI snippets in the recipes section are the runnable client halves; point them at a
live `fak serve` and the same forwarding + adjudication path executes.

### Example 3 (tool-call: schema-valid, still adjudicated, denied by policy) — runnable

A well-formed tool call is still a *proposed* effect. This is the load-bearing example for
the issue's fourth acceptance point, and it runs offline with no model:

```bash
# A schema-valid refund_payment candidate (well-formed args) under a read-only policy:
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool refund_payment --args '{"amount":500,"currency":"USD"}'
# fak: loaded capability floor from examples/customer-support-readonly-policy.json
# verdict=DENY reason=POLICY_BLOCK by=monitor

# The same policy ALLOWS a read-only tool with equally valid args:
go run ./cmd/fak preflight --policy examples/customer-support-readonly-policy.json \
  --tool search_kb --args '{"q":"refund window"}'
# verdict=ALLOW reason=NONE by=monitor
```

The `refund_payment` arguments are perfectly schema-valid — structured generation would
happily produce them — and fak still **denies the effect** by structure, with a named
reason (`POLICY_BLOCK`), not a model judgment call. That is the whole point: *well-formed is
not the same as allowed.* The in-gateway version of this is asserted in
`TestChatProxyForwardsStructuredOutputFieldsToRideEngine`, where the constrained generation
proposes both `allow_read` and `deny_write` and the gate drops `deny_write` after decoding.

---

## Non-goals

- **Not** forking or embedding the app libraries (OpenAI SDK, LangChain, Pydantic AI,
  Instructor, BAML, LlamaIndex) into fak.
- **Not** replacing library retry/validation behavior — Instructor keeps retrying, Pydantic
  keeps validating; fak adjudicates effects, it does not take over the client's loop.
- **Not** re-implementing XGrammar/vLLM/SGLang guided decoding in fak's hot path. The
  minimal native constraint spine is scoped separately in
  [#2596](https://github.com/anthony-chaudhary/fak/issues/2596).

---

## Cross-references

- [Structured generation SOTA, fak fit, and backlog split](../notes/STRUCTURED-GENERATION-SOTA-BACKLOG-2026-07-04.md) — the research note behind this cookbook, with sources and the popularity signals.
- [Run your agent through fak](README.md) — the integration index; the universal repoint recipe and the 60-second offline proof of the gate.
- [Agent-framework integration](../fak/agent-framework-integration.md) — the per-framework cookbook for LangChain, LlamaIndex, AutoGen, CrewAI (proxy or explicit adjudication).
- [fak + LiteLLM](litellm.md) and [Routers & gateways](routers.md) — front a proxy/router that already fans out to many backends.
- [Interoperability stance](interoperability.md) — the honest per-wire grade for each surface.
- [Debugging a verdict](debugging.md) — why was my schema-valid call denied? Reproduce it offline with `fak preflight --explain`.
- [CLAIMS.md](https://github.com/anthony-chaudhary/fak/blob/main/CLAIMS.md) — the claim-by-claim scope, including the shipped structured-output passthrough.
