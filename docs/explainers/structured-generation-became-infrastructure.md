---
title: "Structured generation didn't die — it became infrastructure"
description: "Structured generation looks like a quiet topic, but it didn't disappear: it was absorbed into structured outputs, tool calling, MCP, provider-native schema modes, and engine guided decoding. Where the mechanism went, why it feels quieter, and why fak's job is not producing valid JSON but adjudicating whether a valid candidate is allowed to act."
slug: structured-generation-became-infrastructure
keywords:
  - structured generation
  - structured outputs
  - constrained decoding
  - guided decoding
  - tool calling
  - JSON schema
  - grammar constraints
  - response_format
  - fak adjudication
  - effect boundary
date: 2026-07-17
---

# Structured generation didn't die — it became infrastructure

*Who this is for: anyone who noticed "structured generation" fall out of the
conversation and wondered whether it was a dead direction — and anyone deciding
where fak fits next to a structured-output library or a guided-decoding engine.*

## The short answer

Structured generation did not fall out of use. It got absorbed into three louder
surfaces and stopped being advertised under its old name:

- **Provider-native structured outputs** — OpenAI `response_format` / Responses
  `text.format`, strict function calling, and the native output modes of Anthropic,
  Gemini, and xAI.
- **Serving-engine guided decoding** — vLLM / SGLang / TensorRT-LLM applying
  token-mask engines like XGrammar, llguidance, or Outlines.
- **Agent / tool ecosystems** — LangChain, Pydantic AI, Instructor, BAML, LlamaIndex,
  and MCP, which use schemas as product plumbing rather than as a research headline.

The phrase went quiet because the mechanism became table stakes — not because anyone
stopped needing it.

## Why it feels quieter

1. **The syntax part became table stakes.** Engines and APIs now advertise
   "Structured Outputs" or "tool calling," not "structured generation." Producing
   schema-valid tokens is a solved, commoditized layer.
2. **The interesting failures moved up-stack.** The hard problems are now semantic
   correctness, large/nested schemas, reasoning-channel coexistence, streaming,
   speculative decoding, and security under attacker-chosen grammars.
3. **Frameworks hid it behind types.** A LangChain / Pydantic / Instructor / BAML
   user usually sees a Python type, a Pydantic model, a Zod schema, or a BAML
   function — not a grammar engine.
4. **Agent discussion displaced decoder discussion.** MCP, tool registries,
   reasoning models, and agent harnesses are where schema enforcement now shows up.
5. **"100% valid JSON" stopped being the buying criterion.** Current benchmarks show
   that valid output can still be inaccurate, rejected by a provider's schema subset,
   slow, or unsafe.

### Directional popularity signals

Repo-star and issue-search snapshots (collected 2026-07-04) are **directional only** —
they do not measure market share, and are noisy — but they show the phrase shift
clearly. Structured output lives inside agent frameworks and serving engines now:
`langchain/langchain` (~141k stars), `vllm-project/vllm` (~85k), `run-llama/llama_index`
(~51k), `sgl-project/sglang` (~30k), and the dedicated structured-generation lineage
(`guidance` ~22k, `outlines` ~14k, `instructor` ~13k, `baml` ~8.5k, `xgrammar` ~1.8k,
`llguidance` ~0.8k). Over 2024→2026, GitHub issue mentions of `"structured output"`,
`"tool calling"`, and `"agent" "MCP"` grew far faster than `"structured generation"` or
`"constrained decoding"` — the topic didn't shrink, it was renamed by its consumers.

## Three layers, three jobs

The confusion comes from collapsing three different jobs into one word. They are
complementary, not competing — and fak is only the third:

| Layer | Example | Guarantees | Does **not** guarantee |
|---|---|---|---|
| **Structured-output library** | Instructor, Pydantic AI, BAML, LangChain strategies | A typed object you can program against; parse / validate / retry; provider portability | Token-level enforcement (some modes are prompt-then-validate); that the value is *correct*; that the action is *allowed* |
| **Guided-decoding engine** | XGrammar(-2), llguidance, Outlines in vLLM / SGLang / TensorRT-LLM | Every emitted token stays inside the grammar → **syntactically valid** JSON/CFG, fast, at scale | Semantic accuracy; provider-subset acceptance; and — critically — that a well-formed call is *safe to execute* |
| **fak gate** | `fak serve` at the effect boundary | Adjudicates whether a schema-valid candidate is **allowed to have the requested effect**: default-deny floor, provenance/IFC taint, deny-as-value verdicts, audit | The structure itself — fak rides the library/engine that produces it, it does not re-implement them |

A grammar is not a security policy. A schema constrains *shape*; it says nothing about
whether `transfer_funds` should fire. That gap is fak's entire job.

### What the benchmarks warn

- **JSONSchemaBench** turns "supports JSON Schema" into a measured claim (10K real
  schemas across frameworks) — structural coverage is now benchmarked, not asserted.
- **ExtractBench** is the caution against equating valid with useful: in its 2026
  study, structured-output mode dropped overall validity from 51% to 37%, and one
  model's credit-agreement pass rate from 86.9% to 70.0% — schema restrictions and
  long-document grounding interact badly.
- **CodeSpear / CodeShield** is the security caution: grammar-constrained decoding can
  itself become an attack surface. Valid ≠ safe.

## Where fak fits

fak's differentiator is **not** generating valid JSON. It is adjudicating whether a
valid candidate is allowed to act. Concretely, that means:

- **Ride mode (today).** Point an OpenAI-compatible client at `fak serve`; fak
  preserves `response_format` / `text.format`, `tools`, `strict`, and `logit_bias` to
  the vLLM / SGLang / provider upstream, and adjudicates the tool calls that come back
  exactly as it always does. The structure is produced by the best engine available;
  fak governs the effect. (See the [OpenAI-SDK set-base-URL recipe](../../examples/openai-sdk-minimal/README.md).)
- **A minimal native spine (deliberately small).** fak owns only the smallest compiler
  that turns its tool schemas into a `model.LogitMask` for the in-kernel engine —
  enough to prove the seam and keep the kernel honest, explicitly **not** an attempt to
  out-engineer XGrammar-2. Production structure should ride the mature engines.
- **Evidence discipline.** fak measures structural validity, semantic/value accuracy,
  policy safety, and latency *separately*, because a valid JSON object can still be
  wrong, rejected, slow, or forbidden.

## The recommendation

Use the existing libraries and engines for **structure** — they are excellent,
commoditized, and improving fast (XGrammar-2 claims 100% schema accuracy in its
tool-calling evals). Put **fak at the effect boundary**, where the question is not
"is this valid JSON?" but "is this valid call allowed to act, given who asked and what
it would touch?" Structure and adjudication are different jobs; fak does not replace
vLLM, SGLang, Outlines, or Instructor, and does not ask you to give them up.

---

*Sourced from the repo research note
[`docs/notes/STRUCTURED-GENERATION-SOTA-BACKLOG-2026-07-04.md`](../notes/STRUCTURED-GENERATION-SOTA-BACKLOG-2026-07-04.md),
which carries the full citation list (XGrammar-2, vLLM/SGLang docs, llguidance,
Outlines, OpenAI/LangChain/Pydantic AI/Instructor/BAML docs, JSONSchemaBench,
ExtractBench, CodeSpear/CodeShield) and the native-spine backlog split. Popularity
figures are 2026-07-04 snapshots and are directional, not market-share claims.*
