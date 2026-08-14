---
title: "Built-in compaction audit"
description: "Why built-in compaction in agent products is useful as an overflow valve but weak as durable memory, especially for coding, research, and search workflows."
date: 2026-07-06
status: research-note
---

# Built-in compaction audit

## Bottom line

Built-in compaction is usually good at one narrow job: preventing a long
agent thread from hitting the context window. It is not very good as a
continuity mechanism. In Codex, Claude Code/API, OpenCode, and similar
agents, compaction is mostly a model-authored summary of prior context,
followed by dropping or hiding the raw history. That makes it lossy,
self-certified, hard to inspect, and poorly aligned with the facts a later
turn may need.

The right framing is:

- compaction is an emergency overflow valve;
- it is not a memory system;
- it is not a provenance system;
- it is not a proof that important state survived.

For coding and research agents, the weak point is not only "the summary may
forget details." The deeper problem is that the compactor has to predict
future salience from a noisy, already-long context, without durable pointers,
retention contracts, source provenance, or an external witness.

## What current products expose

### Codex

OpenAI's Codex docs say all thread information must fit in the model context
window, and that for longer tasks Codex may automatically compact context by
"summarizing relevant information and discarding less relevant details." The
CLI also exposes `/compact`, whose documented effect is to replace earlier
turns with a concise summary. Codex has configuration and hooks around this:
`model_auto_compact_token_limit`, `compact_prompt`,
`experimental_compact_prompt_file`, and `PreCompact` / `PostCompact` hooks.

Sources:

- [Codex prompting](https://developers.openai.com/codex/prompting)
- [Codex `/compact`](https://developers.openai.com/codex/cli/slash-commands)
- [Codex config reference](https://developers.openai.com/codex/config-reference)
- [Codex sample config](https://developers.openai.com/codex/config-sample)
- [Codex hooks](https://developers.openai.com/codex/hooks)

Read: Codex gives useful knobs and lifecycle hooks, but the core mechanism is
still summary plus discard. There is no public contract that says which facts
must survive, no source-level witness for what was dropped, and no guarantee
that a later fact can be demand-paged from the original transcript.

### Claude Code and Claude API

Claude Code enables auto-compaction by default (`autoCompactEnabled: true`) and
documents it as summarizing history when the thread approaches context limits.
The Claude API now has beta server-side compaction: when a configured input
token threshold is reached, Claude generates a `compaction` block, and
subsequent requests drop all content blocks prior to that block. The API also
allows custom compaction instructions and `pause_after_compaction`, which is
useful because the default summary may need application-specific correction.

Claude's own docs also acknowledge the operational failure mode: auto-compact
can "thrash" when large files or tool output immediately refill the context,
and the recommended recovery is to read smaller chunks, run `/compact` with a
focus, move work to a subagent, or clear the conversation.

Sources:

- [Claude Code costs](https://code.claude.com/docs/en/costs)
- [Claude Code settings](https://code.claude.com/docs/en/settings)
- [Claude Code troubleshooting](https://code.claude.com/docs/en/troubleshooting)
- [Claude API compaction](https://platform.claude.com/docs/en/build-with-claude/compaction)
- [Claude context windows](https://platform.claude.com/docs/en/build-with-claude/context-windows)

Read: Anthropic's API design is more explicit than most. The `compaction`
block is a real structure, and `pause_after_compaction` is a good hook for
application control. But it still replaces raw earlier context with a summary.
The summary can be customized, but customization is not the same as a
machine-checkable retention invariant.

### OpenCode

OpenCode documents a `compaction` config object with `auto`, `prune`, and
`reserved`. It also has a hidden `compaction` system agent that compacts long
context into a smaller summary, and plugin hooks that can inject context into
the compaction prompt or replace that prompt entirely.

OpenCode's public issue tracker shows the practical edge cases:

- `/compact` can fail before summarization if the compaction request itself is
  larger than the model's context window.
- a compaction replay bug injected a synthetic user message ("What did we do so
  far?"), causing unwanted recap behavior and confusing role semantics.
- users have asked for more control over compaction thresholds because late
  compaction can happen after long-context quality has already degraded.

Sources:

- [OpenCode config](https://opencode.ai/docs/config/)
- [OpenCode agents](https://opencode.ai/docs/agents/)
- [OpenCode plugins](https://opencode.ai/docs/plugins/)
- [OpenCode issue #29857](https://github.com/anomalyco/opencode/issues/29857)
- [OpenCode issue #13838](https://github.com/anomalyco/opencode/issues/13838)
- [OpenCode issue #11314](https://github.com/anomalyco/opencode/issues/11314)

Read: OpenCode is unusually extensible, but the need for compaction plugins is
itself evidence that generic compaction cannot reliably know which state must
survive. Hooks help operators patch the summary prompt; they do not make the
result lossless.

## Why built-in compaction is weak

### 1. Summaries are not reversible

Compaction is a rewrite. Once raw turns are replaced by a summary, the next
agent turn cannot tell whether a missing constraint was genuinely irrelevant,
accidentally omitted, softened, merged into another statement, or contradicted
by later evidence.

This is especially bad for coding:

- exact file paths and line references matter;
- a rejected approach can be as important as the chosen one;
- "do not touch X" constraints are small and easy to omit;
- tool failures, flaky-test details, and environment caveats often matter
  later;
- a summary can preserve intent while losing the evidence needed to act safely.

It is also bad for research and search:

- source URLs, publication dates, and quote boundaries matter;
- why a source was rejected matters;
- conflicts between sources should remain visible, not blended;
- "I searched X and found nothing" is negative evidence, but summaries tend to
  drop it;
- final citations become weaker when the raw source trail is compacted into an
  unsourced recap.

### 2. The compactor has to predict future salience

The relevant fact later is often not the fact that looked important at the time
of compaction. A small version number, a command flag, a branch name, an error
substring, or a rejected source can become decisive ten turns later.

This is the same structural problem as "lost in the middle": long-context
models do not robustly use information just because it is present. The TACL
paper "Lost in the Middle" found that models often perform best when relevant
information is near the beginning or end of context and worse when it sits in
the middle. Chroma's context-rot report found performance degradation as input
length grows. A later arXiv paper argues that sheer input length can hurt
performance even with perfect retrieval.

Sources:

- [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- [Context Rot](https://www.trychroma.com/research/context-rot)
- [Context Length Alone Hurts LLM Performance Despite Perfect Retrieval](https://arxiv.org/html/2510.05381v1)

Compaction helps with the length side of this problem, but it does so by asking
another model pass to guess what should survive. That replaces one unreliable
attention problem with a lossy selection problem.

### 3. It often fires too late

Default compaction usually triggers near a context limit or configured token
threshold. By then, the session may already be suffering from context rot,
slow responses, stale plans, accumulated tool noise, and poor retrieval inside
the prompt.

Late compaction has two bad outcomes:

- the compaction input is already polluted, so the summary can preserve noise;
- the request can be too large or too close to the limit to summarize safely.

The Claude Code troubleshooting page explicitly describes thrashing after
successful auto-compaction when large content immediately refills the context.
OpenCode issue #29857 describes a large restored session where `/compact` could
not even start because the summarization request exceeded the model window.

### 4. It collapses provenance

For research, provenance is not decoration. The agent needs to preserve:

- source identity;
- retrieval query;
- timestamp or freshness;
- whether the source was primary or secondary;
- exact fact extracted;
- uncertainty and conflicts;
- what was not found.

A generic compaction summary tends to flatten all of that into "what we know."
That is convenient for chat continuity and bad for auditability. It makes the
next agent more likely to cite memory of a source rather than the source
itself.

Long-horizon search work shows the same issue. In "Lost in the Maze," existing
agentic search systems fail from poor context management: they either overflow
context, exhaust budgets, stop early, or flood the model with noisy search
content. The proposed fix is not "summarize everything harder"; it is a more
structured search/browse/summarize design that controls what enters context.

Source:

- [Lost in the Maze: Overcoming Context Limitations in Long-Horizon Agentic Search](https://arxiv.org/html/2510.18939v1)

### 5. It can mutate roles and framing

Agent transcripts are structured: user request, assistant plan, tool call, tool
result, hook output, system/developer instruction, and external source are not
interchangeable. If compaction replays state as a generic user message or a
plain assistant recap, it changes how the next model interprets authority and
intent.

The OpenCode synthetic-user-message issue is a concrete example. The model was
reportedly shown a fake user prompt asking what had been done, causing it to
answer a question the user did not ask and possibly reason as if that question
was user intent.

### 6. It compounds over repeated compactions

Repeated compaction creates a summary-of-summary chain. Each pass can:

- normalize uncertainty into certainty;
- drop old caveats;
- merge separate facts;
- forget who said what;
- preserve stale plans because they are prominent;
- erase why a current plan exists.

The failure is hard to see because every summary can look coherent. Coherence
is not fidelity.

### 7. It can fight prompt caching and cost controls

Prompt caching rewards stable prompt prefixes. OpenAI documents that cache hits
require exact prefix matches, with static content placed at the beginning.
Anthropic documents a prefix hierarchy where changes at an earlier level
invalidate that level and everything after it.

A compaction summary rewrites earlier conversation bytes. Depending on where it
lands and how the client uses cache breakpoints, this can reduce the chance of
cache reuse. A smaller prompt is not automatically cheaper if it destroys a
large warm cached prefix. This is the economic reason fak's
[long-session economics](../explainers/long-session-economics.md) argues for
preserving byte-identical prefixes where possible instead of blindly rewriting
history.

Sources:

- [OpenAI prompt caching](https://developers.openai.com/api/docs/guides/prompt-caching)
- [Claude tool use with prompt caching](https://platform.claude.com/docs/en/agents-and-tools/tool-use/tool-use-with-prompt-caching)
- [Long-session economics](../explainers/long-session-economics.md)

## What "good" would look like

Better context management should have stronger invariants than "the model wrote
a reasonable recap." The useful target is not one bigger summary. It is a
layered memory/control system:

1. Keep raw evidence addressable. Summaries can point to source spans, files,
   tool outputs, and retrieval results instead of replacing them forever.
2. Preserve typed structure. User constraints, system instructions, tool
   results, citations, decisions, and open TODOs should survive as different
   object types.
3. Add retention contracts. Let the system mark facts as must-keep, may-drop,
   stale, superseded, or demand-pageable.
4. Verify compaction. After compaction, run probes for preserved constraints,
   active files, open risks, source citations, and negative findings.
5. Compact earlier and selectively. Avoid waiting until the window is nearly
   full; keep noisy tool output out of the main context from the start.
6. Separate research memory from prose. Store source cards and evidence tables,
   then generate prose from them, not the other way around.
7. Keep cache economics explicit. Track whether compaction saved tokens,
   preserved cached prefixes, or merely shifted cost into cache misses.

## Practical audit checklist

When evaluating a product's built-in compaction, ask:

- What exact trigger fires compaction?
- Can the operator lower the trigger before quality degrades?
- Does compaction preserve raw source/tool spans by pointer?
- Is the summary typed, or just prose?
- Can a user pin facts that must survive?
- Can the system prove a pin survived?
- Does it preserve citations, rejected sources, and negative searches?
- Does it preserve role boundaries and authority levels?
- Can it fail closed if the compaction request is itself too large?
- Does it expose pre/post compaction hooks?
- Does it measure cache hit impact after compaction?
- Does it test repeated compaction, not just one compaction?

## Conclusion

Built-in compaction is necessary because long agent sessions otherwise overflow
or degrade. But the default version in agent products is mostly a lossy
summarizer wrapped in product UX. That is acceptable for casual continuity and
dangerous for long-horizon coding, research, and search unless paired with
addressable evidence, typed state, retention controls, and post-compaction
verification.

For fak, the useful contrast is the one already developing in `ctxplan`: do not
make the model remember everything through prose. Keep a bounded working set,
pin what must stay exact, make old state recoverable by reference, and audit
what was actually retained.

## See also

For how fak manage *aligns with* Codex's built-in compaction at runtime — the
two-wire model (fak's lossless cache-preserving cut on the Anthropic wire vs
delegating to Codex's native `model_auto_compact_token_limit=96000` on the
Responses wire), why guarded sessions "compact when the window looks light,"
and a fleet audit of the local rollout store — see
[CODEX-TURN-COMPACTION-ALIGNMENT-2026-07-15](./CODEX-TURN-COMPACTION-ALIGNMENT-2026-07-15.md).
