---
title: "Compaction vs. the industry: why fak drops-and-splices instead of summarizing"
description: "How fak's cache-prefix-preserving history compaction differs from the built-in compaction in Claude Code, the Anthropic API context-editing feature, Cursor, OpenAI Codex, GitHub Copilot, Aider, and LangChain — what is genuinely different, what is an unverified hypothesis, and where 'fak is better' is a category error."
date: 2026-06-25
---

# Compaction vs. the industry

*An audit of how fak's context-history compaction differs from the built-in
compaction shipping in coding agents and LLM frameworks circa 2025–2026. Every
fak claim below was verified against the source by an adversarial reader; every
industry claim is web-sourced; the central technical claim was deliberately
attacked by a skeptic and survives only with a fence. The fences are the
finding, not a footnote.*

## TL;DR

Almost everyone else compacts a long conversation by **summarizing or clearing
old turns and re-sending a rewritten prompt**. By the providers' own
documentation, a rewrite **breaks the prompt cache** — OpenAI's docs say it
outright: *"when you drop, summarize or compact earlier turns in a conversation,
you'll break the cache."* fak takes the one route that keeps the prefix
byte-stable: it **drops whole old middle turns and splices the original bytes
back together**, then *proves* the cached prefix is byte-for-byte unchanged
before shipping.

What fak **proves**: the bytes it ships are byte-identical to what you sent, so a
cache hit is *possible*. What fak does **not** prove, and cannot from its own
wire: that Anthropic's cache actually *reuses* that prefix after the middle is
dropped. That second step — the "cache cascade" — is sound engineering reasoning
about Anthropic's documented cache behavior, **not a measured fact**, and it is
the hinge the entire cost story turns on (see §"The one claim that could be
wrong"). This note is careful to keep those two apart, because the industry
comparison is only honest if fak's own claim is.

## The crux: a rewrite busts the cache; a byte-splice keeps it eligible

Provider prompt caches (Anthropic, OpenAI) only discount a prefix that is
**byte-for-byte identical** to what they saw on the previous turn. Any
compaction that **summarizes** ("replace the old turns with a 2-paragraph
summary") rewrites the prompt body → new bytes → cache miss → the provider
re-charges the whole prefix at full price on the turn it compacts. The summary
pays off on *later* turns, but you eat a full re-prefill every time you compact.

fak's `CompactAnthropicHistory` (`internal/agent/anthropic_compact.go`) refuses
to rewrite. On the real-Anthropic passthrough it:

1. Anchors a **protected prefix** at the *first* `cache_control` breakpoint — the
   stable cached head the provider reuses every turn.
2. Drops the **middle** turns (the un-cacheable span the provider re-bills
   anyway), replacing them with one short stub message.
3. Splices the result by **copying the original bytes** of the prefix and the
   recent kept window verbatim (`b.Write(raw[:prefixEnd])`), never
   re-serializing — so a content array is never split and no partially-cached
   message is ever re-marshalled.
4. **Proves it** before shipping: re-decodes the spliced body, and asserts
   `bytes.Equal(raw[:prefixEnd], out[:prefixEnd])`. If either check fails it
   returns the input unchanged.

A 900-turn dogfood (`experiments/agent-live/compact-100k-session-dogfood-2026-06-25.json`)
shows the wire effect: a 142,516-token inbound body forwards as 6,597 tokens
(95.4% shed), the spliced body re-decodes valid, and the 10,660-byte protected
prefix is SHA256-identical with compaction off vs on. **Read that artifact
precisely: it runs against a `<mock-upstream>`.** It proves fak ships a
byte-identical prefix and a smaller body; it does *not* prove the real provider
reused the prefix. That is the right thing for it to prove — see below.

## Where each approach lands

| Tool / feature | Mechanism | Busts prefix cache? | Information loss |
|---|---|---|---|
| **fak** `--compact-history-budget` | Drop middle turns, **byte-splice** original prefix | **No** to the bytes; provider *reuse* unmeasured | Turns dropped (gone); byte- & adjudication-faithful |
| **Anthropic API context-editing** (`clear_tool_uses_20250919`) | Clears old **tool results** server-side, placeholder stub; paired with memory tool | **Yes** — docs: *"Invalidates cached prompt prefixes when content is cleared"* | Lossy but **recoverable** via memory tool |
| **Aider** | Recursive LLM summarization (`ChatSummary`) | **Yes** | Lossy (summary retains gist) |
| **LangChain** `ConversationSummaryMemory` | Summarize-and-resend | **Yes** | Lossy |
| **LangChain** `trim_messages()` | Sliding-window **drop** (no rewrite) | **No** — closest analog to fak | Drops turns |
| **OpenAI Codex CLI** | LLM summary replaces history | **Yes** — one cold miss, then re-caches under `prompt_cache_key` | Lossy |
| **GitHub Copilot CLI** | Summarize, then **deliberately** reset prefix at a cache boundary | **Yes, by design** (a "natural cache boundary" for model routing) | Lossy |
| **Cursor / Windsurf** | Lossy flash-model summarization + offload-to-file fallback | Likely (message restructuring); undocumented | Lossy, file-recoverable |

The honest read: fak is **not** the only design that *can* preserve cache —
LangChain's `trim_messages` and Copilot's deliberate-boundary strategy both
reason about it. fak is the only one in the set whose compaction is a **proven
byte-identity splice rather than a summary**, ships it **on by default in the
proxy path** (not a library primitive you wire yourself or an opt-in beta), and
**reports the provenance** of the saving honestly (below).

## Three more ways the design is genuinely different (all code-verified)

**1. Faithful to the security boundary, not just the bytes.** Compaction runs
only on `req.Raw` (the outbound wire body). The decoded `req.Messages` the kernel
adjudicates — `admitInboundResults`, `adjudicateProposed` — still see the **full**
history (`internal/gateway/messages.go:172-180, 466, 496`). Shedding old turns to
save tokens never weakens taint-tracking or tool-call adjudication. The
summarizers compact the *same* transcript the model reasons over; fak compacts
only the copy that goes on the wire.

**2. It separates what it controls from what it observes.** fak reports the
tokens it **shed** (witnessed — fak authored it) next to the provider's
**`cache_read`** (observed — relayed verbatim, *"attribute nothing to fak from it
alone"*) as distinct `/metrics` counters. It guarantees *"the prefix I ship is
byte-identical"* and explicitly does **not** claim *"therefore you got a cache
hit."* No summarizer-based tool makes that distinction; they conflate "I
compacted" with "I saved money." (This is also the discipline that keeps the
cascade claim honest — see below.)

**3. Fail-safe by construction.** On *any* ambiguity — no breakpoint, too few
messages, a splice that would orphan a `tool_result`, a prefix that didn't come
out byte-equal — it forwards the original prompt unchanged. It can fail to
*help*, but it structurally cannot *break a turn* or silently corrupt the cache.
A summarizer that mis-summarizes degrades the model's context invisibly.

## The one claim that could be wrong: the cache cascade

This is the load-bearing assumption, stated plainly so it can be attacked.

fak keeps the **head** breakpoint's bytes identical, but dropping the middle
**shifts the byte position of every later breakpoint**, including a recent one
Claude Code marks near the end. fak's reasoning: when the recent breakpoint no
longer matches, Anthropic's cache **walks backward** (documented as up to ~20
content blocks per breakpoint, 4 breakpoints per request) and **cascades** to the
still-valid head prefix, so the dominant cache hit survives and only the dropped
middle is re-billed.

**That cascade is an informed hypothesis about Anthropic's internal cache
lookup, not a measured fact.** The dogfood proves the prefix bytes survive; it
runs against a mock upstream and is structurally blind to whether the real cache
cascades or instead misses at the shifted breakpoint and **re-bills the dropped
middle as fresh input**. If the cascade fails, the shed-token savings can
collapse toward zero on the compacting turn, and the prefix-preservation work
buys nothing on that turn.

What would settle it: correlate fak's **witnessed** shed-tokens against the
provider's **observed** `cache_read_input_tokens` on real Anthropic traffic, on
the same request, compaction off vs on. **The instrument already exists** —
`/metrics` emits both halves side by side, by design:

- `fak_gateway_compaction_shed_tokens_total` — WITNESSED, *"What fak SENT — not a
  claim about what the provider billed."*
- `fak_gateway_compaction_cache_read_tokens_total` — OBSERVED, provider-reported
  `cache_read`, *"attribute nothing to fak from it alone."*

So this is **not blocked on code**; it is one credentialed Anthropic session away
from a number. Run a long real session through `fak guard -- claude` with
compaction on, scrape those two series, and a high `cache_read` next to a high
`shed` *is* the cascade landing; a cratered `cache_read` next to a high `shed` is
the cascade failing (the dropped middle re-billed). That this is a scrape-and-read
rather than a build is the direct payoff of fence 2 (separating witnessed from
observed). Tracked as epic
[#745](https://github.com/anthony-chaudhary/fak/issues/745). Until that run
exists, the correct phrasing stays: *fak makes a cache hit eligible by proving
byte-identity; whether the provider's cascade realizes it is unmeasured.*

## Ambiguities resolved (each adversarially)

Five distinctions the first-pass audit blurred, each investigated and then
attacked by a skeptic. The phrasing below is what survived the attack.

**1. Cache cascade — see above.** Verdict: *needs-qualification.* Proven:
byte-identity, valid re-decode, tokens shed. Unproven: provider reuse. Confidence
medium; it is the audit's single biggest open risk.

**2. fak vs. Anthropic context-editing is orthogonal, not "better."** They
optimize different axes. Context-editing clears **tool results** (often the
biggest blocks) server-side and pairs with a **memory tool** so cleared content
is *recoverable*; it deliberately *accepts* a cache invalidation to gain a
smaller ongoing prefix. fak drops **whole turns** *irrecoverably* and *preserves*
the cache prefix. "fak is better than context-editing" is a **category error**.
The coherent statement: *choose fak when the bottleneck is re-prefill cost on a
long session with a frozen middle; choose context-editing when the bottleneck is
tool-result bloat and you need semantic recovery of what you cleared.* (The "84%
token reduction" figure that floats around context-editing is from an Anthropic
agent eval; treat it as theirs, on their task — not a fak comparison point. The
`clear_at_least` amortization parameter is documented but I did not
independently verify its cache-write economics — medium confidence.)

**3. "Lossless" was overloaded — here are the four axes.** Replace the single
word with a position on each:

| Axis | fak (drop+splice) | A summarizer |
|---|---|---|
| **Cache bytes** | lossless (byte-identical prefix, proven) | lossy (rewrite → cache miss) |
| **Trust boundary** | kernel sees full history + a stub *documenting* the drop → loss is **kernel-observable**, not hidden | summarizer's loss is **silent** to any kernel |
| **Model context** | **lossy** — dropped turns are gone from the model's view | **lossy** — but a summary retains the *gist*, so arguably loses *less* signal per token |
| **Recoverability** | **irrecoverable** (turn is gone) | summary present (lossy-but-there); context-editing+memory is *recoverable* |
| **Measurement** | **opaque** — fak does not measure whether dropping turns degrades the model's reasoning | same gap, rarely acknowledged |

So fak is *byte-lossless* and *adjudication-faithful*, but **not** "lossless" in
the sense that matters to the model's reasoning. The corrected one-liner: *fak
loses turns the way a summarizer loses detail — but it loses them in bytes the
kernel can still audit, and it never pretends the loss didn't happen.* The honest
weakness: a dropped turn loses **more** than a good summary of that turn; fak's
bet is that the *middle* turns are the stale ones least worth keeping.

**4. Anthropic-specific by implementation, not necessity.** The *problem* —
truncating history busts an exact-match prefix cache — exists on the OpenAI wire
too (automatic caching, ≥1024 tokens, 128-token-aligned, exact-match required).
fak doesn't solve it there because (a) it only fires compaction on the
real-Anthropic passthrough, and (b) on the OpenAI wire it rebuilds the body from
decoded messages, so there are no stable original bytes to splice, and (c) there
is no `cache_control` breakpoint to anchor on. An OpenAI-wire equivalent is
*possible in principle* — preserve/reconstruct the original request bytes and
anchor on the **longest stable prefix** instead of a breakpoint — but it trades
fak's clean byte-equality proof for a harder "longest-stable-prefix" argument and
is **not implemented**. So this is a current-scope boundary, not a law of nature.

**5. Drop-the-middle may help the *content* problem too, but that is unmeasured.**
A long session has two distinct problems: the **billing** problem (re-sending N
tokens costs money; the cache discounts the prefix) and the **content** problem
(a model drowning in 200k–500k tokens of stale history degrades — "context rot",
lost-in-the-middle). fak's compaction is *aimed* at billing. Dropping the stale
middle *plausibly* also helps content (fewer distracting tokens reach the model)
— but fak measures **no** task-success delta, so that is a hypothesis, not a
claim. Where a cache-busting **signal-restoring summarizer** could beat fak: a
session so long that the content problem dominates, where replacing 100k stale
tokens with a 2k high-signal recap wins on task success *even while losing on
cost*. fak's own roadmap acknowledges this as the **cut-vs-reset crossover**
(below): once the prefix is stale, a cut stops helping and a fresh reset would
reclaim more.

## The honest fences (current scope)

- **Passthrough-only.** The byte-splice fires only when fronting the **real
  Anthropic API** (`s.anthropicPassthrough`). Identity on every other wire.
- **Anthropic-wire-shaped.** Built around `cache_control` breakpoints; no OpenAI
  equivalent is implemented (see resolution 4).
- **Needs a breakpoint to anchor.** No `cache_control` anywhere → it does nothing
  (`CompactReasonNoBreakpoint`). It relies on the client marking the cached head.
- **Provider reuse is unmeasured** (resolution 1 / epic #745). fak proves
  byte-identity; the cache-hit realization is the credentialed follow-on.
- **No task-success measurement** (resolution 5). fak measures tokens shed, not
  whether dropped turns hurt the model.
- **Reset-vs-cut is shadow-only.** The smarter policy — start a fresh session
  with a carryover seed when the cache goes *stale* — is implemented but
  **recommend-only** (`reset_shadow.go` / `reset_score.go`); it logs a verdict and
  acts on nothing. Today fak does the cut, not the reset. See
  `BUDGET-TRIGGERED-SESSION-RESET-2026-06-25.md`.
- **The Claude Code CLI's own `/compact` is not authoritatively documented.** The
  *API* context-editing is (and it busts cache); the CLI's internal compaction is
  best understood as LLM summarization, but that is medium-confidence inference.

## What is genuinely still open (for the next pass)

1. **Measure the cascade.** Correlate witnessed shed vs observed `cache_read` on
   real Anthropic traffic, off vs on. Settles the audit's biggest risk (#745).
   **Not blocked on code** — `fak_gateway_compaction_shed_tokens_total` and
   `fak_gateway_compaction_cache_read_tokens_total` already emit both halves; it
   needs one credentialed session scraped, not a feature.
2. **Measure task-success under compaction.** Does dropping the middle hurt the
   model? Without it, "helps the content problem" stays a hypothesis.
3. **A scorecard row.** The `tools/industry_scorecard.data/` taxonomy has **no
   row** for client-side built-in compaction head-to-head (the `agent`/`memory`
   groups cover fleet serving and KV-cache). A `rows-compaction.json` group would
   make this comparison durable and re-checkable. *(Proposed, not yet added.)*
4. **Promote reset from shadow to live**, gated on the stale-prefix signal — the
   measured form of resolution 5.
5. **An OpenAI-wire path?** Decide whether the longest-stable-prefix variant
   (resolution 4) is worth the weaker proof, or whether OpenAI's automatic cache
   makes it unnecessary.

## See also

- [`explainers/long-sessions-keep-the-cache-hit.md`](../explainers/long-sessions-keep-the-cache-hit.md)
  — the reader-facing version of the crux.
- [`explainers/frozen-trajectory-cache-cliff.md`](../explainers/frozen-trajectory-cache-cliff.md)
  — why a rewrite is a cliff; source of the 20-block / 4-breakpoint cache limits.
- `BUDGET-TRIGGERED-SESSION-RESET-2026-06-25.md` — the cut-vs-reset crossover and
  the reset machinery (shadow / default-off today).
- [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md) — the durability axis.
- `internal/agent/anthropic_compact.go` · `internal/gateway/messages.go` ·
  `internal/gateway/reset_score.go` · `internal/gateway/reset_shadow.go` — the code.
- `experiments/agent-live/compact-100k-session-dogfood-2026-06-25.json` — the
  wire-byte dogfood (mock upstream; proves byte-identity, not provider reuse).
- Tracking: [#745](https://github.com/anthony-chaudhary/fak/issues/745).
