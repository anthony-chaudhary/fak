---
title: "Long sessions: shed history, keep the cache hit"
description: "Why a long Claude Code session gets expensive, why the obvious fix makes it worse, and how fak sheds old turns while keeping the provider's prompt-cache prefix byte-identical so the discount survives."
---

# Long sessions: shed history, keep the cache hit

> The sibling explainers ([addressable KV cache](addressable-kv-cache.md),
> [the frozen-trajectory cache cliff](frozen-trajectory-cache-cliff.md)) cover the
> theory of cache reuse. This one is the practical version: the one flag on
> `fak guard` that stops a growing session from getting more expensive every turn,
> and exactly what it does and does not promise.

*For anyone running a long Claude Code (or similar) session and watching the cost
climb. No `fak` internals needed — only the basic fact that an agent re-sends its
whole conversation every turn.*

## The problem, in one breath

An agent has no memory between turns. So every turn, the client re-sends the entire
conversation so far. A short chat barely notices. A 100k-token session re-sends 100k
tokens just to ask the next question. The bill grows with the square of the work.

Providers soften this with a **prompt cache**: if the front of your prompt is exactly
what they saw last time, they charge a fraction for that part. The catch is in the
word *exactly*. The cache only applies while the cached prefix stays byte-for-byte
identical to what you sent before.

## Why the obvious fix backfires

The natural way to shrink a long transcript is to summarize the old turns and send the
summary instead. That feels right and costs more.

Summarizing rewrites the body of the prompt. A rewrite reorders bytes. Reordered bytes
break the byte-for-byte match the cache depends on, so the provider stops discounting
and re-charges the whole thing at full price. You did work to save money and the bill
went up.

## What fak does instead

`fak guard` takes a different route, **on by default**. Instead of rewriting the
prompt, it **drops** the old middle turns and splices the bytes back together. The
cacheable front of the prompt is copied through untouched, byte-for-byte (a `memcpy`,
never a re-serialize), so the provider's cache prefix still matches and the discount
holds. On any ambiguity it does nothing and forwards the original prompt unchanged, so
it never breaks a turn.

There is one important conservative case: if the candidate middle span itself contains
`cache_control`, `fak` treats that span as provider-warm and refuses to drop it
(`cached_span`). A smaller prompt is not automatically cheaper when the current full
history is already being served from the provider cache. In that case keeping the cache
hit beats bursting the cache just to send fewer bytes.

You don't have to ask for it: it fires automatically once a conversation sprawls past
~48k resident tokens (a typical short session is left untouched). Pass a tighter budget
to shed sooner, or `--compact-history-budget 0` to disable it entirely:

```bash
fak guard --compact-history-budget 8000 -- claude   # tighter than the ~48k default
```

## When would you burst the cache?

Only when the remaining session horizon repays the burst. A simple planning rule is:

```text
break_even_turns = ceil((write_mult - read_mult) * invalidated_suffix_tokens
                         / (read_mult * dropped_cached_tokens))
```

For a 50-turn session, compare that number with the turns still ahead. With Anthropic-like
1h cache economics (`write_mult=1.25`, `read_mult=0.1`), dropping 20k tokens that were
already cache-read but invalidating a 40k-token warm suffix takes 23 future turns to pay
back. If you are on turn 40 of 50, do not burst. If you are on turn 20 of 50, it might be
worth a measured, explicit burst. If a future segment-addressable vCache surgery could
invalidate only a 5k-token section instead of the 40k suffix, the same drop repays in
three turns. That is the value of sub-vCache surgery: not magic savings, just a much
smaller invalidated suffix.

The live Anthropic prompt-cache path is prefix-shaped, so arbitrary middle surgery is not
available there today. `fak` can do byte-splice drops only when the cached prefix remains
intact, and it now refuses to silently delete cache-marked middle spans without an explicit
horizon/economics gate.

## What it guarantees, and what it only observes

`fak` is careful about the line between the two.

It **guarantees** one thing: the prefix it ships is byte-identical to what you sent. If
it can't preserve that, it refuses to compact and forwards the original. That makes a
cache hit *possible*. It does not *force* one. Whether the provider actually reuses the
cache is the provider's decision.

So `fak` reports both numbers side by side instead of claiming the win. `/metrics`
exposes `fak_gateway_compaction_*`: the tokens `fak` shed (what it sent) next to the
provider's reported `cache_read` (what came back). The `fak guard` exit line summarizes
both. If `cache_read` is low while the prefix was byte-identical, the miss is
provider-side: a cache TTL expiry, an eviction, or your client moving its own
breakpoint. It is not something `fak` broke, and you see it either way instead of
silently overpaying.

## Checked against real sessions

The break-even rule and the cold-resume planner both rest on one assumption: once a session
sits idle past the cache TTL, the next turn pays to re-establish the prefix. `fak resume
validate` checks that assumption against your own history. It reads a corpus of real session
transcripts, and for every back-to-back pair of turns it compares what the planner would have
predicted (cold if the gap crossed the TTL, warm if not) against what the provider actually
billed (did the next turn re-read the prior prefix, or re-write it?).

Run over this machine's session history, about 82,000 real turn-to-turn boundaries, the
planner's warm-or-cold call matched the provider 97.7% of the time. Where a turn did come back
cold, it re-wrote essentially the whole prompt, which is exactly what the cold-resume price tag
assumes. The one systematic miss is in the safe direction: the planner calls a 5-to-15-minute
gap cold, but the provider often still had the prefix warm. Anthropic's 5-minute cache is a
floor that refreshes on use, so in practice it survives longer than the clock. This
conservative band reflects the governor policy derived in the vCache design note
(`docs/notes/VCACHE-VIRTUAL-API-CACHE-2026-06-24.md` §5.4), which shows pinning is only
justified when λT > ln((W5+μ+L)/L) ≈ 0.81–1.5 requests/TTL depending on latency value and
rate-limit pressure — a tight band where the planner errs toward declaring cold. The planner
errs toward declaring cold, which is the conservative way to be wrong: it never claims a warm
cache that isn't there.

There is a subtlety the planner gets slightly wrong. A genuine multi-hour resume usually starts
a fresh transcript file, and `fak resume validate` now measures those first turns directly.
They confirm the cold case at scale, but with a twist: a resume re-caches only about two-thirds
of the carried transcript and re-sends the rest as ordinary input. The planner prices the whole
thing at the higher cache-write rate, so it over-states the cost of a cold resume by roughly a
third of that premium. The same scan also turns up plenty of resumes that opened straight onto a
still-warm prefix from the prior session, a saving the planner does not yet count. Folding both
corrections back into the price tag is the next step.

```bash
fak resume validate --corpus ~/.claude/projects   # back-test the planner on your own sessions
```

Tracking: [#745](https://github.com/anthony-chaudhary/fak/issues/745).

*Last updated: 2026-06-26*
