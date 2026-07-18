---
title: "The cache cliff: why a 96.6% prompt-cache hit rate can still be a warning sign"
description: "The high prompt-cache hit rate everyone quotes — fak's own fleet audit measured 96.6% of ingested tokens cached, median 99% — is real, but it is purchased with a frozen, append-only trajectory. It is a prefix match, so it holds only while the harness never touches history. This story tells the cliff: the ceiling rises to 99% with length, then falls off toward 0% the moment you edit history, and why that means the honest metric is reread work deleted, not hit rate quoted. Every figure comes from the deterministic tools/cache_curve.py."
slug: cache-cliff
keywords:
  - prompt cache hit rate
  - cache cliff
  - KV cache reuse
  - frozen trajectory
  - context management
  - addressable KV cache
  - agent serving
  - cache benchmark honesty
date: 2026-07-17
---

# The cache cliff

**Short answer.** The prompt-cache hit rate vendors quote — 90%+, "we cache almost
everything" — is real, and fak's own fleet audit is right there with it: **96.6% of all
ingested tokens cached, median 99%** across 199 real sessions. But that number is high
*because the trajectory is frozen*, not because caching is free. It is a **prefix
match**: it holds only while the harness never edits history. Start managing context —
compaction, re-ordering, cross-agent fan-out — and the same default cache falls off a
cliff toward **0%**. The honest metric is not the hit rate you quote; it is the reread
work you actually delete.

*For anyone comparing serving stacks on "cache hit rate" and wondering why the numbers
are all suspiciously high. By the end you will know why the ceiling is high, exactly
which moves send it to zero, and why fak reports a different number on purpose. Every
figure here comes from [`tools/cache_curve.py`](../../../tools/cache_curve.py)
(deterministic, stdlib-only) and the fleet audit; the mechanism is in the
[frozen-trajectory cache-cliff explainer](../../explainers/frozen-trajectory-cache-cliff.md).*

## Why the public number is high: it's the frozen ceiling

A prompt cache keys on the longest matching **prefix** of the request. If turn N is
just turn N-1 plus a few appended tokens, the whole earlier context is a cache hit. So
for a single, linear, **append-only** agent — one that never edits its own history — the
hit rate does not just stay high, it *rises with length*:

| turns | frozen-ceiling cache-hit |
|---|---|
| 10 | 82% |
| 50 | 96% |
| 200 | 99% |

That ceiling is the number that gets quoted. And it is not hypothetical: the fak fleet
audit (`session_audit.py audit --since-days 30`, 199 sessions) measured **96.6% of all
ingested tokens served from cache, with a median of 99%**. Both are real. Neither is a
caching triumph to celebrate — they are the *signature of a harness that never touches
history*.

There is even a counter-intuitive corollary that trips people up: for a frozen agent,
**more tool calls raise the hit rate**, not lower it (~81% mean for sessions with no
tool calls, ~98% for sessions with 16+). Each tool result is just more appended prefix.
"More tool density lowers the cache" is *false* — as long as you stay frozen.

## The cliff: what sends it to 0%

The ceiling assumes the harness never edits history. But context management *is*
editing history — and the default cache is keyed on position in a frozen prefix, so any
edit at depth D invalidates everything from D onward. Push edit-depth into the prefix
and the hit rate falls off a cliff:

| edit-depth into prefix | cache-hit (from a 99% ceiling) |
|---|---|
| 0% (append-only) | 99.0% |
| 5% | 94.1% |
| 25% (compact ¼) | 74.3% |
| 50% | 49.5% |
| 100% (rewrite the head) | 0.0% |

Two more axes bend it the same way:

- **Compaction / re-ordering.** The moment you compress or reshuffle history to fit a
  budget, you have edited the prefix — and forfeited every cached token after the edit.
- **Cross-agent fan-out.** Give each agent its own trajectory and each one's *personal*
  hit rate stays high, but the shared setup is re-prefilled once per agent: 0% reuse
  *across the fleet*, the exact work a multi-agent serving stack most wants back.

So the high number is fragile in precisely the situations a real agent workload creates.
Quoting it as a headline is quoting the best case of a structure that is designed to
degrade.

## The honest metric: reread work deleted, not hit rate quoted

This is why fak does not lead with a hit-rate percentage. A percentage measures how well
you stayed inside the frozen regime. The thing worth measuring is how much re-computation
you **delete** once you *leave* it — which needs a cache keyed on **content and
identity**, not position in a frozen prefix. That is the addressable, coherence-checked
KV cache fak owns: it can evict, compact, and reorder history and still reuse the spans
that are genuinely unchanged (the eviction is bit-exact — see
[`max|Δ| = 0`](bit-exact.md)). The percentage is a side effect; the deleted reread is the
result.

## Run it yourself

Every number above is reproducible with no key, model, GPU, or network:

```
python tools/cache_curve.py
```

It is deterministic and stdlib-only — it prints the frozen ceiling curve, the
edit-depth decay, and the tool-density and fan-out axes, so you can check the cliff
rather than take the ceiling on faith.

## The honest fences

- **The ceiling numbers are a model, the 96.6% is measured.** The `cache_curve.py`
  curves are an analytic prefix-match model; the 96.6% / median-99% is a real audit of
  199 sessions. They agree because real fak sessions *are* mostly frozen today — which is
  the point, not a coincidence.
- **The cliff is not a fak defect; it is the default-cache defect fak addresses.** A
  position-keyed prompt cache *cannot* survive history edits. The story is why the
  quoted number misleads and what metric to use instead — not a benchmark loss.

---

**Related:** [`frozen-trajectory cache-cliff explainer`](../../explainers/frozen-trajectory-cache-cliff.md)
(the full mechanism and scaling laws) ·
[`max|Δ| = 0`](bit-exact.md) (how fak deletes the reread the cliff exposes, bit-exactly) ·
[`tools/cache_curve.py`](../../../tools/cache_curve.py) (the runnable demonstrator).

_Dimension H (Benchmark-as-story) of the
[concept-popularization epic](../../notes/CONCEPT-POPULARIZATION-EPIC-2026-07-02.md)._
