# The fak-OWN cache-value path in `fak guard` (and why F is often ~0)

**One question this answers:** when I run `fak guard -- claude`, is fak delivering
its *own* cache value, or just relaying the provider's? And if the exit summary says
"fak-slice diagnostic — F is ~0", what do I do about it?

This note exists so an agent does not have to re-trace five files to learn the path.

## Two different cache acts, opposite verbs — keep them straight

`fak guard` on the Anthropic passthrough does **two** distinct things to the cache.
They are easy to conflate; they are not the same, and they do not conflict.

| act | verb | who owns the saving | how it's labelled | measured on |
|-----|------|---------------------|-------------------|-------------|
| **PRESERVE-prefix** (default) | *keep* the `cache_control` head **byte-identical** so the upstream prompt-cache **hit survives** | the **provider** (Anthropic) | **OBSERVED** — fak relays it, never claims it | `docs/nightrun/cache-savings.jsonl`, `mechanism: provider_prompt_cache` |
| **SHED-middle** (compaction) | *purposefully* **drop** the old, un-cacheable middle turns the provider re-bills every turn | **fak** (authored) | **WITNESSED** — fak did it | `docs/nightrun/cache-savings.jsonl`, `mechanism: compaction_shed` |

The trap is thinking shed "breaks the cache". It does not: shed **keeps** the
preserved head anchor and only cuts the tokens **after** the `cache_control`
breakpoint — the span the provider was *not* caching anyway. Preserve keeps the
hit; shed kills the re-bill. Both are "cache management"; they point in opposite
directions and both are wins.

`--compact-history-budget` (default `48000`, `internal/gateway/gateway.go`) is the
SHED lever. `--compact-anchor-head` (default on, `#1407`) re-anchors the protected
prefix on the stable system/tools head so a real Claude Code body — whose only
`cache_control` breakpoint sits near the *end* and would otherwise protect the whole
conversation — can still shed its middle.

## Why F (fak's slice) is frequently ~0 in guard mode

`fak guard -- claude` is a **proxy**: fak runs no KV kernel of its own, so the
in-kernel KV-prefix reuse witness (Track 1, `docs/nightrun/cache-value.jsonl`) is
structurally 0 — there is no fak kernel to reuse a prefix. That is honest, not a bug.

So in proxy mode fak's *only* own-value mechanism is **compaction-shed**. When it does
not fire, F is ~0 and every dollar of the session is provider-owned. The dominant
reason it does not fire is **anchor-starved**: the incoming body's `cache_control`
breakpoint protects more than the budget, so there is nothing left to shed. The exit
summary names this explicitly and tells you the fix (`--compact-anchor-head`, on by
default — so if you see it, either it was disabled or the traffic carries no stable
head to re-anchor on).

## How to SEE the value (the one command)

```
fak cachevalue report            # two-track P&L + owner split over ALL sessions
```

- **Track 2 (OBSERVED $)** is the provider prompt-cache saving — real, and usually
  the bulk of the dollars in guard mode.
- **Owner attribution** splits `provider_teq` vs `fak_teq`. `fak_share` is fak's
  own slice. In pure proxy sessions this is often 0 by the structure above; a session
  where compaction shed a real middle shows a non-zero `fak_teq` / `compaction_shed`.

The per-session `fak guard` exit summary also prints:
- the **cache attribution** line (`provider ~X + fak ~Y`, OBSERVED vs WITNESSED), and
- the **fak-slice diagnostic** naming *why* F is ~0 this session, now pointing back
  at `fak cachevalue report`.

## To make fak's own value actually fire (operator lever)

The goal is a non-zero `fak_share`. The path:

1. Run a real long `fak guard -- claude` session (history must sprawl past ~48k
   resident tokens for there to be a middle worth shedding).
2. On the exit summary (or live `/metrics`), read `CompactionBailReasons` /
   `CompactionAnchorStarved`. That names the exact gate holding shed at 0.
3. If `anchor-starved`: confirm `--compact-anchor-head` is on and the traffic has a
   stable system/tools head. The burst-economics gate (`CacheBurstPaysBack`, `#1408`)
   also requires either a wired bounded turn-horizon or an observably TTL-idle trace —
   a plain warm session with no horizon will not burst. Wiring a session-turn horizon
   into the guard is the lever that lets the long-session path fire.
4. Re-run `fak cachevalue report`: a `compaction_shed` row → non-zero `fak_teq` →
   fak's own live cache value, shown.

## Source map (so the next agent does not re-trace it)

- lever + defaults: `internal/gateway/gateway.go` (`DefaultCompactHistoryBudget`),
  flags in `cmd/fak/guard.go` / `cmd/fak/serve.go`
- fire + witness: `internal/gateway/messages.go` (`compactAnthropicRawWithReason`),
  `internal/gateway/metrics.go` (`compactShed` → `AdjudicationSummary`)
- bail vocabulary: `internal/agent/anthropic_compact.go` (`CompactReason*`),
  `internal/agent/anthropic_cachebp.go` (`BreakpointReason*`)
- ledger write: `cmd/fak/cachevalue_savings.go` (`appendObservedCacheSavings`)
- report + owner split: `internal/cachevaluereport/` , `fak cachevalue report`
- exit summary: `cmd/fak/guard_format.go` (`formatCacheAttribution`,
  `formatFakSliceDiagnostic`)
