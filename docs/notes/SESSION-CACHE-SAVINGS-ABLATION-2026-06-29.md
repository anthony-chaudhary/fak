# Cache savings in one session: what fak did, and what it didn't

*2026-06-29 — a read of the `fak guard` exit summary for resumed session `c6723f7c`.*

The guard printed a big number on the way out:

```
fak guard: cache saving — saved ~25.5M input-token-equiv across 122 turn(s)
           (NET of the write premium, 86% of the uncached cost)
```

It is a real saving and it is honestly labelled, but it is easy to read it as
*"fak saved 25.5M tokens."* It didn't. **Anthropic's prompt cache saved them; fak
measured and priced the saving.** This note ablates the session three ways so the
attribution is clear, then shows the small set of things fak actually changed.

![Cache-savings ablation for session c6723f7c](session-cache-ablation-2026-06-29.svg)

## The three-way ablation

| Scenario | Prompt-cache cost (input-token-equiv) | vs no cache |
|---|---|---|
| **No cache** — projected, every prompt token billed cold at 1× | **~29.65M** | baseline |
| **Normal cache** — plain `claude`, Anthropic prompt cache on | **~4.15M** | **−86%** |
| **fak guard** — the same cache, plus fak's floor | **~4.15M** on the token axis | **−86%, identical** + 1,104 round-trips saved |

The first two columns are the whole story on the token axis. fak's column is the
interesting one *because it is the same as the column to its left* — by design.

ASCII, for a terminal:

```
No cache    (projected)  ████████████████████████████████  29.65M
Normal cache (claude)    ████▌                              4.15M   ── 86% saved
fak guard   (cache+floor)████▌                              4.15M   ── identical
                              └─ req.Raw forwarded verbatim, prefix byte-identical
```

## Where the 25.5M actually comes from

The number is computed in [`internal/gateway/cache_pricing.go`](../../internal/gateway/cache_pricing.go)
(via `vcachegov.ProveTelemetrySavings`) from three counters the upstream reports on
every turn:

```
saved    = 0.9·cache_read  −  0.25·cache_creation      (read rebate − write premium)
baseline = input + cache_read + cache_creation          (the uncached counterfactual)
savedPct = saved / baseline = 0.86   (printed)
```

Working backwards from the two printed figures:

- `baseline = 25.5M / 0.86 ≈ 29.65M` — what the prompt would have cost with no cache.
- `actual   = baseline − saved ≈ 4.15M` — what it actually cost.

A cache read is billed at `0.1×` and a 5-minute cache write at `1.25×` of the base
input price. The session is heavily **read-dominated**: if cache-creation is small,
`cache_read ≈ 28.3M`, meaning roughly 96% of all prompt tokens across the 122 turns
were served from the warm prefix instead of re-billed. That is what a long,
stable-context agent session looks like when the cache works.

The split of `cache_read` vs `cache_creation` is not recoverable from the summary
line alone; it lives on `/metrics` (`fak_vcache_saved_token_equiv` and the raw
`cache_read_input_tokens` / `cache_creation_input_tokens`). The two printed numbers
pin `baseline` and `actual`, which is what the ablation needs.

### OBSERVED, not WITNESSED

This is the load-bearing distinction. The token counts are **OBSERVED** — relayed
verbatim from Anthropic. fak prices them with the published multipliers and reports
the dollars/tokens as *cost evidence*, never as a fak trust claim. The provenance
comment in `cache_pricing.go` is explicit: *"fak relays the provider's token counts;
it does not author them."* So "fak saved 25.5M" is the wrong sentence. The right one
is **"the provider cache saved 25.5M; fak made it legible."**

## Why "normal cache" and "fak guard" are identical on the token axis

`fak guard -- claude` runs in **passthrough**: it forwards `req.Raw` byte-for-byte to
the real Anthropic API (`internal/gateway/messages.go:512`). The client's
`cache_control` breakpoint — and every byte before it — is unchanged through the
kernel hop. So the prefix the provider hashes for its cache is identical to what
plain `claude` would send, and the cache read/write the provider performs is exactly
the same. **fak cannot, and does not, move the 25.5M.** Remove fak entirely and the
cache saving is the same. (This is the same passthrough fact that keeps outbound
context rewrites inert on the live route today.)

## What fak itself changed this session

Three levers, none of which touch the headline number:

1. **Tool-floor prune (token axis, real but small).** fak dropped **131 unreachable
   tool definitions** from `tools[]` across 131 turns — tools the floor would
   `DEFAULT_DENY` anyway, so removing the advertisement never shrinks the reachable
   action set. The pruner splices *after* the cache breakpoint and re-proves the
   protected prefix is byte-identical, so it never bursts the cache. This saves
   *uncached* tokens: order ~tens of thousands total (≈ 20–66k depending on def
   size), which is **~0.1% of the 25.5M headline.** It is fak's only genuine
   token saving here, and it is a rounding error next to the provider cache.

2. **Floor effect (latency axis — the real win).** fak stopped **1,104 bad tool
   calls before they cost a wasted round-trip**: 1,101 repaired in-flight and 3
   denied outright. That is 87% of the 1,272 proposed calls. This saves *round-trips
   and wall-clock*, not prompt-cache tokens — a different axis from the chart's left
   panel, and the place fak earned its keep this session. (Three calls were also
   blocked on policy: `POLICY_BLOCK` ×1, `SELF_MODIFY` ×2.)

3. **Compaction (idle, correctly).** Enabled with a 48k budget, it **fired 0 times**
   (136 bailed, almost all `under_budget`). Compaction only sheds tokens when context
   sprawls past the cut; nothing did, so it shed 0. "0 fired" here means *working and
   idle*, not disabled.

Plus the audit journal: 171 hash-chained rows appended (3,195 total). Observability,
not savings.

## The honest one-liner

> The provider's prompt cache did ~99.9% of the token saving and would have done it
> without fak. fak's contribution this session was to **price that saving honestly**,
> trim a sliver of uncached tool-def tokens, and — its real job — **catch 1,104 bad
> tool calls before they wasted a round-trip.**

## Reproduce / verify

```sh
# the tamper-evident decision trail behind the summary
fak audit verify "$APPDATA/fak/guard-audit.jsonl"

# the same saved-token-equiv engine, offline, over telemetry
fak vcache observe   # → fak_vcache_saved_token_equiv, matches the live gateway

# raw provider counters live on the gateway
curl -s localhost:<port>/metrics | grep -E 'cache_read|cache_creation|saved_token_equiv'
```

## See also

- [`internal/gateway/cache_pricing.go`](../../internal/gateway/cache_pricing.go) — the pricing model and its OBSERVED-vs-WITNESSED provenance.
- [`internal/vcachegov/proof.go`](../../internal/vcachegov/proof.go) — `ProveTelemetrySavings`, the engine shared by the live summary and `fak vcache observe`.
- [O(1)-context-window economics](../explainers/o1-context-window-economics.md) — the broader cost model this sits inside.
