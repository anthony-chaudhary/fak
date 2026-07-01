# Offensive cache-breakpoint placement is a net, fak-specific saving on the Claude path (2026-07-01)

> The offline economic witness for the offensive half of #806. It answers, with a green
> deterministic test, the one question the placement's correctness tests never priced:
> **does fak's cache-breakpoint placement actually save money on the Claude/Anthropic path,
> and is that saving fak's own?**

## What was missing

The offensive cache-breakpoint placement (`agent.PlaceAnthropicCacheBreakpoint`, wired into
the Anthropic passthrough at `internal/gateway/messages.go:480` via
`Server.maybeCompactAnthropicRaw`) was thoroughly witnessed for **correctness** — where the
breakpoint lands, prefix byte-identity, volatility handling, idempotence
(`internal/agent/anthropic_cachebp_test.go`) — and the prompt-cache **pricing model** was
witnessed in isolation (`internal/gateway/cache_pricing_test.go`). But nothing tied the two
together: there was no witness that *placement → provider caches the head → net dollars
saved*, and no witness that the saving is **fak-specific** rather than a cache the client
would have earned anyway.

This note records that witness. The test is
`internal/gateway/provider_cache_fak_placement_savings_test.go`
(`TestFakPlacementUnlocksProviderCacheSavings`).

## The claim, scoped honestly

For a caller that sends **no `cache_control`** — a raw OpenAI-shaped client, a minimal SDK, a
hand-rolled `/v1/messages` request — provider prefix caching is entirely on the table: the
stable system/tools head is re-prefilled at full price every turn. fak's offensive placement
splices a `cache_control` breakpoint onto that stable head, so the provider serves turns
2..N from cache at 0.1× instead of 1.0×.

**Scope fence (load-bearing):** this is the win for callers that send *no* breakpoint. It is
**not** a win over Claude Code, which already marks its own head — there fak returns identity
(`already_set`) and the provider cache is the *client's*, not fak's. This witness never claims
that slice.

## Provenance (per [net-true-value](../standards/net-true-value.md))

| Element | Provenance | How |
|---|---|---|
| Placement | **WITNESSED** | the real `Server.maybeCompactAnthropicRaw` transform on the real body; removing the inserted breakpoint span recovers the original bytes exactly. |
| Counterfactual | **WITNESSED** | the same body with the lever off is forwarded byte-for-byte; with no breakpoint the provider caches nothing. |
| Causality | **WITNESSED** | `cache_read` is derived from the *forwarded bytes* of each arm: 0 without fak's breakpoint, >0 with it. |
| Magnitude | **MODELED** | the shipped `gateway.CachePricing` model + Anthropic's published 0.1× read / 1.25× 5m-write multipliers, **net of the one-time write premium**. |

The baseline is the **real alternative** (the identical caller with no fak → no provider
cache), not a strawman; the saving is **net** of the write premium; the scope is stated; and
it is reproducible with one command. This is the offline dollar witness — the *live*
provider-side `cache_read_input_tokens` capture on a credentialed host remains the follow-on
(epic #745), exactly as the compaction claim (CLAIMS.md) already fences.

## Captured witness

Representative run (8-turn session, ~2000-token stable head, Opus-4.8 example base rates
`{$5/$25 per MTok}` — the pricing model takes the rate as a parameter, so the token-equivalent
saving is rate-independent and the `$` scales with the input price):

```
=== RUN   TestFakPlacementUnlocksProviderCacheSavings
FAK-SPECIFIC PROMPT-CACHE SAVINGS WITNESS (offensive placement #806, Claude/Anthropic path)
  scope         : one caller that sent NO cache_control; 8-turn session; 5m cache tier; Opus-4.8 example rates {$5/$25 per MTok}
  placement     : WITNESSED — real Server.maybeCompactAnthropicRaw spliced an ephemeral breakpoint on the stable head, prefix byte-identical
  counterfactual: WITNESSED — same body, lever off => forwarded unchanged; no breakpoint => provider cache_read=0 across all 8 turns
  causality     : WITNESSED — cache_read derived from the forwarded bytes: 0 without fak, 14364 tok with fak (creation 2052 tok on turn 1)
  magnitude     : MODELED   — shipped CachePricing + published Anthropic 0.1x read / 1.25x write economics, net of the write premium
  ---- economics ----
  baseline (no fak) : $0.122480   (head re-prefilled at full price every turn)
  with fak          : $0.060407
  net saving        : $0.062073  = 50.7% of the prompt-cache spend over 8 turns
  turn-1 saving     : $-0.002565  (NEGATIVE — the honest one-time write premium; break-even at turn 2)
  attribution       : provider(read rebate) 12928 + provider(write premium) -513 = net 12415 token-equiv, 100% fak-unlocked
--- PASS: TestFakPlacementUnlocksProviderCacheSavings (0.00s)
```

The two halves reconcile exactly (asserted at `eps=1e-12`): net token-equiv `12415 × $5/MTok =
$0.062073`, the same figure the per-turn dollar difference produces. The turn-1 line is
deliberately negative — the one-time cache-write premium — and break-even lands at turn 2, the
same break-even `CachePricing.SavingsUSD` models.

## Reproduce

```bash
# native go test is blocked on the Windows dev box; run under WSL (see AGENTS.md)
go test ./internal/gateway -run TestFakPlacementUnlocksProviderCacheSavings -v -count=1
```

Deterministic: no key, no network, no GPU. The `-v` flag prints the witness block above.

## See also

- [`docs/cache-value-rollup.md`](../cache-value-rollup.md) — the two-track (WITNESSED kernel
  reuse / OBSERVED provider-$) reader-facing layer this witness feeds.
- [`CLAIMS.md`](../../CLAIMS.md) — the shipped/simulated honesty ledger (the M2 placement and
  compaction claims this note prices).
- `internal/agent/anthropic_cachebp.go` — the offensive placement itself; its correctness is
  witnessed in `internal/agent/anthropic_cachebp_test.go`.
