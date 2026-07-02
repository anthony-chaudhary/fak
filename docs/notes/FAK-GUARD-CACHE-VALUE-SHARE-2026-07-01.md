# fak guard's share of cache value on real sessions — proven (2026-07-01)

**Question.** Of the cache value fak reports on real sessions, what percent is
fak-authored (fak's own mechanisms) versus the provider's prompt cache? The
headline `fak guard` prints ("saved N tokens") had never been split into a
proven percentage on real traffic — epic #1844's core complaint.

**Answer (measured, from the durable ledgers as of 2026-07-02T02:00Z):**

| scope | provider token-equiv | fak token-equiv | fak share |
|---|---:|---:|---:|
| real `fak guard -- claude` sessions (11 sessions, 2026-07-01..02 UTC) | 153,100,871 | 0 | **0.0000%** |
| all real sessions in the ledgers (guard + `fak run` kernel sessions) | 153,100,871 | 3,441 | **0.0022%** |

On the flagship `fak guard -- claude` path, **100% of the reported cache value
is the provider's prompt cache** — the same cache the client would get with no
fak in the path. fak's own witnessed slice on real traffic is 3,441 tokens of
KV-prefix reuse, all from `fak run` smollm2 kernel sessions (Track 1), none
from guard.

## Witness chain (why this is proven, not asserted)

1. **The provider slice** is OBSERVED: 11 `provider_prompt_cache` rows in
   `docs/nightrun/cache-savings.jsonl` (schema `fak-cache-savings-ledger/1`),
   appended at guard exit by `appendObservedCacheSavingsTo`
   (`cmd/fak/cachevalue_savings.go`) from the live `AdjudicationSummary`.
   Net token-equiv sums to 153,100,871.2.
2. **The fak-on-guard zero is a witness, not a recording gap.** The same
   writer emits BOTH mechanisms from the same summary in one call
   (`cachevaluereport.NewSavingsRows`): a `provider_prompt_cache` row when
   cache tokens exist and a `fak`/`compaction_shed` row when shed > 0. The 11
   provider rows prove the writer ran at every guard exit; the 0 compaction
   rows therefore prove `CompactionShedTokens == 0` on every one of those
   sessions (anchor-starvation, #1407). KV-prefix and vDSO are structurally 0
   on the guard proxy path — `Decide` increments neither
   (`cmd/fak/guard_format.go`), the #1844 audit finding.
3. **The fak slice that does exist** is WITNESSED: Track-1 kernel rows
   (`docs/nightrun/cache-value.jsonl`), 9 `fak run` smollm2 sessions
   (2026-06-28, 2026-06-30), 3,441 reused KV-prefix tokens (W26: 260,
   W27: 3,181).
4. **Reproduce:** `fak cachevalue report` — the owner-attribution table now
   prints `fak_share` per period (pre-divided, refused with `-` when a period
   total is not positive), and the JSON report carries `fak_share_pct`.

## Scope fences (what this number does NOT say)

- **Placement is identity on this population.** The #806 witness
  (`internal/gateway/provider_cache_fak_placement_savings_test.go`) proves
  fak's breakpoint placement unlocks a 100%-fak-attributable provider-cache
  saving — but only for callers that send NO `cache_control`. Claude Code
  marks its own head, so on these 11 real sessions fak's placement is
  `already_set` identity and claims none of the provider slice.
- **Managed cache (1h-TTL upgrade, 8b618eec) did not fire.** It is gated to
  API-key-billed sessions; these sessions are subscription-billed, and the
  usage ledger carries no TTL-upgrade counters for them.
- **Share is over recorded rows only.** `docs/nightrun/gateway-usage.jsonl`
  records only MCP `serve` exits (all-zero counters) — guard traffic that
  predates the savings ledger (before 2026-07-01) is not in the denominator
  (usage-plane gaps, epic #1601).

## What would move the number (the #1844 frontier)

The fak share rises above ~0 on the guard path only via fak-authored wire-side
levers: C2 (live ablation arm to prove deltas on real traffic), C6 (1h
`cache_control` TTL upgrade on the stable prefix), C7 (uncached-remainder
shrink), #1603 (breakpoint plan for no-cache_control callers — already
witnessed in test, needs a real no-breakpoint caller population), and
compaction de-starvation (#1407). Each lands with a sweep row per the epic's
acceptance, and this note's table is the baseline it gets compared against.
