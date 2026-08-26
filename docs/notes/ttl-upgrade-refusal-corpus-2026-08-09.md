---
title: "Managed-cache TTL-upgrade refusal taxonomy across the gateway-usage corpus"
description: "Histogram of every recorded 1h TTL-upgrade refusal and an estimate of cacheable prefix tokens left without 1h protection, for #3633."
date: 2026-08-09
---

# TTL-upgrade refusal corpus study (2026-08-09)

## Verdict

**The gate refuses often, but this corpus does not show byte-safety making false
refusals.** The only session that recorded refusal reasons refused 46 upgrade
attempts and upgraded 19: a **70.8% refusal share**. `volatile_head` accounts for
45/46 refusals (**97.8%**) and an estimated 3.11M cacheable-prefix tokens left on
the shorter TTL; `no_stable_breakpoint` accounts for 1/46 and about 69k tokens.
That is materially conservative in effect, but the session-level ledger cannot
distinguish a correctly detected volatile byte from a false positive. Keep the
fail-safe byte rule and pursue the bounded redaction/spec work in #2191 rather
than weakening it.

“Left on the table” here means **not upgraded from the existing 5-minute cache
TTL to 1 hour**. It does not mean the provider necessarily stored no cache entry.
The ledger witnesses fak authoring an upgrade, not provider acceptance.

## Corpus and fields

The committed corpus is `docs/nightrun/gateway-usage.jsonl`:

- schema: `fak-gateway-usage-ledger/1`
- 3,353 session-exit rows, 2026-07-01 through 2026-07-11 UTC
- blob witness: `f65df29615cdaaea24a28a99b57dac7874b13cb9`
- exact refusal counts: `counters.cache_ttl_upgrade_reasons`
- authored upgrades: `counters.cache_ttl_upgrades_upgraded`
- token proxy: `counters.cached_prompt_tokens / counters.cached_turns`

All 3,353 rows were decoded. Exactly one row has a non-empty
`cache_ttl_upgrade_reasons` map: line 2,956, generated
`2026-07-10T21:15:38Z`, `session_type:"guard"`, `context:"claude"`. It records
51 cached turns, 3,519,466 cached prompt tokens, 46 refusals, and 19 authored
upgrades. Thus the observed mean cacheable prefix is
`3,519,466 / 51 = 69,009.14` tokens per cached turn. The ledger stores refusal
counts as a per-session histogram rather than per-turn records, so every refusal
is counted but individual refused turns cannot be listed separately.

## Refusal-cause histogram and token estimate

| Refusal class | Refused attempts | Share of refusals | Estimated prefix tokens left at 5m |
|---|---:|---:|---:|
| `volatile_head` | 45 | 97.8% | 3,105,411 |
| `no_stable_breakpoint` | 1 | 2.2% | 69,009 |
| **Total** | **46** | **100%** | **3,174,420** |

Estimate formula for each class:

```text
reason_count * (cached_prompt_tokens / cached_turns)
```

This is an exposure estimate, not provider billing. It assumes a refused attempt
would have protected a prefix comparable to the session's average provider-cache
read. The current ledger has no per-refusal prefix-token field, so a more precise
estimate is not recoverable from this corpus. The estimate should not be summed
as unique bytes: a long-lived prefix may recur on multiple refused turns.

Reproduction query (Python standard library only, run from the repository root):

```python
import json

rows = [json.loads(line) for line in open(
    "docs/nightrun/gateway-usage.jsonl", encoding="utf-8"
)]
witnesses = [r for r in rows if r["counters"].get("cache_ttl_upgrade_reasons")]
assert len(rows) == 3353 and len(witnesses) == 1
c = witnesses[0]["counters"]
mean_prefix = c["cached_prompt_tokens"] / c["cached_turns"]
for reason, count in sorted(c["cache_ttl_upgrade_reasons"].items()):
    print(reason, count, round(count * mean_prefix))
```

## Interpretation and recommendation

The observed refusal share is not rare, so the mechanism is conservative in
operational effect and leaves a meaningful amount of prefix reuse without 1-hour
protection. But **this study does not prove byte-safety is over-refusing**:
`cache_ttl_upgrade_reasons` records the closed decision class, not the bytes that
triggered it or an independent stable/volatile judgment. The dominant result is
instead diagnostic: nearly the entire refusal tax is one tractable class,
`volatile_head`.

Recommendation:

1. Do not relax identity-on-ambiguity from this evidence.
2. Use #2191 to specify and soak a semantics-preserving volatile-head
   redaction/hoist, then compare refusal rate and provider-accepted cache value.
3. If this analysis must become recurring or billing-grade, add a per-attempt
   prefix-token estimate and provider-acceptance witness before introducing live
   alarming; that instrumentation is outside #3633.

## Generation classification

This leaf is **`gen/now`**, not an unclassified future bet: the live issue has a
bounded research-only done condition, an already committed corpus, a closed
reason vocabulary, and no dependency on changing runtime behavior.

- **Promotion evidence:** the durable ledger already exposes the exact refusal
  histogram and token counters needed to execute the study now; the bounded
  artifact and acceptance gate are explicit in #3633.
- **Demotion/retirement evidence:** none for this research leaf. The runtime
  follow-on remains separate because this corpus cannot validate redaction
  safety or provider acceptance.
- **Invalidating assumption:** if a per-refusal prefix-token witness shows that
  refused attempts are not comparable to the session's average cached turn,
  the 3.17M estimate is biased and must be replaced. If independent byte labels
  show the `volatile_head` detections are false positives, the recommendation to
  preserve the fail-safe rule is also invalidated.
