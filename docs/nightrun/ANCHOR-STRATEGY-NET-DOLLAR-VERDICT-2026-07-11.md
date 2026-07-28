---
title: "Anchor-strategy net-dollar verdict (#2809)"
description: "Head-anchored compaction versus the first-breakpoint idle default, measured on real traffic: a conservative net-dollar floor over one pinned ledger week."
---

# Anchor-strategy net-dollar verdict — `CompactAnchorFirstBP` vs `CompactAnchorHead` (#2809)

**Issue:** #2809 (parent epic #2783, cache-value savings program). Siblings: #2780 (savings
audit), #1844 (ablate), #745 (compaction value), #1490/#1072 (attribution), #2393 (deterministic
compaction).

**Question (issue title):** *was the #1407 de-starvation switch — turning compaction from the
`CompactAnchorFirstBP` idle default onto `CompactAnchorHead`, which is what lets the lever actually
fire on real Claude Code traffic — net positive on net dollars, after the cache-write burst it pays?*

## Verdict

**YES — head-anchoring is net-beneficial on real traffic, with a conservative net-dollar floor of
`+$2,735.2977` over the window 2026-07-04..2026-07-11** (pinned to committed ledger `1c55b6668`).

This is a *lower bound*, not a point estimate: head-anchoring's gross shed cleared its cache-write
burst even when charged the **entire** window's provider write premium — an over-charge (see
methodology). The true net-of-burst benefit is `≥` this floor.

| quantity | value | source |
|---|---|---|
| compaction fires (head arm, N) | **605** | `compaction_shed` rows with `compaction_saved_usd > 0` |
| gross shed (head-arm benefit) | **`+$3,115.7683`** | Σ `compaction_saved_usd` (shed valued at input price) |
| window provider write premium (burst upper bound) | **`$380.4706`** | Σ `write_premium_usd`, all `provider_prompt_cache` rows |
| **conservative net floor** | **`+$2,735.2977`** | gross shed − entire window write premium |
| window | **2026-07-04 .. 2026-07-11** | ledger `date` range of the folded rows |

## Provenance / how to reproduce

Real traffic, committed ledger: **`docs/nightrun/cache-savings.jsonl`** (schema
`fak-cache-savings-ledger/1`, 2,045 rows over 2026-07-01..2026-07-11 as of committed ledger
`1c55b6668`). The two sides of a
head-anchor fire land on **separate mechanism rows**: `compaction_shed` (the fak-authored shed, no
burst on that row) and `provider_prompt_cache` (the cache-write premium — where the burst lands, but
not attributed per fire).

# Pinned to committed ledger 1c55b6668 so these numbers reproduce EXACTLY and forever — nightrun
# folds new rows into cache-savings.jsonl ~7×/day, so reading the live working-tree file drifts as
# traffic accumulates (that is expected, not a defect). Feed the committed object on stdin:
#   git show 1c55b6668:docs/nightrun/cache-savings.jsonl | python anchor_net_floor.py
```python
import sys, json
gross = wp = 0.0; fires = 0; dates = []
for line in sys.stdin:
    line = line.strip()
    if not line: continue
    r = json.loads(line)
    if r.get('mechanism') == 'compaction_shed' and (r.get('compaction_saved_usd', 0) or 0) > 0:
        gross += r['compaction_saved_usd']; fires += 1; dates.append(r['date'])
    wp += r.get('write_premium_usd', 0) or 0
print(f"fires={fires} gross_shed=${gross:.4f} window_write_premium=${wp:.4f} "
      f"floor=${gross-wp:+.4f} window={min(dates)}..{max(dates)}")
# -> fires=605 gross_shed=$3115.7683 window_write_premium=$380.4706 floor=$+2735.2977 window=2026-07-04..2026-07-11
```

The verdict rendering (three-way, worded, floor-gated) is the pure `ablate.ObservedAnchorArm` seam
(`internal/ablate/anchor_observed.go`); `go test ./internal/ablate -run TestObservedAnchorArm` pins
the verdict *rule* — a directional `IS net-beneficial` claim is earned only when the conservative
floor is strictly positive — against a deterministic frozen fixture, decoupled from the mutable
ledger by design so the assertion never drifts as rows fold. The SHA-pinned command above ties the
*live numbers* to that rule; the test guards the *rule* itself.

## Methodology — why a *conservative floor*, and why that is the honest shape here

The clean A/B (`ablate.AnchorABSweep`, `anchor_ab.go`) prices **one session under both anchors** and
differences them. That needs a **counterfactual** — the same traffic replayed under FirstBP *and*
Head — and real traffic never carries one: a live session runs under whichever anchor is configured,
never both. The recorded window is **entirely the head arm** (605 fires; **0**
`compaction_anchor_starved` rows at the pinned SHA — no FirstBP-dormant arm to difference against). So the matched A/B
has no pairs to fold, and forcing single-arm rows into it would fabricate the missing arm.

What the ledger *does* witness cleanly:

- **Gross shed** (`compaction_saved_usd`) is exact — the dollars head-anchoring saved by firing.
- **Burst** (`write_premium_usd`) is only bounded: the fire-caused cache-write burst is a *subset*
  of the window total, which also includes cache writes no fire caused (initial priming, TTL refresh,
  non-fire turns).

Charging the head arm the **entire** window write premium is therefore a deliberate over-charge —
and FirstBP-idle would itself write cache, indeed re-prime the *larger* unshed prefix on TTL expiry,
so its own burst could equal or exceed head's. Netting the full premium out yields a floor that the
true net-of-burst benefit cannot fall below: `true ≥ gross_shed − window_write_premium = +$2,735.30`.
A positive floor is a hard directional win requiring no per-fire attribution; only a *non-positive*
floor would have been undistinguished and kicked the question to the attribution siblings.

## Generation-closure evidence

- **Promotion evidence (earned `net-beneficial`):** 605 real compaction fires over a full week of
  guarded-Claude traffic (pinned ledger `1c55b6668`), gross shed `+$3,115.77`, floor `+$2,735.30`
  positive even under the worst-case burst charge. The floor is not a knife-edge: it stayed strictly
  positive and *rose monotonically* across every committed ledger snapshot as the window filled —
  `+$216.22` @ 61 fires (07-04..07-06, `03e7b042a`), `+$332.86` @ 202 fires (..07-08, `a3b5b5935`),
  `+$2,684.25` @ 598 fires (`90c4cff06`), `+$2,735.30` @ 605 fires (..07-11, `1c55b6668`). The #1407
  de-starvation switch paid for its burst and then some, with widening margin, throughout.
- **Demotion / retirement evidence (what would flip it):** a future window whose window write
  premium exceeds the gross shed drives the floor `≤ 0` — the verdict then downgrades to
  `NOT DISTINGUISHABLE` and the question routes to per-fire burst attribution (#1490/#1072). A window
  with 0 fires (FirstBP dormancy, #1407) reads `UNWITNESSED`. Re-run the reproducing command against a
  later committed SHA (`git show <sha>:docs/nightrun/cache-savings.jsonl | python …`) to re-check
  against the then-current ledger.
- **Invalidating assumption:** the floor assumes the fire-caused burst is bounded by the total
  provider write premium and that shed is correctly priced at the input rate. If a future accounting
  change books fire bursts on the `compaction_shed` rows themselves (or splits 5m/1h creation tiers
  per fire), the window-total upper bound no longer holds and the arm must fold the per-fire burst
  directly — at which point the clean `AnchorABSweep` shape, fed a real replayed counterfactual,
  supersedes this conservative bound.

## Scope

In scope: the net-dollar verdict on `CompactAnchorFirstBP` vs `CompactAnchorHead` over real traffic.
Out of scope (unchanged): the anchoring strategies themselves, and the attribution / savings-audit
siblings (#2780, #1844, #745, #1490/#1072, #2393).
