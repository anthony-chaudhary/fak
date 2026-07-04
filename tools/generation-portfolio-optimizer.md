# Multi-Generation Portfolio Optimizer

A **portfolio-level** scorer that reads the labeled generation portfolio (every
open `generation`-labeled issue, partitioned by stream) and balances five named
axes — **now throughput, next-gen foundation, option value, dependency risk, and
stale debt** — into a per-item score *and* a per-stream allocation reading. It
answers the question the [#1625](https://github.com/anthony-chaudhary/fak/issues/1625)
epic opened and issue [#1652](https://github.com/anthony-chaudhary/fak/issues/1652)
names: once the portfolio is labeled, it can be **optimized instead of manually
sorted**. Closes [#1652](https://github.com/anthony-chaudhary/fak/issues/1652)
under epic [#1625](https://github.com/anthony-chaudhary/fak/issues/1625).

Generation stream: `gen/second-next`. This is an **architectural design memo +
lifecycle model** — the second-next proof bar (simulation / compatibility policy
/ a dependency edge with a demotion criterion), never default runtime exposure.
The runnable `fak` verb is named below as promotion evidence, not shipped here.

## Where this sits (keep the three scorers distinct)

This is the portfolio-grain sibling of two existing per-item artifacts and the
one existing per-lane metric; keep them distinct so they compose instead of
overlapping:

| Artifact | Question it answers | Grain |
|---|---|---|
| [Generation-Fit Grooming Score](generation-fit-grooming-checklist.md) (#1648) | *Is the generation label even right for this issue?* | one issue, at intake |
| [Generation Readiness Gates](../docs/notes/GENERATION-READINESS-GATES-2026-06-30.md) (#1644) | *Is the evidence strong enough to promote this item?* | one item, at promotion |
| `debt_score` in [`docs/generation.md`](../docs/generation.md#debt-metric) | *How much intake drift does one lane carry?* | one lane, at milestone report |
| **This optimizer (#1652)** | *Given the whole labeled portfolio, which mix should attention go to, and which items rebalance it?* | the whole portfolio, at dispatch planning |

The canonical stream / milestone / evidence / promotion-verb definitions this
optimizer scores against live in [`docs/generation.md`](../docs/generation.md);
this file is the checkable rubric, not a second source of truth. In particular
the **stale debt** axis (§4) is deliberately the *item-grain projection* of that
doc's lane-grain `debt_score`, so the two never disagree.

## Orthogonality (the optimizer changes none of the three axes)

The optimizer produces a *dispatch-attention* reading. It must never be read as
priority, a branch strategy, or a runtime exposure decision.

- **Priority.** A high or low portfolio score is not urgency. A `gen/future`
  item that the optimizer flags as *starved* is not thereby high-priority, and a
  `gen/now` item with a big `now_throughput` term is not thereby the most urgent
  thing in the repo. The score encodes **horizon balance**, not value: the whole
  point (§5) is to stop a greedy priority sort from starving the foundation and
  option horizons. Treating a low score as "deprioritize" — especially on
  `gen/future` — is the anti-pattern the epic exists to kill.
- **Shared trunk.** The optimizer reads issue metadata and emits a report. It
  authorizes no branch, no worktree, no stale-trunk exception. Every stream still
  lands through `main`, by explicit path, DCO-signed, with the `(fak <leaf>)`
  stamp. A high score never earns a side branch; a starvation flag never earns
  one either.
- **Runtime feature gates.** The optimizer scores *planning* metadata (labels,
  milestone, issue body, cheap sidecars), not runtime exposure. A `gen/next` item
  may ship inert behind a default-off gate and still score well on
  `nextgen_foundation`; that is correct posture, not a contradiction. The
  optimizer itself, per its `gen/second-next` posture, ships as an operator-only
  read (a milestone-report column or an explicit verb) — **never a default-on
  behavior that reorders dispatch on its own**. It informs the operator; it does
  not act.

## The five axes

For each item `i` in the labeled portfolio, compute five components. Three are
**assets** (they raise the score); two are **liabilities** (they lower it). Every
input is a cheap read — `gh issue view <n> --json labels,milestone,body,title`
plus the same commit/label sidecars `debt_score` already reads — so the whole
portfolio scores without cloning or rereading the parent epic.

| # | Axis | Sign | Cheap proxy (0..1) |
|---|---|---|---|
| 1 | `now_throughput` | asset | Value deliverable to the *current* release train. `1.0` for `gen/now` + `G0` milestone + acceptance naming a trunk witness (captured command/test); decays with horizon distance; `~0` for `gen/future` **by definition, not as a demerit** (the other axes price a future item). |
| 2 | `nextgen_foundation` | asset | How much nearer-horizon work depends on this. Proxy: normalized referenced-by count from nearer streams (a `Parent:`/`blocks`/`depends on #N` edge from a `now`/`next` item), capped at `1.0`. A named prerequisite of shipping work scores high even at a far horizon. |
| 3 | `option_value` | asset | The architectural optionality preserved — the value of keeping a door open. Proxy: `1.0` if the body names **both** a promotion path **and** a demotion criterion; halved if only one; `~0` if neither (an option you cannot price or close is a permanent park, not an asset). Decays toward 0 when no invalidating assumption is named — an unpriceable option is not option value. |
| 4 | `dependency_risk` | liability | Cross-generation dependency exposure. Proxy: `1.0` if the item asserts a cross-generation edge (ABI / wire schema / compat policy) with **no** demotion criterion; **halved to `0.5`** if the edge carries an explicit demotion/kill criterion (the additive-only test from [`docs/generation-abi-compatibility-policy.md`](../docs/generation-abi-compatibility-policy.md)); `0` if it declares no cross-gen edge. Mitigation, not elimination — the risk is real even when bounded. |
| 5 | `stale_debt` | liability | The drag the item adds sitting unpromoted / unwitnessed. **Item-grain projection of `debt_score`**: `3·missing_witness + 2·(label⇄milestone mismatch) + 2·(unpromoted later-horizon bet) + 1·stale-risk`, then normalized to `0..1`. One source of truth with [`docs/generation.md`](../docs/generation.md#debt-metric); if that metric's inputs change, this axis rebinds to them, never forks. |

### Per-item score

The horizon lives in the **weights**, not in a single global ranking — that is
what keeps the score orthogonal to priority (each stream values the same axis
differently):

```text
item_score(i) =
    W[stream].now   * now_throughput(i)      # asset
  + W[stream].found * nextgen_foundation(i)  # asset
  + W[stream].opt   * option_value(i)        # asset
  - W[stream].dep   * dependency_risk(i)     # liability
  - W[stream].debt  * stale_debt(i)          # liability

asset(i) = the three + terms ;  debt(i) = the two - terms (magnitude)
net(i)   = asset(i) - debt(i)
```

Default per-stream weights (priors, tunable — see the invalidating assumption):

| stream | `now` | `found` | `opt` | `dep` | `debt` |
|---|---|---|---|---|---|
| `gen/now` | 1.0 | 0.3 | 0.1 | 0.5 | 1.0 |
| `gen/next` | 0.5 | 1.0 | 0.4 | 0.7 | 0.8 |
| `gen/second-next` | 0.1 | 0.6 | 1.0 | 1.0 | 0.5 |
| `gen/future` | 0.0 | 0.3 | 1.0 | 0.3 | 0.3 |

Reading the table: `now` rewards throughput and punishes debt hardest; a
`second-next` bet is weighted toward option value and foundation but carries the
**heaviest dependency-risk weight**, because managing cross-generation dependency
*is* a second-next item's job; `future` rewards pure option value and barely
penalizes debt (a low-debt parked research memo is fine to hold).

## The portfolio view — optimize instead of sort

A per-item score is still just a sort, and a greedy sort is exactly the manual
behavior the issue's *Why* calls out: chase the highest `now_throughput` every
time and the foundation / option horizons starve. The **portfolio** behavior is
an allocation reading with two robust flags:

- **starvation** — a stream's clipped-positive net share `net⁺(stream) / Σ net⁺`
  falls below its floor. This fires when the greedy-now sort has crowded out a
  horizon: no foundation bet, no option bet in flight.
- **debt-drag** — a stream's `net(stream) < 0`: its stale-debt liability exceeds
  its asset sum. The stream is carrying more unwitnessed/stale drag than value;
  the optimizer names the worst items to `add-witness`, `park`, or `retire`.

Default target bands (share of active attention, tunable; **guidance minimums,
not quotas** — the operator may consciously waive a floor):

| stream | floor | ceiling |
|---|---|---|
| `gen/now` | 0.45 | 0.70 |
| `gen/next` | 0.15 | 0.35 |
| `gen/second-next` | 0.05 | 0.20 |
| `gen/future` | 0.02 | 0.10 |

Crowding (a stream over its ceiling) is emitted as a **soft advisory only**, not
a hard flag — with a small portfolio, per-stream shares are lumpy and a hard
crowding flag would false-fire. The two hard flags (starvation, debt-drag) are
the failure modes the issue cares about: a future starved by greedy-now, and
debt quietly accumulating in a stream.

## Machine-readable schema

An agent or a future `fak` verb emits one object per portfolio scoring. This is
the machine form of the tables above; a report column or `issue_triage` scope can
produce and consume it without prose parsing.

```json
{
  "schema": "fak-generation-portfolio/1",
  "streams": {
    "now":         {"items": 12, "asset": 9.40, "debt": 2.10, "net": 7.30, "share": 0.56, "band": [0.45, 0.70], "state": "healthy"},
    "next":        {"items": 7,  "asset": 5.10, "debt": 1.80, "net": 3.30, "share": 0.25, "band": [0.15, 0.35], "state": "healthy"},
    "second_next": {"items": 4,  "asset": 3.00, "debt": 0.60, "net": 2.40, "share": 0.18, "band": [0.05, 0.20], "state": "healthy"},
    "future":      {"items": 1,  "asset": 0.20, "debt": 0.05, "net": 0.15, "share": 0.01, "band": [0.02, 0.10], "state": "starved"}
  },
  "flags": [
    {"stream": "future", "kind": "starvation", "detail": "future net+ share 0.01 below floor 0.02 — no live research/option bet"}
  ],
  "items": [
    {"issue": 1652, "stream": "second_next",
     "now_throughput": 0.1, "nextgen_foundation": 0.6, "option_value": 1.0,
     "dependency_risk": 0.5, "stale_debt": 0.0,
     "asset": 1.37, "debt": 0.50, "net": 0.87, "suggested_verb": null}
  ]
}
```

`state ∈ {healthy, starved, debt_drag}`. `suggested_verb` on an item is one of
`add-witness | promote | demote | retire | park | null`, chosen from the
dominant liability (missing witness → `add-witness`; net-negative later-horizon
bet with a failed assumption → `demote`/`retire`).

### Field bindings (no epic reread)

| Axis | GitHub / repo source | Rule |
|---|---|---|
| `now_throughput` | `.labels`, `.milestone`, `.body` "Acceptance"/"Witness" | `gen/now` + `G0` + a named trunk witness ⇒ high; decays with horizon. |
| `nextgen_foundation` | `.body` `Parent:`/`depends on #N`; reverse refs from nearer streams | referenced-by count from `now`/`next`, normalized. |
| `option_value` | `.body` promotion path + demotion criterion + invalidating assumption | both present ⇒ 1.0; one ⇒ 0.5; none ⇒ ~0. |
| `dependency_risk` | `.body` cross-gen edge (ABI/schema/compat) + demotion criterion | edge without criterion ⇒ 1.0; with ⇒ 0.5; none ⇒ 0. |
| `stale_debt` | same sidecars as `debt_score` | item-grain `3·missing_witness + 2·mismatch + 2·unpromoted + 1·stale`, normalized. |

## Worked before/after readout (the planning witness)

A hand-computed four-to-five-item portfolio shows the flags firing then clearing
— the before/after the issue's *Witness* section asks a planning artifact for.

**Before — a greedy-now portfolio (starvation + debt-drag fire).** Three items,
all `gen/now`, no foundation or option bet in flight; one item is missing its
witness.

```text
I1 gen/now   now=1.0 found=0.2                     asset=1.06 debt=0.00 net= 1.06
I2 gen/now   now=1.0                               asset=1.00 debt=0.00 net= 1.00
I3 gen/now   now=0.8            debt=1.0(no wit.)   asset=0.80 debt=1.00 net=-0.20  -> add-witness

streams (net+ share of Σnet+ = 2.06):
  now          net+ 2.06  share 1.00   [.45,.70]  advisory: over ceiling (crowded)
  next         net+ 0.00  share 0.00   [.15,.35]  FLAG starvation
  second_next  net+ 0.00  share 0.00   [.05,.20]  FLAG starvation
  future       net+ 0.00  share 0.00   [.02,.10]  FLAG starvation
reading: 3 starvation flags — the portfolio is greedily allocated to now;
         no foundation or option bet in flight. I3 is net-negative (no witness).
```

**After — operator witnesses I3 and adds two horizon bets** (a `gen/next`
foundation seam I5, and this optimizer #1652 itself as the `gen/second-next`
bet I4):

```text
I1 gen/now         now=1.0 found=0.2                          net= 1.06
I2 gen/now         now=1.0                                    net= 1.00
I3 gen/now         now=0.8            (witnessed, debt=0)      net= 0.80
I5 gen/next        now=0.4 found=1.0 opt=0.3 dep=0.2          asset=1.32 debt=0.14 net= 1.18
I4 gen/2nd (#1652) now=0.1 found=0.6 opt=1.0 dep=0.5(mitig.)  asset=1.37 debt=0.50 net= 0.87

streams (net+ share of Σnet+ = 4.91):
  now          net+ 2.86  share 0.58   [.45,.70]  healthy
  next         net+ 1.18  share 0.24   [.15,.35]  healthy
  second_next  net+ 0.87  share 0.18   [.05,.20]  healthy
  future       net+ 0.00  share 0.00   [.02,.10]  FLAG starvation (still no future bet)
reading: now-crowding + next & second-next starvation CLEARED; no net-negative
         items remain. One honest flag stands: future is still starved — a real
         signal the operator can accept (floor is guidance) or address next.
```

Before the optimizer, "is the portfolio balanced?" was an operator hunch over a
manual sort; after it, the same five axes, per-stream weights, and two flag rules
apply to every portfolio, and the imbalance surfaces as a named flag plus the
specific items that rebalance it — the greedy-now failure the issue's *Why*
describes becomes a fired flag instead of a silent drift.

## Promotion / demotion / assumption (for this artifact)

- **Promotion evidence** (what moves this design memo toward `gen/now`): wire the
  five axes and two flags into a runnable surface — `fak generation portfolio`
  (pure logic in `internal/genportfolio/`, thin shell in `cmd/fak/`, per the
  Go-not-Python rule) or a `fak milestone report` column — that reads
  `gh issue list --json labels,milestone,body` and emits the
  `fak-generation-portfolio/1` object, plus **one captured dogfood readout** over
  the live open `generation` lanes. A green contract test over the worked example
  above (its arithmetic is the fixture) *plus* that readout is the promotion
  witness. Until then this stays a `gen/second-next` planning model, applied by
  hand or by a planning agent.
- **Demotion / retirement evidence**: retire this optimizer if the five axes stop
  mapping to real dispatch decisions — if operators never act on a starvation or
  debt-drag flag, an unread optimizer is decorative and should be removed, not
  defended. Demote (park) it if `debt_score` (lane grain) and the grooming score
  (issue grain) already drive every allocation call and this portfolio grain adds
  no signal a milestone report lacks. Fold-and-delete if a future `fak` verb
  subsumes the rubric into its help text.
- **Invalidating assumption**: the optimizer assumes the **labeled portfolio is
  complete and truthful** — that every real generation bet is a labeled issue and
  that `option_value` / `dependency_risk` are legible from issue bodies. If
  unlabeled work dominates the real portfolio, or if those two axes migrate to a
  project field or into code where the cheap read cannot see them, the optimizer
  scores a biased subset and must be rebound to whatever surface then carries the
  signal. A second, sharper assumption: the **weight table and the bands are
  hand-set priors, not fit to outcomes**. If realized promotions and regressions
  do not correlate with the score, recalibrate the weights against realized
  throughput and retired assumptions — do not defend the priors.

## Continue here

A future agent needs no epic reread. To advance #1652's follow-on:

1. Implement `fak generation portfolio` — pure logic in `internal/genportfolio/`,
   thin shell in `cmd/fak/generation_portfolio.go` — that reads
   `gh issue list --state open --label generation --json number,labels,milestone,body`,
   computes the five axes per the field-binding table, applies the default
   per-stream weights, and emits `fak-generation-portfolio/1`.
2. Gate it with a contract test that reproduces the worked before/after readout
   exactly (the numbers above are the fixture) and asserts the two flags fire and
   clear as shown.
3. Keep the **compatibility edge with its demotion criterion**: the `stale_debt`
   axis MUST remain the item-grain projection of `debt_score` in
   [`docs/generation.md`](../docs/generation.md#debt-metric). If that metric's
   inputs change, rebind this axis to them in the same pass — the demotion
   criterion for this whole optimizer is "the axes no longer map to the canonical
   `docs/generation.md` definitions," which retires it rather than letting it
   drift into a second source of truth.
4. Capture one dogfood readout over the live open `generation` lanes; that
   readout plus the green contract test is the promotion witness named above.
