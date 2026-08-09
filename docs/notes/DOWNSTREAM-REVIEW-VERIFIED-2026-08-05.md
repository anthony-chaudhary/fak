# Verifying the downstream-adopter review — three gaps that were not gaps, and the one that is

**Date:** 2026-08-05 · **Verifies:**
[`DOWNSTREAM-FAK-ADOPTION-REVIEW-2026-08-05.md`](DOWNSTREAM-FAK-ADOPTION-REVIEW-2026-08-05.md)
(landed the same day as `bbcc4cbf14`)

**Why this note exists.** That review ranked fifteen public-safe imports from a contract-delivered
repository that adopted fak's guard/lane/worktree/shipgate concepts, and closed with five *"next
checkable steps"* — its own admission that the findings were **proposals, not verdicts**. This note
runs the first three of those steps and reports what they returned. Three of its headline
recommendations turn out to be **already implemented in fak**, in the one file the review did not
read. One survives, and gets sharper. Two of its open questions are now settled with measurement.

⛔ **The correction matters more than the additions.** A review that names a mechanism fak already
has spends a reader's attention on work that is done, and — worse — invites someone to build a
second copy of it. Everything below is stated against a file and a line so the next reader can
re-derive it rather than inherit it.

**Public-safety boundary is unchanged** from the reviewed note and restated in § 6: mechanisms and
doctrine only, no client identity, no domain, no engagement specifics, no absolute cost totals.

---

## 1. Three gaps that are not gaps — the review read the producers and the data, never the consumer

The review assessed fak's scorecard surface by reading the ~28 `tools/*_scorecard.py` producers and
`tools/scorecard_baseline.json` (the pinned data). It never read
[`tools/scorecard_control_pane.py`](../../tools/scorecard_control_pane.py) — the **fold** that turns
those cards into a verdict. Every discipline it proposed adding is already there.

| Review's claim | What is actually at HEAD | Verdict |
|---|---|---|
| §A1 *"a zero-for-unmeasured convention is fak's densest risk surface"* | `metric_from_payload` sets **`"debt": None`** on any error (`:442`), and also when the payload lacks the debt key (`:464`, `error: "missing <key> in payload"`). The fold then partitions on **type**: `measured = [m for m in metrics if isinstance(m.get("debt"), int)]` (`:552`) and `total_debt = sum(...)` over `measured` only (`:554`). **An unmeasured card cannot enter the sum as `0`.** | ⛔ **not a gap** |
| §A2 *"fak's gates do not print their candidate count / have no could-not-run code"* | The pane renders **`(N measured, M errored)`** beside every total, and the ratchet returns **exit `2`** — `RATCHET UNPINNED: no baseline to ratchet against` (`:992`) — distinct from `1` for a real regression. | ⛔ **not a gap** for the pane (the review's separate per-package "convert the admission into a test" point stands on its own) |
| §A4 *"what fak does not have is the floor/objective partition"* | Two independent no-regression axes, both hard by default. `if int(payload.get("errored", 0)) > 0: return 1, "RATCHET FAIL: N scorecard(s) unmeasured"` (`:984`) — **fak reds the gate on an unmeasured probe, where the downstream loop merely routes to a supervisor.** Plus the grade ratchet, which reds on a per-metric letter drop *"EVEN WHEN the raw-unit total held — the regression a flat `total_debt` would otherwise hide"*, and a `WARN early-warning` line for a metric that **"rose … vs baseline — hidden under a green portfolio"** (`:954`). | ⛔ **overstated** — fak has a *uniform* no-regression rule rather than a designated floor set |

⭐ fak also carries a piece of discipline the downstream repo has no equivalent of:
`build_break_hint` distinguishes *"your uncommitted WIP does not compile, so every Go-backed card
errored at once"* from *"this card is broken"* — differential attribution at the instrument layer,
the same instinct as `cmd/fak/trunk_red_ledger.go`.

## 2. What survives — and it unifies the two rules that looked separate

The review's §A3 (`probes_trusted` — a selftest that builds a document designed to trip each check
and fails if the check stays quiet) is **a real gap**, and running it down shows it is the same gap
as the half of §A4 that survives:

> **Both of fak's ratchet axes read the numerator and neither reads the domain.**
> `total_debt` and `grade_debt` are the only two things `compute_trend` compares against the pin
> (`:656–674`), and there is **no occurrence of `candidates`, `denominator`, `domain`, `scanned`,
> `examined`, or any corpus-size concept anywhere in the control pane** (grepped).

Two different failures therefore arrive wearing the same green:

- **A card that goes quiet.** Its detector stops matching, it exits 0, it honestly reports `0`. That
  is an `int`, so it is *measured*, so it enters the sum and **lowers** `total_debt`. The ratchet
  reads an improvement. This is exactly the case `debt: None` was built to catch and cannot: the
  pane is sound against a card that **crashes** and blind to a card that **goes quiet**.
- **A card whose domain was deleted.** The downstream framing — *"against a single blended total the
  cheapest way to reduce debt is to delete the citation, the link, or the page the metric counts"* —
  passes **both** fak axes, because deleting counted things lowers raw debt and cannot lower a
  grade. Neither ratchet can see that the corpus shrank.

**Why this is not hypothetical here.** `tools/scorecard_baseline.json` pins 44 metrics; **24 of them
(54%) sit at exactly `0`**, and `total_debt` is their plain sum (604 = sum of metrics, checked).
More than half the board is already in the state where *quiet* and *clean* are the same reading.

**The landing, and it is one change not two.** Have each card emit the size of the domain it
quantified over alongside its debt, and give the ratchet a third axis that **floors coverage**: a
card whose candidate count falls is a regression even when its debt falls. That single denominator
closes the quiet-probe hole and the deleted-domain hole together, and it is the smallest thing that
makes `probes_trusted` mechanical rather than aspirational.

## 3. A real defect, found downstream and fixed here — multi-line lane trees

The review's §C7 reported that `fak orient` misreports the lane for a repo whose `dos.toml` wraps
its lane trees across lines, and could not confirm it: *"`internal/devindex/orient.go:124`
`splitLaneTree` splits on **comma**, so the newline case needs a direct test."*

⛔ **That looked at the wrong function.** `splitLaneTree` splits `leaf.Tree`, a string that
`parseLanes` itself built with `strings.Join(globs, ", ")` — it can never see a newline. The TOML
scanner is `parseLanes`, and the bug is real:

- `parseLanes` is a line scanner, so the multi-line spelling puts a lane's **name** and its **globs**
  on different lines.
- The name alone reaches `c.declared[name] = true` (`internal/devindex/devindex.go:361`) — which is
  what every *"is this lane real?"* check consults — and the glob parse then hits
  `if len(globs) == 0 { continue }` (`:381`) before the leaf is appended.
- Net effect: **the lane validates, contributes no prefixes and no exact entries, and
  `LaneForPath` falls through to `unknown` for every path it owns.** Nothing errors, no gate reds.

Pinned by `internal/devindex/lanes_multiline_test.go`, which fails on the pre-fix parser with
`multi-line lane 'docsroot' produced no leaf; got [gateway tools]`, and fixed by `joinLaneArrays`
(`:284`), which folds a multi-line array to the one-line shape the scanner expects before it runs.

⭐ **The general shape is the reason to keep this note.** fak's own `dos.toml` writes every tree
inline, so **fak's tree is structurally incapable of exercising this path** — the bug was invisible
here and obvious downstream. That is the recurring value of an adopter: they hold the configurations
you do not.

⚠️ The **other** half of §C7 is confirmed and unfixed: `cmd/fak/orient.go:49` reads
`refs/fak/locks` via `internal/leaseref` (`:92`), while DOS journals to `.dos/lane-journal.jsonl`.
**Two lease substrates in one workspace, and neither reader can see the other's holders** — so
`fak orient --leases` prints `none` for an actively held DOS lane. Documenting them as disjoint is
the minimum; bridging the reader is the fix.

## 4. The lease-liveness contradiction, settled

The review flagged that its §C5 (*"never judge a lease's liveness by its `pid`"*) contradicted a
standing note advising the opposite, and left it as step 2. Measured against
`.dos/lane-journal.jsonl` in this checkout — 1,922 records, ops `ENFORCE 1777 / ACQUIRE 60 /
REFUSE 43 / RELEASE 42`:

- ⛔ **The downstream repo is right, and the reason is checkable without taking a lease.** A single
  holder journals *several distinct pids* — `fable-superloop` → 3, `goal-5046` → 3, and eight more
  holders → 2 each. A long-lived agent session would journal one. **The pid is the per-invocation
  CLI child**, so it reads dead for every lease including a fresh one, and `dead pid ⇒ stale lease`
  fires on all of them. The standing note reached the right verdict on a 9-day-old lease for the
  wrong reason and would have misfired on a live one. It has been corrected.
- ⭐ **But a sound check is available and unused by either side.** Every lease row carries
  **`lease.proc_starttime` alongside `pid` — 60 of 60**. pid + start-time is a precise process
  identity that survives pid reuse, which is strictly better than the downstream fallback of judging
  by age. Nothing reads it.
- **The wedge is larger than either note said.** `ttl_minutes` is `null` on **60 of 60** acquires,
  and 60 ACQUIRE against 42 RELEASE leaves **18 leases never released** across 32 lanes. The newest
  acquire is 2026-07-27 — every wedge is over a week old.

## 5. The method lesson, which is the transferable part

The first pass declared four gaps by reading the **producers** (`tools/*_scorecard.py`) and the
**data** (`scorecard_baseline.json`), and never the **consumer** (`scorecard_control_pane.py`). All
the discipline lived in the consumer. This is the same failure the downstream repo names in its own
vocabulary and is worth stating in fak's:

> ⭐ **A gap claim quantifies over a search domain, so it inherits every rule about domains.** *"fak
> has no equivalent"* is `∀x∈D ¬P(x)`, and it is vacuously easy to satisfy by choosing a small `D`.
> A negative finding must state where it looked, exactly as a clean gate must print what it
> quantified over.

⇒ The operational rule, and it costs one line per finding: **an absence claim names the files it
searched.** Every negative in this note does.

## 6. What was filtered out

Unchanged from the reviewed note, applied as a hard filter: client identity, the engagement, the
product domain and vendor stack, internal ids and wave names, and absolute cost totals tied to the
client's account are all removed. Nothing above depends on them. The measured figures retained here
are all from **fak's own tree and journal**, not the downstream repo's.

## 7. Next checkable steps

1. **Land the coverage axis (§2).** One denominator per card, floored by the ratchet. Closes the
   quiet-probe and deleted-domain holes together; unblocks `probes_trusted` as a mechanical check
   rather than a convention. Highest value remaining in either note.
2. **Decide the two lease substrates (§3).** Unify, bridge, or document `refs/fak/locks` vs
   `.dos/lane-journal.jsonl` as disjoint — but not leave `fak orient --leases` printing `none` over
   a held lane.
3. **Reap on pid + `proc_starttime` (§4).** The materials are journaled on 60/60 leases; 18 lanes
   are wedged for want of a reader. Pair with a TTL, since `ttl_minutes` is `null` on every acquire.
4. **Steps 4 and 5 of the reviewed note are untouched here** — the ordinal landing deadline into
   `tools/issue_worker_prompt.py`, and filing the guard-result-shape gap (a refusal-terminated
   session is indistinguishable from a clean one). Both still stand as written.
