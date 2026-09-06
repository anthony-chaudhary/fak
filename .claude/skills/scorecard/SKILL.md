---
name: scorecard
description: "The generic scoring doctrine the whole fak scorecard family instantiates — how to BUILD a new deterministic, tree-cross-checked, debt-driving scorecard and how to RUN any of them as a repeatable RSI pass. Every sibling (code-quality, docs, doc-appeal, seo, demo-quality, repo-hygiene, observability, learning, industry, agent-readiness, product, persona) is the same machine pointed at a different surface: pure KPIs over a data-dir or the git-tracked tree, cross-checked against reality so the score can't be gamed by editing data, folded into one *-debt. Use when this named workflow matches the task."
metadata:
  opencode: claude-only
---

# scorecard — the generic scoring doctrine (build one, run any)

> **What this is.** fak measures itself with a *family* of scorecards, one per
> surface a reviewer cares about. They look different on the outside but they are the
> **same machine**: a deterministic measurement that turns "is this surface getting
> better or worse" into one integer you can move and prove you moved. This skill is
> the shared doctrine — the anatomy every scorecard follows, the RSI loop every
> scorecard skill runs, and the recipe for adding a new one. The per-surface skills
> (`quality-score`, `appeal-score`, `industry-score`, `agent-readiness`,
> `product-score`, `persona-score`) are instances; this is the template they share.

The whole idea in one line: **a scorecard reads reality, folds it into a `*-debt`
integer the cross-check won't let you fake, and a paired skill drives that integer
down by adding the real thing — never by gaming the detector.**

## Raw debt is an unbounded discovery ledger

Every `<surface>_debt` value is an **unbounded non-negative integer**: the exhaustive
count of all currently detected HARD defects and coverage gaps in the named corpus
under a named metric version. “Unbounded” means the schema and process impose no
maximum; a finite repository still produces a finite count.

The 0–100 score and A–F grade are separate, bounded presentation signals. They must
never replace, clamp, normalize away, sample, or top-N-truncate raw debt. A renderer
may preview a work-list only if the machine payload still emits every defect and the
preview labels both its limit and the full count. Zero means “metric version V found
no debt in corpus C now,” not “no debt exists” or “stop discovering.” When a new check
finds more debt, bump the metric version and preserve the prior snapshot so same-version
improvement and cross-version discovery stay distinguishable.

---

## Discover the live family; do not hard-code a roster

The registry printed by the first-class Go front door is the truth source:

```bash
fak score --help
```

Select the scorecard named by the task from that live registry and capture its machine-readable
result before editing:

```bash
fak score <name> --json > <name>-before.json
```

Retire binding FAIL debt before advisory polish.

Do not preserve a fixed list in this skill: the registry grows. If a registered scorecard lacks
`--json`, record that as a concrete gap rather than silently omitting it. The historical
`python tools/scorecard_control_pane.py` path is a generation/debug fallback only; before trusting
its generated pane, compare its discovered inputs with `fak score --help`.

## The five laws every scorecard obeys

1. **Deterministic.** Two clones at one commit score *identically*. No clock, no
   network, no randomness in the score. Read-only over the data; the only writes are
   a generated doc folder under `--markdown-dir`/`--markdown`.

2. **Cross-checked against reality — ungameable by editing data.** A data-driven
   scorecard (industry / product / persona) keeps its rows in a JSON data dir, but
   **every claim in a row is verified against the real tree**: a path that must
   exist, a CLAIMS tag the concept actually carries, a command whose `./cmd/<dir>`
   resolves, a doc that mentions a token. So you cannot drop debt by editing the
   data — you fix the real thing. A tree-reading scorecard (code / agent-readiness)
   reads the git-tracked tree directly, which is the same property by construction.

3. **One headline `*-debt` integer.** The count of concrete, re-derivable HARD
   defects (plus coverage gaps, for the catalog scorecards). Driving it to zero is
   the goal; the `*-debt` is what folds into the control pane. **SOFT** signals lower
   the score but are *never* debt — they're judgment nudges, not work-list items.

4. **A pure core, an impure shell.** The KPIs are pure functions over facts
   (`kpi_*(...) -> {kpi, group, score, detail, defects, soft}`); a thin disk/git
   shell gathers the facts and calls them. This is what makes the core unit-testable
   with fixtures and the live tree a single smoke test.

5. **A control-pane payload.** Every scorecard's `--json` emits the same envelope so
   the fold and any loop runner can read it uniformly:

   ```json
   { "schema": "...", "ok": false, "verdict": "ACTION", "finding": "...",
     "reason": "...", "next_action": "...",
     "corpus": { "score": 0, "grade": "A", "<surface>_debt": 0, ... },
     "kpis": [ ... ] }
   ```

   The control pane reads `corpus.<debt>` and `corpus.grade`; keep those keys.

---

## Default portfolio contract for scoreboard-debt work

When a task names **scoreboard debt**, **scorecard debt**, score improvement, or
a 2×/50% target, the individual card is not the whole acceptance surface. Run
the canonical portfolio pane before edits and after the final candidate:

```bash
python tools/scorecard_control_pane.py --json > <allocated-scratch>/pane-before.json
# run the owning leaf scorecard with the same workspace/corpus scope
python tools/scorecard_control_pane.py --json > <allocated-scratch>/pane-after.json
```

Treat the receipts as comparable only when workspace, corpus, and detector
versions match. The pane's `index_coverage` is a completion gate:
`index_coverage.ok` must be true, `unindexed_count` must be zero, and each
excluded implementation must carry a production-owned reason. A card absent
from default reporting is unfinished integration, not an improved card.

Report both the owning card's leaf-debt delta and the pane's `total_debt` delta.
For a 2× / 50% reduction, prove `debt_after <= debt_before / 2` against the
declared authoritative baseline. A grade change, narrower corpus, or stale
baseline comparison does not prove 2×.

Preserve the measuring stick. Detector logic, discovery, exclusions, weights,
thresholds, and the pinned baseline cannot count as debt retirement unless the
issue explicitly requests a separately reviewed measurement migration. Never
`--pin` merely to make residual debt or a regression green. Store same-scope
before/after receipts for substantive cohorts and bind them to the shipped
commit or issue witness.

## Running one as an RSI pass (the loop every scorecard skill shares)

Whichever surface you're on, the pass is the same five steps:

1. **Run it** — `python tools/<x>_scorecard.py` builds the work-list; `--json` is the
   machine payload; `--critical` / `--gaps` (catalog scorecards) rank the backlog.
2. **Retire `*-debt` worst-first** — fix the heaviest KPI (or worst-served row)
   first, by **adding the real affordance / writing the real test / correcting the
   real overclaim**. Never weaken a claim, a guard, or a verdict to score.
3. **Weigh the SOFT signals, then stop** — fix the cheap real ones; don't chase them
   to zero. A token added only to move a metric is the gaming this refuses.
4. **Re-measure + prove** — `--compare baseline.json` prints the debt delta and the
   2×/3× verdict; regenerate the committed snapshot so the doc matches the tree.
5. **Commit only the scorecard lane, by explicit path** — run `fak sync check` (or `fak sync reconcile --apply`), then commit via `fak commit --path`:
   `fak commit --path tools/<x>_scorecard.py --path tools/<x>_scorecard_test.py --path <data dir> --path <doc dir> -F msg [--push]`. Never
   `git add -A`. Push via `fak sync push`. End the subject with the `(fak <leaf>)` trailer.

---

## The clean read (measure the committed tree, not the working-tree dirt)

A scorecard number is only trustworthy if two people at the same commit get the
same number. On a shared, multi-session tree that is **not** automatic — a
working tree carries untracked scratch, nested mirror checkouts
(`.fak/tmp/…`, `.dos/_dos_park/_iso_build/…`), mangled root paths a prior session
dumped, and your own half-finished WIP. Two pollution classes can corrupt a read:

1. **File-enumeration pollution** — scratch / mirror / mangled paths get counted,
   inflating the occurrence-counters (slop, disambiguation) many-fold.
2. **Build-break pollution** — uncommitted WIP that doesn't compile makes every
   Go-backed card (`go run ./cmd/fak …`) error and drop out of the fold, so a
   regression there can't be caught.

**The canonical clean read — run from the repo ROOT, no extra checkout needed:**

```
python tools/scorecard_control_pane.py            # the folded portfolio read
python tools/scorecard_control_pane.py --json      # machine payload
```

Class (1) is handled **in-place, by construction** — the occurrence-counters
report the committed floor on a dirty tree just as on a clean one, via one of two
immunity mechanisms:

- **`git ls-files`** (the strongest, used by `code-slop`, `repo-hygiene`,
  `agent-readiness`, `stability`, …): enumerate tracked paths directly and read
  working-tree bytes. An untracked path is *structurally unreachable*.
- **Tracked-subtree scoping** (used by `concept-disambiguation`, …): walk only the
  source subtrees (`internal/`, `cmd/`, `docs/`) and skip scratch dirs (`.git`,
  caches, `vendor`, `.dispatch-runs`, `.goal-runs`), so a nested mirror checkout at
  `.fak/…` or `.dos/…` is never descended into.

(This is why the fix for a polluted read is to make the *measurement* immune,
**never** to delete the untracked junk by hand — it may be a peer's scratch on a
shared tree.) When you build a NEW tree-reading scorecard, prefer **`git
ls-files`** over a root `rglob` — it is immune by construction, not by remembering
to exclude every scratch dir the fleet invents.

Class (2) has one requirement: **the tree must COMPILE.** If your working tree
has WIP that doesn't build, either commit/stash it first, or measure a pristine
checkout of HEAD that **keeps `.git`** (every git-based card needs it):

```
git worktree add --detach /path/to/clean HEAD     # or: git clone --local . /path/to/clean
python tools/scorecard_control_pane.py --workspace /path/to/clean
```

Do **not** use `git archive HEAD | tar -x` for this: it strips `.git`, so
`repo-hygiene` and the churn-based cards error on the missing repo.

### Regression note — a build break is not a card bug

When a Go-backed card errors, the control pane now says so directly (it tags the
errored Go cards and points here). To triage by hand:

- Run **`go build ./...`**. If it **FAILS**, the errored cards are a working-tree
  **build break** — commit/stash your WIP or measure a clean HEAD checkout; the
  cards are fine.
- If `go build ./...` **PASSES** but a card still errors, it is a **real card
  bug** — debug that card's `--json` directly.

The live-smoke test `test_live_collect_and_fold` asserts **zero** errored cards on
the real tree, so a genuine card breakage reds CI; this note keeps a transient WIP
build-break from being mistaken for one.

---

## Building a NEW scorecard (the recipe)

When a surface isn't measured yet, add one. The fastest path is to copy the closest
existing instance and re-point it: **`product_scorecard.py` / `persona_readiness_scorecard.py`**
for a *catalog* (data-dir rows cross-checked against the tree), **`internal/agentreadinessscore`**
for a *tree-reading* scorecard (no data dir).

1. **Pick the shape.** Catalog (a roster of rows that evolves — concepts, personas,
   competitors) → data dir + coverage. Tree-reading (a fixed set of affordances the
   tree must have) → no data dir.

2. **Define the KPIs as pure functions.** Each returns
   `{kpi, group, score (0-100), detail, defects: [str], soft: [str]}`. Prefix every
   defect string with the row/area id (`"<id>: …"`) so per-row debt is recoverable.
   Group the KPIs (e.g. well-formed / reality / honesty) and weight them.

3. **Cross-check against the real tree.** This is the ungameable part — load tree
   facts once (paths that exist, CLAIMS tags, cmd dirs, documented verbs, doc text)
   and have each KPI verify the data against them. If a check can pass by editing
   only the data file, it's not a real check.

4. **Fold to a payload.** Compute the weighted composite, the `A–F` grade
   (`grade_letter`), the `<surface>_debt` unbounded integer (exhaustive sum of HARD
   defects + coverage gaps, never clamped or truncated), and emit the control-pane envelope with `corpus.<surface>_debt` and
   `corpus.grade`.

5. **Add renderers + flags.** `render` (terminal work-list), `--json`, `--compare
   baseline.json` (prove the drop), and a doc generator (`--markdown` or
   `--markdown-dir`). For catalog scorecards add `--chart` / `--critical` / `--gaps`.

6. **Write the test.** Fixtures for each KPI's defect trigger AND its clean case, the
   fold to `*-debt`, plus a **live smoke** asserting a named metric-version/corpus
   result. If it is zero, call it the current detector floor, not proof of completion.

7. **Wire it into the control pane + re-pin.** Add a row to `SCORECARDS` in
   `tools/scorecard_control_pane.py` binding `{key, debt, script, label}`. Adding a
   scorecard raises the portfolio total, so **re-pin the baseline**:
   `python tools/scorecard_control_pane.py --pin` (and commit
   `tools/scorecard_baseline.json` in the same lane). Without the re-pin the `--check`
   ratchet reads the new debt as a regression.

8. **Write the paired RSI skill** under `.claude/skills/<x>-score/SKILL.md` — the
   five-step loop above, the one anti-gaming rule, and the commit-by-path discipline.
   It is an instance of *this* doctrine; link back here.

---

## The anti-gaming law (the thing that makes a scorecard worth anything)

A scorecard is only as honest as its cross-check. The single rule that protects every
instance: **retire a defect by changing reality, never by changing the detector.** A
missing-affordance defect is fixed by adding the affordance, not the keyword; an
overclaim is fixed by correcting the claim, not by relaxing the check; an untested
package is fixed by a real test, not a stubbed one. If "fixing" a defect would mean
gaming the substring match instead of improving the surface, **stop — that's not a
real gap**, and weakening the check to make it green is the one move that turns the
whole family into theater.

## The growth-invariant KPI shape (when "stays constant as it grows" is the goal)

Most scorecards measure an *absolute* surface and report a **count** — and that count
mechanically WORSENS as the repo grows (a 3×-bigger tree has ~3× the surfaces, so the
raw defect count climbs even when discipline is unchanged). That is correct for "how
clean is the tree right now," but WRONG when the goal is "does effort stay *flat* as we
grow" (steerability, or any future "constant-as-it-scales" surface). For that goal:

1. **Every KPI is a ratio, density, or distribution percentile — never a raw count.** A
   2×-larger surface with the same discipline must score *identically*. A percentile is
   definitionally scale-free; a rate (offenders / total) is scale-free; a Gini is
   scale-free in value. A raw count is not — drop it.

2. **The headline is a 0–100 INDEX (a weighted mean of the invariant KPIs), not a debt
   pile.** A debt count can't answer "is effort flat" because it trends with size. The
   index can. You still emit a `*-debt` integer for control-pane membership, but —

3. **A defect is emitted only when an invariant rate crosses a FIXED threshold.** The
   *score* is the rate; the *debt* is the count of threshold crossings. So the debt
   stays orthogonal to size: a clean small tree and a clean 10×-bigger tree both emit 0.

4. **DROP any signal that is secretly size-coupled.** Some ratios still trend with
   growth: a *spread* (files-per-package) widens as packages mature; a *mean*
   (LOC-per-package) drifts as the denominator grows non-linearly; a *denominator-coupled
   share* (one package's fan-in / total packages) falls "for free" as the repo grows. If
   it moves from growth alone with no change in discipline, it is SOFT (advisory) or
   dropped — never a gate.

5. **Stay orthogonal to count-based siblings in the shared portfolio.** If another
   scorecard already emits HARD debt for a defect class (code_quality owns god-files and
   tests), do NOT re-emit it — score it on the invariant rate (SOFT) and let the sibling
   own the count. Re-emitting double-counts the same defect in `total_debt`.

`steerability` is the reference instance: its only HARD KPIs are the two whose cheapest
fix is genuinely real (split a dispatch monolith; commit the ratchet) — both 0 on a
disciplined tree — and its value lives in the index + the SOFT drift signals.

---

Keep this skill refined as the method evolves: when a new scorecard teaches a better
KPI shape, a sharper cross-check, or a new failure mode (a SOFT signal that should be
HARD, a debt that double-counts), fold the lesson back here so the next scorecard
starts from it.
