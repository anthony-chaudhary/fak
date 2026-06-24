---
name: scorecard
description: The generic scoring doctrine the whole fak scorecard family instantiates — how to BUILD a new deterministic, tree-cross-checked, debt-driving scorecard and how to RUN any of them as a repeatable RSI pass. Every sibling (code-quality, docs, doc-appeal, seo, demo-quality, repo-hygiene, observability, learning, industry, agent-readiness, product, persona) is the same machine pointed at a different surface: pure KPIs over a data-dir or the git-tracked tree, cross-checked against reality so the score can't be gamed by editing data, folded into one *-debt integer + an A–F grade, emitting a control-pane payload, retired worst-first by ADDING the real thing, and proven with a before/after compare. Use when adding a NEW scorecard for a surface that isn't measured yet, when refining the scoring method itself, or when you need the shared contract behind quality-score / appeal-score / industry-score / agent-readiness / product-score / persona-score.
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

---

## The family (each the same machine, a different surface)

| Scorecard | Tool | Debt key | Surface it measures |
|---|---|---|---|
| code-quality | `tools/code_quality_scorecard.py` | `code_debt` | the Go module (gofmt, tests, god-files) |
| docs | `tools/docs_scorecard.py` | `doc_debt` | the doc corpus |
| doc-appeal | `tools/doc_appeal_scorecard.py` | `appeal_debt` | a doc's prose voice (human, not LLM) |
| seo / aeo | `tools/seo_aeo_scorecard.py` | `seo_debt` | answer-engine / search surface |
| demo-quality | `tools/demo_quality_scorecard.py` | `demo_debt` | the runnable demos |
| repo-hygiene | `tools/repo_hygiene_scorecard.py` | `hygiene_debt` | the tree's shape |
| observability | `tools/observability_scorecard.py` | — | dashboards / alerts / metrics doc |
| industry | `tools/industry_scorecard.py` | `parity_debt` | fak vs the SOTA field (OUTWARD) |
| agent-readiness | `tools/agent_readiness_scorecard.py` | `friction_debt` | can an AI agent adopt fak (one persona) |
| product | `tools/product_scorecard.py` | `product_debt` | can a person use each concept today |
| persona | `tools/persona_readiness_scorecard.py` | `persona_debt` | are the top-10 personas served |

The ones that fold into the unified ratchet are wired in
`tools/scorecard_control_pane.py` (`SCORECARDS`), which sums every `*-debt` into one
portfolio number and gates regressions.

---

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
5. **Commit only the scorecard lane, by explicit path** — `git commit -s -F msg --
   tools/<x>_scorecard.py tools/<x>_scorecard_test.py <data dir> <doc dir>`. Never
   `git add -A`. End the subject with the `(fak <leaf>)` trailer.

---

## Building a NEW scorecard (the recipe)

When a surface isn't measured yet, add one. The fastest path is to copy the closest
existing instance and re-point it: **`product_scorecard.py` / `persona_readiness_scorecard.py`**
for a *catalog* (data-dir rows cross-checked against the tree), **`agent_readiness_scorecard.py`**
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
   (`grade_letter`), the `<surface>_debt` integer (sum of HARD defects + coverage
   gaps), and emit the control-pane envelope with `corpus.<surface>_debt` and
   `corpus.grade`.

5. **Add renderers + flags.** `render` (terminal work-list), `--json`, `--compare
   baseline.json` (prove the drop), and a doc generator (`--markdown` or
   `--markdown-dir`). For catalog scorecards add `--chart` / `--critical` / `--gaps`.

6. **Write the test.** Fixtures for each KPI's defect trigger AND its clean case, the
   fold to `*-debt`, plus a **live smoke** asserting the real tree's current floor
   (usually zero debt) — that smoke is the regression sentinel.

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

Keep this skill refined as the method evolves: when a new scorecard teaches a better
KPI shape, a sharper cross-check, or a new failure mode (a SOFT signal that should be
HARD, a debt that double-counts), fold the lesson back here so the next scorecard
starts from it.
