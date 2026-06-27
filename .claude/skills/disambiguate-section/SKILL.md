---
name: disambiguate-section
description: Apply the concept-disambiguation pass to ANOTHER section of fak - an under-covered watched family (gateway-engine, guard-gate, ...) or a newly-discovered overloaded root (evict, decision, render, plan, pool, layout) - and, when a section is bigger than one sitting, file a GitHub ticket that scopes it with real context. Two modes over the same scorecard (tools/concept_disambiguation_scorecard.py): APPLY picks the worst-covered section from --gaps, surfaces its genuinely-DISTINCT confusable concepts (filtering inflections), adds grounded + glossary-anchored rows, draws the distinctions, re-measures to PROVE coverage rose, and commits the lane by explicit path; TICKET enumerates the sections that still need the pass (under-covered families + missed roots + cross-cluster canonical collisions) and opens one GH issue per section with the confusable tokens, file evidence, and scorecard-tied acceptance criteria. Use after running disambiguation-score when the backlog is too large for one pass, to onboard a new overloaded root, or to turn the coverage backlog into tracked work.
---

# disambiguate-section - extend crystal clarity to the next section (or file the ticket)

> A companion to **[disambiguation-score](../disambiguation-score/SKILL.md)** (which RUNS
> the scorecard and retires debt in the families already watched) and an instance of the
> generic **[scorecard](../scorecard/SKILL.md)** doctrine. This skill is about REACH:
> pointing the same discipline at a section it does not yet cover - and, when that section
> is a whole epic of work, filing a GH ticket that scopes it honestly instead of leaving it
> as an anonymous coverage number.

## When to use

- The coverage backlog for a family is large and you want to chew the worst one (APPLY).
- A new overloaded root has appeared in the tree that no family watches yet - the
  scorecard's critic flagged `evict`, `decision`, `render`, `plan`, `pool`, `layout`
  (APPLY: add the family + rows, or TICKET: scope it).
- You want to turn "11.4% coverage" into tracked, contextual work other agents can pick up
  (TICKET).

## The confusable map this works from

Run the scorecard to get the live backlog; everything below is re-derived, never hand-typed:

```
python tools/concept_disambiguation_scorecard.py --gaps        # per-family unpositioned tokens
python tools/concept_disambiguation_scorecard.py --json        # corpus.coverage.per_family / .uncovered
python tools/concept_disambiguation_scorecard.py --critical    # clarity-debt (should be 0 on a clean tree)
```

The work splits three ways, worst-first:
1. **Under-covered watched families** - a family with a big `discovered - covered` gap
   (e.g. gateway-engine, guard-gate, context-ctx). The fix is rows.
2. **Missed overloaded roots** - a root no family watches (the critic's list). The fix is a
   new `families[]` entry in `_meta.json` + rows.
3. **Cross-cluster canonical collisions** - one token, two unrelated meanings in different
   packages (`Manifest`, `Verdict`, `Backend`, `Observer`, `Index`, `Store`, `Refusal`,
   `KVCache` vs `Session.Cache`). The fix is a RENAME or a paired pair of rows that draw the
   line, plus a glossary entry.

## APPLY mode (retire one section's coverage debt)

1. **Pick the section** - the worst `discovered - covered` from `--gaps`, or a new root.
2. **Separate signal from inflection** - the discovery is deliberately broad and counts some
   grammatical variants (`gated`/`gates`/`guards`) and one-off compounds. These are NOT new
   concepts. Either position the genuinely-distinct ones (e.g. `kernel` the coordinator vs
   the compute `matkernel`/`sessionq4kkernel`; `vCacheScore` vs `vCache`; `memoryKVCache` vs
   `KVCache`) OR add the pure inflection to the family's `ignore` list in `_meta.json` so the
   universe counts concepts, not grammar. Adding an inflection to `ignore` is honest precision,
   NOT gaming - it removes a non-concept from the universe; dropping a real concept's root to
   hide it IS gaming.
3. **Add grounded + anchored rows** - for each distinct concept: a real `grounding` token
   (verify it appears in the production corpus, STRICT identity), a `definition`, a
   `distinction`, a `distinct_from` that resolves to a real catalog id. To earn a `crystal`
   verdict, write the distinction into `docs/fak/concept-glossary.md` and point
   `glossary_anchor` at it; otherwise leave the anchor empty for a `defined` verdict.
4. **Re-measure + prove** - `--compare baseline.json` shows coverage up and clarity-debt
   still 0; regenerate the doc folder.
5. **Re-pin + commit the lane by explicit path** - update `tools/scorecard_baseline.json`
   (the `disambiguation` metric drops as coverage rises), then
   `git commit -s -- tools/concept_disambiguation_scorecard.data/ docs/fak/concept-glossary.md
   docs/concept-disambiguation-scorecard/ tools/scorecard_baseline.json`. Never `git add -A`.
   End the subject with `(fak concept-disambiguation)`.

## TICKET mode (turn the backlog into tracked work)

Use this when a section is too big for one sitting. The helper is `gh`; writing issues is the
only side effect - keep it deliberate.

1. **One label, one epic** - ensure the `disambiguation` label exists
   (`gh label create disambiguation -c "#5319e7" -d "concept-disambiguation: crystal clarity over similar-sounding names"`),
   and an epic parent groups the children.
2. **One issue per section** - for each under-covered family / missed root / collision set,
   write a body with FOUR parts:
   - **Context** - why this section is confusable (the shared root, how many distinct meanings).
   - **The confusable names** - a table `concept | what it is | file:line evidence`, citing
     ONLY tokens you verified with Grep (no fabricated symbols).
   - **Acceptance criteria** - measurable against the scorecard: e.g. "family `X` coverage
     rises from a/b to >= c", "N grounded+anchored rows added", "glossary section written",
     "clarity-debt stays 0", "for a collision: the canonical_unique KPI is clean (rename
     landed) or both rows draw the line".
   - **How** - point at this skill's APPLY mode + `disambiguation-score`.
3. **Wire it** - create children, then edit the epic body to list the child numbers. Label
   each with `disambiguation` + `documentation`, priority per impact.
4. **Report** - print the created issue numbers/URLs; do not claim more than `gh` returned.

## The anti-gaming rule (specific to reach)

**Raise coverage by disambiguating real concepts, never by shrinking the universe dishonestly.**
Adding a pure inflection (`gated`) to `ignore` is fine - it is not a concept. Dropping a
family root so a real concept (`matkernel`) stops being discovered is gaming. A ticket that
"closes" a section by deleting its family entry, not by adding rows, is gaming. The grounding
cross-check is STRICT identity on purpose: cite the token the code actually uses.
