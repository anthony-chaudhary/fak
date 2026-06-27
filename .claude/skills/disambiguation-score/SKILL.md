---
name: disambiguation-score
description: One repeatable pass that keeps fak's growing namespace CRYSTAL-CLEAR - each similar-sounding concept (cache / vCache / KV cache / cachemeta / the provider prompt-cache; guard vs gate; the two witnesses) given one canonical name, a written definition, and an explicit line drawn against the siblings it is confused with. Runs the concept-disambiguation scorecard (tools/concept_disambiguation_scorecard.py) over a data dir of confusable-concept rows cross-checked against the tree (the grounding token must appear in the production corpus; the glossary anchor must exist; a distinct_from reference must resolve), turns each HARD defect into a required fix (rename a true canonical collision, write a missing definition, draw + anchor a missing distinction, replace a fabricated grounding) and each uncovered tree token into a coverage row to ADD, retires disambiguation-debt worst-first, re-measures to PROVE the debt dropped, and commits only the scorecard lane by explicit path. The NAMING-clarity counterpart of conflation-score (number provenance) and product-score (can a person use the concept). Use after adding a confusable name, when two concepts blur, or on a /loop cadence to keep the vocabulary from fogging as it grows.
---

# disambiguation-score - the crystal-clarity / naming pass

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine. Read that
> for the five laws and the RSI loop; this file is the per-surface specialization.

## Why this scorecard exists

fak has grown a large vocabulary, and several roots are badly overloaded. The word
"cache" alone names at least a dozen distinct things (KV cache, vCache, cachemeta, the
provider prompt-cache, ViewCache, enginecache, RadixKV, ...); "guard" and "gate" blur;
"witness" means two unrelated ideas. As the system grows, a reader cannot tell the
concepts apart - and a name a reader cannot pin is a name they will misuse.

The crystal-clarity law, in one line: **every distinct concept gets one canonical
name, a written definition, and an explicit line drawn against the siblings it is
confused with - and that distinction lives in a doc that exists, not tribal memory.**

This is orthogonal to its two easily-confused siblings (an irony the glossary notes):
`conflation-score` keeps every reported NUMBER labeled by provenance (WITNESSED vs
OBSERVED); `product-score` asks whether a person can USE a concept. This one asks
whether the NAME is unambiguous.

## The KPIs (what it grades)

- **well_formed (HARD)** - required fields present, enums in the closed vocabulary,
  unique id, family declared.
- **canonical_unique (HARD)** - no two concepts share THE canonical name. A real
  collision is fixed by a RENAME, never a note.
- **defined (HARD)** - every concept carries one distinguishing definition sentence.
- **disambiguated (HARD)** - the heart: a concept in a family with >= 1 positioned
  sibling must carry a `distinction` line AND a `distinct_from` that resolves to a real
  catalog id. A confusable concept that never says what it is NOT is the core debt.
- **grounded (HARD)** - the grounding token must appear in the production corpus by
  STRICT identity (a fabricated `policyloaded` that merely contains `policy` is refused).
- **anchored (HARD)** - a concept claiming `crystal` clarity must have its distinction
  written in a glossary_anchor that exists; any non-empty anchor must resolve on disk.
- **clarity_consistent (HARD)** - the declared verdict must match the evidence
  (crystal / defined / drifting / colliding / undocumented). Calling an unanchored
  concept `crystal` is the overclaim this catches.
- **kind_grounding / hierarchy (SOFT)** - advisory: kind agrees with grounding; an
  optional `parent` resolves and does not cycle.

Plus **coverage**: for each watched family the scorecard DISCOVERS the confusable
tokens in the tree (a token counts only when it spans >= `min_files` production files,
or is a package/dir name, or a doc heading), and `coverage_debt` is the count not yet
positioned. This is the ungameable engine - you cannot shrink the universe by editing
data, only by disambiguating the real concept.

## The pass (the shared five-step loop)

1. **Run it** - `python tools/concept_disambiguation_scorecard.py` (work-list),
   `--json` (payload), `--critical` (worst-first), `--gaps` (the coverage backlog by
   family), `--compare baseline.json` (prove the drop).
2. **Retire disambiguation-debt worst-first** - fix the heaviest KPI by changing
   REALITY: rename a true canonical collision, write a missing definition, draw the
   `distinction` + point `distinct_from` at a real sibling id, replace a fabricated
   grounding with a real token. Then close coverage (`--gaps`): add a row for the worst
   family's unpositioned tokens, writing the distinction into
   `docs/fak/concept-glossary.md` to earn a `crystal` verdict.
3. **Weigh the SOFT signals** - fix the cheap real ones (kind/grounding, a dangling
   parent); do not chase them to zero.
4. **Re-measure + prove** - `--compare` prints the debt delta; the positioned rows must
   stay clean (0 clarity-debt) while coverage rises.
5. **Commit only the scorecard lane, by explicit path** -
   `git commit -s -- tools/concept_disambiguation_scorecard.py
   tools/concept_disambiguation_scorecard_test.py tools/concept_disambiguation_scorecard.data/
   docs/fak/concept-glossary.md docs/concept-disambiguation-scorecard/
   tools/scorecard_control_pane.py tools/scorecard_baseline.json`. Never `git add -A`.
   End the subject with the `(fak <leaf>)` trailer.

## The anti-gaming rule (specific to this surface)

**Disambiguate a concept by making the name + the written distinction real, never by
weakening the cross-check.** A coverage gap is closed by ADDING an honest, grounded,
anchored row - not by dropping the family root so the token stops being discovered, and
not by listing a `distinct_from` id that does not exist. A `crystal` verdict is earned
by writing the distinction into the glossary - not by pointing the anchor at a doc that
happens to exist. The grounding cross-check is STRICT identity on purpose: if a real
concept's token genuinely is not in the production corpus, the honest fix is to rename
the concept toward the name the code actually uses, not to relax the match.

## Adding a surface

When a new overloaded root appears (a fresh `*-thing` family proliferating across the
tree), add a family entry to `_meta.json` (`roots`, optional `ignore` inflections,
optional `exclude` substrings, `min_files`); the scorecard then discovers its
confusable tokens and the coverage backlog grows until you position them. Keep the
glossary (`docs/fak/concept-glossary.md`) as the single anchor the `crystal` rows cite.

## ASCII gotcha

Keep the scorecard source + data + glossary ASCII-only (use ` - ` for a dash, straight
quotes). The repo's provenance pre-commit hook crashes on non-cp1252 UTF-8 before its
own escape fires.
