# Slop-debt sub-categories + a systemic way to retire it (2026-07-03)

Grounding: `python tools/code_slop_scorecard.py --json` on the tracked tree,
score 56.4/100 (grade F), **826 HARD units of slop-debt**. The distribution is
not spread across the six axes — it is one axis:

| KPI | debt | share |
|---|---:|---:|
| duplication | 810 | 98.1% |
| dead_code | 16 | 1.9% |
| comment_slop / vacuous_tests / stub_masquerade / churn_bloat | 0 | 0% |

So "attack slop debt" today means "attack `duplication`", and every proposal
below is about giving that one 810-unit bucket the structure it is missing.

## The systemic failure: one bucket, one weight, no gradient

The clone detector already does real classification work — the FP-gate pipeline
in `kpi_duplication` (`tools/code_slop_scorecard.py:751`) recognizes
signature-only headers, flag-plumbing, dispatch-arm boilerplate, and entry-point
scaffolds, and routes some to `soft`. But every group that *survives* the gates
collapses to the same scalar: **+1 debt, −2 score, regardless of shape.** A
46-line GEMM clone across 21 sites and a 6-line `for i := 0; i < len(s)` loop are
worth exactly the same.

Three consequences, all systemic:

1. **The number can't direct work.** "810" tells you nothing about *where* the
   extractable value is. An agent on a `/slop-score` loop retires whatever the
   worst-first list surfaces, which is sorted by first-site path, not by payoff.
2. **The debt is not monotonic under good refactors.** Extract one real helper
   and you may split a merged group, or expose a shorter sub-clone, nudging the
   count sideways. There is no "these N are the real ones" invariant to drive to
   zero.
3. **The gates are all-or-nothing.** A group is HARD or `soft`; there is no
   "advisory, low-value, tracked-but-not-gating" middle, so genuinely-idiomatic
   duplication either inflates the F or gets silently dropped.

## Empirical sub-categories (derived from the 802 emitted groups)

Classifying the current defect list by span, fan-out, site-location, and the
sample line:

**By span (source lines of the largest site):**
| span | groups | reading |
|---|---:|---|
| 5–6 | 174 | idiom-scale; rarely worth a helper |
| 7–15 | 526 | the bulk; mixed |
| 16–30 | 86 | almost always a real extractable unit |
| 31+ | 16 | unambiguous copy-paste of a whole function |

**By fan-out (# sites):**
| sites | groups | reading |
|---|---:|---|
| 2 | 521 | a pair; extraction may not pay for the indirection |
| 3–5 | 221 | a helper clearly pays |
| 6–12 | 41 | a helper is overdue |
| 13+ | 19 | a missing abstraction; highest priority |

**By location / kind:**
- **240 same-file self-clones** — a block duplicated *within one file*. Different
  fix (local helper, or a loop) and different risk than cross-file. Currently
  indistinguishable in the count.
- **125 high-value prod clones** (≥10 lines, ≥3 sites, no `_test.go` site) — the
  real target. This is the number worth driving to zero.
- **~10 embedded-string / import-block clones** — e.g. a shared Optuna Python
  snippet in benchmark harnesses, or an identical `import (...)` block. These are
  *data*, not logic; extracting them is often wrong.
- **2 all-`cmd/*/main.go` scaffolds** already routed to `soft`.

### Proposed duplication sub-category taxonomy

| sub-KPI | definition | weight class | why separate |
|---|---|---|---|
| `dup_extractable` | ≥ MIN_SPAN lines **and** ≥3 sites **and** ≥1 non-test site | **HARD, full weight** | the real missing-helper debt; the number to zero out |
| `dup_pair` | exactly 2 sites, ≥ MIN_SPAN lines | HARD, half weight | a pair may not justify indirection; real but lower payoff |
| `dup_local` | all sites in one file | HARD, half weight | distinct fix (local helper / loop); should not read like cross-cutting rot |
| `dup_test` | every site in `_test.go` | SOFT (advisory) | table-driven test scaffolding is often deliberate |
| `dup_data` | window is a string-literal / import block (no control-flow tokens) | SOFT (advisory) | it is data, not logic; extraction usually wrong |
| `dup_idiom` | < MIN_SPAN, or matches the existing FP gates | dropped (as today) | not extractable slop |

This is a **re-projection of data the detector already computes** — span,
site-set, file-set, and token-kind are all in hand at emit time. No new scan.

The load-bearing change is that `slop-debt` becomes a *weighted* sum, and the
control-pane surfaces `dup_extractable` as its own line, so the worst-first loop
retires the 125 real clones before it ever touches a 2-site idiom.

## Applying the same lens to the other axes (so the taxonomy generalizes)

The duplication split is the urgent one, but the *pattern* — split a flat count
into "load-bearing vs advisory vs local" — is the reusable idea:

- **dead_code (16)** → `dead_prod` (an unexported symbol referenced nowhere, in
  non-test code — real) vs `dead_scaffold` (a parked helper whose `runX` is
  tested but whose `case "x"` is intentionally commented out pending a peer wire;
  see the memory `peer-wave owns main.go scaffold wiring`). Today these collide;
  the second is a false positive that wastes an agent's turn.
- **vacuous_tests (0)** → already effectively split by the `_is_reexec_helper`
  exemption (`c78fe627`). That exemption *is* a sub-category; it just isn't named
  as one. Naming it makes the pattern legible and reusable.
- **stub_masquerade** → the SOFT→HARD promotion ladder (#781) is a
  *maturity* sub-category (soak-window state), orthogonal to *kind*.

## Systemic approaches (not one-off retirement)

The current loop is: run scorecard → retire worst-first → re-measure. That
retires *instances*. The systemic moves attack the *generators*:

**1. Weight by payoff, not by count.** Ship the sub-category weights above so the
debt integer tracks *extractable value*, and a 46-line ×21 clone dominates a
6-line ×2 idiom. This alone re-points every `/slop-score` loop at the real work.

**2. A "helper-extraction" arm that acts, not just scores.** The 125
`dup_extractable` groups are mechanically similar: N copies of a token-identical
block → one function + N call sites. This is the `modularize` skill's motion
(behavior-preserving code motion, `goimports -w`, `gofmt`, `go vet`, `go test`,
commit by path) applied to clones instead of god-functions. A `dedup` skill
that: picks the top `dup_extractable` group, proposes the helper signature,
does the extraction in one package, verifies, and proves the debt dropped —
turns scoring into shipping.

**3. Generator-level prevention (the real leverage).** Many clones are *born*
from a template, not copy-pasted by hand. Verified against the tracked tree
(`.dos/_dos_park`, `.fak/tmp`, `.tmp` copies excluded — the scorecard already
skips them):
   - the `cmd/*/main.go` flag-parse + `run()` skeleton (already `soft`-gated),
   - the `render.go` post-formatting structs (`Emoji string // leading status
     glyph` appears across `benchpost`/`dojopost`/… render files) — a shared
     `internal/postfmt` row/render struct,
   - **`ParseLedger` — 5 tracked definitions across 5 packages**
     (`cachevalueledger`, `cadencereport`, `dojo`, `experiments`, `nightrun`),
     each an identical parse *shape* returning a different row type (`[]Row`,
     `[]LedgerRow`, `[]Experiment`, `[]CollectRow`). A real generics /
     shared-parser candidate — the clearest single generator.

   NOTE on false positives at generator scale: the clone report's high-fan-out
   headline groups can OVERSTATE a generator. `gradeFor` shows as a ×34 clone,
   but the tracked tree has only **2 definitions with *different* signatures**
   (`gradeFor(score float64)` vs `gradeFor(b bucket, snap, sig)`); the ×34 is the
   token-window matching the grade-*bucket body* shape across many unrelated
   functions, not 34 copies of one function. This is itself evidence for the
   sub-category split: a fan-out count over token windows is not a count of
   extractable call sites, so `dup_extractable` must be defined on **distinct
   definition sites**, not raw window occurrences.

   The systemic fix is to file one extraction ticket per *verified* generator
   (shared `internal/ledger`, `internal/postfmt`) so the clone can't be reborn.
   A `dedup --by-generator` mode that groups `dup_extractable` by sample-line
   shape — then confirms each cluster against distinct tracked definitions —
   surfaces these directly while filtering the token-window mirages.

**4. Make the sub-category boundaries themselves ratchetable (score-2x).** Once
`dup_extractable` is driven low, harden: lower `CLONE_MIN_GROUP_SPAN`, or
promote `dup_pair` from half-weight to full. The `score-2x` doctrine (debt down,
then bar up) already exists; sub-categories give it finer knobs to tighten than
a single global threshold.

## Recommended sequence

1. **Land the duplication sub-category split** in `kpi_duplication` (pure
   re-projection; emit `dup_extractable` / `dup_pair` / `dup_local` /
   `dup_test` / `dup_data` alongside the flat count, weighted). Prove the total
   is unchanged in *coverage* but now *ordered by payoff*. — the keystone.
2. **Add the `dedup` skill** that retires `dup_extractable` worst-first with the
   `modularize` code-motion recipe + DOS commit-audit witness.
3. **File the generator-extraction tickets** (grade / postfmt / ledger / render
   structs) so the highest-fan-out clones die at the source.
4. **Generalize** the dead_code split (`dead_prod` vs `dead_scaffold`) to stop
   the parked-scaffold false positives.

The through-line: today slop-debt is one number that says *how much*; the
sub-categories make it say *which kind, and worth what* — which is the only form
of the number an agent loop can act on without a human re-sorting it every time.
