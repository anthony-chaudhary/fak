---
name: lightgap-score
description: "One repeatable pass that answers \"is fak actually worth adopting, for whom, and what does it cost you to find out?\" on an UNBOUNDED scale anchored at two ends — the next-best option a given buyer would really use, and the best that is physically possible. Runs the Go-backed `fak score lightgap` over a modular data directory (tools/lightgap_scorecard.data/): 8 facets x 7 buyer segments, each cell scored w_net = artanh(beta) - artanh(load), where beta is the fraction of the alternative-to-ceiling. Use when this named workflow matches the task."
---

# lightgap-score — how good is fak really, for whom, and against what

> **What this does.** Every other scorecard in this repo tops out at 100, and a
> number that tops out cannot tell you whether you are near a wall. This one does
> not top out. Each cell is anchored at **two** points a buyer can actually check:
> the thing they would do **instead** of adopting fak, and the **best that is
> physically possible** on that axis. Then it subtracts, on the same scale, what it
> costs *them* to learn it. The output is not a grade — it is a per-buyer verdict,
> and the verdicts disagree with each other on purpose.

The shape: **run the card → read the dents before the peaks → close the worst
UNCOVERED comparison by RUNNING IT → re-anchor any ceiling the field moved →
regenerate the doc folder → commit only the lightgap lane.**

---

## The one thing that makes this card different from every other scorecard here

**Its debt does not retire by editing a file.** `parity_debt`, `doc_debt`,
`code_debt` and the rest all retire by fixing something in the tree. `lightgap_debt`
counts **material comparisons nobody has run** — a facet a buyer weights heavily
where no committed artifact compares fak to what that buyer would actually use. The
only honest way to close one is to **run the experiment and commit the artifact**.

That is why `--check` names the experiment for every gap. If you find yourself
"fixing" `lightgap_debt` by lowering a weight, deleting an `unrun` entry, or
inventing a number, stop: you have converted an open follow-on into a false claim,
which is the exact failure this card exists to prevent.

---

## The model in four lines

```
beta  = (fak - next_best) / (ceiling - next_best)      # gap to physics, closed
load  = (fak_hours - alt_hours) / tolerance_hours      # patience consumed, signed
w_net = artanh(beta) - artanh(load)                    # unbounded, signed, additive
                                                       # => adopt iff beta > load
```

Why rapidity (`artanh`) and not a ratio or a percentage:

- **Zero at the alternative.** A positive score has to be *earned* against what the
  buyer already has. Matching the incumbent scores 0, not 50.
- **Signed and unbounded below.** Being worse than the alternative is a negative
  score, not a low one. That is what lets `local-first x raw-speed` come back
  `-4.35` and BLOCK the whole segment instead of averaging away.
- **Diverges at the ceiling.** Closing the last 1% of the gap to physics costs
  unboundedly more than the first 50%. That divergence is the honest description of
  a moat — and of how far fak is from having one.
- **Additive, so the tax subtracts on the same scale.** `load` is not a fudge
  factor applied afterwards; it is the same transform on the adopter's patience,
  which is what makes `beta > load` a real decision rule rather than a slogan.

`alt_hours` is the term that carries "same result, far less restructuring" — a cell
can score positive at `beta = 0` purely because the alternative is a rewrite and fak
is a wrapper. When that happens the card says so explicitly: **that is a procurement
claim, not a capability claim.**

### Three anchor shapes

| shape | when | effect |
|---|---|---|
| default | there is real headroom between the alternative and the limit | `beta = (F-N)/(c-N)` |
| `pure_tax` | **the incumbent IS the ceiling** (fak fronts SGLang; llama.cpp is already installed) | `beta = (F-N)/|N|`, can never be positive — mediation only takes a cut |
| `parity_at_ceiling` | both options sit at the definitional floor (AgentDojo ASR 0) | `beta = 0`; only differential adoption cost moves the score |

Getting the shape wrong is the most damaging error available here, because it
silently changes what the denominator means. A `pure_tax` cell that scores positive
is a bug in the anchor, not a win — the test suite asserts this.

---

## The data directory (modular — one concern per file)

```
tools/lightgap_scorecard.data/
  _meta.json           bands, the 8 facets, and the 7 buyer segments. Each segment
                       carries tolerance_hours (the patience denominator), switch_bar
                       (what is worth switching for), weights (the buyer's attention
                       budget, must sum to 1.0), next_best_summary (what they would
                       use instead), and `unrun` (why a weighted facet has no cell,
                       plus the experiment that would close it).
  _ceilings.json       one ceiling per facet: metric, unit, direction, c, kind
                       (physical | definitional | lower-bound), derivation, caveat.
                       An undefended ceiling turns the denominator into a wish.
  _alternatives.json   the next-best registry, each with a class
                       (sota | tuned | floor | naive) and a source.
  cells-<segment>.json one file per buyer. Each cell names its alternative, its fak
                       value + provenance + source, its adoption cost + basis, and
                       usually a fence.
```

Edit a file, re-run the tool. `docs/lightgap-scorecard/` is GENERATED by
`--markdown-dir`; never hand-edit a page.

---

## The rules that override everything

1. **Never score against a strawman.** `naive-reprefill` is in the alternatives file
   *because* the headline 60.3x figure is measured against it, and scoring against it
   would be dishonest. It is catalogued as the zero point and **no cell may use it**.
   The comparison is always what the buyer would genuinely deploy. The test suite
   enforces this.
2. **Never invent a fak number.** Every value traces to
   [`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md) or a committed
   generated artifact. **Check the authority row against the generated doc** — they
   drift. (The support matrix is the live example: the authority row still says
   19/56 while the CI-freshness-gated `docs/HARDWARE-MATRIX.md` says 32/56. The
   generated doc wins.) If no comparison exists, the honest output is an `unrun`
   entry, which becomes debt.
3. **Respect the authority's own fences.** `BENCHMARK-AUTHORITY.md:66` carries a
   RETRACTION plus explicit pricing fences on shed tokens. A derivation that violates
   a fence is not a number, however arithmetically tidy it looks.
4. **Claim caps are not negotiable.** MODELED/PROJECTED cap at CRUISE; OBSERVED and
   `lower-bound` ceilings cap at RELATIVISTIC. Raw `w_net` is always reported so the
   cap is auditable; capped `w_eff` is what decisions use. An authored corpus must not
   read like a measurement.
5. **A cell that is too good is a bug.** If a facet pins at the same value across
   every segment, the input is almost certainly `F = c`. Re-derive it honestly. (This
   already happened once: observability sat at a flat `1.0` because all five decision
   classes are *recorded* — but not **in one artifact**, so single-artifact
   reconstruction is 4/5, and the peaks differentiate by segment once that is fixed.)
6. **No overall score. Ever.** Averaging a BLOCKED local-first result against an
   ADOPT platform-team result hides precisely what the reader came for. The test
   suite asserts the payload emits no aggregate.

---

## Step 1 — Run it

```bash
fak score lightgap                         # the sphere + per-use-case verdicts
fak score lightgap --dents                 # every cell where fak LOSES, worst first
fak score lightgap --unrun                 # the comparisons nobody has run
fak score lightgap --check                 # honesty gate; exit 1 on lightgap_debt
fak score lightgap --segment local-first   # one buyer, in full
fak score lightgap --facet raw-speed       # one axis, across buyers
fak score lightgap --ceilings              # every anchor and its derivation
fak score lightgap --json                  # machine payload
```

Read `--dents` **before** the summary. A scorecard whose own subject wins every cell
is measuring the wrong things; the dents are where the card earns its keep.

## Step 2 — Pick the worst-first move

| What you see | What it means | The honest move |
|---|---|---|
| **UNCOVERED** in `--check` | a facet this buyer weights has never been compared to their real alternative | **Run the experiment `--check` names.** Commit the artifact, then add the cell. |
| A segment came back **UNDECIDABLE** | too much of that buyer's weighted attention is unmeasured | Same — this is the highest-value work on the board. |
| A segment came back **BLOCKED** | a material axis is REGRESSIVE | Do NOT re-weight. Either fix the axis or state the boundary in the positioning. |
| `DEGENERATE_CEILING` | ceiling equals the alternative | The anchor is wrong. Pick the real limit, or mark the cell `pure_tax`/`parity_at_ceiling`. |
| `CEILING_BREACHED` | a `pure_tax` cell scored above its own ceiling | The shape or the numbers are wrong. Mediation cannot beat what it mediates. |
| `ALT_UNAFFORDABLE` | the alternative costs more than the buyer's whole tolerance | It is not their next-best option. Find the one they would actually use, or declare the facet `unrun` with that reason. |
| `UNFENCED_MODELED_LEAD` | an authored estimate reports as a large win with no fence | Add the fence, or re-derive. |
| Any other defect | the model is misconfigured | Fix it. A misconfigured model produces numbers that look fine and mean nothing. |

## Step 3 — Re-anchor when the field moves

A ceiling is a claim with an expiry. Two ways it goes stale:

- **A `lower-bound` ceiling** (`c` = best system currently known) moves whenever
  somebody ships better. Re-check the derivation and update `c`; every cell on that
  facet re-scores. This is the mechanism that stops a stale ceiling from flattering
  fak forever.
- **A `physical` or `definitional` ceiling** does not move, but its *derivation* can
  be wrong. The raw-speed roofline is `BW / bytes-per-weight-pass` (150 GB/s ÷ 1.54 GB
  Q8_0 ≈ 97.4 tok/s on an M3 Pro) — a different host or quantization is a different
  ceiling, so a cell scored on one must say so.

Also re-check **`next_best_summary`** per segment. If the thing a buyer would
otherwise use has changed, every cell in that file is anchored to the wrong zero.

## Step 4 — Prove it, regenerate, verify

```bash
fak score lightgap --json > /tmp/lightgap-after.json
go test ./internal/lightgapscore
go test ./cmd/fak -run Lightgap
fak score lightgap --markdown-dir docs/lightgap-scorecard
```

State the before/after in the buyer's terms, not the tool's: *"solo-max moved
UNDECIDABLE → ADOPT-WITH-SCARS once the long-session head-to-head landed; debt 9 → 8"*.

The test suite pins the **arithmetic identities** (zero at the alternative,
divergence at the ceiling, additivity, direction-agnostic beta) and the **honesty
invariants** (claim caps, sourced cells, unmeasured facets become debt). It
deliberately does **not** pin the debt integer or any `w_net`, so the card improving
never reds the suite.

## Step 5 — Commit only the lightgap lane, by explicit path

```bash
fak commit --path .claude/skills/lightgap-score/SKILL.md \
  --path tools/lightgap_scorecard.data --path docs/lightgap-scorecard \
  -m "feat(lightgap): <what improved> (fak claude)"
```

- Stage by explicit path, never `git add -A` — this is a shared trunk.
- A data + generated-doc diff takes `docs(...)` or `chore(scorecard): …`; end the
  subject with the `(fak tools)` trailer or it stays NOT_SHIPPED.
- On Windows pass the message via `-F <file>`, and keep `-m`/`-F` **before** the `--`.
- Stay on the trunk (`main`); push promptly.

---

## Control-pane status

`lightgap_debt` is emitted at the top level of `--json` and is control-pane
compatible (`find_int` locates it), but the card is **deliberately held out** of
`tools/scorecard_control_pane.py` for now — see the reason in `EXCLUDED_SCORECARDS`
in `tools/scorecard_control_pane_test.py`. Two things block the fold: its debt unit
is "an experiment nobody ran" rather than "a defect in something that exists", and
registering it adds 9 to `total_debt`, which reds the shared ratchet until
`scorecard_baseline.json` is re-pinned — a pin that cannot be taken honestly while
go-backed cards are dropping out of the fold on a non-compiling tree. **Register and
pin on a green tree.**

The card emits no corpus grade on purpose (it has no mean); `derive_grade(debt)` is
the correct last-resort lens.

## When to run this

- **Before any positioning or README work** — this card tells you which buyer the
  claim is true for, and it is usually not all of them.
- **When a head-to-head lands** — turn an `unrun` entry into a cell and watch the
  segment verdict move.
- **When a competitor moves a `lower-bound` ceiling** — every cell on that facet
  re-scores against the new limit.
- **When someone asks "should we use this?"** — `--segment <their-shape>` is the
  answer, including the part where the answer is BLOCKED.
