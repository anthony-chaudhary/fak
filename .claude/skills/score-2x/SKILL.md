---
name: score-2x
description: The generic 2×-then-harden loop the whole scorecard family runs — score a surface, drive its debt down 2× with genuine fixes, rescore to PROVE the drop, and when the surface saturates (grade A, zero debt, nothing left to retire) HARDEN the metric itself — tighten a real threshold, promote a SOFT KPI to HARD, or add a dimension — so the score stays a live gradient instead of a frozen A. The conductor over the per-surface instruments (quality-score, industry-score, persona-score, steerability-score, …): it owns the one move no single *-score skill owns — raising the bar when a metric stops discriminating, then re-pinning the control-pane ratchet. Use to run a 2× pass on any surface, to decide whether a saturated scorecard needs hardening, or on a /loop cadence to keep every metric honest in BOTH directions (debt down, bar up).
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[surface] [--target 2x|3x] [--harden]  (surface = code-quality|industry|persona|steerability|…; no surface = list scorecards, pick the one with the most room)"
metadata:
  opencode: claude-only
---

# /score-2x — drive a metric 2×, then harden it when it saturates

> **What this is.** Every `*-score` skill measures one surface and retires its debt
> worst-first. This is the loop *around* all of them — the process the operator
> described in one line: **"fresh score on X. work issues that improve it 2× if
> possible. rescore. if the score is too high, refresh and make the metric harder."**
> The per-surface skills are the instruments; this is the conductor. Its distinctive
> move — the one no single scorecard skill owns — is the last step: when a surface
> saturates (grade A, zero debt, nothing genuine left to retire), the metric has
> stopped doing work, so you **raise the bar** and the loop has a real gradient again.

This is the Goodhart defense made into a routine. A scorecard pinned at A rewards the
status quo; it no longer tells you the surface improved, only that it stopped getting
measured. The 2×-then-harden loop keeps the number a *live* measure: debt ratchets
**down** while the surface is dirty, the bar ratchets **up** when the surface is clean.

The shape: **pick a surface + record the baseline → retire debt to halve it (genuine
fixes only) → rescore with `--compare` to prove the drop → decide: ship the 2×, or
(if saturated) harden the metric and re-pin → commit ONLY that lane by explicit path.**

This skill writes nothing on its own beyond the genuine fixes, the regenerated
snapshot, and — on the harden branch — the threshold/KPI edit plus the re-pinned
`tools/scorecard_baseline.json`. It never weakens a check to score, and never
hardens a metric in a way that can't be defended as a genuinely higher standard.

---

## Step 0 — Pick the surface (and read its instrument)

The scorecard family lives in `tools/*_scorecard.py`; the shared doctrine is
[`scorecard`](../scorecard/SKILL.md) and the unified ratchet is
`tools/scorecard_control_pane.py`. Each has a paired RSI skill (`quality-score`,
`industry-score`, `persona-score`, `steerability-score`, `appeal-score`,
`conflation-score`, `agent-readiness`, `stability-score`, …).

- **A surface named?** Use it. (`code-quality` → `tools/code_quality_scorecard.py`.)
- **No surface?** Run the control pane to see the portfolio, then pick the surface
  with the **most room** — the worst grade / heaviest `*-debt`, *unless* a surface is
  already at grade A (then it's a harden candidate, Step 4):

  ```bash
  python tools/scorecard_control_pane.py            # portfolio: every metric's debt + grade + trend
  ```

Open that surface's `*-score` SKILL.md and obey **its** anti-gaming rules — they are
surface-specific (don't write a vacuous test for `tests` debt; don't delete a `// TODO`
for `hygiene`; don't weaken a guard for `ship_integrity`). This loop never overrides
the instrument's own honesty rules; it sequences them.

## Step 1 — Fresh score: record the baseline

Capture the machine payload BEFORE touching anything. The whole point is to prove a
delta, so you need the before-number on disk, not in your head:

```bash
python tools/<x>_scorecard.py --json > /tmp/<x>-base.json    # Bash > preserves UTF-8; PowerShell > mangles glyphs
```

Read `corpus.score`, `corpus.grade`, `corpus.<x>_debt`, and `corpus.breakdown` (per-KPI
debt, worst first). The per-KPI `defects` arrays ARE the work-list. **Write down
`debt = N`, `score = S`, `grade = G`.** If `N == 0` and `G == A`, the surface is
already saturated — skip to Step 4 (harden); there is no 2× to do.

## Step 2 — Retire debt worst-first to HALVE it (genuine fixes only)

Target the heaviest KPI (`breakdown[0]`) and work down. The bar for a 2× pass is
**halving the actionable HARD debt** (N → ≤N/2) — doubling a 70-ish score is
impossible since it caps at 100, so the honest "2×" always lives on the *debt* axis.
3× is N → ≤⌈(N+2)/3⌉ (the `--compare` ladder computes both targets for you).

Fix only via the surface skill's **safe, genuine classes** — add the real affordance,
write the real test, correct the real overclaim, split the real monolith. Per the
doctrine's one law: **retire a defect by changing reality, never by changing the
detector.** If "fixing" it would mean gaming a substring match, stop — it's not a real
gap, and weakening the check to make it green is the move that turns the family into
theater.

**"if possible" is load-bearing.** Some passes can't halve: the surface is already
near-clean, or the only remaining debt is RISKY (a refactor too big for one safe
lane), peer-owned, or historical (a residual that ages out). Report it as `not yet`
with evidence — the witness you're missing and the next checkable step — never fake
the halving and never convert an honest partial into a claimed 2×.

## Step 3 — Rescore and PROVE the drop

Re-run the **full** scorecard (not a `--no-toolchain` fast path) against the baseline:

```bash
python tools/<x>_scorecard.py --compare /tmp/<x>-base.json
```

The `--compare` renderer prints the per-group debt deltas and a VERDICT ladder:

- `VERDICT: ≥3× reduction achieved (… target ≤T3).`
- `VERDICT: ≥2× (not yet 3×) — … 3× needs ≤T3.`
- `VERDICT: not yet 2× — need <x>-debt ≤T2 (now M); 3× target ≤T3.`

State the delta plainly: `<x>-debt N → M (−k), score S → S', grade G → G'`. Regenerate
the committed snapshot so the doc matches the tree (each scorecard documents its own
`--markdown` / `--markdown-dir` form):

```bash
python tools/<x>_scorecard.py --markdown --stamp YYYY-MM-DD > docs/<X>-SCORECARD.md
```

> Scorecards without `--compare` (e.g. `code_quality_scorecard.py`) state the program
> target in prose — compute the delta by hand from the two `--json` reads: `debt N→M`.

If you hit the 2× (or 3×) target: this is a normal ship — **go to Step 5**. If the
`--compare` ratio reads `∞ (zero)` / debt is already 0 / grade is A and there is
nothing genuine left to retire: the surface is **saturated** — **go to Step 4**.

## Step 4 — Saturated? HARDEN the metric, then re-pin (the distinctive move)

A metric at grade A with zero debt has stopped discriminating: a clean small tree and
a clean 10×-bigger tree both read A, and the next pass would be rewarded for nothing.
"The score is too high" is the signal to **raise the bar** — make the metric measure
a genuinely higher standard, so it surfaces real new work and the loop continues.

**The saturation signals (any of):** `<x>_debt == 0` across two consecutive passes;
`--compare` ratio prints `∞ (zero)`; grade A with an empty work-list; the control pane
shows this metric flat at its pinned floor while the field/repo has plainly moved.

**The four genuine harden moves** (pick the one that reflects a real higher standard):

1. **Tighten a threshold constant.** The scorecard's HARD bars are named constants —
   e.g. in `code_quality_scorecard.py`: `FILE_HARD_MAX = 1500`, `FUNC_HARD_MAX = 200`,
   `TEST_MIN_FUNCS = 4`. Lower a god-file/god-function ceiling (1500 → 1200) or raise a
   coverage floor; the tighter bar surfaces outliers that were passing, and the loop
   returns to Step 1 to retire them. Update the constant's inline comment + the
   scorecard's own doc to the new bar.
2. **Promote a SOFT signal to HARD.** A surface that's clean on every HARD KPI but
   carries a pile of advisory SOFT signals (drift, `godoc`, `hygiene`) is telling you
   the *real* frontier moved to what used to be advisory. Promote it **only when its
   cheapest fix is genuinely real** — never one whose cheap fix is gaming (doc-comment
   spam, deleting a `// TODO`). That guard is *why* those stay SOFT; respect it.
3. **Add a new KPI / dimension.** The field added an axis the scorecard doesn't grade
   (industry), or a new concept/persona/competitor row joined the catalog. Add the KPI
   as a pure function cross-checked against the real tree, per the doctrine recipe.
4. **Raise a coverage target.** For a catalog scorecard, lift the coverage floor (the
   share of the field/personas/concepts that must be positioned) so an un-positioned
   row becomes debt again.

**Anti-gaming for hardening (the inverse trap).** Making a metric "harder" is only
honest if the new bar is **defensible as a genuinely higher standard** — one the repo
or the field should actually hold. Do NOT: invent a KPI that's trivially passed to pad
the count; add a size-coupled signal that reddens just from growth (the doctrine's
growth-invariant law forbids it — a 2×-bigger tree at the same discipline must still
score the same); or tighten a threshold so far past reality that it floods the
work-list with cosmetic noise. A good harden surfaces a *handful* of real new
outliers, not a wall of them. If you can't name why the tighter bar is a better
standard, don't tighten it — leave the A and say the surface is genuinely clean.

**Re-pin the ratchet.** Hardening RAISES this metric's debt (by design — it just found
new work), so the control-pane `--check` gate would read it as a regression until you
re-floor it. After the harden edit lands and you've retired the first round of newly
surfaced debt, re-pin:

```bash
python tools/scorecard_control_pane.py --check      # see the per-metric trend (now RED on this metric — expected)
python tools/scorecard_control_pane.py --pin        # re-floor the portfolio at the new, harder baseline
```

Commit `tools/scorecard_baseline.json` in the **same lane** as the threshold/KPI edit
(`--pin` blesses the new floor — without it, every later pass reads the harder bar as a
permanent regression). Then the loop returns to Step 1 against the harder metric.

## Step 5 — Commit ONLY the lane, by explicit path

Stay on `main` (the `OFF_TRUNK` guard refuses a branch). Commit by explicit path —
never `git add -A` on this shared tree:

```bash
git pull --no-rebase --no-edit
git add <your genuine-fix paths> docs/<X>-SCORECARD.md      # + tools/<x>_scorecard.py tools/scorecard_baseline.json on a harden
git commit -s -m "<conventional subject>" -m "<body: N→M debt (or 'harden: bar X→Y'), what changed>" -m "(fak <leaf>)"
dos commit-audit HEAD                                        # MUST print [diff-witnessed] / verdict OK
git push
```

- **Subject honesty.** A debt-retiring pass → the verb that matches the fix
  (`test(scope):`, `style(fmt):`, `refactor(scope):`). A harden pass → `chore(scorecard): tighten <bar> X→Y (fak <leaf>)`. Lead with a recognized verb or `dos commit-audit`
  ABSTAINs. End every ship commit with the `(fak <leaf>)` trailer so the `dos verify`
  referee binds it.
- **`dos commit-audit HEAD` is the gate.** `[diff-witnessed]` is the green light;
  `subject-only` / CLAIM_UNWITNESSED means your diff doesn't back your subject —
  reword to what the diff did (or add the missing change).
- **Default is to ship.** Once the lane is green (`make ci`), commit AND push; don't
  wait to be asked. Defer to the guard (`OFF_TRUNK`, a peer `MERGE_HEAD` in flight —
  wait for it to clear, then re-try the pathspec commit). Never force-push.

---

## Why this is RSI and not vibes

Each pass moves a number **re-derived from disk** (and, where the surface has one, from
a witness the agent didn't author — the Go toolchain's exit codes, a red→green
`go test`, `dos review` / `dos commit-audit`). The score is the gauge; the witnesses
are why you can trust it moved. The loop is monotone in BOTH directions:

- **While the surface is dirty,** debt ratchets down — halve, then halve again — and
  the control-pane `--check` gate refuses a regression above the pinned floor.
- **When the surface is clean,** the bar ratchets up — a tighter threshold, a promoted
  KPI, a new dimension — and `--pin` re-floors at the harder standard.

A metric that only goes one way eventually freezes at A and stops measuring. This loop
is what keeps "is X getting better" a question with a live, ungameable answer — the
debt can't drift past the cross-check, and the bar can't saturate into theater.

## The honesty rules (internalize — the loop is only as good as its honesty)

1. **2× lives on the debt axis, "if possible."** Can't halve safely this pass? Report
   `not yet` with the missing witness and next step — never fake the ratio.
2. **Retire by changing reality, never the detector.** The instrument's surface-specific
   anti-gaming rules bind here unchanged.
3. **Harden only to a defensible higher standard.** A tighter bar must surface a few
   REAL outliers, stay growth-invariant, and be one the repo/field should hold — or
   leave the A.
4. **Re-pin in the harden lane.** A harden without `--pin` poisons the `--check` ratchet
   for every later pass; pin the new floor in the same commit.
5. **Re-measure with the FULL scorecard and `dos commit-audit` the commit** before
   claiming either a drop or a shipped harden.

## When to run this

- To run a **2× pass** on any one surface (the per-surface skill does the retiring;
  this sequences baseline → prove → ship).
- When a scorecard has **saturated at grade A** and you need to decide whether — and
  how — to make the metric harder.
- After a benchmark/feature lands that should move a surface, to prove it did.
- On a **/loop cadence** across the portfolio: each tick, pick the surface with the
  most room, drive it 2×; when every surface is clean, harden the weakest bar.
