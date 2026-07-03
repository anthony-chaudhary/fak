---
name: milestone-score
description: One repeatable pass that makes the milestone report's CLIMB and ROADMAP retirable by the RSI loop — the milestone counterpart of quality-score (code) and stability-score (trust under iteration). Runs the milestone scorecard (`fak milestone-scorecard --json`) over the report's OWN two dimensions — the distance-from-MATURED climb shortfall across the M0..M7 support-maturity grid PLUS the un-progressed tracked-epic roadmap gaps — folds them into one deterministic milestone_debt integer + a worst-first milestone_worklist, retires debt worst-first (climb the lowest-rung cell to M4, then close the most-open discrete epic), re-pins the climb ratchet on a real climb improvement, re-measures with --compare to PROVE the debt dropped, and commits only the milestone lane by explicit path. COMPOSES — does NOT duplicate — support-maturity (which fences each cell to its regime ceiling); milestone_debt scores raw distance-to-MATURED across the grid as the headline climb, alongside the roadmap. Use after a cell climbs a rung, after an epic closes its children, on the milestone report cadence, or on a /loop cadence to keep the milestone program retirable like every other surface.
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
---

# /milestone-score — make the milestone climb + roadmap retirable, then retire it

> **What this does.** The milestone report (`fak milestone report`) RECORDS the
> project's position — the M0..M7 maturity grid and the tracked-epic roadmap — but a
> trend report does not grade a retire-able debt, so the RSI loop could not drive it
> like it drives code-quality or stability. This pass gives the report its scorecard
> surface: a deterministic `milestone_debt` integer and a worst-first worklist, then
> retires that debt one real climb or epic-close at a time.

It is an instance of the generic [`scorecard`](../scorecard/SKILL.md) doctrine pointed
at one surface: **the milestone program's climb to MATURED and its roadmap convergence**.
The tool is the Go verb `fak milestone-scorecard` (cmd/fak/milestonescorecard.go over
internal/milestonereport), and it folds into `tools/scorecard_control_pane.py` next to
the rest of the family — registered as `milestone` (debt `milestone_debt`) alongside the
`milestone_climb` ratchet gate.

## The two dimensions (and why this is NOT a parallel maturity scale)

`milestone_debt = climb_debt + roadmap_debt`, folded from the SAME two dimensions the
report already folds — no re-derived scale:

- **climb** — `climb_debt` = sum over every cell below the MATURED floor (M4 Correct) of
  `(M4 - current_rung)` rung-steps. One defect per owed rung-step, so the kernel's
  `len(Defects)` count fold equals the rung-weighted shortfall. This is the report's OWN
  matured-floor framing (raw distance-to-MATURED across the grid), NOT the
  declared-target/regime-ceiling framing `support_maturity_debt` uses — so the two
  scorecards stay distinct: support-maturity fences each cell to its regime ceiling and
  scores the declared-target shortfall; milestone scores the headline project climb to a
  single MATURED bar. They COMPOSE, they do not duplicate.
- **roadmap** — `roadmap_debt` = sum over the MEASURED DISCRETE tracked epics of their
  open children (`total - closed`). An ongoing PROGRAM (kernel-optimization,
  cache-optimization) has no 100% so it never owes roadmap debt; an UNREADABLE epic is
  excluded — you cannot retire what you cannot measure, and that gap is the report's
  ACTION verdict's job, not the scorecard's.

The worst-first `milestone_worklist` orders climb cells by rung shortfall (lowest rung,
biggest gap, first) ahead of roadmap gaps by open-child count — the climb is the deeper,
structural debt, so it retires first.

## Run it as an RSI pass (the five steps)

1. **Run it** — `fak milestone-scorecard --json` for the control-pane payload (read
   `corpus.milestone_debt` + `corpus.milestone_worklist`); `fak milestone-scorecard` for
   the rendered snapshot. Save a baseline first:
   `fak milestone-scorecard --json > /tmp/milestone-before.json`. The roadmap half resolves
   each tracked epic's children live via `gh`; to fold hermetically (offline / no `gh`),
   pass `--epics-from <file>` with a data file carrying a pre-resolved `counts` block.
2. **Retire `milestone_debt` worst-first** — drain `corpus.milestone_worklist` top-down:
   climb the lowest-rung cell up to MATURED (M4) — fence it, run it, then prove it
   correct — then close the most-open DISCRETE epic's children. Never move a cell to M4
   by re-labeling; a rung climb is witnessed by the cell actually meeting the rung's bar.
3. **Re-pin the climb ratchet on a REAL climb** — the two witnessed climb KPIs
   (`matured_cells`, `milestone_progress`) are pinned in
   `docs/milestones/baseline.json` and a regression in EITHER reds CI via the
   `milestone_climb` control-pane gate. After a genuine climb improvement, re-pin:
   `fak milestone-scorecard --pin`. Never `--pin` to silence a regression — that ratchets
   the floor DOWN.
4. **Re-measure + prove** — `fak milestone-scorecard --compare /tmp/milestone-before.json`
   prints the debt delta; the worklist must have shrunk and `milestone_debt` must not have
   risen. There is no per-surface markdown snapshot to regenerate (the report itself is the
   snapshot); the proof is the `--compare` delta plus the re-pinned baseline.
5. **Commit only the milestone lane, by explicit path** —
   `git commit -s -- docs/milestones/baseline.json <the cell/epic evidence you moved>`.
   End the subject with the `(fak milestonereport)` trailer (or `(fak <leaf>` for the
   concrete code a rung climb touched). Never `git add -A`.

## The anti-gaming law

Retire a defect by moving the real cell or closing the real epic, never by editing the
grid labels or the tracked-epic set to flatter the number. A climb defect is retired by a
cell genuinely meeting the next rung's bar (witnessed by the report's own collection), and
a roadmap defect is retired by a child issue actually closing — not by un-tracking the
epic or redefining MATURED. If "fixing" a defect would mean relaxing a rung's definition
or dropping an epic from the tracked set, **stop — that is not a real retirement.**

## Re-pinning the climb baseline (handle with care)

The climb ratchet (`milestone_climb`, debt `climb_ratchet_debt`) is a DISTINCT gate from
`milestone_debt`: a cell count that holds while `milestone_debt` rises still reds the
control pane here. Do **not** blindly `--pin` after a milestone-score pass: confirm
`matured_cells` did not decrease and `milestone_progress` did not regress FIRST, then
re-pin only the genuine climb. A blind re-pin blesses a regression the way a portfolio
re-pin hides a per-metric one.
