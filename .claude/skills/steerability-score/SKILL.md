---
name: steerability-score
description: One repeatable pass that keeps fak as STEERABLE as it grows — the one scorecard whose every KPI is growth-invariant, so a 2x-larger repo with the same discipline scores the same. Runs the steerability scorecard (tools/steerability_scorecard.py) over the working tree, reads the 0-100 steerability index + the advisory drift signals (coupling hubs, p90 sizes, long-function rate, package drift, churn hot spots), drives the index UP and the worst drift axis DOWN by adding REAL modularity (split a cmd dispatch monolith along its verb seams, break a coupling hub, document a package header) — never by gaming a detector, re-measures to PROVE the index rose, and commits only the scorecard lane by explicit path. The growth-invariant counterpart of code-quality (absolute defects) and repo-hygiene (tree shape). Use after a structural change (a new package, a coupling edge into a hub, a growing dispatch file), when the project "feels" harder to change than its size warrants, or on a /loop cadence to keep steering effort flat as the kernel grows.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--range HEAD~N..HEAD] [--target INDEX]  (no args = full measure + one improvement iteration)"
---

# /steerability-score — keep steering effort flat as the repo grows

> **What this does.** Every sibling scorecard reports an absolute *count* of
> defects, which mechanically climbs as the repo grows — a 3×-bigger tree has ~3×
> the surfaces. None of them answers the question this skill owns: as fak doubles
> in size, does the **effort to steer, change, and navigate it stay roughly flat**,
> and if it drifts, do we *know* and can we *correct*? This is that pass. It is an
> instance of the shared [`scorecard`](../scorecard/SKILL.md) doctrine, pointed at
> the one surface measured in *shape*, not *size*.

The shape: **run the scorecard → read the index + the worst drift axis → raise the
index by adding REAL modularity (never gaming) → re-measure and prove the index rose
→ commit ONLY your lane by explicit path.**

The headline is a 0–100 **steerability index** — a weighted mean of growth-invariant
KPIs — not a debt pile. That is the design: a count would trend just from getting
bigger, so it can't answer "is steering effort *flat*." The index can. Drive it UP.

---

## The measure (eleven KPIs, four groups, all growth-invariant)

`tools/steerability_scorecard.py` folds these into the index + a `steerability_debt`
integer (for control-pane membership). **HARD** KPIs emit debt; **SOFT** KPIs score
the index but never emit debt — because their only cheap fix would be gaming.

| Group | KPI | HARD/SOFT | what it measures (growth-invariant form) |
|---|---|---|---|
| modularity | `file_size_dist` | SOFT | p90 file length vs a fixed reference (percentile — scale-free) |
| modularity | `func_size_dist` | SOFT | fraction of functions over the soft length line (a rate) |
| modularity | `god_file_rate` | SOFT | rate of files over the hard ceiling (code_quality owns the COUNT) |
| modularity | `god_func_rate` | SOFT | rate of functions over the hard ceiling (code_quality owns the COUNT) |
| coupling | `fan_in_gini` | SOFT | Gini of the internal-import fan-in graph (flat = steerable) |
| coupling | `hub_share` | SOFT | the single most-depended-on package's share of all packages |
| coupling | `dispatch_god_file` | **HARD** | a `cmd/*` dispatch file over the hard ceiling |
| navigability | `package_doc_frac` | SOFT | fraction of packages with a `// Package x` doc-comment |
| correction | `ratchet_present` | **HARD** | the control-pane baseline parses + this scorecard is wired |
| correction | `worst_pkg_drift` | SOFT | worst package's LOC growth vs the pinned baseline |
| correction | `churn_concentration` | SOFT, HEAD-relative | Gini of recent-commit churn (a hot spot) |

**Why only two HARD KPIs, both ~0 on a disciplined tree.** This scorecard stays
ORTHOGONAL to `code_quality`: god-files and tests are already *that* scorecard's HARD
debt, and both fold into the same control-pane total. Re-emitting them here would
double-count the same monolith in the portfolio sum. So steerability **scores**
size/coupling on the invariant rate (SOFT) and leaves the raw count to `code_quality`.
The only things it emits debt for are the two whose cheapest fix is genuinely real
work — splitting a dispatch monolith (`dispatch_god_file`) and committing the
correction ratchet (`ratchet_present`). On a healthy tree both are zero, and the
signal lives in the **index** and the **drift signals**. That is correct, not a bug.

`churn_concentration` is the one **HEAD-relative** KPI (the `code_quality.ship_integrity`
precedent): it reads recent git history, so its number moves as commits land even on a
byte-identical tree. Pin `--range HEAD~N..HEAD` for a stable read. It can never anchor
the baseline.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/steerability_scorecard.py            # human scorecard (index + per-KPI + drift signals)
python tools/steerability_scorecard.py --json      # machine payload (the loop uses this)
```

Read `corpus.index`, `corpus.index_by_group`, and `corpus.breakdown` (per-KPI, worst
first). The per-KPI `soft` arrays are the **drift work-list** — where steerability is
heading even when no hard debt exists yet. **Record the baseline:** write down
`index = S` and the worst group, because the whole point is to prove the delta.

## Step 2 — Raise the index worst-axis-first, using REAL structural moves only

Attack the lowest group score (`corpus.index_by_group`), then the lowest KPI in it.
The genuine moves, by KPI:

- **`dispatch_god_file` (the one HARD coupling defect).** A `cmd/*` file over the
  ceiling means every new verb fights the same monolith. Split the verb table into
  per-command files (the [`modularize`](../modularize/SKILL.md) move: behavior-
  preserving code motion along real seams). This is the highest-value steerability
  fix — it directly cuts the blast radius of adding a command.
- **`hub_share` / `fan_in_gini` (coupling, SOFT but the heart of steerability).** A
  package imported by a large fraction of the tree is a chokepoint every change routes
  through. The real fix is to **break the dependency**: extract the narrow interface
  the importers actually need into a small, stable package, or invert the dependency
  so the hub no longer has to change when its consumers do. NEVER add a façade
  re-export package to split the hub's name while the real coupling is unchanged —
  that games the Gini without improving steerability.
- **`file_size_dist` / `func_size_dist` (modularity, SOFT).** Split a large file along
  a concern seam; extract a long function into a named helper. Same discipline as
  `modularize`. A cosmetic split with no real seam games a percentile and is refused.
- **`package_doc_frac` (navigability, SOFT).** Add a *real* `// Package x …` header
  that says what the package is for and when to reach for it — one sentence a reader
  or agent uses to orient. NEVER `// Package x provides x.` spam: that games the
  fraction without aiding navigation (the `godoc` lesson). If a package's purpose is
  genuinely obvious from its name, leave it and say so.
- **`ratchet_present` (the one HARD correction defect).** If it fires, the control-pane
  baseline is missing/malformed or this scorecard isn't wired into the fold. Fix it by
  committing a real baseline (`python tools/scorecard_control_pane.py --pin`) and the
  `SCORECARDS` row — the genuine correction affordance, not a `touch`.

`worst_pkg_drift` and `churn_concentration` are *advisory lenses*, not fix targets:
they tell you WHERE to look (a ballooning package, a churn hot spot), and the fix is
one of the moves above applied there.

## Step 3 — Validate every structural change (the honesty gate)

Any code motion (splitting a file, extracting a helper, breaking a coupling edge) must
keep behavior identical. Native `go test` is OS-blocked on the Windows dev box, so
validate under WSL:

```bash
go build ./...                                   # compiles (catches a broken import graph)
go vet ./...
wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./<changed-pkg>/... -count=1'
```

A refactor you have not built and tested is not done. Never edit the frozen ABI
(`internal/abi`) or add a dependency to move a number.

## Step 4 — Re-measure and PROVE the index rose

```bash
python tools/steerability_scorecard.py --json
python tools/steerability_scorecard.py --compare baseline.json   # the index/debt delta + verdict
```

State the delta plainly: `index S → S' (+k)`, and the group that moved. Pin `--range`
to the SAME window you measured the baseline with, so the HEAD-relative churn KPI
doesn't masquerade as your improvement. Regenerate the committed snapshot:

```bash
python tools/steerability_scorecard.py --markdown --stamp YYYY-MM-DD > docs/STEERABILITY-SCORECARD.md
```

(Use the Bash tool's `>` — it preserves UTF-8; a PowerShell `>` re-encodes to UTF-16
and mangles the `·`/`×`/`—` glyphs.)

## Step 5 — Commit ONLY your lane, by explicit path

```bash
git pull --no-rebase --no-edit
git commit -s -F msg -- tools/steerability_scorecard.py tools/steerability_scorecard_test.py docs/STEERABILITY-SCORECARD.md
dos commit-audit HEAD                             # MUST print [diff-witnessed] / verdict OK
git push
```

If your pass did real refactoring in a kernel package, commit that package in its OWN
lane (the `modularize` discipline), separate from the scorecard lane. Subject honesty:
a refactor → `refactor(<scope>):`; a scorecard-only measure/snapshot → `chore(steer):`
or `docs(steer):`. End every ship commit with a `(fak steerability)` (or the touched
leaf's) trailer so the `dos verify` referee binds it. Stay on `main`; never force-push.

---

## The RSI loop

Each pass: measure → raise the lowest group → prove the index rose → commit witnessed.
Next pass, the *new* lowest group surfaces — the loop walks steerability UP because the
index is re-derived from the tree every time and can't be talked past. Because every
KPI is growth-invariant, **a clean pass holds the index flat even as the repo grows** —
which is the entire goal: the same level of steerability at 2× the size.

## Anti-gaming laws (the index is only as honest as the pass)

1. **Never add a façade / re-export package to split a coupling hub's name.** The real
   coupling is unchanged; you've gamed the Gini. Break the dependency for real.
2. **Never do a cosmetic file/function split with no concern seam** to move a
   percentile. Split along a real boundary or leave it.
3. **Never spam `// Package x provides x.`** to move `package_doc_frac` — it's SOFT for
   exactly this reason. Write a header that actually orients, or leave it.
4. **Never re-emit god-file/test debt here.** `code_quality` owns that count; this
   scorecard scores the RATE. Double-counting the same monolith is the wiring bug this
   orthogonality exists to prevent.
5. **Never let the HEAD-relative `churn_concentration` masquerade as a structural win.**
   Pin `--range` to the baseline's window before claiming the index moved.
6. **Re-measure and `dos commit-audit` your commit** before claiming the index rose.

## When to run this

- To **baseline** steerability (first run records the index + the worst group).
- After a **structural change**: a new package, a new import edge into a hub, a growing
  `cmd/*` dispatch file, a package that's ballooning.
- When the project **feels harder to change than its size warrants** — the index tells
  you which axis (coupling, modularity, navigability) is the culprit.
- On a `/loop` cadence to keep steering effort flat as the kernel grows between releases.

The scorecard is read-only; this skill's only writes are your genuine structural fixes,
`docs/STEERABILITY-SCORECARD.md`, and the tool itself. It never edits the frozen ABI and
never games a SOFT KPI.
