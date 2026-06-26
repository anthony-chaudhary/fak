---
name: slop-score
description: One repeatable RSI pass over CODE SLOP — the slop the compiler can't see. Runs the code-slop scorecard (tools/code_slop_scorecard.py), reads the slop-debt work-list, retires debt worst-first using ONLY genuine fixes (extract a copy-paste clone into a shared helper, delete or wire a dead unexported symbol, add a real assertion to a vacuous test, drop tautological doc comments + commented-out code), re-measures to PROVE the number dropped, regenerates the committed snapshot, grounds the ship in DOS (dos commit-audit on the new commit), and commits by explicit path. The slop sibling of /quality-score (defects) and /appeal-score (prose voice). Use to baseline code slop, drive slop-debt toward 0 worst-first, or on a /loop cadence to keep the kernel from re-accreting clones, dead weight, and assertion-free tests.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--no-toolchain] [--range A..B] [--target N]  (no args = full measure + one worst-first retirement)"
---

# /slop-score — make "less slop" provable, then prove it moved

> An instance of the generic **[scorecard](../scorecard/SKILL.md)** doctrine (read that
> for the five laws and the RSI loop). It is the slop sibling of **`quality-score`**
> (classic defects: gofmt, vet, untested packages) and **`appeal-score`** (prose voice).
> Where `quality-score` grades the slop the *compiler* and `go vet` already catch,
> this grades the slop they *can't* see: code that builds, vets clean, and has a test
> present, yet rots the kernel from the inside — copy-paste clones, tests that assert
> nothing, dead unexported symbols, commented-out cruft.

The shape: **run the scorecard → read the slop-debt work-list → retire debt worst-first
(genuine fixes only) → re-measure and prove the drop → regenerate the snapshot →
DOS-witness the commit → commit ONLY your lane by explicit path.**

The headline metric is **slop-debt** (`corpus.slop_debt`): the count of concrete,
re-derivable HARD defects across the slop axes, re-derived from disk every pass so it
can't drift or be talked past. Drive it toward zero and "less slop" becomes a number
you moved, not a claim you made. The epic that tracks the program is **#775**; Track A
(drive the debt down) is what this skill runs.

---

## The measure (six axes — four HARD, two SOFT; deterministic, same tree → same score)

`tools/code_slop_scorecard.py` folds these into a `slop_score` (0–100, A–F) and the
`slop_debt` integer. **HARD** axes emit slop-debt (they gate `ok`); **SOFT** axes score
but never gate. `ok` is False iff any HARD defect exists.

| KPI | HARD/SOFT | one unit of slop-debt is… | the genuine fix |
|---|---|---|---|
| `duplication` | HARD | a normalized Go token-window copy-pasted into 2+ distinct sites | extract ONE shared helper both sites call (behavior-preserving) — **#776** |
| `dead_code` | HARD | an unexported symbol defined but referenced nowhere else | delete it, or *wire* it if it has a real intended caller (e.g. `cmdTraj` into `main.go`) — **#777** |
| `vacuous_tests` | HARD | a `Test*` func body that makes zero assertions | add a real assertion that can actually FAIL on a regression — **#778** |
| `comment_slop` | HARD | a tautological doc comment (`// X does X`) or a commented-out code block | delete the cruft (NOT an honest `// TODO`/`[STUB]` marker — see anti-gaming) |
| `stub_masquerade` | **SOFT** | an exported func whose body is only `return nil` / `panic("unimplemented")` | advisory only — the honest `[STUB]` ledger is a *feature*; promotion to HARD is **#781** (gated on a multi-release zero-FP soak), not this pass |
| `churn_bloat` | **SOFT** | recent commits that ADD `.go` files without ever removing any (HEAD-relative) | advisory trend over `--range`, not a tree defect; never gate it, never "fix" it by deleting files |

`Benchmark*` and `Fuzz*` funcs are **not** graded by `vacuous_tests` (a benchmark's job
is to time, not assert) — so never demote a `Test` to a `Benchmark` to dodge the axis.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/code_slop_scorecard.py            # human scorecard (per-KPI + work-list)
python tools/code_slop_scorecard.py --json     # machine payload (the loop reads this)
```

It exits **non-zero whenever slop-debt > 0** (this is the gate the epic wires fully at
zero, #779). Read `corpus.score`, `corpus.slop_debt`, and `corpus.breakdown` (per-KPI
debt, worst first). The per-KPI `defects` arrays ARE the work-list: each entry names the
exact files+lines of a clone group, a dead symbol, or a vacuous test.

The scorecard **walks the working tree** (`rglob("*.go")`, not `git ls-files`) so an
uncommitted change scores — re-measure reflects what's on disk right now. The committed
snapshot `docs/CODE-SLOP-SCORECARD.md` can read LOWER than a live run when HEAD has grown
since the snapshot was last regenerated; the live number is the truth, and Step 4
re-stamps the snapshot to match. Skip the Go toolchain probe (faster, static-only):
`--no-toolchain`. Tune the SOFT churn window: `--range HEAD~40..HEAD`.

**Record the baseline.** Before touching anything, write down `slop_debt = N`,
`score = S`, and the heaviest KPI. The whole point is to prove the delta, so you need
the before-number.

## Step 2 — Retire slop-debt worst-first, using GENUINE fixes only

Attack the heaviest KPI first (`corpus.breakdown[0]`), but only via fixes that are *safe
on a shared trunk* and *genuine* (they improve the code, not just the number):

- **`duplication` (usually heaviest).** For a named clone group, extract the shared
  body into ONE helper (a package-level func, or a small internal package if the sites
  span packages) and have every site call it. Keep behavior byte-identical. Most clones
  live in the `cmd/*bench` and demo mains (flag wiring, `readHFConfig`, `writeFile`,
  output blocks) — those are the high-count groups. Do ONE group per pass and validate
  hard (Step 3); a half-done extraction that changes behavior is worse than the clone.
- **`dead_code`.** A dead unexported symbol is either (a) genuinely orphaned → delete it,
  or (b) something that was meant to be wired and never was (e.g. a `cmdTraj` subcommand
  missing from its `main.go` dispatch) → wire it to its real caller. Decide per symbol by
  reading it; don't blanket-delete something a peer is mid-wiring (check `git log`/blame).
- **`vacuous_tests`.** Add a real, table-driven assertion to the named test — one that
  can FAIL if the function under test regresses. A fabricated `assert true` is worse than
  no assertion and is exactly the gaming the measure exists to refuse.
- **`comment_slop`.** Delete a `// Foo does Foo`-style tautology or a block of
  commented-out code. **Never** delete an honest `// TODO`/`[STUB]` marker — this repo
  keeps a `[STUB]` ledger where that marker is a deliberate honesty feature.

Never edit the frozen ABI (`internal/abi`), add a dependency, or touch the *detector* to
move a number. Hardening the detector (AST clone detection #780, the SOFT→HARD promotions
#781) is deliberate Track-B work with its own issues — it is NOT a way to drop debt.

## Step 3 — Validate every change (the honesty gate)

Native `go test` is OS-blocked on the Windows dev box, so validate under WSL:

```bash
go build ./...                                   # compiles (catches structural breaks)
go vet ./...                                     # typechecks incl. _test.go files
wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./<changed-pkg>/... -count=1'
```

A de-dup extraction or a new assertion you have NOT run is not done — it might not even
compile. Run it; if it fails or won't build, fix or revert it. Re-run `gofmt -l <files>`
so your edits stay format-clean (the `quality-score` `format` KPI catches them otherwise).

## Step 4 — Re-measure, PROVE the drop, regenerate the snapshot

The slop scorecard has **no `--compare` flag** — prove the drop by re-running and reading
`slop_debt` against your recorded baseline:

```bash
python tools/code_slop_scorecard.py --json      # confirm corpus.slop_debt fell N → M
```

State the delta plainly: `slop-debt N → M (−k), score S → S'`. Then regenerate the
committed snapshot so the doc matches the tree, and confirm freshness:

```bash
python tools/code_slop_scorecard.py --markdown --stamp YYYY-MM-DD > docs/CODE-SLOP-SCORECARD.md
python tools/code_slop_scorecard.py --check-doc                  # must exit 0 now
```

(Use the Bash tool's `>` — it preserves UTF-8. A PowerShell `>` re-encodes to UTF-16 and
mangles the `·`/`—` glyphs.) If you lowered the debt, **re-pin the control-pane floor** so
the gain ratchets and a future regression back up is caught:

```bash
python tools/scorecard_control_pane.py --pin     # rewrites tools/scorecard_baseline.json
```

Commit `tools/scorecard_baseline.json` in the same lane (the control pane reads `slop` via
`corpus.slop_debt`).

## Step 5 — Ground the ship in DOS, then commit ONLY your lane

```bash
git pull --no-rebase --no-edit                   # merge integrates alongside dirty peer files
git commit -s -F <msgfile> -- <your explicit paths>   # options BEFORE --, paths AFTER
dos commit-audit HEAD                             # MUST print [diff-witnessed] / verdict OK
git push
```

- **Subject honesty.** A de-dup extraction → `refactor(<scope>): extract … (#776)`. A
  deletion of dead code → `refactor(<scope>): drop dead … (#777)`. New assertions →
  `test(<scope>): assert … (#778)`. Lead with a recognized verb (the commit-message
  witness ABSTAINs on a noun-led subject). End every ship commit with a `(fak <leaf>)`
  trailer so the `dos verify` referee binds it. Cite the Track-A child issue you closed
  (or `#775` for an epic-level increment).
- **`dos commit-audit HEAD` is the gate.** `subject-only` / CLAIM_UNWITNESSED means your
  diff doesn't back your subject — reword to what the diff actually did. `[diff-witnessed]`
  is the green light.
- **Shared-trunk discipline.** Stay on `main` (the `OFF_TRUNK` guard refuses a branch).
  Stage and commit by explicit path in ONE shell call so a peer's bare commit can't sweep
  your staged files — NEVER `git add -A`. If a peer's `MERGE_HEAD` is set
  (`cannot do a partial commit during a merge`), wait for it to clear, then re-try the
  pathspec commit. Never force-push.

---

## The RSI loop (why this is a self-improvement skill, not a one-shot)

Each pass: measure → retire the heaviest genuine debt → prove the drop → regenerate +
re-pin → commit witnessed. Run it again and the *new* heaviest KPI surfaces — the loop
walks slop-debt down monotonically because the number is re-derived from the working tree
every time. The discipline that makes it RSI and not vibes: **every pass moves a number
grounded in evidence the agent did not author** — the scorecard's re-derivation from disk,
a `go test` red→green under WSL, and `dos commit-audit` on the commit.

The full gate (the bare run + `--check-doc` wired into `make demo-scorecards` so a
regression turns CI RED) lands once slop-debt hits 0 — that is **#779**, blocked until
Track A drains the duplication/dead_code/vacuous_tests debt this skill retires.

## Anti-gaming laws (the measure is only as honest as the pass)

1. **Never write a vacuous test to pass `vacuous_tests`, and never demote a `Test` to a
   `Benchmark`/`Fuzz` to dodge it.** An assertion must be able to FAIL on a real regression.
2. **Never "de-dup" by deleting a copy that's actually needed at both sites.** Extract a
   shared helper both call; validate build/vet/test. Deleting working code to drop a clone
   count is a regression wearing a green number.
3. **Never delete an honest `// TODO` / `[STUB]` marker** to move `comment_slop` or
   `stub_masquerade` — those are deliberate honesty features (the latter is SOFT for this
   exact reason).
4. **Never touch the detector or the baseline to drop debt.** Detector hardening (#780,
   #781) is deliberate Track-B work with its own gates, not a debt-dodge. Lower the floor
   (`--pin`) only AFTER a genuine drop you proved.
5. **Never add a dependency or edit the frozen ABI** to move a number.
6. **Re-measure with the FULL scorecard** (not `--no-toolchain`) before claiming a drop,
   and **`dos commit-audit` your commit** before claiming it shipped.

## When to run this

- To **baseline** code slop (first run records `slop_debt` + score).
- To drive **slop-debt toward 0 worst-first** (the #775 Track-A program), one KPI per pass.
- After a burst of new `cmd/*` benches/demos (they land with copy-pasted flag-wiring and
  output blocks → `duplication` spikes; this pass retires it genuinely).
- On a `/loop` cadence to keep the kernel from re-accreting slop between releases.

The scorecard is read-only; this skill's only writes are your genuine fixes,
`docs/CODE-SLOP-SCORECARD.md`, `tools/scorecard_baseline.json`, and the tool itself. It
never edits the frozen ABI, never games a SOFT axis, and never weakens the detector to score.
