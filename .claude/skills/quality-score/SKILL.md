---
name: quality-score
description: One repeatable RSI pass over CODE quality — the code-side counterpart of refresh-readme. Runs the code-quality scorecard (tools/code_quality_scorecard.py), reads the code-debt work-list, retires debt worst-first using ONLY the safe, genuine classes (gofmt, real tests for untested packages, safe god-function extraction), re-measures to PROVE the number dropped, grounds the ship in DOS (dos commit-audit on the new commit, dos review for the ship_integrity KPI), and commits by explicit path. Use to baseline code quality, drive the code-2x program (halve code-debt, then halve again), or on a /loop cadence to keep the kernel from rotting. The code's checking layer, the way refresh-readme is the README's.
disable-model-invocation: false
user-invocable: true
allowed-tools: Read, Bash, Write, Edit, Grep, Glob
argument-hint: "[--no-toolchain] [--no-dos] [--target N]  (no args = full measure + one improvement iteration)"
---

# /quality-score — make "better code" provable, then prove it moved

> **What this does.** The docs have a measuring stick (`docs_scorecard.py` +
> `/refresh-readme`); until this skill, the *Go code* had none — so "cleaner
> code", "better architecture", "more tested" were unfalsifiable claims with no
> number to move. This is the code's checking layer. It makes "improve the
> code" a **repeatable, evidence-grounded pass**: measure the code-debt, retire
> it worst-first with safe genuine fixes, then re-measure to prove the number
> dropped — and ground the proof in DOS, not in a sentence.

The shape: **run the scorecard → read the code-debt work-list → retire debt
worst-first (safe classes only) → re-measure and prove the drop → DOS-witness
the commit → commit ONLY your lane by explicit path.**

The headline metric is **code-debt**: the count of concrete, re-derivable HARD
defects across ten KPIs. The program target is the **code-2x**: halve code-debt,
then halve again — recursive self-improvement with a number that can't lie,
because every unit is re-derived from disk and the Go toolchain on each pass.

---

## The measure (ten KPIs, each 0–100, deterministic — same tree → same score)

`tools/code_quality_scorecard.py` folds these into a composite score + the
code-debt integer. **HARD** KPIs emit code-debt (they gate `ok`); **SOFT** KPIs
score but never gate.

| KPI | HARD/SOFT | one unit of code-debt is… |
|---|---|---|
| `build` | HARD | `go build ./...` does not exit 0 |
| `vet` | HARD | a `go vet ./...` diagnostic |
| `format` | HARD | a file `gofmt -l` reports as unformatted |
| `deps` | HARD | an external require in `go.mod` (or a present `go.sum`) — the zero-dep invariant broke |
| `honesty` | HARD | a `- [` line in `CLAIMS.md` not carrying exactly one `[SHIPPED]/[SIMULATED]/[STUB]` tag |
| `architecture` | HARD | an *egregious* god-file (>1500 lines) or god-function (>200 lines) |
| `tests` | HARD | a non-trivial package (≥4 funcs) with zero `_test.go` |
| `ship_integrity` | HARD | a RESIDUAL commit in `dos review` — a claim the diff could not witness |
| `godoc` | **SOFT** | an undocumented exported symbol (advisory — see anti-gaming) |
| `hygiene` | **SOFT** | a `TODO/FIXME/HACK/XXX` marker in shipped code (advisory — see anti-gaming) |

**`godoc` and `hygiene` are SOFT on purpose.** The cheap way to move either is
*gaming, not quality*: doc-comment spam (`// X does X`) for godoc, and DELETING
a `// TODO` (hiding the gap, not closing it) for hygiene — and this repo keeps
an honest `[STUB]` ledger where a "TODO: implement" marker is a *feature* of
that honesty. They score (a doc-poor or marker-heavy tree grades lower) but they
never gate, so the pass is never rewarded for cosmetic churn. Same WARN/HARD
split `docs_scorecard` draws for jargon.

---

## Step 1 — Run the scorecard (it builds your work-list)

From the repo root:

```bash
python tools/code_quality_scorecard.py            # human scorecard (per-KPI + work-list)
python tools/code_quality_scorecard.py --json     # machine payload (the loop uses this)
```

It exits non-zero whenever code-debt > 0. Read `corpus.score`, `corpus.code_debt`,
and `corpus.breakdown` (per-KPI debt, worst first). The per-KPI `defects` arrays
ARE the work-list.

Fast/static-only mode (skips `go build`/`vet`/`gofmt`): `--no-toolchain`. Skip
the DOS probe: `--no-dos`. The default (full) is what you commit against.

**Record the baseline.** Before touching anything, write down `code_debt = N` and
`score = S`. The whole point is to prove the delta, so you need the before-number.

## Step 2 — Retire code-debt worst-first, using SAFE GENUINE classes only

Attack the heaviest KPI first (`corpus.breakdown[0]`), but only via fixes that
are *safe on a shared trunk* and *genuine* (they improve the code, not just the
number). In rough order of safety:

- **`format` (safest).** `gofmt -w <the flagged files>`. Semantics-preserving;
  `go build ./...` stays green. Confirm only those files are clean and not
  peer-modified first (`git status --short <files>`).
- **`tests` (safe + high-value).** For each untested package, find a PURE,
  deterministic function (math, parsing, formatting, scoring — no GPU, network,
  or model-file) and write ONE real, table-driven test with assertions that can
  actually FAIL. Same package clause (`package main` for `cmd/*`). **Be honest:**
  if a package is only a `main()` wiring flags + I/O into already-tested
  `internal/` packages, it has no testable seam — leave it and say so. A
  fabricated `assert true` test is worse than no test, and it is exactly the
  gaming the SOFT KPIs exist to refuse. Validate (Step 3) before trusting it.
- **`vet` / `build`.** Fix the diagnostic at its cause; never silence it.
- **`architecture` (RISKY — care).** Extract a god-function's body into a
  testable helper (which also retires a `tests` unit). This is real refactoring:
  do it in ONE lane, keep behavior identical, and validate hard. If you can't do
  it safely this pass, leave it for a focused pass — do NOT half-do a refactor.
- **`ship_integrity`.** A RESIDUAL is usually a *historical* commit (e.g. a
  regenerated binary asset the diff can't witness); it ages out of the window as
  the trunk advances. You don't "fix" someone else's past commit — you ensure
  YOUR commits are diff-witnessed (Step 5), so you never *add* residual.

Never edit the frozen ABI (`internal/abi`) or add a dependency to move a number.

## Step 3 — Validate every change (the honesty gate)

Native `go test` is OS-blocked on the Windows dev box, so validate under WSL:

```bash
go build ./...                                   # compiles (catches structural breaks)
go vet ./...                                     # typechecks incl. _test.go files
wsl -e bash -lc 'cd /mnt/c/work/fak && go test ./<changed-pkg>/... -count=1'
```

A new test you have not RUN is not done — it might not even compile. Run it.
If it fails or won't build, fix or drop it; never ship a red/uncompiling test.
Re-run `gofmt -l <your new files>` — your additions must be gofmt-clean (the
`format` KPI will catch them next pass otherwise).

## Step 4 — Re-measure and PROVE the drop

```bash
python tools/code_quality_scorecard.py --json
```

Confirm `code_debt` dropped from N toward your target and `score` rose. State the
delta plainly: `code-debt N → M (−k), score S → S'`. The code-2x bar for a pass
is **halving the actionable hard-debt** (doubling a 70-ish score is impossible —
it caps at 100, so the honest "2×" lives on the debt axis, the way the docs
program states "cut doc-debt 100×"). Regenerate the committed snapshot:

```bash
python tools/code_quality_scorecard.py --markdown --stamp YYYY-MM-DD > docs/CODE-QUALITY-SCORECARD.md
```

(Use the Bash tool's `>` — it preserves UTF-8. A PowerShell `>` re-encodes to
UTF-16 and mangles the `·`/`×`/`—` glyphs.)

## Step 5 — Ground the ship in DOS, then commit ONLY your lane

The `ship_integrity` KPI already reads `dos review` — but witness YOUR OWN
commit too, so "I fixed it" is backed by evidence the committing agent can't
author:

```bash
git pull --no-rebase --no-edit                   # merge integrates fine alongside dirty peer files
git add <your explicit paths>                    # NEVER git add -A on this shared tree
git commit -s -m "<conventional subject>" -m "<body: N→M debt, what changed>" -m "(fak <leaf>)"
dos commit-audit HEAD                             # MUST print [diff-witnessed] / verdict OK
git push
```

- **Subject honesty.** New tests + fmt → `test(<scope>):` or `style(fmt):`. A
  refactor that changes structure → `refactor(<scope>):`. Don't use `feat(`/`fix(`
  for a test/format pass — keep the prefix honest to what changed. End every
  ship commit with a `(fak <leaf>)` trailer so the `dos verify` referee binds it.
- **`dos commit-audit HEAD` is the gate.** If it says `subject-only` /
  CLAIM_UNWITNESSED, your diff doesn't back your subject — reword the subject to
  what the diff actually did (or add the missing change). A `[diff-witnessed]`
  verdict is the green light. This is the same rung the `ship_integrity` KPI
  scores, applied to your own work.
- **Shared-trunk discipline.** Stay on `main` (the `OFF_TRUNK` guard refuses a
  branch). Commit by explicit path. If a peer's `MERGE_HEAD` is set
  (`cannot do a partial commit during a merge`): wait for it to clear, then
  re-try the pathspec commit. Never force-push.

---

## The RSI loop (why this is a self-improvement skill, not a one-shot)

Each pass: measure → retire the heaviest safe debt → prove the drop → commit
witnessed. Run it again next pass and the *new* heaviest KPI surfaces — the loop
walks the debt down monotonically because the number is re-derived from disk
every time, so it can't drift or be talked past. Halve, then halve again. On a
`/loop` cadence it keeps the kernel from re-accreting debt between releases.

The discipline that makes it RSI and not vibes: **every pass moves a number that
is grounded in evidence the agent did not author** — the Go toolchain's exit
codes (`build`/`vet`/`format`), a `go test` red→green under WSL (`tests`), and
`dos review` / `dos commit-audit` (`ship_integrity`). The score is the gauge;
the witnesses are why you can trust it moved.

## Anti-gaming laws (internalize these — the measure is only as honest as the pass)

1. **Never write a vacuous test to drop `tests` debt.** A test must be able to
   FAIL on a real regression. No testable seam → leave it and say so.
2. **Never delete a `// TODO` to "improve" `hygiene`.** It's SOFT for this exact
   reason — close the gap or leave the honest marker.
3. **Never spam doc comments to move `godoc`.** Also SOFT, also on purpose.
4. **Never silence a `vet` diagnostic or `//nolint` past it.** Fix the cause.
5. **Never add a dependency or touch the frozen ABI** to move a number.
6. **Re-measure with the FULL scorecard** (not `--no-toolchain`) before claiming
   a drop, and **`dos commit-audit` your commit** before claiming it shipped.

## When to run this

- To **baseline** code quality (first run records score + code-debt).
- To drive the **code-2x program** — one halving per focused pass.
- After a burst of new `cmd/*` tools or benchmarks (they land untested → `tests`
  debt spikes; this pass retires it genuinely).
- On a `/loop` cadence to keep the kernel from rotting between releases.

The scorecard is read-only; this skill's only writes are your genuine fixes,
`docs/CODE-QUALITY-SCORECARD.md`, and the tool itself. It never edits the frozen
ABI and never games a SOFT KPI.
