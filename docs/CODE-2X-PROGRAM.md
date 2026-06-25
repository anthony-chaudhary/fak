---
title: "fak code-2x program — the code-debt trajectory ledger"
description: "The durable trajectory ledger for fak's code-quality RSI loop: the baseline, the target, and what each pass actually moved to halve code-debt, with proof."
---

# The code-2x program — a consistent quality measure, baselined and halved

The code-2x program is fak's trajectory ledger for recursively halving code-debt: the hand-kept record of the baseline, the target, and what each quality pass actually moved, with the evidence that proves it. Code-debt is the count of concrete, re-derivable HARD defects that `tools/code_quality_scorecard.py` scores from disk and the Go toolchain — never from a self-report — so the same tree always yields the same number. Because a mature codebase's composite score caps at 100 and cannot double, the "2×" target lives on the debt axis: cut defects by at least half, then halve again. Each pass is committed by explicit path and witnessed with `dos commit-audit`, so the ledger records improvements that were proven, not asserted.

> The durable ledger for the code-quality RSI loop. The living per-KPI snapshot
> is [`CODE-QUALITY-SCORECARD.md`](CODE-QUALITY-SCORECARD.md) (auto-regenerated
> each pass); this file is the hand-kept *trajectory* — baseline, target, and
> what each pass actually moved, with the evidence that proves it.

## The measure

`tools/code_quality_scorecard.py` scores the Go module on **ten deterministic,
evidence-grounded KPIs** and folds them into a composite score (0–100) plus the
headline metric, **code-debt**: the count of concrete, re-derivable HARD defects.
Same tree → same number, because every unit is re-derived from disk and the Go
toolchain, never from a self-report.

| KPI | HARD/SOFT | a unit of code-debt |
|---|---|---|
| `build` | HARD | `go build ./...` ≠ exit 0 |
| `vet` | HARD | a `go vet ./...` diagnostic |
| `format` | HARD | a `gofmt -l` unformatted file |
| `deps` | HARD | an external `go.mod` require (zero-dep invariant broke) |
| `honesty` | HARD | a `CLAIMS.md` `- [` line not carrying exactly one tag |
| `architecture` | HARD | an egregious god-file (>1500 ln) / god-function (>200 ln) |
| `tests` | HARD | a non-trivial package (≥4 funcs) with no `_test.go` |
| `ship_integrity` | HARD | a `dos review` RESIDUAL commit — a claim the diff can't witness |
| `godoc` | SOFT | an undocumented exported symbol (advisory — anti-gaming) |
| `hygiene` | SOFT | a `TODO/FIXME/HACK/XXX` marker (advisory — anti-gaming) |

**Why two KPIs are SOFT.** The cheap way to move `godoc` or `hygiene` is *gaming,
not quality*: doc-comment spam, or deleting a `// TODO` to hide a gap rather than
close it — and this repo keeps an honest `[STUB]` ledger where a "TODO: implement"
marker is a feature of that honesty. They score (a doc-poor or marker-heavy tree
grades lower) but never gate, so the loop is never rewarded for cosmetic churn.

**The DOS grounding.** `ship_integrity` reads `dos review --json` and counts each
RESIDUAL commit — a subject the diff could not back — as debt, from evidence the
committing agent could not author. Every commit in this program is additionally
witnessed with `dos commit-audit HEAD` (must print `[diff-witnessed]`).

## The 2× target (honest framing)

On an already-mature codebase a composite score sits high (~70–85) and **cannot**
double — it caps at 100. So the meaningful "2×" lives on the **debt axis**: cut
code-debt by ≥50% (≥2× fewer defects), exactly how this repo frames its docs
program ("cut doc-debt 100×"). Halve, then halve again — recursive.

## Trajectory

| Date | Pass | Score | Code-debt | `format` | `tests` | `architecture` | `ship_integrity` | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 2026-06-22 | **baseline** | 76.6 (C) | **34** | 4 | 23 | 6 | 1 | commit `e908f92` (the measure) |
| 2026-06-22 | pass 1 | 83.4 (B) | **7** | 0 | 0 | 6 | 1 | `4d8bc04` (fmt), `de07750` (tests) |

**Pass 1 result: code-debt 34 → 7 — a 4.9× reduction (≫ the 2× target).** Score
76.6 → 83.4 (C → B).

What moved it, and why each is genuine (not a gamed number):

- **`format` 4 → 0** — `gofmt -w` on four unformatted files (semantics-preserving;
  `go build ./...` stayed green). Commit `4d8bc04`, `[diff-witnessed]`.
- **`tests` 23 → 0** — one table-driven unit test added to every untested
  non-trivial package, each exercising a *pure* helper (`parseInts`, `lcgIDs`,
  `argmax`, `prefillTokens`, `ratio`, …) with hand-computed expected values that
  fail on regression. **Validated red→green under WSL** (`go test ./cmd/...
  ./internal/webbench/...` exit 0) — the run caught one wrong expected value
  (`demorace` single_agent, `D+R=9` not `6`) and it was fixed to match verified
  behavior. No fabricated `assert-true` tests; functions needing a GPU / model
  file / network were deliberately left untested. Commit `de07750`,
  `[diff-witnessed]` ("test claim witnessed by a touched test file").

## Remaining debt (next passes)

- **`architecture` (6)** — 2 egregious god-files (`internal/ggufload/gguf.go`
  2298 ln, `internal/model/weights.go` 1588 ln) + 4 god-functions in `cmd/*`
  mains (`cmdServe` 246, `fakchat:main` 280, `modelbench:main` 458,
  `simpledemo:main` 315). These are *real refactors* (extract a god-function's
  body into a testable helper — which also adds a `tests` unit), and per the
  `/quality-score` skill's safety rules they belong in a **focused pass**, not a
  drive-by during a test sweep. Deferred deliberately.
- **`ship_integrity` (1)** — a historical commit (`3df9627`, a regenerated
  social-preview PNG the diff can't witness). Not fixable after the fact; it ages
  out of the `HEAD~20..HEAD` window as the trunk advances. The discipline is to
  never *add* residual — every commit here is `dos commit-audit`-clean.

## Measure integrity (adversarial hardening pass)

A quality measure is only worth trusting if it resists *gaming* — moving the
number without improving the code. After pass 1, a 4-lens adversarial review (19
agents: attack → independent verify) was run against the scorecard itself. It
confirmed **12 real defects**; all were fixed (`tools/code_quality_scorecard.py`,
+11 unit tests, ruff-clean):

- **Scanner un-gamed (HIGH).** The AST-free function-length scan counted raw
  `{`/`}` after only stripping `//`, so one `s := "}"` line collapsed a 250-line
  god-function to length ~3 (erasing architecture debt) — and a stray `{` in a
  literal could forge one. Replaced with a literal/comment-aware lexer
  (`_code_only`) that blanks string/rune/backtick/`/* */` spans across lines, and
  a net-per-line depth scan that no longer early-breaks on a balanced
  `interface{}` in a multi-line signature.
- **Empty `_test.go` no longer credits a package (HIGH).** `tests` now requires a
  real `Test`/`Benchmark`/`Fuzz`/`Example` function, not a bare `package foo`
  marker file.
- **Triviality gate fixed (HIGH).** The test-debt gate double-counted exported
  funcs and folded in types/vars; now counts function declarations once.
- **Toolchain/`dos` absence fails *open* (HIGH).** A missing `go`/`gofmt`/`dos`
  now scores the KPI *skipped* (100, soft "unmeasured" note), not a build
  failure — so a box without the toolchain doesn't grade the same tree lower.
- **`deps` catches `replace`-to-external (MED)**, **`honesty` ignores `- [`
  inside fenced code blocks (MED)**, and the determinism contract now documents
  that `ship_integrity` is HEAD-relative (history, not tree).

Three findings were *refuted* by the verify stage (a god-file split into two
honest files is a legitimate fix, not gaming; CRLF gofmt handling; `//` inside a
string) — so the score's anti-gaming posture is itself now witnessed, not
asserted.

## Repeat it

`/quality-score` (`.claude/skills/quality-score/SKILL.md`) is the repeatable RSI
pass: measure → retire the heaviest safe debt → re-measure to prove the drop →
DOS-witness the commit → commit by explicit path. Run it after a burst of new
`cmd/*` tools (they land untested), to drive the next halving, or on a `/loop`
cadence to keep the kernel from re-accreting debt.
