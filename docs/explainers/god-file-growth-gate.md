---
title: "The god-file growth gate: a ratchet, not a sweep"
description: "A monolith is never written in one commit — it accretes one line at a time, faster than anyone pays it down, because nothing refuses the next line. Hermes' gateway/run.py reached 20,320 lines that way. fak's scorecard names the debt line (1500/file, 200/function); GOD_FILE_GROWTH turns the doctrine preventive — a CI ratchet that grandfathers today's offenders at-size and refuses new growth. Files stay firm at 1500 (aligned with the file-only sibling gate); functions block loose at 400, and a bounded slack band lets a grandfathered offender absorb an ordinary edit — so the surface stays bounded without false-blocking a plausible change to already-large code."
slug: god-file-growth-gate
keywords:
  - god file
  - god function
  - code size ceiling
  - CI ratchet
  - architecture debt
  - refactor gate
  - monolith
  - steerability
  - modularize
date: 2026-07-05
---

# The god-file growth gate: a ratchet, not a sweep

*For anyone running a coding-agent fleet on a shared trunk, or wondering why a size
limit that already exists as a scorecard number needs a second life as a gate. No
setup needed to follow along; the worked example runs with a stock `go` toolchain
and no model. By the end you'll know what makes a monolith form, why a one-time
cleanup never holds, and how a grandfathered ratchet stops the next line of growth
without a flag day.*

Nobody writes a 20,000-line file. They write a 300-line file, and then someone adds
a function, and then someone adds a case to a switch inside that function, and a year
later `git blame` is a hundred authors deep and the file is 20,320 lines long. That
is the real number for `gateway/run.py` in Hermes, a peer agent stack. Its `cli.py`
is 16,128 lines; `run_agent.py` ~6,030; `conversation_loop.py` 5,297;
`hermes_state.py` 5,862. The core of the system is concentrated in a handful of
files that no single person can hold in their head.

The revealing part isn't the size — it's that Hermes' *own* code-quality rubric asks
for the fix. It literally says "refactor god-files into clean modules." The doctrine
is correct and written down. The files grew anyway. That's the whole lesson: **a rule
that only describes the good state does nothing to defend it.** The monolith accretes
faster than it's paid down precisely because, at the moment someone adds the line that
tips a 1,499-line file to 1,501, nothing refuses.

## fak already names the debt line

fak's code-quality scorecard (`tools/code_quality_scorecard.py`) has counted this as
hard architecture debt from the start:

```python
FILE_HARD_MAX = 1500    # hard debt: an egregious god-file
FUNC_HARD_MAX = 200     # hard debt: an egregious god-function
```

And fak owns the *paydown* side too. `/modularize` is a focused pass that splits a
monolith along real concern seams as behavior-preserving code motion — moving a
top-level declaration into another file *in the same package* is a semantic no-op in
Go, because package (not file) is the scope, so `go build` + `go vet` + `go test` is a
near-complete gate. `tools/godsplit_plan.py` plans the boundaries (the doc-comment-aware
cut lines) and flags the four things that make a move *not* a no-op (per-file build
tags, `init()` order, aliased imports, raw strings). The one gap in that "near-complete"
gate — a decl silently *dropped* by an incomplete cut, unreferenced in-module so `go build`
never says `undefined:` — is closed by `tools/refactor_verify.py`, which diffs the package's
top-level declaration set before/after and fails on any definition that left without landing
somewhere. In one run that loop took fak from architecture-debt 12 → 0.

So the debt line is named and the cleanup is a repeatable recipe. What was missing is the
thing that keeps a clean tree clean: a rule that runs on *every* change and refuses new
growth. A scorecard *measures* debt; it doesn't *stop* it accruing. Left to a periodic
sweep, the tree oscillates — clean after a `/modularize` run, creeping back up until the
next one, because between sweeps nothing holds the line.

## Measure strict, block loose

Here's the subtlety that keeps the gate from becoming the thing everyone disables — and it
lands differently on files than on functions.

**Files stay firm at 1500.** A brand-new file monolith is the single thing you most want to
refuse, so the *file* line is not loosened: this gate and its file-only sibling
`internal/godfileceiling` (the hard "NO NEW GOD-FILE" ratchet, issue #2898) both hold new files
at 1500, matching the scorecard's `FILE_HARD_MAX`. That's deliberate — the two gates are kept
aligned on the file line, and raising it in only one would be dead code, since the stricter
sibling still refuses.

**Functions block loose.** The scorecard's `FUNC_HARD_MAX=200` is a *measurement* threshold —
it answers "how much function debt is on the tree?", and a 250-line function genuinely is a
notch of debt worth seeing on the dashboard. But a *build gate* that reds on that same
250-line function isn't surfacing debt, it's blocking work: the function is large, not runaway,
and refusing it just teaches the fleet to reach for the escape hatch on every push. So the
*function* ceiling is split from the scorecard: the gate blocks only *egregious* new functions,
its ceiling defaulting to 400 (operator-tunable via `FAK_GODFUNC_MAX_LINES`). The file ceiling
is tunable too (`FAK_GODFILE_MAX_LINES`), but the sibling gate bounds files regardless.

**Grandfathered offenders get slack.** On top of the ceilings, an already-frozen offender may
drift within a bounded **slack band** (`FAK_GODFILE_GROWTH_SLACK_PCT`, default 20%) before the
gate re-engages — the fix for the single commonest false block, an ordinary edit to a function
that was already large. Measure strict so you can see the debt; block loose so the gate only
ever fires on a monolith actually forming. And for the genuine one-off the ceilings still
refuse, `ALLOW_GOD_FILE=1` admits a single run rather than forcing the author onto a feature
branch.

(The sibling `internal/godfileceiling` gate, issue #2898, is the LOC-ceiling half — file-only,
non-tunable const 1500. This gate, #2868, adds the *function* dimension and the growth ratchet
on top.)

## The ratchet: grandfather the stock, refuse the flow

A one-time sweep to zero is the wrong shape, and not just because it's a lot of work at
once. The moment it lands, the tree starts re-accreting, and you're back to periodic
cleanups. The gate that *holds* has to be a **ratchet**: it fixes the current level and
only ever moves one way.

`GOD_FILE_GROWTH` (`internal/hooks/gate_godfile.go`) does exactly that. When it shipped,
every tracked non-test `.go` file already over the file ceiling, and every function already
over the function ceiling, was **grandfathered** at its then-size into a frozen baseline
(`internal/hooks/godfile_baseline.go`). The gate then refuses exactly two things:

1. a **new** file crossing the file ceiling (1500 lines, matching the sibling gate), or a
   **new** function crossing the function ceiling (default 400) — anything not in the baseline
   at all;
2. **growth** of a grandfathered offender past its frozen size *plus its slack band* —
   a function frozen at 1000 lines with the default 20% slack reds at 1201, and 1200 is
   clean.

Everything else passes. Shrinking is always clean. And the frozen anchor itself may only ever
*shrink or lose entries* — the regen test (`FAK_GODFILE_BASELINE_REGEN=1`) refuses a
baseline that adds an entry or raises a ceiling, so drift within the slack band never gets
baked into the anchor. The result is a surface that's **bounded, not strictly monotonic**:
a grandfathered offender may drift within its band, but a runaway still reds and the anchor
only ratchets *down*. The band is the deliberate trade — the last increment of strictness
for far fewer false blocks on a busy shared trunk, where an ordinary edit to an already-large
function is the single commonest way a well-behaved change hits the gate. Want the strict
ratchet back? `FAK_GODFILE_GROWTH_SLACK_PCT=0`.

This is deliberately close to the shape of the Python-tool ratchet (`NEW_PYTHON_TOOL`) that
already ships in the same gate set — grandfather the stock, refuse the flow — so there's
one mental model for "hold this line at its current level" across the hygiene surface.

## What the gate refuses to be gamed by

The file rule is a line count, but the function rule can't be — a line count is trivial
to defeat by stuffing logic onto fewer, longer lines, or by hiding it in a giant string
literal. So `longFuncs` parses the file with `go/parser` and measures each function's
span from its declaration line through its closing brace. The AST makes the span
un-gameable by literals or comments, the same reason the scorecard blanks literals before
its own brace scan. A file that doesn't parse fails *open* on the function scan (it won't
compile anyway, and the file-line rule still applies) — the gate is a quality signal, and
a quality signal that wedges the loop on a transient parse error is worse than one that
stays quiet.

Function identity is receiver-qualified: `internal/gateway/gateway.go:New`,
`internal/agent/chat.go:(*HTTPPlanner).Complete`. Two same-named methods in one file
can't shadow each other's ceiling.

## Where it runs

Two places, both hard:

- **`fak hygiene`** (pre-push) runs `GOD_FILE_GROWTH` default-on in the tree-gate set
  (`internal/hooks/tree.go`), so a growth offense is caught locally *before* the trunk
  goes red.
- **`make ci`** runs `TestGodfileGate_LiveTreeClean`: the real tracked tree, judged
  against the frozen baseline, must yield zero findings. The day a god-file grows or a
  new one lands, this reds CI naming the offender and the line it crossed.

Every finding carries the same recover-by hint: don't grow the monolith, split it —
`/modularize` owns the recipe — and regenerate the baseline (tighter, never looser)
only *after* a genuine split lands. The gate points at the cure it's already got.

## A worked example: what reds the build, and what doesn't

Take the real case the slack band was built for: `cmd/fak/guard.go:cmdGuard`, a grandfathered
god-function frozen at 1181 lines. A peer adds a handful of lines to it — the kind of ordinary
edit that lands on a shared trunk every hour — pushing it to, say, 1245. Under a strict ratchet
that reds the build the instant it crosses 1181, forcing whoever touched it into a mid-flight
split of a function nobody set out to refactor. Under the slack band (1181 + 20% = 1417) it's
clean, and the fleet keeps moving. Measuring the function as a notch of debt is the scorecard's
job; *blocking* a five-line edit to it was only ever noise.

Now say a change balloons the same function to 1500 lines — past the band. *That* reds, by
name, on the turn it happened:

```
GOD_FILE_GROWTH  cmd/fak/guard.go
grandfathered god-function GREW: cmd/fak/guard.go:cmdGuard spans 1500 lines, past its frozen
ceiling 1181 (+20% growth slack = 1417). Split along concern seams instead of growing the
monolith — /modularize owns the recipe … If the size is legitimate, widen the growth slack
or raise the ceiling for the run (FAK_GODFILE_GROWTH_SLACK_PCT / FAK_GODFILE_MAX_LINES /
FAK_GODFUNC_MAX_LINES) or admit it with ALLOW_GOD_FILE=1.
```

The agent (or the human reviewing the PR) reads that while the code is still in front of it,
and either lifts the new logic into a helper, runs `/modularize` on `guard.go` first to earn
the headroom (which *lowers* the frozen ceiling, so the split pays for itself), or — if the
size is genuinely justified — widens the slack or admits the one run. The gate fires on the
runaway and stays quiet on the ordinary edit. Multiply that across a fleet of agents on a
shared trunk, every turn, and the function cannot *run away* upward — while a normal day's
work never trips it. (New *files* over 1500 are refused too, by the sibling
`godfileceiling` gate; this gate's own slack band is what governs the *function* case above.)

You can watch the gate judge the live tree with no model in the loop:

```bash
go test ./internal/hooks -run TestGodfileGate_LiveTreeClean -v
# ok — zero god-file/function growth offenses on the tracked tree
```

A clean tree is silent; the first file or function to cross its ceiling names itself and
the span it grew to.

## Why this is the steerability story, not just a lint

A monolith isn't only ugly — it's *unsteerable*. The larger a file or function, the more
of it a change has to load, the more surface a reviewer (human or model) has to hold, and
the more places a small edit can have a non-local effect. fak tracks this as a first-class
signal: the steerability scorecard (`tools/steerability_scorecard.py`, surfaced in
`#steering-guard` via `fak steering`) folds architecture debt into a steerability index,
and `/modularize` is the pass that pays it down.

`GOD_FILE_GROWTH` is the *preventive* half of that same story. `/modularize` retires the
debt that exists; the gate stops the next unit of debt from forming. Together they make
the steerability index a ratchet instead of a treadmill — you don't re-clean the same
files every month, because the line held the whole time.

That's the difference between Hermes' rubric and fak's gate. Both *say* "keep files
small." One of them refuses the line that would break the rule. Hermes' 20,320-line
`gateway/run.py` is what "we have a doctrine" looks like without the gate; fak's frozen
baseline plus a red build on the first line over the ceiling is what "we have a floor"
looks like with it.

*The model proposes the next line; the kernel disposes of the one that grows the
monolith.* See [linting agent code at the kernel](code-linting-at-the-kernel.md) for the
sibling idea (a bad *write* checked at the same boundary) and the `/modularize` skill for
the paydown recipe this gate points every offender toward.
