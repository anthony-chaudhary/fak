---
title: "The god-file growth gate: a ratchet, not a sweep"
description: "A monolith is never written in one commit — it accretes one line at a time, faster than anyone pays it down, because nothing refuses the next line. Hermes' gateway/run.py reached 20,320 lines that way. fak already names the ceiling (1500 lines/file, 200/function) and owns the paydown loop (/modularize); GOD_FILE_GROWTH turns the doctrine preventive — a CI ratchet that grandfathers today's offenders at-size and refuses only new growth, so the god-code surface is monotonically non-increasing."
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

## fak already names the ceiling

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
tags, `init()` order, aliased imports, raw strings). In one run that loop took fak from
architecture-debt 12 → 0.

So the ceiling is named and the cleanup is a repeatable recipe. What was missing is the
thing that keeps a clean tree clean: a rule that runs on *every* change and refuses new
growth. A scorecard *measures* debt; it doesn't *stop* it accruing. Left to a periodic
sweep, the tree oscillates — clean after a `/modularize` run, creeping back up until the
next one, because between sweeps nothing holds the line.

## The ratchet: grandfather the stock, refuse the flow

A one-time sweep to zero is the wrong shape, and not just because it's a lot of work at
once. The moment it lands, the tree starts re-accreting, and you're back to periodic
cleanups. The gate that *holds* has to be a **ratchet**: it fixes the current level and
only ever moves one way.

`GOD_FILE_GROWTH` (`internal/hooks/gate_godfile.go`) does exactly that. When it shipped,
every tracked non-test `.go` file already over 1500 lines, and every function already
over 200, was **grandfathered** at its then-size into a frozen baseline
(`internal/hooks/godfile_baseline.go`). The gate then refuses exactly two things:

1. a **new** file crossing 1500 lines, or a **new** function crossing 200 — anything
   not in the baseline at all;
2. **growth** of a grandfathered offender past its frozen size — `gateway.go` was
   3066 lines when frozen, so 3067 reds; 3065 is clean.

Everything else passes. Shrinking is always clean. And the baseline itself may only ever
*shrink or lose entries* — the regen test (`FAK_GODFILE_BASELINE_REGEN=1`) refuses a
baseline that adds an entry or raises a ceiling. So the god-code surface is
**monotonically non-increasing**: it can only go down, one genuine split at a time.

This is deliberately the same shape as the Python-tool ratchet (`NEW_PYTHON_TOOL`) that
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

## A worked example: the line that reds the build

Say an agent is editing `internal/gateway/gateway.go` — frozen at 3066 lines — and adds
a 40-line handler, pushing it to 3106. Nothing about that diff looks wrong in isolation;
it's a plausible, well-formed change. But it grows a grandfathered god-file, so the gate
refuses it by name:

```
GOD_FILE_GROWTH  internal/gateway/gateway.go
grandfathered god-file GREW: internal/gateway/gateway.go is 3106 lines, frozen ceiling 3066.
Split along concern seams instead of growing the monolith — /modularize owns the recipe …
```

The agent (or the human reviewing the PR) reads that on the turn it made the change,
while the file is still in front of it, and either lands the handler in a new
concern-scoped file in the same package — a no-op move — or runs `/modularize` on
`gateway.go` first to earn the headroom (which *lowers* the frozen ceiling, so the split
pays for itself). The monolith never gets the extra 40 lines. Multiply that across a
fleet of agents on a shared trunk, every turn, and the file simply cannot drift upward.

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
