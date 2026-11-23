---
title: "Level 4 — Prove faster: the keep-bit"
description: "How a correct kernel earns the right to land: a mechanical, non-forgeable keep-bit decides KEEP or REVERT from a measured witness, not a reviewer."
slug: kernel-development-level-4
keywords:
  - non-forgeable keep-bit
  - shipgate.Evaluate
  - rsicycle
  - honest baseline
  - tuned vs naive baseline
  - benchmark authority
  - keep or revert
  - measurement window
  - prove faster
  - kernel speed gate
date: 2026-11-18
---

# Level 4 — Prove faster: the keep-bit

The keep-bit is a mechanical, non-forgeable gate that decides KEEP or REVERT for a kernel from a measured witness. It is not a reviewer's opinion, so a contributor cannot land a correct-but-slower kernel.

*You are on **Level 4 of 6** of the [kernel-development ladder](README.md).*

> **What this is.** The speed rung. Rung 3 proved your kernel is *correct*; this rung
> decides whether it is allowed to *land*. The rule is one line: **a mechanical gate,
> not a reviewer's opinion, decides.** You do not get to say your kernel is faster — you
> feed a measured witness to a keep-bit that is unexported and set only from that
> witness. By the end you can name the keep-bit, run it, read KEEP vs REVERT, and say
> honestly what a passing keep-bit does *not* prove.

## Questions this page answers

- How do I know my kernel is actually *faster*, not just different?
- Why can't I trust my own benchmark?
- What is a **keep-bit**, and why must it be **non-forgeable**?
- What is a **baseline**, and which one is the honest one?
- Why are the naive/cold and tuned/warm baselines never blended?
- What does a passing keep-bit *not* prove?

## Why can't I trust my own benchmark?

Because the author is the worst verifier of their own speedup — the same reason rung 3
distrusts the author's eye. You wrote the kernel, you want it to be fast, and a
stopwatch in your own hands will happily confirm the belief you started with. A
correct-but-slower kernel is the exact failure mode this rung exists to catch: it
passes every correctness witness, it *looks* like progress, and it quietly makes the
system worse.

fak's answer is the same discipline as [verify, don't trust](../verify-dont-trust.md):
replace "take the author's word for the speedup" with "check a witness the author
cannot forge." For speed, that witness is a **measured metric** — a before/after pair
the environment produced, not the loop. `EXTENDING.md` states the contract plainly: a
correct optimization still has to *earn its keep*, and the keep decision is
**non-forgeable**.

## What is a keep-bit, and why must it be non-forgeable?

A **keep-bit** is a single boolean: *did this candidate earn the right to stay?* In fak
it is a real Go field — `Witness.improvedBit`, unexported, in
`internal/shipgate/shipgate.go`. It is set in exactly one place, `shipgate.Evaluate`,
and only from the measured signals in the `Witness` struct:

```go
// internal/shipgate/shipgate.go
func Evaluate(w Witness) (Decision, Witness) {
	p := ProfileFor(w.Class)
	w.improvedBit = (!p.needGain || w.improved()) &&
		(!p.needSuite || w.SuiteGreen) &&
		(!p.needTruth || w.TruthClean)
	if w.improvedBit {
		return KEEP, w
	}
	return REVERT, w
}
```

The bit is **unexported** on purpose: no setter, no exported field, no `SetKept(true)`.
The only way to make it true is to hand `Evaluate` a `Witness` whose `Before`/`After`
show a strict gain *and* whose suite is green *and* whose truth syscall is clean. A test
pins this: `TestKeepBitNonForgeable` asserts a zero `Witness` — one nobody measured —
never reports `Kept()`. You cannot assert your way to a KEEP; you can only *measure*
your way to one. That is what makes it safe to accept optimizations from anyone,
including autonomous coding agents: the gate does not read your commit message, it
reads the numbers.

## What are the three signals the keep-bit reads?

`Evaluate` folds three independent signals, and the profile decides which are required
(`EvidenceProfile` in the same file):

| Signal | Field | Meaning | Required by |
|---|---|---|---|
| **strict gain** | `Before` / `After` (+ `LowerBetter`) | the metric moved the right way, strictly | `ClassFull`, `ClassProofCarrying` |
| **green suite** | `SuiteGreen` | the test suite passed on a clean worktree | `ClassFull` |
| **clean truth** | `TruthClean` | the truth syscall (`dos commit-audit`) was clean | **every** class |

The truth floor is non-negotiable: `TestEveryProfileKeepsTheTruthFloor` asserts no
evidence class can KEEP on a dirty truth — a result that cannot be reproduced from
committed state is not a result. And a *safe no-op* (green suite, clean truth, metric
unmoved) is still REVERT: `TestEvaluateRevertsNoGain` covers both "no change" and
"worse".

## What is a baseline, and which one is the honest one?

A **baseline** is the thing you measured *against*. The whole game is picking the
honest one, because the flattering one is always available and always a lie. There are
two, and they answer different questions:

| Baseline | What it is | What a win against it means | Honest as a headline? |
|---|---|---|---|
| **naive / cold** | no cache, no reuse, stateless serving | "reuse deletes redundant work" | **No** — it is context, not the decision-grade comparison |
| **tuned / warm** | the best already-shipped stack (warm per-agent KV, tuned engine) | "you beat the thing people actually run" | **Yes** — this is the headline |

`BENCHMARK-AUTHORITY.md` is the single source of truth for which number is which, and
it draws the line explicitly: the public multi-agent reuse headline is **4.1× vs
tuned**, while the *same run* is 60.3× vs naive stateless serving — the authority
names the tuned per-agent KV as "the decision-grade baseline; naive stateless is
context, not the headline alternative." The [fleet benchmark
suite](../fleet-benchmarks.md) states the framing law for the whole repo: compare
against the **best already-shipped baseline**, state the absolute number, and mark
every `naive`/`cold` multiple so it can never read as a SOTA win.

## Why are the two baselines never blended?

Because blending them produces a number true of nothing. Average a 60× cold multiple
with a 4.1× warm one and you get a figure matching no workload anyone runs — arithmetic
on two different questions. Keep them **strictly apart**: print both, label both, let
the reader pick the one matching their case. The same rule separates **measured** from
**modeled**: a real dedup hit or cache fill is *measured*; the *dollar price* of a turn
is *modeled* by a transparent cost model. Never let a modeled number wear a measured
number's clothes ([what a saved token is worth](../what-a-saved-token-is-worth.md)).

```
   Which baseline is honest for MY claim?
        ├── "reuse deletes work"      ──►  naive / cold   (context, label it)
        └── "I beat what people run"  ──►  tuned / warm   (the headline)
                                          └── never average the two.
```

## The keep/revert loop

This is the whole cycle, and the decision is made by the gate, not by you:

```
   ┌──────────────┐   ┌───────────────────┐   ┌────────────────────────┐
   │  candidate   │──►│ measure against   │──►│  shipgate.Evaluate     │
   │  (your .go)  │   │ the honest        │   │  (the keep-bit)        │
   └──────────────┘   │ baseline          │   └───────────┬────────────┘
                      └───────────────────┘               │
              ┌───────────────────────────────────────────┴───────────┐
              ▼                                                       ▼
   ┌──────────────────────────────┐          ┌──────────────────────────────┐
   │  KEEP (exit 0): strict gain  │          │  REVERT (exit 3): no gain,   │
   │  ∧ green suite ∧ clean truth │          │  worse, red suite, dirty     │
   └──────────────────────────────┘          └──────────────────────────────┘
```

A run of non-keeps does not spin forever: `shipgate.Gate` (`NewGate(k)`, `Record`,
`ConsecutiveNonKeeps`) trips a breaker after `k` consecutive non-keeps and returns
`ESCALATE` — the loop cannot find a witnessed improvement, so it hands off to a human
instead of thrashing.

## Predict, then run: the real Gate-3 entry point

`EXTENDING.md` names the entry point: `go run ./cmd/rsicycle …`. `cmd/rsicycle` is a
one-shot that drives **one** keep-or-revert decision through `shipgate.Evaluate`. Its
flags are the three environment-authored witnesses — `-metric`, `-before`, `-after`,
`-lower-better`, `-suite-green`, `-truth-clean` — and the verdict is carried in the
**exit code**: `0` KEEP, `3` REVERT.

**Predict first.** A strict gain with a green suite and a clean truth should KEEP and
exit 0. A flat metric should REVERT and exit 3. A strict gain with a *dirty* truth
should REVERT and exit 3 — the truth floor blocks it even though the metric moved.

Now run it from the repo root:

```bash
go run ./cmd/rsicycle -metric tok_per_s -before 100 -after 140 -suite-green -truth-clean
go run ./cmd/rsicycle -metric tok_per_s -before 100 -after 100 -suite-green -truth-clean
go run ./cmd/rsicycle -metric tok_per_s -before 100 -after 140 -suite-green
```

**What actually happened** (run on this host, 2026-11-18):

```
# run 1 — strict gain, green suite, clean truth
score=rsicycle value=140.000 grade=kept metric_delta=40.000 suite_green=true truth_clean=true
DECISION=KEEP kept=true
# run 2 — flat metric
score=rsicycle value=100.000 grade=no-gain metric_delta=0.000 suite_green=true truth_clean=true
DECISION=REVERT kept=false
# run 3 — strict gain, but dirty truth
score=rsicycle value=140.000 grade=truth-dirty metric_delta=40.000 suite_green=true truth_clean=false
DECISION=REVERT kept=false
```

The prediction held on all three. `grade=` names *why* — `kept`, `no-gain`,
`truth-dirty`, `suite-red` — and `DECISION=` is the verdict. One honest caveat: `go run`
**masks the exit code** (it prints `exit status 3` and itself returns 1). Build the
binary and the raw code is visible:

```bash
go build -o /tmp/rsicycle.exe ./cmd/rsicycle
/tmp/rsicycle.exe -metric tok_per_s -before 100 -after 100 -suite-green -truth-clean; echo $?
# → DECISION=REVERT kept=false   → 3
```

The same keep-gate is exposed as a CLI verb, `dos improve`, whose help text calls it
"the witness-gated keep-bit that closes that hole" — the loop grading its own homework.
It reads the same three signals and carries the same exit-code contract (`0` KEEP, `3`
REVERT, `4` ESCALATE):

```bash
dos improve --suite-passed --truth-clean --work 43 --baseline-work 40   # KEEP, exit 0
dos improve --suite-passed --truth-clean --work 40 --baseline-work 40   # REVERT, exit 3
```

`EXTENDING.md` also tells you to run the environment preflight first; on this host
`python tools/extend_preflight.py` prints `[PASS] gate3-prove-faster: keep-bit wired:
cmd/rsicycle one-shot + shipgate.Evaluate` and `NOT READY — required checks failed:
git-guards-installed`. The Gate-3 line is the one that matters: the keep-bit is wired;
the FAIL is a host-setup issue, not a missing gate.

## Worked example: "my kernel is 1.4× faster on my laptop"

This is the claim every kernel author makes, and it is **not yet a claim** — it is an
observation with no baseline, no window, and no witness. Walk it through.

**Why is it not a claim?** "1.4× faster" is a ratio, and a ratio needs a denominator.
Faster than *what*? If the denominator is a cold, un-tuned run, the 1.4× may be
entirely the cache warming up, not your kernel. If it is a single run on a laptop that
was also compiling in the background, it may be noise. A number without a named
baseline and a named window is a story, not a measurement.

**What must the baseline be?** The **tuned / warm** one — the best already-shipped
stack, measured on the same host, in the same session, with the resource you are
measuring freed first. `EXTENDING.md` is explicit: "Free the resource you're measuring
first (e.g. VRAM before a GPU bench) — contention, not architecture, is the usual cause
of a surprising regression."

**What must the window be?** Long enough to average out warm-up and scheduling noise,
and *identical* for both arms — a before/after pair measured over different windows is
not a comparison. **What must the keep-bit see?** A `Witness` with `Before`/`After` from
that same window, `SuiteGreen` true, `TruthClean` true. If the honest baseline already
had the win, the metric does not move and the verdict is REVERT.

| | Bad setup | Good setup |
|---|---|---|
| Baseline | cold / naive, or none named | tuned / warm, same host, same session |
| Window | one run, different lengths | N runs, identical window, warm-up discarded |
| Resource | laptop also compiling | measured resource freed first |
| Metric | "feels faster" | a named metric (`tok_per_s`, `p50_ns`) with `LowerBetter` set |
| Truth | uncommitted working tree | clean tree, `dos commit-audit` clean |
| Verdict | a story | `shipgate.Evaluate` → KEEP or REVERT |

## What a passing keep-bit does NOT prove

A green keep-bit is a *bounded* result, and the honesty discipline is load-bearing:

- **It does not prove the win generalizes.** The witness measured *one* metric on *one*
  host over *one* window. A win on your laptop is not a win on a GPU node, a different
  dtype, or a different sequence length. `BENCHMARK-AUTHORITY.md` grades any broader
  claim against the net-true-value standard, and a scoped row "does not imply an
  unmeasured model, backend, hardware, concurrency level, or production SLO."
- **It does not prove the baseline was the best one.** The keep-bit compares against
  the baseline *you fed it*. Feed it a weak baseline and it will happily KEEP a change
  that loses to the tuned stack. The gate is only as honest as the `Before` you supply.
- **It does not prove the number reproduces from committed state.** `TruthClean` is a
  *signal you pass in*, not something `Evaluate` re-derives — a clean truth means the
  truth syscall was clean when you ran it, not a standing guarantee.
- **It does not prove the change is worth its complexity.** A strict gain of 0.1% that
  adds a thousand lines is a KEEP by the bit and a bad trade by judgment. The bit
  answers "is it faster?", not "is it worth it?".

## Checkpoint

1. What is the keep-bit, and why is it unexported?
2. Name the three signals `shipgate.Evaluate` folds, and which one no class may skip.
3. Which baseline is the honest headline, and why are the two never blended?
4. Your kernel is 1.4× faster against a cold baseline. Is that a claim? What is missing?
5. Give one thing a passing keep-bit does *not* prove.

### Answers

1. It is `Witness.improvedBit` in `internal/shipgate/shipgate.go` — a single boolean
   set only inside `shipgate.Evaluate` from measured signals. It is unexported so there
   is no setter: you cannot assert your way to a KEEP, only measure your way to one
   (`TestKeepBitNonForgeable`).
2. Strict gain (`Before`/`After` + `LowerBetter`), green suite (`SuiteGreen`), clean
   truth (`TruthClean`). The **truth floor** is required by every evidence class
   (`TestEveryProfileKeepsTheTruthFloor`).
3. The **tuned / warm** baseline — the best already-shipped stack. The naive/cold
   baseline is context, not the headline. They are never blended because averaging two
   different questions yields a number true of nothing; print both, label both.
4. No — it is an observation, not a claim. Missing: a named honest baseline (tuned /
   warm, same host and session), an identical window for both arms, the measured
   resource freed first, and a clean truth so the result reproduces from committed state.
5. Any one of: that the win generalizes beyond the measured host/metric/window; that
   the baseline was the best one; that the number reproduces from committed state
   (`TruthClean` is a passed-in signal); that the change is worth its complexity.

## See also

- [Level 3 — Prove correct](level-3-prove-correct.md) · [Level 5 — The adjudicator
  rung](level-5-the-adjudicator-rung.md) — the rungs either side of this one.
- [`EXTENDING.md`](../../../EXTENDING.md) — "Gate 3 — Prove it's faster (the keep-bit)"
  in full; [`BENCHMARK-AUTHORITY.md`](../../../BENCHMARK-AUTHORITY.md) — the tuned-vs-naive
  baseline discipline and the single source of truth for every number.
- [The fak fleet benchmark suite](../fleet-benchmarks.md) — the honest-baseline framing
  law; [what a saved token is worth](../what-a-saved-token-is-worth.md) — the economics
  behind the metric; [verify, don't trust](../verify-dont-trust.md) — the witness discipline.


---

*[← Level 3](level-3-prove-correct.md) · [full ladder](README.md) · next → [Level 5](level-5-the-adjudicator-rung.md)*
