---
title: "Session-observability for RSI: from cost data to value data"
description: "Why our coding-session transcripts are an under-observed RSI data asset, the capture->structure->link->aggregate->learn ladder that turns them into something a loop can learn from, and the sessionobs scorecard that grades how far up the ladder the pipeline has climbed."
---

# Session-observability for RSI

The fleet runs thousands of Claude Code coding sessions a day. Every one leaves a
transcript JSONL on the host that produced it. That is a large, continuously
growing record of *how agents actually do the work* in this repo, and it is the
most direct RSI signal we own: not a synthetic benchmark, but our own behavior,
graded by reality every time a commit lands or a session stops.

We already mine part of it. `tools/session_audit.py` folds those transcripts into
exact **cost** observability: tokens, tool mix, cache reuse, dollars, per-model and
per-namespace rollups. That answers *what did a session spend*. It is genuinely
useful and it is not the thing this note is about.

The thing this note is about is the question cost data cannot answer: **what did a
session achieve?** A loop that wants to improve how we work has to be able to tell a
session that shipped a witnessed commit apart from one that burned 200 turns and hit
a STOP. Until it can, every session looks the same — a pile of tokens and tool calls
with no verdict attached. You cannot learn "this behavior produces value" from a
corpus where value is invisible.

## The missing rung: the outcome link

The single piece of data that is missing is the link from a session to its
**outcome**. Concretely, for each session: did it land a commit? Was that commit's
claim later confirmed by a witness, or is it a `CLAIMED_CLOSED` that no diff
supports? Did the session end at a STOP, a guard refusal, or an interrupt with
nothing shipped? Or was it a read-only session that explored and answered a question
and was never meant to mutate anything?

The fleet already computes exactly this vocabulary at the *issue* level — the
dispatch-status `closure_rate` partitions closes into `TRUE_RESOLVED` /
`DATA_RESOLVED` / `CLAIMED_CLOSED` by strict diff-witness. The session corpus has no
equivalent. A session is the unit where the *behavior* lives (which tools, in which
order, how many retries, which guard refusals provoked), but it carries no outcome
label, so the behavior cannot be correlated with anything.

Attach that label and the corpus becomes learnable. Leave it off and the corpus is a
cost ledger, not an RSI substrate.

## The observability ladder

Session data becomes RSI-useful one rung at a time. Naming the rungs is what lets us
measure how far up we are instead of asserting "we have observability" or "we
don't".

1. **capture** — the data exists and is retained. (Done: transcripts on disk.)
2. **structure** — each session is an analysis-shaped record, not an opaque blob.
   (Done for cost: `session_audit.py` parses per-session token/tool structure.)
3. **link** — each record is tied to its outcome, so value and waste are
   distinguishable. (Missing. This is the rung that unlocks the rest.)
4. **aggregate** — each record carries behavior signals (guard refusals, tool
   errors, interrupts, commits, stop/goal events) that a loop can contrast across
   many sessions. (Partly possible; not collected with the outcome.)
5. **learn** — a registered RSI loop reads a committed, scrubbed corpus and changes
   behavior on a strict, witnessed gain. (Missing.)

A pipeline that has climbed to rung 2 has cost observability. A pipeline that has
climbed to rung 5 has RSI. The gap between them is the work.

## The scorecard

`internal/sessionobs` grades the climb, following the repo's
[scorecard doctrine](../../.claude/skills/scorecard/SKILL.md). It is a pure,
deterministic function — `Score(corpus []Record, pipe Pipeline) Report` — with no
clock and no RNG, so two callers with the same inputs score identically. The headline
integer is `sessionobs_debt`: the count of HARD rungs not yet built.

The KPIs are the rungs:

| KPI | rung | what it checks |
|---|---|---|
| `corpus_nonempty` | capture | there are session records at all (the row count is the gate) |
| `records_structured` | structure | records carry per-session turn structure, not empty husks |
| `outcome_link_rate` | link | the share of records tied to an outcome (the headline) |
| `value_waste_separable` | link | both value and waste classes are present, so a loop has a contrast |
| `behavior_signal_present` | aggregate | records carry behavior features, not just an outcome label (SOFT) |
| `corpus_committed` | learn | a scrubbed corpus is durable in-tree, not stranded on one host's disk |
| `loop_consumes` | learn | a registered loop actually reads the corpus |
| `registered_in_control_pane` | learn | the scorecard folds into the ratchet (SOFT) |

You retire debt the only honest way: by **building the missing rung** — ingest the
outcomes, commit a scrubbed corpus, wire a loop — never by weakening a check. A red
`outcome_link_rate` is fixed by linking real outcomes, not by lowering the threshold.
A red `value_waste_separable` is fixed by ingesting sessions that actually stopped,
not by relabeling. This is the same anti-gaming law every fak scorecard obeys.

### The Record is scrubbed by construction

A `sessionobs.Record` carries only structured signal — counts, durations, an outcome
class, behavior flags. It never carries raw prompt or result prose. That is
deliberate and load-bearing: the scrubbed corpus is what a fleet can **commit and
fold across hosts** without leaking private session content. The prose stays on the
host that produced it; only the signal travels. This is what makes rung 4 (aggregate
across the fleet) and rung 5 (a committed corpus a deterministic loop reads) possible
at all.

## What an RSI loop does with it

Once the corpus has outcomes and signals, the loop is a contrast: hold the
value-side sessions (shipped, witnessed) against the waste-side (stopped, refused)
and ask which behaviors separate them. Candidate questions the corpus can finally
answer, none of which cost data can:

- Do sessions that read `AGENTS.md` early ship more often than those that don't?
- What is the tool-sequence signature of a session that fights the guard into a STOP,
  and can we surface that pattern as an early-warning the agent acts on?
- How many turns does a TRUE_RESOLVED ship take, and which sessions are in the
  expensive tail for no extra value?
- Which guard-refusal classes precede waste, so the recovery table in `AGENTS.md`
  can be sharpened where it actually fails?

A kept refinement requires a strict, re-measured gain plus a witness it did not
author — the same honesty gate as the [guard verdict RSI
loop](guard-verdict-rsi-loop.md). The loop closes on our own usage, or it does not
close.

## Build plan

This lands in increments, debt-retiring worst-first.

- **Increment 1 (shipped): the deterministic scorer core.** `internal/sessionobs`
  with `Record` / `Outcome` / `Signals` / `Pipeline` / `ClassifyOutcome` / `Score`,
  fully unit-tested against fixtures for each rung. This is the measurement: point it
  at any corpus + pipeline facts and it tells you, honestly, how far up the ladder
  you are. Run against today's reality it reports high debt — the link and learn
  rungs are unbuilt — which is the correct, non-gamed baseline.
- **Increment 2: the ingester + outcome linker.** A `fak sessions` command (the
  impure shell) that reads the host's transcripts, derives each session's behavior
  signals, links it to a git/witness outcome via `ClassifyOutcome`, and writes the
  scrubbed `Record` corpus. This retires the `outcome_link_rate`,
  `value_waste_separable`, and `corpus_committed` rungs.
- **Increment 3: the loop.** A registered RSI pass that reads the committed corpus,
  proposes one behavior refinement on a strict witnessed gain, and folds the
  scorecard into the control-pane ratchet. This retires `loop_consumes` and
  `registered_in_control_pane`, closing the ladder.

The honesty boundary holds at every step: the score is deterministic, the corpus is
scrubbed, and debt is retired by building the rung, never by editing the detector.
