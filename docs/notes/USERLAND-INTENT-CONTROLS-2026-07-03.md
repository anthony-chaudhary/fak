---
title: "Userland intent controls: compile intent to knobs, react to witnessed progress (2026-07-03)"
description: "A design for the control layer users actually want: intent verbs (`retry --context more`) that compile to many concrete knobs at once, and a reactive policy algebra (`when stalled then checkpoint; reset`) whose predicates are queries over witnessed signals — not absolute turn/token limits a user should never have to think about. Grounded in the seams that already exist: the session drive table, compose.go, the ctxvalue step-advice vocabulary, the reset carryover builder, and the SESLOG/GRAPH tickets."
---

# Userland intent controls: intent → knobs, signals → verbs

> Date: 2026-07-03. Scope: design + the in-repo seams it composes. Nothing in §2–§5 is
> shipped; §7 is the honest fence. Sits directly on
> [`SESSION-CONTROL-STATE-AS-FIRST-CLASS`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md)
> (the mechanism substrate) and inside the harness-native program (#2387, esp. TURN #2388,
> SESLOG #2392, GRAPH #2397).

## 0. The idea in one paragraph

fak's control surface is now **mechanism-complete but intent-blind**. The drive table
(`internal/session`, #620) gives a live session real knobs — run-state, budget, pace,
priority, all writable mid-flight, all read at the turn boundary. But every knob is an
absolute number (`turns_left: 3`, `max_tokens_per_turn: 4096`), and absolute numbers are
exactly the interface a user should never see: nobody knows whether their task needs 12
turns or 40, and being asked to guess is the "dumb limit" experience. Meanwhile the
*signals* a smarter layer would react to already exist — `step_advice` from the ctxvalue
report, liveness/VERIFIED-progress from `dos_status`, the compaction-bail reasons — but
they are all advice-only; nothing acts on them. The missing layer is two compilers: an
**intent compiler** that turns a user's one-word direction (`retry with more context`)
into a coherent multi-knob write, and a **policy engine** that turns witnessed signals
into drive verbs (`when not progressing then checkpoint, reset with distilled carryover,
escalate on the second failure`) — with the policy expressed as a typed, explainable
query/AST over the session's own ledger, not a bag of hardcoded if-statements.

## 1. The gap, stated from the tree

What exists (all live seams, verified against the tree today):

| Seam | What it proves | What it does NOT do |
|---|---|---|
| `internal/session` drive table + `/v1/fak/session/{id}/{verb}` routes | knobs are live-writable per turn; `Decide` gates the boundary | knobs are absolute numbers; no vocabulary above them |
| `internal/session/compose.go` | ONE knob (throttle ratio) can derive TWO budgets (planner window + worker fraction) coherently | only the pace axis; nothing composes "context" or "effort" |
| `internal/gateway/ctxvalue.go` `StepClass` (`any/bounded/checkpoint/rebuild/unknown`) | a closed, evidence-named advice vocabulary per session | inert by design — "the gateway never enforces it" |
| `fak_session_reset` / the guard carryover builder | a session can be reborn into a fresh window with a distilled seed | reachable only on budget-drain; no user verb; no context-size dial |
| `dos_status` (liveness · VERIFIED progress · region · resume) | progress can be answered from the ledger, never from self-report | nothing inside the loop consumes it |
| `ArmMetrics.StoppedBySession` closed stop reasons | "why did it stop" is a field | stopping is the only automated reaction that exists |

So: signals without actuation on one side, actuation without vocabulary on the other.
The design is the bridge, in two layers, with the dumb limits demoted to a third.

## 2. Layer 1 — intent axes: the userland vocabulary

The user names a **dimension and a direction**, never a number:

```
fak session retry <id> --context less|equal|more [--effort ...] [--patience ...]
fak session <id> dig-deeper | wrap-up | be-patient
```

`retry --context more` is the flagship because it shows why intent must compile to
*many* knobs at once. "More context" is not "raise a byte cap." It compiles to a
coherent bundle:

- **recall**: raise the memq recall query's `k` and byte `budget` (the recall side);
- **planner**: widen the ctxplan `SessionPlanner.Budget` window (the resident side);
- **carryover**: on the reset/fork, seed with more survival classes — full tool results
  and file excerpts, not just the distilled summary (the COMPACT #2393 class contract);
- **compaction**: lower aggressiveness (raise `--compact-history-budget` for the child).

And `--context less` is not a degraded mode — it is often the *right* prescription for a
session that has confused itself: distilled-carryover-only, tight window, aggressive
class-based compaction. Fresh eyes as a control verb.

Three properties make this a compiler and not a vibe:

1. **Relative to the measured baseline, never absolute.** `more` =
   f(what the previous attempt actually used), read from the ctxvalue accumulator and
   the usage ledger — e.g. carryover budget = 1.5× the prior attempt's median resident,
   clamped by the envelope (§6). The verb is anchored to evidence, so it means the same
   thing on a small task and a huge one.
2. **Pure and explainable.** `Compile(axis, direction, baseline) → []KnobWrite`, a
   sibling of `compose.go` (which is the proof-of-concept: one throttle ratio already
   derives two budgets). `fak session retry --explain` prints the knob bundle before
   applying — the memq explain-before-run discipline.
3. **Small closed axis set.** `context` (the four knobs above) · `effort` (per-turn
   output cap, reasoning effort, model tier via the router) · `patience` (turn budget,
   stall thresholds, backoff caps) · `breadth` (tool fan-out, subagent count). Each axis
   is a named composition function; adding an axis is adding a function, not a flag.

`retry` itself is precisely: **fork from the last checkpoint** (SESLOG #2392 makes this
O(1); until then, the reset-carryover path), with a fresh trace, the recompiled context
envelope, and — load-bearing — the previous attempt's *failure evidence* distilled into
the seed: what was tried, what was witnessed to fail (`dos_commit_audit` verdicts, test
output), so the retry is informed, never a blind rerun.

## 3. Layer 2 — the reactive policy algebra ("if it's not progressing, do something")

The goal's second half asks for the standing version: the user shouldn't have to watch
the session to issue the verb. That is a **policy**: predicate over signals → verb from
a closed set. The repo already knows how to build this kind of thing — memq's lesson is
"build SQL, not a specific query," and the refusal vocabulary's lesson is "closed
tokens, never prose." Applied here:

**Predicates** are pure functions over the *witnessed* signal catalog only:

```
progress.verified_delta(turns=5) == 0     -- ledger-VERIFIED rung, never self-report
liveness == STALLED                        -- the dos_status fold
ctx.step_advice == checkpoint | rebuild    -- the ctxvalue StepClass, per turn
budget.remaining_pct < 20
tool.error_rate(turns=4) > 0.5
touch.cycle(files, turns=6)                -- the thrash detector (§4)
```

Each predicate names its evidence source and its provenance label (WITNESSED vs
OBSERVED, the conflation-scorecard discipline); a policy can be required to fire only on
witnessed rungs.

**Verbs** are the closed action vocabulary — mostly verbs that already exist:
`throttle / pause / drain / budget / pace / priority` (the drive table), plus
`checkpoint` (land in-flight state — the StepClassCheckpoint action), `reset(carryover=
distilled|full)` (the session-reset seam), `retry(axis=direction)` (§2), and
`escalate(reason)` / `refuse(reason)` with a closed reason token.

**Rules** are a small AST, not config soup:

```
when progress.verified_delta(turns=5) == 0 and ctx.step_advice == checkpoint
for 2 turns
then checkpoint; retry(context=less)
cooldown 1/attempt, max 2

when tool.error_rate(turns=4) > 0.5 and touch.cycle(turns=6)
then pause; escalate(THRASH_DETECTED)
```

Semantics that keep it sound:

- **Boundary-only evaluation.** Rules are read in the same slot as `Decide` — one lock
  read per turn boundary, never mid-decode. The DRAINING nuance carries over verbatim:
  a fired stop is requested instantly, taken at the boundary.
- **Narrow-never-widen, by construction** (the #2387 doctrine). An autonomous rule may
  only narrow drive (cut budget, pause, checkpoint, reset, escalate). Widening — more
  budget, more context — exists only inside a **pre-authorized retry envelope** the
  user/operator granted up front ("up to 2 retries, context may grow to 2× baseline"):
  bounded, journaled, and spent like a budget. A session can never talk itself into
  more rope; it can only spend rope a human already measured out.
- **Explain-before-arm, journal-on-fire.** `fak policy explain` prints the compiled
  plan (memq's explain). Every firing appends a receipt — predicate, the evidence
  values it saw, verb, drive `Rev` before/after — to the session ledger, the same
  receipt discipline as VALID #2391. "Why did my session reset at 3am" is a lookup.

## 4. The graph/query reading of "not progressing well"

Two graph structures make the stall predicate sharp instead of vibes:

1. **Progress as a query over the session ledger.** SESLOG (#2392) makes the session a
   content-addressed, hash-chained event log. Then "is it progressing" is literally a
   query: *did any node's done-bit flip to VERIFIED in the last N turns? did the work
   frontier change?* — where done-bits come from the task graph (GRAPH #2397) and
   verification from `dos_verify`. This is the same move memq made for recall: the
   ledger is the database; progress predicates are queries over it. No new
   reconstruction layer, no trusting the transcript's narration.
2. **Thrash as a cycle in the touch graph.** Build the bipartite graph (files ×
   turns) from tool calls: a session revisiting the same nodes — edit A, revert A,
   edit A — without a verified edge landing is *cycling*, which is distinguishable
   from a session touching new nodes slowly (exploring) even when both burn the same
   tokens. Token-rate heuristics cannot tell these apart; the graph can. This is the
   AST-shaped answer to "if it's not progressing well": the predicate walks structure,
   not counters.

## 5. Why this is where fak provides real value

Intent-level control is only sound on top of an ungameable progress signal. "Auto-
continue until done, retry if stuck" conditioned on the agent's own narration is a
prompt-injection surface and an optimism amplifier — a model that *says* "almost done"
buys itself budget. fak is the stack where the conditioning signals are **witnessed**:
progress is the ledger-VERIFIED rung (`dos_status` has no `claimed` field by
construction), done is `dos_verify` from git evidence, commit claims are
`dos_commit_audit`-graded. That is the moat: everyone can ship a `--max-turns` flag;
only a trust kernel can safely ship *"keep going while it's genuinely working, intervene
when it genuinely isn't"* — because only it can tell the difference without asking the
agent. Dumb limits are what you deploy when you can't trust any signal. fak can.

## 6. Where the dumb limits go

They don't disappear — they get **demoted to the outermost envelope**: the hard token/
spend/wall-clock ceiling an operator sets once, sized generously, hit rarely (the same
role `DefaultLedgerLimit` plays — a backstop, not a control). Inside the envelope,
everything is intent and policy. The user-facing default is a shipped **policy pack**
("standard drive"): checkpoint on `step_advice=checkpoint`; on `rebuild`+stall, reset
with distilled carryover; escalate after two fruitless retries; thrash → pause+escalate.
Users steer with intent verbs, operators tune by editing the pack, and nobody is asked
to guess `maxTurns` again.

## 7. Build order + honest fences

1. **Intent compiler** — `internal/session/intent.go` (pure, `compose.go` sibling) +
   `fak session retry --context …` CLI. Smallest shippable slice: the `context` axis
   compiled onto the reset-carryover path. *(No new mechanism — composition of #620 +
   the carryover builder.)*
2. **Signal catalog** — fold `StepClass`, drive `Rev` stream, and the `dos_status`
   liveness/progress rungs into one per-turn `signals.Snapshot` with provenance labels.
3. **Policy engine v0** — Go-declared rule table (no DSL parser yet), evaluated beside
   `Decide` in `gateTurn`, journaled firings, narrow-only verbs + the retry envelope.
4. **Graph predicates** — touch-graph thrash from existing trace events now;
   ledger-query progress once SESLOG #2392 / GRAPH #2397 land.
5. **DSL/AST surface + `fak policy explain`** last — syntax is the least risky part;
   the semantics above are the contract.

Fences, owned plainly: the served `req.Raw` turn still does not read `Decide` (the #620
fence — this design inherits it; the harness `RunArm` loop is the first consumer, same
sequencing as ctxplan). `effort`'s model-tier knob needs the router's demote/promote
path, which is not-yet live-wired. Progress-graph predicates are blocked on #2392/#2397;
until then the stall predicate runs on the weaker witnessed pair (liveness + verified
commit delta). Nothing here is a scheduler: priority stays a field a supervisor reads.

## Related

- [`SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md)
  — the drive-state substrate every verb here writes to.
- [`O1-TURN-CONTEXT-PLANNER-2026-06-23.md`](O1-TURN-CONTEXT-PLANNER-2026-06-23.md) — the
  context-side budget the `context` axis compiles onto.
- `internal/session/compose.go` — the existing one-knob→many-budgets composition this
  generalizes; `internal/gateway/ctxvalue.go` — the closed advice vocabulary the policy
  engine finally consumes.
- Harness-native program #2387: TURN #2388 (the boundary these rules evaluate at),
  SESLOG #2392 (the ledger the progress query runs over), GRAPH #2397 (the done-bit
  graph), COMPACT #2393 (the carryover classes `--context` selects among), VALID #2391
  (the receipt discipline for policy firings).
