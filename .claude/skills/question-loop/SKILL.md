---
name: question-loop
description: The super-loop-family member that ASKS instead of ships. It launches detached workers whose only job is to ask 5–10 hard, honest questions about what the repo is doing — the question no other agent has asked, the one everyone's afraid to ask, the one that's opposite what the repo claims, the steelman of the other side — and append them to a durable ledger (docs/questions/asked.jsonl). A SEPARATE next-step loop, in a SEPARATE context window, turns qualifying questions into gh tickets. Use when this named workflow matches the task.
allowed-tools: Read, Bash, Write
metadata:
  opencode: claude-only   # the ask/next-step context separation + the honesty boundary are load-bearing
---

# /question-loop — the super loop that ASKS

> A super loop in the `/super-loop` operator sense — a **bulk detached launcher** —
> but where `/super-loop` launches workers that **ship fixes**, this launches
> workers that **ask questions**. Asking is the whole job. Answering, fixing, and
> ticketing happen elsewhere, on purpose.

## Disambiguation (labeling matters here — read before running)

| Not to be confused with | It does | `/question-loop` does |
|---|---|---|
| `/super-loop` | launches detached workers that **ship fixes** to issues | launches detached workers that **ask questions** |
| `/run-it-all-night` | collects the next most important **datum** on the box | asks the next most important **question** about the plan |
| idea-scout (`fak idea-scout`) | mines **external** feeds (arXiv/GitHub/HN) for related **ideas** | generates **internal** provocations about **our own** choices |
| `research` issues | design-shaped **work items** on the backlog | **questions** — explicitly *not yet work* |
| Go `superloop.Super` (`internal/superloop`) | an **interior** node that walks *other* loops worst-first and mutates nothing at its own altitude | a **leaf** that **produces** questions at its own altitude |

The last row is the load-bearing one: `superloop.Classify` grades a loop on five
ordered properties and reports the **first** it fails. This loop is a bare **leaf**
— no member loops, and it acts at its own altitude (it emits questions) — so
`Classify(question-loop).Reason` reads `has_members does not hold` (the first rung;
`LeafFacts` fails all five), and it is *separately* not an `interior_node`. Either
way it is deliberately **not** registered as a `superloop.Super`; it is a "super
loop" only in the launcher sense. See
[`docs/questions/README.md`](../../../docs/questions/README.md) for the full map.

## The two loops (separate agents, separate context windows)

This is the deep part the operator asked for. Two loops, never sharing a context:

1. **ASK loop** — role *questioner*. Fuel:
   [`ask-hard-questions.md`](../../goal-prompts/ask-hard-questions.md). Each tick
   appends 5–10 `open` questions to the ledger and stops. Never proposes fixes,
   files tickets, or answers itself.
2. **NEXT-STEP loop** — role *ticketer*. Fuel:
   [`questions-to-tickets.md`](../../goal-prompts/questions-to-tickets.md). Each
   tick takes **one** `open` question, decides its fate, and — only if it earns it
   — files **one** gh ticket. Never invents questions.

Why separate: a questioner that also holds the ticketing pen slides into solution
mode — it downgrades the uncomfortable question into a to-do it knows how to close.
The ASK role reads widely, open issues INCLUDED (the `UNASKED` category is literally
"not on any issue," so it must see the backlog to spot a blind spot); what it must
not do is triage, close, or file. A **fresh context per ASK tick** keeps the
provocations honest — never anchored to the next-step pipeline's decisions.

## The four categories

- **UNASKED** — a blind spot no agent has raised; the thing not on any issue.
- **AFRAID** — the elephant in the room everyone's thinking but avoids saying.
- **CONTRARIAN** — directly challenges something the repo CLAIMS is correct.
- **STEELMAN** — the strongest form of the OPPOSITE side of a design choice.

## Run it — attended

- **See what's been asked:** read the tail of `docs/questions/asked.jsonl`, or
  filter one category (`grep '"CONTRARIAN"' docs/questions/asked.jsonl`).
- **Ask a batch now:** follow [`ask-hard-questions.md`](../../goal-prompts/ask-hard-questions.md)
  yourself — read widely, append 5–10 `open` rows, commit the ledger by path,
  stop. Do not answer them in the same pass.
- **Resolve one:** follow [`questions-to-tickets.md`](../../goal-prompts/questions-to-tickets.md)
  — take one `open` question, answer / dismiss / ticket it, commit the ledger.

## Run it — unattended cadence

The operator wants ~5–10 questions every few hours, with the ticketer running
more slowly behind it. Launch the two detached, at the gardening tier (these are
cheap, non-issue-drain loops), through the proven launcher:

```
# ASK tick — every few hours
pwsh tools/launch_goal_detached.ps1 -PointerFile .claude/goal-prompts/ask-hard-questions.md \
  -Workspace C:\work\fak -WorkKind gardening

# NEXT-STEP tick — less often, drains the open questions into tickets
pwsh tools/launch_goal_detached.ps1 -PointerFile .claude/goal-prompts/questions-to-tickets.md \
  -Workspace C:\work\fak -WorkKind gardening
```

Wire the cadence with whatever scheduler the box already uses (the same cron /
wave machinery the dispatch fleet runs on); the launcher passes the no-DoS spawn
gate (`dispatch_preflight.py`) on every launch, so an over-scheduled tick refuses
rather than piling up. Preview an unattended launch with `-PlanOnly` first.

## Labeling the output (enforced, not just asked-for)

The ledger's ids, categories, statuses, and public-safety are machine-checked by
`fak question-ledger` (`next-id`, `dedupe-check`, `lint`, `stats`; tests auto-run
by `make ci`) — the labeling authority both loops call, so "very careful about
labeling" is a gate, not a hope.

A ticketed question gets the **`question-loop`** label as **provenance** ("this
work came from a witnessed question") plus real priority/area/class — it's meant
to be actual work, so it is NOT hidden from dispatch. See them all through the
`question-tickets` view in `.github/issue-views.json`.

## The honesty boundary (do not cross)

- **Asking is not solving.** The ASK loop commits only the ledger, never code.
- **A launch is not a ship.** A question-ticket's witness is the filed issue AND
  the committed `ticketed` status change — not that a worker started.
- **Thin and honest beats padded.** Fewer than 5 non-obvious questions this tick?
  Ask fewer. A question an existing ticket already asks is work, not a question.
- **Public-safe.** No machine-absolute path, host, or PII in a question or ticket
  body — the `PUBLIC_LEAK` gate refuses it.
- **Commit by explicit path**, never `git add -A` (shared multi-session trunk).

## When NOT to use

- To **ship fixes** in bulk → `/super-loop`. This loop only asks.
- To **collect data** on the box → `/run-it-all-night`.
- To mine **external** sources for ideas → `fak idea-scout`.
- To walk *existing* loops worst-first → the Go `superloop` surface
  (`fak superloop walk|drive`). This loop is a leaf, not that interior node.
