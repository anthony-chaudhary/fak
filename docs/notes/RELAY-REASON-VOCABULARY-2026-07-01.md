---
title: "Relay reason vocabulary"
description: "Data-only closed reason vocabulary for perpetual-session relay tombstones."
date: 2026-07-01
---

# Relay reason vocabulary

Status: data-only spec. No gate consumes these tokens yet, and this file is not
the exclusive `dos.toml` reason table. A future floor must opt in deliberately,
with its own maintenance witness, before any relay refuses or drains on these
tokens.

The vocabulary is closed so relay legs never explain their state with prose.
Every row follows the `dos check-reason` shape: a token, a category, a summary,
and a fix that tells the next leg or operator what action is valid.

## Shape

```json
{
  "token": "RELAY_*",
  "category": "advisory|TRUE_DRAIN|STALE_CLAIM|OPERATOR_GATE",
  "summary": "one-line condition that fired",
  "fix": "operator or successor-leg recovery action"
}
```

Categories are intentionally small:

| Category | Meaning |
|---|---|
| `advisory` | A threshold or precondition was observed, but the current leg may continue to a safe point. |
| `TRUE_DRAIN` | The relay ended or rotated cleanly against a durable witness. |
| `STALE_CLAIM` | A transcript or baton claim is not corroborated by durable state. |
| `OPERATOR_GATE` | Automatic continuation is unsafe; a human or stronger witness must intervene. |

## Tokens

| Token | Category | Summary | Fix |
|---|---|---|---|
| `RELAY_ARMED` | advisory | The soft relay threshold was crossed; rotate at the next verified safe point. | Finish the current atomic action, externalize load-bearing state, and rotate when the safe-point predicate passes. |
| `RELAY_ROTATED` | TRUE_DRAIN | The leg wrote a baton and tombstone at a safe boundary, then handed control to the next leg. | Start the successor leg from the baton and re-verify its cursor before trusting any progress field. |
| `RELAY_GOAL_DONE` | TRUE_DRAIN | `done_when` was satisfied against the durable store, so the relay ended normally. | Close the relay; do not launch another leg unless the durable done witness is later invalidated. |
| `RELAY_NOT_EXTERNALIZED` | STALE_CLAIM | Rotation was refused because load-bearing state still lived only in the transcript. | Commit, file, or ledger the missing state, then rerun the externalize check; park if it cannot be made durable. |
| `RELAY_PARKED_UNSAFE` | OPERATOR_GATE | The hard ceiling arrived before any verified safe point, so the leg parked instead of overrunning context. | Resume with an operator or stronger witness, recover only durable state, and write a clean baton before relaunching automation. |
| `RELAY_BATON_STALE` | STALE_CLAIM | Successor verification found the baton cursor no longer matches git, ledger, or the configured durable store. | Discard the stale baton fields, re-derive the cursor from ground truth, and write a fresh baton before continuing. |
| `RELAY_NO_PROGRESS` | OPERATOR_GATE | The relay made no verified progress for the configured number of consecutive legs. | Stop automatic rotation, inspect the blocker and hysteresis settings, then relaunch only after a new progress witness or narrowed objective exists. |

## Non-goals

- This file does not add `dos.toml` reasons.
- This file does not authorize any new refusal path.
- This file does not make transcript summaries trustworthy; relay successors still
  re-derive from durable witnesses.
