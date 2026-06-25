---
title: "Budget-triggered session reset: the human-like restart fak does at a token budget"
description: "When a long session crosses its token budget, fak distills the durable facts, drops the ephemera, keeps the recent tail, and continues on a fresh session — the carryover is a pluggable contributor registry, and the auto-continue is one opt-in flag."
date: 2026-06-25
---

# Budget-triggered session reset

*What a human does when a chat gets too long — start over, but keep what matters — made
into a mechanism fak can run at a token budget, transparently, for a Claude Code or Codex
session it proxies.*

## TL;DR

A user sets a token budget (e.g. `fak serve --context-budget-tokens 150000`). When a served
session crosses it, fak does not just stop. With `--reset-on-budget` on, it **distills the
conversation into a compact carryover, re-arms a fresh session, and the client's next request
continues transparently** — no 409, no manual restart. The carryover is the smart part: it
keeps the **durable facts** (preferences, identity, conventions — classified by the SHIPPED
`ctxmmu` durability prior), a **task recap**, a **warm-prefix descriptor**, and the **recent
verbatim tail**, and it drops the ephemera ("it's 3pm" evaporates; "I prefer concise answers"
survives). The carryover is a **pluggable contributor registry**, so new items register
without editing the core.

This was ~90% assembly. fak already had the **detection** (`session.Table` budgets +
`DebitUsage` minting a continuation id on context exhaustion) and the **directive**
(`gateway` emitting a `SessionResetDirective` to a supervisor). What was missing was the
**smarts and the auto-continue**: the directive carried no carryover content, and nothing
re-armed the session. This change fills exactly that gap.

## The seam (what was already there)

| Rung | Where | Status before |
|---|---|---|
| Per-session token budget | `internal/session` `Budget{TurnsLeft, TokensLeft, ContextTokensLeft}` | shipped |
| Budget exhaustion → drain + continuation id | `session/usage.go` `DebitUsage` | shipped |
| Served proxy debits per turn | `gateway/session_admit.go` `debitServedSessionTurn` | shipped |
| Next-turn refusal at the boundary | `gateway` `beginServedSessionTurn` → 409 | shipped |
| Passive reset directive to a supervisor | `gateway` `SessionResetDirective` | shipped (but **empty** — no carryover) |
| **Durability classifier** ("what to keep") | `ctxmmu.classifyDurability` | shipped (`docs/CONTEXT-IS-NOT-MEMORY.md`) |
| **The carryover builder** | — | **this change** |
| **The re-arm verb** | — | **this change** |
| **The auto-continue** | — | **this change** |

## What this change adds

**1. The re-arm verb — `session.Table.Recontinue(parent, child, fresh Budget)`.**
The one write that may follow a *terminal* budget drain. The existing control verbs refuse a
Stopped session by design ("you start a new session, you do not un-stop one"), and `Reset`
*deletes* the record. `Recontinue` instead mints a NEW live session under the continuation
trace, links it back via `ParentTrace` with `Generation` incremented, stamps the closed
reason `BUDGET_RESET`, and leaves the drained parent intact so the exhaustion stays
observable. (`internal/session/table.go`, `decide.go`.)

**2. The carryover builder — `internal/sessionreset` (a pluggable contributor registry).**
`BuildSeed(Input) Seed` folds every registered `Contributor` deterministically over the
drained transcript. Four built-ins register by default:

- `durability_facts` — keeps the `durable` lines, drops `turn`/`session`, by reusing the
  SHIPPED classifier through the new `ctxmmu.ClassifyText(role, content)` (the chat-shaped
  entry to the same `classifyDurability` the admit gate runs — not a fork).
- `task_distill` — a deterministic, extractive "where we are" recap (objective + latest ask).
  A model-call distiller that summarizes the middle is a named follow-on.
- `warm_prefix` — a descriptor for replaying the stable prefix from the vCache prefix-DAG
  (`vcachechain.ReplayCost`), so the fresh window does not re-prefill the system preamble.
  **Honest scope:** vCache is a decision layer (off the live path), so v1 emits the recall
  PLAN, not a live KV splice. Live same-model KV reuse is the named follow-on.
- `verbatim_tail` — the last N turns verbatim, like a human keeping the last thing on screen.

`sessionreset.Register(c)` is the open seam: "other items that come up" register without
editing `BuildSeed`. The package is a tier-2 mechanism — it imports only `ctxmmu`,
`vcachechain`, and stdlib (NOT the wire `agent` type), so the tier-4 gateway converts at the
boundary.

**3. The gateway auto-continue — `Config.ResetOnBudget` (opt-in, fail-open).**
At the served boundary (`gateway/session_admit.go`, both the Anthropic `/v1/messages` and the
OpenAI `/v1/chat/completions` wires), when a request is refused for a budget drain AND the
host wired `ResetOnBudget`, the gateway asks the host for a fresh trace + seed, splices the
seed ahead of the live messages (after any system framing), switches to the fresh trace, and
**proceeds**. With the hook nil — the default — the path is byte-identical to today's 409 +
directive. The gateway stays session-internals-blind: the host (`cmd/fak`) owns
`session.Recontinue` and `sessionreset.BuildSeed`.

**4. The host wiring — `fak serve --reset-on-budget`.**
`cmd/fak`'s `resetServedSessionOnBudget` builds the seed and re-arms the continuation; the
flag gates it (and requires `--context-budget-tokens`). Default OFF: the slice ships dark and
flips on after a live soak, because it rewrites in-flight transcript history.

## Honesty ledger

- **Reuses, does not reinvent.** The "what to keep" decision is the shipped `ctxmmu`
  durability prior verbatim, witnessed to agree with the admit path on the bite pairs.
- **Live KV reuse is deferred.** The `warm_prefix` contributor stamps `live_kv_reuse:deferred`
  in its audit Meta. v1 carries the recall plan, not a live KV splice.
- **The task distiller is extractive, not generative.** v1 is deterministic (no model call);
  a model-call middle-summary is a tracked follow-on.
- **Default behavior is unchanged.** With `--reset-on-budget` off (the default), the 409 +
  `SessionResetDirective` path is byte-identical to before.

## Follow-ons (filed as issues)

- Live same-model KV-included reuse (wire `vcachechain` to a live serve splice).
- A model-call task distiller (summarize the middle, not just extract the ends).
- An MCP `fak_session_reset` tool variant (cooperative reset for any MCP client).
- An operator webhook + a pre-threshold (e.g. 80%) warning before exhaustion.
- A cross-session / fleet-wide budget pool (today each session's budget is independent).

## See also

- [`CONTEXT-IS-NOT-MEMORY.md`](../CONTEXT-IS-NOT-MEMORY.md) — the durability axis the
  `durability_facts` contributor reuses.
- `SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md` — the drive-state table the budget and
  `Recontinue` verb live on.
- `PORTABLE-SESSION-IMAGE-AND-SNAPSHOT-2026-06-24.md` — the freeze/rehydrate machinery a
  future KV-included reset would compose with.
