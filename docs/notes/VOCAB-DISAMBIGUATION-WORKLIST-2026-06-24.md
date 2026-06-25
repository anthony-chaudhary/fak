---
title: "Vocabulary disambiguation worklist: the overloaded words and where they fork"
description: "The actionable companion to the agent-anatomy explainer. The term-conflation audit found seven core words that each name several distinct concepts; this is the per-term worklist with file locations and the proposed split, split into a docs lane (#721) and a code lane (#722)."
---

# Vocabulary disambiguation worklist

> Date: 2026-06-24.
> Companion to [`EXPLAINER-what-is-an-agent`](EXPLAINER-what-is-an-agent-2026-06-24.md),
> which holds the conceptual map. This note is the **actionable** half: the
> term-conflation audit's per-term findings with real file locations and the
> proposed split. Filed as two issues — **#721** (the docs-lane glossary) and
> **#722** (the code-lane renames + the trust-framing fix). Nothing here is a
> mandate; each row is a proposal to rename or to document, decided per type.

## Why this exists

[`SCALING-LAWS §6`](SCALING-LAWS-OF-AGENTS-2026-06-19.md) already proved the move for
one word ("the word *cache* names at least seven different things"). The audit found
the same disease across the agent vocabulary. The cost is real: a reader who reads
`recall.Session` as "the session" or `internal/agent` as "the agent fak guards" has
the trust model backwards. The fix is cheap — one authoritative split per word — and
the worklist below is what it takes.

## A. `session` — five distinct Go types under one word

| sense | type | file | proposed action |
|---|---|---|---|
| token **decoder** over a KV cache | `model.Session` / `model.BatchSession` | `internal/model/kv.go`, `internal/model/batch.go` | rename → `Decoder` / `Generator` (#722) |
| reloaded **core image** (survives process death) | `recall.Session` | `internal/recall/recall.go` | rename → `CoreImage` (#722) |
| live **drive state** (run-state / budget / priority / pace) | `session.Table` / `session.State` | `internal/session/` | **keep** — this is the canonical `session` |
| the **wire DTO** of the drive state | `gateway.SessionState` | `internal/gateway/gateway.go` | keep (it is explicitly the wire form) |
| per-session **context planner** | `agent.SessionPlanner` | `internal/agent/ctxplan_session.go` | lower priority → consider `ContextPlanner` |

Sharpest split: pick on the question the object answers about a run — *decode tokens*
(decoder), *reload after death* (core image), *is it running and how hard* (drive
state, canonical), *which spans are resident this turn* (planner).

## B. `agent` — the kernel vs the untrusted guest (the trust-framing one)

Four senses: (1) fak itself, the *kernel*; (2) the external untrusted AI loop fak
mediates, the *guest*; (3) `fak agent`, the CLI verb for fak's own demo loop; (4) the
`internal/agent` package, which is mostly the wire servers + the loop. The
load-bearing rule: the kernel is the reference monitor, the guest is the untrusted
program — never one word for both. Action (#722): add `internal/agent/doc.go`
("the host-side loop + wire servers", not "the agent fak guards") and audit the
`AGENTS.md` / FAQ self-references.

## C. `context` — window vs view vs admission-target

The *window* (a fixed token budget), the *view* (`ctxplan` made resident this turn),
and the *admission target* (what "enters context" means at the `ctxmmu` gate).
[`CONTEXT-IS-NOT-MEMORY`](../CONTEXT-IS-NOT-MEMORY.md) already owns the
context-vs-memory durability cut; the missing rows are window-vs-view-vs-admission
(#721). State plainly: "fits in the window" is not "entered the view."

## D. `model` — routed LLM vs the in-kernel transformer

The *routed LLM* (a provider+weights the policy picks — really an `Engine` binding;
`docs/model-routing.md` already concedes "it selects an engine, not a receiver")
versus `internal/model.Model`, fak's own in-kernel transformer with a forward pass and
a KV cache. Action (#721): in routing docs say "engine" / "LLM choice"; reserve
"model" for `internal/model`. "fak runs a model" (owns weights) ≠ "fak routes to a
model" (picks an endpoint).

## E. `memory` — KV working vs durable recall vs procedural

*Working memory* = the KV cache; *durable / recall memory* = cross-session facts that
earn promotion (`internal/recall`); *procedural memory* = a cached skill view
(`contextq.SkillContextRecord`). [`MEMORY-LAYERS-EXPLAINER`](../MEMORY-LAYERS-EXPLAINER.md)
owns the four-layer KV decomposition. Rule (#721): "working memory" always means the
KV cache, never the durable store.

## F. `tool` vs `skill`

A *tool* is an effect-bearing call the adjudicator gates per invocation (a syscall); a
*skill* is a host-side procedure, upstream of the gate, that issues tool calls. fak
gates tools, not skills. The kernel sees only tool calls; a skill is upstream of the
gate and produces them (#721).

## G. `steering`

*Loop steering* (a deny disposition redirecting the loop — defensive, kernel-owned)
vs *planner bias* (benefit signals over which spans go resident) vs *adversarial
steering* (an attacker's prompt). Reserve "steer" for the first; rename the planner
use to "bias/weight" and the attacker use to "manipulate/hijack" (#721).

## Lanes

- **#721 (docs):** the canonical glossary — rows C–G plus the session/agent rows as
  documentation. Could live in a new `docs/GLOSSARY.md` or a pinned section.
- **#722 (code):** the two code-side fixes — the `internal/agent` trust-framing
  `doc.go`, and the non-canonical `Session` renames (A).

## Already tracked elsewhere — do not refile

- **The agent communicator has no first-class Go type** (the zero-byte rank space is a
  lease-and-witness pattern) — the MPI-comm epic owns this.
- **The ctxplan live loop is off by default** (`FAK_CTXPLAN_SEAM`) — tracked by the
  ctxplan seam epic (#555).
- **Session `Decide` is consumed on the harness arm, not the served gateway turn** —
  the [`SESSION-CONTROL-STATE-AS-FIRST-CLASS`](SESSION-CONTROL-STATE-AS-FIRST-CLASS-2026-06-24.md) §7
  follow-on epic owns this.
- **Identity is a Python module, not an `abi` principal**, and `fleet_sessions` still
  re-derives the directive — the #618 follow-up owns this.

## Related

- [`EXPLAINER-what-is-an-agent`](EXPLAINER-what-is-an-agent-2026-06-24.md) — the
  conceptual map this worklist serves.
- [`SCALING-LAWS-OF-AGENTS`](SCALING-LAWS-OF-AGENTS-2026-06-19.md) §6 — the same
  disambiguation move for the word *cache*.
