---
title: "idea-scout triage: PrologMCP — a stateful Prolog solver exposed over MCP; a capability paper (symbolic delegation), not an MCP-security paper. Close / not adopted: PrologMCP is a server fak GOVERNS, not a kernel component; recorded as a clean exemplar of the well-behaved stateful MCP server fak's server-facing seams already anticipate (2026-06-24)"
description: "Triage of the idea-scout candidate arXiv:2606.14935 (Mensfelt, Prabhakaran, Haret, Trencsenyi, Stathis — 'PrologMCP: A Standardized Prolog Tool Interface for LLM Agents'): a task-agnostic open-source MCP server that exposes Prolog as a stateful tool with a compact interface, structured error reporting, and per-session isolation, driving a translate-run-inspect-repair loop. Verdict: close / not adopted — it is a CAPABILITY paper (symbolic delegation: the LLM translates, the solver infers), not the mcp-security topic it surfaced under; it is a server that would sit BEHIND fak's gateway, not a part of fak's kernel. Recorded residual: PrologMCP's three named design properties each map to a shipped fak server-facing seam, so it is a useful worked example that validates those seams from the server side. Scout-calibration note: surfaced on a bare MCP+server keyword match; working as designed (flagged for human triage), n=1 is not enough to retune the scorer. No change to tools/idea_scout.py."
---

# idea-scout triage — PrologMCP / a Prolog solver as a stateful MCP tool (issue #584)

> Closes the daily idea-scout candidate [#584](https://github.com/anthony-chaudhary/fak/issues/584)
> (`tools/idea_scout.py`, filed 2026-06-24). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: close / not adopted — out of kernel scope. PrologMCP is a server fak
> GOVERNS, not a kernel component. Recorded as a clean exemplar of the well-behaved
> stateful MCP server fak's server-facing seams already anticipate; no code change.**

**Source:** https://arxiv.org/abs/2606.14935 — "PrologMCP: A Standardized Prolog Tool
Interface for LLM Agents", Agnieszka Mensfelt, Adarsh Prabhakaran, Adrian Haret, Vince
Trencsenyi, Kostas Stathis (submitted 2026-06-12 v1, 2026-06-19 v2). Read from the arXiv
abstract via WebFetch on 2026-06-24; this is a surface read of the abstract, not a paper
audit or a reproduction — the reported numbers below are as the abstract states them.

## What it is

A **capability** paper, not a security one. Its premise: frontier reasoning-tuned LLMs
still fail on deductive tasks *at depth*, and buying accuracy with longer internal
reasoning scales poorly. The complementary route is **symbolic delegation** — the LLM
*translates* the problem into a logic program, and a *Prolog solver* performs the
inference. Today those autoformalization pipelines are bespoke, tied to one task or one
agent.

PrologMCP's contribution is to make that delegation a **task-agnostic, open-source MCP
server**: it exposes Prolog as a **stateful tool through the Model Context Protocol**,
with three named design properties — a **compact tool interface**, **structured error
reporting**, and **per-session isolation** — which together make the
**translate-run-inspect-repair** loop a reusable primitive any MCP-capable agent can drive
(translate to a program, run it, inspect the structured error, repair the formalization,
re-run). Evaluated on **PARARULE-Plus** (a general subset and a harder
reasoning-failure-mode subset) across Claude Sonnet 4.6, GPT-4.1, and o4-mini; the
abstract reports the symbolic Formalizer matching or beating the reasoning LLMs —
~1.00 on the general subset and ~1.00/0.99 vs ~0.95/0.94 on the challenging subset.

## Why it surfaced next to fak — and why that is a weak match

The scout surfaced it under topic **`mcp-security`** (score 50) on the terms
*model context protocol / mcp (title) / server* plus a freshness bonus (≤30d). That is a
**bare keyword + title match**: the paper has "MCP" in the title and is a "server," but it
is about **a new capability to expose** (symbolic inference), not about the **security of
the MCP boundary**. So it is materially different from the prior idea-scout MCP hits this
note sits beside — [MCPPrivacyDetector](RESEARCH-mcp-privacy-leakage-triage-2026-06-23.md)
(a protocol-induced *leakage threat*) and
[CLAWAUDIT](RESEARCH-agent-runtime-source-audit-triage-2026-06-24.md) (a *runtime-boundary*
audit that is genuine prior art for fak's thesis). PrologMCP is neither a threat to fak
nor prior art for its adjudication thesis.

The load-bearing distinction: **PrologMCP and fak are complementary layers, not the same
thing and not competitors.** PrologMCP is a *server* that makes a trusted symbolic tool
available; fak is the *kernel* that adjudicates calls to **any** server. A deployed
PrologMCP would sit **behind fak's gateway** as one more governed MCP server — fak would
adjudicate the calls into it and quarantine the results out of it. fak does **not** adopt
the capability into the kernel: fak is an agent kernel, not an autoformalization / solver
pipeline, and folding a Prolog runtime into the one-binary, zero-dep kernel would
contradict that design.

## The useful residual: PrologMCP validates fak's server-facing seams

The reason this is worth a note rather than a one-line close is that PrologMCP is a **clean
worked example of the kind of stateful MCP server fak is built to govern** — and its three
named design properties each line up with a **shipped** fak seam. That is not adoption; it
is independent evidence that fak's server-facing contract matches what a well-designed MCP
server actually wants.

| PrologMCP design property | fak's counterpart (the governing side) | Shipped artifact |
|---|---|---|
| Per-session isolation | Session isolation + recall **result-quarantine**; a tainted server result is sink-gated before it can flow on | `internal/ifc` (`Ref.Taint` source-stamped, rank-30 pre-call sink gate), `internal/recall` (CLAIMS.md IFC line) |
| Structured error reporting | **Deny-as-value**: a refusal/error is a typed disposition (RETRYABLE / WAIT / ESCALATE / TERMINAL) the loop consumes, from a closed 12-reason vocabulary — not free text | adjudicator units 19, 20, 74 |
| Compact tool interface | A **typed, build-time-checked leaf registration** + grammar-constrained tool calls — a small surface the adjudicator can reason about | `internal/registrations`, `internal/grammar` units 52–57 |
| The *repair* rung of translate-run-inspect-**repair** | In-syscall **call-repair** (positional→named auto-repair; unrepairable ⇒ `Deny(MISROUTE)`) — fak already does the structural-repair step of exactly this loop, with no model turn | `internal/grammar` units 52–57 |
| "Stateful tool" (vs a pure query) | The vDSO tier-1 registry **only** memoizes calls gated on `readOnlyHint`+`idempotentHint`; a stateful / non-idempotent tool like a Prolog session is correctly **never** cached | `internal/vdso` units 25–38 |

There is also a mild, generic **threat surface** worth naming and dismissing: a Prolog
solver exposed as a tool is an **unbounded-compute** surface — a non-terminating query or a
combinatorial blow-up is a denial-of-service vector. But that is exactly the
stateful, potentially-unbounded tool fak's admission / resource gates already exist to
bound; it warrants **no new defense**, only the standard resource floor every adjudicated
tool gets.

## Triage decision

- **Adopt the capability into the kernel?** **No.** Out of kernel scope. PrologMCP is a
  *server* fak governs, not a kernel component, and a Prolog/autoformalization runtime in
  the one-binary, zero-dep kernel would contradict the design. (If a fak deployment ever
  *wants* symbolic delegation, the right shape is to run PrologMCP as a downstream MCP
  server behind fak's gateway — no kernel change.)
- **Defend against (a threat)?** **Only generically.** A symbolic solver is an
  unbounded-compute tool surface; fak already bounds that class with its admission /
  resource gates. No new defense.
- **Cite as prior art?** **No.** It is not prior art for fak's adjudication thesis — it is
  an MCP-ecosystem capability paper. (This is the honest difference from the adjacent
  CLAWAUDIT triage, which *was* cited: that paper is about the runtime *boundary*, fak's
  exact thesis; this one is about a tool to put *behind* the boundary.)

**Scout calibration (no code change).** The candidate surfaced under `mcp-security` on a
bare "MCP" + "server" + title match, but it is a capability paper, not a security one. This
is the scout **working as designed** — it surfaces plausibly-related MCP work and *flags it
for human triage* precisely so a human can catch a capability-not-security hit. A single
false-positive (n=1) is not enough signal to retune the `mcp-security` topic's scorer;
doing so would overfit. If the `mcp-security` topic keeps surfacing capability-not-security
papers, *that* recurrence becomes a real `tools/idea_scout.py` tuning task. For now: no
change to `tools/idea_scout.py` — the scout scored and surfaced correctly; the human verdict
is "close / not adopted."

**Action:** close #584 as triaged → not adopted / out of kernel scope (this note). No code
change.
