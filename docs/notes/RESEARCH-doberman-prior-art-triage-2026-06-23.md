---
title: "idea-scout triage: Doberman-Core — prior art for the capability-gate thesis (2026-06-23)"
description: "Triage of the idea-scout candidate fu351/Doberman-Core: an independent agent-security gate that intercepts tool calls and returns PASS/AUTH/BLOCK before execution. Verdict: prior art to cite, not a capability to adopt and not a threat to defend against."
---

# idea-scout triage — Doberman-Core (issue #528)

> Closes the daily idea-scout candidate [#528](https://github.com/anthony-chaudhary/fak/issues/528)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite.**

**Source:** https://github.com/fu351/Doberman-Core (Apache-2.0, Python 3.11+, 37★, last push 2026-06-17)

## What it is

Doberman is an open-source security layer for AI coding agents. It sits between an
agent and its tools as a **transparent MCP proxy** — `agent ──▶ Doberman ──▶ real MCP
tool servers` — and returns a `PASS` / `AUTH` / `BLOCK` verdict on every tool call
*before* it reaches a downstream system. Two stated principles drive it: **fail-closed**
(an error defaults to denial) and **raise-only** (a guardrail can tighten but never
loosen without explicit human sign-off).

Concrete features named in its README: hard blocks (secret exfiltration, destructive
commands, force-push on protected branches, invisible-Unicode token smuggling), soft
`AUTH` step-ups (sensitive paths, opaque shell payloads, homoglyph/OOD anomalies),
tunable modes (Light / Balanced / Strict / Paranoid), audit logging of every decision,
tiered auth (confirm / TOTP / scoped elevation), a pre-inference prompt-injection turn
gate, and an ASR/FPR benchmark harness. (Verified from the public README via WebFetch
on 2026-06-23; this is a surface read, not a code audit.)

## Why it surfaced next to fak

It is the closest external peer the scout has found to fak's core thesis. Both projects
adjudicate a tool call **at the boundary, before dispatch, default-deny, fail-closed** —
and both refuse the destructive-by-argument cases (force-push, secret exfil, `rm`-shaped
commands) by structure rather than by asking the model to behave. The overlap is real and
worth naming honestly, in the spirit of fak's `[0/29 novel — the contribution is the
assembly]` prior-art discipline (`CLAIMS.md`).

| | **Doberman-Core** | **fak** |
|---|---|---|
| Boundary | Out-of-process **MCP proxy / stdio sidecar** | **In-process** Go syscall boundary (no IPC, no second process on the decide path) |
| Runtime | Python 3.11+ (`pip install doberman-core`) | One static Go binary (`fak`), zero deps, no `go.sum` |
| Decide model | PASS / AUTH / BLOCK, fail-closed, raise-only | Allow / Deny / Defer, default-deny, closed 12-reason refusal vocabulary |
| Step-up | Tiered auth (confirm / TOTP / scoped elevation) | Deny-as-value disposition (RETRYABLE / WAIT / ESCALATE / TERMINAL) the loop consumes |
| Injection stance | Pre-inference screening turn gate **plus** the gate | Detector is **evadable by design and is not the floor**; the capability lock + result quarantine is |
| Reuse / perf | Not a stated goal | Same boundary doubles as the KV/prefix-cache reuse gate |
| License | Apache-2.0 | (this repo) |

The honest contrast is the *placement* of the gate, not its existence: Doberman validates
the "wrap the tools, deny before dispatch" pattern as an MCP sidecar an adopter bolts on;
fak fuses the same gate into the kernel address space so there is no proxy hop and the
same boundary pays for cache reuse. That fak's central pattern was independently and
concurrently arrived at by another project is evidence the pattern is *right*, and is
exactly the kind of peer fak's launch plan expects to share the `awesome-llm-security` /
`awesome-harness-engineering` shelf with (`docs/launch/landscape-research.md`).

## Triage decision

- **Adopt?** No. It is a Python out-of-process sidecar; fak is an in-process Go kernel
  with a different deployment surface and a fused security/reuse boundary. Pulling it in
  would contradict the one-binary, no-IPC-on-the-decide-path design, not extend it.
- **Defend against?** No. It is a defensive peer, not a threat surface.
- **Cite as prior art?** **Yes** — recorded here as the closest external peer to the
  capability-gate-before-dispatch thesis, alongside the prior art already tagged in
  `CLAIMS.md` (DOS `dos_refuse_reasons`, eBPF verdict, kernel vDSO).

**Action:** close #528 as triaged → prior art cited (this note). No code change: nothing
in `tools/idea_scout.py` is wrong — the candidate was surfaced and scored correctly.
