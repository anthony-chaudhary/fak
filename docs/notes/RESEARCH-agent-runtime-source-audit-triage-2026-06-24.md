---
title: "idea-scout triage: CLAWAUDIT — a source-code audit of the agent runtime layer; prior art validating fak's 'the runtime IS the boundary' thesis + an audit methodology to mirror on fak's own kernel (2026-06-24)"
description: "Triage of the idea-scout candidate arXiv:2606.21071 (Zhang, Li, Guo, Cai — 'Local LLM Agents as Vulnerable Runtimes'): CLAWAUDIT statically audits the implementation layer of local LLM agents (prompt builder, parser, tool dispatcher, skill loader, memory writer, network client, permission gate) — the same seven components fak's kernel IS. Verdict: prior art to cite (independent validation that the runtime layer is the unexamined safety boundary) AND a methodology to mirror (fak's claims are design-level; the IMPLEMENTATION of its own seven components is in-scope for exactly this audit). The static-rule tool itself is not adopted (Semgrep/CodeQL rules target the OpenClaw-family source tree, not fak's Go kernel)."
---

# idea-scout triage — CLAWAUDIT / agent runtime source audit (issue #585)

> Closes the daily idea-scout candidate [#585](https://github.com/anthony-chaudhary/fak/issues/585)
> (`tools/idea_scout.py`, filed 2026-06-24). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite AND an audit methodology to mirror on fak's own runtime.
> The static-rule tool is not adopted.**

**Source:** https://arxiv.org/abs/2606.21071 — "Local LLM Agents as Vulnerable Runtimes:
A Source-Code Audit of the Agent Runtime Layer", Zhengsong Zhang, Zongze Li, Jiawei Guo,
Haipeng Cai (submitted 2026-06-19). Read from the arXiv abstract via WebFetch on
2026-06-24; this is a surface read of the abstract, not a paper audit or a reproduction.

## What it is

The paper's thesis is the load-bearing part: local LLM agents (it names OpenClaw and
Nanobot) "have become privileged software runtimes that mediate between user intent, model
outputs, and host-level actions" — the shell, filesystem, browser, stored credentials,
messaging apps. Prior agent-security work characterizes them from the *outside* (prompt
injection, malicious skills, marketplace risk, black-box eval). But **the implementation
layer that performs the mediation — the prompt builder, parser, tool dispatcher, skill
loader, memory writer, network client, and permission gate — has remained an unexamined
safety boundary.** To the authors' knowledge, no prior work audits the agent's *source
tree* for implementation-level weaknesses in those components.

Their artifact, **CLAWAUDIT**, is a static auditing framework that measures vulnerability
exposure in a local-agent runtime. It derives a **five-category taxonomy from STRIDE** and
writes agent-specific static-analysis rules absent from general rule sets, instantiated in
two backends: **47 Semgrep YAML rules and 30 CodeQL queries**. They evaluate on
**OPENCLAWBENCH** — 446 source-level advisories mined from the OpenClaw repo, split
temporally into 229 train (rule-derivation) and 217 held-out test advisories. On the
held-out set the custom rules raise **Semgrep recall 21.7% → 66.8%** and **CodeQL recall
13.8% → 75.1%**, with train/test gaps ≤ 4 pp (the rules generalize to unseen vulns). They
flag the honest limit themselves: the recall-oriented rules need **manual triage**, so
semantic filtering is required before production use.

## Why it surfaced next to fak

Because fak *is* the seven components the paper names. fak's whole thesis is that this
mediation layer is THE security boundary — "an agent kernel that sits between an AI agent
and the tools it calls, and adjudicates every tool call before it runs" (`AGENTS.md`). The
paper is independent, concurrent evidence that fak is pointed at the right boundary: the
implementation layer, not a wrapper around it.

| Paper's component | fak's subsystem | How fak treats it |
|---|---|---|
| Prompt builder | `internal/ctxmmu`, `internal/ctxplan` | Context/turn assembly is kernel-owned, not string-concatenated per turn |
| Parser | adjudicator call-repair + `internal/grammar`, `internal/normgate` | Malformed/poisoned tool calls are repaired or refused by structure, not parsed loosely |
| Tool dispatcher | `internal/adjudicator` (`decide.go`) | Every call is adjudicated **before** dispatch — default-deny |
| Skill loader | `internal/registrations`, `tools/new_leaf.py`, `internal/architest` ABI guard | No dynamic skill `eval`; a leaf is a typed, build-time-checked registration, not runtime-loaded code |
| Memory writer | `internal/recall` | Memory is a kernel surface, result-quarantined, not a free-write sink |
| Network client | `internal/gateway` | Egress goes through the gateway, not an ad-hoc per-tool HTTP call |
| Permission gate | `internal/policy` + adjudicator | A default-deny capability floor with a closed 12-reason refusal vocabulary the model can't talk past |

## The sharp, honest insight

fak's security **claims are design-level**: default-deny adjudication, an in-process
(no-IPC) boundary, a structured tool-call parser/repair, a closed refusal vocabulary,
result quarantine. The paper's contribution is **orthogonal to a design claim** — even a
correctly-*designed* boundary can carry implementation-level weaknesses (a parser
confusion, a path-traversal in the loader, an off-by-one in the permission gate) that a
design statement does not cover. And fak is squarely *in scope* for that audit: fak is
itself "a privileged software runtime that mediates between user intent, model outputs, and
host-level actions."

So the useful residual the scout surfaced is a **methodology to mirror**, not a tool to
import: per component, fak should be able to say whether its defense is **structural**
(immune by construction — e.g. no dynamic skill `eval`, in-process boundary with no IPC
parse surface, default-deny gate) or **implementation-correctness-bound** (audit-reachable
— where a bug in fak's own Go would matter). fak already runs Go-native static gates
(`go vet`, the `architest`/`boundarylint`/`pathlint`/`urllint` guards, the claims honesty
ledger); the paper motivates folding the seven components into one STRIDE-shaped checklist
so the structural-vs-audit split is explicit rather than assumed. That is a future doc/scorecard
increment, not part of closing this triage.

## Triage decision

- **Adopt the tool?** No. CLAWAUDIT's value is its 47 Semgrep + 30 CodeQL rules, derived
  from and benchmarked against the **OpenClaw** source tree (the OpenClaw-/Nanobot-family
  agent runtimes). Those rules target that codebase's patterns; they do not transfer to
  fak's single static Go kernel, and pulling in a Python/Semgrep/CodeQL audit pipeline as a
  dependency would contradict the one-binary, zero-dep design. fak's static-analysis seam
  is already Go-native.
- **Defend against (a methodology to mirror)?** **Yes, partial** — recorded above as a real
  residual: fak's claims cover the *design* of the boundary; the *implementation* of its own
  seven components is the exact surface CLAWAUDIT-style auditing covers. The next honest
  increment is a per-component structural-vs-audit map, not a code change here.
- **Cite as prior art?** **Yes** — the first systematic, source-code-level framing that the
  agent runtime layer is the security boundary (not a wrapper around it). It independently
  and concurrently validates fak's central thesis and belongs alongside the prior art
  already tagged in `CLAIMS.md` (DOS `dos_refuse_reasons`, the eBPF verdict, the kernel
  vDSO), next to the [Doberman-Core](RESEARCH-doberman-prior-art-triage-2026-06-23.md)
  capability-gate peer and the [Tool-Guard](RESEARCH-tool-guard-isolated-planning-triage-2026-06-23.md)
  description-poisoning triage.

**Action:** close #585 as triaged → prior art cited + audit methodology recorded (this
note). No code change to `tools/idea_scout.py`: the candidate was surfaced and scored
correctly — this was a real, on-topic, high-relevance hit.
