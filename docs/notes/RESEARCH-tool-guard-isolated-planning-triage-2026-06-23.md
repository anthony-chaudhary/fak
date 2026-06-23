---
title: "idea-scout triage: Tool-Guard / isolated planning — a threat axis fak does not yet gate (2026-06-23)"
description: "Triage of the idea-scout candidate arXiv:2606.20922 (Shi et al., 'Think Twice Before You Act'): cross-tool description poisoning steers a planner via tool METADATA even when the poisoned tool is never called, and the Tool-Guard defense quarantines a tool's INFLUENCE after a misaligned invocation. Verdict: a threat to defend against (a real residual gap — fak gates results and calls, not descriptions) AND prior art to cite (the description-axis dual of fak's result-axis quarantine)."
---

# idea-scout triage — Tool-Guard / isolated planning (issue #527)

> Closes the daily idea-scout candidate [#527](https://github.com/anthony-chaudhary/fak/issues/527)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: a threat to defend against (records a real residual gap) AND prior art to cite.**

**Source:** https://arxiv.org/abs/2606.20922 — "Think Twice Before You Act: Protecting
LLM Agents Against Tool Description Poisoning via Isolated Planning", Shanghao Shi, Xiao
Wang, Chaoyu Zhang, Hao Li, Wenjing Lou, Thomas Hou (submitted 2026-06-18). Read from the
arXiv abstract via WebFetch on 2026-06-23; this is a surface read of the abstract, not a
paper audit or a reproduction.

## The attack: cross-tool description poisoning

The paper names a tool-agent attack surface distinct from prompt injection. **Cross-tool
description poisoning** manipulates *planner-visible tool metadata* — a tool's
**description** in the catalog the planner sees — to steer the agent's trajectory, **even
if the poisoned tool itself is never chosen**. The load-bearing observation: a poisoned
description *persists in the planning context across steps*, so it exerts continuous
influence over every subsequent tool choice, not a one-shot nudge. The authors first show
that several existing prompt-injection defenses **transfer poorly** to this threat, because
those defenses screen tool *inputs/outputs* (or a single turn's user text), not the
standing catalog metadata the planner re-reads each step.

## The defense: Tool-Guard via isolated planning

**Tool-Guard** is a system-level defense built on a new primitive the paper calls
**isolated planning**: when a tool *invocation* is detected as misaligned or suspicious,
that tool is moved onto a quarantined **"influenced list"**, which *breaks the further
influence of its description on planning* — while still letting the tool be invoked to
support the task. The novelty is the axis of the quarantine: it isolates a tool's
**description-time influence over the plan**, not the tool's call or its result bytes.
Reported to cut attack success substantially on AgentDojo and ASB while keeping task
utility (per the abstract; not re-measured here).

## Why it surfaced next to fak, and where fak actually stands

fak's security floor adjudicates two boundaries — and **neither is the one this attack
targets**:

| Boundary | fak surface (CLAIMS.md) | Does it cover description poisoning? |
|---|---|---|
| The tool **call** | Capability lock at `k.Decide` / `Syscall` (default-deny, closed 12-reason refusal) | **Partial.** Catches a poisoned description that tries to induce a *destructive call* — but the attack's whole point is steering the trajectory *without* the poisoned tool ever being chosen, so the call gate never fires on it. |
| The tool **result** | Result-admit gate + `normgate` rescan: secret/poison-shaped tool **output** is QUARANTINED, paged to a stub, K/V span evicted (CLAIMS.md result-quarantine, KV-quarantine bridge) | **No.** Quarantine acts on bytes coming *out* of a tool. A tool *description* is metadata going *into* the planner; it never passes through the result-admit gate. |
| The tool **description** | — (no description-admit gate today) | **This is the gap.** Catalog metadata enters the planning context unscreened and unprovenanced. |

So the honest read is that cross-tool description poisoning is a **third axis** —
description-time, planning-context — orthogonal to fak's call-time capability lock and its
result-time quarantine. fak does not currently gate it. Recording that residual honestly is
the point of this note; claiming the existing floor already covers it would be the kind of
unwitnessed claim `CLAIMS.md` exists to prevent.

## Triage decision

- **Adopt Tool-Guard as-is?** **No — not its mechanism.** Isolated planning gates on a
  *misalignment detector* ("invocations detected as misaligned or suspicious"). fak's
  standing stance is that such a detector is **evadable by design and is never the floor**
  (the floor is the capability lock + structural quarantine; the injection detector is a
  best-effort rung — CLAIMS.md `normgate`, the AgentDojo battery). Wiring a detect-then-
  quarantine loop in as the *defense* would contradict that thesis. The *idea* worth
  keeping is the quarantine **axis**, not the detector that triggers it.
- **Defend against?** **Yes — this is the operative verdict.** The threat is real and fak's
  current floor does not cover the description axis. The structural (non-detector) dual of
  isolated planning is a **description-admit gate**: screen/provenance-stamp tool-catalog
  metadata the same way the result-admit gate screens tool output, and bound a low-trust
  tool's description so it cannot steer the plan toward a *different* tool. That is a
  candidate capability, scoped here, not built in this triage increment (see Next step).
- **Cite as prior art?** **Yes.** Tool-Guard's isolated planning is the closest external
  peer to fak's quarantine thesis pushed onto a new axis — the **description-time dual** of
  fak's **result-time** result-admit/KV-quarantine gate. Worth naming in the spirit of the
  `[0/29 novel — the contribution is the assembly]` prior-art discipline (`CLAIMS.md`): the
  primitive (quarantine an untrusted artifact's influence) is shared; fak's and Tool-Guard's
  contributions are *where* they place it (result bytes / KV span vs. plan influence).

**Action:** close #527 as triaged → **defend-against + prior-art cited** (this note). No
code change in this increment: `tools/idea_scout.py` surfaced and scored the candidate
correctly (topic `prompt-injection-defense`, score 63), and the right small artifact for a
research/security triage is the recorded verdict + the named gap, not a half-built gate.

**Next step (the smallest honest follow-on, if pursued):** a `description-admit` rung —
provenance-stamp each tool's catalog metadata at registration and refuse (or trust-bound)
an unprovenanced/low-trust description from influencing planner tool-selection — filed as
its own scoped issue against `internal/ctxmmu` / `internal/provenance`, with an AgentDojo/ASB
cross-tool-poisoning fixture as its witness. It reuses the existing admit/quarantine and
provenance seams rather than importing Tool-Guard's detector.
