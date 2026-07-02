---
title: "idea-scout triage: 'Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents' — the SURVEY that names fak's own family (out-of-band = deterministic policy mediating the agent's actions, organized as reference monitoring + Biba integrity + least privilege; CaMeL/FIDES/Progent/RTBAS/FORGE) AND warns that all of them are validated only on STATIC benchmarks — the same methodology that made in-band defenses look strong until adaptive attacks broke twelve at >90%; prior art to cite — fak IS an out-of-band defense, so the taxonomy situates fak's shipped seams, and the adaptive-evaluation warning independently backs fak's deliberate structural-over-detection bet while raising the bar fak's OWN agentdojo ASR battery should meet (adaptive, not static); no capability adopted (fak already is out-of-band) and not a threat (same side of the trust boundary) (2026-06-26)"
description: "Triage of idea-scout candidate arXiv:2606.26479 (Narisetty, Kore, Kattamanchi, Kumarapu — 'Adaptive Evaluation of Out-of-Band Defenses Against Prompt Injection in LLM Agents', submitted 2026-06-25). The paper makes two contributions: (1) it organizes the 2024-2026 out-of-band defense family — defend a tool-using agent against indirect prompt injection NOT by training the model to refuse but by enforcing security OUTSIDE the model with a deterministic policy that mediates the agent's actions (CaMeL, FIDES, Progent, RTBAS, FORGE; capabilities, information-flow labels, reference monitors) — as instances of classical integrity protection (Biba), reference monitoring, and least privilege, yielding a structured coverage comparison; (2) it WARNS that every one is validated only on static benchmarks — the same methodology that made in-band defenses look strong until adaptive, defense-aware attacks broke twelve of them at >90% — and specifies the threat model + protocol an adaptive evaluation needs, then runs it as an independent reproduction/extension of Progent's adaptive-attack analysis on AgentDojo with an open-weight agent (Qwen2.5-7B) on a single H200: averaged over three runs Progent cut mean ASR ~6x (25.8%->4.2%) and a hand-crafted adaptive attack did not raise it (2.6%) — one small-scale data point on a weak model with a single black-box template, a white-box GCG attack remains open, 'consistent with but does not establish' that out-of-band enforcement is a harder adaptive target than in-band detection. Verdict: prior art to cite — fak IS a member of this out-of-band family (default-deny capability floor = least privilege + reference monitor; IFC Ref.Taint sink-gating = Biba integrity; result-admit quarantine = containment that holds when the model is fooled), so the survey is the academic NAME for the family fak's security.md already calls 'FIDES/CaMeL-class' and 'MELON/IFC/capability-allow-list', and the adaptive-evaluation warning independently supports fak's repeated honest concession that detection is deliberately non-load-bearing (~100% evadable) while RAISING the bar fak's own internal/agentdojo ASR battery should meet — adaptive (defense-aware), not static. No capability adopted (fak already enforces out-of-band); not a threat (a parallel defense on the same side of the trust boundary as fak)."
---

# idea-scout triage — out-of-band injection-defense taxonomy + adaptive evaluation (issue #909)

> Closes the daily idea-scout candidate [#909](https://github.com/anthony-chaudhary/fak/issues/909)
> (`tools/idea_scout.py`, filed 2026-06-26). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — this is the SURVEY paper that names fak's own family.
> fak IS an out-of-band defense (a deterministic policy that mediates the agent's
> actions outside the model), so the paper's taxonomy — reference monitoring + Biba
> integrity + least privilege — is the academic name for the seams fak already ships,
> and its adaptive-evaluation warning independently backs fak's deliberate
> structural-over-detection bet while raising the bar fak's own AgentDojo ASR battery
> should meet (adaptive, not static). No capability is adopted (fak already enforces
> out-of-band); it is not a threat (a parallel defense on fak's side of the boundary).**

**Source:** https://arxiv.org/abs/2606.26479 — "Adaptive Evaluation of Out-of-Band
Defenses Against Prompt Injection in LLM Agents", Praneeth Narisetty, Shiva Nagendra
Babu Kore, Uday Kumar Reddy Kattamanchi, Jayaram Kumarapu (submitted 2026-06-25). Read
from the arXiv abstract via WebFetch on 2026-06-26; this is a surface read of the
abstract, not a paper audit or a reproduction (the abstract gives the taxonomy axes, the
named systems, the adaptive-evaluation protocol, and one self-hosted result with its own
stated caveats — not the per-system coverage table or the attack code).

## The paper, in one pass

The diagnosis is the premise fak was built on: recent work (2024-2026) has **converged on
defending a tool-using agent against indirect prompt injection NOT by training the model
to refuse, but by enforcing security OUTSIDE the model — a deterministic policy that
mediates the agent's actions.** Named systems: **CaMeL, FIDES, Progent, RTBAS, FORGE**,
realized with **capabilities, information-flow labels, and reference monitors**; several
report near-elimination of attacks on the **AgentDojo** benchmark.

Two contributions:

1. **Organize the family as classical security.** The out-of-band defenses are cast as
   instances of **integrity protection (Biba)**, **reference monitoring**, and **least
   privilege** — yielding a structured comparison of what each does and does *not* cover.
2. **Warn that the evidence is static, and specify the adaptive protocol.** Every one of
   these systems is validated only on a **fixed set of injection attempts** — the *same*
   methodology that made **in-band** defenses look strong until **adaptive, defense-aware
   attacks broke twelve of them at over 90% success**. The paper specifies the threat
   model + protocol an adaptive evaluation requires, then runs it as an independent
   reproduction/extension of **Progent's own adaptive-attack analysis** on AgentDojo with
   an **open-weight agent (Qwen2.5-7B) self-hosted on a single H200** — a setting Progent's
   authors did not test. Averaged over three runs the defense held: Progent cut mean
   attack success **~6x (25.8% -> 4.2%)**, and a hand-crafted **adaptive** attack did not
   raise it (**2.6%**).

The authors fence their own result hard: *"one small-scale data point on a weak model with
a single black-box attack template; a stronger optimized (white-box GCG) attack remains
open … consistent with, but does not establish, the hypothesis that deterministic
out-of-band enforcement is a harder target for an adaptive attacker than in-band
detection."* The contribution is the **framing + protocol**, not a robustness proof.

## Where fak actually stands

This is the most *categorically on-target* paper the scout has surfaced: it is not a rival
mechanism — it is the **survey of the family fak belongs to**. fak's security posture is,
in the paper's own three axes, an out-of-band defense:

| Paper's out-of-band primitive | The shipped fak seam that realizes it |
|---|---|
| **Reference monitoring** — a deterministic policy mediates *every* action before it runs | fak's **default-deny capability floor**: every tool call crosses the one in-process adjudication boundary and `k.Decide` admits it against a declarative manifest or **denies by structure** (closed 12-reason vocab; `internal/adjudicator`, `internal/policy`, `TestFoldDefaultDenyEmptyPolicy`). This is a reference monitor at the tool-call seam — "a default-deny capability floor the model can't talk past" ([`AGENTS.md`](https://github.com/anthony-chaudhary/fak/blob/main/AGENTS.md), `CLAIMS.md`). |
| **Least privilege** | The **policy manifest** is exactly a least-privilege declaration: the adopter declares *which* capabilities the task needs (`--policy FILE`, `POLICY.md`); everything else is denied. |
| **Integrity protection (Biba)** — untrusted (low-integrity) inputs may not influence high-integrity actions | fak's **IFC `Ref.Taint` sink-gating**: tainted tool results are tracked and `ifc.SinkGate` blocks a tainted->sink flow *pre-call* (`internal/tracesink`), with **kernel-authored provenance** (trust is taken from the model, not asserted by it). A poisoned low-integrity tool output cannot drive a high-integrity egress — the Biba "no read-up / no write-down" shape at the egress floor. The security.md row already names this **"FIDES/CaMeL-class"**. |
| **Containment that holds when the model is fooled** (the family's shared promise) | **Context-MMU result-admit quarantine** makes a poisoned tool result **non-load-bearing** — held out of the model's context entirely (`k.AdmitResult`, `internal/normgate` rank-5 canonicalize-and-rescan, the `internal/wirescreen` pre-send redactor). |

So fak is not *adjacent* to CaMeL/FIDES/Progent — by this paper's own taxonomy it is a
**member of the same out-of-band class**, assembled into one userspace binary rather than
a research prototype. The paper is the literature handle for what `CLAIMS.md` already calls
**"the MELON/IFC/capability-allow-list family"** and what
[`docs/industry-scorecard/security.md`](../industry-scorecard/security.md) already calls
**"FIDES/CaMeL-class"** — it is the survey that *names and structures* that family.

## The sharp, honest insight

The paper's **second** contribution is the one that actually moves fak, in two directions
at once:

- **It independently validates fak's most-repeated honest concession.** fak's security
  story already insists that **detection is deliberately non-load-bearing**: the
  security.md row concedes the detector is **"~100% evadable on a SOTA evasion battery"**
  and FP-prone, and states fak's guarantee is the **structural capability floor +
  containment** (which never run the detector), *not* a detection rate. This paper is the
  external, scholarly form of that exact argument: **in-band detection looked strong on
  static benchmarks until adaptive attacks broke twelve defenses at >90%**, and the
  hypothesis it tests is that **deterministic out-of-band enforcement is a harder target
  for an adaptive attacker than in-band detection.** That is fak's structural-over-detection
  bet, stated by an independent group — strong prior art to cite next to the concession so
  the security posture is read as a *principled choice the literature shares*, not a
  weakness.

- **It raises the bar fak's own evaluation must clear.** The paper's warning applies to
  fak too: a static AgentDojo-style battery is *exactly* the methodology it cautions
  against. fak ships `internal/agentdojo` (an ASR battery) and a security.md row whose
  honest fence is already **"no measured ASR-under-containment number … no third-party
  containment audit."** The paper hands fak the missing acceptance protocol: an **adaptive
  (defense-aware) evaluation** — the attacker knows the policy and adapts — is the bar an
  out-of-band defense should be measured at, *not* a fixed injection set. The matching
  honest fence travels with the citation: the paper's *own* adaptive result is one
  small-scale black-box data point on a weak model with **a white-box GCG attack still
  open**, so it **does not establish** out-of-band robustness — it establishes the
  *methodology* and a caution. fak must cite it as a *protocol to adopt for evaluation*,
  never as evidence that fak's containment is adaptively robust (it has no such number).

## Triage decision

- **Adopt a new capability? No — fak already IS an out-of-band defense.** CaMeL/FIDES/
  Progent/RTBAS/FORGE are the family fak's shipped seams already instantiate (reference
  monitor = the capability floor; least privilege = the manifest; Biba integrity = IFC
  sink-gating; containment = MMU quarantine). There is no new mechanism to import; the
  paper *describes* fak's class. The one thing worth importing is **evaluation
  methodology**, recorded below as a follow-on — not a feature.
- **Defend against (is this a threat)? No — it is a parallel defense on the SAME side of
  the trust boundary.** Like fak, it protects the operator from a compromised agent
  runtime. (Contrast the [agentic-surveillance triage](RESEARCH-agentic-surveillance-evasion-triage-2026-06-25.md),
  #772, where the topology was inverted against the user — here it is aligned, as with the
  [AgenticOS](RESEARCH-agenticos-intent-os-triage-2026-06-26.md) (#770) and
  [joint intent+harm](RESEARCH-intent-harm-joint-verification-triage-2026-06-26.md) (#910)
  prior-art triages.)
- **Cite as prior art? Yes — strongly, on two distinct grounds.** (1) It is the **survey
  that names fak's family** — the academic handle for "out-of-band, deterministic policy
  mediating the agent's actions," organized as reference monitoring + Biba integrity +
  least privilege — and belongs next to `CLAIMS.md`'s `0/29-NOVEL` "the contribution is the
  ASSEMBLY" discipline (every primitive is established; fak's novelty is the fused
  assembly). (2) Its **adaptive-evaluation warning** both backs fak's structural-over-
  detection bet *and* specifies the protocol fak's own ASR claims should eventually meet.

**Action:** close #909 as triaged → **prior art cited (the survey that names and
structures fak's own out-of-band defense family — reference monitoring + Biba integrity +
least privilege over CaMeL/FIDES/Progent/RTBAS/FORGE — plus an adaptive-evaluation warning
that independently supports fak's deliberate structural-over-detection bet) + the
methodological caution recorded as the bar fak's own `internal/agentdojo` ASR battery
should eventually meet (adaptive, defense-aware — not a static injection set); no
capability adopted, fak already enforces out-of-band, and it is not a threat** (this note).
No code change in this increment: `tools/idea_scout.py` surfaced and scored the candidate
correctly (topic `prompt-injection-defense`, score 70 — a real, on-topic, high-relevance
hit), and the right small artifact for a research/security triage is the recorded verdict
+ the primitive-by-primitive mapping, not a feature import for a class fak already ships.

**Next step (the smallest honest follow-on, if pursued):** record this paper as the
literature anchor on the security row in
[`docs/industry-scorecard/security.md`](../industry-scorecard/security.md) (via its data
row `tools/industry_scorecard.data/rows-security.json`) — the `tool-sandboxing-structural-
containment` row already cites MELON (arXiv:2502.05174) and names the "FIDES/CaMeL-class";
adding this survey as the family's taxonomy citation, and turning the existing
"no measured ASR-under-containment number" fence into a tracked task to run an **adaptive
(defense-aware)** AgentDojo evaluation rather than a static one, is the next scoped edit.
Filed as its own industry-scorecard-lane change (which regenerates the doc), not built in
this triage increment.
