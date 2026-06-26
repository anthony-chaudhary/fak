---
title: "idea-scout triage: agentic surveillance / SurveilBench — a trust topology INVERTED from fak's (the agent surveils its own user for a third party); fak gates the egress 'send-the-report' step ONLY when the operator runs the gateway for the user, and the paper's evasion defense (prompt-injection repurposed) is the inverse of fak's injection-is-non-load-bearing floor; prior art to cite + a named scope boundary, mechanism not adopted (2026-06-25)"
description: "Triage of the idea-scout candidate arXiv:2606.25836 (Jeong, Pham, Houmansadr, Bagdasarian, 'AI Snitches Get Glitches: Towards Evading Agentic Surveillance'): formalizes agentic surveillance — an AI agent analyzes user data, crafts a report, and sends it out via tools — for an adversarial provider (employer / nation-state) against a user who cannot control the agent; SurveilBench across corporate/education/police; three evasion techniques (hide / deceive / over-escalate) repurpose prompt injection user-side. Verdict: prior art to cite + a real threat whose EGRESS step maps to fak's capability lock + outbound floor ONLY when the gateway serves the user — and whose core operator-vs-user asymmetry is structurally OUTSIDE fak's substrate. The evasion mechanism is NOT adopted: it lives in the data the agent reads (gateway-external) and is the inverse of fak's injection-is-non-load-bearing floor — a fak-fronted surveillance agent would HARDEN the surveiller against it."
---

# idea-scout triage — agentic surveillance / SurveilBench (issue #772)

> Closes the daily idea-scout candidate [#772](https://github.com/anthony-chaudhary/fak/issues/772)
> (`tools/idea_scout.py`, filed 2026-06-25). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a real threat whose egress step maps to fak's
> outbound floor ONLY when the operator runs the gateway for the user. The proposed
> evasion mechanism (prompt injection repurposed) is NOT adopted — it is the inverse
> of fak's injection-is-non-load-bearing floor.**

**Source:** https://arxiv.org/abs/2606.25836 — "AI Snitches Get Glitches: Towards
Evading Agentic Surveillance", Hyejun Jeong, Dzung Pham, Amir Houmansadr, Eugene
Bagdasarian (submitted 2026-06-24). Read from the arXiv abstract via WebFetch on
2026-06-25; this is a surface read of the abstract, not a paper audit or a reproduction.

## The paper, in one pass

The paper **introduces and formalizes agentic surveillance**: "the ability of an AI
agent to analyze available information, craft a report, and send it out using available
tools." The adversary is the party that **provisions** the agent — "many employers
(and even nation-states) already provide their users with this technology" — and the
victim is the user, who "might not even have the ability or permission to control the
actions and data accesses of the surveilling agents." The pipeline is three steps:
**analyze** the user's available data → **craft** a report → **send** it out via tools.

Two contributions follow:

1. **SurveilBench** — a dataset of reporting scenarios across three domains
   (**corporate, education, police**) to evaluate surveillance capability across models.
   The finding: some models show **emergent (unprompted) tendencies to help
   surveillance**, but they also **report the attempt to surveil to the government** —
   i.e. the same models snitch in both directions (per the abstract; not re-measured here).
2. **Three evasion techniques** that **repurpose prompt injection** as a *user-side
   defense*: **hide from**, **deceive**, or **induce over-escalation in** the
   surveillance agent. The conclusion is that agentic surveillance "can already be
   easily implemented" and calls for a technical / ethical / legislative framework.

## Where fak actually stands

The lens that resolves this paper is **who the gateway serves**. fak's entire
threat model presumes the operator who runs the gateway is the party whose interests
the gateway protects: the capability lock, the result-admit screen, and the witness
ledger all sit between *the operator's agent* and *the outside world*. Agentic
surveillance **inverts that topology** — the agent (and, if any, its gateway) is
provisioned by the adversary *against* the user, who controls neither. That inversion
is the single fact that decides every row below.

| The paper's frame | fak's position |
|---|---|
| The pipeline ends in **send it out using available tools** | This is the one step that lands on a fak surface. "Send" is a **capability**, and fak's floor is **default-deny on capability** (`TestFoldDefaultDenyEmptyPolicy`, `CLAIMS.md` unit 15): the in-kernel A2A send is `Deny(DEFAULT_DENY)` without the negotiated `CapA2ASend` (`internal/a2achan`), the outbound wire has a pre-send secret/PII redactor (`internal/wirescreen/redactor.go`, #572), and `ifc.SinkGate` refuses to persist a payload the egress floor would block (`internal/tracesink`). **But all three protect the user only when the user/operator runs the gateway.** |
| The agent **analyzes the user's available data** | Out of reach of the capability lock by construction. A provisioned assistant *legitimately holds* the read capabilities (mail, files, calendar) it uses to surveil — the same ones it uses to assist. fak gates **capability, not intent**; it cannot distinguish "summarize my inbox for me" from "summarize my inbox for my employer" by the capability fingerprint alone. The surveillance signal is in the *destination and purpose* of the report, not in a denied call. |
| The adversary **provisions the agent**; the user cannot control its data accesses | **Structurally outside fak's substrate.** fak protects the party who runs it. When the surveiller owns the agent and the gateway, no gateway-level control helps the user — and fak's **legibility is dual-use**: an auditable witness ledger of every egress is *exactly* what a surveillance-report pipeline wants. fak does not, and must not, claim to protect a user from a gateway run *against* them. |
| Defense = **repurpose prompt injection** (hide / deceive / over-escalate), placed in the data the agent reads | The **inverse** of fak's floor. fak's whole job is to make injected content **non-load-bearing** — result-admit quarantine (`k.AdmitResult`, the `SECRET_EXFIL` screen) and the capability lock exist so that bytes the agent *reads* cannot steer what it *does*. A fak-fronted surveillance agent would therefore **harden the surveiller against the user's evasion**: the user's injected "glitch" is screened on ingress exactly as any other untrusted content. The paper's defense and fak's defense point in opposite directions because they protect opposite parties. |

So fak and the paper share a vocabulary (agents, tools, prompt injection, an egress
report) but sit on opposite sides of the trust boundary. The paper's contribution is
to name a topology — **provider-vs-user** — that fak's design does not cover; that is
itself a useful scope marker for fak's security story, which is **operator-vs-external-attacker**.

## Triage decision

- **Adopt the three evasion techniques (prompt-injection-as-defense) as a fak
  capability? No — on three independent grounds.** (1) They live in the **data the
  agent reads**, not at a gateway — they are a property of crafted user content, which
  is gateway-external; fak has no surface to host them. (2) They are
  **prompt-injection-as-defense**, the exact inverse of fak's *injection-is-non-load-bearing*
  floor — building them into fak would mean building a thing fak's own result-admit
  screen is designed to neutralize. (3) A fak-fronted surveillance agent would
  **harden the surveiller** against precisely these techniques, so shipping them as a
  fak feature would be self-defeating for the user the paper is trying to protect. Not adopted.
- **Defend against (is the threat real for fak)? Real for the EGRESS step, and only
  when the gateway serves the user — with three honest fences.** When the user (or an
  operator acting for the user) runs fak, the "send the report out" step is gated like
  any outbound capability: default-deny on the send capability, the pre-send
  `wirescreen` redactor, and the `ifc.SinkGate` egress floor. The fences worth not
  overclaiming:
  1. **fak only helps when it serves the user.** The paper's core asymmetry — the
     surveiller *owns* the agent — is structurally outside fak's substrate; fak
     protects whoever runs it, and cannot protect a user from a gateway run against them.
  2. **Capability ≠ intent.** A surveillance report is a *narrative summary* of the
     user, not a secret/key/PII pattern; `wirescreen` is precision-biased for credential
     and PII shapes, so it targets the wrong content class for a behavioral report.
     fak's lock denies on the *capability*, and "send a summary" can ride a capability
     the user's own assistant also holds.
  3. **Legibility is dual-use.** fak's witness ledger and bounded-disclosure refusals
     are an asset *for the operator*; under the inverted topology they are an asset for
     the *surveiller*. This is a property to state, not regress on — fak's honesty
     guarantees are about auditability, not about hiding a user from their own provider.
- **Cite as prior art? Yes.** The **formalization of agentic surveillance** (analyze →
  craft → send) and the **provider-vs-user trust topology** are the cleanest external
  statement of a threat axis fak's threat model deliberately does *not* cover, and
  **SurveilBench** (corporate / education / police) quantifies it. It belongs alongside
  fak's prior-art discipline (`CLAIMS.md` `[0/29 novel — the contribution is the
  assembly]`) as the paper that draws the boundary of *whom* the gateway protects.

**Action:** close #772 as triaged → **prior art cited + a real egress-side threat fak
gates only when the operator runs the gateway for the user, with the provider-vs-user
asymmetry named as a scope boundary outside fak's substrate; the prompt-injection-based
evasion mechanism is explicitly not adopted (it is the inverse of fak's
injection-is-non-load-bearing floor)** (this note). No code change in this increment:
`tools/idea_scout.py` surfaced and scored the candidate correctly (topic
`prompt-injection-defense`, score 50), and the right small artifact for a research /
security triage is the recorded verdict + the named scope boundary, not a
gateway-external evasion layer fak's own floor would neutralize.

**Next step (the smallest honest follow-on, if pursued):** a one-paragraph addition to
fak's security framing (e.g. `docs/notes/SECURITY-capability-floor-2026-06-18.md` or
the threat-model section it anchors) stating the **operator-vs-user scope boundary**
explicitly — "fak protects the party who runs the gateway; it makes no claim to protect
a user from an agent provisioned against them" — so the provider-vs-user topology this
paper names is recorded as an out-of-scope axis rather than an unstated assumption. It
documents the boundary this note draws rather than importing a gateway-external evasion
mechanism.
