---
title: "idea-scout triage: defensive misdirection (CMPE) — theory-side support for detector-is-not-the-floor, but its deceptive mechanism is the inverse of fak's legible-refusal floor (2026-06-23)"
description: "Triage of the idea-scout candidate arXiv:2606.20470 (Soosahabi & Namsani, 'Analyzing Defensive Misdirection Against Model-Guided Automated Attacks on Agentic AI Systems'): a probabilistic attacker-judge model shows predictable detect-and-block refusals let ASR approach 1 as the query budget grows, and proposes detect-and-misdirect (CMPE) — replacing honest refusals with strategically misleading responses. Verdict: prior art to cite + a threat-model framing that REINFORCES fak's detector-is-not-the-floor thesis; the misdirection mechanism is NOT adopted — it re-centers an evadable detector and requires the kernel to lie, the inverse of fak's legible closed-vocabulary refusal and witness ledger."
---

# idea-scout triage — defensive misdirection / CMPE (issue #553)

> Closes the daily idea-scout candidate [#553](https://github.com/anthony-chaudhary/fak/issues/553)
> (`tools/idea_scout.py`, filed 2026-06-23). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite + a threat-model framing that reinforces fak's
> detector-is-not-the-floor thesis. The proposed misdirection mechanism is NOT adopted.**

**Source:** https://arxiv.org/abs/2606.20470 — "Analyzing Defensive Misdirection Against
Model-Guided Automated Attacks on Agentic AI Systems", Reza Soosahabi & Vivek Namsani
(submitted 2026-06-18). Read from the arXiv abstract via WebFetch on 2026-06-23; this is a
surface read of the abstract, not a paper audit or a reproduction.

## The paper, in one pass

The work models the attack-defense setting as three probabilistic components: a **target
system**, its **defense mechanism**, and the **attacker's automated judge** (the model-guided
evaluator that scores each probe and steers the next one). Two results:

1. **The negative result.** Conventional **detect-and-block** defenses can let attacker
   success rate (ASR) **approach 1 as the query budget grows**, because *predictable refusals
   provide useful feedback to automated search*. A deterministic refusal is a clean label —
   "this probe was blocked" — that the attacker's judge hill-climbs on, refining prompts until
   one slips through.
2. **The proposed defense.** **Detect-and-misdirect**: a detected-malicious interaction gets a
   *controlled, non-operational* response engineered to induce **false positives in the
   attacker's judge** — lowering the positive predictive value of attacker-selected candidates
   and yielding a **bounded asymptotic ASR**. The proof-of-concept is **CMPE** (Contextual
   Misdirection via Progressive Engagement): replace predictable refusal text with *safe but
   strategically misleading* responses. Reported to cut estimated ASR upper bounds by up to
   two orders of magnitude and nearly eliminate verified success on PAIR / GPTFuzz runs (per
   the abstract; not re-measured here).

## Where fak actually stands

fak's standing thesis — **detector-is-not-the-floor** — is the lens that resolves this paper
cleanly. The floor is the **capability lock** (default-deny at `k.Decide`, a refusal drawn
from a **closed 12-reason vocabulary** with **bounded disclosure**) plus structural
containment (result-admit / KV-quarantine); the injection *detector* (`normgate`, the
AgentDojo battery) is deliberately a best-effort rung that **never load-bears** (`CLAIMS.md`).

| The paper's frame | fak's position |
|---|---|
| Detect-and-block where **the detector IS the floor** → automated judge hill-climbs on refusal feedback → ASR→1 | fak's floor is **not** a detector. The capability lock denies on the *capability* (`SELF_MODIFY`, `POLICY_BLOCK`, …), not on a verdict about whether the prompt "looks malicious". Probing refusal *text* does not move an attacker closer to a reachable destructive call — the lock denies the same way no matter how the request is phrased. |
| Predictable refusals leak exploitable signal | fak's refusals are predictable **on purpose**, and that is safe **precisely because** the leaked information is a *structural fact* ("this capability is not in policy"), not a tunable detector confidence the judge can climb. Bounded disclosure already withholds near-miss signal — `SELF_MODIFY` returns *only the offending glob*, not "you were close" (`CLAIMS.md`, units 19–20). |
| Defense = make the system **lie** (misleading responses) | The inverse of fak's design. The whole trust substrate is **legible, verifiable, honest** refusals + a witness ledger ("honesty over a green-looking lie"); deny-as-value hands the loop a structured disposition it consumes (`CLAIMS.md`, unit 74). A kernel that deceives its own caller has no honest disposition to emit and corrodes the auditability the closed vocabulary + DOS verify-from-git exist to provide. |

So the paper's **negative result is theory-side support for fak's thesis**: it formalizes
*why* a detect-and-block-as-floor system is hill-climbable, and the mechanism it identifies —
the attacker's judge exploiting refusal feedback — only bites when the refusal/detection is
the thing standing between the attacker and the capability. In fak it is not. This is the
**theory-side** companion to issue #526's **evaluation-side** support for the same thesis
(an AUC-0.998 probe is not detection); both say *do not put the detector on the floor*.

## Triage decision

- **Adopt detect-and-misdirect / CMPE as a fak defense?** **No — on two independent grounds.**
  (1) It **re-centers a detector**: you must *detect* "malicious" to decide when to misdirect,
  re-importing the evadable-detector-as-floor problem fak keeps off the load-bearing path.
  (2) It requires the kernel to **emit strategically misleading responses** — to lie to its own
  caller. That is the direct inverse of fak's legible closed-vocabulary refusal, bounded-
  disclosure witness, deny-as-value disposition, and the DOS witness ledger. Deception buys a
  bounded ASR at the cost of the auditability that is fak's actual product. Not adopted.
- **Defend against (is the threat real for fak)?** **Real, but structurally out of scope for
  the floor** — with one residual worth recording. The attack (an automated judge hill-climbing
  on refusal feedback) is real for *detect-and-block-as-floor* systems; fak's floor is immune
  because the lock is not a detector. The **residual fak should not regress on**: a refusal
  reason must never leak a *tunable* signal (a detector confidence, a near-miss score, a
  "distance-to-allow") that an automated judge could climb. The closed 12-reason vocabulary +
  bounded disclosure already enforce this; this note records it as an explicit invariant to
  guard, not a new gate to build.
- **Cite as prior art?** **Yes.** The probabilistic attacker-judge model is a clean external
  formalization of *why predictable detection leaks* — worth naming alongside the existing
  prior-art discipline (`CLAIMS.md` `[0/29 novel — the contribution is the assembly]`). fak and
  this paper agree on the diagnosis (predictable detection is hill-climbable) and diverge on the
  cure: the paper hides the signal by lying; fak removes the signal's leverage by not putting a
  detector on the floor.

**Action:** close #553 as triaged → **prior art cited + threat-model framing reinforcing
detector-is-not-the-floor; misdirection mechanism explicitly not adopted** (this note). No code
change in this increment: `tools/idea_scout.py` surfaced and scored the candidate correctly
(topic `prompt-injection-defense`, score 53), and the right small artifact for a research /
security triage is the recorded verdict + the named invariant, not a half-built deception layer.

**Next step (the smallest honest follow-on, if pursued):** a one-time audit asserting that no
entry in the closed refusal vocabulary leaks a hill-climbable score — i.e. every `deny` reason
is a categorical structural fact, never a confidence or a near-miss distance — filed as its own
scoped test against `internal/abi/reasons.go` / the adjudicator. It guards the invariant this
note names rather than importing the paper's detector-and-deceive loop.
