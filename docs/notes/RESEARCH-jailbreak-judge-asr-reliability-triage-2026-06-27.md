---
title: "idea-scout triage: 'How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring' — a measurement-validity study showing the AUTOMATED JUDGE that assigns nearly every reported attack-success-rate (ASR) is itself uncalibrated and adversarially flippable: dedicated safety classifiers over-flag (precision 0.835, recall 0.974); three LLM-as-judges keep high precision (0.81-0.94) but erratic recall (0.06-0.65); benign framing wrappers that leave the harmful text untouched flip every LLM-judge 57-100% of the time (a single prepended refusal sentence accounts for 39-88%); the classifier resists surface attacks (≤6.7%) but a white-box GCG attack flips 70% of its confident true positives. Verdict: PRIOR ART TO CITE — independent, measured evidence for fak's witness-not-self-report / detector-is-not-the-floor stance, and specifically a validation of fak's CHOICE OF SUCCESS CRITERION: fak's own ASR (asr_fullstack=0, asr_detection=0.763 over the 38-case corpus) is NOT assigned by a judge model at all — internal/agentdojo/agentdojo.go computes 'succeeded := injectionReachedContext && sinkExecuted' from deterministic adjudicator verdicts (VerdictQuarantine / VerdictDeny; the code comment says 'Deterministic, in-process, no model'), so it is structurally immune to all three documented attack families (framing wrappers, prepended-refusal, GCG-on-judge-weights) because there is no judge model in the success decision. Mechanism NOT adopted (a measurement-validity finding, no kernel surface) and not a threat (no adversary against fak; the adversary is against other people's judges). One honest fence carried forward: this immunity is specific to fak's structural flow-reachability witness; the day fak ever reports a judge-mediated ASR (a content-harm probe, or a comparison against an external benchmark whose ASR is LLM-graded), it must follow the paper's recommendations — report judge precision/recall on a human-labeled slice, report ASR corrected for judge precision, and include an adversarial check of the judge. No code change (2026-06-27)"
description: "Triage of idea-scout candidate arXiv:2606.25487 (Yang Gao, 'How Reliable Is Your Jailbreak Judge? Calibration and Adversarial Robustness of Automated ASR Scoring', v2 2026-06-24). The paper checks the automated JUDGE that assigns almost every reported jailbreak/prompt-injection ASR — a safety classifier or a prompted chat model — against 596 human-labeled completions from the HarmBench classifier validation set, then attacks the judges. Findings: the dedicated classifier over-flags (P 0.835, R 0.974); three LLM-as-judges hold high precision (0.81-0.94) but show erratic recall (0.06-0.65), so the same responses yield very different ASR by judge; benign framing wrappers that leave the harmful text intact flip every LLM-judge 57-100% (a single prepended refusal sentence = 39-88%); the classifier resists these surface attacks (≤6.7%) but a small-budget white-box GCG attack on its open weights flips 70% of its confident true positives (21/30); an 80-sample two-annotator audit confirms the flips left the harm intact. Recommendation: report judge P/R on a human-labeled slice, report ASR corrected for judge precision, and adversarially check the judge. Verdict: PRIOR ART TO CITE — measured, external evidence for fak's detector-is-not-the-floor / witness-not-self-report thesis (siblings: AUC-is-not-detection #probe-auc, defensive-misdirection, and the FRONTIER survey's 'benchmarks gameable to near-perfect without solving tasks' anchor), and a direct validation of fak's success-criterion design: fak's ASR numbers (asr_fullstack=0 = 0/38, asr_detection=0.763 = 29/38, benign_completion_rate=1) are produced by a deterministic, in-process, no-model adjudication witness — internal/agentdojo/agentdojo.go: 'succeeded := injectionReachedContext && sinkExecuted', where injectionReachedContext is a detector VerdictQuarantine outcome and sinkExecuted is a sink-gate VerdictDeny outcome, catch_reasons drawn from the closed structural vocabulary (TRUST_VIOLATION 29 / MALFORMED 9), not an LLM grading harmful text. Because no judge model sits in fak's success decision, the three attack families the paper documents (benign-framing wrappers, prepended-refusal sentences, white-box GCG on judge weights) cannot move fak's number: you cannot frame your way past a VerdictDeny that gates PROVENANCE (tainted->sink), not content. Mechanism NOT adopted (a measurement-validity result, no kernel mechanism) and not a threat (the adversary in the paper targets judges fak does not run). Honest fence: the immunity is specific to fak's structural flow-reachability witness — if fak ever reports a judge-mediated ASR (a future content-harm probe, or any external benchmark whose ASR is LLM-graded and that fak compares against), it must adopt the paper's protocol (judge P/R on a human-labeled slice + precision-corrected ASR + an adversarial judge check). No change to tools/idea_scout.py — surfaced and scored correctly (topic prompt-injection-defense, score 47)."
---

# idea-scout triage — how reliable is your jailbreak judge? (issue #1097)

> Closes the daily idea-scout candidate [#1097](https://github.com/anthony-chaudhary/fak/issues/1097)
> (`tools/idea_scout.py`, filed 2026-06-27). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — measured, external evidence that the automated JUDGE
> assigning nearly every reported ASR is uncalibrated and adversarially flippable, and a
> direct validation of fak's choice of SUCCESS CRITERION. fak's own ASR
> (`asr_fullstack=0`, `asr_detection=0.763` over the 38-case corpus) is assigned by no
> judge model at all: `internal/agentdojo/agentdojo.go` computes `succeeded :=
> injectionReachedContext && sinkExecuted` from deterministic adjudicator verdicts (the
> code comment reads "Deterministic, in-process, no model"), so it is structurally immune
> to the three attack families the paper documents (framing wrappers, prepended-refusal,
> white-box GCG on judge weights) — there is no judge model in the decision to flip.
> Mechanism not adopted (a measurement-validity finding, no kernel surface) and not a
> threat (the adversary targets other people's judges). Honest fence: the immunity is
> specific to fak's structural flow-reachability witness; the day fak reports a
> judge-mediated ASR it must adopt the paper's protocol. No code change.**

**Source:** https://arxiv.org/abs/2606.25487 — "How Reliable Is Your Jailbreak Judge?
Calibration and Adversarial Robustness of Automated ASR Scoring", Yang Gao (v2, submitted
2026-06-24). Read from the arXiv abstract as surfaced by the scout on 2026-06-27; this is a
surface read of the abstract, not a paper audit or a reproduction of its tables.

## What it is

A **measurement-validity** study, not a serving system, an attack on a model, or a
defense. Its target is the **metric**, not the model: almost every LLM-jailbreak and
prompt-injection paper reports an **attack-success rate (ASR)**, and that number is
assigned not by people but by an **automated judge** — either a **safety classifier**
trained for the task or a **general chat model prompted to grade**. The judge is rarely
itself checked. The paper checks it, then attacks it.

Using **596 human-labeled completions** from the HarmBench classifier validation set, it
compares the two judge families against human majority votes and finds they **fail in
opposite ways**:

- **The dedicated classifier over-flags** — precision **0.835**, recall **0.974** (it
  rarely misses a true positive but raises many false alarms).
- **Three LLM-as-judges keep high precision (0.81-0.94) but show erratic recall
  (0.06-0.65)** — so *the same set of responses produces very different ASR depending on
  which judge scores it*. The metric is not stable across judges.

And they differ sharply in **adversarial robustness**:

- **Benign framing wrappers that leave the harmful text untouched** (e.g. a polite
  preamble) flip **every LLM-judge between 57% and 100%** of the time; a **single
  prepended refusal sentence accounts for 39-88%** of that.
- **The dedicated classifier resists those surface attacks** (at most **6.7%**), but a
  **white-box GCG attack on its open weights flips 70% of its confident true positives**
  (21 of 30; 95% CI 54-86%) at a *small* optimization budget.
- A **two-annotator audit of 80 sampled flips** confirms the attacks **leave the harm
  intact** — every flip still contained the harmful content. The judge changed its verdict;
  the response did not.

Conclusion and recommendation: because a large and growing share of reported ASR comes
from LLM-judges, **many such numbers are unreliable both on average and under deliberate
pressure**. Papers should **report judge precision and recall on a human-labeled slice**,
**report ASR corrected for judge precision**, and **include an adversarial check of the
judge**. (The broader 2026 literature agrees the problem is real but the *fine-tuned
HarmBench classifier* is among the more stable single judges — ~93% human agreement, low
variance — while LLM-judges and substring heuristics are weaker; this paper sharpens that
into a calibration + adversarial-robustness audit of both families.)

## The three triage questions

fak is an **agent kernel**: one Go binary at the tool-call seam that adjudicates every tool
call before it runs — a default-deny **capability floor** (security gate) plus
do-the-shared-setup-once cross-turn **reuse** (performance gate). Against that mission:

- **Adopt as a capability? No.** There is no kernel mechanism here. "Audit the calibration
  and adversarial robustness of an ASR judge" is a **measurement protocol for evaluations**,
  not a runtime fak governs or a serving primitive fak implements. The one importable thing
  is a *fence for fak's own metric reporting*, recorded below — not a feature.
- **Defend against as a threat? No.** There is no adversary against fak here. The attacks in
  the paper target **judges** (a safety classifier; prompted chat models) — components fak
  **does not run** in its success decision (next section). The "GCG on open weights" attack
  needs a judge model with exposed weights on the scoring path; fak has none.
- **Cite as prior art? Yes — at the metric-validity level.** It is independent, *measured*
  evidence for a load-bearing fak design choice: fak's success criterion is a structural
  action/flow witness, not an LLM-judged harm score (next two sections).

## Where fak actually stands — fak's ASR has no judge in it

The paper's central object is the sentence *"that number is assigned not by people but by an
automated judge."* fak's number is assigned by **neither**. `internal/agentdojo` — fak's
adaptive AgentDojo-style ASR battery — computes attacker success from the **adjudicator's
deterministic verdicts**, and says so in code:

> *"Deterministic, in-process, **no model**."* — `internal/agentdojo/agentdojo.go:230`

The success criterion is two structural facts AND-ed together
(`agentdojo.go:243-278`):

```go
injectionReachedContext := !quarantined          // a detector VerdictQuarantine outcome (:254)
sinkExecuted            := !sinkDenied            // a sink-gate VerdictDeny outcome     (:268)
succeeded := injectionReachedContext && sinkExecuted   // the attacker wins iff the SINK ran (:277)
```

and `Score` simply counts `succeeded` over the matrix: `r.ASR = Succeeded / Total`
(`agentdojo.go:299`). Mapped against the paper:

| The paper's object | The fak seam at the same altitude |
|---|---|
| ASR assigned by a **safety classifier** (P 0.835 / R 0.974) or a **prompted chat model** (erratic R 0.06-0.65) | ASR assigned by the **adjudicator's verdict**: `succeeded` iff the egress/destructive **sink call was admitted** (`VerdictDeny` ⇒ not admitted). No classifier, no chat-model grader. |
| **Benign framing wrappers flip the judge 57-100%**; a prepended refusal sentence 39-88% | The success decision reads a **`VerdictDeny` on PROVENANCE** (the session is tainted by the untrusted read, so the sink is denied) — *not* the wording of the response. You cannot "frame" your way past a provenance-gated deny; the same outcome as [MIRROR #1007](RESEARCH-mirror-novelty-mcts-redteam-triage-2026-06-27.md) / [ShareLock #911](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md): *barred regardless of phrasing*. |
| **White-box GCG flips 70% of the classifier's confident TPs** (needs the judge's open weights on the scoring path) | There is **no judge model with weights** on fak's success path to optimize against. The verdict is a deterministic gate decision, not a differentiable score. |
| **The metric is unstable across judges** (same responses, different ASR) | The metric is **reproducible by construction**: a fixed corpus (`corpus_hash` sha256-pinned), a fixed adjudicator, a deterministic in-process run — re-running yields the same `asr_fullstack` / `asr_detection`. |

The recorded numbers (`experiments/agent-live/agentdojo-fak-fullstack-20260625.json`, and
the 2026-06-27 re-run): **`asr_fullstack = 0` (0/38)**, **`asr_detection = 0.763` (29/38)**,
`catch_reasons = {TRUST_VIOLATION: 29, MALFORMED: 9}` — a **closed structural reason
vocabulary**, not a harm grade — and **`benign_completion_rate = 1`** (2/2 benign controls
complete, sink executed). That last figure matters against the paper's *precision* axis: the
dedicated classifier the paper audits **over-flags** (P 0.835); fak's structural gate did
**not** flag either benign control, because "did the gate deny a tainted→sink flow" is a
different, sharper question than "does this text look harmful."

## The sharp, honest insight

The paper proves that the **dominant way ASR is computed in the literature** — an LLM-judge
grading whether a completion is harmful — is *uncalibrated on average* and *adversarially
flippable without changing the harm*. fak computes ASR a **different way**: a deterministic
**flow-reachability + action-admission witness**. The two ways are not interchangeable, and
the paper is the measured argument for fak's choice:

- **Every attack family the paper lands works on the JUDGE, and fak has no judge to land
  them on.** Framing wrappers and prepended-refusal sentences flip a *grader reading the
  words*; GCG flips a *classifier with weights*. fak's success bit is `sinkExecuted` — a
  property of the **adjudicator's verdict on the action**, set by PROVENANCE taint, not by
  the words of the response or any model's score. So all three attacks are **out of
  fak's metric by construction**, the same way ShareLock's information-theoretic dispersal
  and MIRROR's novelty are out of fak's *defense* by construction: they optimize against a
  detector/judge that fak keeps **off the load-bearing path**.

- **This is the metric-side twin of detector-is-not-the-floor.** fak's standing position is
  that a learned/text-reading **detector** is best-effort and evadable, hosted *above* the
  floor, never *on* it ([defensive-misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md),
  [AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md),
  and the [FRONTIER survey](FRONTIER-AGENT-LEADERBOARD-SURVEY-2026-06.md) HAL anchor —
  *"8 major agent benchmarks gameable to near-perfect without solving tasks"*). This paper
  adds the **measurement** corollary: a learned/text-reading **judge** is best-effort and
  evadable, so an ASR built on it is best-effort too. fak refuses the detector on the floor
  *and* refuses the judge in the metric, for the same reason.

- **The honest boundary — and where fak is NOT covered.** fak's immunity is specific to its
  **structural flow witness**. It does **not** mean "fak's ASR is the only valid one"; it
  means "fak's ASR measures a *different, action-level* event that a judge cannot flip."
  Two things fak's structural witness does **not** measure, and must not pretend to: (1) it
  does not score **content harm in the model's text** (the thing the paper's judges grade) —
  fak gates the *action*, so a response that *says* something harmful but issues no denied
  sink is, correctly, not an attack *success* in fak's sense (it is a different threat
  model). (2) The `asr_detection=0.763` layer is a *reachability* count (did the content
  detectors quarantine), still structural — but the detectors themselves remain the
  ~100%-evadable best-effort rung fak keeps off the floor. The number fak reports is the
  **full-stack** one (`0`), and only with the corpus boundary stated.

## The one importable thing — a fence for fak's own metric reporting (recorded, not a feature)

There is a genuine, fak-shaped takeaway, recorded here precisely so it is **not mistaken for
a defect to fix today**. fak's ASR is structural and judge-free *now*. The paper's
recommendations become **mandatory** the moment fak reports any ASR that is **judge-mediated**:

1. A future **content-harm probe** (if fak ever scores the model's text, not just the
   action) would be a judge — and would inherit this fragility unless it reports **judge
   precision/recall on a human-labeled slice**, reports **ASR corrected for judge
   precision**, and ships an **adversarial check of the judge**.
2. Any **comparison against an external benchmark** whose ASR is **LLM-graded** (much of the
   AgentDojo / jailbreak literature) must carry the caveat that the *external* number is
   judge-assigned and may be uncalibrated/flippable — fak's structural `0` and an external
   LLM-judged ASR are **not the same kind of number** and must never be blended.

This is a **fence to hold**, not a hole in this increment: today fak's success criterion is
deterministic and model-free, which is exactly why this paper is a citation *for* it.

## Triage decision

- **Adopt?** No — a measurement-validity protocol, no kernel mechanism. The importable
  piece is a metric-reporting fence (above), recorded, not built.
- **Defend against?** No — the adversary targets *judges* (classifiers / prompted graders)
  that fak does not run in its success decision; there is no attack surface here on fak.
- **Cite as prior art? Yes** — measured, external evidence that judge-assigned ASR is
  uncalibrated and adversarially flippable, validating fak's deterministic, model-free
  success criterion (`agentdojo.go` "no model"; `succeeded := injectionReachedContext &&
  sinkExecuted`) and the metric-side of detector-is-not-the-floor. Sibling to
  [AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md),
  [defensive-misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md), and the
  [FRONTIER survey](FRONTIER-AGENT-LEADERBOARD-SURVEY-2026-06.md) HAL "benchmarks gameable"
  anchor.

**Action:** close [#1097](https://github.com/anthony-chaudhary/fak/issues/1097) as triaged
→ **prior art recorded; not adopted as a capability, not a threat**, with one metric-reporting
fence carried forward. No code change in this increment: the right artifact for a
measurement-validity candidate is the recorded verdict + the witnessed mapping to fak's
existing success criterion, not a speculative feature.

**Next step (the smallest honest follow-on, if pursued):** when the [MIRROR #1007 /
#909](RESEARCH-mirror-novelty-mcts-redteam-triage-2026-06-27.md) evaluation upgrade lands an
**adaptive cross-surface ASR-under-containment** number, keep that number on the **same
structural witness** this note documents (`sinkExecuted` / `VerdictDeny`), and if any rung of
it ever becomes judge-mediated, attach this paper's three checks (judge P/R on a
human-labeled slice + precision-corrected ASR + an adversarial judge check) to the
`internal/agentdojo` battery. Filed as an agentdojo-lane fence, not built here.

**Scout calibration (no code change).** The candidate surfaced under topic
`prompt-injection-defense` (score **47**) on genuine `defense` / `jailbreak` / `prompt
injection (abs)` terms plus a recency/freshness bonus (≤30d). This is a **closer-to-mission**
hit than the day's other two security candidates land on fak's *metric* rather than only its
mechanism — it is exactly the "agent-security-evaluation" lane the scout exists to keep
current. The scout behaved **as designed** (judges *new and on-topic*, never *worth
building*, and hands the call to human triage — see [`docs/idea-scout.md`](../idea-scout.md)),
so there is **no change to `tools/idea_scout.py`** — only this recorded verdict.

## Related

Same-day idea-scout batch (filed together, triaged separately):
[#1098](https://github.com/anthony-chaudhary/fak/issues/1098) ("Agents That Know Too Much: A
Data-Centric Survey of Privacy in LLM Agents" — whose headline finding, *only information-flow
control covers both compositional and cross-session inference leakage*, is independent
survey-level validation of fak's IFC `Ref.Taint` sink-gating) ·
[#1099](https://github.com/anthony-chaudhary/fak/issues/1099) ("Position Rebinding Cache
Reuse" — maps onto fak's non-prefix KV-reuse correctness work,
`internal/gateway/kv_nonprefix_reuse_gap_test.go` and the [Kamera
triage](RESEARCH-kamera-position-invariant-kv-triage-2026-06-26.md)). Thesis siblings:
[AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md) ·
[defensive-misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md) ·
[ShareLock #911](RESEARCH-sharelock-multitool-threshold-poisoning-triage-2026-06-26.md) ·
[MIRROR #1007](RESEARCH-mirror-novelty-mcts-redteam-triage-2026-06-27.md) ·
[FRONTIER agent-leaderboard survey #1068](FRONTIER-AGENT-LEADERBOARD-SURVEY-2026-06.md)
(the HAL "benchmarks gameable" anchor; the ASR-0.000 floor with its corpus boundary). Witness
artifacts: `internal/agentdojo/agentdojo.go` ·
`experiments/agent-live/agentdojo-fak-fullstack-20260625.json`.
