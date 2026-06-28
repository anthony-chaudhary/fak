---
title: "idea-scout triage: 'Where Do CoT Training Gains Land in LLM based Agents?' — a CoT-faithfulness / interpretability study of agent training, not a serving mechanism, an attack, or a defense. Verdict: prior art to cite — independent evidence for fak's witness-not-self-report / detector-is-not-the-floor stance (verbalized CoT is post-hoc; the action is largely predictable from the prompt before any reasoning), with a clean named boundary on the span-attention faithfulness question fak's own #861/#862 is still auditing. Not adopted as a capability (a training/measurement finding, no kernel surface) and not a threat. Recorded residual: a CoT-token-economy hypothesis for the token-saving program (dojo #951) — NOT adopted, because acting on it needs a per-model, per-task accuracy acceptance gate fak does not have, and the paper itself shows CoT still changes the action on the hard tail. No code change (2026-06-27)"
description: "Triage of the idea-scout candidate arXiv:2606.26935 (Jingyu Liu, Zhiwen Wang, Yuxin Jing, Huanyu Zhou, Yong Liu — 'Where Do CoT Training Gains Land in LLM based Agents?'): an interpretability study that asks what CoT training actually improves in language-model agents — getting better at CHANGING the action through generated reasoning, or getting better at predicting the action DIRECTLY from the prompt without CoT. It compares prompt actions (no-CoT prediction) against CoT actions across training checkpoints and finds prompt-action quality improves substantially, i.e. much of the measured gain lands in the model's direct prompt->action mapping rather than in reasoning-driven action change — consistent with prior CoT-unfaithfulness / post-hoc-rationalization findings. Verdict: prior art to cite. fak never reads the model's verbalized reasoning for a security verdict and never trusts a self-report; it gates the resulting ACTION at the tool-call seam (default-deny capability floor) and re-checks claimed work against the diff (dos verify / dos commit-audit). This paper is independent training-time evidence for that detector-is-not-the-floor / witness-not-self-report bet — the action is largely fixed by the prompt before the CoT is written, so a CoT-reading detector would be auditing a post-hoc narrative. It also draws a sharper line for fak's own contested-faithfulness work (#860-#866, span-attention as a proposal signal not a causal explanation): the same skepticism that demotes attention-as-explanation demotes CoT-as-explanation. Not adopted as a capability (an interpretability/measurement result, no kernel mechanism to fold in) and not a threat (no adversary). One recorded residual, NOT adopted: if prompt-action quality is high, the CoT tokens are non-load-bearing on the easy mass of steps, which is a token-economy lever for the dojo gym (#951) / token-saving-defaults program — but only behind a per-model, per-task accuracy acceptance gate fak does not yet have, and the paper's own finding is that CoT still changes the action on the hard tail, so blind CoT-dropping is unsafe and explicitly not shipped here. Scout-calibration note: surfaced under topic agent-model-arch (score 50) on genuine training(title)/checkpoint/reasoning terms + freshness — correctly on-topic and closer-to-mission than the day's #1009 pedagogy hit (it touches the self-report/faithfulness axis fak's security thesis rests on), though still a training/interpretability study rather than a serving mechanism. The scout worked as designed; no change to tools/idea_scout.py."
---

# idea-scout triage — where do CoT training gains land in agents? (issue #1008)

> Closes the daily idea-scout candidate [#1008](https://github.com/anthony-chaudhary/fak/issues/1008)
> (`tools/idea_scout.py`, filed 2026-06-27). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — independent training-time evidence for fak's
> witness-not-self-report / detector-is-not-the-floor stance. Verbalized CoT is
> largely post-hoc; across checkpoints the action becomes more predictable DIRECTLY
> from the prompt, before any reasoning is written. That is exactly why fak gates the
> ACTION at the tool-call seam and never reads the model's reasoning for a verdict.
> Not adopted as a capability (an interpretability result, no kernel surface) and not a
> threat. One recorded residual — a CoT-token-economy hypothesis — is NOT adopted in
> this increment. No code change.**

**Source:** https://arxiv.org/abs/2606.26935 — "Where Do CoT Training Gains Land in
LLM based Agents?", Jingyu Liu, Zhiwen Wang, Yuxin Jing, Huanyu Zhou, Yong Liu
(submitted 2026-06-25). Read from the arXiv abstract as surfaced by the scout on
2026-06-27; this is a surface read of the abstract, not a paper audit.

## What it is

An **interpretability / training-analysis** study, not a serving system, a protocol, an
attack, or a defense. Its premise leans on prior work that **verbalized CoT is not always
faithful** — it can be **post-hoc rationalization**, where the model already "knows" the
answer before it writes the reasoning. The paper turns that into a question about
**training**: when CoT training improves an agent, *where does the gain land*? Is the model
getting better at **changing its action through the generated reasoning** (CoT actions), or
better at **predicting the action directly from the prompt** with no CoT at all (prompt
actions)?

Its method is to measure both across training **checkpoints**: a **prompt action**
(predict the action without CoT) versus a **CoT action** (predict the action with CoT). The
reported finding is that **prompt-action quality improves substantially** across
checkpoints — i.e. a large share of the measured CoT-training gain shows up in the model's
**direct prompt→action mapping**, not in reasoning-driven action change. That is consistent
with the post-hoc / unfaithfulness reading: the answer is increasingly fixed by the prompt
before the CoT is generated.

It proposes a **measurement and a finding about model training**. It does not propose a
kernel component, a runtime mechanism, an MCP server, or a serving primitive.

## The three triage questions

fak is an **agent kernel**: one Go binary at the tool-call seam that adjudicates every tool
call before it runs — a default-deny **capability floor** (security gate) plus
do-the-shared-setup-once cross-turn **reuse** (performance gate). Against that mission:

- **Adopt as a capability? No.** There is no mechanism to fold into the kernel. "Compare
  prompt-action vs CoT-action across checkpoints" is a **model-training measurement**, run
  against weights during training, not a runtime fak governs or a serving primitive fak
  implements. fak does not train models; it serves and gates tool calls.
- **Defend against as a threat? No.** There is no adversary, attack, or failure mode here.
  It is an interpretability result on the same side of, and largely orthogonal to, fak's
  trust boundary.
- **Cite as prior art? Yes — at the *thesis* level, not the mechanism level.** The paper is
  independent, training-time evidence for a load-bearing fak design choice (next section).

## Why this is prior art to cite — witness-not-self-report, from the training side

fak's security floor is built on a deliberate refusal to trust the model's **stated
reasoning**. The kernel never reads the CoT to decide whether a tool call is allowed; it
gates the **action** the call would perform (default-deny capability floor) and, on the
result side, quarantines and IFC-taints what comes back (`AdmitResult` + `wirescreen` /
`SinkGate`). The `dos verify` / `dos commit-audit` referee makes the same move one altitude
up: it grades a claim against the **diff git actually recorded**, never against the commit
message the author wrote. The recorded position is **detector-is-not-the-floor** — a learned
or text-reading detector is best-effort and evadable, hosted *above* the floor, never *on*
it (`CLAIMS.md` "Honest ceiling": detection is ~100% evadable, non-load-bearing; cf.
[defensive-misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md) and
[AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md)).

This paper is the same skepticism arriving from the **training** direction. If CoT training
gains land mostly in the **prompt→action** mapping — if the action is increasingly fixed by
the prompt **before** the reasoning is verbalized — then a security or routing decision that
**reads the CoT** is auditing a **post-hoc narrative**, not the cause of the action. That is
precisely the failure mode fak's witness-not-self-report design avoids: it gates the action,
not the explanation of the action. The paper is worth citing as external, independent
support for that bet — alongside the unfaithfulness/post-hoc literature it builds on.

**One sharper boundary — it tightens fak's own contested-faithfulness work.** fak's
reward-over-spans epic ([#860](https://github.com/anthony-chaudhary/fak/issues/860)-#866)
treats **attention-as-explanation as contested** and demotes raw span attention to a *cheap
proposal signal* that must be confirmed by exact-eviction leave-one-out before it can drive
anything (see
[witnessed-span-attention-validity](RESEARCH-witnessed-span-attention-validity-2026-06-26.md)).
This paper is the **CoT-as-explanation** sibling of that same demotion: the verbalized chain,
like the attention map, is a plausible-looking narrative that need not be the causal driver.
The honest fence is identical — **a self-generated explanation (CoT or attention) is a
proposal to be verified against an action-level witness, never an accepted causal account**.
The two should be cross-linked, not conflated: this is a training-checkpoint behavioral
finding about CoT; #861/#862 is a per-decode mechanistic question about attention mass.

## The one recorded residual — a token-economy hypothesis, NOT adopted

There is a genuine, fak-shaped follow-on thought, and it is recorded here precisely so it is
**not mistaken for a shipped lever**. If prompt-action quality is high — if the model already
predicts the right action directly from the prompt on the easy mass of steps — then the CoT
tokens are **non-load-bearing** on those steps, and *skipping the CoT* would save tokens.
Token economy is fak's lane: the `fak dojo` predict→run→measure→eval→calibrate gym
([#951](https://github.com/anthony-chaudhary/fak/issues/951)) and the token-saving-defaults
program exist to find and gate exactly this kind of lever.

It is **not adopted in this increment**, for two honest reasons:

1. **Dropping CoT is not bit-exact-off.** fak's safe token-savers default on only when they
   are witnessed bounded-loss or exactness-preserving; removing CoT changes the model's
   output distribution and can change the action, so it can only ride behind a **per-model,
   per-task accuracy acceptance gate** (a quality probe, not the bit-exact last-logit oracle).
   fak does not have that gate today.
2. **The paper's own finding forbids blind dropping.** The gain landing in prompt-action does
   **not** mean CoT is useless — it means CoT is increasingly redundant **on the steps the
   model already solves**. On the hard tail, generated reasoning still **changes the action**;
   that is the half of the comparison the abstract sets up. A blind "always skip CoT" lever
   would regress exactly those steps. The only honest form of this lever is *conditional*
   (skip CoT only where a calibrated predictor says prompt-action suffices), which is a
   measured dojo experiment, not a default.

So the residual is filed as a **candidate dojo hypothesis**, gated on an accuracy-acceptance
witness fak must first build — not a feature, and explicitly not shipped here.

## Triage decision

- **Adopt?** No — an interpretability/training-measurement result, no kernel mechanism.
- **Defend against?** No — not a threat.
- **Cite as prior art?** **Yes** — independent training-time evidence for
  witness-not-self-report / detector-is-not-the-floor, and the CoT-as-explanation sibling of
  the contested-attention demotion in #861/#862. Recorded as a thesis-level citation, with
  one token-economy residual explicitly **not adopted**.

**Action:** this note is the recorded triage; close [#1008](https://github.com/anthony-chaudhary/fak/issues/1008)
as triaged → **prior art recorded, not adopted as a capability or defended as a threat**.
No code change in this increment: the right artifact for a measurement-result candidate is
the recorded verdict and the named boundary, not a speculative feature.

**Scout calibration (no code change).** The candidate surfaced under topic
`agent-model-arch` (score **50**) on genuine `training (title)` / `checkpoint` / `reasoning`
terms plus a recency/freshness bonus (≤30d, 2d old). Unlike the day's #1009 pedagogy hit
(an on-topic-by-keyword, off-mission-by-content match), this one is **closer to mission** —
it lands on the self-report/faithfulness axis fak's security thesis rests on — even though it
is still a training/interpretability study rather than a serving or security mechanism. The
scout behaved **as designed**: it judges *new and on-topic*, never *worth building*, and
handed the call to human triage (see [`docs/idea-scout.md`](../idea-scout.md)). The scoring
was correct, so there is **no change to `tools/idea_scout.py`** — only this recorded verdict.
