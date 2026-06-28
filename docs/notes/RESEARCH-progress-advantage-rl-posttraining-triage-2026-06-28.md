---
title: "idea-scout triage: 'Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents' — an RL-post-training reward-modeling result (an implicit step-level advantage read free off the policy/reference log-prob ratio), not a serving mechanism, an attack, or a defense. Verdict: prior art to cite — the reward-modeling-side arrival of fak's 'a free byproduct of the existing pipeline replaces a dedicated trained judge' thesis (dos verify grades the recorded diff, no trained reward model), with a SHARP boundary: the paper's signal is the policy's OWN log-probability — a model-internal, confidence-like score that, by fak's detector-is-not-the-floor stance, hosts ABOVE the trust floor, never ON it. Not adopted as a capability (fak does not RL-post-train models and has no paired reference policy for its served fleet, so the 'free lunch' is unavailable on fak's actual models; no kernel surface) and not a threat. One recorded residual — progress advantage as a CHEAP PROPOSAL signal for fak's trajectory graders (the failure-attribution application maps onto dos verify's failure-to-witness + the dojo eval step) — is NOT adopted, gated behind a paired-policy availability fak lacks and fenced as never-the-floor. No code change (2026-06-28)"
description: "Triage of the idea-scout candidate arXiv:2606.26080 (Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh, Tanwi Mallick, Sharon Li — 'Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents', submitted 2026-06-24): process reward models give fine-grained step-level evaluation of LLMs, but building them for agentic settings is prohibitively hard (long-horizon interactions, irreversible actions, stochastic environment feedback defeat human annotation and Monte Carlo estimation at scale). The paper shows RL post-training ALREADY contains the ingredients for step-level scoring, eliminating dedicated reward-model training: under a general stochastic MDP, the log-probability ratio between the RL-trained policy and its reference policy EXACTLY recovers the optimal advantage function — a signal it terms progress advantage that is annotation-free, domain-agnostic, and free as a byproduct of the standard RL post-training pipeline. Validated across test-time scaling, uncertainty quantification, and failure attribution on five benchmarks and four model families; it beats confidence-based baselines and, with no task-specific training, surpasses dedicated trained reward models. Verdict: prior art to cite — at the thesis level, with a sharp boundary. The paper's central economics ('the existing pipeline already contains the step-level signal; you do not need to train a dedicated judge') is the reward-modeling-side sibling of fak's witness referee: dos verify / dos commit-audit grade a claim against the diff git ACTUALLY recorded — a derived, annotation-free witness — instead of training or trusting a reward/judge model or believing the worker's self-report. The sharp fence: progress advantage is the policy's OWN log-probability (the paper benchmarks it against confidence-based baselines), i.e. a model-internal self-referential score; fak's recorded position is detector-is-not-the-floor / witness-not-self-report, so a signal read off the model's own probabilities is exactly the self-report fak keeps ABOVE its load-bearing trust floor, never on it — the same demotion fak applies to CoT-as-explanation (#1008) and attention-as-explanation (#861/#862). fak's witness is external, action-level, git-recorded fact the model cannot author; the progress advantage is internal confidence the model does author. Not adopted as a capability: fak serves and gates tool calls, it does not RL-post-train models, and the 'free lunch' is computable only by whoever ran the post-training (you need both the RL policy and its reference policy log-probs) — fak's served fleet is third-party/local models with no paired reference policy, so the signal is unavailable on fak's actual models; there is no kernel surface to fold in. Not a threat (no adversary). One recorded residual, NOT adopted: fak's dispatch fleet and dojo gym (#951) DO grade agent trajectories step by step (closure honesty + the witness ledger; the dojo predict->run->measure->eval->calibrate loop), and the paper's failure-attribution application — 'which step of a failed trajectory lost the most progress' — maps directly onto fak's failure-to-witness need; progress advantage is a candidate CHEAP PROPOSAL signal there, but it is not adopted because (1) it needs paired policy/reference log-probs fak does not have for its served models, and (2) per fak's own thesis a model-internal score can only host above the evidence floor, never replace the diff witness. Scout-calibration note: surfaced under topic agent-model-arch (score 44) on a genuine training(title) term + freshness (3d old, <=30d) — correctly on-topic; the scout judges new-and-on-topic, never worth-building, and handed the call to human triage. It is the third recent agent-model-arch hit that is a TRAINING-time result fak cannot adopt as a capability (after #1008 CoT-training; cf. the #861/#862 reward-over-spans line) — an on-topic-by-keyword/off-mission-by-content pattern for the `training` scorer worth watching, but NOT a scoring bug (relevance is judged correctly; 'fak does not train' is a triage-time call, and these training papers still yield citable thesis-level prior art), so no change to tools/idea_scout.py."
---

# idea-scout triage — progress advantage: a free RL-post-training step reward (issue #1122)

> Closes the daily idea-scout candidate [#1122](https://github.com/anthony-chaudhary/fak/issues/1122)
> (`tools/idea_scout.py`, filed 2026-06-28). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt as a
> capability, defend against as a threat, or cite as prior art (see
> [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — at the *thesis* level, with a sharp boundary. The
> paper's "the existing pipeline already contains the step-level signal, so you need no
> dedicated trained judge" is the reward-modeling-side sibling of fak's witness referee
> (`dos verify` grades the recorded diff, not a trained reward model). But its signal is
> the policy's OWN log-probability — a model-internal, confidence-like score — and fak's
> recorded position is detector-is-not-the-floor / witness-not-self-report: a self-report
> read off the model's own probabilities hosts ABOVE fak's trust floor, never on it. Not
> adopted as a capability (fak does not RL-post-train, and has no paired reference policy
> for its served models, so the "free lunch" is unavailable on fak's actual fleet; no
> kernel surface) and not a threat. One recorded residual — progress advantage as a cheap
> PROPOSAL signal for fak's trajectory graders — is NOT adopted. No code change.**

**Source:** https://arxiv.org/abs/2606.26080 — "Neglected Free Lunch from Post-training:
Progress Advantage for LLM Agents", Changdae Oh, Wendi Li, Seongheon Park, Samuel Yeh,
Tanwi Mallick, Sharon Li (submitted 2026-06-24). Read from the arXiv abstract as surfaced
by the scout on 2026-06-28; this is a surface read of the abstract, not a paper audit.

## What it is

A **reward-modeling / RL-post-training** result, not a serving system, a protocol, an
attack, or a defense. The premise: **process reward models** (PRMs) give fine-grained,
**step-level** evaluation of an LLM's reasoning, but building one for an **agentic**
setting is prohibitively hard — long-horizon interactions, **irreversible actions**, and
**stochastic environment feedback** defeat both human annotation and Monte-Carlo
estimation at scale.

The claim is that **RL post-training already contains the ingredients** for step-level
scoring, so a dedicated PRM need not be trained at all. Under a **general stochastic
MDP**, the paper derives that the **log-probability ratio between the RL-trained policy
and its reference policy exactly recovers the optimal advantage function** — a signal it
names **progress advantage**. It is therefore **annotation-free, domain-agnostic, and
available as a byproduct of the standard RL post-training pipeline** (the same implicit-
reward shape the DPO/PPO-reference family uses: `r(s,a) ∝ log[ π_RL(a|s) / π_ref(a|s) ]`).

The paper **validates** the signal on three applications — **test-time scaling**,
**uncertainty quantification**, and **failure attribution** — across **five benchmarks**
and **four model families**, reporting that it consistently beats **confidence-based
baselines** and, with **no task-specific training**, **surpasses dedicated trained reward
models**.

It proposes a **derivation + a measurement**, used at inference to *score* an existing
agent's steps. It does not propose a kernel component, a runtime mechanism, an MCP server,
or a serving primitive fak would implement.

## The three triage questions

fak is an **agent kernel**: one Go binary at the tool-call seam that adjudicates every
tool call before it runs — a default-deny **capability floor** (security gate) plus
do-the-shared-setup-once cross-turn **reuse** (performance gate). Against that mission:

- **Adopt as a capability? No.** There is no mechanism to fold into the kernel, and the
  "free lunch" is **not even computable on fak's models**. Progress advantage needs the
  log-probs of **both** the RL-trained policy **and its reference policy** for the *same*
  served model — it is free **only to whoever ran the RL post-training**. fak does not
  train models; it serves and gates tool calls over **third-party / local** weights for
  which there is **no paired reference policy**. The "byproduct of the standard RL
  post-training pipeline" is a pipeline fak does not run, so the byproduct is absent. This
  is the same off-mission shape as the [CoT-training-gains triage (#1008)](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md)
  and the [data-centric privacy survey (#1098)](RESEARCH-agent-privacy-data-centric-survey-triage-2026-06-27.md):
  a training/measurement result, no runtime mechanism fak governs.
- **Defend against as a threat? No.** There is no adversary, attack, or failure mode. It
  is a reward-modeling result on the same side of fak's trust boundary.
- **Cite as prior art? Yes — at the *thesis* level, not the mechanism level**, and with a
  boundary sharp enough that it does **not** get mistaken for an endorsement of using a
  model-internal score on the trust floor (next two sections).

## Why this is prior art to cite — a free pipeline byproduct beats a trained judge

fak's whole referee design is a refusal to **train or trust a dedicated judge**. The
`dos verify` / `dos commit-audit` truth syscall grades a claimed result against the
**diff git actually recorded** — a derived, annotation-free witness already produced by
the normal commit pipeline — never against a learned reward model and never against the
commit message the author wrote (a forgeable self-report). The recorded economics is
exactly the paper's headline: **the existing pipeline already contains the step-level
signal you would otherwise pay to train a judge for**. The paper makes that argument from
the **RL-reward-modeling** direction (the post-training run already encodes the optimal
advantage in the policy/reference ratio); fak makes it from the **evidence** direction
(the version-control run already encodes the step's effect in the diff). Both replace an
annotation-hungry trained judge with a **free byproduct of the pipeline that was going to
run anyway** — and the paper's empirical result that the free signal *outperforms
dedicated trained reward models* is independent support for that "don't build the judge"
bet.

## The sharp boundary — the signal is a self-report, and fak keeps those off the floor

The citation stops at the economics. **What the two signals are is opposite.** Progress
advantage is the **policy's own log-probability** — a **model-internal, self-referential**
score (the paper benchmarks it precisely against **confidence-based baselines**, i.e. it
lives in the same family as the model's own confidence). fak's recorded position is
**detector-is-not-the-floor / witness-not-self-report** (`CLAIMS.md` "Honest ceiling":
detection is ~100% evadable, non-load-bearing; cf.
[defensive-misdirection](RESEARCH-defensive-misdirection-triage-2026-06-23.md) and
[AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md)). A
score read off the model's **own probabilities** is exactly the kind of self-report fak
hosts **above** its load-bearing floor, never **on** it.

So the honest fence is the same one drawn for the **CoT-as-explanation** sibling
([#1008](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md)) and the contested
**attention-as-explanation** demotion in fak's reward-over-spans epic
([#861](https://github.com/anthony-chaudhary/fak/issues/861)/[#862](https://github.com/anthony-chaudhary/fak/issues/862)):
**a self-generated signal — CoT text, attention mass, or a policy log-prob ratio — is a
*proposal* to be verified against an action-level witness, never an accepted causal
account.** fak's witness is **external, action-level, and git-recorded** — a fact the
model cannot author; progress advantage is **internal confidence the model does author**.
They sit on opposite sides of the trust boundary and must be cross-linked, not conflated.

## The one recorded residual — a proposal signal for the trajectory graders, NOT adopted

There is a genuine, fak-shaped follow-on, recorded here precisely so it is **not mistaken
for a shipped lever**. fak's **dispatch fleet** and **dojo gym** already grade agent
**trajectories step by step** — closure honesty + the witness ledger on the fleet side,
and the `fak dojo` **predict→run→measure→eval→calibrate** loop
([#951](https://github.com/anthony-chaudhary/fak/issues/951)) on the gym side. The paper's
**failure-attribution** application — *which step of a failed trajectory lost the most
progress?* — maps directly onto a real fak need (attributing a failed dispatch to the step
that broke it, today done by the diff/witness referee, not a per-step score). Progress
advantage is a candidate **cheap proposal signal** to *rank* steps for that attribution.

It is **not adopted in this increment**, for two honest reasons:

1. **It is unavailable on fak's models.** Computing it needs paired RL-policy and
   reference-policy log-probs for the served model; fak's fleet is third-party / local
   weights with no paired reference policy, so the signal cannot be produced where the
   need is. (This is the same "the prerequisite pipeline isn't fak's" wall as the adopt
   question above.)
2. **It can only ever be a proposal, never the floor.** Even where computable, a
   model-internal score is, by fak's own thesis, hosted **above** the evidence floor — it
   may *propose* which step to look at, but the **attribution that counts** is still the
   action-level, git-recorded witness (`dos verify`). A trajectory grader that *decided*
   on the log-prob ratio would be trusting a self-report.

So the residual is filed as a **candidate proposal signal** for the dispatch/dojo
trajectory graders — gated behind a paired-policy availability fak lacks, and explicitly
fenced as **never-the-floor**. Not a feature, and explicitly not shipped here.

## Triage decision

- **Adopt?** No — an RL-post-training reward-modeling result; no kernel mechanism, and the
  signal is not even computable on fak's served (untrained-by-fak, no-reference-policy)
  models.
- **Defend against?** No — not a threat.
- **Cite as prior art?** **Yes** — at the thesis level: independent, reward-modeling-side
  evidence for fak's "a free byproduct of the existing pipeline beats a dedicated trained
  judge" referee bet (`dos verify` over the diff), with a **sharp boundary** that the
  paper's signal is a model-internal self-report fak keeps **above** the floor — the
  policy-log-prob sibling of the CoT-as-explanation demotion in
  [#1008](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md) and the
  attention-as-explanation demotion in #861/#862.

**Action:** this note is the recorded triage; close [#1122](https://github.com/anthony-chaudhary/fak/issues/1122)
as triaged → **prior art recorded, not adopted as a capability or defended as a threat**.
No code change in this increment: the right artifact for a reward-modeling-result candidate
is the recorded verdict and the named boundary, not a speculative feature.

**Scout calibration (no code change).** The candidate surfaced under topic
`agent-model-arch` (score **44**) on a genuine `training (title)` term plus a freshness
bonus (≤30d, 3d old) — **correctly on-topic**. It is the **third** recent `agent-model-arch`
hit that is a **training-time** result fak cannot adopt as a capability (after the
[#1008 CoT-training triage](RESEARCH-cot-training-gains-agents-triage-2026-06-27.md);
cf. the #861/#862 reward-over-spans line) — an **on-topic-by-keyword / off-mission-by-
content** pattern for the `training` scorer worth *watching*. It is **not** a scoring bug:
the scout judges *new and on-topic*, never *worth building*, and these training papers
still yield **citable thesis-level prior art** (this one included), so suppressing the
`training` term would lose signal, not just noise. The scout behaved **as designed** and
handed the call to human triage (see [`docs/idea-scout.md`](../idea-scout.md)); there is
**no change to `tools/idea_scout.py`** — only this recorded verdict.
