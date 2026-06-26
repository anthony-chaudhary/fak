---
title: "idea-scout triage: 'Verifying Intent and Harm' — a verification-centric defense that jointly evaluates prompt intent AND response harm because attacks live in the GAP between them; this is fak's two-sided adjudication thesis stated at the detector altitude (Decide folds the pre-call INTENT chain; AdmitResult is documented in-code as 'the dual of Decide' and folds the result-side HARM chain), so it is prior art to cite + an evaluation discipline for fak's best-effort detector rung — the three-LLM-analyst mechanism is NOT adopted (detector-is-not-the-floor; a learned verifier degrades under the paper's own adaptive attacker, fak's structural default-deny does not) (2026-06-26)"
description: "Triage of idea-scout candidate arXiv:2606.26377 (Thota, Lei, Thangaraj, Jonnalagadda, Nilizadeh — 'Verifying Intent and Harm: A Unified Defense Against LLM-Generated Threats', submitted 2026-06-24): prompt-only and response-only defenses miss attacks that exploit the separation between adversarial intent expressed in the prompt and actionable harm that manifests only in the response; the paper proposes a verification-centric framework with a specialized Intent Analyst (scores prompt intent), a specialized Harm Analyst (scores response harm), and a Judge that resolves conflicts before delivery (reported F1 0.90->0.95, ASR down to 4.1%, benign-sensitive FPR 0.12->0.06 across jailbreaks/prompt-injection/phishing/cyber-abuse/harmful-content, tested under adaptive attackers who know the verifier). Verdict: prior art to cite — the join-intent-with-harm thesis is fak's two-sided adjudication design at detector altitude: fak's kernel already splits the same axis structurally (k.Decide folds ONLY the pre-call INTENT chain; k.AdmitResult is the documented EXPORTED DUAL of Decide and folds the result-side HARM chain — quarantine + IFC + wirescreen/SinkGate egress floor), and the witnessed commit audit (dos verify / dos commit-audit) re-checks the claimed intent against the actual diff-borne harm. The three-LLM-analyst+Judge MECHANISM is NOT adopted: it is a learned detector, and fak's recorded position is detector-is-not-the-floor (the paper's own adaptive-attacker result shows the learned verifier is evadable; fak's default-deny capability floor the model 'can't talk past' is not). Named residual: the Harm Analyst classifies CONTENT harm (the model emitting phishing/harmful TEXT) that fak's capability gate does not score at the tool-call seam — a real best-effort-detector rung fak runs but does not make load-bearing. Not a threat — a parallel defense on the same side of the trust boundary."
---

# idea-scout triage — joint intent+harm verification (issue #910)

> Closes the daily idea-scout candidate [#910](https://github.com/anthony-chaudhary/fak/issues/910)
> (`tools/idea_scout.py`, filed 2026-06-26). The scout judges whether a candidate is
> *new and on-topic*; this note is the human triage it hands off — adopt, defend
> against, or cite as prior art (see [`docs/idea-scout.md`](../idea-scout.md)).
> **Verdict: prior art to cite — "join the prompt's intent with the response's harm,
> because the attack lives in the GAP between them" IS fak's two-sided adjudication
> thesis stated at the detector altitude. fak already splits that exact axis
> structurally: `k.Decide` folds the pre-call INTENT chain, `k.AdmitResult` is the
> in-code documented DUAL that folds the result-side HARM chain. The three-LLM-analyst
> + Judge mechanism is NOT adopted (a learned detector is best-effort, not the floor),
> and the paper's own adaptive-attacker result is the evidence why.**

**Source:** https://arxiv.org/abs/2606.26377 — "Verifying Intent and Harm: A Unified
Defense Against LLM-Generated Threats", Poojitha Thota, Yun Lei, Santhosh Thangaraj,
Siddhartha Reddy Jonnalagadda, Shirin Nilizadeh (submitted 2026-06-24). Read from the
arXiv abstract via WebFetch on 2026-06-26; this is a surface read of the abstract, not a
paper audit or a reproduction. The abstract reports headline numbers (below) but the read
does not verify the datasets, the baseline tuning, or the adaptive-attack setup behind
them — those are taken as the authors' claims, not as confirmed results.

## The paper, in one pass

The diagnosis is one fak has made from the start, here aimed at the *defense* layer:
existing defenses inspect **either** the user prompt **or** the generated response, **not
both** — and **many real attacks exploit precisely the separation between adversarial
intent expressed in the prompt and actionable harm that manifests only in the response.**
A prompt that looks benign in isolation, a response that looks benign in isolation, and an
*interaction* that is unsafe only when the two are read together: that interaction slips
past every single-sided filter.

The fix is **verification-centric and two-sided**: evaluate prompt intent **and** response
harm jointly, *before the response is delivered.* The framework is three cooperating
components:

- a **specialized Intent Analyst** that scores the adversarial intent of the user prompt;
- a **specialized Harm Analyst** that scores the harmful content of the generated response;
- a **Judge** that resolves conflicts between the two (benign-looking prompt + harmful
  response, or hostile-looking prompt + harmless response).

Reported across five threat categories (jailbreaks, prompt injection, phishing, cyber
abuse, harmful content): average **F1 0.90 → 0.95** over the strongest baseline; **0.87 →
0.95** over a single-agent + chain-of-thought baseline; **attack success rate down to
4.1%**; **false-positive rate on benign-but-sensitive requests 0.12 → 0.06**. The authors
further test **adaptive attackers who know the verifier's structure.**

## Where fak actually stands

fak and the paper share the **same core insight** — *defense must be two-sided because the
intent and the harm are separable, and the attack hides in the gap.* They differ on
**what does the joining**: the paper joins them with three **learned LLM analysts**; fak
joins them with a **structural, two-chain adjudication seam** plus a **git-witnessed**
re-check of claim-vs-action.

| Paper's frame | fak's position |
|---|---|
| **Diagnosis:** prompt-only and response-only defenses miss attacks that exploit the *separation* between adversarial intent (prompt) and actionable harm (response) | **fak's two-sided design premise, verbatim.** fak does not bet either side alone: the kernel folds a **pre-call** chain over the *requested* call and a **result-side** chain over the *produced* result, exactly because a benign-looking request can yield a harmful result and vice versa. |
| **Intent Analyst** — score the adversarial intent of the prompt before acting | ≈ **`k.Decide`** (`internal/kernel/kernel.go`), documented as folding *"ONLY the Adjudicator chain"* — the pure pre-call adjudication of the requested capability against the default-deny policy floor (`internal/adjudicator`, `internal/policy`, `--policy FILE`). fak scores intent **structurally** (does this call's *capability* fall inside the declared task floor?), not by a learned intent classifier. |
| **Harm Analyst** — score the harm of the generated response before delivery | ≈ **`k.AdmitResult`**, documented in-code as *"the EXPORTED dual of Decide"* — it arms the result-side stack (context-MMU result-admit **quarantine**, per-trace **IFC taint** ledger, the pre-send `internal/wirescreen` redactor, the `ifc.SinkGate` egress floor in `internal/tracesink`) over a result a client produced and handed back. fak's harm side gates **what the result is allowed to DO / where it may flow**, structurally. |
| **Judge** — resolve conflicts between the two analysts before delivery | ≈ fak's **deny-as-value typed dispositions + the hash-chained witness/decision journal**: the verdict is the *most-restrictive* admission folded across the chains, recorded as a legible refusal from a closed vocabulary — and, at ship time, the **`dos verify` / `dos commit-audit`** referee re-judges the agent's *claimed* intent against the *actual* diff-borne harm (claim-vs-action), which is the same intent↔harm reconciliation one tick later in the lifecycle. |
| **Mechanism:** three specialized **LLM** analysts + an LLM Judge (F1 0.95, ASR 4.1%), tested under an **adaptive attacker who knows the verifier** | **Not adopted — this is the detector rung, and fak's floor is not a detector.** A learned verifier is **best-effort** and **evadable**; the paper's own adaptive-attack section is the evidence. fak's position, recorded across prior triages, is **detector-is-not-the-floor**: the default-deny **capability** floor the model *"can't talk past"* is what bounds harm, and a classifier rides *above* it as a non-load-bearing signal. |

So the two agree on the **shape of the defense** (two-sided, verify-before-deliver) and
differ on the **trust placed in the joiner**: a learned panel of analysts vs. a structural
capability gate witnessed by git.

## The sharp, honest insight

The paper is strong **independent validation that the intent/harm split is the right axis
to defend on** — an external group, working the detector side, concluded that a defense
must read *both* the prompt's intent and the response's harm because the unsafe signal is
in the *interaction*, not in either half. That is precisely why fak's kernel exposes
**two** adjudication entry points (`Decide` / `AdmitResult`) rather than one filter, and
why `AdmitResult` is written as *"the dual of Decide"*: the architecture already encodes
"intent here, harm there, judge the pair."

But the paper also supplies its **own** best argument for why fak does **not** make that
joiner a learned model. The framework's quality rests on the analysts' classification
accuracy (F1 0.95, FPR 0.06) — real, useful detector numbers — and the authors themselves
stress-test it against **adaptive attackers who know the verifier's structure**, the exact
regime where a learned detector's guarantee decays (cf. fak's recorded
[detector-is-not-the-floor](RESEARCH-defensive-misdirection-triage-2026-06-23.md) and
[AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md) priors).
fak's structural floor does not move under an adaptive prompt: you cannot *argue* a call
past a default-deny capability gate. The paper's verifier and fak's floor therefore sit at
**different rungs** — the paper is the kind of high-quality detector fak would run **above**
its floor, never the floor itself. This matches `CLAIMS.md`'s `0/29-NOVEL` discipline:
every primitive (intent scoring, harm scoring, a judge) is established/emerging — the
contribution is the **assembly**, and fak's assembly puts the structural gate, not the
classifier, on the load-bearing path.

The matching honest fence cuts the other way too. The **Harm Analyst classifies CONTENT
harm** — the model *emitting* a phishing email or harmful instructions as *text* — and that
is a harm fak's capability gate **does not score** at the tool-call seam: if no gated
capability is exercised, fak's structural floor has nothing to deny, and the redaction /
egress rung (`wirescreen` / `SinkGate`) bounds *flow*, not *semantic harmfulness of the
words*. Scoring whether the response text itself is harmful is exactly a **best-effort
detector** rung — one fak can host above the floor, but the paper is a reminder that this
rung is genuinely useful (5-category F1 0.95) and that fak's coverage of pure
content-harm-in-text is a named residual, not a solved problem.

## Triage decision

- **Adopt the three-analyst + Judge framework as a fak feature? No — it is the detector
  rung, not fak's floor.** fak's load-bearing defense is the **structural** default-deny
  capability gate the model cannot talk past (`Decide` / `AdmitResult`), witnessed by the
  decision journal and the `dos verify` claim-vs-action referee. A learned verifier panel
  is best-effort and, per the paper's own adaptive-attacker test, evadable; promoting it to
  the floor would re-center an evadable classifier, the exact move fak's
  detector-is-not-the-floor cluster refuses. fak may run such a verifier **above** the
  floor as a non-load-bearing signal — that is the existing best-effort-detector seam, not
  a new architecture.
- **Defend against (is this a threat to fak)? No — it is a parallel defense on the SAME
  side of the trust boundary.** The framework protects the operator/user from unsafe model
  output, exactly as fak does; there is no adversarial mechanism here to fence. (Contrast
  the [agentic-surveillance triage](RESEARCH-agentic-surveillance-evasion-triage-2026-06-25.md),
  #772, where the topology was *inverted* against the user — here it is aligned.)
- **Cite as prior art? Yes.** It is the cleanest **detector-altitude** statement of fak's
  two-sided premise (intent and harm are separable; defend the *interaction*, verify before
  deliver), and its evaluation is a usable **discipline** for any best-effort detector fak
  hosts above its floor: report F1 / ASR / **benign-sensitive FPR** *and* an
  **adaptive-attacker** result, the same bar the
  [AUC-is-not-detection](RESEARCH-probe-auc-evaluation-protocol-triage-2026-06-23.md) prior
  set. It belongs alongside the detector-is-not-the-floor priors and the `Decide`/`AdmitResult`
  dual as the paper that says — from the *defense-detector* community — *the intent/harm
  axis fak split structurally is the right one.*

**Action:** close #910 as triaged → **prior art cited (independent detector-side validation
of fak's two-sided intent/harm adjudication — `Decide` folds intent, `AdmitResult` is its
documented dual and folds harm — plus an evaluation discipline, report F1 + ASR +
benign-FPR + an adaptive-attacker result, for fak's best-effort detector rung), with the
three-LLM-analyst + Judge MECHANISM explicitly NOT adopted (detector-is-not-the-floor; the
paper's own adaptive attacker is the evidence that a learned verifier is evadable where
fak's default-deny capability floor is not), and the Harm-Analyst's content-harm-in-text
coverage named as a real best-effort residual fak hosts above its floor but does not make
load-bearing** (this note). No code change in this increment: `tools/idea_scout.py`
surfaced and scored the candidate correctly (topic `prompt-injection-defense`, score 53 — a
real, on-topic, high-relevance hit), and the right small artifact for a research/security
triage is the recorded verdict + the component-by-component mapping, not a new detector on
the kernel's load-bearing path.

**Next step (the smallest honest follow-on, if pursued):** a one-line cross-link from the
best-effort-detector rung's doc naming this framework as the **content-harm detector
reference** — two-sided (intent + harm), verify-before-deliver, evaluated with an
adaptive-attacker result — so that when fak documents a classifier *above* its floor, it
cites the bar this paper sets (F1/ASR/benign-FPR + adaptive attack) rather than re-deriving
it. Filed as its own scoped edit, not built in this triage increment.
