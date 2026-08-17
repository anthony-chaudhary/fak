# Benefit–harm standard for defaults

**Status:** shipped doctrine; automated default-admission checker is not yet shipped  
**Scope:** any behavior FAK enables without an operator deliberately selecting it, plus any change that broadens such behavior  
**Related standards:** [net-true value](net-true-value.md), [observer effect](observer-effect.md), [support-maturity honesty fence](support-maturity-honesty-fence.md)

## Rule

**A default is an intervention, not free convenience.** Enable it only when its expected benefit over the real next-best behavior outweighs its total expected harm for the population that receives it. Use the least invasive effective setting, expose important residual risks, monitor effects after release, and define a stop or rollback rule before broadening exposure.

“First, do no harm” is a useful instinct, but an absolute no-harm test would reject every useful intervention. Medicine instead contributes a stronger operating model: compare benefits and harms, account for uncertainty and patient circumstances, reduce avoidable harm, watch real outcomes after introduction, and change course when the balance moves. For FAK, the “patient” is the operator and workload; the relevant outcomes include task correctness, authority, privacy, latency, cost, attention, recoverability, and trust.

This standard complements net-true value. Net-true asks whether a claimed gain survives all costs against the right baseline. Benefit–harm asks the prior product question: **should users be exposed by default, at what dose, with which contraindications, monitoring, and withdrawal criteria?**

## Admission record

Before adding or widening a default, record all nine fields. A prose design note, issue contract, or test table is sufficient if each answer is explicit and witnessed.

1. **Indication** — Which concrete user problem and workload population is this intended to help?
2. **Comparator and non-intervention** — What happens with the current default, including doing nothing? Compare against the strongest practical alternative, not a strawman.
3. **Benefit** — Which user outcome improves, by how much, for how many workloads, and how soon? Separate measured outcome from mechanism proxy.
4. **Harms and interactions** — Include correctness loss, hidden work, stale state, authority expansion, privacy leakage, latency/cost tails, lock-in, operator confusion, and interactions with other stacking defaults. Estimate severity, probability, duration, and affected population.
5. **Uncertainty** — Which populations and tail cases are missing from the evidence? Uncertainty lowers admissible exposure; popularity does not convert missing evidence into safety.
6. **Contraindications** — Name conditions where the behavior must remain off or fail closed. Do not make every user discover these through an incident.
7. **Dose and safeguards** — Choose the narrowest scope, frequency, authority, and resource budget that produces the benefit. Prefer preview, bounded trials, read-only operation, idempotence, isolation, and an observable fallback.
8. **Consent and control** — State whether the operator can understand the material tradeoff, inspect activation, override it, and return to the prior behavior. Irreversible or authority-expanding behavior requires deliberate opt-in even when average benefit is positive.
9. **Surveillance and stop rule** — Name the benefit and harm signals collected after release, the review horizon, the threshold that disables or narrows the default, and the tested rollback path. Aggregate success must not hide severe tail harm.

## Verdict

Assign one exposure class. “Useful” alone is never a DEFAULT verdict.

| Class | Use when | Required posture |
|---|---|---|
| **DEFAULT** | Representative evidence shows a favorable balance; serious harms are prevented or bounded; rollback and surveillance exist. | Enable at the minimum effective dose. Publish activation and material residual risk. |
| **CONDITIONAL DEFAULT** | Benefit is favorable only when a machine-checkable indication is present. | Detect the indication, refuse on contraindication or uncertainty, and fall back safely. |
| **OPT-IN** | Value is workload-specific, evidence is immature, the action expands authority, or harms may be material/irreversible. | Require deliberate selection; provide preview or bounded trial where possible. |
| **EXCLUDE** | The same outcome has a less harmful route, serious harm cannot be bounded, benefit is not net-true, or there is no credible surveillance/withdrawal path. | Do not ship as an offered behavior until the balance changes. |

A default graduates by evidence: **opt-in trial → conditional default → broad default**. It can move backward at any time. Rollback is a normal lifecycle transition, not an admission of failure.

## Applying the standard to FAK's improvement families

These are default postures, not permanent verdicts. A concrete feature can move only with its own admission record.

| Improvement family | Expected benefit | Material side effects | Starting posture |
|---|---|---|---|
| Structural policy and capability floor | Blocks unauthorized tool calls before model persuasion can matter. | False denials, workflow interruption, and a misleading sense that policy covers semantics it does not inspect. | **DEFAULT** for the minimum deny floor; capability expansion stays explicit. Keep an allow witness and explain the denial. |
| Exact tool-result or prefix reuse | Avoids repeated latency, tokens, and external work. | Stale answers, cross-session or cross-tenant disclosure, and replaying a result after hidden state changed. | **CONDITIONAL DEFAULT** only with scoped identity, deterministic/declared-safe inputs, freshness rules, and miss fallback; otherwise opt-in or exclude. |
| Context shedding, elision, and compression | Extends useful sessions and lowers provider input cost. | Removes evidence, instructions, provenance, or an anomaly the model still needs. Harms can be delayed and hard to attribute. | **CONDITIONAL DEFAULT** at a conservative budget with protected invariants, restore path, semantic-quality surveillance, and stop thresholds. Never infer safety from token reduction alone. |
| Model routing or substitution | Lowers cost/latency and can reserve stronger models for harder turns. | Silent capability loss, changed safety behavior, nondeterministic quality tails, and provider/data-boundary changes. | **CONDITIONAL DEFAULT** only inside a witnessed non-inferiority envelope with escalation fallback. Provider or data-boundary changes require explicit operator control. |
| Speculative tool execution and prefetch | Hides tool latency. | Executes unwanted work, duplicates mutation, spends budget, leaks intent, or races the chosen branch. | **OPT-IN** until side-effect freedom, cancellation, deduplication, budget, and privacy are proven. Mutating or externally visible calls are contraindicated by default. |
| Retry, resume, and autonomous continuation | Recovers progress from transient failure and reduces supervision. | Duplicate effects, runaway spend, stale-intent continuation, and repeated harmful action. | **CONDITIONAL DEFAULT** only for idempotent or deduplicated steps with bounded attempts, freshness checks, and visible stop control; authority expansion is opt-in. |
| Telemetry and post-release learning | Detects regressions and rare harms that pre-release tests miss. | Privacy loss, observer overhead, retention risk, and metrics steering the product away from user outcomes. | **DEFAULT** only for minimum-necessary, scoped, inspectable telemetry with retention limits; sensitive content collection is opt-in. Monitor the monitor's cost and effect. |
| Destructive automation or automatic landing | Removes toil and shortens completion time. | Data loss, unintended publication, supply-chain impact, or an irreversible shared-state change. | **OPT-IN** behind preview, narrow capability, witness, and tested recovery. Exclude actions with no credible rollback unless the operator explicitly accepts that condition. |

## Evidence and review rules

- **Measure both columns.** A success metric without a paired harm metric is not an admission record. Examples: cache hit rate pairs with stale/cross-scope incidents; token reduction pairs with task-quality and restore rates; autonomous completion pairs with duplicate-effect, intervention, and spend-tail rates.
- **Weight severity, not just frequency.** A rare credential disclosure can outweigh many cheap cache hits. Report tail outcomes and affected cohorts, not only averages.
- **Test interactions.** Defaults stack like co-medications. Admit the combination users actually receive; do not sum isolated benchmark wins while ignoring shared latency, context, authority, or failure modes.
- **Prefer reversible exposure.** Start with the smallest cohort and dose that can falsify the expected balance. A kill switch that was never exercised is weaker evidence than a tested rollback.
- **Keep judgment accountable.** Automation may gather the record and enforce hard contraindications, but a broad-default decision with material residual harm names an owner and review date.

## Worked admission: speculative mutating tool calls

- **Indication:** reduce perceived latency when the next tool call is highly predictable.
- **Comparator:** wait for the model's selected call; no external action occurs on an unchosen branch.
- **Benefit:** latency saved only when prediction is correct and the call lies on the critical path.
- **Harms:** an incorrect prediction can send a message, mutate data, consume a scarce quota, or reveal intent; cancellation cannot undo many effects.
- **Uncertainty:** prediction precision on representative long-tail traffic and tool-specific reversibility are not established by a read-only microbenchmark.
- **Contraindication:** any call that is mutating, externally visible, non-idempotent, privacy-sensitive, or not safely deduplicated.
- **Dose/safeguards:** permit only bounded, cancellable, side-effect-free prefetch with a separate speculative budget and no cache publication before selection.
- **Control:** explicit activation and an inspectable trace.
- **Surveillance/stop:** compare critical-path latency saved against wrong-branch execution, wasted spend, and cancellation failures; disable on any boundary violation.
- **Verdict:** **OPT-IN** for proven read-only tools; **EXCLUDE** mutating speculation from defaults.

## Medical-industry sources borrowed, and limits

Accessed 2026-08-17:

- [WHO, *Patient safety*](https://www.who.int/news-room/fact-sheets/detail/patient-safety) — frames safety as preventing and reducing avoidable harm, emphasizes system design, incident learning, and patient engagement rather than relying on individual vigilance.
- [FDA, *Factors to Consider When Making Benefit-Risk Determinations in Medical Device Premarket Approval and De Novo Classifications*](https://www.fda.gov/media/112570/download) — separates probable benefit, probable risk, uncertainty, patient perspectives, mitigations, and postmarket controls in structured decisions.
- [NICE NG197, *Shared decision making*](https://www.nice.org.uk/guidance/ng197) — compares options, including doing nothing, and incorporates what matters to the person receiving the intervention.
- [EMA, *Pharmacovigilance: Overview*](https://www.ema.europa.eu/en/human-regulatory-overview/pharmacovigilance-overview) — treats authorization as the start of ongoing detection, assessment, understanding, prevention, and benefit–risk review.

The analogy has limits. FAK is not a medical device, users are not patients, these sources do not confer clinical or regulatory compliance, and software harms are not commensurable with bodily injury. We borrow the decision structure—indication, comparator, benefit, adverse effects, uncertainty, contraindication, dose, consent, surveillance, and withdrawal—not medical authority or terminology as decoration.
