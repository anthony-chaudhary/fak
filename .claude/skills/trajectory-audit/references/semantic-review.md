# Semantic trajectory review

Use this reference after `fak trajectory audit` selects the cohort. It is a
content-review protocol, not another transcript parser and not a semantic score.

## Evidence unit

Review one **event window** at a time:

- the assistant plan or statement immediately before the signal;
- the tool/action and typed outcome;
- the next assistant interpretation and action; and
- enough earlier context to identify the active constraint or already-known fact.

Do not infer confusion from keywords alone. “Retry,” “actually,” or “not sure” is
a locator, not a verdict. Tool output containing `error` is likewise not a
failure unless the tool contract says the call failed.

## Semantic checks

| Check | Evidence that proves the defect | Counter-evidence / sensible baseline |
|---|---|---|
| Contradictory plan | The model takes an action incompatible with its still-active stated plan without acknowledging a new fact. | It names the new evidence and revises the plan. |
| Answered-question repetition | The model asks or investigates the same resolved question again without identifying why prior evidence became invalid. | Prior evidence expired, conflicted, or did not cover the needed scope. |
| Stale-state assertion | The model claims state that a preceding authoritative tool result directly contradicts. | It distrusts an ambiguous result and performs a narrower readback. |
| Unchanged failed action | Same effective tool/action and failure recur without changed arguments, environment, hypothesis, or expected pending-state transition. | Bounded polling of pending asynchronous work, or a changed condition before retry. |
| Constraint/state loss | The model violates or forgets an explicit task scope, file boundary, invariant, or completion requirement it had previously recognized. | It preserves the constraint or explicitly obtains authorization to change it. |
| Unresolved uncertainty | The model states uncertainty and then makes a consequential claim/action without resolving or bounding it. | It gathers evidence, marks the claim unproven, or chooses a reversible action. |
| Self-correction | The model acknowledges a wrong assumption only after avoidable work, mutation, or repeated failure. | Fast correction before side effects is effective recovery, not debt. |

A model does not need to say “I am confused” for confusion to be proven. Conversely,
an explicit “my mistake” is not automatically a defect if correction was prompt,
contained, and cheaper than the next-best alternative.

## Depth tiers

### Depth 1 — default spot check

Deduplicate sessions, then inspect:

- top 3 by repeated failures;
- up to 3 with identical non-wait failures;
- top 3 by mutation churn;
- 1 low-signal control from the same source/model/time cohort.

One session can satisfy several strata. Record selected/eligible counts for each.
Inspect adjacent event windows only. This is the minimum for confusion or model-
regression questions.

### Depth 2 — expanded family review

Trigger when:

- Depth 1 proves any confusion;
- at least 2 events in one signal family remain insufficient evidence;
- a control exhibits the same alleged defect; or
- current and baseline exposure differs by more than 2x.

Inspect all sessions in the implicated signal family, capped at 20, plus a second
control for each represented source/model cohort. Compare rates using an explicit
exposure denominator (sessions or tool calls); raw counts from unequal cohorts are
not comparable quality evidence.

### Depth 3 — bounded cohort review

Use only for release/blocking claims or when Depth 2 proves the same pattern in at
least 3 sessions. Inspect the full topical cohort admitted by `--since` and
`--user-contains`, obtain independent review of the semantic classifications, and
publish agreement/disagreement counts. If the cohort is too large, narrow it with
a declared user-prompt literal or time window rather than silently sampling.

## Event classification

Choose exactly one semantic class:

- **proven semantic confusion** — authoritative adjacent content satisfies a
  semantic defect check and the sensible baseline action was available;
- **effective recovery** — the model changes its hypothesis/condition, gathers
  evidence, or safely bounds uncertainty before proceeding;
- **task-inherent repetition** — repetition is necessary relative to the next-best
  action, such as bounded pending-work polling;
- **analyzer false positive** — the aggregate signal does not represent the
  alleged event (for example, status output lexically contains `error`);
- **insufficient evidence** — the event window cannot establish cause.

Then separately assign the remediation owner from the parent skill:
`skill-solvable`, `tool-solvable`, `task-inherent / efficient reuse`, or
`insufficient evidence`.

## Privacy-safe report card

Use one row per sampled event or homogeneous event group:

```text
session: <opaque session id>
source/model cohort: <source> / <model>
signal: <repeated_failure|non_wait_failure|mutation_churn|control>
aggregate count: <n>
tool + normalized class: <tool> / <class or none>
mutated target: <repository-relative target or none>
semantic class: <one class above>
baseline action: <short privacy-safe paraphrase>
evidence: <short paraphrase of before/outcome/after; no raw content>
remediation owner: <owner class>
```

Finish with:

- depth reached and why;
- sessions selected / eligible by stratum;
- proven, recovery, inherent, false-positive, and insufficient counts;
- controls reviewed and whether they reproduced the pattern;
- baseline exposure denominator and comparability;
- residual uncertainty and the exact next escalation trigger.

Durable reports and issues must not contain raw prompts, reasoning, tool arguments,
tool outputs, credentials, or machine-private absolute paths. Short direct quotes
are allowed only in an operator-only response when needed to answer an explicit
content question; prefer paraphrases and never promote those quotes to committed
artifacts.
