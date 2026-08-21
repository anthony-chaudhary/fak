---
title: "Mystery-free adjustment atlas"
description: "A plain-language map from a fak outcome to the rule that decided it, the input that owns it, and a one-variable experiment that proves your understanding."
---

# Mystery-free adjustment atlas

Use this page when you want to understand or tune fak without guessing. The method is
small: **predict → run → locate → adjust one thing → rerun**. The goal is not to hide
complexity. It is to expose the shortest accurate causal chain from an outcome to the
input that controls it.

## The five fields

Write these five lines before changing a process:

```text
Outcome:       What observable result am I explaining?
Decider:       Which named rule, rung, budget, or measurement produced it?
Owned input:   Which configuration or call input does that decider read?
Experiment:    What one reversible change should alter the outcome?
Source:        Where is the authoritative config, code, or guide?
```

The fields prevent common mistakes:

- A **measurement** says what happened; it does not by itself prove why.
- The **decider** is the first causal place to inspect. A nearby subsystem is not enough.
- An **owned input** is read by that decider. Changing an unrelated downstream knob is
  not an experiment.
- A useful **experiment changes one variable** and predicts the direction of the result
  before the rerun.
- The **source** lets you go one level deeper when the explanation is still not simple
  enough to teach back accurately.

Use temporary files, dry runs, or read-only queries first. Do not use a production policy,
live session, or shared cache as a learning sandbox.

## Process map

| Process | Outcome to observe | Decider to locate | Owned input to adjust | Small counterfactual | Go deeper |
|---|---|---|---|---|---|
| Tool policy | `fak preflight … --explain` prints `verdict`, `reason`, and the winning rung. | The `=>` row; earlier `DEFER` rows did not decide. | The named tool/args and the matching manifest entry: `deny`, `allow`, `allow_prefix`, argument rule, or fail-closed posture. | Copy the manifest to a temporary file, change one matching rule, and rerun the identical call. | [`docs/fak/policy-guide.md`](policy-guide.md), [`internal/adjudicator`](../../internal/adjudicator) |
| Model routing | A served call names the selected route/model in runtime metadata and metrics; validate a candidate before serving it. | The first matching rule in `--route-manifest`, after the call is classified into a routing subject. | Subject fields in the call and the ordered manifest rule that matches them. | Copy the manifest, alter one rule or one subject field, then compare the selected model for the same request. | `fak serve help policy`, [`internal/modelroute`](../../internal/modelroute) |
| Context shaping | Runtime token/context counters and debug stats show retained, compacted, elided, or reset context. | The first reached context budget or enabled shaping rule. | `--context-budget-tokens`, `--ctx-view-budget`, compaction settings, and elision settings. | On a disposable session, lower one budget, replay the same turns, and predict which threshold fires earlier. | `fak serve help context`, [`docs/explainers/context.md`](../explainers/context.md) |
| Cache and reuse | `fak info` reports reused tokens, effective cost, and savings; `fak ablate` compares the same trace with reuse on and off. | The paired same-trace delta, not a standalone warm-run number. | Cache attachment/invalidation settings and whether the workload preserves a stable reusable prefix. | Run `fak ablate --trace TRACE.json`; change one cache or prefix condition and repeat the same trace. | `fak serve help cache`, [`docs/explainers/addressable-kv-cache.md`](../explainers/addressable-kv-cache.md) |
| Model serving | `fak doctor serve` reports readiness rows and remediation; startup/debug metrics report the live engine path. | The first unready resource/tier row or the explicitly selected backend/model setting. | Model reference, memory headroom, backend, parallelism, and hardware-specific serve flags. | Change one sizing or backend assumption in a dry readiness check before starting a listener. | `fak serve help model`, [`docs/fak/server-config.md`](server-config.md) |
| Session budgets | `fak session --status SESSION_ID` reports the current budget and control state; `fak resume --explain` reports replay-vs-reset posture. | The exhausted turn/token/context limit or resume governor verdict. | `--turn-limit`, `--token-limit`, `--context-limit`, reset behavior, or the session control signal. | Create a disposable session with one smaller limit, run the same workload, and predict the earlier stop/reset point. | `fak help session`, `fak resume --help` |
| Repository architecture | `fak architecture --leaf LEAF` reports tier, direct dependencies, fan-out, and violations. | The concrete import edge or declared tier rule named by the report. | The importing package boundary, dependency direction, or architecture declaration. | Before editing, predict which dependency row a proposed import would add; rerun the leaf report after the isolated change. | [`docs/architecture.md`](../architecture.md), [`internal/archreport`](../../internal/archreport) |

A command that needs a live server, session, or trace is an **operating-state witness**.
Its absence is “not observed yet,” not a passing result. The policy and architecture rows
are offline and safe to try immediately; the other rows should use a disposable local run
or a captured trace rather than production state.

## Worked example: an unknown tool

**Predict.** An unknown tool is denied because a fail-closed policy has no affirmative
rule for it.

**Run.** From the repository root:

```powershell
fak preflight --policy examples/customer-support-readonly-policy.json --tool mystery_action --args "{}" --explain
```

**Locate.** The output says `DENY / DEFAULT_DENY`; the winning `=>` row is
`adjudicator.Adjudicator`. The other rungs say `DEFER`, so they did not allow or deny the
call.

**Adjust one thing.** Add only `mystery_action` to `allow` in a temporary copy:

```powershell
$p = Get-Content examples/customer-support-readonly-policy.json | ConvertFrom-Json
$p.allow += "mystery_action"
$tmp = Join-Path $env:TEMP "fak-learning-policy.json"
$p | ConvertTo-Json -Depth 10 | Set-Content $tmp
fak preflight --policy $tmp --tool mystery_action --args "{}" --explain
```

**Rerun and teach back.** The identical call now returns `ALLOW` because the policy rung
found one affirmative rule. Nothing about the model, routing, cache, or later execution
changed. If that sentence does not account for the observed trace, keep following the
winning source; do not add another knob.

## Teach-back check

You understand an adjustment when another person can answer all five fields and reproduce
the predicted change. “This flag seemed to help” is not enough. A mystery-free claim has:

1. the before outcome;
2. the named decider;
3. the one owned input changed;
4. the after outcome under the same workload; and
5. the authoritative source or captured trace.

Return to the [`LEARNING-PATH.md`](../../LEARNING-PATH.md) for prerequisite-ordered depth.
