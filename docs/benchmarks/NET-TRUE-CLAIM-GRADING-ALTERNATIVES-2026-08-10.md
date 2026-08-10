# Net-true claim grading alternatives — 2026-08-10

Status: **INCOMPLETE**. Issue [#6180](https://github.com/anthony-chaudhary/fak/issues/6180) tracks real integration/evaluation-system runs and independent resource/cost witnesses.

## Capability boundary and workload

`internal/claimcheck.Grade`, exposed by `fak claim-check`, returns `net-true`, `strawman`, or `not-yet` after separately checking the real baseline, net accounting, scope, provenance label, reproducible witness, and realized/default-or-gated state. This contract covers grading only; witness-plan generation, worktree fingerprinting, and finding reuse remain separate capability debt.

Every arm grades the committed nine-case fixture: two honest net-true claims (including honestly gated realization), one strawman-baseline claim, and six not-yet claims missing baseline, witness, provenance, net accounting, scope, or realized deployment. Correctness requires every exact verdict and a non-empty failing-question class for every rejection.

## Arms

| Arm | Class | Local status |
|---|---|---:|
| fak native net-true claim grader | native | available |
| accept claim when any witness exists | tuned no-grader baseline | available, incorrect |
| fak + Prometheus | first-class integration | unavailable |
| fak + OpenTelemetry | first-class integration | unavailable |
| OPA/Rego | external | unavailable |
| OpenAI Evals graders | external | unavailable |
| LangSmith evaluators | external | unavailable |
| Braintrust scorers | external | unavailable |
| DeepEval metrics | external | unavailable |

The baseline is a realistic minimal review heuristic—accept a claim if it cites any artifact—but it cannot detect strawman baselines or missing scope/net/provenance/realization. Unavailable products keep `Available=false` and all measurements at zero; local imitations do not witness them.

## Completion evidence

Complete arms report exact verdicts, each wrong-verdict class, reason mismatches, latency/throughput, CPU/RSS, input/network/storage bytes, model/evaluator tokens, setup/operator time, service charges, and total cost. Versions, prompts/policies, raw decisions, and independent read-back must be pinned.

`TestCompareLocalKeepsClaimEvaluationAlternativesExplicit` locks inventory, native corpus correctness, baseline failure, and unavailable zeros. `BenchmarkGradeFixture` grades all nine claims per iteration. Local timing is not a cross-product claim and no system is ranked until #6180 carries real-boundary witnesses.
