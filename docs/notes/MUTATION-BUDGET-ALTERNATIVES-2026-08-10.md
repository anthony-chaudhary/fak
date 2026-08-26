# GitHub mutation-budget alternatives — 2026-08-10

## Verdict

**INCOMPLETE.** The local packet executes fak's estimator and a direct-call/no-reserve baseline only. It does not claim Octokit, `gh api`, or Envoy results. Issue [#6136](https://github.com/anthony-chaudhary/fak/issues/6136) remains open for independently captured real-client witnesses.

## Same-workload contract

Every arm receives twelve remaining GitHub API calls, a five-call reserve, the same reset time, and one plan containing eight closes, five comments, and two fetches. The correct result is to hold the 15-call plan before mutation while retaining separate mutation/fetch accounting.

| Arm | Class | Local availability | Honest result |
|---|---|---:|---|
| fak native mutation reserve | native | yes | correct HOLD; 15 total, 13 mutation, 2 fetch calls |
| direct API calls without reserve | tuned no-feature baseline | yes | incorrect: attempts all 15 and consumes reserve |
| GitHub Octokit rate-limit handling | external | no | zero measurements; real client/response required |
| `gh api` rate-limit handling | external | no | zero measurements; real CLI/headers required |
| Envoy global rate limit | external | no | zero measurements; real service/configuration required |

Repository inspection found no equivalent first-class fak integration beyond the native GitHub path, so no integration arm is fabricated. Mocks and package availability are not results.

## Local native witness

```text
go test ./internal/mutationbudget -bench BenchmarkEstimateMutationHour -benchmem -run '^$' -count=5
```

Windows/amd64, AMD Ryzen 9 9950X: 351.4, 361.4, 448.4, 431.7, 410.1 ns/op. Median: **410.1 ns/estimate**, **368 B/op**, **6 allocs/op**. This is in-process policy cost, not GitHub network latency or API billing.

## Reproduce

```text
go test ./internal/mutationbudget -run TestCompareLocalKeepsAPIBudgetAlternativesExplicit
go test ./internal/mutationbudget -bench BenchmarkEstimateMutationHour -benchmem -run '^$' -count=5
go test ./internal/nativebench
```
