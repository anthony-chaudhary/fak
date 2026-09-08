<!-- fak-ci-key: manual-runs-not-release-evidence -->
# fix(ci): prevent routed and pinned workflow runs from impersonating release CI

```routing
lane: ci
paths: [".github/workflows/ci.yml", ".github/workflows/ci-fast.yml"]
expected_steps: 5
priority: P0
class: infra
```

## Parent context

Follow-up to closed #6400 and companion to the release-consumer admission ticket. Parent quality program: #3831.

## Current state

`ci.yml` accepts `workflow_dispatch` with caller-controlled `go_test_args_json`; that event runs only the routed test job while the full build/vet/claims and race jobs are skipped. `ci-fast.yml` separately accepts a caller-controlled `sha` and checks out that revision without proving it equals the GitHub run's represented `headSha`. Both workflows can therefore publish a successful run whose event, job set, or tested checkout is not the full release claim implied by the workflow name and GitHub SHA.

Closed issue #6400 made routed CI executable and requested source-SHA read-back, but it did not isolate routed evidence from release evidence. This issue is the producer-side contract; release consumers are hardened separately.

## Why this is next

The producer ambiguity is already exploitable by ordinary supported workflow inputs. Fixing the emitted identity first gives every release consumer a deterministic fact to validate.

## Classification

- Centrality: Core release integrity.
- P1 context: advanced — every successful CI result names the exact code and test regime it actually executed.
- P2 net value: advanced — narrow diagnostics remain available without being convertible into release credit.
- P3 adaptation: preserved — keep existing push/PR and CI-only routing behavior, adding only provenance separation.
- P4 operations: advanced — operators can distinguish full, fast, and routed runs from GitHub metadata and durable job output.

## Core through-line

Workflow event and requested checkout -> validate exact tested SHA and declared run intent -> execute the corresponding fixed job set -> emit an unambiguous run/check identity that cannot be consumed as a stronger CI class.

## Working spine

`workflow_dispatch`/push event -> checkout and intent validation -> fixed CI class -> exact-SHA run metadata and check name -> downstream release admission.

## Gold-plating boundary

- No CI sharding, cache redesign, runner migration, or test-suite optimization.
- No release decision changes; those belong to the release-consumer issue.
- Do not remove the CI-only route. Give it a distinct non-release identity and exact-SHA receipt.
- Do not infer a tested SHA from the dispatch ref when the checkout differs.

## Done condition

- [ ] A routed `workflow_dispatch` run cannot present the same release-eligible signal as full `ci.yml`.
- [ ] A pinned `ci-fast` dispatch fails closed unless the checked-out SHA equals the SHA recorded for consumer admission.
- [ ] Full push/PR CI retains its existing build, vet, test, claims, and race behavior.
- [ ] Workflow tests or lint pin the event/job/provenance contract.

## Definition of done

Both workflows emit evidence whose CI class and tested revision are machine-checkable, and no supported manual input can make a narrower or older execution indistinguishable from trusted release CI.

## Verifiable Witness

```text
go test ./internal/testroute ./internal/workflowlint -count=1
gh workflow run ci.yml --ref main -f go_test_args_json='["./pkg/abi","-run","TestDoesNotExist"]'
gh run view <run-id> --json event,headSha,status,conclusion,jobs,url
gh workflow run ci-fast.yml --ref main -f sha=<older-known-green-sha> -f intent=release
gh run view <run-id> --json event,headSha,status,conclusion,jobs,url
```

The two manual runs may succeed as diagnostics, but their metadata/check identity must make them ineligible for release admission. A checked-out SHA mismatch must be explicit and non-crediting.

## Witness

The focused workflow-routing tests pass, and GitHub run read-back proves manual diagnostic runs expose a non-release class plus the exact tested SHA.

## Acceptance gate

The focused Go suites pass, the two manual negative probes remain diagnostic-only, and a normal push run preserves the existing full and fast job sets.

## Closure binding

The resolving commit cites this issue and carries `(fak ci)`.

## Likely files

- `.github/workflows/ci.yml`
- `.github/workflows/ci-fast.yml`

## Lane

`ci`

## Expected steps

5
