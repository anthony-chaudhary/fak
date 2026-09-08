<!-- fak-release-key: exact-tested-sha-ci-admission -->
# fix(release): admit CI only for the exact tested SHA, trusted event, and required job set

```routing
lane: release
paths: ["tools/release_context.py", "tools/release_context_test.py", "tools/release_decide.py", "tools/release_decide_test.py", "cmd/fak/release_ship.go", "cmd/fak/release_ship_test.go", "cmd/fak/release_status.go", "cmd/fak/release_status_ci_test.go"]
expected_steps: 8
priority: P0
class: infra
```

## Parent context

Follow-up to closed #1374 and consumer companion to the CI producer-provenance ticket. Parent quality program: #3831.

## Current state

Release context treats the first decisive successful workflow run as green without requiring exact HEAD; its staleness logic applies only to failures. Release status and ship paths query one `ci.yml` run by commit and do not request or validate event type or required jobs. Combined with manual routed CI and pinned-checkout workflows, a successful diagnostic or old-checkout run can be admitted as release evidence for code it did not fully test.

Closed issue #1374 introduced the fast decisive release signal for latency. It did not define an authenticated relationship among GitHub `headSha`, the actual checkout, event class, and required job set. This issue owns consumer-side fail-closed admission; the workflow producer contract is separate.

## Why this is next

Release status, decision, and ship currently consume ambiguous successful runs. Once producer facts are explicit, this is the smallest consumer change that prevents those facts from being promoted beyond what they prove.

## Classification

- Centrality: Core release integrity.
- P1 context: advanced — release decisions retain exact test provenance rather than a workflow-name approximation.
- P2 net value: advanced — prevents a false green cut without slowing trusted exact-SHA fast CI.
- P3 adaptation: preserved — reuse the existing release context, decision, status, and ship seams.
- P4 operations: advanced — rejected CI reports a stable reason: wrong event, wrong SHA, missing jobs, stale result, or untrusted intent.

## Core through-line

Candidate release SHA -> read completed CI facts including event, represented SHA, tested SHA, intent, and jobs -> validate a closed trusted-run policy -> emit green admission or a typed refusal consumed identically by decide, status, and ship.

## Working spine

GitHub run JSON -> one pure trusted-CI admission fold -> release context signal -> decide/status/ship consumers -> release proceed or typed hold.

## Gold-plating boundary

- No release cadence, versioning, tagging, artifact publishing, or notification redesign.
- No generic GitHub check-policy framework.
- Do not accept a green ancestor unless an explicit, bounded policy proves every intervening commit covered; exact HEAD is the default.
- Do not special-case current known workflow IDs without testing event and job-set negatives.

## Done condition

- [ ] Manual routed runs and mismatched pinned checkouts are rejected as release evidence.
- [ ] Green admission requires exact tested SHA, trusted event/intent, and the declared required job set.
- [ ] Missing fields and stale successful runs fail closed with stable reasons.
- [ ] `release decide`, `release status`, and `release ship` share the same admission semantics.
- [ ] Tests cover old-checkout/current-head mismatch, narrow manual success, missing job, pending exact-HEAD run, stale green, red, and valid exact-HEAD green.

## Definition of done

Every release entry point uses one fail-closed CI admission contract and cannot accept success unless the tested revision, event/intent, and required jobs exactly satisfy that contract.

## Verifiable Witness

```text
python tools/release_context_test.py
python tools/release_decide_test.py
go test ./cmd/fak -run 'TestRelease(StatusCI|Ship).*CI' -count=1
```

Fixtures representing a successful `workflow_dispatch` routed run, a successful run whose tested SHA differs from `headSha`, and a successful run missing required jobs must all refuse release admission. A trusted exact-SHA run with the full declared job set must pass.

## Witness

The Python and Go fixture suites independently read the same CI facts and agree on rejection or admission for every exact-SHA, event, freshness, and required-job case.

## Acceptance gate

All three focused suites pass and their negative fixtures produce stable non-green admission reasons without a force flag.

## Closure binding

The resolving commit cites this issue and carries `(fak release)`.

## Likely files

- `tools/release_context.py`
- `tools/release_context_test.py`
- `tools/release_decide.py`
- `tools/release_decide_test.py`
- `cmd/fak/release_ship.go`
- `cmd/fak/release_ship_test.go`
- `cmd/fak/release_status.go`
- `cmd/fak/release_status_ci_test.go`

## Lane

`release`

## Expected steps

8
