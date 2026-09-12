<!-- fak-gateway-key: rich-dashboard-first-click-probe-race-v1 -->

# test(gateway): synchronize rich-dashboard first-click progress fixture

Tracked by [#12825](https://github.com/anthony-chaudhary/fak/issues/12825).

## Parent context

Parent: #8708 (dashboard startup, idle, click, and refresh test envelope).
Coordinates with: #8708.
Coordinates with: #12670 (separate corpus fixture fix whose package gate exposed this failure).
This is an additional prerequisite, outside the selected ten campaign issues.
There is no start-blocking dependency on either issue. The coordinator assigns
the write lease before implementation.

## Current state

The full gateway package at base `d49abccee991b2bc6aea9a819e9829b05a9d1823`
with the unrelated #12670 fixture patch failed
`TestRichDashboardDormantUntilFirstClickThenRedirects`: the first request
returned HTTP 303 where the test required HTTP 200 and a progress render.
The dashboard source and test were unchanged from that base. The coordinator's
full-package receipt recorded exit 1 after 197.262s; this is not a standalone
deterministic reproduction, and the three separate latency failures are outside scope.

The schedule is explicit in existing code:

1. The test's probe increments its counter and immediately returns success.
2. `richDashboardManager.ensure` sets `starting`, unlocks, and launches `activate`.
3. Activation may set `ready` before `ensure` calls `snapshot`.
4. The handler correctly redirects a ready snapshot; the test assumes activation
   has not completed and demands a progress page instead.

Closed #9762 fixed ABI registration isolation, a different failure class.
Closed #8690 introduced the lazy activation and readiness redirect behavior.
Searches of open and closed issues by exact test name, file, dashboard race,
flaky dashboard, and the marker found no exact existing tracker for this schedule.

## Problem frame

- Portfolio tier: 1, all-in-one serving validation.
- Centrality: Enabling.
- For: contributors running the gateway package gate.
- Problem: an uncontrolled successful probe makes the progress-render assertion depend on scheduling.
- Today: an immediate healthy redirect can turn an unrelated change's package gate red.
- Better because: an explicit probe barrier establishes the starting state before the captured render is asserted.
- P1: preserved; no prompts, private endpoints, or host paths enter evidence.
- P2: advanced; deterministic validation avoids false-red retries.
- P3: preserved; production activation and readiness decisions stay unchanged.
- P4: advanced; the existing HTTP handler render and redirect are both executed.

## Working spine

Block probe completion -> capture and assert progress -> release probe ->
observe ready -> assert the requested redirect and one activation probe.

## Core through-line

One test owns the scheduling condition its progress assertion requires.

## Scope

Scope class: atomic.

Modify only `TestRichDashboardDormantUntilFirstClickThenRedirects` in
`internal/gateway/rich_dashboard_test.go`. Add a release channel to its existing
probe closure; wait for release or context cancellation. Release only after
the first response's progress assertions succeed, then retain the existing
ready wait, redirect destination, and probe-count assertions. Ensure early
test failure cancels or releases the blocked probe without leaking a goroutine.

## Gold-plating boundary

No production handler or manager change, sleeps, timing tolerance, skips,
either-200-or-303 assertion, latency-gate edits, ABI registration changes,
corpus fixture changes, or new generic helper. Keep #12670 separate.

## Done condition

The focused captured-render test deterministically observes progress while
probe completion is blocked, then readiness and HTTP 303 to the original
destination after release, with exactly one probe and safe failure cleanup.

## Definition of Done

- [ ] The existing captured progress and redirect assertions pass with explicit probe
ordering and cancellation-safe cleanup; the test-only patch is independently reviewed.

## Outcome

The gateway progress-render witness no longer depends on probe scheduling.

## Why now

This unchanged fixture failed the current gateway package gate and prevents
the coordinator from accepting otherwise unrelated prerequisite changes.

## Acceptance gate

The focused Witness command passes all 20 iterations with both existing render
states asserted. Source review confirms an explicit release channel controls
probe completion and no production logic or assertion has been weakened.

## Witness

`go test ./internal/gateway -run '^TestRichDashboardDormantUntilFirstClickThenRedirects$' -count=20 -v`

Run through the supported WSL test entry point in an isolated worktree. Preserve
the original full-package failure excerpt and the focused passing log. Channel
ordering, not elapsed time, must establish the progress state. A full package
gate is scheduled separately by the coordinator; do not launch broad tests on
the saturated host for this leaf.

## Likely files

- `internal/gateway/rich_dashboard_test.go` (only implementation write path).
- `docs/tickets/gateway/TICKET-02-rich-dashboard-first-click-probe-race.md` (tracking record).

## Lane

gateway-tests

## Work unit

leaf

## Work size

S1

## Required model tier

T2

## Expected steps

2

## Work estimate

Estimate: 1 point.

## Overall completion contribution

Contribution: 1/1 point for this prerequisite; does not change the selected-ten denominator.

## Completion standard

development

## Closure binding

Close only after the exact test-only diff and captured witness are independently
reviewed and landed through the coordinator's guarded route. Provenance:
`EXEMPT_TEST_ONLY`.

## Implementation and validation

The test now blocks its existing probe on a release channel until the first
progress-render assertions pass. It then releases the probe and retains the
existing ready, redirect-destination, and single-probe assertions. The probe
also selects its context cancellation; the existing deferred manager close
cancels that context if an assertion exits the test early. Production code
and the separate #12670 fixture are unchanged by this patch.

The focused witness passed all 20 iterations under WSL Go 1.26.6 with
`FAK_FAST=0`, directly testing the app-provisioned detached worktree. The log
contains 20 progress responses (HTTP 200, request `gw-2`) and 20 ready
redirects (HTTP 303, request `gw-3`); package result: 0.346s, exit 0.
The coordinator retains the original full-package RED receipt. The reserved full gateway gate subsequently passed on base `24d99052e14ccf0ce93a005becb3288ff3a24fcb` plus the separate #12670 and #12825 fixture patches in a durable Linux LF managed worktree: 69.177s, exit 0, with `FAK_STRICT_SERVE_LATENCY=1`. Both fixture tests and all three unchanged latency gates passed.
