<!-- fak-gateway-key: ep-fanout-roster-admission-v1 -->
<!-- github-issue: 12836 -->

# fix(gateway): admit roster-bound chat requests before EP follower fanout

Tracked by [#12836](https://github.com/anthony-chaudhary/fak/issues/12836).

## Current state

EP admission is part of #5633's original acceptance scope. The implementation
now retains the bounded original request body and defers boot follower release
until roster admission. Explicitly bound requests use only their selected
account; denied bindings release no follower. Unbound requests retain the
original follower body and constant route. The Responses continuation restriction
applies only when the request actually uses EP followers. Independent real TCP
regressions and the full strict gateway suite passed on 2026-09-12. This closes
the #12836 admission finding within #5633's original acceptance scope.

## Original source finding

Read-only source review at public commit `a649f9fd7be257013d167bb5cb4b9104dd02c775`
shows the expert-parallel request bridge starts follower requests before the
normal handler decodes its request and chooses a planner:

| Handler | Existing early fanout call |
| --- | --- |
| OpenAI chat | `internal/gateway/http.go:595` |
| Anthropic Messages | `internal/gateway/messages.go:181` |
| Legacy completions | `internal/gateway/completions.go:51` |
| Responses | `internal/gateway/responses.go:274` |
| Gemini | `internal/gateway/gemini.go:129` |

`internal/gateway/http_epfanout.go:59` checks the follower header and configured
follower addresses, reads and restores the inbound body, then starts goroutines.
Lines 96 and 106 construct a POST with that body and invoke `epFanoutClient.Do`.
The helper does not consult the account roster or its principal allowlist.

The initial #5633 integration candidate bound chat accounts after the ordinary wire
decoder, before buffered or streaming planner dispatch. Therefore the existing
EP bridge remained earlier than that new account-admission seam. An explicitly
bound request refused by account admission could already have been mirrored when
`FAK_EP_FANOUT_ADDRS` is configured.

**Original evidence boundary:** the early fanout ordering predates #5633; #5633 did not
introduce the bridge or move it earlier. This is an integration finding from
source inspection, not a demonstrated network disclosure. At initial triage no
real TCP test of the EP-plus-roster configuration had run. No GPU behavior,
collective failure, or production incident is claimed.

## Classification

- Portfolio tier: 2, serving-only request admission; also supports tier 1.
- Centrality: Core.
- Priority: P1.
- Work unit: S1 leaf.

## Problem frame

- For: operators combining an authenticated multi-account gateway with the EP bridge.
- Problem: account refusal occurs after a separate path can send the request body.
- Today: the planner account is gated, but follower transmission has an earlier boundary.
- Better because: every outbound copy observes account admission before sending bytes,
  using the existing principal and roster contract instead of another authorization rule.
- Witness: a real loopback follower records zero requests for a roster-denied call.
- P1: advanced; admission covers the whole served request's outbound copies.
- P2: advanced; model serving retains a single account-admission rule.
- P3: preserved; use the existing principal allowlist and fail-closed refusal.
- P4: preserved; make no acceleration or physical-hardware claim from HTTP tests.

## Parent context

#5632, the gateway front-door epic. This finding was recorded separately during
#5633 integration and is being resolved within #5633's original acceptance scope.

## Why this is next

The new per-request account binding makes admission ordering across the EP
bridge directly relevant. Capture its independent witness before treating a
gateway configured with both features as fully qualified.

## Dependencies

Coordinates with: #5633.

Related completed bridge coverage: #5528 and #5523. Those issues added wire
coverage; their scope did not establish roster-account admission before fanout.
No blocking dependency: the follower-order reproducer is an executable disjoint
slice. Coordinate final integration with #5633's request binding contract.

Duplicate search on 2026-09-12 checked open and closed issues for EP fanout
admission, roster fanout, expert-parallel admission, and the exact combination
of `FAK_EP_FANOUT_ADDRS` and roster. No matching leaf was found; #5528 is related
closed coverage work and #5632 is the broad gateway epic.

## Working spine

Authenticated request -> existing roster admission -> permitted follower release
and planner selection -> real HTTP witness with explicit destination counts.

## Core through-line

Confirm and document that a roster-denied request permits no follower copy.
Enforce that contract at the common EP seam using the same account admission as
the chat path. If some bound targets should not enter EP at all, express that
decision explicitly before dispatch while preserving legitimate local EP work.

## Gold-plating boundary

Keep the fix to the request ordering/admission seam and one focused integration
fixture. Do not redesign EP collectives, introduce a routing policy or CLI flag,
change native owned-loop exemptions, or redesign account routing.

## Done condition

- [x] A real loopback test first demonstrates whether the ordering permits an
      outbound follower request for a principal-refused explicit binding.
- [x] A denied request returns a non-success response and neither follower,
      bound account, nor boot upstream receives its request body.
- [x] Exercise the five covered inbound wires in the focused fixture; include
      buffered and streaming requests on OpenAI chat and Messages.
- [x] Allowed local EP calls retain their follower release and draining behavior.
- [x] No-roster and explicit-miss behavior retain their existing contract.
- [x] Follower requests remain nonrecursive and use constant route identities.

## Definition of done

The real TCP regression proves zero unauthorized follower transmissions, allowed
EP requests retain their behavior, and the affected gateway gate passes.

## Acceptance gate

The focused real TCP witness below and the existing EP bridge tests pass with
no-roster/miss compatibility preserved. Retain explicit red/green evidence.

## Witness

Independent regression in `internal/gateway/chat_route_test.go`; passed:

`go test ./internal/gateway -run '^TestChatWireRosterRoutePrecedesEPFanout$' -count=1 -v`

Use `httptest.Server` for a real TCP follower and upstreams, a restricted account
roster, and an authenticated principal excluded from the requested account.
Set `FAK_EP_FANOUT_ADDRS` to the loopback follower. Assert destination counts
after the handler and its follower wait complete; avoid sleep-based absence
claims. Capture red and green receipts. Then run existing EP bridge coverage
and the gateway package through the supported WSL/CI test environment.

## Likely files and lane

- `internal/gateway/chat_route.go` for bounded deferred follower release
- The five existing chat wire handlers
- `internal/gateway/chat_route_test.go` for the independent real TCP fixture

```routing
lane: gateway
paths: internal/gateway/chat_route.go, internal/gateway/http.go, internal/gateway/messages.go, internal/gateway/completions.go, internal/gateway/responses.go, internal/gateway/gemini.go, internal/gateway/chat_route_test.go
expected_steps: 3
```

## Expected steps

3: capture the real ordering witness, enforce the shared admission before
follower transmission, and verify permitted/miss compatibility.

## Closure binding

Close only with a DCO-signed `(fak gateway)` commit, the real TCP red/green
receipt, and a green affected gateway gate. Source-order reasoning alone does
not establish that the runtime defect is fixed.

## Work estimate

Estimate: 2 points.

## Overall completion contribution

Contribution: 2/2 points for this bounded admission-order follow-up, not a
percentage of the broader gateway epic.

## Completion standard

development

The real TCP regression establishes the admission contract within the operating
envelope below. This is software validation, not physical EP collective or GPU
qualification.

## Target operating envelope

- roster-denial wire coverage: >= 5 wires
- denied follower transmissions: <= 0 requests

## Witnessed operating envelope

- roster-denial wire coverage: = 5 wires
- denied follower transmissions: = 0 requests
- streaming EP admission coverage: = 2 wires (Chat and Messages)
- handler source order inspected: = 5 handlers

## Evidence status

The pre-fix regression failed behaviorally on all five EP wires, and native
Messages used the boot planner. The final three route regressions passed with
race detection on parent `13025078471a3382b11165d7c7595c12253055ea` (2.841s).
Their independent test source SHA-256 is
`7024565f1f15f45a59761520c201f260d5509d4ed9eea1c5806b1b6f57bd8196`.

The unchanged full strict gateway test binary passed on a quiet leased Linux
amd64 node in 64 seconds, with `-test.count=1 -test.timeout=20m`, no test filter,
normal GOMAXPROCS, and unchanged latency thresholds. Binary SHA-256:
`37f97a6c3a3dc3a06ebb30c79c6a1b12a58e3348d010f29020d8e7331e488626`.
Full receipt SHA-256:
`5ce7492974b263e4ae49ebf79aed2bcac335d3977b0592ac220d5c85da790c18`.
The binary was built with Go 1.26.6; fixture subprocesses used the node's
Go 1.27.0-X:nodwarf5 toolchain. The source archive contained the exact staged
tree and required fixtures. A prior loaded-workstation run failed three latency
gates and remains retained; it is not counted as a passing result.
