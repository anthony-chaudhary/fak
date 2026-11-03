# Wiring audit — internal/beam (debtwave2-internal-beam)

Read-only wiring audit for the `beam` debt lane (ticket
`docs/tickets/debtwave2-internal-beam/TICKET-debtwave2-internal-beam.md`,
baseline census 2026-11-02: maturity 5/8, rung tested, total_debt 6.78
(principal 6.0 + carrying 0.78), comment ratio 36.3%, health issues
excess_comments, disconnected_wiring, unproven_runtime). Audited 2026-11-03 at
HEAD `5adc05a4b5`; every line number below was re-checked against that HEAD.

## Finding: PRESENT-wired (integrated=true, baseline flag stale)

The baseline `disconnected_wiring` flag was stale: a production call path invokes this
package end-to-end. Read-only import scan from the repo root:

| Question | Answer | Re-check command (repo root) |
|---|---|---|
| Who imports `internal/beam`? | Exactly two production edges: `internal/agent/inkernel_beam.go:10`, `internal/gateway/native_beam.go:9` | `rg -n "github.com/anthony-chaudhary/fak/internal/beam" --glob '*.go' -g '!*_test.go'` |
| Is there a cmd/ entry point to it? | Yes. Critical-path read: `cmd/fak up -> agent -> beam`, distance 2, `on_critical_path=true` | `go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane beam --json` → `critical_path` |
| Does the reader agree? | `evidence.integrated=true`, `related.dependents=[agent,gateway]` | same witness command → `lanes[beam]` |

Live call chain (per-request opt-in; omitting the ext field preserves the old sampler):

1. Ext field: `internal/gateway/wire.go:465` (`BeamSearch *agent.NativeBeamSearchParams json:"beam_search,omitempty"`).
2. Ingress validation: `internal/gateway/http.go:1044` calls `validateChatBeamSearch`, defined at `internal/gateway/native_beam.go:19`.
3. Option application: `internal/gateway/http.go:962` passes `agent.WithNativeBeamSearch(chatBeamParams(req))` (option defined at `internal/agent/inkernel_beam.go:50`).
4. Planner dispatch: `internal/agent/inkernel_planner.go:2260` routes to `completeNativeBeamSearch` when `sp.NativeBeamSearch != nil` (function at `internal/agent/inkernel_beam.go:153`).
5. Search invocation: `internal/agent/inkernel_beam.go:220` calls `beam.Search(ctx, prepared.ids, cfg, p.nativeBeamNextLogits)`.

Test coverage at all three layers:

- `internal/beam/search_test.go`: TestSearchSurvivorsBeatGreedyAndOwnInputs, TestSearchEOSLengthPenaltyAndCompletedLifecycle, TestSearchStableTiesAndFiniteLogSoftmax, TestSearchRefusesBeforeForwardAndNeverReturnsPartial, TestSearchCancellationClosesPublication; `internal/beam/indirection_test.go`: TestBeamZeroAllocKV plus slot/fork/reorder/batch-driver tests.
- `internal/agent/inkernel_beam_test.go`: TestNativeBeamOrdinaryCompleteOwnsReplayAndRanks, TestNativeBeamDefaultNilAndCapturedOptions, TestNativeBeamUnsupportedAndCanceledBeforePublication, TestNativeBeamHTTPProviderRefusesAtSingleOptionFold, TestNativeBeamWorkspaceReservationIsBoundedAndIdempotent.
- `internal/gateway/native_beam_test.go`: TestNativeBeamActualChatCarrierAndRankedResponse, TestNativeBeamUnsupportedIngressNeverCallsPlanner, TestNativeBeamRouteAndMissingResultRefuse, TestNativeBeamLegacyRefusalAndOmittedDefault, TestNativeBeamWireCannotResurrectRedactedWinnerOrAlternatives, TestNativeBeamWireCannotResurrectBodyRefusal, TestNativeBeamPublicProjectionOwnsOnlyBoundedClosedFields.

## Why no new wiring was landed

The wiring already exists end-to-end (ext field, ingress validation, option fold, planner
dispatch, `beam.Search`), so forcing additional wiring would be hollow: a duplicate edge or
presence-only registration adds no call path and only churn. The ticket boundary also keeps
edits inside `internal/beam/`, which forbids touching the agent/gateway seams anyway. This
ticket therefore lands the audit evidence only.

## Reader impact

Numbers are frozen at target after commit `92bef49e81` (on origin/main since 2026-11-03
00:22 PST, subject "refactor(beam): retire debtwave2 lane debt - prune formulaic comment
bloat, keep invariant forms (fak internal/beam)") pruned the real driver: formulaic
Contract:/INVARIANT: header noise, keeping every non-obvious why. Current reader output
(re-run the witness below to re-confirm): maturity 8/8 (rung integrated, gap 0), total_debt
0, comment_ratio 0.232, excess_comments=false, integrated=true, dependents=[agent,gateway],
benchmarked=true, test_status=passing. Census inertness of this document: the debt-lanes
census skips non-Go files (`internal/debtlane/scan.go` L1061:
`if !strings.HasSuffix(d.Name(), ".go") { return nil }`), so committing this `.md` moves no
scored evidence field.

## Open follow-on

`unproven_runtime` (health `proof_status=unproven`) remains open: no hardware/runtime
execution witness yet. It is an open follow-on, not broken, and scoring-neutral while
maturity is capped at target 8. Next checkable step: a runtime/HIL witness campaign, out of
scope for this ticket.

## Next checkable step

Re-run the witness and expect maturity 8 / total_debt 0:

    go run ./cmd/fak debt-lanes --workspace . --target-repo fak --lane beam --json

(strip any leading non-JSON wrapper lines from stdout before parsing; the JSON starts at the
first `{`.) Keep `go test ./internal/beam/... -count=1` green; reader next_action: "beam is
at target maturity (8.0/8.0); keep tests and benchmarks green".
