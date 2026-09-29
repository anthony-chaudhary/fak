# The public model router's §2.1 conformance note

This note records, in the public `fak` repository, how the model-selection plane
here satisfies the balance/failover separation the private router states as its
governing contract. It is a **cross-repo citation, never an import**: the public
tree depends on no symbol from `fak-private`, and `fak-private` must not import
`fak/internal/*`.

## The rule

The private authority map `docs/ROUTER-FAILOVER-AUTHORITY.md` §2.1 states:

> **A balancer never produces a severity verdict, and never consumes one.** …
> It is not a function of any `Failure`, any status code, or any `BackendHealth`
> ejection state. If a proposal reads a health verdict to decide *order*, it has
> built a second severity plane and is refused.

The private balance plane (`platform/routing`, `BalancePolicy.Select`) obeys this
and has witnesses for it (LB-07). This note is the public side of the same
contract.

## Where the public plane stands

| Surface | Before | Now |
| --- | --- | --- |
| `internal/modelroute` `PlanIssueAudit` selection | excluded candidates on `AuditSkipProviderUnhealthy` / `AuditSkipCooldownActive`; ordered survivors by `auditHealthRank` | records the health and cooldown verdicts on the receipt but never lets them remove a candidate or decide order; the only selection-stage exclusion is a **capacity SATURATED pre-flight admission** fact. Order comes from the capacity **measurement** (`auditCapacityRank`) plus declared preference/priority/cost. |
| `internal/gateway` `FleetMembership.Pick` / `ReplicaDispatch` placement | mutable round-robin cursors (`m.rr`, `r.next`) over a health-admitted set | a pure keyed rendezvous (`hrwScore(key, memberID)`) over the declared/admissible set. No counter, no clock; two identical `(key, set)` inputs are byte-identical. |

The load-bearing correction is the same one the private plane makes: a candidate
that is merely slow or briefly unhealthy **stays selectable**. Failover — the
retry/escalation path — still avoids a persistently unhealthy provider, because
the health loop evicts it from the admissible set; the *balancer* simply stops
re-deriving that verdict.

## The keyed pick replaces the counter

The private argument against a round-robin counter (`docs/notes/ROUTER-LOAD-BALANCE-2026-09-28.md`
§3) is that a counter makes the decision a function of history, which breaks
replay and makes the balancer unexplainable. The public replacement is a keyed
rendezvous over the request's stable identity:

- `internal/gateway` `hrwScore` — FNV-1a 64 + SplitMix64 avalanche over
  `key || ':' || memberID`, construction-identical to the private
  `routing.BalanceScore` / `ComputeHRWScore`. The constant is **duplicated with
  this citation**, not shared across the boundary.
- `internal/modelroute` `auditCapacityRank` — order from the capacity
  measurement; `AuditSkipCapacitySaturated` is the sole selection-stage refusal.

## Witness

- `internal/modelroute/audit_route_v1_test.go`
  - `TestPlanIssueAuditSelectionIgnoresHealthAndCooldownVerdicts`
  - `TestPlanIssueAuditCapacityIsPreflightAdmissionNotReactiveEjection`
  - `TestPlanIssueAuditSkipsSameFamilyAliasesAndKeepsUnhealthyProvidersSelectable`
  - `TestPlanIssueAuditCooldownIsRecordedNotConsumedAndPermutationDeterminism`
- `internal/gateway/fleet_membership_models_compat_test.go`
  - `TestFleetMembershipKeyedSelectionIsReplayable`
  - `TestFleetMembershipKeyedSelectionSpreadsAndPreservesFailover`

Run: `go test ./internal/modelroute/... ./internal/gateway/... -count=1`.
