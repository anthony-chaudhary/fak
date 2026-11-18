<!-- fak-cadence-key: role-context-window-profiles -->
# feat(executionroute): a role's context window is a predefined TYPE, not an ad-hoc per-call budget

```routing
lane: executionroute
paths: ["internal/executionroute/rolecontext.go", "internal/executionroute/rolecontext_test.go", "internal/executionroute/roles.go", "docs/tickets/20261117-role-context-window-profiles/TICKET-01-role-context-window-profiles.md"]
expected_steps: 4
```

**Origin (operator, verbatim, 11/17/2026):**

> "I think thinking about more like predefined types of context Windows like the
> way you have the implementer and the judge and stuff like that are actually
> kind of a good idea maybe. It's sort of like the idea of a persistent identity
> but it's along a different axis Maybe"

## Core through-line

Two spines already existed and had never met.

- **The delegation axis** — `internal/executionroute/roles.go` owns WHO does the
  work: a closed `Role` set (`scout` / `worker` / `judge` / `primary`), each with
  its own `modelroute.Subject` and its own independent `Budget`.
- **The context-window axis** — `internal/ctxplan/layout.go` owns HOW MUCH CONTEXT
  a turn materializes: the four-area `Layout` (`base` / `current` / `recent` /
  `deep`), each an `AreaPolicy{MaxSpans, MaxTokens, Precision}` with precisions
  `exact` / `planned` / `pointer`.

Nothing bound them. `executionroute` carried **no** context-window field and did
not import `ctxplan`; `ctxplan.Layout` was not role-keyed. So a role's context
window was whatever the caller passed per call — an ad-hoc budget, not a
predefined type. That is the gap the operator named.

## What landed

`internal/executionroute/rolecontext.go` is the binding, and only the binding:

- `RoleContextProfile` — a named window TYPE: `Name`, `Role`, `Layout`
  (`ctxplan.Layout`), `ResidentTokens`, `PrecisionPosture`, `Rationale`.
- `DefaultRoleContextProfiles()` — a fixed, compile-time table with one row per
  member of the closed `Role` set, in declared role order.
- `ContextProfileForRole(Role)` — resolution, with `ok=false` for an unprofiled
  role.
- `ValidateRoleContextProfiles([]RoleContextProfile)` — the closure validator.
- `resolveContextProfile(RolePlan)` — the fail-closed resolver used by routing.

`RolePlan` gains an optional `ContextProfile string` override; `RoleDecision`
gains the resolved `ContextProfile RoleContextProfile`, so the window a role ran
under is auditable from the envelope.

## The four predefined types

| Role | Profile name | Posture | Deep probe | Base precision | Resident target |
|---|---|---|---|---|---|
| `scout` | `scout-probe-narrow` | `narrow` | 8 | exact | 2000 |
| `worker` | `worker-implementer-bounded` | `bounded` | 16 | exact | 8000 |
| `judge` | `judge-adjudicator-wide` | `wide` | 64 | exact | 16000 |
| `primary` | `primary-driver-driving` | `driving` | 32 | exact | 16000 |

The differences are mechanical, not cosmetic:

- **scout** is the cheap probe: its `recent` and `deep` areas are
  `PrecisionPointer`, so cold spans cost a recoverable pointer rather than
  resident bytes. It must not pay for history it will not read.
- **worker** is the implementer: exact task and exact current state, a moderate
  *planned* recent tail, and a shallow deep probe — wide enough to act, narrow
  enough that a bounded doer cannot wander.
- **judge** adjudicates: the widest deep probe and an **exact** base, because the
  rubric or contract it scores against must be verbatim — a rubric read as a
  pointer is not a rubric.
- **primary** drives: the broadest recent tail (it holds the conversation
  together) with *planned* deep history, because it delegates depth to the others.

## Fail-closed, not fail-silent

`RouteRoles` refuses an unregistered profile name and refuses a role with no
registered default. It never downgrades to a zero `Layout`: a zero `Layout` plans
against `ctxplan`'s defaults, which would make an unprofiled role *look*
profiled. `TestRouteRolesRejectsUnknownContextProfile` is the guard.

## Witness

```
go test ./internal/executionroute/ -run TestRoleContextProfile -count=1
```

Six tests: registry closure (every closed role has exactly one profile, no
orphan), per-role distinctness (the implementer and the judge do **not** share a
window — the judge's deep probe is strictly wider and its base is exact), live
resolution on the routing path, override honored, unknown profile refused, and
the validator naming each defect class.

## Non-goals

- **Repo heaviness / checkout speed** is a *different* axis, owned by the
  `checkout-100x-20261113` campaign (`blob:none` partial-clone lane object source
  is its P0 keystone). This ticket does not touch lane provisioning.
- **Serving/deployment posture** (`raw|lean|legacy|auto` over 9 levers) is owned
  by `ctx-plane-flex` / `ctx-plane-profile-custom`. Different axis.
- **Model selection** stays with `RolePlan.Subject` / `modelroute`. This ticket
  adds only the context-window dimension.
- No new context planner: `ctxplan.Layout` is reused as-is, unmodified.

## Scaling

Not applicable. The registry is a fixed compile-time table with one entry per
member of a closed set (4 today), no append path, no caller-keyed map, and no
per-request allocation. Resolution is a bounded linear scan over ≤4 entries —
O(1) in every input dimension. The only axis that could grow is the closed `Role`
set, and adding a role is a deliberate source edit gated by `knownRole` plus
`TestRoleContextProfileRegistryIsClosed`.
