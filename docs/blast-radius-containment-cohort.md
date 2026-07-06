# Blast-radius containment — ticket cohort

**Status:** already filed on `anthony-chaudhary/fak` as issues **#2712–#2720** (2026-07-05).
This file is the portable, self-contained spec so another agent/fleet can (a) recreate the
cohort in another repo, or (b) read the whole design without touching the API.

**Shape:** one epic (#2712) + one load-bearing spine (W1 / #2713) + seven fan-out children
(W2–W8 / #2714–#2720).

**Ship order:** W1 first (everything reads its ledger) → W2/W3 (recognize + estimate) →
the load-bearing W4/W5/W6 (hold + elect + release) → W7/W8 (surface + bound).

## The problem in one line
With N concurrent agents on a shared trunk, one agent's discovery of a *shared-root-cause*
failure silently taxes the other N-1: they **rediscover** it (burning a cycle each),
**collide** on the fix (N racing PRs to one file), or **stall globally** even though disjoint
work is shippable. Today's containment is all siloed — `guardrsi/livelock.go` is per-trace,
`attemptbudget` is per-issue, `blockerpost` is a human Slack post, `affectedtests` is
test-selection-only — so none of it recognizes "shared." This cohort makes the fleet discover
the shared failure **once**, hold **only** the affected agents, elect **one** fixer, and
**auto-release** on a *witnessed* fix.

## How to create these tickets
Each section below carries a metadata block (title, labels, milestone, parent, depends-on)
followed by the issue body verbatim. To file one:

    gh api repos/<OWNER>/<REPO>/issues \
      -f title="<title>" \
      -F body=@<body-file>.md \
      -f 'labels[]=<label>' -f 'labels[]=<label>' ... \
      -F milestone=<N>

Notes (fak host specifics):
- Prefer `gh api` over `gh issue create` — the latter spelling can trip the guard.
- Labels and the milestone must already exist in the target repo. Milestone `6` here =
  "10x agentic coding loop with witnessed self-correction" (fak-specific; drop or remap
  `-F milestone` elsewhere).
- An outward-facing `gh api` write may hit the preview-confirm gate: re-propose the
  *byte-identical* call with `"_fak_confirm":"<token>"` added to the tool input. That is a
  pause, not a denial.
- Bodies use `##` section headers (Current state / Working spine / In scope / Out of scope /
  Done condition / Witness / Acceptance gate / Closure binding / Likely files / Lane / …) so
  each leaf passes the fak issue-contract dispatchability check.

## Cohort map
| Ticket | # | Seam | Package / verb |
|--------|---|------|----------------|
| Epic | 2712 | umbrella: recognize → broadcast → scope-hold → auto-release | — |
| W1 spine | 2713 | fleet-wide known-bad ledger + record/match | `internal/knownbad` |
| W2 | 2714 | recognize: cross-trace `FailureHash` correlation | `internal/guardrsi` |
| W3 | 2715 | estimate: blast radius = import graph ∩ live leases | `internal/blastradius` |
| W4 | 2716 | scope-hold only intersecting issues (`BLOCKED_BY_KNOWN_BAD`) | `internal/dispatchtick` |
| W5 | 2717 | elect exactly one fixer via an exclusive lease | `fak knownbad claim` |
| W6 | 2718 | witness-gated auto-release of held agents | `fak knownbad resolve` |
| W7 | 2719 | operator blast card (1 cause → N affected, 1 fixing) | `internal/blockerpost` |
| W8 | 2720 | TTL + revoke so a stale known-bad can't wedge the fleet | `internal/knownbad` |

Load-bearing for epic closure: **W1 + W4 + W5 + W6**.

---

## Epic — #2712

- **title:** epic(dev-ex): blast-radius containment — one agent's new bug shouldn't stall the other N-1 (recognize → broadcast → scope-hold → auto-release)
- **labels:** epic, dev-ex, dispatch, fleet-400iph, operator, priority/P1
- **milestone:** 6
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2712

### Body

## Parent context

Parent: agent working conditions II #2136 (milestone "10x agentic coding loop with witnessed self-correction"), under the safe parallel-agent throughput program (`fleet-400iph`). Sibling of the per-issue attempt budget (`internal/attemptbudget`, #1777/#1778) and the per-session livelock nudge (`internal/guardrsi/livelock.go`).

## Current state

With N concurrent agents on the shared trunk, a failure with a **shared root cause** — a broken shared helper, a red trunk build, a poisoned resource, a flaky-then-hard shared service, a bad migration — is discovered by ONE agent, but its blast radius is the whole fleet. Today the other N-1 agents each, independently and blind to one another:

1. **Rediscover** the same failure — each burns a full cycle plus tokens re-finding what agent-1 already knows.
2. **Collide on the fix** — several may each try to fix the same shared file, producing N racing edits/PRs to one tree (the `hot-tree` failure at fleet scale).
3. **Stall globally** — several park or livelock on it even though disjoint, still-shippable work exists.

The existing containment is all **siloed**, so none of it recognizes "shared":

- `internal/guardrsi/livelock.go` counts identical failures **per trace (one session)** and nudges that one agent. It even computes a `FailureHash` — but never correlates it across traces.
- `internal/attemptbudget` budgets attempts **per issue-id**. A root cause hitting 9 different issues looks like 9 unrelated failing issues, each cooling down on its own window.
- `internal/blockerpost` posts a blocker to Slack for a **human**; it is outbound-only, not auto-triggered by correlation, and not an agent-consumable signal.
- `internal/affectedtests` (`fak affected`) already computes the **dependency blast radius of a change** but is used only for test selection — never joined to live worker leases to answer "who is affected right now."
- `dos arbitrate` / lease-lanes can keep work disjoint but nothing declares a tree "quarantined until the fix is witnessed."

## Why this is next

Blast radius is the dominant tail risk of the 400-issues/hour program: throughput collapses not when one issue is hard, but when one shared bug silently taxes every concurrent worker. We already pay for N agents — we should pay the *discovery* cost once, not N times, and never pay the *collision* or *global-stall* cost at all.

## Working spine

A fleet-wide **known-bad signature** substrate the whole fleet reads and writes:

1. **Recognize** a failure is shared — cross-trace correlation of the `FailureHash` guardrsi already computes, plus a worker-declared path on first discovery.
2. **Estimate** the blast radius — dependency graph ∩ live leases → the set of affected in-flight agents and queued issues.
3. **Broadcast** it as an agent-consumable signal, not just a human Slack post.
4. **Scope-hold** ONLY the agents whose declared tree intersects the blast radius; let disjoint agents keep shipping (progress, not global stall).
5. **Elect one fixer** via a lease on the broken tree; park the rest with a pointer to the fixer, not into a collision.
6. **Auto-release** the held agents when the fix is **witnessed** (dos verify / green `fak affected` on the broken package) — witness-gated, never on a self-report.
7. **Surface** it to the operator as one blast-framed card: "1 root cause -> N affected, 1 fixing, N-1 parked, witness pending."
8. **Expire / falsify** a signature so a stale or misattributed known-bad can never wedge the fleet forever (the opposite failure mode).

## In scope

The coordination substrate plus the eight seams above, filed as the child issues below. The spine (W1) is the minimal working end-to-end; every other child extends one named seam that already exists.

## Children (fan-out)

W1 is the load-bearing spine; W2-W8 each extend one named, already-existing seam.

- **W1 (spine)** #2713 — fleet-wide known-bad signature ledger + `fak knownbad record/match`. `internal/knownbad`.
- **W2 (recognize)** #2714 — promote a cross-trace repeated `FailureHash` to a known-bad candidate. `internal/guardrsi`.
- **W3 (estimate)** #2715 — blast-radius estimator: join the `affectedtests` import graph with live leases. `internal/blastradius`.
- **W4 (scope-hold)** #2716 — hold ONLY the intersecting issues (`BLOCKED_BY_KNOWN_BAD`), let disjoint agents ship. `internal/dispatchtick`.
- **W5 (elect fixer)** #2717 — exactly one fixer per known-bad via an exclusive lease. `fak knownbad claim`.
- **W6 (auto-release)** #2718 — witness-gated release of held agents on a proven fix. `fak knownbad resolve`.
- **W7 (surface)** #2719 — operator blast card: 1 root cause -> N affected, 1 fixing, N-1 parked. `internal/blockerpost`.
- **W8 (bound)** #2720 — TTL + revoke so a stale/misattributed known-bad can't wedge the fleet. `internal/knownbad`.

Load-bearing for epic closure: W1 + W4 + W5 + W6.

## Out of scope

Fixing any specific bug; changing the guard's per-call adjudication; introducing a new transport (reuse the journal, the dos lease, and the blockerpost seams already in the tree).

## Done condition / witness

Epic closes when W1 (spine) plus the scope-hold (W4), single-fixer (W5), and auto-release (W6) children are shipped, and a dogfood run shows: one injected shared failure is recorded once, the disjoint agents keep shipping, exactly one fixer is elected, and the held agents auto-release on the witnessed fix — with a captured ledger and command readout.

## Acceptance gate

`make ci` green; a reproducible dogfood transcript (recorded under `docs/nightrun/` or a focused test) demonstrating the "discover once -> scope-hold -> one fixer -> witnessed release" path end to end.

## Closure binding

This epic closes only via a `Closes #<epic>` commit that lands the spine plus the three load-bearing children (W4/W5/W6) with the dogfood witness. Child issues bind to their own `(fak <leaf>)`-stamped commits.

## Dependencies

- related: W1..W8 (filed as children of this epic)
- related: #2136 (parent program), #1777, #1778 (attemptbudget siblings)

## Likely files

- `internal/guardrsi/livelock.go`
- `internal/attemptbudget/attemptbudget.go`
- `internal/blockerpost/blockerpost.go`
- `internal/affectedtests/`
- `internal/dispatchtick/dispatchtick.go`

## Lane

dispatch

## Work unit

epic

---

## W1 — spine — #2713

- **title:** feat(knownbad): fleet-wide known-bad signature ledger + `fak knownbad record/match` (blast-radius spine, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2713

### Body

## Parent context

Parent: blast-radius containment epic #2712. This is the epic's **load-bearing spine** — the fleet-wide substrate every other child (W2..W8) reads and writes. Nothing else in the epic works until this exists.

## Current state

The fleet has no shared place to record "this failure is known-bad and shared." Every containment surface today is siloed: `internal/guardrsi/livelock.go` is per-trace, `internal/attemptbudget` is per-issue-id, `internal/blockerpost` is a human Slack post. So when agent-1 discovers a shared bug, agents 2..N have no signal to read before they burn a full cycle rediscovering it.

## Why this is next

This is the smallest thing that turns "N agents each rediscover the bug" into "1 agent records it, the rest read it." Every other child (scope-hold, single-fixer, auto-release, operator card) is a consumer of this ledger; it must land first.

## Working spine

A new pure package `internal/knownbad` plus a `fak knownbad` verb:

- **Signature**: a stable, content-free id derived from `(failure_class, normalized tree globs, optional guardrsi FailureHash)` — same shape as `guardrsi.ArgsDigest`/`failureHash` (sha256 over a canonical key). Two agents hitting the same shared cause produce the same signature.
- **Record**: `fak knownbad record --tree <globs> --reason <class> --note "..."` appends one JSONL record `{schema:"fak.known-bad.v1", signature, reason_class, tree_globs, discovered_by, discovered_at_unix, ttl_seconds, status:"open"}` to a fleet-visible ledger (journal-style append, same idiom as the other fak ledgers).
- **Match/query**: `fak knownbad match --tree <globs> [--json]` returns whether the requested tree intersects any *live* (open, unexpired) known-bad signature, with the matching record(s). Exit non-zero (or a JSON `matched:true`) so a worker OR the dispatcher can short-circuit before burning a cycle.

Pure fold core (signature derivation, tree-glob intersection, liveness by `now` supplied as data) in `internal/knownbad`; impure shell (ledger read/write, clock, flags) in `cmd/fak/knownbad.go`.

## In scope

- `internal/knownbad`: signature derivation, the record shape, tree-glob intersection (reuse the same glob semantics `dos arbitrate`/lease trees use), and the pure `Match(records, req, nowUnix) -> matches` fold.
- `cmd/fak/knownbad.go`: `record` and `match` subcommands over a JSONL ledger; `--json`; deterministic, clock-injected.
- Unit tests for signature stability, intersection, and liveness.

## Out of scope

Cross-trace auto-promotion (that is W2), the blast-radius agent set (W3), dispatcher hold wiring (W4), fixer election (W5), auto-release (W6), the operator card (W7), and TTL/GC policy beyond a plain unexpired check (W8). W1 ships the substrate and the two verbs; the consumers fan out.

## Done condition

`fak knownbad record --tree internal/foo/** --reason build` writes a record; a second shell `fak knownbad match --tree internal/foo/bar.go` reports `matched:true` with that record; `fak knownbad match --tree internal/other/**` reports `matched:false`. The pure core has passing tests for signature stability and intersection.

## Witness

Captured terminal transcript of the record-then-match sequence above (both the intersecting and disjoint match), plus `go test ./internal/knownbad/...` green. Record the transcript in the commit body or a focused test.

## Acceptance gate

`make ci` green; `go test ./internal/knownbad/... ./cmd/fak/...` green; the record→match transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` that lands `internal/knownbad` + `cmd/fak/knownbad.go` + tests with the transcript witness.

## Dependencies

- blocks: #2712 (epic spine — the other children depend on this)

## Likely files

- `internal/knownbad/knownbad.go`
- `internal/knownbad/knownbad_test.go`
- `cmd/fak/knownbad.go`
- `cmd/fak/main.go`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The JSONL append-ledger idiom (used by the other fak ledgers) is an acceptable fleet-visible store for v0; a shared path can be resolved the same way `internal/journal` resolves its store.
- Tree-glob intersection can reuse the same semantics the dos lease/arbitrate path already uses, so a known-bad tree and a lease tree compare apples to apples.

## Confusion risks

- Do NOT reuse `guardrsi.LivelockDetector` state (that is per-trace and in-memory); this is a durable, cross-trace ledger. W2 bridges the two.
- `knownbad` is about a *runtime-discovered shared failure*, distinct from the router's static `BLOCKED_BY_HUMAN` per-issue label (`fak dispatch skipped`).

## Coordination

New package + new file; low collision risk. Touches `cmd/fak/main.go` (dispatch table) — a hot file; commit by narrow pathspec.

## Trigger

Filed once at epic-spine creation.

## Batch policy

One issue per spine; deduped by the `fak knownbad` verb key.

---

## W2 — recognize — #2714

- **title:** feat(guardrsi): promote a cross-trace repeated FailureHash to a fleet known-bad candidate (blast-radius W2, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2714

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 (this feeds it). Extends `internal/guardrsi/livelock.go`.

## Current state

`internal/guardrsi/livelock.go` already computes a content-free `FailureHash` for a repeated failing tool call — but the `LivelockDetector` is keyed **per trace** (`byTrace map[string]livelockRun`), so it only ever notices one agent looping on its own. When 9 agents each hit the *same* shared failure once, no counter crosses a threshold: each trace sees a single failure, and nothing joins the nine identical `FailureHash` values into "this is shared."

## Why this is next

This is the automatic **recognize** step. Without it, a shared failure only becomes known-bad if a worker manually declares it (the W1 record path). Cross-trace correlation makes the fleet notice the shared cause on its own, from signals guardrsi already emits.

## Working spine

A fleet-wide correlator (new `internal/guardrsi/fleetcorrelate.go`, pure) keyed by `FailureHash`, counting **distinct TraceIDs** within a rolling window. When a `FailureHash` is observed from >= K distinct traces inside the window, it emits a known-bad candidate that the gateway forwards to `fak knownbad record` (#2713) — carrying the failure's reason class and the union of the traces' declared trees as the signature tree. Same emissions from a single trace never promote (that stays the existing per-trace livelock nudge).

Pure fold: `Correlate(observations []{TraceID, FailureHash, Reason, TreeGlobs, TS}, k, windowSecs, now) -> []Candidate`. The gateway holds the observation buffer under its existing server mutex (same discipline the `LivelockDetector` uses today).

## In scope

- `internal/guardrsi/fleetcorrelate.go`: the distinct-trace-over-window fold + `Candidate` shape.
- Wiring the gateway's existing livelock observation point to also feed the correlator and, on promotion, call the W1 record seam.
- Tests: K distinct traces promote; K emissions from one trace do not; observations outside the window age out.

## Out of scope

The ledger itself (W1 #2713), the blast-radius agent set (W3), any dispatcher behavior (W4). This only *produces* candidates.

## Done condition

The fold, given 3 distinct traces emitting the same `FailureHash` within the window, returns exactly one candidate; given 3 emissions from a single trace, returns none; observations older than the window do not count toward K.

## Witness

`go test ./internal/guardrsi/...` green with the three cases above; a captured JSON candidate from a table test. Cite the test name in the commit body.

## Acceptance gate

`make ci` green; `go test ./internal/guardrsi/...` green.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak guardrsi)` that lands `internal/guardrsi/fleetcorrelate.go` + tests + the gateway wiring.

## Dependencies

- after: #2713 (needs the knownbad record seam to forward candidates to)
- related: #2712

## Likely files

- `internal/guardrsi/livelock.go`
- `internal/guardrsi/fleetcorrelate.go`
- `internal/gateway/gateway.go`

## Lane

guardrsi

## Work unit

leaf

## Expected steps

6

## Assumptions

- The gateway is the single place all traces' livelock observations already pass through, so one correlator instance sees the whole fleet's failures.
- The union of the correlated traces' declared trees is a good-enough signature tree for v0; W3 refines the true blast radius.

## Confusion risks

- Keep the per-trace `LivelockDetector` untouched — this is an *additional* cross-trace aggregator, not a replacement. One nudges a single stuck agent; the other promotes a shared cause.
- Distinct-trace count, NOT total-emission count: 5 loops in one session must not look like 5 agents.

## Coordination

Touches `internal/gateway/gateway.go` (contended) — narrow pathspec commit; arbitrate the gateway keyword lane.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per correlation seam; deduped by the `(fak guardrsi)` cross-trace-correlator work.

---

## W3 — estimate — #2715

- **title:** feat(blast): blast-radius estimator — join the affectedtests import graph with live leases (W3, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2715

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Joins `internal/affectedtests` (dependency graph) with the live lease set.

## Current state

`fak affected` (`internal/affectedtests`) already computes the **dependency blast radius of a change** — the changed packages plus every package that transitively imports one — but only to select tests. The dos lease ledger (`internal/leaseref`, `dos lease-lane live`) knows which trees live workers hold. **Nothing joins them**: given a broken package/tree, the fleet cannot answer "which in-flight agents are actually affected?" So today a hold is all-or-nothing guesswork.

## Why this is next

Scope-hold (W4), single-fixer (W5), and the operator card (W7) all need the *affected set*. This is the **estimate** step: turn a broken tree into the concrete list of live leases and queued issues that intersect its dependency blast radius, so the fleet can hold precisely those and let the rest run.

## Working spine

`fak blast estimate <path|package> [--json]`:

1. Expand the broken package to its dependents via the `internal/affectedtests` import graph (the dependency blast radius).
2. Read the live lease set (`internal/leaseref` / the dos lease ledger).
3. Return the leases whose tree intersects the blast radius, plus the queued issues whose declared paths intersect — the **affected set**.

Pure join core in a new `internal/blastradius` (or under `internal/knownbad`): `Estimate(graph, brokenPkg, leases, issues) -> AffectedSet`. Impure shell gathers the graph, the leases, and the issue paths.

## In scope

- `internal/blastradius`: the pure `Estimate` join + `AffectedSet` shape.
- `cmd/fak/blast.go`: `estimate` subcommand, `--json`.
- Reuse `internal/affectedtests` for the graph and `internal/leaseref` for the leases — no new graph/lease code.
- Tests over a synthetic graph + lease set.

## Out of scope

Acting on the affected set — holding (W4), electing a fixer (W5), rendering the operator card (W7). This verb only *reports* who is affected.

## Done condition

Given a synthetic import graph, a broken package P, and a set of live leases, `Estimate` returns exactly the leases whose tree intersects P's dependents and excludes the disjoint ones; the queued-issue intersection behaves the same.

## Witness

`go test ./internal/blastradius/...` green; a captured `fak blast estimate --json` over a fixture showing the affected leases and the excluded disjoint ones. Cite in the commit body.

## Acceptance gate

`make ci` green; `go test ./internal/blastradius/... ./cmd/fak/...` green.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak blast)` landing `internal/blastradius` + `cmd/fak/blast.go` + tests.

## Dependencies

- after: #2713
- related: #2712, W4 (#scope-hold consumes this)

## Likely files

- `internal/blastradius/blastradius.go`
- `internal/blastradius/blastradius_test.go`
- `cmd/fak/blast.go`
- `internal/affectedtests/`
- `internal/leaseref/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- The `affectedtests` import graph is reusable outside the test-selection path (it already returns a package->dependents structure).
- Lease trees and package import paths can be reconciled to a common tree/glob form for intersection.

## Confusion risks

- Blast radius here is the *dependency* blast radius (who imports the broken package), NOT the changed-file set of a diff. Reuse the dependents direction of the graph.
- A lease tree may be globs; a package is a path — intersect at the tree/glob level, the same way W1 does.

## Coordination

Mostly new package + new verb; touches `cmd/fak/main.go` (dispatch table) — narrow pathspec.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per estimator seam; deduped by the `(fak blast)` verb.

---

## W4 — scope-hold — #2716

- **title:** feat(dispatch): scope-hold only the blast-radius-intersecting issues via BLOCKED_BY_KNOWN_BAD (W4, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2715
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2716

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 and the blast-radius estimator W3. Extends `internal/dispatchtick` and `cmd/fak/dispatch_skipped.go`.

## Current state

`internal/attemptbudget` holds a *single* issue after N failed attempts; `fak dispatch skipped` surfaces issues the router statically labelled `BLOCKED_BY_HUMAN`. Neither reacts to a *live, runtime-discovered* shared blocker: an issue whose tree intersects a known-bad signature is dispatched anyway, so the worker walks straight into the shared bug. The naive fix — pause the whole fleet — is the opposite failure (global stall); we want to hold **only** the affected agents.

## Why this is next

This is the **scope-hold** step and the whole point of "progress, not stall": disjoint agents keep shipping while only the blast-radius-intersecting work waits. Without it the ledger is informational only.

## Working spine

The dispatch router consults the known-bad ledger (#2713) and the blast estimate (W3) before offering an issue:

- An issue whose declared paths intersect a **live** known-bad signature is skipped with a NEW closed-vocabulary reason `BLOCKED_BY_KNOWN_BAD`, carrying the signature id and the elected fixer (W5) as the "next action."
- An issue disjoint from every live signature dispatches normally.

Add `BLOCKED_BY_KNOWN_BAD` to the router's skip-reason set and to the `fak dispatch skipped` card (a distinct row from the human-blocked set), and register it in the dos refuse-reason vocabulary so the skip is a structured, verifiable refusal.

## In scope

- `internal/dispatchtick`: the known-bad intersection check + the new skip reason.
- `cmd/fak/dispatch_skipped.go`: render the known-bad-blocked rows (separate from `BLOCKED_BY_HUMAN`).
- Register `BLOCKED_BY_KNOWN_BAD` in the closed refuse vocabulary (`dos.toml [reasons]` / the reason set).
- Tests: intersecting issue skipped with the reason; disjoint issue still dispatchable.

## Out of scope

Electing the fixer (W5), releasing the hold (W6), the operator card (W7). This only decides skip-vs-dispatch per issue against the live ledger.

## Done condition

With one live known-bad over tree T, `fak dispatch route` skips exactly the issues whose paths intersect T (reason `BLOCKED_BY_KNOWN_BAD`) and still routes issues disjoint from T for dispatch.

## Witness

`go test ./internal/dispatchtick/... ./cmd/fak/...` green; a captured `fak dispatch route --json` (or the skipped card) over a fixture with one live signature, showing the scoped skip + a disjoint dispatchable issue. `dos check-reason BLOCKED_BY_KNOWN_BAD` returns known. Cite in the commit body.

## Acceptance gate

`make ci` green; the scoped-skip fixture transcript captured; the reason resolves in the closed vocabulary.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak dispatch)` landing the router check + the skip reason + the card + the vocabulary entry + tests.

## Dependencies

- after: #2713 (ledger), W3 (estimate)
- related: #2712

## Likely files

- `internal/dispatchtick/dispatchtick.go`
- `cmd/fak/dispatch_skipped.go`
- `dos.toml`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- The router already has each issue's declared paths (it routes by lane/paths today), so intersection needs no new issue metadata.
- Adding a reason to the closed vocabulary is the accepted way to make a new structured skip verifiable (the refuse-reason discipline).

## Confusion risks

- Hold ONLY the intersecting subset — never all issues. A global pause is the failure mode this issue exists to avoid.
- `BLOCKED_BY_KNOWN_BAD` (dynamic, runtime) is distinct from `BLOCKED_BY_HUMAN` (static, router-labelled) — different rows, different next-actions.

## Coordination

Touches `internal/dispatchtick/dispatchtick.go` and `dos.toml` (both contended) — arbitrate the dispatch keyword lane; narrow pathspec commit.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per scope-hold seam; deduped by the `BLOCKED_BY_KNOWN_BAD` reason.

---

## W5 — elect fixer — #2717

- **title:** feat(knownbad): elect exactly one fixer per known-bad via an exclusive lease (W5, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, hot-tree, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2717

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Uses the dos exclusive-lease seam (`internal/leaseref` / `dos arbitrate`).

## Current state

When a shared bug is found, several agents may each decide to fix the same shared file — N racing edits/PRs to one tree, the `hot-tree` collision at fleet scale. `dos arbitrate` can keep trees disjoint, but nothing declares "exactly one agent owns fixing THIS known-bad tree"; the affected agents have no pointer to a fixer, so the default is to collide or to each wait blindly.

## Why this is next

This is the **elect one fixer** step. Scope-hold (W4) parks the affected agents; without a single elected fixer, either nobody fixes it (everyone parked) or everybody does (collision). Exactly-one is the invariant that makes parking safe.

## Working spine

A known-bad signature can be **claimed** by exactly one fixer:

- `fak knownbad claim <signature>` acquires an **exclusive dos lease** on the signature's broken tree and records the claimant on the ledger record.
- A second `claim` on an already-claimed signature is REFUSED (structured reason) and returns the current fixer's identity — so the loser gets a pointer, not a collision.
- The claim is the thing W4's skip card and W7's operator card point at ("parked — @fixer owns the fix").

Exactly-one is enforced by the exclusive lease (the arbiter already guarantees a single exclusive holder over an intersecting tree), not by ledger bookkeeping alone.

## In scope

- `cmd/fak/knownbad.go`: the `claim` subcommand over the exclusive-lease seam.
- `internal/knownbad`: record the claimant + claim time on the signature.
- Tests: two concurrent claims -> exactly one wins, the other is refused with the winner's identity.

## Out of scope

Releasing the claim / resolving the signature (W6), the dispatcher skip (W4), the operator card (W7). This only elects and records the single fixer.

## Done condition

Two agents race `fak knownbad claim <sig>`: exactly one acquires the exclusive lease and is recorded as fixer; the other exits refused, printing the winner's identity. A claim on an unclaimed signature succeeds and stamps the claimant.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including the race case (two claims, one winner); a captured transcript of the second claim's refusal naming the winner. Cite in the commit body.

## Acceptance gate

`make ci` green; the two-claims-one-winner test green; the refusal transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the `claim` verb + the exclusive-lease wiring + the race test.

## Dependencies

- after: #2713 (ledger)
- related: #2712, W4 (#scope-hold points at the claimant), W6 (#auto-release drops the claim)

## Likely files

- `cmd/fak/knownbad.go`
- `internal/knownbad/knownbad.go`
- `internal/leaseref/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The dos exclusive lease over an intersecting tree is a sufficient exactly-one mutex (the arbiter refuses a second exclusive holder), so no extra distributed-lock code is needed.
- A dead claimant's lease is reaped by the existing lease-liveness path; a stale claim then becomes re-claimable (see the shared-trunk lease-reap gotcha).

## Confusion risks

- Exactly-one is enforced by the LEASE, not by the ledger write — do not "claim" by only appending a record (two agents could both append). The lease acquisition is the gate.
- A refused claim must return the WINNER (so the loser has a pointer), not just "refused."

## Coordination

New subcommand + lease integration; touches `cmd/fak/knownbad.go` (owned by W1) — sequence after #2713 lands. Low tree-collision risk otherwise.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per fixer-election seam; deduped by the `fak knownbad claim` verb.

---

## W6 — auto-release — #2718

- **title:** feat(knownbad): witness-gated auto-release of held agents on a proven fix (W6, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, priority/P1
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2716, #2717
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2718

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713, the scope-hold W4, and the fixer election W5. Reuses `internal/affectedtests` + `dos verify`.

## Current state

A held/parked agent (W4) has no automatic way back. Once the elected fixer (W5) lands the fix, the known-bad signature stays `open`, so the affected agents stay parked forever unless something clears it. Clearing it on a fixer's **self-report** ("I fixed it") is exactly the failure this repo's witness discipline forbids — an unproven claim must not release the fleet.

## Why this is next

This is the **auto-release** step — the one that makes parking safe to enter, because it is guaranteed to end on real evidence. It is the closing arm of the loop: without it, scope-hold is a one-way door.

## Working spine

`fak knownbad resolve <signature>` flips a signature `open -> resolved` **only when the fix is witnessed**:

- Witness = a green `fak affected` over the broken package AND/OR `dos verify` binding the fixer's commit to the signature's tree. No witness -> stays `open`, reported as `not yet` with the missing witness.
- On resolve: drop the fixer's exclusive lease (W5) and clear the scope-hold so W4 stops skipping the previously-blocked issues — they become dispatchable again on the next tick.

The witness is the gate; `resolve` is refused (structured reason) without it.

## In scope

- `cmd/fak/knownbad.go`: the `resolve` subcommand, witness-gated.
- `internal/knownbad`: the `open -> resolved` transition + release of the hold/lease.
- Reuse `internal/affectedtests` (green over the tree) and the `dos verify` seam for the witness.
- Tests: resolve refused with no witness; resolve succeeds + releases with a witnessed green over the tree.

## Out of scope

Detecting the shared failure (W2), estimating blast radius (W3), the skip itself (W4), the operator card (W7), TTL/expiry (W8). This only closes a signature on evidence.

## Done condition

`fak knownbad resolve <sig>` with no witness leaves the signature `open` and prints the missing witness; with a witnessed green `fak affected` over the tree it flips to `resolved`, drops the fixer's lease, and the previously `BLOCKED_BY_KNOWN_BAD` issues route as dispatchable on the next `fak dispatch route`.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including both the refused-without-witness and released-with-witness cases; a captured transcript showing a skipped issue becoming dispatchable after resolve. Cite in the commit body.

## Acceptance gate

`make ci` green; the two resolve cases green; the skip->dispatchable transcript captured.

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the witness-gated `resolve` + the hold/lease release + tests.

## Dependencies

- after: #2713 (ledger), W4 (the hold to release), W5 (the lease to drop)
- related: #2712

## Likely files

- `cmd/fak/knownbad.go`
- `internal/knownbad/knownbad.go`
- `internal/affectedtests/`

## Lane

dispatch

## Work unit

leaf

## Expected steps

7

## Assumptions

- A green `fak affected` over the broken package is an acceptable machine witness that the shared bug is gone for v0; `dos verify` on the fixer commit strengthens it.
- Releasing the hold is a ledger status flip the dispatcher already re-reads each tick (no push needed).

## Confusion risks

- Never resolve on a self-report or a bare commit subject — the witness must be an independent green/verify. Absent it, report `not yet`, not `done`.
- Resolve must release BOTH the scope-hold (W4) and the fixer's lease (W5); dropping one without the other leaves the fleet half-stuck.

## Coordination

Extends `cmd/fak/knownbad.go` (owned by W1/W5) — sequence after those land. Reuses existing affected/verify seams.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per auto-release seam; deduped by the `fak knownbad resolve` verb.

---

## W7 — surface — #2719

- **title:** feat(knownbad): operator blast card — 1 root cause → N affected, 1 fixing, N-1 parked (W7, epic #2712)
- **labels:** dispatch, dev-ex, operator, observability, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713, #2715
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2719

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713 and the blast-radius estimator W3. Extends `internal/blockerpost` and the dispatch scorecard/TUI.

## Current state

`internal/blockerpost` posts a single blocker to Slack with severity tiers (status/operator/clear); the dispatch TUI and scorecard show per-issue rows. Neither shows the **blast frame** — the one thing an operator needs to see at a glance: "1 root cause -> N affected, 1 fixing, N-1 parked, witness pending." Today a shared bug taxing the fleet is invisible as a *shared* event; it reads as N unrelated stuck workers.

## Why this is next

This is the **surface** step. Recognition (W2), estimation (W3), and holding (W4) can all be working while the operator still can't tell one shared bug from nine coincidences. One blast-framed card turns the whole containment state into a single legible line.

## Working spine

One card per **live** known-bad signature, fed from the ledger (#2713) + the blast estimate (W3):

- Fields: root-cause/signature, affected count, the elected fixer (W5), parked count, and witness status (pending/resolved).
- Severity: `status` (muted) while a fixer is claimed and progressing; escalate to `operator` (surfaced mention) when a signature has NO fixer after a threshold, or the witness is overdue.
- Empty ledger -> a quiet all-clear line (the existing `clear` tier), so a scheduled run honestly says "no shared blockers."

Rendered by a `fak knownbad report` verb and wired into the `blockerpost` feeder + a dispatch scorecard/TUI row.

## In scope

- `cmd/fak/knownbad.go` (or `cmd/fak/blockers.go`): a `report` that renders the per-signature blast card.
- `internal/blockerpost`: a fold from the live known-bad set to a `Blocker` (reuse `Severity`/`Blocker`/`Blocks`).
- A dispatch scorecard/TUI row for live signatures.
- Tests over a fixture (1 fixer + 5 affected -> one card; no-fixer -> operator severity; empty -> clear).

## Out of scope

Deciding the hold (W4), electing the fixer (W5), resolving (W6). This only *renders* the state the other children produce.

## Done condition

Given a live known-bad with 1 fixer + 5 affected, `fak knownbad report` renders one card "root cause X -> 6 affected, 1 fixing (@who), 5 parked, witness: pending"; a signature with no fixer past the threshold renders at `operator` severity; an empty ledger renders the quiet all-clear.

## Witness

`go test ./internal/blockerpost/... ./cmd/fak/...` green for the three render cases; a captured `fak knownbad report` (dry-run) card for each. Cite in the commit body.

## Acceptance gate

`make ci` green; the three render cases green; the cards captured (dry-run, posts nothing).

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` (or `(fak blockers)`) landing the report renderer + the blockerpost fold + the scorecard row + tests.

## Dependencies

- after: #2713 (ledger), W3 (affected set)
- related: #2712, W5 (fixer identity shown on the card)

## Likely files

- `internal/blockerpost/blockerpost.go`
- `internal/blockerpost/render.go`
- `cmd/fak/knownbad.go`
- `cmd/fak/dispatch_scorecard.go`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- The `blockerpost` severity tiers (status/operator/clear) map cleanly onto the fixer-present / no-fixer / empty cases, so no new transport is needed.
- The affected/fixer/parked counts come straight from W3 + the ledger; the card is pure rendering.

## Confusion risks

- This is a READ/render surface — it must not decide holds or elect fixers; it reflects the ledger + estimate.
- Default to dry-run: a scheduled render must never page on a `status`-tier card, only on `operator`.

## Coordination

Touches `internal/blockerpost` + a dispatch scorecard file (contended) — narrow pathspec; sequence after W3 + W5 so the fields exist.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per operator-surface seam; deduped by the `fak knownbad report` verb.

---

## W8 — bound — #2720

- **title:** feat(knownbad): bounded TTL + revoke so a stale known-bad can't wedge the fleet (W8, epic #2712)
- **labels:** dispatch, dev-ex, fleet-400iph, security, priority/P2
- **milestone:** 6
- **parent:** #2712
- **depends-on:** #2713
- **filed:** https://github.com/anthony-chaudhary/fak/issues/2720

### Body

## Parent context

Parent: blast-radius containment epic #2712. Depends on the known-bad ledger spine #2713. Hardens the liveness check W1 stubs. This is the safety valve for the whole epic.

## Current state

W1 (#2713) records a known-bad signature and matches against *live* ones, but a v0 liveness check is just "status == open." A signature that never expires — or was **misattributed** (a flaky test read as a shared bug, a fix that landed but was never resolved) — would scope-hold the affected agents **forever**. That is the exact inverse of today's blindness: instead of nobody noticing the bug, nobody can escape a phantom one.

## Why this is next

Every other child makes the hold *tighter*; this one makes it *bounded*. It is what lets an operator (and the fleet) trust scope-hold — a hold that cannot self-expire is a liability, not a feature. Ship it alongside the spine's consumers so no signature can wedge the fleet.

## Working spine

Two bounded escape hatches on every signature:

- **TTL**: each record carries a bounded `ttl_seconds` (default e.g. 30-60m). `fak knownbad match` treats a record past `discovered_at + ttl` as expired -> `matched:false`, so the scope-hold auto-lifts even if nobody resolves it. A live shared bug re-fires and re-records; a phantom one just ages out.
- **Revoke**: `fak knownbad revoke <signature> --reason <class>` falsifies an open record immediately (e.g. "it was flaky, not shared") -> stops matching now. A structured refuse reason is emitted if a worker tries to `claim`/`resolve` an already expired/revoked signature.

Both paths release the scope-hold (W4) the same way `resolve` (W6) does — the difference is `resolve` needs a witness, TTL/revoke are the *unproven-so-release* safety valves.

## In scope

- `internal/knownbad`: TTL-aware liveness in the `Match` fold (`now` supplied as data); the `revoke` transition.
- `cmd/fak/knownbad.go`: the `revoke` subcommand + a default TTL on `record`.
- A closed refuse reason for acting on an expired/revoked signature.
- Tests: past-TTL record does not match; `revoke` stops matching immediately; both lift the hold.

## Out of scope

Recognition (W2), estimate (W3), skip (W4), election (W5), witnessed resolve (W6), the card (W7). This only bounds and falsifies signatures.

## Done condition

A record past its `ttl_seconds` reports `matched:false` (hold auto-lifts); `fak knownbad revoke <sig>` flips an open record to `revoked` so it stops matching immediately; a `claim`/`resolve` against an expired/revoked signature returns a structured refuse reason.

## Witness

`go test ./internal/knownbad/... ./cmd/fak/...` green including the past-TTL non-match, the revoke, and the refuse-on-expired cases; a captured transcript of a match flipping to `false` once `now` passes the TTL. Cite in the commit body.

## Acceptance gate

`make ci` green; the TTL + revoke cases green; the refuse reason resolves in the closed vocabulary (`dos check-reason`).

## Closure binding

Closed by a `Closes #<this>` commit stamped `(fak knownbad)` landing the TTL-aware liveness + `revoke` + the refuse reason + tests.

## Dependencies

- after: #2713 (ledger)
- related: #2712, W4 (the hold both paths lift), W6 (the witnessed-resolve sibling)

## Likely files

- `internal/knownbad/knownbad.go`
- `internal/knownbad/knownbad_test.go`
- `cmd/fak/knownbad.go`
- `dos.toml`

## Lane

dispatch

## Work unit

leaf

## Expected steps

6

## Assumptions

- A bounded default TTL is safe because a still-live shared bug re-fires and re-records; the cost of a too-short TTL is one extra rediscovery, the cost of no TTL is a permanent wedge.
- The `Match` fold already takes `now` as data (W1), so TTL is a pure comparison, not a clock read.

## Confusion risks

- TTL/revoke are the UNWITNESSED release valves; `resolve` (W6) is the WITNESSED close. Keep them distinct — a revoke is "this was never really a shared bug," a resolve is "the shared bug is proven gone."
- A too-long TTL re-creates the wedge this issue exists to prevent; pick a bounded default and make it overridable per record.

## Coordination

Extends `internal/knownbad` + `cmd/fak/knownbad.go` (owned by W1) and `dos.toml` — sequence after #2713; narrow pathspec on `dos.toml`.

## Trigger

Filed once at epic fan-out.

## Batch policy

One issue per TTL/falsifiability seam; deduped by the `fak knownbad revoke` verb.

