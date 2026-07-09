# Capacity-aware reroute — the missing control-path arrow (#3520)

**Status:** pure kernel SHIPPED in `internal/modelroute` (0 dirty deps at author time);
live fleet-GPU dispatch + `cmd/fak` wiring remain OPEN (see below). Diff-witnessed by
`internal/modelroute/capacity_test.go` (17 cases).

## What shipped

`internal/modelroute/capacity.go` — a pure, deterministic capacity kernel that fills the
gap the hardware docs name explicitly: the policy plane and the physical plane "meet only
at the meter, never at the control"
(`docs/explainers/hardware-limits-and-capacity.md` §2). It is the control arrow between
them, and nothing more.

- `AssessCapacity(CapacityDemand, CapacityCeiling) CapacityVerdict` — decides keep-local
  (`CAPACITY_OK`) vs route-elsewhere (`CAPACITY_REROUTE`), or conservative-degrade
  (`CAPACITY_UNKNOWN`) when no actionable signal is present. Two orthogonal ceilings:
  - **Faithful param ceiling** — model size (billions) vs `CapacityCeiling.FaithfulParamsB`
    (default candidate `LocalFaithfulCeilingBillion = 7.0`, sourced from
    `hardware-limits-and-capacity.md:72`, "`fak` faithful ≤ 7B on the 36 GB Mac").
  - **Context-window over-subscription** — prompt + planned output vs the headroom-adjusted
    window (`DefaultDeviceHeadroom = 0.15`, sourced from `hardware-limits-and-capacity.md:303`).
- `CapacityReason` — a closed, `known…`-validated newtype (mirrors `Reduction`), so the
  reroute reason is emittable / verifiable / refusable per the doctrine note §2.6.
- `Manifest.RouteWithCapacity(Subject, CapacityDemand, CapacityCeiling)` — folds the kernel
  over `Route`: on reroute it stamps the OPEN `Labels["capacity"]="reroute"` signal on a
  **copy** of the Subject so an operator's top-of-list rule
  (`match:{labels:{capacity:reroute}} -> plan:{members:[{model:"fleet-large"}]}`) selects
  the fleet lane. Returns the typed verdict alongside the `Decision`.

## Design choices (and why)

- **Zero edits to existing files.** The kernel lives in new files and returns the typed
  reason in a new `CapacityVerdict` *alongside* `Decision`, rather than mutating the
  widely-constructed `Decision` struct. Keeps the change collision-free in the shared tree
  and leaves every existing route byte-identical.
- **Reroute via the existing `Labels` channel**, not a new match predicate — `Route`'s hot
  path is untouched; a Subject with no capacity signal routes exactly as today
  (`TestRouteWithCapacity_NoSignalRoutesByteIdentical`).
- **Fail OPEN.** An unknown capacity never reroutes — an absent signal is not evidence of
  overflow (docs Planks 1–5). `ContextWindow == 0` means unbounded/unknown, not zero-capacity.
- **Locality stays the roster's job.** The kernel compares declared numbers a caller
  measured; it never reasons about `Target.Kind`/hardware. The caller supplies the model
  size already parsed to billions (e.g. `turnbench.paramsBillions`), so the routing spine
  takes on no size-string parsing and no benchmark-leaf dependency.

## Open (honestly not done here)

- **`cmd/fak/route.go` wiring** — consulting `RouteWithCapacity` on the live dispatch path.
  That file was mid-edit by another agent at author time; deferred rather than swept.
- **A live fleet-GPU collective.** The reroute *target* (Platform 4, the 8-GPU server where
  "single-box ceilings stop binding", `docs/HARDWARE-MATRIX.md`) is a documented, planned
  lane — the Plank-6 capacity escape — but no live cross-device NCCL path exists yet
  (`hardware-limits-and-capacity.md`). This kernel emits the *directive*; realizing the
  destination is separate, tracked work.
- **Sourcing the per-task capacity facts** (model size, resolved window) into the caller
  that builds `CapacityDemand`. `Target.ContextTokens` already exists post-`Resolve`; a
  first-class per-model param size does not yet live on the routing path.
