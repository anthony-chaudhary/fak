---
title: "Mirage deep study — refreshed 2026-08-25"
description: "Mirage is not a replacement for fak's model/tool-call kernel. It is a complementary resource substrate: one POSIX-like namespace over local files,"
---
# Mirage deep study — refreshed 2026-08-25

**Source:** [`strukto-ai/mirage`](https://github.com/strukto-ai/mirage)
**Pinned revision:** [`2ed4257af98fc1a206a5444057d1290892190e69`](https://github.com/strukto-ai/mirage/commit/2ed4257af98fc1a206a5444057d1290892190e69) (checked 2026-08-25)
**Release checked:** `v0.0.5` at `d4f2ff5de2ec8e6eeb3b283f794d7b85bae53835` (published 2026-08-15)
**License:** Apache-2.0 (`LICENSE` at the pinned revision)
**Acquisition:** full git clone in allocated scratch; 8,406 tracked files and 1,564 commits at the pin. GitHub issues, pull requests, discussions, releases, and repository metadata were also queried on 2026-08-17.

## Exhaustive inventory refresh — 2026-08-25

The committed [`fak-study-inventory-map/1`](../research/inventory/strukto-ai-mirage.json) indexes all **8,921 tracked files**, **1,556 directories**, and **1,116,124 text lines** at the pinned revision across 17 top-level subsystems. The largest denominators are `python/` (3,461 files), `typescript/` (3,083), `integ/` (1,324), `examples/` (364), `docs/` (327), and `spec/` (285); the map also records the remaining 11 roots and representative paths. The local walk covers README/docs, architecture/design, runtime, tests/fixtures, history-bearing files, roadmap tokens, and license/provenance.

Non-tree evidence is complete at the pinned commit timestamp: **206 issues** (99 open, 107 closed at read-back), **706 pull requests** (13 open, 24 closed, 669 merged), **zero discussions**, five releases/tags, and all 1,798 commits. GitHub issue/PR lists were paged beyond their complete denominators, GraphQL returned `totalCount: 0` for discussions, and REST returned all five releases. No dedicated `ROADMAP` or `CHANGELOG` file exists; explicit roadmap issues [#721](https://github.com/strukto-ai/mirage/issues/721) and [#705](https://github.com/strukto-ai/mirage/issues/705), proposal [#861](https://github.com/strukto-ai/mirage/issues/861), the full issue/PR ledger, release records, Git history, and the unfinished-work-marker scan supply those classes without presenting proposals as shipped behavior.

Provenance remains suitable for clean-room adaptation: the root and Python licenses are Apache-2.0, package manifests declare Apache-2.0, and `licenses/` retains bundled MIT and BSD-3-Clause notices. A direct code port still requires file-level provenance review.

The candidate matrix was re-run against current fak. Three `fak capabilities` queries covered external-input drift, backend/workspace capabilities, and VFS/tool semantics. The original six dispositions still hold: [#7256](https://github.com/anthony-chaudhary/fak/issues/7256) remains the one distinct borrow; typed backend support maps to [#5310](https://github.com/anthony-chaudhary/fak/issues/5310); staged workspaces and observer continuity map to [#4234](https://github.com/anthony-chaudhary/fak/issues/4234); policy placement is shipped; and a universal VFS remains a rejected default. No additional dispatchable borrow survived, so no new issue was filed.
## Verdict

Mirage is not a replacement for fak's model/tool-call kernel. It is a complementary **resource substrate**: one POSIX-like namespace over local files, object stores, SaaS APIs, databases, and agent memory, with caching, policy, observation, snapshots, and a shell. Its strongest transferable idea is not “add a VFS.” It is: **when a run depends on mutable external resources, capture what the agent actually saw, pin or fingerprint it, and refuse silent replay drift before the next effect.**

The study found three valuable mechanisms. Two already have stronger or correctly scoped fak work underway; one survives as a current gap:

1. **External-input snapshot drift gate — borrow by adaptation.** Mirage records per-path version pins or fingerprints at snapshot time and checks them lazily before reads and before the first write. `STRICT` refuses changed content, `WARN` reports and continues, and `OFF` is the only opt-out. fak has rich artifact digests, effect receipts, resume witnesses, and source-drift work, but no equivalent gate binding a resumed run's mutable external inputs to the bytes or versions it observed. This is the surviving borrow, filed as fak [#7256](https://github.com/anthony-chaudhary/fak/issues/7256).
2. **Per-operation backend capability declaration — bounded-superset inspiration.** Mirage centralizes `supports_op(op_name, path)` and uses it at dispatch seams rather than trusting each backend to fail consistently. fak already has typed provider/tool capabilities, feature negotiation, unsupported-capability degradation, and the open MountView execution gap #5310. A generic VFS capability matrix would be peripheral today; the useful spirit belongs in existing typed adapter seams, not a new substrate.
3. **Transactional workspace branches — exclude as duplicate.** Mirage's open roadmap (#138, #165, #175, #183) proposes copy-on-write staged writes, tombstones, rollback, and commit across mounts. fak already tracks the closer agent-kernel version in #4234 and #4236: reversible effectful speculation under isolated branches with judged merge-or-abort. Mirage adds evidence for those issues, not a competing ticket.

## What Mirage actually ships

### One workspace, many resource kinds

`python/mirage/workspace/workspace/workspace.py` resolves virtual paths to mounts, dispatches operations, maintains revision state, and exposes snapshot/load. `docs/home/architecture.mdx` describes the workspace as orchestration over resources, cache, observer, and policy. The repository includes Python and TypeScript implementations plus broad integration suites for object stores, databases, SaaS systems, FUSE, and shell behavior.

This is useful as a **bounded superset** for agents whose task is dominated by heterogeneous remote data. It is not fak's best default: putting every tool/resource behind filesystem semantics would flatten typed tool contracts, add a large compatibility surface, and move fak away from its central checkpoint at model/tool calls.

### Snapshot fidelity and drift detection

The concrete mechanism is split across:

- `python/mirage/workspace/snapshot/api.py`: serializes workspace state and optionally includes content;
- `python/mirage/workspace/snapshot/drift.py`: installs revision pins or fingerprints and implements the lazy drift checker;
- `python/mirage/types.py:693`: declares `DriftPolicy.STRICT|WARN|OFF`;
- `python/mirage/workspace/workspace/workspace.py:464-584`: snapshot/load entry points;
- `python/tests/workspace/test_snapshot_drift.py`: proves strict refusal before reads and before an operation writes, warn/off behavior, and byte fallback;
- `python/tests/e2e/test_snapshot_drift_live.py`: checks live ETag drift and false-positive resistance.

The important design detail is **effect ordering**. Drift is checked not only when data is read, but before a write can clobber a remote object that changed after the snapshot. If the backend supports historical revisions, Mirage prefers the pinned revision. Otherwise it compares a recorded fingerprint; included bytes can preserve the exact view the agent saw.

### Typed operation support

`python/mirage/workspace/mount/mount.py:590` computes operation support from mount mode, path, resource capabilities, and command availability. Dispatch and metadata code call that seam before operations (`workspace/dispatcher/dispatcher.py:437`; `workspace/executor/builtins/metadata.py:662`). This is stronger than scattered “try it and inspect the error,” and Mirage's open issue #281 explicitly pushes optional capabilities into generic commands.

fak already applies this pattern at more relevant seams: orchestration capability degradation, provider/tool feature negotiation, policy capability floors, and typed unsupported results. The directly adjacent unfinished work is #5310, which parses a MountView refusal but does not yet execute it at the call-side adjudicator.

### Observer and policy placement

`python/mirage/observe/observer.py` sends file events to a store and subscriber set; `python/mirage/workspace/workspace/policy.py` routes policy by mount; runtime policy types carry structured decisions. Mirage recently closed #675 after proving that shell-only policy checks let FUSE and programmatic operations bypass denial. That history reinforces fak's existing architecture: put policy at the shared operation checkpoint, not in one user interface. fak's gateway/result observers are already read-only, sampled, and fail-open where appropriate, so no new borrow survives here.

## Proposed and incomplete ideas mined

Open issues were treated as proposals, not shipped behavior:

- **#721 — consistency, versioning, and storage roadmap:** staged evolution from snapshot/version identity toward durable shared state.
- **#165 / #138 / #175 / #183 — transactional workspace:** branch-local overlay, tombstones, rollback, then mount-agnostic commit.
- **#333 — version-aware caches:** revision-keyed entries and dirty-aware eviction for snapshots/rollback.
- **#366 — Redis cross-process cache semantics:** asks what consistency contract a shared cache should expose.
- **#210 — fork drops pending drift checks:** a useful negative witness that snapshot metadata must survive cloning.
- **#211 — fork replaces a custom observer:** a useful negative witness that cloned execution contexts must retain explicitly configured observability.
- **#281 — optional streaming capability:** backend variation should be declared and negotiated rather than discovered through runtime crashes.

These proposals strengthen the external-drift candidate and caution against a monolithic VFS import. Mirage itself is still converging on cross-process cache semantics, transactional mounts, and fork fidelity.

## Current-fak witness

The comparison queried fak's code, docs, claims index, and open issue corpus on 2026-08-17.

| Mirage mechanism | Current fak evidence | Classification |
|---|---|---|
| Snapshot fingerprints and strict pre-effect drift checks | fak records digests and provenance in artifact, replay, cache, and effect-receipt surfaces; #3951/#4098 address drift of borrowed source excerpts; no run-level contract records mutable external inputs and checks them before resumed effects. | **GAP — adapt** |
| Typed `supports_op` at the backend seam | `internal/providerfeatures`, gateway capability fields, orchestration capability degradation, and policy tooling already negotiate typed support; #5310 covers the closest missing MountView enforcement. | **PARTIAL/PLANNED — feed existing issue** |
| CoW staged workspace rollback | #4234 and #4236 already specify reversible effectful branches, isolation, judged merge/abort, receipts, and cleanup. | **PLANNED — duplicate** |
| Policy at every operation door | fak's adjudicator/kernel/gateway checkpoint is already the core architecture; Mirage #675 is a cautionary confirmation. | **SHIPPED principle** |
| Observer retention through forks | fak has read-only gateway/model observers and orchestration receipts, but Mirage's exact VFS-fork defect is not central to fak. Telemetry continuity should be tested when #4234 lands rather than becoming a standalone VFS-shaped issue. | **DEFER into #4234 witness** |
| Unified filesystem for all resources | No equivalent, intentionally. Typed tools preserve semantics better for fak's default use case. | **EXCLUDE default; bounded optional integration only** |

## Borrow candidate that survives

### External-input drift contract for resumable runs — fak #7256

**For:** operators resuming or replaying an agent run that read mutable remote state.
**Problem:** fak can prove its own artifacts and effects, but cannot currently prove that remote inputs are still the versions the agent observed.
**Today:** resume can continue against changed bytes or metadata without a typed pre-effect refusal.
**Better because:** a compact input manifest records `{resource identity, observed version or digest, capture time, optional retained bytes}` and a resume gate checks it before the next effect; strict is the default, warn/off are explicit.
**Witness:** a fake mutable resource changes after capture; strict resume refuses before the effect sink is called, warn emits a typed drift receipt and continues, and retained/pinned bytes replay the original observation.

This should adapt the behavior, not port Mirage code. fak is Go and its resource identities are typed tool/artifact references rather than virtual paths. Apache-2.0 would permit a direct port, but clean adaptation is smaller and fits fak's existing effect-receipt and resume seams.

## Best-default frontier and bounded superset

- **Best default for fak:** keep typed tools/resources and add a small, optional external-input manifest plus strict pre-effect drift gate for replay/resume. No filesystem emulation is required.
- **Bounded superset:** an adapter may expose a Mirage workspace or another VFS as one typed tool for data-heavy cohorts. Its cache, policy, and snapshot claims must remain subordinate to fak's kernel checkpoint and must surface typed capability/drift receipts.
- **Rejected frontier move:** replacing fak's tool contracts with POSIX paths. The compatibility and semantic-loss cost dominates for the common agent-control path.

## Provenance and limits

No Mirage source bytes were copied into fak. All implementation references above are observations at the pinned revision; candidate implementation is adaptation/inspiration. GitHub metadata is time-sensitive and was observed on 2026-08-17: 3,499 stars, 259 forks, 96 issues total, 8 pull requests total, and latest release `v0.0.5`. The repository was created 2026-05-06, so its breadth is notable but its long-term compatibility and operational stability are not yet established.

The guarded parallel study used fak orchestration run `orch-37f308f70b0c4e604789ed82` with a three-role ultracode DAG, required leases, independent witnesses, and effect readback. Final claims here were independently re-read from the pinned clone and current fak tree rather than accepted from worker logs.
