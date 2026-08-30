# AI-Ops storage-qualification field study

Status: complete source study; borrow decisions are snapshot-scoped.

## Snapshot and scope

- Observed at: 2026-08-29 America/Los_Angeles.
- Source checkout: `C:\Users\antho\OneDrive\Desktop\work\AI-Ops`.
- Source state studied: `origin/master@ca8aef4a3d44d1d1206e3936ed3332f4bcd4eb86` (2026-06-19). The local checkout was `dbe9447`, one commit behind; the upstream-only change updates release packaging, not runtime architecture.
- Source license: no root `LICENSE`, `COPYING`, or `NOTICE` file was present at the studied revision. This pass therefore permits inspiration and independent reimplementation only; it does not authorize direct code porting.
- Repository surfaces opened: root product/design docs; Python CLI, config, discovery, doctor, provisioning, SSH and shell-sanitization; runner, analysis, profiles, catalog, FIO, publishing; Go UI API/runner/config/Bench adapter; tests; package/release files; Git history, tags/releases, issues and PRs.
- GitHub state: 48 published tags through `v0.10.48`; no repository issues or pull requests were returned by GitHub at observation time.

## Result

AI-Ops is an artifact-first SSD qualification controller for agentic-AI workloads. It resolves a declarative configuration into a staged CPU, DGX, Bench, or FIO plan; executes it locally or remotely; records run/profile state; reconstructs typed analysis data from artifacts; classifies bottlenecks and derives SSD requirements; then renders scorecards and optional publication output. The Go web UI is a ports-and-adapters shell over the Python CLI and artifact tree rather than a second qualification engine.

The useful lesson for fak is not the repository's broad orchestration shapeâ€”fak already has stronger typed receipts, workload profiles, correctness gates, and replayable tuning in `internal/computetune`. The useful remaining axis is narrower: AI-Ops turns workload observations into an explicit **resource qualification recommendation** (required latency, IOPS, bandwidth, endurance, and capacity), while fak's current compute tuning selects an implementation candidate for a declared profile. That is a distinct decision product worth considering for storage-backed inference envelopes.

## Architecture and control flow

1. `aiops` enters through `aiops.cli:main`; config loading prefers JSON and retains a legacy YAML path (`pyproject.toml:18-19`; `aiops/cli.py:46-66@ca8aef4`).
2. Plan builders convert resolved configuration into CPU, DGX, or FIO-comparison steps (`aiops/runner/plan_builders.py:101-176@ca8aef4`).
3. `RunPlan.execute` owns preflight, lifecycle, step dispatch, interruption, observer notification, and finalization (`aiops/runner/sequencer.py:209-225,329-358,529-593@ca8aef4`).
4. `RunTracker` persists aggregate run state and profile transitions, projecting them into SQLite catalog records (`aiops/runner/run_tracker.py:53-161@ca8aef4`; `aiops/catalog/db.py:24-79@ca8aef4`).
5. Observers receive run, step, analysis, publish, error, and interrupt events; observer failures are isolated from benchmark execution (`aiops/runner/observers.py:31-73@ca8aef4`; `aiops/runner/sequencer.py:36-45@ca8aef4`).
6. Post-processing reloads artifacts into typed `RunData`, then applies taxonomy, weight, bottleneck, efficiency, and SSD-spec stages (`aiops/analysis/loader.py:356-400@ca8aef4`; `taxonomy_classifier.py:240-278`; `weight_matrix.py:32-38`; `bottleneck.py:152-354`; `ssd_spec.py:92-234`).
7. Reporting and publication run downstream of the analysis artifact boundary (`aiops/cli.py:442,550,850@ca8aef4`; `aiops/publish/orchestrator.py:22-78@ca8aef4`).
8. The Go UI launches the Python CLI with a temporary config, captures stdout/stderr, streams live or replayed logs by SSE, and reads generated analysis/scorecard files (`ui/internal/runner/runner.go:247-261,469-480@ca8aef4`; `ui/internal/api/handler_runs_sse.go:24-118,186-197@ca8aef4`).

## Design patterns worth retaining

- **Artifact-first stage boundaries.** Execution, analysis, reporting, and publication can fail or rerun independently.
- **Plan/executor split.** Plan builders describe work; the sequencer owns lifecycle; tool adapters execute concrete steps.
- **Failure-isolated observers.** Notifications and telemetry do not become benchmark control authority.
- **Durable state separate from logs.** Run and profile status is queryable without scraping terminal output.
- **Ports-and-adapters UI.** The UI consumes the CLI contract rather than duplicating qualification logic.
- **Historical plus live streaming.** SSE replay closes the late-subscriber race for completed and active runs.

## Candidate matrix

| Candidate | Source | Axis | Fak witness and verdict | Portfolio |
|---|---|---|---|---|
| Replayable workload profile plus deterministic candidate selection | `aiops/profiles/workload.py:1-120@ca8aef4`; plan/step flow above | Replay determinism and offline selection | `internal/computetune/tune.go:22-176` already defines typed profiles, validates candidates before timing, preserves failures, selects by median, and emits a manifest digest. **PRESENT-on-axis; no issue.** | DEFAULT in existing computetune |
| Artifact-first run manifest and resumable profile ledger | `aiops/runner/run_tracker.py:53-161@ca8aef4`; `sequencer.py:209-593` | Inspectable lifecycle state across stages | Fak already has pervasive typed receipts, manifests, ledgers, and replay artifacts; the target uses a product-specific file/SQLite projection rather than a stronger general mechanism. **PRESENT-on-axis; no issue.** | Existing surfaces |
| Observer lifecycle isolated from the benchmark controller | `aiops/runner/observers.py:31-73@ca8aef4`; `sequencer.py:36-45` | Optional side effects cannot abort primary work | Fak's journal, telemetry, and hooks already isolate reporting from core execution in multiple leaves. AI-Ops adds no stronger typed contract. **PRESENT-on-axis; no issue.** | Existing surfaces |
| Derive explicit SSD qualification requirements from measured workload behavior | `aiops/analysis/ssd_spec.py:92-234@ca8aef4`; `bottleneck.py:152-354`; `weight_matrix.py:32-193` | Convert observations into a resource purchasing/placement envelope, not merely a winning implementation | `internal/computetune/tune.go:22-176` selects the best candidate for a declared compute profile and records provenance, but does not derive minimum storage latency/IOPS/bandwidth/endurance/capacity requirements from observed traces. **PARTIAL-on-axis.** | OPTIONAL-MODULE / research issue, never a silent core default |
| Go UI as a CLI/artifact adapter | `ui/internal/runner/runner.go:247-261,469-480@ca8aef4` | Avoid duplicated business logic across UI and engine | Fak's CLI-first architecture and external adapters already follow this separation. **PRESENT-on-axis; no issue.** | Existing architecture |
| Live SSE with history replay | `ui/internal/api/handler_runs_sse.go:24-118,186-197@ca8aef4` | Late subscribers receive prior events then continue live | Relevant to UI productization, but not central to fak's current kernel or operator surfaces, and no uncovered required seam was established in this study. **WATCH.** | Revisit with a concrete browser UI requirement |

## Borrow to file

### Storage qualification envelope from captured inference traces

- Technique: derive a typed minimum storage envelopeâ€”latency, random/sequential throughput, endurance, and capacityâ€”from a captured native-inference workload trace, with provenance back to measurements and declared quality constraints.
- Source anchor: `aiops/analysis/ssd_spec.py:92-234@ca8aef4` plus bottleneck and weighting stages above.
- Fak seam: `internal/computetune/tune.go:22-176` has profile/candidate selection and receipts but stops at implementation choice; storage-backed inference work appears across paging, remote-memory, and storage-to-GPU issues without this reusable derivation contract.
- Their-worldview reason: AI-Ops serves qualification engineers who must translate workload behavior into a drive-selection decision; a benchmark score alone is insufficient.
- Fak opportunity: an optional analysis leaf that consumes an already-witnessed native-fak trace and emits a quality-constrained storage envelope. It must not invent requirements from synthetic or simulated evidence and must identify the fak-native engine in its receipt.
- Disconfirming check: prove an existing fak artifact already emits all five minimum requirements with trace provenance and operating-envelope constraints; if so, close as PRESENT-on-axis.
- License route: INSPIRE/independent implementation only unless the source repository gains an explicit compatible license.
- Filed follow-on: #10267 (`research(compute): derive a storage qualification envelope from witnessed native traces`).

## Risks and negative knowledge

1. **No explicit source license.** Direct porting is excluded.
2. **Release-manifest drift.** The only upstream-only commit repairs `release.py` contents by adding `install.py`, the JSON config, and the full UI directory. Manually duplicated release inventories are a demonstrated maintenance risk.
3. **Cross-language contract weight.** The Go UI depends on executable discovery, CLI arguments, temporary config schema, stdout/stderr behavior, and artifact paths.
4. **Central coordination files.** `aiops/cli.py` and `RunPlan.execute` carry many cross-cutting responsibilities.
5. **Observer errors are debug-only.** Isolation protects runs but can hide failed notifications or catalog updates.
6. **Configuration migration remains in the primary path.** JSON and deprecated YAML handling coexist.
7. **External execution coupling.** Paramiko, Bench, FIO, subprocesses, and remote hosts widen the operational failure envelope.
8. **Go-side test asymmetry.** The tracked UI has much less test coverage than the Python package; only the configuration package exposed a tracked Go test in this read.
9. **Local test run was not green.** A full `python -m pytest -q` was attempted twice on the Windows checkout. It progressed beyond 50% with multiple failures and generated `.fak/negframe/journal.jsonl`; output capture was interrupted before a trustworthy final count. This is evidence of a non-clean local run, not a precise failure inventory. The static architecture findings do not depend on a green suite.

## Worldview and divergence

AI-Ops treats hardware qualification as a first-class decision product: the user wants to know which storage device is adequate and why. Fak primarily treats performance evidence as a way to choose and prove an execution strategy. Those goals overlap at workload capture but diverge at the decision output. The worthwhile adaptation is therefore an optional resource-envelope analyzer, not importing AI-Ops's orchestration stack or making storage qualification a core kernel responsibility.

## Completeness critic

Material surfaces opened: all Python package directories; root docs and plans; package/release configuration; tests; Go UI source; Git history/releases; GitHub issue and PR state; source-license/provenance files. No material source subsystem remains intentionally unopened. Generated archives, caches, `.venv`, historical `legacy-test-examples`, and completed run outputs were not treated as source of architectural truth because maintained source and tests cover their producers/consumers. The latest source delta was inspected directly from `origin/master` without mutating the external checkout.

## Refresh triggers

Re-run this study if AI-Ops adds a license, changes the analysis formula or artifact schema, gains active issues/PR discussion, or ships a revision after `ca8aef4` that changes runtime architecture. Re-witness the fak axis before implementing any filed borrow.
