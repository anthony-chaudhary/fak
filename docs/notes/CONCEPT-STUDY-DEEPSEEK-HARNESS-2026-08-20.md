# DeepSeek Harness study: reversible composition, durable turns, and the corrected borrowing backlog

**Observed:** 2026-08-20  
**Primary source:** [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)  
**Pinned revision:** `dsh-v0.1.0-rc.8` / `141eb6fef83422698aef7a981029e843e8161534` (2026-08-19)  
**License:** MIT at the repository root. The pinned tree also vendors locally modified MIT components with their own notices. This study copies zero source bytes; every prospective transfer is **INSPIRE** (TypeScript design into existing Go seams).

## Verdict

DeepSeek Harness is a young but unusually coherent plugin-first agent product. A Cordis context owns typed services, events, and reversible effects; ordered profile layers assemble the product; every model-visible change enters the durable session log; tool calls pass through one lossless execution pipeline; and live preset edits create standing generations rather than mutating sessions already in flight.

Those are good mechanisms. The first disposition pass undercounted them: it treated thematic overlap with an epic as if the epic's done condition contracted the exact borrowed behavior. A 2026-08-20 leaf-grain readback corrected that mistake. **Thirteen net-new FAK leaves survive**: the original observed-version write ticket [#8222](https://github.com/anthony-chaudhary/fak/issues/8222), plus twelve independently dispatchable loop, ledger, scheduler, compaction, and harness-lifecycle tickets:

- input/model truth: [#8260](https://github.com/anthony-chaudhary/fak/issues/8260) claims admitted input before assembly, [#8262](https://github.com/anthony-chaudhary/fak/issues/8262) reconstructs the exact model-visible request, [#8263](https://github.com/anthony-chaudhary/fak/issues/8263) persists interrupted assistant prefixes, and [#8264](https://github.com/anthony-chaudhary/fak/issues/8264) wires the already-shipped scheduler classes into the live native loop;
- tool scheduling/recovery: [#8266](https://github.com/anthony-chaudhary/fak/issues/8266) overlaps effect-safe bodies while committing in model order, [#8267](https://github.com/anthony-chaudhary/fak/issues/8267) closes cancellation over started and queued calls, and [#8268](https://github.com/anthony-chaudhary/fak/issues/8268) repairs incomplete calls as completed, never-started, or outcome-unknown;
- compaction truth: [#8265](https://github.com/anthony-chaudhary/fak/issues/8265) requires every replacement to cite the exact event set it shadows;
- harness lifecycle: [#8269](https://github.com/anthony-chaudhary/fak/issues/8269) adds owner-bound reversible scopes, [#8270](https://github.com/anthony-chaudhary/fak/issues/8270) reclaims only unpinned superseded generations, [#8271](https://github.com/anthony-chaudhary/fak/issues/8271) audits mounted extension readiness, and [#8272](https://github.com/anthony-chaudhary/fak/issues/8272) transactionally swaps mounted extension graphs.

Existing tickets remain the right route where their actual contracts cover the axis: session-generation persistence/checkpoints ([#6343](https://github.com/anthony-chaudhary/fak/issues/6343), [#6438](https://github.com/anthony-chaudhary/fak/issues/6438)); content-addressed skills/assets ([#6796](https://github.com/anthony-chaudhary/fak/issues/6796), [#6807](https://github.com/anthony-chaudhary/fak/issues/6807), [#7230](https://github.com/anthony-chaudhary/fak/issues/7230)); product/install rollback ([#7217](https://github.com/anthony-chaudhary/fak/issues/7217), [#7218](https://github.com/anthony-chaudhary/fak/issues/7218)); descendant containment ([#2354](https://github.com/anthony-chaudhary/fak/issues/2354)); and subagent runtime admission ([#3266](https://github.com/anthony-chaudhary/fak/issues/3266)). Dynamic browser modules, a redacted live settings editor, and model-authored in-process plugins remain outside FAK's current product/security boundary.

## For / Problem / Today / Better because / Witness

- **For:** operators running long-lived agent sessions while configuration, tools, providers, or harness assets change.
- **Problem:** a session can silently become a hybrid of old and new composition, or its durable transcript can cease to describe what the model actually saw.
- **Today:** FAK has stronger policy, cache, tool-snapshot, compact-pair, and session-generation primitives, but its native file mutations carry paths without observed target/version tokens and some session primitives are not yet dogfooded or persisted end to end.
- **Better because:** DeepSeek Harness supplies concrete failure reports and regression shapes, while leaf-grain routing turns each uncovered invariant into a checkable FAK contract without copying its dynamic product architecture.
- **Witness:** every candidate below has a pinned upstream `path:line@revision`, a FAK self-query result, a disposition, and an issue or explicit revisit trigger; the thirteen new leaves are #8222, #8260, and #8262-#8272.

Problem centrality is **Enabling**, with the individual owned-loop, mutation, compaction, and extension-lifecycle leaves **Core** at their seams. P1 managed context is directly served; P2 net-true efficiency is validated by bounded receipts and measured scheduler/GC work; P3 bounded adaptation is served by claimed inputs, standing generations, observed-version writes, and owner scopes; P4 integrated operations is served by fail-loud activation and durable readback. The next-best alternative was either a feature checklist or an umbrella-only routing pass. The corrected comparison keeps product-boundary exclusions while filing each uncovered behavior at dispatchable grain.

## What was studied

The full clone contained 7,807 files and 12,940 commits reachable from the pinned revision. The `rc.7` to `rc.8` range alone touched 1,604 files (`54,064` insertions, `10,533` deletions), so the revision pin is load-bearing.

| Source class | Coverage |
|---|---|
| Architecture and lifecycle | `docs/architecture.md`, `docs/agent-lifecycle.md`, `docs/tool-execution-pipeline.md`, subsystem and defensive-pattern docs |
| Runtime source | boot/profile composition, agent loop/inbox, ordered tool scheduling, context projection, compaction, session repair/persistence, agent presets, settings, client modules, Cordis/Loader/Include vendor adaptations |
| Tests | transactional config reload, preset generations, plugin HMR disposal, session/backend snapshots, browser composition snapshots |
| History and releases | full reachable history, `rc.7`→`rc.8` delta, both public release tags, recent path history |
| Agent Notes | implemented, proposed, rejected, and frozen notes; to-do and known-limitation sweeps |
| Field feedback | GitHub metadata and 3,616 Discussions; public Issues are disabled and the public PR count is zero |
| Provenance | root MIT license, `vendor/README.md`, pinned vendor revisions/local-change ledger, no submodules |

Public review history is the main blind spot. Merge subjects preserve internal PR numbers, but their review threads are not public. Discussion reports are user evidence, not proof that a fix has landed. This was a static source study: no package installation or end-to-end launch was attempted, so operational conclusions come from committed tests, docs, and issue reproductions rather than a locally executed DSH stack.

## Architecture map

### Cordis is the composition kernel

Services, event handlers, and registrations are effects owned by the plugin that created them and unwind when that plugin is disposed (`docs/architecture.md:11-13@141eb6fe`). A profile is an ordinary plugin package plus ordered bundles; profile, home, and CLI patches then layer over that base (`docs/architecture.md:19-27@141eb6fe`, `packages/boot/app-boot/src/profile.ts:5-21@141eb6fe`). Boot settles the tree and audits enabled rows that failed to activate instead of declaring readiness around silent `PENDING` plugins (`packages/boot/app-boot/src/index.ts:651-801@141eb6fe`).

Cordis itself is vendored and locally hardened. The pinned change ledger names reentrant disposal, transactional loader/include updates, exact HMR watching, serialized configuration mutation, atomic writes, lazy configuration resolution, and rescoping (`vendor/README.md:3-23@141eb6fe`). That provenance makes direct copying possible under MIT with notices, but FAK has no need to import a TypeScript lifecycle runtime.

### Session truth is the model-visible boundary

The architecture states that model-visible changes must be logged (`docs/architecture.md:96@141eb6fe`). Tool identity and call arguments are logged before execution; arguments are normalized; the final `tool/result` stores a single lossless model-facing JSON snapshot (`docs/tool-execution-pipeline.md:6-26,60@141eb6fe`). Assistant messages retain `sourceEventSeqs`, and compaction can trigger a retry generation instead of rewriting the prior turn (`docs/agent-lifecycle.md:74-76@141eb6fe`).

This is the repository's most transferable invariant. It is also where public defect reports are most instructive: [Discussion #3591](https://github.com/deepseek-ai/deepseek-harness/discussions/3591) reports a dangling `tool/call` after parallel scheduler failure, while [#3615](https://github.com/deepseek-ai/deepseek-harness/discussions/3615) reports stale agent-scoped tools after recomposing a blank session. The architecture is sound; the reports show why the invariant needs repair and cross-layer regression witnesses.

### The loop separates execution concurrency from observation order

The agent claims its target inbox before asynchronous prompt assembly, so messages arriving later cannot leak into the step already being built (`packages/core/agent-loop/src/agent.ts:210-240@141eb6fe`). Tool bodies marked parallel may overlap, but preparation, results, extra context, post-hooks, and finalizers become durable only across contiguous model-order slots; cancellation drains started calls and writes typed never-started results for the rest (`packages/core/agent-loop/src/tool-calls.ts:25-47,137-256@141eb6fe`). Crash repair uses the durable `tool/call` event as the dividing line: an unclosed call with that event has unknown outcome and must not be blindly retried, while a call never recorded as started is retryable (`packages/core/session/src/repair.ts:82-122@141eb6fe`).

FAK's owned loop already emits typed turn/call/result progress and executes a model turn's tool calls serially in model order (`internal/agent/loop.go:595-722`, `internal/agent/loop_observe.go:23-67`). That is correctness-preserving but leaves safe overlap on the table. [#5796](https://github.com/anthony-chaudhary/fak/issues/5796) owns tool-call width metrics, not execution eligibility, exclusive barriers, ordered commit, or cancellation closure; those exact leaves are now [#8266](https://github.com/anthony-chaudhary/fak/issues/8266) and [#8267](https://github.com/anthony-chaudhary/fak/issues/8267). The admitted-transcript and session-ledger epics likewise did not contract claim-before-assembly or per-call ambiguous-outcome repair; [#8260](https://github.com/anthony-chaudhary/fak/issues/8260), [#8264](https://github.com/anthony-chaudhary/fak/issues/8264), and [#8268](https://github.com/anthony-chaudhary/fak/issues/8268) now do. [#8263](https://github.com/anthony-chaudhary/fak/issues/8263) separately preserves a non-empty interrupted assistant prefix that the native loop currently drops on completion error.

DeepSeek Harness also refuses provenance-free surface replacement: a replacing event must cite every shadowed event sequence (`packages/core/session/src/surface.ts:210-247@141eb6fe`). FAK's [#3025](https://github.com/anthony-chaudhary/fak/issues/3025)-[#3027](https://github.com/anthony-chaudhary/fak/issues/3027) cluster covers retained spans, source cards, and fidelity probes, but none requires exact equality between the dropped event set and replacement citations; [#8265](https://github.com/anthony-chaudhary/fak/issues/8265) owns that missing invariant. DeepSeek Harness's derived projection cache also flushes the source log before writing a cache row, so the cache may lag durable truth but never run ahead of it (`packages/session/session-projection-cache/src/index.ts:104-198@141eb6fe`). A generic projection-cache layer remains premature until FAK has a real cached projection consumer.

### Live edits advance generations, not active sessions

Agent presets mount immutable standing generations. New sessions join the current generation; a blank session may recompose; a session that has produced output may not switch because that would strand tools already called (`packages/preset/agent-presets/README.md:145-150@141eb6fe`). Replacement configuration is applied transactionally: failed imports, applies, or in-place updates restore the previous entry/tree (`vendor/loader/src/config/entry.ts:141-242@141eb6fe`, `packages/boot/app-boot/tests/config-reload.spec.ts:62-175@141eb6fe`).

The current implementation also documents the debt honestly: a generation stamp covers the composition file but not adjacent skills/assets, and superseded generations are never reclaimed (`packages/preset/agent-presets/README.md:149-150@141eb6fe`). FAK's content-addressed asset closure is already contracted by [#6796](https://github.com/anthony-chaudhary/fak/issues/6796), [#6807](https://github.com/anthony-chaudhary/fak/issues/6807), and [#7230](https://github.com/anthony-chaudhary/fak/issues/7230). Reclamation is not: `pkg/managedharness` appends generation records without pin/refcount/GC, so [#8270](https://github.com/anthony-chaudhary/fak/issues/8270) now owns current/last-good/session/checkpoint roots and bounded collection. Product-install rollback is covered by #7217/#7218, while mounted extension effects still need owner scopes and transactional graph replacement ([#8269](https://github.com/anthony-chaudhary/fak/issues/8269), [#8272](https://github.com/anthony-chaudhary/fak/issues/8272)).

### Settings distinguish raw, effective, and safe-to-edit state

Settings descriptors expose a monotonic raw-document revision; path mutations let a redacted client edit one field without reconstructing and deleting hidden secrets; the revision check occurs at the front of the serialized write queue (`packages/settings/settings/src/index.ts:75-78,194-202,622-626@141eb6fe`). Raw-document changes and resolved-value changes are separate events (`packages/settings/settings/src/index.ts:713-728@141eb6fe`).

The wire safety boundary is incomplete: secret roles reachable only through union/intersection/transform schemas can pass through redaction, and serialized schema defaults can expose secret defaults (`packages/settings/settings/README.md:41-45@141eb6fe`). FAK should not borrow this surface until it has a live redacted settings editor; if that product arrives, fail-closed schema description is a prerequisite rather than a follow-up.

### Client composition is content-addressed but product-specific

Client modules form a content-hashed dependency graph, activate in topological order, incrementally reconcile changed content, fail loudly on initial activation, and contain steady-state listener failures (`docs/subsystems/client-modules.md:42-61@141eb6fe`). Generated Typert routes bind Host→Client methods to runtime codecs and explicit scopes/allowlists. This is strong web-product engineering, but it is not a kernel primitive for FAK's one Go binary.

### Self-modification is explicitly shell-equivalent

The `tool-cordis` extension lets a model define host and browser halves against a live capability catalog, then run/stop/undefine them through ordinary plugin cleanup. Its own trust statement says the VM is not a security boundary and should be treated like bash access; asynchronous code can escape the evaluation timeout (`packages/extensions/tool-cordis/README.md:21-27,100-104@141eb6fe`). FAK should keep the useful ideas—live capability catalogs and revision-pinned activation—and exclude in-process model-authored plugins from the default capability floor.

## Candidate ledger

FAK was queried three ways before disposition: `fak capabilities` for product capabilities, `fak dev index docs|leaves|verbs|claims` for named surfaces, and raw `rg` over implementation/tests/issues. `PRESENT`, `PARTIAL`, `ABSENT`, and `DIVERGENT` below refer to that three-way witness, not keyword similarity.

| Candidate | Pinned DeepSeek Harness evidence | FAK evidence and axis | Verdict | Portfolio / action |
|---|---|---|---|---|
| Model-visible iff logged | `docs/architecture.md:96`; `packages/core/agent-loop/src/agent.ts:426-513@141eb6fe` | `internal/promptaudit` records exact segment provenance/digests but has no production caller; the ledger stores request shape/digest, not an exact reconstructable model-boundary receipt | **PARTIAL** | **DEFAULT / INSPIRE** through [#8262](https://github.com/anthony-chaudhary/fak/issues/8262); #7736 remains promptaudit dogfood, not this ledger leaf |
| Immutable session composition generation | `packages/preset/agent-presets/README.md:145-150@141eb6fe` | epoch/checkpoint persistence remains [#6343](https://github.com/anthony-chaudhary/fak/issues/6343)/[#6438](https://github.com/anthony-chaudhary/fak/issues/6438); recursive skill/asset identity is #6796/#6807/#7230 | **PARTIAL** | **DEFAULT** through those existing exact contracts; use #3615 as the hybrid-generation failure witness |
| Transactional last-good replacement | `vendor/loader/src/config/entry.ts:141-246`; `vendor/loader/src/config/group.ts:59-106@141eb6fe` | product/install activation and rollback are #7217/#7218; `pkg/managedharness` preflights before pointer mutation and has no mounted-effect rollback | **PRESENT product / ABSENT extension graph** | **DEFAULT / INSPIRE** for the uncovered mounted-graph axis through [#8272](https://github.com/anthony-chaudhary/fak/issues/8272) |
| Owner-bound reversible plugin effects | `docs/architecture.md:11-13`; `vendor/cordis/src/fiber.ts:222-560@141eb6fe` | `pkg/harnesskit` declares lifecycle/ownership but exposes no operational owner scope, reverse unwind, or reentrant close | **PARTIAL** | **DEFAULT / INSPIRE** through [#8269](https://github.com/anthony-chaudhary/fak/issues/8269); linked factories and sidecars need cleanup without Go dynamic loading |
| Layered product profiles | `docs/architecture.md:19-27`; `profile.ts:5-21@141eb6fe` | `internal/harnessprofile` and the universal harness/profile notes already resolve layered harness assets | **PRESENT** | **DEFAULT**; retain explicit layer order and receipts |
| Lossless tool-call/result boundary | `docs/tool-execution-pipeline.md:6-26,60@141eb6fe` | `internal/toolcatalog/toolcatalog.go:196-231,481-510` validates request snapshots; `internal/agent/anthropic_compact.go` preserves tool-use/result pairs under compaction | **PRESENT** | **DEFAULT**; DSH #3591 is negative evidence for existing tests |
| Claim input before asynchronous assembly | `packages/core/agent-loop/src/agent.ts:210-240@141eb6fe` | the native loop destructively drains steer before asynchronous render/send with no durable claimed-set→request receipt | **PARTIAL** | **DEFAULT / INSPIRE** through [#8260](https://github.com/anthony-chaudhary/fak/issues/8260) |
| Consume scheduler-classified live input | same claim boundary plus typed inbox events | #2402 shipped `now`/`next`/non-querying classes, but its `DrainQueryingTurn`/`TakeInterrupts` consumers have no production caller | **PARTIAL** | **DEFAULT** through [#8264](https://github.com/anthony-chaudhary/fak/issues/8264); wire the shipped vocabulary, do not invent another bus |
| Preserve interrupted assistant prefix | `packages/core/agent-loop/src/agent.ts:332-377@141eb6fe` | native streaming returns on completion error before appending the assistant; a non-empty visible prefix is absent from durable session truth | **ABSENT** | **DEFAULT / INSPIRE** through [#8263](https://github.com/anthony-chaudhary/fak/issues/8263) |
| Overlap safe tool bodies, commit in model order | `packages/core/agent-loop/src/tool-calls.ts:25-47,137-256@141eb6fe` | `internal/agent/loop.go:595-722` executes serially; #5796 owns width metrics but not scheduler eligibility/order | **PARTIAL** | **DEFAULT / INSPIRE** through [#8266](https://github.com/anthony-chaudhary/fak/issues/8266), with [#8267](https://github.com/anthony-chaudhary/fak/issues/8267) closing started/queued cancellation |
| Repair started calls as outcome-unknown | `packages/core/session/src/repair.ts:82-122@141eb6fe` | FAK emits in-memory `tool_started` progress but has no durable per-call start/result repair boundary | **PARTIAL** | **DEFAULT / INSPIRE** through [#8268](https://github.com/anthony-chaudhary/fak/issues/8268); never infer safe retry from a missing result |
| Provenance-preserving surface replacement | `packages/core/session/src/surface.ts:210-247`; `packages/compaction/compaction/src/types.ts:92-119@141eb6fe` | #3025-#3027 cover retained spans/cards/probes but not exact equality between dropped events and replacement citations | **PARTIAL** | **DEFAULT / INSPIRE** through [#8265](https://github.com/anthony-chaudhary/fak/issues/8265) |
| Projection cache never ahead of durable log | `packages/session/session-projection-cache/src/index.ts:104-198@141eb6fe` | FAK has the content-addressed session-ledger program (#2392) but no equivalent generic cached projection plane needing this barrier | **ABSENT outside current seam** | **WATCH**; borrow the durability-before-cache invariant if a real session projection cache lands |
| Opaque observed file target/version | `packages/fs/fs/src/types.ts:11-33,117-188`; `packages/fs/fs-local/src/index.ts:74-103,166-254@141eb6fe` | `internal/codetools/confine.go:46-96` re-confines paths and `mutation.go:42-147` atomically replaces them, but `args.go:58-88` carries no observed version and the check/mutate window has no per-target serialization | **ABSENT** | **DEFAULT / INSPIRE** through new [#8222](https://github.com/anthony-chaudhary/fak/issues/8222); stale or identity-mismatched edits must refuse without changing disk |
| Reclaim superseded composition generations | `packages/preset/agent-presets/README.md:145-154@141eb6fe` | `pkg/managedharness` retains every generation record/artifact; #7216/#7218 name rollback roots but no reclaim rule | **ABSENT** | **DEFAULT / INSPIRE** through [#8270](https://github.com/anthony-chaudhary/fak/issues/8270) |
| Fail-loud dependency/readiness audit | `packages/boot/app-boot/src/index.ts:651-801@141eb6fe`; [#3600](https://github.com/deepseek-ai/deepseek-harness/discussions/3600) | #6792 covers static resolution and #4203 host install hints; no post-mount registry proves every enabled extension reached `RUNNING` | **PRESENT static / ABSENT runtime** | **DEFAULT / INSPIRE** for mounted readiness through [#8271](https://github.com/anthony-chaudhary/fak/issues/8271) |
| Descendant cannot kill the supervisor | [Discussion #387](https://github.com/deepseek-ai/deepseek-harness/discussions/387) | process-isolation contract [#2171](https://github.com/anthony-chaudhary/fak/issues/2171); descendant containment [#2354](https://github.com/anthony-chaudhary/fak/issues/2354) | **PARTIAL** | **DEFAULT** through the existing security epic; no duplicate |
| Process-tree completion means observed quiescence | `packages/subprocess/subprocess/src/types.ts:158-193`; `subprocess-local/src/index.ts:79-101@141eb6fe` | `internal/procguard`, `internal/toolprocgate` descendant-state/leak events, and `internal/microagent/toolexec_test.go:122-163` kill and independently poll the whole tree | **PRESENT** | **DEFAULT** existing process spine; no new issue |
| Publish a child only after capability acceptance | `packages/subagent/subagent/src/index.ts:420-481`; `continuation.ts:226-243,355-466@141eb6fe` | inherited capability contract [#2355](https://github.com/anthony-chaudhary/fak/issues/2355)/[#2358](https://github.com/anthony-chaudhary/fak/issues/2358); runtime packaging remains [#3266](https://github.com/anthony-chaudhary/fak/issues/3266) | **PARTIAL** | **DEFAULT** through the existing runtime issue; no duplicate |
| Poison shared state after ambiguous cancellation | `packages/lsp/lsp-stdio/src/instance.ts:93-150,192-239@141eb6fe` | FAK exposes no shared stateful LSP instance; its provider calls are request-scoped HTTP and logical-session reset is separately typed | **DIVERGENT** | **WATCH** if FAK adds a pooled stateful protocol client; no generic mechanism now |
| Functional sandbox enforcement probe | `native/landlock-run/packages/entry/src/main.c:264-297`; `sandbox-local/tests/landlock.e2e.ts:55-91@141eb6fe` | `internal/guard/landlock_acceptance_linux_test.go:13-205` executes real deny/allow writes; the hook-floor is deliberately opt-in and fail-open as defense in depth | **PRESENT/DIVERGENT** | Keep the real probe; do not silently relabel FAK's narrower fail-open promise as DSH's fail-closed sandbox |
| Cache/route identity must remain fresh and visible | [Discussion #3565](https://github.com/deepseek-ai/deepseek-harness/discussions/3565) reports a stale route and zero cache reads | FAK's core gateway exposes provider/account/model placement and cache lineage; cache reuse is a primary measured capability | **PRESENT/AHEAD** | **DEFAULT**; retain as field evidence for route-disclosure and freshness regressions |
| CAS/path mutation over redacted settings | `settings/src/index.ts:75-78,194-202,622-626@141eb6fe` | no FAK live redacted settings editor was found; policy/config mutations use different guarded surfaces | **ABSENT outside product boundary** | **EXCLUDE** now; **WATCH** only if a multi-client live editor lands |
| Content-hashed web-client module graph | `docs/subsystems/client-modules.md:42-61@141eb6fe` | no dynamic web-client composition plane; FAK intentionally ships one Go binary | **DIVERGENT** | **EXCLUDE** kernel; an external UI may use the recipe independently |
| Model-authored in-process plugins | `tool-cordis/README.md:21-27,100-104@141eb6fe` | default-deny structural policy and typed tool programs treat model output as untrusted | **DIVERGENT** | **EXCLUDE** as a security boundary; keep capability-catalog ideas only |
| Product-visible real-composition snapshots | `docs/testing.md:9-47@141eb6fe` | AGENTS proof-by-default requires captured behavior/render witnesses; project tests already exercise composed surfaces | **PRESENT** | **RECIPE**: when a plugin affects a product surface, test the real composition, disposal, and emitted artifact |

## Field failures that sharpen existing FAK work

- [#3591](https://github.com/deepseek-ai/deepseek-harness/discussions/3591): a parallel tool scheduler can leave a durable call without its result, making the transcript provider-invalid. This validates FAK's pair-preserving compaction and content-addressed event-log work.
- [#3615](https://github.com/deepseek-ai/deepseek-harness/discussions/3615): recomposing a blank session can retain old agent-scoped tools and restrictions. This is the exact hybrid-generation failure that #6987/#6343 should disconfirm.
- [#3565](https://github.com/deepseek-ai/deepseek-harness/discussions/3565): manual compaction/resume can reuse a six-day-old provider route, report zero cache-read tokens, and create a large billed-token surprise. This validates route freshness and disclosure as part of cache correctness.
- [#3600](https://github.com/deepseek-ai/deepseek-harness/discussions/3600): plugin installation can exit successfully after optional native platform packages fail to download, deferring failure to first delegation. FAK's dependency preflight must keep success tied to the capability actually becoming runnable.
- [#441](https://github.com/deepseek-ai/deepseek-harness/discussions/441): boot rewrites the profile file and can race when processes share `DSH_HOME`. Compare-before-write and unique temporary names are the proven repair shape; FAK already avoids making a mutable profile copy part of ordinary boot.
- [#3173](https://github.com/deepseek-ai/deepseek-harness/discussions/3173): operators want an opt-in degraded boot mode, but the proposal retains fail-loud default behavior and visible diagnostics. FAK should keep the same asymmetry for optional integrations versus security-critical capabilities.

## Borrow, watch, and exclude

### Default

- Bind native `Read` receipts and later `Edit`/overwrite effects through an opaque observed target/version token; #8222 owns the smallest Go spine and real-loop witness.
- Claim live input before assembly (#8260/#8264), record the exact model-visible request (#8262), and preserve non-empty interrupted output (#8263).
- Overlap only effect-safe tool bodies, preserve exclusive barriers/model order, and close cancellation structurally (#8266/#8267).
- Treat a durable `tool/call` start without a durable result as outcome-unknown, never as permission to replay a side effect; #8268 owns the per-call repair leaf.
- Require compaction replacements to cite the exact source-event set they shadow; #8265 owns the equality witness.
- Bind extension effects to owner scopes, fail loud on post-mount readiness, swap candidate graphs transactionally, and reclaim only unpinned generations (#8269-#8272).

### Recipe or optional module

- Keep owner-bound effect/disposal scopes independent of dynamic Go loading: linked factories and sidecars use the abstraction, while the single-binary boundary remains intact.
- Use revision-checked path mutations if FAK gains a multi-client redacted settings UI; require fail-closed schema description first.
- Let a separately built web client adopt content-hashed topological modules without making that graph part of the kernel ABI.

### Watch

- Monitor releases after `rc.8` and discussions #3565, #3591, and #3615. A revisit triggers when upstream lands a regression witness or FAK opens a new composition-generation seam not covered by #6987/#6343/#6438.
- Apply the durability-before-cache watermark only when FAK adds a real cached session projection consumer.

### Exclude

- Do not treat a VM around model-authored host code as a security boundary.
- Do not move browser-module composition, live settings authoring, or TypeScript framework lifecycle into the Go kernel.
- Do not copy Cordis/Loader source. The useful transfer is clean-room design into existing Go seams; zero vendored bytes is the lower-cost route.

## Issue and witness ledger

| Result | Count | Disposition |
|---|---:|---|
| New issues filed | **13** | [#8222](https://github.com/anthony-chaudhary/fak/issues/8222), [#8260](https://github.com/anthony-chaudhary/fak/issues/8260), and [#8262](https://github.com/anthony-chaudhary/fak/issues/8262)-[#8272](https://github.com/anthony-chaudhary/fak/issues/8272); #8261 is an unrelated peer ticket |
| Existing open routes retained | **19** | #7736, #2392, #6343, #6438, #2354, #3266, #5796, #2401, #2217, #3025-#3027, #5158, #6777, #6796, #6807, #7216, #7218, #7230 |
| Closed work independently confirmed | **13** | #6987, #4203, #2171, #2355, #2358, #2402, #2416, #2417, #3353, #5921, #6792, #6904, #7217 |
| Product-boundary exclusions | **3 families** | dynamic web-client modules, multi-client redacted settings authoring, and model-authored in-process plugins |
| Source bytes copied | **0** | all transfers are INSPIRE |

The correction pass re-read the live body and done condition of every umbrella ticket used by the first disposition, then independently re-audited the context/loop, execution/recovery, and plugin/composition clusters. Twelve umbrella mappings failed that test and became #8260/#8262-#8272. Routes stayed existing only where the leaf contract actually names the borrowed axis; deliberate product-boundary differences stayed excluded rather than becoming speculative tickets. Every newly filed body carries pinned `path:line@revision` evidence, the current FAK seam, a first failing check, a live/offline dogfood witness, labels, milestone, category/layer, and operating envelopes; all twelve were read back from GitHub after creation.

There is no public upstream issue/PR filing path to use: repository Issues are disabled, public PR history is empty, and upstream asks external contributors to use Discussions. No external message was necessary because this pass found FAK-side routing evidence, not an upstream defect newly discovered here.

## Sources

- [Pinned repository revision](https://github.com/deepseek-ai/deepseek-harness/tree/141eb6fef83422698aef7a981029e843e8161534) and [`dsh-v0.1.0-rc.8`](https://github.com/deepseek-ai/deepseek-harness/releases).
- [Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/architecture.md), [agent lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/agent-lifecycle.md), [tool pipeline](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/tool-execution-pipeline.md), and [testing doctrine](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/testing.md).
- [Agent preset composition](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/preset/agent-presets/README.md), [settings](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/settings/settings/README.md), and [self-modifying Cordis tool trust stance](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/extensions/tool-cordis/README.md).
- [Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions), observed 2026-08-20; exact reports are linked beside each claim above.

## Exhaustive inventory refresh (2026-08-25)

Issue [#8989](https://github.com/anthony-chaudhary/fak/issues/8989) closes the denominator gap without moving the original study cutoff. The machine-readable map is [`docs/research/inventory/deepseek-ai-deepseek-harness.json`](../research/inventory/deepseek-ai-deepseek-harness.json), generated from the detached checkout at commit `141eb6fef83422698aef7a981029e843e8161534` (`dsh-v0.1.0-rc.8`). It walks every regular file outside `.git` and vendored dependency trees and accounts for README/docs, architecture/design, runtime, tests/fixtures, changelog/history/releases, roadmap/to-do evidence, and license/provenance.

The non-tree audit used the same commit timestamp as its cutoff. GitHub reports Issues and Pull Requests disabled for this repository, so those two classes are checked absent rather than inferred from an empty REST response. Discussions were paged to exhaustion and filtered by `createdAt`: 3,310 existed at the cutoff, comprising 3,226 open and 84 closed. The local history contains 12,940 commits reachable from the pin, two merged tags, and the pinned `dsh-v0.1.0-rc.8` release. Commands, counts, cutoff, and source URLs are recorded in the map's `non_tree_study` object and registry `source_evidence`.

### Candidate-to-FAK decision matrix

The exhaustive read-back did not add a fourteenth candidate. It did correct the old one-gap summary: thirteen leaf-grain candidates were filed, and the current disposition is independently readable from GitHub.

| Borrowed pattern | FAK decision | Follow-on |
|---|---|---|
| Observed-version mutation guard | Implemented | #8222 closed |
| Claim admitted input before prompt assembly | Implemented | #8260 closed |
| Exact model-visible request reconstruction | Implemented | #8262 closed |
| Persist interrupted assistant prefixes | Implemented | #8263 closed |
| Scheduler-classified steer input | Retain as unfinished product gap | #8264 open |
| Compaction replacement provenance | Retain as unfinished trust gap | #8265 open |
| Effect-safe tool overlap with ordered commit | Implemented | #8266 closed |
| Ordered skipped results after cancellation | Implemented | #8267 closed |
| Durable incomplete-tool repair | Implemented | #8268 closed |
| Owner-scoped extension effects | Implemented | #8269 closed |
| Superseded generation reclamation | Implemented | #8270 closed |
| Mounted-extension readiness audit | Implemented | #8271 closed |
| Transactional extension-graph swap | Implemented | #8272 closed |

Candidate-specific `fak capabilities` self-queries are pinned in the map. They preserve two product-boundary decisions from the original study: FAK keeps its static-binary and default-deny architecture rather than copying a general runtime plugin marketplace, and it borrows lifecycle/effect semantics only where they strengthen native FAK ownership. No new issue is warranted beyond the two already-open follow-ons, #8264 and #8265.

### Completeness critic

The critic found no unaccounted source class: local-tree classes have path evidence or checked-absent results; history, release, tag, and discussion evidence is pinned to the same revision timestamp; the forge compound class explicitly accounts for issues, pull requests, and discussions; self-query, candidate decisions, and issue tracking are traceable rather than asserted. The row-specific acceptance witness is `fak study-monitor --inventory-check --json`, whose `deepseek-ai/deepseek-harness` result must report `ready: true` at indexed revision `141eb6fef83422698aef7a981029e843e8161534`.
