# DeepSeek Harness study: reversible composition, durable turns, and one guarded-filesystem gap

**Observed:** 2026-08-20  
**Primary source:** [`deepseek-ai/deepseek-harness`](https://github.com/deepseek-ai/deepseek-harness)  
**Pinned revision:** `dsh-v0.1.0-rc.8` / `141eb6fef83422698aef7a981029e843e8161534` (2026-08-19)  
**License:** MIT at the repository root. The pinned tree also vendors locally modified MIT components with their own notices. This study copies zero source bytes; every prospective transfer is **INSPIRE** (TypeScript design into existing Go seams).

## Verdict

DeepSeek Harness is a young but unusually coherent plugin-first agent product. A Cordis context owns typed services, events, and reversible effects; ordered profile layers assemble the product; every model-visible change enters the durable session log; tool calls pass through one lossless execution pipeline; and live preset edits create standing generations rather than mutating sessions already in flight.

Those are good mechanisms. One honest net-new FAK gap survived the comparison: FAK's native coding tools confine and atomically replace paths, but do not bind a mutation to the file identity/version the agent observed. That focused gap is filed as [#8222](https://github.com/anthony-chaudhary/fak/issues/8222). The other highest-value ideas are already present, already tracked, or intentionally outside FAK's product boundary:

- prompt provenance and model-visible receipts already have a tested implementation and an open dogfood gap in [#7736](https://github.com/anthony-chaudhary/fak/issues/7736), while content-addressed session events are [#2392](https://github.com/anthony-chaudhary/fak/issues/2392);
- session-pinned composition generations and restore manifests were scoped in closed epic [#6987](https://github.com/anthony-chaudhary/fak/issues/6987), with persistence work still explicit in [#6343](https://github.com/anthony-chaudhary/fak/issues/6343) and [#6438](https://github.com/anthony-chaudhary/fak/issues/6438);
- host-readiness dependency hints shipped through [#4203](https://github.com/anthony-chaudhary/fak/issues/4203);
- process isolation has a closed contract in [#2171](https://github.com/anthony-chaudhary/fak/issues/2171) and descendant containment remains open in [#2354](https://github.com/anthony-chaudhary/fak/issues/2354);
- owned-loop input, transcript, and crash-recovery work already routes through [#2388](https://github.com/anthony-chaudhary/fak/issues/2388), [#2392](https://github.com/anthony-chaudhary/fak/issues/2392), and [#2217](https://github.com/anthony-chaudhary/fak/issues/2217); compaction provenance is explicit in [#3026](https://github.com/anthony-chaudhary/fak/issues/3026) and [#3027](https://github.com/anthony-chaudhary/fak/issues/3027);
- ordered overlap of effect-safe tool bodies is a measured optimization candidate under existing tool-width epic [#5796](https://github.com/anthony-chaudhary/fak/issues/5796), not a reason to weaken FAK's current serial effect ordering;
- runtime plugin unloading, a dynamic browser-module graph, and model-authored in-process plugins conflict with FAK's static single-binary and default-deny boundaries.

The useful outcome is therefore **one narrow leaf plus a routing result**, not feature-checklist backlog: bind FAK-owned file mutations to observed versions; keep DeepSeek Harness as a monitor for generation/prompt/session failure evidence; use its real-composition test discipline as a recipe; and revisit reversible plugin ownership only if FAK deliberately adopts runtime-loadable capabilities.

## For / Problem / Today / Better because / Witness

- **For:** operators running long-lived agent sessions while configuration, tools, providers, or harness assets change.
- **Problem:** a session can silently become a hybrid of old and new composition, or its durable transcript can cease to describe what the model actually saw.
- **Today:** FAK has stronger policy, cache, tool-snapshot, compact-pair, and session-generation primitives, but its native file mutations carry paths without observed target/version tokens and some session primitives are not yet dogfooded or persisted end to end.
- **Better because:** DeepSeek Harness supplies concrete failure reports and regression shapes for those existing seams without requiring FAK to copy its dynamic product architecture.
- **Witness:** every candidate below has a pinned upstream `path:line@revision`, a FAK self-query result, a disposition, and an issue or explicit revisit trigger; the sole new gap is #8222.

Problem centrality is **Enabling**, with #8222 itself **Core** to the FAK-owned coding seam. P1 managed context is directly served; P2 net-true efficiency is validated by the prompt-cache failures; P3 bounded adaptation is served by standing generations and observed-version writes; P4 integrated operations is served by fail-loud activation and durable receipts. The next-best alternative was to treat the repository as a feature checklist and file duplicates. The witnessed comparison reduced the study to one independently dispatchable gap.

## What was studied

The full clone contained 7,807 files and 12,940 commits reachable from the pinned revision. The `rc.7` to `rc.8` range alone touched 1,604 files (`54,064` insertions, `10,533` deletions), so the revision pin is load-bearing.

| Source class | Coverage |
|---|---|
| Architecture and lifecycle | `docs/architecture.md`, `docs/agent-lifecycle.md`, `docs/tool-execution-pipeline.md`, subsystem and defensive-pattern docs |
| Runtime source | boot/profile composition, agent loop/inbox, ordered tool scheduling, context projection, compaction, session repair/persistence, agent presets, settings, client modules, Cordis/Loader/Include vendor adaptations |
| Tests | transactional config reload, preset generations, plugin HMR disposal, session/backend snapshots, browser composition snapshots |
| History and releases | full reachable history, `rc.7`→`rc.8` delta, both public release tags, recent path history |
| Agent Notes | implemented, proposed, rejected, and frozen notes; TODO and known-limitation sweeps |
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

FAK's owned loop already emits typed turn/call/result progress and executes a model turn's tool calls serially in model order (`internal/agent/loop.go:595-710`, `internal/agent/loop_observe.go:23-67`). That is correctness-preserving but leaves safe overlap on the table. Existing [#5796](https://github.com/anthony-chaudhary/fak/issues/5796) owns tool-call width and the before/after meter; any optimization there should borrow DeepSeek Harness's split—overlap only bodies proven effect-safe, keep exclusive barriers, and commit every model-visible observation in model order. FAK's open admitted-transcript and session-ledger work ([#2401](https://github.com/anthony-chaudhary/fak/issues/2401), [#2392](https://github.com/anthony-chaudhary/fak/issues/2392), [#2217](https://github.com/anthony-chaudhary/fak/issues/2217)) is the existing route for claim-before-assembly and started/unknown-outcome recovery, so no duplicate issue is needed.

DeepSeek Harness also refuses provenance-free surface replacement: a replacing event must cite every shadowed event sequence (`packages/core/session/src/surface.ts:210-247@141eb6fe`). Its derived projection cache flushes the source log before writing a cache row, so the cache may lag durable truth but never run ahead of it (`packages/session/session-projection-cache/src/index.ts:104-198@141eb6fe`). FAK already has open source-card/fidelity work in [#3026](https://github.com/anthony-chaudhary/fak/issues/3026)/[#3027](https://github.com/anthony-chaudhary/fak/issues/3027), and its content-addressed session-ledger program is [#2392](https://github.com/anthony-chaudhary/fak/issues/2392). A generic projection-cache layer would be premature until that durable source exists and a real cached projection justifies it.

### Live edits advance generations, not active sessions

Agent presets mount immutable standing generations. New sessions join the current generation; a blank session may recompose; a session that has produced output may not switch because that would strand tools already called (`packages/preset/agent-presets/README.md:145-150@141eb6fe`). Replacement configuration is applied transactionally: failed imports, applies, or in-place updates restore the previous entry/tree (`vendor/loader/src/config/entry.ts:141-242@141eb6fe`, `packages/boot/app-boot/tests/config-reload.spec.ts:62-175@141eb6fe`).

The current implementation also documents the debt honestly: a generation stamp covers the composition file but not adjacent skills/assets, and superseded generations are never reclaimed (`packages/preset/agent-presets/README.md:149-150@141eb6fe`). Those are reasons to borrow the invariant, not the exact lifecycle implementation.

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
| Model-visible iff logged | `docs/architecture.md:96`; `docs/agent-lifecycle.md:74-76@141eb6fe` | `internal/promptaudit` records exact segment provenance/digests but has no production caller; `internal/toolcatalog/toolcatalog.go:65-94` content-addresses model-visible tool snapshots | **PARTIAL** | **DEFAULT** through existing [#7736](https://github.com/anthony-chaudhary/fak/issues/7736) and [#2392](https://github.com/anthony-chaudhary/fak/issues/2392); no duplicate issue |
| Immutable session composition generation | `packages/preset/agent-presets/README.md:145-150@141eb6fe` | session update design closed in [#6987](https://github.com/anthony-chaudhary/fak/issues/6987); epoch/checkpoint persistence remains [#6343](https://github.com/anthony-chaudhary/fak/issues/6343)/[#6438](https://github.com/anthony-chaudhary/fak/issues/6438) | **PARTIAL** | **DEFAULT**; use #3615 as failure evidence, not a new feature request |
| Transactional last-good replacement | `vendor/loader/src/config/entry.ts:141-242`; `config-reload.spec.ts:62-175@141eb6fe` | `internal/gateway/session_move.go:284-359` journals checkpoint/admit/restore/cutover with rollback; #6987 owns runtime generations | **PRESENT/PARTIAL** | **DEFAULT** existing seams; no new issue |
| Owner-bound reversible plugin effects | `docs/architecture.md:11-13`; `vendor/cordis/src/fiber.ts:222-695@141eb6fe` | `internal/abi/registry.go:75-78,135-197,457-609` is populated at init and production-read-only; `internal/tuiplugin/registry.go:90-175` has no production unregister | **DIVERGENT** | **WATCH** only if runtime plugin loading becomes a product requirement; FAK deliberately excludes Go dynamic loading |
| Layered product profiles | `docs/architecture.md:19-27`; `profile.ts:5-21@141eb6fe` | `internal/harnessprofile` and the universal harness/profile notes already resolve layered harness assets | **PRESENT** | **DEFAULT**; retain explicit layer order and receipts |
| Lossless tool-call/result boundary | `docs/tool-execution-pipeline.md:6-26,60@141eb6fe` | `internal/toolcatalog/toolcatalog.go:196-231,481-510` validates request snapshots; `internal/agent/anthropic_compact.go` preserves tool-use/result pairs under compaction | **PRESENT** | **DEFAULT**; DSH #3591 is negative evidence for existing tests |
| Claim input before asynchronous assembly | `packages/core/agent-loop/src/agent.ts:210-240@141eb6fe` | FAK's typed steer classes shipped in #2402, but the native loop still lacks DSH's one owned inbox/claim transaction; admitted-transcript ownership remains #2401 under #2388 | **PARTIAL** | **DEFAULT** through existing #2401/#2388; require the claimed input set to match the prompt receipt once the owned transcript is complete |
| Overlap safe tool bodies, commit in model order | `packages/core/agent-loop/src/tool-calls.ts:25-47,137-256@141eb6fe` | `internal/agent/loop.go:595-710` executes calls serially in model order; #5796 already owns measured tool-call width | **PARTIAL** | **RECIPE** under existing #5796: overlap only effect-safe bodies, preserve exclusive barriers and ordered commits, and keep only a measured latency win |
| Repair started calls as outcome-unknown | `packages/core/session/src/repair.ts:82-122@141eb6fe` | FAK emits `tool_started` before its dispatch/deny switch and has a write-ahead model-turn retry checkpoint, but no durable per-tool started/result repair boundary; #2392/#2217 own the ledger/kill-safe ladder | **PARTIAL** | **DEFAULT** through existing #2392/#2217; never infer safe retry for a side-effecting call from a missing result |
| Provenance-preserving surface replacement | `packages/core/session/src/surface.ts:210-247@141eb6fe` | FAK preserves tool-use/result pairs, but source-card and post-compaction provenance witnesses remain open in #3026/#3027 | **PARTIAL** | **DEFAULT** through existing #3026/#3027; no duplicate issue |
| Projection cache never ahead of durable log | `packages/session/session-projection-cache/src/index.ts:104-198@141eb6fe` | FAK has the content-addressed session-ledger program (#2392) but no equivalent generic cached projection plane needing this barrier | **ABSENT outside current seam** | **WATCH**; borrow the durability-before-cache invariant if a real session projection cache lands |
| Opaque observed file target/version | `packages/fs/fs/src/types.ts:11-33,117-188`; `packages/fs/fs-local/src/index.ts:74-103,166-254@141eb6fe` | `internal/codetools/confine.go:46-96` re-confines paths and `mutation.go:42-147` atomically replaces them, but `args.go:58-88` carries no observed version and the check/mutate window has no per-target serialization | **ABSENT** | **DEFAULT / INSPIRE** through new [#8222](https://github.com/anthony-chaudhary/fak/issues/8222); stale or identity-mismatched edits must refuse without changing disk |
| Fail-loud dependency/readiness audit | `packages/boot/app-boot/src/index.ts:651-801@141eb6fe`; [#3600](https://github.com/deepseek-ai/deepseek-harness/discussions/3600) | data-driven install hints and shared host-readiness preflight shipped in [#4203](https://github.com/anthony-chaudhary/fak/issues/4203) | **PRESENT** for FAK's static dependency model | **DEFAULT**; dynamic plugin activation audit remains out of scope |
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
- Preserve exact model-visible prompt/tool/session receipts and finish their existing dogfood/persistence issues.
- Preserve session-pinned generations: build a replacement completely, advance one explicit pointer, and let old sessions drain on their original composition.
- Treat a durable `tool/call` start without a durable result as outcome-unknown, never as permission to replay a side effect; #2392/#2217 own the event-log and kill-safe route.
- Keep real-composition snapshot tests for every operator-visible plugin change, including unload/disposal behavior.

### Recipe or optional module

- Use owner-bound effect/disposal scopes inside any future runtime-loadable module layer, but introduce no generic dynamic loader merely to obtain the abstraction.
- Use revision-checked path mutations if FAK gains a multi-client redacted settings UI; require fail-closed schema description first.
- Let a separately built web client adopt content-hashed topological modules without making that graph part of the kernel ABI.

### Watch

- Monitor releases after `rc.8` and discussions #3565, #3591, and #3615. A revisit triggers when upstream lands a regression witness or FAK opens a new composition-generation seam not covered by #6987/#6343/#6438.
- Revisit reversible plugin ownership only if FAK changes its explicit no-dynamic-loading stance.

### Exclude

- Do not treat a VM around model-authored host code as a security boundary.
- Do not move browser-module composition, live settings authoring, or TypeScript framework lifecycle into the Go kernel.
- Do not copy Cordis/Loader source. The useful transfer is clean-room design into existing Go seams; zero vendored bytes is the lower-cost route.

## Issue and witness ledger

| Result | Count | Disposition |
|---|---:|---|
| New issues filed | **1** | [#8222](https://github.com/anthony-chaudhary/fak/issues/8222), observed target/version binding for native coding-tool mutations |
| Existing open issues reinforced | **11** | #7736, #2392, #6343, #6438, #2354, #3266, #5796, #2401, #2217, #3026, #3027 |
| Closed work independently confirmed | **6** | #6987, #4203, #2171, #2355, #2358, #2402 |
| Runtime-plugin candidates parked behind a trigger | **1 family** | revisit only after an explicit dynamic-loading product decision |
| Source bytes copied | **0** | all transfers are INSPIRE |

The parallel architecture reader declared nine findings. Fresh pinned-tree read-back confirmed the five findings used here (reversible effects, transactional replacement, standing generations, settings revision/redaction, and the self-modification trust stance); four unneeded claims were not folded. The execution reader declared ten findings; fresh read-back confirmed the six candidates used for the late countercheck (process-tree quiescence, file identity/version, ambiguous cancellation, child publication, host-owned child cleanup, and functional sandbox probing), while four unneeded claims were not folded. The context/loop reader declared 33 findings; fresh read-back confirmed the five contracts used here (claim-before-assembly, ordered tool commit, outcome-unknown repair, replacement provenance, and projection watermarks), while 28 unneeded claims were not folded. Combined coverage: **confirmed 16 of 52 declared results; refuted 0; unwitnessed 36; no-claim 0**.

There is no public upstream issue/PR filing path to use: repository Issues are disabled, public PR history is empty, and upstream asks external contributors to use Discussions. No external message was necessary because this pass found FAK-side routing evidence, not an upstream defect newly discovered here.

## Sources

- [Pinned repository revision](https://github.com/deepseek-ai/deepseek-harness/tree/141eb6fef83422698aef7a981029e843e8161534) and [`dsh-v0.1.0-rc.8`](https://github.com/deepseek-ai/deepseek-harness/releases/tag/dsh-v0.1.0-rc.8).
- [Architecture](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/architecture.md), [agent lifecycle](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/agent-lifecycle.md), [tool pipeline](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/tool-execution-pipeline.md), and [testing doctrine](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/docs/testing.md).
- [Agent preset composition](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/preset/agent-presets/README.md), [settings](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/settings/settings/README.md), and [self-modifying Cordis tool trust stance](https://github.com/deepseek-ai/deepseek-harness/blob/141eb6fef83422698aef7a981029e843e8161534/packages/extensions/tool-cordis/README.md).
- [Discussions](https://github.com/deepseek-ai/deepseek-harness/discussions), observed 2026-08-20; exact reports are linked beside each claim above.
