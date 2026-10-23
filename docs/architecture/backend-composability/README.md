# AMD, Vulkan and Halo to multi-backend composability

Source-review snapshot: 2026-10-23. Repository: [anthony-chaudhary/fak](https://github.com/anthony-chaudhary/fak).

This index turns four public-source inventories into bounded work for NVIDIA and other backends without deleting existing AMD qualification. The result is **31 canonical planning records: 24 new-leaf records (two filed, 22 proposed) and 7 existing-issue updates**. The active HIL and cooperative-matrix repairs are each included once; two model proposals still need exact scope resolution. HIL [#13958](https://github.com/anthony-chaudhary/fak/issues/13958) and cooperative-matrix [#13959](https://github.com/anthony-chaudhary/fak/issues/13959) are verified bindings; remaining creation inputs are proposals. This catalog edits documentation only; separate implementation commits and their limited evidence are identified below.

- [Reviewed file/symbol seams and preservation requirements](INVENTORY.md)
- [Machine-readable ticket bodies, statuses, dependencies and source links](ticket-plan.json)

## Exact source pins and evidence boundaries

| Source | Snapshot status | What it establishes |
|---|---|---|
| [93b295a8](https://github.com/anthony-chaudhary/fak/tree/93b295a81683f81868d1b9cc4c340496700cc0da) | Inventory baseline | Exact public code for the lexical census and targeted semantic findings. |
| [b7ccbe9f](https://github.com/anthony-chaudhary/fak/commit/b7ccbe9f9d336a70ad0511159daf0219e3339e0a) / [#13956](https://github.com/anthony-chaudhary/fak/issues/13956) | Published gated BF16 source | Opt-in finite BF16 source/protocol packet, not general CUDA V4.1 admission. Full native compile/link/device/lifecycle/performance qualification remains deferred. |
| [68fbd200](https://github.com/anthony-chaudhary/fak/commit/68fbd200b2f4ebee0fe44b6a846947cb6db83136) | Later hosted delta, observed 14:22 UTC | Appended NVFP4 dtype 21 and CPU E2M1/E4M3FN reference oracle only. No loader, encoder, native scale-layout, kernel or hardware-performance credit. |
| Tail-RoPE candidate | Unpublished active work; limited CPU source/protocol proof | Corrected patch `e0c6d577f0d1418539aaa78a4c2a17f0595363c225c61fd57575811174d0efe3` is source-accepted. One compile, two passing baselines and fourteen intended mutation refusals completed in 20.43 s with owned processes drained. CUDA wrapper/native sources remained fixtures; no ABI, lifecycle, arithmetic/FTZ, GPU or model qualification. Exact capability remains false. No duplicate leaf. |
| Workflow candidate `61425860c4a4c3eb85bdd4b6f34b453fdb8d276b` / [#13955](https://github.com/anthony-chaudhary/fak/issues/13955) | Unpublished; workflow publication scope blocked | Final no-run CUDA executable linkage belongs to this existing owner. It is not hosted main and cannot be worked around by changing credentials or permissions. |
| HIL reference-truthfulness repair | Published [30e1434f](https://github.com/anthony-chaudhary/fak/commit/30e1434ffc444009619b285142e00c8c46b77c00), issue [#13958](https://github.com/anthony-chaudhary/fak/issues/13958) open | Exact patch `05c5986b97f18fbdeb6b07d101b72f6acb66ae6d34273947123d52facb85c3fe`: baseline intended failures, candidate three CPU roots/15 pass events, no skips. Full Darwin/CLI/GPU/model checks remain deferred; source publication is not hardware proof. |
| Cooperative-matrix evidence repair | [#13959](https://github.com/anthony-chaudhary/fak/issues/13959) filed; source accepted, uncommitted | Two-file patch `676d1ad28b97d99ff4acfe4c386119b0822f16830b67870ad90c79c204c39268` removes fabricated floors and hardware wording while preserving numerical/reference interfaces. Compiler, formatter and Go tests were not run at this snapshot. |

Exact source comparisons: [93b → BF16](https://github.com/anthony-chaudhary/fak/compare/93b295a81683f81868d1b9cc4c340496700cc0da...b7ccbe9f9d336a70ad0511159daf0219e3339e0a) and [BF16 → NVFP4](https://github.com/anthony-chaudhary/fak/compare/b7ccbe9f9d336a70ad0511159daf0219e3339e0a...68fbd200b2f4ebee0fe44b6a846947cb6db83136). The latter modifies only `compute.go`, `nvfp4.go` and `nvfp4_test.go`. No baseline counts were silently recomputed at the later revision. Later hosted main `dc0bdb6bc6420d089946d4b210afe0901826f105` adds the separate guard lifecycle repair [#13957](https://github.com/anthony-chaudhary/fak/issues/13957) on top of 68fbd20 without compute/model changes. It does not expand this inventory's semantic evidence. Refresh hosted tip and issue status before implementing or publishing a leaf.

## What was covered

| Slice | Search coverage | Semantic review boundary |
|---|---|---|
| HAL / compute / public ABI | All 1,106 tracked scoped files read and hashed: 605 source, 488 tests, 13 other; 9,258 lexical declaration anchors | Contracts, optional capabilities, native/reference distinctions, errors/readback/fences, selected-device profiles/observations, format claims, named branches, shaders and candidate boundaries. Not exhaustive numeric proof or a compiled call graph. |
| Memory / scheduling / residency | 2,948 source/test files available to the search; 143 file entries in the reviewed-family catalog | 24 ownership, transfer, pools, graph, topology, tier and capacity seam groups. Not a line-by-line audit of every search file. |
| Model / loader / agent integration | 2,255 Go paths, 763 non-test; 117 vendor-hit files, 67 with non-comment hits; 693 match lines | Targeted dispatch, profile consumers, load estimates, reports, snapshots, lineage and qualification gates. No `internal/agentruntime` or `pkg/agentruntime` tree exists at the pin. |
| Tooling / build / receipts / docs | Ten bounded seam groups; no exhaustive file-count claim | HIL truthfulness, comparison admission, validation routing, manifests, roofline policy, telemetry identity, environment stamps and benchmark authority. |

Counts overlap and must not be summed as unique files. Vendor-name matches are review leads, not automatically defects. Private implementation and private ownership/dedup are outside the evidence in this public packet. No GPU run, model load, purchase, device-access change, or benchmark was performed by the inventory.

## Design rules

1. Reuse the optional interfaces and explicit policies already present. Do not replace the HAL wholesale or mechanically rename Vulkan/Strix code.
2. Keep facts separate from policies, requested configuration, simulation, native execution, admission and measured qualification. Unknown remains unknown, not a Halo default or zero-valued success.
3. Keep allocator, recording-domain and selected-device identity separate from backend names. Retain owners until checked completion; uncertain completion is not success and must not permit reuse.
4. Preserve AMD vendor/device/subgroup/kernel guards, supplied-table arithmetic, exact rounding/status protocols, model/artifact/numerical identity, complete continuation and bounded memory accounting.
5. A capability interface is not permission to remove V4.1 prefix, MTP rollout or quant-route whitelists. Qualify the exact backend/artifact/shape/operation envelope first.
6. Missing capability may decline before mutation. Failure after selecting/mutating a device operation must not silently replay on CPU, serial execution or an external engine.
7. Software fixtures, compilation, final linkage, software-ICD registration, physical numeric correctness and workload performance are separate evidence stages. Never close the latter with the former.

## Canonical ownership across seams

- **HAL:** device-memory profile and exported resolver; selected-device fact providers; native checked adapters; gpustats. DeviceMemoryProfile contains cache/alignment/subgroup facts, not bandwidth.
- **Memory:** execution-domain adapters, accounting classes, owned asynchronous completion, pools/tiers/registration/topology and ring retirement. Keep active ownership work and existing guarded synchronous paths intact.
- **Model:** neutral dispatch and consumers of explicit profiles/storage plans/reports. CPU-miss uses declared/calibrated ExpertCPUMissProfile bandwidth plus actual artifact geometry, not cache-size-derived bandwidth.
- **Tooling:** HIL/evidence truthfulness, comparable benchmark arms, build taxonomy, target-specific validation and selected-device receipt joins. Existing NVIDIA receipt owners remain canonical.

## Priority and dependency DAG

Order is risk/dependency guidance, not a claim on paths or an instruction to create every issue at once.

The cooperative-matrix evidence correction is also actively being prepared; reserve its existing implementation rather than starting a duplicate.

1. **P0, published source with remaining qualification:** `hil-reference-execution-truthfulness`. Track [#13958](https://github.com/anthony-chaudhary/fak/issues/13958), published [30e1434f](https://github.com/anthony-chaudhary/fak/commit/30e1434ffc444009619b285142e00c8c46b77c00), its actual CPU evidence and explicit integration debt. Do not start a second implementation.
2. **P1, small correctness/evidence fixes:** `memory-strix-backing-owner-release`, `hal-coopmat-honest-model`, HIL liveness identity and comparison compatibility. These can be proved with CPU fixtures; they do not qualify hardware.
3. **P1, composable policy/consumer leaves:** profile resolver, Qwen batch dispatcher, explicit CPU-miss profile, QSA consumer, GGUF storage plan and neutral agent reports.
4. **P1, lifecycle/native prerequisites:** complete CUDA owner, owned transfer contract and ring retirement; failure-atomic tiers; discovered topology; selected CUDA memory facts; staged checked D2H. Keep current owner/hot-file gates.
5. **Existing owners:** build manifests #3712, required-target receipts #10934, CUDA control/profile receipts #9121/#13938/#9497, Windows gpustats #11318, mixed-precision matrix #13102, tiers #11323 and topology #11321.
6. **P2 / unresolved:** registration evidence, explicit token-pool capacity, roofline policy, off-live registry matcher, shared-panel producer resolution and advisory cache intent.

Logical edges:

- `hal-export-memory-profile-resolver` → `model-qsa-profile`.
- `memory-owned-transfer-contract` → `memory-ring-owned-transfer-integration`.
- Existing complete-owner stack readiness → CUDA forward-owner adapter and ring integration. A later native CUDA transfer adapter must preserve this contract and be separately scoped; no overlap credit yet.
- Active CUDA kernel-file work reconciled + actual final-link gate → `hal-cuda-checked-readback-native` → `hal-cuda-checked-readback-consumer`.
- `gguf-storage-policy` → separately scoped real NVIDIA loader wiring and measured phase-peak qualification, not immediate fit admission.
- Explicit producer/test paths resolved → `model-shared-panel-trait`; replace the compatibility producer before removing its model fallback.

The selected CUDA memory provider can implement the existing provider independently; it is not a false prerequisite for CPU-only QSA policy tests. CPU-miss is independent of the cache-profile resolver. Same-package HIL writes and shared CUDA kernel files require serialization even when the logical DAG has no edge.

## Existing issues, no duplicate umbrellas

The JSON contains fetched public issue states and exact links. Preserve these distinctions:

- Open [#10442](https://github.com/anthony-chaudhary/fak/issues/10442) owns cross-backend cleanup; [#13949](https://github.com/anthony-chaudhary/fak/issues/13949), [#13950](https://github.com/anthony-chaudhary/fak/issues/13950), [#13951](https://github.com/anthony-chaudhary/fak/issues/13951), [#13952](https://github.com/anthony-chaudhary/fak/issues/13952) and [#13953](https://github.com/anthony-chaudhary/fak/issues/13953) own distinct Vulkan decomposition work. No replacement owner stack.
- Closed [#12440](https://github.com/anthony-chaudhary/fak/issues/12440) provided compute profile policy; new resolver/provider/model-consumer work is distinct. [#12444](https://github.com/anthony-chaudhary/fak/issues/12444) is a closed duplicate, not a spare new task.
- Closed [#12188](https://github.com/anthony-chaudhary/fak/issues/12188), [#11038](https://github.com/anthony-chaudhary/fak/issues/11038), [#12736](https://github.com/anthony-chaudhary/fak/issues/12736), [#10758](https://github.com/anthony-chaudhary/fak/issues/10758), [#5627](https://github.com/anthony-chaudhary/fak/issues/5627), [#12191](https://github.com/anthony-chaudhary/fak/issues/12191) and [#12709](https://github.com/anthony-chaudhary/fak/issues/12709) are predecessors, not proof the identified remaining seams are resolved.
- Open [#9987](https://github.com/anthony-chaudhary/fak/issues/9987), [#13800](https://github.com/anthony-chaudhary/fak/issues/13800), [#12865](https://github.com/anthony-chaudhary/fak/issues/12865), [#12839](https://github.com/anthony-chaudhary/fak/issues/12839), [#12442](https://github.com/anthony-chaudhary/fak/issues/12442), [#12983](https://github.com/anthony-chaudhary/fak/issues/12983), [#13891](https://github.com/anthony-chaudhary/fak/issues/13891), [#13893](https://github.com/anthony-chaudhary/fak/issues/13893) and [#13900](https://github.com/anthony-chaudhary/fak/issues/13900) retain CUDA MTP/prefix/state/quant/numerical/provenance/Engram ownership. This inventory neither closes them nor re-creates them.
- [#13542](https://github.com/anthony-chaudhary/fak/issues/13542) retains gfx1151 FP4/FP8 ISA-claim correction, including matching model/roofline claims; an architecture label does not establish arithmetic support.

Issue state is a 2026-10-23 snapshot, not a support matrix. Unknown state is explicitly `not_reverified_in_packet`, never silently treated as open or closed.

## Ticket and publication contract

The docs-only tracker fits open [#1678](https://github.com/anthony-chaudhary/fak/issues/1678), whose heterogeneous-fleet reference-architecture and gap-mapping scope covers this index. It is a nonclosing documentation contribution, not a blanket parent or implementation grant for every source leaf. Individual source tickets retain their actual component owners.

Each JSON record carries a stable dedup key, proposed bounded paths, source evidence, action (`create_leaf`, `track_created_leaf` or `update_existing`), related issue states, exact acceptance/hardware boundaries and a native issue-body input. New identifiers are planning keys, not fabricated GitHub issue numbers. The created HIL and cooperative-matrix records point to verified #13958/#13959 and explicitly forbid another create operation. `admission_ready` is false until current-source/ownership/dedup and required native metadata are verified.

The created HIL issue carries a new proposed, unearned 1/13 allocation under #12472, with its rationale in the hosted issue; it is not represented as a historical budget. Other parent relationships and point allocations remain unassigned, not invented. Closed predecessors are lineage. The publication owner must resolve normal routing/work metadata and any required parent budget/contribution convention. A verified existing convention may be reused, or a new reasoned baseline may be transparently proposed from the actual scoped leaves, explicitly unearned and not represented as historical fact. Historical points are not an intrinsic blocker, and no additional human approval is required merely to prepare authorized project planning. Zero earned contribution is claimed; no fabricated parent or validation bypass is permitted.

Before mutation: refresh public source, exact dedup key/title/outcome and issue state; resolve paths; preserve active writers; use normal native validation. For an existing issue, add the relevant bounded evidence/scope without replacing unrelated content or changing state. For a new leaf, create once and bind its returned number/URL. Record the actual publication commit only after hosted verification. Keep source-only publication Ref-only when broader qualification remains outstanding.

The public files deliberately contain no private source, internal preparation notes, machine paths, runtime agent/session identifiers or credential information. They do not launch hardware or alter device access.

Documentation publication tracker: [#13960](https://github.com/anthony-chaudhary/fak/issues/13960), a nonclosing documentation contribution under [#1678](https://github.com/anthony-chaudhary/fak/issues/1678). Component repairs retain their own issue owners.
