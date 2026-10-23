# Backend composability: source-pinned semantic inventory

This is the reviewed engineering map behind [the index](README.md) and [canonical ticket inputs](ticket-plan.json). It preserves four complementary scopes rather than treating every AMD/Vulkan/Halo identifier as a bug. The common baseline is public commit `93b295a81683f81868d1b9cc4c340496700cc0da`; later hosted BF16 and CPU-only NVFP4 deltas are explicitly separate. The inventories ran no compiler, numerical test, hardware job, or performance experiment.

Read the index first for current snapshot status, cross-component ownership, dependency ordering, unresolved scope and publication requirements. File counts overlap. Enumerating all files in a slice is not the same as semantically reviewing every line or all full-repository callers.

## Later implementation-status update

The inventories below retain their original source-only provenance. Separately, the active HIL exact patch `05c5986b97f18fbdeb6b07d101b72f6acb66ae6d34273947123d52facb85c3fe` passed three focused CPU roots / 15 test events with zero skips; the original baseline failed its intended assertions. Darwin runtime, GPU and full-model integration remain unrun. The corrected tail-RoPE patch `e0c6d577f0d1418539aaa78a4c2a17f0595363c225c61fd57575811174d0efe3` is independently source-accepted; limited CPU source/protocol evidence now passed one compile, two baselines and fourteen intended mutation refusals, with owned processes drained. Wrapper/native fixtures were not executed; ABI/lifecycle/device/model qualification remains unrun. The HIL corrective issue is now [#13958](https://github.com/anthony-chaudhary/fak/issues/13958), published at [30e1434f](https://github.com/anthony-chaudhary/fak/commit/30e1434ffc444009619b285142e00c8c46b77c00), with full integration debt retained. Cooperative-matrix [#13959](https://github.com/anthony-chaudhary/fak/issues/13959) is filed with a source-accepted two-file patch, but compiler/formatter/Go tests and source publication remain unrun. These later statuses supersede pending preparation notes, not the baseline source findings.

## Compute/HAL composability inventory

### Executive result

The reusable seams mostly already exist. The safe direction is to supply selected-device facts and narrowly scoped native adapters, not rename Vulkan or remove qualified AMD implementations.

Highest-priority small leaves:
1. Correct cooperative-matrix simulation being labeled as a hardware witness (2 files).
2. Export the existing fail-closed device-memory resolver for model consumers (2 files).
3. Separate neutral V4.1 status decoding from Vulkan error attribution (2 files).

The next leaves are a selected-device CUDA memory-facts provider and staged checked CUDA D2H readback. Existing issues already own the mixed-precision matrix, Windows native GPU telemetry, Vulkan ownership decomposition, and interrupted-request cleanup. Canonical drafts and acceptance criteria are in [ticket-plan.json](ticket-plan.json).

No GPU, compiler, test suite, production checkout, repository write, or hosted issue write was used for this inventory. Source findings are not hardware parity results. The assembled documentation does not alter implementation source.

### Scope, pins and method

- Baseline: public main 93b295a81683f81868d1b9cc4c340496700cc0da, read from immutable Git objects. All 1,106 tracked files under internal/compute/** and pkg/abi/** were enumerated, read and hashed: 10,489,204 bytes; 605 source files, 488 tests, 13 other files.
- The local unpublished workflow commit 61425860 has no diff in these two scoped trees. It is not treated as hosted main or compute progress.
- File/symbol census contains 9,258 lexical declaration anchors. Directory totals: internal/compute root 875; shaders 113; strix 104; metal 3; testdata 3; arm64tile 2; deltanetavx512 2; slotstream 2; pkg/abi 2.
- Baseline path/blob/hash/build-tag/symbol metadata was enumerated for the stated scope; coverage counts are in the index.
- These are lexical indices, not a compiler call graph or proof that every semantic path was audited. Method-name matches need receiver/build-tag review. File family neutral_other means “not classified by the filename heuristic,” not “semantically neutral.”
- The semantic review covered core contracts; optional interfaces; CPU reference vs native distinctions; CUDA, Vulkan, ROCm, Metal and Strix adapters; error/readback/fence behavior; device-profile and observation providers; quantization/cooperative-matrix claims; vendor-named branch sites; shaders and active candidates. Deep numeric proof of all kernels and exhaustive full-repository caller analysis are excluded.
- Public issue dedup used targeted searches plus explicit issue reads. A negative search is not proof that no duplicate exists. Private ownership is unverified; private issue enumeration was not attempted.
- Model orchestration, storage/loader ownership, HIL/workflow/receipt schemas, internal/abi internals, internal/metalgemm implementation and internal/roofline internals have separate owners or are outside this scope. Interfaces and edges into them are identified here without claiming their complete review.

#### Later hosted delta, independently verified

At 14:22 UTC the connected GitHub main read returned 68fbd200b2f4ebee0fe44b6a846947cb6db83136. Baseline census remains explicitly pinned to 93b; later changes are appendices, not silently folded into its counts.

- b7ccbe9f9d336a70ad0511159daf0219e3339e0a is a direct child of 93b and publishes the gated finite CUDA BF16 source packet (#13956). Three changed files: cuda_kernels.cu, cuda_v41_bf16_publish.go, cuda_v41_bf16_publish_contract_test.go. The exact accepted patch is SHA-256 6ec82e5942658832d3e22c8542f8843d862fa4e9028ec5990463a4deb3c48023. Hosted source is not native execution qualification.
- Exact later comparison: [b7ccbe9f → 68fbd200](https://github.com/anthony-chaudhary/fak/compare/b7ccbe9f9d336a70ad0511159daf0219e3339e0a...68fbd200b2f4ebee0fe44b6a846947cb6db83136). Only compute.go, nvfp4.go and nvfp4_test.go changed. NVFP4 is appended as dtype 21, preserving prior IDs. DecodeNVFP4Reference/GemvNVFP4Reference provide a CPU unswizzled row-major E2M1 reference with E4M3FN per-16-value scales and an FP32 global scale. It does not describe a padded/2-D device scale layout, loader, encoder, native kernel, hardware result or performance. No duplicate NVFP4 oracle ticket is proposed.
- CUDA tail-RoPE remains active candidate work. Its corrected exact binding is now independently source-accepted; its bounded CPU source/protocol runner has two passing baselines and fourteen intended mutation refusals, not wrapper/native or GPU execution. Do not build a new implementation from the superseded 12d13 or 519df packet hashes. Production kernel semantics were unchanged by the reported helper fix. Its final binding must be refreshed by its owner before any dependent kernel-file edit.

### Semantic inventory

#### 1. Backend-neutral contracts to preserve and compose

| Files / symbols | Reusable mechanism | Gap or boundary |
|---|---|---|
| compute.go: Dtype, QuantSpec, Layout, Tensor, Buffer, HostBuffer, Caps, Backend | Operation/data/ownership interface independent of driver API; append-only dtype identity | An enum or broad Caps flag does not prove all format/layout/operation combinations. FP4 and NVFP4 must remain distinct. Lookup is explicit; Pick intentionally falls back to Default, so strict admission should use Lookup. |
| compute.go: TensorCloner, BackendResourceTeardown, BackendRequestRetirer, Fence, AsyncUploader | Immutable cloning, bounded retirement, ownership and completion seams | Adapters must not counterfeit completion or replay failed selected device work on CPU. Existing #10442 owns interrupted-request cleanup. |
| compute.go: RankUploader, CollectiveInitializer, ProcessGroupBackend, CollectiveBackend | Per-rank upload/group collectives | CUDA's device-zero leaf assumptions cannot be transplanted into multi-GPU paths. Selected allocation/device identity must be explicit. |
| pkg/abi/abi.go and tests: EngineDriver, RegionBackend, KVBackend, KVBackendFactory, KVResidency, PageOutBackend, Resolver, Adjudicator aliases/registrations | Stable external aliases to internal/abi | This is not compute.Backend. No vendor coupling found in scoped aliases; do not widen the frozen ABI with CUDA pointers/streams. |
| device_memory_profile.go: DeviceMemoryProfile, DeviceMemoryProfileProvider, CachePlanningAvailable, DeviceCacheTileTokens | Optional established facts, unknown/partial fail-closed state, generic aligned cache planning | Resolver private; no production provider method in baseline census. Fallback facts AMD-only. Export policy separately from native discovery. |
| physical_observation.go: BackendRuntimeIdentity, BackendExecutionSnapshot, BackendExecutionDelta, BackendExecutionWindow | Identity-bound before/after evidence, observation-presence flags and single-use window end | Production provider found only on Vulkan. CUDA transfer counters are not a complete neutral execution window. No zero-filled “unobserved equals zero work” adapter. |
| devicefault.go: DeviceFaultLatch; request_lifecycle.go | Fault epoch and request-cleanup mechanisms | Preserve bounded drain, poison/no replay, ownership retention after uncertain completion. #10442 and existing CUDA fault work own larger cleanup changes. |
| host_callback.go: GraphCaptureSession; wc_copy.go: WCDeviceDMASurface | Graph callback and write-combining DMA contracts | #12950 already established generic seams. Declared interface/host model does not show a production CUDA implementation. |
| qwen35_sequence_contract.go; qwen35_decode_batch_contract.go; qwen35_mtp_contract.go | Prefill/embedding/all-logit/prefix/feature-row and batch/MTP semantic extensions | Availability differs by extension. CUDA+Vulkan embedding rows exist; broad sequence or dtype support is not every optional capability. Model dispatch cleanup has another owner. |
| v41_*.go semantic interfaces and pure decoders | Finite publication; exact arithmetic; status protocol; ownership and no-replay contracts | At baseline native implementations of the reviewed V4.1 optional methods were Vulkan-only. Later BF16 and active RoPE are two specific CUDA leaves, not general V4.1 readiness. |

Primary anchors:
- [Core contracts](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/compute.go#L190)
- [Memory facts and resolver](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/device_memory_profile.go#L5)
- [Physical observation](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/physical_observation.go)
- [Public ABI aliases](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/pkg/abi/abi.go)

#### 2. V4.1 operation inventory and accidental attribution coupling

The capability census indexed exact interface/method/file anchors. Reviewed families include:

- BF16 boundary/checked/publish/attention-BF16; finite F32; exact RMSNorm; query prefix/norm; FFN, KV and index-query publication.
- Tail RoPE QK/rows/exact; exact logit and scaling.
- mHC initial/split/pre/post/RMS/prenorm.
- Embedding gather/Q2K; Engram gate/row decode.
- Compressor pool/ring/row logging.
- Index score/select/scaled weights/router top-K/gate.
- Routed SwiGLU/panels/chunks/down/IDs/fused Q2/Q3 token panels.
- Shared attention; step batches; deferred status; owned readback.

Preserve supplied-table RoPE, trailing adjacent-pair semantics, bitwise prefix preservation, fresh immutable F32 outputs, independently rounded operations, closed publication sites and arithmetic/protocol distinction. Current generic RoPE validators contain signed-32/combined-uint32 bounds reflecting existing shader admission; widening them for CUDA requires separate bounds proof, not a casual uint64 substitution.

Accidental/vendor-adapter coupling co-located in generic-looking files:

| File / symbol | Finding | Safe composition |
|---|---|---|
| v41_deferred_status.go: v41DeferredVerdict / v41DeferredProtocolError | backend string parameter still produces Vulkan classes and sentinel errors | Extract pure verdict core, retain old Vulkan-compatible wrapper. Top-ranked 2-file draft. |
| v41_attention_bf16_read.go; v41_ffn_norm_publication.go; v41_query_norm_publication.go; v41_index_select_read.go; v41_index_query_publication.go; v41_kv_norm_publication.go | Helper errors hardcode Backend: vulkan | Reuse arithmetic contracts; move/call attribution at adapter boundary when another backend needs the helper. No giant rename. |
| v41_indexer_score.go: vulkanV41IndexerScoreStatusError; compressor_pool/ring, engram_gate/row_decode, exact_rmsnorm wrappers | Vendor error mapping is deliberately Vulkan-specific despite neutral surrounding interfaces | Do not copy Vulkan transport status numbers into CUDA; share only protocol/arithmetic result. |
| v41_mhc_prenorm.go: kernel-event helper | Generic-looking file consumes vulkanKernelTiming/vulkanKernelEvent | Keep Vulkan instrumentation separate from neutral mHC math in a future bounded move. |
| vulkan_errors.go: BackendError, VulkanErrorClass; cuda_errors.go: CUDAOpError | Generic type name BackendError is structurally Vulkan-specific; CUDA has separate error shape | Avoid breaking existing errors.Is/As behavior. Decode neutral cause first, then backend-owned wrapper. |

[Deferred decoder evidence](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/v41_deferred_status.go#L56)

#### 3. Hardware implementations, not rename targets

| Family | Scope and examples | Preservation invariants / NVIDIA meaning |
|---|---|---|
| CUDA | cuda.go, cuda_kernels.cu/.h, cuda_specialized.go, graph_cuda.go, cuda_graph*, CUDA KV/MoE/quant/multi-GPU/NCCL files | cudaMu and native stream/context ownership; device-only vs managed memory; allocation bases and extents; fence generation; graph/address lifetime; true per-rank identity. cuBLAS/tensor-core kernels are CUDA-native, not substitutes for missing optional contracts. |
| ROCm | rocm_backend.go/.h, rocm_kernels.hip, rocm KV/arch files | HIP/rocBLAS and build gates; single-stream locking/device state; checked rocmCheck; architecture-specific supported formats. Do not infer parity from HIP/CUDA syntax similarity. |
| Vulkan | vulkan*.go, platform cgo/purego adapters, vulkan shim/native headers, SPIR-V admission and shader registry | Descriptor/queue/fence/resource ownership; runtime module admission; physical identity; shader feature/shape gates. Vulkan is a cross-vendor API; some modules remain specifically AMD-qualified. |
| Shaders | 113 files under shaders, predominantly Vulkan GLSL plus attention.metal, including shared attention, Q4K wave32/cooperative matrix, Q8 and GDN | Workgroup, subgroup, shared-memory padding, push ABI and numeric behavior are kernel implementation constraints. Shader presence or SPIR-V parsing is not device qualification. Do not globally replace subgroup assumptions with warp=32. |
| Metal | metal.go, metal KV/sequence adapters, metal subdirectory | Apple buffer ownership/shared memory is not CUDA managed-memory behavior. The separate internal/metalgemm implementation was not exhaustively reviewed here. |
| Strix | strix/*, strix_kernel_bridge.go | gfx1151, 40 CU/32 MiB MALL, wave32, KFD/GEM/USWC/GTT/HSA AQL, code objects, doorbells, UMA cache/fence and WMMA behavior remain hardware-specific. Extract parameterized algorithms only after proving the parameter contract. |

Some feature support varies across current backend implementations. Any exact operation/format decision should use method/capability checks and admission, not this table as a permanent support matrix. Existing #13102 owns that declarative matrix.

Specific qualified guards to retain:
- vulkan_q4k_wave32.go uses AMD vendor 0x1002/gfx1151 device 0x1586, subgroup feature and 32-lane requirements, plus row-shape thresholds. A separately qualified NVIDIA variant must not erase those gates.
- vulkan_spirv_admission.h validates shader bytes/capabilities/push ABI; strict RoPE admission protects contraction requirements. It does not prove driver numerics.
- host_environment.go / host_environment_drm.go implement an explicitly RADV/AMD Linux selected-device observer, including driver identity and DRM binding checks. Add a distinct NVIDIA observer if needed; do not weaken the RADV evidence check into an all-Vulkan claim.
- vram_carveout.go: DedicatedVRAMReporter is neutral; current sysfs logic is specifically integrated AMD Vulkan carveout accounting. Preserve no GTT/system-DRAM double count. It is not a generic discrete NVIDIA VRAM probe.

#### 4. Generic-looking simulations and architecture tables

- coopmat.go: queryHardwareCapabilities uses static vendor/architecture tables, including assumed subgroup/LDS/microtiles. ConfigureTileGeometry/Validate use fixed 64x64x32, 128 threads and padding=2. ExecuteGEMM is host-loop reference execution. EstimatePrefillThroughput assumes model TFLOPS/efficiency and forces sub-300 results to 325; coopmat_test.go calls that [HW-WITNESSED]. Correct evidence labeling and remove the floor, while preserving the host reference and AMD code. No production caller of NewCooperativeMatrixEngine/EstimatePrefillThroughput was found outside that file in the scoped search.
- strix/compute_hal.go contains RDNA35_WMMA_GEMM/RDNA35_GEMV host arithmetic and modeled roofline alongside hardware types. These are not native dispatch evidence.
- cuda_int4_skinny_rdna.go is tag-free pure-Go RDNA W4A16 reference/layout work despite the CUDA filename. Existing #13155 owns this lane.
- radv_attention.go has reusable F16/F32 conversion/contiguizing helpers plus Strix-specific modeled channel/128-byte assumptions. hip_attention.go wraps the host pass. uma_strided_copy.go specifically matches Strix. Do not transplant UMA locality assumptions to discrete PCIe.
- AMD GPUDirect files include emulated registration/doorbell/completion paths. cuda_gpudirect_storage.go likewise models host queues with hardcoded sm120/RTX5090/BAR assumptions. Neither filename proves native direct-storage DMA. Memory/storage inventory owns canonical follow-on tickets; no duplicate here.

[Cooperative-matrix estimate](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/coopmat.go#L839)
[Claiming test](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/compute/coopmat_test.go#L387)

#### 5. Memory facts, readback and physical observation gaps

**Canonical memory profile producer.** DeviceMemoryProfile has positive-value/known-flag gates. gfx1151 fallback supplies 32 MiB/128 B/32 lanes, other known AMD architectures only subgroup facts. Unknowns stay unknown. Provider results take precedence, including a partial/unknown result. No production DeviceMemoryProfile() implementation was found in baseline. Closed #12440 already owns the generic seam; new leaves export the resolver and then supply selected CUDA facts without inventing cache residency or allocator alignment. QSA consumers belong to model inventory and depend on the exported resolver. CPU-miss bandwidth needs separate declared/calibrated bandwidth and actual slab geometry through ExpertCPUMissProfile; DeviceMemoryProfile does not provide or imply bandwidth.

**Checked ordinary CUDA readback.** cuda.go Read invokes void native D2H calls, then advances fenceGen and returns a slice. cuda_kernels.cu prints a failed cudaMemcpy but cannot return its status; the per-device wrapper can return early after selection failure. Contrast Vulkan's checked Read (#12709) and ROCm's rocmCheck. Proposed staged leaves first add checked native ABI (3 files), then integrate Go Read (2 files), preserving typed errors, withholding output and not advancing successful state on failure. Active BF16/RoPE own independent checked paths; do not replace them or collide with their kernel edits. This is a source-level failure-propagation gap, not a reproduced device failure.

**Observation.** Vulkan implements the neutral physical snapshot/window provider. CUDA exposes transfer counters but not the full selected-identity observation/window semantics in the baseline. A future adapter must maintain monotonic counters, observation-presence flags, correct reset/wrap handling and scoped attribution. HIL receipt-schema work is separate. Host telemetry does not prove dispatch or correctness.

**GPU stats.** SystemGPUStats tries Apple then nvidia-smi. Unavailable/invalid information must stay unavailable, not zero memory. GPUStat's ordinal is not stable physical identity; HarnessGPUVRAM prefers active backend capacity, otherwise aggregates, which must not be mistaken for selected-device evidence. Existing open #11318 owns Windows DXGI/PDH fallback; no new issue needed.

#### 6. Model-facing interfaces that need composition rather than renaming

- qwen35_gdn.go: shared allocation/geometry pieces are reusable, but VulkanQwen35GDNConvTiledChannelTransposer is vendor-named, and Qwen35GDNAllocationError carries CUDA attribution. Separate semantic state/allocation contracts from adapter attribution in a future bounded consumer-driven change.
- glm_kda.go: VulkanGLMKDAStepper exposes VulkanGLMKDAWave32Available/VulkanGLMKDAStep and matched-ablation variants. These encode an implementation family, not a universal recurrence API. Derive neutral recurrence semantics first, retain current Vulkan variants.
- qwen35_sequence_fence_policy.go is intentionally a Vulkan legacy per-layer-fence ablation; do not globalize its environment policy.
- Model batch dispatch/build-tag neutralization has a separate owner. Existing #13881, #11036, #9257, #8820, #13548 and routed-expert issues already cover portions of broader backend/model work. An optional method's existence is not full end-to-end admission.

### Candidate boundaries and no-duplication rules

**BF16 hosted b7ccbe9 / #13956:** cuda+cgo+fak_cuda_v41_bf16_publish and separate native FAK_CUDA_V41_BF16_PUBLISH remain opt-in. Fresh device-only F32 output, finite ordinary widened-BF16, integer RNE, 16-byte status, checked completion, selected live device-zero ownership and quarantine after uncertain drain are the intended boundary. No CPU replay/default promotion/model admission/whole-V4.1 claim. Publication notes explicitly leave full CUDA compile/link/device numerics, transfer/lifecycle/fault/concurrency/performance unrun; do not upgrade reduced CPU/source checks into hardware evidence.

**Tail-RoPE active:** supplied F32 sin/cos table, trailing adjacent pairs and exact prefix; separate rounding/no table regeneration/no extra BF16; fresh device-only rows; live allocation/extent/alias validation; checked completion; idle-stream refusal; terminal quarantine. Exact capability stays false pending arithmetic/FTZ device witness. Corrected binding is independently source-accepted; limited CPU source/protocol evidence passed two baselines and fourteen intended mutation refusals; native/ABI/lifecycle/device/model proof remains unrun. No duplicate implementation ticket.

**NVFP4 hosted 68fbd20:** CPU reference only, distinct dtype and scale identity. Reuse the oracle for a later kernel, but do not credit a loader/native scale layout/device kernel. No duplicate oracle ticket.

### Existing issues and ownership

- #12440 closed: generic memory-profile seam. Follow-ons must not reopen merely to add NVIDIA.
- #12188 closed: original universal cooperative-matrix framework. New narrow evidence/model correction references it.
- #13102 open: explicit mixed-precision kernel matrix. Update its scoped acceptance instead of duplicating.
- #11318 open: Windows native GPU telemetry fallback.
- #10442 open: interrupted-request backend cleanup.
- #13949–#13953 open: Vulkan kernel specs, checked submission ownership/completion, descriptor retirement, allocation accounting, profiling ownership. Keep existing owners and AMD behavior.
- #12709 closed: checked Vulkan readback precedent, not CUDA implementation.
- #12950 closed: graph callback/WC DMA neutral seams, not proof all adapters exist.
- #13936 open: CUDA end-to-end no-fallback acceptance family; #13956 BF16; #13955 no-run link lane; #4180 Blackwell umbrella.
- #13542's gfx1151 FP4/FP8 ISA correction and roofline details are owned by tooling inventory; do not create a duplicate. Preserve modeled/simulated labels.

The issue-state snapshot is embedded in ticket-plan.json; state is not proof of readiness or physical qualification.

## Memory, residency and scheduling portability inventory

Source: public `anthony-chaudhary/fak` at `93b295a81683f81868d1b9cc4c340496700cc0da`, inspected 2026-10-23. Exact files were read with `git show`; the publisher checkout and its unpublished HEAD were not modified. This is source inspection and ticket preparation, not a build, hardware qualification, or performance result.

### Main conclusion

FAK already has useful backend-neutral contracts. Preserve them and add missing native adapters rather than generalizing every AMD-named object into another HAL. The important missing pieces are complete CUDA execution-domain ownership, classed/owned asynchronous transfers, honest physical-memory provenance, and failure-atomic tier movement. Halo's coherent UMA, GTT, AQL and MALL policies are not CUDA's discrete-memory, event or graph semantics.

The most actionable newly observed resource defect is in `strix.UMAPointerBackingProvider`: allocation registers an owned `UMABuffer` with its manager, while `FreeSlab` is a no-op. That is a source-proven retained-owner mismatch in this allocator, not a proven live-serving leak. The same provider calls its Go-heap allocation direct GTT and increments a simulated ioctl count. These require separate ownership and evidence fixes before reuse as a generic allocator.

### Existing contracts to reuse

- `internal/compute/compute.go`: `Backend`, opaque `Tensor`/`Buffer`, optional `TensorCloner`, `BackendResourceTeardown`, `BackendRequestRetirer`, `RankUploader`, `Fence`, `AsyncUploader`. Do not make host pointer conversion part of the generic tensor contract.
- `capacity.go`: `MemoryClass`, `MemoryScope`, `MemoryDemand`, `MemoryPlan`, `DeviceCapacity`, `HostCapacity`, `DeviceCeiling`, `FitVerdict`, typed allocation errors. Preserve separate host/device accounting, unknown-capacity states, explicit headroom and overflow-safe totals. A logical plan is not an allocation reservation.
- `epoch.go`: `HardwareEpoch`, `BackendEpochReporter`, `EpochTracker`. Cached topology, allocation requirements and graph identity must be tied to the actual device/context generation.
- `forward_execution_owner.go`: `ForwardExecutionOwnerBackend` identifies a complete recording domain, not an operation mutex. All aliases/wrappers of a shared native context must report the same owner.
- `gpubudget.go`: shared CUDA/Vulkan weight-budget parser. Its 90% auto fraction and unknown-to-unbounded legacy behavior are deliberate policy, not detected hardware facts. Any fail-closed policy change must be explicit and compatibility-tested.
- `kvresidency.go`: `PlanKVResidency` and `KVResidencyMemoryPlan` already separate host/device KV budgets without naming a vendor.
- `graph_pipeline.go`, `vulkan_transient_plan.go`: typed `Graph`, `TransientValue`, `PlanGraphTransients`, escape/lifetime analysis and deterministic allocation receipts are already backend-neutral despite the latter filename. Keep the Vulkan binding adapter separate; a rename alone is not portability work.
- `internal/polymodel` plus `internal/model/paging_ring.go`: bounded residency admission, pinning, immutable descriptors, exact quantized footprints, hold/release and victim settlement are reusable. Do not create another expert cache.

### File/symbol inventory and classification

Each row names the source seam, current coupling, the reusable mechanism and the preservation requirement. `native` means a real driver path exists in source, not that it was executed here; `model` means a host-side planner, simulator or policy object.

1. **Native Vulkan allocation and backing.** `vulkan_shim.cpp`: `Buffer`, `WeightArenaBlock`, `allocateCompatibleMemory`, `allocBuffer`, `fvk_malloc_weight`, `fvk_malloc_hostvis_weight`, `fvk_free`; `vulkan_memory_guard.h`; `vulkan_plan_vulkan.go`. Vulkan heap/type flags, dedicated requirements, alignments and coherent mapping are intentional API specialization. Implicit global ownership is accidental coupling. Reuse allocation identities, reservation rollback and shared-backing ownership; keep descriptor users alive until successful completion. Existing #13952, #12217, #12223, #13760/#13761 own this work. Do not duplicate their native extraction.

2. **Native allocation facts.** `vulkan_memory_capacity_snapshot.go`: `VulkanMemoryCapacitySnapshot`; `vulkan_buffer_requirements.go`: `VulkanBufferMemoryRequirements`. These return device-generation-tagged observations and exact resource requirements, not fit guarantees. A neutral facts envelope can have a CUDA adapter, but CUDA has no equivalent Vulkan memory-type-bit lattice. Preserve complete-or-empty results, known bits, interval, generation, quarantine/busy refusal and no silent flush/trim/retry.

3. **CUDA allocation and managed spill.** `cuda_backend_state.go`: `cudaBuf`, `ucKey`, `uploadCache`, `cudaBackend`; `cuda_tensor_allocation.go`: `dallocWeight`, `dev`, `devTr`, `devTrDeviceOnly`; `cuda.go` classed allocation helpers. CUDA device-local versus managed memory is intentional. Global upload cache, one context/stream and process-wide pooling make instance-local ownership assumptions unsafe. Preserve the host GC owner in pointer-keyed caches, dtype/layout keys, quantization side-channel bytes, strict device-only refusal, no managed spill for strict mutable state, and budget decrements exactly once.

4. **CUDA native transient pool.** `cuda_kernels.cu`: `g_pool`, `fcuda_malloc`, `fcuda_free`, `fcuda_graph_prewarm`, `fcuda_trim_pool_large`; `cuda.go`: `Recycle`, `releaseTransientBuffers`. Current exact-size free-list reuse relies on one FIFO stream and externally serialized calls. Do not convert it to several streams or per-instance pools by changing a mutex. Any shared pool needs allocation/context identity and dependency-complete reuse; `cudaMallocAsync`/mempools are a possible future adapter, not a drop-in safety proof.

5. **Whole-forward execution ownership.** `forward_execution_owner.go`, `vulkan_forward_execution_owner.go`, `internal/model/backend_execution_owner.go`. Vulkan implements the optional owner; no CUDA implementation was found. CUDA's `g_stream`, graph exec, transients and upload cache are nevertheless shared. A bounded CUDA provider leaf should join the existing contract and generic lifecycle tests, not rewrite the active 24-path Vulkan/model ownership stack. Parent #10442 remains canonical for interrupted-request cleanup.

6. **Request retirement and failure quarantine.** `compute.go`: `RetireBackendRequestResources`, `TeardownBackendResources`; `vulkan.go`: request retirement/teardown; `vulkan_shim.cpp`: checked submission, retained frees, quarantined ownership; CUDA `DeviceFaultLatch` and invalid `cudaBuf` bit. The guarantees are generic: idempotent cleanup, no reuse on uncertain completion, original failure preserved, subsequent request clean or typed refusal. Implementations remain API-specific. Existing #10442, #6412, #10426, #13950/#13951 and the active ownership stack are dependencies, not new duplicate tickets.

7. **Expert residency and asynchronous staging.** `internal/model/paging_ring.go`: `uploadStagedAsync`, `stageWithError`, `awaitStaged`, `discardStaged`, `hold`, `release`; `expert_ring_hal.go`, `expert_ring_prefetch.go`, `expert_ring_model_scope.go`, `expert_ring_shared.go`. The ring is neutral. `uploadStagedAsync` intentionally refuses async for complete-owner backends and prefers classed synchronous uploads because async cannot yet carry class and an undemanded fence can outlive the forward. Preserve this safety gate. No production `UploadAsync` implementation was found in compute. Closed #5627/#5614/#13044 established seams and model-side behavior, not NVIDIA transfer overlap qualification.

8. **Transfer completion semantics.** `compute.go`: `Fence.Wait()/Done()`. Current narrow contract lacks typed wait/query error and explicit source ownership or execution-domain retention. A future classed owned transfer must retain host source and destination until completion; distinguish pending, complete and failed/unknown; settle undemanded transfers before domain retirement; preserve byte-accounting class; propagate/retain failure without freeing uncertain resources. Do not infer that `Caps.Async` implies async H2D.

9. **Native Vulkan staging.** `vulkan_shim.cpp`: transfer staging, `fvk_h2d`, `fvk_d2h`, direct coherent mapped-weight path; `vulkan_borrowed_expert_upload.go`: `SupportsBorrowedExpertUploads`. Synchronous borrowed-source upload is intentional and permits caller release on return. On CUDA, use an explicit pinned-host/native completion contract before permitting borrowed async sources. A `runtime.Pinner` is not a CUDA page-lock or RDMA registration receipt.

10. **Mapped write-combined host copies.** `devicecopy.go`: `CopyFromWriteCombined`, optional hook and CPU feature/env gating; `strix/streamcopy*.go`, `strix/coherency_fence.go`. WC access policy, x86 fences and AMD AQL signaling belong in concrete adapters. The generic operation should describe source mapping/cache/coherency capabilities and completion, not assume all shared memory needs an AMD/x86 fence. The pure-Go unrolled copy is not proof that non-temporal machine instructions executed.

11. **Direct slab registration.** `direct_slab.go`: `DirectSlabConfig`, `DirectSlabAllocator`, `SlabAllocation.GetSGE`, `IsZeroCopy`; consumer in `amd_gpudirect_apu.go`. Constructor allocates Go bytes, derives `registered`/`direct` from configuration or env, and can invent `lkey=0x1000`. This is insufficient evidence of device mapping or verbs registration. Separate host allocation from a verified registration handle/provider, and keep synthetic mode explicitly labeled. Closed #10758/#11910 are historical context, not current physical proof.

12. **Strix slab caching.** `strix/pool_allocator.go`: `BackingProvider`, `UMAPointerBackingProvider`, `HeapBackingProvider`, `BlockCachingAllocator`, `Trim`, `Reset`; `strix/uma_pointer.go`: `UMAPointerManager`. Binning, warm reuse and pressure trimming are reusable. Halo 64 KiB–32 MiB bins, 2 MiB slab alignment and 64 GiB watermark are profile policy. `BlockDescriptor.DevPtr == HostPtr` cannot become generic GPU semantics. Track actual provider/owned handle per slab, release through its originating provider, and retain/reconcile ownership if release fails. The no-op `UMAPointerBackingProvider.FreeSlab` is a concrete source-level defect.

13. **Strix UMA and native/synthetic distinction.** `strix/uma_mmu.go`, `uma_pointer.go`, `uma_alloc.go`, `kfd_alloc*.go`, `kfd_ioctl.go`, `kfd_types.go`. `UMABuffer`/pointer manager use aligned Go heap; `UMAZeroCopyAllocator` has explicit native/hook/simulated paths and `Mode()`. Preserve those distinctions. Host/device numeric pointer equality is neither device accessibility nor coherency proof. CUDA UVA, mapped host memory and managed memory are different capabilities.

14. **Ping-pong prefetch and cache models.** `strix/double_buffer.go`, `buffer_types.go`, `weight_stream.go`, `stream_types.go`: `DoubleBufferManager`, `MALLCacheModel`, `WeightStreamPipeline`. Scheduler lifecycle, cancellation and in-use-buffer exclusion are reusable; 2×16 MiB buffers, fixed MALL set ranges, 40 CUs and modeled DRAM rates are intentional Halo policy. `allocateAligned` and goroutine transfer simulation do not pin bytes in hardware cache. Generalize a plan and transfer/completion adapter only when a production consumer exists.

15. **MALL residency and placement.** `strix/mall_partitioner.go`, `mall_tiler.go`, `mall_tiling.go`, `mall_types.go`, `partition_types.go`, `kv_pack.go`, `set_permutation.go`, `line_alignment.go`, `block_layout.go`. Cache set math, tile geometry and MALL budget are hardware-specific and should stay in an AMD adapter. Reuse checked geometry and capacity policy with actual backend facts, not by relabeling MALL as NVIDIA L2. Canonical device-profile follow-on is owned by the HAL inventory; model QSA consumer by model inventory.

16. **CUDA capture and replay.** `cuda.go`: `GraphBegin`, `GraphEndLaunch`, `GraphReset`, `GraphAbort`, `graphPrewarmDepth=128`; `cuda_kernels.cu`: `g_capture_open`, `g_exec`; `cuda_kv.go`; `graph_cuda.go`; `constant_upload.go`. Capture-free allocation, stable addresses, full parameter uploads during capture, fixed KV capacity and abort-on-unwind are intentional requirements. A universal graph interface must represent support/refusal and stable resource ownership, not assume Vulkan recorder equivalence. Pool depth is a capacity policy and should eventually derive from a bounded live-resource plan. Closed #1479 is historical; graph enablement is still explicit in this source.

17. **Vulkan graph/cache planners.** `vulkan_graph.go`: `VulkanCommandGraph`, layer graph builder and staging-pool models; `vulkan_config.go`: `PowerOfTwoGraphCache`, `GraphCacheKey`, `StrixHaloMTPConfig`; `mtp_rdna35.go`. Power-of-two LRU caching is reusable; 1–64 draft buckets, 512 ubatch and Strix config are profile policy. Graph objects/counters are host models unless a native consumer is proved. Preserve graph destruction callback, in-flight retirement and context/shape/epoch identity before sharing cache machinery with CUDA.

18. **Batch submission and hazard planning.** `vulkan_plan.go`: `ForwardPlan`, `BatchRecorder`, `RecordForward`, `AsyncOverlap`; `vulkan_batch_hazards.go`: `VulkanBufferAccess`, `LowerVulkanBatchHazards`; `vulkan_batch_window.go`; `vulkan.go`: `BeginBatch`, `FlushBatch`, `CycleBatch`. Logical read/write ranges and dependency edges are reusable. Vulkan pipeline/access masks and native lowering are not CUDA events. Unknown aliases retain the conservative barrier floor. Existing #12392 and #13950/#13951 govern submission/retirement; avoid another command-ring ticket.

19. **Neutral batch/prefill planning.** `prefill.go`, `prefill_batch.go`, `speculative_graph.go`, `decode_batchpad.go`: `PrefillGraphCapturer`, `BatchedPrefillBackend`, `SpeculativeGraphPlanner`, length-bucketed/compacted slot schedules. These mechanisms already avoid vendor policy; keep scheduler reachability, padding/work accounting and separate primary/speculative graph identities. Do not infer device batching performance from a schedule simulator.

20. **AMD transfer topology and leases.** `amd_gpudirect.go`, `amd_gpudirect_sysfs.go`, `amd_gpudirect_apu_topology.go`, `amd_gpudirect_disagg.go`, `amd_gpudirect_rdma.go`, `amd_gpudirect_rdmapool.go`, `amd_gpudirect_kvpaging*.go`, `amd_gpudirect_moe.go`: AMD nodes/fabrics, DMA-BUF/registration/queue handles, APU profiles, `KVTransferLease` and prefill/decode publication owner. Vendor discovery/AQL/DMABUF adapters are intentional. Lease state, expiry, directed connectivity and registration lifetimes can be neutral; never translate names into evidence of real DMA or actual measured bandwidth. Keep route admission separate from transport execution.

21. **NVIDIA topology is itself workstation-coupled.** `cuda_gpudirect_topology.go`: `CUDADeviceNode`, `NVMeStorageNode`, `EvaluateP2PRoute`. It infers chipset/direct CPU from `M2A_CPU`, `M2B_SB`, `0000:00:03.*`/`0000:00:01.*`, hardcodes X570 notes and numerical latency/bandwidth. That is a board heuristic, not a general discovered route. Reuse open #11321: replace it with discovered ancestor/ACS/NUMA/link facts, preserve unknown explicitly and separate estimates from observed qualification.

22. **Hierarchical memory and CUDA direct-storage models.** `hierarchical_memory.go`, `cuda_gpudirect_storage.go`, `kv_page_offload.go`, `dram_ring.go`: `HierarchicalMemoryManager`, `CUDADirectStorageMemorySlab`, `KVPageOffloader`, `DRAMStagingRing`. Generic HMM defaults to RTX5090-like capacities/synthetic addresses and a concrete CUDA slab. Block backing is Go bytes; transfers are modeled. Tier changes mutate accounting before ignoring `WriteBlock`/swap errors. Reuse open #11323 for provider injection and failure-atomic movement, with #11320/#11321 context. `DRAMStagingRing`'s write-coalescing mechanism is already neutral; its dwell/capacity are policy.

23. **CPU NUMA and host memory.** `numa_topology*.go`, `numa_replica_plan*.go`, `numa_replica_store*.go`, `numa_affinity*.go`, `decode_numapool.go`, `decode_numaschedule.go`, `decode_interleave*.go`, `farmem*.go`, `mempolicy*.go`, `hostmem*.go`. CPU topology/replica scheduling is vendor-neutral and should stay separate from GPU NUMA locality. Preserve cpuset/mempolicy constraints, per-node known bits, memory-only-node exclusion, admission before replication and platform refusal. Explicit synthesized nodes are model/test data, not observed hardware. NVIDIA affinity must attach device PCI/NUMA identity rather than infer one GPU from host nodes.

24. **Context-MMU defaults versus mechanism.** `internal/ctxmmu/allocator.go`: `SharedTokenPool`, `NewSharedTokenPoolWithConfig`, `PhysicalBlockAllocator`; `kv_pool.go`: `KVPoolConfig`, `KVPool`; `expert_residency.go`: `ExpertResidencyPolicy`. CoW/reference counts, two-phase reservations, pinning and page geometry are neutral. Default 512K tokens, 64 KiB/token, 35 GiB GTT and 6/2 GiB expert rings are Halo profiles, not universal defaults. No non-test public caller of the SharedTokenPool constructors was found in the exact-revision scan. Add an explicit validated capacity constructor before new consumers rather than silently changing legacy defaults. Closed #12191 owns the substrate; #12600 preserves checkpoint restorability. Expert policy explicitly says it is off the live serve path.

### Prioritized implementation packets

See [ticket-plan.json](ticket-plan.json) for bounded write sets, acceptance, dedup and separate hardware gates. Suggested order:

1. Fix slab ownership/release with a host regression (new narrow ticket).
2. Add CUDA complete-domain owner adapter under #10442, without editing active Vulkan/model ownership work.
3. Make HMM transfer state failure-atomic under existing #11323.
4. Correct topology evidence/unknown handling under existing #11321.
5. Separate direct-slab registration facts from synthetic configuration (follow-on to closed #10758).
6. Specify an owned, classed transfer-completion contract, then integrate ring retirement, then implement native CUDA transfer. These are separate packets; do not enable async by deleting the current guards.
7. Add a validated, explicit Context-MMU capacity constructor for new cross-device consumers.

### NVIDIA mapping guidance

- Query actual `cudaDeviceProp` capabilities (including cache, warp, managed-memory/UVA fields) rather than parse `sm_*` into a memory-capacity guess. The HAL/profile component owns this adapter. [NVIDIA device properties](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/structcudaDeviceProp.html)
- Async copies require suitable pinned/page-locked host buffers to overlap. CUDA events and stream waits provide completion/dependency semantics; event query failures are not “done.” Retain source and destination owners until that result is known. [NVIDIA asynchronous execution](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html), [event semantics](https://docs.nvidia.com/cuda/cuda-driver-api/cuda_driver_api/group__CUDA__EVENT.html)
- Stream-ordered allocation still requires all cross-stream users to finish before freeing. It cannot justify recycling an unknown-completion buffer. [NVIDIA stream-ordered allocator](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html)
- Treat UVA, mapped/page-locked host allocation and managed memory as separate facts; do not assume Halo host/device pointer identity. [NVIDIA unified and system memory](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/understanding-memory.html)

### Coverage and exclusions

Exact public source searches covered compute, model, ctxmmu and polymodel; 2,948 source/test files were materialized read-only outside the publisher checkout to support search. The 24 groups above are the reviewed memory/resource/scheduling seams, not a claim that every line of every file was audited. The bounded reviewed families and symbols are listed in this section; the catalog covered 143 file entries.

Kernel arithmetic, BF16/RoPE, format/model dispatch, build/toolchain/ops setup, and the active 24-path Vulkan ownership stack belong to parallel inventories or existing packets. Model QSA and CPU-miss Halo constants are delegated to the model slice; production device profiles/telemetry to HAL. No hosted writes, code changes in the publisher checkout, hardware jobs, purchases, device-access changes, or native execution occurred. Private-repository ownership/dedup is unverified. Public issue state was individually fetched for canonical references; search result state alone was not trusted.

## Model, session, and artifact composability inventory

Source: public FAK commit `93b295a81683f81868d1b9cc4c340496700cc0da`, inspected 2026-10-23. This is a source audit, not an executed correctness, performance, or hardware qualification. The checkout was read with `git show`, `git grep`, and tree inspection only. No checkout changes, builds, tests, GPU jobs, commits, or GitHub writes were performed.

### Outcome

The important distinction is between portable mechanisms still coupled to a vendor label and deliberate qualification boundaries. The first group should become composable; the second must remain closed until the exact backend, artifact, numerical semantics, ownership, and resource envelope are proved.

Best small follow-ons:

1. Expose the already-neutral independent Qwen decode dispatcher in non-Vulkan builds.
2. Stop supplying a Halo bandwidth/expert-size profile to every V4.1 CPU-miss session.
3. Make the model QSA scratch consumer use the existing device profile rather than 32 MiB MALL.
4. Select GGUF load estimates by actual retained/transformed/streamed storage policy, not a Vulkan flag.
5. Add neutral optional weight-memory/release reporting while retaining existing Vulkan receipts.

CUDA MTP, dense-device prefix persistence, recurrent-state ownership, quant admission, and numerical-profile intake already have owning issues. Do not create replacements for them.

### Scope and completeness

The path manifest enumerates 2,255 Go files under `internal/model` and `internal/ggufload`, plus `internal/agent/inkernel*`, `internal/agent/prefix_cache_identity.go`, and `internal/agent/chat.go`. The lexical census scans all 763 non-test Go files in that scope; relevant test witnesses were inspected selectively, not exhaustively. The census found 693 source-pinned vendor-name matching lines in 117 production files (67 with noncomment hits). A lexical hit is not a defect. Relevant capability, snapshot, loader, and owner seams were then inspected directly. This does not claim that a name scan proves the absence of all semantic hardware assumptions.

There is no `internal/agentruntime` or `pkg/agentruntime` path at this public pin. The public integration surfaces here are `internal/agent/inkernel*`. CLI, compute kernels, schedulers, fleet tooling, and external/private adapters belong to separate inventory slices. Private ownership/dedup remains unverified. Everything in this report comes from public code or public issues.

### Invariants every leaf must preserve

- FAK owns execution, memory, state, scheduling, and cache. An explicitly selected external engine may be a comparator; it is never an automatic recovery path.
- A complete prefix is not just KV: Qwen also needs convolution/recurrent state and exact token/raw-hidden lineage; V4.1 needs committed history, temporal attention, and source-publication state. Tenant/agent/epoch isolation remains intact.
- Model/artifact identity, numerical profile, backend instance/recording owner, layer placement, and state epoch must agree before reuse or transfer. A backend name is not an object-ownership identity.
- Missing capabilities decline before mutation. A selected operation failure retires/poisons its owners and does not replay through CPU or serial execution after mutation.
- Persistent weights, temporary staging, host copies, device copies, expert rings, shared UMA pools, and deferred retirement must all remain bounded and charged by their real lifetime.
- Storage format support, upload support, resident-kernel support, qualified route support, and measured hardware support are different facts.
- Device-only policy and the tool capability floor are not relaxed by portability work. Unknown hardware stays unknown; synthetic tests do not mint physical evidence.

### A. Concrete vendor coupling

#### A1. Pure model batch orchestration is hidden by Vulkan build tags

`internal/model/qwen35_batch_decode_vulkan.go:1,15-17,36-99` is guarded by `vulkan && (windows || linux) && cgo`, but its dispatcher consumes the neutral `compute.Qwen35IndependentDecodeBatchBackend` and contains no Vulkan driver call. `qwen35_batch_decode.go:3-20` uses one init-installed global hook. A CUDA-only build cannot reach this orchestration even if its backend implements the exact existing capability.

Keep B=2..8, same model/backend owner, independent KV/GDN/lineage, positive observed shared panels, zero activation D2H/serial fallback, and post-mutation terminal-failure behavior. Do not simply add a second unconditional init hook, which could shadow the existing Metal hook. Existing execution owners: [#12743](https://github.com/anthony-chaudhary/fak/issues/12743), [#12389](https://github.com/anthony-chaudhary/fak/issues/12389). A model-only portable-dispatch follow-on is distinct from a CUDA kernel implementation.

#### A2. V4.1 CPU-miss routing inherits one Halo calibration

`internal/model/v41_cpu_miss.go:12-21,37-45` defines `v41CPUMissHaloProfile` with 3.5e9 transfer B/s, 25e9 host B/s, and a 12,810,240-byte expert, then supplies it to `ExpertCPUMissPolicyFromEnv` for every eligible session. `expert_cpu_miss.go:105-120` accepts a caller profile; the underlying solver is already parameterized. The feature is default off, so this affects explicitly enabled CPU-miss sessions, not every decode.

Bind a declared/calibrated profile to the selected execution envelope and actual slab geometry. Do not treat PCIe dGPU, UMA, and coherent CPU/GPU systems as independent identical channels. Preserve exact quality, deterministic partitioning, the per-layer cap, host/device join, and lookup reconciliation. [#11038](https://github.com/anthony-chaudhary/fak/issues/11038) is the closed solver predecessor, not proof this live caller is generalized.

#### A3. Model QSA scratch bypasses the device-profile seam

`qwen35_hal.go:1077-1200`, especially `1141-1145`, uses `compute.StrixHaloInfinityCacheBytes` as a generic 32 MiB limit. This path reads query and full KV to the host and computes sparse attention on the host; it is not proof of a GPU-resident gather. The caller at `1039-1047` falls back to backend dense attention on a returned error; device-only host refusal is earlier at `1078`.

Use explicit cache/scratch facts and preserve pre-mutation decline versus selected failure. Do not claim a cache-size fact proves actual cache residency, or relabel the host path as device work. The existing compute profile is `DeviceMemoryProfile`/`DeviceMemoryProfileProvider`; its resolver is currently unexported. A small producer/resolver prerequisite belongs in compute, followed by a two-file model consumer. [#12440](https://github.com/anthony-chaudhary/fak/issues/12440) is closed for compute-policy witnesses; [#12444](https://github.com/anthony-chaudhary/fak/issues/12444) is its accidental duplicate. Neither removes this model-level constant. [#12185](https://github.com/anthony-chaudhary/fak/issues/12185) is the historical Halo QSA leaf.

#### A4. GGUF estimator selection is encoded as Vulkan

`internal/ggufload/preflight.go:70-90,235-261,296-488` exposes `VulkanMixedQ4K` and a Vulkan-only unified-pool guard. `preflight_streamed_experts.go:11-32` requires `Backend.Name()=="vulkan"` even though its payload, ring, and staging calculations describe storage policy. Reusing this estimate for CUDA is safe only when actual loader choices are the same; removing the name check alone can undercount a different CUDA transformation path.

Represent the selected retained formats, conversion policy, embedding representation, streamed-dense policy, single-residency policy, and ring bounds explicitly. Keep header-only sizing, unknown/unsupported refusals, split-checkpoint limitations, aligned staging, phase peaks, and simultaneous host/device accounting. Keep `host_residency.go`'s carve-out and shared-pool rules, rather than summing advertised apertures as physical RAM. [#12736](https://github.com/anthony-chaudhary/fak/issues/12736) is the closed lifetime-calibration predecessor.

#### A5. Generic agent memory reports require Vulkan-named methods

`internal/agent/inkernel_device_weight_memory.go:31-53` requires `VulkanWeightArenaCounters` and `VulkanOffloadWeightCounters`; its output DTO is already neutral. `inkernel_retained_release.go:17-21` similarly requires `VulkanRetainedReleaseSummary`. A non-Vulkan provider must borrow Vulkan method names or appear unsupported.

Add neutral optional reader contracts, prefer them when present, and preserve a compatibility read of the existing methods. Missing reports remain absent, not zero. Read-only on-demand release reporting must not enter token paths or take planner locks. This is distinct from [#13059](https://github.com/anthony-chaudhary/fak/issues/13059)'s CLI banner.

#### A6. MTP orchestration and receipts are Vulkan-named

`inkernel_planner_speculative.go:50-72,281-301,930-944`, `inkernel_planner.go:1688-1725`, `inkernel_mtp_diagnostic.go`, and `chat.go:511-514` expose `EnableVulkanMTP`, `VulkanMTPExecution`, and `mtp-vulkan`. The underlying draft operation in `model/qwen35_mtp_vulkan.go:54-77` is already capability-based and requires physical transfer counters.

Do not rename serialized evidence in place or label CUDA as `mtp-vulkan`. After CUDA target/draft qualification under [#9987](https://github.com/anthony-chaudhary/fak/issues/9987), add a versioned neutral execution envelope with the actual backend, retaining legacy fields only for their original semantics. This is a later three-file agent leaf, not a reason to prematurely enable CUDA.

#### A7. Shared-expert panel has an implicit Vulkan compatibility grant

`model/v41_shared_expert_resident.go:74-97` first honors `SupportsV41SharedExpertPanel(dtype,rows,out,in)` and otherwise grants a list of dtypes when the backend is named Vulkan. The optional exact-shape capability is already the composable extension seam. A backend merely named Vulkan should not be the long-term evidence of support.

Move the established Vulkan support table into an explicit backend implementation before removing compatibility admission. Preserve 1..64-row/int32-overflow bounds, weight dtype checks, BF16 deferred statuses, checked fault order, and lifetime ownership. Do not broaden support based only on `DeviceMemory` or upload capability. Coordinate with grouped-output and active BF16 work rather than bundling it.

#### A8. Model-side RDNA cache-hint DTOs carry hardware opcodes

`model/quant_forward.go:828-935` contains `DefaultMALLCapacityBlocks=128`, `CachePolicyHint{SLC,GLC,NT}`, and `ClassifyKVBlockCacheHint`, which assumes 64-token blocks and labels the first 128 blocks MALL-pinned. Session wrappers do not consult the backend. These are advisory planning/hint surfaces; their labels do not establish hardware pinning or execution.

Separate generic intent (reuse/streaming) from the RDNA translator and derive bytes from actual KV geometry/profile. Preserve old explicit RDNA descriptors as compatibility data; never reinterpret SLC/GLC/NT as CUDA cache operators. This is lower priority than the live QSA consumer. The compute/strix inventory owns the physical translator.

#### A9. Declarative registry is not a general device capability resolver

`model/registry.go:37-46,55-99` has CUDA/Metal/Vulkan presence fields, a CUDA-specific architecture comparator, and no default refusal for an unknown backend requirement. An unknown backend with empty requirements can match. `MinDeviceArch` can also be compared to `CUDAArch` for a non-CUDA requirement. Source search found use only inside the registry, not a production loader caller at this pin.

A small fail-closed registry regression/fix is justified, but do not market it as a serving correction. Longer-term multiple-device/UMA placement belongs in the separate runtime profile work, not a broad rewrite of this leaf.

### B. Deliberate qualification guards: preserve, then qualify

#### B1. V4.1 complete prefix whitelist

`kv_prefix_snapshot.go:99-130` admits only `cpu-ref` and `vulkan` for V4.1 complete backend snapshots. `config.go:1237-1272` separates bare architecture support from backend-qualified support. This is explicit physical/software qualification, not an accidental vendor conditional. `PrefixSnapshot` carries the exact backend instance, complete continuation, and owner; CUDA admission requires an actual CUDA restore/continue witness, not a `Name` edit.

The source pin itself fixes foreign state-only snapshot restoration in `kv_prefix_snapshot.go` plus `v41_prefix_artifact_identity_test.go`. Do not duplicate that owner/model fix. Keep the active numerical-profile work separate. Dense device prefix persistence remains [#13800](https://github.com/anthony-chaudhary/fak/issues/13800).

#### B2. Resident MTP target gate

`qwen35_mtp_target_vulkan.go:106-131` requires Vulkan plus retained F32/Q4_K, unsplit layer placement, sequence, all-logits, raw-hidden, and prefix-replay capabilities. `qwen35_mtp_draft.go:77-95,738-775` enforces same backend and captured lineage. Removing the name gate is insufficient even if a fake implements every interface. Decompose [#9987](https://github.com/anthony-chaudhary/fak/issues/9987) into a versioned qualification contract, CUDA implementation, and independent physical continuation/rollback witness.

#### B3. MTP rollout/canary gate

`qwen38_mtp_canary.go:289-325` admits only its certified model/format/backend envelope and requires unexpired registered evidence. The Metal/CPU vocabulary is a rollout boundary. [#12839](https://github.com/anthony-chaudhary/fak/issues/12839) owns exact evidence lookup. Capability discovery alone must not change automatic activation. [#12865](https://github.com/anthony-chaudhary/fak/issues/12865) owns MTP cross-execution-identity restore; [#12442](https://github.com/anthony-chaudhary/fak/issues/12442) owns the recurrent resource lease.

#### B4. Quant admission is route-specific, not generic upload support

`quant_kquant.go:41-86` accepts only Metal or CPU and preserves explicit native CPU weight bands. This census deliberately follows the actual Metal serving route, not `SupportsHALKQuant`. Open [#12983](https://github.com/anthony-chaudhary/fak/issues/12983) already states a cross-backend goal. Extend that owner with per-route qualified kernels, rather than trusting generic upload support or creating a second census.

`hal.go:332-358,437-446`, `quant_registry.go`, `moe_device_gateup.go`, and `artifact_quant.go:145-217` already distinguish raw storage, dense/routed operations, and device dtype qualification. Preserve the direct Q5_K/Q6_K and raw-IQ pathways; missing kernels must not become an unannounced host dequant loop.

#### B5. Generic HAL semantic restrictions

`qwen35_backend_contract.go:197-255` refuses unsupported LayerNorm, RoPE, and sliding-window semantics before load, and requires a real GDN operation plus an exact path identity. Existing owners [#12607](https://github.com/anthony-chaudhary/fak/issues/12607), [#12608](https://github.com/anthony-chaudhary/fak/issues/12608), [#12609](https://github.com/anthony-chaudhary/fak/issues/12609) must not be bypassed by a generic backend registry.

The canonical `Qwen35GDNCapabilityIdentity` already coexists with legacy CUDA/Vulkan route strings. Prefer the canonical identity for new implementations, preserve legacy receipt meaning, and fix misleading diagnostic wording only as a narrow follow-on.

#### B6. Artifact and numerical identity

`authenticated_artifact.go`, `v41_engram_binding.go`, `v41_engram_attach.go`, `v41_engram_digest_tree.go`, and `v41_prefix_artifact_identity_test.go` bind actual content/extent/ownership. `native_selection_identity.go` is only a functional route identity: it explicitly says its ModelRef is not an artifact digest. Do not substitute one for the other.

[#13893](https://github.com/anthony-chaudhary/fak/issues/13893) owns 1017 target/Engram/draft provenance and remains broader than the shipped owner fix. [#13891](https://github.com/anthony-chaudhary/fak/issues/13891) owns release/format numerical semantics. Do not infer BF16/FP8 equivalence from matching dimensions or apply an official FP8 boundary indiscriminately to GGUF K-quants. DSpark serving remains closed.

### C. Already-portable seams and historical presets

- **Execution ownership:** `backend_execution_owner.go` structurally consumes `compute.ForwardExecutionOwnerBackend`; scopes bind sessions to one recording domain and release before model-global retirement. Keep the selected-error cleanup semantics and panic identity. This needs backend implementations, not a replacement global mutex.
- **Immutable weights:** `device_weight_residency.go`, `hal.go`, `paging_ring.go`, and `expert_checkpoint_tier.go` use backend instance identity, cached immutable weights, bounded staging, and ring budgets. `SupportsBorrowedExpertUploads` is the explicit synchronous-copy lifetime contract; do not infer it from vendor or UMA.
- **Single residency:** `q4k_single_residency.go:27-40` is already gated by successful upload, all-device routing, non-lazy host storage, operator policy, and integrated topology. Generalizing topology must preserve every route/authority gate and must not free bytes a CPU fallback still needs.
- **Engram:** `v41_engram_gate_device.go:21-46` and `v41_engram_row_decode.go:239-248` use neutral optional row-decode/gate capabilities. `v41_engram_source_owner.go`, `v41_engram_lookahead.go`, and `v41_engram_prefetch.go` preserve source holds, cancellation/join, bounded panel geometry, and owned cache data. Reuse these for CUDA; do not upload the full table or introduce a duplicate cache. [#13900](https://github.com/anthony-chaudhary/fak/issues/13900) and [#13899](https://github.com/anthony-chaudhary/fak/issues/13899) already own host owned-panel/look-ahead work. Closed [#12950](https://github.com/anthony-chaudhary/fak/issues/12950) supplies callback/DMA adapter contracts, not proof of a physical CUDA graph implementation.
- **Error compatibility:** many V4.1 device bindings call `ConvertCUDAPanic`. The representative Engram projection first handles neutral BackendError, then legacy CUDA panics, then retires other plain-error/panic backends. A CUDA-named compatibility converter is not evidence that ROCm errors are ignored. Avoid a broad active-BF16/rope error-handler refactor.
- **Host swap:** `qwen35_paged_swap.go` serializes the full host hybrid continuation with checksum/geometry validation before publication. Do not transplant its host bytes into a device KV-only restore.
- **ROCmFP4:** `rocmfp4.go` identifies a real storage layout (32 E2M1 values plus one FP16 scale, 18 bytes); `fp4meta.go` separates format/scales/accumulator/claim scope. Preserve the encoding name and do not alias it to NVFP4 or MXFP4. The gfx1151 native arithmetic claim is already questioned by [#13542](https://github.com/anthony-chaudhary/fak/issues/13542). Its model metadata preset must be reconciled with that owner, not used to infer CUDA support.
- **PLE:** `qwen35.go:807-845` has a named Halo study default including a 2 GiB bound and throughput target; target geometry must come from the artifact. [#13836](https://github.com/anthony-chaudhary/fak/issues/13836) owns loader/config parity; [#13835](https://github.com/anthony-chaudhary/fak/issues/13835) owns shared PLE/HashK derivation. Do not duplicate either or turn a study target into an observed speed.
- **DSpark:** `v41_dspark_residency.go:275-355` is explicitly a pinned study with a configurable host budget, a 128 GiB Halo preset, and upstream-reported acceptance. Keep the historical preset, label it, and do not treat it as detected hardware or enable serving.
- **Trace adapters:** `expert_route_trace_analyze.go:493-530` exports a specifically Strix benchmark schema. Keep that adapter and add a separate neutral schema only for a real new consumer; do not silently reinterpret old traces.
- **CUDA-direct descriptors:** `qwen38_cudadirect_swap.go` belongs to [#11320](https://github.com/anthony-chaudhary/fak/issues/11320). It contains conceptual defaults and zero-copy labels; no device registration or zero-copy hardware qualification follows from a descriptor. It is not a reusable generic backend capability without observed transport evidence.
- **Backend-specific evidence:** `qwen35_state_identity.go` is an exact-P32 Metal observation; `native_inference_receipt.go` and `inkernel_native_receipt.go` have CUDA immutable-upload counters. Preserve these exact evidence scopes. A neutral envelope may embed them, but no cross-backend evidence should be manufactured by renaming.

## AMD, Vulkan and Halo portability inventory: tooling, build, evidence and documentation

Source baseline: public fak commit `93b295a81683f81868d1b9cc4c340496700cc0da`, inspected 2026-10-23. This is a source-inspection inventory, not a GPU qualification receipt.

### Summary

The main portability problem in this slice is not a shortage of backend names. Existing code already has CUDA, Vulkan, Metal, CPU and vendor-specific telemetry adapters. The gaps are inconsistent evidence contracts: configuration becomes “physical” hardware in HIL, all device changes can require Halo-specific validation, comparison acceptance omits comparability checks, and build documentation lags actual artifacts.

Highest-priority repair: non-Metal HIL micro-doses always execute CPU reference code, but environment variables can cause the report to claim successful physical CUDA/Vulkan execution. A three-file CPU-only repair packet is active with a dedicated implementation owner; do not start a duplicate writer.

### Boundaries and evidence status

- Read-only git show, git grep and public GitHub issue inspection only. No checkout, fetch, shared-tree edit, compilation, device probe, model load, hardware job, service mutation, or hosted write.
- Public source coverage: HIL inventory and micro-doses; CLI/backend diagnostics; environment hints; build manifests and accelerator workflows; hardware-validation routing and receipt admission; benchmark comparability; roofline and bandwidth evidence; benchmark job environment metadata; public benchmark/documentation authorities.
- HAL/kernel correctness, memory/residency/scheduler algorithms, model/session execution and private factory policies are separate inventory owners.
- Private issue ownership is unverified. No private issue enumeration or private source inspection was performed. Public findings do not contain proprietary implementation detail.
- Existing current work is reserved: #13955 / candidate 61425860 owns CUDA final no-run executable linkage; #13956 owns BF16 checked-status publication; the pending NVIDIA receipt work must not be re-created.
- Exact code links use the pinned public revision. Open-issue searches can be incomplete and issue state can change; the publication owner must recheck the exact duplicate marker and owning issue before writing.

### Inventory and disposition

#### T01. Environment hints are not executed accelerator evidence

Sources:
- [internal/hil/probe_other.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/probe_other.go#L16-L30): any nonempty CUDA_VISIBLE_DEVICES, FAK_CUDA=1, any nonempty VK_ICD_FILENAMES or FAK_VULKAN=1 sets PhysicalAvailable=true.
- [internal/hil/microdose_platform_other.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/microdose_platform_other.go#L1-L7): every non-darwin/arm64/cgo build calls runHostReferenceMicroDose.
- [internal/hil/microdose_other.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/microdose_other.go#L9-L24): CPU probes copy the supplied HardwareInfo.
- [internal/hil/microdose.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/microdose.go#L19-L58): top-level hardware/bias/guidance uses that probe result; allPassed begins true and cancellation can leave zero completed doses.

Accidental coupling: “configured API” is promoted to “physical device available” and then “executed physical GPU.” It can mislabel CUDA_VISIBLE_DEVICES=-1, an invalid ICD path, or a software Vulkan ICD. The problem applies to both AMD and NVIDIA, not just absent NVIDIA support.

Preserve: CPU reference probes remain useful, but their identity and evidence class must remain reference-only. No real CUDA/Vulkan HIL executor exists on this branch.

Disposition: new bounded corrective leaf under #12472, prerequisite to #12473. Do not duplicate #11724, which concerns a separate AMD roofline producer. New ticket draft hil-reference-execution-truthfulness in ticket-plan.json.

#### T02. LAN liveness invents Halo hardware identity

Source: [internal/hil/probe_lan.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/probe_lan.go#L175-L200). Successful /healthz after a TCP connection returns Appliance “AMD Ryzen AI MAX+ 395” and GPU “AMD Radeon 8060S (40 CUs, gfx1151, 64 GB UMA).” The API accepts an explicit hostOverride, so the same canned identity can be attached to a non-Halo service.

Intentional: the bounded zero-config Halo discovery candidate is a deployment adapter.
Accidental: liveness is treated as remote hardware attestation; the override does not make it generic.

Composable seam: keep reachability/transport independent of optional observed hardware identity. With no capability payload, report hardware unknown. If a typed capability document exists, retain its source and observation time; a hostname or healthy HTTP response is never sufficient.

Disposition: new two-file corrective leaf under #12472. #12475 owns caching/liveness monitoring and #12476 owns gateway streaming, not this false identity. hil-lan-liveness-not-hardware-identity in ticket-plan.json.

#### T03. Comparison acceptance lacks a matched experiment envelope

Sources: [ComparisonArm](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/types.go#L126-L137) and [AuditComparison](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/hil/comparison.go#L18-L85).

The function checks positive values and nonempty metric strings, then trusts IsPhysicalSilicon plus an evidence-type string. It does not require equal metrics or units, finite numbers, positive sample counts, receipt references, or a matched model/workload/cache/batch envelope. NaN passes the <=0 check. Different physical GPUs are allowed without separating a hardware comparison from an engine comparison. The output nevertheless says “Admissible for canonical benchmark comparison and issue closure.”

Intentional: cross-hardware comparisons can be useful.
Accidental: any pair of claimed physical observations becomes an achieved-win claim, even when units, statistic or workload differ.

Reuse: modelperfobs/cache_state.go already has independently resettable cache layers, typed transition evidence, before/after backend identities and explicit unproved states. modelperfobs/bandwidth.go already distinguishes theoretical versus measured roofs, host/device/transfer traffic and counter scope. Do not invent a second generic cache-state system.

Disposition: hil-comparison-compatible-evidence closes the immediate metric/units/sample/proof-reference hole in three files; a later envelope adapter can join existing cache/model/run identities. Existing #12474 is the claims-linter consumer and must depend on a trustworthy auditor, not count this as already solved. Independent execution authority remains #12381.

#### T04. Generic device validation routes to Halo

Sources:
- [internal/validate/strix.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/validate/strix.go#L50-L123): every internal/compute file, all internal/roofline files, and model files containing cuda/metal/rocm/etc. are GPU-relevant.
- The same file, lines 186-217, wires only Strix discovery, controller authority and RunStrixValidation.
- [internal/validate/hwreceipt.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/validate/hwreceipt.go#L29-L64) and lines 157-179: FAK_VALIDATE_STRIX_RECEIPT, Strix schemas, and amd-strix-validate producer are the receipt authority.

Intentional: a Halo kernel must retain exact source/archive/controller/freshness/cleanup checks and its physical Halo gate.
Accidental: GPU-relevant is a boolean and collapses backend, architecture, source-only changes, toolchain needs and actual physical targets into one Halo requirement. A CUDA-specific change cannot be qualified by Halo evidence, while a CPU-only manifest change should not launch a device job.

Disposition: existing #10934 already owns target-partitioned canary receipts. Update it with these exact source locations instead of opening another umbrella. First implement a pure typed requirement planner; preserve current enforcement until target-specific receipt adapters are wired and fail closed. Common code can require multiple target witnesses, not whichever node happens to be reachable. Receipt schema/producer/target compatibility must be explicit, not a permissive generic JSON parser.

Software witness: classifier tables for CUDA-only, Vulkan-only, Metal-only, common HAL, source-only diagnostics, docs/test-only paths, and unknown changes; forged or wrong-target receipts refused. Hardware witness later: the exact candidate on each required target with selected-device identity and no fallback. No hardware launch is authorized by this inventory.

#### T05. Build taxonomy is stale and has two authority copies

Sources:
- docs/build-variants.json and .github/build-variants.json are identical copies at the baseline.
- [LoadVariantManifest](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/buildwitness/variants.go#L53-L69) prefers docs, then .github.
- Manifest describes Vulkan only as vulkan-windows and requires an AMD Vulkan driver; it has no Linux Vulkan row. Metal says no CI witness.
- .github/workflows/release-vulkan.yml and release-artifacts.yml already build Linux Vulkan and check bundled-asset registration under Lavapipe. release-artifacts.yml already uses cgo on darwin/arm64 for Metal.
- .github/workflows/cuda-build.yml has a per-SM toolkit matrix, but baseline's go build of non-main internal/compute is not executable-link proof.

Intentional: CUDA toolkit/SM matrices, Metal platform constraints, Vulkan shader assets and driver requirements remain backend-specific adapters.
Accidental: Vulkan is treated as AMD in taxonomy; source manifests omit shipped Linux artifacts; copy fallback can drift; registration/link/runtime outcomes are narrated as one “supported” state.

Disposition: update #3712 with the small three-file manifest-consistency repair build-variant-accelerator-taxonomy-reconcile. Preserve separate pure-Go darwin compile checks versus the shipped cgo Metal artifact. #13955 exclusively owns actual CUDA final executable linkage. #11657/#11658/#11659 own dynamic loader, multi-SM artifacts and runtime hydration. #12792/#12489 own container/release qualification. No duplicate install/build frameworks or workflow-permission workaround.

#### T06. Generic roofline names contain a Halo-only policy

Sources:
- [internal/nativeperf/validator.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/nativeperf/validator.go#L250-L332): ValidateRooflineAttainment intentionally requires gfx1151, fixed 80-percent attainment and fixed logit cosine threshold.
- Lines 150-155 deserialize a nested measured roof into TheoreticalPeakBandwidthGBps. Lines 303-306 fall back to Target80PctBandwidthGBps as if it were the full denominator.
- [internal/roofline/empirical.go](https://github.com/anthony-chaudhary/fak/blob/93b295a81683f81868d1b9cc4c340496700cc0da/internal/roofline/empirical.go#L333-L424): the probe is a modeled Halo MALL/WMMA fixture, not a generic empirical NVIDIA probe. It correctly retains simulated status when hardware is merely detected.

Intentional: the original #11572 Halo campaign's target and threshold are a policy contract, not constants to delete.
Accidental: generic receipt names conceal the policy; measured and theoretical denominators are conflated; NVIDIA must not inherit Halo's bandwidth, MALL or numeric floor.

Disposition: nativeperf-explicit-roofline-policy is a software-only policy/denominator leaf preserving the legacy Halo entry point and requiring an explicit target/denominator contract for any new device. Do not assign hardware credit to a checksum or nonempty signature. #12381 owns independent execution authority. #11724 owns physical-vs-modeled producer truthfulness. Existing #13542 must also account for empirical.go's modeled FP8 WMMA ceilings and knee points; do not file a second ISA ticket.

#### T07. Telemetry adapters are reusable, but the selected device must be bound

Sources:
- internal/modelperfobs/bandwidth_collect.go:203-214 chooses an explicitly requested AMD device first, then explicitly requested NVIDIA, otherwise NVIDIA device zero before AMD.
- internal/modelperfobs/bandwidth_collect_nvidia.go:12-77 records UUID and device name; utilization.memory correctly stays controller-active percentage and is never relabeled GB/s.
- internal/modelperfobs/bandwidth_collect_nvidia_profile.go separates kernel-active-time aggregation, uninstrumented-run requirements, device/engine evidence and capture metadata.
- cmd/fak/model_observe.go:257,488 is the current standalone collection caller.
- internal/nativeperf/gpu_clocks.go provides observed/failed/unreadable status, coverage counts and numeric validation.

Intentional: nvidia-smi, ROCm/sysfs and Apple counter parsers are provider adapters; their units and missing-counter behavior differ.
Portability risk: default “device 0” in a standalone sampler is not evidence for the runtime's selected GPU, especially with CUDA_VISIBLE_DEVICES remapping, heterogeneous GPUs or MIG. This review did not establish a current serving caller that misbinds device zero, so do not file that as a proven runtime defect.

Disposition: strengthen existing #9121, #9497 and #13938 acceptance: join telemetry to actual executed backend/device/instance identity, include selector provenance and typed unavailable values, reject ambiguous joins, and keep profiler time distinct from end-to-end wall time. Existing #11318 owns Windows native telemetry fallback; HAL owner should handle it.

#### T08. Job metadata mixes API classes, architecture aliases and actual hardware

Sources:
- internal/benchcatalog/environment.go:40-48 represents GPU requirements/observations as Class + Count. This is intentionally generic and already treats missing axes as unknown, not wildcards; requirements and receipts are digest-bound.
- internal/benchloop/fleetnodes.go stores a scrubbed evidence-backed node baseline. It is historical registration, not live availability.
- cmd/fak/benchloop_fleet_dispatch.go:457-473 has machine-ID-specific CUDA command/SM selection. One H100-named catalog identifier explicitly maps to an A100 sm_80 execution host.
- tools/schemas/machine-specs.v1.json:47-56 permits arrays but its explicit accelerator metadata is NVIDIA-centric (compute_capability/cuda_version), with ambiguous memory_gb units.
- internal/hil/types.go HardwareKind mixes API (“cuda”, “vulkan”), architecture family (“strix_halo”) and CPU implementation (“cpu_simd”).

Composable seam: requested workload capabilities, logical placement alias and observed execution target are different fields. An observed target tuple should preserve vendor, device model, ISA, API/backend, driver/runtime, memory bytes and memory model, count/topology/partition where relevant, plus observation source/freshness. Missing data stays unknown. Preserve only public-safe identifiers in public artifacts; private routes, addresses and credentials remain outside them.

Disposition: no new scheduler implementation ticket from this static review. Add these exact acceptance requirements to #10934/#13938 and the owning private scheduler/control-plane work after its owner can be verified through an allowed route. Do not weaken existing Class/Count fail-closed admission while enriching its tuple. Runtime placement/residency algorithms belong to the other inventory slice.

#### T09. Backend/config diagnostics already provide useful separation

Sources: cmd/fak/backends.go; internal/runtimecap/runtimecap.go (Report, RequestedBackend and backend refusal around lines 601-623); internal/runtimecap/execution_receipt.go; internal/envconfiglint/admitted.go; cmd/fak/serve_backend_preflight.go.

Preserve:
- backends explicitly documents registry presence as not GPU completeness.
- runtimecap separates compiled/unavailable/unsupported, exact requested backend and explicit CPU degradation/remote placement.
- vendor environment variables are valid compatibility inputs. Do not rename CUDA_VISIBLE_DEVICES or VK_ICD_FILENAMES into a fake universal driver variable.
- an AMD-specific preflight message is appropriate when emitted for an actual AMD APU.

Generalization rule: record requested, effective and ignored configuration plus exact selected backend/device; use typed capability absence and declared fallback rather than name-based silent substitution. #13701/#13698 already own typed decline/environment-stamp vocabulary; reuse it instead of creating a third warning system.

#### T10. Documents and benchmark qualification must preserve scope

Authorities: BENCHMARK-AUTHORITY.md, docs/benchmarks/QWEN-PERFORMANCE-INDEX.md, docs/benchmarks/hardware-latest.json, README.md, docs/compute/vulkan-linux.md, docs/benchmarks/HARDWARE-IN-THE-LOOP-ROADMAP.md and per-run receipts.

Keep hardware-specific runbooks and historical results hardware-specific. A 64 GiB Halo result is not a 128 GiB Halo result, much less an NVIDIA result. Raw single-stream decode, aggregate batched output, speculative accepted-token rates, cache-hit latency and cold unique decode remain separate quantities.

One accepted comparison envelope needs:
1. exact source/build/backend/runtime and selected GPU identity;
2. exact model/tokenizer/quantization bytes and prompt/output work;
3. concurrency, batch width and per-stream versus aggregate units;
4. per-layer cache state with observed cold/invalidate transitions, not merely requested reset;
5. timing boundaries, repetition/statistic, warmup and profiler overhead;
6. quality/no-fallback evidence plus typed unavailable telemetry;
7. genuine execution evidence class and the applicable device/driver/environment tuple.

Build success, final linking, software-ICD registration, CPU fixture tests, physical numeric parity and workload performance qualification are six different evidence stages. A skipped device test or empty/cancelled suite earns no physical pass. A successful software fixture does not erase a missing hardware witness. Existing pending NVIDIA receipts should update the exact authority/index/README freshness contract only after acceptance, not by copying Halo claims.
