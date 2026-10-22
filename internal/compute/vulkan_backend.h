/* vulkan_backend.h — the flat C ABI between the Go cgo wrapper (vulkan.go) and the
 * Vulkan compute shim (vulkan_shim.cpp). It mirrors cuda_backend.h function-for-function
 * so vulkan.go is a near-mechanical port of cuda.go: the Go side owns the Backend
 * interface and re-checks every result; this header carries only data (device buffer
 * handles + shapes), never trust.
 *
 * The device "pointer" here is an opaque handle (a void* the shim maps to a VkBuffer +
 * its bound VkDeviceMemory), NOT a host-dereferenceable address — same contract as the
 * CUDA device pointer. All ops are f32: this is an *Approx* peer of the cpuref
 * *Reference*, held to the argmax-exact + logit-cosine gate, NOT to bit-identity (Vulkan
 * GLSL fma/reduction order differs from the model's fdot tree, which is what makes the
 * Approx classification real and honest). Quantized weight dtypes are not implemented on
 * device yet; the Go MatMul refuses them with a clear message.
 *
 * Compiled ONLY under `-tags vulkan`; the default `go build ./cmd/fak` excludes it, so the
 * shipped artifact stays one pure-Go binary. */
#ifndef FAK_VULKAN_BACKEND_H
#define FAK_VULKAN_BACKEND_H
#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>
#ifdef __cplusplus
extern "C" {
#endif

/* fvk_init creates the Vulkan instance, picks a physical device (prefers a discrete GPU,
 * falls back to the first available), creates the logical device + a compute queue, builds
 * one compute pipeline per kernel from the SPIR-V modules in spirv_dir, and reports the
 * device name (into name, up to namelen-1 chars). *is_discrete is set to 1 for a discrete
 * GPU, 0 otherwise (the Go side surfaces it in Tier so a CPU/software device — e.g. Mesa
 * lavapipe — can be told apart from the real GPU). Returns 0 on success, non-zero if no
 * Vulkan compute device is reachable or pipeline creation fails. */
int fvk_init(char *name, int namelen, int *is_discrete, const char *spirv_dir);
/* Query identity from the physical device selected by fvk_init. Values come
 * directly from VkPhysicalDeviceProperties, never process configuration. */

/* Explicit idle-only diagnostic probe. Creates one unbound VkBuffer at a time,
 * queries actual requirements and destroys it before publication. Driver host
 * allocation is possible. No VkDeviceMemory allocation, bind, submit or fence.
 * Caller holds the complete execution owner and external vulkanMu. */
#define FVK_BUFFER_REQUIREMENTS_MAX_PROBES 16
#define FVK_BUFFER_REQUIREMENTS_OK 0
#define FVK_BUFFER_REQUIREMENTS_UNAVAILABLE 1
#define FVK_BUFFER_REQUIREMENTS_INVALID 2
#define FVK_BUFFER_REQUIREMENTS_BUSY 3
#define FVK_BUFFER_REQUIREMENTS_QUARANTINED 4
#define FVK_BUFFER_REQUIREMENTS_DRIVER_FAILED 5
#define FVK_BUFFER_REQUIREMENTS_STORAGE 1
#define FVK_BUFFER_REQUIREMENTS_TRANSFER 2
typedef struct {
    uint64_t size_bytes;
    uint32_t profile;
} fvk_buffer_requirements_request;
typedef struct {
    uint64_t requested_bytes, required_bytes, alignment;
    uint32_t profile, memory_type_bits;
    uint32_t requires_dedicated, prefers_dedicated;
} fvk_buffer_requirements_row;
typedef struct {
    uint64_t device_generation;
    uint32_t count, memory_type_count;
    fvk_buffer_requirements_row rows[FVK_BUFFER_REQUIREMENTS_MAX_PROBES];
} fvk_buffer_requirements_snapshot;
/* Non-OK clears out; driver_result retains the exact create error, if any. */
int fvk_buffer_requirements_probe(const fvk_buffer_requirements_request *requests,
    uint32_t count, fvk_buffer_requirements_snapshot *out, int32_t *driver_result);

int fvk_device_identity(char *name, int namelen, uint32_t *vendor_id,
                        uint32_t *device_id, uint32_t *driver_version,
                        uint32_t *api_version);
/* Selected VkPhysicalDeviceType, or OTHER (0) when unavailable. Metadata only:
 * integrated=1, discrete=2, virtual=3, CPU=4. Used by physical qualification. */
int fvk_device_type(void);

/* Atomic capacity observation, called with the external vulkanMu held. This is
 * current allocator/heap evidence, not a future-fit quote or a reservation.
 * It does not allocate Vulkan storage, submit, fence, trim, or change owners.
 * The driver budget can change due to other processes even during this query. */
#define FVK_MEMORY_CAPACITY_MAX_HEAPS 16
#define FVK_MEMORY_CAPACITY_MAX_TYPES 32
#define FVK_MEMORY_CAPACITY_OK 0
#define FVK_MEMORY_CAPACITY_UNAVAILABLE 1
#define FVK_MEMORY_CAPACITY_INVALID 2
typedef struct {
    uint64_t size_bytes, budget_bytes, usage_bytes;
    uint32_t flags, budget_known;
} fvk_memory_capacity_heap;
typedef struct {
    uint32_t property_flags, heap_index;
} fvk_memory_capacity_type;
typedef struct {
    uint64_t device_generation;
    uint64_t limit_bytes, live_bytes, peak_bytes, refused_allocations;
    uint64_t max_buffer_bytes, max_storage_buffer_range, max_memory_allocation_size;
    uint32_t max_memory_allocation_count;
    uint32_t guard_initialized, guard_configured, guard_configuration_valid, guard_accounting_valid;
    int32_t submission_status;
    uint32_t quarantined, batch_active, inflight_active, restore_active;
    uint32_t vendor_id, device_id, driver_version, api_version, device_type;
    uint32_t heap_count, type_count;
    char device_name[256];
    fvk_memory_capacity_heap heaps[FVK_MEMORY_CAPACITY_MAX_HEAPS];
    fvk_memory_capacity_type types[FVK_MEMORY_CAPACITY_MAX_TYPES];
} fvk_memory_capacity_snapshot;
/* Returns OK only with a complete fixed-size record; other results clear out. */
int fvk_memory_capacity_snapshot_read(fvk_memory_capacity_snapshot *out);


/* Query the DRM render node of the initialized, selected physical device. Returns
 * 1 only for a supported Linux query with a valid major/minor pair; otherwise
 * returns 0 and clears supplied outputs. This is identity, not memory capacity. */
int fvk_device_drm_render_node(uint64_t *major, uint64_t *minor);

/* device memory + transfers (the residency seam). The returned handle is an opaque
 * VkBuffer wrapper; it is NOT a host pointer. */
void *fvk_malloc(size_t bytes);
/* fvk_malloc_hostvis allocates a storage buffer in HOST-VISIBLE memory directly, skipping the
 * device-local attempt fvk_malloc makes. The residency-budget path uses it to place cold
 * weights host-side DELIBERATELY (in a chosen order) instead of letting hot weights lose the
 * device-local allocation race and spill unpredictably. Same opaque handle + STORAGE_USAGE as
 * fvk_malloc; the compute path reads it the same way, just slower (over PCIe). */
void *fvk_malloc_hostvis(size_t bytes);
/* fvk_malloc_hostvis_weight is fvk_malloc_hostvis for an immutable weight; on UMA direct upload
 * a HOST_COHERENT allocation is persistently mapped and fvk_h2d writes through the mapping. */
void *fvk_malloc_hostvis_weight(size_t bytes);
void fvk_free(void *d);
void fvk_h2d(void *d, const void *h, size_t bytes);
/* fvk_d2h returns 0 only after the requested bytes reached host memory. Negative
 * values preserve VkResult failures when one is available. The positive private
 * status identifies an otherwise-unattributed staging allocation failure. */
#define FVK_D2H_STAGING_ALLOCATION_FAILED 1
int fvk_d2h(void *h, const void *d, size_t bytes);
/* V4.1-only checked D2H: flushes any batch through the checked submit path,
 * records a shader/transfer memory barrier, then checks allocation/begin/end/
 * submit/wait. A failed submit or wait quarantines resources until process exit.
 * Used for attention status and that operation's fresh device output only. */
/* Synchronous exclusive host loan: 0 MISS unchanged, 1 success in place,
 * -1 selected fatal with status. Reuses existing pool/staging backing only. */
int fvk_v41_try_pool_norm_panel(float* host, const void* gain, int rows, int width, float eps, int* status);
// probe!=0 checks existing capacity without claims; caller serializes probe/allocation/commit.
int fvk_v41_try_ffn_norm_publication(float* host, float* published, const void* gain, int rows, int width, float eps, int probe, int* status);
// Query-only: two heterogeneous existing pool buffers, unchanged capacity keys.
// Pool-only scalar KV chain; selected transport is fatal and never replayable.
int fvk_v41_try_kv_norm_publication(const float* host, float* image, const void* gain, float eps, int probe, int* status);
int fvk_v41_try_query_norm_publication(float* host, float* published, const void* gain, int rows, int width, int capacityRows, float eps, int probe, int* status);
int fvk_v41_d2h(void *h, const void *d, size_t bytes);
/* fvk_v41_d2h for count ordered, disjoint regions of one host image behind one
 * checked host fence: srcs[i] (bytes[i]) lands at host + offsets[i]. */
int fvk_v41_try_index_query_publication(float* raw, float* published, const void* source,
                                        const void* weight, const void* input, int probe, int* status);
int fvk_v41_d2h_many(void *host, size_t host_bytes, int count, void *const *srcs,
                     const size_t *offsets, const size_t *bytes);
/* One-shot deterministic failure injection for backend tests. */
void fvk_debug_d2h_staging_failure_once(int enabled);
void fvk_d2d(void *dst, const void *src, size_t bytes);
/* fvk_d2d_off copies `bytes` from src[0..] into dst at byte offset dst_off — the
 * device-resident KV append (write a new K/V row at the tail of the layer buffer). */
void fvk_d2d_off(void *dst, size_t dst_off, const void *src, size_t bytes);
/* General device->device range copy. Both handles are opaque Buffer* values; offsets are
 * byte offsets into those buffers, never pointer arithmetic on the handles themselves. */
void fvk_d2d_range(void *dst, size_t dst_off, const void *src, size_t src_off, size_t bytes);
/* Bounded V4.1 gather into fresh, disjoint destination storage. Destination
 * offsets are implicit, in input order. Repeated source regions are allowed.
 * Every region is checked before commands are recorded. Outside a batch this
 * uses one checked submission; inside it preserves the existing owner fence.
 * Returns 2 for invalid geometry, otherwise the checked submission status. */
#define FVK_V41_GATHER_MAX_REGIONS 1024
int fvk_v41_d2d_gather(void *dst, int count, void *const *srcs,
                       const size_t *src_offsets, const size_t *bytes);
void fvk_sync(void);
/* Drop all currently recycled device buffers. Safe after a session has closed and flushed. */
void fvk_trim_pool(void);
/* Drop recycled buffers when the pool grows past max_buffers; intended for token boundaries. */
void fvk_trim_pool_if_over(size_t max_buffers);
/* Debug-only introspection for backend tests. Returns the VkMemoryPropertyFlags that the
 * opaque buffer was actually allocated with, or 0 for nil. */
uint32_t fvk_debug_buffer_props(const void *d);
int fvk_debug_buffer_is_host_visible(const void *d);
int fvk_debug_buffer_is_device_local(const void *d);

/* Allocation provenance for one live opaque buffer. Requested flags are the
 * original allocator request; property_flags and heap_flags describe the actual
 * selected memory type and its heap. Neither flag set proves a disjoint physical
 * pool. host_visible_fallback records only allocBuffer's successful device-local
 * to host-visible retry, not Go-side placement/recovery decisions. */
typedef struct fvk_buffer_backing_info {
    uint32_t requested_property_flags;
    uint32_t memory_type_index;
    uint32_t property_flags;
    uint32_t heap_index;
    uint32_t heap_flags;
    int host_visible_fallback;
    int weight_arena_bound;
} fvk_buffer_backing_info;
/* Returns 1 only for a live buffer with valid selected type/heap indices in the
 * initialized backend; otherwise returns 0 and clears a supplied output. The
 * caller must keep d alive and serialize with allocation/free, like other ABI
 * calls. Indices are local to the current selected device, not persistent IDs. */
int fvk_buffer_backing(const void *d, fvk_buffer_backing_info *out);

/* One buffer binding's nominal bytes and the full VkDeviceMemory allocation
 * request backing it. reservation_bytes is allocationSize passed to Vulkan, not
 * physical residency, driver overhead, available capacity or a per-binding charge.
 * allocation_id is an opaque nonzero shim token, never a handle/address. Bindings
 * in one weight-arena block share its token and full reservation; a pooled buffer
 * retains its token while the same memory allocation is retained. */
typedef struct fvk_buffer_reservation_info {
    uint64_t buffer_bytes;
    uint64_t reservation_bytes;
    uint64_t allocation_id;
    uint64_t binding_offset;
} fvk_buffer_reservation_info;
/* Returns 1 only for complete reservation metadata; otherwise returns 0 and clears
 * a supplied output. The caller must supply a live owned Buffer (including a live
 * borrowed-view owner) and hold the same allocation/free/init lock. Arbitrary or
 * retired pointers cannot be validated. Identity is meaningful only within the
 * current initialized-device lifetime; no generation validates reinitialization.
 * The ID sequence never wraps/reuses tokens; exhaustion makes new reservations
 * unobservable without changing allocation success, fallback or pooling policy.
 * Reads host metadata only, with no Vulkan call, owner mutation or batch flush.
 * No whole-backend inventory, physical-pool disjointness or capacity sum follows.
 * Additive ABI: existing structs and symbols are unchanged; rebuild the shim. */
int fvk_buffer_reservation(const void *d, fvk_buffer_reservation_info *out);

/* Partial, pull-only inventory of native release-requested bindings. The caller
 * holds the allocation/free/init lock (Go: vulkanMu) throughout this call. No
 * Vulkan calls, fences, transfers, owner changes or allocations occur here.
 * Only deferred frees and the two native idle pools are covered; active owners,
 * Go transient pools, stages, scratch, commands and descriptors are excluded.
 * A successful empty result never proves physical release or device health.
 * Capacity and inspection are bounded by FVK_RETAINED_RELEASE_LIMIT. Inspection
 * charges both bucket visits (even empty buckets) and buffer entries. A truncated
 * empty prefix does not establish that the covered containers are empty. Order is
 * unspecified. Truncation means only a returned prefix was observed; totals
 * deduplicate backing AllocationIDs in that prefix, never the whole backend.
 * reservation_bytes is requested VkDeviceMemory bytes, not physical residency.
 * Identical binding/container duplicates are coalesced; conflicting ownership or
 * metadata fails closed. Valid-capacity rows clear on failure; status explains unavailable,
 * stale-generation, or invalid metadata. Old-generation entries are not hidden.
 * Generation identifies an initialization attempt, not a reusable logical lease.
 * logical_lease_generation is always zero (unavailable), including after pool reuse.
 * No handle is accepted or returned; this cannot validate borrowed/stale tensors.
 * Outputs must be non-overlapping, with rows pointing to capacity writable records;
 * rows may be null only at capacity zero. Capacity above the limit is invalid
 * and rows are not accessed; a null summary is invalid and outputs are untouched.
 * A rebuilt shim is required. Existing reservation ABI and allocator policy stay
 * unchanged. Quarantine supersedes pool/deferred usability; container still says
 * where the retained owner was observed. No physically-freed state is inferred. */
#define FVK_RETAINED_RELEASE_LIMIT 1024u
enum fvk_retained_snapshot_status {
    FVK_RETAINED_OK = 1, FVK_RETAINED_UNAVAILABLE = 2,
    FVK_RETAINED_INVALID = 3, FVK_RETAINED_STALE_GENERATION = 4
};
enum fvk_retained_container {
    FVK_RETAINED_DEFERRED = 1, FVK_RETAINED_DEVICE_POOL = 2,
    FVK_RETAINED_OFFLOAD_POOL = 3
};
enum fvk_retained_backing_state {
    FVK_BACKING_DEFERRED = 1, FVK_BACKING_POOL_RETAINED = 2,
    FVK_BACKING_QUARANTINED = 3
};
enum fvk_retained_release_state { FVK_RELEASE_REQUESTED = 1 };
typedef struct fvk_retained_release_record {
    fvk_buffer_reservation_info reservation;
    uint64_t device_generation;
    uint64_t logical_lease_generation; /* unavailable, always zero */
    uint32_t release_state;
    uint32_t backing_state;
    uint32_t container;
} fvk_retained_release_record;
typedef struct fvk_retained_release_summary {
    uint64_t device_generation;
    uint64_t returned_unique_reservation_bytes;
    uint32_t returned_records;
    uint32_t returned_unique_allocations;
    uint32_t truncated;
    uint32_t status;
} fvk_retained_release_summary;
int fvk_retained_release_snapshot(fvk_retained_release_record *rows, uint32_t capacity,
                                  fvk_retained_release_summary *summary);

/* V2 preserves the legacy record/function ABI above. A rebuilt shim exporting the
 * new symbol is required. All bounded-prefix/status/locking contracts still apply.
 * native_checkout_generation starts at 1 after successful native binding and
 * advances only when a native idle pool hands that Buffer out again. Zero means
 * unavailable (including exhaustion); it never wraps or restarts on a pooled move.
 * Compare only the same device_generation/allocation_id/binding_offset across
 * snapshots. This is not an expected-generation token or stale-alias validation.
 * Go transient-pool reuse is excluded, so logical_lease_generation remains zero.
 * Deferred-to-pool moves, quarantine and ordinary writes do not advance checkout. */
typedef struct fvk_retained_release_record_v2 {
    fvk_buffer_reservation_info reservation;
    uint64_t device_generation;
    uint64_t logical_lease_generation;
    uint64_t native_checkout_generation;
    uint32_t release_state;
    uint32_t backing_state;
    uint32_t container;
} fvk_retained_release_record_v2;
int fvk_retained_release_snapshot_v2(fvk_retained_release_record_v2 *rows, uint32_t capacity,
                                     fvk_retained_release_summary *summary);

/* Observe the one shim-global, persistently mapped ordinary H2D/D2H stage.
 * Returns 1 for a coherent empty owner (buffer_bytes = 0) or a retained stage;
 * returns 0 for uninitialized, inconsistent or unavailable ownership/backing.
 * Both outputs are required; supplied outputs are cleared even on failure.
 * buffer_bytes is the retained nominal buffer length, not a transfer length or
 * VkDeviceMemory reservation. No handle or persistent allocation ID is exposed.
 * The caller must serialize with all shim operations, including growth and init,
 * using the same owner lock. Valid only within the current initialized device
 * lifetime: no generation token validates an owner across shim reinitialization.
 * This reads host metadata only; it performs no Vulkan call or owner mutation,
 * does not flush a pending batch, and does not establish device quiescence.
 * It excludes restore/Q4_K staging, scratch and pools, and proves neither device
 * health nor disjoint physical memory pools. New callers require a rebuilt shim;
 * the existing fvk_buffer_backing_info layout and entry points are unchanged. */
int fvk_transfer_stage_backing(uint64_t *buffer_bytes, fvk_buffer_backing_info *out);

/* Observe the same shim-global ordinary transfer-stage owner with its complete
 * allocation reservation. Both outputs are required and cleared on every call.
 * Returns 1 with two zero records for a coherent initialized empty owner, or
 * with both complete records for a retained stage; otherwise returns 0 with
 * zero records. reservation_bytes is the full allocation request, not residency
 * or a per-receiver charge; allocation_id is the existing opaque shim token.
 * The same owner lock and current initialized-device lifetime precondition as
 * fvk_transfer_stage_backing apply. Retained metadata remains observable after
 * sticky submission failure, without proving device health or quiescence.
 * Reads host metadata only, without Vulkan calls, allocation, transfer, flush,
 * owner mutation or reinitialization validation. Existing record layouts and
 * allocator policy are unchanged; callers require a matching rebuilt shim. */
int fvk_transfer_stage_reservation(fvk_buffer_reservation_info *out, fvk_buffer_backing_info *backing);

/* y[P,out] = x[P,in] @ W[out,in]^T   (all row-major f32). */
void fvk_matmul_f32(const void *dW, const void *dX, void *dY, int out, int in, int P);
/* first argmax of x[1,in] @ W[out,in]^T without materializing the logits vector. */
int fvk_matmul_argmax_f32(const void *dW, const void *dX, int out, int in);

/* fvk_have_q8 reports 1 if the device + driver support the 8-bit-storage / int8-arithmetic
 * features the Q8 GEMM shader needs (queried at init); 0 otherwise. The Go side falls back to
 * f32 weights when this is 0, so Q8 is an optional fast path, never a correctness dependency. */
int fvk_have_q8(void);
/* Optional decode-only Q8 Gated-DeltaNet input-projection specialization. Its
 * pipeline may be absent while the generic Q8 kernels remain fully available. */
int fvk_have_qwen35_gdn_q8_in_proj(void);
int fvk_have_glm_kda_wave32(void);
int fvk_have_cooperative_matrix(void);
/* Q6_K availability is tied to the optional q6k_matmul SPIR-V module. */
int fvk_have_q6k_matmul(void);
/* Q5_K availability is tied to the optional q5k_matmul SPIR-V module. */
int fvk_have_q5k_matmul(void);
/* Q3_K availability is tied to the optional q3k_matmul SPIR-V module. */
int fvk_have_q3k_matmul(void);
/* Value-head tiling kernel availability (optional qwen35_value_head_tile module); without it the
 * tiling records device copies instead. */
int fvk_have_qwen35_value_head_tile(void);
uint32_t fvk_max_compute_work_group_count_x(void);
/* Per-resource storage-buffer cap discovered at init. fvk_max_buffer_bytes is the effective
 * single-buffer ceiling fak must respect: min(maxStorageBufferRange, maxMemoryAllocationSize)
 * when both are known, otherwise the known cap, or 0 when unknown. */
uint64_t fvk_max_buffer_bytes(void);
uint64_t fvk_max_storage_buffer_range(void);
uint64_t fvk_max_memory_allocation_size(void);
uint64_t fvk_total_device_local_memory(void);
/* Current device-local heap budget/usage from VK_EXT_memory_budget when the driver exposes
 * it. Returns 1 and fills budget/usage/free_bytes on success; returns 0 when the extension is
 * unavailable, the query is not initialized, or the driver reports no usable budget. */
int fvk_device_local_memory_budget(uint64_t *budget, uint64_t *usage, uint64_t *free_bytes);
int fvk_have_memory_budget(void);

/* Q8_0 quantized GEMM: y[P,out] = dequant(Wq[out,in], scales[out,in/32]) applied to x[P,in].
 * Wq is int8 weight codes (out*in bytes), Wscale is per-block f32 scales (out*(in/32) floats),
 * both pre-quantized and uploaded ONCE. The activation x is f32 and quantized per-block on the
 * device. This is the 4× weight-memory cut over fvk_matmul_f32 and the Q8-vs-Q8 parity path. */
void fvk_q8_matmul_f32(const void *dWcodes, const void *dWscale, const void *dX, void *dY,
                       int out, int in, int P);
void fvk_q8_matmul_2d_f32(const void *dWcodes, const void *dWscale, const void *dX, void *dY,
                          int out, int in, int P, unsigned int gridX, unsigned int gridY);
/* Two Q8_0 projections over the same f32 X in one dispatch. */
void fvk_q8_matmul2_f32(const void *dW0codes, const void *dW0scale,
                        const void *dW1codes, const void *dW1scale,
                        const void *dX, void *dY0, void *dY1,
                        int out0, int out1, int in, int P);
/* Three Q8_0 projections over the same f32 X in one dispatch. */
void fvk_q8_matmul3_f32(const void *dW0codes, const void *dW0scale,
                        const void *dW1codes, const void *dW1scale,
                        const void *dW2codes, const void *dW2scale,
                        const void *dX, void *dY0, void *dY1, void *dY2,
                        int out0, int out1, int out2, int in, int P);
/* Four Q8_0 projections over one decode activation. Returns zero when the
 * optional pipeline records successfully and a non-zero fail-closed status otherwise. */
int fvk_qwen35_gdn_q8_in_proj_f32(
    const void *dW0codes, const void *dW0scale,
    const void *dW1codes, const void *dW1scale,
    const void *dW2codes, const void *dW2scale,
    const void *dW3codes, const void *dW3scale,
    const void *dX, void *dY0, void *dY1, void *dY2, void *dY3,
    int out0, int out1, int out2, int out3, int in);
/* RMSNorm(X) followed by two Q8_0 projections over the normalized row (FFN gate/up, Q8). */
void fvk_rmsnorm_q8_matmul2_f32(const void *dW0codes, const void *dW0scale,
                                const void *dW1codes, const void *dW1scale,
                                const void *dX, const void *dNorm, void *dY0, void *dY1,
                                int out0, int out1, int in, int P, float eps);
/* RMSNorm(X) followed by Q/K/V Q8_0 projections over the normalized row (attention in, Q8). */
void fvk_rmsnorm_q8_matmul3_f32(const void *dWqcodes, const void *dWqscale,
                                const void *dWkcodes, const void *dWkscale,
                                const void *dWvcodes, const void *dWvscale,
                                const void *dX, const void *dNorm, void *dQ, void *dK, void *dV,
                                int qOut, int kOut, int vOut, int in, int P, float eps);
/* dD[P,out] += (silu(g) * u) @ Wcodes[out,in]^T with a Q8_0 down projection (FFN tail, Q8). */
void fvk_swiglu_q8_matmul_add_f32(const void *dWcodes, const void *dWscale,
                                  const void *dG, const void *dU, void *dD,
                                  int out, int in, int P);
/* first argmax of RMSNorm(x) @ W[out,in]^T without materializing normalized x or logits. */
int fvk_rmsnorm_matmul_argmax_f32(const void *dW, const void *dX, const void *dNorm,
                                  int out, int in, float eps);
/* dY[P,out] += x[P,in] @ W[out,in]^T; used to fuse projection + residual add. */
void fvk_matmul_add_f32(const void *dW, const void *dX, void *dY, int out, int in, int P);
/* Two projections over the same X in one dispatch. */
void fvk_matmul2_f32(const void *dW0, const void *dW1, const void *dX,
                     void *dY0, void *dY1, int out0, int out1, int in, int P);
/* Q/K/V projections over the same X in one dispatch. */
void fvk_matmul3_f32(const void *dWq, const void *dWk, const void *dWv, const void *dX,
                     void *dQ, void *dK, void *dV, int qOut, int kOut, int vOut, int in, int P);
/* per-row RMSNorm: y[r,:] = x[r,:] * rsqrt(mean(x[r,:]^2) + eps) * w[:]  (rows x n). */
void fvk_rmsnorm_f32(const void *dX, const void *dW, void *dY, int rows, int n, float eps);
/* RMSNorm(X) followed by one projection over the normalized row. */
void fvk_rmsnorm_matmul_f32(const void *dW, const void *dX, const void *dNorm,
                            void *dY, int out, int in, int P, float eps);
/* RMSNorm(X) followed by two projections over the normalized row. */
void fvk_rmsnorm_matmul2_f32(const void *dW0, const void *dW1, const void *dX, const void *dNorm,
                             void *dY0, void *dY1, int out0, int out1, int in, int P, float eps);
/* RMSNorm(X) followed by Q/K/V projections over the normalized row. */
void fvk_rmsnorm_matmul3_f32(const void *dWq, const void *dWk, const void *dWv,
                             const void *dX, const void *dNorm, void *dQ, void *dK, void *dV,
                             int qOut, int kOut, int vOut, int in, int P, float eps);

/* RoPE (HF non-interleaved rotate_half) on x[nHeads*headDim] at absolute position pos. */
void fvk_rope_f32(void *dX, int pos, int nHeads, int headDim, double theta);

/* One-position V4.1 tail-adjacent RoPE, with immutable F32 Q[heads,headDim],
 * KV[headDim], and table[rotaryDim/2,2] packed [sin,cos]. The two full-sized
 * outputs must not alias one another or any input. Prefix bits are copied.
 * Returns 0 after recording/completing dispatch; 1 uninitialized, 2 invalid
 * geometry/buffers/aliases, 3 unavailable pipeline, 4 failed dispatch, or the
 * negative VkResult of an observed submission failure. Pending batches still
 * require checked submission/readback. Handles must be live backend buffers.
 * New native builds require the dedicated module; absence/ABI failure refuses
 * initialization. Deploy only a coupled rebuilt archive and verified V4/complete61
 * bundle. fvk_init itself does not verify the external build receipt. */
#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
/* Experimental exact environment for the existing mandatory tail module.
 * Requires admitted loaded bytes and all three selected-device F32 controls.
 * Symbols are absent from ordinary archives; this is not physical qualification. */
int fvk_v41_tail_rope_exact_available(void);
int fvk_v41_tail_rope_exact_module_abi(const void* data, size_t bytes);
#endif
int fvk_have_v41_tail_rope_qk(void);
int fvk_v41_tail_rope_qk_f32(const void *dQ, const void *dKV,
                            void *dQOut, void *dKVOut, const void *dSinCos,
                            int heads, int headDim, int rotaryDim);

/* Strict binary32 twin of the tail kernel (registry V12/complete72 module
 * v41_tail_rope_exact.spv): same ABI, arguments and return codes as
 * fvk_v41_tail_rope_qk_f32, plus declared F32 DenormPreserve,
 * SignedZeroInfNanPreserve and RoundingModeRTE. Optional at init: loaded only
 * when the selected device reports all three controls and the strict ABI proof
 * admits the exact module bytes; otherwise it stays unavailable (return 3).
 * fvk_v41_tail_rope_strict_abi runs that proof on caller bytes, with no device. */
int fvk_have_v41_tail_rope_exact(void);
int fvk_v41_tail_rope_exact_f32(const void *dQ, const void *dKV,
                               void *dQOut, void *dKVOut, const void *dSinCos,
                               int heads, int headDim, int rotaryDim);
int fvk_v41_tail_rope_strict_abi(const void *data, size_t bytes);

/* Shared-latent sink attention for one query position. Five storage bindings:
 * Q[heads,headDim], sharedKV[selectedRows,headDim], sink[heads] (or live [1]
 * dummy when hasSink=0), out[heads,headDim], status[heads,4] uint32. All buffer
 * extents must be exact; out/status are disjoint from each other and inputs.
 * Push ABI: int32 heads, headDim, selectedRows, mode, hasSink; float32 scale.
 * Modes: plain=0, compressed=1. Status fields: stage, selectedSlotPlusOne,
 * elementPlusOne, offendingValueBits; stage 0 success, 1 score, 2 exp term,
 * 3 denominator, 4 weight, 5 value accumulator, 6 output. Zero indices mean -1.
 * Each head's record is initialized on dispatch. Positive return codes match
 * tail RoPE above; negative returns preserve checked submission VkResult.
 * Dispatch success is provisional until status and output are checked/read.
 * The module is mandatory in the rebuilt V4/complete61 archive/bundle pair;
 * absence/corruption/ABI mismatch fails initialization, with no legacy fallback.
 * Source-pinned V4 receipt verification is an explicit pre-launch gate, not
 * an automatic receipt check performed by fvk_init. */
int fvk_have_v41_shared_attention(void);
int fvk_v41_shared_attention_f32(const void *dQ, const void *dSharedKV,
                                 const void *dSink, void *dOut, void *dStatus,
                                 int heads, int headDim, int selectedRows,
                                 int mode, int hasSink, float scale);
/* Optional determinism class for the same entrypoint (registry V16/79):
 * class 0 (default) is the serial bit-exact kernel above; class 1 is the
 * run-to-run Wave32 subgroup-tree module, used only for headDim <= 512.
 * set_class returns 0, 1 not ready, 2 invalid class, 3 module unavailable.
 * FAK_V41_SHARED_ATTENTION_DETERMINISM=run-to-run selects class 1 at init. */
int fvk_have_v41_shared_attention_fast(void);
int fvk_v41_shared_attention_class(void);
int fvk_v41_shared_attention_set_class(int cls);

/* Optional V4.1 MoE router top-k, outside the trusted shader registry. Five
 * exact storage bindings: logits[rows,experts] F32, bias[experts] F32 (a [1]
 * dummy when hasBias=0), experts[rows,topK] uint32, weights[rows,topK] F32,
 * status[rows,4] uint32. Outputs are disjoint from each other and inputs.
 * Push ABI: int32 rows, experts, topK, full, hasBias; float32 routeScale.
 * Status fields: stage, expertPlusOne, offendingValueBits, 0; stage 0 success,
 * 1 logit, 2 score, 3 bias, 4 selection, 5 normalization, 6 weight. Return
 * codes match shared attention; success is provisional until status is read. */
int fvk_have_v41_router_topk(void);
int fvk_v41_router_topk_f32(const void *dLogits, const void *dBias,
                            void *dExperts, void *dWeights, void *dStatus,
                            int rows, int experts, int topK,
                            int full, int hasBias, float routeScale);

/* V4.1 router gate projection fused ahead of fvk_v41_router_topk_f32 for one
 * token row, one dispatch. Seven storage bindings: weight[experts,hidden] F32,
 * input[hidden] F32, bias as above, logits[experts] F32 scratch, experts[topK]
 * uint32, weights[topK] F32, status[4] uint32. The caller seeds status words 0
 * and 3 with UINT32_MAX (word 3 counts workgroup arrivals). Logits match the
 * one-token F32 matmul bit for bit; routing matches the router top-k. Push ABI:
 * int32 experts, hidden, topK, full, hasBias; float32 routeScale. Bounds:
 * experts <= 1024, hidden <= 65536. Return codes match the router top-k. The
 * verdict is the Go device self-check result: 0 unknown, 1 equal, 2 differed. */
int fvk_have_v41_router_gate_topk(void);
int fvk_v41_router_gate_verdict(void);
void fvk_v41_router_gate_set_verdict(int verdict);
int fvk_v41_router_gate_topk_f32(const void *dWeight, const void *dInput, const void *dBias,
                                 void *dLogits, void *dExperts, void *dWeights, void *dStatus,
                                 int experts, int hidden, int topK,
                                 int full, int hasBias, float routeScale);

/* V4.1 scalar-order indexer score. Four exact F32 storage bindings:
 * Q[heads,headDim], keys[rows,headDim], weights[heads], scores[rows]. Inputs
 * are caller-validated finite and immutable; output must not alias any input.
 * Push ABI is int32 rows, heads, headDim, offsets 0/4/8. Runtime admission needs
 * core Vulkan 1.2 binary32 denorm/signed-zero-Inf-NaN/RTE properties and matching
 * SPIR-V execution modes plus scalar NoContraction proof. Query failure or
 * unknown proof is unsupported, with a nonempty unavailable reason.
 * rows=0 accepts null key/output handles and validates live Q/weight extents
 * without allocation or dispatch. Nonempty return 0 is provisional until the
 * checked batch fence/readback completes; errors have the RoPE status meanings.
 * Arithmetic-produced Inf/NaN remain scores; downstream selection owns policy.
 * Integration requires the separate V5/complete62 source-bound build receipt;
 * this does not reinterpret frozen V4/complete61 and fvk_init verifies no receipt. */
int fvk_have_v41_indexer_score(void);
const char *fvk_v41_indexer_score_unavailable_reason(void);
int fvk_v41_indexer_score_f32(const void *dQ, const void *dKeys,
                              const void *dWeights, void *dScores,
                              int rows, int heads, int headDim);

/* Production have is the dispatch prerequisites (registry V5+ module, binary32
 * float controls, built pipeline). A separately built test archive may also
 * expose these physical-witness symbols; no runtime environment flag enables
 * them or changes the production capability result. */
#ifdef FAK_V41_INDEXER_SCORE_WITNESS
int fvk_v41_indexer_score_witness_ready(void);
int fvk_v41_indexer_score_witness_f32(const void *dQ, const void *dKeys,
                                      const void *dWeights, void *dScores,
                                      int rows, int heads, int headDim);
#endif

/* V4.1 mHC stream operations (one module, mode in push constants). Four F32
 * storage bindings A, B, C (read-only) and Out (fresh, disjoint from every
 * input); an unused input binding repeats A. coeffs rows are pre[4] | post[4] |
 * comb[16] (comb row-major [src,dst]). Modes (push ABI int32 mode, rows,
 * hidden, iters; float32 eps, rmsEps):
 *   0 initial: A=x[rows,hidden] -> Out[rows,4,hidden], four BF16 copies
 *   1 split:   A=mixes[rows,24] B=scale[3] C=base[24] -> Out=coeffs[rows,24]
 *   2 pre:     A=streams[rows,4,hidden] B=coeffs -> Out[rows,hidden], BF16
 *   3 post:    A=x[rows,hidden] B=streams C=coeffs -> Out[rows,4,hidden], BF16
 *   4 rms-split: A=raw mixes[rows,24] B=streams[rows,4,hidden] C=scale[3]|base[24]
 *              -> coeffs; mixes scaled by 1/sqrt(mean(streams^2)+rmsEps) first
 * The module is optional: fvk_have_v41_mhc is 0 when it is absent or its
 * pipeline fails, and initialization does not fail on it. Return codes match
 * tail RoPE above; return 0 is provisional until the checked readback. */
int fvk_have_v41_mhc(void);
int fvk_v41_mhc_f32(int mode, const void *dA, const void *dB, const void *dC,
                    void *dOut, int rows, int hidden, int iters, float eps, float rmsEps);

/* V4.1 one-query index selection after the indexer score. Four storage
 * bindings: scores[rows] F32 (compared as order-preserving integer keys),
 * scratch[2*blocks+rows] uint32, out[min(topK,rows)] F32 exact row integers
 * (-1 empty), status[1] uint32 (nonzero = a NaN score). One workgroup ranks
 * by score descending then position ascending, with optional candidate-block
 * masking (topKBlocks>0, blocks of blockSize, newest block pinned). Push ABI is
 * int32 rows, topK, topKBlocks, blockSize. Returns the tail RoPE status codes.
 * Optional module outside the shader registry: absence leaves have=0. */
int fvk_have_v41_index_select(void);
int fvk_v41_index_select_f32(const void *dScores, void *dScratch, void *dOut, void *dStatus,
                             int rows, int topK, int topKBlocks, int blockSize);

/* V4.1 Q2_K token-embedding row gather. Three storage bindings: the raw Q2_K
 * table [vocab, hidden] (84-byte super-blocks, GGUF order), ids[rows] uint32,
 * out[rows, hidden] F32. Only the selected rows are dequantized, with the host
 * scalar Q2_K operation order and exact f16 widening. Push ABI is int32 rows,
 * hidden, vocab. Returns the tail RoPE status codes. Optional module outside
 * the shader registry: absence leaves have=0. */
int fvk_have_v41_q2k_embed_gather(void);
int fvk_v41_q2k_embed_gather_f32(const void *dTable, const void *dIds, void *dOut,
                                 int rows, int hidden, int vocab);

/* V4.1 bit-exact RMSNorm (compressor and index-key norms). Three storage
 * bindings: x[rows,width] F32, weight[width] F32, y[rows,width] F32 (disjoint
 * from both inputs). Each row equals the host serial RMSNorm bit for bit for a
 * power-of-two width <= 65536 and a positive normal eps. Push ABI is int32 rows,
 * width, float32 eps, invWidth. Returns the tail RoPE status codes. Optional
 * module (registry V8): absence or missing binary32 float controls leave have=0. */
int fvk_have_v41_exact_rmsnorm(void);
int fvk_v41_exact_rmsnorm_f32(const void *dX, const void *dW, void *dY,
                              int rows, int width, float eps);

/* V4.1 Engram gate/add over the resident mHC panel (registry V11). Five
 * storage bindings: streams[rows,4,hidden], projection[rows,5*hidden],
 * norms[8*hidden] (qNorm | kNorm), next[rows,4,hidden] and status[rows,4], all
 * F32; both outputs are disjoint from every input and from each other. Push
 * ABI is int32 rows, hidden, pow2, float32 eps, invHidden, hiddenF,
 * invSqrtHidden. Returns the tail RoPE status codes. Optional module: absence
 * or missing binary32 float controls leave have=0. */
int fvk_have_v41_engram_gate(void);
int fvk_v41_engram_gate_f32(const void *dStreams, const void *dProjection, const void *dNorms,
                            void *dNext, void *dStatus, int rows, int hidden, float eps,
                            float invSqrtHidden);

/* V4.1 Engram table-row decode (registry V14). Three storage bindings: the
 * encoded rows (rows x 264 packed E4M3/E8M0, 84 Q2_K or 1024 F32 bytes for
 * mode 0/1/2), the F32 panel [rows,dim] and one uint32 status word per row,
 * pairwise disjoint; 1 <= dim <= 256. The poison entry writes a quiet NaN into
 * target[index] when any of the first rows status words is nonzero. Push ABI
 * is int32 mode, rows, dim, target. Returns the tail RoPE status codes.
 * Optional module like the Engram gate. */
int fvk_have_v41_engram_row_decode(void);
int fvk_v41_engram_row_decode(const void *dSrc, void *dDst, void *dStatus, int mode, int rows, int dim);
int fvk_v41_engram_row_poison(const void *dStatus, void *dTarget, int rows, int targetCount, int index);

/* V4.1 compressor gated pool (registry V11). Three storage bindings: kv and
 * scores [ratio,width] and pooled[width], all F32; the output is disjoint from
 * both inputs. 2 <= ratio <= 4096. Push ABI is int32 ratio, width. Returns the
 * tail RoPE status codes. Optional module like the Engram gate. */
int fvk_have_v41_compressor_pool(void);
int fvk_v41_compressor_pool_f32(const void *dKV, const void *dScores, void *dOut,
                                int ratio, int width);

/* V4.1 device-resident compressor ring (optional v41_compressor_ring module).
 * Stores kvRow/scoreRow [width] into ring row slot of kvRing/scoreRing
 * [ratio,width]; with emit=1 the same dispatch pools the ring with the gated
 * pool arithmetic into out [2,width] (pooled row, then per-dimension status).
 * With emit=0 out is unused. Push ABI is int32 ratio, width, slot, emit. */
int fvk_have_v41_compressor_ring(void);
int fvk_v41_compressor_ring_f32(const void *dKVRow, const void *dScoreRow, void *dKVRing,
                                void *dScoreRing, void *dOut, int ratio, int width,
                                int slot, int emit);

/* V4.1 routed-expert activation: BF16 gate/up publication, asymmetric clamp,
 * silu(g)*u, route weight and BF16 down-operand publication. Four storage
 * bindings: gate[count] F32, up[count] F32, out[count] F32 and a 16-byte status
 * (magic, count, lowest activation fault, lowest down-operand fault), all
 * pairwise disjoint. Synchronous: the status is filled with UINT32_MAX in the
 * same checked command, and an active batch is refused. Push ABI is uint32
 * count, float32 limit, weight. Returns the tail RoPE status codes. Optional
 * module (registry V10): absence or missing binary32 float controls leave have=0. */
int fvk_have_v41_routed_id(void);
/* Experimental V18 Q2_K same-weight 2/4/8-token panel. Three disjoint output
 * bindings W/X/Y, push ABI int32 out,in,tokens. No numeric status or host read:
 * the caller retains its activation/publication checks. Optional and default-off. */
int fvk_have_v41_q2k_token_pair(void);
int fvk_v41_q2k_token_pair_shape(int out, int in, int tokens);
int fvk_v41_q2k_token_pair_f32(const void* gate, const void* up, const void* input, void* gateOut, void* upOut, int out, int in, int tokens);
int fvk_have_v41_q2k_token_panel(void);
int fvk_v41_q2k_token_panel_shape(int out, int in, int tokens);
int fvk_v41_q2k_token_panel_f32(const void *dW, const void *dX, void *dY, int out, int in, int tokens);

/* V19 Q3_K token panel shares the opt-in/push ABI; weight storage is padded
 * to four bytes, matching Q3_K upload and the final uint shader load. */
int fvk_have_v41_q3k_token_panel(void);
int fvk_v41_q3k_token_panel_shape(int out, int in, int tokens);
int fvk_v41_q3k_token_panel_f32(const void *dW, const void *dX, void *dY, int out, int in, int tokens);
int fvk_v41_routed_id_max_picks(void);
int fvk_v41_matvec_id_f32(int dtype, void *const *dW, int slots, const void *dX, void *dY, const void *dIds,
                          int out, int in, int picks, int xStride);
int fvk_v41_routed_swiglu_id_f32(const void *dGate, const void *dUp, void *dOut, const void *dIds, void *dStatus,
                                 int count, int picks, float limit);
int fvk_v41_routed_publish_sum_f32(const void *dDown, void *dSum, const void *dIds, void *dStatus,
                                   int count, int picks, uint32_t site);
/* V4.1 fused routed FFN (fak-private#3341, registry V17): two dispatches per routed
 * layer instead of the id path's five. gate/up binds eight gate slot weights, eight
 * up slot weights, X, ids, mid and status (20 bindings, 20-byte push); down/sum binds
 * eight down slot weights, mid, ids, sum and status (12 bindings, 16-byte push). Each
 * stage is the id stage term for term, so every pick's output and status record track
 * the id path's. Push ABIs are (outDim,inDim,picks,xStride,limit) and
 * (outDim,midDim,site,picks). Returns the tail RoPE status codes. Optional module:
 * absence or missing binary32 float controls leave have=0. */
int fvk_have_v41_routed_fused(void);
int fvk_v41_routed_gate_up_swiglu_id_f32(void *const *dGate, void *const *dUp, int slots,
                                         const void *dX, const void *dIds, void *dMid, void *dStatus,
                                         int out, int in, int picks, int xStride, float limit);
int fvk_v41_routed_down_sum_f32(void *const *dDown, int slots, const void *dMid, void *dSum,
                                const void *dIds, void *dStatus, int out, int midDim, int picks, uint32_t site);
int fvk_have_v41_routed_swiglu(void);
int fvk_v41_routed_swiglu_f32(const void *dGate, const void *dUp, void *dOut, void *dStatus,
                              int count, float limit, float weight);
/* Experimental borrowed-panel row activation. Exact opt-in
 * FAK_V41_ROUTED_PANEL_ROW=1 and a marked optional module are required.
 * The four bindings are unchanged; push is count,rowOffset,limit,weight.
 * Gate/up contain rows*count F32 values; output contains one count-wide row.
 * Status remains one scalar-operation record with row-local fault indices.
 * No allocation ownership transfers and no host readback occurs here. */
int fvk_have_v41_routed_swiglu_panel_row(void);
int fvk_have_v41_routed_swiglu_panel_chunk(void);
int fvk_v41_routed_swiglu_panel_chunk_shape(int rows, int count, int row, int chunk);
int fvk_v41_routed_swiglu_panel_chunk_f32(const void* gate, const void* up, void* out, void* status,
                                       int rows, int count, int row, int chunk, float limit, const float* weights);
int fvk_v41_routed_swiglu_panel_chunk_module_abi(const void* data, size_t bytes);

int fvk_v41_routed_swiglu_panel_row_shape(int rows, int count, int row);
int fvk_v41_routed_swiglu_panel_row_f32(const void *dGate, const void *dUp, void *dOut, void *dStatus,
                                       int rows, int count, int row, float limit, float weight);
/* Optional marked strict F32 multiply + ordinary checked BF16 at fixed site 25.
 * Three disjoint buffers; input remains borrowed until deferred completion. */
int fvk_have_v41_index_scale_bf16(void);
int fvk_v41_index_scale_bf16_count(int count);
int fvk_v41_index_scale_bf16_f32(const void *dInput, void *dOut, void *dStatus, int count, float scale);
int fvk_v41_index_scale_bf16_module_abi(const void *data, size_t bytes);

/* Optional marked row-addressed ordinary RoutedDown (site 21) publication.
 * Input is borrowed F32 [rows,count], output is one fresh count-wide row;
 * status is one 16-byte checked BF16 record with row-local fault index.
 * No allocation ownership transfer or host readback occurs here. */
// Optional fixed-site21 2/4/8-row publication. One row-local status per row.
int fvk_have_v41_routed_down_panel_chunk(void);
int fvk_v41_routed_down_panel_chunk_shape(int rows, int count, int first, int chunk);
int fvk_v41_routed_down_panel_chunk_f32(const void *dInput, void *dOut, void *dStatus,
                                      int rows, int count, int first, int chunk);
int fvk_v41_routed_down_panel_chunk_module_abi(const void *data, size_t bytes);
int fvk_have_v41_routed_down_panel_row(void);
int fvk_v41_routed_down_panel_row_shape(int rows, int count, int row);
int fvk_v41_routed_down_panel_row_f32(const void *dInput, void *dOut, void *dStatus,
                                   int rows, int count, int row);
/* Ordinary V4.1 attention BF16 publication, registry V9/current68. The
 * unchanged integer checked module accepts only sites 3 (attention input)
 * and 4 (attention output) through this entrypoint. Input/output/status are
 * mutually disjoint; status is freshly initialized in the checked command.
 * Output stays private until checked status transport and decoding succeed. */
int fvk_have_v41_attention_bf16(void);
/* Sticky submission status, then pending=4, healthy batch/restore=2, ready=0. */
int fvk_v41_attention_bf16_state(void);
int fvk_v41_attention_bf16_count_supported(int count);
int fvk_v41_attention_bf16_f32(const void *dInput, void *dOut, void *dStatus, int count, uint32_t site);
int fvk_v41_attention_bf16_module_abi(const void *data, size_t bytes);
/* Ordinary V4.1 BF16 publication for every other full-profile boundary, on the
 * same checked module and state. Its closed site policy is final norm 1/2,
 * query norm 5/6 and 7..27; attention 3/4 stays on the entry above. */
int fvk_v41_bf16_publish_f32(const void *dInput, void *dOut, void *dStatus, int count, uint32_t site);

/* Optional raw F32 finiteness status: input is borrowed through checked
 * completion; status is a fresh disjoint 16-byte allocation owned by its caller.
 * No row output, host readback, BF16 conversion or CPU fallback is performed. */
int fvk_have_v41_f32_finite(void);
int fvk_v41_f32_finite_count_supported(int count);
int fvk_v41_f32_finite(const void *dInput, void *dStatus, int count, uint32_t site);
int fvk_v41_f32_finite_module_abi(const void *data, size_t bytes);
/* Caller holds the execution-owner mutex. Sticky quarantine of an interrupted
 * acquisition/dispatch/retirement; this never proves drain or clears a fault. */
void fvk_v41_f32_finite_abandon(void);



/* Experimental BF16 bit-copy spine. Production archives deliberately omit
 * these symbols; a witness-tagged Go build cannot link the wrong archive.
 * A dedicated reviewed module is additional to, never part of, trusted registry.
 * F32 storage [count] -> fresh F32 [count], integer RNE, two buffers/4B push. */
#ifdef FAK_V41_BF16_BOUNDARY_WITNESS
int fvk_have_v41_bf16_boundary(void);
int fvk_v41_bf16_boundary_f32(const void *dInput, void *dOut, int count);
int fvk_v41_bf16_boundary_module_abi(const void *data, size_t bytes);
/* Separate finite-only publication: three mutually disjoint buffers, 8-byte
 * count/site push and a freshly initialized 16-byte little-endian status.
 * Site 1 is final-norm input, 2 is output. Reject active batching without flush.
 * The caller withholds output until checked status D2H and decoding succeed;
 * failed output contents are unspecified. These final-norm symbols remain isolated
 * even though V9 shares the checked integer module with ordinary attention. */
int fvk_have_v41_bf16_checked(void);

/* Warm query-prefix composition. All pointers are backend-owned Buffer handles;
 * host_collapse is borrowed synchronously and never retained. No generic batch
 * is acquired or completed. Sites1/2 of the sibling operation stay unchanged. */
#define FVK_V41_QUERY_PREFIX_BUSY_BATCH 20
#define FVK_V41_QUERY_PREFIX_BUSY_RESTORE 21
typedef struct fvk_v41_query_prefix_buffers {
    const void* weights[3]; /* attention gain, wq_a, query gain */
    void* input;
    void* values[7]; /* BF16 input, norm, collapsed, projection, BF16, norm, qLat */
    void* status[4]; /* distinct 16-byte records for sites3..6 */
} fvk_v41_query_prefix_buffers;
int fvk_have_v41_query_prefix(void);
int fvk_v41_query_prefix_state(void);
int fvk_v41_query_prefix_geometry_supported(int hidden, int rank);
int fvk_v41_query_prefix_f32(const float* host_collapse,
    const fvk_v41_query_prefix_buffers* buffers, int hidden, int rank,
    float eps, uint32_t* failed_stage);

int fvk_v41_bf16_checked_count_supported(int count);
int fvk_v41_bf16_checked_f32(const void *dInput, void *dOut, void *dStatus, int count, uint32_t site);
int fvk_v41_bf16_checked_module_abi(const void *data, size_t bytes);
#endif

/* Experimental exact binary32 scalar multiply. Finite input and factor are
 * the caller precondition, including signed zeros/subnormals; finite overflow
 * to infinity is allowed. Fresh disjoint F32 [count], two buffers/8B push.
 * Ordinary archives omit these symbols and never load this witness module. */
#ifdef FAK_V41_LOGIT_SCALE_WITNESS
int fvk_have_v41_logit_scale(void);
int fvk_v41_logit_scale_count_supported(int count);
int fvk_v41_logit_scale_f32(const void *dInput, void *dOut, int count, float scale);
int fvk_v41_logit_scale_module_abi(const void *data, size_t bytes);
#endif

/* Ordinary strict twin of that scalar multiply (registry V13/complete73 module
 * v41_scale_exact.spv): same contract, arguments and return codes. Optional at
 * init: loaded only when the selected device reports binary32 DenormPreserve,
 * SignedZeroInfNanPreserve and RoundingModeRTE and the exact scalar proof
 * admits the module bytes; otherwise it stays unavailable (return 3).
 * fvk_v41_scale_exact_module_abi runs that proof on caller bytes, no device. */
int fvk_have_v41_scale_exact(void);
int fvk_v41_scale_exact_count_supported(int count);
int fvk_v41_scale_exact_f32(const void *dInput, void *dOut, int count, float scale);
int fvk_v41_scale_exact_module_abi(const void *data, size_t bytes);

/* SwiGLU: y = silu(g) * u, elementwise, length n. */
void fvk_swiglu_f32(const void *dG, const void *dU, void *dY, int n);
void fvk_swiglu_limit_f32(const void *dG, const void *dU, void *dY, int n, float limit);
/* dY[P,out] += (silu(g[P,in]) * u[P,in]) @ W[out,in]^T. */
void fvk_swiglu_matmul_add_f32(const void *dW, const void *dG, const void *dU,
                               void *dY, int out, int in, int P);

/* dst += src (length n); and dst[r,:] += bias[:] (rows x width). */
void fvk_add_f32(void *dDst, const void *dSrc, int n);
void fvk_add_bias_f32(void *dDst, const void *dBias, int rows, int width);

/* Decode FlashAttention: q[nH*hd] (one position), K/V [nPos, nKV*hd] row-major; causal by
 * construction (the cache holds exactly the attendable keys). grp = nH/nKV. out[nH*hd].
 * 4 buffers (Q, K, V, Out) with online softmax and zero global scratch allocations. */
void fvk_attention_f32(const void *dQ, const void *dK, const void *dV, void *dOut,
                       int nPos, int nH, int nKV, int hd, float scale);

/* argmax over logits[n]: returns the SMALLEST index attaining the maximum value (the
 * cpuref first-max tie-break), copied back to the host as the single scalar fence. */
int fvk_argmax_f32(const void *dLogits, int n);

/* ---- batched submission (the throughput lever) --------------------------------
 * Without batching, every primitive op submits its own command buffer and waits on a
 * fence — ~300 GPU round-trips per token, which dominates wall-clock. Between
 * fvk_batch_begin() and the next host fence, the compute ops (matmul/rmsnorm/rope/
 * swiglu/add/add_bias/attention) instead RECORD into one shared command buffer with a
 * compute->compute barrier between them, and are submitted ONCE when the batch flushes.
 * A host-fence op (fvk_d2h / fvk_argmax_f32) auto-flushes the pending batch first, so a
 * caller just brackets a token's forward pass with begin()/…/Read and pays one submit.
 * fvk_batch_flush() forces a submit+wait (idempotent if nothing is pending). Calling
 * begin() when already batching is a no-op; the model loop brackets each token. */
void fvk_batch_begin(void);
void fvk_batch_flush(void);
/* fvk_batch_cycle() closes the open batch and opens the next without waiting on it: the
 * device runs the closed batch while the host records the next, and only the batch before
 * it is retired. Every host fence, one-shot command and buffer destruction drains the
 * in-flight batch first. FAK_VULKAN_PIPELINE_FLUSH=0 restores flush+begin. */
void fvk_batch_cycle(void);
int fvk_batch_inflight(void);
bool fvk_batch_active(void);
/* ---- exact batch resource hazards (#12218) -------------------------------------
 * The default batched recorder fences every recorded dispatch against its predecessor
 * with one coarse compute->compute barrier because it has no per-dispatch hazard input.
 * fvk_batch_hazards_arm(n) selects the exact-barrier lowering: n!=0 arms it (the Go
 * lowering has already proven the window's declarations complete), n==0 forces the
 * coarse barrier for the next and every subsequent batch (fail closed). The flag is
 * process-wide and defaults to 0, so an unarmed build is byte-identical to the prior
 * behavior. fvk_batch_hazards_armed() reports the live value. */
void fvk_batch_hazards_arm(int armed);
int fvk_batch_hazards_armed(void);
/* Request retirement uses the same completion fence without counting an extra public
 * FlushBatch call when the token path already closed its batch. */
void fvk_retire_request(void);
/* Sticky VkResult from recording/submission/completion; nonzero poisons the
 * device context. Reset never clears a device failure or pending batch. */
int fvk_submission_status(void);
void fvk_submission_reset(void);
int fvk_batch_flush_status(void);
/* V4.1 step batch (#13820): begin returns 0 when it opened the step's recording
 * window, nonzero when another batch or a sticky fault refuses it. While open,
 * fvk_h2d records staged uploads, fvk_d2h/fvk_v41_d2h submit the recorded chain
 * once on the checked path and resume, status-owning kernels record into it,
 * and an operation owning its own command buffer fences it first.
 * End submits any remaining recorded work and returns the sticky status. */
int fvk_v41_step_batch_begin(void);
int fvk_v41_step_batch_end(void);
int fvk_v41_step_batch_active(void);
/* True only for a batch the V4.1 step window does not own. */
bool fvk_batch_foreign_active(void);
/* Cumulative host transfer bytes, independent of optional profiling. */
uint64_t fvk_h2d_bytes(void);
/* Subset of fvk_h2d_bytes written directly into mapped device-local weight memory (UMA). */
uint64_t fvk_direct_h2d_bytes(void);
uint64_t fvk_d2h_bytes(void);
/* Always-on cumulative device-work counters: recorded compute dispatches, D2H
 * payload bytes, and host wall nanoseconds from queue submit to fence signal. */
void fvk_device_work_counters(uint64_t *dispatches, uint64_t *d2h_bytes, uint64_t *submit_wait_ns);
/* Complete backend-owned transfer tuple. Returns 0 after accounting overflow or
 * before initialization; all six outputs are mandatory. Observed zero directions
 * are returned as zero values with success=1. */
int fvk_transfer_counters(uint64_t *h2d_count, uint64_t *h2d_bytes,
                          uint64_t *d2h_count, uint64_t *d2h_bytes,
                          uint64_t *d2d_count, uint64_t *d2d_bytes);
/* Process-owned allocations on device-local Vulkan heaps. The live output is
 * mandatory. The closing snapshot is a gauge, not a peak. begin/end creates one
 * non-nestable peak window and requires a healthy, quiescent submit context. */
int fvk_device_allocation_snapshot(uint64_t *live_bytes);
int fvk_device_allocation_window_begin(uint64_t *token);
int fvk_device_allocation_window_end(uint64_t token, uint64_t *live_bytes,
                                     uint64_t *peak_bytes);

int fvk_qwen35_split_qg_panel_f32(const void* qg, void* q, void* gate,
    int tokens, int nHeads, int headDim);
/* Tiles a [tokens, nV*headDim] GDN core panel into the GGUF ssm_out input-column order. */
int fvk_qwen35_value_head_tile_f32(const void* src, void* dst,
    int tokens, int nK, int nV, int headDim);
int fvk_qwen35_partial_rope_panel_f32(const void* q, const void* k,
    void* qOut, void* kOut, int tokens, int startPos, int nQHeads,
    int nKHeads, int headDim, int rotaryDim, double theta);
/* K/V contain prefix+tokens contiguous rows; query/output contain tokens rows. */
int fvk_qwen35_causal_attention_panel_f32(const void* q, const void* k,
    const void* v, void* out, int tokens, int prefix, int nHeads,
    int nKVHeads, int headDim, float scale);
int fvk_sigmoid_mul_f32(void* x, const void* gate, int n);

int fvk_qwen35_gdn_preprojected_f32(
    const void* mixed, const void* z, const void* beta, const void* alpha,
    const void* conv1d, const void* a_log, const void* dt_bias, const void* norm,
    void* conv_state, void* recurrent_state, void* core,
    int tokens, int conv_dim, int n_k, int n_v, int k_hd, int v_hd, int kernel, float eps);
int fvk_qwen35_gdn_conv_tiled_transpose_f32(
    const void* mixed, const void* conv1d, void* conv_state, void* conv_out,
    int tokens, int conv_dim, int kernel);
int fvk_glm_kda_step_f32(
    void* state, const void* q, const void* k, const void* value,
    const void* alpha, const void* beta, void* output, int heads, int variant);
void fvk_q4k_matmul_f32(const void *dQ4K, const void *dX, void *dY, int out, int in, int P);
void fvk_q6k_matmul_f32(const void *dQ6K, const void *dX, void *dY,
                        int out, int in, int P);
void fvk_q5k_matmul_f32(const void *dQ5K, const void *dX, void *dY,
                        int out, int in, int P);
void fvk_q3k_matmul_f32(const void *dQ3K, const void *dX, void *dY,
                        int out, int in, int P);
void fvk_rmsnorm_q4k_matmul2_f32(const void *dW0, const void *dW1, const void *dX, const void *dNorm,
                                 void *dY0, void *dY1, int out0, int out1, int in, int P, float eps);
void fvk_swiglu_q4k_matmul_add_f32(const void *dW, const void *dG, const void *dU,
                                   void *dD, int out, int in, int P);
void fvk_q2k_matmul_f32(const void *dQ2K, const void *dX, void *dY, int out, int in, int P);
/* Native IQ-quant matvec (ticket 05f); fmt is the fvk IQ format id (vulkan_iq.go).
 * mode 0: Y[t] = W0 X[t] for t < tokens; mode 1: Y += W0 x; mode 2: Y = W0 x, Y1 = W1 x. */
/* Prefill fp16 cooperative-matrix GEMM over packed weights (mm_cm1_*.spv). fmt: 0 Q2_K,
 * 1 IQ2_XXS, 2 IQ2_XS, 3 IQ2_S, 4 IQ3_XXS, 5 IQ3_S, 6 IQ1_S, 7 IQ4_XS, 8 Q4_K. fvk_mm_cm1_f32 returns FVK_MM_CM1_OK after recording Y[t] = W X[t] for every
 * t < tokens, else the status that refused the call without recording anything. */
#define FVK_MM_CM1_OK 0
#define FVK_MM_CM1_NOT_INITIALIZED 1
#define FVK_MM_CM1_NO_COOPMAT 2
#define FVK_MM_CM1_NO_SHADER_F16 3
#define FVK_MM_CM1_NO_F16_SHAPE 4
#define FVK_MM_CM1_NO_MODULE 5
#define FVK_MM_CM1_BAD_FORMAT 6
#define FVK_MM_CM1_BAD_SHAPE 7
#define FVK_MM_CM1_DISPATCH_FAILED 8
int fvk_mm_cm1_status(int fmt);
int fvk_mm_cm1_f32(int fmt, const void *dW, const void *dX, void *dY, int out, int in, int tokens);
int fvk_iq_matvec_available(int fmt);
void fvk_iq_matvec_f32(int fmt, const void *dW0, const void *dX, void *dY, const void *dW1, void *dY1,
                       int out0, int out1, int in, int tokens, int mode);
typedef struct fvk_dispatch_profile {
    uint64_t compute_dispatches;
    uint64_t q4k_matmul_dispatches;
    uint64_t q2k_matmul_dispatches;
    uint64_t other_compute_dispatches;
    uint64_t compute_barriers;
    uint64_t d2d_copies;
    uint64_t batch_submits;
    uint64_t batch_flushes;
    uint64_t one_shot_submits;
    uint64_t other_matmul_dispatches;
    uint64_t other_norm_dispatches;
    uint64_t other_rope_dispatches;
    uint64_t other_swiglu_dispatches;
    uint64_t other_add_dispatches;
    uint64_t other_attention_dispatches;
    uint64_t other_argmax_dispatches;
    uint64_t other_gdn_dispatches;
    uint64_t other_unclassified_dispatches;
    uint64_t one_shot_compute_submits;
    uint64_t one_shot_h2d_submits;
    uint64_t one_shot_d2h_submits;
    uint64_t one_shot_d2d_submits;
} fvk_dispatch_profile;

void fvk_dispatch_profile_snapshot(fvk_dispatch_profile *out);
void fvk_dispatch_profile_reset(void);

/* GPU timestamp profile (FAK_VULKAN_TIMESTAMP_PROFILE=1). Rows are cumulative per
 * (kid, groups) shape; untimed_* count submits outside a timed batch and
 * barriers_elided counts dispatches recorded without their barrier. */
typedef struct fvk_ts_row {
    uint32_t kid, gx, gy, pad;
    uint64_t calls;
    double ns;
} fvk_ts_row;
typedef struct fvk_ts_totals {
    int32_t enabled, usable, init_tried, pad;
    uint64_t batches, ops, truncated, untimed_compute, untimed_transfer, barriers_elided;
    double gpu_ns, wall_ns;
} fvk_ts_totals;
int fvk_ts_enabled(void);
int fvk_ts_snapshot(fvk_ts_totals *tot, fvk_ts_row *rows, int cap);
const char *fvk_ts_kernel_name(uint32_t kid);

#ifdef __cplusplus
}
#endif
#endif
