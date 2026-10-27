//go:build ignore

// vulkan_shim.cpp — the Vulkan compute hardware seam behind the typed compute.Backend.
//
// Compiled offline (clang++) into a static/shared lib that the cgo wrapper (vulkan.go,
// //go:build vulkan) links. It mirrors cuda_kernels.cu function-for-function: every op is
// f32, and this is an *Approx* peer of the cpuref *Reference* — held to the argmax-exact +
// logit-cosine gate, NOT to bit-identity. GLSL fma/reduction order differs from the model's
// fdot tree, which makes the Approx classification honest.
//
// Design (correctness-first, like CUDA's first cut):
//   - one VkInstance / VkPhysicalDevice (prefer DISCRETE_GPU) / VkDevice / compute VkQueue.
//   - device memory = a VkBuffer + bound VkDeviceMemory; the opaque handle the C ABI hands
//     to Go is a Buffer* (NOT a host pointer), matching the CUDA device-pointer contract.
//   - one VkPipeline per kernel, built from the SPIR-V modules in spirv_dir at init.
//   - every op records a one-shot command buffer, submits, and waits on a fence — the
//     synchronous Ready()==true model. A buffer pool recycles allocations (cudaMalloc is
//     slow; so is vkAllocateMemory) so steady-state decode pays ~zero alloc cost.
//   - all entry points are serialized by the Go-side vulkanMu mutex, so the single command
//     pool + queue need no internal locking.
//
// The default `go build` excludes this; only `-tags vulkan` links it.

#include "vulkan_backend.h"
#include "vulkan_memory_guard.h"
#include "vulkan_spirv_admission.h"

using fak::vulkan::spirv::swigluPushConstantABI;
using fak::vulkan::spirv::v41TailRoPEABI;
using fak::vulkan::spirv::v41SharedAttentionABI;
using fak::vulkan::spirv::v41IndexerScoreABI;
using fak::vulkan::spirv::v41BF16BoundaryABI;
using fak::vulkan::spirv::v41BF16CheckedABI;
using fak::vulkan::spirv::v41F32FiniteABI;
using fak::vulkan::spirv::v41LogitScaleABI;
using fak::vulkan::spirv::spirvModuleFraming;
using fak::vulkan::spirv::v41TokenGridModuleABI;

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>
#include <unordered_map>
#include <atomic>
#include <algorithm>
#include <chrono>
#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

struct DispatchProfileCounters {
    std::atomic<uint64_t> compute{0}, q4k{0}, q2k{0}, other{0};
    std::atomic<uint64_t> otherMatmul{0}, otherNorm{0}, otherRope{0}, otherSwiGLU{0};
    std::atomic<uint64_t> otherAdd{0}, otherAttention{0}, otherArgmax{0}, otherGDN{0}, otherUnclassified{0};
    std::atomic<uint64_t> barriers{0}, d2d{0}, batchSubmits{0}, batchFlushes{0}, oneShotSubmits{0};
    std::atomic<uint64_t> barriersElided{0};
    std::atomic<uint64_t> oneShotCompute{0}, oneShotH2D{0}, oneShotD2H{0}, oneShotD2D{0};
};
static DispatchProfileCounters g_dp;
static const bool g_dp_on = [] { const char* v = std::getenv("FAK_VULKAN_DISPATCH_PROFILE"); return v && v[0] == '1' && v[1] == '\0'; }();
// One startup authority for native dispatch and Go upload attribution.
extern "C" int fvk_dispatch_profile_enabled(void) {
    return g_dp_on ? 1 : 0;
}
static inline void dp_inc(std::atomic<uint64_t>& v) { if (g_dp_on) v.fetch_add(1, std::memory_order_relaxed); }
// Always-on device-work counters, independent of FAK_VULKAN_DISPATCH_PROFILE: every
// recorded compute dispatch, and host wall from queue submit to fence signal.
static std::atomic<uint64_t> g_deviceDispatches{0};
static std::atomic<uint64_t> g_submitWaitNs{0};
// Every batch and one-shot queue submit, also profile-independent; with the profile on
// it moves exactly with batch_submits + one_shot_submits.
static std::atomic<uint64_t> g_deviceSubmits{0};
static inline void dpBatchSubmit() { g_deviceSubmits.fetch_add(1, std::memory_order_relaxed); dp_inc(g_dp.batchSubmits); }
uint64_t fvk_device_submits(void) { return g_deviceSubmits.load(std::memory_order_relaxed); }

static inline uint64_t submitWaitNow() {
    return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
// Always-on host-sync accounting (writers hold vulkanMu; Go reads the atomics lock-free).
// g_deviceIdleClock is one word so a lock-free read is coherent: bit 0 set means idle
// since origin value>>1, clear means value>>1 idle ns booked so far.
static std::atomic<uint64_t> g_deviceIdleClock{0};
static uint32_t g_syncOutstanding = 0;
// Pipelined batches submitted without a host wait, and drained. Between the two the
// device may finish unobserved, so the idle clock misses that idle.
static std::atomic<uint64_t> g_syncPipeSubmits{0}, g_syncPipeDrains{0};
// Last compute kernel recorded per path since that path's last host sync.
static uint32_t g_syncBatchStage = UINT32_MAX, g_syncOneShotStage = UINT32_MAX;

#define VKCHECK(call) do { VkResult _r = (call); if (_r != VK_SUCCESS) { \
  fprintf(stderr, "fak-vulkan: %s:%d VkResult=%d\n", __FILE__, __LINE__, (int)_r); abort(); } } while (0)

// ---- global Vulkan state --------------------------------------------------------

// forward decl: batchFlush (in the anonymous namespace) recycles buffers freed mid-batch
// via the C-ABI fvk_free defined far below. Declaring it here keeps the call well-formed.
extern "C" void fvk_free(void* d);

namespace {

VkInstance        g_instance = VK_NULL_HANDLE;
VkPhysicalDevice  g_phys     = VK_NULL_HANDLE;
VkDevice          g_dev      = VK_NULL_HANDLE;
VkQueue           g_queue    = VK_NULL_HANDLE;
uint32_t          g_qfam     = 0;
VkCommandPool     g_cmdpool  = VK_NULL_HANDLE;
VkFence           g_submitFence = VK_NULL_HANDLE;
VkPhysicalDeviceMemoryProperties g_memprops{};
bool              g_ready    = false;
VkDeviceSize      g_maxStorageBufferRange = 0;
VkDeviceSize      g_maxMemoryAllocationSize = 0;
VkDeviceSize      g_maxBufferBytes = 0;
uint32_t          g_maxComputeWorkGroupCountX = 0;
uint32_t          g_maxComputeWorkGroupCountY = 0;
bool              g_v41Q3TokenGridModule = false;
bool              g_v41Q2TokenGridModule = false;
VkDeviceSize      g_totalDeviceLocalMemory = 0;
bool              g_haveMemoryBudget = false;
bool              g_batching = false;
// V4.1 and restore submissions opt into checked command/fence calls. A failed submit
// or wait leaves resources quarantined for this process lifetime;
// there is no native context-recovery/teardown API that could safely recycle them.
bool              g_v41SubmissionPendingFailure = false;
void              batchBegin();
void              batchFlush();
VkResult          g_submissionStatus = VK_SUCCESS;
std::atomic<uint64_t> g_h2dBytes{0};
std::atomic<uint64_t> g_d2hBytes{0};
std::atomic<uint64_t> g_h2dCount{0};
std::atomic<uint64_t> g_d2hCount{0};
std::atomic<uint64_t> g_d2dCount{0};
std::atomic<uint64_t> g_d2dBytes{0};
// Weight bytes written straight into a mapped DEVICE_LOCAL|HOST_VISIBLE arena (no staging copy).
std::atomic<uint64_t> g_directH2DBytes{0};
// UMA (integrated GPU, e.g. a carve-out APU): weights go to a device-local, host-visible memory
// type and are written through the mapping, the llama.cpp ggml-vulkan UMA path. FAK_VULKAN_UMA_DIRECT
// overrides the device-type default (0/off disables, 1/on forces).
bool g_umaDirectWeights = false;
bool                  g_transferCountersValid = true;
uint64_t              g_deviceAllocationLiveBytes = 0;
bool                  g_deviceAllocationAccountingValid = true;
bool                  g_allocationWindowActive = false;
uint64_t              g_allocationWindowToken = 0;
uint64_t              g_nextAllocationWindowToken = 1;
uint64_t              g_allocationWindowPeakBytes = 0;

// Tokens name VkDeviceMemory allocations, not Buffer wrappers, arena indices or
// addresses. Retention/pooling keeps the token; a new allocation never reuses it.
// No cross-device-lifetime contract is provided. Exhaustion affects metadata only.
// Observation identity only. Init attempts invalidate old observation generations;
// exhaustion disables this inventory rather than wrapping or changing allocation policy.
uint64_t g_memoryDeviceGeneration = 0;
bool g_memoryDeviceGenerationReady = false;
void beginMemoryDeviceGeneration() {
    g_memoryDeviceGenerationReady = false;
    static bool exhausted = false;
    if (exhausted || g_memoryDeviceGeneration == std::numeric_limits<uint64_t>::max()) {
        exhausted = true;
        g_memoryDeviceGeneration = 0;
    } else {
        ++g_memoryDeviceGeneration;
    }
}
uint64_t g_nextMemoryAllocationID = 1;
uint64_t nextMemoryAllocationID() {
    uint64_t id = g_nextMemoryAllocationID;
    if (id != 0) {
        g_nextMemoryAllocationID = id == std::numeric_limits<uint64_t>::max() ? 0 : id + 1;
    }
    return id;
}

bool checkedCounterAdd(std::atomic<uint64_t>& counter, uint64_t value) {
    uint64_t current = counter.load(std::memory_order_relaxed);
    if (value > std::numeric_limits<uint64_t>::max() - current) {
        g_transferCountersValid = false;
        return false;
    }
    counter.store(current + value, std::memory_order_relaxed);
    return true;
}

bool environmentFlagIs(const char* name, char flag) {
#ifdef _WIN32
    char value[2]{};
    DWORD length = GetEnvironmentVariableA(name, value, (DWORD)sizeof(value));
    return length == 1 && value[0] == flag;
#else
    const char* value = std::getenv(name);
    return value && value[0] == flag && value[1] == '\0';
#endif
}

bool environmentFlagEnabled(const char* name) { return environmentFlagIs(name, '1'); }

// CPU reads of uncached GTT staging run near 1 GB/s on Strix Halo, which made
// V4.1 decode readback ~43 ms/step; cached is the default and =0 opts out.
const bool g_preferCachedTransferStage =
    !environmentFlagIs("FAK_VULKAN_STAGE_HOST_CACHED", '0');

VulkanMemoryGuard g_memoryGuard;

bool configureMemoryGuard() {
    return g_memoryGuard.configure(std::getenv("FAK_VULKAN_MEMORY_LIMIT_BYTES"));
}

VkResult allocateGuardedMemory(const VkMemoryAllocateInfo* info, VkDeviceMemory* memory) {
    if (!configureMemoryGuard() || !g_memoryGuard.reserve(info->allocationSize)) {
        *memory = VK_NULL_HANDLE;
        return VK_ERROR_OUT_OF_DEVICE_MEMORY;
    }
    VkResult result = vkAllocateMemory(g_dev, info, nullptr, memory);
    if (result != VK_SUCCESS) {
        g_memoryGuard.release(info->allocationSize);
        *memory = VK_NULL_HANDLE;
    } else {
        g_memoryGuard.committed();
    }
    return result;
}

void freeGuardedMemory(VkDeviceMemory memory, VkDeviceSize bytes) {
    vkFreeMemory(g_dev, memory, nullptr);
    g_memoryGuard.release(bytes);
}

void reportMemoryGuard() {
    if (g_memoryGuard.configured && g_memoryGuard.configurationValid) {
        fprintf(stderr, "fak-vulkan-memory-guard: {\"scope\":\"all_vk_device_memory\",\"limit_bytes\":%llu,\"live_bytes\":%llu,\"peak_bytes\":%llu,\"refused_allocations\":%llu,\"configured\":true,\"configuration_valid\":true,\"accounting_valid\":%s}\n",
            (unsigned long long)g_memoryGuard.limit, (unsigned long long)g_memoryGuard.live,
            (unsigned long long)g_memoryGuard.peak, (unsigned long long)g_memoryGuard.refused,
            g_memoryGuard.accountingValid ? "true" : "false");
    }
}

bool memoryTypeUsesDeviceLocalHeap(uint32_t memoryType) {
    if (memoryType >= g_memprops.memoryTypeCount) return false;
    uint32_t heap = g_memprops.memoryTypes[memoryType].heapIndex;
    return heap < g_memprops.memoryHeapCount &&
        (g_memprops.memoryHeaps[heap].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) != 0;
}

void trackDeviceAllocation(VkDeviceSize bytes, uint32_t memoryType) {
    if (!memoryTypeUsesDeviceLocalHeap(memoryType)) return;
    uint64_t allocationBytes = static_cast<uint64_t>(bytes);
    if (!g_deviceAllocationAccountingValid ||
        allocationBytes > std::numeric_limits<uint64_t>::max() - g_deviceAllocationLiveBytes) {
        g_deviceAllocationAccountingValid = false;
        return;
    }
    g_deviceAllocationLiveBytes += allocationBytes;
    if (g_allocationWindowActive && g_deviceAllocationLiveBytes > g_allocationWindowPeakBytes) {
        g_allocationWindowPeakBytes = g_deviceAllocationLiveBytes;
    }
}

void untrackDeviceAllocation(VkDeviceSize bytes, bool deviceLocalHeap) {
    if (!deviceLocalHeap || !g_deviceAllocationAccountingValid) return;
    uint64_t allocationBytes = static_cast<uint64_t>(bytes);
    if (allocationBytes > g_deviceAllocationLiveBytes) {
        g_deviceAllocationAccountingValid = false;
        return;
    }
    g_deviceAllocationLiveBytes -= allocationBytes;
}

// A device buffer: VkBuffer + its memory + byte size. The opaque handle Go holds is a
// Buffer* — never a host address.
struct Buffer {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t         bytes = 0;
    VkMemoryPropertyFlags props = 0;
    // Keep provenance separate from props: existing placement/pooling uses that
    // historical policy field, which need not contain every selected-type flag.
    VkMemoryPropertyFlags requestedProps = 0;
    uint32_t       memoryTypeIndex = UINT32_MAX;
    bool           hostVisibleFallback = false;
    VkDeviceSize   memoryOffset = 0;
    bool           weightArenaBound = false;
    size_t         weightArenaBlock = std::numeric_limits<size_t>::max();
    VkDeviceSize   allocationBytes = 0;
    uint64_t       allocationID = 0;
    uint64_t       deviceGeneration = 0;
    uint64_t       nativeCheckoutGeneration = 0; // diagnostic native checkout only; zero unavailable
    bool           allocationDeviceLocal = false;
    void*          mapped = nullptr; // host view of a direct-upload weight arena binding
    bool           ownsMapping = false; // mapped is this buffer's own map of mem, unmapped on destroy
    bool           offloadPage = false; // mapped host-visible expert-ring page, recycled via g_hvOffloadPool
    bool           offloadPristine = false; // handed out by fvk_malloc_offload_weight, not written since
};

// Called only at native pool checkout, under the existing external allocator lock.
// Exhaustion is sticky unavailable rather than wrapping into an old checkout ID.
// Go transient-pool ownership changes do not pass through this seam.
static void advanceNativeCheckoutGeneration(Buffer* b) {
    if (b->nativeCheckoutGeneration != 0) {
        b->nativeCheckoutGeneration = b->nativeCheckoutGeneration == std::numeric_limits<uint64_t>::max()
            ? 0 : b->nativeCheckoutGeneration + 1;
    }
}

// Immutable model weights keep their descriptor-visible VkBuffer identity while sharing a
// bounded set of VkDeviceMemory blocks. This is the issue-local #12217 adapter pending the
// generic #11096 VMM contract: bump-only offsets are never individually reused, and the blocks
// are released only after the last referencing buffer is destroyed. Vulkan calls are externally
// serialized by vulkanMu, matching VMA virtual-block's externally synchronized contract.
static constexpr VkDeviceSize WEIGHT_ARENA_BLOCK_BYTES = 256ull * 1024ull * 1024ull;
struct WeightArenaBlock {
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkDeviceSize capacity = 0;
    uint64_t allocationID = 0;
    uint64_t deviceGeneration = 0;
    VkDeviceSize used = 0;
    uint32_t memoryTypeIndex = UINT32_MAX;
    size_t liveBuffers = 0;
    bool allocationDeviceLocal = false;
    void* mapped = nullptr; // persistent map of a host-visible block (UMA direct upload)
};
std::vector<WeightArenaBlock> g_weightArena;
uint64_t g_weightArenaMemoryAllocations = 0;
uint64_t g_weightArenaBufferBindings = 0;
uint64_t g_weightArenaReservedBytes = 0;
uint64_t g_weightArenaLiveBytes = 0;
uint64_t g_weightArenaPeakReservedBytes = 0;

void releaseWeightArena() {
    if (g_v41SubmissionPendingFailure) return;
    for (const WeightArenaBlock& block : g_weightArena) {
        if (block.liveBuffers != 0) return;
    }
    for (WeightArenaBlock& block : g_weightArena) {
        if (block.mem) {
            untrackDeviceAllocation(block.capacity, block.allocationDeviceLocal);
            freeGuardedMemory(block.mem, block.capacity);
        }
    }
    g_weightArena.clear();
    g_weightArenaReservedBytes = 0;
}

// size-bucketed free list for device-local buffers (mirrors the CUDA g_pool/g_live arena).
// Host-visible buffers are deliberately not pooled here: the residency-budget path may
// allocate cold weights host-visible, and reusing one as a later device-local tensor would
// silently downgrade residency.
std::unordered_map<size_t, std::vector<Buffer*>> g_pool;
size_t g_poolCount = 0;
// UMA with a small VRAM carve-out: expert-ring pages live in mapped host-visible (GTT) memory.
// DEVICE_LOCAL pages there fill the carve-out first, and once it is full the kernel cannot place
// GPU page tables, so a large ring dies in amdgpu_gem_va_update_vm with -ENOMEM.
bool g_offloadHostVisible = false;
std::unordered_map<size_t, std::vector<Buffer*>> g_hvOffloadPool;
uint64_t g_offloadHostVisiblePages = 0;

// Reusable HOST_VISIBLE transfer staging. H2D/D2H are serialized by the Go-side mutex and
// submit synchronously here, so one persistently mapped buffer is enough; grow it on demand.
Buffer* g_stage = nullptr;
void*   g_stageMapped = nullptr;
size_t  g_stageCap = 0;

// Restore uses a separate, strictly bounded staging allocation, never shared
// with ordinary H2D/D2H. Cleanup destroys it only when no submission is uncertain;
// otherwise it stays quarantined until process exit.
struct RestoreTransaction {
    Buffer*        stage = nullptr;
    void*          mapped = nullptr;
    size_t         cap = 0;
    size_t         maxEntries = 0;
    size_t         used = 0;
    size_t         entries = 0;
    size_t         payloadBytes = 0;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    int            successfulSubmits = 0;
};
RestoreTransaction g_restore;
int g_restoreFailAfterSubmits = -1;

// One compute kernel: pipeline + layout + descriptor set layout + how many storage buffers
// it binds + push-constant byte size.
// Matches shaders/attention.comp. Registration and dispatch share this ABI.
struct AttentionPush {
    int nPos; int nH; int nKV; int hd; float scale; int mode; int tileCount;
    int causal; int windowSize;
};
static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(AttentionPush) == 36,
              "decode attention push ABI must contain nine 32-bit scalars");
static_assert(offsetof(AttentionPush, nPos) == 0 && offsetof(AttentionPush, nH) == 4 &&
              offsetof(AttentionPush, nKV) == 8 && offsetof(AttentionPush, hd) == 12 &&
              offsetof(AttentionPush, scale) == 16 && offsetof(AttentionPush, mode) == 20 &&
              offsetof(AttentionPush, tileCount) == 24 && offsetof(AttentionPush, causal) == 28 &&
              offsetof(AttentionPush, windowSize) == 32, "decode attention push offsets changed");

struct Kernel {
    VkShaderModule        shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl    = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipe   = VK_NULL_HANDLE;
    int                   nbuf   = 0;
    uint32_t              pcsize = 0;
};

enum KId { K_MATMUL, K_MATMUL_ADD, K_MATMUL_ARGMAX, K_MATMUL_ARGMAX_BLOCKS, K_MATMUL2, K_MATMUL3, K_RMSNORM, K_RMSNORM_MATMUL, K_RMSNORM_MATMUL2, K_RMSNORM_MATMUL3, K_RMSNORM_MATMUL_ARGMAX_BLOCKS, K_ROPE, K_SWIGLU, K_SWIGLU_MATMUL_ADD, K_ADD, K_ADD_BIAS, K_ATTENTION, K_ARGMAX, K_ARGMAX_PAIRS, K_Q8_MATMUL, K_Q8_MATMUL_DECODE, K_Q8_MATMUL2, K_Q8_MATMUL3, K_RMSNORM_Q8_MATMUL2, K_RMSNORM_Q8_MATMUL3, K_SWIGLU_Q8_MATMUL_ADD, K_QWEN35_GDN_Q8_IN_PROJ, K_QWEN35_GDN_CONV, K_QWEN35_GDN_RECURRENT, K_QWEN35_GDN_PREFILL_TILED, K_QWEN35_GDN_PREFILL_NORM, K_QWEN35_GDN_VERIFY_TILED, K_GLM_KDA_REREAD, K_GLM_KDA_WAVE32, K_Q4K_MATMUL, K_Q4K_MATMUL_WAVE32, K_Q4K_MATMUL_COOPMAT, K_Q6K_MATMUL, K_Q5K_MATMUL, K_Q3K_MATMUL, K_RMSNORM_Q4K_MATMUL2, K_SWIGLU_Q4K_MATMUL_ADD, K_Q2K_MATMUL, K_RMSNORM_Q2K_MATMUL2, K_QWEN35_SPLIT_QG_PANEL, K_QWEN35_PARTIAL_ROPE_PANEL, K_QWEN35_CAUSAL_ATTENTION_PANEL, K_ATTENTION_SPLIT_KV, K_SIGMOID_MUL, K_Q2K_MATVEC, K_RMSNORM_Q8_MATMUL2_COOP, K_IQ4XS_MATVEC, K_IQ3XXS_MATVEC, K_IQ2S_MATVEC, K_IQ3S_MATVEC, K_IQ2XXS_MATVEC, K_IQ2XS_MATVEC, K_IQ1S_MATVEC, K_V41_TAIL_ROPE_QK, K_V41_SHARED_ATTENTION, K_V41_INDEXER_SCORE, K_V41_MHC, K_V41_ROUTER_TOPK, K_V41_INDEX_SELECT, K_V41_Q2K_EMBED_GATHER, K_V41_EXACT_RMSNORM, K_V41_BF16_BOUNDARY, K_V41_LOGIT_SCALE, K_V41_BF16_CHECKED, K_V41_ROUTED_SWIGLU, K_V41_ENGRAM_GATE, K_V41_COMPRESSOR_POOL, K_V41_TAIL_ROPE_EXACT, K_V41_SCALE_EXACT, K_V41_ENGRAM_ROW_DECODE, K_V41_Q2K_MATVEC_ID, K_V41_Q3K_MATVEC_ID, K_V41_ROUTED_SWIGLU_ID, K_V41_ROUTED_PUBLISH_SUM, K_V41_SHARED_ATTENTION_FAST, K_V41_ROUTED_GATE_UP_SWIGLU_ID, K_V41_ROUTED_DOWN_SUM, K_V41_Q2K_TOKEN_PANEL, K_V41_Q3K_TOKEN_PANEL, K_V41_ROUTED_SWIGLU_PANEL_ROW, K_V41_ROUTED_DOWN_PANEL_ROW, K_V41_INDEX_SCALE_BF16, K_V41_ROUTED_SWIGLU_PANEL_CHUNK, K_V41_ROUTED_DOWN_PANEL_CHUNK, K_V41_Q2K_TOKEN_PAIR, K_MM_CM1_Q2K, K_MM_CM1_IQ2XXS, K_MM_CM1_IQ2XS, K_MM_CM1_IQ2S, K_MM_CM1_IQ3XXS, K_MM_CM1_IQ3S, K_MM_CM1_IQ1S, K_MM_CM1_IQ4XS, K_MM_CM1_Q4K, K_V41_F32_FINITE, K_QWEN35_VALUE_HEAD_TILE, K_V41_ROUTER_GATE_TOPK, K_V41_COMPRESSOR_RING, K_V41_SHARED_ATTENTION_SPLIT, K_COUNT };
Kernel g_kern[K_COUNT];

// Every non-Q4_K/Q2_K kernel belongs to exactly one primary operation family. Fused
// kernels follow their public operation prefix so these fields sum to `other`.
std::atomic<uint64_t>& dpOtherFamily(KId id) {
    switch (id) {
    case K_MATMUL: case K_MATMUL_ADD: case K_MATMUL_ARGMAX: case K_MATMUL_ARGMAX_BLOCKS:
    case K_MATMUL2: case K_MATMUL3: case K_Q8_MATMUL: case K_Q8_MATMUL_DECODE: case K_Q8_MATMUL2: case K_Q8_MATMUL3: case K_Q6K_MATMUL: case K_Q5K_MATMUL: case K_Q3K_MATMUL: case K_V41_Q3K_MATVEC_ID: case K_V41_Q3K_TOKEN_PANEL:
    case K_IQ4XS_MATVEC: case K_IQ1S_MATVEC: case K_IQ2XS_MATVEC: case K_IQ2XXS_MATVEC: case K_IQ3S_MATVEC: case K_IQ2S_MATVEC: case K_IQ3XXS_MATVEC:
    case K_MM_CM1_Q2K: case K_MM_CM1_IQ2XXS: case K_MM_CM1_IQ2XS: case K_MM_CM1_IQ2S: case K_MM_CM1_IQ3XXS: case K_MM_CM1_IQ3S: case K_MM_CM1_IQ1S: case K_MM_CM1_IQ4XS: case K_MM_CM1_Q4K:
        return g_dp.otherMatmul;
    case K_RMSNORM: case K_RMSNORM_MATMUL: case K_RMSNORM_MATMUL2: case K_RMSNORM_MATMUL3:
    case K_RMSNORM_MATMUL_ARGMAX_BLOCKS: case K_RMSNORM_Q8_MATMUL2: case K_RMSNORM_Q8_MATMUL3: case K_RMSNORM_Q8_MATMUL2_COOP:
    case K_RMSNORM_Q4K_MATMUL2: case K_V41_EXACT_RMSNORM:
        return g_dp.otherNorm;
    case K_ROPE: case K_QWEN35_PARTIAL_ROPE_PANEL: case K_V41_TAIL_ROPE_QK: case K_V41_TAIL_ROPE_EXACT:
        return g_dp.otherRope;
    case K_SWIGLU: case K_SWIGLU_MATMUL_ADD: case K_SWIGLU_Q8_MATMUL_ADD: case K_SIGMOID_MUL:
    case K_SWIGLU_Q4K_MATMUL_ADD: case K_V41_ROUTED_SWIGLU: case K_V41_ROUTED_SWIGLU_ID: case K_V41_ROUTED_SWIGLU_PANEL_ROW: case K_V41_ROUTED_SWIGLU_PANEL_CHUNK:
        return g_dp.otherSwiGLU;
    case K_ADD: case K_ADD_BIAS:
        return g_dp.otherAdd;
    case K_ATTENTION: case K_QWEN35_CAUSAL_ATTENTION_PANEL: case K_ATTENTION_SPLIT_KV:
    case K_V41_SHARED_ATTENTION: case K_V41_SHARED_ATTENTION_FAST: case K_V41_SHARED_ATTENTION_SPLIT: case K_V41_INDEXER_SCORE:
        return g_dp.otherAttention;
    case K_ARGMAX: case K_ARGMAX_PAIRS: case K_V41_ROUTER_TOPK: case K_V41_INDEX_SELECT:
    case K_V41_ROUTER_GATE_TOPK:
        return g_dp.otherArgmax;
    case K_QWEN35_GDN_Q8_IN_PROJ: case K_QWEN35_GDN_CONV: case K_QWEN35_GDN_RECURRENT:
    case K_QWEN35_GDN_PREFILL_TILED: case K_QWEN35_GDN_PREFILL_NORM:
    case K_QWEN35_GDN_VERIFY_TILED: case K_QWEN35_VALUE_HEAD_TILE:
    case K_GLM_KDA_REREAD: case K_GLM_KDA_WAVE32:
        return g_dp.otherGDN;
    case K_V41_ROUTED_DOWN_PANEL_CHUNK: case K_V41_ROUTED_DOWN_PANEL_ROW: case K_V41_INDEX_SCALE_BF16: case K_V41_MHC: case K_V41_Q2K_EMBED_GATHER: case K_V41_BF16_BOUNDARY: case K_V41_LOGIT_SCALE: case K_V41_BF16_CHECKED:
    case K_V41_ENGRAM_GATE: case K_V41_COMPRESSOR_POOL: case K_V41_SCALE_EXACT: case K_V41_ENGRAM_ROW_DECODE: case K_V41_ROUTED_PUBLISH_SUM:
    case K_V41_COMPRESSOR_RING:
    case K_QWEN35_SPLIT_QG_PANEL: case K_Q4K_MATMUL: case K_Q4K_MATMUL_WAVE32: case K_Q4K_MATMUL_COOPMAT: case K_Q2K_MATMUL: case K_RMSNORM_Q2K_MATMUL2: case K_Q2K_MATVEC: case K_V41_Q2K_MATVEC_ID: case K_V41_Q2K_TOKEN_PANEL: case K_V41_Q2K_TOKEN_PAIR: case K_COUNT:
        return g_dp.otherUnclassified;
    }
    return g_dp.otherUnclassified;
}

static inline void dpDispatch(const Kernel& k) {
    g_deviceDispatches.fetch_add(1, std::memory_order_relaxed);
    (g_batching ? g_syncBatchStage : g_syncOneShotStage) = (uint32_t)(&k - g_kern);
    if (!g_dp_on) return;
    const KId id = static_cast<KId>(&k - g_kern);
    if (id == K_Q4K_MATMUL || id == K_Q4K_MATMUL_WAVE32 || id == K_Q4K_MATMUL_COOPMAT) {
        g_dp.q4k.fetch_add(1, std::memory_order_relaxed);
    } else if (id == K_Q2K_MATMUL || id == K_RMSNORM_Q2K_MATMUL2 || id == K_Q2K_MATVEC || id == K_V41_Q2K_MATVEC_ID || id == K_V41_Q2K_TOKEN_PANEL || id == K_V41_Q2K_TOKEN_PAIR) {
        g_dp.q2k.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_dp.other.fetch_add(1, std::memory_order_relaxed);
        dpOtherFamily(id).fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t n = g_dp.compute.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 255) == 0) {
        fprintf(stderr,
            "fak-vulkan profile compute=%llu q4k=%llu other=%llu other_ops=matmul:%llu,norm:%llu,rope:%llu,swiglu:%llu,add:%llu,attention:%llu,argmax:%llu,gdn:%llu,unclassified:%llu barriers=%llu d2d=%llu batch_submits=%llu batch_flushes=%llu one_shot_submits=%llu one_shot_calls=compute:%llu,h2d:%llu,d2h:%llu,d2d:%llu h2d_bytes=%llu d2h_bytes=%llu\n",
            (unsigned long long)n, (unsigned long long)g_dp.q4k.load(), (unsigned long long)g_dp.other.load(),
            (unsigned long long)g_dp.otherMatmul.load(), (unsigned long long)g_dp.otherNorm.load(),
            (unsigned long long)g_dp.otherRope.load(), (unsigned long long)g_dp.otherSwiGLU.load(),
            (unsigned long long)g_dp.otherAdd.load(), (unsigned long long)g_dp.otherAttention.load(),
            (unsigned long long)g_dp.otherArgmax.load(), (unsigned long long)g_dp.otherGDN.load(),
            (unsigned long long)g_dp.otherUnclassified.load(), (unsigned long long)g_dp.barriers.load(),
            (unsigned long long)g_dp.d2d.load(), (unsigned long long)g_dp.batchSubmits.load(),
            (unsigned long long)g_dp.batchFlushes.load(), (unsigned long long)g_dp.oneShotSubmits.load(),
            (unsigned long long)g_dp.oneShotCompute.load(), (unsigned long long)g_dp.oneShotH2D.load(),
            (unsigned long long)g_dp.oneShotD2H.load(), (unsigned long long)g_dp.oneShotD2D.load(),
            (unsigned long long)g_h2dBytes.load(), (unsigned long long)g_d2hBytes.load());
    }
}

// FAK_VULKAN_TIMESTAMP_PROFILE gate, read once at load; the profile itself is below.
const bool g_tsOn = [] { const char* v = std::getenv("FAK_VULKAN_TIMESTAMP_PROFILE"); return v && v[0] == '1' && v[1] == '\0'; }();
// Work submitted outside a timed batch, with no timestamp pair of its own, is in no
// timestamp row, so the profile counts it rather than silently reporting full coverage.
std::atomic<uint64_t> g_tsUntimedCompute{0}, g_tsUntimedTransfer{0};
// A dispatch recorded without its barrier overlaps its predecessor, so its delta is not its own time.
uint64_t g_tsBarriersElided = 0;
static inline void tsCountUntimed(bool compute) {
    if (!g_tsOn) return;
    (compute ? g_tsUntimedCompute : g_tsUntimedTransfer).fetch_add(1, std::memory_order_relaxed);
}

// timed: the submit carried its own timestamp pair (tsOneShotBegin), so its time is in a row.
static inline void dpOneShot(std::atomic<uint64_t>& family, bool timed = false) {
    if (!timed) tsCountUntimed(&family == &g_dp.oneShotCompute);
    g_deviceSubmits.fetch_add(1, std::memory_order_relaxed);
    if (!g_dp_on) return;
    family.fetch_add(1, std::memory_order_relaxed);
    g_dp.oneShotSubmits.fetch_add(1, std::memory_order_relaxed);
}

// ---- host-sync sites ------------------------------------------------------------------
// A site is the outermost entry point that made the host wait on the device; the stage is
// the last compute kernel recorded on the waited path (kSyncStageNone: no dispatch).
enum SyncSite : uint32_t {
    kSyncSiteSubmitWait, kSyncSiteBatchFlush, kSyncSiteFlushStatus, kSyncSiteReadback,
    kSyncSiteStepSuspend, kSyncSiteStepEnd, kSyncSitePipelineDrain, kSyncSiteUpload, kSyncSiteCount
};
const char* const kSyncSiteNames[kSyncSiteCount] = {
    "submit_wait", "batch_flush", "flush_status", "readback", "step_suspend", "step_end", "pipeline_drain", "upload"};
constexpr uint32_t kSyncSiteUnset = kSyncSiteCount;
constexpr uint32_t kSyncStageNone = (uint32_t)K_COUNT;
constexpr uint32_t kSyncStageCount = kSyncStageNone + 1;
struct SyncCell { std::atomic<uint64_t> syncs{0}, waitNs{0}, d2hBytes{0}; };
SyncCell g_syncCells[kSyncSiteCount][kSyncStageCount];
uint32_t g_syncSite = kSyncSiteUnset;
bool g_syncBatchFlush = false;
SyncCell* g_syncLast = nullptr;

struct SyncScope {
    uint32_t prevSite;
    bool prevBatch;
    explicit SyncScope(uint32_t site, bool batchFlush = false) : prevSite(g_syncSite), prevBatch(g_syncBatchFlush) {
        if (g_syncSite == kSyncSiteUnset) g_syncSite = site;
        if (batchFlush) g_syncBatchFlush = true;
    }
    ~SyncScope() { g_syncSite = prevSite; g_syncBatchFlush = prevBatch; }
};

inline uint32_t syncTakeBatchStage() { const uint32_t s = g_syncBatchStage; g_syncBatchStage = UINT32_MAX; return s; }
inline uint32_t syncTakeStage() {
    if (g_syncBatchFlush) return syncTakeBatchStage();
    const uint32_t s = g_syncOneShotStage;
    g_syncOneShotStage = UINT32_MAX;
    return s;
}

inline uint64_t syncIdleNs(uint64_t now) {
    const uint64_t v = g_deviceIdleClock.load(std::memory_order_relaxed);
    if (!(v & 1)) return v >> 1;
    return now > (v >> 1) ? now - (v >> 1) : 0;
}

// A submit with nothing outstanding closes the open idle interval at `at`.
inline void syncSubmitted(uint64_t at) {
    if (g_syncOutstanding++ != 0) return;
    const uint64_t v = g_deviceIdleClock.load(std::memory_order_relaxed);
    if (v & 1) g_deviceIdleClock.store(syncIdleNs(at) << 1, std::memory_order_relaxed);
}

// A completed host wait: the last outstanding signal opens an idle interval, and the wait
// books to the scoped site (or `site`) and `stage`.
inline void syncSignaled(uint32_t site, uint32_t stage, uint64_t waitStart, bool scoped = true) {
    const uint64_t now = submitWaitNow();
    if (g_syncOutstanding > 0 && --g_syncOutstanding == 0) {
        const uint64_t v = g_deviceIdleClock.load(std::memory_order_relaxed);
        if (!(v & 1)) g_deviceIdleClock.store(((now - (v >> 1)) << 1) | 1, std::memory_order_relaxed);
    }
    if (scoped && g_syncSite != kSyncSiteUnset) site = g_syncSite;
    SyncCell& c = g_syncCells[site < kSyncSiteCount ? site : kSyncSiteSubmitWait][stage < kSyncStageCount ? stage : kSyncStageNone];
    c.syncs.fetch_add(1, std::memory_order_relaxed);
    c.waitNs.fetch_add(now > waitStart ? now - waitStart : 0, std::memory_order_relaxed);
    g_syncLast = &c;
}

inline void syncNoteD2H(uint64_t bytes) {
    if (g_syncLast) g_syncLast->d2hBytes.fetch_add(bytes, std::memory_order_relaxed);
}

// Q8 fast-path availability (set in fvk_init from the device's 8-bit-storage + int8 features).
int g_have_q8 = 0;
// The four-projection GDN specialization is optional even when generic Q8 is
// available: an absent or rejected pipeline must preserve the composed path.
int g_have_qwen35_gdn_q8_in_proj = 0;
// Mandatory in a current build; published only after complete initialization.
int g_have_v41_tail_rope_qk = 0;
#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
bool g_v41TailRoPEExactModule = false;
#endif
// Optional strict binary32 twin (registry V12/complete72); published only after
// complete initialization, the selected device's F32 controls and strict ABI proof.
int g_have_v41_tail_rope_exact = 0;
int g_have_v41_shared_attention = 0;
// Optional run-to-run determinism class (V41C-12): pinned Wave32 subgroup
// trees, published only with subgroup size control and the oracle's ABI proof.
// Class 0 keeps the serial bit-exact kernel; class 1 selects the fast module.
int g_have_v41_shared_attention_fast = 0;
bool g_v41SharedAttentionFastSubgroups = false;
int g_v41SharedAttentionClass = 0;
// Class 2 (split-k) is the optional two-pass module; its partials live in a
// shim-owned scratch buffer that grows on demand and is never aliased.
int g_have_v41_shared_attention_split = 0;
Buffer* g_v41SplitPartials = nullptr;
// Optional: the router module is not in the trusted shader registry yet, so it
// is published only when its module is present and its ABI matches.
int g_have_v41_router_topk = 0;
// Optional fused router gate projection + top-k. The device verdict is the
// Go self-check result against matmul + router top-k on this device:
// 0 unknown, 1 bit-identical, 2 disagreed. Reset with the device.
int g_have_v41_router_gate_topk = 0;
int g_v41RouterGateVerdict = 0;
bool g_v41RouterGateSubgroups = false;
// Independently admitted only after dynamic binary32 properties and module proof.
int g_have_v41_indexer_score = 0;
int g_have_v41_mhc = 0;
int g_have_v41_index_select = 0;
int g_have_v41_q2k_embed_gather = 0;
// Optional: needs the same binary32 float controls as the indexer score.
int g_have_v41_exact_rmsnorm = 0;
int g_have_v41_bf16_boundary = 0;
int g_have_v41_logit_scale = 0;
int g_have_v41_bf16_checked = 0;
// Optional: registry V10 module; needs the binary32 float controls it declares.
int g_have_v41_routed_swiglu = 0;
// Optional: registry V11 modules; same binary32 float controls as the exact RMSNorm.
int g_have_v41_engram_gate = 0;
int g_have_v41_compressor_pool = 0;
// Optional: device-resident compressor ring (store + gated pool), same float controls.
int g_have_v41_compressor_ring = 0;
// Optional: registry V13 strict scalar multiply; same binary32 float controls,
// admitted by the exact scalar proof the witness logit-scale module uses.
int g_have_v41_scale_exact = 0;
// Optional: registry V14 Engram table-row decode; same binary32 float controls.
int g_have_v41_engram_row_decode = 0;
// Optional: registry V15 routed-expert id modules (one dispatch per stage over
// every pick); the checked pair needs the same binary32 float controls.
int g_have_v41_routed_id = 0;
bool g_v41IndexerFloatControls = false;
std::string g_v41IndexerScoreReason = "Vulkan has not initialized";
// Fixed-size GLM KDA kernels require an explicitly requested 32-lane subgroup.
// A local size of 128 alone is not a Wave32 contract: RADV may otherwise choose Wave64.
int g_have_glm_kda_wave32 = 0;
// Wave32 cooperative Q4_K decode kernel (subgroup arithmetic + effective/required subgroup size 32).
int g_have_q4k_wave32 = 0;
// Single-token Q2_K matvec (q2k_matvec.spv); optional, default-on when the SPIR-V loads.
int g_have_q2k_matvec = 0;
int g_have_rmsnorm_q8_matmul2_coop = 0;
// Native IQ-quant matvec kernels (ticket 05f), indexed by the fvk IQ format id shared with
// vulkan_iq.go. Each is optional: absent SPIR-V or its FAK_VULKAN_<FMT>=0 kill switch leaves
// the format unavailable, and Upload then expands that weight to Q8_0 as before.
enum { FVK_IQ4_XS = 0, FVK_IQ3_XXS = 1, FVK_IQ2_S = 2, FVK_IQ3_S = 3, FVK_IQ2_XXS = 4, FVK_IQ2_XS = 5, FVK_IQ1_S = 6, FVK_IQ_FORMATS };
const KId kIQKernel[FVK_IQ_FORMATS] = {K_IQ4XS_MATVEC, K_IQ3XXS_MATVEC, K_IQ2S_MATVEC, K_IQ3S_MATVEC, K_IQ2XXS_MATVEC, K_IQ2XS_MATVEC, K_IQ1S_MATVEC};
const char* const kIQSpv[FVK_IQ_FORMATS] = {"iq4xs_matvec.spv", "iq3xxs_matvec.spv", "iq2s_matvec.spv", "iq3s_matvec.spv", "iq2xxs_matvec.spv", "iq2xs_matvec.spv", "iq1s_matvec.spv"};
const char* const kIQEnv[FVK_IQ_FORMATS] = {"FAK_VULKAN_IQ4XS", "FAK_VULKAN_IQ3XXS", "FAK_VULKAN_IQ2S", "FAK_VULKAN_IQ3S", "FAK_VULKAN_IQ2XXS", "FAK_VULKAN_IQ2XS", "FAK_VULKAN_IQ1S"};
int g_have_iq[FVK_IQ_FORMATS] = {};
// Rows per 64-lane workgroup: every iq*_matvec.comp gives each subgroup NR = 8 rows, so a
// Wave32 workgroup covers 16 (a Wave64 one strides over the rest).
const uint32_t kIQMatvecRows = 16;
const uint32_t kIQMatvecMaxGroups = 4096;
bool g_q4k_wave32_required_subgroup = false;
// Optional, default-off recurrent prefill variant. All access is serialized by
// the Go Vulkan mutex, including debug mode/counter operations.
int g_have_gdn_prefill_tiled = 0;
bool g_gdn_prefill_required_subgroup = false;
uint32_t g_gdn_prefill_max_groups_y = 0;
int g_gdn_prefill_mode = -1;
uint64_t g_gdn_prefill_tiled_calls = 0;
uint64_t g_gdn_prefill_scalar_calls = 0;
// Verify-width (1..7 token) register-resident variant, admitted per geometry by
// an on-device self-check. -1 = follow the verdict, 0 = force stock, 1 = request
// the candidate (still requiring a true verdict). g_gdn_verify_verdict holds the
// cached self-check result for the one admitted production geometry.
int g_have_gdn_verify_tiled = 0;
int g_gdn_verify_mode = -1;
int g_gdn_verify_verdict = 0;
int g_gdn_verify_checked = 0;
float g_gdn_verify_deviation = 0.0f;
uint64_t g_gdn_verify_tiled_calls = 0;
uint64_t g_gdn_verify_scalar_calls = 0;
// GDN calls on the production geometry that FAK_VULKAN_GDN_PREFILL_TILED=0 kept
// on the stock kernel.
uint64_t g_gdn_optout_calls = 0;
// Test-only forced-disagreement hook: when nonzero the self-check reports failure
// and the route must fall back to the stock kernel.
int g_gdn_verify_force_disagree = 0;
// Prefill-class (>=8 token) self-check verdict, cached per geometry alongside the
// verify-width verdict. The tiled route requires a true verdict once the default
// flip is earned; until then the explicit request path still requires it.
int g_gdn_tiled_verdict = 0;
float g_gdn_tiled_deviation = 0.0f;
int g_have_coopmat = 0;
// Candidate Q4_K cooperative-matrix prefill arm (2D block-tiled). Requires the native
// cooperative-matrix capability AND a 2D-grid shader that built; default retains scalar.
int g_have_q4k_coopmat = 0;
// Prefill fp16 cooperative-matrix GEMM over packed K-quant / i-quant weights (mm_cm1_*.spv).
// g_mmcm1Status[f] is 0 when format f is servable, else the first failed requirement
// (FVK_MM_CM1_* status codes in vulkan_backend.h) so a refused route names its reason.
enum { FVK_MMCM1_Q2K = 0, FVK_MMCM1_IQ2XXS, FVK_MMCM1_IQ2XS, FVK_MMCM1_IQ2S, FVK_MMCM1_IQ3XXS, FVK_MMCM1_IQ3S, FVK_MMCM1_IQ1S, FVK_MMCM1_IQ4XS, FVK_MMCM1_Q4K, FVK_MMCM1_FORMATS };
const KId kMMCM1Kernel[FVK_MMCM1_FORMATS] = {K_MM_CM1_Q2K, K_MM_CM1_IQ2XXS, K_MM_CM1_IQ2XS, K_MM_CM1_IQ2S, K_MM_CM1_IQ3XXS, K_MM_CM1_IQ3S, K_MM_CM1_IQ1S, K_MM_CM1_IQ4XS, K_MM_CM1_Q4K};
const char* const kMMCM1Spv[FVK_MMCM1_FORMATS] = {"mm_cm1_q2k.spv", "mm_cm1_iq2xxs.spv", "mm_cm1_iq2xs.spv", "mm_cm1_iq2s.spv", "mm_cm1_iq3xxs.spv", "mm_cm1_iq3s.spv", "mm_cm1_iq1s.spv", "mm_cm1_iq4xs.spv", "mm_cm1_q4k.spv"};
const uint32_t kMMCM1TileRows = 128;   // BM in mm_cm1_quant.glsl
const uint32_t kMMCM1TileTokens = 128; // BN in mm_cm1_quant.glsl
int g_mmcm1Status[FVK_MMCM1_FORMATS] = {FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED,
    FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED,
    FVK_MM_CM1_NOT_INITIALIZED, FVK_MM_CM1_NOT_INITIALIZED};
bool g_have_shader_f16 = false;
bool g_have_coopmat_f16_16 = false;
// Portable packed Q6_K is optional so older SPIR-V bundles remain loadable.
int g_have_q6k_matmul = 0;
int g_have_q5k_matmul = 0;
int g_have_q3k_matmul = 0;
// Optional GDN core value-head tiling for a native-order ssm_out; absent SPIR-V keeps the copy route.
int g_have_qwen35_value_head_tile = 0;
int g_have_attention_split_kv = 0;

VkDescriptorPool g_descpool = VK_NULL_HANDLE;

// 20 covers the widest fused kernel: the fused routed gate/up binds eight gate
// slot weights, eight up slot weights, X, ids, mid and status. Sizes the
// per-dispatch descriptor/buffer stack arrays; well under the device's
// storage-buffer binding limit and the 32768-descriptor pool's headroom.
static constexpr int MAX_DISPATCH_BUFS = 20;

struct DescriptorSetRecord {
    VkDescriptorSetLayout layout = VK_NULL_HANDLE;
    VkDescriptorSet       set    = VK_NULL_HANDLE;
    int                   nbuf   = 0;
    VkBuffer              buffers[MAX_DISPATCH_BUFS]{};
};

std::unordered_map<VkDescriptorSetLayout, std::vector<DescriptorSetRecord>> g_descSetPool;

void clearDescriptorBindingCache() {
    for (auto& kv : g_descSetPool) {
        for (DescriptorSetRecord& rec : kv.second) {
            rec.nbuf = 0;
            for (int i = 0; i < MAX_DISPATCH_BUFS; ++i) rec.buffers[i] = VK_NULL_HANDLE;
        }
    }
}

uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags want);

// findDirectWeightMemType picks a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type backed by a
// device-local heap (the UMA carve-out, never the system-RAM GTT heap), or UINT32_MAX.
uint32_t findDirectWeightMemType(uint32_t typeBits) {
    const VkMemoryPropertyFlags want = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) && (g_memprops.memoryTypes[i].propertyFlags & want) == want &&
            memoryTypeUsesDeviceLocalHeap(i)) {
            return i;
        }
    }
    return UINT32_MAX;
}

uint32_t findMemType(uint32_t typeBits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g_memprops.memoryTypeCount; ++i) {
        if ((typeBits & (1u << i)) &&
            (g_memprops.memoryTypes[i].propertyFlags & want) == want) {
            return i;
        }
    }
    return UINT32_MAX;
}

bool allocPressure(VkResult r) {
    return r == VK_ERROR_OUT_OF_DEVICE_MEMORY ||
           r == VK_ERROR_OUT_OF_HOST_MEMORY ||
           r == VK_ERROR_TOO_MANY_OBJECTS;
}

// Compatible-type enumeration adapted from ROCmFPX/ggml Vulkan allocation:
// https://github.com/ROCmFPX/ROCmFPX/blob/9279b36aa69fd1ecc420211fc24b5c011c6889ed/ggml/src/ggml-vulkan/ggml-vulkan-buffers.cpp
// Copyright (c) 2023-2026 The ggml authors
// Copyright (c) 2025-2026 Carlo Pasquale (Charlie12345), ROCmFPX additions
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// Fak retains its first-attempt/pool-drain policy and retries only allocation
// pressure, never device loss or another fatal error. This selector performs no
// accounting or binding; the caller publishes those only after success. Preferred
// flags order otherwise-compatible candidates; they never become requirements.
template <typename Allocate>
VkResult allocateCompatibleMemory(uint32_t typeBits, VkMemoryPropertyFlags want,
                                  VkMemoryPropertyFlags preferred,
                                  const VkPhysicalDeviceMemoryProperties& properties,
                                  VkDeviceSize allocationSize, bool tryAll, Allocate allocate,
                                  VkDeviceMemory* out, uint32_t* selectedType,
                                  bool* allocationAttempted) {
    *allocationAttempted = false;
    *out = VK_NULL_HANDLE;
    *selectedType = UINT32_MAX;
    VkResult result = VK_ERROR_FEATURE_NOT_PRESENT;
    const uint32_t count = std::min(properties.memoryTypeCount, uint32_t(32));
    const uint32_t heapCount = std::min(properties.memoryHeapCount, uint32_t(VK_MAX_MEMORY_HEAPS));
    const uint32_t passes = preferred ? 2 : 1;
    for (uint32_t pass = 0; pass < passes; ++pass) {
        const bool preferredPass = preferred && pass == 0;
        for (uint32_t i = 0; i < count; ++i) {
            const VkMemoryType& type = properties.memoryTypes[i];
            if (!(typeBits & (uint32_t(1) << i)) || (type.propertyFlags & want) != want) continue;
            const bool hasPreferred = preferred && (type.propertyFlags & preferred) == preferred;
            if (preferred && hasPreferred != preferredPass) continue;
            // VUID-vkAllocateMemory-pAllocateInfo-01713: use the Vulkan allocation
            // requirement, not the smaller logical payload, for heap eligibility.
            if (type.heapIndex >= heapCount || allocationSize > properties.memoryHeaps[type.heapIndex].size) continue;
            VkDeviceMemory candidate = VK_NULL_HANDLE;
            *allocationAttempted = true;
            result = allocate(i, &candidate);
            if (result == VK_SUCCESS) {
                *out = candidate;
                *selectedType = i;
                return result;
            }
            if (!tryAll || !allocPressure(result)) return result;
        }
    }
    return result;
}

bool deviceExtensionSupported(const char* name) {
    if (!g_phys || !name || !*name) return false;
    uint32_t n = 0;
    if (vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &n, nullptr) != VK_SUCCESS || n == 0) {
        return false;
    }
    std::vector<VkExtensionProperties> props(n);
    if (vkEnumerateDeviceExtensionProperties(g_phys, nullptr, &n, props.data()) != VK_SUCCESS) {
        return false;
    }
    for (const auto& p : props) {
        if (strcmp(p.extensionName, name) == 0) return true;
    }
    return false;
}

bool queryDeviceLocalMemoryBudget(VkDeviceSize* budget, VkDeviceSize* usage) {
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    if (!g_phys || !g_haveMemoryBudget) return false;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT bp{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 props2{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
    props2.pNext = &bp;
    vkGetPhysicalDeviceMemoryProperties2(g_phys, &props2);
    VkDeviceSize b = 0;
    VkDeviceSize u = 0;
    for (uint32_t i = 0; i < props2.memoryProperties.memoryHeapCount; ++i) {
        if (props2.memoryProperties.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            b += bp.heapBudget[i];
            u += bp.heapUsage[i];
        }
    }
    if (b == 0) return false;
    if (budget) *budget = b;
    if (usage) *usage = u;
    return true;
#else
    (void)budget;
    (void)usage;
    return false;
#endif
}

void drainInflight();

void destroyBuffer(Buffer* b) {
    if (g_v41SubmissionPendingFailure) return; // referenced scratch/arena may still be in flight
    if (!b) return;
    if (b->buf) {
        drainInflight(); // a pipelined batch may still bind this buffer
        // Descriptor sets outlive temporary buffers; a recycled handle is not
        // proof that its old binding is still valid. Cover every destruction path.
        clearDescriptorBindingCache();
        vkDestroyBuffer(g_dev, b->buf, nullptr);
    }
    if (b->weightArenaBound) {
        if (b->weightArenaBlock < g_weightArena.size()) {
            WeightArenaBlock& block = g_weightArena[b->weightArenaBlock];
            if (block.liveBuffers > 0) --block.liveBuffers;
        }
        if (g_weightArenaLiveBytes >= b->bytes) g_weightArenaLiveBytes -= b->bytes;
        else g_weightArenaLiveBytes = 0;
    } else if (b->mem) {
        untrackDeviceAllocation(b->allocationBytes, b->allocationDeviceLocal);
        if (b->ownsMapping) vkUnmapMemory(g_dev, b->mem);
        freeGuardedMemory(b->mem, b->allocationBytes);
    }
    delete b;
    releaseWeightArena();
}

Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags preferred = 0);

bool g_debugD2HStagingFailureOnce = false;

size_t stageCapacity(size_t bytes) {
    size_t cap = 64 * 1024;
    while (cap < bytes && cap <= (((size_t)-1) / 2)) cap *= 2;
    return cap < bytes ? bytes : cap;
}

Buffer* stagingBuffer(size_t bytes, int* failureStatus = nullptr) {
    if (failureStatus) *failureStatus = VK_SUCCESS;
    if (bytes == 0) return nullptr;
    if (failureStatus && g_debugD2HStagingFailureOnce) {
        g_debugD2HStagingFailureOnce = false;
        *failureStatus = FVK_D2H_STAGING_ALLOCATION_FAILED;
        return nullptr;
    }
    if (g_stage && g_stageCap >= bytes && g_stageMapped) return g_stage;

    if (g_stage) {
        if (g_stageMapped) vkUnmapMemory(g_dev, g_stage->mem);
        destroyBuffer(g_stage);
        g_stage = nullptr;
        g_stageMapped = nullptr;
        g_stageCap = 0;
    }

    size_t cap = stageCapacity(bytes);
    g_stage = allocBuffer(cap,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
        g_preferCachedTransferStage ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0);
    if (!g_stage) {
        if (failureStatus) *failureStatus = FVK_D2H_STAGING_ALLOCATION_FAILED;
        return nullptr;
    }
    VkResult r = vkMapMemory(g_dev, g_stage->mem, 0, cap, 0, &g_stageMapped);
    if (r != VK_SUCCESS || !g_stageMapped) {
        fprintf(stderr, "fak-vulkan: vkMapMemory(stage %zu bytes) failed VkResult=%d\n", cap, (int)r);
        destroyBuffer(g_stage);
        g_stage = nullptr;
        g_stageMapped = nullptr;
        g_stageCap = 0;
        if (failureStatus) {
            *failureStatus = r == VK_SUCCESS ? FVK_D2H_STAGING_ALLOCATION_FAILED : (int)r;
        }
        return nullptr;
    }
    g_stageCap = cap;
    return g_stage;
}

// drainPool destroys every recycled (free-list) buffer, returning their device allocations
// to the driver. Called under VRAM / allocation-count pressure: Vulkan caps the NUMBER of
// live vkAllocateMemory objects (maxMemoryAllocationCount, often a few thousand), and the
// per-token weight re-upload churns through many distinct-size buffers, so an unbounded pool
// would exhaust that count and vkAllocateMemory would start returning VK_ERROR_OUT_OF_*.
size_t drainPool() {
    size_t freed = 0;
    for (auto& kv : g_pool) {
        for (Buffer* b : kv.second) { destroyBuffer(b); ++freed; }
        kv.second.clear();
    }
    for (auto& kv : g_hvOffloadPool) {
        for (Buffer* b : kv.second) { destroyBuffer(b); ++freed; }
        kv.second.clear();
    }
    if (freed > 0) clearDescriptorBindingCache();
    g_poolCount = 0;
    return freed;
}

Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage,
                    VkMemoryPropertyFlags preferred) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    size_t reqBytes = bytes ? bytes : 1;
    if ((usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) &&
        g_maxBufferBytes > 0 && (VkDeviceSize)reqBytes > g_maxBufferBytes) {
        fprintf(stderr,
            "fak-vulkan: refusing storage buffer %zu bytes; device single-resource cap=%llu "
            "(maxStorageBufferRange=%llu maxMemoryAllocationSize=%llu); chunk/defuse required\n",
            bytes,
            (unsigned long long)g_maxBufferBytes,
            (unsigned long long)g_maxStorageBufferRange,
            (unsigned long long)g_maxMemoryAllocationSize);
        return nullptr;
    }
    Buffer* b = new Buffer();
    b->bytes = bytes;
    b->requestedProps = props;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = reqBytes;
    bi.usage = usage;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult cr = vkCreateBuffer(g_dev, &bi, nullptr, &b->buf);
    if (cr != VK_SUCCESS) {
        fprintf(stderr, "fak-vulkan: vkCreateBuffer(%zu bytes) failed VkResult=%d\n", bytes, (int)cr);
        delete b;
        return nullptr;
    }
    VkMemoryRequirements req{};
    vkGetBufferMemoryRequirements(g_dev, b->buf, &req);
    bool allocationAttempted = false;
    auto tryAlloc = [&](VkMemoryPropertyFlags want, VkDeviceMemory* out, uint32_t* selectedType, bool tryAll = false) -> VkResult {
        return allocateCompatibleMemory(req.memoryTypeBits, want, preferred, g_memprops, req.size, tryAll, [&](uint32_t index, VkDeviceMemory* candidate) {
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = index;
                return allocateGuardedMemory(&ai, candidate);
            }, out, selectedType, &allocationAttempted);
    };

    // First attempt; on out-of-memory / too-many-allocations, drain the recycle pool (which
    // holds buffers nothing references right now) and retry before considering a slower
    // host-visible storage fallback for device-local tensors.
    VkMemoryPropertyFlags actualProps = props;
    uint32_t selectedMemoryType = UINT32_MAX;
    VkResult r = tryAlloc(props, &b->mem, &selectedMemoryType);
    if (allocPressure(r)) {
        drainPool();
        r = tryAlloc(props, &b->mem, &selectedMemoryType, true);
    }
    if ((allocPressure(r) || (r == VK_ERROR_FEATURE_NOT_PRESENT && !allocationAttempted)) &&
        (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        VkMemoryPropertyFlags fallback =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkResult fr = tryAlloc(fallback, &b->mem, &selectedMemoryType, true);
        if (fr == VK_SUCCESS) {
            fprintf(stderr,
                "fak-vulkan: device-local alloc(%zu bytes) failed VkResult=%d; using host-visible storage\n",
                bytes, (int)r);
            r = fr;
            actualProps = fallback;
            b->hostVisibleFallback = true;
        } else {
            r = fr;
        }
    }
    if (r == VK_ERROR_FEATURE_NOT_PRESENT && !allocationAttempted) {
        fprintf(stderr, "fak-vulkan: no compatible memory type for %zu bytes\n", bytes);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    if (r != VK_SUCCESS) {
        fprintf(stderr, "fak-vulkan: vkAllocateMemory(%zu bytes) failed VkResult=%d\n", bytes, (int)r);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    VkResult br = vkBindBufferMemory(g_dev, b->buf, b->mem, 0);
    if (br != VK_SUCCESS) {
        fprintf(stderr, "fak-vulkan: vkBindBufferMemory(%zu bytes) failed VkResult=%d\n", bytes, (int)br);
        freeGuardedMemory(b->mem, req.size);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    b->props = actualProps;
    b->memoryTypeIndex = selectedMemoryType;
    b->allocationBytes = req.size;
    b->allocationID = nextMemoryAllocationID();
    b->deviceGeneration = g_memoryDeviceGeneration;
    b->nativeCheckoutGeneration = 1;
    b->allocationDeviceLocal = memoryTypeUsesDeviceLocalHeap(selectedMemoryType);
    trackDeviceAllocation(req.size, selectedMemoryType);
    return b;
}

// mapHostVisibleWeight gives a weight's own host-visible allocation (the GTT fallback when the
// UMA carve-out is too small) a persistent map, so uploads write through it like the direct arena.
// Only HOST_COHERENT types qualify: a non-coherent type would need explicit flushes, so it keeps
// the staging path, as does a failed map or UMA direct upload being off.
void mapHostVisibleWeight(Buffer* b) {
    if (!g_umaDirectWeights || !b || !b->mem || b->mapped || b->weightArenaBound ||
        b->memoryOffset != 0 || b->memoryTypeIndex >= g_memprops.memoryTypeCount) return;
    const VkMemoryPropertyFlags want =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((g_memprops.memoryTypes[b->memoryTypeIndex].propertyFlags & want) != want) return;
    void* mapped = nullptr;
    if (vkMapMemory(g_dev, b->mem, 0, VK_WHOLE_SIZE, 0, &mapped) != VK_SUCCESS || !mapped) return;
    b->mapped = mapped;
    b->ownsMapping = true;
}

// device-local storage buffer used for all tensors (the residency seam).
const VkBufferUsageFlags STORAGE_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

VkDeviceSize alignArenaOffset(VkDeviceSize value, VkDeviceSize alignment) {
    if (alignment <= 1) return value;
    VkDeviceSize remainder = value % alignment;
    if (remainder == 0) return value;
    VkDeviceSize padding = alignment - remainder;
    if (value > std::numeric_limits<VkDeviceSize>::max() - padding) {
        return std::numeric_limits<VkDeviceSize>::max();
    }
    return value + padding;
}

Buffer* allocWeightArenaBuffer(size_t bytes, VkDeviceSize maxArenaBytes) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    size_t reqBytes = bytes ? bytes : 1;
    if (g_maxBufferBytes > 0 && (VkDeviceSize)reqBytes > g_maxBufferBytes) return nullptr;

    Buffer* b = new Buffer();
    b->bytes = bytes;
    b->requestedProps = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = reqBytes;
    bi.usage = STORAGE_USAGE;
    bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkResult cr = vkCreateBuffer(g_dev, &bi, nullptr, &b->buf);
    if (cr != VK_SUCCESS) {
        delete b;
        return nullptr;
    }

    VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
    VkMemoryRequirements2 req2{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
    req2.pNext = &dedicated;
    VkBufferMemoryRequirementsInfo2 info{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
    info.buffer = b->buf;
    vkGetBufferMemoryRequirements2(g_dev, &info, &req2);
    const VkMemoryRequirements& req = req2.memoryRequirements;
    if (dedicated.requiresDedicatedAllocation) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        Buffer* own = allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
        if (own && own->hostVisibleFallback) mapHostVisibleWeight(own);
        return own;
    }

    const uint32_t preferredType = g_umaDirectWeights ? findDirectWeightMemType(req.memoryTypeBits) : UINT32_MAX;
    if (maxArenaBytes == 0) maxArenaBytes = g_totalDeviceLocalMemory;
    if (maxArenaBytes == 0) maxArenaBytes = g_maxMemoryAllocationSize;
    if (maxArenaBytes == 0) maxArenaBytes = WEIGHT_ARENA_BLOCK_BYTES;

    // Prefer existing UMA backing, then reuse any compatible actual backing.
    const uint32_t memoryTypeCount = std::min(g_memprops.memoryTypeCount, uint32_t(32));
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < g_weightArena.size(); ++i) {
            WeightArenaBlock& block = g_weightArena[i];
            const uint32_t memoryType = block.memoryTypeIndex;
            if (memoryType >= memoryTypeCount ||
                !(req.memoryTypeBits & (uint32_t(1) << memoryType)) ||
                !(g_memprops.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) continue;
            if ((memoryType == preferredType) != (pass == 0)) continue;
            VkDeviceSize offset = alignArenaOffset(block.used, req.alignment);
            if (offset == std::numeric_limits<VkDeviceSize>::max() ||
                offset > block.capacity || req.size > block.capacity - offset) continue;
            VkResult br = vkBindBufferMemory(g_dev, b->buf, block.mem, offset);
            if (br != VK_SUCCESS) continue;
            block.used = offset + req.size;
            ++block.liveBuffers;
            b->mem = block.mem;
            b->props = g_memprops.memoryTypes[memoryType].propertyFlags;
            b->memoryTypeIndex = block.memoryTypeIndex;
            b->memoryOffset = offset;
            b->mapped = block.mapped ? (char*)block.mapped + offset : nullptr;
            b->weightArenaBound = true;
            b->weightArenaBlock = i;
            b->nativeCheckoutGeneration = 1;
            ++g_weightArenaBufferBindings;
            g_weightArenaLiveBytes += bytes;
            return b;
        }
    }

    if (g_weightArenaReservedBytes >= maxArenaBytes) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    VkDeviceSize remaining = maxArenaBytes - g_weightArenaReservedBytes;
    VkDeviceSize blockBytes = WEIGHT_ARENA_BLOCK_BYTES;
    if (g_maxMemoryAllocationSize > 0 && blockBytes > g_maxMemoryAllocationSize) {
        blockBytes = g_maxMemoryAllocationSize;
    }
    if (blockBytes < req.size) blockBytes = req.size;
    if (blockBytes > remaining) blockBytes = remaining;
    if (blockBytes < req.size) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }

    VkDeviceMemory memory = VK_NULL_HANDLE;
    uint32_t memoryType = UINT32_MAX;
    bool allocationAttempted = false;
    auto tryAlloc = [&](bool tryAll) -> VkResult {
        auto allocate = [&](uint32_t index, VkDeviceMemory* candidate) {
            VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            ai.allocationSize = blockBytes;
            ai.memoryTypeIndex = index;
            return allocateGuardedMemory(&ai, candidate);
        };
        // Preserve the UMA first choice. Heap eligibility uses the entire block;
        // retry only allocation pressure, with fresh private output per attempt.
        const uint32_t preferredBits = preferredType < 32 ? (uint32_t(1) << preferredType) : 0;
        if (preferredBits) {
            VkResult result = allocateCompatibleMemory(preferredBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
                g_memprops, blockBytes, false, allocate, &memory, &memoryType, &allocationAttempted);
            if (allocationAttempted && (!tryAll || !allocPressure(result))) return result;
        }
        return allocateCompatibleMemory(req.memoryTypeBits & ~preferredBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, 0,
            g_memprops, blockBytes, tryAll, allocate, &memory, &memoryType, &allocationAttempted);
    };
    VkResult ar = tryAlloc(false);
    if (allocPressure(ar)) {
        drainPool();
        ar = tryAlloc(true);
    }
    if (ar != VK_SUCCESS) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    VkResult br = vkBindBufferMemory(g_dev, b->buf, memory, 0);
    if (br != VK_SUCCESS) {
        freeGuardedMemory(memory, blockBytes);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }

    WeightArenaBlock block{};
    block.mem = memory;
    block.capacity = blockBytes;
    block.allocationID = nextMemoryAllocationID();
    block.deviceGeneration = g_memoryDeviceGeneration;
    block.used = req.size;
    block.memoryTypeIndex = memoryType;
    block.liveBuffers = 1;
    block.allocationDeviceLocal = memoryTypeUsesDeviceLocalHeap(memoryType);
    const VkMemoryPropertyFlags directMapProps = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    if ((g_memprops.memoryTypes[memoryType].propertyFlags & directMapProps) == directMapProps) {
        // Direct uploads do not flush noncoherent ranges; those types use staging.
        // A failed map leaves mapped null, and uploads take the staging path.
        if (vkMapMemory(g_dev, memory, 0, VK_WHOLE_SIZE, 0, &block.mapped) != VK_SUCCESS) block.mapped = nullptr;
    }
    trackDeviceAllocation(blockBytes, memoryType);
    g_weightArena.push_back(block);
    b->mem = memory;
    b->props = g_memprops.memoryTypes[memoryType].propertyFlags;
    b->memoryTypeIndex = memoryType;
    b->memoryOffset = 0;
    b->mapped = block.mapped;
    b->weightArenaBound = true;
    b->weightArenaBlock = g_weightArena.size() - 1;
    b->nativeCheckoutGeneration = 1;
    ++g_weightArenaMemoryAllocations;
    ++g_weightArenaBufferBindings;
    g_weightArenaReservedBytes += blockBytes;
    if (g_weightArenaReservedBytes > g_weightArenaPeakReservedBytes) {
        g_weightArenaPeakReservedBytes = g_weightArenaReservedBytes;
    }
    g_weightArenaLiveBytes += bytes;
    return b;
}

size_t scratchCapacity(size_t bytes) {
    size_t cap = 4 * 1024;
    while (cap < bytes && cap <= (((size_t)-1) / 2)) cap *= 2;
    return cap < bytes ? bytes : cap;
}

Buffer*                       g_gdn_conv_out = nullptr;
Buffer*                       g_gdn_prefill_readout = nullptr;

struct PartialRoPECacheEntry {
    uint32_t thetaBits = 0;
    int rotary = 0;
    size_t positions = 0;
    Buffer* table = nullptr;
    std::vector<float> inverseFrequency;
};

std::vector<PartialRoPECacheEntry> g_partialRoPECaches;

void freePartialRoPECaches() {
    for (auto& entry : g_partialRoPECaches) {
        if (entry.table) destroyBuffer(entry.table);
    }
    g_partialRoPECaches.clear();
    clearDescriptorBindingCache();
}

Buffer* partialRoPETable(float theta, int rotary, size_t requiredPositions) {
    uint32_t thetaBits = 0;
    static_assert(sizeof(thetaBits) == sizeof(theta), "float key must be exact");
    memcpy(&thetaBits, &theta, sizeof(thetaBits));

    PartialRoPECacheEntry* found = nullptr;
    for (auto& entry : g_partialRoPECaches) {
        if (entry.thetaBits == thetaBits && entry.rotary == rotary) {
            found = &entry;
            break;
        }
    }
    if (!found) {
        g_partialRoPECaches.push_back(PartialRoPECacheEntry{});
        found = &g_partialRoPECaches.back();
        found->thetaBits = thetaBits;
        found->rotary = rotary;
        found->inverseFrequency.resize((size_t)rotary / 2);
        for (size_t pair = 0; pair < found->inverseFrequency.size(); ++pair) {
            found->inverseFrequency[pair] = std::pow(theta,
                -2.0f * (float)pair / (float)rotary);
        }
    }
    if (found->table && found->positions >= requiredPositions) return found->table;

    const size_t pairs = (size_t)rotary / 2;
    if (pairs == 0) return nullptr;
    size_t maxPositions = std::numeric_limits<size_t>::max() / pairs / (2 * sizeof(float));
    // The shader forms the flattened table subscript in uint arithmetic.
    const size_t shaderMaxPositions = (size_t)std::numeric_limits<uint32_t>::max() / pairs;
    if (shaderMaxPositions < maxPositions) maxPositions = shaderMaxPositions;
    if (g_maxBufferBytes > 0) {
        const size_t deviceMaxPositions = (size_t)g_maxBufferBytes / pairs / (2 * sizeof(float));
        if (deviceMaxPositions < maxPositions) maxPositions = deviceMaxPositions;
    }
    if (requiredPositions > maxPositions) return nullptr;

    size_t capacity = found->positions ? found->positions : 64;
    if (capacity > maxPositions) capacity = maxPositions;
    while (capacity < requiredPositions) {
        if (capacity > maxPositions / 2) {
            capacity = maxPositions;
            break;
        }
        capacity *= 2;
    }
    if (capacity < requiredPositions) return nullptr;
    const size_t floats = capacity * pairs * 2;
    const size_t bytes = floats * sizeof(float);
    if ((g_maxBufferBytes > 0 && (VkDeviceSize)bytes > g_maxBufferBytes) ||
        bytes > (size_t)std::numeric_limits<VkDeviceSize>::max()) {
        return nullptr;
    }

    // A recorded batch may still reference the old table. Complete it before replacing
    // that allocation, then resume batching for the caller's next operation.
    const bool resumeBatch = g_batching;
    if (resumeBatch) batchFlush();
    VkMemoryPropertyFlags hostvis =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer* replacement = allocBuffer(bytes, hostvis, STORAGE_USAGE);
    if (!replacement) {
        if (resumeBatch) batchBegin();
        return nullptr;
    }
    void* mapped = nullptr;
    VkResult mappedResult = vkMapMemory(g_dev, replacement->mem, 0, bytes, 0, &mapped);
    if (mappedResult != VK_SUCCESS || !mapped) {
        destroyBuffer(replacement);
        if (resumeBatch) batchBegin();
        return nullptr;
    }
    float* values = static_cast<float*>(mapped);
    if (found->table && found->positions > 0) {
        void* oldMapped = nullptr;
        const size_t oldBytes = found->positions * pairs * 2 * sizeof(float);
        VkResult oldMappedResult = vkMapMemory(g_dev, found->table->mem, 0, oldBytes, 0, &oldMapped);
        if (oldMappedResult != VK_SUCCESS || !oldMapped) {
            vkUnmapMemory(g_dev, replacement->mem);
            destroyBuffer(replacement);
            if (resumeBatch) batchBegin();
            return nullptr;
        }
        memcpy(values, oldMapped, oldBytes);
        vkUnmapMemory(g_dev, found->table->mem);
    }
    for (size_t pos = found->positions; pos < capacity; ++pos) {
        for (size_t pair = 0; pair < pairs; ++pair) {
            const float angle = (float)pos * found->inverseFrequency[pair];
            const size_t offset = (pos * pairs + pair) * 2;
            values[offset] = std::sin(angle);
            values[offset + 1] = std::cos(angle);
        }
    }
    vkUnmapMemory(g_dev, replacement->mem);
    if (found->table) destroyBuffer(found->table);
    found->table = replacement;
    found->positions = capacity;
    clearDescriptorBindingCache();
    if (resumeBatch) batchBegin();
    return found->table;
}

Buffer* gdnConvOutScratch(size_t bytes) {
    if (bytes == 0) bytes = 4;
    if (g_gdn_conv_out && g_gdn_conv_out->bytes >= bytes) return g_gdn_conv_out;

    if (g_gdn_conv_out) {
        if (g_batching) batchFlush();
        destroyBuffer(g_gdn_conv_out);
        g_gdn_conv_out = nullptr;
        clearDescriptorBindingCache();
    }
    size_t cap = scratchCapacity(bytes);
    if (g_maxBufferBytes > 0 && cap > g_maxBufferBytes) cap = bytes;
    g_gdn_conv_out = allocBuffer(cap, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    if (!g_gdn_conv_out && cap != bytes) {
        g_gdn_conv_out = allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    }
    if (!g_gdn_conv_out) {
        fprintf(stderr, "fak-vulkan: GDN conv_out scratch allocation failed (%zu bytes)\n", bytes);
        return nullptr;
    }
    return g_gdn_conv_out;
}

// Keep one grow-only readout allocation across layers and calls. Flush before
// replacing a buffer referenced by recorded commands, then resume recording.
Buffer* gdnPrefillReadoutScratch(size_t bytes) {
    if (bytes == 0 || (g_maxBufferBytes > 0 && bytes > g_maxBufferBytes)) return nullptr;
    if (g_gdn_prefill_readout && g_gdn_prefill_readout->bytes >= bytes) return g_gdn_prefill_readout;
    const bool resumeBatch = g_batching;
    if (g_gdn_prefill_readout) {
        if (resumeBatch) batchFlush();
        destroyBuffer(g_gdn_prefill_readout);
        g_gdn_prefill_readout = nullptr;
        clearDescriptorBindingCache();
    }
    size_t cap = scratchCapacity(bytes);
    if (g_maxBufferBytes > 0 && cap > g_maxBufferBytes) cap = bytes;
    g_gdn_prefill_readout = allocBuffer(cap, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    if (!g_gdn_prefill_readout && cap != bytes) {
        g_gdn_prefill_readout = allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    }
    if (resumeBatch && !g_batching) batchBegin();
    return g_gdn_prefill_readout;
}

void freeGdnScratch() {
    if (g_gdn_conv_out) {
        destroyBuffer(g_gdn_conv_out);
        g_gdn_conv_out = nullptr;
        clearDescriptorBindingCache();
    }
    if (g_gdn_prefill_readout) {
        destroyBuffer(g_gdn_prefill_readout);
        g_gdn_prefill_readout = nullptr;
        clearDescriptorBindingCache();
    }
}

// ---- one-shot command buffer helper --------------------------------------------
VkCommandBuffer allocBeginCmd();
// A one-shot command is a host fence: the in-flight batch completes first.
VkCommandBuffer beginCmd() {
    drainInflight();
    return allocBeginCmd();
}

VkCommandBuffer allocBeginCmd() {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_cmdpool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VKCHECK(vkAllocateCommandBuffers(g_dev, &ai, &cmd));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VKCHECK(vkBeginCommandBuffer(cmd, &bi));
    return cmd;
}

void submitWait(VkCommandBuffer cmd) {
    VKCHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKCHECK(vkResetFences(g_dev, 1, &g_submitFence));
    const uint64_t submitted = submitWaitNow();
    VKCHECK(vkQueueSubmit(g_queue, 1, &si, g_submitFence));
    syncSubmitted(submitted);
    VKCHECK(vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX));
    g_submitWaitNs.fetch_add(submitWaitNow() - submitted, std::memory_order_relaxed);
    syncSignaled(kSyncSiteSubmitWait, syncTakeStage(), submitted);
}

// Only a completed batch may transfer its command to the batch-local idle slot.
bool retainCompletedBatchCommand(VkCommandBuffer cmd);
void endSubmitWait(VkCommandBuffer cmd) {
    submitWait(cmd);
    if (!retainCompletedBatchCommand(cmd)) vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
}

VkCommandBuffer v41BeginCmdChecked() {
    drainInflight();
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_cmdpool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkResult r = vkAllocateCommandBuffers(g_dev, &ai, &cmd);
    if (r != VK_SUCCESS) {
        g_submissionStatus = r;
        return VK_NULL_HANDLE;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(cmd, &bi);
    if (r != VK_SUCCESS) {
        if (cmd) vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
        g_submissionStatus = r;
        return VK_NULL_HANDLE;
    }
    return cmd;
}

bool v41EndSubmitWaitChecked(VkCommandBuffer cmd, bool submit) {
    VkResult r = vkEndCommandBuffer(cmd);
    if (r == VK_SUCCESS && submit) r = vkResetFences(g_dev, 1, &g_submitFence);
    if (r == VK_SUCCESS && submit) {
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1;
        si.pCommandBuffers = &cmd;
        const uint64_t submitted = submitWaitNow();
        r = vkQueueSubmit(g_queue, 1, &si, g_submitFence);
        if (r != VK_SUCCESS) {
            // In particular VK_ERROR_DEVICE_LOST is equivalent to successful
            // submission for pending/in-use resource state. Treat every failed
            // submit conservatively: no failure result is a quiescence proof.
            g_v41SubmissionPendingFailure = true;
        } else {
            syncSubmitted(submitted);
            r = vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX);
            if (r != VK_SUCCESS) g_v41SubmissionPendingFailure = true;
            else g_submitWaitNs.fetch_add(submitWaitNow() - submitted, std::memory_order_relaxed);
            if (r == VK_SUCCESS) syncSignaled(kSyncSiteSubmitWait, syncTakeStage(), submitted);
        }
    }
    if (!g_v41SubmissionPendingFailure &&
        !(r == VK_SUCCESS && submit && retainCompletedBatchCommand(cmd))) {
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
    }
    if (r != VK_SUCCESS) {
        // Positive VkResult values (notably VK_TIMEOUT=2) must not collide with
        // this entrypoint's positive geometry/capability statuses.
        if (r > VK_SUCCESS) fprintf(stderr, "fak-vulkan: V4.1 incomplete submission VkResult=%d\n", int(r));
        g_submissionStatus = r < VK_SUCCESS ? r : VK_ERROR_UNKNOWN;
    }
    return r == VK_SUCCESS;
}

VkResult restoreBeginCommand() {
    if (g_submissionStatus != VK_SUCCESS) return g_submissionStatus;
    if (g_restore.cmd != VK_NULL_HANDLE) return VK_SUCCESS;
    drainInflight();
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = g_cmdpool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkResult r = vkAllocateCommandBuffers(g_dev, &ai, &g_restore.cmd);
    if (r != VK_SUCCESS) {
        g_restore.cmd = VK_NULL_HANDLE;
        return r;
    }
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    r = vkBeginCommandBuffer(g_restore.cmd, &bi);
    if (r != VK_SUCCESS) {
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &g_restore.cmd);
        g_restore.cmd = VK_NULL_HANDLE;
    }
    return r;
}

void restoreDiscardCommand() {
    if (g_v41SubmissionPendingFailure) return;
    if (g_restore.cmd != VK_NULL_HANDLE) {
        (void)vkEndCommandBuffer(g_restore.cmd);
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &g_restore.cmd);
        g_restore.cmd = VK_NULL_HANDLE;
    }
    g_restore.used = 0;
    g_restore.entries = 0;
    g_restore.payloadBytes = 0;
}

void restoreCleanup() {
    if (g_v41SubmissionPendingFailure) return;
    restoreDiscardCommand();
    if (g_restore.stage) {
        if (g_restore.mapped) vkUnmapMemory(g_dev, g_restore.stage->mem);
        destroyBuffer(g_restore.stage);
    }
    g_restore = RestoreTransaction{};
}

// ---- batched submission state ---------------------------------------------------
// When g_batching is true, compute dispatches RECORD into g_batchCmd (with a
// compute->compute barrier between them) instead of each submitting its own buffer. The
// descriptor sets recorded into the open buffer must outlive recording, so they are parked
// in g_batchSets and freed after the single submit. g_batchOps counts recorded ops so an
// empty flush is a cheap no-op.
VkCommandBuffer               g_batchCmd  = VK_NULL_HANDLE;
// Drained command reuse, adapted from ggml's completed command-pool recycling:
// ggml-org/llama.cpp 5e03a1ff450cae37a774684de55d14fb48b1f90b,
// ggml/src/ggml-vulkan/ggml-vulkan.cpp:1114-1137 (MIT).
// Copyright (c) 2023-2026 The ggml authors. See THIRD_PARTY_NOTICES.md.
// Reset one buffer, never the shared pool: restore and one-shot commands have
// independent ownership. This slot is separate from the active batch census.
VkCommandBuffer g_batchIdleCmd = VK_NULL_HANDLE;
VkDevice g_batchIdleDevice = VK_NULL_HANDLE;
VkCommandPool g_batchIdlePool = VK_NULL_HANDLE;
bool retainCompletedBatchCommand(VkCommandBuffer cmd) {
    if (!g_batching || cmd != g_batchCmd || g_v41SubmissionPendingFailure ||
        g_batchIdleCmd != VK_NULL_HANDLE) return false;
    // batchFlush clears the active alias before any subsequent batchBegin.
    g_batchIdleCmd = cmd;
    g_batchIdleDevice = g_dev;
    g_batchIdlePool = g_cmdpool;
    return true;
}
void discardIdleBatchCommand() {
    if (g_v41SubmissionPendingFailure) return;
    if (g_batchIdleCmd != VK_NULL_HANDLE) {
        vkFreeCommandBuffers(g_batchIdleDevice, g_batchIdlePool, 1, &g_batchIdleCmd);
    }
    g_batchIdleCmd = VK_NULL_HANDLE;
    g_batchIdleDevice = VK_NULL_HANDLE;
    g_batchIdlePool = VK_NULL_HANDLE;
}
VkCommandBuffer beginReusedBatchCommand() {
    // The slot has a successful fence proof, but belongs only to its original
    // device and pool. Observation-generation metadata is not execution policy.
    if (g_batchIdleDevice != g_dev || g_batchIdlePool != g_cmdpool) {
        g_submissionStatus = VK_ERROR_UNKNOWN;
        return VK_NULL_HANDLE;
    }
    VkResult r = vkResetCommandBuffer(g_batchIdleCmd, 0);
    if (r == VK_SUCCESS) {
        VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        r = vkBeginCommandBuffer(g_batchIdleCmd, &bi);
    }
    if (r != VK_SUCCESS) {
        g_submissionStatus = r < VK_SUCCESS ? r : VK_ERROR_UNKNOWN;
        if (r == VK_ERROR_DEVICE_LOST) g_v41SubmissionPendingFailure = true;
        // Ordinary errors cannot make this unsubmitted command pending. Device
        // loss instead retains the handle with all context owners until exit.
        discardIdleBatchCommand();
        return VK_NULL_HANDLE;
    }
    const VkCommandBuffer cmd = g_batchIdleCmd;
    g_batchIdleCmd = VK_NULL_HANDLE;
    g_batchIdleDevice = VK_NULL_HANDLE;
    g_batchIdlePool = VK_NULL_HANDLE;
    return cmd;
}
std::vector<DescriptorSetRecord> g_batchSets;
std::vector<Buffer*>          g_batchFreed;   // buffers freed mid-batch, recycled after submit
int                           g_batchOps  = 0;
bool                          g_batchHasV41 = false;
uint64_t                      g_batchD2DCount = 0;
uint64_t                      g_batchD2DBytes = 0;
bool                          g_batchD2DValid = true;

// V4.1 step batch (#13820). While open, uploads record into the batch through a
// per-batch host-visible arena and readbacks record their copy, submit once on
// the checked path and resume recording, so a decode step submits only where the
// host must observe a result. Status-owning kernels record their fill and
// dispatch into it; their status readback submits.
bool                          g_stepBatch = false;
Buffer*                       g_upArena = nullptr;
void*                         g_upArenaMapped = nullptr;
size_t                        g_upArenaCap = 0;
size_t                        g_upArenaUsed = 0;

// A full compute->compute barrier: every recorded op may read the previous op's output
// buffer, so each dispatch is fenced against the prior by a global shader-write->shader-read
// barrier. Coarse but correct; per-buffer barriers are a later refinement.
void recordComputeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    dp_inc(g_dp.barriers);
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Exact batch resource hazards (#12218). Default-off: when unarmed the recorder keeps the
// coarse global barrier and the recorded bytes are identical to the prior behavior. The Go
// ledger proves a window's declarations complete before arming, so the shim never has to
// infer hazard metadata it was not handed; an unarmed or incomplete window falls closed.
int g_batchHazardsArmed = 0;

// Per-dispatch sync verdicts keyed by the dispatch's ordinal within the batch window. The
// key is the op ordinal the shim itself assigns (g_batchOps BEFORE the increment), so a
// verdict can only ever elide the exact dispatch it was computed for. A missing entry
// fails closed to "needs sync": because Go declares only the matmul-family seams, every
// other dispatch (RMSNorm/RoPE/attention/...) simply has no entry and keeps its barrier.
// A positional queue would misalign the moment an undeclared dispatch interleaves, so the
// verdict is addressed, not streamed.
std::unordered_map<int, char> g_batchHazardSync; // ordinal -> 1 needs sync, 0 elidable

bool nextHazardSync(int ordinal) {
    if (!g_batchHazardsArmed) return true;
    auto it = g_batchHazardSync.find(ordinal);
    if (it == g_batchHazardSync.end()) return true;   // fail closed
    return it->second != 0;
}

// Record the barrier between two adjacent dispatches. When hazards are armed, a dispatch
// whose Go-side plan already proved it disjoint from its predecessor needs no device fence
// at all, so the elided case records nothing. The coarse path records the global barrier.
void recordDispatchBarrier(VkCommandBuffer cmd, int ordinal) {
    if (g_batchHazardsArmed && !nextHazardSync(ordinal)) {
        dp_inc(g_dp.barriersElided);
        if (g_tsOn) ++g_tsBarriersElided;
        return;
    }
    recordComputeBarrier(cmd);
}

// ---- GPU timestamp stage profile (FAK_VULKAN_TIMESTAMP_PROFILE=1, default off) ----------
// Serialized diagnostic only: enabling this disables pipelined batch cycles, even if
// timestamp queries are unavailable. The single query pool must not be reset while a
// prior batch is in flight. Bottom-of-pipe intervals include intervening barriers/copies;
// they are not isolated kernel-active time or the production pipeline's critical path.
// Only instrumented batched commands are covered; one-shot dispatches are omitted.
// Cumulative intervals are folded per (kernel id, groupsX, groupsY) shape and printed
// every FAK_VULKAN_TIMESTAMP_PROFILE_EVERY batches (default 32). Unset, no query commands
// are emitted and the command stream is byte-identical to the unprofiled one.
const char* const kKernelNames[] = {
    "matmul", "matmul_add", "matmul_argmax", "matmul_argmax_blocks", "matmul2", "matmul3",
    "rmsnorm", "rmsnorm_matmul", "rmsnorm_matmul2", "rmsnorm_matmul3", "rmsnorm_matmul_argmax_blocks",
    "rope", "swiglu", "swiglu_matmul_add", "add", "add_bias", "attention", "argmax", "argmax_pairs",
    "q8_matmul", "q8_matmul_decode", "q8_matmul2", "q8_matmul3", "rmsnorm_q8_matmul2", "rmsnorm_q8_matmul3",
    "swiglu_q8_matmul_add", "qwen35_gdn_q8_in_proj", "qwen35_gdn_conv", "qwen35_gdn_recurrent",
    "qwen35_gdn_prefill_tiled", "qwen35_gdn_prefill_norm", "qwen35_gdn_verify_tiled",
    "glm_kda_reread", "glm_kda_wave32", "q4k_matmul", "q4k_matmul_wave32", "q4k_matmul_coopmat",
    "q6k_matmul", "q5k_matmul", "q3k_matmul", "rmsnorm_q4k_matmul2", "swiglu_q4k_matmul_add", "q2k_matmul",
    "rmsnorm_q2k_matmul2", "qwen35_split_qg_panel", "qwen35_partial_rope_panel",
    "qwen35_causal_attention_panel", "attention_split_kv", "sigmoid_mul", "q2k_matvec", "rmsnorm_q8_matmul2_coop",
    "iq4xs_matvec", "iq3xxs_matvec", "iq2s_matvec", "iq3s_matvec", "iq2xxs_matvec", "iq2xs_matvec", "iq1s_matvec", "v41_tail_rope_qk", "v41_shared_attention", "v41_indexer_score", "v41_mhc", "v41_router_topk", "v41_index_select", "v41_q2k_embed_gather", "v41_exact_rmsnorm", "v41_bf16_boundary", "v41_logit_scale", "v41_bf16_checked", "v41_routed_swiglu", "v41_engram_gate", "v41_compressor_pool", "v41_tail_rope_exact", "v41_scale_exact", "v41_engram_row_decode", "v41_q2k_matvec_id", "v41_q3k_matvec_id", "v41_routed_swiglu_id", "v41_routed_publish_sum", "v41_shared_attention_fast", "v41_routed_gate_up_swiglu_id", "v41_routed_down_sum", "v41_q2k_token_panel", "v41_q3k_token_panel", "v41_routed_swiglu_panel_row", "v41_routed_down_panel_row", "v41_index_scale_bf16", "v41_routed_swiglu_panel_chunk", "v41_routed_down_panel_chunk", "v41_q2k_token_pair",
    "mm_cm1_q2k", "mm_cm1_iq2xxs", "mm_cm1_iq2xs", "mm_cm1_iq2s", "mm_cm1_iq3xxs", "mm_cm1_iq3s", "mm_cm1_iq1s", "mm_cm1_iq4xs", "mm_cm1_q4k", "v41_f32_finite",
    "qwen35_value_head_tile",
    "v41_router_gate_topk",
    "v41_compressor_ring",
    "v41_shared_attention_split",
};
static_assert(sizeof(kKernelNames) / sizeof(kKernelNames[0]) == K_COUNT, "kernel name table out of sync with KId");

// Pseudo-kernel ids for the transfer and fill commands a timed batch records.
enum TsPseudoKernel : uint32_t {
    kTsD2DCopy = (uint32_t)K_COUNT, kTsH2DUpload, kTsD2DGather, kTsFill, kTsReadback
};
const char* tsKernelName(uint32_t kid) {
    if (kid < (uint32_t)K_COUNT) return kKernelNames[kid];
    switch (kid) {
    case kTsD2DCopy: return "d2d_copy";
    case kTsH2DUpload: return "h2d_upload";
    case kTsD2DGather: return "d2d_gather";
    case kTsFill: return "fill";
    case kTsReadback: return "readback";
    default: return "unknown";
    }
}
const uint32_t kTsMax = 8192;
VkQueryPool g_tsPool = VK_NULL_HANDLE;
bool g_tsUsable = false, g_tsInitTried = false;
double g_tsPeriodNs = 1.0;
uint64_t g_tsMask = ~0ull;
uint32_t g_tsUsed = 0;
struct TsSlot { uint32_t kid, gx, gy; };
std::vector<TsSlot> g_tsSlots;
struct TsAgg { uint64_t n = 0; double ns = 0; };
std::unordered_map<uint64_t, TsAgg> g_tsAgg;
uint64_t g_tsBatches = 0, g_tsOps = 0, g_tsTruncated = 0;
double g_tsGpuNs = 0, g_tsWallNs = 0;
std::chrono::steady_clock::time_point g_tsBatchStart;
uint64_t g_tsEvery = [] { const char* v = std::getenv("FAK_VULKAN_TIMESTAMP_PROFILE_EVERY"); long n = v ? atol(v) : 0; return (uint64_t)(n > 0 ? n : 32); }();

void tsInit() {
    g_tsInitTried = true;
    fprintf(stderr, "fak-vulkan ts-profile mode=serialized_batches timing=bottom_of_pipe_intervals coverage=batched_commands_only pipeline_cycles=disabled\n");
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys, &props);
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qfs.data());
    uint32_t bits = g_qfam < qn ? qfs[g_qfam].timestampValidBits : 0;
    if (bits == 0 || props.limits.timestampPeriod <= 0) {
        fprintf(stderr, "fak-vulkan ts-profile unavailable: timestampValidBits=%u period=%f\n", bits, props.limits.timestampPeriod);
        return;
    }
    g_tsPeriodNs = props.limits.timestampPeriod;
    g_tsMask = bits >= 64 ? ~0ull : ((1ull << bits) - 1);
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = kTsMax;
    if (vkCreateQueryPool(g_dev, &qi, nullptr, &g_tsPool) != VK_SUCCESS) {
        g_tsPool = VK_NULL_HANDLE;
        fprintf(stderr, "fak-vulkan ts-profile unavailable: vkCreateQueryPool failed\n");
        return;
    }
    g_tsSlots.reserve(kTsMax);
    g_tsUsable = true;
}

void tsBatchBegin(VkCommandBuffer cmd) {
    if (!g_tsOn) return;
    if (!g_tsInitTried) tsInit();
    if (!g_tsUsable) return;
    vkCmdResetQueryPool(cmd, g_tsPool, 0, kTsMax);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsPool, 0);
    g_tsUsed = 1;
    g_tsSlots.clear();
    g_tsBatchStart = std::chrono::steady_clock::now();
}

// A compute op submitted outside a batch gets its own timestamp pair, so it lands in a
// row instead of the untimed count. Unread slots of a recorded batch would be clobbered,
// so with any pending the op stays untimed (and counted).
const uint32_t kTsOneShotSlots = 3; // begin, status fill, dispatch
bool tsOneShotBegin(VkCommandBuffer cmd) {
    if (!g_tsOn || g_batching) return false;
    if (!g_tsInitTried) tsInit();
    if (!g_tsUsable || g_tsUsed != 0) return false;
    vkCmdResetQueryPool(cmd, g_tsPool, 0, kTsOneShotSlots);
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsPool, 0);
    g_tsUsed = 1;
    g_tsSlots.clear();
    g_tsBatchStart = std::chrono::steady_clock::now();
    return true;
}

void tsAfterDispatch(VkCommandBuffer cmd, uint32_t kid, uint32_t gx, uint32_t gy) {
    if (!g_tsOn || !g_tsUsable) return;
    if (g_tsUsed >= kTsMax) { ++g_tsTruncated; return; }
    vkCmdWriteTimestamp(cmd, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, g_tsPool, g_tsUsed);
    g_tsSlots.push_back(TsSlot{kid, gx, gy});
    ++g_tsUsed;
}

void tsReport() {
    std::vector<std::pair<uint64_t, TsAgg>> rows(g_tsAgg.begin(), g_tsAgg.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    double perBatch = g_tsBatches ? 1.0 / (double)g_tsBatches : 0;
    fprintf(stderr, "fak-vulkan ts-profile batches=%llu ops/batch=%.1f gpu_ms/batch=%.3f wall_ms/batch=%.3f truncated=%llu mode=serialized_batches timing=bottom_of_pipe_intervals coverage=batched_commands_only\n",
        (unsigned long long)g_tsBatches, g_tsOps * perBatch, g_tsGpuNs * perBatch / 1e6, g_tsWallNs * perBatch / 1e6,
        (unsigned long long)g_tsTruncated);
    size_t shown = 0;
    for (const auto& r : rows) {
        if (shown++ >= 40) break;
        uint32_t kid = (uint32_t)(r.first >> 48), gx = (uint32_t)((r.first >> 20) & 0xFFFFFFF), gy = (uint32_t)(r.first & 0xFFFFF);
        fprintf(stderr, "fak-vulkan ts-stage kernel=%s groups=%ux%u calls/batch=%.2f ms/batch=%.3f us/call=%.2f share=%.1f%%\n",
            tsKernelName(kid), gx, gy, r.second.n * perBatch, r.second.ns * perBatch / 1e6,
            r.second.n ? r.second.ns / (double)r.second.n / 1e3 : 0, g_tsGpuNs > 0 ? 100.0 * r.second.ns / g_tsGpuNs : 0);
    }
}

// A recorded batch whose timestamps are never read (failed query, failed or skipped
// submit) counts its slots as truncated, so its lost GPU time cannot pass as complete.
void tsBatchLost() {
    if (g_tsOn && g_tsUsable && g_tsUsed >= 2) g_tsTruncated += g_tsUsed - 1;
    g_tsUsed = 0;
}

// A one-shot (batch=false) folds into the rows and totals but is not a batch, so the
// per-batch ts-stage figures stay per decode batch.
void tsCollect(bool batch) {
    if (!g_tsOn || !g_tsUsable || g_tsUsed < 2) return;
    std::vector<uint64_t> ts(g_tsUsed);
    if (vkGetQueryPoolResults(g_dev, g_tsPool, 0, g_tsUsed, ts.size() * sizeof(uint64_t), ts.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
        tsBatchLost();
        return;
    }
    double wall = (double)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - g_tsBatchStart).count();
    for (uint32_t i = 1; i < g_tsUsed; ++i) {
        uint64_t d = ((ts[i] - ts[i - 1]) & g_tsMask);
        const TsSlot& s = g_tsSlots[i - 1];
        uint64_t key = ((uint64_t)s.kid << 48) | ((uint64_t)(s.gx & 0xFFFFFFF) << 20) | (uint64_t)(s.gy & 0xFFFFF);
        TsAgg& a = g_tsAgg[key];
        a.n++;
        a.ns += d * g_tsPeriodNs;
    }
    g_tsGpuNs += ((ts[g_tsUsed - 1] - ts[0]) & g_tsMask) * g_tsPeriodNs;
    g_tsWallNs += wall;
    g_tsOps += g_tsUsed - 1;
    if (batch && (++g_tsBatches % g_tsEvery) == 0) tsReport();
    g_tsUsed = 0;
}

void tsBatchCollect() { tsCollect(true); }

// Reads the pair tsOneShotBegin opened, or counts its slots lost when the submit failed.
void tsOneShotEnd(bool completed) {
    if (completed) tsCollect(false);
    else tsBatchLost();
}

void batchBegin() {
	if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS) return;
	if (g_batching) return;          // already recording — the model brackets each token
	// A V4.1 window must latch command allocation/begin failures, not abort.
	g_batchCmd = g_batchIdleCmd != VK_NULL_HANDLE ? beginReusedBatchCommand() :
        (g_stepBatch ? v41BeginCmdChecked() : allocBeginCmd());
	if (g_batchCmd == VK_NULL_HANDLE) return;
	tsBatchBegin(g_batchCmd);
	g_batching = true;
	g_batchOps = 0;
	g_batchHasV41 = g_stepBatch;
	g_batchD2DCount = 0;
	g_batchD2DBytes = 0;
	g_batchD2DValid = true;
	g_batchSets.clear();
	// A new window owns a fresh hazard verdict queue; anything left from the prior window
	// must not leak into this one (the Go ledger re-lowers each window independently).
	g_batchHazardSync.clear();
}

// ---- pipelined batch cycle (FAK_VULKAN_PIPELINE_FLUSH, default on) ----------------
// A synchronous mid-token flush leaves the device idle while the host records the next
// batch, then pays a submit and a fence wake. batchCycle submits the closed batch WITHOUT
// waiting and retires only the batch before it, so the host records batch N+1 while the
// device runs batch N. The in-flight batch keeps its command buffer, descriptor sets and
// parked frees until its own fence; every host-sync path (one-shot command, host copy,
// buffer destruction, flush) drains it first, so no caller can observe it.
struct InflightBatch {
    bool active = false;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkFence fence = VK_NULL_HANDLE;
    std::vector<DescriptorSetRecord> sets;
    std::vector<Buffer*> freed;
    uint64_t d2dCount = 0;
    uint64_t d2dBytes = 0;
    bool d2dValid = true;
    uint32_t syncStage = UINT32_MAX;
};
InflightBatch g_inflight;
VkFence g_pipeFences[2] = {VK_NULL_HANDLE, VK_NULL_HANDLE};
const bool g_pipelineFlush = [] {
    const char* v = std::getenv("FAK_VULKAN_PIPELINE_FLUSH");
    return !(v && v[0] == '0' && v[1] == '\0');
}();
std::atomic<uint64_t> g_pipeCycles{0}, g_pipeFallbacks{0}, g_pipeDrains{0}, g_pipeWaitNs{0};

void drainInflight() {
    if (!g_inflight.active) return;
    InflightBatch done = std::move(g_inflight);
    g_inflight = InflightBatch{};
    const auto t0 = std::chrono::steady_clock::now();
    VKCHECK(vkWaitForFences(g_dev, 1, &done.fence, VK_TRUE, UINT64_MAX));
    g_syncPipeDrains.fetch_add(1, std::memory_order_release);
    syncSignaled(kSyncSitePipelineDrain, done.syncStage,
        (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(t0.time_since_epoch()).count(), false);
    if (g_dp_on) {
        g_pipeWaitNs.fetch_add((uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - t0).count(), std::memory_order_relaxed);
        g_pipeDrains.fetch_add(1, std::memory_order_relaxed);
    }
    if (!done.d2dValid || !checkedCounterAdd(g_d2dCount, done.d2dCount) ||
        !checkedCounterAdd(g_d2dBytes, done.d2dBytes)) {
        g_transferCountersValid = false;
    }
    for (auto it = done.sets.rbegin(); it != done.sets.rend(); ++it) {
        if (it->set) g_descSetPool[it->layout].push_back(*it);
    }
    if (g_batchIdleCmd == VK_NULL_HANDLE) {
        g_batchIdleCmd = done.cmd;
        g_batchIdleDevice = g_dev;
        g_batchIdlePool = g_cmdpool;
    } else {
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &done.cmd);
    }
    // Sets return to the pool before any release can clear the binding cache. The fence
    // proved these frees complete, so they recycle now even while a batch records.
    const bool batching = g_batching;
    g_batching = false;
    for (Buffer* b : done.freed) fvk_free(b);
    g_batching = batching;
}

VkFence pipelineFence() {
    const int slot = g_inflight.active && g_inflight.fence == g_pipeFences[0] ? 1 : 0;
    if (g_pipeFences[slot] == VK_NULL_HANDLE) {
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        if (vkCreateFence(g_dev, &fci, nullptr, &g_pipeFences[slot]) != VK_SUCCESS) {
            g_pipeFences[slot] = VK_NULL_HANDLE;
        }
    }
    return g_pipeFences[slot];
}

// Queue the closing batch behind the in-flight one before waiting on either.
void submitBehindInflightWait(VkCommandBuffer cmd) {
    VKCHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cmd;
    VKCHECK(vkResetFences(g_dev, 1, &g_submitFence));
    const uint64_t syncAt = submitWaitNow();
    VKCHECK(vkQueueSubmit(g_queue, 1, &si, g_submitFence));
    syncSubmitted(syncAt);
    const uint32_t syncStage = syncTakeStage();
    drainInflight();
    VKCHECK(vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX));
    syncSignaled(kSyncSiteBatchFlush, syncStage, syncAt);
    if (!retainCompletedBatchCommand(cmd)) vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
}

void batchFlush();

void batchCycle() {
    if (g_v41SubmissionPendingFailure || !g_batching) return;
    VkFence fence = VK_NULL_HANDLE;
    const bool eligible = g_pipelineFlush && !g_batchHasV41 && !g_stepBatch && !g_tsOn &&
        !g_batchHazardsArmed && g_submissionStatus == VK_SUCCESS;
    if (eligible && g_batchOps == 0) return;
    if (eligible) fence = pipelineFence();
    if (fence == VK_NULL_HANDLE) {
        if (g_dp_on) g_pipeFallbacks.fetch_add(1, std::memory_order_relaxed);
        batchFlush();
        batchBegin();
        return;
    }
    dpBatchSubmit();
    VKCHECK(vkEndCommandBuffer(g_batchCmd));
    VKCHECK(vkResetFences(g_dev, 1, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &g_batchCmd;
    const uint64_t syncAt = submitWaitNow();
    VKCHECK(vkQueueSubmit(g_queue, 1, &si, fence));
    syncSubmitted(syncAt);
    g_syncPipeSubmits.fetch_add(1, std::memory_order_relaxed);
    const uint32_t syncStage = syncTakeBatchStage();
    drainInflight();
    g_inflight.active = true;
    g_inflight.cmd = g_batchCmd;
    g_inflight.fence = fence;
    g_inflight.syncStage = syncStage;
    g_inflight.sets.swap(g_batchSets);
    g_inflight.freed.swap(g_batchFreed);
    g_inflight.d2dCount = g_batchD2DCount;
    g_inflight.d2dBytes = g_batchD2DBytes;
    g_inflight.d2dValid = g_batchD2DValid;
    g_batchCmd = VK_NULL_HANDLE;
    g_batching = false;
    g_batchOps = 0;
    g_batchD2DCount = 0;
    g_batchD2DBytes = 0;
    g_batchD2DValid = true;
    if (g_dp_on && (g_pipeCycles.fetch_add(1, std::memory_order_relaxed) + 1) % 1024 == 0) {
        fprintf(stderr, "fak-vulkan pipeline cycles=%llu fallbacks=%llu drains=%llu drain_wait_ms=%.3f\n",
            (unsigned long long)g_pipeCycles.load(), (unsigned long long)g_pipeFallbacks.load(),
            (unsigned long long)g_pipeDrains.load(), g_pipeWaitNs.load() / 1e6);
    }
    batchBegin();
    // The new batch's first access must follow every access of the in-flight one.
    if (g_batching) recordComputeBarrier(g_batchCmd);
}

void batchFlush() {
    SyncScope syncScope(kSyncSiteBatchFlush, true);
    if (g_v41SubmissionPendingFailure) return; // never retry an uncertain submit
    if (!g_batching) { drainInflight(); return; }
    bool hadWork = g_batchOps > 0;
    if (g_batchHasV41) {
        drainInflight();
        if (hadWork) dpBatchSubmit();
        const bool completed = v41EndSubmitWaitChecked(g_batchCmd, hadWork);
        if (!completed || !hadWork || g_v41SubmissionPendingFailure) tsBatchLost();
        if (g_v41SubmissionPendingFailure) return; // retain command, sets and parked buffers
        if (completed && hadWork) tsBatchCollect();
        if (completed && hadWork && (!g_batchD2DValid ||
            !checkedCounterAdd(g_d2dCount, g_batchD2DCount) ||
            !checkedCounterAdd(g_d2dBytes, g_batchD2DBytes))) g_transferCountersValid = false;
    } else if (hadWork) {
        dpBatchSubmit();
        if (g_inflight.active) submitBehindInflightWait(g_batchCmd);
        else endSubmitWait(g_batchCmd);   // single submit + fence for the whole recorded chain
        tsBatchCollect();
        if (!g_batchD2DValid ||
            !checkedCounterAdd(g_d2dCount, g_batchD2DCount) ||
            !checkedCounterAdd(g_d2dBytes, g_batchD2DBytes)) {
            g_transferCountersValid = false;
        }
    } else {
        VKCHECK(vkEndCommandBuffer(g_batchCmd));
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &g_batchCmd);
        tsBatchLost();
        drainInflight();
    }
    // Push back in reverse dispatch order. The pool is LIFO, so the next token's first
    // dispatch reuses the descriptor set that represented the previous token's first
    // equivalent dispatch, maximizing identical binding reuse.
    for (auto it = g_batchSets.rbegin(); it != g_batchSets.rend(); ++it) {
        if (it->set) g_descSetPool[it->layout].push_back(*it);
    }
    g_batchSets.clear();
    g_batchCmd = VK_NULL_HANDLE;
    // Clear the batching flag BEFORE recycling, so the fvk_free calls below take the real
    // recycle path instead of re-parking into g_batchFreed. The recorded ops have executed
    // (endSubmitWait fenced), so every parked buffer is now safe to return to the pool.
	g_batching = false;
	g_batchOps = 0;
	g_batchHasV41 = false;
	g_batchD2DCount = 0;
	g_batchD2DBytes = 0;
	g_batchD2DValid = true;
	g_upArenaUsed = 0;
	std::vector<Buffer*> freed;   freed.swap(g_batchFreed);
	for (Buffer* b : freed)   fvk_free(b);
}

// A copy into a buffer an earlier recorded op read or wrote: order it after
// every prior shader and transfer access.
void recordTransferBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    dp_inc(g_dp.barriers);
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

bool stepBatchRecording() {
    return g_stepBatch && g_batching && g_batchCmd != VK_NULL_HANDLE && !g_v41SubmissionPendingFailure;
}

// Reserve bytes in the upload arena. A full arena is reused only after the
// recorded copies that read it have completed; it grows only when empty.
bool stepUploadReserve(size_t bytes, size_t* offset) {
    const size_t aligned = (g_upArenaUsed + 15) & ~(size_t)15;
    if (g_upArena && aligned >= g_upArenaUsed && aligned <= g_upArenaCap && bytes <= g_upArenaCap - aligned) {
        *offset = aligned;
        return true;
    }
    if (g_upArenaUsed > 0) {
        SyncScope syncScope(kSyncSiteUpload);
        batchFlush();
        if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || g_batching) return false;
        batchBegin();
        if (!stepBatchRecording()) return false;
    }
    if (!g_upArena || bytes > g_upArenaCap) {
        size_t cap = g_upArenaCap ? g_upArenaCap : ((size_t)1 << 20);
        while (cap < bytes) {
            if (cap > std::numeric_limits<size_t>::max() / 2) { cap = bytes; break; }
            cap *= 2;
        }
        if (g_maxBufferBytes > 0 && cap > g_maxBufferBytes) cap = bytes;
        if (g_upArena) {
            if (g_upArenaMapped) vkUnmapMemory(g_dev, g_upArena->mem);
            destroyBuffer(g_upArena);
            g_upArena = nullptr;
            g_upArenaMapped = nullptr;
            g_upArenaCap = 0;
        }
        Buffer* arena = allocBuffer(cap,
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
        if (!arena) return false;
        void* mapped = nullptr;
        if (vkMapMemory(g_dev, arena->mem, 0, cap, 0, &mapped) != VK_SUCCESS || !mapped) {
            destroyBuffer(arena);
            return false;
        }
        g_upArena = arena;
        g_upArenaMapped = mapped;
        g_upArenaCap = cap;
    }
    *offset = 0;
    return true;
}

// Record one staged host->device copy into the open step batch. False leaves
// nothing recorded, so the caller keeps the synchronous upload.
bool stepRecordUpload(Buffer* dst, const void* host, size_t bytes) {
    if (!stepBatchRecording() || !dst || dst->mapped || !host || bytes == 0 || bytes > dst->bytes) return false;
    size_t offset = 0;
    if (!stepUploadReserve(bytes, &offset)) return false;
    memcpy((unsigned char*)g_upArenaMapped + offset, host, bytes);
    if (g_batchOps > 0) recordTransferBarrier(g_batchCmd);
    VkBufferCopy region{offset, 0, bytes};
    vkCmdCopyBuffer(g_batchCmd, g_upArena->buf, dst->buf, 1, &region);
    tsAfterDispatch(g_batchCmd, kTsH2DUpload, 0, 0);
    ++g_batchOps;
    g_upArenaUsed = offset + bytes;
    if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes)) {
        g_transferCountersValid = false;
    }
    return true;
}

// Record the readback copy behind every prior recorded op, submit the step's
// recorded chain once on the checked path, then resume recording.
int stepRecordReadback(void* host, Buffer* src, size_t bytes) {
    SyncScope syncScope(kSyncSiteReadback);
    int stagingStatus = VK_SUCCESS;
    Buffer* stage = stagingBuffer(bytes, &stagingStatus);
    if (!stage) {
        if (stagingStatus == VK_ERROR_DEVICE_LOST) {
            g_submissionStatus = VK_ERROR_DEVICE_LOST;
            g_v41SubmissionPendingFailure = true;
        }
        return stagingStatus == VK_SUCCESS ? FVK_D2H_STAGING_ALLOCATION_FAILED : stagingStatus;
    }
    recordComputeBarrier(g_batchCmd);
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(g_batchCmd, src->buf, stage->buf, 1, &region);
    tsAfterDispatch(g_batchCmd, kTsReadback, 0, 0);
    ++g_batchOps;
    batchFlush();
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || g_batching) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
    }
    memcpy(host, g_stageMapped, bytes);
    syncNoteD2H(bytes);
    if (!checkedCounterAdd(g_d2hCount, 1) || !checkedCounterAdd(g_d2hBytes, bytes)) {
        g_transferCountersValid = false;
    }
    batchBegin();
    return (int)g_submissionStatus;
}

// An operation that owns its own command buffer never joins a batch. Inside a
// step batch it fences the recorded chain first and resumes recording after.
struct V41StepSuspend {
    bool resume = false;
    V41StepSuspend() {
        if (!g_stepBatch || !g_batching) return;
        SyncScope syncScope(kSyncSiteStepSuspend);
        batchFlush();
        resume = !g_batching && !g_v41SubmissionPendingFailure && g_submissionStatus == VK_SUCCESS;
    }
    ~V41StepSuspend() { if (resume && g_stepBatch) batchBegin(); }
};

// UMA direct upload: the destination is coherent device-local memory the CPU maps, so the source
// (often file-backed checkpoint pages) is copied once, with no staging buffer or transfer submit.
void copyHostToMappedWeight(Buffer* dst, const void* host, size_t bytes) {
    memcpy(dst->mapped, host, bytes);
    if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes) ||
        !checkedCounterAdd(g_directH2DBytes, bytes)) {
        g_transferCountersValid = false;
    }
}

// staging copy host<->device through one persistent HOST_VISIBLE scratch buffer.
void copyHostToDevice(Buffer* dst, const void* host, size_t bytes) {
    SyncScope syncScope(kSyncSiteUpload);
    if (g_v41SubmissionPendingFailure) return;
    if (bytes == 0 || !dst || !host) return;
    drainInflight();
    if (dst->mapped) {
        copyHostToMappedWeight(dst, host, bytes);
        return;
    }
    Buffer* stage = stagingBuffer(bytes);
    if (!stage) return;
    memcpy(g_stageMapped, host, bytes);
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, stage->buf, dst->buf, 1, &region);
    dpOneShot(g_dp.oneShotH2D);
    endSubmitWait(cmd);
    if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes)) {
        g_transferCountersValid = false;
    }
}

int copyDeviceToHost(void* host, Buffer* src, size_t bytes) {
    SyncScope syncScope(kSyncSiteReadback);
    if (g_v41SubmissionPendingFailure) return (int)g_submissionStatus;
    if (bytes == 0) return VK_SUCCESS;
    if (!host || !src) return VK_ERROR_INITIALIZATION_FAILED;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    int stagingStatus = VK_SUCCESS;
    Buffer* stage = stagingBuffer(bytes, &stagingStatus);
    if (!stage) return stagingStatus == VK_SUCCESS ? FVK_D2H_STAGING_ALLOCATION_FAILED : stagingStatus;
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, src->buf, stage->buf, 1, &region);
    dpOneShot(g_dp.oneShotD2H);
    endSubmitWait(cmd);
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    memcpy(host, g_stageMapped, bytes);
    syncNoteD2H(bytes);
    if (!checkedCounterAdd(g_d2hCount, 1) || !checkedCounterAdd(g_d2hBytes, bytes)) {
        g_transferCountersValid = false;
    }
    return VK_SUCCESS;
}

// ---- SPIR-V load + pipeline build ----------------------------------------------
std::vector<char> readFile(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "fak-vulkan: cannot open SPIR-V %s\n", path.c_str()); return {}; }
    fseek(f, 0, SEEK_END);
    long sz = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> data(sz);
    if (fread(data.data(), 1, sz, f) != (size_t)sz) { fclose(f); return {}; }
    fclose(f);
    return data;
}

// This is a narrow ABI/NoContraction admission check, not a build-receipt or
// full SPIR-V validator. The verified current bundle remains a pre-launch gate.
std::vector<char> readV41TailRoPEModule(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) return {};
    if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return {}; }
    const long size = ftell(f);
    // This single small fixed kernel does not admit an unbounded allocation from
    // a malformed module file. Ordinary kernel loading is deliberately untouched.
    if (size < 20 || size > 1024 * 1024 || size % 4 != 0 || fseek(f, 0, SEEK_SET) != 0) {
        fclose(f); return {};
    }
    std::vector<char> code(static_cast<size_t>(size));
    const bool complete = fread(code.data(), 1, code.size(), f) == code.size();
    const bool closed = fclose(f) == 0;
    return complete && closed ? code : std::vector<char>{};
}

// Exact immutable module attestation; ordinary scalar/BF16 admission is unchanged.
#include "v41_index_scale_bf16_spirv.h"
#include "v41_routed_swiglu_chunk_spirv.h"
#include "v41_routed_down_chunk_spirv.h"
#include "v41_q2k_token_pair_spirv.h"

// mm_cm1_quant.glsl multiplies 16x16x16 fp16 A/B into fp32 C at subgroup scope; the
// device must list that exact combination, not merely expose the extension.
static bool coopMatF16Shape16Supported() {
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    auto query = reinterpret_cast<PFN_vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR>(
        vkGetInstanceProcAddr(g_instance, "vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR"));
    if (!query) return false;
    uint32_t n = 0;
    if (query(g_phys, &n, nullptr) != VK_SUCCESS || n == 0) return false;
    std::vector<VkCooperativeMatrixPropertiesKHR> props(n, {VK_STRUCTURE_TYPE_COOPERATIVE_MATRIX_PROPERTIES_KHR});
    if (query(g_phys, &n, props.data()) != VK_SUCCESS) return false;
    for (uint32_t i = 0; i < n; ++i) {
        const VkCooperativeMatrixPropertiesKHR& p = props[i];
        if (p.MSize == 16 && p.NSize == 16 && p.KSize == 16 &&
            p.AType == VK_COMPONENT_TYPE_FLOAT16_KHR && p.BType == VK_COMPONENT_TYPE_FLOAT16_KHR &&
            p.CType == VK_COMPONENT_TYPE_FLOAT32_KHR && p.ResultType == VK_COMPONENT_TYPE_FLOAT32_KHR &&
            p.scope == VK_SCOPE_SUBGROUP_KHR) {
            return true;
        }
    }
#endif
    return false;
}

bool buildKernel(Kernel& k, const std::string& spvPath, int nbuf, uint32_t pcsize, uint32_t subgroupSize = 0) {
    const bool fixedV41 = &k == &g_kern[K_V41_F32_FINITE] || &k == &g_kern[K_V41_Q2K_TOKEN_PAIR] || &k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK] || &k == &g_kern[K_V41_INDEX_SCALE_BF16] || &k == &g_kern[K_V41_TAIL_ROPE_QK] || &k == &g_kern[K_V41_TAIL_ROPE_EXACT] || &k == &g_kern[K_V41_SHARED_ATTENTION] || &k == &g_kern[K_V41_SHARED_ATTENTION_FAST] || &k == &g_kern[K_V41_INDEXER_SCORE] || &k == &g_kern[K_V41_BF16_BOUNDARY] || &k == &g_kern[K_V41_LOGIT_SCALE] || &k == &g_kern[K_V41_BF16_CHECKED] || &k == &g_kern[K_V41_SCALE_EXACT] || &k == &g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID] || &k == &g_kern[K_V41_ROUTED_DOWN_SUM];
    std::vector<char> code = fixedV41 ? readV41TailRoPEModule(spvPath) : readFile(spvPath);
    if (&k == &g_kern[K_V41_Q3K_TOKEN_PANEL]) g_v41Q3TokenGridModule = false;
    if (&k == &g_kern[K_V41_Q2K_TOKEN_PANEL]) g_v41Q2TokenGridModule = false;
    if (!spirvModuleFraming(code)) return false;
    const int q3GridABI = &k == &g_kern[K_V41_Q3K_TOKEN_PANEL] ? v41TokenGridModuleABI(code, 0x51334731u) : 0;
    const int q2GridABI = &k == &g_kern[K_V41_Q2K_TOKEN_PANEL] ? v41TokenGridModuleABI(code, 0x51324731u) : 0;
    // A distinct optional filename is insufficient: require the encoded row
    // addressing marker before any pipeline can advertise this ABI.
    if (&k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW] &&
        (nbuf != 4 || pcsize != 16 || v41TokenGridModuleABI(code, 0x52535231u) != 1)) return false;
    if (&k == &g_kern[K_V41_ROUTED_DOWN_PANEL_ROW] &&
        (nbuf != 3 || pcsize != 12 || v41TokenGridModuleABI(code, 0x52445031u) != 1)) return false;
    if (&k == &g_kern[K_V41_INDEX_SCALE_BF16] &&
        (nbuf != 3 || pcsize != 8 || !g_v41IndexerFloatControls || !v41IndexScaleBF16ABI(code))) return false;
    if (&k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK] &&
        (nbuf != 4 || pcsize != 48 || !g_v41IndexerFloatControls || !v41RoutedSwiGLUChunkABI(code))) return false;
    if (&k == &g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK] &&
        (nbuf != 3 || pcsize != 16 || !v41RoutedDownChunkABI(code))) return false;
    if (&k == &g_kern[K_V41_Q2K_TOKEN_PAIR] &&
        (nbuf != 5 || pcsize != 12 || !g_v41IndexerFloatControls || !v41Q2KTokenPairABI(code))) return false;
    if (q2GridABI < 0) return false;
    if (q3GridABI < 0) return false;
    if (&k == &g_kern[K_V41_F32_FINITE] &&
        (nbuf != 2 || pcsize != 8 || !v41F32FiniteABI(code))) return false;
    if (&k == &g_kern[K_V41_BF16_BOUNDARY] &&
        (nbuf != 2 || pcsize != 4 || !v41BF16BoundaryABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible integer BF16 boundary module in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_BF16_CHECKED] &&
        (nbuf != 3 || pcsize != 8 || !v41BF16CheckedABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible checked BF16 module in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_LOGIT_SCALE] &&
        (nbuf != 2 || pcsize != 8 || !v41LogitScaleABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible exact logit scale module in %s\n", spvPath.c_str());
        return false;
    }
    // The strict scalar twin's F32 execution modes are valid only on a device
    // reporting the matching controls; the proof binds the exact loaded bytes.
    if (&k == &g_kern[K_V41_SCALE_EXACT] &&
        (nbuf != 2 || pcsize != 8 || !g_v41IndexerFloatControls || !v41LogitScaleABI(code))) {
        fprintf(stderr, "fak-vulkan: strict V4.1 scalar multiply module lacks admitted modes or selected-device controls in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_SWIGLU] && !swigluPushConstantABI(code)) {
        fprintf(stderr, "fak-vulkan: incompatible SwiGLU push-constant ABI in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_TAIL_ROPE_QK] &&
        (nbuf != 5 || pcsize != 12 || !v41TailRoPEABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible V4.1 tail RoPE module in %s\n", spvPath.c_str());
        return false;
    }
#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
    if (&k == &g_kern[K_V41_TAIL_ROPE_QK] &&
        (!g_v41IndexerFloatControls || !v41TailRoPEABI(code, true))) {
        fprintf(stderr, "fak-vulkan: exact tail RoPE witness lacks admitted module modes or selected-device controls in %s\n", spvPath.c_str());
        return false;
    }
#endif
    // The strict twin's three F32 execution modes are valid only on a device
    // reporting the matching controls; the proof binds the exact loaded bytes.
    if (&k == &g_kern[K_V41_TAIL_ROPE_EXACT] &&
        (nbuf != 5 || pcsize != 12 || !g_v41IndexerFloatControls || !v41TailRoPEABI(code, true))) {
        fprintf(stderr, "fak-vulkan: strict V4.1 tail RoPE module lacks admitted modes or selected-device controls in %s\n", spvPath.c_str());
        return false;
    }
    if ((&k == &g_kern[K_V41_SHARED_ATTENTION] || &k == &g_kern[K_V41_SHARED_ATTENTION_FAST]) &&
        (nbuf != 5 || pcsize != 24 || !v41SharedAttentionABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible V4.1 shared attention module in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_INDEXER_SCORE] &&
        (nbuf != 4 || pcsize != 12 || !v41IndexerScoreABI(code))) {
        g_v41IndexerScoreReason = "indexer module lacks fixed ABI, binary32 execution modes or scalar NoContraction proof";
        fprintf(stderr, "fak-vulkan: incompatible V4.1 indexer score module in %s\n", spvPath.c_str());
        return false;
    }
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size();
    smi.pCode = reinterpret_cast<const uint32_t*>(code.data());
    if (vkCreateShaderModule(g_dev, &smi, nullptr, &k.shader) != VK_SUCCESS) return false;

    std::vector<VkDescriptorSetLayoutBinding> binds(nbuf);
    for (int i = 0; i < nbuf; ++i) {
        binds[i].binding = (uint32_t)i;
        binds[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        binds[i].descriptorCount = 1;
        binds[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo dli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    dli.bindingCount = (uint32_t)nbuf;
    dli.pBindings = binds.data();
    if (vkCreateDescriptorSetLayout(g_dev, &dli, nullptr, &k.dsl) != VK_SUCCESS) return false;

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &k.dsl;
    if (pcsize > 0) { pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr; }
    if (vkCreatePipelineLayout(g_dev, &pli, nullptr, &k.layout) != VK_SUCCESS) return false;

    VkPipelineShaderStageCreateInfo stage{VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO};
    stage.stage = VK_SHADER_STAGE_COMPUTE_BIT;
    stage.module = k.shader;
    stage.pName = "main";
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfoEXT requiredSubgroup{
        VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO_EXT};
    if (subgroupSize != 0) {
        requiredSubgroup.requiredSubgroupSize = subgroupSize;
        stage.pNext = &requiredSubgroup;
    }
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = k.layout;
    if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe) != VK_SUCCESS) return false;

    k.nbuf = nbuf;
    k.pcsize = pcsize;
    if (&k == &g_kern[K_V41_Q3K_TOKEN_PANEL]) g_v41Q3TokenGridModule = q3GridABI == 1;
    if (&k == &g_kern[K_V41_Q2K_TOKEN_PANEL]) g_v41Q2TokenGridModule = q2GridABI == 1;
#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
    if (&k == &g_kern[K_V41_TAIL_ROPE_QK]) g_v41TailRoPEExactModule = true;
#endif
    return true;
}

DescriptorSetRecord acquireDescriptorSet(Kernel& k, bool checkedV41 = false) {
    auto& bucket = g_descSetPool[k.dsl];
    if (!bucket.empty()) {
        DescriptorSetRecord rec = bucket.back();
        bucket.pop_back();
        return rec;
    }

    VkDescriptorSetAllocateInfo dsi{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsi.descriptorPool = g_descpool;
    dsi.descriptorSetCount = 1;
    dsi.pSetLayouts = &k.dsl;
    DescriptorSetRecord rec{};
    rec.layout = k.dsl;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    const VkResult allocated = vkAllocateDescriptorSets(g_dev, &dsi, &ds);
    if (allocated != VK_SUCCESS) {
        if (checkedV41) {
            g_submissionStatus = allocated < VK_SUCCESS ? allocated : VK_ERROR_UNKNOWN;
            // Device loss may affect previously recorded/submitted work. Keep
            // all owners, command buffers and pools quarantined for this process.
            if (allocated == VK_ERROR_DEVICE_LOST) g_v41SubmissionPendingFailure = true;
        }
        fprintf(stderr, "fak-vulkan: descriptor alloc failed\n");
        return rec;
    }
    rec.set = ds;
    return rec;
}

void recycleDescriptorSet(DescriptorSetRecord rec) {
    if (rec.set) g_descSetPool[rec.layout].push_back(rec);
}

// dispatch: bind `bufs` (nbuf of them) + push constants, run groupsX*groupsY workgroups.
bool dispatch(Kernel& k, Buffer** bufs, const void* pc, uint32_t pcsize, uint32_t groupsX, uint32_t groupsY = 1, bool forceChecked = false, Buffer* initializedStatus = nullptr, uint32_t statusBytes = 16) {
    if (g_v41SubmissionPendingFailure) return false;
    // Only these concrete synchronous kernels may request the bounded status
    // initialization. They never flush a batch; they record only into a V4.1 step batch.
    const bool finiteOnly = &k == &g_kern[K_V41_F32_FINITE];
    const bool routedChunk = &k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK];
    const bool downChunk = &k == &g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK];
    const bool routedID = &k == &g_kern[K_V41_ROUTED_SWIGLU_ID] || &k == &g_kern[K_V41_ROUTED_PUBLISH_SUM] ||
        &k == &g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID] || &k == &g_kern[K_V41_ROUTED_DOWN_SUM];
    if ((&k == &g_kern[K_V41_BF16_CHECKED] || &k == &g_kern[K_V41_ROUTED_DOWN_PANEL_ROW] || &k == &g_kern[K_V41_INDEX_SCALE_BF16] || &k == &g_kern[K_V41_ROUTED_SWIGLU] || &k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW] || routedID || routedChunk || downChunk || finiteOnly) && initializedStatus == nullptr) return false;
    if (initializedStatus != nullptr) {
        // The status binding is the last one of each status-owning kernel. Only
        // routed id, activation and down chunks can own multiple 16-byte records.
        const bool checkedFinite = finiteOnly && k.nbuf == 2;
        const bool checkedBF16 = (&k == &g_kern[K_V41_BF16_CHECKED] || &k == &g_kern[K_V41_ROUTED_DOWN_PANEL_ROW] || &k == &g_kern[K_V41_INDEX_SCALE_BF16] || downChunk) && k.nbuf == 3;
        const bool routedSwiGLU = (&k == &g_kern[K_V41_ROUTED_SWIGLU] || &k == &g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW] || routedChunk) && k.nbuf == 4;
        const bool routedSwiGLUID = &k == &g_kern[K_V41_ROUTED_SWIGLU_ID] && k.nbuf == 5;
        const bool routedPublishSum = &k == &g_kern[K_V41_ROUTED_PUBLISH_SUM] && k.nbuf == 4;
        const bool routedFusedGateUp = &k == &g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID] && k.nbuf == 20;
        const bool routedFusedDownSum = &k == &g_kern[K_V41_ROUTED_DOWN_SUM] && k.nbuf == 12;
        if ((!checkedFinite && !checkedBF16 && !routedSwiGLU && !routedSwiGLUID && !routedPublishSum && !routedFusedGateUp && !routedFusedDownSum) || !forceChecked || (g_batching && !g_stepBatch) ||
            !bufs || bufs[k.nbuf - 1] != initializedStatus || initializedStatus->buf == VK_NULL_HANDLE ||
            statusBytes < 16 || statusBytes % 16 != 0 || (statusBytes != 16 && !routedID && !routedChunk && !downChunk) ||
            ((routedChunk || downChunk) && ((groupsY != 2 && groupsY != 4 && groupsY != 8) || statusBytes != 16u * groupsY)) ||
            initializedStatus->bytes < statusBytes) return false;
    }
    // Ordinary scalar kernels inside a checked step also fail closed on
    // descriptor allocation; skipping a dispatch must never publish stale data.
    const bool checkedV41 = forceChecked || g_stepBatch || &k == &g_kern[K_V41_TAIL_ROPE_QK] || &k == &g_kern[K_V41_TAIL_ROPE_EXACT] || &k == &g_kern[K_V41_SHARED_ATTENTION] || &k == &g_kern[K_V41_SHARED_ATTENTION_FAST] || &k == &g_kern[K_V41_SHARED_ATTENTION_SPLIT] || &k == &g_kern[K_V41_INDEXER_SCORE] || &k == &g_kern[K_V41_MHC] || &k == &g_kern[K_V41_ROUTER_TOPK] || &k == &g_kern[K_V41_INDEX_SELECT] || &k == &g_kern[K_V41_Q2K_EMBED_GATHER] || &k == &g_kern[K_V41_EXACT_RMSNORM] || &k == &g_kern[K_V41_BF16_BOUNDARY] || &k == &g_kern[K_V41_LOGIT_SCALE] || &k == &g_kern[K_V41_ENGRAM_GATE] || &k == &g_kern[K_V41_COMPRESSOR_POOL] || &k == &g_kern[K_V41_COMPRESSOR_RING] || &k == &g_kern[K_V41_SCALE_EXACT] || &k == &g_kern[K_V41_ENGRAM_ROW_DECODE] || &k == &g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID] || &k == &g_kern[K_V41_ROUTED_DOWN_SUM];
    if (k.nbuf > MAX_DISPATCH_BUFS) {
        fprintf(stderr, "fak-vulkan: dispatch skipped; kernel has %d buffers, max %d\n",
                k.nbuf, MAX_DISPATCH_BUFS);
        return false;
    }
    for (int i = 0; i < k.nbuf; ++i) {
        if (!bufs[i]) {
            fprintf(stderr, "fak-vulkan: dispatch skipped; buffer %d is null\n", i);
            return false;
        }
    }
    DescriptorSetRecord rec = acquireDescriptorSet(k, checkedV41);
    if (!rec.set) {
        return false;
    }
    bool sameBindings = rec.nbuf == k.nbuf;
    for (int i = 0; i < k.nbuf; ++i) {
        if (rec.buffers[i] != bufs[i]->buf) {
            sameBindings = false;
            break;
        }
    }
    if (!sameBindings) {
        VkDescriptorBufferInfo dbi[MAX_DISPATCH_BUFS]{};
        VkWriteDescriptorSet wr[MAX_DISPATCH_BUFS]{};
        for (int i = 0; i < k.nbuf; ++i) {
            dbi[i].buffer = bufs[i]->buf;
            dbi[i].offset = 0;
            dbi[i].range  = VK_WHOLE_SIZE;
            wr[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            wr[i].dstSet = rec.set;
            wr[i].dstBinding = (uint32_t)i;
            wr[i].descriptorCount = 1;
            wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
            rec.buffers[i] = bufs[i]->buf;
        }
        rec.nbuf = k.nbuf;
        for (int i = k.nbuf; i < MAX_DISPATCH_BUFS; ++i) rec.buffers[i] = VK_NULL_HANDLE;
        vkUpdateDescriptorSets(g_dev, (uint32_t)k.nbuf, wr, 0, nullptr);
    }

    if (g_batching) {
        if (g_batchCmd == VK_NULL_HANDLE) { recycleDescriptorSet(rec); return false; }
        if (checkedV41) g_batchHasV41 = true;
        // RECORD into the open batch buffer: barrier against the prior op, then dispatch.
        // The descriptor set must outlive recording, so park it for post-submit free.
        // When the Go ledger armed exact hazards, it also queued this dispatch's sync
        // verdict by ordinal; an elided verdict (proven disjoint) records no device fence.
        if (initializedStatus != nullptr) {
            // Step batch only: the same fill-then-conservative-dependency contract
            // as the one-shot path, recorded; the status readback submits it.
            if (g_batchOps > 0) recordTransferBarrier(g_batchCmd);
            vkCmdFillBuffer(g_batchCmd, initializedStatus->buf, 0, statusBytes, UINT32_MAX);
            tsAfterDispatch(g_batchCmd, kTsFill, 0, 0);
            recordComputeBarrier(g_batchCmd);
        } else if (g_batchOps > 0) {
            recordDispatchBarrier(g_batchCmd, g_batchOps);
        }
        vkCmdBindPipeline(g_batchCmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(g_batchCmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &rec.set, 0, nullptr);
        if (pcsize > 0) vkCmdPushConstants(g_batchCmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
        dpDispatch(k);
        vkCmdDispatch(g_batchCmd, groupsX, groupsY, 1);
        tsAfterDispatch(g_batchCmd, (uint32_t)(&k - g_kern), groupsX, groupsY);
        g_batchSets.push_back(rec);
        ++g_batchOps;
        return true;
    }

    // Unbatched: one-shot submit + fence (the original per-op path).
    VkCommandBuffer cmd = checkedV41 ? v41BeginCmdChecked() : beginCmd();
    if (cmd == VK_NULL_HANDLE) { recycleDescriptorSet(rec); return false; }
    const bool timed = tsOneShotBegin(cmd);
    if (initializedStatus != nullptr) {
        // Completion alone is not memory visibility. This same-command fill
        // and conservative dependency cover status atomics AND prior input writes.
        vkCmdFillBuffer(cmd, initializedStatus->buf, 0, statusBytes, UINT32_MAX);
        if (timed) tsAfterDispatch(cmd, kTsFill, 0, 0);
        recordComputeBarrier(cmd);
    }
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &rec.set, 0, nullptr);
    if (pcsize > 0) vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
    dpDispatch(k);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
    if (timed) tsAfterDispatch(cmd, (uint32_t)(&k - g_kern), groupsX, groupsY);
    dpOneShot(g_dp.oneShotCompute, timed);
    if (checkedV41) {
        const bool completed = v41EndSubmitWaitChecked(cmd, true);
        if (timed) tsOneShotEnd(completed && !g_v41SubmissionPendingFailure);
        if (!g_v41SubmissionPendingFailure) recycleDescriptorSet(rec);
        return completed;
    }
    endSubmitWait(cmd);
    if (timed) tsOneShotEnd(g_submissionStatus == VK_SUCCESS);
    recycleDescriptorSet(rec);
    return g_submissionStatus == VK_SUCCESS;
}

inline Buffer* B(const void* h) { return (Buffer*)h; }
inline Buffer* B(void* h)       { return (Buffer*)h; }

// ---- GDN register-resident self-check --------------------------------------------------
//
// One geometry-checked self-check per (geometry, width class), run on the same
// device that will execute the candidate, once before the first forward and never
// inside one. Both the stock recurrent shader and the register-resident candidate
// run on identical synthetic operands; the verdict admits a class only when the
// normwise max relative deviation of both `core` and the final recurrent state
// stays within kGdnVerifyDeviationMax. The candidates reassociate the FP32 sums,
// so bit identity is explicitly not demanded (contrast upstream, which does demand
// it). The stock kernel is used on any disagreement.
//
// Two width classes exist: the verify-width class (1..7 tokens) uses the DVPL=2
// verify shader, and the prefill class (>=8 tokens) uses the 16-token tiled shader.
// Both are checked here so a selected candidate of either class falls back to stock
// when it disagrees on the device in use.
static constexpr float kGdnVerifyDeviationMax = 1e-6f;
constexpr int kGdnVerifyWidthCount = 4;
constexpr int kGdnVerifyWidths[kGdnVerifyWidthCount] = {1, 4, 16, 64};
// True entries are the prefill-tiled class (>=8), false entries the verify class.
constexpr bool kGdnVerifyIsTiled[kGdnVerifyWidthCount] = {false, false, true, true};

struct VerifyScratch {
    Buffer *convOut = nullptr, *core[kGdnVerifyWidthCount] = {};
    ~VerifyScratch() {
        if (convOut) destroyBuffer(convOut);
        for (int i = 0; i < kGdnVerifyWidthCount; ++i) {
            if (core[i]) destroyBuffer(core[i]);
        }
    }
};

// Deterministic wide-magnitude operands. A fixed LCG keeps the check reproducible
// across runs and devices without pulling in <random>.
float verifyOperand(uint32_t& s, float scale) {
    s = s * 1664525u + 1013904223u;
    return (float)((int32_t)(s >> 9) - (int32_t)(1u << 22)) / (float)(1u << 22) * scale;
}

float gdnVerifyDeviation(const float* got, const float* ref, size_t n) {
    float maxDelta = 0.0f, maxRef = 0.0f;
    for (size_t i = 0; i < n; ++i) {
        float delta = fabsf(got[i] - ref[i]);
        float refAbs = fabsf(ref[i]);
        if (delta > maxDelta) maxDelta = delta;
        if (refAbs > maxRef) maxRef = refAbs;
    }
    if (maxRef <= 0.0f) return (maxDelta <= 0.0f) ? 0.0f : 1.0f;
    return maxDelta / maxRef;
}

// Run both kernels on identical operands for one width and return the worst
// (core, state) normwise max relative deviation. Returns a negative value when a
// dispatch or allocation failed. `tiledClass` selects the prefill-tiled candidate;
// otherwise the DVPL=2 verify-width candidate runs.
float gdnVerifyRunWidth(VerifyScratch& scratch, int slot, int tokens, bool tiledClass,
                        int conv_dim, int n_k, int n_v, int k_hd, int v_hd, int kernel, float eps) {
    const size_t convOutBytes = (size_t)tokens * conv_dim * sizeof(float);
    const size_t coreBytes = (size_t)tokens * n_v * v_hd * sizeof(float);
    const size_t stateBytes = (size_t)n_v * k_hd * v_hd * sizeof(float);
    const size_t convStateBytes = (size_t)(kernel - 1) * conv_dim * sizeof(float);
    const size_t stateCount = stateBytes / sizeof(float);

    if (scratch.convOut) { destroyBuffer(scratch.convOut); scratch.convOut = nullptr; }
    scratch.convOut = allocBuffer(convOutBytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    scratch.core[slot] = allocBuffer(coreBytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, STORAGE_USAGE);
    if (!scratch.convOut || !scratch.core[slot]) {
        // Free whatever did allocate so a failed width never leaks into the next.
        if (scratch.convOut) { destroyBuffer(scratch.convOut); scratch.convOut = nullptr; }
        if (scratch.core[slot]) { destroyBuffer(scratch.core[slot]); scratch.core[slot] = nullptr; }
        return -1.0f;
    }

    const size_t mixedCount = (size_t)tokens * conv_dim;
    const size_t zCount = (size_t)tokens * n_v * v_hd;
    const size_t gateCount = (size_t)tokens * n_v;
    const size_t convWCount = (size_t)conv_dim * kernel;
    std::vector<float> mixed(mixedCount), z(zCount), beta(gateCount), alpha(gateCount);
    std::vector<float> convW(convWCount), aLog(n_v), dtBias(n_v), norm(v_hd);
    std::vector<float> convStateWrite((size_t)(kernel - 1) * conv_dim);
    std::vector<float> stateInit(stateCount);
    uint32_t seed = 0x5eed1234u + (uint32_t)(tokens * 2654435761u);
    for (float& v : mixed) v = verifyOperand(seed, 0.35f);
    for (float& v : z) v = verifyOperand(seed, 0.6f);
    for (float& v : beta) v = verifyOperand(seed, 0.5f);
    for (float& v : alpha) v = verifyOperand(seed, 0.4f);
    for (float& v : convW) v = verifyOperand(seed, 0.2f);
    for (float& v : aLog) v = verifyOperand(seed, 0.1f) - 1.0f;
    for (float& v : dtBias) v = verifyOperand(seed, 0.1f);
    for (float& v : norm) v = verifyOperand(seed, 0.1f) + 1.0f;
    for (float& v : convStateWrite) v = verifyOperand(seed, 0.05f);
    for (float& v : stateInit) v = verifyOperand(seed, 0.02f);

    const VkMemoryPropertyFlags hostVis =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer* dMixed = allocBuffer(mixedCount * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dZ = allocBuffer(zCount * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dBeta = allocBuffer(gateCount * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dAlpha = allocBuffer(gateCount * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dConvW = allocBuffer(convWCount * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dALog = allocBuffer((size_t)n_v * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dDtBias = allocBuffer((size_t)n_v * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dNorm = allocBuffer((size_t)v_hd * sizeof(float), hostVis, STORAGE_USAGE);
    Buffer* dConvState = allocBuffer(convStateBytes ? convStateBytes : 4, hostVis, STORAGE_USAGE);
    // dScalarState / dVerifyState are the two final recurrent states; the core pair
    // lives in scratch.core[slot] (stock) and dVerifyCore (register-resident).
    Buffer* dScalarState = allocBuffer(stateBytes, hostVis, STORAGE_USAGE);
    Buffer* dVerifyState = allocBuffer(stateBytes, hostVis, STORAGE_USAGE);
    Buffer* dVerifyCore = allocBuffer(coreBytes, hostVis, STORAGE_USAGE);
    Buffer* dReadout = allocBuffer(coreBytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    bool ok = dMixed && dZ && dBeta && dAlpha && dConvW && dALog && dDtBias && dNorm &&
              dConvState && dScalarState && dVerifyState && dVerifyCore && dReadout;
    if (ok) {
        copyHostToDevice(dMixed, mixed.data(), mixedCount * sizeof(float));
        copyHostToDevice(dZ, z.data(), zCount * sizeof(float));
        copyHostToDevice(dBeta, beta.data(), gateCount * sizeof(float));
        copyHostToDevice(dAlpha, alpha.data(), gateCount * sizeof(float));
        copyHostToDevice(dConvW, convW.data(), convWCount * sizeof(float));
        copyHostToDevice(dALog, aLog.data(), (size_t)n_v * sizeof(float));
        copyHostToDevice(dDtBias, dtBias.data(), (size_t)n_v * sizeof(float));
        copyHostToDevice(dNorm, norm.data(), (size_t)v_hd * sizeof(float));
        copyHostToDevice(dScalarState, stateInit.data(), stateBytes);
        copyHostToDevice(dVerifyState, stateInit.data(), stateBytes);
    }

    struct ConvPC { int tokens, conv_dim, kernel; } cpc{tokens, conv_dim, kernel};
    struct RecPC { int tokens, conv_dim, n_k, n_v, k_hd, v_hd; float eps; } rpc{tokens, conv_dim, n_k, n_v, k_hd, v_hd, eps};

    // Arm A: stock recurrent shader. Its conv state runs in its own buffer.
    Buffer* scalarConvState = allocBuffer(convStateBytes ? convStateBytes : 4, hostVis, STORAGE_USAGE);
    if (ok && scalarConvState) copyHostToDevice(scalarConvState, convStateWrite.data(), convStateBytes);
    ok = ok && scalarConvState;
    if (ok) {
        Buffer* cbufs[4] = {dMixed, dConvW, scalarConvState, scratch.convOut};
        ok = dispatch(g_kern[K_QWEN35_GDN_CONV], cbufs, &cpc, sizeof(cpc), (uint32_t)((conv_dim + 63) / 64));
    }
    if (ok) {
        Buffer* rbufs[9] = {scratch.convOut, dZ, dBeta, dAlpha, dALog, dDtBias, dNorm, dScalarState, scratch.core[slot]};
        ok = dispatch(g_kern[K_QWEN35_GDN_RECURRENT], rbufs, &rpc, sizeof(rpc), (uint32_t)n_v);
    }

    // Arm B: the register-resident candidate for this class + shared norm pass.
    if (ok) {
        copyHostToDevice(dConvState, convStateWrite.data(), convStateBytes);
        Buffer* cbufs[4] = {dMixed, dConvW, dConvState, scratch.convOut};
        ok = dispatch(g_kern[K_QWEN35_GDN_CONV], cbufs, &cpc, sizeof(cpc), (uint32_t)((conv_dim + 63) / 64));
    }
    if (ok) {
        Buffer* rbufs[9] = {scratch.convOut, dZ, dBeta, dAlpha, dALog, dDtBias, dNorm, dVerifyState, dReadout};
        if (tiledClass) {
            ok = dispatch(g_kern[K_QWEN35_GDN_PREFILL_TILED], rbufs, &rpc, sizeof(rpc), 4, (uint32_t)n_v);
        } else {
            ok = dispatch(g_kern[K_QWEN35_GDN_VERIFY_TILED], rbufs, &rpc, sizeof(rpc),
                          (uint32_t)((v_hd + 7) / 8), (uint32_t)n_v);
        }
    }
    if (ok) {
        Buffer* rbufs[9] = {dReadout, dZ, dBeta, dAlpha, dALog, dDtBias, dNorm, dVerifyState, dVerifyCore};
        ok = dispatch(g_kern[K_QWEN35_GDN_PREFILL_NORM], rbufs, &rpc, sizeof(rpc), (uint32_t)n_v, (uint32_t)tokens);
    }

    float deviation = -1.0f;
    if (ok) {
        std::vector<float> stockCore(coreBytes / sizeof(float)), verifyCore(coreBytes / sizeof(float));
        std::vector<float> stockState(stateCount), verifyState(stateCount);
        bool read = copyDeviceToHost(stockCore.data(), scratch.core[slot], coreBytes) == VK_SUCCESS &&
                    copyDeviceToHost(verifyCore.data(), dVerifyCore, coreBytes) == VK_SUCCESS &&
                    copyDeviceToHost(stockState.data(), dScalarState, stateBytes) == VK_SUCCESS &&
                    copyDeviceToHost(verifyState.data(), dVerifyState, stateBytes) == VK_SUCCESS;
        if (read) {
            float coreDev = gdnVerifyDeviation(verifyCore.data(), stockCore.data(), stockCore.size());
            float stateDev = gdnVerifyDeviation(verifyState.data(), stockState.data(), stateCount);
            deviation = coreDev > stateDev ? coreDev : stateDev;
        }
    }

    if (dReadout) destroyBuffer(dReadout);
    if (dVerifyCore) destroyBuffer(dVerifyCore);
    if (dVerifyState) destroyBuffer(dVerifyState);
    if (dScalarState) destroyBuffer(dScalarState);
    if (scalarConvState) destroyBuffer(scalarConvState);
    for (Buffer* b : {dMixed, dZ, dBeta, dAlpha, dConvW, dALog, dDtBias, dNorm, dConvState}) if (b) destroyBuffer(b);
    // Release this width's durable result before the next width allocates, so
    // the check's device peak stays at one width class instead of all four.
    if (scratch.core[slot]) { destroyBuffer(scratch.core[slot]); scratch.core[slot] = nullptr; }
    return deviation;
}

// Run every width class once and cache one verdict per class. Returns the worst
// candidate deviation (>= 0 on a clean run, -1 on failure).
float gdnVerifySelfCheck() {
    const int conv_dim = 10240, n_k = 16, n_v = 48, k_hd = 128, v_hd = 128, kernel = 4;
    g_gdn_verify_checked = 1;
    if (g_gdn_verify_force_disagree) {
        g_gdn_verify_verdict = 0;
        g_gdn_tiled_verdict = 0;
        g_gdn_verify_deviation = 1.0f;
        g_gdn_tiled_deviation = 1.0f;
        return 1.0f;
    }
    if (!g_ready || !g_have_gdn_prefill_tiled) {
        g_gdn_verify_verdict = 0;
        g_gdn_tiled_verdict = 0;
        g_gdn_verify_deviation = -1.0f;
        g_gdn_tiled_deviation = -1.0f;
        return -1.0f;
    }
    VerifyScratch scratch;
    float worst = 0.0f;
    for (int i = 0; i < kGdnVerifyWidthCount; ++i) {
        const bool tiledClass = kGdnVerifyIsTiled[i];
        // The tiled class needs the prefill pipeline; the verify class needs its own.
        if (tiledClass ? !g_have_gdn_prefill_tiled : !g_have_gdn_verify_tiled) {
            if (tiledClass) { g_gdn_tiled_verdict = 0; g_gdn_tiled_deviation = -1.0f; }
            else { g_gdn_verify_verdict = 0; g_gdn_verify_deviation = -1.0f; }
            continue;
        }
        float dev = gdnVerifyRunWidth(scratch, i, kGdnVerifyWidths[i], tiledClass, conv_dim, n_k, n_v, k_hd, v_hd, kernel, 1e-5f);
        int verdict = (dev >= 0.0f && dev <= kGdnVerifyDeviationMax) ? 1 : 0;
        if (tiledClass) {
            g_gdn_tiled_verdict = verdict;
            g_gdn_tiled_deviation = dev;
        } else {
            g_gdn_verify_verdict = verdict;
            g_gdn_verify_deviation = dev;
        }
        if (dev < 0.0f) return -1.0f;
        if (dev > worst) worst = dev;
    }
    return worst;
}

} // namespace

// Pure synthetic seam: shares the actual selector above, never calls Vulkan,
// mutates backend state, or fabricates live allocation/dispatch observations.
extern "C" int fvk_debug_memory_type_selection(uint32_t typeBits, uint32_t want, uint32_t preferred,
        uint32_t typeCount, const uint32_t* flags, const int* results,
        const uint32_t* heapIndices, uint32_t heapCount, const uint64_t* heapSizes,
        uint64_t allocationSize, int tryAll, uint32_t* attempts, uint32_t* attemptCount, uint32_t* selectedType,
        int* publishedHandle) {
    VkPhysicalDeviceMemoryProperties properties{};
    properties.memoryTypeCount = typeCount;
    properties.memoryHeapCount = heapCount;
    const uint32_t count = std::min(typeCount, uint32_t(32));
    for (uint32_t i = 0; i < count; ++i) {
        properties.memoryTypes[i].propertyFlags = flags[i];
        properties.memoryTypes[i].heapIndex = heapIndices[i];
    }
    for (uint32_t i = 0; i < std::min(heapCount, uint32_t(VK_MAX_MEMORY_HEAPS)); ++i) {
        properties.memoryHeaps[i].size = heapSizes[i];
    }
    *attemptCount = 0;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    bool allocationAttempted = false;
    VkResult result = allocateCompatibleMemory(typeBits, want, preferred, properties, allocationSize,
        tryAll != 0, [&](uint32_t i, VkDeviceMemory* out) {
            attempts[(*attemptCount)++] = i;
            VkResult status = static_cast<VkResult>(results[i]);
            // Failed Vulkan output parameters are undefined. Poison even a
            // failed synthetic result to prove it is never published or freed.
            *out = (VkDeviceMemory)(uintptr_t)(i + 1);
            return status;
        }, &memory, selectedType, &allocationAttempted);
    *publishedHandle = memory != VK_NULL_HANDLE ? 1 : 0;
    return static_cast<int>(result);
}



// ---- C ABI ----------------------------------------------------------------------
extern "C" {

void fvk_memory_guard_snapshot(uint64_t* limit, uint64_t* live, uint64_t* peak,
                               uint64_t* refused, int* configured, int* configuration_valid,
                               int* accounting_valid) {
    *limit = g_memoryGuard.limit;
    *live = g_memoryGuard.live;
    *peak = g_memoryGuard.peak;
    *refused = g_memoryGuard.refused;
    *configured = g_memoryGuard.configured;
    *configuration_valid = g_memoryGuard.configurationValid;
    *accounting_valid = g_memoryGuard.accountingValid;
}


int fvk_memory_capacity_snapshot_read(fvk_memory_capacity_snapshot* out) {
    if (!out) return FVK_MEMORY_CAPACITY_INVALID;
    *out = {};
    if (!g_ready || !g_dev || !g_phys || !g_memoryDeviceGenerationReady ||
        !g_memoryDeviceGeneration) return FVK_MEMORY_CAPACITY_UNAVAILABLE;

    static_assert(VK_MAX_MEMORY_HEAPS == FVK_MEMORY_CAPACITY_MAX_HEAPS, "heap ABI");
    static_assert(VK_MAX_MEMORY_TYPES == FVK_MEMORY_CAPACITY_MAX_TYPES, "type ABI");
    VkPhysicalDeviceMemoryProperties memory = g_memprops;
    bool budgetKnown = false;
    uint64_t budgets[FVK_MEMORY_CAPACITY_MAX_HEAPS]{};
    uint64_t usages[FVK_MEMORY_CAPACITY_MAX_HEAPS]{};
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    if (g_haveMemoryBudget) {
        VkPhysicalDeviceMemoryBudgetPropertiesEXT budget{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
        VkPhysicalDeviceMemoryProperties2 properties{
            VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2};
        properties.pNext = &budget;
        vkGetPhysicalDeviceMemoryProperties2(g_phys, &properties);
        memory = properties.memoryProperties;
        budgetKnown = true;
        for (uint32_t i = 0; i < FVK_MEMORY_CAPACITY_MAX_HEAPS; ++i) {
            budgets[i] = budget.heapBudget[i];
            usages[i] = budget.heapUsage[i];
        }
    }
#endif
    if (!memory.memoryHeapCount || memory.memoryHeapCount > FVK_MEMORY_CAPACITY_MAX_HEAPS ||
        !memory.memoryTypeCount || memory.memoryTypeCount > FVK_MEMORY_CAPACITY_MAX_TYPES ||
        memory.memoryHeapCount != g_memprops.memoryHeapCount ||
        memory.memoryTypeCount != g_memprops.memoryTypeCount) return FVK_MEMORY_CAPACITY_INVALID;

    fvk_memory_capacity_snapshot result{};
    for (uint32_t i = 0; i < memory.memoryHeapCount; ++i) {
        const VkMemoryHeap& heap = memory.memoryHeaps[i];
        if (!heap.size || heap.size != g_memprops.memoryHeaps[i].size ||
            heap.flags != g_memprops.memoryHeaps[i].flags) return FVK_MEMORY_CAPACITY_INVALID;
        result.heaps[i] = {heap.size, budgets[i], usages[i], heap.flags, budgetKnown ? 1u : 0u};
    }
    for (uint32_t i = 0; i < memory.memoryTypeCount; ++i) {
        const VkMemoryType& type = memory.memoryTypes[i];
        if (type.heapIndex >= memory.memoryHeapCount ||
            type.heapIndex != g_memprops.memoryTypes[i].heapIndex ||
            type.propertyFlags != g_memprops.memoryTypes[i].propertyFlags)
            return FVK_MEMORY_CAPACITY_INVALID;
        result.types[i] = {type.propertyFlags, type.heapIndex};
    }
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(g_phys, &properties);
    result.device_generation = g_memoryDeviceGeneration;
    result.limit_bytes = g_memoryGuard.limit;
    result.live_bytes = g_memoryGuard.live;
    result.peak_bytes = g_memoryGuard.peak;
    result.refused_allocations = g_memoryGuard.refused;
    result.guard_initialized = g_memoryGuard.initialized ? 1u : 0u;
    result.guard_configured = g_memoryGuard.configured ? 1u : 0u;
    result.guard_configuration_valid = g_memoryGuard.configurationValid ? 1u : 0u;
    result.guard_accounting_valid = g_memoryGuard.accountingValid ? 1u : 0u;
    result.max_buffer_bytes = g_maxBufferBytes;
    result.max_storage_buffer_range = g_maxStorageBufferRange;
    result.max_memory_allocation_size = g_maxMemoryAllocationSize;
    result.max_memory_allocation_count = properties.limits.maxMemoryAllocationCount;
    result.submission_status = static_cast<int32_t>(g_submissionStatus);
    result.quarantined = g_v41SubmissionPendingFailure ? 1u : 0u;
    result.batch_active = g_batching ? 1u : 0u;
    result.inflight_active = g_inflight.active ? 1u : 0u;
    result.restore_active = (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) ? 1u : 0u;
    result.vendor_id = properties.vendorID;
    result.device_id = properties.deviceID;
    result.driver_version = properties.driverVersion;
    result.api_version = properties.apiVersion;
    result.device_type = static_cast<uint32_t>(properties.deviceType);
    result.heap_count = memory.memoryHeapCount;
    result.type_count = memory.memoryTypeCount;
    memcpy(result.device_name, properties.deviceName, sizeof(result.device_name));
    result.device_name[sizeof(result.device_name)-1] = 0;
    *out = result;
    return FVK_MEMORY_CAPACITY_OK;
}


int fvk_buffer_requirements_probe(const fvk_buffer_requirements_request* requests,
    uint32_t count, fvk_buffer_requirements_snapshot* out, int32_t* driver_result) {
    if (out) *out = {};
    if (driver_result) *driver_result = static_cast<int32_t>(VK_SUCCESS);
    if (!out || !driver_result || !requests || !count ||
        count > FVK_BUFFER_REQUIREMENTS_MAX_PROBES) return FVK_BUFFER_REQUIREMENTS_INVALID;
    if (!g_ready || !g_dev || !g_phys || !g_memoryDeviceGenerationReady ||
        !g_memoryDeviceGeneration) return FVK_BUFFER_REQUIREMENTS_UNAVAILABLE;
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS)
        return FVK_BUFFER_REQUIREMENTS_QUARANTINED;
    if (g_batching || g_stepBatch || g_inflight.active || g_restore.stage ||
        g_restore.cmd != VK_NULL_HANDLE || g_allocationWindowActive)
        return FVK_BUFFER_REQUIREMENTS_BUSY;
    if (!g_maxBufferBytes || !g_memprops.memoryTypeCount || g_memprops.memoryTypeCount > 32)
        return FVK_BUFFER_REQUIREMENTS_INVALID;

    // Validate and copy the entire bounded descriptor set before driver calls.
    // Both profiles conservatively use the existing single-resource cap, even
    // though transfer-only buffers need not have a storage-range restriction.
    fvk_buffer_requirements_request inputs[FVK_BUFFER_REQUIREMENTS_MAX_PROBES]{};
    for (uint32_t i = 0; i < count; ++i) {
        const auto request = requests[i];
        if (!request.size_bytes || request.size_bytes > uint64_t(INT64_MAX) ||
            request.size_bytes > g_maxBufferBytes ||
            (request.profile != FVK_BUFFER_REQUIREMENTS_STORAGE &&
             request.profile != FVK_BUFFER_REQUIREMENTS_TRANSFER))
            return FVK_BUFFER_REQUIREMENTS_INVALID;
        inputs[i] = request;
    }
    fvk_buffer_requirements_snapshot result{};
    result.device_generation = g_memoryDeviceGeneration;
    result.memory_type_count = g_memprops.memoryTypeCount;
    result.count = count;
    const uint32_t validTypes = g_memprops.memoryTypeCount == 32 ? UINT32_MAX :
        ((uint32_t(1) << g_memprops.memoryTypeCount) - 1);
    for (uint32_t i = 0; i < count; ++i) {
        VkBufferCreateInfo create{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        create.size = inputs[i].size_bytes;
        create.usage = inputs[i].profile == FVK_BUFFER_REQUIREMENTS_STORAGE ?
            STORAGE_USAGE : (VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
        create.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer buffer = VK_NULL_HANDLE;
        const VkResult status = vkCreateBuffer(g_dev, &create, nullptr, &buffer);
        if (status != VK_SUCCESS) {
            *driver_result = static_cast<int32_t>(status);
            return FVK_BUFFER_REQUIREMENTS_DRIVER_FAILED;
        }
        VkMemoryDedicatedRequirements dedicated{VK_STRUCTURE_TYPE_MEMORY_DEDICATED_REQUIREMENTS};
        VkMemoryRequirements2 requirements{VK_STRUCTURE_TYPE_MEMORY_REQUIREMENTS_2};
        requirements.pNext = &dedicated;
        VkBufferMemoryRequirementsInfo2 info{VK_STRUCTURE_TYPE_BUFFER_MEMORY_REQUIREMENTS_INFO_2};
        info.buffer = buffer;
        vkGetBufferMemoryRequirements2(g_dev, &info, &requirements);
        // Every successfully created object is destroyed before validation,
        // a subsequent probe, or publication. It was never bound/submitted.
        vkDestroyBuffer(g_dev, buffer, nullptr);
        const VkMemoryRequirements& r = requirements.memoryRequirements;
        if (r.size < inputs[i].size_bytes || !r.alignment ||
            (r.alignment & (r.alignment - 1)) || !r.memoryTypeBits ||
            (r.memoryTypeBits & ~validTypes) ||
            dedicated.requiresDedicatedAllocation > VK_TRUE ||
            dedicated.prefersDedicatedAllocation > VK_TRUE)
            return FVK_BUFFER_REQUIREMENTS_INVALID;
        result.rows[i] = {inputs[i].size_bytes, r.size, r.alignment,
            inputs[i].profile, r.memoryTypeBits, dedicated.requiresDedicatedAllocation,
            dedicated.prefersDedicatedAllocation};
    }
    *out = result;
    return FVK_BUFFER_REQUIREMENTS_OK;
}

int fvk_device_identity(char* name, int namelen, uint32_t* vendor_id,
                        uint32_t* device_id, uint32_t* driver_version,
                        uint32_t* api_version) {
    if (g_phys == VK_NULL_HANDLE) return 0;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys, &props);
    if (name && namelen > 0) {
        strncpy(name, props.deviceName, namelen - 1);
        name[namelen - 1] = 0;
    }
    if (vendor_id) *vendor_id = props.vendorID;
    if (device_id) *device_id = props.deviceID;
    if (driver_version) *driver_version = props.driverVersion;
    if (api_version) *api_version = props.apiVersion;
    return 1;
}

int fvk_device_drm_render_node(uint64_t* major, uint64_t* minor) {
    if (major) *major = 0;
    if (minor) *minor = 0;
#if defined(__linux__) && defined(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)
    if (!major || !minor || !g_ready || g_phys == VK_NULL_HANDLE ||
        !deviceExtensionSupported(VK_EXT_PHYSICAL_DEVICE_DRM_EXTENSION_NAME)) {
        return 0;
    }
    VkPhysicalDeviceDrmPropertiesEXT drm{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DRM_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 props{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props.pNext = &drm;
    vkGetPhysicalDeviceProperties2(g_phys, &props);
    if (drm.hasRender != VK_TRUE || drm.renderMajor <= 0 || drm.renderMinor < 0) {
        return 0;
    }
    *major = static_cast<uint64_t>(drm.renderMajor);
    *minor = static_cast<uint64_t>(drm.renderMinor);
    return 1;
#else
    return 0;
#endif
}

// V4.1 token panels are default-on: exactly "0" is each family's off switch.
static bool v41TokenPanelEnvEnabled(const char* name) {
    const char* value = std::getenv(name);
    return !(value && value[0] == '0' && value[1] == '\0');
}

int fvk_init(char* name, int namelen, int* is_discrete, const char* spirv_dir) {
    // A re-init is not a device teardown and cannot establish quiescence. Keep
    // the poisoned context and its loaded capability bound until process exit.
    if (g_v41SubmissionPendingFailure) return 9;
    if (!configureMemoryGuard()) {
        fprintf(stderr, "fak-vulkan: invalid FAK_VULKAN_MEMORY_LIMIT_BYTES (expected 1..INT64_MAX)\n");
        return 10;
    }
    reportMemoryGuard();
    discardIdleBatchCommand(); // release against the old owning device/pool before re-init
    beginMemoryDeviceGeneration();
    g_have_v41_tail_rope_qk = 0;
#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
    g_v41TailRoPEExactModule = false;
#endif
    g_have_v41_tail_rope_exact = 0;
    g_have_v41_shared_attention = 0;
    g_have_v41_shared_attention_fast = 0;
    g_have_v41_shared_attention_split = 0;
    g_v41SplitPartials = nullptr; // owned by the device generation being torn down
    g_v41SharedAttentionClass = 0;
    g_have_v41_router_topk = 0;
    g_have_v41_router_gate_topk = 0;
    g_v41RouterGateVerdict = 0;
    g_have_v41_indexer_score = 0;
    g_have_v41_mhc = 0;
    g_have_v41_index_select = 0;
    g_have_v41_q2k_embed_gather = 0;
    g_have_v41_exact_rmsnorm = 0;
    g_have_v41_bf16_boundary = 0;
    g_have_v41_logit_scale = 0;
    g_have_v41_bf16_checked = 0;
    g_have_v41_routed_swiglu = 0;
    g_have_v41_engram_gate = 0;
    g_have_v41_compressor_pool = 0;
    g_have_v41_compressor_ring = 0;
    g_have_v41_scale_exact = 0;
    g_have_v41_engram_row_decode = 0;
    g_have_v41_routed_id = 0;
    bool v41IndexerPipelineBuilt = false;
    bool v41MHCPipelineBuilt = false;
    g_v41IndexerFloatControls = false;
    g_v41IndexerScoreReason = "Vulkan initialization did not complete";
    g_have_qwen35_gdn_q8_in_proj = 0;
    g_have_q6k_matmul = 0;
    g_have_q5k_matmul = 0;
    g_have_attention_split_kv = 0;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "fak";
    app.apiVersion = VK_API_VERSION_1_2;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    if (vkCreateInstance(&ici, nullptr, &g_instance) != VK_SUCCESS) return 1;

    uint32_t n = 0;
    vkEnumeratePhysicalDevices(g_instance, &n, nullptr);
    if (n == 0) return 2;
    std::vector<VkPhysicalDevice> devs(n);
    vkEnumeratePhysicalDevices(g_instance, &n, devs.data());

    // prefer a discrete GPU; fall back to the first device.
    VkPhysicalDevice chosen = devs[0];
    bool discrete = false;
    for (auto d : devs) {
        VkPhysicalDeviceProperties p{};
        vkGetPhysicalDeviceProperties(d, &p);
        if (p.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { chosen = d; discrete = true; break; }
    }
    g_phys = chosen;
    {
        VkPhysicalDeviceProperties chosenProps{};
        vkGetPhysicalDeviceProperties(g_phys, &chosenProps);
        g_umaDirectWeights = chosenProps.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        if (const char* uma = getenv("FAK_VULKAN_UMA_DIRECT")) {
            if (!strcmp(uma, "0") || !strcmp(uma, "off")) g_umaDirectWeights = false;
            if (!strcmp(uma, "1") || !strcmp(uma, "on")) g_umaDirectWeights = true;
        }
    }
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys, &props);
    VkPhysicalDeviceMaintenance3Properties maint3{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_3_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlPropertiesEXT subgroupProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES_EXT};
    VkPhysicalDeviceSubgroupProperties subgroupBasicProps{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceFloatControlsProperties floatControls{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT_CONTROLS_PROPERTIES};
    // Core 1.2 route only. Never infer float behavior from vendor/device names;
    // absent version or unknown/false properties leave the candidate unavailable.
    if (props.apiVersion >= VK_API_VERSION_1_2) subgroupBasicProps.pNext = &floatControls;
    maint3.pNext = &subgroupProps;
    subgroupProps.pNext = &subgroupBasicProps;
    VkPhysicalDeviceProperties2 props2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    props2.pNext = &maint3;
    vkGetPhysicalDeviceProperties2(g_phys, &props2);
    if (props.apiVersion < VK_API_VERSION_1_2) {
        g_v41IndexerScoreReason = "Vulkan 1.2 float-control properties unavailable";
    } else if (floatControls.shaderDenormPreserveFloat32 != VK_TRUE) {
        g_v41IndexerScoreReason = "binary32 DenormPreserve unsupported or unknown";
    } else if (floatControls.shaderSignedZeroInfNanPreserveFloat32 != VK_TRUE) {
        g_v41IndexerScoreReason = "binary32 SignedZeroInfNanPreserve unsupported or unknown";
    } else if (floatControls.shaderRoundingModeRTEFloat32 != VK_TRUE) {
        g_v41IndexerScoreReason = "binary32 RoundingModeRTE unsupported or unknown";
    } else {
        g_v41IndexerFloatControls = true;
        g_v41IndexerScoreReason = "indexer module has not been proven";
    }
    g_maxStorageBufferRange = props2.properties.limits.maxStorageBufferRange;
    g_maxMemoryAllocationSize = maint3.maxMemoryAllocationSize;
    g_maxBufferBytes = g_maxStorageBufferRange;
    g_maxComputeWorkGroupCountX = props2.properties.limits.maxComputeWorkGroupCount[0];
    g_maxComputeWorkGroupCountY = props2.properties.limits.maxComputeWorkGroupCount[1];
    if (g_maxMemoryAllocationSize > 0 &&
        (g_maxBufferBytes == 0 || g_maxMemoryAllocationSize < g_maxBufferBytes)) {
        g_maxBufferBytes = g_maxMemoryAllocationSize;
    }
    if (name && namelen > 0) { strncpy(name, props.deviceName, namelen - 1); name[namelen - 1] = 0; }
    if (is_discrete) *is_discrete = discrete ? 1 : 0;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);
    g_totalDeviceLocalMemory = 0;
    for (uint32_t i = 0; i < g_memprops.memoryHeapCount; ++i) {
        if (g_memprops.memoryHeaps[i].flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            g_totalDeviceLocalMemory += g_memprops.memoryHeaps[i].size;
        }
    }

    // find a compute-capable queue family.
    uint32_t qn = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, nullptr);
    std::vector<VkQueueFamilyProperties> qfs(qn);
    vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &qn, qfs.data());
    bool found = false;
    for (uint32_t i = 0; i < qn; ++i) {
        if (qfs[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g_qfam = i; found = true; break; }
    }
    if (!found) return 3;

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = g_qfam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;

    // Q8 fast-path features: the q8_matmul shader needs 8-bit SSBO storage + int8 arithmetic.
    // Query them; if both are present, chain them into device creation and flag g_have_q8 so
    // the Go side may upload Q8 weights. If absent, we create the device without them and the
    // backend stays f32-only — Q8 is an optional accelerator, never a correctness dependency.
    VkPhysicalDevice8BitStorageFeatures f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features fi8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT subgroupFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR coopFeatures{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
    subgroupFeatures.pNext = &coopFeatures;
#endif
    f8.pNext = &fi8;
    fi8.pNext = &subgroupFeatures;
    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &f8;
    vkGetPhysicalDeviceFeatures2(g_phys, &feat2);
    g_have_q8 = (f8.storageBuffer8BitAccess && fi8.shaderInt8) ? 1 : 0;
    g_have_shader_f16 = fi8.shaderFloat16 == VK_TRUE;

    std::vector<const char*> enabledDeviceExts;
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    bool haveCoopMatExt = deviceExtensionSupported(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    g_have_coopmat = (haveCoopMatExt && coopFeatures.cooperativeMatrix && g_have_q8) ? 1 : 0;
#else
    g_have_coopmat = 0;
#endif
#ifdef VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
    bool haveSubgroupSizeControlExt = deviceExtensionSupported(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
    bool subgroupSizeControlAllowed =
        haveSubgroupSizeControlExt && subgroupFeatures.subgroupSizeControl &&
        subgroupProps.minSubgroupSize <= 32 && subgroupProps.maxSubgroupSize >= 32 &&
        (subgroupProps.requiredSubgroupSizeStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0;
    g_have_glm_kda_wave32 = subgroupSizeControlAllowed ? 1 : 0;
    if (g_have_coopmat && !subgroupSizeControlAllowed) {
        g_have_coopmat = 0;
    }
#else
    bool subgroupSizeControlAllowed = false;
    g_have_glm_kda_wave32 = 0;
    g_have_coopmat = 0;
#endif
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    if (g_have_coopmat) {
        enabledDeviceExts.push_back(VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME);
    }
#endif

    bool isGfx1151 = props.vendorID == 0x1002u && props.deviceID == 0x1586u;
    bool haveSubgroupBasic =
        (subgroupBasicProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroupBasicProps.supportedOperations & VK_SUBGROUP_FEATURE_BASIC_BIT) != 0;
    bool haveSubgroupArithmetic =
        (subgroupBasicProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroupBasicProps.supportedOperations & VK_SUBGROUP_FEATURE_ARITHMETIC_BIT) != 0;
    bool effectiveSubgroup32 = (subgroupBasicProps.subgroupSize == 32);
    bool requiredSubgroup32 = subgroupSizeControlAllowed;

    g_have_q4k_wave32 =
        (isGfx1151 && haveSubgroupBasic && haveSubgroupArithmetic &&
         (effectiveSubgroup32 || requiredSubgroup32)) ? 1 : 0;
    g_q4k_wave32_required_subgroup = (g_have_q4k_wave32 && requiredSubgroup32);

    bool haveSubgroupShuffle =
        (subgroupBasicProps.supportedStages & VK_SHADER_STAGE_COMPUTE_BIT) != 0 &&
        (subgroupBasicProps.supportedOperations & VK_SUBGROUP_FEATURE_SHUFFLE_BIT) != 0;
    const auto& limits = props2.properties.limits;
    g_v41SharedAttentionFastSubgroups =
        g_have_glm_kda_wave32 && haveSubgroupBasic && haveSubgroupArithmetic && haveSubgroupShuffle &&
        (subgroupBasicProps.supportedOperations & VK_SUBGROUP_FEATURE_BALLOT_BIT) != 0 &&
        limits.maxComputeSharedMemorySize >= 8192 * 4 + 64;
    g_v41RouterGateSubgroups = haveSubgroupBasic && haveSubgroupArithmetic;
    g_gdn_prefill_max_groups_y = limits.maxComputeWorkGroupCount[1];
    g_have_gdn_prefill_tiled =
        (haveSubgroupBasic && haveSubgroupShuffle && (effectiveSubgroup32 || requiredSubgroup32) &&
         limits.maxComputeWorkGroupInvocations >= 256 && limits.maxComputeWorkGroupSize[0] >= 128 &&
         limits.maxComputeWorkGroupSize[1] >= 8 && limits.maxComputeSharedMemorySize >= 20864 &&
         limits.maxComputeWorkGroupCount[0] >= 48 && limits.maxComputeWorkGroupCount[1] >= 48) ? 1 : 0;
    g_gdn_prefill_required_subgroup = g_have_gdn_prefill_tiled && requiredSubgroup32;
    // Both width classes of the register-resident candidate (verify-width and
    // prefill-tiled) share the subgroup and shared-memory prerequisites. The
    // per-geometry on-device self-check decides whether each is selected.
    // Verify-width (1..7 token) register-resident variant. It shares the tiled
    // path's subgroup and shared-memory prerequisites but needs only one 32-wide
    // workgroup per eight value dims (no 16-token LDS panel), so it reuses the
    // same admission flag. The per-geometry device self-check decides whether it
    // is actually selected at run time.

    bool needSubgroupControl = (g_have_glm_kda_wave32 != 0) || g_q4k_wave32_required_subgroup || g_gdn_prefill_required_subgroup || (g_have_coopmat != 0);
    if (needSubgroupControl) {
#ifdef VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
        enabledDeviceExts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
#endif
    }
#ifdef VK_EXT_MEMORY_BUDGET_EXTENSION_NAME
    g_haveMemoryBudget = deviceExtensionSupported(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (g_haveMemoryBudget) {
        enabledDeviceExts.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    }
#else
    g_haveMemoryBudget = false;
#endif

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.enabledExtensionCount = (uint32_t)enabledDeviceExts.size();
    dci.ppEnabledExtensionNames = enabledDeviceExts.empty() ? nullptr : enabledDeviceExts.data();
    VkPhysicalDevice8BitStorageFeatures e8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features ei8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    VkPhysicalDeviceSubgroupSizeControlFeaturesEXT esubgroup{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_FEATURES_EXT};
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    VkPhysicalDeviceCooperativeMatrixFeaturesKHR ecoop{
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_COOPERATIVE_MATRIX_FEATURES_KHR};
#endif
    if (g_have_q8) {
        e8.storageBuffer8BitAccess = VK_TRUE;
        ei8.shaderInt8 = VK_TRUE;
        ei8.shaderFloat16 = (g_have_coopmat && g_have_shader_f16) ? VK_TRUE : VK_FALSE;
        e8.pNext = &ei8;
        dci.pNext = &e8;
    }
    if (needSubgroupControl) {
        esubgroup.subgroupSizeControl = VK_TRUE;
        if (g_have_q8) {
            ei8.pNext = &esubgroup;
        } else {
            dci.pNext = &esubgroup;
        }
    }
#ifdef VK_KHR_COOPERATIVE_MATRIX_EXTENSION_NAME
    if (g_have_coopmat) {
        ecoop.cooperativeMatrix = VK_TRUE;
        esubgroup.pNext = &ecoop;
    }
#endif
    VkResult dr = vkCreateDevice(g_phys, &dci, nullptr, &g_dev);
    if (dr != VK_SUCCESS && (g_haveMemoryBudget || g_have_coopmat)) {
        enabledDeviceExts.clear();
        g_haveMemoryBudget = false;
        g_have_coopmat = 0;
        needSubgroupControl = (g_have_glm_kda_wave32 != 0) || g_q4k_wave32_required_subgroup || g_gdn_prefill_required_subgroup;
        if (needSubgroupControl) {
#ifdef VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME
            enabledDeviceExts.push_back(VK_EXT_SUBGROUP_SIZE_CONTROL_EXTENSION_NAME);
#endif
            esubgroup.pNext = nullptr;
        } else if (g_have_q8) {
            ei8.pNext = nullptr;
        } else {
            dci.pNext = nullptr;
        }
        dci.enabledExtensionCount = (uint32_t)enabledDeviceExts.size();
        dci.ppEnabledExtensionNames = enabledDeviceExts.empty() ? nullptr : enabledDeviceExts.data();
        dr = vkCreateDevice(g_phys, &dci, nullptr, &g_dev);
    }
    if (dr != VK_SUCCESS) return 4;
    vkGetDeviceQueue(g_dev, g_qfam, 0, &g_queue);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g_qfam;
    if (vkCreateCommandPool(g_dev, &cpi, nullptr, &g_cmdpool) != VK_SUCCESS) return 5;

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    if (vkCreateFence(g_dev, &fci, nullptr, &g_submitFence) != VK_SUCCESS) return 6;

    // a generous descriptor pool (freed per-dispatch; FREE_DESCRIPTOR_SET_BIT lets us).
    // Sized for a full BATCHED token: ~30 layers × ~9 dispatches ≈ 270 descriptor sets live
    // at once (freed only after the per-token batch submits), each binding up to 5 buffers.
    // 8192 sets / 32768 descriptors is generous headroom over one token.
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32768};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    dpi.maxSets = 8192;
    dpi.poolSizeCount = 1;
    dpi.pPoolSizes = &psz;
    if (vkCreateDescriptorPool(g_dev, &dpi, nullptr, &g_descpool) != VK_SUCCESS) return 7;

    std::string dir(spirv_dir ? spirv_dir : ".");
    auto P = [&](const char* f) { return dir + "/" + f; };
    // (kernel, spirv, nbuf, push-constant bytes)
    bool ok = true;
    ok &= buildKernel(g_kern[K_MATMUL],    P("matmul.spv"),    3, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_MATMUL_ADD], P("matmul_add.spv"), 3, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_MATMUL_ARGMAX], P("matmul_argmax.spv"), 3, 2 * sizeof(int));
    ok &= buildKernel(g_kern[K_MATMUL_ARGMAX_BLOCKS], P("matmul_argmax_blocks.spv"), 4, 2 * sizeof(int));
    ok &= buildKernel(g_kern[K_MATMUL2],   P("matmul2.spv"),   5, 4 * sizeof(int));
    ok &= buildKernel(g_kern[K_MATMUL3],   P("matmul3.spv"),   7, 5 * sizeof(int));
    ok &= buildKernel(g_kern[K_RMSNORM],   P("rmsnorm.spv"),   3, 2 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_RMSNORM_MATMUL], P("rmsnorm_matmul.spv"), 4, 3 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_RMSNORM_MATMUL2], P("rmsnorm_matmul2.spv"), 6, 4 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_RMSNORM_MATMUL3], P("rmsnorm_matmul3.spv"), 8, 5 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_RMSNORM_MATMUL_ARGMAX_BLOCKS], P("rmsnorm_matmul_argmax_blocks.spv"), 5, 2 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_ROPE],      P("rope.spv"),      1, 3 * sizeof(int) + sizeof(float));
    // A missing or incompatible V4.1 module is a broken current bundle, not
    // an optional capability miss. Receipt verification is a pre-launch gate.
    ok &= buildKernel(g_kern[K_V41_TAIL_ROPE_QK], P("v41_tail_rope_qk.spv"), 5, 3 * sizeof(int));
    // Archive and complete61 shader bundle cut over and roll back together.
    // Never silently substitute legacy59/60 when this required module fails.
    ok &= buildKernel(g_kern[K_V41_SHARED_ATTENTION], P("v41_shared_attention.spv"), 5, 24);
    const bool v41SharedAttentionFastBuilt = g_v41SharedAttentionFastSubgroups &&
        buildKernel(g_kern[K_V41_SHARED_ATTENTION_FAST], P("v41_shared_attention_fast.spv"), 5, 24, 32);
    // Optional split-k class: the same Wave32 pin plus a partials binding and a pass word.
    FILE* v41SplitProbe = fopen(P("v41_shared_attention_split.spv").c_str(), "rb");
    if (v41SplitProbe) fclose(v41SplitProbe);
    const bool v41SharedAttentionSplitBuilt = v41SharedAttentionFastBuilt && v41SplitProbe != nullptr &&
        buildKernel(g_kern[K_V41_SHARED_ATTENTION_SPLIT], P("v41_shared_attention_split.spv"), 6, 28, 32);
    // Trusted since registry V7/complete66, but still optional at init: an
    // absent module leaves the capability false and device-only V4.1 refuses.
    const std::string v41RouterModule = P("v41_router_topk.spv");
    FILE* v41RouterProbe = fopen(v41RouterModule.c_str(), "rb");
    const bool v41RouterModulePresent = v41RouterProbe != nullptr;
    if (v41RouterProbe) fclose(v41RouterProbe);
    const bool v41RouterPipelineBuilt = v41RouterModulePresent && buildKernel(g_kern[K_V41_ROUTER_TOPK], v41RouterModule, 5, 24);
    const std::string v41RouterGateModule = P("v41_router_gate_topk.spv");
    FILE* v41RouterGateProbe = fopen(v41RouterGateModule.c_str(), "rb");
    const bool v41RouterGateModulePresent = v41RouterGateProbe != nullptr;
    if (v41RouterGateProbe) fclose(v41RouterGateProbe);
    const bool v41RouterGatePipelineBuilt = g_v41RouterGateSubgroups && v41RouterGateModulePresent &&
        buildKernel(g_kern[K_V41_ROUTER_GATE_TOPK], v41RouterGateModule, 7, 24);
    // This registration is coupled to the separately versioned V5/complete62
    // receipt migration. The frozen V4/complete61 identity is not extended.
    if (g_v41IndexerFloatControls) {
        g_v41IndexerScoreReason = "indexer module missing or pipeline creation failed";
        v41IndexerPipelineBuilt = buildKernel(g_kern[K_V41_INDEXER_SCORE], P("v41_indexer_score.spv"), 4, 12);
        if (v41IndexerPipelineBuilt) g_v41IndexerScoreReason.clear();
    }
    // Optional: an absent or rejected mHC module leaves the capability false so
    // the model refuses its device-only route; it never fails initialization.
    v41MHCPipelineBuilt = buildKernel(g_kern[K_V41_MHC], P("v41_mhc.spv"), 4, 6 * sizeof(int));
    // V4.1 index selection, trusted since registry V7/complete66 and optional
    // at init: an absent module leaves the capability false, never fails init.
    const bool v41IndexSelectBuilt = buildKernel(g_kern[K_V41_INDEX_SELECT], P("v41_index_select.spv"), 4, 16);
    // V4.1 Q2_K embedding row gather, trusted since registry V7/complete66 and
    // optional at init like the router and index-select modules.
    const bool v41Q2KEmbedGatherBuilt = buildKernel(g_kern[K_V41_Q2K_EMBED_GATHER], P("v41_q2k_embed_gather.spv"), 3, 12);
    // V4.1 bit-exact RMSNorm for the compressor and index-key norms, trusted
    // since registry V8/complete67. Optional at init, and only built when the
    // binary32 float controls its execution modes declare are supported.
    const bool v41ExactRMSNormBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_EXACT_RMSNORM], P("v41_exact_rmsnorm.spv"), 3, 16);
    // V4.1 routed-expert activation (BF16 gate/up, clamp, SwiGLU, route weight,
    // BF16 down operand), trusted since registry V10/complete69. Optional at init
    // and built only when its declared binary32 float controls are supported.
    const bool v41RoutedSwiGLUBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_ROUTED_SWIGLU], P("v41_routed_swiglu.spv"), 4, 12);
    // Independent experimental opt-in; absent or rejected row-address modules
    // never disable the existing scalar activation or affect initialization.
    const char* routedRowEnv = std::getenv("FAK_V41_ROUTED_PANEL_ROW");
    if (routedRowEnv && routedRowEnv[0] == '1' && routedRowEnv[1] == '\0' && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW], P("v41_routed_swiglu_panel_row.spv"), 4, 16);
    }
    // Integer-only publication needs no floating-point execution modes.
    if (routedRowEnv && routedRowEnv[0] == '1' && routedRowEnv[1] == '\0') {
        buildKernel(g_kern[K_V41_ROUTED_DOWN_PANEL_ROW], P("v41_routed_down_panel_row.spv"), 3, 12);
    }
    const char* routedChunkEnv = std::getenv("FAK_V41_ROUTED_PANEL_CHUNK");
    if (routedChunkEnv && routedChunkEnv[0] == '1' && routedChunkEnv[1] == '\0' && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK], P("v41_routed_swiglu_panel_chunk.spv"), 4, 48);
    }
    const char* downChunkEnv = std::getenv("FAK_V41_ROUTED_DOWN_CHUNK");
    if (downChunkEnv && downChunkEnv[0] == '1' && downChunkEnv[1] == '\0') {
        buildKernel(g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK], P("v41_routed_down_panel_chunk.spv"), 3, 16);
    }
    const char* indexScaleBF16Env = std::getenv("FAK_V41_INDEX_SCALE_BF16");
    if (indexScaleBF16Env && indexScaleBF16Env[0] == '1' && indexScaleBF16Env[1] == '\0' && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_INDEX_SCALE_BF16], P("v41_index_scale_bf16.spv"), 3, 8);
    }
    // V4.1 Engram gate/add and compressor gated pool, trusted since registry
    // V11/complete71. Optional at init and only built with the binary32 float
    // controls their execution modes declare.
    const bool v41EngramGateBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_ENGRAM_GATE], P("v41_engram_gate.spv"), 5, 28);
    const bool v41CompressorPoolBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_COMPRESSOR_POOL], P("v41_compressor_pool.spv"), 3, 8);
    const bool v41CompressorRingBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_COMPRESSOR_RING], P("v41_compressor_ring.spv"), 5, 16);
    // V4.1 strict binary32 tail RoPE (compressed publication), trusted since
    // registry V12/complete72. Optional at init: built only with the F32 controls
    // its execution modes declare. The witness archive keeps its own strict
    // mandatory module, so it does not also load this twin.
    bool v41TailRoPEExactBuilt = false;
#ifndef FAK_V41_TAIL_ROPE_EXACT_WITNESS
    v41TailRoPEExactBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_TAIL_ROPE_EXACT], P("v41_tail_rope_exact.spv"), 5, 3 * sizeof(int));
#endif
    // V4.1 strict binary32 scalar multiply (indexer weight scale), trusted since
    // registry V13/complete73. Optional at init: built only with the F32 controls
    // its execution modes declare.
    const bool v41ScaleExactBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_SCALE_EXACT], P("v41_scale_exact.spv"), 2, 8);
    // V4.1 Engram table-row decode (packed E4M3/E8M0, Q2_K, F32 rows), trusted
    // since registry V14/complete74. Optional at init: built only with the F32
    // controls its execution modes declare.
    const bool v41EngramRowDecodeBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_ENGRAM_ROW_DECODE], P("v41_engram_row_decode.spv"), 3, 16);
    // V4.1 routed-expert id modules (llama.cpp mul_mat_vec_id port), trusted
    // since registry V15/complete78. Optional at init: all four or none, and the
    // checked pair only with the F32 controls their execution modes declare.
    const bool v41RoutedIDBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_Q2K_MATVEC_ID], P("v41_q2k_matvec_id.spv"), 11, 16) &&
        buildKernel(g_kern[K_V41_Q3K_MATVEC_ID], P("v41_q3k_matvec_id.spv"), 11, 16) &&
        buildKernel(g_kern[K_V41_ROUTED_SWIGLU_ID], P("v41_routed_swiglu_id.spv"), 5, 12) &&
        buildKernel(g_kern[K_V41_ROUTED_PUBLISH_SUM], P("v41_routed_publish_sum.spv"), 4, 12);
    // The optional fused pair must not disable the four-kernel ID fallback.
    // Its capability separately requires both pipelines to be present.
    if (v41RoutedIDBuilt &&
        buildKernel(g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID], P("v41_routed_gate_up_swiglu_id.spv"), 20, 20)) {
        buildKernel(g_kern[K_V41_ROUTED_DOWN_SUM], P("v41_routed_down_sum.spv"), 12, 16);
    }
    // Token tiles are default-on and built before first use; each family's
    // exact "0" skips only its own pipeline. Missing optional modules leave
    // the existing paths alone.
    const bool q2TokenEnabled = v41TokenPanelEnvEnabled("FAK_V41_Q2K_TOKEN_PANEL");
    const bool q3TokenEnabled = v41TokenPanelEnvEnabled("FAK_V41_Q3K_TOKEN_PANEL");
    if (q2TokenEnabled && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_Q2K_TOKEN_PANEL], P("v41_q2k_token_panel.spv"), 3, 12);
    }
    if (q3TokenEnabled && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_Q3K_TOKEN_PANEL], P("v41_q3k_token_panel.spv"), 3, 12);
    }
    const char* q2PairEnv = std::getenv("FAK_V41_Q2K_TOKEN_PAIR");
    if (q2PairEnv && q2PairEnv[0] == '1' && q2PairEnv[1] == '\0' && g_v41IndexerFloatControls) {
        buildKernel(g_kern[K_V41_Q2K_TOKEN_PAIR], P("v41_q2k_token_pair.spv"), 5, 12);
    }
    // Registry V9/current68 promotes only the unchanged finite checked module.
    // A current archive and its complete bundle cut over together. Keep the
    // capability false unless mandatory initialization completes below.
    const bool v41BF16CheckedBuilt = buildKernel(g_kern[K_V41_BF16_CHECKED], P("v41_bf16_checked.spv"), 3, 8);
    ok &= v41BF16CheckedBuilt;
    // Optional raw-finite status. Older bundles remain usable; no model selects it here.
    buildKernel(g_kern[K_V41_F32_FINITE], P("v41_f32_finite.spv"), 2, 8);
    // The raw all-bit operation remains an isolated witness-only extra.
    bool v41BF16BoundaryBuilt = false;
#ifdef FAK_V41_BF16_BOUNDARY_WITNESS
    v41BF16BoundaryBuilt = buildKernel(g_kern[K_V41_BF16_BOUNDARY], P("v41_bf16_boundary.spv"), 2, 4);
#endif
    bool v41LogitScaleBuilt = false;
#ifdef FAK_V41_LOGIT_SCALE_WITNESS
    v41LogitScaleBuilt = g_v41IndexerFloatControls &&
        buildKernel(g_kern[K_V41_LOGIT_SCALE], P("v41_logit_scale.spv"), 2, 8);
#endif


    ok &= buildKernel(g_kern[K_SWIGLU],    P("swiglu.spv"),    3, sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_SWIGLU_MATMUL_ADD], P("swiglu_matmul_add.spv"), 4, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_ADD],       P("add.spv"),       2, sizeof(int));
    ok &= buildKernel(g_kern[K_ADD_BIAS],  P("add_bias.spv"),  2, 2 * sizeof(int));
    ok &= buildKernel(g_kern[K_ATTENTION], P("attention.spv"), 5, sizeof(AttentionPush));
    ok &= buildKernel(g_kern[K_ARGMAX],    P("argmax.spv"),    2, sizeof(int));
    ok &= buildKernel(g_kern[K_ARGMAX_PAIRS], P("argmax_pairs.spv"), 3, sizeof(int));
    ok &= buildKernel(g_kern[K_QWEN35_GDN_CONV], P("qwen35_gdn_conv.spv"), 4, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_QWEN35_GDN_RECURRENT], P("qwen35_gdn_recurrent.spv"), 9, 6 * sizeof(int) + sizeof(float));
    if (g_have_gdn_prefill_tiled) {
        const uint32_t subgroup = g_gdn_prefill_required_subgroup ? 32 : 0;
        const bool tiled = buildKernel(g_kern[K_QWEN35_GDN_PREFILL_TILED], P("qwen35_gdn_prefill_tiled.spv"), 9, 6 * sizeof(int) + sizeof(float), subgroup);
        const bool norm = buildKernel(g_kern[K_QWEN35_GDN_PREFILL_NORM], P("qwen35_gdn_prefill_norm.spv"), 9, 6 * sizeof(int) + sizeof(float));
        g_have_gdn_prefill_tiled = tiled && norm;
        if (g_have_gdn_prefill_tiled) {
            // The verify-width variant is independent of the tiled pipeline: a regex or
            // driver that rejects one module must not disable the other's kernel.
            g_have_gdn_verify_tiled =
                buildKernel(g_kern[K_QWEN35_GDN_VERIFY_TILED], P("qwen35_gdn_verify_tiled.spv"), 9, 6 * sizeof(int) + sizeof(float), subgroup) ? 1 : 0;
        } else {
            g_have_gdn_verify_tiled = 0;
        }
    }
    if (g_have_glm_kda_wave32) {
        ok &= buildKernel(g_kern[K_GLM_KDA_REREAD], P("glm_kda_recurrent_reread.spv"), 7, sizeof(int), 32);
        ok &= buildKernel(g_kern[K_GLM_KDA_WAVE32], P("glm_kda_recurrent_wave32.spv"), 7, sizeof(int), 32);
    }
    ok &= buildKernel(g_kern[K_QWEN35_SPLIT_QG_PANEL], P("qwen35_split_qg_panel.spv"), 3, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_QWEN35_PARTIAL_ROPE_PANEL], P("qwen35_partial_rope_panel.spv"), 5, 7 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_QWEN35_CAUSAL_ATTENTION_PANEL], P("qwen35_causal_attention_panel.spv"), 4, 5 * sizeof(int) + sizeof(float));
    g_have_attention_split_kv = buildKernel(g_kern[K_ATTENTION_SPLIT_KV],
        P("attention_split_kv.spv"), 5, 7 * sizeof(int) + sizeof(float)) ? 1 : 0;
    ok &= buildKernel(g_kern[K_SIGMOID_MUL], P("sigmoid_mul.spv"), 2, sizeof(int));
    ok &= buildKernel(g_kern[K_Q4K_MATMUL], P("q4k_matmul.spv"), 3, 3 * sizeof(int));
    if (g_have_q4k_wave32) {
        uint32_t reqSize = g_q4k_wave32_required_subgroup ? 32 : 0;
        if (!buildKernel(g_kern[K_Q4K_MATMUL_WAVE32], P("q4k_matmul_wave32.spv"), 3, 3 * sizeof(int), reqSize)) {
            g_have_q4k_wave32 = 0;
        }
    }
    // Candidate Q4_K cooperative-matrix prefill arm: only load when the native
    // cooperative-matrix capability and Wave32 subgroup control are present, so an
    // unsupported device keeps the scalar path instead of a broken pipeline.
    if (g_have_coopmat) {
        uint32_t reqSize = g_q4k_wave32_required_subgroup ? 32 : 0;
        if (!buildKernel(g_kern[K_Q4K_MATMUL_COOPMAT], P("q4k_matmul_coopmat.spv"), 3, 3 * sizeof(int), reqSize)) {
            g_have_q4k_coopmat = 0;
        } else {
            g_have_q4k_coopmat = 1;
        }
    }
    g_have_q6k_matmul = buildKernel(g_kern[K_Q6K_MATMUL], P("q6k_matmul.spv"),
                                    3, 3 * sizeof(int)) ? 1 : 0;
    g_have_q5k_matmul = buildKernel(g_kern[K_Q5K_MATMUL], P("q5k_matmul.spv"),
                                    3, 3 * sizeof(int)) ? 1 : 0;
    g_have_q3k_matmul = buildKernel(g_kern[K_Q3K_MATMUL], P("q3k_matmul.spv"),
                                    3, 3 * sizeof(int)) ? 1 : 0;
    g_have_qwen35_value_head_tile = buildKernel(g_kern[K_QWEN35_VALUE_HEAD_TILE],
                                    P("qwen35_value_head_tile.spv"), 2, 4 * sizeof(int)) ? 1 : 0;
    buildKernel(g_kern[K_RMSNORM_Q4K_MATMUL2], P("rmsnorm_q4k_matmul2.spv"), 6, 4 * sizeof(int) + sizeof(float));
    buildKernel(g_kern[K_SWIGLU_Q4K_MATMUL_ADD], P("swiglu_q4k_matmul_add.spv"), 4, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_Q2K_MATMUL], P("q2k_matmul.spv"), 7, 4 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_RMSNORM_Q2K_MATMUL2], P("q2k_matmul.spv"), 7, 4 * sizeof(int) + sizeof(float));
    // Optional single-token Q2_K matvec (same interface as q2k_matmul.spv). Absent SPIR-V or
    // FAK_VULKAN_Q2K_MATVEC=0 keeps the original one-thread-per-row decode kernel.
    {
        const char* q2kmv = std::getenv("FAK_VULKAN_Q2K_MATVEC");
        bool q2kmvOff = q2kmv && q2kmv[0] == '0' && q2kmv[1] == '\0';
        g_have_q2k_matvec = (!q2kmvOff && buildKernel(g_kern[K_Q2K_MATVEC], P("q2k_matvec.spv"), 7, 4 * sizeof(int) + sizeof(float))) ? 1 : 0;
    }
    for (int f = 0; f < FVK_IQ_FORMATS; ++f) {
        const char* sw = std::getenv(kIQEnv[f]);
        bool off = sw && sw[0] == '0' && sw[1] == '\0';
        g_have_iq[f] = (!off && haveSubgroupArithmetic &&
                        buildKernel(g_kern[kIQKernel[f]], P(kIQSpv[f]), 7, 4 * sizeof(int) + sizeof(float))) ? 1 : 0;
    }
    g_have_coopmat_f16_16 = g_have_coopmat && coopMatF16Shape16Supported();
    for (int f = 0; f < FVK_MMCM1_FORMATS; ++f) {
        if (!g_have_coopmat) g_mmcm1Status[f] = FVK_MM_CM1_NO_COOPMAT;
        else if (!g_have_shader_f16) g_mmcm1Status[f] = FVK_MM_CM1_NO_SHADER_F16;
        else if (!g_have_coopmat_f16_16) g_mmcm1Status[f] = FVK_MM_CM1_NO_F16_SHAPE;
        else if (!buildKernel(g_kern[kMMCM1Kernel[f]], P(kMMCM1Spv[f]), 3, 3 * sizeof(int), 32)) g_mmcm1Status[f] = FVK_MM_CM1_NO_MODULE;
        else g_mmcm1Status[f] = FVK_MM_CM1_OK;
    }
    if (!ok) return 8;
    // Q8 kernel is built only when the device advertised the int8/8-bit-storage features; its
    // SPIR-V uses them, so loading it without the enabled device feature would be invalid. If
    // it fails to build, disable the Q8 path rather than failing init (f32 stays available).
    if (g_have_q8) {
        if (g_have_coopmat) {
            if (!buildKernel(g_kern[K_Q8_MATMUL], P("q8_matmul.spv"), 4, 3 * sizeof(int), 32)) {
                g_have_coopmat = 0;
            }
        }
        if (!buildKernel(g_kern[K_Q8_MATMUL_DECODE], P("q8_matmul_decode.spv"), 4, 3 * sizeof(int)) ||
            !buildKernel(g_kern[K_Q8_MATMUL2], P("q8_matmul2.spv"), 7, 4 * sizeof(int)) ||
            !buildKernel(g_kern[K_Q8_MATMUL3], P("q8_matmul3.spv"), 10, 5 * sizeof(int)) ||
            !buildKernel(g_kern[K_RMSNORM_Q8_MATMUL2], P("rmsnorm_q8_matmul2.spv"), 8, 4 * sizeof(int) + sizeof(float)) ||
            !buildKernel(g_kern[K_RMSNORM_Q8_MATMUL3], P("rmsnorm_q8_matmul3.spv"), 11, 5 * sizeof(int) + sizeof(float)) ||
            !buildKernel(g_kern[K_SWIGLU_Q8_MATMUL_ADD], P("swiglu_q8_matmul_add.spv"), 5, 5 * sizeof(int))) {
            g_have_q8 = 0;
        }
    }
    if (g_have_q8) {
        // Cooperative (8 outputs x 32 lanes) gate/up kernel; bit-identical to the scalar one.
        // Optional: absent SPIR-V or FAK_VULKAN_Q8_GATEUP_COOP=0 keeps rmsnorm_q8_matmul2.spv.
        const char* coop = std::getenv("FAK_VULKAN_Q8_GATEUP_COOP");
        bool coopOff = coop && coop[0] == '0' && coop[1] == '\0';
        g_have_rmsnorm_q8_matmul2_coop = (!coopOff && buildKernel(g_kern[K_RMSNORM_Q8_MATMUL2_COOP],
            P("rmsnorm_q8_matmul2_coop.spv"), 8, 4 * sizeof(int) + sizeof(float))) ? 1 : 0;
        g_have_qwen35_gdn_q8_in_proj = buildKernel(
            g_kern[K_QWEN35_GDN_Q8_IN_PROJ], P("qwen35_gdn_q8_in_proj.spv"),
            13, 5 * sizeof(int)) ? 1 : 0;
    }

    g_ready = true;
    g_memoryDeviceGenerationReady = g_memoryDeviceGeneration != 0;
    g_have_v41_tail_rope_qk = 1;
    g_have_v41_shared_attention = 1;
    g_have_v41_shared_attention_fast = v41SharedAttentionFastBuilt ? 1 : 0;
    g_have_v41_shared_attention_split = v41SharedAttentionSplitBuilt ? 1 : 0;
    if (const char* cls = std::getenv("FAK_V41_SHARED_ATTENTION_DETERMINISM")) {
        if (std::strcmp(cls, "run-to-run") == 0 && g_have_v41_shared_attention_fast) g_v41SharedAttentionClass = 1;
        else if (std::strcmp(cls, "split-k") == 0 && g_have_v41_shared_attention_split) g_v41SharedAttentionClass = 2;
        else g_v41SharedAttentionClass = 0;
    }
    g_have_v41_router_topk = v41RouterPipelineBuilt ? 1 : 0;
    g_have_v41_router_gate_topk = v41RouterGatePipelineBuilt ? 1 : 0;
    g_have_v41_indexer_score = v41IndexerPipelineBuilt ? 1 : 0;
    g_have_v41_mhc = v41MHCPipelineBuilt ? 1 : 0;
    g_have_v41_index_select = v41IndexSelectBuilt ? 1 : 0;
    g_have_v41_q2k_embed_gather = v41Q2KEmbedGatherBuilt ? 1 : 0;
    g_have_v41_exact_rmsnorm = v41ExactRMSNormBuilt ? 1 : 0;
    g_have_v41_bf16_boundary = v41BF16BoundaryBuilt ? 1 : 0;
    g_have_v41_logit_scale = v41LogitScaleBuilt ? 1 : 0;
    g_have_v41_bf16_checked = v41BF16CheckedBuilt ? 1 : 0;
    g_have_v41_routed_swiglu = v41RoutedSwiGLUBuilt ? 1 : 0;
    g_have_v41_engram_gate = v41EngramGateBuilt ? 1 : 0;
    g_have_v41_compressor_pool = v41CompressorPoolBuilt ? 1 : 0;
    g_have_v41_compressor_ring = v41CompressorRingBuilt ? 1 : 0;
    g_have_v41_tail_rope_exact = v41TailRoPEExactBuilt ? 1 : 0;
    g_have_v41_scale_exact = v41ScaleExactBuilt ? 1 : 0;
    g_have_v41_engram_row_decode = v41EngramRowDecodeBuilt ? 1 : 0;
    g_have_v41_routed_id = v41RoutedIDBuilt ? 1 : 0;
    // One per-geometry device self-check, before the first forward and never
    // inside one. Its verdict is cached and read by the verify-width route.
    gdnVerifySelfCheck();
    return 0;
}

void* fvk_malloc(size_t bytes) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    if (bytes == 0) bytes = 4;
    auto it = g_pool.find(bytes);
    if (it != g_pool.end() && !it->second.empty()) {
        Buffer* b = it->second.back();
        it->second.pop_back();
        advanceNativeCheckoutGeneration(b);
        if (g_poolCount > 0) --g_poolCount;
        return b;
    }
    return allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
}

// A paged weight (an expert-ring page-in) is evicted and replaced while the model runs, so it
// must not take the bump-only weight arena: its offsets are never reused, and an evicted page
// would leave its bytes reserved until every arena buffer died. It takes the size-bucketed pool
// instead, so an evicted page's buffer serves the next same-size page-in.
uint64_t g_offloadWeightReused = 0;
uint64_t g_offloadWeightFresh = 0;

// A host-visible page is recycled only through fvk_free, which parks mid-batch frees until the
// batch fences, so a page handed out here is never read by a recorded op: its first write needs
// no batch flush.
static Buffer* mallocHostVisibleOffloadPage(size_t bytes) {
    auto it = g_hvOffloadPool.find(bytes);
    if (it != g_hvOffloadPool.end() && !it->second.empty()) {
        Buffer* b = it->second.back();
        it->second.pop_back();
        advanceNativeCheckoutGeneration(b);
        ++g_offloadWeightReused;
        b->offloadPristine = true;
        return b;
    }
    Buffer* b = allocBuffer(bytes,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, STORAGE_USAGE);
    if (!b) return nullptr;
    mapHostVisibleWeight(b);
    if (!b->mapped) {
        destroyBuffer(b);
        return nullptr;
    }
    b->offloadPage = true;
    b->offloadPristine = true;
    ++g_offloadWeightFresh;
    ++g_offloadHostVisiblePages;
    return b;
}

int fvk_set_offload_host_visible(int on) {
    g_offloadHostVisible = on != 0 && g_umaDirectWeights;
    return g_offloadHostVisible ? 1 : 0;
}

int fvk_offload_host_visible(void) { return g_offloadHostVisible ? 1 : 0; }

uint64_t fvk_offload_host_visible_pages(void) { return g_offloadHostVisiblePages; }

void* fvk_malloc_offload_weight(size_t bytes) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    if (bytes == 0) bytes = 4;
    if (g_offloadHostVisible) {
        if (Buffer* b = mallocHostVisibleOffloadPage(bytes)) return b;
    }
    auto it = g_pool.find(bytes);
    if (it != g_pool.end() && !it->second.empty()) {
        Buffer* b = it->second.back();
        it->second.pop_back();
        advanceNativeCheckoutGeneration(b);
        if (g_poolCount > 0) --g_poolCount;
        ++g_offloadWeightReused;
        return b;
    }
    Buffer* b = allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    if (b) ++g_offloadWeightFresh;
    return b;
}

void fvk_offload_weight_stats(uint64_t* reused, uint64_t* fresh) {
    if (reused) *reused = g_offloadWeightReused;
    if (fresh) *fresh = g_offloadWeightFresh;
}

void* fvk_malloc_weight(size_t bytes, uint64_t max_arena_bytes) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    if (bytes == 0) bytes = 4;
    return allocWeightArenaBuffer(bytes, (VkDeviceSize)max_arena_bytes);
}

void fvk_weight_arena_stats(uint64_t* memory_allocations, uint64_t* buffer_bindings,
                            uint64_t* reserved_bytes, uint64_t* live_bytes,
                            uint64_t* peak_reserved_bytes) {
    if (memory_allocations) *memory_allocations = g_weightArenaMemoryAllocations;
    if (buffer_bindings) *buffer_bindings = g_weightArenaBufferBindings;
    if (reserved_bytes) *reserved_bytes = g_weightArenaReservedBytes;
    if (live_bytes) *live_bytes = g_weightArenaLiveBytes;
    if (peak_reserved_bytes) *peak_reserved_bytes = g_weightArenaPeakReservedBytes;
}

void* fvk_malloc_hostvis(size_t bytes) {
    if (g_v41SubmissionPendingFailure) return nullptr;
    if (bytes == 0) bytes = 4;
    // Host-visible directly — the residency-budget path uses this for cold weights so they go
    // host-side by CHOICE, not by losing the device-local race. No pool: weights are immutable
    // and long-lived, so the per-size recycle bucket (sized for hot transient shapes) doesn't
    // apply. allocBuffer with these flags makes no device-local attempt and never spills.
    VkMemoryPropertyFlags hostvis =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    return allocBuffer(bytes, hostvis, STORAGE_USAGE);
}

// fvk_malloc_hostvis_weight is fvk_malloc_hostvis for an immutable weight: with UMA direct upload
// on, a HOST_COHERENT allocation is persistently mapped so fvk_h2d writes straight through it.
// Host-visible buffers are never pooled (fvk_free destroys them), so a mapping cannot leak into an
// unrelated reuse.
void* fvk_malloc_hostvis_weight(size_t bytes) {
    Buffer* b = B(fvk_malloc_hostvis(bytes));
    mapHostVisibleWeight(b);
    return b;
}

// Per-size-bucket recycle cap. The forward pass reissues the SAME handful of sizes every
// token (weights, the fixed activation widths), so a small cap recycles hot shapes while
// bounding the live vkAllocateMemory count — beyond it, free for real (destroy the buffer).
static const size_t POOL_BUCKET_CAP = 64;

void fvk_free(void* d) {
    if (!d) return;
    Buffer* b = B(d);
    // During a batch, a freed buffer may still be referenced by a recorded-but-unsubmitted
    // op (e.g. the KV grow copies old->new then frees old; the copy hasn't executed yet).
    // Park it and recycle only after the batch flushes, so it can't be handed back out and
    // rebound mid-batch.
    if (g_batching || g_v41SubmissionPendingFailure) { g_batchFreed.push_back(b); return; }
    drainInflight();
    if (b->weightArenaBound) {
        destroyBuffer(b);
        clearDescriptorBindingCache();
        return;
    }
    if (b->offloadPage) {
        b->offloadPristine = false;
        auto& bucket = g_hvOffloadPool[b->bytes];
        if (bucket.size() < POOL_BUCKET_CAP) {
            bucket.push_back(b);
        } else {
            destroyBuffer(b);
            clearDescriptorBindingCache();
        }
        return;
    }
    if (b->props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
        destroyBuffer(b);
        clearDescriptorBindingCache();
        return;
    }
    auto& bucket = g_pool[b->bytes];
    if (bucket.size() < POOL_BUCKET_CAP) {
        bucket.push_back(b); // recycle for reuse
        ++g_poolCount;
    } else {
        destroyBuffer(b);    // bucket full — return the allocation to the driver
        clearDescriptorBindingCache();
    }
}

// h2d uploads host data: if a batch is open it must flush first (the new device data must be
// visible to subsequently-recorded ops, and the staging copy itself needs its own submit).
// In practice weights upload BEFORE the per-token batch begins (cached), so this rarely flushes.
void fvk_h2d(void* d, const void* h, size_t bytes) {
    SyncScope syncScope(kSyncSiteUpload);
    if (g_v41SubmissionPendingFailure) return;
    Buffer* page = B(d);
    if (page && page->offloadPristine && page->mapped && h && bytes <= page->bytes) {
        page->offloadPristine = false;
        copyHostToMappedWeight(page, h, bytes);
        return;
    }
    if (stepRecordUpload(B(d), h, bytes)) return;
    bool resumeBatch = g_batching;
    if (resumeBatch) batchFlush();
    copyHostToDevice(B(d), h, bytes);
    if (resumeBatch) batchBegin();
}

// Bounded immutable-residency restore transaction. The Go adapter owns fresh
// destination buffers and publishes them only after every submit succeeds.
// Each submit reuses the same staging allocation only after its fence completes.
int fvk_restore_begin(size_t max_bytes, size_t max_entries) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready || !g_dev || !g_cmdpool || !g_submitFence || g_batching ||
        max_bytes < 4 || (max_bytes & 3) != 0 || max_entries == 0 ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) {
        return 1;
    }
    VkMemoryPropertyFlags hostvis =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    Buffer* stage = allocBuffer(max_bytes, hostvis,
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    if (!stage) return 2;
    void* mapped = nullptr;
    VkResult r = vkMapMemory(g_dev, stage->mem, 0, max_bytes, 0, &mapped);
    if (r != VK_SUCCESS || !mapped) {
        destroyBuffer(stage);
        return r == VK_SUCCESS ? 3 : (int)r;
    }
    g_restore.stage = stage;
    g_restore.mapped = mapped;
    g_restore.cap = max_bytes;
    g_restore.maxEntries = max_entries;
    return 0;
}

int fvk_restore_add(void* dst_handle, size_t dst_offset, const void* src, size_t bytes) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    Buffer* dst = B(dst_handle);
    if (!g_restore.stage || !g_restore.mapped || !dst || !src || bytes == 0 ||
        (dst_offset & 3) != 0 || (bytes & 3) != 0 ||
        dst_offset > dst->bytes || bytes > dst->bytes - dst_offset) {
        return 4;
    }
    if (g_restore.entries >= g_restore.maxEntries) return 5;
    size_t aligned = (g_restore.used + 3) & ~(size_t)3;
    if (aligned > g_restore.cap || bytes > g_restore.cap - aligned) return 6;
    VkResult r = restoreBeginCommand();
    if (r != VK_SUCCESS) return (int)r;
    memcpy((unsigned char*)g_restore.mapped + aligned, src, bytes);
    VkBufferCopy region{aligned, dst_offset, bytes};
    vkCmdCopyBuffer(g_restore.cmd, g_restore.stage->buf, dst->buf, 1, &region);
    g_restore.used = aligned + bytes;
    ++g_restore.entries;
    g_restore.payloadBytes += bytes;
    return 0;
}

int fvk_restore_submit(void) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_restore.stage || g_restore.cmd == VK_NULL_HANDLE || g_restore.entries == 0) return 7;
    if (g_restoreFailAfterSubmits >= 0 &&
        g_restore.successfulSubmits >= g_restoreFailAfterSubmits) {
        restoreDiscardCommand();
        g_submissionStatus = VK_ERROR_DEVICE_LOST;
        return (int)VK_ERROR_DEVICE_LOST;
    }
    VkCommandBuffer cmd = g_restore.cmd;
    // The checked helper frees only commands known not to be pending. Keep all
    // restore ownership and accounting intact if submit/wait completion is uncertain.
    // ts-untimed: counted by dpOneShot after a completed submit.
    const bool completed = v41EndSubmitWaitChecked(cmd, true);
    if (g_v41SubmissionPendingFailure) return (int)g_submissionStatus;
    g_restore.cmd = VK_NULL_HANDLE;
    size_t submittedBytes = g_restore.payloadBytes;
    size_t submittedEntries = g_restore.entries;
    g_restore.used = 0;
    g_restore.entries = 0;
    g_restore.payloadBytes = 0;
    if (!completed) return (int)g_submissionStatus;
    ++g_restore.successfulSubmits;
    if (!checkedCounterAdd(g_h2dCount, submittedEntries) ||
        !checkedCounterAdd(g_h2dBytes, submittedBytes)) {
        g_transferCountersValid = false;
    }
    dpOneShot(g_dp.oneShotH2D);
    return 0;
}

void fvk_restore_finish(void) { restoreCleanup(); }
void fvk_restore_abort(void) { restoreCleanup(); }
int fvk_restore_active(void) {
    return (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) ? 1 : 0;
}
void fvk_debug_restore_fail_after_submits(int successful_submits) {
    g_restoreFailAfterSubmits = successful_submits;
}

// d2h is a true host fence (the final logits Read): flush the recorded batch so the compute
// has actually executed, then copy device->host.
int fvk_d2h(void* h, const void* d, size_t bytes) {
    SyncScope syncScope(kSyncSiteReadback);
    if (stepBatchRecording() && h && d && bytes > 0 && bytes <= B((void*)d)->bytes &&
        g_submissionStatus == VK_SUCCESS) {
        return stepRecordReadback(h, B((void*)d), bytes);
    }
    if (g_batching) batchFlush();
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    return copyDeviceToHost(h, B((void*)d), bytes);
}

// V4.1 status/output readback deliberately avoids the generic aborting command
// helpers. The shared staging allocation and transfer counters remain canonical.
int fvk_v41_d2h(void* h, const void* d, size_t bytes) {
    SyncScope syncScope(kSyncSiteReadback);
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    Buffer* src = B(d);
    if (!h || !src || src->buf == VK_NULL_HANDLE || src->mem == VK_NULL_HANDLE ||
        bytes == 0 || bytes > src->bytes) return VK_ERROR_INITIALIZATION_FAILED;
    if (stepBatchRecording()) return stepRecordReadback(h, src, bytes);
    if (g_batching) {
        // A marked result may be read while another batch is open. Its host
        // fence must still use the checked path, even if no V4.1 op was added.
        g_batchHasV41 = true;
        batchFlush();
    }
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    int stagingStatus = VK_SUCCESS;
    Buffer* stage = stagingBuffer(bytes, &stagingStatus);
    if (!stage) {
        if (stagingStatus == VK_ERROR_DEVICE_LOST) {
            g_submissionStatus = VK_ERROR_DEVICE_LOST;
            g_v41SubmissionPendingFailure = true;
        }
        return stagingStatus == VK_SUCCESS ? FVK_D2H_STAGING_ALLOCATION_FAILED : stagingStatus;
    }
    VkCommandBuffer cmd = v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return (int)g_submissionStatus;
    recordComputeBarrier(cmd); // shader writes must be visible to transfer reads
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, src->buf, stage->buf, 1, &region);
    dpOneShot(g_dp.oneShotD2H);
    if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
    memcpy(h, g_stageMapped, bytes);
    syncNoteD2H(bytes);
    if (!checkedCounterAdd(g_d2hCount, 1) || !checkedCounterAdd(g_d2hBytes, bytes)) {
        g_transferCountersValid = false;
    }
    return VK_SUCCESS;
}

// A synchronous loan of existing pool backing. MISS has no allocator/transport
// effects. Host input/output may alias: staging copies all input before readback.
static bool v41NormPanelLive(const Buffer* b) {
    if (!b || b->buf == VK_NULL_HANDLE || b->mem == VK_NULL_HANDLE) return false;
    if (b->weightArenaBound) {
        return b->weightArenaBlock < g_weightArena.size() &&
            g_weightArena[b->weightArenaBlock].deviceGeneration == g_memoryDeviceGeneration &&
            g_weightArena[b->weightArenaBlock].mem == b->mem;
    }
    return b->deviceGeneration == g_memoryDeviceGeneration;
}

int fvk_v41_try_pool_norm_panel(float* host, const void* gain, int rows, int width,
                                float eps, int* status) {
    if (!status) return -1;
    *status = VK_SUCCESS;
    auto fail = [&]() {
        *status = g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
        return -1;
    };
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS) return fail();
    if (!g_ready || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return fail();
    if (!host || rows < 2 || rows > 64 || width <= 0 || !std::isfinite(eps) || eps <= 0 ||
        (size_t)width > std::numeric_limits<size_t>::max() / (4u * (size_t)rows)) return fail();
    const size_t bytes = 4u * (size_t)rows * (size_t)width;
    Buffer* w = B(gain);
    if (!v41NormPanelLive(w) || w->bytes < 4u * (size_t)width) return fail();
    const bool step = stepBatchRecording();
    if (g_batching && !step) return 0;
    auto it = g_pool.find(bytes);
    if (it == g_pool.end() || it->second.size() < 2 || it->second.size() > POOL_BUCKET_CAP || g_poolCount < 2) return 0;
    auto& bucket = it->second;
    Buffer* x = bucket[bucket.size()-1];
    Buffer* y = bucket[bucket.size()-2];
    auto eligible = [&](Buffer* b) {
        return v41NormPanelLive(b) && b->bytes == bytes && b->allocationBytes >= bytes &&
            b->allocationID != 0 && b->allocationDeviceLocal && !b->weightArenaBound &&
            !b->offloadPage && !b->mapped && b->nativeCheckoutGeneration != 0 &&
            b->nativeCheckoutGeneration != std::numeric_limits<uint64_t>::max();
    };
    if (!eligible(x) || !eligible(y) || x == y || x->buf == y->buf || x->mem == y->mem ||
        !v41NormPanelLive(g_stage) || !g_stageMapped || g_stageCap < bytes) return 0;
    size_t aligned = 0;
    if (step) {
        aligned = (g_upArenaUsed + 15u) & ~(size_t)15u;
        if (!v41NormPanelLive(g_upArena) || !g_upArenaMapped || aligned < g_upArenaUsed ||
            aligned > g_upArenaCap || bytes > g_upArenaCap - aligned) return 0;
    }
    // Ordinary pooled buffers own whole allocations. Conservative distinct memory
    // rejects physical aliasing with all protected operands without guessing ranges.
    Buffer* protectedBuffers[] = {w, g_stage, step ? g_upArena : nullptr};
    for (Buffer* p : protectedBuffers) {
        if (p && (x == p || y == p || x->buf == p->buf || y->buf == p->buf ||
                  x->mem == p->mem || y->mem == p->mem)) return 0;
    }
    for (size_t i = 0; i + 2 < bucket.size(); ++i) {
        if (bucket[i] == x || bucket[i] == y) return 0;
    }
    // All refusal predicates precede this atomic pair claim. The two pops cannot
    // allocate or fail, so a partial claim never escapes and no rollback allocates.
    bucket.pop_back(); bucket.pop_back(); g_poolCount -= 2;
    advanceNativeCheckoutGeneration(x); advanceNativeCheckoutGeneration(y);
    struct Loan {
        Buffer* x; Buffer* y;
        ~Loan() { fvk_free(x); fvk_free(y); }
    } loan{x,y};
    if (step) {
        if (!stepRecordUpload(x, host, bytes)) return fail();
    } else {
        memcpy(g_stageMapped, host, bytes);
        VkCommandBuffer cmd = v41BeginCmdChecked();
        if (cmd == VK_NULL_HANDLE) return fail();
        VkBufferCopy region{0,0,bytes};
        vkCmdCopyBuffer(cmd, g_stage->buf, x->buf, 1, &region);
        dpOneShot(g_dp.oneShotH2D);
        if (!v41EndSubmitWaitChecked(cmd, true)) return fail();
        if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes)) g_transferCountersValid = false;
    }
    struct { int rows, n; float eps; } pc{rows,width,eps};
    Buffer* bufs[] = {x,w,y};
    if (!dispatch(g_kern[K_RMSNORM], bufs, &pc, sizeof(pc), (uint32_t)rows, 1, true)) return fail();
    const int readStatus = fvk_v41_d2h(host, y, bytes);
    if (readStatus != VK_SUCCESS) { *status = readStatus; return -1; }
    // Readback resumes a step batch. A failed resume must not publish a result.
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || (step && !stepBatchRecording())) return fail();
    return 1;
}

int fvk_v41_try_ffn_norm_publication(float* host, float* published, const void* gain, int rows, int width,
                                     float eps, int probe, int* status) {
    if (!status) return -1;
    *status = VK_SUCCESS;
    auto fail = [&]() {
        *status = g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
        return -1;
    };
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS) return fail();
    if (!g_ready || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return fail();
    if (!host || rows < 2 || rows > 64 || width <= 0 || !std::isfinite(eps) || eps <= 0 ||
        (size_t)width > std::numeric_limits<size_t>::max() / (4u * (size_t)rows)) return fail();
    const size_t bytes = 4u * (size_t)rows * (size_t)width;
    // Explicit host cost bound, not a memory reservation. A miss allocates nothing.
    if (bytes > 1310720u || !fvk_have_v41_attention_bf16() ||
        !fvk_v41_attention_bf16_count_supported((int)(bytes / 4))) return 0;
    if (!probe && !published) return fail();
    Buffer* w = B(gain);
    if (!v41NormPanelLive(w) || w->bytes < 4u * (size_t)width) return fail();
    const bool step = stepBatchRecording();
    if (g_batching && !step) return 0;
    auto it = g_pool.find(bytes);
    if (it == g_pool.end() || it->second.size() < 2 || it->second.size() > POOL_BUCKET_CAP || g_poolCount < 2) return 0;
    auto sit = g_pool.find(16);
    if (sit == g_pool.end() || sit == it || sit->second.empty() ||
        sit->second.size() > POOL_BUCKET_CAP || g_poolCount < 3) return 0;
    auto& statusBucket = sit->second;
    Buffer* verdict = statusBucket.back();
    auto& bucket = it->second;
    Buffer* x = bucket[bucket.size()-1];
    Buffer* y = bucket[bucket.size()-2];
    auto eligible = [&](Buffer* b, size_t required) {
        return v41NormPanelLive(b) && b->bytes == required && b->allocationBytes >= required &&
            b->allocationID != 0 && b->allocationDeviceLocal && !b->weightArenaBound &&
            !b->offloadPage && !b->mapped && b->nativeCheckoutGeneration != 0 &&
            b->nativeCheckoutGeneration != std::numeric_limits<uint64_t>::max();
    };
    if (!eligible(x, bytes) || !eligible(y, bytes) || !eligible(verdict, 16) || x == y || x->buf == y->buf || x->mem == y->mem || verdict == x || verdict == y ||
        verdict->buf == x->buf || verdict->buf == y->buf || verdict->mem == x->mem || verdict->mem == y->mem ||
        !v41NormPanelLive(g_stage) || !g_stageMapped || g_stageCap < bytes + 16u) return 0;
    size_t aligned = 0;
    if (step) {
        aligned = (g_upArenaUsed + 15u) & ~(size_t)15u;
        if (!v41NormPanelLive(g_upArena) || !g_upArenaMapped || aligned < g_upArenaUsed ||
            aligned > g_upArenaCap || bytes > g_upArenaCap - aligned) return 0;
    }
    // Ordinary pooled buffers own whole allocations. Conservative distinct memory
    // rejects physical aliasing with all protected operands without guessing ranges.
    Buffer* protectedBuffers[] = {w, g_stage, step ? g_upArena : nullptr};
    for (Buffer* p : protectedBuffers) {
        if (p && (x == p || y == p || verdict == p || x->buf == p->buf || y->buf == p->buf || verdict->buf == p->buf ||
                  x->mem == p->mem || y->mem == p->mem || verdict->mem == p->mem)) return 0;
    }
    auto aliasesLoan = [&](Buffer* other) {
        if (!other) return false;
        for (Buffer* claimed : {x, y, verdict}) {
            if (other == claimed || other->buf == claimed->buf || other->mem == claimed->mem) return true;
        }
        return false;
    };
    for (size_t i = 0; i + 2 < bucket.size(); ++i) {
        if (aliasesLoan(bucket[i])) return 0;
    }
    for (size_t i = 0; i + 1 < statusBucket.size(); ++i) {
        if (aliasesLoan(statusBucket[i])) return 0;
    }
    // Probe and commit run under the same caller lock. The probe changes no owner,
    // generation, counters or payload, permitting bounded host allocation afterward.
    if (probe) return 1;
    // All refusal predicates precede this atomic triple claim. The three pops cannot
    // allocate or fail, so a partial claim never escapes and no rollback allocates.
    bucket.pop_back(); bucket.pop_back(); statusBucket.pop_back(); g_poolCount -= 3;
    advanceNativeCheckoutGeneration(x); advanceNativeCheckoutGeneration(y); advanceNativeCheckoutGeneration(verdict);
    struct Loan {
        Buffer* x; Buffer* y; Buffer* verdict;
        ~Loan() { fvk_free(x); fvk_free(y); fvk_free(verdict); }
    } loan{x,y,verdict};
    if (step) {
        if (!stepRecordUpload(x, host, bytes)) return fail();
    } else {
        memcpy(g_stageMapped, host, bytes);
        VkCommandBuffer cmd = v41BeginCmdChecked();
        if (cmd == VK_NULL_HANDLE) return fail();
        VkBufferCopy region{0,0,bytes};
        vkCmdCopyBuffer(cmd, g_stage->buf, x->buf, 1, &region);
        dpOneShot(g_dp.oneShotH2D);
        if (!v41EndSubmitWaitChecked(cmd, true)) return fail();
        if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes)) g_transferCountersValid = false;
    }
    struct { int rows, n; float eps; } pc{rows,width,eps};
    Buffer* bufs[] = {x,w,y};
    if (!dispatch(g_kern[K_RMSNORM], bufs, &pc, sizeof(pc), (uint32_t)rows, 1, true)) return fail();
    const int publishStatus = fvk_v41_bf16_publish_f32(y, x, verdict, (int)(bytes / 4), 10);
    if (publishStatus != VK_SUCCESS) { *status = publishStatus; return -1; }
    const int readStatus = fvk_v41_d2h(host, y, bytes);
    if (readStatus != VK_SUCCESS) { *status = readStatus; return -1; }
    // Readback resumes a step batch. A failed resume must not publish a result.
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || (step && !stepBatchRecording())) return fail();
    void* sources[] = {verdict, x};
    const size_t offsets[] = {0, 16};
    const size_t lengths[] = {16, bytes};
    const int publicationRead = fvk_v41_d2h_many(published, bytes + 16, 2, sources, offsets, lengths);
    if (publicationRead != VK_SUCCESS) { *status = publicationRead; return -1; }
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || (step && !stepBatchRecording())) return fail();
    return 1;
}

// Query-only heterogeneous pool loan: preserves both original capacity keys.
int fvk_v41_try_query_norm_publication(float* host, float* published, const void* gain, int rows, int width, int capacityRows,
                                     float eps, int probe, int* status) {
    if (!status) return -1;
    *status = VK_SUCCESS;
    auto fail = [&]() {
        *status = g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
        return -1;
    };
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS) return fail();
    if (!g_ready || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return fail();
    if (!host || rows < 2 || rows > 64 || width != 1280 || capacityRows < rows || capacityRows > 64 || !std::isfinite(eps) || eps <= 0 ||
        (size_t)width > std::numeric_limits<size_t>::max() / (4u * (size_t)rows)) return fail();
    const size_t bytes = 4u * (size_t)rows * (size_t)width;
    // Explicit host cost bound, not a memory reservation. A miss allocates nothing.
    if (bytes > 327680u || !fvk_have_v41_attention_bf16() ||
        !fvk_v41_attention_bf16_count_supported((int)(bytes / 4))) return 0;
    if (!probe && !published) return fail();
    Buffer* w = B(gain);
    if (!v41NormPanelLive(w) || w->bytes < 4u * (size_t)width) return fail();
    const bool step = stepBatchRecording();
    if (!step) return 0; // Cold serving Prefill owns this checked recording window.
    const size_t smallBytes = 4u * (size_t)capacityRows * 1280u;
    const size_t largeBytes = 4u * (size_t)capacityRows * 5120u;
    auto it = g_pool.find(smallBytes);
    auto lit = g_pool.find(largeBytes);
    if (it == g_pool.end() || lit == g_pool.end() || it == lit || it->second.empty() || lit->second.empty() ||
        it->second.size() > POOL_BUCKET_CAP || lit->second.size() > POOL_BUCKET_CAP) return 0;
    auto sit = g_pool.find(16);
    if (sit == g_pool.end() || sit == it || sit == lit || sit->second.empty() ||
        sit->second.size() > POOL_BUCKET_CAP || g_poolCount < 3) return 0;
    auto& statusBucket = sit->second;
    Buffer* verdict = statusBucket.back();
    auto& bucket = it->second;
    auto& largeBucket = lit->second;
    Buffer* x = bucket.back();
    Buffer* y = largeBucket.back();
    auto eligible = [&](Buffer* b, size_t required) {
        return v41NormPanelLive(b) && b->bytes == required && b->allocationBytes >= required &&
            b->allocationID != 0 && b->allocationDeviceLocal && !b->weightArenaBound &&
            !b->offloadPage && !b->mapped && b->nativeCheckoutGeneration != 0 &&
            b->nativeCheckoutGeneration != std::numeric_limits<uint64_t>::max();
    };
    if (!eligible(x, smallBytes) || !eligible(y, largeBytes) || !eligible(verdict, 16) || x == y || x->buf == y->buf || x->mem == y->mem || verdict == x || verdict == y ||
        verdict->buf == x->buf || verdict->buf == y->buf || verdict->mem == x->mem || verdict->mem == y->mem ||
        !v41NormPanelLive(g_stage) || !g_stageMapped || g_stageCap < bytes + 16u) return 0;
    size_t aligned = 0;
    if (step) {
        aligned = (g_upArenaUsed + 15u) & ~(size_t)15u;
        if (!v41NormPanelLive(g_upArena) || !g_upArenaMapped || aligned < g_upArenaUsed ||
            aligned > g_upArenaCap || bytes > g_upArenaCap - aligned) return 0;
    }
    // Ordinary pooled buffers own whole allocations. Conservative distinct memory
    // rejects physical aliasing with all protected operands without guessing ranges.
    Buffer* protectedBuffers[] = {w, g_stage, step ? g_upArena : nullptr};
    for (Buffer* p : protectedBuffers) {
        if (p && (x == p || y == p || verdict == p || x->buf == p->buf || y->buf == p->buf || verdict->buf == p->buf ||
                  x->mem == p->mem || y->mem == p->mem || verdict->mem == p->mem)) return 0;
    }
    auto aliasesLoan = [&](Buffer* other) {
        if (!other) return false;
        for (Buffer* claimed : {x, y, verdict}) {
            if (other == claimed || other->buf == claimed->buf || other->mem == claimed->mem) return true;
        }
        return false;
    };
    for (size_t i = 0; i + 1 < bucket.size(); ++i) {
        if (aliasesLoan(bucket[i])) return 0;
    }
    for (size_t i = 0; i + 1 < largeBucket.size(); ++i) {
        if (aliasesLoan(largeBucket[i])) return 0;
    }
    for (size_t i = 0; i + 1 < statusBucket.size(); ++i) {
        if (aliasesLoan(statusBucket[i])) return 0;
    }
    // Probe and commit run under the same caller lock. The probe changes no owner,
    // generation, counters or payload, permitting bounded host allocation afterward.
    if (probe) return 1;
    // All refusal predicates precede this atomic triple claim. The three pops cannot
    // allocate or fail, so a partial claim never escapes and no rollback allocates.
    bucket.pop_back(); largeBucket.pop_back(); statusBucket.pop_back(); g_poolCount -= 3;
    advanceNativeCheckoutGeneration(x); advanceNativeCheckoutGeneration(y); advanceNativeCheckoutGeneration(verdict);
    struct Loan {
        Buffer* x; Buffer* y; Buffer* verdict;
        ~Loan() { fvk_free(x); fvk_free(y); fvk_free(verdict); }
    } loan{x,y,verdict};
    if (!stepRecordUpload(x, host, bytes)) return fail();
    const int inputDispatch = fvk_v41_bf16_publish_f32(x, y, verdict, (int)(bytes / 4), 5);
    if (inputDispatch != VK_SUCCESS) { *status = inputDispatch; return -1; }
    // Preserve the input envelope before the output kernel resets the same status.
    // Numeric verdicts are consumed row-first by the owner after all transport.
    const int inputRead = fvk_v41_d2h(published, verdict, 16);
    if (inputRead != VK_SUCCESS) { *status = inputRead; return -1; }
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || !stepBatchRecording()) return fail();
    struct { int rows, n; float eps; } pc{rows,width,eps};
    Buffer* bufs[] = {y,w,x};
    if (!dispatch(g_kern[K_RMSNORM], bufs, &pc, sizeof(pc), (uint32_t)rows, 1, true)) return fail();
    const int publishStatus = fvk_v41_bf16_publish_f32(x, y, verdict, (int)(bytes / 4), 6);
    if (publishStatus != VK_SUCCESS) { *status = publishStatus; return -1; }
    const int readStatus = fvk_v41_d2h(host, x, bytes);
    if (readStatus != VK_SUCCESS) { *status = readStatus; return -1; }
    // Readback resumes a step batch. A failed resume must not publish a result.
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || (step && !stepBatchRecording())) return fail();
    void* sources[] = {verdict, y};
    const size_t offsets[] = {0, 16};
    const size_t lengths[] = {16, bytes};
    const int publicationRead = fvk_v41_d2h_many(published + 4, bytes + 16, 2, sources, offsets, lengths);
    if (publicationRead != VK_SUCCESS) { *status = publicationRead; return -1; }
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || (step && !stepBatchRecording())) return fail();
    return 1;
}

// fvk_v41_d2h for several regions behind one host fence (#13820): each copy
// lands at its caller-chosen staging offset, so k status records and rows cost
// one submit instead of k. Regions are ordered and disjoint in the host image.
// Single-row KV chain. Claims only existing pool capacity; no allocator calls.
int fvk_v41_try_kv_norm_publication(const float* host, float* image, const void* gain, float eps, int probe, int* status) {
    SyncScope syncScope(kSyncSiteReadback);
    if (!status) return -1;
    *status=VK_SUCCESS;
    auto fail=[&]() { *status=g_submissionStatus!=VK_SUCCESS?(int)g_submissionStatus:(int)VK_ERROR_UNKNOWN; return -1; };
    if (g_v41SubmissionPendingFailure || g_submissionStatus!=VK_SUCCESS) return fail();
    if (!g_ready || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration || g_restore.stage || g_restore.cmd!=VK_NULL_HANDLE) return fail();
    constexpr size_t bytes=2048u, hostBytes=4128u;
    if (!host || !std::isfinite(eps) || eps<=0) return fail();
    if (!stepBatchRecording() || !fvk_have_v41_attention_bf16() || !fvk_v41_attention_bf16_count_supported(512)) return 0;
    Buffer* w=B(gain);
    size_t aligned=(g_upArenaUsed+15u)&~(size_t)15u;
    if (!v41NormPanelLive(w) || w->bytes<bytes || !v41NormPanelLive(g_stage) || !g_stageMapped || g_stageCap<hostBytes || g_stage->bytes<hostBytes ||
        !v41NormPanelLive(g_upArena) || !g_upArenaMapped || aligned<g_upArenaUsed || aligned>g_upArenaCap || bytes>g_upArenaCap-aligned) return 0;
    auto aliases=[](Buffer* a,Buffer* b){return a==b || a->buf==b->buf || a->mem==b->mem;};
    Buffer* protectedBuffers[]={w,g_stage,g_upArena};
    for(size_t i=0;i<3;++i) for(size_t j=0;j<i;++j) if(aliases(protectedBuffers[i],protectedBuffers[j])) return 0;
    auto it=g_pool.find(bytes),sit=g_pool.find(16u);
    if(it==g_pool.end() || sit==g_pool.end() || it->second.size()<2 || sit->second.empty() || it->second.size()>POOL_BUCKET_CAP || sit->second.size()>POOL_BUCKET_CAP || g_poolCount<3) return 0;
    Buffer* x=it->second.back(); Buffer* y=it->second[it->second.size()-2]; Buffer* verdict=sit->second.back();
    Buffer* claims[]={x,y,verdict};
    auto eligible=[](Buffer* b,size_t n){return v41NormPanelLive(b) && b->bytes==n && b->allocationBytes>=n && b->allocationID && b->allocationDeviceLocal && !b->weightArenaBound && !b->offloadPage && !b->mapped && b->nativeCheckoutGeneration && b->nativeCheckoutGeneration!=std::numeric_limits<uint64_t>::max();};
    if(!eligible(x,bytes) || !eligible(y,bytes) || !eligible(verdict,16u)) return 0;
    for(size_t i=0;i<3;++i){for(size_t j=0;j<i;++j) if(aliases(claims[i],claims[j])) return 0; for(Buffer* p:protectedBuffers) if(aliases(claims[i],p)) return 0;}
    size_t visits=0,seen[3]={0,0,0};
    auto inspect=[&](Buffer* p,bool pool){
        if(++visits>1024 || !p) return false;
        for(Buffer* q:protectedBuffers) if(aliases(p,q)) return false;
        for(size_t i=0;i<3;++i) if(aliases(p,claims[i]) && (!pool || p!=claims[i] || ++seen[i]!=1)) return false;
        return true;
    };
    for(const auto& bucket:g_pool){if(++visits>1024)return 0;for(Buffer* p:bucket.second)if(!inspect(p,true))return 0;}
    for(const auto& bucket:g_hvOffloadPool){if(++visits>1024)return 0;for(Buffer* p:bucket.second)if(!inspect(p,false))return 0;}
    for(Buffer* p:g_batchFreed)if(!inspect(p,false))return 0;
    for(size_t n:seen)if(n!=1)return 0;
    if(probe)return 1;
    if(!image)return fail();
    it->second.pop_back();it->second.pop_back();sit->second.pop_back();g_poolCount-=3;
    for(Buffer* b:claims)advanceNativeCheckoutGeneration(b);
    struct Loan{Buffer* x;Buffer* y;Buffer* verdict;~Loan(){fvk_free(x);fvk_free(y);fvk_free(verdict);}}loan{x,y,verdict};
    if(!stepRecordUpload(x,host,bytes))return fail();
    int rc=fvk_v41_bf16_publish_f32(x,y,verdict,512,7);
    if(rc!=VK_SUCCESS){*status=rc;return -1;}
    if(!stepBatchRecording())return fail();
    // Preserve the first envelope before reusing the status allocation. This
    // transfer is in the same submission, with an explicit WAR dependency.
    recordComputeBarrier(g_batchCmd);
    VkBufferCopy first{0,0,16};vkCmdCopyBuffer(g_batchCmd,verdict->buf,g_stage->buf,1,&first);tsAfterDispatch(g_batchCmd,kTsReadback,0,0);++g_batchOps;
    recordTransferBarrier(g_batchCmd);
    struct{int rows,n;float eps;}pc{1,512,eps};Buffer* normBuffers[]={y,w,x};
    if(!dispatch(g_kern[K_RMSNORM],normBuffers,&pc,sizeof(pc),1,1,true))return fail();
    rc=fvk_v41_bf16_publish_f32(x,y,verdict,512,8);
    if(rc!=VK_SUCCESS){*status=rc;return -1;}
    if(!stepBatchRecording())return fail();
    recordComputeBarrier(g_batchCmd);
    Buffer* sources[]={x,verdict,y};const size_t offsets[]={16u,2064u,2080u},lengths[]={bytes,16u,bytes};
    for(size_t i=0;i<3;++i){VkBufferCopy region{0,offsets[i],lengths[i]};vkCmdCopyBuffer(g_batchCmd,sources[i]->buf,g_stage->buf,1,&region);tsAfterDispatch(g_batchCmd,kTsReadback,0,0);}
    g_batchOps+=3;batchFlush();
    if(g_v41SubmissionPendingFailure || g_submissionStatus!=VK_SUCCESS || g_batching)return fail();
    batchBegin();
    if(g_v41SubmissionPendingFailure || g_submissionStatus!=VK_SUCCESS || !stepBatchRecording())return fail();
    memcpy(image,g_stageMapped,hostBytes);
    const size_t copied[]={16u,bytes,16u,bytes};
    for (size_t n : copied) syncNoteD2H(n);
    for(size_t n:copied)if(!checkedCounterAdd(g_d2hCount,1)||!checkedCounterAdd(g_d2hBytes,n))g_transferCountersValid=false;
    return 1;
}

// Fixed single-row index projection publication. The caller holds vulkanMu
// across probe, bounded Go allocation and commit. No stage growth or allocation.
int fvk_v41_try_index_query_publication(float* raw, float* published, const void* source,
                                        const void* weight, const void* input, int probe, int* status) {
    SyncScope syncScope(kSyncSiteReadback);
    if (!status) return -1;
    *status = VK_SUCCESS;
    auto fail = [&]() {
        *status = g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
        return -1;
    };
    if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS) return fail();
    if (!g_ready || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return fail();
    constexpr size_t bytes = 32768u, hostBytes = 65552u;
    if (!stepBatchRecording() || !fvk_have_v41_attention_bf16() ||
        !fvk_v41_attention_bf16_count_supported(8192)) return 0;
    Buffer* y = B(source); Buffer* w = B(weight); Buffer* x = B(input);
    if (!v41NormPanelLive(y) || !v41NormPanelLive(w) || !v41NormPanelLive(x) ||
        y->bytes < bytes || !y->nativeCheckoutGeneration ||
        !v41NormPanelLive(g_stage) || !g_stageMapped || g_stageCap < hostBytes ||
        g_stage->bytes < hostBytes) return 0;
    auto aliases = [](Buffer* a, Buffer* b) {
        return a == b || a->buf == b->buf || a->mem == b->mem;
    };
    Buffer* protectedBuffers[] = {y,w,x,g_stage};
    for (size_t i=0;i<4;++i) for (size_t j=0;j<i;++j)
        if (aliases(protectedBuffers[i],protectedBuffers[j])) return 0;
    auto it=g_pool.find(bytes), sit=g_pool.find(16u);
    if (it==g_pool.end() || sit==g_pool.end() || it->second.empty() || sit->second.empty() ||
        it->second.size()>POOL_BUCKET_CAP || sit->second.size()>POOL_BUCKET_CAP || g_poolCount<2) return 0;
    Buffer* out=it->second.back(); Buffer* verdict=sit->second.back();
    auto eligible=[](Buffer* b,size_t n) {
        return v41NormPanelLive(b) && b->bytes==n && b->allocationBytes>=n && b->allocationID &&
            b->allocationDeviceLocal && !b->weightArenaBound && !b->offloadPage && !b->mapped &&
            b->nativeCheckoutGeneration && b->nativeCheckoutGeneration!=std::numeric_limits<uint64_t>::max();
    };
    if (!eligible(out,bytes) || !eligible(verdict,16u) || aliases(out,verdict)) return 0;
    for (Buffer* p:protectedBuffers) if (aliases(out,p) || aliases(verdict,p)) return 0;
    if (g_upArena && (aliases(out,g_upArena) || aliases(verdict,g_upArena))) return 0;
    // A bounded ownership census includes empty historical buckets. Refuse if
    // the complete census cannot fit; no partial census grants a pool claim.
    size_t visits=0, outSeen=0, verdictSeen=0;
    auto inspect=[&](Buffer* p,bool pool) {
        if (++visits>1024 || !p) return false;
        for (Buffer* protectedBuffer:protectedBuffers) if (aliases(p,protectedBuffer)) return false;
        if (aliases(p,out)) { if (!pool || p!=out || ++outSeen!=1) return false; }
        if (aliases(p,verdict)) { if (!pool || p!=verdict || ++verdictSeen!=1) return false; }
        return true;
    };
    for (const auto& bucket:g_pool) {
        if (++visits>1024) return 0;
        for (Buffer* p:bucket.second) if (!inspect(p,true)) return 0;
    }
    for (const auto& bucket:g_hvOffloadPool) {
        if (++visits>1024) return 0;
        for (Buffer* p:bucket.second) if (!inspect(p,false)) return 0;
    }
    for (Buffer* p:g_batchFreed) if (!inspect(p,false)) return 0;
    if (outSeen!=1 || verdictSeen!=1) return 0;
    if (probe) return 1;
    if (!raw || !published || raw==published) return fail();
    it->second.pop_back(); sit->second.pop_back(); g_poolCount-=2;
    advanceNativeCheckoutGeneration(out); advanceNativeCheckoutGeneration(verdict);
    struct Loan { Buffer* out; Buffer* verdict; ~Loan(){ fvk_free(out); fvk_free(verdict); } } loan{out,verdict};
    const int publishStatus=fvk_v41_bf16_publish_f32(y,out,verdict,8192,22);
    if (publishStatus!=VK_SUCCESS) { *status=publishStatus; return -1; }
    if (!stepBatchRecording()) return fail();
    // Same stage layout and synchronization as d2h_many, with a fixed scatter
    // into two independent host owners instead of a retained combined slab.
    recordComputeBarrier(g_batchCmd);
    Buffer* sources[]={verdict,y,out};
    const size_t offsets[]={0u,16u,16u+bytes};
    const size_t lengths[]={16u,bytes,bytes};
    for (size_t i=0;i<3;++i) {
        VkBufferCopy region{0,offsets[i],lengths[i]};
        vkCmdCopyBuffer(g_batchCmd,sources[i]->buf,g_stage->buf,1,&region);
        tsAfterDispatch(g_batchCmd,kTsReadback,0,0);
    }
    g_batchOps+=3;
    batchFlush();
    if (g_v41SubmissionPendingFailure || g_submissionStatus!=VK_SUCCESS || g_batching) return fail();
    batchBegin();
    if (g_v41SubmissionPendingFailure || g_submissionStatus!=VK_SUCCESS || !stepBatchRecording()) return fail();
    memcpy(raw,(const char*)g_stageMapped+16,bytes);
    memcpy(published,g_stageMapped,16);
    memcpy(published+4,(const char*)g_stageMapped+16+bytes,bytes);
    for (size_t n : lengths) syncNoteD2H(n);
    for (size_t n:lengths) {
        if (!checkedCounterAdd(g_d2hCount,1) || !checkedCounterAdd(g_d2hBytes,n)) g_transferCountersValid=false;
    }
    return 1;
}

int fvk_v41_d2h_many(void* host, size_t hostBytes, int count, void* const* srcs,
                     const size_t* offsets, const size_t* bytes) {
    SyncScope syncScope(kSyncSiteReadback);
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    if (!host || hostBytes == 0 || count <= 0 || !srcs || !offsets || !bytes) return VK_ERROR_INITIALIZATION_FAILED;
    size_t end = 0;
    for (int i = 0; i < count; ++i) {
        Buffer* src = B(srcs[i]);
        if (!src || src->buf == VK_NULL_HANDLE || src->mem == VK_NULL_HANDLE || bytes[i] == 0 ||
            bytes[i] > src->bytes || offsets[i] < end || offsets[i] > hostBytes ||
            bytes[i] > hostBytes - offsets[i]) return VK_ERROR_INITIALIZATION_FAILED;
        end = offsets[i] + bytes[i];
    }
    const bool recording = stepBatchRecording();
    if (!recording && g_batching) {
        g_batchHasV41 = true;
        batchFlush();
    }
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    int stagingStatus = VK_SUCCESS;
    Buffer* stage = stagingBuffer(hostBytes, &stagingStatus);
    if (!stage) {
        if (stagingStatus == VK_ERROR_DEVICE_LOST) {
            g_submissionStatus = VK_ERROR_DEVICE_LOST;
            g_v41SubmissionPendingFailure = true;
        }
        return stagingStatus == VK_SUCCESS ? FVK_D2H_STAGING_ALLOCATION_FAILED : stagingStatus;
    }
    VkCommandBuffer cmd = recording ? g_batchCmd : v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return (int)g_submissionStatus;
    recordComputeBarrier(cmd); // shader writes must be visible to transfer reads
    for (int i = 0; i < count; ++i) {
        VkBufferCopy region{0, offsets[i], bytes[i]};
        vkCmdCopyBuffer(cmd, B(srcs[i])->buf, stage->buf, 1, &region);
        if (recording) tsAfterDispatch(cmd, kTsReadback, 0, 0);
    }
    if (recording) {
        g_batchOps += count;
        batchFlush();
        if (g_v41SubmissionPendingFailure || g_submissionStatus != VK_SUCCESS || g_batching) {
            return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : (int)VK_ERROR_UNKNOWN;
        }
    } else {
        dpOneShot(g_dp.oneShotD2H);
        if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
    }
    memcpy(host, g_stageMapped, hostBytes);
    for (int i = 0; i < count; ++i) syncNoteD2H(bytes[i]);
    for (int i = 0; i < count; ++i) {
        if (!checkedCounterAdd(g_d2hCount, 1) || !checkedCounterAdd(g_d2hBytes, bytes[i])) {
            g_transferCountersValid = false;
        }
    }
    if (recording) batchBegin();
    return VK_SUCCESS;
}

void fvk_debug_d2h_staging_failure_once(int enabled) {
    g_debugD2HStagingFailureOnce = enabled != 0;
}

// Test-only kernel selection for the decode matvec A/B (bit-parity) witnesses. Each call
// selects the optimized kernel when enabled != 0 AND its pipeline was built, returns the
// previous selection, and never changes which pipelines exist.
int fvk_debug_select_q8_gateup_coop(int enabled) {
    int prev = g_have_rmsnorm_q8_matmul2_coop;
    g_have_rmsnorm_q8_matmul2_coop = (enabled && g_kern[K_RMSNORM_Q8_MATMUL2_COOP].pipe != VK_NULL_HANDLE) ? 1 : 0;
    return prev;
}
int fvk_debug_select_iq_matvec(int fmt, int enabled) {
    if (fmt < 0 || fmt >= FVK_IQ_FORMATS) return 0;
    int prev = g_have_iq[fmt];
    g_have_iq[fmt] = (enabled && g_kern[kIQKernel[fmt]].pipe != VK_NULL_HANDLE) ? 1 : 0;
    return prev;
}
int fvk_debug_select_q2k_matvec(int enabled) {
    int prev = g_have_q2k_matvec;
    g_have_q2k_matvec = (enabled && g_kern[K_Q2K_MATVEC].pipe != VK_NULL_HANDLE) ? 1 : 0;
    return prev;
}
int fvk_debug_select_qwen35_value_head_tile(int enabled) {
    int prev = g_have_qwen35_value_head_tile;
    g_have_qwen35_value_head_tile = (enabled && g_kern[K_QWEN35_VALUE_HEAD_TILE].pipe != VK_NULL_HANDLE) ? 1 : 0;
    return prev;
}

// device->device copies (RoPE's copy-then-rotate, the KV append) are RECORDED into the open
// batch with a preceding barrier, so they stay ordered against the compute that produced the
// source — no premature submit. Unbatched, they take the one-shot path.
void fvk_d2d_range(void* dst, size_t dst_off, const void* src, size_t src_off, size_t bytes) {
    if (g_v41SubmissionPendingFailure) return;
    if (bytes == 0 || !dst || !src) return;
    if (g_batching) {
        if (g_batchOps > 0) recordDispatchBarrier(g_batchCmd, g_batchOps);
        VkBufferCopy region{src_off, dst_off, bytes};
        dp_inc(g_dp.d2d);
        vkCmdCopyBuffer(g_batchCmd, B((void*)src)->buf, B(dst)->buf, 1, &region);
        tsAfterDispatch(g_batchCmd, kTsD2DCopy, 0, 0);
        if (g_batchD2DCount == std::numeric_limits<uint64_t>::max() ||
            bytes > std::numeric_limits<uint64_t>::max() - g_batchD2DBytes) {
            g_batchD2DValid = false;
        } else {
            ++g_batchD2DCount;
            g_batchD2DBytes += bytes;
        }
        ++g_batchOps;
        return;
    }
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{src_off, dst_off, bytes};
    dp_inc(g_dp.d2d);
    vkCmdCopyBuffer(cmd, B((void*)src)->buf, B(dst)->buf, 1, &region);
    dpOneShot(g_dp.oneShotD2D);
    endSubmitWait(cmd);
    if (!checkedCounterAdd(g_d2dCount, 1) || !checkedCounterAdd(g_d2dBytes, bytes)) {
        g_transferCountersValid = false;
    }
}

// Single-table counterpart of the bounded multi-source gather below. Prompt
// length is not limited by its fixed multi-source metadata budget.
int fvk_d2d_gather_rows(void* dstHandle, const void* srcHandle, const intptr_t* ids,
                        int count, size_t rowBytes) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready) return 1;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (!ids || count <= 0 || rowBytes == 0 || (rowBytes & 3u) != 0 ||
        size_t(count) > std::numeric_limits<size_t>::max() / rowBytes ||
        size_t(count) > std::numeric_limits<size_t>::max() / sizeof(VkBufferCopy)) return 2;
    Buffer* dst = B(dstHandle);
    Buffer* src = B((void*)srcHandle);
    const auto live = [](const Buffer* b) {
        return b && b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE && b->bytes > 0 &&
            b->bytes <= std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset;
    };
    const size_t total = size_t(count) * rowBytes;
    if (!live(dst) || !live(src) || total > dst->bytes || rowBytes > src->bytes ||
        src == dst || src->buf == dst->buf ||
        (src->mem == dst->mem && src->memoryOffset < dst->memoryOffset + dst->bytes &&
         dst->memoryOffset < src->memoryOffset + src->bytes) ||
        (g_batching && (g_batchCmd == VK_NULL_HANDLE || g_batchOps == std::numeric_limits<int>::max()))) return 2;
    const size_t rows = src->bytes / rowBytes;
    for (int i = 0; i < count; ++i) {
        if (ids[i] < 0 || size_t(ids[i]) >= rows) return 2;
    }
    // Allocate before touching the command buffer. No C++ allocation exception
    // may cross the C ABI, and no partial gather may precede an invalid row.
    VkBufferCopy* regions = (VkBufferCopy*)std::malloc(size_t(count) * sizeof(VkBufferCopy));
    if (!regions) return (int)VK_ERROR_OUT_OF_HOST_MEMORY;
    for (int i = 0; i < count; ++i) {
        regions[i] = VkBufferCopy{size_t(ids[i]) * rowBytes, size_t(i) * rowBytes, rowBytes};
    }
    const bool recording = g_batching;
    VkCommandBuffer cmd = recording ? g_batchCmd : v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) {
        std::free(regions);
        return (int)g_submissionStatus;
    }
    // Repeated source rows are legal; destination rows and source/destination
    // allocations are disjoint (VUID 00117/12482). Keep the conservative barrier
    // rather than relying on a single-row hazard-elision verdict.
    recordComputeBarrier(cmd);
    vkCmdCopyBuffer(cmd, src->buf, dst->buf, uint32_t(count), regions);
    std::free(regions); // vkCmdCopyBuffer consumes the region metadata now.
    dp_inc(g_dp.d2d);
    if (recording) {
        g_batchHasV41 = true;
        tsAfterDispatch(cmd, kTsD2DGather, 0, 0);
        ++g_batchOps;
        // Transfer counters retain logical rows/bytes; dispatch profiling counts
        // the single native command, just as the multi-source gather does.
        if (uint64_t(count) > std::numeric_limits<uint64_t>::max() - g_batchD2DCount ||
            total > std::numeric_limits<uint64_t>::max() - g_batchD2DBytes) {
            g_batchD2DValid = false;
        } else {
            g_batchD2DCount += uint64_t(count);
            g_batchD2DBytes += total;
        }
    } else {
        recordComputeBarrier(cmd);
        dpOneShot(g_dp.oneShotD2D);
        if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
        if (!checkedCounterAdd(g_d2dCount, uint64_t(count)) || !checkedCounterAdd(g_d2dBytes, total)) {
            g_transferCountersValid = false;
        }
    }
    return (int)g_submissionStatus;
}

// A fresh gather output has no read/write dependency between its disjoint
// destination regions. Source regions may repeat (VUID 00117/12482): only
// destination overlap and source/destination memory overlap are forbidden.
// Fixed metadata bounds avoid an allocation failure after recording starts.
int fvk_v41_d2d_gather(void* dstHandle, int count, void* const* srcs,
                       const size_t* srcOffsets, const size_t* bytes) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready) return 1;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (count <= 0 || count > FVK_V41_GATHER_MAX_REGIONS || !srcs || !srcOffsets || !bytes) return 2;
    Buffer* dst = B(dstHandle);
    const auto live = [](const Buffer* b) {
        return b && b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE && b->bytes > 0 &&
            b->bytes <= std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset;
    };
    if (!live(dst)) return 2;

    Buffer* sources[FVK_V41_GATHER_MAX_REGIONS];
    int groups[FVK_V41_GATHER_MAX_REGIONS];
    uint32_t sizes[FVK_V41_GATHER_MAX_REGIONS]{};
    uint32_t starts[FVK_V41_GATHER_MAX_REGIONS];
    VkBufferCopy ordered[FVK_V41_GATHER_MAX_REGIONS];
    int sourceCount = 0;
    size_t total = 0;
    for (int i = 0; i < count; ++i) {
        Buffer* src = B(srcs[i]);
        if (!live(src) || bytes[i] == 0 || ((srcOffsets[i] | bytes[i]) & 3u) != 0 || srcOffsets[i] > src->bytes ||
            bytes[i] > src->bytes - srcOffsets[i] || total > dst->bytes || bytes[i] > dst->bytes - total) return 2;
        // Require the entire fresh output allocation to be disjoint, including
        // distinct VkBuffer objects bound into one VkDeviceMemory allocation.
        if (src == dst || src->buf == dst->buf ||
            (src->mem == dst->mem && src->memoryOffset < dst->memoryOffset + dst->bytes &&
             dst->memoryOffset < src->memoryOffset + src->bytes)) return 2;
        int group = 0;
        while (group < sourceCount && sources[group]->buf != src->buf) ++group;
        if (group == sourceCount) sources[sourceCount++] = src;
        groups[i] = group;
        ++sizes[group];
        total += bytes[i];
    }
    if (total != dst->bytes || (g_batching && (g_batchCmd == VK_NULL_HANDLE ||
        g_batchOps > std::numeric_limits<int>::max() - sourceCount))) return 2;
    uint32_t at = 0;
    for (int group = 0; group < sourceCount; ++group) {
        starts[group] = at;
        at += sizes[group];
        sizes[group] = 0;
    }
    size_t destination = 0;
    for (int i = 0; i < count; ++i) {
        const int group = groups[i];
        ordered[starts[group] + sizes[group]++] = VkBufferCopy{srcOffsets[i], destination, bytes[i]};
        destination += bytes[i];
    }

    const bool recording = g_batching;
    VkCommandBuffer cmd = recording ? g_batchCmd : v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return (int)g_submissionStatus;
    // This must not trust a single-operation hazard-elision verdict: the
    // aggregate reads several sources. The next ordinary consumer keeps its
    // existing transfer-write dependency barrier after the whole gather.
    recordComputeBarrier(cmd);
    if (recording) g_batchHasV41 = true;
    for (int group = 0; group < sourceCount; ++group) {
        vkCmdCopyBuffer(cmd, sources[group]->buf, dst->buf, sizes[group], ordered + starts[group]);
        dp_inc(g_dp.d2d); // physical copy commands, not logical regions
        if (recording) {
            tsAfterDispatch(cmd, kTsD2DGather, 0, 0);
            ++g_batchOps;
        }
    }
    if (recording) {
        // Preserve existing transfer accounting: a logical span remains one
        // region, even though native dispatch profiling counts fewer commands.
        if (uint64_t(count) > std::numeric_limits<uint64_t>::max() - g_batchD2DCount ||
            total > std::numeric_limits<uint64_t>::max() - g_batchD2DBytes) {
            g_batchD2DValid = false;
        } else {
            g_batchD2DCount += uint64_t(count);
            g_batchD2DBytes += total;
        }
    } else {
        // A later unbatched consumer may start without a batch predecessor.
        // Publish transfer writes for both shader and transfer reads now.
        recordComputeBarrier(cmd);
        dpOneShot(g_dp.oneShotD2D);
        if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
        if (!checkedCounterAdd(g_d2dCount, uint64_t(count)) || !checkedCounterAdd(g_d2dBytes, total)) {
            g_transferCountersValid = false;
        }
    }
    return (int)g_submissionStatus;
}

void fvk_d2d(void* dst, const void* src, size_t bytes) {
    fvk_d2d_range(dst, 0, src, 0, bytes);
}

void fvk_d2d_off(void* dst, size_t dst_off, const void* src, size_t bytes) {
    fvk_d2d_range(dst, dst_off, src, 0, bytes);
}

// Reassemble a chunked projection's token-major panel without a copy command
// and dependency barrier for every token. The source rows are contiguous;
// destination gaps belong to other chunks and must remain untouched.
int fvk_d2d_strided_rows(void* dstHandle, size_t dstOffset, size_t dstStride,
                         const void* srcHandle, int count, size_t rowBytes) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready) return 1;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (count <= 0 || rowBytes == 0 || dstStride < rowBytes ||
        ((dstOffset | dstStride | rowBytes) & 3u) != 0 ||
        size_t(count) > std::numeric_limits<size_t>::max() / sizeof(VkBufferCopy)) return 2;
    Buffer* dst = B(dstHandle);
    Buffer* src = B((void*)srcHandle);
    const auto live = [](const Buffer* b) {
        return b && b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE && b->bytes > 0 &&
            b->bytes <= std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset;
    };
    if (!live(dst) || !live(src) || size_t(count) > src->bytes / rowBytes ||
        dstOffset > dst->bytes || rowBytes > dst->bytes - dstOffset ||
        size_t(count - 1) > (dst->bytes - dstOffset - rowBytes) / dstStride ||
        src == dst || src->buf == dst->buf ||
        (src->mem == dst->mem && src->memoryOffset < dst->memoryOffset + dst->bytes &&
         dst->memoryOffset < src->memoryOffset + src->bytes) ||
        (g_batching && (g_batchCmd == VK_NULL_HANDLE || g_batchOps == std::numeric_limits<int>::max()))) return 2;
    // The source/destination capacity divisions prove these products fit.
    const size_t total = size_t(count) * rowBytes;
    VkBufferCopy* regions = (VkBufferCopy*)std::malloc(size_t(count) * sizeof(VkBufferCopy));
    if (!regions) return (int)VK_ERROR_OUT_OF_HOST_MEMORY;
    for (int i = 0; i < count; ++i) {
        regions[i] = VkBufferCopy{size_t(i) * rowBytes, dstOffset + size_t(i) * dstStride, rowBytes};
    }
    const bool recording = g_batching;
    VkCommandBuffer cmd = recording ? g_batchCmd : v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) {
        std::free(regions);
        return (int)g_submissionStatus;
    }
    // The source is the preceding GEMM output. Preserve a conservative
    // shader-write -> transfer-read dependency independent of hazard elision.
    recordComputeBarrier(cmd);
    vkCmdCopyBuffer(cmd, src->buf, dst->buf, uint32_t(count), regions);
    std::free(regions);
    dp_inc(g_dp.d2d);
    if (recording) {
        g_batchHasV41 = true;
        tsAfterDispatch(cmd, kTsD2DGather, 0, 0);
        ++g_batchOps;
        if (uint64_t(count) > std::numeric_limits<uint64_t>::max() - g_batchD2DCount ||
            total > std::numeric_limits<uint64_t>::max() - g_batchD2DBytes) {
            g_batchD2DValid = false;
        } else {
            g_batchD2DCount += uint64_t(count);
            g_batchD2DBytes += total;
        }
    } else {
        recordComputeBarrier(cmd);
        dpOneShot(g_dp.oneShotD2D);
        if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
        if (!checkedCounterAdd(g_d2dCount, uint64_t(count)) || !checkedCounterAdd(g_d2dBytes, total)) {
            g_transferCountersValid = false;
        }
    }
    return (int)g_submissionStatus;
}

// batch entry points (C ABI).
void fvk_batch_begin(void) { batchBegin(); }
void fvk_batch_flush(void) { dp_inc(g_dp.batchFlushes); batchFlush(); }
void fvk_batch_cycle(void) { dp_inc(g_dp.batchFlushes); batchCycle(); }
int fvk_batch_inflight(void) { return g_inflight.active ? 1 : 0; }
bool fvk_batch_active(void) { return g_batching; }
void fvk_batch_hazards_arm(int armed) { g_batchHazardsArmed = armed != 0 ? 1 : 0; }
int fvk_batch_hazards_armed(void) { return g_batchHazardsArmed; }
/* The ordinal the NEXT recorded dispatch will carry. Because a barrier is emitted before
 * op k against op k-1, that barrier's verdict key is k = the value returned here. A Go
 * seam declares its verdict under this key so the verdict can only ever elide the exact
 * dispatch it was computed for, regardless of interleaved undeclared dispatches. */
int fvk_batch_next_ordinal(void) { return g_batchOps; }
void fvk_batch_hazards_set(int ordinal, int needs_sync) {
    g_batchHazardSync[ordinal] = needs_sync != 0 ? 1 : 0;
}
void fvk_batch_hazards_reset(void) {
    g_batchHazardSync.clear();
}
uint64_t fvk_batch_barriers_elided(void) {
    return g_dp.barriersElided.load(std::memory_order_relaxed);
}
void fvk_retire_request(void) { batchFlush(); }
int fvk_submission_status(void) { return (int)g_submissionStatus; }
void fvk_submission_reset(void) {
    // A Vulkan failure is a context failure, not a per-request diagnostic.
    // Callers must retire the context instead of clearing an unconfirmed fence.
}
int fvk_batch_flush_status(void) {
    SyncScope syncScope(kSyncSiteFlushStatus);
    fvk_batch_flush();
    const int status = (int)g_submissionStatus;
    if (status == VK_SUCCESS && g_stepBatch) batchBegin();
    return (int)g_submissionStatus;
}
bool fvk_batch_foreign_active(void) { return g_batching && !g_stepBatch; }
int fvk_v41_step_batch_begin(void) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready || g_v41SubmissionPendingFailure || g_batching || g_stepBatch ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 1;
    g_stepBatch = true;
    g_batchHazardsArmed = 0;
    batchBegin();
    if (!g_batching) {
        g_stepBatch = false;
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 1;
    }
    return 0;
}
int fvk_v41_step_batch_end(void) {
    if (!g_stepBatch) return (int)g_submissionStatus;
    SyncScope syncScope(kSyncSiteStepEnd);
    batchFlush();
    g_stepBatch = false;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    return g_batching ? (int)VK_ERROR_UNKNOWN : 0;
}
int fvk_v41_step_batch_active(void) { return g_stepBatch ? 1 : 0; }
uint64_t fvk_h2d_bytes(void) { return g_h2dBytes.load(std::memory_order_relaxed); }
uint64_t fvk_direct_h2d_bytes(void) { return g_directH2DBytes.load(std::memory_order_relaxed); }
uint64_t fvk_d2h_bytes(void) { return g_d2hBytes.load(std::memory_order_relaxed); }
void fvk_device_work_counters(uint64_t* dispatches, uint64_t* d2h_bytes, uint64_t* submit_wait_ns) {
    if (dispatches) *dispatches = g_deviceDispatches.load(std::memory_order_relaxed);
    if (d2h_bytes) *d2h_bytes = g_d2hBytes.load(std::memory_order_relaxed);
    if (submit_wait_ns) *submit_wait_ns = g_submitWaitNs.load(std::memory_order_relaxed);
}
uint64_t fvk_device_idle_ns(void) { return syncIdleNs(submitWaitNow()); }
void fvk_sync_pipeline_counters(uint64_t* submits, uint64_t* drains) {
    const uint64_t d = g_syncPipeDrains.load(std::memory_order_acquire);
    if (drains) *drains = d;
    if (submits) *submits = g_syncPipeSubmits.load(std::memory_order_relaxed);
}
int fvk_sync_site_count(void) { return (int)kSyncSiteCount; }
int fvk_sync_stage_count(void) { return (int)kSyncStageCount; }
const char* fvk_sync_site_name(int site) {
    return site >= 0 && site < (int)kSyncSiteCount ? kSyncSiteNames[site] : "";
}
const char* fvk_sync_stage_name(int stage) {
    if (stage == (int)kSyncStageNone) return "none";
    return stage >= 0 && stage < (int)K_COUNT ? kKernelNames[stage] : "";
}
int fvk_sync_site_rows(uint32_t* site, uint32_t* stage, uint64_t* syncs, uint64_t* wait_ns,
                       uint64_t* d2h_bytes, int cap) {
    int n = 0;
    for (uint32_t s = 0; s < kSyncSiteCount; ++s) {
        for (uint32_t k = 0; k < kSyncStageCount; ++k) {
            const SyncCell& c = g_syncCells[s][k];
            const uint64_t count = c.syncs.load(std::memory_order_relaxed);
            if (count == 0) continue;
            if (n < cap && site && stage && syncs && wait_ns && d2h_bytes) {
                site[n] = s;
                stage[n] = k;
                syncs[n] = count;
                wait_ns[n] = c.waitNs.load(std::memory_order_relaxed);
                d2h_bytes[n] = c.d2hBytes.load(std::memory_order_relaxed);
            }
            ++n;
        }
    }
    return n;
}
int fvk_transfer_counters(uint64_t* h2d_count, uint64_t* h2d_bytes,
                           uint64_t* d2h_count, uint64_t* d2h_bytes,
                           uint64_t* d2d_count, uint64_t* d2d_bytes) {
    if (!g_ready || g_submissionStatus != VK_SUCCESS || !g_transferCountersValid ||
        !h2d_count || !h2d_bytes || !d2h_count || !d2h_bytes ||
        !d2d_count || !d2d_bytes) {
        return 0;
    }
    *h2d_count = g_h2dCount.load(std::memory_order_relaxed);
    *h2d_bytes = g_h2dBytes.load(std::memory_order_relaxed);
    *d2h_count = g_d2hCount.load(std::memory_order_relaxed);
    *d2h_bytes = g_d2hBytes.load(std::memory_order_relaxed);
    *d2d_count = g_d2dCount.load(std::memory_order_relaxed);
    *d2d_bytes = g_d2dBytes.load(std::memory_order_relaxed);
    return 1;
}

int fvk_device_allocation_snapshot(uint64_t* live_bytes) {
    if (!g_ready || g_submissionStatus != VK_SUCCESS ||
        !g_deviceAllocationAccountingValid || !live_bytes) {
        return 0;
    }
    *live_bytes = g_deviceAllocationLiveBytes;
    return 1;
}

static bool backendObservationQuiescent() {
    return !g_batching && !g_inflight.active && g_batchCmd == VK_NULL_HANDLE && g_batchOps == 0 &&
        g_batchSets.empty() && g_batchFreed.empty() &&
        g_batchD2DCount == 0 && g_batchD2DBytes == 0 && g_batchD2DValid &&
        !g_restore.stage && !g_restore.mapped && g_restore.cmd == VK_NULL_HANDLE &&
        g_restore.cap == 0 && g_restore.maxEntries == 0 && g_restore.used == 0 &&
        g_restore.entries == 0 && g_restore.payloadBytes == 0 &&
        g_restore.successfulSubmits == 0;
}

int fvk_device_allocation_window_begin(uint64_t* token) {
    if (!g_ready || g_submissionStatus != VK_SUCCESS ||
        !backendObservationQuiescent() || !g_deviceAllocationAccountingValid ||
        !g_transferCountersValid || g_allocationWindowActive || !token ||
        g_nextAllocationWindowToken == 0) {
        return 0;
    }
    g_allocationWindowActive = true;
    g_allocationWindowToken = g_nextAllocationWindowToken++;
    g_allocationWindowPeakBytes = g_deviceAllocationLiveBytes;
    *token = g_allocationWindowToken;
    return 1;
}

int fvk_device_allocation_window_end(uint64_t token, uint64_t* live_bytes,
                                     uint64_t* peak_bytes) {
    if (!g_allocationWindowActive || token == 0 || token != g_allocationWindowToken) {
        return 0;
    }
    g_allocationWindowActive = false;
    g_allocationWindowToken = 0;
    if (g_submissionStatus != VK_SUCCESS || !backendObservationQuiescent() ||
        !g_deviceAllocationAccountingValid || !g_transferCountersValid ||
        !live_bytes || !peak_bytes) {
        return 0;
    }
    *live_bytes = g_deviceAllocationLiveBytes;
    *peak_bytes = g_allocationWindowPeakBytes;
    return 1;
}
// ---- RADV/Vulkan performance-query phase observation ---------------------------
//
// Fail-closed: reports availability from the live device extension set only.
// Unsupported counters are never replaced with constants; callers must observe
// `*available == 0` and publish a typed-unavailable receipt.

// The KHR performance-query counters entry point is a device-level command the
// Vulkan loader does not export for direct linking; resolve it at runtime and
// cache the result. A missing entry point is fail-closed (null), never a stub
// that fabricates a reading.
#ifdef VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME
static PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR
    phaseCounterEnumerate() {
    static PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR fn = nullptr;
    static bool resolved = false;
    if (!resolved) {
        resolved = true;
        if (g_dev) {
            fn = (PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR)
                vkGetDeviceProcAddr(g_dev, "vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR");
        }
    }
    return fn;
}
#endif

int fvk_phase_performance_query_available(void) {
    if (!g_ready || g_submissionStatus != VK_SUCCESS) return 0;
#ifdef VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME
    if (!deviceExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)) return 0;
    return phaseCounterEnumerate() ? 1 : 0;
#else
    return 0;
#endif
}

int fvk_phase_counter_count(void) {
#ifdef VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME
    if (!g_ready || g_submissionStatus != VK_SUCCESS ||
        !deviceExtensionSupported(VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME)) {
        return -1;
    }
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR enumerate =
        phaseCounterEnumerate();
    if (!enumerate) return -1;
    uint32_t n = 0;
    if (enumerate(g_phys, 0, &n, nullptr, nullptr) != VK_SUCCESS) {
        return -1;
    }
    return (int)n;
#else
    return -1;
#endif
}

int fvk_phase_counter_describe(int index, char* name, size_t name_len,
                               char* unit, size_t unit_len, int* scope) {
#ifdef VK_KHR_PERFORMANCE_QUERY_EXTENSION_NAME
    if (index < 0 || !name || name_len == 0 || !unit || unit_len == 0 || !scope) {
        return 0;
    }
    PFN_vkEnumeratePhysicalDeviceQueueFamilyPerformanceQueryCountersKHR enumerate =
        phaseCounterEnumerate();
    if (!enumerate) return 0;
    uint32_t n = 0;
    if (!g_ready || g_submissionStatus != VK_SUCCESS ||
        enumerate(g_phys, 0, &n, nullptr, nullptr) != VK_SUCCESS || (uint32_t)index >= n) {
        return 0;
    }
    std::vector<VkPerformanceCounterKHR> counters(n, VkPerformanceCounterKHR{
        VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_KHR});
    std::vector<VkPerformanceCounterDescriptionKHR> descs(n, VkPerformanceCounterDescriptionKHR{
        VK_STRUCTURE_TYPE_PERFORMANCE_COUNTER_DESCRIPTION_KHR});
    if (enumerate(g_phys, 0, &n, counters.data(), descs.data()) != VK_SUCCESS) {
        return 0;
    }
    strncpy(name, descs[index].name, name_len - 1);
    name[name_len - 1] = 0;
    const char* unitName = "unknown";
    switch (counters[index].unit) {
        case VK_PERFORMANCE_COUNTER_UNIT_GENERIC_KHR: unitName = "generic"; break;
        case VK_PERFORMANCE_COUNTER_UNIT_PERCENTAGE_KHR: unitName = "percentage"; break;
        case VK_PERFORMANCE_COUNTER_UNIT_NANOSECONDS_KHR: unitName = "nanoseconds"; break;
        case VK_PERFORMANCE_COUNTER_UNIT_BYTES_KHR: unitName = "bytes"; break;
        case VK_PERFORMANCE_COUNTER_UNIT_BYTES_PER_SECOND_KHR: unitName = "bytes_per_second"; break;
        case VK_PERFORMANCE_COUNTER_UNIT_CYCLES_KHR: unitName = "cycles"; break;
        default: unitName = "unknown"; break;
    }
    strncpy(unit, unitName, unit_len - 1);
    unit[unit_len - 1] = 0;
    *scope = (int)counters[index].scope;
    return 1;
#else
    (void)index; (void)name; (void)name_len; (void)unit; (void)unit_len; (void)scope;
    return 0;
#endif
}

void fvk_sync(void) {
    if (g_v41SubmissionPendingFailure) return; drainInflight(); if (g_dev) vkDeviceWaitIdle(g_dev); }

int fvk_have_q8(void) { return g_have_q8; }
int fvk_have_qwen35_gdn_q8_in_proj(void) { return g_have_qwen35_gdn_q8_in_proj; }
int fvk_have_glm_kda_wave32(void) { return g_have_glm_kda_wave32; }
int fvk_have_cooperative_matrix(void) { return g_have_coopmat; }
int fvk_have_q6k_matmul(void) { return g_have_q6k_matmul; }
int fvk_have_q5k_matmul(void) { return g_have_q5k_matmul; }
int fvk_have_q3k_matmul(void) { return g_have_q3k_matmul; }
int fvk_have_qwen35_value_head_tile(void) { return g_have_qwen35_value_head_tile; }
uint32_t fvk_max_compute_work_group_count_x(void) { return g_maxComputeWorkGroupCountX; }
uint64_t fvk_max_buffer_bytes(void) { return (uint64_t)g_maxBufferBytes; }
uint64_t fvk_max_storage_buffer_range(void) { return (uint64_t)g_maxStorageBufferRange; }
uint64_t fvk_max_memory_allocation_size(void) { return (uint64_t)g_maxMemoryAllocationSize; }
uint64_t fvk_total_device_local_memory(void) { return (uint64_t)g_totalDeviceLocalMemory; }
int fvk_have_memory_budget(void) { return g_haveMemoryBudget ? 1 : 0; }

int fvk_device_local_memory_budget(uint64_t* budget, uint64_t* usage, uint64_t* free_bytes) {
    VkDeviceSize b = 0;
    VkDeviceSize u = 0;
    if (!queryDeviceLocalMemoryBudget(&b, &u)) return 0;
    if (budget) *budget = (uint64_t)b;
    if (usage) *usage = (uint64_t)u;
    if (free_bytes) *free_bytes = (uint64_t)(b > u ? b - u : 0);
    return 1;
}

uint32_t fvk_debug_buffer_props(const void* d) {
    if (!d) return 0;
    return (uint32_t)B((void*)d)->props;
}

int fvk_debug_buffer_is_host_visible(const void* d) {
    if (!d) return 0;
    return (B((void*)d)->props & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
}

int fvk_debug_buffer_is_device_local(const void* d) {
    if (!d) return 0;
    return (B((void*)d)->props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
}

int fvk_buffer_backing(const void* d, fvk_buffer_backing_info* out) {
    if (!out) return 0;
    *out = {};
    if (!g_ready || !g_dev || !d) return 0;
    const Buffer* b = B((void*)d);
    if (!b->buf || !b->mem || b->memoryTypeIndex >= g_memprops.memoryTypeCount) return 0;
    const VkMemoryType& type = g_memprops.memoryTypes[b->memoryTypeIndex];
    if (type.heapIndex >= g_memprops.memoryHeapCount) return 0;
    out->requested_property_flags = b->requestedProps;
    out->memory_type_index = b->memoryTypeIndex;
    out->property_flags = type.propertyFlags;
    out->heap_index = type.heapIndex;
    out->heap_flags = g_memprops.memoryHeaps[type.heapIndex].flags;
    out->host_visible_fallback = b->hostVisibleFallback ? 1 : 0;
    out->weight_arena_bound = b->weightArenaBound ? 1 : 0;
    return 1;
}

int fvk_buffer_reservation(const void* d, fvk_buffer_reservation_info* out) {
    if (!out) return 0;
    *out = {};
    fvk_buffer_backing_info backing{};
    if (!fvk_buffer_backing(d, &backing)) return 0;
    const Buffer* b = B((void*)d);
    VkDeviceSize reserved = b->allocationBytes;
    uint64_t id = b->allocationID;
    if (b->weightArenaBound) {
        if (b->weightArenaBlock >= g_weightArena.size()) return 0;
        const WeightArenaBlock& block = g_weightArena[b->weightArenaBlock];
        if (block.mem != b->mem || block.memoryTypeIndex != b->memoryTypeIndex ||
            block.liveBuffers == 0 || block.used > block.capacity ||
            b->memoryOffset > block.used || b->bytes > block.used - b->memoryOffset) return 0;
        reserved = block.capacity;
        id = block.allocationID;
    } else if (b->memoryOffset != 0) {
        return 0;
    }
    if (id == 0 || reserved == 0 || b->memoryOffset > reserved ||
        b->bytes > reserved - b->memoryOffset) return 0;
    out->buffer_bytes = static_cast<uint64_t>(b->bytes);
    out->reservation_bytes = static_cast<uint64_t>(reserved);
    out->allocation_id = id;
    out->binding_offset = static_cast<uint64_t>(b->memoryOffset);
    return 1;
}

// Read only authoritative retained containers. No global allocation registry or
// event log is needed; scan and duplicate checks are bounded even for large queues.
extern "C++" {
static void setRetainedCheckout(fvk_retained_release_record&, const Buffer*) {}
static void setRetainedCheckout(fvk_retained_release_record_v2& row, const Buffer* b) {
    row.native_checkout_generation = b->nativeCheckoutGeneration;
}
static bool retainedCheckoutMatches(const fvk_retained_release_record&, const Buffer*) { return true; }
static bool retainedCheckoutMatches(const fvk_retained_release_record_v2& row, const Buffer* b) {
    return row.native_checkout_generation == b->nativeCheckoutGeneration;
}
template <typename Record>
static int retainedReleaseSnapshot(Record* rows, uint32_t capacity,
                                  fvk_retained_release_summary* summary) {
    if (!summary) return FVK_RETAINED_INVALID;
    *summary = {};
    summary->device_generation = g_memoryDeviceGeneration;
    auto fail = [&](uint32_t status) {
        if (rows && capacity <= FVK_RETAINED_RELEASE_LIMIT) {
            for (uint32_t i = 0; i < capacity; ++i) rows[i] = {};
        }
        *summary = {};
        summary->device_generation = g_memoryDeviceGeneration;
        summary->status = status;
        return static_cast<int>(status);
    };
    if (capacity > FVK_RETAINED_RELEASE_LIMIT || (capacity && !rows))
        return fail(FVK_RETAINED_INVALID);
    for (uint32_t i = 0; i < capacity; ++i) rows[i] = {};
    if (!g_ready || !g_dev || !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration)
        return fail(FVK_RETAINED_UNAVAILABLE);
    uint32_t inspected = 0;
    uint32_t failure = 0;
    bool stopped = false;
    auto visit = [&]() {
        if (inspected == FVK_RETAINED_RELEASE_LIMIT) {
            summary->truncated = 1;
            stopped = true;
            return false;
        }
        ++inspected;
        return true;
    };
    auto observe = [&](Buffer* b, uint32_t container) {
        if (stopped || !visit()) return;
        if (!b) { failure = FVK_RETAINED_INVALID; stopped = true; return; }
        uint64_t generation = b->deviceGeneration;
        if (b->weightArenaBound) {
            if (b->weightArenaBlock >= g_weightArena.size()) {
                failure = FVK_RETAINED_INVALID; stopped = true; return;
            }
            generation = g_weightArena[b->weightArenaBlock].deviceGeneration;
        }
        if (generation != g_memoryDeviceGeneration || !generation) {
            failure = FVK_RETAINED_STALE_GENERATION; stopped = true; return;
        }
        fvk_buffer_reservation_info reservation{};
        if (!fvk_buffer_reservation(b, &reservation)) {
            failure = FVK_RETAINED_INVALID; stopped = true; return;
        }
        bool seenAllocation = false;
        for (uint32_t i = 0; i < summary->returned_records; ++i) {
            const auto& prior = rows[i];
            if (prior.reservation.allocation_id != reservation.allocation_id) continue;
            seenAllocation = true;
            if (prior.reservation.reservation_bytes != reservation.reservation_bytes) {
                failure = FVK_RETAINED_INVALID; stopped = true; return;
            }
            if (prior.reservation.binding_offset == reservation.binding_offset) {
                if (prior.reservation.buffer_bytes != reservation.buffer_bytes || prior.container != container ||
                    !retainedCheckoutMatches(prior, b)) {
                    failure = FVK_RETAINED_INVALID; stopped = true;
                }
                return; // exact duplicate binding in the same authoritative container
            }
        }
        if (summary->returned_records == capacity) {
            summary->truncated = 1;
            stopped = true;
            return;
        }
        if (!seenAllocation) {
            if (reservation.reservation_bytes > std::numeric_limits<uint64_t>::max() -
                summary->returned_unique_reservation_bytes) {
                failure = FVK_RETAINED_INVALID; stopped = true; return;
            }
            summary->returned_unique_reservation_bytes += reservation.reservation_bytes;
            ++summary->returned_unique_allocations;
        }
        auto& row = rows[summary->returned_records++];
        row.reservation = reservation;
        row.device_generation = generation;
        setRetainedCheckout(row, b);
        row.release_state = FVK_RELEASE_REQUESTED;
        row.container = container;
        row.backing_state = g_v41SubmissionPendingFailure ? FVK_BACKING_QUARANTINED :
            (container == FVK_RETAINED_DEFERRED ? FVK_BACKING_DEFERRED : FVK_BACKING_POOL_RETAINED);
    };
    for (Buffer* b : g_batchFreed) { observe(b, FVK_RETAINED_DEFERRED); if (stopped) break; }
    if (!stopped) for (const auto& bucket : g_pool) {
        if (!visit()) break; // Empty historical buckets must also consume work budget.
        for (Buffer* b : bucket.second) { observe(b, FVK_RETAINED_DEVICE_POOL); if (stopped) break; }
        if (stopped) break;
    }
    if (!stopped) for (const auto& bucket : g_hvOffloadPool) {
        if (!visit()) break;
        for (Buffer* b : bucket.second) { observe(b, FVK_RETAINED_OFFLOAD_POOL); if (stopped) break; }
        if (stopped) break;
    }
    if (failure) return fail(failure);
    summary->status = FVK_RETAINED_OK;
    return FVK_RETAINED_OK;
}
} // extern "C++": preserve legacy C linkage while sharing bounded enumeration.

int fvk_retained_release_snapshot(fvk_retained_release_record* rows, uint32_t capacity,
                                  fvk_retained_release_summary* summary) {
    return retainedReleaseSnapshot(rows, capacity, summary);
}

int fvk_retained_release_snapshot_v2(fvk_retained_release_record_v2* rows, uint32_t capacity,
                                     fvk_retained_release_summary* summary) {
    return retainedReleaseSnapshot(rows, capacity, summary);
}

int fvk_transfer_stage_backing(uint64_t* buffer_bytes, fvk_buffer_backing_info* out) {
    if (buffer_bytes) *buffer_bytes = 0;
    if (out) *out = {};
    if (!buffer_bytes || !out || !g_ready || !g_dev) return 0;
    if (!g_stage) return !g_stageMapped && g_stageCap == 0 ? 1 : 0;
    if (!g_stageMapped || g_stageCap == 0 || g_stage->bytes != g_stageCap) return 0;

    // Copy only after complete owner/backing validation. The caller's vulkanMu
    // keeps stagingBuffer from retiring the private handle during this read.
    fvk_buffer_backing_info backing{};
    if (!fvk_buffer_backing(g_stage, &backing)) return 0;
    *buffer_bytes = static_cast<uint64_t>(g_stageCap);
    *out = backing;
    return 1;
}

int fvk_transfer_stage_reservation(fvk_buffer_reservation_info* out, fvk_buffer_backing_info* backing) {
    if (out) *out = {};
    if (backing) *backing = {};
    if (!out || !backing) return 0;

    uint64_t buffer_bytes = 0;
    fvk_buffer_backing_info local_backing{};
    if (!fvk_transfer_stage_backing(&buffer_bytes, &local_backing)) return 0;
    if (buffer_bytes == 0) return 1;

    // The same owner lock covers both reads. Publish neither record until the
    // existing owner and reservation observers agree on the retained length.
    fvk_buffer_reservation_info reservation{};
    if (!fvk_buffer_reservation(g_stage, &reservation) ||
        reservation.buffer_bytes != buffer_bytes) return 0;
    *out = reservation;
    *backing = local_backing;
    return 1;
}

void fvk_trim_pool(void) {
    if (g_v41SubmissionPendingFailure) return;
    if (!g_dev) return;
    if (g_batching) batchFlush();
    drainPool();
    freeGdnScratch();
    freePartialRoPECaches();
    releaseWeightArena();
}

void fvk_trim_pool_if_over(size_t max_buffers) {
    if (g_v41SubmissionPendingFailure) return;
    if (!g_dev) return;
    if (g_poolCount > max_buffers) {
        if (g_batching) batchFlush();
        drainPool();
    }
}

// matmulDecodeGroups returns the workgroup count of matmul.comp's one-row chunk
// split (push P = -1), or 0 when the shape needs the staged one-group-per-row path.
static uint32_t matmulDecodeGroups(int out, int in) {
    if (out <= 0 || in <= 0) return 0;
    const uint64_t chunks = (uint64_t(in) + 1023u) / 1024u;
    if (chunks > 256u) return 0;
    const uint64_t perGroup = 256u / chunks;
    const uint64_t groups = (uint64_t(out) + perGroup - 1u) / perGroup;
    return groups > g_maxComputeWorkGroupCountX ? 0 : uint32_t(groups);
}

void fvk_matmul_f32(const void* dW, const void* dX, void* dY, int out, int in, int P) {
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
    if (P == 1) {
        if (const uint32_t groups = matmulDecodeGroups(out, in)) {
            pc.P = -1;
            dispatch(g_kern[K_MATMUL], bufs, &pc, sizeof(pc), groups);
            return;
        }
    }
    dispatch(g_kern[K_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_q8_matmul_2d_f32(const void* dWcodes, const void* dWscale, const void* dX, void* dY,
                          int out, int in, int P, uint32_t gridX, uint32_t gridY) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: q8_matmul requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[4] = {B((void*)dWcodes), B((void*)dWscale), B((void*)dX), B(dY)};
    Kernel& kernel = (g_have_coopmat && g_kern[K_Q8_MATMUL].pipe != VK_NULL_HANDLE)
        ? g_kern[K_Q8_MATMUL]
        : g_kern[K_Q8_MATMUL_DECODE];
    dispatch(kernel, bufs, &pc, sizeof(pc), gridX, gridY);
}

void fvk_q8_matmul_f32(const void* dWcodes, const void* dWscale, const void* dX, void* dY,
                       int out, int in, int P) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: q8_matmul requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[4] = {B((void*)dWcodes), B((void*)dWscale), B((void*)dX), B(dY)};
    if (g_have_coopmat && g_kern[K_Q8_MATMUL].pipe != VK_NULL_HANDLE && P > 1) {
        uint32_t tileN = 32u;
        uint32_t tileM = 32u;
        uint32_t gridX = ((uint32_t)out + tileN - 1u) / tileN;
        uint32_t gridY = ((uint32_t)P + tileM - 1u) / tileM;
        dispatch(g_kern[K_Q8_MATMUL], bufs, &pc, sizeof(pc), gridX, gridY);
    } else {
        uint32_t outputsPerGroup = 8u;
        uint32_t outGroups = ((uint32_t)out + outputsPerGroup - 1u) / outputsPerGroup;
        Kernel& kernel = g_kern[K_Q8_MATMUL_DECODE];
        dispatch(kernel, bufs, &pc, sizeof(pc), outGroups, (uint32_t)P);
    }
}

void fvk_q8_matmul2_f32(const void* dW0codes, const void* dW0scale,
                        const void* dW1codes, const void* dW1scale,
                        const void* dX, void* dY0, void* dY1,
                        int out0, int out1, int in, int P) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: q8_matmul2 requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int out0, out1, inDim, P; } pc{out0, out1, in, P};
    Buffer* bufs[7] = {
        B((void*)dW0codes), B((void*)dW0scale),
        B((void*)dW1codes), B((void*)dW1scale),
        B((void*)dX), B(dY0), B(dY1),
    };
    uint32_t totalOut = (uint32_t)(out0 + out1);
    uint32_t outGroups = (totalOut + 255u) / 256u;
    dispatch(g_kern[K_Q8_MATMUL2], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

void fvk_q8_matmul3_f32(const void* dW0codes, const void* dW0scale,
                        const void* dW1codes, const void* dW1scale,
                        const void* dW2codes, const void* dW2scale,
                        const void* dX, void* dY0, void* dY1, void* dY2,
                        int out0, int out1, int out2, int in, int P) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: q8_matmul3 requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int out0, out1, out2, inDim, P; } pc{out0, out1, out2, in, P};
    Buffer* bufs[10] = {
        B((void*)dW0codes), B((void*)dW0scale),
        B((void*)dW1codes), B((void*)dW1scale),
        B((void*)dW2codes), B((void*)dW2scale),
        B((void*)dX), B(dY0), B(dY1), B(dY2),
    };
    uint32_t totalOut = (uint32_t)(out0 + out1 + out2);
    uint32_t outGroups = (totalOut + 255u) / 256u;
    dispatch(g_kern[K_Q8_MATMUL3], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

int fvk_qwen35_gdn_q8_in_proj_f32(
    const void* dW0codes, const void* dW0scale,
    const void* dW1codes, const void* dW1scale,
    const void* dW2codes, const void* dW2scale,
    const void* dW3codes, const void* dW3scale,
    const void* dX, void* dY0, void* dY1, void* dY2, void* dY3,
    int out0, int out1, int out2, int out3, int in) {
    if (!g_ready) return 1;
    if (!g_have_qwen35_gdn_q8_in_proj ||
        g_kern[K_QWEN35_GDN_Q8_IN_PROJ].pipe == VK_NULL_HANDLE) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (out0 <= 0 || out1 <= 0 || out2 <= 0 || out3 <= 0 || in <= 0 || (in & 31) != 0)
        return 2;

    uint64_t in64 = (uint64_t)in;
    uint64_t blocks = in64 / 32u;
    uint64_t outs[4] = {(uint64_t)out0, (uint64_t)out1, (uint64_t)out2, (uint64_t)out3};
    const void* codes[4] = {dW0codes, dW1codes, dW2codes, dW3codes};
    const void* scales[4] = {dW0scale, dW1scale, dW2scale, dW3scale};
    void* outputs[4] = {dY0, dY1, dY2, dY3};
    if (!dX || in64 * sizeof(float) > B((void*)dX)->bytes) return 2;
    uint64_t totalOut = 0;
    for (int i = 0; i < 4; ++i) {
        if (!codes[i] || !scales[i] || !outputs[i]) return 2;
        uint64_t codeBytes = outs[i] * in64;
        uint64_t scaleBytes = outs[i] * blocks * sizeof(float);
        uint64_t outputBytes = outs[i] * sizeof(float);
        if (codeBytes > B((void*)codes[i])->bytes ||
            scaleBytes > B((void*)scales[i])->bytes ||
            outputBytes > B(outputs[i])->bytes) return 2;
        totalOut += outs[i];
    }
    if (totalOut == 0 || totalOut > UINT32_MAX - 7u) return 2;

    struct { int out0, out1, out2, out3, inDim; } pc{out0, out1, out2, out3, in};
    Buffer* bufs[13] = {
        B((void*)dW0codes), B((void*)dW0scale),
        B((void*)dW1codes), B((void*)dW1scale),
        B((void*)dW2codes), B((void*)dW2scale),
        B((void*)dW3codes), B((void*)dW3scale),
        B((void*)dX), B(dY0), B(dY1), B(dY2), B(dY3),
    };
    dispatch(g_kern[K_QWEN35_GDN_Q8_IN_PROJ], bufs, &pc, sizeof(pc),
             (uint32_t)((totalOut + 7u) / 8u));
    return (int)g_submissionStatus;
}

void fvk_rmsnorm_q8_matmul2_f32(const void* dW0codes, const void* dW0scale,
                                const void* dW1codes, const void* dW1scale,
                                const void* dX, const void* dNorm, void* dY0, void* dY1,
                                int out0, int out1, int in, int P, float eps) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: rmsnorm_q8_matmul2 requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int out0, out1, inDim, P; float eps; } pc{out0, out1, in, P, eps};
    Buffer* bufs[8] = {
        B((void*)dW0codes), B((void*)dW0scale),
        B((void*)dW1codes), B((void*)dW1scale),
        B((void*)dX), B((void*)dNorm), B(dY0), B(dY1),
    };
    uint32_t totalOut = (uint32_t)(out0 + out1);
    if (g_have_rmsnorm_q8_matmul2_coop && (totalOut + 7u) / 8u <= g_maxComputeWorkGroupCountX && P <= 65535) {
        dispatch(g_kern[K_RMSNORM_Q8_MATMUL2_COOP], bufs, &pc, sizeof(pc), (totalOut + 7u) / 8u, (uint32_t)P);
        return;
    }
    uint32_t outGroups = (totalOut + 255u) / 256u;
    dispatch(g_kern[K_RMSNORM_Q8_MATMUL2], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

void fvk_rmsnorm_q8_matmul3_f32(const void* dWqcodes, const void* dWqscale,
                                const void* dWkcodes, const void* dWkscale,
                                const void* dWvcodes, const void* dWvscale,
                                const void* dX, const void* dNorm, void* dQ, void* dK, void* dV,
                                int qOut, int kOut, int vOut, int in, int P, float eps) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: rmsnorm_q8_matmul3 requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int qOut, kOut, vOut, inDim, P; float eps; } pc{qOut, kOut, vOut, in, P, eps};
    Buffer* bufs[11] = {
        B((void*)dWqcodes), B((void*)dWqscale),
        B((void*)dWkcodes), B((void*)dWkscale),
        B((void*)dWvcodes), B((void*)dWvscale),
        B((void*)dX), B((void*)dNorm), B(dQ), B(dK), B(dV),
    };
    uint32_t totalOut = (uint32_t)(qOut + kOut + vOut);
    uint32_t outGroups = (totalOut + 255u) / 256u;
    dispatch(g_kern[K_RMSNORM_Q8_MATMUL3], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

void fvk_swiglu_q8_matmul_add_f32(const void* dWcodes, const void* dWscale,
                                  const void* dG, const void* dU, void* dD,
                                  int out, int in, int P) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: swiglu_q8_matmul_add requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int outDim, inDim, P, outGroupBase, tokenBase; } pc{out, in, P, 0, 0};
    Buffer* bufs[5] = {
        B((void*)dWcodes), B((void*)dWscale), B((void*)dG), B((void*)dU), B(dD),
    };
    // Cooperative two-dimensional grid: eight outputs per 256-thread workgroup on X and one
    // token row per workgroup on Y. Both dimensions are bounded by their device limits; the
    // static outGroupBase/tokenBase push constants (0 here) keep large-output/large-token
    // shapes expressible without flattening onto a single >65,535 dimension.
    uint32_t outputsPerGroup = 8u;
    uint32_t outGroups = ((uint32_t)out + outputsPerGroup - 1u) / outputsPerGroup;
    dispatch(g_kern[K_SWIGLU_Q8_MATMUL_ADD], bufs, &pc, sizeof(pc), outGroups, (uint32_t)P);
}

void fvk_rmsnorm_q4k_matmul2_f32(const void* dW0, const void* dW1,
                                 const void* dX, const void* dNorm,
                                 void* dY0, void* dY1,
                                 int out0, int out1, int in, int P, float eps) {
    struct { int out0, out1, inDim, P; float eps; } pc{out0, out1, in, P, eps};
    Buffer* bufs[6] = {
        B((void*)dW0), B((void*)dW1),
        B((void*)dX), B((void*)dNorm),
        B(dY0), B(dY1),
    };
    uint32_t totalOut = (uint32_t)(out0 + out1);
    uint32_t outGroups = (totalOut + 255u) / 256u;
    dispatch(g_kern[K_RMSNORM_Q4K_MATMUL2], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

void fvk_swiglu_q4k_matmul_add_f32(const void* dW, const void* dG, const void* dU,
                                   void* dD, int out, int in, int P) {
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[4] = {
        B((void*)dW), B((void*)dG), B((void*)dU), B(dD),
    };
    uint32_t outGroups = ((uint32_t)out + 255u) / 256u;
    dispatch(g_kern[K_SWIGLU_Q4K_MATMUL_ADD], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
}

int fvk_matmul_argmax_f32(const void* dW, const void* dX, int out, int in) {
    Buffer* idx = (Buffer*)fvk_malloc(sizeof(int));
    if (!idx) {
        fprintf(stderr, "fak-vulkan: matmul_argmax idx allocation failed\n");
        abort();
    }
    if (out <= 256) {
        struct { int outDim, inDim; } pc{out, in};
        Buffer* bufs[3] = {B((void*)dW), B((void*)dX), idx};
        dispatch(g_kern[K_MATMUL_ARGMAX], bufs, &pc, sizeof(pc), 1);
    } else {
        int blocks = (out + 255) / 256;
        Buffer* vals = (Buffer*)fvk_malloc((size_t)blocks * sizeof(float));
        Buffer* inds = (Buffer*)fvk_malloc((size_t)blocks * sizeof(int));
        if (!vals || !inds) {
            fprintf(stderr, "fak-vulkan: matmul_argmax partial allocation failed (%d blocks)\n", blocks);
            abort();
        }
        struct { int outDim, inDim; } pc0{out, in};
        Buffer* bufs0[4] = {B((void*)dW), B((void*)dX), vals, inds};
        dispatch(g_kern[K_MATMUL_ARGMAX_BLOCKS], bufs0, &pc0, sizeof(pc0), (uint32_t)blocks);
        struct { int n; } pc1{blocks};
        Buffer* bufs1[3] = {vals, inds, idx};
        dispatch(g_kern[K_ARGMAX_PAIRS], bufs1, &pc1, sizeof(pc1), 1);
        if (g_batching) batchFlush();
        fvk_free(vals);
        fvk_free(inds);
    }
    if (g_batching) batchFlush();
    int h = 0;
    copyDeviceToHost(&h, idx, sizeof(int));
    fvk_free(idx);
    return h;
}

int fvk_rmsnorm_matmul_argmax_f32(const void* dW, const void* dX, const void* dNorm,
                                  int out, int in, float eps) {
    Buffer* idx = (Buffer*)fvk_malloc(sizeof(int));
    if (!idx) {
        fprintf(stderr, "fak-vulkan: rmsnorm_matmul_argmax idx allocation failed\n");
        abort();
    }
    int blocks = (out + 255) / 256;
    Buffer* vals = (Buffer*)fvk_malloc((size_t)blocks * sizeof(float));
    Buffer* inds = (Buffer*)fvk_malloc((size_t)blocks * sizeof(int));
    if (!vals || !inds) {
        fprintf(stderr, "fak-vulkan: rmsnorm_matmul_argmax partial allocation failed (%d blocks)\n", blocks);
        abort();
    }
    struct { int outDim, inDim; float eps; } pc0{out, in, eps};
    Buffer* bufs0[5] = {B((void*)dW), B((void*)dX), B((void*)dNorm), vals, inds};
    dispatch(g_kern[K_RMSNORM_MATMUL_ARGMAX_BLOCKS], bufs0, &pc0, sizeof(pc0), (uint32_t)blocks);
    struct { int n; } pc1{blocks};
    Buffer* bufs1[3] = {vals, inds, idx};
    dispatch(g_kern[K_ARGMAX_PAIRS], bufs1, &pc1, sizeof(pc1), 1);
    if (g_batching) batchFlush();
    fvk_free(vals);
    fvk_free(inds);
    int h = 0;
    copyDeviceToHost(&h, idx, sizeof(int));
    fvk_free(idx);
    return h;
}

void fvk_matmul_add_f32(const void* dW, const void* dX, void* dY, int out, int in, int P) {
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
    dispatch(g_kern[K_MATMUL_ADD], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_matmul2_f32(const void* dW0, const void* dW1, const void* dX,
                     void* dY0, void* dY1, int out0, int out1, int in, int P) {
    struct { int out0, out1, inDim, P; } pc{out0, out1, in, P};
    Buffer* bufs[5] = {B((void*)dW0), B((void*)dW1), B((void*)dX), B(dY0), B(dY1)};
    dispatch(g_kern[K_MATMUL2], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_matmul3_f32(const void* dWq, const void* dWk, const void* dWv, const void* dX,
                     void* dQ, void* dK, void* dV, int qOut, int kOut, int vOut, int in, int P) {
    struct { int qOut, kOut, vOut, inDim, P; } pc{qOut, kOut, vOut, in, P};
    Buffer* bufs[7] = {B((void*)dWq), B((void*)dWk), B((void*)dWv), B((void*)dX), B(dQ), B(dK), B(dV)};
    dispatch(g_kern[K_MATMUL3], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_rmsnorm_f32(const void* dX, const void* dW, void* dY, int rows, int n, float eps) {
    struct { int rows, n; float eps; } pc{rows, n, eps};
    Buffer* bufs[3] = {B((void*)dX), B((void*)dW), B(dY)};
    dispatch(g_kern[K_RMSNORM], bufs, &pc, sizeof(pc), (uint32_t)rows);
}

void fvk_rmsnorm_matmul_f32(const void* dW, const void* dX, const void* dNorm,
                            void* dY, int out, int in, int P, float eps) {
    struct { int outDim, inDim, P; float eps; } pc{out, in, P, eps};
    Buffer* bufs[4] = {B((void*)dW), B((void*)dX), B((void*)dNorm), B(dY)};
    dispatch(g_kern[K_RMSNORM_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_rmsnorm_matmul2_f32(const void* dW0, const void* dW1, const void* dX, const void* dNorm,
                             void* dY0, void* dY1, int out0, int out1, int in, int P, float eps) {
    struct { int out0, out1, inDim, P; float eps; } pc{out0, out1, in, P, eps};
    Buffer* bufs[6] = {B((void*)dW0), B((void*)dW1), B((void*)dX), B((void*)dNorm), B(dY0), B(dY1)};
    dispatch(g_kern[K_RMSNORM_MATMUL2], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_rmsnorm_matmul3_f32(const void* dWq, const void* dWk, const void* dWv,
                             const void* dX, const void* dNorm, void* dQ, void* dK, void* dV,
                             int qOut, int kOut, int vOut, int in, int P, float eps) {
    struct { int qOut, kOut, vOut, inDim, P; float eps; } pc{qOut, kOut, vOut, in, P, eps};
    Buffer* bufs[8] = {B((void*)dWq), B((void*)dWk), B((void*)dWv), B((void*)dX), B((void*)dNorm), B(dQ), B(dK), B(dV)};
    dispatch(g_kern[K_RMSNORM_MATMUL3], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_rope_f32(void* dX, int pos, int nHeads, int headDim, double theta) {
    struct { int pos, nHeads, headDim; float theta; } pc{pos, nHeads, headDim, (float)theta};
    Buffer* bufs[1] = {B(dX)};
    uint32_t total = (uint32_t)nHeads * (uint32_t)(headDim / 2);
    dispatch(g_kern[K_ROPE], bufs, &pc, sizeof(pc), (total + 127) / 128);
}

#ifdef FAK_V41_TAIL_ROPE_EXACT_WITNESS
int fvk_v41_tail_rope_exact_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41TailRoPEABI(code, true) ? 1 : 0;
}

int fvk_v41_tail_rope_exact_available(void) {
    return g_ready && g_v41IndexerFloatControls && g_v41TailRoPEExactModule &&
        g_have_v41_tail_rope_qk && g_kern[K_V41_TAIL_ROPE_QK].pipe != VK_NULL_HANDLE;
}
#endif

int fvk_have_v41_tail_rope_qk(void) {
    return g_ready && g_have_v41_tail_rope_qk && g_kern[K_V41_TAIL_ROPE_QK].pipe != VK_NULL_HANDLE;
}

int fvk_device_type(void) {
    if (!g_ready || g_phys == VK_NULL_HANDLE) return int(VK_PHYSICAL_DEVICE_TYPE_OTHER);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(g_phys, &properties);
    return int(properties.deviceType);
}

int fvk_have_v41_tail_rope_exact(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_tail_rope_exact &&
        g_kern[K_V41_TAIL_ROPE_EXACT].pipe != VK_NULL_HANDLE;
}

int fvk_v41_tail_rope_strict_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41TailRoPEABI(code, true) ? 1 : 0;
}

// Both tail modules share one ABI; the caller has proven its kernel available.
static int v41TailRoPEDispatch(Kernel& kernel, const void* dQ, const void* dKV,
                               void* dQOut, void* dKVOut, const void* dSinCos,
                               int heads, int headDim, int rotaryDim) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || (rotaryDim & 1)) return 2;
    const uint64_t qCount = uint64_t(heads) * uint64_t(headDim);
    const uint64_t total = qCount + uint64_t(headDim);
    const uint64_t groups = (total + 255u) / 256u;
    if (total > UINT32_MAX || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[5] = {B(dQ), B(dKV), B(dQOut), B(dKVOut), B(dSinCos)};
    const uint64_t counts[5] = {qCount, uint64_t(headDim), qCount, uint64_t(headDim), uint64_t(rotaryDim)};
    for (int i = 0; i < 5; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(float) ||
            counts[i] * sizeof(float) > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int out = 2; out <= 3; ++out) {
        for (int i = 0; i < 5; ++i) if (i != out && aliases(bufs[out], bufs[i])) return 2;
    }
    struct { int heads, headDim, rotaryDim; } pc{heads, headDim, rotaryDim};
    static_assert(sizeof(pc) == 12, "V4.1 tail RoPE push ABI must be three int32 values");
    if (!dispatch(kernel, bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_v41_tail_rope_qk_f32(const void* dQ, const void* dKV,
                            void* dQOut, void* dKVOut, const void* dSinCos,
                            int heads, int headDim, int rotaryDim) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_tail_rope_qk()) return 3;
    return v41TailRoPEDispatch(g_kern[K_V41_TAIL_ROPE_QK], dQ, dKV, dQOut, dKVOut, dSinCos, heads, headDim, rotaryDim);
}

int fvk_v41_tail_rope_exact_f32(const void* dQ, const void* dKV,
                               void* dQOut, void* dKVOut, const void* dSinCos,
                               int heads, int headDim, int rotaryDim) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_tail_rope_exact()) return 3;
    return v41TailRoPEDispatch(g_kern[K_V41_TAIL_ROPE_EXACT], dQ, dKV, dQOut, dKVOut, dSinCos, heads, headDim, rotaryDim);
}

int fvk_have_v41_shared_attention(void) {
    return g_ready && g_have_v41_shared_attention && g_kern[K_V41_SHARED_ATTENTION].pipe != VK_NULL_HANDLE;
}

int fvk_have_v41_shared_attention_fast(void) {
    return g_ready && g_have_v41_shared_attention_fast && g_kern[K_V41_SHARED_ATTENTION_FAST].pipe != VK_NULL_HANDLE;
}

int fvk_have_v41_shared_attention_split(void) {
    return g_ready && g_have_v41_shared_attention_split && g_kern[K_V41_SHARED_ATTENTION_SPLIT].pipe != VK_NULL_HANDLE;
}

int fvk_v41_shared_attention_class(void) { return g_v41SharedAttentionClass; }

int fvk_v41_shared_attention_set_class(int cls) {
    if (!g_ready) return 1;
    if (cls != 0 && cls != 1 && cls != 2) return 2;
    if (cls == 1 && !fvk_have_v41_shared_attention_fast()) return 3;
    if (cls == 2 && !fvk_have_v41_shared_attention_split()) return 3;
    g_v41SharedAttentionClass = cls;
    return 0;
}

// Split-k class: a split pass over (row tiles x head groups) writes per-(head, tile)
// softmax partials into shim scratch, then a per-head combine pass merges them in
// tile order and writes out and status. Returns false when the shape is outside
// the module's envelope or a dispatch fails; the caller then runs the run-to-run
// class, which overwrites out and status. Neither pass is declared to the Go hazard
// ledger, so both keep the coarse barrier and any following elision fails closed.
static bool v41SharedAttentionSplit(Buffer** bufs, int heads, int headDim, int selectedRows,
                                    int mode, int hasSink, float scale) {
    constexpr int kTile = 16, kHeadGroup = 64, kPart = 512 + 4, kMaxRows = 8192, kMinRows = 32;
    if (!fvk_have_v41_shared_attention_split() || headDim != 512 ||
        selectedRows < kMinRows || selectedRows > kMaxRows) return false;
    const uint32_t blocks = uint32_t((selectedRows + kTile - 1) / kTile);
    const uint32_t groups = uint32_t((heads + kHeadGroup - 1) / kHeadGroup);
    const uint64_t partBytes = uint64_t(heads) * blocks * kPart * sizeof(float);
    if (blocks > g_maxComputeWorkGroupCountX || partBytes > (uint64_t(1) << 31)) return false;
    if (!g_v41SplitPartials || g_v41SplitPartials->bytes < partBytes) {
        Buffer* grown = (Buffer*)fvk_malloc(size_t(partBytes));
        if (!grown) return false;
        if (g_v41SplitPartials) fvk_free(g_v41SplitPartials);
        g_v41SplitPartials = grown;
    }
    Buffer* split[6] = {bufs[0], bufs[1], bufs[2], bufs[3], bufs[4], g_v41SplitPartials};
    struct Push {
        int32_t heads, headDim, selectedRows, mode, hasSink;
        float scale;
        int32_t pass;
    } pc{heads, headDim, selectedRows, mode, hasSink, scale, 0};
    static_assert(sizeof(Push) == 28, "V4.1 split attention push ABI must be 28 bytes");
    if (!dispatch(g_kern[K_V41_SHARED_ATTENTION_SPLIT], split, &pc, sizeof(pc), blocks, groups)) return false;
    pc.pass = 1;
    return dispatch(g_kern[K_V41_SHARED_ATTENTION_SPLIT], split, &pc, sizeof(pc), uint32_t(heads));
}

int fvk_v41_shared_attention_f32(const void* dQ, const void* dSharedKV,
                                 const void* dSink, void* dOut, void* dStatus,
                                 int heads, int headDim, int selectedRows,
                                 int mode, int hasSink, float scale) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_shared_attention()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (heads <= 0 || headDim <= 0 || selectedRows <= 0 ||
        (mode != 0 && mode != 1) || (hasSink != 0 && hasSink != 1) ||
        !std::isfinite(scale) || scale == 0.0f) return 2;
    const uint64_t qCount = uint64_t(heads) * uint64_t(headDim);
    const uint64_t kvCount = uint64_t(selectedRows) * uint64_t(headDim);
    const uint64_t statusWords = uint64_t(heads) * 4u;
    if (qCount > INT32_MAX || kvCount > INT32_MAX || statusWords > UINT32_MAX ||
        uint64_t(heads) > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[5] = {B(dQ), B(dSharedKV), B(dSink), B(dOut), B(dStatus)};
    const uint64_t counts[5] = {qCount, kvCount, hasSink ? uint64_t(heads) : 1u, qCount, statusWords};
    for (int i = 0; i < 5; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
            counts[i] * sizeof(uint32_t) != bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int out = 3; out <= 4; ++out) {
        for (int i = 0; i < 5; ++i) if (i != out && aliases(bufs[out], bufs[i])) return 2;
    }
    struct Push {
        int32_t heads, headDim, selectedRows, mode, hasSink;
        float scale;
    } pc{heads, headDim, selectedRows, mode, hasSink, scale};
    static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Push) == 24,
                  "V4.1 attention push ABI must be five int32 values and one float32");
    static_assert(offsetof(Push, heads) == 0 && offsetof(Push, headDim) == 4 &&
                  offsetof(Push, selectedRows) == 8 && offsetof(Push, mode) == 12 &&
                  offsetof(Push, hasSink) == 16 && offsetof(Push, scale) == 20,
                  "V4.1 attention push ABI member offsets must be fixed");
    // One 128-lane workgroup per head; its lane 0 initializes the four status
    // words and publishes the ordered first fault. There is no atomic winner race.
    // The run-to-run class holds Q in 16 lane registers, so wider heads stay serial.
    if (g_v41SharedAttentionClass == 2 && v41SharedAttentionSplit(bufs, heads, headDim, selectedRows, mode, hasSink, scale)) {
        return (int)g_submissionStatus;
    }
    // Split-k shapes outside its envelope, or a failed split dispatch, take the
    // run-to-run class: the split pass writes only shim scratch, never out or status.
    const bool fast = g_v41SharedAttentionClass != 0 && fvk_have_v41_shared_attention_fast() && headDim <= 512;
    if (!dispatch(g_kern[fast ? K_V41_SHARED_ATTENTION_FAST : K_V41_SHARED_ATTENTION], bufs, &pc, sizeof(pc), uint32_t(heads))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_router_topk(void) {
    return g_ready && g_have_v41_router_topk && g_kern[K_V41_ROUTER_TOPK].pipe != VK_NULL_HANDLE;
}

int fvk_v41_router_topk_f32(const void* dLogits, const void* dBias,
                            void* dExperts, void* dWeights, void* dStatus,
                            int rows, int experts, int topK,
                            int full, int hasBias, float routeScale) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_router_topk()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || experts <= 0 || topK <= 0 || topK > experts || topK > 8 ||
        (full != 0 && full != 1) || (hasBias != 0 && hasBias != 1) ||
        !std::isfinite(routeScale) || !(routeScale > 0.0f)) return 2;
    const uint64_t logitCount = uint64_t(rows) * uint64_t(experts);
    const uint64_t pickCount = uint64_t(rows) * uint64_t(topK);
    const uint64_t statusWords = uint64_t(rows) * 4u;
    if (logitCount > INT32_MAX || pickCount > INT32_MAX || statusWords > UINT32_MAX ||
        uint64_t(rows) > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[5] = {B(dLogits), B(dBias), B(dExperts), B(dWeights), B(dStatus)};
    const uint64_t counts[5] = {logitCount, hasBias ? uint64_t(experts) : 1u, pickCount, pickCount, statusWords};
    for (int i = 0; i < 5; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
            counts[i] * sizeof(uint32_t) != bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int out = 2; out <= 4; ++out) {
        for (int i = 0; i < 5; ++i) if (i != out && aliases(bufs[out], bufs[i])) return 2;
    }
    struct Push {
        int32_t rows, experts, topK, full, hasBias;
        float routeScale;
    } pc{rows, experts, topK, full, hasBias, routeScale};
    static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Push) == 24,
                  "V4.1 router push ABI must be five int32 values and one float32");
    // One 64-lane workgroup per row; lane 0 alone writes status and picks,
    // after an exact-compare merge, so there is no atomic winner race.
    if (!dispatch(g_kern[K_V41_ROUTER_TOPK], bufs, &pc, sizeof(pc), uint32_t(rows))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_router_gate_topk(void) {
    return g_ready && g_have_v41_router_gate_topk && g_kern[K_V41_ROUTER_GATE_TOPK].pipe != VK_NULL_HANDLE;
}

int fvk_v41_router_gate_verdict(void) { return g_v41RouterGateVerdict; }

void fvk_v41_router_gate_set_verdict(int verdict) {
    if (verdict == 1 || verdict == 2) g_v41RouterGateVerdict = verdict;
}

int fvk_v41_router_gate_topk_f32(const void* dWeight, const void* dInput, const void* dBias,
                                 void* dLogits, void* dExperts, void* dWeights, void* dStatus,
                                 int experts, int hidden, int topK,
                                 int full, int hasBias, float routeScale) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_router_gate_topk()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (experts <= 0 || experts > 1024 || hidden <= 0 || hidden > 65536 ||
        topK <= 0 || topK > experts || topK > 8 ||
        (full != 0 && full != 1) || (hasBias != 0 && hasBias != 1) ||
        !std::isfinite(routeScale) || !(routeScale > 0.0f)) return 2;
    const uint64_t chunks = (uint64_t(hidden) + 1023u) / 1024u;
    const uint64_t perGroup = 64u / chunks;
    const uint64_t groups = (uint64_t(experts) + perGroup - 1u) / perGroup;
    const uint64_t weightCount = uint64_t(experts) * uint64_t(hidden);
    if (weightCount > INT32_MAX || groups == 0 || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[7] = {B(dWeight), B(dInput), B(dBias), B(dLogits), B(dExperts), B(dWeights), B(dStatus)};
    const uint64_t counts[7] = {weightCount, uint64_t(hidden), hasBias ? uint64_t(experts) : 1u,
                                uint64_t(experts), uint64_t(topK), uint64_t(topK), 4u};
    for (int i = 0; i < 7; ++i) {
        // Read-only operands may sit in larger storage; outputs are exact.
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            (i < 3 ? counts[i] * sizeof(uint32_t) > bufs[i]->bytes : counts[i] * sizeof(uint32_t) != bufs[i]->bytes) ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int out = 3; out <= 6; ++out) {
        for (int i = 0; i < 7; ++i) if (i != out && aliases(bufs[out], bufs[i])) return 2;
    }
    struct Push {
        int32_t experts, hidden, topK, full, hasBias;
        float routeScale;
    } pc{experts, hidden, topK, full, hasBias, routeScale};
    static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Push) == 24,
                  "V4.1 router gate push ABI must be five int32 values and one float32");
    if (!dispatch(g_kern[K_V41_ROUTER_GATE_TOPK], bufs, &pc, sizeof(pc), uint32_t(groups), 1, true)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

static bool v41IndexerScorePrerequisites() {
    return g_ready && g_submissionStatus == VK_SUCCESS && !g_v41SubmissionPendingFailure &&
        g_v41IndexerFloatControls && g_have_v41_indexer_score &&
        g_kern[K_V41_INDEXER_SCORE].pipe != VK_NULL_HANDLE;
}

int fvk_have_v41_indexer_score(void) {
    // Production capability: the module is in the trusted registry since
    // V5/complete62, whose receipt checks its float-control execution modes,
    // and dispatch still needs the binary32 float-control properties and a
    // built pipeline. The production-archive witness is the model selection
    // parity TestV41IndexerSelectionVulkanParity (Ref #13668).
    return v41IndexerScorePrerequisites() ? 1 : 0;
}

const char* fvk_v41_indexer_score_unavailable_reason(void) {
    if (!g_ready) return "Vulkan initialization incomplete";
    if (g_submissionStatus != VK_SUCCESS || g_v41SubmissionPendingFailure) return "sticky Vulkan submission fault";
    if (!v41IndexerScorePrerequisites()) {
        return g_v41IndexerScoreReason.empty() ? "indexer pipeline unavailable" : g_v41IndexerScoreReason.c_str();
    }
    return "";
}

static int v41IndexerScoreChecked(const void* dQ, const void* dKeys,
                              const void* dWeights, void* dScores,
                              int rows, int heads, int headDim) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows < 0 || heads <= 0 || headDim <= 0) return 2;
    const uint64_t qCount = uint64_t(heads) * uint64_t(headDim);
    const uint64_t keyCount = uint64_t(rows) * uint64_t(headDim);
    if (qCount > INT32_MAX || keyCount > INT32_MAX || uint64_t(rows) > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[4] = {B(dQ), B(dKeys), B(dWeights), B(dScores)};
    const uint64_t counts[4] = {qCount, keyCount, uint64_t(heads), uint64_t(rows)};
    for (int i = 0; i < 4; ++i) {
        // Empty key/output handles are never dereferenced or bound. Keys may be
        // a longer row log: the shader reads only rows*headDim elements.
        if (counts[i] == 0) { if (bufs[i] != nullptr) return 2; continue; }
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(float) ||
            (i == 1 ? counts[i] * sizeof(float) > bufs[i]->bytes : counts[i] * sizeof(float) != bufs[i]->bytes) ||
            bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    if (rows == 0) return 0; // no capability lookup, allocation, binding or dispatch
    if (!v41IndexerScorePrerequisites()) return 3;
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int i = 0; i < 3; ++i) if (aliases(bufs[3], bufs[i])) return 2;
    struct Push { int32_t rows, heads, headDim; } pc{rows, heads, headDim};
    static_assert(sizeof(int) == 4 && sizeof(float) == 4 && sizeof(Push) == 12 &&
                  offsetof(Push, rows) == 0 && offsetof(Push, heads) == 4 && offsetof(Push, headDim) == 8,
                  "V4.1 indexer score push ABI must be three fixed int32 values");
    if (!dispatch(g_kern[K_V41_INDEXER_SCORE], bufs, &pc, sizeof(pc), uint32_t(rows))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_v41_indexer_score_f32(const void* dQ, const void* dKeys,
                              const void* dWeights, void* dScores,
                              int rows, int heads, int headDim) {
    if (rows != 0 && !fvk_have_v41_indexer_score()) return 3;
    return v41IndexerScoreChecked(dQ, dKeys, dWeights, dScores, rows, heads, headDim);
}

int fvk_have_v41_mhc(void) {
    return g_ready && g_have_v41_mhc && g_kern[K_V41_MHC].pipe != VK_NULL_HANDLE;
}

int fvk_v41_mhc_f32(int mode, const void* dA, const void* dB, const void* dC,
                    void* dOut, int rows, int hidden, int iters, float eps, float rmsEps) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_mhc()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (mode < 0 || mode > 4 || rows <= 0) return 2;
    if ((mode == 1 || mode == 4) && (iters < 1 || !std::isfinite(eps) || !(eps > 0.0f))) return 2;
    if (mode == 4 && (!std::isfinite(rmsEps) || !(rmsEps > 0.0f))) return 2;
    if (mode != 1 && hidden <= 0) return 2;
    const uint64_t r = uint64_t(rows), h = uint64_t(hidden);
    const uint64_t coeffs = r * 24u, row = r * h, streams = r * 4u * h;
    // Element counts per binding A, B, C, Out; an unused binding repeats A.
    uint64_t counts[4] = {0, 0, 0, 0};
    uint64_t threads = 0;
    switch (mode) {
    case 0: counts[0] = row; counts[1] = row; counts[2] = row; counts[3] = streams; threads = row; break;
    case 1: counts[0] = coeffs; counts[1] = 3; counts[2] = 24; counts[3] = coeffs; threads = r; break;
    case 2: counts[0] = streams; counts[1] = coeffs; counts[2] = streams; counts[3] = row; threads = row; break;
    case 3: counts[0] = row; counts[1] = streams; counts[2] = coeffs; counts[3] = streams; threads = streams; break;
    case 4: counts[0] = coeffs; counts[1] = streams; counts[2] = 27; counts[3] = coeffs; threads = r; break;
    }
    if (streams > UINT32_MAX || coeffs > UINT32_MAX) return 2;
    // rms-split reduces a whole row per workgroup; every other mode is one thread per element.
    const uint64_t groups = mode == 4 ? r : (threads + 63u) / 64u;
    if (groups == 0 || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[4] = {B(dA), B(dB), B(dC), B(dOut)};
    for (int i = 0; i < 4; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(float) ||
            counts[i] * sizeof(float) > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int i = 0; i < 3; ++i) if (aliases(bufs[3], bufs[i])) return 2;
    const bool splits = mode == 1 || mode == 4;
    struct Push { int32_t mode, rows, hidden, iters; float eps, rmsEps; } pc{mode, rows, mode == 1 ? 0 : hidden, splits ? iters : 0, splits ? eps : 0.0f, mode == 4 ? rmsEps : 0.0f};
    static_assert(sizeof(Push) == 24 && offsetof(Push, eps) == 16 && offsetof(Push, rmsEps) == 20,
                  "V4.1 mHC push ABI must be four int32 values then two float32");
    if (!dispatch(g_kern[K_V41_MHC], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// A Q2_K prenorm reproduces q2k_matvec's order, so it exists only while the
// one-token Q2_K MatMul it replaces dispatches that kernel.
int fvk_have_v41_mhc_prenorm(int weightQ2K) {
    return fvk_have_v41_mhc() && (!weightQ2K || g_have_q2k_matvec) ? 1 : 0;
}

int fvk_v41_mhc_prenorm_split_f32(int weightQ2K, const void* dW, const void* dStreams,
                                  const void* dParams, void* dScratch, void* dOut,
                                  int rows, int hidden, int iters, float eps, float rmsEps) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_mhc_prenorm(weightQ2K)) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || hidden <= 0 || iters < 1 || !std::isfinite(eps) || !(eps > 0.0f) ||
        !std::isfinite(rmsEps) || !(rmsEps > 0.0f)) return 2;
    const uint64_t r = uint64_t(rows), width = 4u * uint64_t(hidden);
    const uint64_t chunks = (width + 1023u) / 1024u;
    if (weightQ2K ? width % 256u != 0 : chunks > 64u) return 2;
    const uint64_t perGroup = weightQ2K ? 2u : 64u / chunks;
    const uint64_t dotGroups = (24u + perGroup - 1u) / perGroup;
    const uint64_t weightBytes = weightQ2K ? 24u * (width / 256u) * 84u : 24u * width * 4u;
    const uint64_t streams = r * width;
    const uint64_t prenormGroups = r * (dotGroups + 1u), splitGroups = (r + 63u) / 64u;
    if (streams > UINT32_MAX || 24u * width > UINT32_MAX || r * 25u > UINT32_MAX) return 2;
    if (prenormGroups > g_maxComputeWorkGroupCountX || splitGroups > g_maxComputeWorkGroupCountX) return 2;
    // Weight, streams, params[27], scratch[rows,25], coeffs[rows,24].
    Buffer* bufs[5] = {B(dW), B(dStreams), B(dParams), B(dScratch), B(dOut)};
    const uint64_t bytes[5] = {weightBytes, streams * 4u, 27u * 4u, r * 25u * 4u, r * 24u * 4u};
    for (int i = 0; i < 5; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            bytes[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int i = 0; i < 3; ++i) if (aliases(bufs[3], bufs[i])) return 2;
    for (int i = 0; i < 4; ++i) if (aliases(bufs[4], bufs[i])) return 2;
    struct Push { int32_t mode, rows, hidden, iters; float eps, rmsEps; };
    static_assert(sizeof(Push) == 24 && offsetof(Push, eps) == 16 && offsetof(Push, rmsEps) == 20,
                  "V4.1 mHC push ABI must be four int32 values then two float32");
    const Push project{weightQ2K ? 6 : 5, rows, hidden, 0, 0.0f, 0.0f};
    Buffer* projectBufs[4] = {bufs[0], bufs[1], bufs[2], bufs[3]};
    if (!dispatch(g_kern[K_V41_MHC], projectBufs, &project, sizeof(project), uint32_t(prenormGroups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    const Push finish{7, rows, hidden, iters, eps, rmsEps};
    Buffer* finishBufs[4] = {bufs[3], bufs[3], bufs[2], bufs[4]};
    if (!dispatch(g_kern[K_V41_MHC], finishBufs, &finish, sizeof(finish), uint32_t(splitGroups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

#ifdef FAK_V41_INDEXER_SCORE_WITNESS
// Test-archive-only symbols. No production runtime switch can enable them or
// mark Supports true. The tagged Go witness must link this exact test archive.
int fvk_v41_indexer_score_witness_ready(void) { return v41IndexerScorePrerequisites(); }
int fvk_v41_indexer_score_witness_f32(const void* dQ, const void* dKeys,
                                      const void* dWeights, void* dScores,
                                      int rows, int heads, int headDim) {
    return v41IndexerScoreChecked(dQ, dKeys, dWeights, dScores, rows, heads, headDim);
}
#endif

int fvk_have_v41_index_select(void) {
    return g_ready && g_have_v41_index_select && g_kern[K_V41_INDEX_SELECT].pipe != VK_NULL_HANDLE;
}

int fvk_v41_index_select_f32(const void* dScores, void* dScratch, void* dOut, void* dStatus,
                             int rows, int topK, int topKBlocks, int blockSize) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_index_select()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || rows > (1 << 24) || topK <= 0 || topKBlocks < 0 || (topKBlocks > 0 && blockSize <= 0)) return 2;
    const uint64_t blocks = topKBlocks > 0 ? uint64_t(rows - 1) / uint64_t(blockSize) + 1u : 0u;
    const uint64_t selected = uint64_t(topK < rows ? topK : rows);
    Buffer* bufs[4] = {B(dScores), B(dScratch), B(dOut), B(dStatus)};
    const uint64_t counts[4] = {uint64_t(rows), 2u * blocks + uint64_t(rows), selected, 1u};
    for (int i = 0; i < 4; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(uint32_t) ||
            counts[i] * sizeof(uint32_t) > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    for (int i = 0; i < 4; ++i) {
        for (int j = i + 1; j < 4; ++j) if (aliases(bufs[i], bufs[j])) return 2;
    }
    struct Push { int32_t rows, topK, topKBlocks, blockSize; } pc{rows, topK, topKBlocks, blockSize};
    static_assert(sizeof(Push) == 16, "V4.1 index select push ABI must be four int32 values");
    if (!dispatch(g_kern[K_V41_INDEX_SELECT], bufs, &pc, sizeof(pc), 1u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_exact_rmsnorm(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_exact_rmsnorm &&
        g_kern[K_V41_EXACT_RMSNORM].pipe != VK_NULL_HANDLE;
}

int fvk_v41_exact_rmsnorm_f32(const void* dX, const void* dW, void* dY,
                              int rows, int width, float eps) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_exact_rmsnorm()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    // Power-of-two width keeps ss*invWidth the correctly rounded ss/width; a
    // normal eps keeps mean+eps normal. Both are the Go domain check repeated.
    if (rows <= 0 || width <= 0 || width > (1 << 16) || (width & (width - 1)) != 0 ||
        !std::isfinite(eps) || !(eps >= std::numeric_limits<float>::min())) return 2;
    const uint64_t total = uint64_t(rows) * uint64_t(width);
    if (total > INT32_MAX || uint64_t(rows) > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[3] = {B(dX), B(dW), B(dY)};
    const uint64_t counts[3] = {total, uint64_t(width), total};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] * sizeof(float) > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    if (aliases(bufs[2], bufs[0]) || aliases(bufs[2], bufs[1])) return 2;
    struct Push { int32_t rows, width; float eps, invWidth; } pc{rows, width, eps, 1.0f / float(width)};
    static_assert(sizeof(Push) == 16 && offsetof(Push, eps) == 8 && offsetof(Push, invWidth) == 12,
                  "V4.1 exact RMSNorm push ABI must be two int32 values then two float32 values");
    if (!dispatch(g_kern[K_V41_EXACT_RMSNORM], bufs, &pc, sizeof(pc), uint32_t(rows))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

static bool v41ExtentOK(const Buffer* b, uint64_t count) {
    return b && b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE &&
        count <= std::numeric_limits<size_t>::max() / sizeof(float) &&
        count * sizeof(float) <= b->bytes && b->bytes <= g_maxStorageBufferRange &&
        b->bytes <= std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset;
}

static bool v41BuffersAlias(const Buffer* a, const Buffer* b) {
    return a == b || a->buf == b->buf ||
        (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
         b->memoryOffset < a->memoryOffset + a->bytes);
}

int fvk_have_v41_engram_gate(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_engram_gate &&
        g_kern[K_V41_ENGRAM_GATE].pipe != VK_NULL_HANDLE;
}

int fvk_v41_engram_gate_f32(const void* dStreams, const void* dProjection, const void* dNorms,
                            void* dNext, void* dStatus, int rows, int hidden, float eps,
                            float invSqrtHidden) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_engram_gate()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || hidden <= 0 || hidden > (1 << 16) || !std::isfinite(eps) || !std::isfinite(invSqrtHidden)) return 2;
    const uint64_t r = uint64_t(rows), h = uint64_t(hidden);
    const uint64_t panel = r * 4u * h, projection = r * 5u * h, groups = r * 4u;
    if (projection > INT32_MAX || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[5] = {B(dStreams), B(dProjection), B(dNorms), B(dNext), B(dStatus)};
    const uint64_t counts[5] = {panel, projection, 8u * h, panel, groups};
    for (int i = 0; i < 5; ++i) if (!v41ExtentOK(bufs[i], counts[i])) return 2;
    for (int i = 0; i < 3; ++i) {
        if (v41BuffersAlias(bufs[3], bufs[i]) || v41BuffersAlias(bufs[4], bufs[i])) return 2;
    }
    if (v41BuffersAlias(bufs[3], bufs[4])) return 2;
    const bool pow2 = (hidden & (hidden - 1)) == 0;
    struct Push { int32_t rows, hidden, pow2; float eps, invHidden, hiddenF, invSqrtHidden; } pc{
        rows, hidden, pow2 ? 1 : 0, eps, 1.0f / float(hidden), float(hidden), invSqrtHidden};
    static_assert(sizeof(Push) == 28 && offsetof(Push, eps) == 12 && offsetof(Push, invSqrtHidden) == 24,
                  "V4.1 Engram gate push ABI must be three int32 values then four float32 values");
    if (!dispatch(g_kern[K_V41_ENGRAM_GATE], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_engram_row_decode(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_engram_row_decode &&
        g_kern[K_V41_ENGRAM_ROW_DECODE].pipe != VK_NULL_HANDLE;
}

struct V41EngramRowDecodePush { int32_t mode, rows, dim, target; };
static_assert(sizeof(V41EngramRowDecodePush) == 16, "V4.1 Engram row decode push ABI must be four int32 values");

int fvk_v41_engram_row_decode(const void* dSrc, void* dDst, void* dStatus, int mode, int rows, int dim) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_engram_row_decode()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    static const uint64_t strideWords[3] = {66u, 21u, 256u};
    if (mode < 0 || mode > 2 || rows <= 0 || dim <= 0 || dim > 256) return 2;
    const uint64_t r = uint64_t(rows);
    const uint64_t src = r * strideWords[mode], panel = r * uint64_t(dim);
    if (src > INT32_MAX || panel > INT32_MAX) return 2;
    Buffer* bufs[3] = {B(dSrc), B(dDst), B(dStatus)};
    const uint64_t counts[3] = {src, panel, r};
    for (int i = 0; i < 3; ++i) if (!v41ExtentOK(bufs[i], counts[i])) return 2;
    if (v41BuffersAlias(bufs[1], bufs[0]) || v41BuffersAlias(bufs[2], bufs[0]) || v41BuffersAlias(bufs[1], bufs[2])) return 2;
    V41EngramRowDecodePush pc{mode, rows, dim, 0};
    const uint64_t groups = r < g_maxComputeWorkGroupCountX ? r : g_maxComputeWorkGroupCountX;
    if (groups == 0) return 2;
    if (!dispatch(g_kern[K_V41_ENGRAM_ROW_DECODE], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_v41_engram_row_poison(const void* dStatus, void* dTarget, int rows, int targetCount, int index) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_engram_row_decode()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || targetCount <= 0 || index < 0 || index >= targetCount) return 2;
    // The source binding is unused in poison mode; it aliases the status words.
    Buffer* bufs[3] = {B(dStatus), B(dTarget), B(dStatus)};
    if (!v41ExtentOK(bufs[0], uint64_t(rows)) || !v41ExtentOK(bufs[1], uint64_t(targetCount))) return 2;
    if (v41BuffersAlias(bufs[1], bufs[0])) return 2;
    V41EngramRowDecodePush pc{3, rows, 0, index};
    if (!dispatch(g_kern[K_V41_ENGRAM_ROW_DECODE], bufs, &pc, sizeof(pc), 1u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_compressor_pool(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_compressor_pool &&
        g_kern[K_V41_COMPRESSOR_POOL].pipe != VK_NULL_HANDLE;
}

int fvk_v41_compressor_pool_f32(const void* dKV, const void* dScores, void* dOut,
                                int ratio, int width) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_compressor_pool()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (ratio < 2 || ratio > (1 << 12) || width <= 0) return 2;
    const uint64_t total = uint64_t(ratio) * uint64_t(width);
    const uint64_t groups = (uint64_t(width) + 63u) / 64u;
    if (total > INT32_MAX || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[3] = {B(dKV), B(dScores), B(dOut)};
    const uint64_t counts[3] = {total, total, uint64_t(width)};
    for (int i = 0; i < 3; ++i) if (!v41ExtentOK(bufs[i], counts[i])) return 2;
    if (v41BuffersAlias(bufs[2], bufs[0]) || v41BuffersAlias(bufs[2], bufs[1])) return 2;
    struct Push { int32_t ratio, width; } pc{ratio, width};
    static_assert(sizeof(Push) == 8, "V4.1 compressor pool push ABI must be two int32 values");
    if (!dispatch(g_kern[K_V41_COMPRESSOR_POOL], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_compressor_ring(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_compressor_ring &&
        g_kern[K_V41_COMPRESSOR_RING].pipe != VK_NULL_HANDLE && g_kern[K_V41_COMPRESSOR_RING].nbuf == 5 &&
        g_kern[K_V41_COMPRESSOR_RING].pcsize == 16;
}

// Stores one projected (kv, score) row into ring row `slot`; with emit it also
// pools the whole ring into dOut [2,width] in the same dispatch. Without emit
// dOut is unused and may alias the kv row.
int fvk_v41_compressor_ring_f32(const void* dKVRow, const void* dScoreRow, void* dKVRing, void* dScoreRing,
                                void* dOut, int ratio, int width, int slot, int emit) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_compressor_ring()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (ratio < 2 || ratio > (1 << 12) || width <= 0 || slot < 0 || slot >= ratio || (emit != 0 && emit != 1)) return 2;
    const uint64_t total = uint64_t(ratio) * uint64_t(width);
    const uint64_t groups = (uint64_t(width) + 63u) / 64u;
    if (total > INT32_MAX || groups > g_maxComputeWorkGroupCountX) return 2;
    Buffer* bufs[5] = {B(dKVRow), B(dScoreRow), B(dKVRing), B(dScoreRing), B(emit ? dOut : (void*)dKVRow)};
    const uint64_t counts[5] = {uint64_t(width), uint64_t(width), total, total, emit ? 2u * uint64_t(width) : uint64_t(width)};
    for (int i = 0; i < 5; ++i) if (!v41ExtentOK(bufs[i], counts[i])) return 2;
    for (int ring = 2; ring < 4; ++ring) {
        if (v41BuffersAlias(bufs[ring], bufs[0]) || v41BuffersAlias(bufs[ring], bufs[1])) return 2;
    }
    if (v41BuffersAlias(bufs[2], bufs[3])) return 2;
    if (emit) {
        for (int i = 0; i < 4; ++i) if (v41BuffersAlias(bufs[4], bufs[i])) return 2;
    }
    struct Push { int32_t ratio, width, slot, emit; } pc{ratio, width, slot, emit};
    static_assert(sizeof(Push) == 16, "V4.1 compressor ring push ABI must be four int32 values");
    if (!dispatch(g_kern[K_V41_COMPRESSOR_RING], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_routed_swiglu(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_routed_swiglu &&
        g_kern[K_V41_ROUTED_SWIGLU].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_SWIGLU].nbuf == 4 &&
        g_kern[K_V41_ROUTED_SWIGLU].pcsize == 12;
}

int fvk_v41_routed_swiglu_f32(const void* dGate, const void* dUp, void* dOut, void* dStatus,
                              int count, float limit, float weight) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_swiglu()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || count <= 0 || !std::isfinite(limit) || !std::isfinite(weight) ||
        g_maxComputeWorkGroupCountX == 0) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    if (bytes > std::numeric_limits<size_t>::max() || bytes > std::numeric_limits<VkDeviceSize>::max()) return 2;
    Buffer* bufs[4] = {B(dGate), B(dUp), B(dOut), B(dStatus)};
    const uint64_t required[4] = {bytes, bytes, bytes, 16};
    for (int i = 0; i < 4; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    // Conservative whole-allocation ranges; every addition is checked above.
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    struct Push { uint32_t count; float limit, weight; } pc{uint32_t(count), limit, weight};
    static_assert(sizeof(Push) == 12 && offsetof(Push, limit) == 4 && offsetof(Push, weight) == 8,
                  "V4.1 routed SwiGLU push ABI must be uint32 count then float32 limit and weight");
    // groups <= ceil(INT32_MAX/64), so every active grid-stride index stays below 2^31.
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_SWIGLU], bufs, &pc, sizeof(pc), groups, 1, true, bufs[3])) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}
// One row of borrowed gate/up panels, using the unchanged per-lane scalar
// activation arithmetic and one freshly initialized row-local status record.
int fvk_have_v41_routed_swiglu_panel_row(void) {
    const char* enabled = std::getenv("FAK_V41_ROUTED_PANEL_ROW");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW].nbuf == 4 && g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW].pcsize == 16;
}

int fvk_v41_routed_swiglu_panel_row_shape(int rows, int count, int row) {
    if (rows < 1 || rows > 64 || count <= 0 || row < 0 || row >= rows || g_maxComputeWorkGroupCountX == 0) return 0;
    const uint64_t bytes = uint64_t(rows) * uint64_t(count) * 4u;
    return bytes <= INT32_MAX && bytes <= g_maxStorageBufferRange;
}

int fvk_v41_routed_swiglu_panel_row_f32(const void* dGate, const void* dUp, void* dOut, void* dStatus,
                              int rows, int count, int row, float limit, float weight) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_swiglu_panel_row()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !fvk_v41_routed_swiglu_panel_row_shape(rows, count, row) || !std::isfinite(limit) || !std::isfinite(weight) ||
        g_maxComputeWorkGroupCountX == 0) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    if (bytes > std::numeric_limits<size_t>::max() || bytes > std::numeric_limits<VkDeviceSize>::max()) return 2;
    Buffer* bufs[4] = {B(dGate), B(dUp), B(dOut), B(dStatus)};
    const uint64_t required[4] = {uint64_t(rows) * bytes, uint64_t(rows) * bytes, bytes, 16};
    for (int i = 0; i < 4; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    // Conservative whole-allocation ranges; every addition is checked above.
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    struct Push { uint32_t count, rowOffset; float limit, weight; } pc{uint32_t(count), uint32_t(uint64_t(row) * uint64_t(count)), limit, weight};
    static_assert(sizeof(Push) == 16 && offsetof(Push, rowOffset) == 4 && offsetof(Push, limit) == 8 && offsetof(Push, weight) == 12,
                  "V4.1 panel-row SwiGLU push ABI is count, row offset, limit and weight");
    // groups <= ceil(INT32_MAX/64), so every active grid-stride index stays below 2^31.
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_SWIGLU_PANEL_ROW], bufs, &pc, sizeof(pc), groups, 1, true, bufs[3])) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}
int fvk_have_v41_routed_swiglu_panel_chunk(void) {
    const char* enabled = std::getenv("FAK_V41_ROUTED_PANEL_CHUNK");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK].nbuf == 4 && g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK].pcsize == 48;
}

int fvk_v41_routed_swiglu_panel_chunk_shape(int rows, int count, int row, int chunk) {
    if (rows < 1 || rows > 64 || count <= 0 || row < 0 || (chunk != 2 && chunk != 4 && chunk != 8) || row > rows - chunk || uint32_t(chunk) > g_maxComputeWorkGroupCountY || g_maxComputeWorkGroupCountX == 0) return 0;
    const uint64_t bytes = uint64_t(rows) * uint64_t(count) * 4u;
    return bytes <= INT32_MAX && bytes <= g_maxStorageBufferRange;
}

int fvk_v41_routed_swiglu_panel_chunk_f32(const void* dGate, const void* dUp, void* dOut, void* dStatus,
                              int rows, int count, int row, int chunk, float limit, const float* weights) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_swiglu_panel_chunk()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !fvk_v41_routed_swiglu_panel_chunk_shape(rows, count, row, chunk) || !std::isfinite(limit) || !weights ||
        g_maxComputeWorkGroupCountX == 0) return 2;
    for (int i = 0; i < chunk; ++i) if (!std::isfinite(weights[i])) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    if (bytes > std::numeric_limits<size_t>::max() || bytes > std::numeric_limits<VkDeviceSize>::max()) return 2;
    Buffer* bufs[4] = {B(dGate), B(dUp), B(dOut), B(dStatus)};
    const uint64_t required[4] = {uint64_t(rows) * bytes, uint64_t(rows) * bytes, uint64_t(chunk) * bytes, uint64_t(chunk) * 16u};
    for (int i = 0; i < 4; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    // Conservative whole-allocation ranges; every addition is checked above.
    for (int i = 0; i < 4; ++i) for (int j = i + 1; j < 4; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    struct Push { uint32_t count, rowOffset, rows; float limit; float weights[8]; } pc{};
    pc.count = uint32_t(count); pc.rowOffset = uint32_t(uint64_t(row) * uint64_t(count));
    pc.rows = uint32_t(chunk); pc.limit = limit;
    std::memcpy(pc.weights, weights, size_t(chunk) * sizeof(float));
    static_assert(sizeof(Push) == 48 && offsetof(Push, rowOffset) == 4 && offsetof(Push, rows) == 8 &&
                  offsetof(Push, limit) == 12 && offsetof(Push, weights) == 16,
                  "V4.1 chunk push ABI requires eight contiguous binary32 route weights");
    // groups <= ceil(INT32_MAX/64), so every active grid-stride index stays below 2^31.
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_SWIGLU_PANEL_CHUNK], bufs, &pc, sizeof(pc), groups, uint32_t(chunk), true, bufs[3], uint32_t(chunk) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}
int fvk_v41_routed_swiglu_panel_chunk_module_abi(const void* data, size_t bytes) {
    if (!data || bytes == 0 || bytes % 4 != 0) return 0;
    const char* first = static_cast<const char*>(data);
    return v41RoutedSwiGLUChunkABI(std::vector<char>(first, first + bytes)) ? 1 : 0;
}
// V4.1 routed-expert id dispatch (fak#13820): one dispatch per stage over every
// routed pick of a layer. ADAPT (MIT, Copyright (c) 2023-2026 The ggml authors):
// llama.cpp ggml-vulkan mul_mat_vec_id @ccdbe290533f68154ede46e0618622eebcb6c070
// (ggml-vulkan.cpp:7876-7908, one dispatch, grid Y = selected expert) with ids
// remapped through a per-layer slot map as llama.cpp PR #29887's llama_moe_cache
// does. The slot table is the layer's bound ring pages (up to kV41RoutedIDSlots).
static constexpr int kV41RoutedIDSlots = 8;
static int v41BF16CheckedCountSupported(int count);

int fvk_have_v41_routed_id(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_routed_id &&
        g_kern[K_V41_Q2K_MATVEC_ID].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q2K_MATVEC_ID].nbuf == 11 &&
        g_kern[K_V41_Q3K_MATVEC_ID].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q3K_MATVEC_ID].nbuf == 11 &&
        g_kern[K_V41_ROUTED_SWIGLU_ID].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_SWIGLU_ID].nbuf == 5 &&
        g_kern[K_V41_ROUTED_PUBLISH_SUM].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_PUBLISH_SUM].nbuf == 4;
}

int fvk_v41_routed_id_max_picks(void) { return kV41RoutedIDSlots; }

// Fused calls require the optional pair at the capability stage, preserving
// unavailable-before-submission-error precedence without disabling legacy calls.
static int v41RoutedIDState(bool requireFused = false) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_id() || (requireFused && !fvk_have_v41_routed_fused())) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (g_batching && !g_stepBatch) return 2;
    return 0;
}

static bool v41RoutedIDBytesOK(const Buffer* b, uint64_t bytes) {
    return b && b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE && bytes > 0 && bytes <= b->bytes &&
        b->bytes <= std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset;
}

int fvk_have_v41_q2k_token_pair(void) {
    const char* enabled = std::getenv("FAK_V41_Q2K_TOKEN_PAIR");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_Q2K_TOKEN_PAIR].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q2K_TOKEN_PAIR].nbuf == 5 && g_kern[K_V41_Q2K_TOKEN_PAIR].pcsize == 12;
}
int fvk_v41_q2k_token_pair_shape(int out, int in, int tokens) {
    if (tokens < 2 || tokens > 64 || out <= 0 || in <= 0 || in % 256 != 0) return 0;
    const uint64_t wBytes = uint64_t(out) * uint64_t(in / 256) * 84u;
    const uint64_t xBytes = uint64_t(tokens) * uint64_t(in) * 4u;
    const uint64_t yBytes = uint64_t(tokens) * uint64_t(out) * 4u;
    const uint64_t groups = 2u * ((uint64_t(out) + 1u) / 2u);
    const uint64_t tokenGroups = (uint64_t(tokens) + 7u) / 8u;
    return wBytes <= INT32_MAX && xBytes <= INT32_MAX && yBytes <= INT32_MAX &&
        g_maxComputeWorkGroupCountX != 0 && groups <= g_maxComputeWorkGroupCountX &&
        g_maxComputeWorkGroupCountY != 0 && tokenGroups <= g_maxComputeWorkGroupCountY;
}
int fvk_v41_q2k_token_pair_f32(const void* dG, const void* dU, const void* dX, void* dGOut, void* dUOut, int out, int in, int tokens) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_q2k_token_pair()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !fvk_v41_q2k_token_pair_shape(out, in, tokens)) return 2;
    Buffer* bufs[5] = {B(dG), B(dX), B(dGOut), B(dU), B(dUOut)};
    const uint64_t wBytes = uint64_t(out) * uint64_t(in / 256) * 84u;
    const uint64_t xBytes = uint64_t(tokens) * uint64_t(in) * 4u;
    const uint64_t yBytes = uint64_t(tokens) * uint64_t(out) * 4u;
    if (!v41RoutedIDBytesOK(bufs[0], wBytes) || !v41RoutedIDBytesOK(bufs[3], wBytes) ||
        !v41RoutedIDBytesOK(bufs[1], xBytes) || !v41RoutedIDBytesOK(bufs[2], yBytes) || !v41RoutedIDBytesOK(bufs[4], yBytes)) return 2;
    for (int output : {2,4}) {
        for (int other : {0,1,2,3,4}) {
            if (other != output && v41BuffersAlias(bufs[output], bufs[other])) return 2;
        }
    }
    struct Push { int32_t outDim, inDim, tokens; } pc{out,in,tokens};
    static_assert(sizeof(Push) == 12 && offsetof(Push,inDim) == 4 && offsetof(Push,tokens) == 8, "Q2 token pair ABI");
    if (!dispatch(g_kern[K_V41_Q2K_TOKEN_PAIR], bufs, &pc, sizeof(pc), uint32_t(2u*((uint64_t(out)+1u)/2u)), uint32_t((uint64_t(tokens)+7u)/8u), true)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_q2k_token_panel(void) {
    return v41TokenPanelEnvEnabled("FAK_V41_Q2K_TOKEN_PANEL") && g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_Q2K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q2K_TOKEN_PANEL].nbuf == 3;
}

int fvk_v41_q2k_token_panel_shape(int out, int in, int tokens) {
    // Legacy optional modules ignore Y and own only eight token slots.
    if (!g_v41Q2TokenGridModule && tokens != 2 && tokens != 4 && tokens != 8) return 0;
    if ((tokens < 2 || tokens > 64) || out <= 0 || in <= 0 || in % 256 != 0) return 0;
    const uint64_t wBytes = uint64_t(out) * uint64_t(in / 256) * 84u;
    const uint64_t xBytes = uint64_t(tokens) * uint64_t(in) * 4u;
    const uint64_t yBytes = uint64_t(tokens) * uint64_t(out) * 4u;
    const uint64_t groups = (uint64_t(out) + 1u) / 2u;
    const uint64_t tokenGroups = (uint64_t(tokens) + 7u) / 8u;
    return wBytes <= INT32_MAX && xBytes <= INT32_MAX && yBytes <= INT32_MAX &&
        g_maxComputeWorkGroupCountX != 0 && groups <= g_maxComputeWorkGroupCountX &&
        g_maxComputeWorkGroupCountY != 0 && tokenGroups <= g_maxComputeWorkGroupCountY;
}

int fvk_v41_q2k_token_panel_f32(const void* dW, const void* dX, void* dY, int out, int in, int tokens) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_q2k_token_panel()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !fvk_v41_q2k_token_panel_shape(out, in, tokens)) return 2;
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
    if (!v41RoutedIDBytesOK(bufs[0], uint64_t(out) * uint64_t(in / 256) * 84u) ||
        !v41RoutedIDBytesOK(bufs[1], uint64_t(tokens) * uint64_t(in) * 4u) ||
        !v41RoutedIDBytesOK(bufs[2], uint64_t(tokens) * uint64_t(out) * 4u) ||
        v41BuffersAlias(bufs[2], bufs[0]) || v41BuffersAlias(bufs[2], bufs[1])) return 2;
    struct Push { int32_t outDim, inDim, tokens; } pc{out, in, tokens};
    static_assert(sizeof(Push) == 12 && offsetof(Push, inDim) == 4 && offsetof(Push, tokens) == 8,
                  "Q2_K token-panel push ABI must be three int32 values");
    if (!dispatch(g_kern[K_V41_Q2K_TOKEN_PANEL], bufs, &pc, sizeof(pc), uint32_t((uint64_t(out) + 1u) / 2u), uint32_t((uint64_t(tokens) + 7u) / 8u), true)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_q3k_token_panel(void) {
    return v41TokenPanelEnvEnabled("FAK_V41_Q3K_TOKEN_PANEL") && g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_Q3K_TOKEN_PANEL].pipe != VK_NULL_HANDLE && g_kern[K_V41_Q3K_TOKEN_PANEL].nbuf == 3;
}

int fvk_v41_q3k_token_panel_shape(int out, int in, int tokens) {
    // Old optional bundles remain safe: their shader ignores workgroup Y and
    // owns exactly eight accumulators. Larger panels must preselect fallback.
    if (!g_v41Q3TokenGridModule && tokens != 2 && tokens != 4 && tokens != 8) return 0;
    if ((tokens < 2 || tokens > 64) || out <= 0 || in <= 0 || in % 256 != 0) return 0;
    const uint64_t wBytes = (uint64_t(out) * uint64_t(in / 256) * 110u + 3u) & ~uint64_t(3u);
    const uint64_t xBytes = uint64_t(tokens) * uint64_t(in) * 4u;
    const uint64_t yBytes = uint64_t(tokens) * uint64_t(out) * 4u;
    const uint64_t groups = (uint64_t(out) + 63u) / 64u;
    const uint64_t tokenGroups = (uint64_t(tokens) + 7u) / 8u;
    return wBytes <= INT32_MAX && xBytes <= INT32_MAX && yBytes <= INT32_MAX &&
        g_maxComputeWorkGroupCountX != 0 && groups <= g_maxComputeWorkGroupCountX &&
        g_maxComputeWorkGroupCountY != 0 && tokenGroups <= g_maxComputeWorkGroupCountY;
}

int fvk_v41_q3k_token_panel_f32(const void* dW, const void* dX, void* dY, int out, int in, int tokens) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_q3k_token_panel()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !fvk_v41_q3k_token_panel_shape(out, in, tokens)) return 2;
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
    if (!v41RoutedIDBytesOK(bufs[0], (uint64_t(out) * uint64_t(in / 256) * 110u + 3u) & ~uint64_t(3u)) ||
        !v41RoutedIDBytesOK(bufs[1], uint64_t(tokens) * uint64_t(in) * 4u) ||
        !v41RoutedIDBytesOK(bufs[2], uint64_t(tokens) * uint64_t(out) * 4u) ||
        v41BuffersAlias(bufs[2], bufs[0]) || v41BuffersAlias(bufs[2], bufs[1])) return 2;
    struct Push { int32_t outDim, inDim, tokens; } pc{out, in, tokens};
    static_assert(sizeof(Push) == 12 && offsetof(Push, inDim) == 4 && offsetof(Push, tokens) == 8,
                  "Q3_K token-panel push ABI must be three int32 values");
    if (!dispatch(g_kern[K_V41_Q3K_TOKEN_PANEL], bufs, &pc, sizeof(pc), uint32_t((uint64_t(out) + 63u) / 64u), uint32_t((uint64_t(tokens) + 7u) / 8u), true)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// dtype 0 is Q2_K (84-byte blocks), 1 is Q3_K (110-byte blocks). xStride is the
// per-pick activation stride in floats: 0 shares one row, in gives each pick its own.
int fvk_v41_matvec_id_f32(int dtype, void* const* dW, int slots, const void* dX, void* dY, const void* dIds,
                          int out, int in, int picks, int xStride) {
    const int state = v41RoutedIDState();
    if (state != 0) return state;
    if ((dtype != 0 && dtype != 1) || !dW || slots <= 0 || slots > kV41RoutedIDSlots || picks <= 0 ||
        picks > kV41RoutedIDSlots || out <= 0 || in <= 0 || in % 256 != 0 || (xStride != 0 && xStride != in) ||
        g_maxComputeWorkGroupCountX == 0) return 2;
    const uint64_t blockBytes = dtype == 0 ? 84u : 110u;
    const uint64_t wBytes = uint64_t(out) * uint64_t(in / 256) * blockBytes;
    const uint64_t xFloats = uint64_t(in) * (xStride == 0 ? 1u : uint64_t(picks));
    const uint64_t yFloats = uint64_t(out) * uint64_t(picks);
    if (wBytes > UINT32_MAX || xFloats > INT32_MAX || yFloats > INT32_MAX) return 2;
    Buffer* bufs[11];
    for (int i = 0; i < kV41RoutedIDSlots; ++i) {
        Buffer* w = B(dW[i < slots ? i : 0]);
        if (!v41RoutedIDBytesOK(w, wBytes)) return 2;
        bufs[i] = w;
    }
    bufs[8] = B((void*)dX);
    bufs[9] = B(dY);
    bufs[10] = B((void*)dIds);
    if (!v41RoutedIDBytesOK(bufs[8], xFloats * 4u) || !v41RoutedIDBytesOK(bufs[9], yFloats * 4u) ||
        !v41RoutedIDBytesOK(bufs[10], uint64_t(picks) * 4u)) return 2;
    for (int i = 0; i < 11; ++i) {
        if (i != 9 && v41BuffersAlias(bufs[9], bufs[i])) return 2;
    }
    struct Push { int32_t outDim, inDim, picks, xStride; } pc{out, in, picks, xStride};
    static_assert(sizeof(Push) == 16, "V4.1 routed id GEMV push ABI must be four int32 values");
    Kernel& k = dtype == 0 ? g_kern[K_V41_Q2K_MATVEC_ID] : g_kern[K_V41_Q3K_MATVEC_ID];
    const uint64_t groups = dtype == 0 ? (uint64_t(out) + 1u) / 2u : (uint64_t(out) + 63u) / 64u;
    if (groups > g_maxComputeWorkGroupCountX) return 2;
    if (!dispatch(k, bufs, &pc, sizeof(pc), uint32_t(groups), uint32_t(picks))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

static bool v41RoutedIDDistinct(Buffer* const* bufs, int n) {
    for (int i = 0; i < n; ++i) for (int j = i + 1; j < n; ++j) {
        if (v41BuffersAlias(bufs[i], bufs[j])) return false;
    }
    return true;
}

// The per-pick v41_routed_swiglu over every pick: status holds one 16-byte record
// per pick, filled with UINT32_MAX in the same command before the dispatch.
int fvk_v41_routed_swiglu_id_f32(const void* dGate, const void* dUp, void* dOut, const void* dIds, void* dStatus,
                                 int count, int picks, float limit) {
    const int state = v41RoutedIDState();
    if (state != 0) return state;
    if (count <= 0 || picks <= 0 || picks > kV41RoutedIDSlots || !std::isfinite(limit)) return 2;
    const uint64_t total = uint64_t(count) * uint64_t(picks);
    if (total > INT32_MAX) return 2;
    Buffer* bufs[5] = {B((void*)dGate), B((void*)dUp), B(dOut), B((void*)dIds), B(dStatus)};
    const uint64_t required[5] = {total * 4u, total * 4u, total * 4u, uint64_t(picks) * 8u, uint64_t(picks) * 16u};
    for (int i = 0; i < 5; ++i) if (!v41RoutedIDBytesOK(bufs[i], required[i])) return 2;
    if (!v41RoutedIDDistinct(bufs, 5)) return 2;
    struct Push { uint32_t count; float limit; uint32_t picks; } pc{uint32_t(count), limit, uint32_t(picks)};
    static_assert(sizeof(Push) == 12 && offsetof(Push, limit) == 4 && offsetof(Push, picks) == 8,
                  "V4.1 routed SwiGLU id push ABI must be uint32 count, float32 limit, uint32 picks");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_SWIGLU_ID], bufs, &pc, sizeof(pc), groups, uint32_t(picks), true, bufs[4],
                  uint32_t(picks) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// The checked BF16 publication of every pick's down row plus their routed sum
// in ids order. Site policy mirrors fvk_v41_bf16_publish_f32.
int fvk_v41_routed_publish_sum_f32(const void* dDown, void* dSum, const void* dIds, void* dStatus,
                                   int count, int picks, uint32_t site) {
    const int state = v41RoutedIDState();
    if (state != 0) return state;
    const bool validSite = site == 1 || site == 2 || (site >= 5 && site <= 27);
    if (!validSite || count <= 0 || picks <= 0 || picks > kV41RoutedIDSlots ||
        !v41BF16CheckedCountSupported(count)) return 2;
    const uint64_t total = uint64_t(count) * uint64_t(picks);
    if (total > INT32_MAX) return 2;
    Buffer* bufs[4] = {B((void*)dDown), B(dSum), B((void*)dIds), B(dStatus)};
    const uint64_t required[4] = {total * 4u, uint64_t(count) * 4u, uint64_t(picks) * 12u, uint64_t(picks) * 16u};
    for (int i = 0; i < 4; ++i) if (!v41RoutedIDBytesOK(bufs[i], required[i])) return 2;
    if (!v41RoutedIDDistinct(bufs, 4)) return 2;
    struct Push { uint32_t count, site, picks; } pc{uint32_t(count), site, uint32_t(picks)};
    static_assert(sizeof(Push) == 12, "V4.1 routed publish-sum push ABI must be three uint32 values");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_PUBLISH_SUM], bufs, &pc, sizeof(pc), groups, 1, true, bufs[3],
                  uint32_t(picks) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// V4.1 fused routed-FFN dispatch (fak-private#3341): two dispatches per routed
// layer instead of the id path's five. ADAPT (MIT, Copyright (c) 2025-2026
// antirez): antirez/ds4 ds4_cuda.cu moe_gate_up_mid_decode_lut_qwarp32_kernel
// @fc80bd695da76ee14bcb86820bf900a5cdfda806:22143 (gate + up + SwiGLU + route
// weight into one mid buffer) and moe_down_sum_qwarp32_kernel:23869 /
// moe_down_q4K_sum6_qwarp32_kernel:24099 (down projection plus the six-expert
// sum). The id/slot-map binding is the routed id path's (llama.cpp
// mul_mat_vec_id + llama_moe_cache). Each stage is the routed id stage term for
// term, so every pick's output and status record track the id path.
int fvk_have_v41_routed_fused(void) {
    return g_ready && g_v41IndexerFloatControls &&
        g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID].nbuf == 20 &&
        g_kern[K_V41_ROUTED_DOWN_SUM].pipe != VK_NULL_HANDLE && g_kern[K_V41_ROUTED_DOWN_SUM].nbuf == 12;
}

// One Q2_K gate+up GEMV with the BF16 clamp-SwiGLU route-weight publication in
// its epilogue. Grid Y is the pick (its slot read from dIds), grid X tiles the
// inter rows in pairs. Returns the tail status codes.
int fvk_v41_routed_gate_up_swiglu_id_f32(void* const* dGate, void* const* dUp, int slots,
                                          const void* dX, const void* dIds, void* dMid, void* dStatus,
                                          int out, int in, int picks, int xStride, float limit) {
    const int state = v41RoutedIDState(true);
    if (state != 0) return state;
    if (!dGate || !dUp || !std::isfinite(limit) || slots <= 0 || slots > kV41RoutedIDSlots || picks <= 0 ||
        picks > kV41RoutedIDSlots || out <= 0 || in <= 0 || in % 256 != 0 || (xStride != 0 && xStride != in) ||
        g_maxComputeWorkGroupCountX == 0) return 2;
    const uint64_t wBytes = uint64_t(out) * uint64_t(in / 256) * 84u;
    const uint64_t xFloats = uint64_t(in) * (xStride == 0 ? 1u : uint64_t(picks));
    const uint64_t midFloats = uint64_t(out) * uint64_t(picks);
    if (wBytes > UINT32_MAX || xFloats > INT32_MAX || midFloats > INT32_MAX) return 2;
    Buffer* bufs[20];
    for (int i = 0; i < kV41RoutedIDSlots; ++i) {
        Buffer* g = B(dGate[i < slots ? i : 0]);
        Buffer* u = B(dUp[i < slots ? i : 0]);
        if (!v41RoutedIDBytesOK(g, wBytes) || !v41RoutedIDBytesOK(u, wBytes)) return 2;
        bufs[i] = g;
        bufs[8 + i] = u;
    }
    bufs[16] = B((void*)dX);
    bufs[17] = B((void*)dIds);
    bufs[18] = B(dMid);
    bufs[19] = B(dStatus);
    if (!v41RoutedIDBytesOK(bufs[16], xFloats * 4u) || !v41RoutedIDBytesOK(bufs[17], uint64_t(picks) * 12u) ||
        !v41RoutedIDBytesOK(bufs[18], midFloats * 4u) || !v41RoutedIDBytesOK(bufs[19], uint64_t(picks) * 16u)) return 2;
    for (int i = 0; i < 20; ++i) {
        if (i != 18) { if (v41BuffersAlias(bufs[18], bufs[i])) return 2; }
        if (i != 19) { if (v41BuffersAlias(bufs[19], bufs[i])) return 2; }
    }
    struct Push { int32_t outDim, inDim, picks, xStride; float limit; } pc{out, in, picks, xStride, limit};
    static_assert(sizeof(Push) == 20, "V4.1 routed fused gate/up push ABI must be four int32 values and one float32");
    const uint64_t groups = (uint64_t(out) + 1u) / 2u;
    if (groups > g_maxComputeWorkGroupCountX) return 2;
    if (!dispatch(g_kern[K_V41_ROUTED_GATE_UP_SWIGLU_ID], bufs, &pc, sizeof(pc), uint32_t(groups), uint32_t(picks), true,
                  bufs[19], uint32_t(picks) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// One Q3_K down GEMV with the checked BF16 publication of every pick's down row
// and their routed sum in its epilogue. One thread per output row, so the
// reduction order is the q3k id kernel's exactly. Returns the tail status codes.
int fvk_v41_routed_down_sum_f32(void* const* dDown, int slots, const void* dMid, void* dSum,
                                const void* dIds, void* dStatus, int out, int midDim, int picks, uint32_t site) {
    const int state = v41RoutedIDState(true);
    if (state != 0) return state;
    const bool validSite = site == 1 || site == 2 || (site >= 5 && site <= 27);
    if (!dDown || !validSite || slots <= 0 || slots > kV41RoutedIDSlots || picks <= 0 ||
        picks > kV41RoutedIDSlots || out <= 0 || midDim <= 0 || midDim % 256 != 0 ||
        !v41BF16CheckedCountSupported(out) || g_maxComputeWorkGroupCountX == 0) return 2;
    const uint64_t wBytes = uint64_t(out) * uint64_t(midDim / 256) * 110u;
    const uint64_t midFloats = uint64_t(midDim) * uint64_t(picks);
    const uint64_t sumFloats = uint64_t(out);
    if (wBytes > UINT32_MAX || midFloats > INT32_MAX || sumFloats > INT32_MAX) return 2;
    Buffer* bufs[12];
    for (int i = 0; i < kV41RoutedIDSlots; ++i) {
        Buffer* d = B(dDown[i < slots ? i : 0]);
        if (!v41RoutedIDBytesOK(d, wBytes)) return 2;
        bufs[i] = d;
    }
    bufs[8] = B((void*)dMid);
    bufs[9] = B((void*)dIds);
    bufs[10] = B(dSum);
    bufs[11] = B(dStatus);
    if (!v41RoutedIDBytesOK(bufs[8], midFloats * 4u) || !v41RoutedIDBytesOK(bufs[9], uint64_t(picks) * 12u) ||
        !v41RoutedIDBytesOK(bufs[10], sumFloats * 4u) || !v41RoutedIDBytesOK(bufs[11], uint64_t(picks) * 16u)) return 2;
    // Only the first `slots` weight bindings are live; the rest repeat slot 0,
    // so the distinctness check covers the live weights plus mid/ids/sum/status.
    for (int i = 0; i < slots; ++i) {
        for (int j = i + 1; j < slots; ++j) if (v41BuffersAlias(bufs[i], bufs[j])) return 2;
        for (int j = 8; j < 12; ++j) if (v41BuffersAlias(bufs[i], bufs[j])) return 2;
    }
    for (int i = 8; i < 12; ++i) {
        for (int j = i + 1; j < 12; ++j) if (v41BuffersAlias(bufs[i], bufs[j])) return 2;
    }
    struct Push { int32_t outDim, midDim; uint32_t site, picks; } pc{out, midDim, site, uint32_t(picks)};
    static_assert(sizeof(Push) == 16, "V4.1 routed fused down/sum push ABI must be two int32 values and two uint32 values");
    const uint64_t groups = (uint64_t(out) + 63u) / 64u;
    if (groups > g_maxComputeWorkGroupCountX) return 2;
    if (!dispatch(g_kern[K_V41_ROUTED_DOWN_SUM], bufs, &pc, sizeof(pc), uint32_t(groups), 1, true,
                  bufs[11], uint32_t(picks) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_q2k_embed_gather(void) {
    return g_ready && g_have_v41_q2k_embed_gather && g_kern[K_V41_Q2K_EMBED_GATHER].pipe != VK_NULL_HANDLE;
}

int fvk_v41_q2k_embed_gather_f32(const void* dTable, const void* dIds, void* dOut,
                                 int rows, int hidden, int vocab) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_q2k_embed_gather()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (rows <= 0 || hidden <= 0 || (hidden % 256) != 0 || vocab <= 0) return 2;
    const uint64_t rowBytes = uint64_t(hidden / 256) * 84u;
    const uint64_t total = uint64_t(rows) * uint64_t(hidden);
    if (uint64_t(vocab) * rowBytes > UINT32_MAX || total > INT32_MAX) return 2;
    Buffer* bufs[3] = {B(dTable), B(dIds), B(dOut)};
    const uint64_t bytes[3] = {uint64_t(vocab) * rowBytes, uint64_t(rows) * 4u, total * 4u};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            bytes[i] > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    auto aliases = [](const Buffer* a, const Buffer* b) {
        return a == b || a->buf == b->buf ||
            (a->mem == b->mem && a->memoryOffset < b->memoryOffset + b->bytes &&
             b->memoryOffset < a->memoryOffset + a->bytes);
    };
    if (aliases(bufs[2], bufs[0]) || aliases(bufs[2], bufs[1])) return 2;
    struct Push { int32_t rows, hidden, vocab; } pc{rows, hidden, vocab};
    static_assert(sizeof(Push) == 12, "V4.1 Q2_K embedding gather push ABI must be three int32 values");
    uint64_t groups = (total + 255u) / 256u;
    if (groups > g_maxComputeWorkGroupCountX) groups = g_maxComputeWorkGroupCountX;
    if (groups == 0) return 2;
    if (!dispatch(g_kern[K_V41_Q2K_EMBED_GATHER], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

static int v41BF16CheckedModuleABI(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41BF16CheckedABI(code) ? 1 : 0;
}

static int v41HaveBF16Checked(void) {
    return g_ready && g_have_v41_bf16_checked && g_kern[K_V41_BF16_CHECKED].pipe != VK_NULL_HANDLE;
}

static int v41BF16CheckedCountSupported(int count) {
    if (count <= 0 || uint64_t(count) > uint64_t(INT32_MAX) || g_maxComputeWorkGroupCountX == 0) return 0;
    const uint64_t bytes = uint64_t(count) * 4u;
    if (bytes > std::numeric_limits<size_t>::max() || bytes > std::numeric_limits<VkDeviceSize>::max() ||
        (g_maxBufferBytes > 0 && (bytes > g_maxBufferBytes || g_maxBufferBytes < 16))) return 0;
    // groups <= ceil(INT32_MAX/64), hence 64*groups <= 2^31. Every active
    // grid-stride index is <= 2^31-2, so its following increment cannot wrap.
    const uint64_t requested = (uint64_t(count) + 63u) / 64u;
    return std::min<uint64_t>(requested, g_maxComputeWorkGroupCountX) != 0;
}

static int v41BF16CheckedDispatch(const void* dInput, void* dOut, void* dStatus, int count, uint32_t site, bool validSite) {
    if (!g_ready) return 1;
    if (!v41HaveBF16Checked()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if ((g_batching && !g_stepBatch) || !validSite || !v41BF16CheckedCountSupported(count)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[3] = {B(dInput), B(dOut), B(dStatus)};
    const uint64_t required[3] = {bytes, bytes, 16};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    // Use conservative whole-allocation ranges, not only the prefix accessed
    // by this operation. All additions are checked above before comparison.
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    struct Push { uint32_t count, site; } pc{uint32_t(count), site};
    static_assert(sizeof(Push) == 8 && offsetof(Push, count) == 0 && offsetof(Push, site) == 4,
                  "checked BF16 push ABI must contain count/site uint32 values");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_BF16_CHECKED], bufs, &pc, sizeof(pc), groups, 1, true, bufs[2])) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

// Ordinary attention publication has a separate closed site policy. Sharing
// the integer module does not expose the final-norm or query witness APIs.
int fvk_v41_attention_bf16_module_abi(const void* data, size_t bytes) {
    return v41BF16CheckedModuleABI(data, bytes);
}
int fvk_have_v41_attention_bf16(void) { return v41HaveBF16Checked(); }
int fvk_v41_attention_bf16_state(void) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure) return 4;
    if ((g_batching && !g_stepBatch) || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 2;
    return 0;
}
int fvk_v41_attention_bf16_count_supported(int count) { return v41BF16CheckedCountSupported(count); }
int fvk_v41_attention_bf16_f32(const void* dInput, void* dOut, void* dStatus, int count, uint32_t site) {
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    return v41BF16CheckedDispatch(dInput, dOut, dStatus, count, site, site == 3 || site == 4);
}
// The ordinary publication policy mirrors compute.V41BF16PublishSite.
int fvk_v41_bf16_publish_f32(const void* dInput, void* dOut, void* dStatus, int count, uint32_t site) {
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    const bool validSite = site == 1 || site == 2 || (site >= 5 && site <= 27);
    return v41BF16CheckedDispatch(dInput, dOut, dStatus, count, site, validSite);
}

// Optional finite-only status uses the same serialized execution domain and
// checked status fill/barrier/dispatch/completion as ordinary BF16 publication.
int fvk_have_v41_f32_finite(void) {
    return g_ready && g_kern[K_V41_F32_FINITE].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_F32_FINITE].nbuf == 2 && g_kern[K_V41_F32_FINITE].pcsize == 8;
}
int fvk_v41_f32_finite_count_supported(int count) {
    return v41BF16CheckedCountSupported(count) && g_maxStorageBufferRange >= 16 &&
        uint64_t(count) * 4u <= g_maxStorageBufferRange;
}
int fvk_v41_f32_finite_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41F32FiniteABI(code) ? 1 : 0;
}
// Called only while the Go execution-owner mutex remains held, on interrupted
// acquisition, recording or retirement. Never a reset or a quiescence claim:
// fvk_free parks known handles; existing checked drain/context retirement owns
// eventual destruction. The entire native execution domain remains poisoned.
void fvk_v41_f32_finite_abandon(void) {
    if (g_submissionStatus == VK_SUCCESS) g_submissionStatus = VK_ERROR_UNKNOWN;
    g_v41SubmissionPendingFailure = true;
}
int fvk_v41_f32_finite(const void* dInput, void* dStatus, int count, uint32_t site) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_f32_finite()) return 3;
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    if ((site != 1 && site != 2) || !fvk_v41_f32_finite_count_supported(count)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[2] = {B(dInput), B(dStatus)};
    const uint64_t required[2] = {bytes, 16};
    for (int i = 0; i < 2; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    if (bufs[0] == bufs[1] || bufs[0]->buf == bufs[1]->buf ||
        (bufs[0]->mem == bufs[1]->mem && bufs[0]->memoryOffset < bufs[1]->memoryOffset + bufs[1]->bytes &&
         bufs[1]->memoryOffset < bufs[0]->memoryOffset + bufs[0]->bytes)) return 2;
    struct Push { uint32_t count, site; } pc{uint32_t(count), site};
    static_assert(sizeof(Push) == 8 && offsetof(Push, count) == 0 && offsetof(Push, site) == 4,
                  "finite status push ABI must contain count/site uint32 values");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_F32_FINITE], bufs, &pc, sizeof(pc), groups, 1, true, bufs[1])) {
        return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
    }
    return int(g_submissionStatus);
}

// One ordinary RoutedDown publication from a borrowed panel row. The existing
// ordinary publication state and checked module remain admission requirements.
int fvk_have_v41_routed_down_panel_row(void) {
    const char* enabled = std::getenv("FAK_V41_ROUTED_PANEL_ROW");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && v41HaveBF16Checked() &&
        g_kern[K_V41_ROUTED_DOWN_PANEL_ROW].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_ROUTED_DOWN_PANEL_ROW].nbuf == 3 && g_kern[K_V41_ROUTED_DOWN_PANEL_ROW].pcsize == 12;
}

int fvk_v41_routed_down_panel_row_shape(int rows, int count, int row) {
    if (rows < 1 || rows > 64 || row < 0 || row >= rows || !v41BF16CheckedCountSupported(count)) return 0;
    const uint64_t bytes = uint64_t(rows) * uint64_t(count) * 4u;
    return bytes <= INT32_MAX && bytes <= g_maxStorageBufferRange && (g_maxBufferBytes == 0 || bytes <= g_maxBufferBytes);
}

int fvk_v41_routed_down_panel_row_f32(const void* dInput, void* dOut, void* dStatus, int rows, int count, int row) {
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_down_panel_row()) return 3;
    if (!fvk_v41_routed_down_panel_row_shape(rows, count, row)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[3] = {B(dInput), B(dOut), B(dStatus)};
    const uint64_t required[3] = {uint64_t(rows) * bytes, bytes, 16};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    // Site 21 is the existing ordinary RoutedDown policy, never caller-controlled.
    struct Push { uint32_t count, site, rowOffset; } pc{uint32_t(count), 21u, uint32_t(uint64_t(row) * uint64_t(count))};
    static_assert(sizeof(Push) == 12 && offsetof(Push, count) == 0 && offsetof(Push, site) == 4 && offsetof(Push, rowOffset) == 8,
                  "routed down panel row push ABI is count, site and row offset");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_DOWN_PANEL_ROW], bufs, &pc, sizeof(pc), groups, 1, true, bufs[2])) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_routed_down_panel_chunk(void) {
    const char* enabled = std::getenv("FAK_V41_ROUTED_DOWN_CHUNK");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && v41HaveBF16Checked() &&
        g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK].nbuf == 3 && g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK].pcsize == 16;
}

int fvk_v41_routed_down_panel_chunk_shape(int rows, int count, int row, int chunk) {
    if (rows < 1 || rows > 64 || row < 0 || (chunk != 2 && chunk != 4 && chunk != 8) || row > rows - chunk || uint32_t(chunk) > g_maxComputeWorkGroupCountY || !v41BF16CheckedCountSupported(count)) return 0;
    const uint64_t bytes = uint64_t(rows) * uint64_t(count) * 4u;
    return bytes <= INT32_MAX && bytes <= g_maxStorageBufferRange && (g_maxBufferBytes == 0 || bytes <= g_maxBufferBytes);
}

int fvk_v41_routed_down_panel_chunk_f32(const void* dInput, void* dOut, void* dStatus, int rows, int count, int row, int chunk) {
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    if (!g_ready) return 1;
    if (!fvk_have_v41_routed_down_panel_chunk()) return 3;
    if (!fvk_v41_routed_down_panel_chunk_shape(rows, count, row, chunk)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[3] = {B(dInput), B(dOut), B(dStatus)};
    const uint64_t required[3] = {uint64_t(rows) * bytes, uint64_t(chunk) * bytes, uint64_t(chunk) * 16u};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    // Site 21 is the existing ordinary RoutedDown policy, never caller-controlled.
    struct Push { uint32_t count, site, rowOffset, rows; } pc{uint32_t(count), 21u, uint32_t(uint64_t(row) * uint64_t(count)), uint32_t(chunk)};
    static_assert(sizeof(Push) == 16 && offsetof(Push, count) == 0 && offsetof(Push, site) == 4 && offsetof(Push, rowOffset) == 8 && offsetof(Push, rows) == 12,
                  "routed down chunk push ABI is count, site, row offset and rows");
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_ROUTED_DOWN_PANEL_CHUNK], bufs, &pc, sizeof(pc), groups, uint32_t(chunk), true, bufs[2], uint32_t(chunk) * 16u)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_v41_routed_down_panel_chunk_module_abi(const void* data, size_t bytes) {
    if (!data || bytes == 0 || bytes % 4 != 0) return 0;
    const char* first = static_cast<const char*>(data);
    return v41RoutedDownChunkABI(std::vector<char>(first, first + bytes)) ? 1 : 0;
}

// Optional fixed-site 25 strict multiply plus checked BF16 publication.
int fvk_have_v41_index_scale_bf16(void) {
    const char* enabled = std::getenv("FAK_V41_INDEX_SCALE_BF16");
    return enabled && enabled[0] == '1' && enabled[1] == '\0' && g_ready && g_v41IndexerFloatControls && v41HaveBF16Checked() &&
        g_kern[K_V41_INDEX_SCALE_BF16].pipe != VK_NULL_HANDLE &&
        g_kern[K_V41_INDEX_SCALE_BF16].nbuf == 3 && g_kern[K_V41_INDEX_SCALE_BF16].pcsize == 8;
}

int fvk_v41_index_scale_bf16_count(int count) {
    if (!v41BF16CheckedCountSupported(count)) return 0;
    const uint64_t bytes = uint64_t(count) * 4u;
    return bytes <= INT32_MAX && bytes <= g_maxStorageBufferRange && (g_maxBufferBytes == 0 || bytes <= g_maxBufferBytes);
}

int fvk_v41_index_scale_bf16_f32(const void* dInput, void* dOut, void* dStatus, int count, float scale) {
    const int state = fvk_v41_attention_bf16_state();
    if (state != 0) return state;
    if (!g_ready) return 1;
    if (!fvk_have_v41_index_scale_bf16()) return 3;
    if (!std::isfinite(scale) || !fvk_v41_index_scale_bf16_count(count)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[3] = {B(dInput), B(dOut), B(dStatus)};
    const uint64_t required[3] = {bytes, bytes, 16};
    for (int i = 0; i < 3; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            required[i] > bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    for (int i = 0; i < 3; ++i) for (int j = i + 1; j < 3; ++j) {
        if (bufs[i] == bufs[j] || bufs[i]->buf == bufs[j]->buf ||
            (bufs[i]->mem == bufs[j]->mem && bufs[i]->memoryOffset < bufs[j]->memoryOffset + bufs[j]->bytes &&
             bufs[j]->memoryOffset < bufs[i]->memoryOffset + bufs[i]->bytes)) return 2;
    }
    struct Push { uint32_t count, scaleBits; } pc{uint32_t(count), 0};
    static_assert(sizeof(Push) == 8 && offsetof(Push, count) == 0 && offsetof(Push, scaleBits) == 4,
                  "index scale BF16 push ABI is count and binary32 scale bits");
    std::memcpy(&pc.scaleBits, &scale, sizeof(scale));
    const uint32_t groups = uint32_t(std::min<uint64_t>((uint64_t(count) + 63u) / 64u, g_maxComputeWorkGroupCountX));
    if (!dispatch(g_kern[K_V41_INDEX_SCALE_BF16], bufs, &pc, sizeof(pc), groups, 1, true, bufs[2])) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_v41_index_scale_bf16_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes); std::memcpy(code.data(), data, bytes);
    return v41IndexScaleBF16ABI(code) ? 1 : 0;
}

#ifdef FAK_V41_BF16_BOUNDARY_WITNESS
int fvk_v41_bf16_checked_module_abi(const void* data, size_t bytes) { return v41BF16CheckedModuleABI(data, bytes); }
int fvk_have_v41_bf16_checked(void) { return v41HaveBF16Checked(); }
int fvk_v41_bf16_checked_count_supported(int count) { return v41BF16CheckedCountSupported(count); }
int fvk_v41_bf16_checked_f32(const void* dInput, void* dOut, void* dStatus, int count, uint32_t site) {
    return v41BF16CheckedDispatch(dInput, dOut, dStatus, count, site, site == 1 || site == 2);
}

int fvk_have_v41_query_prefix(void) {
    // Availability is module identity, not transient healthy-busy state.
    return fvk_have_v41_bf16_checked() &&
        g_kern[K_RMSNORM].pipe != VK_NULL_HANDLE && g_kern[K_RMSNORM].nbuf == 3 && g_kern[K_RMSNORM].pcsize == 12 &&
        g_kern[K_MATMUL].pipe != VK_NULL_HANDLE && g_kern[K_MATMUL].nbuf == 3 && g_kern[K_MATMUL].pcsize == 12;
}

int fvk_v41_query_prefix_state(void) {
    if (g_submissionStatus != VK_SUCCESS) return int(g_submissionStatus);
    if (g_v41SubmissionPendingFailure) return int(VK_ERROR_UNKNOWN);
    if (g_batching && !g_stepBatch) return FVK_V41_QUERY_PREFIX_BUSY_BATCH;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return FVK_V41_QUERY_PREFIX_BUSY_RESTORE;
    return 0;
}

int fvk_v41_query_prefix_geometry_supported(int hidden, int rank) {
    if (!fvk_v41_bf16_checked_count_supported(hidden) || !fvk_v41_bf16_checked_count_supported(rank)) return 0;
    const uint64_t elements = uint64_t(hidden) * uint64_t(rank);
    const uint64_t bytes = elements * 4u;
    return elements <= uint64_t(INT32_MAX) && bytes <= std::numeric_limits<size_t>::max() &&
        bytes <= std::numeric_limits<VkDeviceSize>::max() &&
        g_maxStorageBufferRange > 0 && bytes <= g_maxStorageBufferRange &&
        (g_maxBufferBytes == 0 || bytes <= g_maxBufferBytes);
}

int fvk_v41_query_prefix_f32(const float* hostCollapse, const fvk_v41_query_prefix_buffers* a,
                            int hidden, int rank, float eps, uint32_t* failedStage) {
    V41StepSuspend suspend;
    if (failedStage) *failedStage = 0;
    const int state = fvk_v41_query_prefix_state();
    if (state != 0) return state;
    if (!fvk_have_v41_query_prefix()) return 3;
    if (!hostCollapse || !a || !failedStage || !std::isfinite(eps) || eps <= 0 ||
        !fvk_v41_query_prefix_geometry_supported(hidden, rank)) return 2;
    const uint64_t hb = uint64_t(hidden) * 4u, rb = uint64_t(rank) * 4u;
    Buffer* weights[3] = {B(a->weights[0]), B(a->weights[1]), B(a->weights[2])};
    Buffer* input = B(a->input);
    Buffer* values[7];
    Buffer* statuses[4];
    Buffer* all[15] = {weights[0], weights[1], weights[2], input};
    uint64_t sizes[15] = {hb, hb * uint64_t(rank), rb, hb};
    for (int i = 0; i < 7; ++i) {
        values[i] = B(a->values[i]); all[i+4] = values[i]; sizes[i+4] = i < 3 ? hb : rb;
    }
    for (int i = 0; i < 4; ++i) {
        statuses[i] = B(a->status[i]); all[i+11] = statuses[i]; sizes[i+11] = 16;
    }
    for (int i = 0; i < 15; ++i) {
        Buffer* b = all[i];
        if (!b || b->buf == VK_NULL_HANDLE || b->mem == VK_NULL_HANDLE || b->bytes < sizes[i] ||
            b->bytes > g_maxStorageBufferRange || b->bytes > std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset) return 2;
    }
    // Read-only weights may share storage. Every mutable range must be disjoint
    // from every other range, including weights and the initial staging owner.
    auto overlaps = [](Buffer* x, Buffer* y) {
        return x == y || x->buf == y->buf || (x->mem == y->mem &&
            x->memoryOffset < y->memoryOffset + y->bytes && y->memoryOffset < x->memoryOffset + x->bytes);
    };
    for (int i = 3; i < 15; ++i) for (int j = 0; j < i; ++j) if (overlaps(all[i], all[j])) return 2;
    *failedStage = 1; // initial staging/copy
    int stagingStatus = VK_SUCCESS;
    Buffer* stage = stagingBuffer(size_t(hb), &stagingStatus);
    if (!stage) {
        if (stagingStatus == VK_ERROR_DEVICE_LOST) {
            g_submissionStatus = VK_ERROR_DEVICE_LOST;
            g_v41SubmissionPendingFailure = true;
        }
        return stagingStatus == VK_SUCCESS ? 4 : stagingStatus;
    }
    if (stage->bytes > std::numeric_limits<VkDeviceSize>::max() - stage->memoryOffset) return 2;
    for (Buffer* b : all) if (overlaps(stage, b)) return 2;
    memcpy(g_stageMapped, hostCollapse, size_t(hb)); // no retained Go pointer
    VkCommandBuffer cmd = v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
    DescriptorSetRecord records[7]{};
    size_t recordCount = 0;
    auto recycle = [&]() {
        for (size_t i = recordCount; i > 0; --i) recycleDescriptorSet(records[i-1]);
        recordCount = 0;
    };
    auto discardUnsubmitted = [&]() {
        // This command has never reached the queue. Do not submit a partial
        // prefix as recovery, and do not call the generic batch flush.
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
        cmd = VK_NULL_HANDLE;
        recycle();
    };
    VkBufferCopy copy{0, 0, VkDeviceSize(hb)};
    vkCmdCopyBuffer(cmd, stage->buf, input->buf, 1, &copy);
    for (Buffer* status : statuses) vkCmdFillBuffer(cmd, status->buf, 0, 16, UINT32_MAX);
    recordComputeBarrier(cmd); // initial input transfer + every fresh status fill
    auto record = [&](Kernel& kernel, Buffer* x, Buffer* y, Buffer* z,
                      const void* push, uint32_t pushBytes, uint32_t groups) {
        if (recordCount >= 7 || kernel.nbuf != 3 || kernel.pcsize != pushBytes ||
            kernel.pipe == VK_NULL_HANDLE || kernel.layout == VK_NULL_HANDLE || groups == 0) return false;
        DescriptorSetRecord rec = acquireDescriptorSet(kernel, true);
        if (!rec.set) return false;
        // Hold the descriptor before any command references it; all seven
        // records survive until checked completion or process-lifetime quarantine.
        records[recordCount++] = rec;
        DescriptorSetRecord& owned = records[recordCount-1];
        Buffer* buffers[3] = {x, y, z};
        VkDescriptorBufferInfo info[3]{};
        VkWriteDescriptorSet write[3]{};
        owned.nbuf = 3;
        for (uint32_t i = 0; i < 3; ++i) {
            info[i].buffer = buffers[i]->buf;
            info[i].offset = 0;
            info[i].range = VK_WHOLE_SIZE;
            write[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            write[i].dstSet = owned.set;
            write[i].dstBinding = i;
            write[i].descriptorCount = 1;
            write[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            write[i].pBufferInfo = &info[i];
            owned.buffers[i] = buffers[i]->buf;
        }
        for (int i = 3; i < MAX_DISPATCH_BUFS; ++i) owned.buffers[i] = VK_NULL_HANDLE;
        vkUpdateDescriptorSets(g_dev, 3, write, 0, nullptr);
        if (recordCount > 1) recordComputeBarrier(cmd);
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, kernel.layout, 0, 1, &owned.set, 0, nullptr);
        vkCmdPushConstants(cmd, kernel.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pushBytes, push);
        dpDispatch(kernel);
        vkCmdDispatch(cmd, groups, 1, 1);
        return true;
    };
    struct BF16Push { uint32_t count, site; };
    struct NormPush { int rows, width; float eps; };
    struct MatmulPush { int out, in, rows; };
    static_assert(sizeof(BF16Push) == 8 && sizeof(NormPush) == 12 && sizeof(MatmulPush) == 12,
                  "query-prefix kernels keep their existing push layouts");
    BF16Push bf[4] = {{uint32_t(hidden),3}, {uint32_t(hidden),4}, {uint32_t(rank),5}, {uint32_t(rank),6}};
    NormPush normH{1,hidden,eps}, normR{1,rank,eps};
    const uint32_t gm = matmulDecodeGroups(rank, hidden);
    MatmulPush mat{rank,hidden,gm ? -1 : 1};
    const uint32_t gh = uint32_t(std::min<uint64_t>((uint64_t(hidden)+63)/64, g_maxComputeWorkGroupCountX));
    const uint32_t gr = uint32_t(std::min<uint64_t>((uint64_t(rank)+63)/64, g_maxComputeWorkGroupCountX));
    bool ok = false;
    do {
        *failedStage = 2;
        if (!record(g_kern[K_V41_BF16_CHECKED],input,values[0],statuses[0],&bf[0],8,gh)) break;
        *failedStage = 3;
        if (!record(g_kern[K_RMSNORM],values[0],weights[0],values[1],&normH,12,1)) break;
        *failedStage = 4;
        if (!record(g_kern[K_V41_BF16_CHECKED],values[1],values[2],statuses[1],&bf[1],8,gh)) break;
        *failedStage = 5;
        if (!record(g_kern[K_MATMUL],weights[1],values[2],values[3],&mat,12,gm ? gm : 1)) break;
        *failedStage = 6;
        if (!record(g_kern[K_V41_BF16_CHECKED],values[3],values[4],statuses[2],&bf[2],8,gr)) break;
        *failedStage = 7;
        if (!record(g_kern[K_RMSNORM],values[4],weights[2],values[5],&normR,12,1)) break;
        *failedStage = 8;
        if (!record(g_kern[K_V41_BF16_CHECKED],values[5],values[6],statuses[3],&bf[3],8,gr)) break;
        ok = true;
    } while (false);
    if (!ok) {
        // Checked descriptor allocation can report device loss and quarantine
        // the device even before this local command is submitted. Honor that
        // global uncertainty: retain this command and every prior descriptor.
        if (!g_v41SubmissionPendingFailure) discardUnsubmitted();
        return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
    }
    *failedStage = 9;
    dpBatchSubmit(); // one combined H2D+compute submission, not two
    tsCountUntimed(true);
    const bool completed = v41EndSubmitWaitChecked(cmd, true);
    // The helper owns command freeing on success/known-pre-submit failure.
    // On uncertainty neither command nor descriptors nor staging may be reused.
    if (!g_v41SubmissionPendingFailure) recycle();
    if (!completed) {
        g_transferCountersValid = false;
        return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
    }
    if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, hb)) g_transferCountersValid = false;
    *failedStage = 0;
    return 0;
}

int fvk_v41_bf16_boundary_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41BF16BoundaryABI(code) ? 1 : 0;
}

int fvk_have_v41_bf16_boundary(void) {
    return g_ready && g_have_v41_bf16_boundary && g_kern[K_V41_BF16_BOUNDARY].pipe != VK_NULL_HANDLE;
}

int fvk_v41_bf16_boundary_f32(const void* dInput, void* dOut, int count) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_bf16_boundary()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (count <= 0) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[2] = {B(dInput), B(dOut)};
    for (int i = 0; i < 2; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            bytes > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    if (bufs[0] == bufs[1] || bufs[0]->buf == bufs[1]->buf ||
        (bufs[0]->mem == bufs[1]->mem && bufs[0]->memoryOffset < bufs[1]->memoryOffset + bufs[1]->bytes &&
         bufs[1]->memoryOffset < bufs[0]->memoryOffset + bufs[0]->bytes)) return 2;
    const uint32_t pc = uint32_t(count);
    uint32_t groups = (uint32_t(count) + 63u) / 64u;
    if (groups > g_maxComputeWorkGroupCountX) groups = g_maxComputeWorkGroupCountX;
    if (groups == 0) return 2;
    if (!dispatch(g_kern[K_V41_BF16_BOUNDARY], bufs, &pc, sizeof(pc), groups)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}
#endif

int fvk_v41_scale_exact_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41LogitScaleABI(code) ? 1 : 0;
}

int fvk_have_v41_scale_exact(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_scale_exact &&
        g_kern[K_V41_SCALE_EXACT].pipe != VK_NULL_HANDLE;
}

int fvk_v41_scale_exact_count_supported(int count) {
    if (!fvk_have_v41_scale_exact() || count <= 0) return 0;
    // No grid-stride loop: an unrepresentable dispatch is refused, never
    // clamped into an unwritten output suffix.
    const uint64_t groups = (uint64_t(count) + 63u) / 64u;
    return groups != 0 && groups <= g_maxComputeWorkGroupCountX;
}

int fvk_v41_scale_exact_f32(const void* dInput, void* dOut, int count, float scale) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_scale_exact()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (!std::isfinite(scale) || !fvk_v41_scale_exact_count_supported(count)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[2] = {B(dInput), B(dOut)};
    for (int i = 0; i < 2; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            bytes > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    if (bufs[0] == bufs[1] || bufs[0]->buf == bufs[1]->buf ||
        (bufs[0]->mem == bufs[1]->mem && bufs[0]->memoryOffset < bufs[1]->memoryOffset + bufs[1]->bytes &&
         bufs[1]->memoryOffset < bufs[0]->memoryOffset + bufs[0]->bytes)) return 2;
    struct Push { uint32_t count; float scale; } pc{uint32_t(count), scale};
    static_assert(sizeof(Push) == 8, "V4.1 exact scale push ABI must be uint32 and float32");
    const uint32_t groups = uint32_t((uint64_t(count) + 63u) / 64u);
    if (!dispatch(g_kern[K_V41_SCALE_EXACT], bufs, &pc, sizeof(pc), groups)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

#ifdef FAK_V41_LOGIT_SCALE_WITNESS
int fvk_v41_logit_scale_module_abi(const void* data, size_t bytes) {
    if (!data || bytes < 20 || bytes > 1024 * 1024 || bytes % 4 != 0) return 0;
    std::vector<char> code(bytes);
    memcpy(code.data(), data, bytes);
    return v41LogitScaleABI(code) ? 1 : 0;
}

int fvk_have_v41_logit_scale(void) {
    return g_ready && g_v41IndexerFloatControls && g_have_v41_logit_scale &&
        g_kern[K_V41_LOGIT_SCALE].pipe != VK_NULL_HANDLE;
}

int fvk_v41_logit_scale_count_supported(int count) {
    if (!fvk_have_v41_logit_scale() || count <= 0) return 0;
    // This shader has no grid-stride loop. Reject an unrepresentable dispatch;
    // clamping would publish an unwritten output suffix.
    const uint64_t groups = (uint64_t(count) + 63u) / 64u;
    return groups != 0 && groups <= g_maxComputeWorkGroupCountX;
}

int fvk_v41_logit_scale_f32(const void* dInput, void* dOut, int count, float scale) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_logit_scale()) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (!std::isfinite(scale) || !fvk_v41_logit_scale_count_supported(count)) return 2;
    const uint64_t bytes = uint64_t(count) * 4u;
    Buffer* bufs[2] = {B(dInput), B(dOut)};
    for (int i = 0; i < 2; ++i) {
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            bytes > bufs[i]->bytes ||
            bufs[i]->bytes > std::numeric_limits<VkDeviceSize>::max() - bufs[i]->memoryOffset) return 2;
    }
    if (bufs[0] == bufs[1] || bufs[0]->buf == bufs[1]->buf ||
        (bufs[0]->mem == bufs[1]->mem && bufs[0]->memoryOffset < bufs[1]->memoryOffset + bufs[1]->bytes &&
         bufs[1]->memoryOffset < bufs[0]->memoryOffset + bufs[0]->bytes)) return 2;
    struct Push { uint32_t count; float scale; } pc{uint32_t(count), scale};
    static_assert(sizeof(Push) == 8, "V4.1 logit scale push ABI must be uint32 and float32");
    const uint32_t groups = uint32_t((uint64_t(count) + 63u) / 64u);
    if (!dispatch(g_kern[K_V41_LOGIT_SCALE], bufs, &pc, sizeof(pc), groups)) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}
#endif

void fvk_swiglu_f32(const void* dG, const void* dU, void* dY, int n) {
    fvk_swiglu_limit_f32(dG, dU, dY, n, 0.0f);
}

void fvk_swiglu_limit_f32(const void* dG, const void* dU, void* dY, int n, float limit) {
    struct { int n; float limit; } pc{n, limit};
    Buffer* bufs[3] = {B((void*)dG), B((void*)dU), B(dY)};
    const bool limited = std::isfinite(limit) && limit > 0.0f;
    if (limited) {
        if (g_submissionStatus != VK_SUCCESS) return;
        if (!g_ready || g_kern[K_SWIGLU].pipe == VK_NULL_HANDLE || n <= 0 ||
            g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) {
            g_submissionStatus = VK_ERROR_UNKNOWN;
            return;
        }
        const uint64_t bytes = uint64_t(n) * sizeof(float);
        for (Buffer* b : bufs) {
            if (!b || b->buf == VK_NULL_HANDLE || b->mem == VK_NULL_HANDLE || bytes > b->bytes ||
                b->bytes > std::numeric_limits<VkDeviceSize>::max() - b->memoryOffset) {
                g_submissionStatus = VK_ERROR_UNKNOWN;
                return;
            }
        }
    }
    const bool accepted = dispatch(g_kern[K_SWIGLU], bufs, &pc, sizeof(pc), ((uint32_t)n + 255) / 256, 1, limited);
    if (limited && !accepted && g_submissionStatus == VK_SUCCESS) g_submissionStatus = VK_ERROR_UNKNOWN;
}

void fvk_swiglu_matmul_add_f32(const void* dW, const void* dG, const void* dU,
                               void* dY, int out, int in, int P) {
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[4] = {B((void*)dW), B((void*)dG), B((void*)dU), B(dY)};
    dispatch(g_kern[K_SWIGLU_MATMUL_ADD], bufs, &pc, sizeof(pc), (uint32_t)P);
}

void fvk_add_f32(void* dDst, const void* dSrc, int n) {
    struct { int n; } pc{n};
    Buffer* bufs[2] = {B(dDst), B((void*)dSrc)};
    dispatch(g_kern[K_ADD], bufs, &pc, sizeof(pc), ((uint32_t)n + 255) / 256);
}

void fvk_add_bias_f32(void* dDst, const void* dBias, int rows, int width) {
    struct { int rows, width; } pc{rows, width};
    Buffer* bufs[2] = {B(dDst), B((void*)dBias)};
    uint32_t total = (uint32_t)rows * (uint32_t)width;
    dispatch(g_kern[K_ADD_BIAS], bufs, &pc, sizeof(pc), (total + 255) / 256);
}

void fvk_attention_f32(const void* dQ, const void* dK, const void* dV, void* dOut,
                       int nPos, int nH, int nKV, int hd, float scale) {
    if (g_v41SubmissionPendingFailure) return;
    // Experimental, default-off selector for #12535. Exact "1" is the only
    // admitted value; every other value preserves the production control path.
    // The variable is read at each call so isolated test subprocesses can select
    // the candidate without widening the public C or Go backend APIs.
    const char* splitEnv = std::getenv("FAK_VULKAN_ATTENTION_CONTEXT_SPLIT");
    const bool contextSplit = splitEnv && splitEnv[0] == '1' && splitEnv[1] == '\0';
    // This is one query at the end of the supplied prefix. The causal bound
    // equals nPos; no sliding-window policy is supplied by this C entry point.
    AttentionPush pc{nPos, nH, nKV, hd, scale, 0, 1, 1, 0};
    Buffer* out = B(dOut);
    Buffer* bufs[5] = {B((void*)dQ), B((void*)dK), B((void*)dV), out, out};
    if (!contextSplit) {
        dispatch(g_kern[K_ATTENTION], bufs, &pc, sizeof(pc), (uint32_t)nH);
        return;
    }

    constexpr uint64_t tileSize = 256u;
    uint64_t tileCount = nPos > 0 ? ((uint64_t)nPos + tileSize - 1u) / tileSize : 1u;
    uint64_t rowFloats = hd > 0 ? (uint64_t)hd + 2u : 0u;
    if (nPos < 0 || nH <= 0 || nKV <= 0 || nH % nKV != 0 || hd <= 0 || hd > 1024 ||
        tileCount > UINT32_MAX || (uint64_t)nH > UINT32_MAX / tileCount ||
        (uint64_t)nH * tileCount > 65535u || rowFloats > SIZE_MAX / sizeof(float) ||
        (uint64_t)nH * tileCount > (SIZE_MAX / sizeof(float)) / rowFloats) {
        g_submissionStatus = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return;
    }
    uint64_t scratchFloats = (uint64_t)nH * tileCount * rowFloats;
    Buffer* scratch = B(fvk_malloc((size_t)(scratchFloats * sizeof(float))));
    if (!scratch) {
        g_submissionStatus = VK_ERROR_OUT_OF_DEVICE_MEMORY;
        return;
    }
    bufs[4] = scratch;
    pc.mode = 1;
    pc.tileCount = (int)tileCount;
    dispatch(g_kern[K_ATTENTION], bufs, &pc, sizeof(pc), (uint32_t)((uint64_t)nH * tileCount));
    // One-shot dispatch executes immediately, so do not let a merge submission
    // obscure its failure. In batch mode both phases are recorded into the same
    // ordered command buffer and the eventual flush preserves submission status.
    if (!g_batching && g_submissionStatus != VK_SUCCESS) {
        fvk_free(scratch);
        return;
    }
    pc.mode = 2;
    dispatch(g_kern[K_ATTENTION], bufs, &pc, sizeof(pc), (uint32_t)nH);
    fvk_free(scratch);
}

int fvk_argmax_f32(const void* dLogits, int n) {
    Buffer* idx = (Buffer*)fvk_malloc(sizeof(int));
    struct { int n; } pc{n};
    Buffer* bufs[2] = {B((void*)dLogits), idx};
    // The logits producer and reduction share the active command buffer. The
    // ordinary dispatch dependency orders their accesses; only the host index
    // read needs a fence. Flush still ends the batch, as this entrypoint did before.
    dispatch(g_kern[K_ARGMAX], bufs, &pc, sizeof(pc), 1);
    if (g_batching) batchFlush();
    int h = 0;
    copyDeviceToHost(&h, idx, sizeof(int));
    fvk_free(idx);
    return h;
}

} // extern "C"

extern "C" int fvk_debug_gdn_prefill_tiled_available(void) {
    return g_ready && g_have_gdn_prefill_tiled;
}

extern "C" int fvk_debug_gdn_verify_tiled_available(void) {
    return g_ready && g_have_gdn_verify_tiled;
}

extern "C" void fvk_debug_gdn_verify_tiled_mode(int mode) {
    g_gdn_verify_mode = (mode == -1 || mode == 1) ? mode : 0;
}

extern "C" void fvk_debug_gdn_verify_tiled_reset(void) {
    g_gdn_verify_tiled_calls = 0;
    g_gdn_verify_scalar_calls = 0;
}

extern "C" void fvk_debug_gdn_verify_tiled_snapshot(uint64_t* verify, uint64_t* scalar) {
    if (verify) *verify = g_gdn_verify_tiled_calls;
    if (scalar) *scalar = g_gdn_verify_scalar_calls;
}

extern "C" int fvk_debug_gdn_verify_tiled_verdict(float* deviation) {
    if (deviation) *deviation = g_gdn_verify_deviation;
    return g_gdn_verify_verdict;
}

extern "C" int fvk_debug_gdn_tiled_verdict(float* deviation) {
    if (deviation) *deviation = g_gdn_tiled_deviation;
    return g_gdn_tiled_verdict;
}

extern "C" int fvk_debug_gdn_verify_tiled_checked(void) {
    return g_gdn_verify_checked;
}

extern "C" void fvk_debug_gdn_verify_tiled_force_disagree(int enabled) {
    g_gdn_verify_force_disagree = enabled ? 1 : 0;
    // Re-run the cached self-check so the forced-disagreement hook takes effect
    // immediately; the forced path never dispatches a kernel.
    gdnVerifySelfCheck();
}

extern "C" void fvk_debug_gdn_prefill_tiled_mode(int mode) {
    g_gdn_prefill_mode = (mode == -1 || mode == 1) ? mode : 0;
}

extern "C" void fvk_debug_gdn_prefill_tiled_reset(void) {
    g_gdn_prefill_tiled_calls = 0;
    g_gdn_prefill_scalar_calls = 0;
    g_gdn_verify_tiled_calls = 0;
    g_gdn_optout_calls = 0;
}

extern "C" uint64_t fvk_debug_gdn_optout_calls(void) {
    return g_gdn_optout_calls;
}

extern "C" void fvk_debug_gdn_prefill_tiled_snapshot(uint64_t* tiled, uint64_t* scalar) {
    if (tiled) *tiled = g_gdn_prefill_tiled_calls;
    if (scalar) *scalar = g_gdn_prefill_scalar_calls;
}

extern "C" int fvk_qwen35_gdn_preprojected_f32(
    const void* mixed, const void* z, const void* beta, const void* alpha,
    const void* conv1d, const void* a_log, const void* dt_bias, const void* norm,
    void* conv_state, void* recurrent_state, void* core,
    int tokens, int conv_dim, int n_k, int n_v, int k_hd, int v_hd, int kernel, float eps) {
    if (!g_ready || !mixed || !z || !beta || !alpha || !conv1d || !a_log || !dt_bias || !norm || !conv_state || !recurrent_state || !core)
        return 1;
    if (tokens <= 0 || conv_dim <= 0 || n_k <= 0 || n_v <= 0 || k_hd <= 0 || v_hd <= 0 || kernel <= 0 || n_v % n_k != 0 || v_hd > 1024)
        return 2;
    Buffer* conv_out = gdnConvOutScratch((size_t)tokens * conv_dim * sizeof(float));
    if (!conv_out) return 3;
    const char* optIn = std::getenv("FAK_VULKAN_GDN_PREFILL_TILED");
    const bool requestedByEnv = optIn && optIn[0] == '1' && optIn[1] == '\0';
    const bool optOut = optIn && optIn[0] == '0' && optIn[1] == '\0';
    const bool productionGeometry = n_k == 16 && n_v == 48 && k_hd == 128 && v_hd == 128 && conv_dim == 10240;
    // Prefill-class (>=8) route. Every candidate selection requires a true
    // self-check verdict. The default follows the flip constant; `=0` forces stock.
    constexpr bool kGdnTiledDefaultFlipEarned = false; // pending the whole-model logit oracle and Strix A/B, halo-gdn-prefill-tiled
    const bool tiledRequested = !optOut && (g_gdn_prefill_mode == 1 ||
        (g_gdn_prefill_mode == -1 && (requestedByEnv || kGdnTiledDefaultFlipEarned)));
    const bool tiled = tiledRequested && g_have_gdn_prefill_tiled && g_gdn_tiled_verdict != 0 &&
        tokens >= 8 && (uint32_t)tokens <= g_gdn_prefill_max_groups_y && productionGeometry;
    // Verify-width (1..7 token) route. `FAK_VULKAN_GDN_PREFILL_TILED=0` forces the
    // stock kernel; otherwise the candidate still requires a true self-check verdict.
    constexpr bool kGdnVerifyDefaultFlipEarned = false; // pending the whole-model logit oracle and Strix A/B, halo-gdn-prefill-tiled
    const bool verifyRequested = !optOut && g_gdn_verify_mode != 0 &&
        (g_gdn_prefill_mode == 1 || g_gdn_verify_mode == 1 || requestedByEnv ||
         kGdnVerifyDefaultFlipEarned);
    const bool verify = verifyRequested && g_have_gdn_verify_tiled && g_gdn_verify_verdict &&
        tokens >= 1 && tokens < 8 && productionGeometry;
    if (optOut && productionGeometry) ++g_gdn_optout_calls;
    Buffer* readout = nullptr;
    if (tiled || verify) {
        readout = gdnPrefillReadoutScratch((size_t)tokens * n_v * v_hd * sizeof(float));
        // Once selected, resource failure is an error, never a silent replay
        // after a partially updated recurrent or convolution state.
        if (!readout) return 4;
    }
    struct ConvPC { int tokens, conv_dim, kernel; } cpc{tokens, conv_dim, kernel};
    Buffer* cbufs[4] = {B((void*)mixed), B((void*)conv1d), B(conv_state), conv_out};
    if (!dispatch(g_kern[K_QWEN35_GDN_CONV], cbufs, &cpc, sizeof(cpc), (uint32_t)((conv_dim + 63) / 64))) return 5;
    struct RecPC { int tokens, conv_dim, n_k, n_v, k_hd, v_hd; float eps; } rpc{tokens, conv_dim, n_k, n_v, k_hd, v_hd, eps};
    Buffer* rbufs[9] = {conv_out, B((void*)z), B((void*)beta), B((void*)alpha), B((void*)a_log), B((void*)dt_bias), B((void*)norm), B(recurrent_state), B(core)};
    if (tiled) {
        rbufs[8] = readout;
        if (!dispatch(g_kern[K_QWEN35_GDN_PREFILL_TILED], rbufs, &rpc, sizeof(rpc), 4, (uint32_t)n_v)) return 6;
        // dispatch inserts the compute write/read barrier when recording a
        // batch; one-shot dispatches wait for completion before this pass.
        rbufs[0] = readout;
        rbufs[8] = B(core);
        if (!dispatch(g_kern[K_QWEN35_GDN_PREFILL_NORM], rbufs, &rpc, sizeof(rpc), (uint32_t)n_v, (uint32_t)tokens)) return 7;
        ++g_gdn_prefill_tiled_calls;
    } else if (verify) {
        // The verify-width variant writes the unnormalised readout to the
        // scratch readout, then the shared norm pass finalises `core`.
        rbufs[8] = readout;
        if (!dispatch(g_kern[K_QWEN35_GDN_VERIFY_TILED], rbufs, &rpc, sizeof(rpc),
                      (uint32_t)((v_hd + 7) / 8), (uint32_t)n_v)) return 6;
        rbufs[0] = readout;
        rbufs[8] = B(core);
        if (!dispatch(g_kern[K_QWEN35_GDN_PREFILL_NORM], rbufs, &rpc, sizeof(rpc), (uint32_t)n_v, (uint32_t)tokens)) return 7;
        // The unified candidate counter tracks either register-resident route; the
        // verify counter is the finer per-width breakdown surfaced by the debug seam.
        ++g_gdn_prefill_tiled_calls;
        ++g_gdn_verify_tiled_calls;
    } else {
        if (!dispatch(g_kern[K_QWEN35_GDN_RECURRENT], rbufs, &rpc, sizeof(rpc), (uint32_t)n_v)) return 8;
        ++g_gdn_prefill_scalar_calls;
        if (tokens >= 1 && tokens < 8 && productionGeometry) ++g_gdn_verify_scalar_calls;
    }
    return 0;
}

extern "C" int fvk_qwen35_gdn_conv_tiled_transpose_f32(
    const void* mixed, const void* conv1d, void* conv_state, void* conv_out,
    int tokens, int conv_dim, int kernel) {
    if (!g_ready || !mixed || !conv1d || !conv_state || !conv_out)
        return 1;
    if (tokens <= 0 || conv_dim <= 0 || kernel <= 0)
        return 2;
    if ((conv_dim * (int)sizeof(float)) % 32 != 0)
        return 2;
    uint64_t mixedBytes = (uint64_t)tokens * conv_dim * sizeof(float);
    uint64_t conv1dBytes = (uint64_t)conv_dim * kernel * sizeof(float);
    uint64_t stateBytes = (uint64_t)(kernel > 1 ? kernel - 1 : 0) * conv_dim * sizeof(float);
    uint64_t outBytes = (uint64_t)tokens * conv_dim * sizeof(float);
    if (B((void*)mixed)->bytes < mixedBytes ||
        B((void*)conv1d)->bytes < conv1dBytes ||
        (stateBytes > 0 && B(conv_state)->bytes < stateBytes) ||
        B(conv_out)->bytes < outBytes) {
        return 2;
    }
    struct ConvPC { int tokens, conv_dim, kernel; } cpc{tokens, conv_dim, kernel};
    Buffer* cbufs[4] = {B((void*)mixed), B((void*)conv1d), B(conv_state), B(conv_out)};
    dispatch(g_kern[K_QWEN35_GDN_CONV], cbufs, &cpc, sizeof(cpc), (uint32_t)((conv_dim + 63) / 64));
    return (int)g_submissionStatus;
}

extern "C" int fvk_glm_kda_step_f32(
    void* state, const void* q, const void* k, const void* value,
    const void* alpha, const void* beta, void* output, int heads, int variant) {
    static constexpr uint64_t D = 128;
    if (!g_ready) return 1;
    if (!g_have_glm_kda_wave32) return 3;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (heads <= 0 || (variant != 0 && variant != 1)) return 2;
    const void* ptrs[] = {state, q, k, value, alpha, beta, output};
    for (const void* ptr : ptrs) if (!ptr) return 2;
    for (size_t i = 0; i < 7; ++i)
        for (size_t j = i + 1; j < 7; ++j)
            if (ptrs[i] == ptrs[j]) return 2;
    uint64_t h = (uint64_t)heads;
    if (h > UINT64_MAX / (D * D * sizeof(float))) return 2;
    uint64_t stateBytes = h * D * D * sizeof(float);
    uint64_t vectorBytes = h * D * sizeof(float);
    uint64_t scalarBytes = h * sizeof(float);
    if (stateBytes > B(state)->bytes || vectorBytes > B((void*)q)->bytes ||
        vectorBytes > B((void*)k)->bytes || vectorBytes > B((void*)value)->bytes ||
        scalarBytes > B((void*)alpha)->bytes || scalarBytes > B((void*)beta)->bytes ||
        vectorBytes > B(output)->bytes) return 2;
    struct { int heads; } pc{heads};
    Buffer* bufs[7] = {B(state), B((void*)q), B((void*)k), B((void*)value),
                       B((void*)alpha), B((void*)beta), B(output)};
    KId id = variant == 0 ? K_GLM_KDA_REREAD : K_GLM_KDA_WAVE32;
    dispatch(g_kern[id], bufs, &pc, sizeof(pc), (uint32_t)heads);
    return (int)g_submissionStatus;
}
// Panel ABI uses signed GLSL indices; validate products before narrowing.
static bool panelFits(const void* ptr, int tokens, int heads, int dim) {
    if (!ptr || tokens <= 0 || heads <= 0 || dim <= 0) return false;
    uint64_t rows = (uint64_t)tokens * (uint64_t)heads;
    if (rows > 2147483647u / (uint64_t)dim) return false;
    return rows * (uint64_t)dim * sizeof(float) <= B(ptr)->bytes;
}
extern "C" int fvk_qwen35_split_qg_panel_f32(const void* qg, void* q, void* gate,
    int tokens, int nHeads, int headDim) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (headDim <= 0 || headDim > 1073741823 || !panelFits(qg,tokens,nHeads,2*headDim) ||
        !panelFits(q,tokens,nHeads,headDim) || !panelFits(gate,tokens,nHeads,headDim) ||
        qg == q || qg == gate || q == gate) return 2;
    struct { int tokens, heads, hd; } pc{tokens,nHeads,headDim};
    Buffer* bufs[] = {B(qg),B(q),B(gate)};
    dispatch(g_kern[K_QWEN35_SPLIT_QG_PANEL],bufs,&pc,sizeof(pc),
        (uint32_t)(((uint64_t)tokens*nHeads*headDim+255)/256));
    return (int)g_submissionStatus;
}
// Tiles a [tokens, nV*headDim] GDN core panel into the GGUF ssm_out input-column order: dst
// head c reads src head (c % nK) * (nV / nK) + c / nK. The optional kernel is one dispatch;
// without its module the same permutation records one multi-region device copy.
extern "C" int fvk_qwen35_value_head_tile_f32(const void* src, void* dst,
    int tokens, int nK, int nV, int headDim) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (nK <= 0 || nV <= 0 || nV % nK != 0 || src == dst ||
        !panelFits(src, tokens, nV, headDim) || !panelFits(dst, tokens, nV, headDim)) return 2;
    if (g_have_qwen35_value_head_tile) {
        struct { int tokens, n_k, n_v, head_dim; } pc{tokens, nK, nV, headDim};
        Buffer* bufs[] = {B((void*)src), B(dst)};
        uint64_t groups = ((uint64_t)tokens * nV * headDim + 255) / 256;
        uint64_t cap = 65535;
        if (g_maxComputeWorkGroupCountX != 0 && g_maxComputeWorkGroupCountX < cap) cap = g_maxComputeWorkGroupCountX;
        if (groups > cap) groups = cap;
        if (!dispatch(g_kern[K_QWEN35_VALUE_HEAD_TILE], bufs, &pc, sizeof(pc), (uint32_t)groups)) return 3;
        return (int)g_submissionStatus;
    }
    const int ratio = nV / nK;
    const size_t headBytes = (size_t)headDim * sizeof(float);
    // panelFits already bounds tokens*nV*headDim to signed shader indexing.
    const int count = tokens * nV;
    if (size_t(count) > std::numeric_limits<size_t>::max() / sizeof(intptr_t)) return 2;
    intptr_t* ids = (intptr_t*)std::malloc(size_t(count) * sizeof(intptr_t));
    if (!ids) return (int)VK_ERROR_OUT_OF_HOST_MEMORY;
    for (int t = 0; t < tokens; ++t) {
        for (int c = 0; c < nV; ++c) {
            const int h = (c % nK) * ratio + c / nK;
            ids[t * nV + c] = t * nV + h;
        }
    }
    const int status = fvk_d2d_gather_rows(dst, src, ids, count, headBytes);
    std::free(ids);
    return status;
}
extern "C" int fvk_qwen35_partial_rope_panel_f32(const void* q, const void* k,
    void* qOut, void* kOut, int tokens, int startPos, int nQHeads,
    int nKHeads, int headDim, int rotaryDim, double theta) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!panelFits(q,tokens,nQHeads,headDim) || !panelFits(k,tokens,nKHeads,headDim) ||
        !panelFits(qOut,tokens,nQHeads,headDim) || !panelFits(kOut,tokens,nKHeads,headDim) ||
        startPos<0 || (uint64_t)startPos+tokens>2147483647u || rotaryDim<0 ||
        rotaryDim>headDim || rotaryDim%2 || !std::isfinite(theta) || theta<=0 ||
        theta>3.402823466e38 || qOut==q || qOut==k || kOut==q || kOut==k || qOut==kOut)
        return 2;
    uint64_t count = (uint64_t)tokens*((uint64_t)nQHeads+nKHeads)*headDim;
    if (count>2147483647u) return 2;
    const float tableTheta = (float)theta;
    if (!std::isfinite(tableTheta) || tableTheta <= 0) return 2;
    const int scalarReference = environmentFlagEnabled(
        "FAK_VULKAN_QWEN35_PARTIAL_ROPE_SCALAR_REFERENCE");
    Buffer* table = B((void*)q); // Valid fifth descriptor; scalar arm never reads it.
    if (!scalarReference && rotaryDim > 0) {
        const uint64_t endPosition = (uint64_t)startPos + (uint64_t)tokens;
        if (endPosition > (uint64_t)std::numeric_limits<size_t>::max()) return 2;
        table = partialRoPETable(tableTheta, rotaryDim, (size_t)endPosition);
        if (!table) return 3;
    }
    struct { int tokens,startPos,qHeads,kHeads,hd,rotary; float theta; int scalarReference; }
        pc{tokens,startPos,nQHeads,nKHeads,headDim,rotaryDim,tableTheta,scalarReference};
    Buffer* bufs[] = {B(q),B(k),B(qOut),B(kOut),table};
    dispatch(g_kern[K_QWEN35_PARTIAL_ROPE_PANEL],bufs,&pc,sizeof(pc),(uint32_t)((count+255)/256));
    return (int)g_submissionStatus;
}
extern "C" int fvk_qwen35_causal_attention_panel_f32(const void* q, const void* k,
    const void* v, void* out, int tokens, int prefix, int nHeads,
    int nKVHeads, int headDim, float scale) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (prefix<0 || tokens<=0 || (uint64_t)tokens+prefix>2147483647u ||
        headDim>1024 || nKVHeads<=0 || nHeads%nKVHeads || !std::isfinite(scale) ||
        !panelFits(q,tokens,nHeads,headDim) || !panelFits(out,tokens,nHeads,headDim) ||
        !panelFits(k,tokens+prefix,nKVHeads,headDim) || !panelFits(v,tokens+prefix,nKVHeads,headDim) ||
        out==q || out==k || out==v) return 2;
    const char* splitArm = std::getenv("FAK_VULKAN_QWEN35_SPLIT_KV");
    const bool splitForced = splitArm && splitArm[0] == '1' && splitArm[1] == '\0';
    const int groupSize = nHeads / nKVHeads;
    const uint64_t nPos = (uint64_t)prefix + 1u;
    const bool splitShape = tokens == 1 && groupSize <= 8 && headDim >= 64 &&
        headDim <= 256 && headDim % 64 == 0 && g_maxComputeWorkGroupCountX != 0 &&
        (uint64_t)nHeads <= g_maxComputeWorkGroupCountX &&
        (uint64_t)nKVHeads <= g_maxComputeWorkGroupCountY;
    if (splitForced && splitShape) {
        if (!g_have_attention_split_kv) {
            return 3;
        } else {
            int chunk = (int)((nPos + 127u) / 128u);
            if (chunk < 256) chunk = 256;
            const int splits = (int)((nPos + (uint64_t)chunk - 1u) / (uint64_t)chunk);
            const uint64_t scratchFloats = (uint64_t)splits * (uint64_t)nHeads *
                ((uint64_t)headDim + 2u);
            if (splits <= 128 && (uint64_t)splits <= g_maxComputeWorkGroupCountX &&
                scratchFloats <= (uint64_t)SIZE_MAX / sizeof(float)) {
                Buffer* scratch = B(fvk_malloc((size_t)(scratchFloats * sizeof(float))));
                if (!scratch) {
                    return 3;
                } else {
                    struct { int nPos,nH,nKV,hd; float scale; int mode,splits,chunk; }
                        splitPC{(int)nPos,nHeads,nKVHeads,headDim,scale,0,splits,chunk};
                    Buffer* splitBufs[] = {B((void*)q),B((void*)k),B((void*)v),B(out),scratch};
                    const bool partial = dispatch(g_kern[K_ATTENTION_SPLIT_KV],splitBufs,
                        &splitPC,sizeof(splitPC),(uint32_t)splits,(uint32_t)nKVHeads);
                    if (!partial) {
                        fvk_free(scratch);
                        return 3;
                    } else {
                        if (splits > 1) {
                            splitPC.mode = 1;
                            if (!dispatch(g_kern[K_ATTENTION_SPLIT_KV],splitBufs,&splitPC,
                                sizeof(splitPC),(uint32_t)nHeads)) {
                                fvk_free(scratch);
                                return 3;
                            }
                        }
                        fvk_free(scratch);
                        return (int)g_submissionStatus;
                    }
                }
            } else {
                return 2;
            }
        }
    }
    struct { int tokens,prefix,heads,kvHeads,hd; float scale; }
        pc{tokens,prefix,nHeads,nKVHeads,headDim,scale};
    Buffer* bufs[] = {B(q),B(k),B(v),B(out)};
    dispatch(g_kern[K_QWEN35_CAUSAL_ATTENTION_PANEL],bufs,&pc,sizeof(pc),(uint32_t)((uint64_t)tokens*nHeads));
    return (int)g_submissionStatus;
}
extern "C" int fvk_sigmoid_mul_f32(void* x, const void* gate, int n) {
    if (!g_ready) return 1;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!panelFits(x,1,1,n) || !panelFits(gate,1,1,n)) return 2;
    Buffer* bufs[] = {B(x),B(gate)};
    dispatch(g_kern[K_SIGMOID_MUL],bufs,&n,sizeof(n),(uint32_t)(((uint64_t)n+255)/256));
    return (int)g_submissionStatus;
}
static int g_q4k_arm_mode = 0; // 0 = default, 1 = candidate, 2 = scalar

extern "C" void fvk_set_q4k_arm_mode(int mode) {
    g_q4k_arm_mode = mode;
}

// Wide single-token Q4_K GEMVs (the LM head) default to Wave32: on gfx1151 the scalar
// thread-per-row path reads a 248320x5120 head at ~24 GB/s (29.9 ms/token) vs 6.7 ms.
static const int kQ4KWave32WideRows = 65536;

static inline bool useQ4KWave32(int P, int out) {
    if (!g_have_q4k_wave32 || P != 1) return false;
    if (g_q4k_arm_mode == 1) return true;
    if (g_q4k_arm_mode == 2) return false;
    const char* arm = std::getenv("FAK_VULKAN_Q4K_ARM");
    if (arm) {
        if (strcmp(arm, "scalar") == 0 || strcmp(arm, "0") == 0) return false;
        if (strcmp(arm, "candidate") == 0 || strcmp(arm, "wave32") == 0 || strcmp(arm, "1") == 0) return true;
    }
    const char* w32 = std::getenv("FAK_VULKAN_Q4K_WAVE32");
    if (w32) {
        if (strcmp(w32, "0") == 0 || strcmp(w32, "false") == 0) return false;
        if (strcmp(w32, "1") == 0 || strcmp(w32, "true") == 0) return true;
    }
    // Physical A/B on gfx1151 resulted in REJECT_RETAIN_SCALAR (0.63x and 0.76x) on narrow
    // rows; default retains scalar below kQ4KWave32WideRows.
    return out >= kQ4KWave32WideRows;
}

extern "C" int fvk_have_q4k_wave32(void) {
    return g_have_q4k_wave32;
}

// Candidate Q4_K cooperative-matrix prefill arm. Default off: the scalar path owns
// decode, and an unsupported device never sees the coopmat pipeline. FAK_VULKAN_Q4K_ARM
// (candidate/scalar) is shared with the Wave32 arm; cooperative matrix is preferred
// for multi-token because it stages the tile once instead of re-reading weights per token.
static inline bool useQ4KCoopMat(int P) {
    if (!g_have_q4k_coopmat || P <= 1) return false;
    if (g_q4k_arm_mode == 2) return false;
    const char* arm = std::getenv("FAK_VULKAN_Q4K_ARM");
    if (arm) {
        if (strcmp(arm, "scalar") == 0 || strcmp(arm, "0") == 0) return false;
        if (strcmp(arm, "candidate") == 0 || strcmp(arm, "coopmat") == 0 || strcmp(arm, "1") == 0) return true;
    }
    const char* cm = std::getenv("FAK_VULKAN_Q4K_COOPMAT");
    if (cm) {
        if (strcmp(cm, "0") == 0 || strcmp(cm, "false") == 0) return false;
        if (strcmp(cm, "1") == 0 || strcmp(cm, "true") == 0) return true;
    }
    if (g_q4k_arm_mode == 1) return true;
    // Default retains the scalar path until a physical A/B qualifies the coopmat arm.
    return false;
}

extern "C" void fvk_q4k_matmul_f32(const void* dQ4K, const void* dX, void* dY,
                         int out, int in, int P) {
    if (useQ4KCoopMat(P)) {
        struct PC { int out, in, p; } pc{out, in, P};
        Buffer* bufs[3] = {B((void*)dQ4K), B((void*)dX), B(dY)};
        // 2D grid: X covers output rows, Y covers tokens (see vulkanQ4KDispatchGrid).
        uint32_t gx = (uint32_t)(((size_t)out + 31) / 32);
        uint32_t gy = (uint32_t)(((size_t)P + 31) / 32);
        dispatch(g_kern[K_Q4K_MATMUL_COOPMAT], bufs, &pc, sizeof(pc), gx, gy);
        return;
    }
    if (useQ4KWave32(P, out)) {
        struct PC { int out, in, p; } pc{out, in, P};
        Buffer* bufs[3] = {B((void*)dQ4K), B((void*)dX), B(dY)};
        uint32_t groups = (uint32_t)(((size_t)out + 1) / 2);
        dispatch(g_kern[K_Q4K_MATMUL_WAVE32], bufs, &pc, sizeof(pc), groups);
        return;
    }
    struct PC { int out, in, p; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dQ4K), B((void*)dX), B(dY)};
    dispatch(g_kern[K_Q4K_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)(((size_t)out * P + 63) / 64));
}
extern "C" void fvk_q6k_matmul_f32(const void* dQ6K, const void* dX, void* dY,
                                    int out, int in, int P) {
    if (!g_have_q6k_matmul || g_kern[K_Q6K_MATMUL].pipe == VK_NULL_HANDLE) return;
    // One token takes the shader's shared-memory staged row path (tokens = -1);
    // the workgroup count is the same ceil(out / 64).
    struct PC { int out, in, p; } pc{out, in, P == 1 ? -1 : P};
    Buffer* bufs[3] = {B((void*)dQ6K), B((void*)dX), B(dY)};
    dispatch(g_kern[K_Q6K_MATMUL], bufs, &pc, sizeof(pc),
             (uint32_t)(((size_t)out * (size_t)P + 63) / 64));
}
extern "C" void fvk_q5k_matmul_f32(const void* dQ5K, const void* dX, void* dY,
                                    int out, int in, int P) {
    if (!g_have_q5k_matmul || g_kern[K_Q5K_MATMUL].pipe == VK_NULL_HANDLE) return;
    struct PC { int out, in, p; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dQ5K), B((void*)dX), B(dY)};
    dispatch(g_kern[K_Q5K_MATMUL], bufs, &pc, sizeof(pc),
             (uint32_t)(((size_t)out * (size_t)P + 63) / 64));
}
extern "C" void fvk_q3k_matmul_f32(const void* dQ3K, const void* dX, void* dY,
                                    int out, int in, int P) {
    if (!g_have_q3k_matmul || g_kern[K_Q3K_MATMUL].pipe == VK_NULL_HANDLE) return;
    struct PC { int out, in, p; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dQ3K), B((void*)dX), B(dY)};
    dispatch(g_kern[K_Q3K_MATMUL], bufs, &pc, sizeof(pc),
             (uint32_t)(((size_t)out * (size_t)P + 63) / 64));
}
static const uint32_t kQ2KMatvecRows = 2; // must match ROWS in q2k_matvec.comp
static inline uint32_t q2kMatvecGroups(int out) { return ((uint32_t)out + kQ2KMatvecRows - 1u) / kQ2KMatvecRows; }
extern "C" void fvk_q2k_matmul_f32(const void* dQ2K, const void* dX, void* dY,
                          int out, int in, int P) {
    struct PC { int out, in, p, aux; float eps; } pc{out, in, P, 0, 0.0f};
    Buffer* bufs[7] = {
        B((void*)dQ2K), B((void*)dX), B((void*)dX), B(dY),
        B((void*)dQ2K), B((void*)dX), B(dY),
    };
    if (P == 1 && g_have_q2k_matvec) {
        dispatch(g_kern[K_Q2K_MATVEC], bufs, &pc, sizeof(pc), q2kMatvecGroups(out));
        return;
    }
    dispatch(g_kern[K_Q2K_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)(((size_t)out * P + 255) / 256));
}
extern "C" void fvk_swiglu_q2k_matmul_add_f32(const void* dQ2K, const void* dG,
                                                const void* dU, void* dD,
                                                int out, int in, int P) {
    if (P != 1) {
        fprintf(stderr, "fak-vulkan: fused Q2_K SwiGLU down projection is decode-only\n");
        abort();
    }
    struct PC { int out, in, p, aux; float eps; } pc{out, in, P, -1, 0.0f};
    Buffer* bufs[7] = {
        B((void*)dQ2K), B((void*)dG), B((void*)dU), B(dD),
        B((void*)dQ2K), B((void*)dG), B(dD),
    };
    if (g_have_q2k_matvec) {
        dispatch(g_kern[K_Q2K_MATVEC], bufs, &pc, sizeof(pc), q2kMatvecGroups(out));
        return;
    }
    dispatch(g_kern[K_Q2K_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)(((size_t)out * P + 255) / 256));
}
extern "C" void fvk_rmsnorm_q2k_matmul2_f32(const void* dW0, const void* dW1,
                                  const void* dX, const void* dNorm,
                                  void* dY0, void* dY1,
                                  int out0, int out1, int in, int P, float eps) {
    struct PC { int out, in, p, aux; float eps; } pc{out0, in, P, out1, eps};
    Buffer* bufs[7] = {
        B((void*)dW0), B((void*)dX), B((void*)dX), B(dY0),
        B((void*)dW1), B((void*)dNorm), B(dY1),
    };
    if (P == 1 && g_have_q2k_matvec) {
        dispatch(g_kern[K_Q2K_MATVEC], bufs, &pc, sizeof(pc), q2kMatvecGroups(out0) + q2kMatvecGroups(out1));
        return;
    }
    uint32_t groups = ((uint32_t)(out0 + out1) + 255u) / 256u;
    dispatch(g_kern[K_RMSNORM_Q2K_MATMUL2], bufs, &pc, sizeof(pc), groups);
}
// Native IQ-quant matvec (iq*_matvec.spv). mode 0: Y[t] = W0 X[t] for t < tokens;
// mode 1: Y += W0 x (tokens == 1); mode 2: Y = W0 x and Y1 = W1 x with W1 the same format
// (tokens == 1). x is already normalized/activated by the caller.
extern "C" int fvk_iq_matvec_available(int fmt) {
    return (fmt >= 0 && fmt < FVK_IQ_FORMATS && g_have_iq[fmt]) ? 1 : 0;
}
extern "C" void fvk_iq_matvec_f32(int fmt, const void* dW0, const void* dX, void* dY,
                                  const void* dW1, void* dY1,
                                  int out0, int out1, int in, int tokens, int mode) {
    if (!fvk_iq_matvec_available(fmt) || tokens < 1 || mode < 0 || mode > 2 || out0 <= 0 ||
        in <= 0 || (in % 256) != 0 || (mode != 0 && tokens != 1) ||
        (mode == 2 && (!dW1 || !dY1 || out1 <= 0))) {
        fprintf(stderr, "fak-vulkan: invalid IQ matvec call fmt=%d mode=%d tokens=%d\n", fmt, mode, tokens);
        abort();
    }
    int aux = mode == 2 ? out1 : (mode == 1 ? -1 : 0);
    struct PC { int out, in, p, aux; float eps; } pc{out0, in, tokens, aux, 0.0f};
    const void* w1 = mode == 2 ? dW1 : dW0;
    void* y1 = mode == 2 ? dY1 : dY;
    Buffer* bufs[7] = {
        B((void*)dW0), B((void*)dX), B((void*)dX), B(dY),
        B((void*)w1), B((void*)dX), B(y1),
    };
    uint32_t groups = ((uint32_t)out0 + kIQMatvecRows - 1u) / kIQMatvecRows;
    if (mode == 2) groups += ((uint32_t)out1 + kIQMatvecRows - 1u) / kIQMatvecRows;
    // The shaders stride over row groups, so a capped grid amortizes their per-workgroup
    // codebook setup while still filling the device.
    if (groups > kIQMatvecMaxGroups) groups = kIQMatvecMaxGroups;
    dispatch(g_kern[kIQKernel[fmt]], bufs, &pc, sizeof(pc), groups, mode == 0 ? (uint32_t)tokens : 1u);
}
// Prefill GEMM over a packed weight: Y[t] = W X[t] for t < tokens (mm_cm1_*.spv). Returns
// FVK_MM_CM1_OK after recording the dispatch, else the status that refused it; a refusal
// records nothing, so the caller's fallback kernel sees untouched buffers.
extern "C" int fvk_mm_cm1_status(int fmt) {
    if (fmt < 0 || fmt >= FVK_MMCM1_FORMATS) return FVK_MM_CM1_BAD_FORMAT;
    return g_mmcm1Status[fmt];
}
extern "C" int fvk_mm_cm1_f32(int fmt, const void* dW, const void* dX, void* dY, int out, int in, int tokens) {
    int st = fvk_mm_cm1_status(fmt);
    if (st != FVK_MM_CM1_OK) return st;
    if (!dW || !dX || !dY || out <= 0 || in <= 0 || (in % 256) != 0 || tokens <= 0) return FVK_MM_CM1_BAD_SHAPE;
    uint32_t gx = ((uint32_t)out + kMMCM1TileRows - 1u) / kMMCM1TileRows;
    uint32_t gy = ((uint32_t)tokens + kMMCM1TileTokens - 1u) / kMMCM1TileTokens;
    if (gx > g_maxComputeWorkGroupCountX || gy > g_maxComputeWorkGroupCountY) return FVK_MM_CM1_BAD_SHAPE;
    struct PC { int out, in, p; } pc{out, in, tokens};
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
    if (!dispatch(g_kern[kMMCM1Kernel[fmt]], bufs, &pc, sizeof(pc), gx, gy)) return FVK_MM_CM1_DISPATCH_FAILED;
    return FVK_MM_CM1_OK;
}
extern "C" void fvk_dispatch_profile_snapshot(fvk_dispatch_profile* out) {
    if (!out) return;
    out->compute_dispatches = g_dp.compute.load();
    out->q4k_matmul_dispatches = g_dp.q4k.load();
    out->q2k_matmul_dispatches = g_dp.q2k.load();
    out->other_compute_dispatches = g_dp.other.load();
    out->compute_barriers = g_dp.barriers.load();
    out->d2d_copies = g_dp.d2d.load();
    out->batch_submits = g_dp.batchSubmits.load();
    out->batch_flushes = g_dp.batchFlushes.load();
    out->one_shot_submits = g_dp.oneShotSubmits.load();
    out->other_matmul_dispatches = g_dp.otherMatmul.load();
    out->other_norm_dispatches = g_dp.otherNorm.load();
    out->other_rope_dispatches = g_dp.otherRope.load();
    out->other_swiglu_dispatches = g_dp.otherSwiGLU.load();
    out->other_add_dispatches = g_dp.otherAdd.load();
    out->other_attention_dispatches = g_dp.otherAttention.load();
    out->other_argmax_dispatches = g_dp.otherArgmax.load();
    out->other_gdn_dispatches = g_dp.otherGDN.load();
    out->other_unclassified_dispatches = g_dp.otherUnclassified.load();
    out->one_shot_compute_submits = g_dp.oneShotCompute.load();
    out->one_shot_h2d_submits = g_dp.oneShotH2D.load();
    out->one_shot_d2h_submits = g_dp.oneShotD2H.load();
    out->one_shot_d2d_submits = g_dp.oneShotD2D.load();
}
extern "C" int fvk_ts_enabled(void) { return g_tsOn ? 1 : 0; }
extern "C" const char* fvk_ts_kernel_name(uint32_t kid) { return tsKernelName(kid); }
// The fold is plain state mutated by recording and batch collection; the caller serializes
// this against every other shim call (Go holds vulkanMu). Returns the row count, which may
// exceed cap: size with cap 0, then fill.
extern "C" int fvk_ts_snapshot(fvk_ts_totals* tot, fvk_ts_row* rows, int cap) {
    if (tot) {
        tot->enabled = g_tsOn ? 1 : 0;
        tot->usable = g_tsUsable ? 1 : 0;
        tot->init_tried = g_tsInitTried ? 1 : 0;
        tot->pad = 0;
        tot->batches = g_tsBatches;
        tot->ops = g_tsOps;
        tot->truncated = g_tsTruncated;
        tot->untimed_compute = g_tsUntimedCompute.load();
        tot->untimed_transfer = g_tsUntimedTransfer.load();
        tot->barriers_elided = g_tsBarriersElided;
        tot->gpu_ns = g_tsGpuNs;
        tot->wall_ns = g_tsWallNs;
    }
    int i = 0;
    if (rows) {
        for (const auto& r : g_tsAgg) {
            if (i >= cap) break;
            rows[i].kid = (uint32_t)(r.first >> 48);
            rows[i].gx = (uint32_t)((r.first >> 20) & 0xFFFFFFF);
            rows[i].gy = (uint32_t)(r.first & 0xFFFFF);
            rows[i].pad = 0;
            rows[i].calls = r.second.n;
            rows[i].ns = r.second.ns;
            ++i;
        }
    }
    return (int)g_tsAgg.size();
}
extern "C" void fvk_dispatch_profile_reset(void) {
    g_dp.compute.store(0); g_dp.q4k.store(0); g_dp.q2k.store(0); g_dp.other.store(0);
    g_dp.barriers.store(0); g_dp.d2d.store(0); g_dp.batchSubmits.store(0);
    g_dp.batchFlushes.store(0); g_dp.oneShotSubmits.store(0);
    g_dp.otherMatmul.store(0); g_dp.otherNorm.store(0); g_dp.otherRope.store(0);
    g_dp.otherSwiGLU.store(0); g_dp.otherAdd.store(0); g_dp.otherAttention.store(0);
    g_dp.otherArgmax.store(0); g_dp.otherGDN.store(0); g_dp.otherUnclassified.store(0);
    g_dp.oneShotCompute.store(0); g_dp.oneShotH2D.store(0);
    g_dp.oneShotD2H.store(0); g_dp.oneShotD2D.store(0);
}
