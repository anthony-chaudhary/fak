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
static inline void dp_inc(std::atomic<uint64_t>& v) { if (g_dp_on) v.fetch_add(1, std::memory_order_relaxed); }

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

bool environmentFlagEnabled(const char* name) {
#ifdef _WIN32
    char value[2]{};
    DWORD length = GetEnvironmentVariableA(name, value, (DWORD)sizeof(value));
    return length == 1 && value[0] == '1';
#else
    const char* value = std::getenv(name);
    return value && value[0] == '1' && value[1] == '\0';
#endif
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
    bool           allocationDeviceLocal = false;
    void*          mapped = nullptr; // host view of a direct-upload weight arena binding
};

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
            vkFreeMemory(g_dev, block.mem, nullptr);
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

enum KId { K_MATMUL, K_MATMUL_ADD, K_MATMUL_ARGMAX, K_MATMUL_ARGMAX_BLOCKS, K_MATMUL2, K_MATMUL3, K_RMSNORM, K_RMSNORM_MATMUL, K_RMSNORM_MATMUL2, K_RMSNORM_MATMUL3, K_RMSNORM_MATMUL_ARGMAX_BLOCKS, K_ROPE, K_SWIGLU, K_SWIGLU_MATMUL_ADD, K_ADD, K_ADD_BIAS, K_ATTENTION, K_ARGMAX, K_ARGMAX_PAIRS, K_Q8_MATMUL, K_Q8_MATMUL_DECODE, K_Q8_MATMUL2, K_Q8_MATMUL3, K_RMSNORM_Q8_MATMUL2, K_RMSNORM_Q8_MATMUL3, K_SWIGLU_Q8_MATMUL_ADD, K_QWEN35_GDN_Q8_IN_PROJ, K_QWEN35_GDN_CONV, K_QWEN35_GDN_RECURRENT, K_QWEN35_GDN_PREFILL_TILED, K_QWEN35_GDN_PREFILL_NORM, K_QWEN35_GDN_VERIFY_TILED, K_GLM_KDA_REREAD, K_GLM_KDA_WAVE32, K_Q4K_MATMUL, K_Q4K_MATMUL_WAVE32, K_Q4K_MATMUL_COOPMAT, K_Q6K_MATMUL, K_Q5K_MATMUL, K_Q3K_MATMUL, K_RMSNORM_Q4K_MATMUL2, K_SWIGLU_Q4K_MATMUL_ADD, K_Q2K_MATMUL, K_RMSNORM_Q2K_MATMUL2, K_QWEN35_SPLIT_QG_PANEL, K_QWEN35_PARTIAL_ROPE_PANEL, K_QWEN35_CAUSAL_ATTENTION_PANEL, K_SIGMOID_MUL, K_Q2K_MATVEC, K_RMSNORM_Q8_MATMUL2_COOP, K_IQ4XS_MATVEC, K_IQ3XXS_MATVEC, K_IQ2S_MATVEC, K_IQ3S_MATVEC, K_IQ2XXS_MATVEC, K_IQ2XS_MATVEC, K_IQ1S_MATVEC, K_V41_TAIL_ROPE_QK, K_V41_SHARED_ATTENTION, K_V41_INDEXER_SCORE, K_COUNT };
Kernel g_kern[K_COUNT];

// Every non-Q4_K/Q2_K kernel belongs to exactly one primary operation family. Fused
// kernels follow their public operation prefix so these fields sum to `other`.
std::atomic<uint64_t>& dpOtherFamily(KId id) {
    switch (id) {
    case K_MATMUL: case K_MATMUL_ADD: case K_MATMUL_ARGMAX: case K_MATMUL_ARGMAX_BLOCKS:
    case K_MATMUL2: case K_MATMUL3: case K_Q8_MATMUL: case K_Q8_MATMUL_DECODE: case K_Q8_MATMUL2: case K_Q8_MATMUL3: case K_Q6K_MATMUL: case K_Q5K_MATMUL: case K_Q3K_MATMUL:
    case K_IQ4XS_MATVEC: case K_IQ1S_MATVEC: case K_IQ2XS_MATVEC: case K_IQ2XXS_MATVEC: case K_IQ3S_MATVEC: case K_IQ2S_MATVEC: case K_IQ3XXS_MATVEC:
        return g_dp.otherMatmul;
    case K_RMSNORM: case K_RMSNORM_MATMUL: case K_RMSNORM_MATMUL2: case K_RMSNORM_MATMUL3:
    case K_RMSNORM_MATMUL_ARGMAX_BLOCKS: case K_RMSNORM_Q8_MATMUL2: case K_RMSNORM_Q8_MATMUL3: case K_RMSNORM_Q8_MATMUL2_COOP:
    case K_RMSNORM_Q4K_MATMUL2:
        return g_dp.otherNorm;
    case K_ROPE: case K_QWEN35_PARTIAL_ROPE_PANEL: case K_V41_TAIL_ROPE_QK:
        return g_dp.otherRope;
    case K_SWIGLU: case K_SWIGLU_MATMUL_ADD: case K_SWIGLU_Q8_MATMUL_ADD: case K_SIGMOID_MUL:
    case K_SWIGLU_Q4K_MATMUL_ADD:
        return g_dp.otherSwiGLU;
    case K_ADD: case K_ADD_BIAS:
        return g_dp.otherAdd;
    case K_ATTENTION: case K_QWEN35_CAUSAL_ATTENTION_PANEL: case K_V41_SHARED_ATTENTION: case K_V41_INDEXER_SCORE:
        return g_dp.otherAttention;
    case K_ARGMAX: case K_ARGMAX_PAIRS:
        return g_dp.otherArgmax;
    case K_QWEN35_GDN_Q8_IN_PROJ: case K_QWEN35_GDN_CONV: case K_QWEN35_GDN_RECURRENT:
    case K_QWEN35_GDN_PREFILL_TILED: case K_QWEN35_GDN_PREFILL_NORM:
    case K_QWEN35_GDN_VERIFY_TILED:
    case K_GLM_KDA_REREAD: case K_GLM_KDA_WAVE32:
        return g_dp.otherGDN;
    case K_QWEN35_SPLIT_QG_PANEL: case K_Q4K_MATMUL: case K_Q4K_MATMUL_WAVE32: case K_Q4K_MATMUL_COOPMAT: case K_Q2K_MATMUL: case K_RMSNORM_Q2K_MATMUL2: case K_Q2K_MATVEC: case K_COUNT:
        return g_dp.otherUnclassified;
    }
    return g_dp.otherUnclassified;
}

static inline void dpDispatch(const Kernel& k) {
    if (!g_dp_on) return;
    const KId id = static_cast<KId>(&k - g_kern);
    if (id == K_Q4K_MATMUL || id == K_Q4K_MATMUL_WAVE32 || id == K_Q4K_MATMUL_COOPMAT) {
        g_dp.q4k.fetch_add(1, std::memory_order_relaxed);
    } else if (id == K_Q2K_MATMUL || id == K_RMSNORM_Q2K_MATMUL2 || id == K_Q2K_MATVEC) {
        g_dp.q2k.fetch_add(1, std::memory_order_relaxed);
    } else {
        g_dp.other.fetch_add(1, std::memory_order_relaxed);
        dpOtherFamily(id).fetch_add(1, std::memory_order_relaxed);
    }
    uint64_t n = g_dp.compute.fetch_add(1, std::memory_order_relaxed) + 1;
    if ((n & 255) == 0) {
        fprintf(stderr,
            "fak-vulkan profile compute=%llu q4k=%llu other=%llu other_ops=matmul:%llu,norm:%llu,rope:%llu,swiglu:%llu,add:%llu,attention:%llu,argmax:%llu,gdn:%llu,unclassified:%llu barriers=%llu d2d=%llu batch_submits=%llu batch_flushes=%llu one_shot_submits=%llu one_shot_calls=compute:%llu,h2d:%llu,d2h:%llu,d2d:%llu\n",
            (unsigned long long)n, (unsigned long long)g_dp.q4k.load(), (unsigned long long)g_dp.other.load(),
            (unsigned long long)g_dp.otherMatmul.load(), (unsigned long long)g_dp.otherNorm.load(),
            (unsigned long long)g_dp.otherRope.load(), (unsigned long long)g_dp.otherSwiGLU.load(),
            (unsigned long long)g_dp.otherAdd.load(), (unsigned long long)g_dp.otherAttention.load(),
            (unsigned long long)g_dp.otherArgmax.load(), (unsigned long long)g_dp.otherGDN.load(),
            (unsigned long long)g_dp.otherUnclassified.load(), (unsigned long long)g_dp.barriers.load(),
            (unsigned long long)g_dp.d2d.load(), (unsigned long long)g_dp.batchSubmits.load(),
            (unsigned long long)g_dp.batchFlushes.load(), (unsigned long long)g_dp.oneShotSubmits.load(),
            (unsigned long long)g_dp.oneShotCompute.load(), (unsigned long long)g_dp.oneShotH2D.load(),
            (unsigned long long)g_dp.oneShotD2H.load(), (unsigned long long)g_dp.oneShotD2D.load());
    }
}

static inline void dpOneShot(std::atomic<uint64_t>& family) {
    if (!g_dp_on) return;
    family.fetch_add(1, std::memory_order_relaxed);
    g_dp.oneShotSubmits.fetch_add(1, std::memory_order_relaxed);
}

// Q8 fast-path availability (set in fvk_init from the device's 8-bit-storage + int8 features).
int g_have_q8 = 0;
// The four-projection GDN specialization is optional even when generic Q8 is
// available: an absent or rejected pipeline must preserve the composed path.
int g_have_qwen35_gdn_q8_in_proj = 0;
// Mandatory in a current build; published only after complete initialization.
int g_have_v41_tail_rope_qk = 0;
int g_have_v41_shared_attention = 0;
// Independently admitted only after dynamic binary32 properties and module proof.
int g_have_v41_indexer_score = 0;
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
const uint32_t kIQMatvecRows = 2; // must match ROWS in every iq*_matvec.comp shader
const uint32_t kIQMatvecMaxGroups = 1024;
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
// Portable packed Q6_K is optional so older SPIR-V bundles remain loadable.
int g_have_q6k_matmul = 0;
int g_have_q5k_matmul = 0;
int g_have_q3k_matmul = 0;

VkDescriptorPool g_descpool = VK_NULL_HANDLE;

// 13 covers the widest fused kernel: Qwen GDN Q8 input projection binds four
// code+scale pairs, X, and four outputs. Sizes the per-dispatch descriptor/buffer
// stack arrays; well under the device's storage-buffer binding limit and the
// 32768-descriptor pool's headroom.
static constexpr int MAX_DISPATCH_BUFS = 13;

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
// accounting or binding; the caller publishes those only after success.
template <typename Allocate>
VkResult allocateCompatibleMemory(uint32_t typeBits, VkMemoryPropertyFlags want,
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
    for (uint32_t i = 0; i < count; ++i) {
        const VkMemoryType& type = properties.memoryTypes[i];
        if (!(typeBits & (uint32_t(1) << i)) || (type.propertyFlags & want) != want) continue;
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

void destroyBuffer(Buffer* b) {
    if (g_v41SubmissionPendingFailure) return; // referenced scratch/arena may still be in flight
    if (!b) return;
    if (b->buf) {
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
        vkFreeMemory(g_dev, b->mem, nullptr);
    }
    delete b;
    releaseWeightArena();
}

Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage);

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
        VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
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
    if (freed > 0) clearDescriptorBindingCache();
    g_poolCount = 0;
    return freed;
}

Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage) {
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
        return allocateCompatibleMemory(req.memoryTypeBits, want, g_memprops, req.size, tryAll, [&](uint32_t index, VkDeviceMemory* candidate) {
                VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ai.allocationSize = req.size;
                ai.memoryTypeIndex = index;
                return vkAllocateMemory(g_dev, &ai, nullptr, candidate);
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
        vkFreeMemory(g_dev, b->mem, nullptr);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    b->props = actualProps;
    b->memoryTypeIndex = selectedMemoryType;
    b->allocationBytes = req.size;
    b->allocationID = nextMemoryAllocationID();
    b->allocationDeviceLocal = memoryTypeUsesDeviceLocalHeap(selectedMemoryType);
    trackDeviceAllocation(req.size, selectedMemoryType);
    return b;
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
        return allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    }

    uint32_t memoryType = g_umaDirectWeights ? findDirectWeightMemType(req.memoryTypeBits) : UINT32_MAX;
    if (memoryType == UINT32_MAX) memoryType = findMemType(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (memoryType == UINT32_MAX) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    if (maxArenaBytes == 0) maxArenaBytes = g_totalDeviceLocalMemory;
    if (maxArenaBytes == 0) maxArenaBytes = g_maxMemoryAllocationSize;
    if (maxArenaBytes == 0) maxArenaBytes = WEIGHT_ARENA_BLOCK_BYTES;

    for (size_t i = 0; i < g_weightArena.size(); ++i) {
        WeightArenaBlock& block = g_weightArena[i];
        if (block.memoryTypeIndex != memoryType) continue;
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
        ++g_weightArenaBufferBindings;
        g_weightArenaLiveBytes += bytes;
        return b;
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

    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = blockBytes;
    ai.memoryTypeIndex = memoryType;
    VkDeviceMemory memory = VK_NULL_HANDLE;
    VkResult ar = vkAllocateMemory(g_dev, &ai, nullptr, &memory);
    if (allocPressure(ar)) {
        drainPool();
        ar = vkAllocateMemory(g_dev, &ai, nullptr, &memory);
    }
    if (ar != VK_SUCCESS) {
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }
    VkResult br = vkBindBufferMemory(g_dev, b->buf, memory, 0);
    if (br != VK_SUCCESS) {
        vkFreeMemory(g_dev, memory, nullptr);
        vkDestroyBuffer(g_dev, b->buf, nullptr);
        delete b;
        return nullptr;
    }

    WeightArenaBlock block{};
    block.mem = memory;
    block.capacity = blockBytes;
    block.allocationID = nextMemoryAllocationID();
    block.used = req.size;
    block.memoryTypeIndex = memoryType;
    block.liveBuffers = 1;
    block.allocationDeviceLocal = memoryTypeUsesDeviceLocalHeap(memoryType);
    if (g_memprops.memoryTypes[memoryType].propertyFlags & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) {
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
VkCommandBuffer beginCmd() {
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
    VKCHECK(vkQueueSubmit(g_queue, 1, &si, g_submitFence));
    VKCHECK(vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX));
}

void endSubmitWait(VkCommandBuffer cmd) {
    submitWait(cmd);
    vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
}

VkCommandBuffer v41BeginCmdChecked() {
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
        r = vkQueueSubmit(g_queue, 1, &si, g_submitFence);
        if (r != VK_SUCCESS) {
            // In particular VK_ERROR_DEVICE_LOST is equivalent to successful
            // submission for pending/in-use resource state. Treat every failed
            // submit conservatively: no failure result is a quiescence proof.
            g_v41SubmissionPendingFailure = true;
        } else {
            r = vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX);
            if (r != VK_SUCCESS) g_v41SubmissionPendingFailure = true;
        }
    }
    if (!g_v41SubmissionPendingFailure) vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
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
std::vector<DescriptorSetRecord> g_batchSets;
std::vector<Buffer*>          g_batchFreed;   // buffers freed mid-batch, recycled after submit
int                           g_batchOps  = 0;
bool                          g_batchHasV41 = false;
uint64_t                      g_batchD2DCount = 0;
uint64_t                      g_batchD2DBytes = 0;
bool                          g_batchD2DValid = true;

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
        return;
    }
    recordComputeBarrier(cmd);
}

// ---- GPU timestamp stage profile (FAK_VULKAN_TIMESTAMP_PROFILE=1, default off) ----------
// Diagnostic only: a bottom-of-pipe timestamp is written at batch start and after every
// recorded dispatch, so (with the recorder's inter-dispatch barriers) the delta between
// consecutive timestamps is that dispatch's device time. Deltas are folded per
// (kernel id, groupsX, groupsY) shape key and printed to stderr every
// FAK_VULKAN_TIMESTAMP_PROFILE_EVERY batches (default 32). Unset, the recorder emits no
// query commands and the recorded command stream is byte-identical to the unprofiled one.
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
    "qwen35_causal_attention_panel", "sigmoid_mul", "q2k_matvec", "rmsnorm_q8_matmul2_coop",
    "iq4xs_matvec", "iq3xxs_matvec", "iq2s_matvec", "iq3s_matvec", "iq2xxs_matvec", "iq2xs_matvec", "iq1s_matvec", "v41_tail_rope_qk", "v41_shared_attention", "v41_indexer_score",
};
static_assert(sizeof(kKernelNames) / sizeof(kKernelNames[0]) == K_COUNT, "kernel name table out of sync with KId");

const bool g_tsOn = [] { const char* v = std::getenv("FAK_VULKAN_TIMESTAMP_PROFILE"); return v && v[0] == '1' && v[1] == '\0'; }();
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
    fprintf(stderr, "fak-vulkan ts-profile batches=%llu ops/batch=%.1f gpu_ms/batch=%.3f wall_ms/batch=%.3f truncated=%llu\n",
        (unsigned long long)g_tsBatches, g_tsOps * perBatch, g_tsGpuNs * perBatch / 1e6, g_tsWallNs * perBatch / 1e6,
        (unsigned long long)g_tsTruncated);
    size_t shown = 0;
    for (const auto& r : rows) {
        if (shown++ >= 40) break;
        uint32_t kid = (uint32_t)(r.first >> 48), gx = (uint32_t)((r.first >> 20) & 0xFFFFFFF), gy = (uint32_t)(r.first & 0xFFFFF);
        fprintf(stderr, "fak-vulkan ts-stage kernel=%s groups=%ux%u calls/batch=%.2f ms/batch=%.3f us/call=%.2f share=%.1f%%\n",
            kid < K_COUNT ? kKernelNames[kid] : "d2d_copy", gx, gy, r.second.n * perBatch, r.second.ns * perBatch / 1e6,
            r.second.n ? r.second.ns / (double)r.second.n / 1e3 : 0, g_tsGpuNs > 0 ? 100.0 * r.second.ns / g_tsGpuNs : 0);
    }
}

void tsBatchCollect() {
    if (!g_tsOn || !g_tsUsable || g_tsUsed < 2) return;
    std::vector<uint64_t> ts(g_tsUsed);
    if (vkGetQueryPoolResults(g_dev, g_tsPool, 0, g_tsUsed, ts.size() * sizeof(uint64_t), ts.data(),
            sizeof(uint64_t), VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT) != VK_SUCCESS) {
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
    if ((++g_tsBatches % g_tsEvery) == 0) tsReport();
    g_tsUsed = 0;
}

void batchBegin() {
	if (g_v41SubmissionPendingFailure) return;
	if (g_batching) return;          // already recording — the model brackets each token
	g_batchCmd = beginCmd();
	tsBatchBegin(g_batchCmd);
	g_batching = true;
	g_batchOps = 0;
	g_batchHasV41 = false;
	g_batchD2DCount = 0;
	g_batchD2DBytes = 0;
	g_batchD2DValid = true;
	g_batchSets.clear();
	// A new window owns a fresh hazard verdict queue; anything left from the prior window
	// must not leak into this one (the Go ledger re-lowers each window independently).
	g_batchHazardSync.clear();
}

void batchFlush() {
    if (g_v41SubmissionPendingFailure) return; // never retry an uncertain submit
    if (!g_batching) return;
    bool hadWork = g_batchOps > 0;
    if (g_batchHasV41) {
        if (hadWork) dp_inc(g_dp.batchSubmits);
        const bool completed = v41EndSubmitWaitChecked(g_batchCmd, hadWork);
        if (g_v41SubmissionPendingFailure) return; // retain command, sets and parked buffers
        if (completed && hadWork) tsBatchCollect();
        if (completed && hadWork && (!g_batchD2DValid ||
            !checkedCounterAdd(g_d2dCount, g_batchD2DCount) ||
            !checkedCounterAdd(g_d2dBytes, g_batchD2DBytes))) g_transferCountersValid = false;
    } else if (hadWork) {
        dp_inc(g_dp.batchSubmits);
        endSubmitWait(g_batchCmd);   // single submit + fence for the whole recorded chain
        tsBatchCollect();
        if (!g_batchD2DValid ||
            !checkedCounterAdd(g_d2dCount, g_batchD2DCount) ||
            !checkedCounterAdd(g_d2dBytes, g_batchD2DBytes)) {
            g_transferCountersValid = false;
        }
    } else {
        VKCHECK(vkEndCommandBuffer(g_batchCmd));
        vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &g_batchCmd);
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
	std::vector<Buffer*> freed;   freed.swap(g_batchFreed);
	for (Buffer* b : freed)   fvk_free(b);
}

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
    if (g_v41SubmissionPendingFailure) return;
    if (bytes == 0 || !dst || !host) return;
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

bool swigluPushConstantABI(const std::vector<char>& code) {
    if (code.size() < 5 * sizeof(uint32_t) || code.size() % sizeof(uint32_t) != 0) return false;
    std::vector<uint32_t> words(code.size() / sizeof(uint32_t));
    memcpy(words.data(), code.data(), code.size());
    if (words[0] != 0x07230203 || words[3] == 0 || words[4] != 0) return false;
    const uint32_t bound = words[3];
    const uint32_t opTypeInt = 21, opTypeFloat = 22, opTypeStruct = 30, opTypePointer = 32;
    const uint32_t opVariable = 59, opDecorate = 71, opMemberDecorate = 72;
    const uint32_t pushConstant = 9, blockDecoration = 2, offsetDecoration = 35;
    std::unordered_map<uint32_t, std::vector<uint32_t>> definitions;
    std::unordered_map<uint32_t, bool> blocks;
    std::unordered_map<uint64_t, uint32_t> offsets;
    uint32_t pushPointer = 0;
    size_t pushVariables = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    for (size_t pos = 5; pos < words.size();) {
        const uint32_t count = words[pos] >> 16;
        const uint32_t op = words[pos] & 0xffff;
        if (count == 0 || count > words.size() - pos) return false;
        const uint32_t* instruction = words.data() + pos;
        if (op == opTypeInt || op == opTypeFloat || op == opTypeStruct || op == opTypePointer || op == opVariable) {
            if ((op == opTypeInt && count != 4) || (op == opTypeFloat && count != 3) ||
                (op == opTypeStruct && count < 2) || (op == opTypePointer && count != 4) ||
                (op == opVariable && count != 4 && count != 5)) return false;
            const uint32_t id = instruction[op == opVariable ? 2 : 1];
            if (!validID(id) || !definitions.emplace(id, std::vector<uint32_t>(instruction, instruction + count)).second) return false;
            if (op == opTypeStruct) {
                for (uint32_t member = 2; member < count; ++member) if (!validID(instruction[member])) return false;
            } else if (op == opTypePointer) {
                if (!validID(instruction[3])) return false;
            } else if (op == opVariable) {
                if (!validID(instruction[1])) return false;
                if (instruction[3] == pushConstant) { pushPointer = instruction[1]; ++pushVariables; }
            }
        } else if (op == opDecorate) {
            if (count < 3 || !validID(instruction[1])) return false;
            if (instruction[2] == blockDecoration) {
                if (count != 3 || !blocks.emplace(instruction[1], true).second) return false;
            }
        } else if (op == opMemberDecorate) {
            if (count < 4 || !validID(instruction[1])) return false;
            if (instruction[3] == offsetDecoration) {
                const uint64_t key = (uint64_t(instruction[1]) << 32) | instruction[2];
                if (count != 5 || !offsets.emplace(key, instruction[4]).second) return false;
            }
        }
        pos += count;
    }
    if (pushVariables != 1) return false;
    auto pointer = definitions.find(pushPointer);
    if (pointer == definitions.end() || (pointer->second[0] & 0xffff) != opTypePointer || pointer->second[2] != pushConstant) return false;
    const uint32_t structID = pointer->second[3];
    auto structure = definitions.find(structID);
    if (structure == definitions.end() || (structure->second[0] & 0xffff) != opTypeStruct || structure->second.size() != 4 || blocks.count(structID) != 1) return false;
    auto integer = definitions.find(structure->second[2]);
    auto floating = definitions.find(structure->second[3]);
    if (integer == definitions.end() || (integer->second[0] & 0xffff) != opTypeInt || integer->second[2] != 32 || integer->second[3] != 1) return false;
    if (floating == definitions.end() || (floating->second[0] & 0xffff) != opTypeFloat || floating->second[2] != 32) return false;
    const uint64_t offsetKey = uint64_t(structID) << 32;
    auto first = offsets.find(offsetKey), second = offsets.find(offsetKey | 1);
    return first != offsets.end() && first->second == 0 && second != offsets.end() && second->second == 4;
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

bool v41TailRoPEABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16 && count >= 3 && a[2] == 17) { // LocalSize
            if (count != 6 || a[3] != 256 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 129 || op == 131 || op == 133) { // FAdd / FSub / FMul
            if (count != 5 || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
        } else if (op == 12) { // no extended arithmetic, especially Fma/trig
            return false;
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 5 || floatResults.size() < 6) return false;
    for (uint32_t id : floatResults) if (!decorated(id, noContraction, 0)) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 5);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 3; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 5 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], slot == 4 ? typeFloat : typeInt, slot == 4 ? 3 : 4);
        if (!scalar || (*scalar)[2] != 32 || (slot != 4 && (*scalar)[3] != 0)) return false;
        const uint32_t access = slot == 2 || slot == 3 ? 25 : 24;
        if (memberAccess.count(key(structID, access)) != 1) return false;
    }
    return seenBindings == 31;
}

// Narrow fixed attention ABI/NoContraction check, not a receipt verifier or
// full SPIR-V semantic validator. The source-pinned V4 bundle is a separate gate.
bool v41SharedAttentionABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults, expImports;
    uint32_t glslImport = 0;
    size_t glslImportCount = 0;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == 0 || kind == 40) return false; // RelaxedPrecision / FPFastMathMode
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16 && count >= 3 && a[2] == 17) { // LocalSize
            if (count != 6 || a[3] != 1 || a[4] != 1 || a[5] != 1) return false;
            localEntry = a[1]; ++localCount;
        } else if (op == 129 || op == 131 || op == 133 || op == 136) { // scalar FAdd/FSub/FMul/FDiv
            if (count != 5 || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
        } else if (op == 11) { // one explicit GLSL.std.450 import
            if (count != 6 || !validID(a[1]) || a[2] != 0x4c534c47 ||
                a[3] != 0x6474732e || a[4] != 0x3035342e || a[5] != 0) return false;
            glslImport = a[1]; ++glslImportCount;
        } else if (op == 12) { // only Exp; never extended Fma or reductions
            if (count != 6 || !validID(a[1]) || !validID(a[2]) || !validID(a[3]) ||
                a[4] != 27 || !validID(a[5])) return false;
            expImports.push_back(a[3]);
        }
        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 5 || floatResults.size() < 6 ||
        glslImportCount != 1 || expImports.empty()) return false;
    for (uint32_t id : expImports) if (id != glslImport) return false;
    for (uint32_t id : floatResults) if (!decorated(id, noContraction, 0)) return false;
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 8);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 5; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    const auto* scale = definition((*structure)[7], typeFloat, 3);
    if (!scale || (*scale)[2] != 32 || !atOffset(pushStruct, 5, 20)) return false;
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 5 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], slot == 4 ? typeInt : typeFloat, slot == 4 ? 4 : 3);
        if (!scalar || (*scalar)[2] != 32 || (slot == 4 && (*scalar)[3] != 0)) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        // Q/KV/sink are read-only; output is read/write for slot-ordered
        // accumulation; status is write-only and read back by the host.
        if (nonWritable != (slot < 3) || nonReadable != (slot == 4)) return false;
    }
    return seenBindings == 31;
}

// Narrow source-bound indexer ABI and binary32-mode admission. This is not a
// general SPIR-V validator or proof of algorithm semantics: the separately
// versioned V5 source/archive/module receipt and physical witness remain required.
// Numeric mode/capability values are from Khronos SPV_KHR_float_controls.
bool v41IndexerScoreABI(const std::vector<char>& code) {
    if (code.size() < 20 || code.size() % 4 != 0) return false;
    std::vector<uint32_t> w(code.size() / 4);
    memcpy(w.data(), code.data(), code.size());
    if (w[0] != 0x07230203 || w[3] == 0 || w[4] != 0) return false;
    const uint32_t bound = w[3];
    const uint32_t typeInt = 21, typeFloat = 22, typeRuntimeArray = 29;
    const uint32_t typeStruct = 30, typePointer = 32, variable = 59;
    const uint32_t block = 2, arrayStride = 6, binding = 33, descriptorSet = 34;
    const uint32_t offset = 35, noContraction = 42, pushConstant = 9, storageBuffer = 12;
    std::unordered_map<uint32_t, std::vector<uint32_t>> defs;
    std::unordered_map<uint64_t, uint32_t> decorations, memberOffsets;
    std::unordered_map<uint64_t, bool> memberAccess;
    std::vector<uint32_t> buffers, floatResults, floatTypes;
    std::vector<std::pair<uint32_t, uint32_t>> floatResultTypes;
    uint32_t modes = 0, capabilities = 0, modeEntry = 0;
    size_t floatControlExtensions = 0, floatAdds = 0, floatMuls = 0;
    uint32_t push = 0, entry = 0, localEntry = 0;
    size_t pushCount = 0, entryCount = 0, localCount = 0;
    auto validID = [bound](uint32_t id) { return id != 0 && id < bound; };
    auto key = [](uint32_t id, uint32_t kind) { return (uint64_t(id) << 32) | kind; };
    for (size_t p = 5; p < w.size();) {
        const uint32_t count = w[p] >> 16, op = w[p] & 0xffff;
        if (count == 0 || count > w.size() - p) return false;
        const uint32_t* a = w.data() + p;
        if (op == typeInt || op == typeFloat || op == typeRuntimeArray ||
            op == typeStruct || op == typePointer || op == variable) {
            if ((op == typeInt && count != 4) || (op == typeFloat && count != 3) ||
                (op == typeRuntimeArray && count != 3) || (op == typeStruct && count < 2) ||
                (op == typePointer && count != 4) || (op == variable && count != 4 && count != 5)) return false;
            const uint32_t id = a[op == variable ? 2 : 1];
            if (!validID(id) || !defs.emplace(id, std::vector<uint32_t>(a, a + count)).second) return false;
            if (op == typeFloat) {
                if (a[2] != 32) return false;
                floatTypes.push_back(id);
            }
            if (op == variable) {
                if (!validID(a[1])) return false;
                if (a[3] == pushConstant) { push = a[1]; ++pushCount; }
                else if (a[3] == storageBuffer) buffers.push_back(id);
                else if (a[3] == 0 || a[3] == 2) return false; // no alternate descriptor classes
            }
        } else if (op == 71) { // OpDecorate
            if (count < 3 || !validID(a[1])) return false;
            const uint32_t kind = a[2];
            if (kind == 0 || kind == 1 || kind == 39 || kind == 40) return false; // no relaxed/rounding/fast-math overrides
            if (kind == block || kind == noContraction) {
                if (count != 3 || !decorations.emplace(key(a[1], kind), 0).second) return false;
            } else if (kind == arrayStride || kind == binding || kind == descriptorSet) {
                if (count != 4 || !decorations.emplace(key(a[1], kind), a[3]).second) return false;
            }
        } else if (op == 72) { // OpMemberDecorate
            if (count < 4 || !validID(a[1])) return false;
            if (a[3] == 0 || a[3] == 39 || a[3] == 40) return false;
            if (a[3] == offset) {
                if (count != 5 || !memberOffsets.emplace(key(a[1], a[2]), a[4]).second) return false;
            } else if (a[3] == 24 || a[3] == 25) { // NonWritable / NonReadable
                if (count != 4 || a[2] != 0 || !memberAccess.emplace(key(a[1], a[3]), true).second) return false;
            }
        } else if (op == 15) { // one GLCompute entry, named main
            if (count < 5 || a[1] != 5 || !validID(a[2]) || a[3] != 0x6e69616d || a[4] != 0) return false;
            entry = a[2]; ++entryCount;
        } else if (op == 16) { // OpExecutionMode, fixed entry/modes/width
            if (count < 3 || !validID(a[1])) return false;
            if (a[2] == 17) {
                if (count != 6 || a[3] != 1 || a[4] != 1 || a[5] != 1) return false;
                localEntry = a[1]; ++localCount;
            } else {
                // SPV_KHR_float_controls: DenormPreserve, SignedZeroInfNanPreserve,
                // RoundingModeRTE. Defaults and conflicting/unknown modes fail closed.
                uint32_t bit = a[2] == 4459 ? 1u : a[2] == 4461 ? 2u : a[2] == 4462 ? 4u : 0u;
                if (bit == 0 || count != 4 || a[3] != 32 || (modes & bit) ||
                    (modeEntry != 0 && modeEntry != a[1])) return false;
                modes |= bit; modeEntry = a[1];
            }
        } else if (op == 17) { // only Shader and the three required float-control capabilities
            if (count != 2) return false;
            uint32_t bit = a[1] == 1 ? 1u : a[1] == 4464 ? 2u : a[1] == 4466 ? 4u : a[1] == 4467 ? 8u : 0u;
            if (bit == 0 || (capabilities & bit)) return false;
            capabilities |= bit;
        } else if (op == 10) { // explicit float-controls extension, no unknown module extensions
            const char expected[] = "SPV_KHR_float_controls";
            if (count != 1 + (sizeof(expected) + 3) / 4 ||
                memcmp(a + 1, expected, sizeof(expected)) != 0 || ++floatControlExtensions != 1) return false;
        } else if (op == 129 || op == 133) { // separately rounded scalar FAdd / FMul only
            if (count != 5 || !validID(a[1]) || !validID(a[2])) return false;
            floatResults.push_back(a[2]);
            if (op == 129) ++floatAdds; else ++floatMuls;
            floatResultTypes.emplace_back(a[1], a[2]);
        } else if (op == 12 || (op >= 48 && op <= 52) || (op >= 109 && op <= 112) || op == 115 || op == 116 ||
                   op == 127 || op == 131 || (op >= 136 && op <= 148) ||
                   op == 24 || (op >= 73 && op <= 75) || op == 331 || op == 332 ||
                   op == 265 || op == 266 || op == 269 || op == 350 || op == 352 || op == 355 || op == 358) {
            // No extended Fma/reductions, floating conversions/quantization, matrices,
            // decoration groups or ID-based execution modes/decorations.
            return false;
        } else if (op == 23) { // only the integer GlobalInvocationID vector is needed
            if (count != 4 || !validID(a[1]) || !validID(a[2])) return false;
            auto scalar = defs.find(a[2]);
            if (scalar == defs.end() || (scalar->second[0] & 0xffff) != typeInt) return false;
        }

        p += count;
    }
    auto definition = [&](uint32_t id, uint32_t op, size_t size) -> const std::vector<uint32_t>* {
        auto it = defs.find(id);
        return it != defs.end() && (it->second[0] & 0xffff) == op && it->second.size() == size ? &it->second : nullptr;
    };
    auto decorated = [&](uint32_t id, uint32_t kind, uint32_t value) {
        auto it = decorations.find(key(id, kind));
        return it != decorations.end() && it->second == value;
    };
    auto atOffset = [&](uint32_t id, uint32_t member, uint32_t value) {
        auto it = memberOffsets.find(key(id, member));
        return it != memberOffsets.end() && it->second == value;
    };
    if (entryCount != 1 || localCount != 1 || entry != localEntry || pushCount != 1 || buffers.size() != 4 || floatResults.size() != 4 ||
        floatAdds != 2 || floatMuls != 2 || modes != 7 || capabilities != 15 || floatControlExtensions != 1 || modeEntry != entry || floatTypes.size() != 1) return false;
    for (const auto& result : floatResultTypes) {
        const auto* scalar = definition(result.first, typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32 || !decorated(result.second, noContraction, 0)) return false;
    }
    const auto* pointer = definition(push, typePointer, 4);
    if (!pointer || (*pointer)[2] != pushConstant) return false;
    const uint32_t pushStruct = (*pointer)[3];
    const auto* structure = definition(pushStruct, typeStruct, 5);
    if (!structure || !decorated(pushStruct, block, 0)) return false;
    for (uint32_t m = 0; m < 3; ++m) {
        const auto* integer = definition((*structure)[m + 2], typeInt, 4);
        if (!integer || (*integer)[2] != 32 || (*integer)[3] != 1 || !atOffset(pushStruct, m, m * 4)) return false;
    }
    uint32_t seenBindings = 0;
    for (uint32_t id : buffers) {
        const auto* var = definition(id, variable, 4);
        auto bind = decorations.find(key(id, binding));
        if (!var || bind == decorations.end() || bind->second >= 4 || !decorated(id, descriptorSet, 0)) return false;
        const uint32_t slot = bind->second, mask = 1u << slot;
        if (seenBindings & mask) return false;
        seenBindings |= mask;
        const auto* ptr = definition((*var)[1], typePointer, 4);
        if (!ptr || (*ptr)[2] != storageBuffer) return false;
        const uint32_t structID = (*ptr)[3];
        const auto* st = definition(structID, typeStruct, 3);
        if (!st || !decorated(structID, block, 0) || !atOffset(structID, 0, 0)) return false;
        const auto* array = definition((*st)[2], typeRuntimeArray, 3);
        if (!array || !decorated((*st)[2], arrayStride, 4)) return false;
        const auto* scalar = definition((*array)[2], typeFloat, 3);
        if (!scalar || (*scalar)[2] != 32) return false;
        const bool nonWritable = memberAccess.count(key(structID, 24)) != 0;
        const bool nonReadable = memberAccess.count(key(structID, 25)) != 0;
        if (nonWritable != (slot < 3) || nonReadable != (slot == 3)) return false;
    }
    return seenBindings == 15;
}

// Reject only malformed SPIR-V framing, not invalid instructions or unsupported
// capabilities. SPIR-V physical layout requires five header words; its magic
// identifies either byte order. Leave version policy and code bytes unchanged.
// https://registry.khronos.org/SPIR-V/specs/unified1/SPIRV.html#_physical_layout_of_a_spir_v_module_and_instruction
static bool spirvModuleFraming(const std::vector<char>& code) {
    if (code.size() < 5 * sizeof(uint32_t) || code.size() % sizeof(uint32_t) != 0) return false;
    uint32_t magic = 0;
    std::memcpy(&magic, code.data(), sizeof(magic));
    return magic == 0x07230203u || magic == 0x03022307u;
}

bool buildKernel(Kernel& k, const std::string& spvPath, int nbuf, uint32_t pcsize, uint32_t subgroupSize = 0) {
    const bool fixedV41 = &k == &g_kern[K_V41_TAIL_ROPE_QK] || &k == &g_kern[K_V41_SHARED_ATTENTION] || &k == &g_kern[K_V41_INDEXER_SCORE];
    std::vector<char> code = fixedV41 ? readV41TailRoPEModule(spvPath) : readFile(spvPath);
    if (!spirvModuleFraming(code)) return false;
    if (&k == &g_kern[K_SWIGLU] && !swigluPushConstantABI(code)) {
        fprintf(stderr, "fak-vulkan: incompatible SwiGLU push-constant ABI in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_TAIL_ROPE_QK] &&
        (nbuf != 5 || pcsize != 12 || !v41TailRoPEABI(code))) {
        fprintf(stderr, "fak-vulkan: incompatible V4.1 tail RoPE module in %s\n", spvPath.c_str());
        return false;
    }
    if (&k == &g_kern[K_V41_SHARED_ATTENTION] &&
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
bool dispatch(Kernel& k, Buffer** bufs, const void* pc, uint32_t pcsize, uint32_t groupsX, uint32_t groupsY = 1) {
    if (g_v41SubmissionPendingFailure) return false;
    const bool checkedV41 = &k == &g_kern[K_V41_TAIL_ROPE_QK] || &k == &g_kern[K_V41_SHARED_ATTENTION] || &k == &g_kern[K_V41_INDEXER_SCORE];
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
        if (g_batchOps > 0) recordDispatchBarrier(g_batchCmd, g_batchOps);
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
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &rec.set, 0, nullptr);
    if (pcsize > 0) vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
    dpDispatch(k);
    vkCmdDispatch(cmd, groupsX, groupsY, 1);
    dpOneShot(g_dp.oneShotCompute);
    if (checkedV41) {
        const bool completed = v41EndSubmitWaitChecked(cmd, true);
        if (!g_v41SubmissionPendingFailure) recycleDescriptorSet(rec);
        return completed;
    }
    endSubmitWait(cmd);
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
extern "C" int fvk_debug_memory_type_selection(uint32_t typeBits, uint32_t want,
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
    VkResult result = allocateCompatibleMemory(typeBits, want, properties, allocationSize,
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

int fvk_init(char* name, int namelen, int* is_discrete, const char* spirv_dir) {
    // A re-init is not a device teardown and cannot establish quiescence. Keep
    // the poisoned context and its loaded capability bound until process exit.
    if (g_v41SubmissionPendingFailure) return 9;
    g_have_v41_tail_rope_qk = 0;
    g_have_v41_shared_attention = 0;
    g_have_v41_indexer_score = 0;
    bool v41IndexerPipelineBuilt = false;
    g_v41IndexerFloatControls = false;
    g_v41IndexerScoreReason = "Vulkan initialization did not complete";
    g_have_qwen35_gdn_q8_in_proj = 0;
    g_have_q6k_matmul = 0;
    g_have_q5k_matmul = 0;
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
    // This registration is coupled to the separately versioned V5/complete62
    // receipt migration. The frozen V4/complete61 identity is not extended.
    if (g_v41IndexerFloatControls) {
        g_v41IndexerScoreReason = "indexer module missing or pipeline creation failed";
        v41IndexerPipelineBuilt = buildKernel(g_kern[K_V41_INDEXER_SCORE], P("v41_indexer_score.spv"), 4, 12);
        if (v41IndexerPipelineBuilt) g_v41IndexerScoreReason.clear();
    }

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
        g_have_iq[f] = (!off && buildKernel(g_kern[kIQKernel[f]], P(kIQSpv[f]), 7, 4 * sizeof(int) + sizeof(float))) ? 1 : 0;
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
    g_have_v41_tail_rope_qk = 1;
    g_have_v41_shared_attention = 1;
    g_have_v41_indexer_score = v41IndexerPipelineBuilt ? 1 : 0;
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
        if (g_poolCount > 0) --g_poolCount;
        return b;
    }
    return allocBuffer(bytes, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
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
    if (b->weightArenaBound) {
        destroyBuffer(b);
        clearDescriptorBindingCache();
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
    if (g_v41SubmissionPendingFailure) return;
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
    if (g_batching) batchFlush();
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    return copyDeviceToHost(h, B((void*)d), bytes);
}

// V4.1 status/output readback deliberately avoids the generic aborting command
// helpers. The shared staging allocation and transfer counters remain canonical.
int fvk_v41_d2h(void* h, const void* d, size_t bytes) {
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (!g_ready || g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return VK_ERROR_INITIALIZATION_FAILED;
    Buffer* src = B(d);
    if (!h || !src || src->buf == VK_NULL_HANDLE || src->mem == VK_NULL_HANDLE ||
        bytes == 0 || bytes > src->bytes) return VK_ERROR_INITIALIZATION_FAILED;
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
    if (!checkedCounterAdd(g_d2hCount, 1) || !checkedCounterAdd(g_d2hBytes, bytes)) {
        g_transferCountersValid = false;
    }
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
        tsAfterDispatch(g_batchCmd, (uint32_t)K_COUNT, 0, 0);
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

void fvk_d2d(void* dst, const void* src, size_t bytes) {
    fvk_d2d_range(dst, 0, src, 0, bytes);
}

void fvk_d2d_off(void* dst, size_t dst_off, const void* src, size_t bytes) {
    fvk_d2d_range(dst, dst_off, src, 0, bytes);
}

// batch entry points (C ABI).
void fvk_batch_begin(void) { batchBegin(); }
void fvk_batch_flush(void) { dp_inc(g_dp.batchFlushes); batchFlush(); }
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
int fvk_batch_flush_status(void) { fvk_batch_flush(); return (int)g_submissionStatus; }
uint64_t fvk_h2d_bytes(void) { return g_h2dBytes.load(std::memory_order_relaxed); }
uint64_t fvk_direct_h2d_bytes(void) { return g_directH2DBytes.load(std::memory_order_relaxed); }
uint64_t fvk_d2h_bytes(void) { return g_d2hBytes.load(std::memory_order_relaxed); }
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
    return !g_batching && g_batchCmd == VK_NULL_HANDLE && g_batchOps == 0 &&
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
    if (g_v41SubmissionPendingFailure) return; if (g_dev) vkDeviceWaitIdle(g_dev); }

int fvk_have_q8(void) { return g_have_q8; }
int fvk_have_qwen35_gdn_q8_in_proj(void) { return g_have_qwen35_gdn_q8_in_proj; }
int fvk_have_glm_kda_wave32(void) { return g_have_glm_kda_wave32; }
int fvk_have_cooperative_matrix(void) { return g_have_coopmat; }
int fvk_have_q6k_matmul(void) { return g_have_q6k_matmul; }
int fvk_have_q5k_matmul(void) { return g_have_q5k_matmul; }
int fvk_have_q3k_matmul(void) { return g_have_q3k_matmul; }
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

void fvk_matmul_f32(const void* dW, const void* dX, void* dY, int out, int in, int P) {
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[3] = {B((void*)dW), B((void*)dX), B(dY)};
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

int fvk_have_v41_tail_rope_qk(void) {
    return g_ready && g_have_v41_tail_rope_qk && g_kern[K_V41_TAIL_ROPE_QK].pipe != VK_NULL_HANDLE;
}

int fvk_device_type(void) {
    if (!g_ready || g_phys == VK_NULL_HANDLE) return int(VK_PHYSICAL_DEVICE_TYPE_OTHER);
    VkPhysicalDeviceProperties properties{};
    vkGetPhysicalDeviceProperties(g_phys, &properties);
    return int(properties.deviceType);
}

int fvk_v41_tail_rope_qk_f32(const void* dQ, const void* dKV,
                            void* dQOut, void* dKVOut, const void* dSinCos,
                            int heads, int headDim, int rotaryDim) {
    if (!g_ready) return 1;
    if (!fvk_have_v41_tail_rope_qk()) return 3;
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
    if (!dispatch(g_kern[K_V41_TAIL_ROPE_QK], bufs, &pc, sizeof(pc), uint32_t(groups))) {
        return g_submissionStatus != VK_SUCCESS ? (int)g_submissionStatus : 4;
    }
    return (int)g_submissionStatus;
}

int fvk_have_v41_shared_attention(void) {
    return g_ready && g_have_v41_shared_attention && g_kern[K_V41_SHARED_ATTENTION].pipe != VK_NULL_HANDLE;
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
    // One scalar invocation per head initializes all four status words, then
    // owns that head's ordered first fault. There is no atomic winner race.
    if (!dispatch(g_kern[K_V41_SHARED_ATTENTION], bufs, &pc, sizeof(pc), uint32_t(heads))) {
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
    // Source-only candidate: no trusted compiler/source/numerical qualification
    // admission is integrated yet. Neither environment nor module bytes can
    // enable production dispatch. A future change must bind that admission to
    // the exact V5 archive/module/compiler/source/device witness identities.
    return 0;
}

const char* fvk_v41_indexer_score_unavailable_reason(void) {
    if (!g_ready) return "Vulkan initialization incomplete";
    if (g_submissionStatus != VK_SUCCESS || g_v41SubmissionPendingFailure) return "sticky Vulkan submission fault";
    if (!v41IndexerScorePrerequisites()) {
        return g_v41IndexerScoreReason.empty() ? "indexer pipeline unavailable" : g_v41IndexerScoreReason.c_str();
    }
    return "trusted source/compiler and numerical qualification admission not integrated";
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
        // Empty key/output handles are never dereferenced or bound.
        if (counts[i] == 0) { if (bufs[i] != nullptr) return 2; continue; }
        if (!bufs[i] || bufs[i]->buf == VK_NULL_HANDLE || bufs[i]->mem == VK_NULL_HANDLE ||
            counts[i] > std::numeric_limits<size_t>::max() / sizeof(float) ||
            counts[i] * sizeof(float) != bufs[i]->bytes || bufs[i]->bytes > g_maxStorageBufferRange ||
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

void fvk_swiglu_f32(const void* dG, const void* dU, void* dY, int n) {
    fvk_swiglu_limit_f32(dG, dU, dY, n, 0.0f);
}

void fvk_swiglu_limit_f32(const void* dG, const void* dU, void* dY, int n, float limit) {
    struct { int n; float limit; } pc{n, limit};
    Buffer* bufs[3] = {B((void*)dG), B((void*)dU), B(dY)};
    dispatch(g_kern[K_SWIGLU], bufs, &pc, sizeof(pc), ((uint32_t)n + 255) / 256);
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
    if (g_batching) batchFlush(); // host fence: the logits must be materialized first
    Buffer* idx = (Buffer*)fvk_malloc(sizeof(int));
    struct { int n; } pc{n};
    Buffer* bufs[2] = {B((void*)dLogits), idx};
    dispatch(g_kern[K_ARGMAX], bufs, &pc, sizeof(pc), 1); // not batching now → submits
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
    // Prefill-class (>=8) route. The explicit request path (debug mode 1 or env
    // "1") requires a true self-check verdict. An unset variable stays stock until a
    // default flip is earned; `=0` also stays stock.
    constexpr bool kGdnTiledDefaultFlipEarned = false; // pending the [HW-WITNESSED] Strix A/B, #12735, #12098
    const bool tiledRequested = g_gdn_prefill_mode == 1 ||
        (g_gdn_prefill_mode == -1 && (requestedByEnv || kGdnTiledDefaultFlipEarned));
    const bool tiled = tiledRequested && g_have_gdn_prefill_tiled && g_gdn_tiled_verdict != 0 &&
        tokens >= 8 && (uint32_t)tokens <= g_gdn_prefill_max_groups_y && productionGeometry;
    // Verify-width (1..7 token) route. `FAK_VULKAN_GDN_PREFILL_TILED=0` forces the
    // stock kernel; an explicit request (debug mode 1 or env "1") selects the
    // candidate but still requires a true self-check verdict. With the variable
    // unset the route follows the verdict only once the default flip is earned for
    // the width class; until then the default stays stock.
    constexpr bool kGdnVerifyDefaultFlipEarned = false; // pending the [HW-WITNESSED] Strix A/B, #12735, #12098
    const bool verifyRequested = !optOut && g_gdn_verify_mode != 0 &&
        (g_gdn_prefill_mode == 1 || g_gdn_verify_mode == 1 || requestedByEnv ||
         kGdnVerifyDefaultFlipEarned);
    const bool verify = verifyRequested && g_have_gdn_verify_tiled && g_gdn_verify_verdict &&
        tokens >= 1 && tokens < 8 && productionGeometry;
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

static inline bool useQ4KWave32(int P) {
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
    // Physical A/B on gfx1151 resulted in REJECT_RETAIN_SCALAR (0.63x and 0.76x); default retains scalar.
    return false;
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
    if (useQ4KWave32(P)) {
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
    struct PC { int out, in, p; } pc{out, in, P};
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
