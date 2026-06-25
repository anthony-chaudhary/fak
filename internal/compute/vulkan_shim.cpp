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
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <string>
#include <vector>
#include <unordered_map>

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

// A device buffer: VkBuffer + its memory + byte size. The opaque handle Go holds is a
// Buffer* — never a host address.
struct Buffer {
    VkBuffer       buf = VK_NULL_HANDLE;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    size_t         bytes = 0;
    VkMemoryPropertyFlags props = 0;
};

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

// One compute kernel: pipeline + layout + descriptor set layout + how many storage buffers
// it binds + push-constant byte size.
struct Kernel {
    VkShaderModule        shader = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl    = VK_NULL_HANDLE;
    VkPipelineLayout      layout = VK_NULL_HANDLE;
    VkPipeline            pipe   = VK_NULL_HANDLE;
    int                   nbuf   = 0;
    uint32_t              pcsize = 0;
};

enum KId { K_MATMUL, K_MATMUL_ADD, K_MATMUL_ARGMAX, K_MATMUL_ARGMAX_BLOCKS, K_MATMUL2, K_MATMUL3, K_RMSNORM, K_RMSNORM_MATMUL, K_RMSNORM_MATMUL2, K_RMSNORM_MATMUL3, K_RMSNORM_MATMUL_ARGMAX_BLOCKS, K_ROPE, K_SWIGLU, K_SWIGLU_MATMUL_ADD, K_ADD, K_ADD_BIAS, K_ATTENTION, K_ARGMAX, K_ARGMAX_PAIRS, K_Q8_MATMUL, K_Q8_MATMUL2, K_Q8_MATMUL3, K_RMSNORM_Q8_MATMUL2, K_RMSNORM_Q8_MATMUL3, K_SWIGLU_Q8_MATMUL_ADD, K_COUNT };
Kernel g_kern[K_COUNT];

// Q8 fast-path availability (set in fvk_init from the device's 8-bit-storage + int8 features).
int g_have_q8 = 0;

VkDescriptorPool g_descpool = VK_NULL_HANDLE;

// 12 covers the widest fused kernel: rmsnorm_q8_matmul3 binds 11 (3 Q8 weight code+scale
// pairs + X + NormW + Q/K/V outputs). At 10 that dispatch was silently skipped, zeroing its
// output. Sizes the per-dispatch descriptor/buffer stack arrays; well under the device's
// storage-buffer binding limit and the 32768-descriptor pool's headroom.
static constexpr int MAX_DISPATCH_BUFS = 12;

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

void destroyBuffer(Buffer* b) {
    if (!b) return;
    if (b->buf) vkDestroyBuffer(g_dev, b->buf, nullptr);
    if (b->mem) vkFreeMemory(g_dev, b->mem, nullptr);
    delete b;
}

Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage);

size_t stageCapacity(size_t bytes) {
    size_t cap = 64 * 1024;
    while (cap < bytes && cap <= (((size_t)-1) / 2)) cap *= 2;
    return cap < bytes ? bytes : cap;
}

Buffer* stagingBuffer(size_t bytes) {
    if (bytes == 0) return nullptr;
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
    if (!g_stage) return nullptr;
    VkResult r = vkMapMemory(g_dev, g_stage->mem, 0, cap, 0, &g_stageMapped);
    if (r != VK_SUCCESS || !g_stageMapped) {
        fprintf(stderr, "fak-vulkan: vkMapMemory(stage %zu bytes) failed VkResult=%d\n", cap, (int)r);
        destroyBuffer(g_stage);
        g_stage = nullptr;
        g_stageMapped = nullptr;
        g_stageCap = 0;
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
    Buffer* b = new Buffer();
    b->bytes = bytes;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes ? bytes : 1;
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
    auto tryAlloc = [&](VkMemoryPropertyFlags want, VkDeviceMemory* out) -> VkResult {
        VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ai.allocationSize = req.size;
        ai.memoryTypeIndex = findMemType(req.memoryTypeBits, want);
        if (ai.memoryTypeIndex == UINT32_MAX) return VK_ERROR_FEATURE_NOT_PRESENT;
        return vkAllocateMemory(g_dev, &ai, nullptr, out);
    };

    // First attempt; on out-of-memory / too-many-allocations, drain the recycle pool (which
    // holds buffers nothing references right now) and retry before considering a slower
    // host-visible storage fallback for device-local tensors.
    VkMemoryPropertyFlags actualProps = props;
    VkResult r = tryAlloc(props, &b->mem);
    if (allocPressure(r)) {
        drainPool();
        r = tryAlloc(props, &b->mem);
    }
    if (r != VK_SUCCESS && (props & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
        (usage & VK_BUFFER_USAGE_STORAGE_BUFFER_BIT)) {
        VkMemoryPropertyFlags fallback =
            VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        VkResult fr = tryAlloc(fallback, &b->mem);
        if (fr == VK_SUCCESS) {
            fprintf(stderr,
                "fak-vulkan: device-local alloc(%zu bytes) failed VkResult=%d; using host-visible storage\n",
                bytes, (int)r);
            r = fr;
            actualProps = fallback;
        }
    }
    if (r == VK_ERROR_FEATURE_NOT_PRESENT) {
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
    return b;
}

// device-local storage buffer used for all tensors (the residency seam).
const VkBufferUsageFlags STORAGE_USAGE =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
    VK_BUFFER_USAGE_TRANSFER_DST_BIT;

std::vector<Buffer*>          g_attentionScratch;
size_t                        g_attentionScratchCursor = 0;

size_t scratchCapacity(size_t bytes) {
    size_t cap = 4 * 1024;
    while (cap < bytes && cap <= (((size_t)-1) / 2)) cap *= 2;
    return cap < bytes ? bytes : cap;
}

Buffer* batchAttentionScratch(size_t bytes) {
    if (bytes == 0) bytes = 4;
    size_t slot = g_attentionScratchCursor++;
    if (slot >= g_attentionScratch.size()) g_attentionScratch.resize(slot + 1, nullptr);

    Buffer* b = g_attentionScratch[slot];
    if (b && b->bytes >= bytes) return b;

    if (b) {
        destroyBuffer(b);
        g_attentionScratch[slot] = nullptr;
        clearDescriptorBindingCache();
    }
    b = allocBuffer(scratchCapacity(bytes), VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, STORAGE_USAGE);
    if (!b) {
        fprintf(stderr, "fak-vulkan: attention scratch allocation failed (%zu bytes)\n", bytes);
        abort();
    }
    g_attentionScratch[slot] = b;
    return b;
}

void freeAttentionScratch() {
    for (Buffer* b : g_attentionScratch) destroyBuffer(b);
    if (!g_attentionScratch.empty()) clearDescriptorBindingCache();
    g_attentionScratch.clear();
    g_attentionScratchCursor = 0;
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

// ---- batched submission state ---------------------------------------------------
// When g_batching is true, compute dispatches RECORD into g_batchCmd (with a
// compute->compute barrier between them) instead of each submitting its own buffer. The
// descriptor sets recorded into the open buffer must outlive recording, so they are parked
// in g_batchSets and freed after the single submit. g_batchOps counts recorded ops so an
// empty flush is a cheap no-op.
bool                          g_batching  = false;
VkCommandBuffer               g_batchCmd  = VK_NULL_HANDLE;
std::vector<DescriptorSetRecord> g_batchSets;
std::vector<Buffer*>          g_batchFreed;   // buffers freed mid-batch, recycled after submit
int                           g_batchOps  = 0;

// Attention scores are scratch, but a token batch records all layers before submit, so each
// attention call in the batch needs a distinct buffer. Keep a reusable ring by call index and
// grow each slot geometrically as the sequence length increases instead of allocating/freeing
// nLayers differently-sized score buffers every token.
// A full compute->compute barrier: every recorded op may read the previous op's output
// buffer, so each dispatch is fenced against the prior by a global shader-write->shader-read
// barrier. Coarse but correct; per-buffer barriers are a later refinement.
void recordComputeBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cmd,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &mb, 0, nullptr, 0, nullptr);
}

void batchBegin() {
	if (g_batching) return;          // already recording — the model brackets each token
	g_batchCmd = beginCmd();
	g_batching = true;
	g_batchOps = 0;
	g_batchSets.clear();
	g_attentionScratchCursor = 0;
}

void batchFlush() {
    if (!g_batching) return;
    bool hadWork = g_batchOps > 0;
    if (hadWork) {
        endSubmitWait(g_batchCmd);   // single submit + fence for the whole recorded chain
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
	std::vector<Buffer*> freed;   freed.swap(g_batchFreed);
	for (Buffer* b : freed)   fvk_free(b);
}

// staging copy host<->device through one persistent HOST_VISIBLE scratch buffer.
void copyHostToDevice(Buffer* dst, const void* host, size_t bytes) {
    if (bytes == 0 || !dst || !host) return;
    Buffer* stage = stagingBuffer(bytes);
    if (!stage) return;
    memcpy(g_stageMapped, host, bytes);
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, stage->buf, dst->buf, 1, &region);
    endSubmitWait(cmd);
}

void copyDeviceToHost(void* host, Buffer* src, size_t bytes) {
    if (bytes == 0 || !host || !src) return;
    Buffer* stage = stagingBuffer(bytes);
    if (!stage) return;
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{0, 0, bytes};
    vkCmdCopyBuffer(cmd, src->buf, stage->buf, 1, &region);
    endSubmitWait(cmd);
    memcpy(host, g_stageMapped, bytes);
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

bool buildKernel(Kernel& k, const std::string& spvPath, int nbuf, uint32_t pcsize) {
    std::vector<char> code = readFile(spvPath);
    if (code.empty()) return false;
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
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = stage;
    cpi.layout = k.layout;
    if (vkCreateComputePipelines(g_dev, VK_NULL_HANDLE, 1, &cpi, nullptr, &k.pipe) != VK_SUCCESS) return false;

    k.nbuf = nbuf;
    k.pcsize = pcsize;
    return true;
}

DescriptorSetRecord acquireDescriptorSet(Kernel& k) {
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
    if (vkAllocateDescriptorSets(g_dev, &dsi, &ds) != VK_SUCCESS) {
        fprintf(stderr, "fak-vulkan: descriptor alloc failed\n");
        return rec;
    }
    rec.set = ds;
    return rec;
}

void recycleDescriptorSet(DescriptorSetRecord rec) {
    if (rec.set) g_descSetPool[rec.layout].push_back(rec);
}

// dispatch: bind `bufs` (nbuf of them) + push constants, run groupsX workgroups.
void dispatch(Kernel& k, Buffer** bufs, const void* pc, uint32_t pcsize, uint32_t groupsX) {
    if (k.nbuf > MAX_DISPATCH_BUFS) {
        fprintf(stderr, "fak-vulkan: dispatch skipped; kernel has %d buffers, max %d\n",
                k.nbuf, MAX_DISPATCH_BUFS);
        return;
    }
    for (int i = 0; i < k.nbuf; ++i) {
        if (!bufs[i]) {
            fprintf(stderr, "fak-vulkan: dispatch skipped; buffer %d is null\n", i);
            return;
        }
    }
    DescriptorSetRecord rec = acquireDescriptorSet(k);
    if (!rec.set) {
        return;
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
        // RECORD into the open batch buffer: barrier against the prior op, then dispatch.
        // The descriptor set must outlive recording, so park it for post-submit free.
        if (g_batchOps > 0) recordComputeBarrier(g_batchCmd);
        vkCmdBindPipeline(g_batchCmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(g_batchCmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &rec.set, 0, nullptr);
        if (pcsize > 0) vkCmdPushConstants(g_batchCmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
        vkCmdDispatch(g_batchCmd, groupsX, 1, 1);
        g_batchSets.push_back(rec);
        ++g_batchOps;
        return;
    }

    // Unbatched: one-shot submit + fence (the original per-op path).
    VkCommandBuffer cmd = beginCmd();
    vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
    vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &rec.set, 0, nullptr);
    if (pcsize > 0) vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, pcsize, pc);
    vkCmdDispatch(cmd, groupsX, 1, 1);
    endSubmitWait(cmd);
    recycleDescriptorSet(rec);
}

inline Buffer* B(const void* h) { return (Buffer*)h; }
inline Buffer* B(void* h)       { return (Buffer*)h; }

} // namespace

// ---- C ABI ----------------------------------------------------------------------
extern "C" {

int fvk_init(char* name, int namelen, int* is_discrete, const char* spirv_dir) {
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
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys, &props);
    if (name && namelen > 0) { strncpy(name, props.deviceName, namelen - 1); name[namelen - 1] = 0; }
    if (is_discrete) *is_discrete = discrete ? 1 : 0;
    vkGetPhysicalDeviceMemoryProperties(g_phys, &g_memprops);

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
    f8.pNext = &fi8;
    VkPhysicalDeviceFeatures2 feat2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feat2.pNext = &f8;
    vkGetPhysicalDeviceFeatures2(g_phys, &feat2);
    g_have_q8 = (f8.storageBuffer8BitAccess && fi8.shaderInt8) ? 1 : 0;

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    VkPhysicalDevice8BitStorageFeatures e8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDeviceShaderFloat16Int8Features ei8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_FLOAT16_INT8_FEATURES};
    if (g_have_q8) {
        e8.storageBuffer8BitAccess = VK_TRUE;
        ei8.shaderInt8 = VK_TRUE;
        e8.pNext = &ei8;
        dci.pNext = &e8;
    }
    if (vkCreateDevice(g_phys, &dci, nullptr, &g_dev) != VK_SUCCESS) return 4;
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
    ok &= buildKernel(g_kern[K_SWIGLU],    P("swiglu.spv"),    3, sizeof(int));
    ok &= buildKernel(g_kern[K_SWIGLU_MATMUL_ADD], P("swiglu_matmul_add.spv"), 4, 3 * sizeof(int));
    ok &= buildKernel(g_kern[K_ADD],       P("add.spv"),       2, sizeof(int));
    ok &= buildKernel(g_kern[K_ADD_BIAS],  P("add_bias.spv"),  2, 2 * sizeof(int));
    ok &= buildKernel(g_kern[K_ATTENTION], P("attention.spv"), 5, 4 * sizeof(int) + sizeof(float));
    ok &= buildKernel(g_kern[K_ARGMAX],    P("argmax.spv"),    2, sizeof(int));
    ok &= buildKernel(g_kern[K_ARGMAX_PAIRS], P("argmax_pairs.spv"), 3, sizeof(int));
    if (!ok) return 8;
    // Q8 kernel is built only when the device advertised the int8/8-bit-storage features; its
    // SPIR-V uses them, so loading it without the enabled device feature would be invalid. If
    // it fails to build, disable the Q8 path rather than failing init (f32 stays available).
    if (g_have_q8) {
        if (!buildKernel(g_kern[K_Q8_MATMUL], P("q8_matmul.spv"), 4, 3 * sizeof(int)) ||
            !buildKernel(g_kern[K_Q8_MATMUL2], P("q8_matmul2.spv"), 7, 4 * sizeof(int)) ||
            !buildKernel(g_kern[K_Q8_MATMUL3], P("q8_matmul3.spv"), 10, 5 * sizeof(int)) ||
            !buildKernel(g_kern[K_RMSNORM_Q8_MATMUL2], P("rmsnorm_q8_matmul2.spv"), 8, 4 * sizeof(int) + sizeof(float)) ||
            !buildKernel(g_kern[K_RMSNORM_Q8_MATMUL3], P("rmsnorm_q8_matmul3.spv"), 11, 5 * sizeof(int) + sizeof(float)) ||
            !buildKernel(g_kern[K_SWIGLU_Q8_MATMUL_ADD], P("swiglu_q8_matmul_add.spv"), 5, 3 * sizeof(int))) {
            g_have_q8 = 0;
        }
    }

    g_ready = true;
    return 0;
}

void* fvk_malloc(size_t bytes) {
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

void* fvk_malloc_hostvis(size_t bytes) {
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
    if (g_batching) { g_batchFreed.push_back(b); return; }
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
    bool resumeBatch = g_batching;
    if (resumeBatch) batchFlush();
    copyHostToDevice(B(d), h, bytes);
    if (resumeBatch) batchBegin();
}

// d2h is a true host fence (the final logits Read): flush the recorded batch so the compute
// has actually executed, then copy device->host.
void fvk_d2h(void* h, const void* d, size_t bytes) {
    if (g_batching) batchFlush();
    copyDeviceToHost(h, B((void*)d), bytes);
}

// device->device copies (RoPE's copy-then-rotate, the KV append) are RECORDED into the open
// batch with a preceding barrier, so they stay ordered against the compute that produced the
// source — no premature submit. Unbatched, they take the one-shot path.
void fvk_d2d_range(void* dst, size_t dst_off, const void* src, size_t src_off, size_t bytes) {
    if (bytes == 0 || !dst || !src) return;
    if (g_batching) {
        if (g_batchOps > 0) recordComputeBarrier(g_batchCmd);
        VkBufferCopy region{src_off, dst_off, bytes};
        vkCmdCopyBuffer(g_batchCmd, B((void*)src)->buf, B(dst)->buf, 1, &region);
        ++g_batchOps;
        return;
    }
    VkCommandBuffer cmd = beginCmd();
    VkBufferCopy region{src_off, dst_off, bytes};
    vkCmdCopyBuffer(cmd, B((void*)src)->buf, B(dst)->buf, 1, &region);
    endSubmitWait(cmd);
}

void fvk_d2d(void* dst, const void* src, size_t bytes) {
    fvk_d2d_range(dst, 0, src, 0, bytes);
}

void fvk_d2d_off(void* dst, size_t dst_off, const void* src, size_t bytes) {
    fvk_d2d_range(dst, dst_off, src, 0, bytes);
}

// batch entry points (C ABI).
void fvk_batch_begin(void) { batchBegin(); }
void fvk_batch_flush(void) { batchFlush(); }

void fvk_sync(void) { if (g_dev) vkDeviceWaitIdle(g_dev); }

int fvk_have_q8(void) { return g_have_q8; }

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

void fvk_trim_pool(void) {
    if (!g_dev) return;
    if (g_batching) batchFlush();
    drainPool();
    freeAttentionScratch();
}

void fvk_trim_pool_if_over(size_t max_buffers) {
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

void fvk_q8_matmul_f32(const void* dWcodes, const void* dWscale, const void* dX, void* dY,
                       int out, int in, int P) {
    if (!g_have_q8) {
        fprintf(stderr, "fak-vulkan: q8_matmul requested but int8/8-bit-storage features are unavailable\n");
        abort();
    }
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[4] = {B((void*)dWcodes), B((void*)dWscale), B((void*)dX), B(dY)};
    uint32_t outGroups = ((uint32_t)out + 255u) / 256u;
    dispatch(g_kern[K_Q8_MATMUL], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
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
    struct { int outDim, inDim, P; } pc{out, in, P};
    Buffer* bufs[5] = {
        B((void*)dWcodes), B((void*)dWscale), B((void*)dG), B((void*)dU), B(dD),
    };
    uint32_t outGroups = ((uint32_t)out + 255u) / 256u;
    dispatch(g_kern[K_SWIGLU_Q8_MATMUL_ADD], bufs, &pc, sizeof(pc), (uint32_t)P * outGroups);
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

void fvk_swiglu_f32(const void* dG, const void* dU, void* dY, int n) {
    struct { int n; } pc{n};
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
    size_t scoreBytes = (size_t)nH * nPos * sizeof(float);
    Buffer* scores = g_batching ? batchAttentionScratch(scoreBytes) : (Buffer*)fvk_malloc(scoreBytes);
    if (!scores) {
        fprintf(stderr, "fak-vulkan: attention scratch allocation failed (%zu bytes)\n", scoreBytes);
        abort();
    }
    struct { int nPos, nH, nKV, hd; float scale; } pc{nPos, nH, nKV, hd, scale};
    Buffer* bufs[5] = {B((void*)dQ), B((void*)dK), B((void*)dV), B(dOut), scores};
    dispatch(g_kern[K_ATTENTION], bufs, &pc, sizeof(pc), (uint32_t)nH);
    if (!g_batching) {
        fvk_free(scores);
    }
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
