// SOURCE-ONLY: this file is not included by any runtime translation unit.
// Include AFTER vulkan_v41_packed_storage.inc at the end of vulkan_shim.cpp.
#include "vulkan_v41_packed_attention.h"
#include <new>

// Deliberately no setter, registry ID, fallback kernel or capability. Replace
// only in a separately reviewed bundle+ABI+float-control+retirement change.
// Proposed append-only stem: v41_packed_main_attention; 5 bindings, 32 bytes.
static Kernel* v41PackedAttentionKernel() { return nullptr; }

int fvk_v41_packed_attention_admission(uint64_t generation) {
    int result = v41PackedState(generation);
    if (result) return result;
    Kernel* k = v41PackedAttentionKernel();
    return k && k->pipe && k->dsl && k->layout && k->nbuf == 5 && k->pcsize == 32 ? 0 : 3;
}

struct V41PackedPush {
    int32_t heads, headDim, selectedRows, mode, hasSink;
    float scale;
    uint32_t rowStrideBytes, formatVersion;
};
static_assert(sizeof(float) == 4 && sizeof(V41PackedPush) == 32, "packed push size");
static_assert(offsetof(V41PackedPush, heads) == 0 && offsetof(V41PackedPush, headDim) == 4 &&
    offsetof(V41PackedPush, selectedRows) == 8 && offsetof(V41PackedPush, mode) == 12 &&
    offsetof(V41PackedPush, hasSink) == 16 && offsetof(V41PackedPush, scale) == 20 &&
    offsetof(V41PackedPush, rowStrideBytes) == 24 && offsetof(V41PackedPush, formatVersion) == 28,
    "packed push offsets");

// Every temporary allocation, including unpublished failure output, has an
// explicit owner BEFORE allocation. Pending submission retains the entire node
// (buffers and descriptor pool) for process retirement. Never recycle uncertainty.
struct V41PackedWork {
    Buffer* buffers[5]{};
    VkDescriptorPool pool = VK_NULL_HANDLE;
    V41PackedWork* next = nullptr;
};
static V41PackedWork* v41PackedRetained = nullptr;
struct V41PackedWorkScope {
    V41PackedWork* work;
    ~V41PackedWorkScope() {
        if (!work || g_v41SubmissionPendingFailure) return;
        // vulkanMu serializes this entire synchronous operation.
        v41PackedRetained = work->next;
        if (work->pool) vkDestroyDescriptorPool(g_dev, work->pool, nullptr);
        for (Buffer* b : work->buffers) if (b) destroyBuffer(b);
        delete work;
    }
};
static int v41PackedMap(Buffer* b, void** p) {
    VkResult r = vkMapMemory(g_dev, b->mem, 0, b->bytes, 0, p);
    return r == VK_SUCCESS && *p ? 0 : (r == VK_SUCCESS ? int(VK_ERROR_UNKNOWN) : int(r));
}

int fvk_v41_packed_attention(const void* source, size_t capacity, size_t initialized_rows,
    uint64_t generation, const uint32_t* rows, uint32_t count,
    const float* q, int32_t heads, const float* sink, int32_t has_sink,
    int32_t mode, float scale, float* output, uint32_t* status) {
    int code = v41PackedState(generation);
    if (code) return code; // before dereferencing stale native handles
    const Buffer* src = B(source);
    if (!v41PackedBacking(src, generation, capacity) || initialized_rows > capacity / 288 ||
        heads <= 0 || !count || count > uint32_t(INT32_MAX - 2047) ||
        (mode != 0 && mode != 1) || (has_sink != 0 && has_sink != 1) ||
        !std::isfinite(scale) || scale == 0 || !q || !rows || !output || !status ||
        (has_sink && !sink)) return 2;
    uint16_t endian = 1;
    if (*reinterpret_cast<unsigned char*>(&endian) != 1) return 3;
    const uint64_t elements = uint64_t(heads) * 512;
    const uint64_t packedBytes = uint64_t(count) * 288;
    const uint64_t sizes[5] = {elements * 4, packedBytes,
        (has_sink ? uint64_t(heads) : 1) * 4, elements * 4, uint64_t(heads) * 16};
    if (elements > INT32_MAX || packedBytes > UINT32_MAX || uint64_t(heads) * 4 > UINT32_MAX) return 2;
    VkPhysicalDeviceProperties props{};
    vkGetPhysicalDeviceProperties(g_phys, &props);
    const auto& limits = props.limits;
    if (uint64_t(heads) > limits.maxComputeWorkGroupCount[0] ||
        limits.maxComputeWorkGroupSize[0] < 128 || limits.maxComputeWorkGroupInvocations < 128 ||
        limits.maxComputeSharedMemorySize < 17936 || limits.maxPushConstantsSize < 32 ||
        limits.maxPerStageDescriptorStorageBuffers < 5 || limits.maxDescriptorSetStorageBuffers < 5) return 3;
    // Descriptor offsets are exactly zero, satisfying 4-byte and device storage
    // offset alignment. Actual allocation requirements are handled by allocBuffer.
    for (uint64_t n : sizes) if (n > SIZE_MAX || n > limits.maxStorageBufferRange) return 2;
    for (uint32_t i = 0; i < count; ++i) if (rows[i] >= initialized_rows) return 2;
    for (uint64_t i = 0; i < elements; ++i) if (!std::isfinite(q[i])) return 2;
    if (has_sink) for (int32_t i = 0; i < heads; ++i) if (!std::isfinite(sink[i])) return 2;
    if ((code = fvk_v41_packed_attention_admission(generation)) != 0) return code;
    Kernel& k = *v41PackedAttentionKernel();
    V41PackedWork* w = new (std::nothrow) V41PackedWork;
    if (!w) return int(VK_ERROR_OUT_OF_HOST_MEMORY);
    w->next = v41PackedRetained;
    v41PackedRetained = w;
    V41PackedWorkScope scope{w};
    try {
        for (int i = 0; i < 5; ++i) {
            w->buffers[i] = allocBuffer(size_t(sizes[i]), VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT);
            Buffer* b = w->buffers[i];
            if (!b) return int(VK_ERROR_OUT_OF_DEVICE_MEMORY);
            if (b->bytes != sizes[i] || b->memoryOffset != 0 || b->deviceGeneration != generation ||
                !b->buf || !b->mem || b->allocationBytes < sizes[i]) return 2;
            if (b->mem == src->mem || b->buf == src->buf) return 2;
            for (int j = 0; j < i; ++j) if (b->mem == w->buffers[j]->mem || b->buf == w->buffers[j]->buf) return 2;
        }
        // CPU inputs copied into owned coherent allocations; no borrowed host
        // pointer survives this call. Output/status are fresh and initialized.
        for (int i : {0, 2, 3, 4}) {
            void* mapped = nullptr;
            if ((code = v41PackedMap(w->buffers[i], &mapped)) != 0) return code;
            if (i == 0) memcpy(mapped, q, size_t(sizes[i]));
            else if (i == 2 && has_sink) memcpy(mapped, sink, size_t(sizes[i]));
            else memset(mapped, i == 4 ? 0xff : 0, size_t(sizes[i]));
            vkUnmapMemory(g_dev, w->buffers[i]->mem);
        }
        VkCommandBuffer cmd = v41BeginCmdChecked();
        if (!cmd) return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
        v41PackedBarrier(cmd);
        for (uint32_t i = 0; i < count; ++i) {
            VkBufferCopy region{VkDeviceSize(rows[i]) * 288, VkDeviceSize(i) * 288, 288};
            vkCmdCopyBuffer(cmd, src->buf, w->buffers[1]->buf, 1, &region);
        }
        VkMemoryBarrier toHost{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        toHost.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        toHost.dstAccessMask = VK_ACCESS_HOST_READ_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &toHost, 0, nullptr, 0, nullptr);
        if (!v41EndSubmitWaitChecked(cmd, true)) return int(g_submissionStatus);
        // Independently inspect ACTUAL gathered scale bytes before dispatch.
        void* mapped = nullptr;
        if ((code = v41PackedMap(w->buffers[1], &mapped)) != 0) return code;
        bool valid = true;
        const auto* bytes = static_cast<const unsigned char*>(mapped);
        for (uint32_t i = 0; i < count; ++i) for (uint32_t s = 256; s < 288; ++s) {
            unsigned char v = bytes[size_t(i) * 288 + s];
            if (!v || v >= 127) valid = false;
        }
        vkUnmapMemory(g_dev, w->buffers[1]->mem);
        if (!valid) return 2;
        VkDescriptorPoolSize poolSize{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 5};
        VkDescriptorPoolCreateInfo pi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        pi.maxSets = 1; pi.poolSizeCount = 1; pi.pPoolSizes = &poolSize;
        VkResult r = vkCreateDescriptorPool(g_dev, &pi, nullptr, &w->pool);
        if (r != VK_SUCCESS) return int(r);
        VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        ai.descriptorPool = w->pool; ai.descriptorSetCount = 1; ai.pSetLayouts = &k.dsl;
        VkDescriptorSet ds = VK_NULL_HANDLE;
        r = vkAllocateDescriptorSets(g_dev, &ai, &ds);
        if (r != VK_SUCCESS) return int(r);
        VkDescriptorBufferInfo bi[5]{};
        VkWriteDescriptorSet writes[5]{};
        for (uint32_t i = 0; i < 5; ++i) {
            bi[i] = VkDescriptorBufferInfo{w->buffers[i]->buf, 0, sizes[i]};
            writes[i] = VkWriteDescriptorSet{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
            writes[i].dstSet = ds; writes[i].dstBinding = i; writes[i].descriptorCount = 1;
            writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &bi[i];
        }
        vkUpdateDescriptorSets(g_dev, 5, writes, 0, nullptr);
        cmd = v41BeginCmdChecked();
        if (!cmd) return g_submissionStatus != VK_SUCCESS ? int(g_submissionStatus) : 4;
        VkMemoryBarrier toShader{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        toShader.srcAccessMask = VK_ACCESS_HOST_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
        toShader.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_HOST_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
            VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &toShader, 0, nullptr, 0, nullptr);
        V41PackedPush pc{heads, 512, int32_t(count), mode, has_sink, scale, 288, 1};
        vkCmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.pipe);
        vkCmdBindDescriptorSets(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, k.layout, 0, 1, &ds, 0, nullptr);
        vkCmdPushConstants(cmd, k.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(pc), &pc);
        vkCmdDispatch(cmd, uint32_t(heads), 1, 1);
        toHost.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT,
            0, 1, &toHost, 0, nullptr, 0, nullptr);
        if (!v41EndSubmitWaitChecked(cmd, true)) return int(g_submissionStatus);
        if ((code = v41PackedMap(w->buffers[4], &mapped)) != 0) return code;
        memcpy(status, mapped, size_t(sizes[4]));
        vkUnmapMemory(g_dev, w->buffers[4]->mem);
        for (uint64_t i = 0; i < uint64_t(heads) * 4; ++i) if (status[i] != 0) return 5;
        if ((code = v41PackedMap(w->buffers[3], &mapped)) != 0) return code;
        memcpy(output, mapped, size_t(sizes[3]));
        vkUnmapMemory(g_dev, w->buffers[3]->mem);
        return 0;
    } catch (...) { return int(VK_ERROR_OUT_OF_HOST_MEMORY); }
}
