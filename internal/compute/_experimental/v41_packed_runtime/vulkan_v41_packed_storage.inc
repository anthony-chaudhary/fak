// SOURCE-ONLY: this file is not included by any runtime translation unit.
// Included once at the END of vulkan_shim.cpp. Not a separate translation unit.
#include "vulkan_v41_packed_storage.h"

int fvk_v41_packed_state(uint64_t* generation) {
    if (generation) *generation = 0;
    if (g_submissionStatus != VK_SUCCESS) return (int)g_submissionStatus;
    if (g_v41SubmissionPendingFailure) return (int)VK_ERROR_UNKNOWN;
    if (!g_ready || !g_dev || !g_cmdpool || !g_submitFence ||
        !g_memoryDeviceGenerationReady || !g_memoryDeviceGeneration) return 3;
    // v41BeginCmdChecked drains through an older unchecked helper. Refuse every
    // open/in-flight owner rather than silently flushing or claiming a fence.
    if (g_batching || g_stepBatch || g_inflight.active || g_batchCmd != VK_NULL_HANDLE ||
        g_restore.stage || g_restore.cmd != VK_NULL_HANDLE) return 4;
    if (generation) *generation = g_memoryDeviceGeneration;
    return 0;
}

static int v41PackedState(uint64_t generation) {
    uint64_t current = 0;
    int status = fvk_v41_packed_state(&current);
    return status ? status : (current == generation ? 0 : 3);
}

static bool v41PackedBacking(const Buffer* b, uint64_t generation, size_t capacity) {
    // Only private Go owner handles reach here; they cannot outlive device
    // teardown. Check generation BEFORE dereferencing any retained native handle.
    return b && capacity && capacity % 288 == 0 && b->bytes == capacity &&
        b->buf != VK_NULL_HANDLE && b->mem != VK_NULL_HANDLE &&
        b->deviceGeneration == generation && b->allocationID &&
        !b->weightArenaBound && !b->offloadPage && !b->mapped &&
        b->memoryOffset == 0 && b->allocationBytes >= capacity;
}

// Include transfer destinations explicitly: the generic compute helper only
// includes transfer READ and is not sufficient for packed overwrite hazards.
static void v41PackedBarrier(VkCommandBuffer cmd) {
    VkMemoryBarrier barrier{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    barrier.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    barrier.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT |
        VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT;
    vkCmdPipelineBarrier(cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
        0, 1, &barrier, 0, nullptr, 0, nullptr);
}

int fvk_v41_packed_zero(void* dst, uint64_t generation, size_t capacity) {
    int status = v41PackedState(generation);
    if (status) return status;
    Buffer* d = B(dst);
    if (!v41PackedBacking(d, generation, capacity)) return 2;
    VkCommandBuffer cmd = v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return (int)g_submissionStatus;
    v41PackedBarrier(cmd);
    vkCmdFillBuffer(cmd, d->buf, 0, capacity, 0);
    v41PackedBarrier(cmd);
    if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
    return 0;
}

int fvk_v41_packed_upload(void* dst, uint64_t generation, size_t capacity,
                          size_t offset, const void* host, size_t bytes) {
    int status = v41PackedState(generation);
    if (status) return status;
    Buffer* d = B(dst);
    if (!v41PackedBacking(d, generation, capacity) || offset % 288 || bytes % 288 ||
        offset > capacity || bytes > capacity - offset || (bytes && !host)) return 2;
    if (!bytes) return 0;
    Buffer* stage = allocBuffer(bytes, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
    if (!stage) return (int)VK_ERROR_OUT_OF_DEVICE_MEMORY;
    void* mapped = nullptr;
    VkResult result = vkMapMemory(g_dev, stage->mem, 0, bytes, 0, &mapped);
    if (result != VK_SUCCESS || !mapped) {
        destroyBuffer(stage);
        return result < VK_SUCCESS ? (int)result : (int)VK_ERROR_UNKNOWN;
    }
    memcpy(mapped, host, bytes);
    vkUnmapMemory(g_dev, stage->mem);
    VkCommandBuffer cmd = v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) { destroyBuffer(stage); return (int)g_submissionStatus; }
    v41PackedBarrier(cmd);
    VkBufferCopy region{0, offset, bytes};
    vkCmdCopyBuffer(cmd, stage->buf, d->buf, 1, &region);
    v41PackedBarrier(cmd);
    bool completed = v41EndSubmitWaitChecked(cmd, true);
    // A pending submission owns both operands. Retain native staging until
    // process exit on uncertain completion, just like checked restore. Do not
    // recycle it, destroy it, or claim its reservation was released.
    if (!g_v41SubmissionPendingFailure) destroyBuffer(stage);
    if (!completed) return (int)g_submissionStatus;
    if (!checkedCounterAdd(g_h2dCount, 1) || !checkedCounterAdd(g_h2dBytes, bytes)) g_transferCountersValid = false;
    dpOneShot(g_dp.oneShotH2D);
    return 0;
}

int fvk_v41_packed_copy(void* dst, size_t dst_capacity, size_t dst_offset,
                       const void* src, size_t src_capacity, size_t src_offset,
                       uint64_t generation, size_t bytes) {
    int status = v41PackedState(generation);
    if (status) return status;
    Buffer* d = B(dst);
    const Buffer* s = B((void*)src);
    if (!v41PackedBacking(d, generation, dst_capacity) ||
        !v41PackedBacking(s, generation, src_capacity) ||
        dst_offset % 288 || src_offset % 288 || bytes % 288 ||
        dst_offset > dst_capacity || bytes > dst_capacity - dst_offset ||
        src_offset > src_capacity || bytes > src_capacity - src_offset) return 2;
    if (!bytes) return 0;
    // Same underlying memory is conservative refusal, even through distinct
    // wrapper/buffer identities. No pointer-only alias inference.
    if (d == s || d->buf == s->buf || d->mem == s->mem) return 2;
    VkCommandBuffer cmd = v41BeginCmdChecked();
    if (cmd == VK_NULL_HANDLE) return (int)g_submissionStatus;
    v41PackedBarrier(cmd);
    VkBufferCopy region{src_offset, dst_offset, bytes};
    vkCmdCopyBuffer(cmd, s->buf, d->buf, 1, &region);
    v41PackedBarrier(cmd);
    if (!v41EndSubmitWaitChecked(cmd, true)) return (int)g_submissionStatus;
    if (!checkedCounterAdd(g_d2dCount, 1) || !checkedCounterAdd(g_d2dBytes, bytes)) g_transferCountersValid = false;
    dpOneShot(g_dp.oneShotD2D);
    return 0;
}
