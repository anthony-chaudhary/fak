//go:build vulkan_argmax_batch_witness

package compute

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

// This is a host-only command/ownership witness, not shader numerical or GPU
// performance evidence. Build the file set directly to avoid native linkage.
// fak-test:runtime medium est=4s lane=optin
// fak-test:justify Compiles the complete pinned shim against official PFN mocks without a Vulkan loader.
func TestVulkanArgmaxBatchFlush(t *testing.T) {
	source := os.Getenv("FAK_ARGMAX_BATCH_SOURCE")
	include := os.Getenv("FAK_ARGMAX_BATCH_VULKAN_INCLUDE")
	if !filepath.IsAbs(source) || !filepath.IsAbs(include) {
		t.Fatal("explicit absolute native source and official Vulkan include paths required")
	}
	tmp := t.TempDir()
	var all strings.Builder
	for _, file := range []struct{ name, env string }{
		{"vulkan_shim.cpp", "FAK_ARGMAX_BATCH_SHIM_SHA256"},
		{"vulkan_backend.h", "FAK_ARGMAX_BATCH_HEADER_SHA256"},
		{"v41_index_scale_bf16_spirv.h", "FAK_ARGMAX_BATCH_INDEX_SCALE_SHA256"},
		{"v41_routed_swiglu_chunk_spirv.h", "FAK_ARGMAX_BATCH_ROUTED_CHUNK_SHA256"},
		{"v41_routed_down_chunk_spirv.h", "FAK_ARGMAX_BATCH_ROUTED_DOWN_CHUNK_SHA256"},
	} {
		raw, err := os.ReadFile(filepath.Join(source, file.name))
		if err != nil {
			t.Fatal(err)
		}
		digest := sha256.Sum256(raw)
		if len(os.Getenv(file.env)) != 64 || hex.EncodeToString(digest[:]) != os.Getenv(file.env) {
			t.Fatalf("unpinned native source: %s", file.name)
		}
		if err := os.WriteFile(filepath.Join(tmp, file.name), raw, 0600); err != nil {
			t.Fatal(err)
		}
		all.Write(raw)
		all.WriteByte('\n')
	}
	allowed := map[string]bool{}
	for _, name := range strings.Fields("vkAllocateDescriptorSets vkUpdateDescriptorSets vkAllocateCommandBuffers vkResetCommandBuffer vkBeginCommandBuffer vkEndCommandBuffer vkFreeCommandBuffers vkResetFences vkQueueSubmit vkWaitForFences vkCmdBindPipeline vkCmdBindDescriptorSets vkCmdPushConstants vkCmdPipelineBarrier vkCmdDispatch vkCmdCopyBuffer") {
		allowed[name] = true
	}
	names := map[string]bool{}
	for _, name := range regexp.MustCompile(`\bvk[A-Z][A-Za-z0-9_]*\b`).FindAllString(all.String(), -1) {
		names[name] = true
	}
	for name := range allowed {
		if !names[name] {
			t.Fatalf("missing allowed native API: %s", name)
		}
	}
	ordered := make([]string, 0, len(names))
	for name := range names {
		ordered = append(ordered, name)
	}
	sort.Strings(ordered)
	var fixture strings.Builder
	fixture.WriteString(vulkanArgmaxBatchPrefix)
	for _, name := range ordered {
		if allowed[name] {
			fmt.Fprintf(&fixture, "static_assert(std::is_same_v<decltype(&checked_%s),PFN_%s>);\n#define %s checked_%s\n", name, name, name, name)
		} else {
			fmt.Fprintf(&fixture, "#define %s CheckedAbort<PFN_%s>::call\n", name, name)
		}
	}
	fixture.WriteString("\n#include \"vulkan_shim.cpp\"\n")
	fixture.WriteString(vulkanArgmaxBatchCases)
	file := filepath.Join(tmp, "argmax-batch.cpp")
	if err := os.WriteFile(file, []byte(fixture.String()), 0600); err != nil {
		t.Fatal(err)
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	binary := filepath.Join(tmp, "argmax-batch")
	ctx, cancel := context.WithTimeout(t.Context(), 120*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-g0", "-I", include, file, "-o", binary)
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	sysproc.ConfigureBackground(cmd)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile actual shim: %v\n%s", err, out)
	}
	run, stop := context.WithTimeout(t.Context(), 10*time.Second)
	defer stop()
	cmd = exec.CommandContext(run, binary)
	cmd.Env = append(os.Environ(), "FAK_VULKAN_DISPATCH_PROFILE=0", "FAK_VULKAN_TIMESTAMP_PROFILE=0")
	sysproc.ConfigureBackground(cmd)
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("actual shim orchestration: %v\n%s", err, out)
	}
	t.Logf("%s", out)
}

const vulkanArgmaxBatchPrefix = `
#include <vulkan/vulkan.h>
#include <array>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

namespace checked_fixture {
[[noreturn]] void die(const char* why) { std::fprintf(stderr, "argmax batch: %s\n", why); std::exit(2); }
void require(bool ok, const char* why) { if (!ok) die(why); }
template<class T> T handle(uintptr_t id) { return (T)id; }
struct Op { int kind; VkBuffer source{}, dest{}; uint32_t count{}, site{}; };
struct Command {
    bool begun = false, ended = false, filled = false, dependency = false, completed = false;
    VkPipeline pipeline{};
    std::vector<VkBuffer> bindings;
    std::vector<uint32_t> push;
    std::vector<Op> ops;
};
std::map<VkCommandBuffer, Command> commands;
std::map<VkCommandBuffer, std::pair<VkDevice,VkCommandPool>> commandOwners;
std::map<VkDescriptorSet, std::vector<VkBuffer>> descriptors;
std::map<VkBuffer, std::array<uint32_t, 4>> memory;
std::map<std::string, int> calls;
std::string failure;
int failureOrdinal = 1;
VkResult failureResult = VK_ERROR_OUT_OF_HOST_MEMORY;
uintptr_t nextHandle = 100;
VkCommandBuffer submitted{};
VkBuffer inputHandle{}, outputHandle{}, statusHandle{}, stageHandle{}, weightHandle{};
VkPipeline checkedPipeline{};
uint32_t expectedCount = 4, expectedSite = 1, expectedGroups = 1;
bool checkedDispatch = false;
VkPipeline producerPipeline{};
int argmaxes = 0;
int fills = 0, dispatches = 0, copies = 0, barriers = 0;
std::array<uint32_t, 4> stage{};
VkResult call(const char* name) {
    const int n = ++calls[name];
    return failure == name && n == failureOrdinal ? failureResult : VK_SUCCESS;
}
Command& command(VkCommandBuffer id) {
    auto it = commands.find(id);
    require(it != commands.end(), "unknown command buffer");
    return it->second;
}
void execute(VkCommandBuffer id) {
    for (const auto& op : command(id).ops) {
        if (op.kind == 0) {
            const float logits[4] = {1.f, 9.f, 9.f, -3.f};
            std::memcpy(memory[op.dest].data(), logits, sizeof(logits));
        } else if (op.kind == 1) {
            // Mock only the payload. Production command/descriptor/barrier and
            // ownership code is included unchanged below; no shader is executed.
            float logits[4]; std::memcpy(logits, memory.at(op.source).data(), sizeof(logits));
            int winner = 0;
            for (int i = 1; i < 4; ++i) if (logits[i] > logits[winner]) winner = i;
            memory[op.dest][0] = uint32_t(winner);
        } else if (op.kind == 2) stage = memory.at(op.source);
        else die("unknown recorded operation");
    }
}
} // namespace checked_fixture

// Every unallowlisted Vulkan function has its official signature and aborts.
// No loader is linked, including for APIs in unused production helper bodies.
template<class> struct CheckedAbort;
template<class R, class... A> struct CheckedAbort<R(VKAPI_PTR*)(A...)> {
    static VKAPI_ATTR R VKAPI_CALL call(A...) { checked_fixture::die("unexpected Vulkan API reached"); }
};

static VKAPI_ATTR VkResult VKAPI_CALL checked_vkAllocateDescriptorSets(VkDevice, const VkDescriptorSetAllocateInfo* info, VkDescriptorSet* out) {
    using namespace checked_fixture;
    require(info && out && info->descriptorSetCount == 1, "descriptor allocation shape");
    const VkResult r = call("vkAllocateDescriptorSets");
    *out = r == VK_SUCCESS ? handle<VkDescriptorSet>(++nextHandle) : VK_NULL_HANDLE;
    if (r == VK_SUCCESS) descriptors[*out] = {};
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkUpdateDescriptorSets(VkDevice, uint32_t n, const VkWriteDescriptorSet* writes, uint32_t copyCount, const VkCopyDescriptorSet*) {
    using namespace checked_fixture;
    call("vkUpdateDescriptorSets");
    require((n == 2 || n == 7) && writes && copyCount == 0, "descriptor update shape");
    for (uint32_t i = 0; i < n; ++i) {
        const auto& w = writes[i];
        require(w.dstBinding == i && w.descriptorCount == 1 && w.descriptorType == VK_DESCRIPTOR_TYPE_STORAGE_BUFFER && w.pBufferInfo, "descriptor role");
        require(w.pBufferInfo->offset == 0 && w.pBufferInfo->range == VK_WHOLE_SIZE, "descriptor range");
        auto& bindings = descriptors.at(w.dstSet);
        if (bindings.size() < n) bindings.resize(n);
        bindings[i] = w.pBufferInfo->buffer;
    }
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkAllocateCommandBuffers(VkDevice device, const VkCommandBufferAllocateInfo* info, VkCommandBuffer* out) {
    using namespace checked_fixture;
    require(info && out && info->commandBufferCount == 1, "command allocation shape");
    const VkResult r = call("vkAllocateCommandBuffers");
    *out = r == VK_SUCCESS ? handle<VkCommandBuffer>(++nextHandle) : VK_NULL_HANDLE;
    if (r == VK_SUCCESS) { commands[*out] = {}; commandOwners[*out] = {device,info->commandPool}; }
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkResetCommandBuffer(VkCommandBuffer id, VkCommandBufferResetFlags flags) {
    using namespace checked_fixture;
    require(flags == 0 && command(id).completed, "reset without successful fence");
    const VkResult r = call("vkResetCommandBuffer");
    if (r == VK_SUCCESS) command(id) = {};
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkBeginCommandBuffer(VkCommandBuffer id, const VkCommandBufferBeginInfo*) {
    using namespace checked_fixture;
    const VkResult r = call("vkBeginCommandBuffer");
    if (r == VK_SUCCESS) command(id).begun = true;
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkEndCommandBuffer(VkCommandBuffer id) {
    using namespace checked_fixture;
    require(command(id).begun, "end without begin");
    const VkResult r = call("vkEndCommandBuffer");
    if (r == VK_SUCCESS) command(id).ended = true;
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkFreeCommandBuffers(VkDevice device, VkCommandPool pool, uint32_t n, const VkCommandBuffer* ids) {
    using namespace checked_fixture;
    call("vkFreeCommandBuffers");
    require(n == 1 && ids && commandOwners.at(ids[0]) == std::make_pair(device,pool) &&
            commands.erase(ids[0]) == 1 && commandOwners.erase(ids[0]) == 1, "command release identity/owner");
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkResetFences(VkDevice, uint32_t n, const VkFence*) {
    using namespace checked_fixture;
    require(n == 1, "fence reset shape");
    return call("vkResetFences");
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkQueueSubmit(VkQueue, uint32_t n, const VkSubmitInfo* infos, VkFence) {
    using namespace checked_fixture;
    require(n == 1 && infos && infos->commandBufferCount == 1, "submit shape");
    submitted = infos->pCommandBuffers[0];
    require(command(submitted).ended, "submit before command end");
    const VkResult r = call("vkQueueSubmit");
    // Even an uncertain submit may have changed memory. A successful-looking
    // envelope must never override the real native failure/pending result.
    execute(submitted);
    return r;
}
static VKAPI_ATTR VkResult VKAPI_CALL checked_vkWaitForFences(VkDevice, uint32_t n, const VkFence*, VkBool32 all, uint64_t timeout) {
    using namespace checked_fixture;
    require(n == 1 && all == VK_TRUE && timeout == UINT64_MAX && submitted != VK_NULL_HANDLE, "wait shape");
    const VkResult r = call("vkWaitForFences");
    if (r == VK_SUCCESS) command(submitted).completed = true;
    return r;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdBindPipeline(VkCommandBuffer id, VkPipelineBindPoint point, VkPipeline pipeline) {
    using namespace checked_fixture;
    call("vkCmdBindPipeline");
    require(point == VK_PIPELINE_BIND_POINT_COMPUTE, "pipeline bind point");
    command(id).pipeline = pipeline;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdBindDescriptorSets(VkCommandBuffer id, VkPipelineBindPoint point, VkPipelineLayout, uint32_t first, uint32_t n, const VkDescriptorSet* sets, uint32_t dynamicCount, const uint32_t*) {
    using namespace checked_fixture;
    call("vkCmdBindDescriptorSets");
    require(point == VK_PIPELINE_BIND_POINT_COMPUTE && first == 0 && n == 1 && sets && dynamicCount == 0, "descriptor bind shape");
    command(id).bindings = descriptors.at(sets[0]);
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdPushConstants(VkCommandBuffer id, VkPipelineLayout, VkShaderStageFlags flags, uint32_t offset, uint32_t size, const void* data) {
    using namespace checked_fixture;
    call("vkCmdPushConstants");
    require(flags == VK_SHADER_STAGE_COMPUTE_BIT && offset == 0 && (size == 4u || size == 20u), "push shape");
    auto& push = command(id).push;
    push.resize(size/4);
    std::memcpy(push.data(), data, size);
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdPipelineBarrier(VkCommandBuffer id, VkPipelineStageFlags srcStage, VkPipelineStageFlags dstStage, VkDependencyFlags flags, uint32_t n, const VkMemoryBarrier* memoryBarriers, uint32_t bufferCount, const VkBufferMemoryBarrier*, uint32_t imageCount, const VkImageMemoryBarrier*) {
    using namespace checked_fixture;
    call("vkCmdPipelineBarrier"); ++barriers;
    const auto stages = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT;
    require(srcStage == stages && dstStage == stages && flags == 0 && n == 1 && memoryBarriers && bufferCount == 0 && imageCount == 0, "conservative barrier stages/shape");
    require(memoryBarriers[0].srcAccessMask == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT) && memoryBarriers[0].dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT), "input/status barrier access");
    command(id).dependency = true;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdDispatch(VkCommandBuffer id, uint32_t x, uint32_t y, uint32_t z) {
    using namespace checked_fixture;
    call("vkCmdDispatch"); ++dispatches;
    auto& c = command(id);
    require(c.begun && x == 1 && y == 1 && z == 1, "dispatch geometry");
    if (c.pipeline == producerPipeline) {
        require(c.push == std::vector<uint32_t>{4,256,1,0,0}, "Q2 producer push");
        require(c.bindings == std::vector<VkBuffer>{weightHandle,inputHandle,inputHandle,outputHandle,weightHandle,inputHandle,outputHandle}, "Q2 producer bindings");
        c.ops.push_back({0, inputHandle, outputHandle, 4, 0});
    } else {
        require(c.pipeline == checkedPipeline && c.push == std::vector<uint32_t>{4} && c.bindings == std::vector<VkBuffer>{outputHandle,statusHandle}, "argmax push/bindings");
        require(c.ops.empty() || c.dependency, "argmax lacks producer dependency");
        c.ops.push_back({1, outputHandle, statusHandle, 4, 0});
        ++argmaxes;
    }
    c.dependency = false;
}
static VKAPI_ATTR void VKAPI_CALL checked_vkCmdCopyBuffer(VkCommandBuffer id, VkBuffer source, VkBuffer dest, uint32_t n, const VkBufferCopy* regions) {
    using namespace checked_fixture;
    call("vkCmdCopyBuffer"); ++copies;
    require(source == statusHandle && dest == stageHandle && n == 1 && regions && regions[0].srcOffset == 0 && regions[0].dstOffset == 0 && regions[0].size == 4, "index readback range");
    command(id).ops.push_back({2, source, dest, 0, 0});
}
`

const vulkanArgmaxBatchCases = `
namespace checked_fixture {
Buffer input{}, output{}, index{}, staging{}, weight{};
void reset(bool checked) {
    commands.clear(); commandOwners.clear(); descriptors.clear(); memory.clear(); calls.clear();
    failure.clear(); failureOrdinal = 1; failureResult = VK_ERROR_OUT_OF_HOST_MEMORY;
    submitted = VK_NULL_HANDLE; dispatches = copies = barriers = argmaxes = 0;
    g_pool.clear(); g_poolCount = 0; g_batchFreed.clear(); g_batchSets.clear(); g_descSetPool.clear();
    g_batching = false; g_batchCmd = VK_NULL_HANDLE; g_batchOps = 0; g_batchHasV41 = false;
    g_stepBatch = checked; g_batchIdleCmd = VK_NULL_HANDLE;
    g_batchIdleDevice = VK_NULL_HANDLE; g_batchIdlePool = VK_NULL_HANDLE;
    g_v41SubmissionPendingFailure = false; g_submissionStatus = VK_SUCCESS;
    g_restore = RestoreTransaction{}; g_debugD2HStagingFailureOnce = false;
    g_d2hCount = 0; g_d2hBytes = 0; g_transferCountersValid = true;
    g_batchHazardsArmed = 0; g_batchHazardSync.clear();
    require(!g_dp_on && !g_tsOn, "profiling must be disabled");
    g_ready = true; g_dev = handle<VkDevice>(1); g_queue = handle<VkQueue>(2);
    g_cmdpool = handle<VkCommandPool>(3); g_submitFence = handle<VkFence>(4); g_descpool = handle<VkDescriptorPool>(5);
    g_maxStorageBufferRange = g_maxMemoryAllocationSize = g_maxBufferBytes = std::numeric_limits<VkDeviceSize>::max();
    input = {}; output = {}; index = {}; staging = {}; weight = {};
    weight.buf = handle<VkBuffer>(14); weight.mem = handle<VkDeviceMemory>(24); weight.bytes = 336; weightHandle = weight.buf;
    Buffer* buffers[] = {&input, &output, &index, &staging};
    for (int i = 0; i < 4; ++i) {
        buffers[i]->buf = handle<VkBuffer>(10+i); buffers[i]->mem = handle<VkDeviceMemory>(20+i);
        buffers[i]->bytes = i == 0 ? 1024 : i == 1 ? 16 : 4;
        buffers[i]->props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        memory[buffers[i]->buf] = {};
    }
    inputHandle = input.buf; outputHandle = output.buf; statusHandle = index.buf; stageHandle = staging.buf;
    stage.fill(0); g_stage = &staging; g_stageMapped = stage.data(); g_stageCap = 4;
    auto& producer = g_kern[K_Q2K_MATMUL];
    producer = {}; producer.pipe = handle<VkPipeline>(30); producer.dsl = handle<VkDescriptorSetLayout>(31); producer.layout = handle<VkPipelineLayout>(32); producer.nbuf = 7; producer.pcsize = 20;
    g_have_q2k_matvec = false;
    producerPipeline = producer.pipe;
    auto& k = g_kern[K_ARGMAX];
    k = {}; k.pipe = handle<VkPipeline>(40); k.dsl = handle<VkDescriptorSetLayout>(41); k.layout = handle<VkPipelineLayout>(42); k.nbuf = 2; k.pcsize = 4;
    checkedPipeline = k.pipe;
    g_pool[4].push_back(&index); g_poolCount = 1;
}
void producer() {
    fvk_q2k_matmul_f32(&weight, &input, &output, 4, 256, 1);
    require(dispatches == 1, "Q2 producer recording refused");
}
void success(bool batch, bool checked, bool armed) {
    reset(checked);
    if (batch) batchBegin();
    producer();
    // An undeclared argmax ordinal must receive the conservative barrier even
    // when the ordinary exact-hazard mechanism is armed.
    if (armed) g_batchHazardsArmed = 1;
    const int result = fvk_argmax_f32(&output, 4);
    require(result == 1 && argmaxes == 1, "wrong first-max or repeated argmax");
    require(calls["vkQueueSubmit"] == (batch ? 2 : 3) && calls["vkWaitForFences"] == (batch ? 2 : 3), "extra compute submit/wait before argmax");
    require(barriers == (batch ? 1 : 0), "producer dependency count");
    require(!g_batching && g_stepBatch == checked && g_submissionStatus == VK_SUCCESS && !g_v41SubmissionPendingFailure, "batch terminal state changed");
    require(g_batchSets.empty() && g_batchFreed.empty() && g_poolCount == 1 && g_pool[4].size() == 1 && g_pool[4][0] == &index, "index lifetime/pool ownership");
    require(copies == 1 && g_d2hCount == 1 && g_d2hBytes == 4, "host index readback count");
}
void failed(const char* api, int ordinal, bool pending) {
    reset(true); batchBegin(); producer();
    failure = api; failureOrdinal = ordinal;
    const int result = fvk_argmax_f32(&output, 4);
    require(result == 0 && g_submissionStatus == failureResult && g_v41SubmissionPendingFailure == pending, "native failure/status changed");
    require(copies == 0 && g_d2hCount == 0, "failed operation published host index");
    if (pending) {
        require(g_batching && g_batchFreed.size() == 1 && g_batchFreed[0] == &index && g_poolCount == 0, "pending index was recycled");
        const auto before = calls;
        require(fvk_malloc(4) == nullptr, "pending index was reusable");
        batchFlush(); require(calls == before, "pending submission retried");
    } else {
        require(!g_batching && g_batchFreed.empty() && g_poolCount == 1 && g_pool[4][0] == &index, "known failure index cleanup");
    }
}
}
int main() {
    using namespace checked_fixture;
    success(false, false, false);
    success(true, false, false);
    success(true, false, true);
    success(true, true, false);
    failed("vkAllocateDescriptorSets", 2, false);
    failed("vkEndCommandBuffer", 1, false);
    failed("vkResetFences", 1, false);
    failed("vkQueueSubmit", 1, true);
    failed("vkWaitForFences", 1, true);
    // The unchanged D2H staging failure does not make a completed index owner
    // pending; the entrypoint's legacy scalar result remains zero.
    reset(false); batchBegin(); producer(); g_debugD2HStagingFailureOnce = true;
    require(fvk_argmax_f32(&output,4) == 0 && !g_batching && g_poolCount == 1 && copies == 0, "readback failure cleanup");
    std::puts("PASS argmax batch: two submits/waits, dependency, index lifetime and checked failures");
}
`
