package compute

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// fak-test:runtime medium est=3s lane=default
func TestVulkanBatchHazardWindowIsolation(t *testing.T) {
	// Execute production native window/retirement code with Vulkan calls captured.
	// This witnesses host ownership and recorded dependencies, not device execution.
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Skip("native window witness requires C++17; not verified")
	}
	b, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	shim := string(b)
	section := func(from, to string) string {
		t.Helper()
		i := strings.Index(shim, from)
		if i < 0 {
			t.Fatalf("missing native window seam %q", from)
		}
		j := strings.Index(shim[i:], to)
		if j < 0 {
			t.Fatalf("missing native window end seam %q", to)
		}
		return shim[i : i+j]
	}
	var source strings.Builder
	source.WriteString(vulkanBatchCycleCapturePrelude)
	// Include real idle-command reuse as well as the batch declarations.
	source.WriteString(section("VkCommandBuffer               g_batchCmd", "// A full compute->compute barrier"))
	source.WriteString(section("void recordComputeBarrier(", "// ---- GPU timestamp stage profile"))
	source.WriteString(section("void batchBegin() {", "// A copy into a buffer an earlier recorded op"))
	source.WriteString(section("void fvk_batch_hazards_arm(", "int fvk_submission_status("))
	source.WriteString(vulkanBatchCycleCaptureMain)
	dir := t.TempDir()
	cpp, binary := filepath.Join(dir, "window.cpp"), filepath.Join(dir, "window")
	if err := os.WriteFile(cpp, []byte(source.String()), 0600); err != nil {
		t.Fatal(err)
	}
	if out, err := exec.Command(compiler, "-std=c++17", "-O0", cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile production window capture: %v\n%s", err, out)
	}
	// Queue submission and fence-wait errors use production VKCHECK/abort, not
	// recoverable returns. Do not model those as a fabricated cleanup path here.
	for _, mode := range []string{"exact", "coarse", "disabled", "timestamp", "step", "fence-failure",
		"reset-failure", "begin-failure", "begin-device-loss"} {
		t.Run(mode, func(t *testing.T) {
			cmd := exec.Command(binary, mode)
			for _, e := range os.Environ() {
				if !strings.HasPrefix(e, "FAK_VULKAN_PIPELINE_FLUSH=") {
					cmd.Env = append(cmd.Env, e)
				}
			}
			pipeline := "1"
			if mode == "disabled" {
				pipeline = "0"
			}
			cmd.Env = append(cmd.Env, "FAK_VULKAN_PIPELINE_FLUSH="+pipeline)
			if out, err := cmd.CombinedOutput(); err != nil {
				t.Fatalf("production window capture: %v\n%s", err, out)
			}
		})
	}
}

const vulkanBatchCycleCapturePrelude = `
#include <unordered_map>
#include <vector>
#include <limits>
#include <cstdint>
#include <cassert>
#include <atomic>
#include <chrono>
#include <utility>
#include <cstdlib>
#include <cstdio>
#include <cstring>
using VkCommandBuffer = int;
using VkFence = int;
using VkDevice = int;
using VkCommandPool = int;
using VkResult = int;
enum { VK_NULL_HANDLE = 0, VK_SUCCESS = 0, VK_ERROR_UNKNOWN = -1,
VK_ERROR_DEVICE_LOST = -4, VK_TRUE = 1,
VK_STRUCTURE_TYPE_MEMORY_BARRIER = 1, VK_ACCESS_SHADER_WRITE_BIT = 2,
VK_ACCESS_TRANSFER_WRITE_BIT = 4, VK_ACCESS_SHADER_READ_BIT = 8,
VK_ACCESS_TRANSFER_READ_BIT = 16, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT = 32,
VK_PIPELINE_STAGE_TRANSFER_BIT = 64, VK_STRUCTURE_TYPE_FENCE_CREATE_INFO = 128,
VK_STRUCTURE_TYPE_SUBMIT_INFO = 256, VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO = 512,
VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT = 1024 };
enum { kSyncSitePipelineDrain = 1, kSyncSiteBatchFlush = 2 };
struct VkMemoryBarrier { int type; int srcAccessMask = 0; int dstAccessMask = 0; };
struct VkFenceCreateInfo { int type; };
struct VkCommandBufferBeginInfo { int type; int flags = 0; };
struct VkSubmitInfo { int type; int commandBufferCount = 0; const int* pCommandBuffers = nullptr; };
struct DescriptorSetRecord { int set = 0; int layout = 0; };
struct Buffer { int id; int descriptor; };
struct Event { char kind; int object; int other; };
std::vector<Event> events;
std::unordered_map<int, bool> signaled;
std::unordered_map<int, int> submitted;
std::unordered_map<int, std::vector<DescriptorSetRecord>> g_descSetPool;
int g_dev = 1, g_cmdpool = 2, g_queue = 3, g_submitFence = 4;
int g_submissionStatus = VK_SUCCESS, nextCommand = 10, nextFence = 100;
int resetResult = VK_SUCCESS, beginResult = VK_SUCCESS;
bool g_v41SubmissionPendingFailure = false, g_batching = false, failFence = false;
bool g_tsOn = false, g_dp_on = true;
int g_tsBarriersElided = 0;
uint32_t batchStage = 0;
std::atomic<uint64_t> g_syncPipeDrains{0}, g_syncPipeSubmits{0}, g_d2dCount{0}, g_d2dBytes{0};
bool g_transferCountersValid = true;
struct { int barriers = 0, barriersElided = 0; } g_dp;
void dp_inc(int& n) { ++n; }
void dpBatchSubmit() { events.push_back({'p', 0, 0}); }
struct SyncScope { SyncScope(int, bool = false) {} };
uint64_t submitWaitNow() { return 1; }
void syncSubmitted(uint64_t) { events.push_back({'s', 0, 0}); }
uint32_t syncTakeBatchStage() { auto s = batchStage; batchStage = UINT32_MAX; return s; }
uint32_t syncTakeStage() { return syncTakeBatchStage(); }
void syncSignaled(int site, uint32_t stage, uint64_t, bool = true) {
    events.push_back({'h', site, (int)stage});
}
bool checkedCounterAdd(std::atomic<uint64_t>& n, uint64_t d) { n.fetch_add(d); return true; }
void tsBatchBegin(int) {}
void tsBatchLost() {}
void tsBatchCollect() {}
int vkEndCommandBuffer(int cmd) { events.push_back({'e', cmd, 0}); return VK_SUCCESS; }
int vkCreateFence(int, const VkFenceCreateInfo*, void*, int* fence) {
    if (failFence) return VK_ERROR_UNKNOWN;
    *fence = nextFence++; return VK_SUCCESS;
}
int vkResetFences(int, int, const int* fence) {
    assert(!submitted.count(*fence) || signaled[*fence]);
    signaled[*fence] = false; return VK_SUCCESS;
}
int vkQueueSubmit(int, int, const VkSubmitInfo* si, int fence) {
    assert(si->commandBufferCount == 1);
    submitted[fence] = *si->pCommandBuffers;
    events.push_back({'q', *si->pCommandBuffers, fence}); return VK_SUCCESS;
}
int vkWaitForFences(int, int, const int* fence, int, uint64_t) {
    assert(submitted.count(*fence)); signaled[*fence] = true;
    events.push_back({'w', *fence, submitted[*fence]}); return VK_SUCCESS;
}
void vkFreeCommandBuffers(int, int, int, const int* cmd) { events.push_back({'c', *cmd, 0}); }
int vkResetCommandBuffer(int cmd, int) {
    bool complete = false;
    for (const auto& s : submitted) if (s.second == cmd && signaled[s.first]) complete = true;
    assert(complete); events.push_back({'r', cmd, 0}); return resetResult;
}
int vkBeginCommandBuffer(int cmd, const VkCommandBufferBeginInfo*) {
    events.push_back({'n', cmd, 0}); return beginResult;
}
void vkCmdPipelineBarrier(int cmd, int src, int dst, int, int count,
                         const VkMemoryBarrier* b, int, void*, int, void*) {
    assert(src == (VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT));
    assert(dst == src && count == 1);
    assert(b->srcAccessMask == (VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT));
    assert(b->dstAccessMask == (VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_READ_BIT));
    events.push_back({'b', cmd, 0});
}
#define VKCHECK(call) do { assert((call) == VK_SUCCESS); } while (0)
int allocBeginCmd() { return nextCommand++; }
int v41BeginCmdChecked() { return allocBeginCmd(); }
void endSubmitWait(int);
bool v41EndSubmitWaitChecked(int, bool);
void fvk_free(void*);
`

const vulkanBatchCycleCaptureMain = `
void endSubmitWait(int cmd) {
    VKCHECK(vkEndCommandBuffer(cmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cmd;
    VKCHECK(vkResetFences(g_dev, 1, &g_submitFence));
    VKCHECK(vkQueueSubmit(g_queue, 1, &si, g_submitFence));
    VKCHECK(vkWaitForFences(g_dev, 1, &g_submitFence, VK_TRUE, UINT64_MAX));
    if (!retainCompletedBatchCommand(cmd)) vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
}
bool v41EndSubmitWaitChecked(int cmd, bool work) {
    if (work) endSubmitWait(cmd);
    else vkFreeCommandBuffers(g_dev, g_cmdpool, 1, &cmd);
    return true;
}
void fvk_free(void* ptr) {
    auto* b = (Buffer*)ptr;
    assert(!g_batching);
    bool descriptorReturned = false;
    for (const auto& d : g_descSetPool[1]) if (d.set == b->descriptor) descriptorReturned = true;
    assert(descriptorReturned);
    events.push_back({'f', b->id, 0});
}
int findEvent(char kind, int object, size_t start = 0) {
    for (size_t i = start; i < events.size(); ++i)
        if (events[i].kind == kind && events[i].object == object) return (int)i;
    return -1;
}
void recordPair(bool exact, int base, Buffer* freed) {
    fvk_batch_hazards_arm(exact);
    for (int i = 0; i < 2; ++i) {
        fvk_batch_hazards_set(g_batchOps, i == 0);
        BatchHazardAttempt attempt{g_batchOps};
        if (g_batchOps > 0) recordDispatchBarrier(g_batchCmd, g_batchOps);
        g_batchSets.push_back({base + i, 1}); ++g_batchOps;
    }
    g_batchFreed.push_back(freed);
}
int main(int argc, char** argv) {
    assert(argc == 2);
    const bool exact = std::strcmp(argv[1], "coarse") != 0;
    const bool fallback = std::strcmp(argv[1], "disabled") == 0 ||
        std::strcmp(argv[1], "timestamp") == 0 || std::strcmp(argv[1], "step") == 0 ||
        std::strcmp(argv[1], "fence-failure") == 0;
    g_tsOn = std::strcmp(argv[1], "timestamp") == 0;
    g_stepBatch = std::strcmp(argv[1], "step") == 0;
    failFence = std::strcmp(argv[1], "fence-failure") == 0;
    Buffer a{1, 11}, b{2, 21};
    batchBegin(); const int cmdA = g_batchCmd; const uint64_t epochA = g_batchHazardEpoch;
    if (!fallback) { batchCycle(); assert(g_batchCmd == cmdA && events.empty()); }
    recordPair(exact, 10, &a); batchStage = 7;
    const uint64_t elisions = g_batchHazardElisions;
    batchCycle();
    if (fallback) {
        assert(!g_inflight.active && g_batching && g_pipeFallbacks == 1);
        assert(findEvent('f', a.id) > findEvent('w', g_submitFence));
        batchFlush(); return 0;
    }
    assert(g_inflight.active && g_inflight.cmd == cmdA);
    const int fenceA = g_inflight.fence, cmdB = g_batchCmd;
    assert(!signaled[fenceA] && findEvent('w', fenceA) == -1 && findEvent('f', a.id) == -1);
    assert(g_inflight.hazards.epoch == epochA && g_inflight.hazards.pending.empty());
    assert(g_inflight.hazards.verdicts.size() == (exact ? 2 : 0));
    assert(g_batchHazardWindow.verdicts.empty() && !g_batchHazardsArmed);
    assert(findEvent('b', cmdB) >= 0); // unconditional dependency before any successor work
    assert(g_batchHazardElisions == elisions);
    const uint64_t epochB = g_batchHazardEpoch;
    assert(epochB != epochA);
    fvk_batch_hazards_reset();
    assert(g_inflight.hazards.verdicts.size() == (exact ? 2 : 0));
    recordPair(exact, 20, &b); batchStage = 9;
    const size_t beforeB = events.size();
    const bool resetFailure = std::strcmp(argv[1], "reset-failure") == 0;
    const bool beginFailure = std::strcmp(argv[1], "begin-failure") == 0;
    const bool deviceLoss = std::strcmp(argv[1], "begin-device-loss") == 0;
    if (resetFailure) resetResult = VK_ERROR_UNKNOWN;
    if (beginFailure || deviceLoss) beginResult = deviceLoss ? VK_ERROR_DEVICE_LOST : VK_ERROR_UNKNOWN;
    batchCycle();
    const int fenceB = g_inflight.fence;
    assert(g_inflight.cmd == cmdB && fenceB != fenceA);
    assert(findEvent('q', cmdB, beforeB) < findEvent('w', fenceA, beforeB));
    assert(findEvent('f', a.id) > findEvent('w', fenceA, beforeB));
    assert(findEvent('f', b.id) == -1 && !signaled[fenceB]);
    if (resetFailure || beginFailure || deviceLoss) {
        // Execute the extracted production reset/begin failure branch after B's
        // successful submit and A's fence, not a synthetic window transition.
        assert(!g_batching && g_batchCmd == VK_NULL_HANDLE);
        assert(g_batchHazardEpoch == epochB && g_batchHazardWindow.epoch == 0);
        assert(g_batchHazardWindow.verdicts.empty() && g_batchHazardSync.empty());
        assert(!g_batchHazardsArmed && g_inflight.hazards.epoch == epochB);
        assert(g_inflight.hazards.verdicts.size() == 2);
        assert(g_inflight.sets.size() == 2 && g_inflight.freed.size() == 1);
        assert(g_batchSets.empty() && g_batchFreed.empty());
        assert(findEvent('b', cmdA, beforeB) == -1);
        assert(findEvent('r', cmdA, beforeB) > findEvent('w', fenceA, beforeB));
        assert((findEvent('n', cmdA, beforeB) == -1) == resetFailure);
        assert(g_submissionStatus == (deviceLoss ? VK_ERROR_DEVICE_LOST : VK_ERROR_UNKNOWN));
        assert(g_v41SubmissionPendingFailure == deviceLoss);
        assert((g_batchIdleCmd == cmdA) == deviceLoss);
        assert((findEvent('c', cmdA, beforeB) == -1) == deviceLoss);
        // Sticky failure prevents a retry or a fresh permission from being armed.
        const size_t stopped = events.size();
        batchBegin(); batchCycle(); fvk_batch_hazards_arm(1); fvk_batch_hazards_set(0, 0);
        assert(events.size() == stopped && g_batchHazardEpoch == epochB);
        assert(!g_batchHazardsArmed && g_batchHazardSync.empty());
        fvk_retire_request();
        if (deviceLoss) {
            assert(events.size() == stopped && g_inflight.active);
            assert(g_batchIdleCmd == cmdA && findEvent('f', b.id) == -1);
        } else {
            assert(!g_inflight.active);
            assert(findEvent('f', b.id) > findEvent('w', fenceB));
        }
        return 0;
    }
    assert(g_batchCmd == cmdA && findEvent('r', cmdA) > findEvent('w', fenceA));
    assert(g_inflight.hazards.epoch == epochB && g_batchHazardWindow.verdicts.empty());
    assert(findEvent('h', kSyncSitePipelineDrain) >= 0);
    assert(events[findEvent('h', kSyncSitePipelineDrain)].other == 7);
    assert(g_syncPipeSubmits == 2 && g_syncPipeDrains == 1);
    // Unknown successor work cannot inherit the preceding window's ordinal permission.
    const auto barriers = g_dp.barriers;
    recordDispatchBarrier(g_batchCmd, 1); assert(g_dp.barriers == barriers + 1);
    // Abandoning a dispatch consumes its pending permission even without recording.
    g_batchOps = 1; fvk_batch_hazards_arm(1); fvk_batch_hazards_set(1, 0);
    { BatchHazardAttempt failed{1}; }
    assert(nextHazardSync(1));
    // A stale attempt cannot erase an identically numbered successor permission.
    { BatchHazardAttempt stale{1, epochA}; fvk_batch_hazards_set(1, 0); }
    assert(!nextHazardSync(1));
    fvk_batch_hazards_reset(); assert(nextHazardSync(1));
    for (int i = 0; i < 257; ++i) { g_batchOps = i; fvk_batch_hazards_set(i, 0); }
    assert(g_batchHazardWindow.exhausted && g_batchHazardWindow.verdicts.size() == 256);
    assert(nextHazardSync(256));
    // Retirement is the real request cancellation/finish surface, not a new scheduler.
    batchStage = 11; fvk_retire_request();
    assert(!g_batching && !g_inflight.active && g_batchHazardWindow.verdicts.empty());
    assert(findEvent('f', b.id) > findEvent('w', fenceB));
    assert(g_syncPipeDrains == 2);
    assert(g_batchIdleCmd != VK_NULL_HANDLE);
    return 0;
}
`
