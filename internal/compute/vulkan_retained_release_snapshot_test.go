//go:build linux

package compute

import (
	"context"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// The fixture extracts production metadata/free/pool functions verbatim. Driver
// handles are synthetic; this proves host lifecycle observation, not Vulkan
// completion, physical reclamation, full-shim compilation or device behavior.
// fak-test:runtime medium est=2s lane=optin
// fak-test:justify Compiles production native retained-owner observation and exercises bounded transitions.
func TestVulkanRetainedReleaseSnapshot(t *testing.T) {
	if os.Getenv("FAK_TEST_VULKAN_RETAINED_SNAPSHOT") != "1" {
		t.Skip("set FAK_TEST_VULKAN_RETAINED_SNAPSHOT=1 for the C++ lifecycle witness")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal(err)
	}
	data, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(data)
	extract := func(start string, semicolon bool) string {
		i := strings.Index(source, start)
		if i < 0 {
			t.Fatalf("production definition missing: %s", start)
		}
		open := strings.Index(source[i:], "{") + i
		depth := 0
		for end := open; end < len(source); end++ {
			if source[end] == '{' {
				depth++
			}
			if source[end] == '}' {
				depth--
				if depth == 0 {
					if semicolon {
						end++
					}
					return source[i : end+1]
				}
			}
		}
		t.Fatalf("unterminated production definition: %s", start)
		return ""
	}
	// Pin observation generation to real init/allocation seams rather than only
	// testing a fixture-owned assignment. The init-failure readiness is independent
	// of g_ready, which existing re-init paths do not clear.
	for _, seam := range []string{
		"if (g_v41SubmissionPendingFailure) return 9;\n    beginMemoryDeviceGeneration();",
		"b->allocationID = nextMemoryAllocationID();\n    b->deviceGeneration = g_memoryDeviceGeneration;",
		"block.allocationID = nextMemoryAllocationID();\n    block.deviceGeneration = g_memoryDeviceGeneration;",
		"g_ready = true;\n    g_memoryDeviceGenerationReady = g_memoryDeviceGeneration != 0;",
	} {
		if !strings.Contains(source, seam) {
			t.Fatalf("missing generation lifecycle seam: %s", seam)
		}
	}
	fixture := retainedSnapshotPreamble + extract("struct Buffer {", true) + "\n" + extract("struct WeightArenaBlock {", true) + retainedSnapshotGlobals +
		extract("void beginMemoryDeviceGeneration() {", false) + "\n" +
		extract("int fvk_buffer_reservation(const void* d,", false) + "\n" +
		extract("int fvk_retained_release_snapshot(fvk_retained_release_record* rows,", false) + "\n" +
		extract("void fvk_free(void* d) {", false) + "\n" +
		extract("void* fvk_malloc(size_t bytes) {", false) + retainedSnapshotMain
	dir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	cpp := filepath.Join(tmp, "snapshot.cpp")
	binary := filepath.Join(tmp, "snapshot")
	if err := os.WriteFile(cpp, []byte(fixture), 0600); err != nil {
		t.Fatal(err)
	}
	ctx, cancel := context.WithTimeout(t.Context(), 20*time.Second)
	defer cancel()
	if out, err := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0", "-Wall", "-Wextra", "-Werror", "-I", dir, cpp, "-o", binary).CombinedOutput(); err != nil {
		t.Fatalf("compile extracted production lifecycle: %v\n%s", err, out)
	}
	if out, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("production lifecycle witness: %v\n%s", err, out)
	}
}

const retainedSnapshotPreamble = `
#include "vulkan_backend.h"
#include <cassert>
#include <cstdint>
#include <limits>
#include <unordered_map>
#include <vector>
using VkBuffer = uintptr_t;
using VkDeviceMemory = uintptr_t;
using VkDeviceSize = uint64_t;
using VkMemoryPropertyFlags = uint32_t;
#define VK_NULL_HANDLE 0
#define VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT 1
#define VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT 2
#define STORAGE_USAGE 0
`
const retainedSnapshotGlobals = `
std::vector<WeightArenaBlock> g_weightArena;
std::vector<Buffer*> g_batchFreed;
std::unordered_map<size_t, std::vector<Buffer*>> g_pool, g_hvOffloadPool;
size_t g_poolCount = 0;
const size_t POOL_BUCKET_CAP = 64;
bool g_ready = true, g_memoryDeviceGenerationReady = true;
bool g_batching = false, g_v41SubmissionPendingFailure = false;
uintptr_t g_dev = 1;
uint64_t g_memoryDeviceGeneration = 1;
int destroyed = 0;
Buffer* B(void* p) { return static_cast<Buffer*>(p); }
int fvk_buffer_backing(const void* p, fvk_buffer_backing_info* out) {
 *out = {};
 const Buffer* b = static_cast<const Buffer*>(p);
 return b && b->buf && b->mem && b->memoryTypeIndex == 0;
}
void clearDescriptorBindingCache() {}
void destroyBuffer(Buffer* b) { ++destroyed; delete b; }
Buffer* allocBuffer(size_t, uint32_t, int) { assert(false); return nullptr; }
`
const retainedSnapshotMain = `
Buffer* makeBuffer(uint64_t id, uint64_t bytes = 16) {
 auto b = new Buffer{};
 b->buf = id; b->mem = id; b->bytes = bytes;
 b->allocationBytes = 128; b->allocationID = id;
 b->deviceGeneration = g_memoryDeviceGeneration; b->memoryTypeIndex = 0;
 return b;
}
int main() {
 fvk_retained_release_record rows[4]{};
 fvk_retained_release_summary s{};
 auto query = [&] { return fvk_retained_release_snapshot(rows, 4, &s); };
 assert(query() == FVK_RETAINED_OK && s.returned_records == 0 && !s.truncated);
 auto b = makeBuffer(7);
 g_batching = true; fvk_free(b);
 assert(query() == FVK_RETAINED_OK && s.returned_records == 1);
 assert(rows[0].release_state == FVK_RELEASE_REQUESTED && rows[0].backing_state == FVK_BACKING_DEFERRED);
 assert(rows[0].logical_lease_generation == 0 && rows[0].device_generation == 1);
 assert(s.returned_unique_reservation_bytes == 128);
 // Model the already-established completion boundary, not a GPU completion proof.
 g_batching = false; g_batchFreed.clear(); fvk_free(b);
 assert(query() == FVK_RETAINED_OK && rows[0].backing_state == FVK_BACKING_POOL_RETAINED);
 auto reused = static_cast<Buffer*>(fvk_malloc(16)); assert(reused == b);
 assert(query() == FVK_RETAINED_OK && s.returned_records == 0 && destroyed == 0);
 fvk_free(reused); assert(query() == FVK_RETAINED_OK && rows[0].reservation.allocation_id == 7);
 // Physical allocation identity survives reuse; logical lease generation is unknown.
 assert(rows[0].logical_lease_generation == 0);
 g_v41SubmissionPendingFailure = true;
 auto uncertain = makeBuffer(8); fvk_free(uncertain);
 assert(query() == FVK_RETAINED_OK && s.returned_records == 2);
 for (uint32_t i=0; i<s.returned_records; ++i) assert(rows[i].backing_state == FVK_BACKING_QUARANTINED);
 assert(fvk_malloc(16) == nullptr && destroyed == 0);
 g_v41SubmissionPendingFailure = false; g_pool.clear(); g_batchFreed.clear();
 delete b; delete uncertain;
 // Shared arena backing is charged once despite distinct logical bindings.
 WeightArenaBlock arena{}; arena.mem = 99; arena.capacity = 128; arena.used = 64;
 arena.allocationID = 99; arena.liveBuffers = 2; arena.memoryTypeIndex = 0; arena.deviceGeneration = 1;
 g_weightArena.push_back(arena);
 auto a = makeBuffer(10), c = makeBuffer(11);
 for (auto x : {a,c}) { x->weightArenaBound = true; x->weightArenaBlock = 0; x->mem = 99; }
 c->memoryOffset = 32; g_batchFreed = {a,c,a};
 assert(query() == FVK_RETAINED_OK && s.returned_records == 2 && s.returned_unique_allocations == 1);
 assert(s.returned_unique_reservation_bytes == 128 && !s.truncated);
 assert(fvk_retained_release_snapshot(rows,1,&s) == FVK_RETAINED_OK && s.truncated && s.returned_records == 1);
 assert(s.returned_unique_reservation_bytes == 128);
 assert(fvk_retained_release_snapshot(nullptr,0,&s) == FVK_RETAINED_OK && s.truncated && s.returned_records == 0);
 // Inspection remains bounded even if all queue entries are duplicates.
 g_batchFreed.assign(FVK_RETAINED_RELEASE_LIMIT + 1, a);
 assert(query() == FVK_RETAINED_OK && s.truncated && s.returned_records == 1);
 g_batchFreed = {a}; g_pool[16] = {a};
 assert(query() == FVK_RETAINED_INVALID && s.returned_records == 0 && rows[0].reservation.allocation_id == 0);
 g_pool.clear(); g_weightArena[0].deviceGeneration = 0;
 assert(query() == FVK_RETAINED_STALE_GENERATION && s.returned_records == 0);
 g_weightArena[0].deviceGeneration = 1;
 beginMemoryDeviceGeneration();
 assert(g_memoryDeviceGeneration == 2 && query() == FVK_RETAINED_UNAVAILABLE);
 g_memoryDeviceGenerationReady = true;
 assert(query() == FVK_RETAINED_STALE_GENERATION && s.returned_records == 0);
 g_batchFreed.clear(); delete a; delete c; g_weightArena.clear();
 auto offload = makeBuffer(33); offload->offloadPage = true; fvk_free(offload);
 assert(query() == FVK_RETAINED_OK && rows[0].container == FVK_RETAINED_OFFLOAD_POOL);
 g_hvOffloadPool.clear(); delete offload;
 auto huge = makeBuffer(44), small = makeBuffer(45);
 huge->allocationBytes = std::numeric_limits<uint64_t>::max() - 8;
 g_batchFreed = {huge,small};
 assert(query() == FVK_RETAINED_INVALID && s.returned_unique_reservation_bytes == 0);
 assert(rows[0].reservation.allocation_id == 0);
 g_batchFreed.clear(); delete huge; delete small;
 for (size_t i = 0; i <= FVK_RETAINED_RELEASE_LIMIT; ++i) g_pool[i] = {};
 assert(query() == FVK_RETAINED_OK && s.truncated && s.returned_records == 0);
 g_pool.clear();
 for (size_t i = 0; i <= FVK_RETAINED_RELEASE_LIMIT; ++i) g_hvOffloadPool[i] = {};
 assert(query() == FVK_RETAINED_OK && s.truncated && s.returned_records == 0);
 g_hvOffloadPool.clear();
 g_ready = false; assert(query() == FVK_RETAINED_UNAVAILABLE); g_ready = true;
 assert(fvk_retained_release_snapshot(rows,FVK_RETAINED_RELEASE_LIMIT+1,&s) == FVK_RETAINED_INVALID);
 g_memoryDeviceGeneration = std::numeric_limits<uint64_t>::max(); beginMemoryDeviceGeneration();
 assert(g_memoryDeviceGeneration == 0 && query() == FVK_RETAINED_UNAVAILABLE);
 beginMemoryDeviceGeneration(); assert(g_memoryDeviceGeneration == 0);
}
`
