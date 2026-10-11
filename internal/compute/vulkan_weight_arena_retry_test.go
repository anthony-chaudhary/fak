//go:build linux

package compute

import (
	"context"
	"crypto/sha256"
	"os"
	"os/exec"
	"path/filepath"
	"testing"
	"time"
)

// This opt-in fixture compiles the complete production shim with allocator-call
// injection. It still requires a C++17 compiler, Vulkan headers and the Vulkan
// loader library; it does not establish device allocation or driver behavior.
// The estimate includes compiling the full production translation unit.
// fak-test:runtime medium est=20s lane=optin
// fak-test:justify Exercises the native arena owner, selected backing and release accounting together.
func TestVulkanWeightArenaCompatibleTypeOwner(t *testing.T) {
	if os.Getenv("FAK_TEST_VULKAN_ARENA_OWNER") != "1" {
		t.Skip("set FAK_TEST_VULKAN_ARENA_OWNER=1 with a C++17 compiler and Vulkan development libraries")
	}
	compiler, err := exec.LookPath("c++")
	if err != nil {
		t.Fatal("the requested arena owner fixture requires c++: ", err)
	}
	// The native red/green runner overlays Go tests onto the parent, but not
	// new testdata files. Carry the fixture with the test so both arms compile
	// identical injected calls against their own real production shim.
	sourceDir, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	tmp := t.TempDir()
	source := filepath.Join(tmp, "vulkan_weight_arena_retry.cpp")
	if err := os.WriteFile(source, []byte(vulkanWeightArenaOwnerFixture), 0o600); err != nil {
		t.Fatal(err)
	}
	emitted, err := os.ReadFile(source)
	if err != nil {
		t.Fatal(err)
	}
	t.Logf("arena fixture sha256=%x", sha256.Sum256(emitted))
	binary := filepath.Join(tmp, "vulkan-weight-arena-owner")
	ctx, cancel := context.WithTimeout(t.Context(), 90*time.Second)
	defer cancel()
	cmd := exec.CommandContext(ctx, compiler, "-std=c++17", "-O0",
		// The unchanged fixture includes ../vulkan_shim.cpp. Resolve that
		// against this checkout's existing testdata directory, not the temp
		// source directory or a copied production implementation.
		"-iquote", filepath.Join(sourceDir, "testdata"), source,
		"-o", binary, "-lvulkan")
	if output, err := cmd.CombinedOutput(); err != nil {
		t.Fatalf("compile production-shim fixture: %v\n%s", err, output)
	}
	ctx, cancelRun := context.WithTimeout(t.Context(), 10*time.Second)
	defer cancelRun()
	if output, err := exec.CommandContext(ctx, binary).CombinedOutput(); err != nil {
		t.Fatalf("injected arena owner: %v\n%s", err, output)
	}
}

// Keep these fixture bytes identical in the candidate and overlaid-parent runs.
const vulkanWeightArenaOwnerFixture = `// Inject only the Vulkan calls used by the immutable-weight arena and its pool
// cleanup. Include the full production translation unit so the owner, selector,
// metadata and lifetime paths cannot drift into a duplicate test implementation.
// The remaining shim symbols still link against the Vulkan loader. No physical
// device allocation, command submission or performance claim follows from this
// fixture, even when it is eventually compiled and run.
#include <vulkan/vulkan.h>
#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <vector>

static VKAPI_ATTR VkResult VKAPI_CALL arenaCreateBuffer(VkDevice, const VkBufferCreateInfo*, const VkAllocationCallbacks*, VkBuffer*);
static VKAPI_ATTR void VKAPI_CALL arenaDestroyBuffer(VkDevice, VkBuffer, const VkAllocationCallbacks*);
static VKAPI_ATTR void VKAPI_CALL arenaGetLegacyRequirements(VkDevice, VkBuffer, VkMemoryRequirements*);
static VKAPI_ATTR void VKAPI_CALL arenaGetRequirements(VkDevice, const VkBufferMemoryRequirementsInfo2*, VkMemoryRequirements2*);
static VKAPI_ATTR VkResult VKAPI_CALL arenaAllocateMemory(VkDevice, const VkMemoryAllocateInfo*, const VkAllocationCallbacks*, VkDeviceMemory*);
static VKAPI_ATTR VkResult VKAPI_CALL arenaBindMemory(VkDevice, VkBuffer, VkDeviceMemory, VkDeviceSize);
static VKAPI_ATTR void VKAPI_CALL arenaFreeMemory(VkDevice, VkDeviceMemory, const VkAllocationCallbacks*);
static VKAPI_ATTR VkResult VKAPI_CALL arenaMapMemory(VkDevice, VkDeviceMemory, VkDeviceSize, VkDeviceSize, VkMemoryMapFlags, void**);
static VKAPI_ATTR void VKAPI_CALL arenaUnmapMemory(VkDevice, VkDeviceMemory);

#define vkCreateBuffer arenaCreateBuffer
#define vkDestroyBuffer arenaDestroyBuffer
#define vkGetBufferMemoryRequirements2 arenaGetRequirements
#define vkGetBufferMemoryRequirements arenaGetLegacyRequirements
#define vkAllocateMemory arenaAllocateMemory
#define vkBindBufferMemory arenaBindMemory
#define vkFreeMemory arenaFreeMemory
#define vkMapMemory arenaMapMemory
#define vkUnmapMemory arenaUnmapMemory
#include "../vulkan_shim.cpp"
#undef vkCreateBuffer
#undef vkDestroyBuffer
#undef vkGetBufferMemoryRequirements2
#undef vkGetBufferMemoryRequirements
#undef vkAllocateMemory
#undef vkBindBufferMemory
#undef vkFreeMemory
#undef vkMapMemory
#undef vkUnmapMemory

namespace {
struct ArenaFixture {
    VkMemoryRequirements req{64, 256, 3};
    VkResult results[3]{VK_SUCCESS, VK_SUCCESS, VK_SUCCESS};
    VkResult bindResult = VK_SUCCESS;
    VkResult mapResult = VK_SUCCESS;
    char mapping[1024]{}; // only offsets are inspected; no simulated large upload
    std::vector<VkDeviceMemory> mapped;
    size_t maps = 0;
    size_t explicitUnmaps = 0;
    size_t implicitUnmaps = 0;
    bool poisonFailedOutput = false;
    std::vector<uint32_t> attempts;
    std::vector<VkDeviceSize> allocationSizes;
    std::vector<VkDeviceMemory> owned;
    std::vector<VkDeviceMemory> freed;
    std::vector<VkBuffer> destroyed;
    std::vector<int> events; // type index for allocation, -1 for pool release
    size_t binds = 0;
    uintptr_t nextBuffer = 100;
    uintptr_t nextMemory = 200;
} fixture;

template <typename H> H handle(uintptr_t value) { return (H)value; }
const VkDeviceMemory poolMemory = handle<VkDeviceMemory>(50);

void resetFixture() {
    assert(g_weightArena.empty() && g_weightArenaReservedBytes == 0);
    assert(g_poolCount == 0 && g_batchFreed.empty());
    assert(fixture.mapped.empty());
    fixture = {};
    g_umaDirectWeights = false;
    g_pool.clear();
    g_memprops = {};
    g_memprops.memoryTypeCount = 3;
    g_memprops.memoryHeapCount = 3;
    for (uint32_t i = 0; i < 3; ++i) {
        g_memprops.memoryTypes[i] = {VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, i};
        g_memprops.memoryHeaps[i] = {WEIGHT_ARENA_BLOCK_BYTES * 4, VK_MEMORY_HEAP_DEVICE_LOCAL_BIT};
    }
    g_memprops.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    g_ready = true;
    g_dev = handle<VkDevice>(1);
    g_maxBufferBytes = WEIGHT_ARENA_BLOCK_BYTES;
    g_maxMemoryAllocationSize = WEIGHT_ARENA_BLOCK_BYTES;
    g_totalDeviceLocalMemory = WEIGHT_ARENA_BLOCK_BYTES * 4;
    g_weightArenaMemoryAllocations = g_weightArenaBufferBindings = 0;
    g_weightArenaLiveBytes = g_weightArenaPeakReservedBytes = 0;
    g_deviceAllocationLiveBytes = 0;
    g_deviceAllocationAccountingValid = true;
    g_nextMemoryAllocationID = 1;
    g_v41SubmissionPendingFailure = g_batching = false;
}

void expectNoPublication() {
    assert(g_weightArena.empty());
    assert(g_weightArenaMemoryAllocations == 0 && g_weightArenaBufferBindings == 0);
    assert(g_weightArenaReservedBytes == 0 && g_weightArenaLiveBytes == 0);
    assert(g_weightArenaPeakReservedBytes == 0 && g_deviceAllocationLiveBytes == 0);
    assert(g_nextMemoryAllocationID == 1 && g_deviceAllocationAccountingValid);
}

void expectBacking(Buffer* b, uint32_t type, uint64_t id, uint64_t offset) {
    fvk_buffer_backing_info backing{};
    fvk_buffer_reservation_info reservation{};
    assert(fvk_buffer_backing(b, &backing) == 1);
    assert(fvk_buffer_reservation(b, &reservation) == 1);
    assert(backing.memory_type_index == type && backing.heap_index == type);
    assert(backing.property_flags == g_memprops.memoryTypes[type].propertyFlags);
    assert(backing.requested_property_flags == VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    assert(backing.heap_flags == VK_MEMORY_HEAP_DEVICE_LOCAL_BIT);
    assert(backing.weight_arena_bound == 1 && backing.host_visible_fallback == 0);
    assert(b->props == backing.property_flags && b->memoryTypeIndex == type);
    assert(reservation.allocation_id == id && reservation.binding_offset == offset);
    assert(reservation.reservation_bytes == WEIGHT_ARENA_BLOCK_BYTES);
    assert(reservation.buffer_bytes == b->bytes);
}

void pressureReuseAndLifetime() {
    resetFixture();
    fixture.results[0] = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    Buffer* recycled = new Buffer();
    recycled->buf = handle<VkBuffer>(50);
    recycled->mem = poolMemory;
    recycled->bytes = recycled->allocationBytes = 16;
    recycled->allocationDeviceLocal = true;
    g_pool[16].push_back(recycled);
    g_poolCount = 1;
    trackDeviceAllocation(16, 0);

    Buffer* first = allocWeightArenaBuffer(60, WEIGHT_ARENA_BLOCK_BYTES);
    assert(first != nullptr);
    assert((fixture.attempts == std::vector<uint32_t>{0, 0, 1}));
    assert((fixture.events == std::vector<int>{0, -1, 0, 1}));
    assert(fixture.freed == std::vector<VkDeviceMemory>{poolMemory});
    assert(g_poolCount == 0 && g_weightArena.size() == 1);
    const VkDeviceMemory selected = first->mem;
    assert(g_weightArena[0].memoryTypeIndex == 1 && g_weightArena[0].allocationID == 1);
    assert(g_weightArena[0].allocationDeviceLocal && g_weightArena[0].used == 64);
    expectBacking(first, 1, 1, 0);
    assert(first->mapped == fixture.mapping && g_weightArena[0].mapped == fixture.mapping);
    assert(fixture.maps == 1 && fixture.mapped == std::vector<VkDeviceMemory>{selected});

    Buffer* second = allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES);
    assert(second != nullptr && second->mem == selected);
    expectBacking(second, 1, 1, 256);
    assert(second->mapped == fixture.mapping + 256 && fixture.maps == 1);
    assert(fixture.attempts.size() == 3 && fixture.binds == 2);
    assert(g_weightArena[0].used == 320 && g_weightArena[0].liveBuffers == 2);
    assert(g_weightArenaMemoryAllocations == 1 && g_weightArenaBufferBindings == 2);
    assert(g_weightArenaReservedBytes == WEIGHT_ARENA_BLOCK_BYTES);
    assert(g_weightArenaPeakReservedBytes == WEIGHT_ARENA_BLOCK_BYTES);
    assert(g_weightArenaLiveBytes == 93 && g_deviceAllocationLiveBytes == WEIGHT_ARENA_BLOCK_BYTES);
    assert(g_nextMemoryAllocationID == 2);

    // Existing-block bind failure must not consume an aligned suffix or publish
    // a binding. The unchanged reservation cap prevents allocating another block.
    fixture.bindResult = VK_ERROR_MEMORY_MAP_FAILED;
    assert(allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
    fixture.bindResult = VK_SUCCESS;
    assert(g_weightArena[0].used == 320 && g_weightArena[0].liveBuffers == 2);
    assert(g_weightArenaBufferBindings == 2 && g_weightArenaLiveBytes == 93);
    assert(fixture.freed.size() == 1 && fixture.attempts.size() == 3);

    // Reuse still obeys the resource mask, actual type properties and table bound.
    const size_t binds = fixture.binds;
    fixture.req.memoryTypeBits = 1;
    assert(allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
    fixture.req.memoryTypeBits = 3;
    g_memprops.memoryTypes[1].propertyFlags = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;
    assert(allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
    g_memprops.memoryTypes[1].propertyFlags |= VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
    g_weightArena[0].memoryTypeIndex = 32;
    g_memprops.memoryTypeCount = 33;
    assert(allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
    g_memprops.memoryTypeCount = 3;
    g_weightArena[0].memoryTypeIndex = 1;
    assert(fixture.binds == binds && fixture.attempts.size() == 3);
    assert(g_weightArenaLiveBytes == 93 && g_weightArena[0].used == 320);

    g_v41SubmissionPendingFailure = true;
    assert(allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
    destroyBuffer(first);
    releaseWeightArena();
    assert(g_weightArena[0].liveBuffers == 2 && fixture.freed.size() == 1);
    assert(fixture.mapped.size() == 1 && fixture.implicitUnmaps == 0 && fixture.explicitUnmaps == 0);
    g_v41SubmissionPendingFailure = false;
    g_batching = true;
    fvk_free(first);
    assert(g_batchFreed.size() == 1 && g_weightArena[0].liveBuffers == 2);
    g_batching = false;
    // Exercise the release half after synthetic quiescence. This does not test
    // the real fence/submit path that normally drains the deferred queue.
    g_batchFreed.clear();
    fvk_free(first);
    assert(g_weightArena[0].liveBuffers == 1 && g_weightArenaLiveBytes == 33);
    assert(g_weightArena[0].used == 320 && fixture.freed.size() == 1);
    fvk_free(second);
    assert(g_weightArena.empty() && g_weightArenaReservedBytes == 0);
    assert(g_weightArenaLiveBytes == 0 && g_deviceAllocationLiveBytes == 0);
    assert((fixture.freed == std::vector<VkDeviceMemory>{poolMemory, selected}));
    assert(fixture.owned.empty() && g_deviceAllocationAccountingValid);
    assert(fixture.mapped.empty() && fixture.implicitUnmaps == 1 && fixture.explicitUnmaps == 0);
}

void blockCapacityAndFailureControls() {
    resetFixture();
    // The payload and req.size fit heap 0, but the actual arena block does not.
    g_memprops.memoryHeaps[0].size = fixture.req.size;
    Buffer* b = allocWeightArenaBuffer(60, WEIGHT_ARENA_BLOCK_BYTES);
    assert(b != nullptr && fixture.attempts == std::vector<uint32_t>{1});
    assert(fixture.allocationSizes == std::vector<VkDeviceSize>{WEIGHT_ARENA_BLOCK_BYTES});
    expectBacking(b, 1, 1, 0);
    fvk_free(b);
    assert(fixture.owned.empty() && g_deviceAllocationLiveBytes == 0);

    // All pressure, fatal first result, fatal after pressure, no eligible heap,
    // and a successful allocation whose bind fails all leave ownership private.
    for (int scenario = 0; scenario < 5; ++scenario) {
        resetFixture();
        fixture.req.memoryTypeBits = 7;
        fixture.poisonFailedOutput = true;
        std::vector<uint32_t> expected;
        switch (scenario) {
        case 0:
            fixture.results[0] = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            fixture.results[1] = VK_ERROR_OUT_OF_HOST_MEMORY;
            fixture.results[2] = VK_ERROR_TOO_MANY_OBJECTS;
            expected = {0, 0, 1, 2};
            break;
        case 1:
            fixture.results[0] = VK_ERROR_DEVICE_LOST;
            expected = {0};
            break;
        case 2:
            fixture.results[0] = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            fixture.results[1] = VK_ERROR_FEATURE_NOT_PRESENT;
            expected = {0, 0, 1};
            break;
        case 3:
            for (auto& heap : g_memprops.memoryHeaps) heap.size = 64;
            break;
        case 4:
            g_memprops.memoryHeaps[0].size = 64;
            fixture.bindResult = VK_ERROR_MEMORY_MAP_FAILED;
            expected = {1};
            break;
        }
        assert(allocWeightArenaBuffer(60, WEIGHT_ARENA_BLOCK_BYTES) == nullptr);
        assert(fixture.attempts == expected);
        assert(fixture.destroyed.size() == 1 && fixture.owned.empty());
        assert(fixture.binds == (scenario == 4 ? 1u : 0u));
        assert(fixture.freed.size() == (scenario == 4 ? 1u : 0u));
        expectNoPublication();
    }
}
// Keep the UMA selection and mapped-view contract in the same injected owner
// witness. Map failure retains the allocation and uses the existing staging path.
void umaPreferenceAndMappedLifetime() {
    for (int scenario = 0; scenario < 7; ++scenario) {
        resetFixture();
        g_umaDirectWeights = true;
        std::vector<uint32_t> expected{1};
        uint32_t selectedType = 1;
        bool expectMap = true;
        bool expectMapped = true;
        switch (scenario) {
        case 0: // Preferred coherent UMA beats the lower-index device-only type.
            break;
        case 1: // After pressure, compatible device-only backing remains reusable.
            fixture.results[1] = VK_ERROR_OUT_OF_DEVICE_MEMORY;
            expected = {1, 1, 0};
            selectedType = 0;
            expectMap = expectMapped = false;
            break;
        case 2: // Dirty failed map output must never become a published host view.
            fixture.mapResult = VK_ERROR_MEMORY_MAP_FAILED;
            expectMapped = false;
            break;
        case 3: // A noncoherent compatible type must not enter unflushed direct copy.
            fixture.req.memoryTypeBits = 2;
            g_memprops.memoryTypes[1].propertyFlags &= ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
            expectMap = expectMapped = false;
            break;
        case 4: // Disabling preference does not disable coherent mapped backing.
            g_umaDirectWeights = false;
            fixture.req.memoryTypeBits = 2;
            break;
        case 5: // A fatal preferred allocation error must not try another type.
            fixture.results[1] = VK_ERROR_DEVICE_LOST;
            break;
        case 6: // Block-size ineligibility is not a driver allocation failure.
            g_memprops.memoryHeaps[1].size = fixture.req.size;
            expected = {0};
            selectedType = 0;
            expectMap = expectMapped = false;
            break;
        }
        Buffer* first = allocWeightArenaBuffer(60, WEIGHT_ARENA_BLOCK_BYTES);
        assert(fixture.attempts == expected);
        if (scenario == 5) {
            assert(first == nullptr && fixture.maps == 0 && fixture.owned.empty());
            expectNoPublication();
            continue;
        }
        assert(first != nullptr);
        expectBacking(first, selectedType, 1, 0);
        assert(first->mapped == (expectMapped ? fixture.mapping : nullptr));
        const VkDeviceMemory memory = first->mem;
        Buffer* second = allocWeightArenaBuffer(33, WEIGHT_ARENA_BLOCK_BYTES);
        assert(second != nullptr && second->mem == memory);
        expectBacking(second, selectedType, 1, 256);
        assert(second->mapped == (expectMapped ? fixture.mapping + 256 : nullptr));
        assert(fixture.attempts == expected && fixture.maps == (expectMap ? 1u : 0u));
        if (expectMapped) {
            const char payload[] = "UMA";
            copyHostToMappedWeight(second, payload, sizeof(payload));
            assert(memcmp(fixture.mapping + 256, payload, sizeof(payload)) == 0);
            assert(fixture.mapping[0] == 0); // reuse must not alias the first binding
        }
        fvk_free(first);
        assert(fixture.freed.empty() && fixture.explicitUnmaps == 0 && fixture.implicitUnmaps == 0);
        fvk_free(second);
        assert(fixture.freed == std::vector<VkDeviceMemory>{memory});
        assert(fixture.mapped.empty() && fixture.owned.empty());
        assert(fixture.explicitUnmaps == 0 && fixture.implicitUnmaps == (expectMapped ? 1u : 0u));
        assert(g_weightArena.empty() && g_deviceAllocationLiveBytes == 0);
    }
}
} // namespace

static VKAPI_ATTR VkResult VKAPI_CALL arenaCreateBuffer(VkDevice, const VkBufferCreateInfo* info,
                                                       const VkAllocationCallbacks*, VkBuffer* out) {
    assert(info->usage == STORAGE_USAGE && info->size > 0);
    *out = handle<VkBuffer>(++fixture.nextBuffer);
    return VK_SUCCESS;
}

static VKAPI_ATTR void VKAPI_CALL arenaDestroyBuffer(VkDevice, VkBuffer buffer, const VkAllocationCallbacks*) {
    assert(buffer != VK_NULL_HANDLE);
    assert(std::find(fixture.destroyed.begin(), fixture.destroyed.end(), buffer) == fixture.destroyed.end());
    fixture.destroyed.push_back(buffer);
}

// Dedicated fallback is outside this witness; fail closed if it becomes reachable
// rather than passing a synthetic device or buffer to the Vulkan loader.
static VKAPI_ATTR void VKAPI_CALL arenaGetLegacyRequirements(VkDevice, VkBuffer, VkMemoryRequirements*) {
    std::abort();
}

static VKAPI_ATTR void VKAPI_CALL arenaGetRequirements(VkDevice, const VkBufferMemoryRequirementsInfo2*,
                                                      VkMemoryRequirements2* out) {
    out->memoryRequirements = fixture.req;
    auto* dedicated = static_cast<VkMemoryDedicatedRequirements*>(out->pNext);
    dedicated->requiresDedicatedAllocation = VK_FALSE;
    dedicated->prefersDedicatedAllocation = VK_FALSE;
}

static VKAPI_ATTR VkResult VKAPI_CALL arenaAllocateMemory(VkDevice, const VkMemoryAllocateInfo* info,
                                                         const VkAllocationCallbacks*, VkDeviceMemory* out) {
    assert(*out == VK_NULL_HANDLE); // fresh output storage, including after a failed candidate
    assert(info->memoryTypeIndex < 3 && info->pNext == nullptr);
    assert(info->allocationSize == WEIGHT_ARENA_BLOCK_BYTES);
    fixture.attempts.push_back(info->memoryTypeIndex);
    fixture.allocationSizes.push_back(info->allocationSize);
    fixture.events.push_back(static_cast<int>(info->memoryTypeIndex));
    const VkResult result = fixture.results[info->memoryTypeIndex];
    // A failed driver result is not an owned allocation, even with dirty output.
    *out = fixture.poisonFailedOutput ? handle<VkDeviceMemory>(999) : VK_NULL_HANDLE;
    if (result == VK_SUCCESS) {
        *out = handle<VkDeviceMemory>(++fixture.nextMemory);
        fixture.owned.push_back(*out);
    }
    return result;
}

static VKAPI_ATTR VkResult VKAPI_CALL arenaBindMemory(VkDevice, VkBuffer, VkDeviceMemory memory,
                                                     VkDeviceSize offset) {
    assert(std::find(fixture.owned.begin(), fixture.owned.end(), memory) != fixture.owned.end());
    assert(offset % fixture.req.alignment == 0);
    assert(offset <= WEIGHT_ARENA_BLOCK_BYTES && fixture.req.size <= WEIGHT_ARENA_BLOCK_BYTES - offset);
    ++fixture.binds;
    return fixture.bindResult;
}

static VKAPI_ATTR void VKAPI_CALL arenaFreeMemory(VkDevice, VkDeviceMemory memory, const VkAllocationCallbacks*) {
    assert(std::find(fixture.freed.begin(), fixture.freed.end(), memory) == fixture.freed.end());
    if (memory == poolMemory) {
        fixture.events.push_back(-1);
    } else {
        auto it = std::find(fixture.owned.begin(), fixture.owned.end(), memory);
        assert(it != fixture.owned.end()); // never free a failed candidate's dirty output
        fixture.owned.erase(it);
    }
    // Vulkan frees implicitly unmap mapped memory. Preserve that production
    // disposal contract and reject double-unmap/free in the injected ownership.
    auto mapped = std::find(fixture.mapped.begin(), fixture.mapped.end(), memory);
    if (mapped != fixture.mapped.end()) {
        fixture.mapped.erase(mapped);
        ++fixture.implicitUnmaps;
    }
    fixture.freed.push_back(memory);
}

static VKAPI_ATTR VkResult VKAPI_CALL arenaMapMemory(VkDevice, VkDeviceMemory memory,
    VkDeviceSize offset, VkDeviceSize size, VkMemoryMapFlags flags, void** out) {
    assert(std::find(fixture.owned.begin(), fixture.owned.end(), memory) != fixture.owned.end());
    assert(std::find(fixture.mapped.begin(), fixture.mapped.end(), memory) == fixture.mapped.end());
    assert(offset == 0 && size == VK_WHOLE_SIZE && flags == 0 && *out == nullptr);
    ++fixture.maps;
    *out = fixture.mapping; // deliberately dirty even on failure
    if (fixture.mapResult == VK_SUCCESS) fixture.mapped.push_back(memory);
    return fixture.mapResult;
}

static VKAPI_ATTR void VKAPI_CALL arenaUnmapMemory(VkDevice, VkDeviceMemory memory) {
    auto it = std::find(fixture.mapped.begin(), fixture.mapped.end(), memory);
    assert(it != fixture.mapped.end());
    fixture.mapped.erase(it);
    ++fixture.explicitUnmaps;
}

int main() {
    pressureReuseAndLifetime();
    blockCapacityAndFailureControls();
    umaPreferenceAndMappedLifetime();
    return 0;
}
`
