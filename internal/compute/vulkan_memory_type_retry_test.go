//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"reflect"
	"strings"
	"testing"
)

// Estimate only; no runtime measurement has been performed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanCompatibleMemoryTypeSelection(t *testing.T) {
	// VkResult constants are part of the Vulkan ABI, not backend observations.
	const success, hostOOM, deviceOOM, deviceLost, mapFailed, tooMany, absent = int32(0), int32(-1), int32(-2), int32(-4), int32(-5), int32(-10), int32(-8)
	const none = ^uint32(0)
	for _, tc := range []struct {
		name       string
		bits, want uint32
		flags      []uint32
		results    []int32
		all        bool
		status     int32
		attempts   []uint32
		selected   uint32
	}{
		{"first_success", 3, 1, []uint32{1, 1}, []int32{success, success}, false, success, []uint32{0}, 0},
		{"first_attempt_stops_before_drain", 3, 1, []uint32{1, 1}, []int32{deviceOOM, success}, false, deviceOOM, []uint32{0}, none},
		{"retry_later_compatible", 7, 1, []uint32{1, 2, 3}, []int32{deviceOOM, success, success}, true, success, []uint32{0, 2}, 2},
		{"type_bits_filter", 2, 1, []uint32{1, 1}, []int32{success, success}, true, success, []uint32{1}, 1},
		{"all_pressure_classes", 15, 1, []uint32{1, 1, 1, 1}, []int32{hostOOM, tooMany, deviceOOM, success}, true, success, []uint32{0, 1, 2, 3}, 3},
		{"exhausted", 3, 1, []uint32{1, 1}, []int32{hostOOM, deviceOOM}, true, deviceOOM, []uint32{0, 1}, none},
		{"callback_feature_absent_is_fatal", 3, 1, []uint32{1, 1}, []int32{absent, success}, true, absent, []uint32{0}, none},
		{"device_lost_stops", 3, 1, []uint32{1, 1}, []int32{deviceLost, success}, true, deviceLost, []uint32{0}, none},
		{"fatal_after_pressure", 7, 1, []uint32{1, 1, 1}, []int32{deviceOOM, mapFailed, success}, true, mapFailed, []uint32{0, 1}, none},
		{"no_compatible_properties", 3, 3, []uint32{1, 2}, []int32{success, success}, true, absent, nil, none},
		{"no_compatible_bits", 0, 1, []uint32{1}, []int32{success}, true, absent, nil, none},
		{"no_types", 1, 1, nil, nil, true, absent, nil, none},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(tc.bits, tc.want, 0, tc.flags, tc.results, make([]uint32, len(tc.flags)), []uint64{1024}, 1024, tc.all)
			if status != tc.status || !reflect.DeepEqual(attempts, tc.attempts) || selected != tc.selected || published != (tc.status == success) {
				t.Fatalf("status=%d attempts=%v selected=%d published=%t; want status=%d attempts=%v selected=%d", status, attempts, selected, published, tc.status, tc.attempts, tc.selected)
			}
		})
	}
	// Type31 is eligible; type32 must never shift by32 or read beyond Vulkan's
	// fixed memoryTypes[32] array, even if synthetic metadata overstates count.
	flags, results := make([]uint32, 33), make([]int32, 33)
	flags[31], flags[32] = 1, 1
	results[31] = deviceOOM
	status, attempts, selected, published := vulkanDebugMemoryTypeSelection(1<<31, 1, 0, flags, results, make([]uint32, len(flags)), []uint64{1024}, 1024, true)
	if status != deviceOOM || !reflect.DeepEqual(attempts, []uint32{31}) || selected != none || published {
		t.Fatalf("type bound escaped: status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published)
	}
}

// Estimate only; no runtime measurement has been performed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanMemoryTypeRetryOwnerSourceContract(t *testing.T) {
	raw, err := os.ReadFile("vulkan_shim.cpp")
	if err != nil {
		t.Fatal(err)
	}
	source := string(raw)
	start := strings.Index(source, "Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage,\n                    VkMemoryPropertyFlags preferred) {")
	end := strings.Index(source, "// device-local storage buffer used for all tensors")
	if start < 0 || end <= start {
		t.Fatal("missing allocation owner")
	}
	owner := source[start:end]
	for _, clause := range []string{
		"allocateCompatibleMemory(req.memoryTypeBits, want, preferred, g_memprops, req.size, tryAll,",
		"VkResult r = tryAlloc(props, &b->mem, &selectedMemoryType);",
		"if (allocPressure(r)) {\n        drainPool();\n        r = tryAlloc(props, &b->mem, &selectedMemoryType, true);",
		"if ((allocPressure(r) || (r == VK_ERROR_FEATURE_NOT_PRESENT && !allocationAttempted)) &&",
		"VkResult fr = tryAlloc(fallback, &b->mem, &selectedMemoryType, true);",
		"} else {\n            r = fr;",
	} {
		if !strings.Contains(owner, clause) {
			t.Errorf("allocation policy lost %q", clause)
		}
	}
	for _, clause := range []string{
		`!environmentFlagIs("FAK_VULKAN_STAGE_HOST_CACHED", '0')`,
		"g_preferCachedTransferStage ? VK_MEMORY_PROPERTY_HOST_CACHED_BIT : 0",
	} {
		if !strings.Contains(source, clause) {
			t.Errorf("cached staging preference lost %q", clause)
		}
	}
	failure := strings.Index(owner, "if (r != VK_SUCCESS) {")
	bind := strings.Index(owner, "VkResult br = vkBindBufferMemory")
	account := strings.Index(owner, "trackDeviceAllocation(req.size, selectedMemoryType);")
	if failure < 0 || bind <= failure || account <= bind {
		t.Fatal("failed candidates must not bind or publish accounting")
	}
}

// Estimate only; no runtime measurement has been performed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanCompatibleMemoryTypeHeapEligibility(t *testing.T) {
	const none = ^uint32(0)
	for _, tc := range []struct {
		name     string
		indices  []uint32
		heaps    []uint64
		results  []int32
		status   int32
		attempts []uint32
		selected uint32
	}{
		{"later_small_heap_skipped_then_valid", []uint32{0, 1, 2}, []uint64{1024, 1023, 2048}, []int32{-2, 0, 0}, 0, []uint32{0, 2}, 2},
		{"first_small_heap_skipped", []uint32{0, 1}, []uint64{1023, 1024}, []int32{0, 0}, 0, []uint32{1}, 1},
		{"later_small_heap_not_attempted", []uint32{0, 1}, []uint64{1024, 1023}, []int32{-2, 0}, -2, []uint32{0}, none},
		{"all_heaps_too_small", []uint32{0}, []uint64{1023}, []int32{0}, -8, nil, none},
		{"invalid_heap_index", []uint32{1, 0}, []uint64{1024}, []int32{0, 0}, 0, []uint32{1}, 1},
		{"no_heaps", []uint32{0}, nil, []int32{0}, -8, nil, none},
	} {
		t.Run(tc.name, func(t *testing.T) {
			flags := make([]uint32, len(tc.indices))
			for i := range flags {
				flags[i] = 1
			}
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(^uint32(0), 1, 0, flags, tc.results, tc.indices, tc.heaps, 1024, true)
			if status != tc.status || !reflect.DeepEqual(attempts, tc.attempts) || selected != tc.selected || published != (tc.status == 0) {
				t.Fatalf("heap eligibility: status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published)
			}
		})
	}
	// Vulkan's fixed memoryHeaps[16] bound wins over synthetic reported count.
	heaps := make([]uint64, 17)
	heaps[16] = 2048
	status, attempts, selected, published := vulkanDebugMemoryTypeSelection(1, 1, 0, []uint32{1}, []int32{0}, []uint32{16}, heaps, 1024, true)
	if status != -8 || len(attempts) != 0 || selected != none || published {
		t.Fatalf("heap bound escaped: status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published)
	}
}

// Estimate only; no runtime measurement has been performed.
// fak-test:runtime fast est=1ms lane=default
func TestVulkanPreferredMemoryTypeSelection(t *testing.T) {
	const visible, coherent, cached = uint32(2), uint32(4), uint32(8)
	required := visible | coherent
	flags := []uint32{required, required | cached, required | cached, required}
	heaps := []uint64{1024}
	heapIndices := make([]uint32, len(flags))
	for _, tc := range []struct {
		name      string
		results   []int32
		tryAll    bool
		status    int32
		attempts  []uint32
		selected  uint32
		published bool
	}{
		{"cached_first", []int32{0, 0, 0, 0}, false, 0, []uint32{1}, 1, true},
		{"cached_absent_falls_back", []int32{0, 0, 0, 0}, false, 0, []uint32{0}, 0, true},
		{"cached_pressure_then_compatible", []int32{0, -2, -1, 0}, true, 0, []uint32{1, 2, 0}, 0, true},
		{"pressure_exhausted", []int32{-2, -2, -1, -2}, true, -2, []uint32{1, 2, 0, 3}, ^uint32(0), false},
		{"cached_fatal_stops", []int32{0, -4, 0, 0}, true, -4, []uint32{1}, ^uint32(0), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			typeBits := uint32(15)
			fixtureFlags := flags
			if tc.name == "cached_absent_falls_back" {
				typeBits = 9
			}
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(typeBits, required, cached, fixtureFlags, tc.results, heapIndices, heaps, 1024, tc.tryAll)
			if status != tc.status || !reflect.DeepEqual(attempts, tc.attempts) || selected != tc.selected || published != tc.published {
				t.Fatalf("status=%d attempts=%v selected=%d published=%t; want status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published, tc.status, tc.attempts, tc.selected, tc.published)
			}
		})
	}
	for _, tc := range []struct {
		name        string
		flags       []uint32
		heapIndices []uint32
		heapSizes   []uint64
	}{
		{"cached_missing_coherent", []uint32{visible | cached, required}, []uint32{0, 0}, []uint64{1024}},
		{"cached_heap_ineligible", []uint32{required | cached, required}, []uint32{0, 1}, []uint64{1023, 1024}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(3, required, cached, tc.flags, []int32{0, 0}, tc.heapIndices, tc.heapSizes, 1024, true)
			if status != 0 || !reflect.DeepEqual(attempts, []uint32{1}) || selected != 1 || !published {
				t.Fatalf("status=%d attempts=%v selected=%d published=%t; want uncached type 1 only", status, attempts, selected, published)
			}
		})
	}
}

// fak-test:runtime hardware est=1s lane=hardware
func TestVulkanCachedTransferStageBacking(t *testing.T) {
	if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") != "1" {
		t.Skip("requires an explicitly admitted Vulkan device")
	}
	if os.Getenv("FAK_VULKAN_STAGE_HOST_CACHED") == "0" {
		t.Skip("cached staging is opted out by FAK_VULKAN_STAGE_HOST_CACHED=0")
	}
	if vulkanDev == nil {
		t.Fatal("required Vulkan backend was not initialized")
	}
	src := NewF32(cpu(), []int{4}, []float32{1, 2, 3, 4})
	device := vulkanDev.Upload(src, F32)
	t.Cleanup(func() { vulkanDev.Free(device) })
	if got := vulkanDev.Read(device); !reflect.DeepEqual(got, []float32{1, 2, 3, 4}) {
		t.Fatalf("transfer round trip = %v", got)
	}
	backings, ok := vulkanDev.VulkanTransferStageBufferBackings()
	if !ok || len(backings) != 1 {
		t.Fatalf("transfer stage backing ok=%t len=%d", ok, len(backings))
	}
	const hostVisible, hostCoherent, hostCached = uint32(2), uint32(4), uint32(8)
	want := hostVisible | hostCoherent
	if backings[0].Backing.PropertyFlags&want != want {
		t.Fatalf("transfer stage memory type=%d flags=%#x, want flags %#x", backings[0].Backing.MemoryTypeIndex, backings[0].Backing.PropertyFlags, want)
	}
	t.Logf("transfer stage memory type=%d flags=%#x host_cached=%t", backings[0].Backing.MemoryTypeIndex, backings[0].Backing.PropertyFlags, backings[0].Backing.PropertyFlags&hostCached != 0)
}
