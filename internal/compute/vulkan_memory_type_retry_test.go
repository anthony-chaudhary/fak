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
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(tc.bits, tc.want, tc.flags, tc.results, make([]uint32, len(tc.flags)), []uint64{1024}, 1024, tc.all)
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
	status, attempts, selected, published := vulkanDebugMemoryTypeSelection(1<<31, 1, flags, results, make([]uint32, len(flags)), []uint64{1024}, 1024, true)
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
	start := strings.Index(source, "Buffer* allocBuffer(size_t bytes, VkMemoryPropertyFlags props, VkBufferUsageFlags usage) {")
	end := strings.Index(source, "// device-local storage buffer used for all tensors")
	if start < 0 || end <= start {
		t.Fatal("missing allocation owner")
	}
	owner := source[start:end]
	for _, clause := range []string{
		"allocateCompatibleMemory(req.memoryTypeBits, want, g_memprops, req.size, tryAll,",
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
			status, attempts, selected, published := vulkanDebugMemoryTypeSelection(^uint32(0), 1, flags, tc.results, tc.indices, tc.heaps, 1024, true)
			if status != tc.status || !reflect.DeepEqual(attempts, tc.attempts) || selected != tc.selected || published != (tc.status == 0) {
				t.Fatalf("heap eligibility: status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published)
			}
		})
	}
	// Vulkan's fixed memoryHeaps[16] bound wins over synthetic reported count.
	heaps := make([]uint64, 17)
	heaps[16] = 2048
	status, attempts, selected, published := vulkanDebugMemoryTypeSelection(1, 1, []uint32{1}, []int32{0}, []uint32{16}, heaps, 1024, true)
	if status != -8 || len(attempts) != 0 || selected != none || published {
		t.Fatalf("heap bound escaped: status=%d attempts=%v selected=%d published=%t", status, attempts, selected, published)
	}
}
