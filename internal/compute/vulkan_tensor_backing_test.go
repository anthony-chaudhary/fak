//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"os"
	"runtime"
	"testing"
)

const (
	vkMemoryPropertyDeviceLocal = uint32(0x1)
	vkMemoryHeapDeviceLocal     = uint32(0x1)
)

func checkVulkanBufferBackingSelfConsistent(t *testing.T, b VulkanBufferBacking) {
	t.Helper()
	if !b.HostVisibleFallback && b.PropertyFlags&b.RequestedPropertyFlags != b.RequestedPropertyFlags {
		t.Fatalf("selected memory type lacks requested flags without a fallback: %+v", b)
	}
	if b.PropertyFlags&vkMemoryPropertyDeviceLocal != 0 && b.HeapFlags&vkMemoryHeapDeviceLocal == 0 {
		t.Fatalf("device-local memory type reported on a non-device-local heap: %+v", b)
	}
}

// fak-test:runtime integration est=250ms lane=optin
// Run with FAK_VULKAN_REQUIRE_DEVICE=1; a skipped run is no witness.
func TestVulkanTensorBufferBackings(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanTensorBufferBackings(Tensor{}); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	v := vk(t)
	host := NewF32(Default(), []int{4}, []float32{1, 2, 3, 4})
	if got, ok := v.VulkanTensorBufferBackings(host); ok || got != nil {
		t.Fatalf("foreign tensor = %v, %t; want unavailable", got, ok)
	}

	activation := v.UploadClass(host, F32, MemoryActivation, "backing activation")
	defer v.Free(activation)
	got, ok := v.VulkanTensorBufferBackings(activation)
	if !ok || len(got) != 1 || got[0].Part != "data" || got[0].Chunk != -1 || got[0].WeightArenaBound {
		t.Fatalf("activation backings = %+v, %t; want one unchunked direct data record", got, ok)
	}
	checkVulkanBufferBackingSelfConsistent(t, got[0])

	weight := v.Upload(host, F32)
	defer v.Free(weight)
	got, ok = v.VulkanTensorBufferBackings(weight)
	if !ok || len(got) != 1 || got[0].Part != "data" || got[0].Chunk != -1 ||
		!got[0].WeightArenaBound || got[0].RequestedPropertyFlags != vkMemoryPropertyDeviceLocal {
		t.Fatalf("weight backings = %+v, %t; want one arena-bound device-local request", got, ok)
	}
	checkVulkanBufferBackingSelfConsistent(t, got[0])

	w := make([]float32, 2*32)
	for i := range w {
		w[i] = float32(i%7) - 3
	}
	q8 := v.Upload(NewF32(Default(), []int{2, 32}, w), Q8_0)
	defer v.Free(q8)
	got, ok = v.VulkanTensorBufferBackings(q8)
	if !ok || len(got)%2 != 0 || len(got) == 0 {
		t.Fatalf("Q8_0 backings = %+v, %t; want complete data/scales pairs", got, ok)
	}
	for i := 0; i < len(got); i += 2 {
		if got[i].Part != "data" || got[i+1].Part != "scales" || got[i].Chunk != got[i+1].Chunk {
			t.Fatalf("Q8_0 backing order = %+v; want data then scales per chunk", got)
		}
		checkVulkanBufferBackingSelfConsistent(t, got[i])
		checkVulkanBufferBackingSelfConsistent(t, got[i+1])
	}
	want := append([]VulkanBufferBacking(nil), got...)
	got[0] = VulkanBufferBacking{}
	again, ok := v.VulkanTensorBufferBackings(q8)
	if !ok || len(again) != len(want) {
		t.Fatalf("repeated Q8_0 query = %+v, %t", again, ok)
	}
	for i := range want {
		if again[i] != want[i] {
			t.Fatalf("repeated query or caller mutation changed evidence: %+v vs %+v", again, want)
		}
	}

	v.Free(weight)
	if got, ok := v.VulkanTensorBufferBackings(weight); ok || got != nil {
		t.Fatalf("freed tensor = %+v, %t; want unavailable", got, ok)
	}
}

// fak-test:runtime integration est=50ms lane=optin
func TestVulkanDRMRenderNodeIdentity(t *testing.T) {
	var absent *vulkanBackend
	if major, minor, ok := absent.VulkanDRMRenderNode(); ok || major != 0 || minor != 0 {
		t.Fatalf("nil backend = %d:%d %t; want unavailable", major, minor, ok)
	}
	v := vk(t)
	major, minor, ok := v.VulkanDRMRenderNode()
	if runtime.GOOS != "linux" {
		if ok || major != 0 || minor != 0 {
			t.Fatalf("%s render node = %d:%d %t; want unavailable", runtime.GOOS, major, minor, ok)
		}
		return
	}
	if !ok {
		if os.Getenv("FAK_VULKAN_REQUIRE_DEVICE") == "1" {
			t.Fatal("required Linux Vulkan device reports no DRM render node")
		}
		t.Skip("selected device does not expose VK_EXT_physical_device_drm render identity")
	}
	if major == 0 {
		t.Fatalf("available render node has zero major: %d:%d", major, minor)
	}
	if m2, n2, ok2 := v.VulkanDRMRenderNode(); !ok2 || m2 != major || n2 != minor {
		t.Fatalf("repeated render node = %d:%d %t; want stable %d:%d", m2, n2, ok2, major, minor)
	}
}
