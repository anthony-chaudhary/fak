//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"slices"
	"testing"
)

// fak-test:runtime integration est=250ms lane=optin
// This uses the real upload/read seam. Run with the required-device environment
// to turn unavailable-device skips into failures; a skipped run is no witness.
func TestVulkanTransferStageBufferBackings(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanTransferStageBufferBackings(); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	v := vk(t)
	values := []float32{3.25, -7.5, 11, 0.125}
	d := v.Upload(NewF32(Default(), []int{len(values)}, values), F32)
	defer v.Free(d)

	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("before stage query: %v", err)
	}
	got, ok := v.VulkanTransferStageBufferBackings()
	if !ok || len(got) != 1 || got[0].BufferBytes < uint64(len(values)*4) {
		t.Fatalf("stage after upload = %v, %t; want one retained buffer", got, ok)
	}
	want := got[0]
	const hostVisibleCoherent = uint32(0x2 | 0x4)
	if b := want.Backing; b.Part != "data" || b.Chunk != -1 ||
		b.RequestedPropertyFlags != hostVisibleCoherent || b.PropertyFlags&hostVisibleCoherent != hostVisibleCoherent ||
		b.HostVisibleFallback || b.WeightArenaBound {
		t.Fatalf("unexpected ordinary transfer-stage backing: %+v", b)
	}
	got[0] = VulkanTransferStageBufferBacking{}
	if again, ok := v.VulkanTransferStageBufferBackings(); !ok || len(again) != 1 || again[0] != want {
		t.Fatalf("repeated stage query = %v, %t; want copied unchanged evidence", again, ok)
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatalf("after stage query: %v", err)
	}
	if before.Counters != after.Counters ||
		!before.DeviceAllocationObserved || !after.DeviceAllocationObserved ||
		before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
		t.Fatal("stage queries changed transfer/dispatch counters or observed live allocation bytes")
	}

	if read := v.Read(d); !slices.Equal(read, values) {
		t.Fatalf("device read = %v; want %v", read, values)
	}
	if afterRead, ok := v.VulkanTransferStageBufferBackings(); !ok || len(afterRead) != 1 || afterRead[0] != want {
		t.Fatalf("stage after read = %v, %t; want the shared H2D/D2H stage evidence", afterRead, ok)
	}
}
