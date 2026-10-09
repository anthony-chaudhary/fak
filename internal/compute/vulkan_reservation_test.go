//go:build vulkan && (windows || linux) && cgo

package compute

import "testing"

// fak-test:runtime integration est=500ms lane=optin
// Run alone on the required physical Vulkan device with no live weight arena.
// The shared-allocation case requires non-dedicated small storage buffers and
// enough configured weight budget for one arena block. A skip is not a witness.
func TestVulkanTensorBufferReservations(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanTensorBufferReservations(Tensor{}); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	v := vk(t)
	host := NewF32(Default(), []int{4}, []float32{1, 2, 3, 4})
	if got, ok := v.VulkanTensorBufferReservations(host); ok || got != nil {
		t.Fatalf("foreign tensor = %v, %t; want unavailable", got, ok)
	}
	if s := v.VulkanWeightArenaStats(); s.LiveBytes != 0 || s.ReservedBytes != 0 {
		t.Fatalf("reservation lifetime fixture requires an empty weight arena: %+v", s)
	}
	readOne := func(tensor Tensor) VulkanBufferReservation {
		t.Helper()
		got, ok := v.VulkanTensorBufferReservations(tensor)
		if !ok || len(got) != 1 {
			t.Fatalf("tensor reservation = %v, %t; want one complete record", got, ok)
		}
		r := got[0]
		if r.Backing.Part != "data" || r.Backing.Chunk != -1 || r.BufferBytes != 16 ||
			r.AllocationID == 0 || r.ReservationBytes < r.BufferBytes ||
			r.BindingOffset > r.ReservationBytes-r.BufferBytes {
			t.Fatalf("invalid reservation: %+v", r)
		}
		got[0] = VulkanBufferReservation{}
		return r
	}

	directA := v.UploadClass(host, F32, MemoryActivation, "reservation direct A")
	defer v.Free(directA)
	directB := v.UploadClass(host, F32, MemoryActivation, "reservation direct B")
	defer v.Free(directB)
	a, b := readOne(directA), readOne(directB)
	if a.Backing.WeightArenaBound || b.Backing.WeightArenaBound ||
		a.BindingOffset != 0 || b.BindingOffset != 0 || a.AllocationID == b.AllocationID {
		t.Fatalf("distinct direct reservations were conflated: A=%+v B=%+v", a, b)
	}

	weightA := v.Upload(host, F32)
	defer v.Free(weightA)
	weightB := v.Upload(host, F32)
	defer v.Free(weightB)
	wa, wb := readOne(weightA), readOne(weightB)
	if !wa.Backing.WeightArenaBound || !wb.Backing.WeightArenaBound ||
		wa.AllocationID != wb.AllocationID || wa.ReservationBytes != wb.ReservationBytes ||
		wa.BindingOffset == wb.BindingOffset || wa.ReservationBytes <= wa.BufferBytes ||
		wa.AllocationID == a.AllocationID || wa.AllocationID == b.AllocationID {
		t.Fatalf("expected two bindings of one separate arena reservation: A=%+v B=%+v", wa, wb)
	}
	if s := v.VulkanWeightArenaStats(); s.ReservedBytes != wa.ReservationBytes || s.LiveBytes != 32 {
		t.Fatalf("buffer reservation disagrees with the single arena owner: %+v, %+v", s, wa)
	}
	before, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if readOne(directA) != a || readOne(weightA) != wa || readOne(weightB) != wb {
		t.Fatal("repeated query or result mutation changed live reservation evidence")
	}
	after, err := v.BackendExecutionSnapshot()
	if err != nil {
		t.Fatal(err)
	}
	if before.Counters != after.Counters || !before.DeviceAllocationObserved ||
		!after.DeviceAllocationObserved || before.DeviceAllocationLiveBytes != after.DeviceAllocationLiveBytes {
		t.Fatal("reservation queries changed observed counters or live device-local allocation bytes")
	}
	v.Free(weightA)
	if got, ok := v.VulkanTensorBufferReservations(weightA); ok || got != nil {
		t.Fatalf("retired tensor = %v, %t; want unavailable", got, ok)
	}
	if readOne(weightB) != wb {
		t.Fatal("retiring one binding changed its live sibling's reservation")
	}
	v.Free(weightB)
	if s := v.VulkanWeightArenaStats(); s.LiveBytes != 0 || s.ReservedBytes != 0 {
		t.Fatalf("last binding did not retire the fixture's arena: %+v", s)
	}
	weightC := v.Upload(host, F32)
	defer v.Free(weightC)
	wc := readOne(weightC)
	if !wc.Backing.WeightArenaBound || wc.AllocationID == wa.AllocationID {
		t.Fatalf("recreated arena reused a retired allocation identity: old=%+v new=%+v", wa, wc)
	}
}
