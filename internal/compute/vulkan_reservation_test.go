//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"context"
	"os"
	"os/exec"
	"strings"
	"testing"
	"time"
)

const vulkanReservationChildEnv = "FAK_TEST_VULKAN_RESERVATION_CHILD"

// fak-test:runtime integration est=2s lane=optin
// The child owns a fresh physical Vulkan context with no live weight arena.
// The shared-allocation case requires non-dedicated small storage buffers and
// enough configured weight budget for one arena block. A skip is not a witness.
func TestVulkanTensorBufferReservations(t *testing.T) {
	var absent *vulkanBackend
	if got, ok := absent.VulkanTensorBufferReservations(Tensor{}); ok || got != nil {
		t.Fatalf("nil backend = %v, %t; want unavailable", got, ok)
	}
	if os.Getenv(vulkanReservationChildEnv) != "1" {
		// Preserve parent eligibility while keeping unrelated process-lifetime
		// allocations out of the absolute arena lifetime assertions below.
		vk(t)
		exe, err := os.Executable()
		if err != nil {
			t.Fatalf("os.Executable: %v", err)
		}
		ctx, cancel := context.WithTimeout(t.Context(), 60*time.Second)
		defer cancel()
		cmd := exec.CommandContext(ctx, exe, "-test.v", "-test.run=^TestVulkanTensorBufferReservations$")
		for _, entry := range os.Environ() {
			key, _, _ := strings.Cut(entry, "=")
			if key != vulkanReservationChildEnv && key != "FAK_VULKAN_REQUIRE_DEVICE" {
				cmd.Env = append(cmd.Env, entry)
			}
		}
		cmd.Env = append(cmd.Env, vulkanReservationChildEnv+"=1", "FAK_VULKAN_REQUIRE_DEVICE=1")
		out, err := cmd.CombinedOutput()
		if ctx.Err() != nil {
			t.Fatalf("reservation subprocess timed out: %v\n%s", ctx.Err(), out)
		}
		if err != nil {
			t.Fatalf("reservation subprocess failed: %v\n%s", err, out)
		}
		if strings.Contains(string(out), "--- SKIP:") ||
			!strings.Contains(string(out), "--- PASS: TestVulkanTensorBufferReservations (") {
			t.Fatalf("reservation subprocess did not execute its required contract:\n%s", out)
		}
		t.Logf("Vulkan reservation fixture output:\n%s", out)
		return
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
