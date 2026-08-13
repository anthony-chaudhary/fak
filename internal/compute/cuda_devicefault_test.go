//go:build cuda

package compute

import (
	"errors"
	"testing"
)

// TestCUDAQwen35GDNDecodeRefusesOnPoisonedSessionWithoutDeviceWork proves the #6412 session
// gate sits BEFORE any device work: a poisoned backend refuses the decode entry with the
// typed device-fault error without validating operands, taking allocations, or crossing the
// C ABI. The backend here is constructed bare — no live CUDA context exists to touch, so the
// test itself is the witness that the refusal happens host-side.
func TestCUDAQwen35GDNDecodeRefusesOnPoisonedSessionWithoutDeviceWork(t *testing.T) {
	be := &cudaBackend{name: "cuda", faultLatch: NewDeviceFaultLatch("cuda", cudaFaultReconstructBudget)}
	if err := AdmitDevice(be, "healthy-pre-check"); err != nil {
		t.Fatalf("healthy session refused: %v", err)
	}
	be.faultLatch.Observe(DeviceFaultExecution, "xid-31", 31)
	_, _, _, err := be.Qwen35GDNDecode(
		Tensor{},
		Tensor{}, Tensor{}, Tensor{}, Tensor{},
		Tensor{}, Tensor{}, Tensor{}, Tensor{}, Tensor{},
		Tensor{}, Tensor{},
		0, 0, 0, 0, 0,
		0,
	)
	var faultErr *DeviceFaultError
	if !errors.As(err, &faultErr) {
		t.Fatalf("poisoned decode error = %T %v, want *DeviceFaultError", err, err)
	}
	if faultErr.Site != "qwen35-gdn-decode" || faultErr.Class != DeviceFaultExecution || faultErr.Code != 31 {
		t.Fatalf("refusal identity = %+v, want site qwen35-gdn-decode class %s code 31", faultErr, DeviceFaultExecution)
	}
	if got := be.faultLatch.Snapshot().RefusedCalls; got != 1 {
		t.Fatalf("refused-call telemetry = %d, want 1", got)
	}
}

// TestCUDABackendExposesSessionFaultLatch proves the registered device backend carries a live
// session latch and publishes it through the DeviceFaultReporter capability, so a serving
// path holding only a compute.Backend can gate on AdmitDevice and drive recovery without
// knowing the concrete backend type.
func TestCUDABackendExposesSessionFaultLatch(t *testing.T) {
	be := cudaGDNBackend(t)
	latch := BackendFaultLatch(be)
	if latch == nil {
		t.Fatal("registered cuda backend reports no session fault latch")
	}
	if latch.Backend() != "cuda" {
		t.Fatalf("latch backend = %q, want cuda", latch.Backend())
	}
}
