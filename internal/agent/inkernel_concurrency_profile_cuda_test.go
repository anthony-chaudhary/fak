//go:build cuda && cgo

package agent

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// fak-test:runtime slow est=10s lane=default
func TestInKernelCUDAConcurrentForward(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		if os.Getenv("FAK_CONCURRENCY_PROFILE_OUT") != "" {
			t.Fatal("CUDA concurrency receipt requested, but exact cuda backend is not registered")
		}
		t.Skip("exact cuda backend not registered; set FAK_CONCURRENCY_PROFILE_OUT to require a CUDA receipt")
	}
	if !be.Caps().DeviceMemory || be.Tier() == "" || be.Class() != compute.Approx {
		t.Fatalf("cuda backend lacks device identity: device_memory=%t tier=%q class=%s", be.Caps().DeviceMemory, be.Tier(), be.Class())
	}
	prof := runInKernelConcurrencyProfile(t, be)
	assertInKernelDeviceMutexProfile(t, prof)
	assertInKernelConcurrencyProfileSaved(t, prof)
	if out := os.Getenv("FAK_CONCURRENCY_PROFILE_OUT"); out != "" {
		if err := prof.Save(out); err != nil {
			t.Fatalf("save CUDA concurrency profile: %v", err)
		}
	}
	t.Logf("CUDA baseline: backend=%s tier=%s mechanism=%s max_concurrent_forward=%d", prof.Backend, prof.BackendTier, prof.Mechanism, prof.MaxConcurrent)
}
