package gateway

import (
	"errors"
	"testing"
)

// fak-test:runtime fast est=5ms lane=default
func TestPairRouter(t *testing.T) {
	r := NewPairRouter()
	r.Register(PairDevice{
		ID: "gpu-remote", Kind: "gpu", Local: false, Healthy: true,
		VRAMBytes: 24 << 30, FreeVRAMBytes: 20 << 30, Capabilities: []string{"fp8", "kv"}, Load: 0.1,
	})
	r.Register(PairDevice{
		ID: "gpu-local-busy", Kind: "gpu", Local: true, Healthy: true,
		VRAMBytes: 24 << 30, FreeVRAMBytes: 18 << 30, Capabilities: []string{"fp8", "kv"}, Load: 0.8,
	})
	r.Register(PairDevice{
		ID: "gpu-unhealthy", Kind: "gpu", Local: true, Healthy: false,
		VRAMBytes: 48 << 30, FreeVRAMBytes: 48 << 30, Capabilities: []string{"fp8", "kv"}, Load: 0.0,
	})
	r.Register(PairDevice{
		ID: "cpu-node", Kind: "cpu", Local: true, Healthy: true,
		VRAMBytes: 64 << 30, FreeVRAMBytes: 64 << 30, Capabilities: []string{"kv"}, Load: 0.0,
	})

	// PreferLocal selects healthy local capability-matched device over remote.
	plan, err := r.Select(PairRouteRequest{
		RequiredCaps: []string{"fp8", "kv"},
		MinVRAMBytes: 16 << 30,
		PreferLocal:  true,
	})
	if err != nil {
		t.Fatalf("Select PreferLocal err=%v", err)
	}
	if plan.SplitUsed || plan.Primary.ID != "gpu-local-busy" {
		t.Fatalf("PreferLocal plan=%+v, want gpu-local-busy", plan)
	}

	// Without PreferLocal, lowest-load healthy capability-matched device wins.
	plan, err = r.Select(PairRouteRequest{
		RequiredCaps: []string{"fp8", "kv"},
		MinVRAMBytes: 16 << 30,
	})
	if err != nil || plan.Primary.ID != "gpu-remote" {
		t.Fatalf("lowest-load plan=%+v err=%v, want gpu-remote", plan, err)
	}

	// Request larger than any single GPU (30 GiB) triggers cross-device split across the 2 healthy fp8 GPUs.
	plan, err = r.Select(PairRouteRequest{
		RequiredCaps: []string{"fp8", "kv"},
		MinVRAMBytes: 30 << 30,
	})
	if err != nil {
		t.Fatalf("cross-device split err=%v", err)
	}
	if !plan.SplitUsed || len(plan.Stages) != 2 {
		t.Fatalf("expected 2-stage cross-device split, got %+v", plan)
	}

	// Unsatisfiable capability returns ErrNoEligiblePairDevice.
	if _, err := r.Select(PairRouteRequest{RequiredCaps: []string{"nvfp4"}}); !errors.Is(err, ErrNoEligiblePairDevice) {
		t.Fatalf("missing cap err=%v, want ErrNoEligiblePairDevice", err)
	}
}
