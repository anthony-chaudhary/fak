package ggufload

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// carveoutBackend is an integrated split-aperture double that reports a dedicated VRAM
// carve-out the OS does not count in MemTotal (fak#13668).
type carveoutBackend struct {
	dualCapacityBackend
	carveout      int64
	carveoutKnown bool
}

func (b carveoutBackend) DedicatedVRAMCarveout() (int64, bool) { return b.carveout, b.carveoutKnown }

func gibBytes(v float64) int64 { return int64(v * float64(int64(1)<<30)) }

func requireUnifiedFitTooBig(t *testing.T, err error) {
	t.Helper()
	var fe *compute.FitError
	if !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
		t.Fatalf("want a typed FitTooBig *compute.FitError, got %T: %v", err, err)
	}
}

// strix3RingPlan is the 2026-10-09 arm-B ring plan: dense base plus device ring on the device,
// a small host scratch.
func strix3RingPlan() compute.MemoryPlan {
	return compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: gibBytes(65.95), Detail: "dense-base+ring"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeDevice, Bytes: gibBytes(0.12), Detail: "kv"},
		{Class: compute.MemoryScratchpad, Scope: compute.MemoryScopeHost, Bytes: gibBytes(1.21), Detail: "scratch"},
	}
}

func TestUnifiedHostResidencyAdmitsDeviceResidentPlanWithinCarveout(t *testing.T) {
	be := carveoutBackend{splitApertureBackend(gibBytes(96), gibBytes(31)), gibBytes(96), true}
	if err := RefuseUnifiedHostResidencyIfTooBig(strix3RingPlan(), be, 0.15); err != nil {
		t.Fatalf("a 66 GiB device-resident plan on MemTotal 31 GiB + 96 GiB carve-out must be admitted: %v", err)
	}
}

func TestUnifiedHostResidencyCarveoutUnknownKeepsMemTotalBound(t *testing.T) {
	for name, be := range map[string]compute.Backend{
		"no-reporter": splitApertureBackend(gibBytes(96), gibBytes(31)),
		"unreadable":  carveoutBackend{splitApertureBackend(gibBytes(96), gibBytes(31)), 0, false},
	} {
		t.Run(name, func(t *testing.T) {
			requireUnifiedFitTooBig(t, RefuseUnifiedHostResidencyIfTooBig(strix3RingPlan(), be, 0.15))
		})
	}
}

func TestUnifiedHostResidencyCarveoutKeeps13280Refused(t *testing.T) {
	cases := map[string]struct {
		carveout, device, host float64
	}{
		"small-carveout": {carveout: 0.5, device: 63, host: 44},
		// The witnessed #13280 box had a 64 GiB carve-out; its host-offload arm still OOM-killed.
		"witnessed-64GiB-carveout": {carveout: 64, device: 60.55, host: 45.45},
	}
	for name, c := range cases {
		t.Run(name, func(t *testing.T) {
			plan := compute.MemoryPlan{
				{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: gibBytes(c.device), Detail: "dense"},
				{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: gibBytes(c.host), Detail: "gguf-host-expert-offload-streamed"},
			}
			be := carveoutBackend{splitApertureBackend(gibBytes(64), gibBytes(62.4)), gibBytes(c.carveout), true}
			requireUnifiedFitTooBig(t, RefuseUnifiedHostResidencyIfTooBig(plan, be, 0.15))
		})
	}
}

func TestUnifiedHostResidencyRefusesDeviceSpillPastMemTotal(t *testing.T) {
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: gibBytes(60), Detail: "dense"},
		{Class: compute.MemoryScratchpad, Scope: compute.MemoryScopeHost, Bytes: gibBytes(1), Detail: "scratch"},
	}
	be := carveoutBackend{splitApertureBackend(gibBytes(96), gibBytes(31)), gibBytes(32), true}
	requireUnifiedFitTooBig(t, RefuseUnifiedHostResidencyIfTooBig(plan, be, 0.15))
}
