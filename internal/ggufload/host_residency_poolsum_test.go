package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestRefuseUnifiedHostResidencyRefusesPlanOverPhysicalPool is the fak#13280 RED->GREEN witness.
//
// The #13280 physical run proved the predecessor premise FALSE: on a split-aperture integrated
// box (strix3: 64 GiB VRAM window + 62.4 GiB system window over 128 GB LPDDR5X) the two windows
// are APERTURE WINDOWS over ONE physical pool, not disjoint carve-outs. The RADV/Vulkan device
// window reports a unified device-local+host-visible heap LARGER than physical RAM, so a plan
// whose device half fits that window and whose host half fits the system window can have a GRAND
// total above MemTotal and be global-OOM-killed before the first token.
//
// This test reconstructs the witnessed strix3 shape (device dense 63.092 GiB <= 64 GiB VRAM,
// host expert cache 8.425 GiB <= 62.4 GiB system) -- the exact plan the predecessor
// TestRefuseUnifiedHostResidencyAdmitsSplitAperture asserted ADMITTED. #13280 on clean silicon
// disproved that: the grand total 71.5 GiB exceeds the 62.4 GiB physical pool. The fix must
// refuse it with a typed *compute.FitError naming the physical (host) pool.
func TestRefuseUnifiedHostResidencyRefusesPlanOverPhysicalPool(t *testing.T) {
	const gib = int64(1 << 30)
	deviceWeights := int64(witnessedDeviceWeightsGiB * float64(gib)) // 63.092 GiB
	hostExperts := int64(witnessedHostExpertsGiB * float64(gib))     // 8.425 GiB
	vram := int64(64 * float64(gib))                                 // 64 GiB device window
	system := int64(witnessedHostTotalGiB * float64(gib))            // 62.4 GiB system window

	if deviceWeights > vram || hostExperts > system {
		t.Fatalf("fixture does not fit the split apertures (device %d<=%d, host %d<=%d)", deviceWeights, vram, hostExperts, system)
	}
	if deviceWeights+hostExperts <= system {
		t.Fatalf("fixture does not reproduce the defect: grand total %d must exceed the physical pool %d", deviceWeights+hostExperts, system)
	}
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceWeights, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: hostExperts, Detail: "gguf-host-expert-offload-streamed"},
	}

	err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, vram, vram, true, system, system, true, 0)
	if err == nil {
		t.Fatalf("expected a typed refusal: grand total %d exceeds the physical pool %d (fak#13280)", deviceWeights+hostExperts, system)
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("refusal must carry a *compute.FitError, got %T: %v", err, err)
	}
	if fe.Verdict != compute.FitTooBig {
		t.Fatalf("Verdict = %v, want FitTooBig", fe.Verdict)
	}
	if !strings.Contains(err.Error(), "unified-memory host residency") {
		t.Fatalf("refusal must carry the shared-pool provenance, got: %v", err)
	}

	// End-to-end through the backend-aware form must reach the SAME typed refusal.
	if err := RefuseUnifiedHostResidencyIfTooBig(plan, splitApertureBackend(vram, system), 0); err == nil {
		t.Fatal("backend-aware form admitted a plan over the physical pool")
	}
}

// TestRefuseUnifiedHostResidencyPhysicalPoolAdmitsPlanWithinPool is the fail-open twin: a plan
// whose grand total is within the physical pool (hostTotal) is admitted, so the physical-pool
// bound does not regress a genuinely-fitting split plan.
func TestRefuseUnifiedHostResidencyPhysicalPoolAdmitsPlanWithinPool(t *testing.T) {
	const gib = int64(1 << 30)
	vram := int64(64 * float64(gib))
	system := int64(witnessedHostTotalGiB * float64(gib))
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 30 * gib, Detail: "dense"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: 20 * gib, Detail: "experts"},
	}
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, vram, vram, true, system, system, true, 0); err != nil {
		t.Fatalf("plan within the physical pool must be admitted, got: %v", err)
	}
}
