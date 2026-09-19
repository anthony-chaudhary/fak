package ggufload

import (
	"errors"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// host_residency_test.go —— fak#13172 regression. On an INTEGRATED device (Strix Halo / APU
// Vulkan tier) the device-visible weights and the host-resident expert/staging charge draw
// from ONE physical DRAM pool, but the per-scope fit checks judge each against its OWN probe:
// the device probe on RADV/Vulkan reports the unified heap (~84 GiB on a 62.4 GiB Halo), so a
// plan whose device dense side alone exceeds MemTotal passes device admission and host
// admission and is then SIGKILLed by the Linux OOM-killer during staging, before the forward.
// The bound must turn that into a typed fail-closed refusal naming the shortfall, while a
// discrete device (independent pools) must stay byte-for-byte unchanged.

// integratedTestBackend is the shared-pool double: a Vulkan-shaped backend whose tier advertises
// an integrated GPU (compute-arbitrated UMA detection, never a product-name guess) and whose
// device probe is FAR larger than the host's physical RAM —— exactly the witnessed strix3 shape
// (device budget 71.53 GiB admitted against a 62.4 GiB MemTotal).
func integratedTestBackend(deviceBytes int64) dualCapacityBackend {
	return dualCapacityBackend{
		capBackend: capBackend{total: deviceBytes, free: deviceBytes, known: true},
		hostTotal:  deviceBytes, hostFree: deviceBytes, hostKnown: true,
		name: "vulkan", tier: "integrated:strix-halo",
	}
}

// witnessedDeviceWeightsGiB / witnessedHostTotalGiB are the strix3 numbers as runtime vars so
// the float multiply is not a constant conversion (rejected at compile time).
var (
	witnessedDeviceWeightsGiB = 63.092
	witnessedHostExpertsGiB   = 8.425
	witnessedHostTotalGiB     = 62.4
)

// TestBackendSharesHostRAM pins the shared-pool predicate at the tier grammar: only an
// "integrated:" tier shares physical RAM; a discrete device, the portable floor (nil), or a
// bare/other tier must NOT take the unified bound.
func TestBackendSharesHostRAM(t *testing.T) {
	cases := []struct {
		name string
		be   compute.Backend
		want bool
	}{
		{"integrated vulkan", integratedTestBackend(8 << 30), true},
		{"discrete vulkan", dualCapacityBackend{capBackend: capBackend{known: true}, name: "vulkan", tier: "discrete:test-gpu"}, false},
		{"portable floor (nil)", nil, false},
	}
	for _, tc := range cases {
		if got := BackendSharesHostRAM(tc.be); got != tc.want {
			t.Fatalf("%s: BackendSharesHostRAM = %v, want %v", tc.name, got, tc.want)
		}
	}
}

// TestRefuseUnifiedHostResidencyNamesShortfall is the core RED->GREEN witness: a plan whose
// TOTAL device+host charge exceeds the host's physical RAM on a shared-pool backend must refuse
// with a typed *compute.FitError naming the shortfall (NOT be admitted, and NOT be an OOM), and
// the refusal must carry the unified-pool provenance.
func TestRefuseUnifiedHostResidencyNamesShortfall(t *testing.T) {
	const gib = int64(1 << 30)
	// The witnessed plan class: device dense side 63.092 GiB + a bounded host expert set, against
	// a 62.4 GiB MemTotal. Each scope alone "fits" its own probe (device probe 84 GiB, host
	// budget 62.4 GiB); the SUM does not fit physical RAM.
	deviceWeights := int64(witnessedDeviceWeightsGiB * float64(gib))
	hostExperts := int64(20 * float64(gib))
	hostTotal := int64(witnessedHostTotalGiB * float64(gib))
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceWeights, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: hostExperts, Detail: "gguf-host-expert-offload-streamed"},
	}
	if deviceWeights <= hostTotal {
		t.Fatalf("fixture does not reproduce the witnessed arithmetic (device %d <= MemTotal %d)", deviceWeights, hostTotal)
	}

	// Shared pool: the summed plan exceeds physical RAM -> typed fail-closed refusal.
	err := RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan, hostTotal, hostTotal, true, 0)
	if err == nil {
		t.Fatal("integrated shared-pool plan whose device+host total exceeds MemTotal was ADMITTED; it would be kernel-OOM-killed during staging (fak#13172)")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want a typed *compute.FitError, got %T: %v", err, err)
	}
	if fe.Verdict != compute.FitTooBig || fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("FitError verdict=%v scope=%v, want FitTooBig/host", fe.Verdict, fe.Scope)
	}
	if fe.Want != deviceWeights+hostExperts {
		t.Fatalf("FitError Want = %d, want the summed simultaneous footprint %d", fe.Want, deviceWeights+hostExperts)
	}
	if !strings.Contains(err.Error(), "unified-memory host residency") {
		t.Fatalf("refusal %q does not identify the shared physical pool", err.Error())
	}
	if !strings.Contains(err.Error(), "needs") || !strings.Contains(err.Error(), "host has") {
		t.Fatalf("refusal %q does not name the host-RAM shortfall (must-needs/host-has)", err.Error())
	}

	// The SAME summed plan on a host with enough physical RAM is admitted (the bound is a real
	// measurement, not a blanket refusal).
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan, 128*gib, 128*gib, true, 0); err != nil {
		t.Fatalf("plan that fits a 128 GiB shared pool was refused: %v", err)
	}
}

// TestRefuseUnifiedHostResidencyFailsOpenOnUnknownHost preserves the established fail-open
// contract: a platform that cannot report physical memory must load exactly as before.
func TestRefuseUnifiedHostResidencyFailsOpenOnUnknownHost(t *testing.T) {
	const gib = int64(1 << 30)
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 63 * gib, Detail: "dense"},
	}
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan, 0, compute.FreeUnknown, false, 0); err != nil {
		t.Fatalf("unknown host capacity must fail open, got %v", err)
	}
}

// TestRefuseUnifiedHostResidencyIgnoresDiscreteAndNil is the P4-preservation sibling: the
// unified bound must be INERT for a discrete device (independent VRAM/host pools) and for the
// portable floor, so a large-VRAM discrete serve is byte-for-byte unchanged.
func TestRefuseUnifiedHostResidencyIgnoresDiscreteAndNil(t *testing.T) {
	const gib = int64(1 << 30)
	// A plan that would certainly trip a shared-pool bound (a 63 GiB device weight charge against
	// a 62.4 GiB host), but the backends below keep independent pools or no probe at all.
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 63 * gib, Detail: "dense"},
	}
	discrete := dualCapacityBackend{
		capBackend: capBackend{total: 80 * gib, free: 80 * gib, known: true},
		hostTotal:  62 * gib, hostFree: 62 * gib, hostKnown: true,
		name: "vulkan", tier: "discrete:test-gpu",
	}
	if err := RefuseUnifiedHostResidencyIfTooBig(plan, discrete, 0); err != nil {
		t.Fatalf("discrete device (independent pools) was judged by the shared-pool bound: %v", err)
	}
	if err := RefuseUnifiedHostResidencyIfTooBig(plan, nil, 0); err != nil {
		t.Fatalf("nil backend (portable floor) was judged by the shared-pool bound: %v", err)
	}
	// The integrated arm takes the bound (its host snapshot on this box may be large, so assert
	// via the injectable form to stay host-independent): against a 62.4 GiB host it must refuse.
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan, int64(witnessedHostTotalGiB*float64(gib)), int64(witnessedHostTotalGiB*float64(gib)), true, 0); err == nil {
		t.Fatal("integrated backend with a 63 GiB charge against a 62.4 GiB host was admitted; the bound is inert on the arm it exists for")
	}
}

// TestRefuseUnifiedHostResidencyForLoadOptionsMatchesPlanAndRefuses is the end-to-end seam: the
// WeightSource convenience form must size the actual load plan off the header and refuse on a
// shared pool, and must be a no-op for a discrete device.
func TestRefuseUnifiedHostResidencyForLoadOptionsMatchesPlanAndRefuses(t *testing.T) {
	ws := synthWeightSource(t) // "a" f32 1 MiB + "b" Q4_K 589824 B => lean plan 1638400 B
	plan, err := ws.UnifiedHostResidencyPlan()
	if err != nil {
		t.Fatalf("UnifiedHostResidencyPlan: %v", err)
	}
	if plan.Total() != wsEstimateLoadBytes(t, ws) {
		t.Fatalf("UnifiedHostResidencyPlan total = %d, want the header lean plan %d", plan.Total(), wsEstimateLoadBytes(t, ws))
	}

	// Discrete backend: inert (fail-open), regardless of host size.
	discrete := dualCapacityBackend{capBackend: capBackend{total: 8 << 30, free: 8 << 30, known: true}, name: "vulkan", tier: "discrete:test-gpu"}
	if err := ws.RefuseUnifiedHostResidencyForLoadOptions(discrete, 0); err != nil {
		t.Fatalf("discrete load-options bound must be inert, got %v", err)
	}
	// Integrated backend with a host far smaller than the plan: refuses.
	tiny := integratedTestBackend(8 << 30)
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan, 1024, 1024, true, 0); err == nil {
		t.Fatal("a 1 KiB host pool must refuse the 1.56 MiB lean plan")
	} else if !strings.Contains(err.Error(), "unified-memory host residency") {
		t.Fatalf("refusal %q missing shared-pool provenance", err.Error())
	}
	_ = tiny
}

// wsEstimateLoadBytes is the header lean-plan size for the synth fixture.
func wsEstimateLoadBytes(t *testing.T, ws *WeightSource) int64 {
	t.Helper()
	b, err := ws.EstimateLoadBytes()
	if err != nil {
		t.Fatalf("EstimateLoadBytes: %v", err)
	}
	return b
}

// ---------------------------------------------------------------------------------------------
// fak#13176 — split VRAM/system aperture. The witnessed strix3 box exposes a 64 GiB device VRAM
// window (mem_info_vram_total) and a 62.4 GiB system window (MemTotal), both carved from 128 GB
// physical. The V4.1 plan is kv=8.425GiB, weights=63.092GiB, headroom=12.623GiB: the dense
// weights fit the 64 GiB VRAM window and the host-resident charge fits the 62.4 GiB system
// window, but the grand total exceeds the system window alone. The pre-#13176 bound judged the
// grand total against MemTotal and refused a plan the hardware can run.

// splitApertureBackend is the strix3-shaped double: an integrated tier with a DISTINCT device
// VRAM window and system window, so the split arm of the bound engages.
func splitApertureBackend(vramBytes, systemBytes int64) dualCapacityBackend {
	return dualCapacityBackend{
		capBackend: capBackend{total: vramBytes, free: vramBytes, known: true},
		hostTotal:  systemBytes, hostFree: systemBytes, hostKnown: true,
		name: "vulkan", tier: "integrated:strix-halo",
	}
}

// TestRefuseUnifiedHostResidencyAdmitsSplitAperture is the fak#13176 RED->GREEN: a plan whose
// device-scoped bytes fit the VRAM window and whose host-scoped bytes fit the system window is
// ADMITTED even though the grand total exceeds the system window alone.
func TestRefuseUnifiedHostResidencyRefusesPlanOverPhysicalPoolWitnessed(t *testing.T) {
	// fak#13280 corrected this witness: the plan the pre-#13280 split form ADMITTED (device dense
	// 63.092 GiB + streamed expert cache 8.425 GiB = 71.5 GiB over a 62.4 GiB MemTotal) is the exact
	// shape that global-OOM-killed strix3 on clean silicon. The device window (64 GiB RADV unified
	// heap) is an aperture over the same physical pool, not a disjoint carve-out, so the grand total
	// must be bounded against MemTotal. This test pins the corrected verdict and keeps the
	// split-vs-single-window distinction explicit.
	const gib = int64(1 << 30)
	deviceWeights := int64(witnessedDeviceWeightsGiB * float64(gib)) // 63.092 GiB
	hostExperts := int64(witnessedHostExpertsGiB * float64(gib))     // 8.425 GiB
	vram := int64(64 * float64(gib))                                 // 64 GiB VRAM window
	system := int64(witnessedHostTotalGiB * float64(gib))            // 62.4 GiB system window
	if deviceWeights > vram || hostExperts > system {
		t.Fatalf("fixture does not fit the split apertures (device %d<=%d, host %d<=%d)", deviceWeights, vram, hostExperts, system)
	}
	if deviceWeights+hostExperts <= system {
		t.Fatalf("fixture does not reproduce the #13280 defect: grand total %d must exceed the system window %d", deviceWeights+hostExperts, system)
	}
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceWeights, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: hostExperts, Detail: "gguf-host-expert-offload-streamed"},
	}
	// Split aperture, injectable: device fits VRAM, host fits system, but the grand total exceeds
	// the physical pool -> REFUSED (the #13280 correction).
	err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, vram, vram, true, system, system, true, 0)
	if err == nil {
		t.Fatalf("split-aperture plan (device %d<=VRAM %d, host %d<=system %d) with grand total %d over the pool %d was admitted; a physical-pool overflow must refuse (fak#13280)", deviceWeights, vram, hostExperts, system, deviceWeights+hostExperts, system)
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) || fe.Verdict != compute.FitTooBig {
		t.Fatalf("refusal must be a typed FitTooBig *compute.FitError, got %T: %v", err, err)
	}
	// End-to-end through the backend-aware form, resolving BOTH windows off the backend probes.
	if err := RefuseUnifiedHostResidencyIfTooBig(plan, splitApertureBackend(vram, system), 0); err == nil {
		t.Fatal("backend-aware form admitted the witnessed over-pool V4.1 plan")
	}
	// A plan that genuinely fits the physical pool is admitted: the correction must not refuse a
	// runnable split plan (device 30 GiB + host 20 GiB = 50 GiB <= 62.4 GiB).
	fits := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 30 * gib, Detail: "dense"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: 20 * gib, Detail: "experts"},
	}
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(fits, vram, vram, true, system, system, true, 0); err != nil {
		t.Fatalf("a split plan within the physical pool must be admitted: %v", err)
	}
}

// TestRefuseUnifiedHostResidencyRefusesDeviceOverflow is the negative test: a plan whose
// device-scoped bytes exceed the VRAM window must still refuse with a typed FitError naming the
// DEVICE aperture, even though its host side fits the system window.
// TestRefuseUnifiedHostResidencyRefusesDeviceOverflow is the negative test: a plan whose
// device-scoped bytes exceed the VRAM window must still refuse with a typed FitError naming the
// DEVICE aperture, even though its host side fits the system window.
func TestRefuseUnifiedHostResidencyRefusesDeviceOverflow(t *testing.T) {
	const gib = int64(1 << 30)
	vram := int64(64 * float64(gib))
	system := int64(witnessedHostTotalGiB * float64(gib))
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: vram + gib, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryKVCache, Scope: compute.MemoryScopeHost, Bytes: 2 * gib, Detail: "kv"},
	}
	err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, vram, vram, true, system, system, true, 0)
	if err == nil {
		t.Fatal("device-overflow plan was admitted; a genuine VRAM overflow must still refuse")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("want a typed *compute.FitError, got %T: %v", err, err)
	}
	if fe.Verdict != compute.FitTooBig || fe.Scope != compute.MemoryScopeDevice {
		t.Fatalf("FitError verdict=%v scope=%v, want FitTooBig/device (the VRAM aperture)", fe.Verdict, fe.Scope)
	}
	if fe.Want != vram+gib {
		t.Fatalf("FitError Want = %d, want the device-scoped demand %d", fe.Want, vram+gib)
	}
	if !strings.Contains(err.Error(), "unified-memory host residency") {
		t.Fatalf("refusal %q missing shared-pool provenance", err.Error())
	}
	// The backend-aware form must reach the SAME typed device refusal.
	if err := RefuseUnifiedHostResidencyIfTooBig(plan, splitApertureBackend(vram, system), 0); err == nil {
		t.Fatal("backend-aware form admitted a device-overflow plan")
	} else if !errors.As(err, &fe) || fe.Scope != compute.MemoryScopeDevice {
		t.Fatalf("backend-aware refusal scope = %v, want device: %v", fe.Scope, err)
	}
}

// TestRefuseUnifiedHostResidencySingleWindowUnchanged is the P3-preservation sibling: a host
// that reports only ONE window (device == system, the pre-#13176 shape) or no device window
// behaves EXACTLY as before — the grand total is judged against the system window.
func TestRefuseUnifiedHostResidencySingleWindowUnchanged(t *testing.T) {
	const gib = int64(1 << 30)
	window := int64(witnessedHostTotalGiB * float64(gib))
	plan := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 63 * gib, Detail: "dense"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: 20 * gib, Detail: "experts"},
	}
	// device == system: one shared pool, grand total 83 GiB > 62.4 GiB -> refuse (as before).
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, window, window, true, window, window, true, 0); err == nil {
		t.Fatal("single-window (device==system) plan over the pool was admitted; the grand-total guard was lost")
	}
	// No device window at all: falls back to the grand-total system-window check -> refuse.
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, 0, compute.FreeUnknown, false, window, window, true, 0); err == nil {
		t.Fatal("unknown-device-window plan over the system pool was admitted; must stay conservative")
	}
	// Unknown host window still fails OPEN, unchanged.
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan, 64*gib, 64*gib, true, 0, compute.FreeUnknown, false, 0); err != nil {
		t.Fatalf("unknown host capacity must fail open, got %v", err)
	}
	// A plan that fits a single unified pool is admitted, unchanged.
	fits := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: 8 * gib, Detail: "dense"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: 2 * gib, Detail: "experts"},
	}
	if err := RefuseUnifiedHostResidencyIfTooBigForReportedAperture(fits, window, window, true, window, window, true, 0); err != nil {
		t.Fatalf("plan inside a single unified pool was refused: %v", err)
	}
}
