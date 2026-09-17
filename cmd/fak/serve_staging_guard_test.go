package main

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// serve_staging_guard_test.go — fak#13171 RED->GREEN at the device cpu-offload staging seam. The
// device --cpu-offload-experts plan charges dense weights MemoryScopeDevice (VRAM) and only the
// routed experts MemoryScopeHost, and the load arm's host guard
// (compute.RefuseHostScopedPlanIfTooBigForHost) judged ONLY plan.HostTotal(). But the loader stages
// the device-scoped dense weights THROUGH host RAM (read -> dequant/transcode -> device upload ->
// free), so a plan whose device dense side alone (weights=63.092GiB) exceeds host RAM passed the
// host guard and the process was kernel-OOM-killed mid-staging before the forward. The staging
// guard must charge the device-scoped transit against the same real host budget and refuse with a
// typed FitError naming the shortfall, while an artifact whose staging charge fits is admitted
// unchanged.

// serveStagingGuardPlan is a device cpu-offload plan in the failing shape: a device-scoped dense
// charge (the bytes that transit host RAM during staging) plus a much smaller host-scoped expert
// pool. HostTotal alone fits a modest budget; the staging charge does not.
func serveStagingGuardPlan(deviceDense, hostExperts int64) compute.MemoryPlan {
	return compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceDense, Detail: "gguf-dense-device"},
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: hostExperts, Detail: "gguf-host-expert-offload"},
	}
}

// The staging charge is the device-scoped transit, NOT the host-scoped expert pool — pinning the
// exact quantity the guard bounds so a future refactor cannot silently shrink it back to HostTotal.
func TestServeDeviceStagingHostChargeIsDeviceTransit(t *testing.T) {
	plan := serveStagingGuardPlan(63<<30, 4<<30)
	if got, want := serveDeviceStagingHostCharge(plan), int64(63<<30); got != want {
		t.Fatalf("staging host charge = %d, want the device-scoped transit %d (HostTotal would be the wrong, pre-#13171 quantity)", got, want)
	}
	if plan.HostTotal() != 4<<30 {
		t.Fatalf("fixture HostTotal = %d, want %d", plan.HostTotal(), 4<<30)
	}
	stagingPlan := serveDeviceStagingHostPlan(plan)
	if got, want := stagingPlan.HostTotal(), int64(63<<30); got != want {
		t.Fatalf("staging plan host total = %d, want the device transit %d (the resident expert pool is judged by the pre-existing host guard, not re-charged here)", got, want)
	}
	if len(stagingPlan) != 1 {
		t.Fatalf("staging plan carries %d rows, want exactly the 1 synthesized transit row", len(stagingPlan))
	}
}

// RED->GREEN negative case: a device cpu-offload plan whose device-scoped dense staging charge
// EXCEEDS the host budget must produce a typed refusal, never an unbounded host materialization.
// The 63.09 GiB-vs-62.4 GiB strix3 shape: the device dense side alone is bigger than physical RAM,
// so staging it through host RAM is the witnessed kernel OOM.
func TestServeDeviceStagingGuardRefusesOversizeTransit(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(63*gib, 4*gib)
	// strix3's real host ceiling, below the dense transit.
	hostFit := serveFitBudget{Base: 62 * gib, Headroom: 0}

	err := refuseDeviceStagingAgainstHostFit(plan, hostFit)
	if err == nil {
		t.Fatalf("oversize device staging transit was ADMITTED against a host budget of %d; the serve would be kernel-OOM-killed mid-staging", hostFit.Base)
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("refusal %v is not a typed *compute.FitError; the operator cannot read the shortfall", err)
	}
	if fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("refusal scope = %q, want %q (the bound is host RAM)", fe.Scope, compute.MemoryScopeHost)
	}
	if fe.Want < 63*gib {
		t.Fatalf("refusal Want = %d, want the device staging transit >= %d", fe.Want, int64(63*gib))
	}
	if fe.Avail != hostFit.avail() {
		t.Fatalf("refusal Avail = %d, want the host budget %d", fe.Avail, hostFit.avail())
	}
}

// The preserved case: a device cpu-offload plan whose device dense staging transit FITS the host
// budget is admitted byte-for-byte unchanged, so a device serve with room to stage is never newly
// refused by this guard.
func TestServeDeviceStagingGuardAdmitsFittingTransit(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(32*gib, 4*gib)
	// A 466 GiB serve box: the 32 GiB dense transit stages comfortably.
	hostFit := serveFitBudget{Base: 466 * gib, Headroom: 0}

	if err := refuseDeviceStagingAgainstHostFit(plan, hostFit); err != nil {
		t.Fatalf("fitting device staging transit was refused: %v", err)
	}
}

// An unprobeable host keeps every capacity rung here fail-open: a zero Base admits unchanged.
func TestServeDeviceStagingGuardFailsOpenOnUnprobeableHost(t *testing.T) {
	plan := serveStagingGuardPlan(63<<30, 4<<30)
	if err := refuseDeviceStagingAgainstHostFit(plan, serveFitBudget{}); err != nil {
		t.Fatalf("unprobeable host refused the staging transit: %v", err)
	}
}

// The guard must be inert when the device side carries nothing to stage (host-only or empty plan),
// so the resident host arm and a plan with no device transit are untouched.
func TestServeDeviceStagingGuardInertWithoutDeviceTransit(t *testing.T) {
	hostOnly := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: 8 << 30, Detail: "gguf-host-expert-offload"},
	}
	if got := serveDeviceStagingHostCharge(hostOnly); got != 0 {
		t.Fatalf("host-only plan staging charge = %d, want 0", got)
	}
	if err := refuseDeviceStagingAgainstHostFit(hostOnly, serveFitBudget{Base: 1 << 20, Headroom: 0}); err != nil {
		t.Fatalf("host-only plan (no device transit) was refused by the staging guard: %v", err)
	}
	if err := refuseDeviceStagingAgainstHostFit(nil, serveFitBudget{Base: 1 << 20, Headroom: 0}); err != nil {
		t.Fatalf("empty plan was refused by the staging guard: %v", err)
	}
}

// serve_staging_guard_test.go (fak#13177 + fak#13186) — SPLIT-APERTURE form. On a Strix Halo the box
// exposes a 64 GiB VRAM carve-out SEPARATE from a 62.43 GiB system window. The #13171 staging guard
// judged the device-scoped dense transit against the SYSTEM window alone, so a 63.22 GiB dense side
// that fits the 64 GiB VRAM window was refused - the [HW-WITNESSED] strix3 refusal that blocked the
// first physical V4.1 token. The split-aware form judges the transit's RESIDENCY against the DEVICE
// window when a distinct one is known (fak#13177), and keeps the single-window judgment
// (byte-for-byte) otherwise.
//
// fak#13186 (the rung this leaf closes): the residency admission alone was NOT sufficient, because
// the transit is staged THROUGH host RAM. The [HW-WITNESSED] strix3 rung admitted the 63.22 GiB
// transit on its 64 GiB VRAM fit and was then kernel-OOM-killed as a ~63 GiB host allocation against
// a ~48.5 GiB host budget. The split form therefore requires BOTH windows: the device window for
// RESIDENCY and the host window for the staging TRANSIT.

// A transit that fits BOTH the device window (residency) and the host window (staging transit) is
// admitted unchanged.
func TestServeDeviceStagingSplitApertureAdmitsTransitFittingBothWindows(t *testing.T) {
	const gib = int64(1) << 30
	// 32 GiB dense transit: fits both a 64 GiB VRAM window and a 62.43 GiB system window.
	plan := serveStagingGuardPlan(32*gib, 4*gib)
	hostFit := serveFitBudget{Base: 62*gib + 439<<20, Headroom: 0}
	deviceTotal := int64(64) * gib
	deviceFree := int64(64) * gib

	if err := refuseDeviceStagingAgainstReportedAperture(plan, hostFit, deviceTotal, deviceFree, true); err != nil {
		t.Fatalf("split-aperture transit fitting both windows was refused: %v", err)
	}
}

// fak#13186 RED->GREEN: the exact witnessed strix3 shape - device dense transit 63.22 GiB, device
// window 64 GiB, system window 62.43 GiB. The transit fits the VRAM window it is destined for but
// EXCEEDS the host window it must be STAGED through; it must be refused typed (host scope) instead
// of admitted-then-kernel-OOM-killed mid-staging.
func TestServeDeviceStagingSplitApertureRefusesTransitExceedingHostStagingWindow(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(63*gib+225<<20, 4*gib)
	// hostFit.Base is the system window (62.43 GiB) the staging transit EXCEEDS.
	hostFit := serveFitBudget{Base: 62*gib + 439<<20, Headroom: 0}
	deviceTotal := int64(64) * gib
	deviceFree := int64(64) * gib

	err := refuseDeviceStagingAgainstReportedAperture(plan, hostFit, deviceTotal, deviceFree, true)
	if err == nil {
		t.Fatalf("split-aperture transit (fits 64 GiB VRAM window, exceeds 62.43 GiB host staging window) was ADMITTED; it would be kernel-OOM-killed mid-staging (fak#13186 regressed)")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("refusal %v is not a typed *compute.FitError; the operator cannot read the shortfall", err)
	}
	if fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("refusal scope = %q, want %q (the bound is the host staging transit)", fe.Scope, compute.MemoryScopeHost)
	}
	if fe.Want < 63*gib {
		t.Fatalf("refusal Want = %d, want the device staging transit >= %d", fe.Want, int64(63*gib))
	}
	if fe.Avail != hostFit.avail() {
		t.Fatalf("refusal Avail = %d, want the host budget %d", fe.Avail, hostFit.avail())
	}
}

// fak#13171 preserved: with NO distinct device window (unknown, or coincident with the host window)
// the transit is judged against the system window exactly as before and still refuses.
func TestServeDeviceStagingSplitAperturePreservesSingleWindowRefusal(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(63*gib, 4*gib)
	hostFit := serveFitBudget{Base: 62 * gib, Headroom: 0}

	// Unknown device window -> single-window judgement -> refuse.
	if err := refuseDeviceStagingAgainstReportedAperture(plan, hostFit, 0, compute.FreeUnknown, false); err == nil {
		t.Fatalf("unknown-device-window oversize transit was ADMITTED; the #13171 kernel-OOM protection regressed")
	}
	// Coincident window (device == host, one pool under two names) -> single-window judgement -> refuse.
	if err := refuseDeviceStagingAgainstReportedAperture(plan, hostFit, hostFit.Base, hostFit.Base, true); err == nil {
		t.Fatalf("coincident-aperture oversize transit was ADMITTED; the #13171 kernel-OOM protection regressed")
	}
}

// A genuine overflow of the DEVICE window still refuses typed - the fak#13171 protection is intact
// in the split arm too. Here the transit exceeds BOTH windows.
func TestServeDeviceStagingSplitApertureRefusesGenuineDeviceOverflow(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(80*gib, 4*gib)
	hostFit := serveFitBudget{Base: 62 * gib, Headroom: 0}
	deviceTotal := int64(64) * gib

	err := refuseDeviceStagingAgainstReportedAperture(plan, hostFit, deviceTotal, deviceTotal, true)
	if err == nil {
		t.Fatalf("80 GiB transit against a 64 GiB device window was ADMITTED")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("refusal %v is not a typed *compute.FitError", err)
	}
	if fe.Want < 80*gib {
		t.Fatalf("refusal Want = %d, want the 80 GiB transit", fe.Want)
	}
}

// The split form is fail-open on an unprobeable host and inert without a device transit, exactly
// like the single-window form.
func TestServeDeviceStagingSplitApertureFailsOpenAndInert(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(63*gib, 4*gib)
	if err := refuseDeviceStagingAgainstReportedAperture(plan, serveFitBudget{}, 64*gib, 64*gib, true); err != nil {
		t.Fatalf("unprobeable host refused the split transit: %v", err)
	}
	hostOnly := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeHost, Bytes: 8 << 30, Detail: "gguf-host-expert-offload"},
	}
	if err := refuseDeviceStagingAgainstReportedAperture(hostOnly, serveFitBudget{Base: 62 * gib}, 64*gib, 64*gib, true); err != nil {
		t.Fatalf("host-only plan (no device transit) was refused by the split guard: %v", err)
	}
}

// serveDeviceCeilingBackend is the fake a device serve fit check sees: it reports BOTH the
// volatile DeviceCapacity reading (total, free, known) and the STABLE device-local ceiling
// (DeviceCeiling). It is the test stand-in for the Vulkan backend on a Strix Halo, where the
// device-local heap capacity and the VK_EXT_memory_budget free reading are different numbers.
type serveDeviceCeilingBackend struct {
	compute.Backend
	total, free int64
	known       bool
	ceiling     int64
	ceilingOK   bool
}

func (b serveDeviceCeilingBackend) Caps() compute.Caps {
	return compute.Caps{CapacityProbe: true, DeviceMemory: true}
}

func (b serveDeviceCeilingBackend) DeviceMemory() (int64, int64, bool) {
	return b.total, b.free, b.known
}

func (b serveDeviceCeilingBackend) DeviceLocalCeiling() (int64, bool) {
	return b.ceiling, b.ceilingOK
}

// serve_staging_guard_test.go (fak#13186) - THE STABLE-CEILING FORM. serveDeviceFitBudget sized
// the device fit budget from the VOLATILE VK_EXT_memory_budget free reading, so a 63.22 GiB
// device-destined dense transit was refused at a low reading (65.88 GiB budget x 0.85 = 56.00
// GiB) and admitted at a high one (84.28 GiB-ish x 0.85 = 71.53 GiB) - where it then hard-OOMs
// on the host staging transit. The transit fits the STABLE 84.28 GiB device-local heap in BOTH
// readings, so the budget must be sized from that ceiling.

// The exact failing shape: a device-destined transit that FITS the stable device-local heap
// capacity but EXCEEDS the volatile budget reading must be ADMITTED. RED before the fix (the
// volatile free reading drove the budget), GREEN after.
func TestServeDeviceFitBudgetUsesStableCeilingNotVolatileFree(t *testing.T) {
	const gib = int64(1) << 30
	// strix3: device-local heap 84.28 GiB; volatile budget reading 65.88 GiB (one real reading).
	be := serveDeviceCeilingBackend{
		total: 84*gib + 286<<20, free: 65*gib + 900<<20, known: true,
		ceiling: 84*gib + 286<<20, ceilingOK: true,
	}
	fit := serveDeviceFitBudget(be)
	if fit.Base != be.ceiling {
		t.Fatalf("device fit budget base = %d, want the STABLE device-local ceiling %d (the volatile free reading %d must not size the budget)", fit.Base, be.ceiling, be.free)
	}
	// The 63.22 GiB dense transit must be admitted against the stable capacity with headroom.
	transit := 63*gib + 225<<20
	if avail := fit.avail(); transit > avail {
		t.Fatalf("63.22 GiB device transit (%d B) exceeds the stable-ceiling budget %d B; it must be admitted", transit, avail)
	}
}

// The two-reading divergence is gone: the SAME (plan, ceiling) admits regardless of the volatile
// free value, because the volatile reading no longer sizes the budget.
func TestServeDeviceFitBudgetInvariantAcrossVolatileReadings(t *testing.T) {
	const gib = int64(1) << 30
	ceiling := 84*gib + 286<<20
	base := func(free int64) serveFitBudget {
		return serveDeviceFitBudget(serveDeviceCeilingBackend{
			total: ceiling, free: free, known: true, ceiling: ceiling, ceilingOK: true,
		})
	}
	low, high := base(56*gib), base(71*gib+500<<20)
	if low.Base != high.Base || low.avail() != high.avail() {
		t.Fatalf("budget diverged across volatile readings: low=%d high=%d (avail %d vs %d)", low.Base, high.Base, low.avail(), high.avail())
	}
}

// A genuine overflow of the STABLE device capacity must still refuse with a typed
// *compute.FitError naming the plan - the fak#13171 kernel-OOM protection is intact.
func TestServeDeviceFitBudgetRefusesGenuineCapacityOverflow(t *testing.T) {
	const gib = int64(1) << 30
	be := serveDeviceCeilingBackend{
		total: 24 * gib, free: 24 * gib, known: true,
		ceiling: 24 * gib, ceilingOK: true,
	}
	fit := serveDeviceFitBudget(be)
	plan := serveStagingGuardPlan(63*gib, 4*gib)
	err := compute.RefuseMemoryPlanIfTooBigForReportedDevice(be, plan, fit.Base, fit.Base, fit.Base > 0, fit.Headroom)
	if err == nil {
		t.Fatalf("63 GiB transit against a 24 GiB stable device ceiling was ADMITTED; the fak#13171 kernel-OOM protection regressed")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("refusal %v is not a typed *compute.FitError", err)
	}
	if fe.Scope != compute.MemoryScopeDevice {
		t.Fatalf("refusal scope = %q, want %q", fe.Scope, compute.MemoryScopeDevice)
	}
	if fe.Want < 63*gib {
		t.Fatalf("refusal Want = %d, want the 63 GiB device transit", fe.Want)
	}
}

// Fail-open preserved: a backend with NO stable-ceiling seam falls back byte-for-byte to the
// volatile DeviceMemoryInfo reading, and an unknown-capacity backend keeps the zero base.
func TestServeDeviceFitBudgetFallsBackWithoutCeilingSeam(t *testing.T) {
	const gib = int64(1) << 30
	volatile := serveCapBackend{total: 24 * gib, free: 20 * gib, known: true}
	if got, want := serveDeviceFitBudget(volatile).Base, int64(20*gib); got != want {
		t.Fatalf("non-ceiling backend base = %d, want the volatile free %d (fallback must be byte-for-byte)", got, want)
	}
	if got := serveDeviceFitBudget(serveCapBackend{}).Base; got != 0 {
		t.Fatalf("unknown-capacity backend base = %d, want 0 (fail-open)", got)
	}
	if got := serveDeviceFitBudget(nil).Base; got != 0 {
		t.Fatalf("nil backend base = %d, want 0 (fail-open)", got)
	}
}
