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

// serve_staging_guard_test.go (fak#13209) - the BOUNDED streamed-dense transit form. The device
// --cpu-offload-experts plan now carries a "gguf-host-dense-streamed" host working-set row
// (ggufload.EstimateCPUOffloadExpertsBoundedDenseMemoryPlan) so the eligible dense k-quant side is
// faulted as bounded checkpoint ranges instead of materialized as one ~63 GiB host anon buffer.
// serveDeviceStagingHostPlan must charge THAT bounded working set -- not the whole plan.DeviceTotal()
// -- and stamp the distinct detail so a physical receipt can tell the two apart. Without the row the
// historical whole-DeviceTotal charge and the plain detail are preserved byte-for-byte.

// serveBoundedDenseStagingPlan is the bounded-transit shape: a device-scoped dense charge (the
// eligible dense side the streamed route covers) plus the bounded host working-set row the streamed
// fold appends, plus a host-scoped routed-expert pool. The bounded row is the observable evidence
// that the bounded route is active.
func serveBoundedDenseStagingPlan(deviceDense, hostExperts, denseBound int64) compute.MemoryPlan {
	return compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceDense, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: hostExperts, Detail: "gguf-host-expert-offload"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: denseBound, Detail: serveDenseBoundedStreamStagingDetail},
	}
}

// With the bounded working-set row present the transit is the BOUNDED dense charge, strictly below
// plan.DeviceTotal(), and carries the bounded detail.
func TestServeDeviceStagingHostPlanBoundsStreamedDenseTransit(t *testing.T) {
	const gib = int64(1) << 30
	// 63.22 GiB device dense side, bounded to a 12 GiB host working set.
	plan := serveBoundedDenseStagingPlan(63*gib+225<<20, 4*gib, 12*gib)

	if !serveDeviceDenseStreamedBounded(plan) {
		t.Fatalf("fixture does not carry the bounded streamed-dense row")
	}
	stagingPlan := serveDeviceStagingHostPlan(plan)
	if len(stagingPlan) != 1 {
		t.Fatalf("staging plan carries %d rows, want exactly the 1 synthesized transit row", len(stagingPlan))
	}
	got := stagingPlan[0]
	if got.Detail != serveDenseStreamedStagingChargeDetail {
		t.Fatalf("transit detail = %q, want %q", got.Detail, serveDenseStreamedStagingChargeDetail)
	}
	if got.Bytes >= plan.DeviceTotal() {
		t.Fatalf("bounded transit = %d, want STRICTLY less than DeviceTotal %d", got.Bytes, plan.DeviceTotal())
	}
	if got.Bytes != 12*gib {
		t.Fatalf("bounded transit = %d, want the declared working set %d", got.Bytes, int64(12*gib))
	}
}

// WITHOUT the bounded row the historical whole-DeviceTotal charge and the plain detail are
// preserved byte-for-byte: the fak#13171 kernel-OOM protection is untouched on the non-streamed shape.
func TestServeDeviceStagingHostPlanPreservesWholeTotalWithoutBoundedRow(t *testing.T) {
	const gib = int64(1) << 30
	plan := serveStagingGuardPlan(63*gib+225<<20, 4*gib)

	stagingPlan := serveDeviceStagingHostPlan(plan)
	if len(stagingPlan) != 1 {
		t.Fatalf("staging plan carries %d rows, want exactly the 1 synthesized transit row", len(stagingPlan))
	}
	got := stagingPlan[0]
	if got.Detail != "gguf-device-staging-host-transit" {
		t.Fatalf("transit detail = %q, want the historical plain detail", got.Detail)
	}
	if got.Bytes != plan.DeviceTotal() {
		t.Fatalf("transit bytes = %d, want the whole DeviceTotal %d byte-for-byte", got.Bytes, plan.DeviceTotal())
	}
}

// A bounded transit that fits the host window is admitted; the same plan WITHOUT the row refuses.
// This is the fak#13209 witness: the bound is what lets the arm reach the load instead of refusing.
func TestServeDeviceStagingBoundedTransitAdmitsWhereWholeTotalRefuses(t *testing.T) {
	const gib = int64(1) << 30
	hostFit := serveFitBudget{Base: 48*gib + 542<<20, Headroom: 0} // ~48.5 GiB host budget

	bounded := serveBoundedDenseStagingPlan(63*gib+225<<20, 4*gib, 12*gib)
	if err := refuseDeviceStagingAgainstHostFit(bounded, hostFit); err != nil {
		t.Fatalf("bounded 12 GiB dense transit against a 48.5 GiB host budget was refused: %v", err)
	}

	unbounded := serveStagingGuardPlan(63*gib+225<<20, 4*gib)
	if err := refuseDeviceStagingAgainstHostFit(unbounded, hostFit); err == nil {
		t.Fatalf("unbounded 63.22 GiB dense transit against a 48.5 GiB host budget was ADMITTED; the fak#13171 protection regressed")
	}
}

// fak#13209 threading seam: the device --cpu-offload-experts arm's option list must carry the
// bounded streamed-dense working set when the env knob is set, derived from the HOST budget. These
// helpers are the NEW plumbing this leaf adds, so stashing serve_model_fit.go makes this test fail
// to build (the RED witness for the threading half).
func TestServeCPUOffloadBoundedDenseOptionsThreadsHostBound(t *testing.T) {
	const gib = int64(1) << 30
	t.Setenv("FAK_STREAM_Q4K", "1")
	// be=nil makes serveStreamedHostFit take the injected fit verbatim (the device-less form),
	// so the bound is a pure function of the host budget under test.
	hostFit := serveFitBudget{Base: 48*gib + 542<<20, Headroom: 0}
	opts := serveCPUOffloadBoundedDenseOptions(nil, hostFit)
	bound, ok := serveBoundedDenseWorkingSetBound(opts)
	if !ok {
		t.Fatalf("option list does not declare the bounded dense working set: %+v", opts)
	}
	want := serveStreamedDenseQ4KWorkingSetBound(hostFit)
	if bound != want {
		t.Fatalf("threaded bound = %d, want the host-derived working set %d", bound, want)
	}
	if bound >= hostFit.avail() {
		t.Fatalf("threaded bound = %d, want STRICTLY below the host budget %d", bound, hostFit.avail())
	}
	// With the knob unset the list is empty, so a non-streamed device cpu-offload serve is
	// byte-identical.
	t.Setenv("FAK_STREAM_Q4K", "0")
	t.Setenv("FAK_METAL_STREAM_Q4K", "0")
	if opts := serveCPUOffloadBoundedDenseOptions(nil, hostFit); len(opts) != 0 {
		t.Fatalf("unset knob produced a non-empty option list: %+v", opts)
	}
	if _, ok := serveBoundedDenseWorkingSetBound(nil); ok {
		t.Fatalf("nil option list reported a bounded dense working set")
	}
}

// serve_staging_guard_test.go (fak#13215) - the COMBINED streamed-expert + bounded-dense transit. The
// DeepSeek-V4.1 Q2_K arm on strix3 runs the STREAMED expert policy (serveStreamedCPUOffloadPlanForAperture
// -> EstimateCPUOffloadExpertsStreamedMemoryPlan, denseResident = -1), which charged the WHOLE 63.22 GiB
// device dense side as the host staging transit and refused typed against the 48.76 GiB host window --
// the exact refusal this leaf removes. The combined estimator threads the SAME bounded dense working set
// the device arm already uses (serveCPUOffloadBoundedDenseOptions), so the streamed plan now carries the
// "gguf-host-dense-streamed" row and the staging guard charges the bounded transit.

// serveCombinedStreamedBoundedDenseStagingPlan is the streamed+bounded shape: the device-scoped dense
// charge the streamed route covers, the bounded host dense working-set row, and the bounded host routed
// expert row the streamed fold emits. The dense row is the observable evidence the combined route is
// active.
func serveCombinedStreamedBoundedDenseStagingPlan(deviceDense, routedResident, denseBound int64) compute.MemoryPlan {
	return compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceDense, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: routedResident, Detail: "gguf-host-expert-offload-streamed"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: denseBound, Detail: serveDenseBoundedStreamStagingDetail},
	}
}

// TestServeDeviceStagingStreamedBoundedTransitAdmitsWhereWholeTotalRefuses is the RED->GREEN witness at
// the staging seam: the combined streamed plan (bounded dense row present) ADMITS a 63.22 GiB device
// dense transit against a 48.76 GiB host window, whereas the historical streamed plan WITHOUT the row
// (denseResident = -1, the fak#13215 refusal) still refuses typed. The charge is the bounded dense
// transit (min(denseEligible, bound) + non-dense remainder), never plan.DeviceTotal().
func TestServeDeviceStagingStreamedBoundedTransitAdmitsWhereWholeTotalRefuses(t *testing.T) {
	const gib = int64(1) << 30
	// strix3's system window: the host budget the staging transit must fit.
	hostFit := serveFitBudget{Base: 48*gib + 778<<20, Headroom: 0}
	deviceDense := 63*gib + 225<<20 // 63.22 GiB device dense side
	routedResident := 4 * gib
	denseBound := 12 * gib

	combined := serveCombinedStreamedBoundedDenseStagingPlan(deviceDense, routedResident, denseBound)
	if !serveDeviceDenseStreamedBounded(combined) {
		t.Fatalf("combined fixture does not carry the bounded streamed-dense row")
	}
	stagingPlan := serveDeviceStagingHostPlan(combined)
	if len(stagingPlan) != 1 {
		t.Fatalf("combined staging plan carries %d rows, want exactly the 1 synthesized transit row", len(stagingPlan))
	}
	if got := stagingPlan[0].Detail; got != serveDenseStreamedStagingChargeDetail {
		t.Fatalf("combined transit detail = %q, want %q", got, serveDenseStreamedStagingChargeDetail)
	}
	if got := stagingPlan[0].Bytes; got != denseBound {
		t.Fatalf("combined transit = %d, want the bounded dense working set %d (not the whole DeviceTotal %d)", got, denseBound, combined.DeviceTotal())
	}
	if err := refuseDeviceStagingAgainstReportedAperture(combined, hostFit, deviceDense, deviceDense, true); err != nil {
		t.Fatalf("combined bounded streamed transit against a 48.76 GiB host window was refused: %v", err)
	}

	// WITHOUT the bounded dense row the historical streamed plan charges the whole DeviceTotal and
	// still refuses typed (the fak#13171 kernel-OOM protection, and the fak#13215 refusal itself).
	legacy := compute.MemoryPlan{
		{Class: compute.MemoryWeights, Scope: compute.MemoryScopeDevice, Bytes: deviceDense, Detail: "gguf-device-dense-load"},
		{Class: compute.MemoryOffload, Scope: compute.MemoryScopeHost, Bytes: routedResident, Detail: "gguf-host-expert-offload-streamed"},
	}
	if serveDeviceDenseStreamedBounded(legacy) {
		t.Fatalf("legacy fixture unexpectedly carries the bounded dense row")
	}
	if got := serveDeviceStagingHostPlan(legacy)[0].Bytes; got != legacy.DeviceTotal() {
		t.Fatalf("legacy transit = %d, want the whole DeviceTotal %d byte-for-byte", got, legacy.DeviceTotal())
	}
	err := refuseDeviceStagingAgainstReportedAperture(legacy, hostFit, deviceDense, deviceDense, true)
	if err == nil {
		t.Fatalf("legacy streamed transit (no bounded dense row) against a 48.76 GiB host window was ADMITTED; the fak#13215 premise is broken")
	}
	var fe *compute.FitError
	if !errors.As(err, &fe) {
		t.Fatalf("legacy refusal %v is not a typed *compute.FitError", err)
	}
	if fe.Scope != compute.MemoryScopeHost {
		t.Fatalf("legacy refusal scope = %q, want %q", fe.Scope, compute.MemoryScopeHost)
	}
}

// serve_staging_guard_test.go (fak#13247) - the COMBINED host working-set sum. The streamed-expert
// resident bound (gguf-host-expert-offload-streamed) and the bounded-dense staging working set
// (gguf-host-dense-streamed) are BOTH simultaneously resident in host RAM. Deriving each independently
// as (1-margin)*avail admitted their SUM at up to ~1.8*avail: the [HW-WITNESSED] strix3 run had a
// 48.65 GiB host budget, a 43.788 GiB expert bound AND a 43.788 GiB dense bound in one serve, and the
// kernel OOM-killed it at 57.8G peak / 28.1G swap before the forward. serveCombinedStreamedDenseResidentBound
// must size the dense bound from the budget REMAINING after the expert bound, so the two together land
// strictly below the budget.

// The combined dense bound is the post-expert remainder less the resident margin, STRICTLY below the
// remainder, and the two working sets sum strictly below the host budget.
func TestServeCombinedStreamedDenseResidentBoundSumsBelowBudget(t *testing.T) {
	const gib = int64(1) << 30
	hostFit := serveFitBudget{Base: 48*gib + 654<<20, Headroom: 0} // strix3 48.654 GiB host budget
	expertBound := serveCPUOffloadStreamedResidentBound(hostFit)

	// The historical independent derivation: both at (1-margin)*avail, summing ABOVE the budget.
	if expertBound+expertBound <= hostFit.avail() {
		t.Fatalf("fixture premise broken: independent bounds %d + %d do not exceed avail %d", expertBound, expertBound, hostFit.avail())
	}

	combined := serveCombinedStreamedDenseResidentBound(hostFit, expertBound)
	if combined >= expertBound {
		t.Fatalf("combined dense bound = %d, want STRICTLY below the independent bound %d", combined, expertBound)
	}
	if combined < 0 {
		t.Fatalf("combined dense bound = %d, want non-negative", combined)
	}
	if expertBound+combined >= hostFit.avail() {
		t.Fatalf("expert %d + combined dense %d = %d, want STRICTLY below avail %d", expertBound, combined, expertBound+combined, hostFit.avail())
	}

	// A non-positive remainder (expert bound at or above avail) yields stream-through, never negative.
	if got := serveCombinedStreamedDenseResidentBound(hostFit, hostFit.avail()); got != 0 {
		t.Fatalf("expert bound at avail -> dense bound = %d, want 0 (stream-through)", got)
	}
	if got := serveCombinedStreamedDenseResidentBound(hostFit, hostFit.avail()+gib); got != 0 {
		t.Fatalf("expert bound above avail -> dense bound = %d, want 0 (stream-through)", got)
	}
	// An unprobeable host yields zero, exactly as every other bound here.
	if got := serveCombinedStreamedDenseResidentBound(serveFitBudget{}, expertBound); got != 0 {
		t.Fatalf("unprobeable host -> dense bound = %d, want 0", got)
	}
	// A negative expert bound clamps to zero rather than inflating the remainder.
	if got := serveCombinedStreamedDenseResidentBound(hostFit, -gib); got != combineWantDenseBound(hostFit, 0) {
		t.Fatalf("negative expert bound -> dense bound = %d, want the zero-expert form", got)
	}
}

// combineWantDenseBound mirrors the remainder arithmetic for the negative-clamp assertion.
func combineWantDenseBound(fit serveFitBudget, expertBound int64) int64 {
	remaining := fit.avail() - expertBound
	if remaining <= 0 {
		return 0
	}
	bound := int64(float64(remaining) * (1 - serveCPUOffloadStreamedResidentMargin))
	if bound >= remaining {
		bound = remaining - 1
	}
	if bound < 0 {
		return 0
	}
	return bound
}
