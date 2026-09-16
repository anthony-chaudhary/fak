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
