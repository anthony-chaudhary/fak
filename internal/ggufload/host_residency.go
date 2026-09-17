package ggufload

import (
	"errors"
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// host_residency.go — the unified-memory host-RAM residency bound for a serve load plan
// (fak#13172). On an INTEGRATED device (a Strix Halo / APU Vulkan tier, "integrated:<device>")
// the device-visible allocation and the host-resident expert/staging charge draw from ONE
// physical DRAM pool. The existing admission rungs check each scope against its OWN probe:
// RefuseMemoryPlanIfTooBig checks device-scoped bytes against the device probe (which on
// RADV/Vulkan reports the unified device-local + host-visible heap, ~84 GiB on a 62.4 GiB
// Halo) and host-scoped bytes against MemAvailable. Neither compares the SIMULTANEOUS SUM to
// physical RAM, so a plan whose device dense side alone (weights=63.092GiB) exceeds MemTotal
// (62.4 GiB) passes both checks and is then SIGKILLed by the Linux OOM-killer during staging,
// before the forward is ever entered (the witnessed d35b541d9 strix3 kill). The preflight
// classifier already closes this hole for its Vulkan-mixed arm
// (refuseIntegratedVulkanUnifiedMemory); this file exposes the SAME measurement to the serve
// load path so the loader emits a typed fail-closed refusal naming the shortfall instead of
// being kernel-OOM-killed.
//
// The bound is deliberately CONSERVATIVE and applies ONLY to a backend whose tier shares
// physical RAM with the host ("integrated:"). A discrete device keeps independent pools and
// must never be judged by this rung, so a discrete GPU with a large VRAM model is unchanged.

// BackendSharesHostRAM reports whether a backend's device memory is drawn from the same
// physical pool as host RAM — an integrated/APU tier ("integrated:<device>"), the tier prefix
// the preflight unified-memory guard keys on. A nil backend, the portable floor, or a discrete
// device returns false, so the host-residency bound stays inert exactly where the pools are
// independent.
func BackendSharesHostRAM(be compute.Backend) bool {
	if be == nil {
		return false
	}
	return strings.HasPrefix(be.Tier(), "integrated:")
}

// RefuseUnifiedHostResidencyIfTooBig bounds a load plan's simultaneous physical-RAM footprint
// when — and only when — the backend shares that RAM (BackendSharesHostRAM).
//
// A split-aperture integrated box (a Strix Halo with a large BIOS/driver VRAM carve-out, e.g.
// strix3: 64 GiB VRAM window + 62.4 GiB system window over 128 GB physical) exposes TWO windows
// over the same DRAM. The plan's DEVICE-scoped bytes live in the VRAM window and its
// HOST-scoped bytes in the system window, and the two together exceed neither window — so
// collapsing them onto the system window alone (fak#13172) refuses a plan the hardware can run
// and blocks the first physical V4.1 token (fak#13176). When both windows are known AND
// distinct, each scope is judged against its OWN window; when the platform reports one window
// (device == system, the pre-#13176 shape) or cannot report the device window, the plan's
// GRAND total is judged against the system window exactly as before — keeping the fail-closed
// OOM protection #13172 landed.
//
// It reuses compute.RefuseMemoryPlanIfTooBigForReportedHost and
// compute.RefuseMemoryPlanIfTooBigForReportedDevice, the same scope-aware comparisons the
// preflight's unified-memory guard uses, so the serve path and the preflight cannot disagree
// about what each aperture can hold. The returned *compute.FitError names the shortfall
// ("memory plan ... needs X, <scope> has Y") and carries the classed plan.
//
// Fail-open, exactly like every other capacity rung here: a non-integrated backend, a nil
// backend, or a host that cannot report physical memory yields nil, so the portable floor and
// every discrete device load exactly as before. headroom in [0,1) reserves the same fraction
// of the budget the matching RefuseMemoryPlanIfTooBig* check uses.
func RefuseUnifiedHostResidencyIfTooBig(plan compute.MemoryPlan, be compute.Backend, headroom float64) error {
	if !BackendSharesHostRAM(be) {
		return nil
	}
	// The device window is the aperture the device-scoped bytes consume (DeviceMemory; on a
	// split box this is mem_info_vram_total, independent of MemTotal). The host window is the
	// system aperture the host-scoped bytes consume. Prefer the backend's OWN host-capacity
	// probe when it advertises one: on an integrated device that probe IS the shared physical
	// pool (Vulkan's HostMemory returns hostSystemMemory, i.e. MemTotal/MemAvailable), and
	// using it lets a caller size and judge the same snapshot. Fall back to the process host
	// probe when the backend does not advertise one, so the bound is total over the backend
	// space.
	devTotal, devFree, devKnown := compute.DeviceMemoryInfo(be)
	hostTotal, hostFree, hostKnown := compute.HostMemoryInfo(be)
	if !hostKnown {
		hostTotal, hostFree, hostKnown = compute.HostSystemMemoryInfo()
	}
	return refuseUnifiedHostResidencyForReportedAperture(plan, devTotal, devFree, devKnown, hostTotal, hostFree, hostKnown, headroom)
}

// RefuseUnifiedHostResidencyIfTooBigForReportedHost is the injectable, exported twin of
// RefuseUnifiedHostResidencyIfTooBig: the caller supplies the already-measured (total, free,
// known) host snapshot, so a plan SIZED against one probe is JUDGED against that same probe
// and the derivation and admission cannot drift. It does NOT itself consult the backend tier —
// the caller has already resolved the shared-pool decision.
//
// It supplies NO device window, so it is the single-window form: the plan's GRAND total is
// judged against the system window exactly as before fak#13176. A caller that holds a distinct
// device aperture uses RefuseUnifiedHostResidencyIfTooBigForReportedAperture.
func RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan compute.MemoryPlan, total, free int64, known bool, headroom float64) error {
	return refuseUnifiedHostResidencyForReported(plan, total, free, known, headroom)
}

// RefuseUnifiedHostResidencyIfTooBigForReportedAperture is the injectable, exported split-aware
// twin: the caller supplies BOTH the device window (deviceTotal, deviceFree, deviceKnown) and
// the host/system window (hostTotal, hostFree, hostKnown). When both windows are known AND
// distinct it bounds the plan's device-scoped bytes against the device window and its
// host-scoped bytes against the system window; otherwise it falls back to the single-window
// grand-total check against the system window. Keeping the two windows explicit makes the
// admission testable without a live GPU or /proc, and keeps a plan SIZED against an aperture
// provably JUDGED against the same aperture.
func RefuseUnifiedHostResidencyIfTooBigForReportedAperture(plan compute.MemoryPlan, deviceTotal, deviceFree int64, deviceKnown bool, hostTotal, hostFree int64, hostKnown bool, headroom float64) error {
	return refuseUnifiedHostResidencyForReportedAperture(plan, deviceTotal, deviceFree, deviceKnown, hostTotal, hostFree, hostKnown, headroom)
}

// refuseUnifiedHostResidencyForReported is the injectable single-window core: it compares the
// plan's GRAND total (every demand — device- or host-scoped, since on one shared pool both are
// physical DRAM) against the host snapshot via the established reported-host refusal, and wraps
// the typed FitError with the shared-pool provenance so an operator sees WHY a device-scoped
// weight charge was judged against host RAM.
func refuseUnifiedHostResidencyForReported(plan compute.MemoryPlan, total, free int64, known bool, headroom float64) error {
	return wrapUnifiedHostResidency(compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, total, free, known, headroom))
}

// refuseUnifiedHostResidencyForReportedAperture is the split-aware core. When the host window is
// unknown the bound fails OPEN (nil), matching every other capacity rung. When a device window
// is known and DIFFERS from the system window the box exposes a genuine split aperture, so each
// scope is checked against its own window; the device check runs first and its typed refusal
// names the device aperture, the host check second and names the system aperture. When no
// device window is known, or the two windows coincide (one pool reported under two names), the
// plan's GRAND total is judged against the system window — the pre-#13176 behavior, so the
// kernel-OOM protection #13172 added is preserved exactly.
//
// The physical-total guard the issue names is the composition of these two checks in the split
// arm: because the device window and the system window are disjoint CARVE-OUTS of the same
// physical DRAM, a plan that fits each window separately cannot exceed the physical total
// without exceeding one of them. No separate physical-total probe exists in compute (adding one
// is out of scope per the issue), so the per-window checks are the conservative physical bound.
func refuseUnifiedHostResidencyForReportedAperture(plan compute.MemoryPlan, deviceTotal, deviceFree int64, deviceKnown bool, hostTotal, hostFree int64, hostKnown bool, headroom float64) error {
	if !hostKnown {
		return nil
	}
	if deviceKnown && deviceTotal > 0 && deviceTotal != hostTotal {
		// Split aperture: device-scoped bytes consume the VRAM window, host-scoped bytes the
		// system window. Each is judged against its OWN scope via the scope-aware refusals:
		// RefuseMemoryPlanIfTooBigForReportedDevice bounds plan.DeviceTotal() against the device
		// window, and the host half bounds plan.HostTotal() against the system window — the
		// host-scoped SUBSET, never the grand total, which is exactly what the split form exists
		// to stop collapsing onto one window.
		if err := compute.RefuseMemoryPlanIfTooBigForReportedDevice(nil, plan, deviceTotal, deviceFree, true, headroom); err != nil {
			return wrapUnifiedHostResidency(err)
		}
		return wrapUnifiedHostResidency(compute.RefuseMemoryPlanIfTooBigForReportedHost(hostScopedPlan(plan), hostTotal, hostFree, true, headroom))
	}
	return wrapUnifiedHostResidency(compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, hostTotal, hostFree, hostKnown, headroom))
}

// wrapUnifiedHostResidency adds the shared-pool provenance to a typed refusal so an operator
// sees WHY a scoped charge was judged against a shared aperture, and passes a nil error
// through unchanged.
func wrapUnifiedHostResidency(err error) error {
	if err == nil {
		return nil
	}
	var fe *compute.FitError
	if errors.As(err, &fe) {
		return fmt.Errorf("integrated unified-memory host residency: %w", err)
	}
	return fmt.Errorf("integrated unified-memory host residency: %w", err)
}

// hostScopedPlan returns ONLY the plan's host-scoped demands, so a host-window check can judge
// the host subset without re-summing the device-scoped bytes that belong to the VRAM aperture.
// It is the split form's host half: on a shared pool the device dense load is not anonymous host
// RAM and must not be charged against the system window a second time. A plan with no
// host-scoped demand yields an empty plan, whose total is zero (admitted against any window).
func hostScopedPlan(plan compute.MemoryPlan) compute.MemoryPlan {
	host := make(compute.MemoryPlan, 0, len(plan))
	for _, d := range plan {
		if d.Bytes > 0 && d.ScopeOrDefault() == compute.MemoryScopeHost {
			host = append(host, d)
		}
	}
	return host
}

// RefuseUnifiedHostResidencyForLoadOptions is the header-only convenience form: it sizes the
// plan for the given load options off the parsed tensor directory (no payload read) and bounds
// it against the shared physical pool. It is the single call a serve load path makes before
// materializing, so the host-residency decision is shared with the plan the loader will
// actually build. A non-shared backend (discrete device, portable floor) is a no-op.
func (s *WeightSource) RefuseUnifiedHostResidencyForLoadOptions(be compute.Backend, headroom float64, opts ...Q4KLoadOption) error {
	if !BackendSharesHostRAM(be) || s == nil {
		return nil
	}
	plan, err := s.UnifiedHostResidencyPlan(opts...)
	if err != nil {
		return err
	}
	return RefuseUnifiedHostResidencyIfTooBig(plan, be, headroom)
}

// UnifiedHostResidencyPlan sizes the plan whose TOTAL simultaneous physical footprint the
// unified-memory bound judges. With WithStreamedExperts the routed expert bulk is charged at
// its bounded resident set (never the full payload); with WithStreamedDenseQ4K the dense side
// stays lazy file ranges, and with the BOUNDED WithStreamedDenseQ4KWorkingSet it is charged at
// that declared host working set rather than the full dense side (fak#13194). With no option
// (the default) it falls back to the resident
// all-weights load plan, so the helper is total over the load-option space the loader accepts.
func (s *WeightSource) UnifiedHostResidencyPlan(opts ...Q4KLoadOption) (compute.MemoryPlan, error) {
	if s == nil {
		return nil, nil
	}
	o := probeQ4KLoadOptions(opts)
	switch {
	case o.streamedExperts:
		return s.EstimateCPUOffloadExpertsStreamedMemoryPlan(o.streamedExpertBytes)
	case o.streamedDenseBounded:
		// Bounded streamed dense: the eligible dense side is charged at its declared host
		// working set (fak#13194), so the residency bound judges what the loader retains rather
		// than the full on-disk dense side. Only a genuinely unbounded request (no working set
		// declared) still falls back to the raw full-charge plan, preserving fail-closed.
		plan, err := s.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4KWorkingSet(o.streamedDenseBytes))
		if errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			return s.EstimateLoadMemoryPlan()
		}
		return plan, err
	case o.streamedDenseQ4K:
		plan, err := s.EstimateQ4KLoadMemoryPlan(WithStreamedDenseQ4K(true))
		if errors.Is(err, ErrQ4KLoadEstimateUnsupported) {
			return s.EstimateLoadMemoryPlan()
		}
		return plan, err
	default:
		return s.EstimateLoadMemoryPlan()
	}
}
