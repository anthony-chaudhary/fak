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

// RefuseUnifiedHostResidencyIfTooBig bounds a load plan's TOTAL simultaneous physical-RAM
// footprint (device-visible bytes PLUS host-resident bytes) against the process host's
// physical RAM when — and only when — the backend shares that RAM (BackendSharesHostRAM).
//
// It reuses compute.RefuseMemoryPlanIfTooBigForReportedHost, the same whole-plan-against-host
// comparison the preflight's unified-memory guard uses, so the serve path and the preflight
// cannot disagree about what the shared pool can hold. The returned *compute.FitError names
// the shortfall ("memory plan ... needs X, host has Y") and carries the classed plan.
//
// Fail-open, exactly like every other capacity rung here: a non-integrated backend, a nil
// backend, or a host that cannot report physical memory yields nil, so the portable floor and
// every discrete device load exactly as before. headroom in [0,1) reserves the same fraction
// of the budget the matching RefuseMemoryPlanIfTooBig* check uses.
func RefuseUnifiedHostResidencyIfTooBig(plan compute.MemoryPlan, be compute.Backend, headroom float64) error {
	if !BackendSharesHostRAM(be) {
		return nil
	}
	// Prefer the backend's OWN host-capacity probe when it advertises one: on an integrated
	// device that probe IS the shared physical pool (Vulkan's HostMemory returns
	// hostSystemMemory, i.e. MemTotal/MemAvailable), and using it lets a caller size and judge
	// the same snapshot. Fall back to the process host probe when the backend does not advertise
	// one, so the bound is total over the backend space.
	total, free, known := compute.HostMemoryInfo(be)
	if !known {
		total, free, known = compute.HostSystemMemoryInfo()
	}
	return refuseUnifiedHostResidencyForReported(plan, total, free, known, headroom)
}

// RefuseUnifiedHostResidencyIfTooBigForReportedHost is the injectable, exported twin of
// RefuseUnifiedHostResidencyIfTooBig: the caller supplies the already-measured (total, free,
// known) host snapshot, so a plan SIZED against one probe is JUDGED against that same probe
// and the derivation and admission cannot drift. It does NOT itself consult the backend tier —
// the caller has already resolved the shared-pool decision.
func RefuseUnifiedHostResidencyIfTooBigForReportedHost(plan compute.MemoryPlan, total, free int64, known bool, headroom float64) error {
	return refuseUnifiedHostResidencyForReported(plan, total, free, known, headroom)
}

// refuseUnifiedHostResidencyForReported is the injectable core: it compares the plan's GRAND
// total (every demand — device- or host-scoped, since on a shared pool both are physical
// DRAM) against the host snapshot via the established reported-host refusal, and wraps the
// typed FitError with the shared-pool provenance so an operator sees WHY a device-scoped
// weight charge was judged against host RAM.
func refuseUnifiedHostResidencyForReported(plan compute.MemoryPlan, total, free int64, known bool, headroom float64) error {
	err := compute.RefuseMemoryPlanIfTooBigForReportedHost(plan, total, free, known, headroom)
	if err == nil {
		return nil
	}
	var fe *compute.FitError
	if errors.As(err, &fe) {
		return fmt.Errorf("integrated unified-memory host residency: %w", err)
	}
	return fmt.Errorf("integrated unified-memory host residency: %w", err)
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
// stays lazy file ranges. With no option (the default) it falls back to the resident
// all-weights load plan, so the helper is total over the load-option space the loader accepts.
func (s *WeightSource) UnifiedHostResidencyPlan(opts ...Q4KLoadOption) (compute.MemoryPlan, error) {
	if s == nil {
		return nil, nil
	}
	o := probeQ4KLoadOptions(opts)
	switch {
	case o.streamedExperts:
		return s.EstimateCPUOffloadExpertsStreamedMemoryPlan(o.streamedExpertBytes)
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
