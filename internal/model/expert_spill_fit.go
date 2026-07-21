package model

import "fmt"

// expert_spill_fit.go — graded expert-spill sizing (the llama.cpp `--n-cpu-moe N` +
// auto-fit equivalent), #5281.
//
// The runtime split (splitKernel, moe_offload.go) already routes each weight host-or-device by
// a predicate, and isExpertWeight is the all-or-nothing default: either every MoE layer's
// experts go to host RAM or none do. What was missing is the GRADED dial between the two
// endpoints — spill only the first N layers' experts to host — and the auto-search that PICKS N
// so the device-resident remainder fits a measured byte budget instead of refusing when neither
// endpoint fits.
//
// This file is that sizing math, kept PURE: it takes a byte budget, the per-layer expert byte
// cost, the always-device dense base cost, and the MoE layer count, and computes the number of
// expert layers to spill. No GPU, no probe, no clock — a weight's home is decided elsewhere
// (isExpertWeight / placementOverride); here we only choose HOW MANY layers to move so the
// choice is deterministic and unit-testable. The spill honors llama.cpp's `--n-cpu-moe`
// semantics: N counts the FIRST N layers whose experts move to host, so the device keeps the
// last (MoELayers-N) layers' experts resident.

// ExpertSpillBudget is the pure input to the sizing math: the device byte budget and the three
// byte quantities that determine device residency at any spill count. All bytes are non-negative
// counts; a negative field is a miscomputed input and is refused (fail-closed) rather than sized.
type ExpertSpillBudget struct {
	// MoELayers is the number of MoE layers whose experts CAN be spilled to host RAM. Spilling N
	// moves the first N of these; the last (MoELayers-N) stay device-resident.
	MoELayers int
	// ExpertBytesPerLayer is the device byte cost of one layer's expert weights — the amount each
	// spilled layer removes from device residency. Zero means spilling frees no device bytes.
	ExpertBytesPerLayer int64
	// DeviceBaseBytes is the byte cost that ALWAYS stays device-resident regardless of the spill
	// count: the dense projections, attention, router, and LM head that every token touches.
	DeviceBaseBytes int64
	// BudgetBytes is the measured device byte budget the resident remainder must fit within.
	BudgetBytes int64
}

// ExpertSpillFit is the sized result: how many layers to spill and the byte residency that spill
// induces. Fits reports whether the device-resident remainder is within the budget; a fail-closed
// auto-fit that cannot make the model fit (the dense base alone exceeds the budget) returns the
// maximal spill with Fits=false rather than a negative or over-count spill.
type ExpertSpillFit struct {
	// SpillLayers is N — the number of first-layer expert sets moved to host RAM. Always in
	// [0, MoELayers].
	SpillLayers int
	// HostSpillBytes is the byte volume moved to host RAM by this spill (SpillLayers * per-layer).
	HostSpillBytes int64
	// DeviceResidentBytes is the byte cost that stays on the device after the spill: the dense
	// base plus the (MoELayers-SpillLayers) resident expert layers.
	DeviceResidentBytes int64
	// Fits is true when DeviceResidentBytes <= BudgetBytes.
	Fits bool
}

// ExpertSpillRangeError is the typed, fail-closed error ResolveExpertSpill returns for an explicit
// user N outside [0, MoELayers]. Callers can match on it to distinguish an out-of-range operator
// input from a malformed budget; the resolve NEVER silently clamps, so a typo cannot degrade into
// an unintended residency.
type ExpertSpillRangeError struct {
	N      int
	Layers int
}

func (e *ExpertSpillRangeError) Error() string {
	return fmt.Sprintf("model: expert spill N = %d out of range [0, %d]", e.N, e.Layers)
}

// validate refuses a budget with any negative field: bytes and layer counts are counts, so a
// negative is a miscomputed input, not a config to size against.
func (b ExpertSpillBudget) validate() error {
	if b.MoELayers < 0 {
		return fmt.Errorf("model: ExpertSpillBudget MoELayers = %d, want >= 0", b.MoELayers)
	}
	if b.ExpertBytesPerLayer < 0 {
		return fmt.Errorf("model: ExpertSpillBudget ExpertBytesPerLayer = %d, want >= 0", b.ExpertBytesPerLayer)
	}
	if b.DeviceBaseBytes < 0 {
		return fmt.Errorf("model: ExpertSpillBudget DeviceBaseBytes = %d, want >= 0", b.DeviceBaseBytes)
	}
	if b.BudgetBytes < 0 {
		return fmt.Errorf("model: ExpertSpillBudget BudgetBytes = %d, want >= 0", b.BudgetBytes)
	}
	return nil
}

// fitAt is the pure residency at a given spill count n. It clamps n to [0, MoELayers] so a
// caller can never induce a negative or over-count residency; SpillLayers reflects the clamped n.
func (b ExpertSpillBudget) fitAt(n int) ExpertSpillFit {
	if n < 0 {
		n = 0
	}
	if n > b.MoELayers {
		n = b.MoELayers
	}
	residentLayers := b.MoELayers - n
	deviceBytes := b.DeviceBaseBytes + int64(residentLayers)*b.ExpertBytesPerLayer
	return ExpertSpillFit{
		SpillLayers:         n,
		HostSpillBytes:      int64(n) * b.ExpertBytesPerLayer,
		DeviceResidentBytes: deviceBytes,
		Fits:                deviceBytes <= b.BudgetBytes,
	}
}

// AutoFitExpertSpill picks the MINIMAL spill count N so the device-resident remainder fits the
// budget — the graded auto-search between the two endpoints:
//
//   - budget fits everything (base + all expert layers <= budget) -> N = 0 (nothing spilled).
//   - budget fits nothing (base alone > budget, or spilling frees no bytes yet base > budget) ->
//     N = MoELayers with Fits=false (fail-closed: spill everything we can, never over-count).
//   - partial -> the least N whose resident remainder is within budget, honoring the first-N-layers
//     spill semantics of `--n-cpu-moe`.
//
// It is monotone: DeviceResidentBytes strictly decreases as N grows (when per-layer > 0), so the
// ceiling-division N is exactly the smallest spill that fits. Deterministic, no device alloc.
func AutoFitExpertSpill(b ExpertSpillBudget) (ExpertSpillFit, error) {
	if err := b.validate(); err != nil {
		return ExpertSpillFit{}, err
	}
	full := b.fitAt(0)
	if full.Fits {
		return full, nil // everything already fits; spill nothing.
	}
	if b.ExpertBytesPerLayer == 0 {
		// Spilling frees no device bytes, so no N can improve the fit; spill all layers and report
		// the honest Fits=false (the dense base alone exceeds the budget).
		return b.fitAt(b.MoELayers), nil
	}
	// Smallest N with DeviceBaseBytes + (MoELayers-N)*perLayer <= budget, i.e.
	// N >= (resident(0) - budget) / perLayer, rounded up. Clamped to MoELayers when even the full
	// spill cannot fit the dense base.
	need := full.DeviceResidentBytes - b.BudgetBytes
	n := int((need + b.ExpertBytesPerLayer - 1) / b.ExpertBytesPerLayer)
	if n > b.MoELayers {
		n = b.MoELayers
	}
	return b.fitAt(n), nil
}

// ResolveExpertSpill sizes residency for an EXPLICIT user-supplied spill count (the operator
// typed `--n-cpu-moe N`). N is validated against [0, MoELayers]; an out-of-range N is refused
// with a typed *ExpertSpillRangeError rather than silently clamped, so an operator's typo cannot
// become an unintended placement. A valid N is honored exactly — the auto-search is bypassed —
// and the returned Fits reports whether the operator's choice actually fits the budget.
func ResolveExpertSpill(b ExpertSpillBudget, userN int) (ExpertSpillFit, error) {
	if err := b.validate(); err != nil {
		return ExpertSpillFit{}, err
	}
	if userN < 0 || userN > b.MoELayers {
		return ExpertSpillFit{}, &ExpertSpillRangeError{N: userN, Layers: b.MoELayers}
	}
	return b.fitAt(userN), nil
}
