package model

import "github.com/anthony-chaudhary/fak/internal/cacheprice"

// prefill_length_crossover.go — length-driven prefill compute-placement (the ktransformers
// `--kt-gpu-prefill-token-threshold` equivalent), #5279.
//
// splitKernel (moe_offload.go) routes each expert GEMM host-or-device by a static, name-only
// predicate: the home is fixed for the whole run and never varies by how long the current prefill
// is. That fixed choice is a lose-lose at the extremes — always host-hybrid is slow on a long
// prompt (CPU MoE bandwidth caps prefill), while always streaming cold experts to the device
// costs one MoE layer of extra VRAM even on a short prompt that never needed it.
//
// This file is the length-driven switch that ktransformers plumbs on one knob:
//
//   - below the crossover  -> host-hybrid: hot experts on the device, cold experts on the host
//     concurrently, no extra VRAM. Cheap for short prompts where CPU MoE is not yet the bottleneck.
//   - at or above the crossover -> device-layerwise: cold experts stream host->device one MoE layer
//     at a time and the expert GEMM runs on device tensor cores. Costs one MoE layer of extra VRAM
//     but prefill then scales with device FLOPs instead of host MoE bandwidth.
//
// It is PURE: a host-free, clock-free choice over token counts and per-token costs. No expert home
// is moved here — splitKernel still does that; here we only choose WHICH regime the current prefill
// should run, so the choice is deterministic and unit-testable. The crossover is honored exactly:
// the token count is compared with >=, matching ktransformers' at-or-above-threshold trigger.
//
// The crossover boundary can be either INJECTED (an operator-measured threshold, ktransformers'
// default 4096) or DERIVED from the per-token costs via cacheprice.TransferBreakEvenLength — the
// same fixed-floor-plus-per-token break-even fak already uses for KV fetch-vs-recompute. Composing
// those two existing mechanisms is the whole delta.

// PrefillPlacementInputs is the pure input to the length-driven switch. All counts are
// non-negative; a degenerate input (zero or negative token count, zero cost slope, no crossover
// that a length can ever clear) fails closed to host-hybrid — the safe default that never needs
// extra VRAM and so can never OOM.
type PrefillPlacementInputs struct {
	// PrefillTokens is the number of tokens in the current prefill chunk (ktransformers' num_tokens).
	// Zero or negative is a degenerate input and fails closed to host-hybrid.
	PrefillTokens int
	// Batch is the number of sequences prefilled together; the total work compared with the crossover
	// is PrefillTokens * Batch. A Batch <= 0 is treated as a single sequence (Batch = 1), never zero
	// work, so a caller that omits it still gets a length-driven choice on PrefillTokens alone.
	Batch int
	// HostPerToken is the per-token cost of the host-hybrid path (CPU MoE). This is the "recompute we
	// save by streaming" side of the break-even. Clamped to >= 0.
	HostPerToken int
	// DevicePerToken is the per-token cost of the device-layerwise path once staged. This is the
	// per-token wire+compute the streaming path still pays. Clamped to >= 0.
	DevicePerToken int
	// StagingOverhead is the fixed, length-independent cost of staging one MoE layer host->device
	// (the "one MoE layer of extra VRAM" toll amortized over the prefill). This is the fixed floor of
	// the break-even. Clamped to >= 0.
	StagingOverhead int
	// InjectedCrossover, when > 0, is an operator-measured token threshold used VERBATIM (bypassing
	// the cost-derived break-even) — ktransformers' `--kt-gpu-prefill-token-threshold`, default 4096.
	// A value <= 0 means "derive the crossover from the per-token costs".
	InjectedCrossover int
}

// PrefillPlacementChoice is the chosen regime plus the boundary that produced it. StageToDevice is
// the one bit splitKernel would consult; the rest explains the choice for a witness and for an
// operator dashboard.
type PrefillPlacementChoice struct {
	// StageToDevice is true for the device-layerwise regime, false for host-hybrid. False is the
	// fail-closed default for every degenerate or never-worthwhile input.
	StageToDevice bool
	// CrossoverTokens is the token boundary honored: the injected threshold when one was supplied, or
	// the cost-derived break-even otherwise. Zero when no length ever favors the device (EverStages
	// false) — meaningless in that case, so read EverStages first.
	CrossoverTokens int
	// EverStages is true when SOME prefill length would choose the device: an injected threshold is
	// supplied, or the derived break-even exists (host per-token strictly exceeds device per-token).
	// When false the device path can never win and the switch is pinned to host-hybrid.
	EverStages bool
	// FromInjected is true when CrossoverTokens came from InjectedCrossover, false when it was derived
	// from the per-token costs. Lets a witness tell an operator override from a measured break-even.
	FromInjected bool
	// TotalTokens is the work compared with the crossover: PrefillTokens * effective Batch. Reported
	// so a witness can confirm the >= comparison without re-deriving the batch handling.
	TotalTokens int
}

// ChoosePrefillPlacement chooses host-hybrid vs device-layerwise for the current prefill. It is the
// pure switch: below the crossover -> host-hybrid, at or above -> device-layerwise, with the
// boundary honored exactly (>=). Degenerate inputs fail closed to host-hybrid:
//
//   - PrefillTokens <= 0 (nothing to prefill) -> host-hybrid, whatever the crossover.
//   - no injected threshold AND no derivable break-even (host per-token <= device per-token, so
//     streaming never pays for itself) -> host-hybrid for every length, EverStages false.
//
// An injected threshold > 0 is honored verbatim; otherwise the break-even is derived from the
// per-token costs and the fixed staging floor via cacheprice.TransferBreakEvenLength. Deterministic,
// no device alloc, no clock.
func ChoosePrefillPlacement(in PrefillPlacementInputs) PrefillPlacementChoice {
	batch := in.Batch
	if batch <= 0 {
		batch = 1
	}
	total := 0
	if in.PrefillTokens > 0 {
		total = in.PrefillTokens * batch
	}

	crossover, everStages, fromInjected := resolvePrefillCrossover(in)

	choice := PrefillPlacementChoice{
		CrossoverTokens: crossover,
		EverStages:      everStages,
		FromInjected:    fromInjected,
		TotalTokens:     total,
	}
	// Fail closed: nothing to prefill, or no length ever favors the device -> host-hybrid.
	if total <= 0 || !everStages {
		choice.StageToDevice = false
		return choice
	}
	// At or above the crossover stages to the device; below stays host-hybrid.
	choice.StageToDevice = total >= crossover
	return choice
}

// resolvePrefillCrossover picks the boundary the switch compares against. A positive
// InjectedCrossover wins verbatim (an operator-measured threshold). Otherwise the break-even is
// derived from the per-token costs and the fixed staging floor; when the host per-token cost does
// not strictly exceed the device per-token cost no length ever amortizes the staging floor, so
// everStages is false and the crossover is a meaningless 0.
func resolvePrefillCrossover(in PrefillPlacementInputs) (crossover int, everStages, fromInjected bool) {
	if in.InjectedCrossover > 0 {
		return in.InjectedCrossover, true, true
	}
	breakEven, ok := cacheprice.TransferBreakEvenLength(in.StagingOverhead, in.HostPerToken, in.DevicePerToken)
	if !ok {
		return 0, false, false
	}
	return breakEven, true, false
}
