package ggufload

import (
	"fmt"
	"math"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// expert_activated_fit.go — the ACTIVATED-working-set device admission (R1 of the MoE
// activated-offload ladder, #5612; docs/MOE-ACTIVATED-OFFLOAD-PLAN.md).
//
// Every device admission in this file's neighbours asks the same question: "do the WEIGHTS fit?"
// For a routed-MoE checkpoint that question is the wrong one. FitOnDevice charges the whole
// checkpoint; FitCPUOffloadExpertsOnDevice charges the dense base and moves the ENTIRE expert band
// to host — binary, all experts device-resident or none. Neither can express the placement the
// bounded expert ring (#5611) actually runs: a permanently resident dense base, plus a BOUNDED
// window of routed-expert bytes that pages against a budget.
//
// A token touches K of E experts per MoE layer (GLM-5.2: 8 of 256, ~3%), so the bytes that must be
// co-resident for decode to make progress are far smaller than the band. This file names the three
// levels of that demand from the header alone, and the fit check admits on the FLOOR:
//
//	floor    dense base + one MoE layer's K experts     — decode CAN run; the ring pages layer to layer
//	token    dense base + K experts on EVERY MoE layer  — a whole token is resident; no intra-token re-page
//	band     dense base + the full routed band          — what today's non-offload admission demands
//
// Between floor and band the model is servable but paging; below the floor no ring budget can save
// it, because a single expert GEMM group cannot be assembled. That is the honest refusal line, and
// it is strictly more permissive than the two checks above it — a checkpoint they refuse can be
// admitted here, and one this refuses they would refuse too.

// ActivatedExpertFit is the header-derived activated-working-set admission for a routed-MoE
// checkpoint at a stated device budget. It reads no tensor payloads: every byte count comes from
// RoutedExpertActiveSet's single pass over the tensor directory.
//
// The three *Fits booleans are the three levels above, evaluated against DeviceBudgetBytes. They
// are monotone — BandFits implies TokenFits implies MinFits — so a caller can report WHICH level a
// checkpoint reached rather than a bare yes/no.
type ActivatedExpertFit struct {
	MoELayers   int // distinct block ordinals carrying a batched routed-expert tensor
	NumExperts  int // E
	ExpertsUsed int // K, clamped to E; 0 when the header omits expert_used_count

	// DeviceBaseBytes is the permanently resident remainder — attention, dense FFN, router,
	// shared experts, embeddings. Nothing in this placement can spill it, so it is the one
	// addend every level below carries.
	DeviceBaseBytes int64
	// RoutedBandBytes is every batched routed-expert tensor, all experts, unsharded.
	RoutedBandBytes int64
	// ActivatedLayerBytes is K experts of ONE MoE layer — the hard floor. Per-expert-per-layer
	// bytes are rounded UP, so an uneven checkpoint over-reserves rather than under-counting
	// into an OOM at the first outsized layer.
	ActivatedLayerBytes int64
	// ActivatedTokenBytes is K experts on every MoE layer: the routed stream one token pulls.
	ActivatedTokenBytes int64

	// DeviceBudgetBytes is the budget the verdicts were computed against, headroom already
	// applied. 0 means unmeasurable — every verdict is then false and the caller must fail open.
	DeviceBudgetBytes int64
	// RingBytes is the routed-expert ring budget (#5611) this placement would size: whatever the
	// device budget leaves after the dense base, never below ActivatedLayerBytes (a smaller ring
	// cannot hold one layer's activation) and never above RoutedBandBytes (a larger one has
	// nothing left to hold). It is exactly the device-scoped expert demand of MemoryPlan.
	RingBytes int64
	// HostBandBytes is RoutedBandBytes - RingBytes: the routed bytes that stay on host.
	HostBandBytes int64

	MinFits   bool // DeviceBaseBytes + ActivatedLayerBytes fits: decode can run, paging
	TokenFits bool // DeviceBaseBytes + ActivatedTokenBytes fits: no intra-token re-page
	BandFits  bool // DeviceBaseBytes + RoutedBandBytes fits: the ring never evicts
}

// ActivatedExpertFitFor derives the activated-working-set admission at a stated device budget
// (headroom already applied; pass 0 when the backend cannot report one). ok=false for a checkpoint
// with no batched routed-expert band — a dense model has no activated working set, and the caller
// should fall back to the ordinary resident admission.
func (s *WeightSource) ActivatedExpertFitFor(deviceBudgetBytes int64) (ActivatedExpertFit, bool, error) {
	as, ok, err := s.RoutedExpertActiveSet()
	if err != nil || !ok {
		return ActivatedExpertFit{}, false, err
	}
	return activatedExpertFit(as, deviceBudgetBytes)
}

// activatedExpertFit is the pure arithmetic, split out so the byte math is testable without a GGUF
// on disk and so ActivatedExpertFitFor stays a header read plus one call.
func activatedExpertFit(as RoutedExpertActiveSet, deviceBudgetBytes int64) (ActivatedExpertFit, bool, error) {
	if as.MoELayers <= 0 || as.NumExperts <= 0 || as.RoutedResident <= 0 {
		return ActivatedExpertFit{}, false, nil
	}
	k := as.ExpertsUsed
	if k > as.NumExperts {
		// A header claiming K > E is nonsense, but clamping keeps the floor a real byte count
		// (the whole layer) instead of an over-reservation that refuses a servable model.
		k = as.NumExperts
	}
	if deviceBudgetBytes < 0 {
		deviceBudgetBytes = 0
	}
	f := ActivatedExpertFit{
		MoELayers:           as.MoELayers,
		NumExperts:          as.NumExperts,
		ExpertsUsed:         k,
		DeviceBaseBytes:     as.NonExpertResident,
		RoutedBandBytes:     as.RoutedResident,
		ActivatedTokenBytes: as.ActivePerToken,
		DeviceBudgetBytes:   deviceBudgetBytes,
	}
	if k > 0 {
		perExpertLayer := ceilDivInt64(as.RoutedResident, int64(as.NumExperts)*int64(as.MoELayers))
		if perExpertLayer != 0 && int64(k) > math.MaxInt64/perExpertLayer {
			return ActivatedExpertFit{}, false, fmt.Errorf("gguf: activated expert layer bytes overflow int64")
		}
		f.ActivatedLayerBytes = perExpertLayer * int64(k)
		if f.ActivatedLayerBytes > as.RoutedResident {
			f.ActivatedLayerBytes = as.RoutedResident
		}
	} else {
		// The header omitted expert_used_count, so the activated set is unknown. Fall back to the
		// whole band: an unknown K must not be admitted as a small one.
		f.ActivatedLayerBytes = as.RoutedResident
		f.ActivatedTokenBytes = as.RoutedResident
	}
	if f.ActivatedTokenBytes < f.ActivatedLayerBytes {
		f.ActivatedTokenBytes = f.ActivatedLayerBytes
	}
	if f.DeviceBaseBytes > math.MaxInt64-f.RoutedBandBytes {
		return ActivatedExpertFit{}, false, fmt.Errorf("gguf: activated expert fit totals overflow int64")
	}

	f.RingBytes = f.ActivatedLayerBytes
	if room := deviceBudgetBytes - f.DeviceBaseBytes; room > f.RingBytes {
		f.RingBytes = room
	}
	if f.RingBytes > f.RoutedBandBytes {
		f.RingBytes = f.RoutedBandBytes
	}
	f.HostBandBytes = f.RoutedBandBytes - f.RingBytes

	if deviceBudgetBytes > 0 {
		f.MinFits = f.DeviceBaseBytes+f.ActivatedLayerBytes <= deviceBudgetBytes
		f.TokenFits = f.DeviceBaseBytes+f.ActivatedTokenBytes <= deviceBudgetBytes
		f.BandFits = f.DeviceBaseBytes+f.RoutedBandBytes <= deviceBudgetBytes
	}
	return f, true, nil
}

// MemoryPlan is the classed form of the placement: the dense base and the sized ring are
// device-scoped weights, the rest of the routed band is host-scoped offload. Feeding it to
// compute.RefuseMemoryPlanIfTooBig yields the same typed *compute.FitError every other admission
// in this package produces — with the demand classes preserved, so an operator surface can show
// WHICH part did not fit.
//
// The device total is DeviceBaseBytes + RingBytes, and RingBytes never drops below
// ActivatedLayerBytes, so a budget under the floor produces a device demand that exceeds it: the
// refusal is a property of the plan, not a separate branch.
//
// This is why there is no Estimate*MemoryPlan sibling for the other placements: a plan here is
// meaningless without the budget it was sized against, so the budget-taking fit carries it.
func (f ActivatedExpertFit) MemoryPlan() compute.MemoryPlan {
	plan := make(compute.MemoryPlan, 0, 3)
	if f.DeviceBaseBytes > 0 {
		plan = append(plan, compute.MemoryDemand{
			Class:  compute.MemoryWeights,
			Bytes:  f.DeviceBaseBytes,
			Detail: "gguf-device-dense-base",
			Scope:  compute.MemoryScopeDevice,
		})
	}
	if f.RingBytes > 0 {
		plan = append(plan, compute.MemoryDemand{
			Class:  compute.MemoryWeights,
			Bytes:  f.RingBytes,
			Detail: "gguf-device-activated-expert-ring",
			Scope:  compute.MemoryScopeDevice,
		})
	}
	if f.HostBandBytes > 0 {
		plan = append(plan, compute.MemoryDemand{
			Class:  compute.MemoryOffload,
			Bytes:  f.HostBandBytes,
			Detail: "gguf-host-expert-band",
			Scope:  compute.MemoryScopeHost,
		})
	}
	return plan
}

// FitActivatedExpertsOnDevice is the device-fit refusal for the bounded activated-expert placement:
// the dense base plus a ring sized to what the device can actually hold, with the remaining routed
// band host-scoped. Use it INSTEAD of FitCPUOffloadExpertsOnDevice wherever the ring (#5611) is
// available, because it admits every checkpoint that one admits and also the band that only pages.
//
// It keeps the fail-open capacity contract: a backend that cannot report memory yields nil (the
// load proceeds), never a refusal. A checkpoint with no batched routed-expert band has no activated
// working set to admit on, so it falls back to FitOnDevice — a dense model is still just a dense
// model. headroom in [0,1) reserves that fraction for KV/activations/scratch, exactly as in
// FitOnDevice.
func (s *WeightSource) FitActivatedExpertsOnDevice(be compute.Backend, headroom float64) error {
	total, free, known := compute.DeviceMemoryInfo(be)
	if !known {
		return nil
	}
	budget := free
	if budget < 0 { // FreeUnknown -> the total ceiling, conservatively
		budget = total
	}
	f, ok, err := s.ActivatedExpertFitFor(compute.BudgetAfterHeadroom(budget, headroom))
	if err != nil {
		return err
	}
	if !ok {
		return s.FitOnDevice(be, headroom)
	}
	return compute.RefuseMemoryPlanIfTooBig(be, f.MemoryPlan(), headroom)
}

func ceilDivInt64(n, d int64) int64 {
	if d <= 0 {
		return 0
	}
	return (n + d - 1) / d
}
