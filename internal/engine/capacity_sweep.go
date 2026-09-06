package engine

// capacity_sweep.go binds the already-shipped capacity planks into one callable
// loop: compute reports HBM pressure, cachemeta decides the placement, and the
// CapacityAdapter executes the demote/spill/evict move. It is intentionally small
// and fail-open so a serving loop can call it without turning unknown capacity into
// a false refusal.

import (
	"context"
	"fmt"
	"sort"

	"github.com/anthony-chaudhary/fak/internal/cachemeta"
	"github.com/anthony-chaudhary/fak/internal/compute"
)

// CapacityPressureCandidate is one live KV span the pressure sweep may move out of
// HBM. Request carries the span's placement economics; Move carries its executable
// identity in the live KV backend. ReclaimBytes is the HBM byte estimate to subtract
// after a successful move; when unset, Request.SizeBytes is used.
type CapacityPressureCandidate struct {
	Request      cachemeta.PlacementRequest
	Move         PlacementMove
	ReclaimBytes int64
}

// CapacityPressureSweep configures one bounded pressure-relief pass. TargetPressure
// is the desired HBM high-water mark in (0,1]; values outside that range default to
// 1.0, preserving the older "only full means pressure" behavior. MaxMoves <= 0 means
// no explicit move cap beyond the candidate list.
type CapacityPressureSweep struct {
	Backend        compute.Backend
	Adapter        *CapacityAdapter
	ResidentBytes  int64
	TargetPressure float64
	MaxMoves       int
	Candidates     []CapacityPressureCandidate

	// DRAMPressure, DRAMCapacityBytes, and DRAMKnown carry the host's live L2/DRAM fullness as
	// probed by the CALLER (the served loop's HostDRAMPressure — see capacity_dram.go), threaded
	// in so the demote TARGET is planned against real host RAM rather than the assumed-empty
	// profile default. Without them the sweep was hardware-aware at its top rung (HBM, via
	// Backend) but BLIND one tier below: a span under HBM pressure that PlanPlacement demotes
	// INTO DRAM was staged there even on a box whose RAM is already full, where the honest move
	// is to skip DRAM for a colder tier with room. This is the "probe DeviceHBMPressure AND
	// HostDRAMPressure" half of issue #1073's wire on the executor side (PlanPlacementForDeviceAndHost
	// is the planless sibling of the same fold).
	//
	// DRAMKnown gates the fold and provides fallback: false (the default — an unsupported
	// host probe, or a caller that does not probe DRAM) folds nothing, so the sweep plans exactly
	// as it did before these fields existed. The probe is the caller's; the sweep only ACTS on it,
	// staying a pure, deterministic function of its config.
	DRAMPressure      float64
	DRAMCapacityBytes int64
	DRAMKnown         bool
}

// CapacityPressureMove records the decision and execution result for one candidate
// the sweep attempted.
type CapacityPressureMove struct {
	Index    int
	Decision cachemeta.PlacementDecision
	Result   PlacementResult
}

// CapacityPressureResult is the typed outcome of one pressure sweep.
type CapacityPressureResult struct {
	Known           bool
	CapacityBytes   int64
	TargetPressure  float64
	InitialPressure float64
	FinalPressure   float64
	ReclaimedBytes  int64
	AppliedMoves    int
	Faults          int
	Moves           []CapacityPressureMove
}

// RunCapacityPressureSweep relieves HBM pressure by planning and executing moves
// for candidate KV spans until the estimated pressure drops below TargetPressure,
// the move cap is reached, or candidates are exhausted. Unknown capacity is a clean
// no-op. Staging faults are recorded in the result but do not abort the sweep, so a
// single bad colder tier cannot hide pressure on the remaining candidates.
func RunCapacityPressureSweep(ctx context.Context, cfg CapacityPressureSweep) (CapacityPressureResult, error) {
	target := normalizeTargetPressure(cfg.TargetPressure)
	pressure, capacity, known := DeviceHBMPressure(cfg.Backend, cfg.ResidentBytes)
	res := CapacityPressureResult{
		Known:           known,
		CapacityBytes:   capacity,
		TargetPressure:  target,
		InitialPressure: pressure,
		FinalPressure:   pressure,
	}
	if !known {
		return res, nil
	}
	if pressure < target || len(cfg.Candidates) == 0 {
		return res, nil
	}
	if cfg.Adapter == nil {
		return res, fmt.Errorf("engine: capacity pressure sweep has no adapter")
	}

	resident := cfg.ResidentBytes
	// Process candidates highest-value-to-relocate first, and fold the bytes each demote
	// commits back into the colder tier's pressure so a wave of demotes cannot overfill a
	// tier the per-span planner sees as empty. Together these turn the sweep from "demote
	// whatever comes first until something breaks" into "spend the scarce colder-tier room
	// on the spans that benefit most, and stop when that room is gone."
	committed := map[cachemeta.ResidencyTier]int64{}
	for _, i := range victimOrder(cfg.Candidates) {
		cand := cfg.Candidates[i]
		if cfg.MaxMoves > 0 && len(res.Moves) >= cfg.MaxMoves {
			break
		}
		if res.FinalPressure < target {
			break
		}
		req := withSweepTierBudget(withSweepHostDRAM(cfg, cand.Request), committed)
		decision := planPlacementForDeviceAtHighWater(cfg.Backend, resident, target, req)
		if !capacityPressureDropAction(decision.Action) {
			continue
		}
		mv := cand.Move
		mv.Decision = decision
		moveRes, err := cfg.Adapter.Execute(ctx, mv)
		if err != nil {
			return res, err
		}
		res.Moves = append(res.Moves, CapacityPressureMove{Index: i, Decision: decision, Result: moveRes})
		if !moveRes.Applied {
			res.Faults++
			continue
		}
		res.AppliedMoves++
		if isColderRelocation(decision.Action) {
			// Charge the demoted bytes against the target tier so the next candidate planned
			// against it sees the room shrink (and, once spent, cascades colder or evicts).
			committed[decision.ToTier] += decision.EstMoveBytes
		}
		reclaimed := cand.ReclaimBytes
		if reclaimed <= 0 {
			reclaimed = cand.Request.SizeBytes
		}
		if reclaimed < 0 {
			reclaimed = 0
		}
		if reclaimed > resident {
			reclaimed = resident
		}
		resident -= reclaimed
		res.ReclaimedBytes += reclaimed
		res.FinalPressure = pressureAfterReclaim(cfg.Backend, resident)
	}
	return res, nil
}

// withSweepHostDRAM folds the caller-probed host-DRAM pressure (and, when given, capacity) into a
// COPY of req when DRAMKnown, so a span under HBM pressure that would demote INTO DRAM is planned
// against real host fullness — a full DRAM routes the demote one tier colder instead of staging
// the span into a tier with no room. DRAMKnown=false returns req unchanged (the fail-open default,
// byte-identical to the pre-DRAM-aware sweep). It mirrors withHostDRAM in capacity_dram.go but
// takes the already-probed values rather than re-reading the host, keeping the sweep a pure,
// deterministic function of its config; a non-positive DRAMCapacityBytes leaves the tier's profile
// ceiling alone (pressure-only fold), matching the "only override a real ceiling" tier contract.
func withSweepHostDRAM(cfg CapacityPressureSweep, req cachemeta.PlacementRequest) cachemeta.PlacementRequest {
	if !cfg.DRAMKnown {
		return req
	}
	req.Pressure = withTierPressure(req.Pressure, cachemeta.TierDRAM, cfg.DRAMPressure)
	if cfg.DRAMCapacityBytes > 0 {
		req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierDRAM, cfg.DRAMCapacityBytes)
	}
	return req
}

// PlanPlacementForDeviceAtHighWater is PlanPlacementForDevice with an operator high-water
// mark. A TargetPressure of 0.80 means observed 80% HBM use is presented to cachemeta as
// "full" pressure, so demotion can happen before the allocator is literally out of memory.
func PlanPlacementForDeviceAtHighWater(b compute.Backend, residentBytes int64, targetPressure float64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	return planPlacementForDeviceAtHighWater(b, residentBytes, normalizeTargetPressure(targetPressure), req)
}

func planPlacementForDeviceAtHighWater(b compute.Backend, residentBytes int64, targetPressure float64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	if pressure, capacity, known := DeviceHBMPressure(b, residentBytes); known {
		req.Pressure = withTierPressure(req.Pressure, cachemeta.TierHBM, scalePressureToTarget(pressure, targetPressure))
		req.Profiles = withTierCapacity(req.Profiles, cachemeta.TierHBM, capacity)
	}
	return cachemeta.PlanPlacement(req)
}

func normalizeTargetPressure(p float64) float64 {
	if p <= 0 || p > 1 {
		return 1
	}
	return p
}

func scalePressureToTarget(pressure, target float64) float64 {
	if target <= 0 || target >= 1 {
		return pressure
	}
	if pressure <= 0 {
		return 0
	}
	scaled := pressure / target
	if scaled > 1 {
		return 1
	}
	return scaled
}

func capacityPressureDropAction(a cachemeta.PlacementAction) bool {
	switch a {
	case cachemeta.ActionDemote, cachemeta.ActionSpill, cachemeta.ActionCompressDemote, cachemeta.ActionEvict:
		return true
	default:
		return false
	}
}

// isColderRelocation reports whether an action RELOCATES the span into a colder tier (and
// therefore consumes that tier's capacity budget). Eviction drops the span to recompute and
// spends no colder-tier room, so it is excluded.
func isColderRelocation(a cachemeta.PlacementAction) bool {
	switch a {
	case cachemeta.ActionDemote, cachemeta.ActionSpill, cachemeta.ActionCompressDemote:
		return true
	default:
		return false
	}
}

// victimOrder returns candidate indices sorted highest-value-to-relocate first, so a bounded
// sweep spends the scarce room in the colder tiers on the spans that gain most from being
// relocated rather than dropped: the most expensive to recompute (tokens x per-token prefill)
// first, then — among equally costly spans — the cheapest to restore (fewest bytes), then the
// coldest (least-recently accessed). The prior sweep processed candidates in enumeration
// order with identical synthetic economics, so under a colder-tier budget whichever span
// happened to come first won the room. This is pure over the candidates' declared economics
// (no clock, no backend read) and stable: a full tie keeps the caller's original order.
func victimOrder(cands []CapacityPressureCandidate) []int {
	order := make([]int, len(cands))
	for i := range order {
		order[i] = i
	}
	sort.SliceStable(order, func(a, b int) bool {
		ri, rj := cands[order[a]].Request, cands[order[b]].Request
		if vi, vj := recomputeCost(ri), recomputeCost(rj); vi != vj {
			return vi > vj // most expensive to recompute relocates first
		}
		if ri.SizeBytes != rj.SizeBytes {
			return ri.SizeBytes < rj.SizeBytes // cheapest to restore first
		}
		return ri.Lifecycle.LastAccessMillis < rj.Lifecycle.LastAccessMillis // coldest first
	})
	return order
}

// recomputeCost is the cost of rebuilding a span from scratch (tokens x per-token prefill) —
// the quantity a demote AVOIDS, and the headline term in a span's value-to-relocate. A span
// with no token/cost info scores 0 (nothing to avoid), so it sorts behind spans that do.
func recomputeCost(req cachemeta.PlacementRequest) int64 {
	if req.Tokens <= 0 || req.PerTokenPrefillNanos <= 0 {
		return 0
	}
	return req.Tokens * req.PerTokenPrefillNanos
}

// withSweepTierBudget raises each colder tier's pressure to reflect the bytes THIS sweep has
// already committed into it, so a span planned against a tier earlier demotes filled sees the
// room shrink — and, once the tier's capacity budget is spent (pressure reaches 1.0), the
// planner cascades the span to the next colder tier or to eviction instead of piling on top
// of demotes its per-span coldestColderWithRoom check cannot yet see. Without this the sweep
// plans every candidate against the tier's ORIGINAL fullness, so N spans each independently
// believe a tier with room for one has room for all, overfilling it into a cascade. The fold
// is fail-open and mirrors withSweepHostDRAM's copy-on-write shape: a tier with no profiled
// capacity (CapacityBytes <= 0) is left untouched — the sweep never invents a ceiling the box
// did not declare — keeping the sweep a pure function of its config plus its own commitments.
func withSweepTierBudget(req cachemeta.PlacementRequest, committed map[cachemeta.ResidencyTier]int64) cachemeta.PlacementRequest {
	for tier, bytes := range committed {
		if bytes <= 0 {
			continue
		}
		prof, ok := req.Profiles[tier]
		if !ok || prof.CapacityBytes <= 0 {
			continue
		}
		p := req.Pressure[tier] + float64(bytes)/float64(prof.CapacityBytes)
		if p > 1 {
			p = 1
		}
		req.Pressure = withTierPressure(req.Pressure, tier, p)
	}
	return req
}

func pressureAfterReclaim(b compute.Backend, residentBytes int64) float64 {
	p, _, known := DeviceHBMPressure(b, residentBytes)
	if !known {
		return 0
	}
	return p
}
