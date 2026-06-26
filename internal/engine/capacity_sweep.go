package engine

// capacity_sweep.go binds the already-shipped capacity planks into one callable
// loop: compute reports HBM pressure, cachemeta decides the placement, and the
// CapacityAdapter executes the demote/spill/evict move. It is intentionally small
// and fail-open so a serving loop can call it without turning unknown capacity into
// a false refusal.

import (
	"context"
	"fmt"

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
	for i, cand := range cfg.Candidates {
		if cfg.MaxMoves > 0 && len(res.Moves) >= cfg.MaxMoves {
			break
		}
		if res.FinalPressure < target {
			break
		}
		decision := planPlacementForDeviceAtHighWater(cfg.Backend, resident, target, cand.Request)
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

// PlanPlacementForDeviceAtHighWater is PlanPlacementForDevice with an operator high-water
// mark. A TargetPressure of 0.80 means observed 80% HBM use is presented to cachemeta as
// "full" pressure, so demotion can happen before the allocator is literally out of memory.
func PlanPlacementForDeviceAtHighWater(b compute.Backend, residentBytes int64, targetPressure float64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	return planPlacementForDeviceAtHighWater(b, residentBytes, normalizeTargetPressure(targetPressure), req)
}

func planPlacementForDeviceAtHighWater(b compute.Backend, residentBytes int64, targetPressure float64, req cachemeta.PlacementRequest) cachemeta.PlacementDecision {
	if pressure, capacity, known := DeviceHBMPressure(b, residentBytes); known {
		req.Pressure = withHBMPressure(req.Pressure, scalePressureToTarget(pressure, targetPressure))
		req.Profiles = withHBMCapacity(req.Profiles, capacity)
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

func pressureAfterReclaim(b compute.Backend, residentBytes int64) float64 {
	p, _, known := DeviceHBMPressure(b, residentBytes)
	if !known {
		return 0
	}
	return p
}
