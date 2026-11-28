// Copyright (c) 2026 fak contributors
// SPDX-License-Identifier: Apache-2.0
//
// INSPIRE-ONLY clean-room adaptation of NVIDIA/Personal-AI-Router
// (NVIDIA/Personal-AI-Router@9ec17912787915f61717e57019e49802fc3b7da7):
// device discovery, capability-aware target selection, and cross-device
// inference distribution planning for the fak gateway router seam.
package gateway

import (
	"errors"
	"sort"
	"sync"
)

// ErrNoEligiblePairDevice is returned when no registered healthy device
// satisfies the request's capability and memory constraints.
var ErrNoEligiblePairDevice = errors.New("gateway/pair: no eligible device")

// PairDevice describes one discovered compute target for capability-aware routing.
type PairDevice struct {
	ID            string
	Name          string
	Kind          string // e.g. "gpu", "cpu", "npu"
	Local         bool
	Healthy       bool
	VRAMBytes     int64
	FreeVRAMBytes int64
	Capabilities  []string
	Load          float64 // 0.0 (idle) .. 1.0+ (saturated)
}

// PairRouteRequest states the capability and memory constraints of one inference request.
type PairRouteRequest struct {
	RequiredCaps []string
	MinVRAMBytes int64
	PreferLocal  bool
	ExcludeIDs   []string
}

// PairRoutePlan records either a single-device placement or a cross-device
// stage split when no single device holds the full MinVRAMBytes.
type PairRoutePlan struct {
	Primary   PairDevice
	Stages    []PairDevice // populated for cross-device split placements
	SplitUsed bool
}

// PairRouter maintains discovered devices and selects capability-aware targets.
type PairRouter struct {
	mu      sync.RWMutex
	devices map[string]PairDevice
}

// NewPairRouter constructs an empty capability-aware device router.
func NewPairRouter() *PairRouter {
	return &PairRouter{devices: make(map[string]PairDevice)}
}

// Register adds or updates a discovered device in the router registry.
func (r *PairRouter) Register(dev PairDevice) {
	if r == nil || dev.ID == "" {
		return
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.devices[dev.ID] = dev
}

// Discover returns a deterministic snapshot of all registered devices sorted by ID.
func (r *PairRouter) Discover() []PairDevice {
	if r == nil {
		return nil
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]PairDevice, 0, len(r.devices))
	for _, d := range r.devices {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].ID < out[j].ID })
	return out
}

func hasAllCaps(have, need []string) bool {
	if len(need) == 0 {
		return true
	}
	set := make(map[string]struct{}, len(have))
	for _, c := range have {
		set[c] = struct{}{}
	}
	for _, req := range need {
		if _, ok := set[req]; !ok {
			return false
		}
	}
	return true
}

// Select chooses the best single healthy, capability-matched device, or falls
// back to a cross-device split across healthy capability-matched devices when
// their combined free VRAM covers MinVRAMBytes.
func (r *PairRouter) Select(req PairRouteRequest) (PairRoutePlan, error) {
	if r == nil {
		return PairRoutePlan{}, ErrNoEligiblePairDevice
	}
	excluded := make(map[string]struct{}, len(req.ExcludeIDs))
	for _, id := range req.ExcludeIDs {
		excluded[id] = struct{}{}
	}

	var capMatched []PairDevice
	var singleCandidates []PairDevice
	for _, d := range r.Discover() {
		if !d.Healthy {
			continue
		}
		if _, skip := excluded[d.ID]; skip {
			continue
		}
		if !hasAllCaps(d.Capabilities, req.RequiredCaps) {
			continue
		}
		capMatched = append(capMatched, d)
		if d.FreeVRAMBytes >= req.MinVRAMBytes {
			singleCandidates = append(singleCandidates, d)
		}
	}

	if len(singleCandidates) > 0 {
		sort.SliceStable(singleCandidates, func(i, j int) bool {
			if req.PreferLocal && singleCandidates[i].Local != singleCandidates[j].Local {
				return singleCandidates[i].Local
			}
			if singleCandidates[i].Load != singleCandidates[j].Load {
				return singleCandidates[i].Load < singleCandidates[j].Load
			}
			if singleCandidates[i].FreeVRAMBytes != singleCandidates[j].FreeVRAMBytes {
				return singleCandidates[i].FreeVRAMBytes > singleCandidates[j].FreeVRAMBytes
			}
			return singleCandidates[i].ID < singleCandidates[j].ID
		})
		return PairRoutePlan{Primary: singleCandidates[0]}, nil
	}

	// Cross-device distribution when no single device holds MinVRAMBytes.
	if req.MinVRAMBytes > 0 && len(capMatched) > 1 {
		sort.SliceStable(capMatched, func(i, j int) bool {
			if capMatched[i].FreeVRAMBytes != capMatched[j].FreeVRAMBytes {
				return capMatched[i].FreeVRAMBytes > capMatched[j].FreeVRAMBytes
			}
			return capMatched[i].ID < capMatched[j].ID
		})
		var stages []PairDevice
		var totalFree int64
		for _, d := range capMatched {
			if d.FreeVRAMBytes <= 0 {
				continue
			}
			stages = append(stages, d)
			totalFree += d.FreeVRAMBytes
			if totalFree >= req.MinVRAMBytes {
				return PairRoutePlan{
					Primary:   stages[0],
					Stages:    stages,
					SplitUsed: true,
				}, nil
			}
		}
	}

	return PairRoutePlan{}, ErrNoEligiblePairDevice
}
