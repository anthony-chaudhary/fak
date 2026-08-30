package compute

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"sort"
)

// TransientValue declares the storage contract for one graph value. LifetimeEnd is an
// optional explicit marker naming the last graph step at which the value must remain live.
// Escapes keeps the value live through the end of the forward pass.
type TransientValue struct {
	Node        NodeID `json:"node"`
	Bytes       int64  `json:"bytes"`
	LifetimeEnd NodeID `json:"lifetime_end,omitempty"`
	Escapes     bool   `json:"escapes,omitempty"`
}

// TransientAllocation is a deterministic suballocation chosen for one graph value. Values
// with disjoint inclusive [Start, End] intervals may share a Slot and Offset.
type TransientAllocation struct {
	Node    NodeID `json:"node"`
	Slot    int    `json:"slot"`
	Offset  int64  `json:"offset"`
	Bytes   int64  `json:"bytes"`
	Start   int    `json:"start"`
	End     int    `json:"end"`
	Escapes bool   `json:"escapes,omitempty"`
}

// TransientAllocationPlan is consumed by a backend suballocator. Mode is either
// "lifetime-reuse" or the existing fail-closed "forward-bump" fallback.
type TransientAllocationPlan struct {
	Mode        string                `json:"mode"`
	Allocations []TransientAllocation `json:"allocations"`
	Reserved    int64                 `json:"reserved_bytes"`
}

// TransientPlanReceipt records the eligibility decision and exact deterministic memory delta.
type TransientPlanReceipt struct {
	Eligible         bool   `json:"eligible"`
	FallbackReason   string `json:"fallback_reason,omitempty"`
	GraphDigest      string `json:"graph_digest"`
	AllocationDigest string `json:"allocation_digest"`
	Values           int    `json:"values"`
	Slots            int    `json:"slots"`
	NaiveBytes       int64  `json:"naive_bytes"`
	ReservedBytes    int64  `json:"reserved_bytes"`
	ReusedValues     int    `json:"reused_values"`
}

const (
	transientPlanLifetimeReuse = "lifetime-reuse"
	transientPlanForwardBump   = "forward-bump"
)

type transientInterval struct {
	value TransientValue
	start int
	end   int
}

type transientSlot struct {
	capacity int64
	end      int
	offset   int64
}

// PlanGraphTransients builds the smallest lifetime-aware reuse plan that is safe for the
// canonical graph. It intentionally falls back to one forward-bump range per value unless
// every declaration is proven: the node exists, its size is positive, its lifetime marker is
// dominated by the definition and follows every use, and escaping values remain distinct.
//
// This is adapted from the lifetime/escape boundary in Modular KGEN StackReuse.cpp at
// modular/modular@1c9fd2e03331f77d3a1034127cb3700b7fa43c02. The implementation and receipt
// remain fak-native and backend-neutral.
func PlanGraphTransients(graph Graph, values []TransientValue, align int64) (TransientAllocationPlan, TransientPlanReceipt, error) {
	canonical, pipelineReceipt, err := CanonicalGraphPipeline().Run(graph)
	if err != nil {
		return TransientAllocationPlan{}, TransientPlanReceipt{}, fmt.Errorf("compute transient plan: canonical graph: %w", err)
	}
	graphDigest := pipelineReceipt.FinalGraphDigest
	if graphDigest == "" {
		graphDigest, err = canonical.Digest()
		if err != nil {
			return TransientAllocationPlan{}, TransientPlanReceipt{}, fmt.Errorf("compute transient plan: graph digest: %w", err)
		}
	}

	intervals, reason := transientIntervals(canonical, values, align)
	if reason != "" {
		plan := forwardBumpTransientPlan(values, align)
		receipt, digestErr := transientPlanReceipt(plan, graphDigest, false, reason)
		return plan, receipt, digestErr
	}

	plan := reuseTransientPlan(intervals, align)
	receipt, err := transientPlanReceipt(plan, graphDigest, true, "")
	return plan, receipt, err
}

func transientIntervals(graph Graph, values []TransientValue, align int64) ([]transientInterval, string) {
	if align < 1 || align&(align-1) != 0 {
		return nil, "alignment must be a positive power of two"
	}
	order, err := stableTopologicalOrder(graph)
	if err != nil {
		return nil, "graph has no stable topological order"
	}
	positions := make(map[NodeID]int, len(order))
	consumers := make(map[NodeID][]int, len(order))
	outputs := make(map[NodeID]bool, len(graph.Outputs))
	for i, node := range order {
		positions[node.ID] = i
		for _, input := range node.Inputs {
			consumers[input] = append(consumers[input], i)
		}
	}
	for _, output := range graph.Outputs {
		outputs[output] = true
	}

	seen := make(map[NodeID]bool, len(values))
	intervals := make([]transientInterval, 0, len(values))
	for _, value := range values {
		start, ok := positions[value.Node]
		if !ok {
			return nil, fmt.Sprintf("value %q is not in the canonical graph", value.Node)
		}
		if seen[value.Node] {
			return nil, fmt.Sprintf("value %q is declared more than once", value.Node)
		}
		seen[value.Node] = true
		if value.Bytes <= 0 {
			return nil, fmt.Sprintf("value %q has non-positive size", value.Node)
		}

		lastUse := start
		for _, consumer := range consumers[value.Node] {
			if consumer > lastUse {
				lastUse = consumer
			}
		}
		end := lastUse
		if value.LifetimeEnd != "" {
			marker, exists := positions[value.LifetimeEnd]
			if !exists {
				return nil, fmt.Sprintf("value %q lifetime marker %q is not in the canonical graph", value.Node, value.LifetimeEnd)
			}
			if marker < start {
				return nil, fmt.Sprintf("value %q lifetime marker does not follow its definition", value.Node)
			}
			if marker < lastUse {
				return nil, fmt.Sprintf("value %q lifetime marker precedes its last use", value.Node)
			}
			end = marker
		}
		escapes := value.Escapes || outputs[value.Node]
		if escapes {
			end = len(order)
		}
		value.Escapes = escapes
		intervals = append(intervals, transientInterval{value: value, start: start, end: end})
	}

	sort.Slice(intervals, func(i, j int) bool {
		if intervals[i].start != intervals[j].start {
			return intervals[i].start < intervals[j].start
		}
		return intervals[i].value.Node < intervals[j].value.Node
	})
	return intervals, ""
}

func reuseTransientPlan(intervals []transientInterval, align int64) TransientAllocationPlan {
	slots := make([]transientSlot, 0, len(intervals))
	allocations := make([]TransientAllocation, 0, len(intervals))
	for _, interval := range intervals {
		slotIndex := -1
		if !interval.value.Escapes {
			for i := range slots {
				if slots[i].end < interval.start && slots[i].capacity >= interval.value.Bytes {
					slotIndex = i
					break
				}
			}
		}
		if slotIndex < 0 {
			slotIndex = len(slots)
			slots = append(slots, transientSlot{capacity: interval.value.Bytes})
		}
		slots[slotIndex].end = interval.end
		allocations = append(allocations, TransientAllocation{
			Node: interval.value.Node, Slot: slotIndex, Bytes: interval.value.Bytes,
			Start: interval.start, End: interval.end, Escapes: interval.value.Escapes,
		})
	}

	var reserved int64
	for i := range slots {
		reserved = alignUp(reserved, align)
		slots[i].offset = reserved
		reserved += slots[i].capacity
	}
	for i := range allocations {
		allocations[i].Offset = slots[allocations[i].Slot].offset
	}
	return TransientAllocationPlan{Mode: transientPlanLifetimeReuse, Allocations: allocations, Reserved: reserved}
}

func forwardBumpTransientPlan(values []TransientValue, align int64) TransientAllocationPlan {
	if align < 1 || align&(align-1) != 0 {
		align = 1
	}
	allocations := make([]TransientAllocation, 0, len(values))
	var reserved int64
	for i, value := range values {
		size := value.Bytes
		if size < 0 {
			size = 0
		}
		reserved = alignUp(reserved, align)
		allocations = append(allocations, TransientAllocation{
			Node: value.Node, Slot: i, Offset: reserved, Bytes: size, Escapes: value.Escapes,
		})
		reserved += size
	}
	return TransientAllocationPlan{Mode: transientPlanForwardBump, Allocations: allocations, Reserved: reserved}
}

func transientPlanReceipt(plan TransientAllocationPlan, graphDigest string, eligible bool, reason string) (TransientPlanReceipt, error) {
	encoded, err := json.Marshal(plan)
	if err != nil {
		return TransientPlanReceipt{}, fmt.Errorf("compute transient plan: encode receipt: %w", err)
	}
	digest := sha256.Sum256(encoded)
	slots := make(map[int]struct{}, len(plan.Allocations))
	var naive int64
	for _, allocation := range plan.Allocations {
		slots[allocation.Slot] = struct{}{}
		naive += allocation.Bytes
	}
	return TransientPlanReceipt{
		Eligible: eligible, FallbackReason: reason, GraphDigest: graphDigest,
		AllocationDigest: hex.EncodeToString(digest[:]), Values: len(plan.Allocations), Slots: len(slots),
		NaiveBytes: naive, ReservedBytes: plan.Reserved, ReusedValues: len(plan.Allocations) - len(slots),
	}, nil
}
