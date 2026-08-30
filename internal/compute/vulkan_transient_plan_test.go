package compute

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"testing"
)

func TestPlanGraphTransientsReusesDisjointLifetimesWithParity(t *testing.T) {
	graph := transientReuseGraphFixture()
	values := []TransientValue{
		{Node: "v0", Bytes: 64, LifetimeEnd: "v1"},
		{Node: "v1", Bytes: 64},
		{Node: "v2", Bytes: 64},
		{Node: "v3", Bytes: 64},
		{Node: "v4", Bytes: 64, Escapes: true},
	}

	plan, receipt, err := PlanGraphTransients(graph, values, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !receipt.Eligible || plan.Mode != transientPlanLifetimeReuse {
		t.Fatalf("eligible plan = (%v, %q), want lifetime reuse: %+v", receipt.Eligible, plan.Mode, receipt)
	}
	if receipt.NaiveBytes != 320 || receipt.ReservedBytes != 192 {
		t.Fatalf("memory receipt = naive %d reserved %d, want 320 and 192: %+v", receipt.NaiveBytes, receipt.ReservedBytes, receipt)
	}
	if receipt.Slots != 3 || receipt.ReusedValues != 2 {
		t.Fatalf("reuse receipt = slots %d reused %d, want 3 and 2: %+v", receipt.Slots, receipt.ReusedValues, receipt)
	}
	alloc := transientAllocationsByNode(plan)
	if alloc["v0"].Slot != alloc["v2"].Slot || alloc["v1"].Slot != alloc["v3"].Slot {
		t.Fatalf("disjoint values did not reuse deterministic slots: %+v", plan.Allocations)
	}
	if alloc["v0"].Slot == alloc["v1"].Slot || alloc["v2"].Slot == alloc["v3"].Slot {
		t.Fatalf("overlapping values shared a slot: %+v", plan.Allocations)
	}
	if alloc["v4"].Slot == alloc["v0"].Slot || alloc["v4"].Slot == alloc["v1"].Slot {
		t.Fatalf("escaping output shared transient storage: %+v", plan.Allocations)
	}
	if alloc["v0"].End != alloc["v1"].Start {
		t.Fatalf("explicit lifetime marker was not retained: v0=%+v v1=%+v", alloc["v0"], alloc["v1"])
	}

	got, err := executeTransientReuseGraph(graph, plan, 3)
	if err != nil {
		t.Fatal(err)
	}
	want := eagerTransientReuseGraph(3)
	if got != want {
		t.Fatalf("planned output = %v, independent eager oracle = %v", got, want)
	}

	plan2, receipt2, err := PlanGraphTransients(graph, values, 16)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(plan2, plan) || !reflect.DeepEqual(receipt2, receipt) {
		t.Fatalf("repeat changed deterministic artifact:\nfirst plan=%+v receipt=%+v\nsecond plan=%+v receipt=%+v", plan, receipt, plan2, receipt2)
	}
}

func TestTransientPlanReceiptUsesAllocationDigestName(t *testing.T) {
	payload, err := json.Marshal(TransientPlanReceipt{AllocationDigest: "abc"})
	if err != nil {
		t.Fatal(err)
	}
	got := string(payload)
	if !strings.Contains(got, `"allocation_digest":"abc"`) {
		t.Fatalf("receipt JSON = %s, want allocation_digest", got)
	}
	if strings.Contains(got, `"plan_digest"`) {
		t.Fatalf("receipt JSON = %s, ambiguous plan_digest must not be emitted", got)
	}
}

func TestPlanGraphTransientsFailsClosedToForwardBump(t *testing.T) {
	graph := transientReuseGraphFixture()
	values := []TransientValue{
		{Node: "v0", Bytes: 64, LifetimeEnd: "missing"},
		{Node: "v1", Bytes: 64},
		{Node: "v4", Bytes: 64, Escapes: true},
	}

	plan, receipt, err := PlanGraphTransients(graph, values, 16)
	if err != nil {
		t.Fatal(err)
	}
	if receipt.Eligible || plan.Mode != transientPlanForwardBump {
		t.Fatalf("ineligible plan = (%v, %q), want forward-bump fallback: %+v", receipt.Eligible, plan.Mode, receipt)
	}
	if !strings.Contains(receipt.FallbackReason, "missing") {
		t.Fatalf("fallback reason = %q, want missing marker", receipt.FallbackReason)
	}
	if receipt.Slots != len(values) || receipt.ReusedValues != 0 || receipt.ReservedBytes != receipt.NaiveBytes {
		t.Fatalf("fallback silently reused storage: plan=%+v receipt=%+v", plan, receipt)
	}
}

func transientReuseGraphFixture() Graph {
	return Graph{
		Nodes: []GraphNode{
			{ID: "v4", Op: GraphOpMultiply, Inputs: []NodeID{"v3", "v3"}},
			{ID: "v2", Op: GraphOpMultiply, Inputs: []NodeID{"v1", "v1"}},
			{ID: "v0", Op: GraphOpInput},
			{ID: "v3", Op: GraphOpAdd, Inputs: []NodeID{"v2", "v2"}},
			{ID: "v1", Op: GraphOpAdd, Inputs: []NodeID{"v0", "v0"}},
		},
		Outputs: []NodeID{"v4"},
	}
}

// eagerTransientReuseGraph is intentionally separate from the allocation-plan executor.
func eagerTransientReuseGraph(x float64) float64 {
	v1 := x + x
	v2 := v1 * v1
	v3 := v2 + v2
	return v3 * v3
}

func executeTransientReuseGraph(graph Graph, plan TransientAllocationPlan, input float64) (float64, error) {
	canonical, _, err := CanonicalGraphPipeline().Run(graph)
	if err != nil {
		return 0, err
	}
	allocations := transientAllocationsByNode(plan)
	storage := make(map[int]float64, len(plan.Allocations))
	for _, node := range canonical.Nodes {
		allocation, ok := allocations[node.ID]
		if !ok {
			return 0, fmt.Errorf("test executor: node %q has no allocation", node.ID)
		}
		read := func(id NodeID) (float64, error) {
			inputAllocation, exists := allocations[id]
			if !exists {
				return 0, fmt.Errorf("test executor: input %q has no allocation", id)
			}
			return storage[inputAllocation.Slot], nil
		}
		var value float64
		switch node.Op {
		case GraphOpInput:
			value = input
		case GraphOpAdd, GraphOpMultiply:
			left, readErr := read(node.Inputs[0])
			if readErr != nil {
				return 0, readErr
			}
			right, readErr := read(node.Inputs[1])
			if readErr != nil {
				return 0, readErr
			}
			if node.Op == GraphOpAdd {
				value = left + right
			} else {
				value = left * right
			}
		default:
			return 0, fmt.Errorf("test executor: unsupported op %q", node.Op)
		}
		storage[allocation.Slot] = value
	}
	output := allocations[canonical.Outputs[0]]
	return storage[output.Slot], nil
}

func transientAllocationsByNode(plan TransientAllocationPlan) map[NodeID]TransientAllocation {
	allocations := make(map[NodeID]TransientAllocation, len(plan.Allocations))
	for _, allocation := range plan.Allocations {
		allocations[allocation.Node] = allocation
	}
	return allocations
}
