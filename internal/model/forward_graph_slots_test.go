package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestPromoteRegionSlotsCarriesBranchAndLoopState(t *testing.T) {
	graph := compute.RegionSlotGraph{Ops: []compute.RegionSlotOp{
		{Kind: compute.RegionSlotDeclare, Slot: "state", Debug: "residual"},
		{Kind: compute.RegionSlotIf, Name: "choose", Then: []compute.RegionSlotOp{
			{Kind: compute.RegionSlotStore, Slot: "state", Value: "left", Debug: "residual.left"},
		}, Else: []compute.RegionSlotOp{
			{Kind: compute.RegionSlotStore, Slot: "state", Value: "right", Debug: "residual.right"},
		}},
		{Kind: compute.RegionSlotLoop, Name: "advance", Body: []compute.RegionSlotOp{
			{Kind: compute.RegionSlotLoad, Name: "prior", Slot: "state"},
			{Kind: compute.RegionSlotStore, Slot: "state", Value: "next", Debug: "residual.next"},
		}},
		{Kind: compute.RegionSlotLoad, Name: "result", Slot: "state"},
	}}

	got, receipt, err := PromoteRegionSlots(graph)
	if err != nil {
		t.Fatal(err)
	}
	wantReceipt := compute.RegionSlotReceipt{Promotions: []compute.RegionSlotPromotion{{Slot: "state", Action: "promote"}}}
	if !reflect.DeepEqual(receipt, wantReceipt) {
		t.Fatalf("receipt = %#v, want %#v", receipt, wantReceipt)
	}
	if countSlotMemoryOps(got.Ops) != 0 {
		t.Fatalf("promoted graph retains slot memory operations: %#v", got)
	}

	branch := got.Ops[0]
	wantBranchCarry := []compute.RegionSlotCarry{{Slot: "state", Input: "undef.state", Output: "choose.state", Debug: "residual.left"}}
	if !reflect.DeepEqual(branch.Carries, wantBranchCarry) {
		t.Fatalf("branch carries = %#v, want %#v", branch.Carries, wantBranchCarry)
	}
	loop := got.Ops[1]
	wantLoopCarry := []compute.RegionSlotCarry{{Slot: "state", Input: "choose.state", Argument: "advance.state.arg", Output: "advance.state", Debug: "residual.next"}}
	if !reflect.DeepEqual(loop.Carries, wantLoopCarry) {
		t.Fatalf("loop carries = %#v, want %#v", loop.Carries, wantLoopCarry)
	}
	if gotValue := loop.Body[0].Value; gotValue != "advance.state.arg" {
		t.Fatalf("loop load value = %q, want region argument", gotValue)
	}
	if gotValue := got.Ops[2].Value; gotValue != "advance.state" {
		t.Fatalf("final load value = %q, want loop result", gotValue)
	}
	if gotDebug := got.Ops[2].Debug; gotDebug != "residual.next" {
		t.Fatalf("final debug binding = %q, want residual.next", gotDebug)
	}
	wantOutput := evalMemorySlotGraph(t, graph.Ops)
	if gotOutput := evalSSARegionGraph(t, got.Ops); gotOutput != wantOutput {
		t.Fatalf("promoted output = %q, eager oracle = %q", gotOutput, wantOutput)
	}

	reordered := compute.RegionSlotGraph{Ops: []compute.RegionSlotOp{
		{Kind: compute.RegionSlotDeclare, Slot: "z"},
		{Kind: compute.RegionSlotDeclare, Slot: "a"},
	}}
	_, orderedReceipt, err := PromoteRegionSlots(reordered)
	if err != nil {
		t.Fatal(err)
	}
	if got := []string{orderedReceipt.Promotions[0].Slot, orderedReceipt.Promotions[1].Slot}; !reflect.DeepEqual(got, []string{"a", "z"}) {
		t.Fatalf("promotion order = %v, want deterministic lexical order", got)
	}
}

func TestPromoteRegionSlotsKeepsUnknownRegionUseMemoryBacked(t *testing.T) {
	graph := compute.RegionSlotGraph{Ops: []compute.RegionSlotOp{
		{Kind: compute.RegionSlotDeclare, Slot: "opaque"},
		{Kind: compute.RegionSlotUnknown, Name: "extension", Body: []compute.RegionSlotOp{
			{Kind: compute.RegionSlotStore, Slot: "opaque", Value: "external"},
		}},
		{Kind: compute.RegionSlotLoad, Name: "result", Slot: "opaque"},
	}}

	got, receipt, err := PromoteRegionSlots(graph)
	if err != nil {
		t.Fatal(err)
	}
	want := compute.RegionSlotReceipt{Promotions: []compute.RegionSlotPromotion{{Slot: "opaque", Action: "keep", Reason: "unknown-region-use"}}}
	if !reflect.DeepEqual(receipt, want) {
		t.Fatalf("receipt = %#v, want %#v", receipt, want)
	}
	if countSlotMemoryOps(got.Ops) != 3 {
		t.Fatalf("unknown-region graph changed memory operations: %#v", got)
	}
}

func countSlotMemoryOps(ops []compute.RegionSlotOp) int {
	count := 0
	for _, op := range ops {
		switch op.Kind {
		case compute.RegionSlotDeclare, compute.RegionSlotLoad, compute.RegionSlotStore:
			count++
		}
		count += countSlotMemoryOps(op.Then)
		count += countSlotMemoryOps(op.Else)
		count += countSlotMemoryOps(op.Body)
	}
	return count
}

func evalMemorySlotGraph(t *testing.T, ops []compute.RegionSlotOp) string {
	t.Helper()
	memory := make(map[string]string)
	var result string
	var run func([]compute.RegionSlotOp)
	run = func(region []compute.RegionSlotOp) {
		for _, op := range region {
			switch op.Kind {
			case compute.RegionSlotDeclare:
				memory[op.Slot] = "undef." + op.Slot
			case compute.RegionSlotStore:
				memory[op.Slot] = op.Value
			case compute.RegionSlotLoad:
				if op.Name == "result" {
					result = memory[op.Slot]
				}
			case compute.RegionSlotIf:
				run(op.Then)
			case compute.RegionSlotLoop:
				run(op.Body)
			case compute.RegionSlotUnknown:
				run(op.Body)
			}
		}
	}
	run(ops)
	return result
}

func evalSSARegionGraph(t *testing.T, ops []compute.RegionSlotOp) string {
	t.Helper()
	values := make(map[string]string)
	var result string
	resolve := func(value string) string {
		if resolved, ok := values[value]; ok {
			return resolved
		}
		return value
	}
	var run func([]compute.RegionSlotOp)
	run = func(region []compute.RegionSlotOp) {
		for _, op := range region {
			switch op.Kind {
			case compute.RegionSlotConst:
				value := resolve(op.Value)
				if op.Name != "" {
					values[op.Name] = value
				}
				if op.Name == "result" {
					result = value
				}
			case compute.RegionSlotIf:
				run(op.Then)
				for _, carry := range op.Carries {
					values[carry.Output] = values["yield."+carry.Slot]
				}
			case compute.RegionSlotLoop:
				for _, carry := range op.Carries {
					values[carry.Argument] = resolve(carry.Input)
				}
				run(op.Body)
				for _, carry := range op.Carries {
					values[carry.Output] = values["yield."+carry.Slot]
				}
			default:
				t.Fatalf("unexpected memory operation in promoted graph: %#v", op)
			}
		}
	}
	run(ops)
	return result
}
