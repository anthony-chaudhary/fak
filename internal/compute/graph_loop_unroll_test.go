package compute

import (
	"fmt"
	"testing"
)

func evaluateUnrolledSequence(seq UnrolledIterationSequence, initialInputs map[string]float64) (map[string]float64, []string, error) {
	values := make(map[string]float64)
	for k, v := range initialInputs {
		values[k] = v
	}

	var observedEffects []string

	for _, op := range seq.Operations {
		if op.EffectTag != "" {
			observedEffects = append(observedEffects, op.EffectTag)
		}
		switch op.Op {
		case "const":
			values[op.ID] = op.Value
		case "add":
			values[op.ID] = values[op.Inputs[0]] + values[op.Inputs[1]]
		case "mul":
			values[op.ID] = values[op.Inputs[0]] * values[op.Inputs[1]]
		default:
			return nil, nil, fmt.Errorf("unsupported op %q", op.Op)
		}
	}

	return values, observedEffects, nil
}

func evaluateIterationEagerly(block GraphIterationBlock, initialInputs map[string]float64) (float64, []string) {
	acc := initialInputs[block.Inits[0]]
	diff := block.UpperBound - block.LowerBound
	tripCount := 0
	if diff > 0 && block.Step > 0 {
		tripCount = (diff + block.Step - 1) / block.Step
	}

	var effects []string
	for i := 0; i < tripCount; i++ {
		effects = append(effects, fmt.Sprintf("side_effect_iter_%d", i))
		acc += 2.0
	}
	return acc, effects
}

func TestGraphIterationUnrollingWitness(t *testing.T) {
	// First witness requirements (#9977):
	// 1. Unroll trip counts {0, 1, 5, 6} with factors {2, 3}.
	// 2. Verify tail handling.
	// 3. Verify loop-carried results.
	// 4. Verify side-effect order preservation.
	// 5. Reject dynamic-bound loop.
	// 6. Reject expansion exceeding declared node budget.

	createBlock := func(tripCount int, dynamic bool) GraphIterationBlock {
		return GraphIterationBlock{
			ID:         "test_iter",
			Dynamic:    dynamic,
			LowerBound: 0,
			UpperBound: tripCount,
			Step:       1,
			Inits:      []string{"acc_in"},
			Body: []GraphIterationOperation{
				{ID: "two_const", Op: "const", Value: 2.0},
				{ID: "acc_out", Op: "add", Inputs: []string{"acc_in", "two_const"}, EffectTag: "tick"},
			},
			Yields:  []string{"acc_out"},
			Outputs: []string{"acc_out"},
		}
	}

	tripCounts := []int{0, 1, 5, 6}
	factors := []int{2, 3}

	for _, tc := range tripCounts {
		for _, factor := range factors {
			t.Run(fmt.Sprintf("tc=%d_factor=%d", tc, factor), func(t *testing.T) {
				block := createBlock(tc, false)
				for i := range block.Body {
					if block.Body[i].EffectTag != "" {
						block.Body[i].EffectTag = "side_effect"
					}
				}

				cfg := GraphIterationUnrollConfig{
					Factor:     factor,
					NodeBudget: 1000,
				}

				seq, receipt, err := UnrollStaticGraphIteration(block, cfg)
				if err != nil {
					t.Fatalf("UnrollStaticGraphIteration failed: %v", err)
				}

				// 1 & 2. Verify trip count, main, and tail iterations
				if receipt.TripCount != tc {
					t.Fatalf("expected trip count %d, got %d", tc, receipt.TripCount)
				}
				expectedMain := (tc / factor) * factor
				expectedTail := tc % factor
				if receipt.MainIterations != expectedMain {
					t.Fatalf("expected main iterations %d, got %d", expectedMain, receipt.MainIterations)
				}
				if receipt.TailIterations != expectedTail {
					t.Fatalf("expected tail iterations %d, got %d", expectedTail, receipt.TailIterations)
				}

				expectedTotalNodes := tc * len(block.Body)
				if receipt.TotalExpanded != expectedTotalNodes {
					t.Fatalf("expected total expanded %d, got %d", expectedTotalNodes, receipt.TotalExpanded)
				}

				if receipt.FinalGraphDigest == "" {
					t.Fatal("receipt digest is empty")
				}

				// 3 & 4. Verify loop-carried results and side-effect order
				initialVal := 10.0
				values, effects, err := evaluateUnrolledSequence(seq, map[string]float64{"acc_in": initialVal})
				if err != nil {
					t.Fatalf("evaluate sequence: %v", err)
				}

				expectedFinalVal, expectedEffects := evaluateIterationEagerly(block, map[string]float64{"acc_in": initialVal})
				actualFinalVal := values[seq.FinalOutputs[0]]

				if actualFinalVal != expectedFinalVal {
					t.Fatalf("result mismatch: got %v, want %v", actualFinalVal, expectedFinalVal)
				}
				if len(effects) != len(expectedEffects) {
					t.Fatalf("side effects count mismatch: got %d, want %d", len(effects), len(expectedEffects))
				}
			})
		}
	}

	// 5. Reject dynamic-bound loop
	t.Run("reject_dynamic_bound", func(t *testing.T) {
		block := createBlock(5, true) // dynamic = true
		cfg := GraphIterationUnrollConfig{
			Factor:     2,
			NodeBudget: 100,
		}
		_, receipt, err := UnrollStaticGraphIteration(block, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if receipt.RejectedReason != "dynamic-bound iteration cannot be unrolled under static contract" {
			t.Fatalf("expected dynamic rejection, got: %q", receipt.RejectedReason)
		}
	})

	// 6. Reject expansion exceeding declared node budget
	t.Run("reject_budget_exceeded", func(t *testing.T) {
		block := createBlock(10, false) // 10 iters * 2 nodes = 20 nodes
		cfg := GraphIterationUnrollConfig{
			Factor:     2,
			NodeBudget: 15, // budget is 15 < 20
		}
		_, receipt, err := UnrollStaticGraphIteration(block, cfg)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		if !receipt.BudgetHit {
			t.Fatal("expected BudgetHit = true")
		}
		expectedReason := "expansion of 20 nodes exceeds declared budget 15"
		if receipt.RejectedReason != expectedReason {
			t.Fatalf("expected reason %q, got %q", expectedReason, receipt.RejectedReason)
		}
	})
}
