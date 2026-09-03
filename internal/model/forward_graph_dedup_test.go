package model

import (
	"fmt"
	"reflect"
	"testing"
)

func evaluateDedupWitnessProgram(program GraphInlineProgram) ([]float32, error) {
	functions := make(map[string]GraphInlineFunction, len(program.Functions))
	for _, fn := range program.Functions {
		functions[fn.Name] = fn
	}

	var call func(string, []float32) ([]float32, error)
	call = func(name string, arguments []float32) ([]float32, error) {
		fn, ok := functions[name]
		if !ok {
			return nil, fmt.Errorf("missing function %q", name)
		}
		if len(arguments) != len(fn.Arguments) {
			return nil, fmt.Errorf("function %q got %d arguments, want %d", name, len(arguments), len(fn.Arguments))
		}

		values := make(map[string]float32)
		for i, arg := range fn.Arguments {
			values[arg] = arguments[i]
		}

		for _, inst := range fn.Instructions {
			if inst.Reference != "" {
				continue
			}
			if inst.Call != "" {
				operands := make([]float32, len(inst.Arguments))
				for i, op := range inst.Arguments {
					operands[i] = values[op]
				}
				results, err := call(inst.Call, operands)
				if err != nil {
					return nil, err
				}
				if len(results) != len(inst.Results) {
					return nil, fmt.Errorf("call to %q returned %d results, want %d", inst.Call, len(results), len(inst.Results))
				}
				for i, res := range inst.Results {
					values[res] = results[i]
				}
				continue
			}

			switch inst.Operation {
			case "const":
				values[inst.Results[0]] = inst.Value
			case "copy":
				values[inst.Results[0]] = values[inst.Arguments[0]]
			case "add":
				values[inst.Results[0]] = values[inst.Arguments[0]] + values[inst.Arguments[1]]
			case "sub":
				values[inst.Results[0]] = values[inst.Arguments[0]] - values[inst.Arguments[1]]
			default:
				return nil, fmt.Errorf("unsupported operation %q", inst.Operation)
			}
		}

		results := make([]float32, len(fn.ReturnValues))
		for i, val := range fn.ReturnValues {
			results[i] = values[val]
		}
		return results, nil
	}

	return call(program.Entry, nil)
}

func TestDeduplicateGraphFunctionsWitness(t *testing.T) {
	// First witness requirements (#9973):
	// 1. Two equivalent leaves (e.g. leaf_a, leaf_b)
	// 2. Two callers that become equivalent ONLY after leaf dedup (caller_a, caller_b)
	// 3. One constant-order negative control (neg_ctrl_1 vs neg_ctrl_2)
	// 4. Deterministic representative selection (lexicographically smallest)
	// 5. Cascading call rewrites
	// 6. Exact outputs matching reference evaluator
	// 7. No merge of the negative control

	program := GraphInlineProgram{
		Entry: "main",
		Functions: []GraphInlineFunction{
			{
				Name:         "main",
				Results:      []string{"out1", "out2", "neg1", "neg2"},
				ReturnValues: []string{"res_a", "res_b", "n1", "n2"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 10, Results: []string{"ten"}},
					{Operation: "const", Value: 3, Results: []string{"three"}},
					{Call: "caller_a", Arguments: []string{"ten", "three"}, Results: []string{"res_a"}},
					{Call: "caller_b", Arguments: []string{"ten", "three"}, Results: []string{"res_b"}},
					{Call: "neg_ctrl_1", Results: []string{"n1"}},
					{Call: "neg_ctrl_2", Results: []string{"n2"}},
				},
			},
			// Caller A calls leaf_a
			{
				Name:         "caller_a",
				Arguments:    []string{"x", "y"},
				Results:      []string{"r"},
				ReturnValues: []string{"ret_a"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 1, Results: []string{"one"}},
					{Operation: "add", Arguments: []string{"x", "one"}, Results: []string{"xp1"}},
					{Call: "leaf_a", Arguments: []string{"xp1", "y"}, Results: []string{"ret_a"}},
				},
			},
			// Caller B calls leaf_b (different callee initially, so not equivalent until leaf dedup!)
			{
				Name:         "caller_b",
				Arguments:    []string{"u", "v"},
				Results:      []string{"s"},
				ReturnValues: []string{"ret_b"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 1, Results: []string{"c1"}},
					{Operation: "add", Arguments: []string{"u", "c1"}, Results: []string{"up1"}},
					{Call: "leaf_b", Arguments: []string{"up1", "v"}, Results: []string{"ret_b"}},
				},
			},
			// Equivalent leaf A: takes (p, q), returns p + q
			{
				Name:         "leaf_a",
				Arguments:    []string{"p", "q"},
				Results:      []string{"out_a"},
				ReturnValues: []string{"sum_a"},
				Instructions: []GraphInlineInstruction{
					{Operation: "add", Arguments: []string{"p", "q"}, Results: []string{"sum_a"}},
				},
			},
			// Equivalent leaf B: takes (m, n), returns m + n
			{
				Name:         "leaf_b",
				Arguments:    []string{"m", "n"},
				Results:      []string{"out_b"},
				ReturnValues: []string{"sum_b"},
				Instructions: []GraphInlineInstruction{
					{Operation: "add", Arguments: []string{"m", "n"}, Results: []string{"sum_b"}},
				},
			},
			// Negative control 1: const 10, then const 20, then sub (10 - 20 = -10)
			{
				Name:         "neg_ctrl_1",
				Results:      []string{"nc1"},
				ReturnValues: []string{"nc1_res"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 10, Results: []string{"c_ten"}},
					{Operation: "const", Value: 20, Results: []string{"c_twenty"}},
					{Operation: "sub", Arguments: []string{"c_ten", "c_twenty"}, Results: []string{"nc1_res"}},
				},
			},
			// Negative control 2 (reversed constant order): const 20, then const 10, then sub (20 - 10 = +10)
			{
				Name:         "neg_ctrl_2",
				Results:      []string{"nc2"},
				ReturnValues: []string{"nc2_res"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 20, Results: []string{"c_twenty"}},
					{Operation: "const", Value: 10, Results: []string{"c_ten"}},
					{Operation: "sub", Arguments: []string{"c_twenty", "c_ten"}, Results: []string{"nc2_res"}},
				},
			},
		},
	}

	wantOutput, err := evaluateDedupWitnessProgram(program)
	if err != nil {
		t.Fatalf("evaluate before dedup: %v", err)
	}

	dedupProgram, receipt, err := DeduplicateGraphFunctions(program)
	if err != nil {
		t.Fatalf("DeduplicateGraphFunctions failed: %v", err)
	}

	gotOutput, err := evaluateDedupWitnessProgram(dedupProgram)
	if err != nil {
		t.Fatalf("evaluate after dedup: %v", err)
	}

	// 6. Assert exact outputs match
	if !reflect.DeepEqual(gotOutput, wantOutput) {
		t.Fatalf("program output changed after dedup: got=%v, want=%v", gotOutput, wantOutput)
	}

	// 1 & 4. Verify deterministic leaf dedup: leaf_a selected as representative, leaf_b merged
	var hasLeafA, hasLeafB bool
	var hasCallerA, hasCallerB bool
	var hasNeg1, hasNeg2 bool

	for _, fn := range dedupProgram.Functions {
		switch fn.Name {
		case "leaf_a":
			hasLeafA = true
		case "leaf_b":
			hasLeafB = true
		case "caller_a":
			hasCallerA = true
		case "caller_b":
			hasCallerB = true
		case "neg_ctrl_1":
			hasNeg1 = true
		case "neg_ctrl_2":
			hasNeg2 = true
		}
	}

	if !hasLeafA || hasLeafB {
		t.Fatalf("expected leaf_a retained and leaf_b merged, got leaf_a=%t, leaf_b=%t", hasLeafA, hasLeafB)
	}

	// 2 & 5. Verify cascading caller dedup: caller_b merged into caller_a
	if !hasCallerA || hasCallerB {
		t.Fatalf("expected caller_a retained and caller_b merged, got caller_a=%t, caller_b=%t", hasCallerA, hasCallerB)
	}

	// 7. Verify negative controls NOT merged
	if !hasNeg1 || !hasNeg2 {
		t.Fatalf("negative control was wrongly merged: neg1=%t, neg2=%t", hasNeg1, hasNeg2)
	}

	// 5. Verify cascading call rewrites in main
	var mainFn GraphInlineFunction
	for _, fn := range dedupProgram.Functions {
		if fn.Name == "main" {
			mainFn = fn
			break
		}
	}
	for _, inst := range mainFn.Instructions {
		if inst.Call == "caller_b" {
			t.Fatalf("main still calls caller_b after dedup")
		}
		if inst.Call == "leaf_b" {
			t.Fatalf("main calls leaf_b")
		}
	}

	// Verify caller_a calls leaf_a
	for _, fn := range dedupProgram.Functions {
		if fn.Name == "caller_a" {
			for _, inst := range fn.Instructions {
				if inst.Call == "leaf_b" {
					t.Fatalf("caller_a calls leaf_b instead of leaf_a")
				}
			}
		}
	}

	if receipt.MergedFunctions != 2 {
		t.Fatalf("expected 2 merged functions (leaf_b and caller_b), got %d", receipt.MergedFunctions)
	}
	if receipt.CascadedCalls < 2 {
		t.Fatalf("expected at least 2 rewritten calls, got %d", receipt.CascadedCalls)
	}
	if receipt.Digest == "" {
		t.Fatal("receipt digest is empty")
	}
}
