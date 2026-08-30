package model

import (
	"reflect"
	"testing"
)

func TestInlineGraphFunctionsSafeLeafPreservesReferenceOutputAndReceipt(t *testing.T) {
	program := graphInlineWitnessProgram()
	want, err := evaluateGraphInlineProgram(program)
	if err != nil {
		t.Fatal(err)
	}

	gotProgram, receipt, err := InlineGraphFunctions(program, 2)
	if err != nil {
		t.Fatal(err)
	}
	got, err := evaluateGraphInlineProgram(gotProgram)
	if err != nil {
		t.Fatal(err)
	}
	if got != want {
		t.Fatalf("output after inlining=%v want oracle=%v", got, want)
	}

	wantDecisions := []GraphInlineDecision{
		{Function: "large", Action: "keep", Reason: "over-threshold"},
		{Function: "leaf", Action: "inline", Reason: "within-threshold"},
		{Function: "recA", Action: "keep", Reason: "recursive-scc"},
		{Function: "recB", Action: "keep", Reason: "recursive-scc"},
		{Function: "referencedLeaf", Action: "inline", Reason: "within-threshold", Retained: true},
	}
	if !reflect.DeepEqual(receipt.Decisions, wantDecisions) {
		t.Fatalf("decisions=%+v want %+v", receipt.Decisions, wantDecisions)
	}
	if receipt.Digest == "" {
		t.Fatal("receipt digest is empty")
	}
	if hasFunction(gotProgram, "leaf") {
		t.Fatal("unreferenced inlined leaf symbol was retained")
	}
	if !hasFunction(gotProgram, "referencedLeaf") {
		t.Fatal("non-call-referenced inlined function symbol was removed")
	}
	if countCalls(gotProgram, "leaf") != 0 || countCalls(gotProgram, "referencedLeaf") != 0 {
		t.Fatalf("eligible calls survive in output: %+v", gotProgram)
	}
	if countCalls(gotProgram, "large") != 1 {
		t.Fatalf("over-threshold call count=%d want 1", countCalls(gotProgram, "large"))
	}

	reordered := program
	reordered.Functions = append([]GraphInlineFunction(nil), program.Functions...)
	for i, j := 0, len(reordered.Functions)-1; i < j; i, j = i+1, j-1 {
		reordered.Functions[i], reordered.Functions[j] = reordered.Functions[j], reordered.Functions[i]
	}
	reorderedProgram, reorderedReceipt, err := InlineGraphFunctions(reordered, 2)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(reorderedProgram, gotProgram) || !reflect.DeepEqual(reorderedReceipt, receipt) {
		t.Fatalf("receipt depends on declaration order:\nfirst=%+v %+v\nsecond=%+v %+v", gotProgram, receipt, reorderedProgram, reorderedReceipt)
	}
}

func TestInlineGraphFunctionsHonorsExplicitControls(t *testing.T) {
	program := GraphInlineProgram{
		Entry: "main",
		Functions: []GraphInlineFunction{
			{Name: "main", Instructions: []GraphInlineInstruction{{Call: "forced"}, {Call: "blocked"}}},
			{Name: "forced", AlwaysInline: true, Instructions: []GraphInlineInstruction{{Operation: "add", Value: 1}, {Operation: "add", Value: 2}, {Operation: "add", Value: 3}}},
			{Name: "blocked", NeverInline: true, Instructions: []GraphInlineInstruction{{Operation: "add", Value: 4}}},
		},
	}
	got, receipt, err := InlineGraphFunctions(program, 1)
	if err != nil {
		t.Fatal(err)
	}
	if countCalls(got, "forced") != 0 || countCalls(got, "blocked") != 1 {
		t.Fatalf("explicit controls not honored: %+v", got)
	}
	want := []GraphInlineDecision{
		{Function: "blocked", Action: "keep", Reason: "never-inline"},
		{Function: "forced", Action: "inline", Reason: "always-inline"},
	}
	if !reflect.DeepEqual(receipt.Decisions, want) {
		t.Fatalf("decisions=%+v want %+v", receipt.Decisions, want)
	}
}

func graphInlineWitnessProgram() GraphInlineProgram {
	return GraphInlineProgram{
		Entry: "main",
		Functions: []GraphInlineFunction{
			{Name: "main", Instructions: []GraphInlineInstruction{
				{Operation: "add", Value: 1},
				{Call: "leaf"},
				{Call: "large"},
				{Reference: "referencedLeaf"},
				{Call: "referencedLeaf"},
			}},
			{Name: "leaf", Instructions: []GraphInlineInstruction{{Operation: "noop"}, {Operation: "add", Value: 2}}},
			{Name: "large", Instructions: []GraphInlineInstruction{{Operation: "add", Value: 3}, {Operation: "add", Value: 4}, {Operation: "mul", Value: 2}}},
			{Name: "recA", Instructions: []GraphInlineInstruction{{Call: "recB"}}},
			{Name: "recB", Instructions: []GraphInlineInstruction{{Call: "recA"}}},
			{Name: "referencedLeaf", Instructions: []GraphInlineInstruction{{Operation: "add", Value: 5}}},
		},
	}
}

func evaluateGraphInlineProgram(program GraphInlineProgram) (float32, error) {
	functions := make(map[string]GraphInlineFunction, len(program.Functions))
	for _, fn := range program.Functions {
		functions[fn.Name] = fn
	}
	var evaluate func(string, float32) (float32, error)
	evaluate = func(name string, value float32) (float32, error) {
		fn, ok := functions[name]
		if !ok {
			return 0, &missingGraphInlineFunction{name: name}
		}
		for _, instruction := range fn.Instructions {
			switch instruction.Operation {
			case "", "noop":
			case "add":
				value += instruction.Value
			case "mul":
				value *= instruction.Value
			default:
				return 0, &unknownGraphInlineOperation{name: instruction.Operation}
			}
			if instruction.Call != "" {
				var err error
				value, err = evaluate(instruction.Call, value)
				if err != nil {
					return 0, err
				}
			}
		}
		return value, nil
	}
	return evaluate(program.Entry, 0)
}

type missingGraphInlineFunction struct{ name string }

func (e *missingGraphInlineFunction) Error() string { return "missing graph function: " + e.name }

type unknownGraphInlineOperation struct{ name string }

func (e *unknownGraphInlineOperation) Error() string { return "unknown graph operation: " + e.name }

func hasFunction(program GraphInlineProgram, name string) bool {
	for _, fn := range program.Functions {
		if fn.Name == name {
			return true
		}
	}
	return false
}

func countCalls(program GraphInlineProgram, callee string) int {
	count := 0
	for _, fn := range program.Functions {
		for _, instruction := range fn.Instructions {
			if instruction.Call == callee {
				count++
			}
		}
	}
	return count
}
