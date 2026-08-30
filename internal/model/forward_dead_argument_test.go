package model

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sort"
	"testing"
)

func TestEliminateDeadGraphArgumentsPreservesOutputAndABIFences(t *testing.T) {
	program := deadArgumentWitnessProgram()
	wantOutput, err := evaluateDeadArgumentProgram(program)
	if err != nil {
		t.Fatal(err)
	}
	wantSignatures, err := serializeFencedGraphSignatures(program)
	if err != nil {
		t.Fatal(err)
	}

	got, receipt, err := EliminateDeadGraphArguments(program)
	if err != nil {
		t.Fatal(err)
	}
	gotOutput, err := evaluateDeadArgumentProgram(got)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(gotOutput, wantOutput) {
		t.Fatalf("graph output after elimination=%v want oracle=%v", gotOutput, wantOutput)
	}
	gotSignatures, err := serializeFencedGraphSignatures(got)
	if err != nil {
		t.Fatal(err)
	}
	if string(gotSignatures) != string(wantSignatures) {
		t.Fatalf("serialized fenced signatures changed:\nbefore=%s\nafter =%s", wantSignatures, gotSignatures)
	}

	wantDecisions := []GraphDeadArgumentDecision{
		{Function: "addressed", Fence: "symbol-reference"},
		{Function: "cached", RemovedArgs: []int{0}},
		{Function: "exported", Fence: "exported-abi"},
		{Function: "external", Fence: "external-abi"},
		{Function: "indirect", Fence: "indirect-call"},
		{Function: "leaf", RemovedArgs: []int{1}, RemovedResults: []int{1}},
		{Function: "main", Fence: "entry-abi"},
		{Function: "middle", RemovedArgs: []int{1}, RemovedResults: []int{1}},
	}
	if !reflect.DeepEqual(receipt.Decisions, wantDecisions) {
		t.Fatalf("decisions=%+v want %+v", receipt.Decisions, wantDecisions)
	}
	if receipt.Digest == "" {
		t.Fatal("receipt digest is empty")
	}

	assertGraphSignature(t, got, "middle", []string{"live"}, []string{"value"})
	assertGraphSignature(t, got, "leaf", []string{"live"}, []string{"value"})
	assertGraphCall(t, got, "main", "middle", []string{"live"}, []string{"out"})
	assertGraphCall(t, got, "middle", "leaf", []string{"live"}, []string{"leafLive"})
	assertGraphSignature(t, got, "cached", nil, []string{"first", "cached"})
}

func deadArgumentWitnessProgram() GraphInlineProgram {
	return GraphInlineProgram{
		Entry: "main",
		Functions: []GraphInlineFunction{
			{
				Name: "main", Results: []string{"answer"}, ReturnValues: []string{"out"},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Value: 4, Results: []string{"live"}},
					{Operation: "const", Value: 99, Results: []string{"dead"}},
					{Call: "middle", Arguments: []string{"live", "dead"}, Results: []string{"out", "discard"}},
					{Call: "exported", Arguments: []string{"live", "dead"}, Results: []string{"exportedOut", "exportedDiscard"}},
					{Call: "external", Arguments: []string{"live", "dead"}, Results: []string{"externalOut", "externalDiscard"}},
					{Call: "indirect", Arguments: []string{"live", "dead"}, Results: []string{"indirectOut", "indirectDiscard"}},
					{Reference: "addressed"},
				},
			},
			{
				Name: "middle", Arguments: []string{"live", "dead"}, Results: []string{"value", "discard"}, ReturnValues: []string{"leafLive", "leafDead"},
				Instructions: []GraphInlineInstruction{{Call: "leaf", Arguments: []string{"live", "dead"}, Results: []string{"leafLive", "leafDead"}}},
			},
			{
				Name: "leaf", Arguments: []string{"live", "dead"}, Results: []string{"value", "discard"}, ReturnValues: []string{"copied", "unused"},
				Instructions: []GraphInlineInstruction{
					{Operation: "copy", Arguments: []string{"live"}, Results: []string{"copied"}},
					{Operation: "const", Value: 7, Results: []string{"unused"}},
				},
			},
			fencedGraphFunction("exported", func(fn *GraphInlineFunction) { fn.Exported = true }),
			fencedGraphFunction("external", func(fn *GraphInlineFunction) { fn.External = true }),
			fencedGraphFunction("indirect", func(fn *GraphInlineFunction) { fn.IndirectCallable = true }),
			fencedGraphFunction("addressed", nil),
			{
				Name: "cached", Arguments: []string{"unused"}, Results: []string{"first", "cached"}, ReturnValues: []string{"zero", "one"}, CachedResultIndexes: []int{1},
				Instructions: []GraphInlineInstruction{
					{Operation: "const", Results: []string{"zero"}},
					{Operation: "const", Value: 1, Results: []string{"one"}},
				},
			},
		},
	}
}

func fencedGraphFunction(name string, mark func(*GraphInlineFunction)) GraphInlineFunction {
	fn := GraphInlineFunction{
		Name: name, Arguments: []string{"live", "dead"}, Results: []string{"value", "discard"}, ReturnValues: []string{"copied", "unused"},
		Instructions: []GraphInlineInstruction{
			{Operation: "copy", Arguments: []string{"live"}, Results: []string{"copied"}},
			{Operation: "const", Value: 8, Results: []string{"unused"}},
		},
	}
	if mark != nil {
		mark(&fn)
	}
	return fn
}

func evaluateDeadArgumentProgram(program GraphInlineProgram) ([]float32, error) {
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
		for i, argument := range fn.Arguments {
			values[argument] = arguments[i]
		}
		for _, instruction := range fn.Instructions {
			if instruction.Reference != "" {
				continue
			}
			if instruction.Call != "" {
				operands := make([]float32, len(instruction.Arguments))
				for i, operand := range instruction.Arguments {
					operands[i] = values[operand]
				}
				results, err := call(instruction.Call, operands)
				if err != nil {
					return nil, err
				}
				if len(results) != len(instruction.Results) {
					return nil, fmt.Errorf("call to %q returned %d results, want %d", instruction.Call, len(results), len(instruction.Results))
				}
				for i, result := range instruction.Results {
					values[result] = results[i]
				}
				continue
			}
			switch instruction.Operation {
			case "const":
				values[instruction.Results[0]] = instruction.Value
			case "copy":
				values[instruction.Results[0]] = values[instruction.Arguments[0]]
			case "":
			default:
				return nil, fmt.Errorf("unsupported operation %q", instruction.Operation)
			}
		}
		results := make([]float32, len(fn.ReturnValues))
		for i, value := range fn.ReturnValues {
			results[i] = values[value]
		}
		return results, nil
	}
	return call(program.Entry, nil)
}

func serializeFencedGraphSignatures(program GraphInlineProgram) ([]byte, error) {
	referenced := make(map[string]bool)
	for _, fn := range program.Functions {
		for _, instruction := range fn.Instructions {
			referenced[instruction.Reference] = instruction.Reference != ""
		}
	}
	type signature struct {
		Name      string   `json:"name"`
		Arguments []string `json:"arguments"`
		Results   []string `json:"results"`
	}
	var signatures []signature
	for _, fn := range program.Functions {
		if fn.Exported || fn.External || fn.IndirectCallable || referenced[fn.Name] {
			signatures = append(signatures, signature{fn.Name, fn.Arguments, fn.Results})
		}
	}
	sort.Slice(signatures, func(i, j int) bool { return signatures[i].Name < signatures[j].Name })
	return json.Marshal(signatures)
}

func assertGraphSignature(t *testing.T, program GraphInlineProgram, name string, arguments, results []string) {
	t.Helper()
	for _, fn := range program.Functions {
		if fn.Name == name {
			if !reflect.DeepEqual(fn.Arguments, arguments) || !reflect.DeepEqual(fn.Results, results) {
				t.Fatalf("%s signature=(%v)->(%v), want (%v)->(%v)", name, fn.Arguments, fn.Results, arguments, results)
			}
			return
		}
	}
	t.Fatalf("function %q missing", name)
}

func assertGraphCall(t *testing.T, program GraphInlineProgram, caller, callee string, arguments, results []string) {
	t.Helper()
	for _, fn := range program.Functions {
		if fn.Name != caller {
			continue
		}
		for _, instruction := range fn.Instructions {
			if instruction.Call == callee {
				if !reflect.DeepEqual(instruction.Arguments, arguments) || !reflect.DeepEqual(instruction.Results, results) {
					t.Fatalf("%s call to %s=(%v)->(%v), want (%v)->(%v)", caller, callee, instruction.Arguments, instruction.Results, arguments, results)
				}
				return
			}
		}
	}
	t.Fatalf("call from %q to %q missing", caller, callee)
}
