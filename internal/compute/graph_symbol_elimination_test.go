package compute

import (
	"reflect"
	"testing"
)

func TestSelectReachableGraphSymbolsMatchesReferenceOracle(t *testing.T) {
	module := GraphSymbolModule{Symbols: []GraphSymbol{
		{Name: "entry", Kind: GraphSymbolFunction, Exported: true, Calls: []string{"direct"}, References: []string{"addressed"}, AttributeReferences: []string{"metadata"}},
		{Name: "unused", Kind: GraphSymbolFunction},
		{Name: "direct", Kind: GraphSymbolFunction, Calls: []string{"recursive_a"}},
		{Name: "addressed", Kind: GraphSymbolFunction},
		{Name: "metadata", Kind: GraphSymbolConstant},
		{Name: "recursive_a", Kind: GraphSymbolFunction, Calls: []string{"recursive_b"}},
		{Name: "recursive_b", Kind: GraphSymbolFunction, Calls: []string{"recursive_a"}},
	}}
	original := cloneGraphSymbolModule(module)

	got, receipt := SelectReachableGraphSymbols(module)
	if !receipt.Selected {
		t.Fatalf("selector fell back: %+v", receipt)
	}
	wantNames := referenceReachableGraphSymbolNames(module)
	if gotNames := graphSymbolNames(got); !reflect.DeepEqual(gotNames, wantNames) {
		t.Fatalf("selected symbols = %v, independent reference = %v", gotNames, wantNames)
	}
	if !reflect.DeepEqual(receipt.Removed, []string{"unused"}) {
		t.Fatalf("removed symbols = %v, want [unused]", receipt.Removed)
	}
	if !reflect.DeepEqual(module, original) {
		t.Fatal("selector mutated its input module")
	}

	second, secondReceipt := SelectReachableGraphSymbols(module)
	if !secondReceipt.Selected || secondReceipt.FinalModuleDigest != receipt.FinalModuleDigest || !reflect.DeepEqual(second, got) {
		t.Fatalf("selector is not deterministic: first=%+v/%+v second=%+v/%+v", got, receipt, second, secondReceipt)
	}
}

func TestSelectReachableGraphSymbolsFallsBackOnUnsupportedModule(t *testing.T) {
	module := GraphSymbolModule{Symbols: []GraphSymbol{{Name: "entry", Kind: GraphSymbolKind("external"), Exported: true}}}
	got, receipt := SelectReachableGraphSymbols(module)
	if receipt.Selected || receipt.FallbackReason == "" {
		t.Fatalf("unsupported module receipt = %+v, want explicit fallback", receipt)
	}
	if !reflect.DeepEqual(got, module) {
		t.Fatalf("fallback module = %+v, want unchanged %+v", got, module)
	}
}

// referenceReachableGraphSymbolNames is intentionally separate from the production selector:
// it uses a simple fixed-point scan rather than the selector's indexed worklist.
func referenceReachableGraphSymbolNames(module GraphSymbolModule) []string {
	reachable := make(map[string]bool)
	for _, symbol := range module.Symbols {
		if symbol.Exported {
			reachable[symbol.Name] = true
		}
	}
	for changed := true; changed; {
		changed = false
		for _, symbol := range module.Symbols {
			if !reachable[symbol.Name] {
				continue
			}
			for _, reference := range append(append(append([]string(nil), symbol.Calls...), symbol.References...), symbol.AttributeReferences...) {
				if !reachable[reference] {
					reachable[reference] = true
					changed = true
				}
			}
		}
	}
	names := make([]string, 0, len(reachable))
	for _, symbol := range module.Symbols {
		if reachable[symbol.Name] {
			names = append(names, symbol.Name)
		}
	}
	return names
}

func graphSymbolNames(module GraphSymbolModule) []string {
	names := make([]string, 0, len(module.Symbols))
	for _, symbol := range module.Symbols {
		names = append(names, symbol.Name)
	}
	return names
}
