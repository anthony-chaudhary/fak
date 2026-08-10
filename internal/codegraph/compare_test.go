package codegraph

import "testing"

func TestCompareLocalKeepsCallGraphAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := []struct{ name, kind string }{
		{"fak native syntactic Go call graph", "native"},
		{"go/ast direct-call scan", "baseline"},
		{"golang.org/x/tools/go/callgraph", "external"},
		{"gopls call hierarchy", "external"},
		{"Go guru callers and callees", "external"},
		{"CodeQL Go call graph", "external"},
		{"SCIP Go code intelligence graph", "external"},
	}
	if len(r.Arms) != len(want) {
		t.Fatalf("arms=%d want %d", len(r.Arms), len(want))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w.name || a.Kind != w.kind {
			t.Fatalf("arm[%d]=%q/%q want %q/%q", i, a.Name, a.Kind, w.name, w.kind)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Nodes != 0 || a.DirectEdges != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed arm %q carries measurements: %+v", a.Name, a)
		}
	}
	native := r.Arms[0]
	if !native.Available || !native.Correct || native.PassedChecks != 4 || native.Nodes != 4 || native.ForwardHits != 3 || native.ReverseHits != 3 {
		t.Fatalf("native result incomplete: %+v", native)
	}
	baseline := r.Arms[1]
	if !baseline.Available || baseline.Correct || baseline.PassedChecks != 2 || baseline.MissedChecks != 2 {
		t.Fatalf("baseline must expose missing graph traversals: %+v", baseline)
	}
}

func BenchmarkSyntacticGoCallGraph(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := runNativeComparison()
		if !a.Correct {
			b.Fatalf("native call graph failed: %+v", a)
		}
	}
}
