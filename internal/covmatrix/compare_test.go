package covmatrix

import "testing"

func TestCompareLocalKeepsCoverageAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := []struct{ name, kind string }{
		{"fak native model-backend-precision coverage matrix", "native"},
		{"hand-maintained support table lookup", "baseline"},
		{"fak + CUDA runtime witness", "integration"},
		{"fak + Metal runtime witness", "integration"},
		{"fak + Vulkan runtime witness", "integration"},
		{"vLLM supported-model matrix", "external"},
		{"llama.cpp backend and quantization matrix", "external"},
		{"Hugging Face Optimum hardware compatibility", "external"},
		{"ONNX Runtime execution-provider matrix", "external"},
		{"TensorRT-LLM support matrix", "external"},
	}
	if len(r.Arms) != len(want) {
		t.Fatalf("arms=%d want %d", len(r.Arms), len(want))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w.name || a.Kind != w.kind {
			t.Fatalf("arm[%d]=%q/%q want %q/%q", i, a.Name, a.Kind, w.name, w.kind)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.FamilyCells != 0 || a.PrecisionCells != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed arm %q carries measurements: %+v", a.Name, a)
		}
	}
	native := r.Arms[0]
	if !native.Available || !native.Correct || native.PassedChecks != 3 || native.UndefinedCells != 0 {
		t.Fatalf("native result is not complete: %+v", native)
	}
	if native.FamilyCells != len(Families)*len(Backends) || native.PrecisionCells != len(Families)*len(Backends)*len(Precisions) {
		t.Fatalf("native omitted cells: %+v", native)
	}
	baseline := r.Arms[1]
	if !baseline.Available || baseline.Correct || baseline.PassedChecks != 2 || baseline.MissedChecks != 1 {
		t.Fatalf("baseline must expose its missed stale check: %+v", baseline)
	}
}

func BenchmarkModelBackendPrecisionCoverageMatrix(b *testing.B) {
	for i := 0; i < b.N; i++ {
		a := runNativeComparison()
		if !a.Correct {
			b.Fatalf("native matrix failed: %+v", a)
		}
	}
}
