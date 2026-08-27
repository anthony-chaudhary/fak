package qwen4exp

import "testing"

func cell(arm string) FrontierCell {
	return FrontierCell{Cohort: "8k-b1-c1", Arm: arm, Backend: "cuda", Artifact: "sha256:x", DType: "bf16", Hardware: "matched", PromptSet: "p", Context: 8192, Batch: 1, Concurrency: 1, Engine: "fak-native", Fallback: "none", Quality: true, Supported: true, LoadMS: 1, TTFTMS: 1, PrefillTPS: 2, DecodeTPS: 1, PeakBytes: 1}
}
func TestFrontierRanksOnlyMatchedQualityPassingCells(t *testing.T) {
	a := cell("a")
	b := cell("b")
	b.DecodeTPS = 2
	u := FrontierCell{Arm: "mlx", Backend: "metal", Supported: false, Reason: "exact artifact unsupported"}
	f := Frontier{Schema: FrontierSchema, Cells: []FrontierCell{a, b, u}}
	r, err := f.Ranked()
	if err != nil {
		t.Fatal(err)
	}
	if len(r) != 2 || r[0].Arm != "b" {
		t.Fatal(r)
	}
}
func TestFrontierRejectsUnmatchedAndNativeFallback(t *testing.T) {
	a := cell("a")
	b := cell("b")
	b.Context = 32768
	if _, err := (Frontier{Schema: FrontierSchema, Cells: []FrontierCell{a, b}}).Ranked(); err == nil {
		t.Fatal("unmatched ranked")
	}
	a.Fallback = "llama.cpp"
	if _, err := (Frontier{Schema: FrontierSchema, Cells: []FrontierCell{a}}).Ranked(); err == nil {
		t.Fatal("fallback accepted")
	}
}
func TestUnsupportedArmMustStayTyped(t *testing.T) {
	f := Frontier{Schema: FrontierSchema, Cells: []FrontierCell{{Arm: "mlx", Supported: false}}}
	if _, err := f.Ranked(); err == nil {
		t.Fatal("untyped unsupported")
	}
}
