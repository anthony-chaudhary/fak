package answershape

import "testing"

func TestCompareLocalKeepsAnswerShapeAlternativesExplicit(t *testing.T) {
	r := CompareLocal()
	want := [][2]string{{"fak native multi-signal answer degeneration detector", "native"}, {"exact repeated-line ratio", "baseline"}, {"fak + OpenAI response guard", "integration"}, {"fak + Anthropic response guard", "integration"}, {"llama.cpp repetition controls", "external"}, {"vLLM repetition penalties", "external"}, {"Hugging Face transformers repetition penalty", "external"}, {"NeMo Guardrails output rail", "external"}}
	if len(r.Arms) != len(want) {
		t.Fatal(len(r.Arms))
	}
	for i, w := range want {
		a := r.Arms[i]
		if a.Name != w[0] || a.Kind != w[1] {
			t.Fatal(i, a)
		}
		if i >= 2 && (a.Available || a.Correct || a.Latency != 0 || a.Cases != 0 || a.CostUSD != 0) {
			t.Fatalf("unwitnessed %+v", a)
		}
	}
	if a := r.Arms[0]; !a.Correct || a.TruePositives != 3 || a.TrueNegatives != 1 {
		t.Fatal(a)
	}
	if a := r.Arms[1]; a.Correct || a.FalseNegatives < 2 {
		t.Fatal(a)
	}
}
func BenchmarkAnswerDegenerationDetection(b *testing.B) {
	for i := 0; i < b.N; i++ {
		if a := nativeArm(); !a.Correct {
			b.Fatal(a)
		}
	}
}
