package residualquant

import "testing"

func TestBenchmarkResidualQuant(t *testing.T) {
	d := PinnedPaperDescriptor()
	req := Request{Descriptor: d, Operation: "inspect", TierBits: 4}
	res := Adjudicate(req)
	if res.Verdict != CaseSupported {
		t.Fatalf("expected supported verdict, got %s", res.Verdict)
	}
}

func BenchmarkResidualQuant(b *testing.B) {
	d := PinnedPaperDescriptor()
	req := Request{Descriptor: d, Operation: "inspect", TierBits: 4}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		res := Adjudicate(req)
		if res.Verdict != CaseSupported {
			b.Fatalf("unexpected verdict: %s", res.Verdict)
		}
	}
}
