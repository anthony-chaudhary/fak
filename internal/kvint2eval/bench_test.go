package kvint2eval

import (
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func TestBenchmarkKVInt2EvalExecution(t *testing.T) {
	raw, err := os.ReadFile(filepath.Join("testdata", "modeled-delegate.json"))
	if err != nil {
		t.Fatal(err)
	}
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		t.Fatal(err)
	}
	res := Evaluate(req)
	if res.Outcome != Dispatch || res.Reason != ProjectionNeedsRun {
		t.Fatalf("unexpected outcome: %s/%s", res.Outcome, res.Reason)
	}
}

func BenchmarkKVInt2Eval(b *testing.B) {
	raw, err := os.ReadFile(filepath.Join("testdata", "modeled-delegate.json"))
	if err != nil {
		b.Fatal(err)
	}
	var req Request
	if err := json.Unmarshal(raw, &req); err != nil {
		b.Fatal(err)
	}

	b.ResetTimer()
	b.ReportAllocs()
	for i := 0; i < b.N; i++ {
		res := Evaluate(req)
		if res.Outcome != Dispatch || res.Reason != ProjectionNeedsRun {
			b.Fatalf("unexpected outcome: %s/%s", res.Outcome, res.Reason)
		}
	}
}
