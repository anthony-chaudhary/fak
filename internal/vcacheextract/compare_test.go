package vcacheextract

import (
	"os"
	"path/filepath"
	"testing"
)

func TestCompareLocalKeepsTokenSanitizationAlternativesExplicit(t *testing.T) {
	got := CompareLocal()
	want := []struct{ name, kind string }{
		{"fak native Codex token sanitizer", "native"},
		{"raw JSONL pass-through", "baseline"},
		{"fak + OpenTelemetry", "integration"},
		{"fak + Prometheus", "integration"},
		{"jq streaming projection", "external"},
		{"Vector VRL remap", "external"},
		{"Fluent Bit filter pipeline", "external"},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, arm)
		}
		if i < 2 {
			if !arm.Available || arm.InputRows != 4 || arm.EligibleRows != 2 || arm.InputBytes == 0 || arm.OutputBytes == 0 {
				t.Fatalf("local[%d]=%+v", i, arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.CounterCorrect || arm.Latency != 0 || arm.InputRows != 0 || arm.EligibleRows != 0 || arm.OutputRows != 0 || arm.MissedRows != 0 || arm.ExtraRows != 0 || arm.ForbiddenFields != 0 || arm.ForbiddenBytes != 0 || arm.ParseFailures != 0 || arm.InputBytes != 0 || arm.OutputBytes != 0 || arm.CPUSeconds != 0 || arm.PeakRSSBytes != 0 || arm.NetworkBytes != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed[%d]=%+v", i, arm)
		}
	}
	native, baseline := got.Arms[0], got.Arms[1]
	if !native.Correct || !native.CounterCorrect || native.OutputRows != 2 || native.ForbiddenFields != 0 || native.ForbiddenBytes != 0 {
		t.Fatalf("native=%+v", native)
	}
	if baseline.Correct || !baseline.CounterCorrect || baseline.OutputRows != 4 || baseline.ExtraRows != 2 || baseline.ForbiddenFields == 0 || baseline.ForbiddenBytes == 0 {
		t.Fatalf("baseline=%+v", baseline)
	}
}

func BenchmarkExtractSanitizedTokenRows(b *testing.B) {
	path := filepath.Join(b.TempDir(), "session.jsonl")
	if err := os.WriteFile(path, comparisonFixture(), 0o600); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(int64(len(comparisonFixture())))
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		rows, err := ExtractRows(path)
		if err != nil {
			b.Fatal(err)
		}
		if !exactComparisonCounters(rows) {
			b.Fatalf("rows=%v", rows)
		}
	}
}
