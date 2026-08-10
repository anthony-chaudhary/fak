package vcacheqa

import "testing"

func TestCompareHonestyLintLocalKeepsStaticAnalyzerAlternativesExplicit(t *testing.T) {
	got := CompareHonestyLintLocal()
	want := []struct{ name, kind string }{
		{"fak native vcache honesty AST lint", "native"},
		{"tuned non-test text scan", "baseline"},
		{"go/analysis analyzer", "external"},
		{"Semgrep", "external"},
		{"CodeQL", "external"},
		{"golangci-lint custom analyzer", "external"},
	}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, arm := range got.Arms {
		if arm.Name != want[i].name || arm.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, arm)
		}
		if i < 2 {
			if !arm.Available || !arm.Correct || arm.TruePositives != 2 || arm.FalsePositives != 0 || arm.FalseNegatives != 0 || arm.LocationErrors != 0 || arm.ParseFailures != 0 || arm.FilesScanned != 2 || arm.InputBytes == 0 {
				t.Fatalf("local[%d]=%+v", i, arm)
			}
			continue
		}
		if arm.Available || arm.Correct || arm.Latency != 0 || arm.TruePositives != 0 || arm.FalsePositives != 0 || arm.FalseNegatives != 0 || arm.LocationErrors != 0 || arm.ParseFailures != 0 || arm.FilesScanned != 0 || arm.InputBytes != 0 || arm.CPUSeconds != 0 || arm.PeakRSSBytes != 0 || arm.NetworkBytes != 0 || arm.OperatorSeconds != 0 || arm.CostUSD != 0 {
			t.Fatalf("unwitnessed[%d]=%+v", i, arm)
		}
	}
}

func BenchmarkHonestyLint(b *testing.B) {
	dir := b.TempDir()
	if _, err := writeHonestyFixture(dir); err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		defects, err := HonestyLint(dir)
		if err != nil {
			b.Fatal(err)
		}
		if !expectedHonestyFindings(defects) {
			b.Fatalf("defects=%+v", defects)
		}
	}
}
