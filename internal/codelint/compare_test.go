package codelint

import "testing"

func TestCompareGoSyntaxLocalKeepsToolchainAlternativesExplicit(t *testing.T) {
	got := CompareGoSyntaxLocal()
	want := []struct{ name, kind string }{{"fak native Go syntax pack", "native"}, {"go/parser first-error-only", "baseline"}, {"go test compile", "external"}, {"gofmt", "external"}, {"go vet", "external"}, {"staticcheck", "external"}, {"golangci-lint", "external"}, {"gopls diagnostics", "external"}}
	if len(got.Arms) != len(want) {
		t.Fatalf("arms=%d", len(got.Arms))
	}
	for i, a := range got.Arms {
		if a.Name != want[i].name || a.Kind != want[i].kind {
			t.Fatalf("arm[%d]=%+v", i, a)
		}
		if i < 2 {
			if !a.Available || !a.Correct || a.Files != 4 || a.CorrectFiles != 4 || a.InputBytes == 0 {
				t.Fatalf("local=%+v", a)
			}
			continue
		}
		if a.Available || a.Correct || a.Latency != 0 || a.Files != 0 || a.CorrectFiles != 0 || a.FalseSyntaxErrors != 0 || a.MissedSyntaxErrors != 0 || a.LocationErrors != 0 || a.ReportedErrors != 0 || a.CPUSeconds != 0 || a.PeakRSSBytes != 0 || a.InputBytes != 0 || a.NetworkBytes != 0 || a.OperatorSeconds != 0 || a.CostUSD != 0 {
			t.Fatalf("unwitnessed=%+v", a)
		}
	}
	if got.Arms[0].ReportedErrors <= got.Arms[1].ReportedErrors {
		t.Fatalf("expected all-error native to report more detail: native=%+v baseline=%+v", got.Arms[0], got.Arms[1])
	}
}
func BenchmarkGoSyntaxPackCorpus(b *testing.B) {
	dir := b.TempDir()
	n, err := writeSyntaxCases(dir)
	if err != nil {
		b.Fatal(err)
	}
	b.ReportAllocs()
	b.SetBytes(n)
	for i := 0; i < b.N; i++ {
		if a := runNativeSyntax(dir, n); !a.Correct {
			b.Fatalf("arm=%+v", a)
		}
	}
}
