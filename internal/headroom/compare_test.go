package headroom

import "testing"

func TestCompareBenchRunsNativeAndNoCompressionOnSameCorpus(t *testing.T) {
	inputs := BenchCorpus()
	report := CompareBench([]string{"none", NativeName}, inputs)
	if !report.Complete || len(report.Arms) != 2 || report.Corpus != len(inputs) {
		t.Fatalf("report=%+v", report)
	}
	if report.Arms[0].Report.OrigTotal != report.Arms[1].Report.OrigTotal {
		t.Fatalf("arms saw different bytes: %+v", report.Arms)
	}
	if report.Arms[0].Report.NewTotal != report.Arms[0].Report.OrigTotal {
		t.Fatalf("no-compression new bytes=%d", report.Arms[0].Report.NewTotal)
	}
	if report.Arms[1].Report.NewTotal >= report.Arms[1].Report.OrigTotal {
		t.Fatalf("native did not reduce bytes: %+v", report.Arms[1].Report)
	}
}

func TestCompareBenchNamesMissingIntegration(t *testing.T) {
	report := CompareBench([]string{"none", NativeName, "lingua"}, BenchCorpus())
	if report.Complete || len(report.Missing) != 1 || report.Missing[0] != "lingua" {
		t.Fatalf("report=%+v", report)
	}
}
