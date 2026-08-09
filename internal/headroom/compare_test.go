package headroom

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestCompareBenchRunsNativeAndNoCompressionOnSameCorpus(t *testing.T) {
	inputs := BenchCorpus()
	report := CompareBench([]string{"none", NativeName}, inputs)
	if !report.ArmsComplete || report.Complete || len(report.Arms) != 2 || report.Corpus != len(inputs) {
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

func TestCompareBenchRunsRegisteredLinguaArm(t *testing.T) {
	inputs := []BenchInput{{Name: "dense", Bytes: []byte("long context containing critical fact")}}
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_ = json.NewEncoder(w).Encode(linguaResponse{Text: "critical fact", Model: "llmlingua-2"})
	}))
	defer srv.Close()
	t.Setenv("FAK_LINGUA_URL", srv.URL)
	report := CompareBench([]string{"none", NativeName, LinguaName}, inputs)
	if !report.ArmsComplete || report.Complete || len(report.Arms) != 3 || len(report.Pending) == 0 {
		t.Fatalf("report=%+v", report)
	}
}

func TestCompareBenchNamesMissingIntegration(t *testing.T) {
	report := CompareBench([]string{"none", NativeName, "missing-compressor"}, BenchCorpus())
	if report.ArmsComplete || report.Complete || len(report.Missing) != 1 || report.Missing[0] != "missing-compressor" {
		t.Fatalf("report=%+v", report)
	}
}
