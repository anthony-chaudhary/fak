package studybench

import (
	"encoding/json"
	"testing"
)

func TestRepresentativeBenchmark(t *testing.T) {
	got, err := RunJSON(FixtureRecords, RepresentativeQueries, 3)
	if err != nil {
		t.Fatal(err)
	}
	var report Report
	if err := json.Unmarshal(got, &report); err != nil {
		t.Fatalf("machine-readable report: %v", err)
	}
	if report.Schema != "study-retrieval-benchmark/1" {
		t.Fatalf("schema = %q", report.Schema)
	}
	if len(report.Queries) != 6 { //boundarylint:ignore CHANGE_DETECTOR_TEST the canonical study benchmark fixture contains exactly six required queries
		t.Fatalf("queries = %d, want six representative kinds", len(report.Queries))
	}
	wantKinds := []string{"source", "mechanism", "candidate", "disposition", "contradiction", "issue_lineage"}
	for i, kind := range wantKinds {
		if report.Queries[i].Kind != kind {
			t.Fatalf("query %d kind = %q, want %q", i, report.Queries[i].Kind, kind)
		}
	}
	if len(report.Methods) != 2 {
		t.Fatalf("methods = %d, want lexical alpha and grep", len(report.Methods))
	}
	for _, method := range report.Methods {
		if method.RecallAtK != 1 {
			t.Errorf("%s recall@k = %v, want 1", method.Method, method.RecallAtK)
		}
		if method.ReturnedBytes == 0 {
			t.Errorf("%s returned no context bytes", method.Method)
		}
		if method.ColdLatencyNS < 0 || method.WarmLatencyNS < 0 {
			t.Errorf("%s reported invalid latency", method.Method)
		}
	}
	if report.Methods[0].BuildBytes == 0 {
		t.Error("lexical alpha did not report index/build cost")
	}
	if report.Methods[1].BuildBytes != 0 {
		t.Error("grep baseline unexpectedly reported an index")
	}
	if report.IndexCriteria.PromotionEvidence == "" || report.IndexCriteria.DemotionEvidence == "" || report.IndexCriteria.InvalidatingAssumption == "" {
		t.Error("generation promotion contract is incomplete")
	}
	t.Log(string(got))
}

func TestLexicalRankingIsDeterministic(t *testing.T) {
	idx, _ := buildIndex(FixtureRecords)
	first := lexicalSearch(FixtureRecords, idx, "candidate index", 3)
	for i := 0; i < 20; i++ {
		next := lexicalSearch(FixtureRecords, idx, "candidate index", 3)
		if len(next) != len(first) {
			t.Fatalf("run %d length changed", i)
		}
		for j := range first {
			if next[j].id != first[j].id || next[j].score != first[j].score {
				t.Fatalf("run %d rank %d changed", i, j)
			}
		}
	}
}
