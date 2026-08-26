package devindex

import (
	"path/filepath"
	"testing"
)

func TestDefaultDiscoveryQuestionsAreBroadAndStable(t *testing.T) {
	qs := DefaultDiscoveryQuestions()
	if len(qs) < 20 {
		t.Fatalf("questions=%d, want >=20", len(qs))
	}
	categories := map[string]bool{}
	ids := map[string]bool{}
	for _, q := range qs {
		if q.ID == "" || q.Query == "" || len(q.Owners) == 0 || q.TopK < 1 {
			t.Fatalf("invalid question: %+v", q)
		}
		if ids[q.ID] {
			t.Fatalf("duplicate id %q", q.ID)
		}
		ids[q.ID] = true
		categories[q.Category] = true
	}
	for _, want := range []string{"concept", "ownership", "cli", "history", "runtime", "docs"} {
		if !categories[want] {
			t.Errorf("missing category %q", want)
		}
	}
}

func TestDiscoveryBenchmarkReportsMeasuredMissesWithoutHidingCoverage(t *testing.T) {
	c, err := Load(filepath.Clean(filepath.Join("..", "..")))
	if err != nil {
		t.Fatal(err)
	}
	report := c.RunDiscoveryBenchmark(DefaultDiscoveryQuestions())
	if report.Schema != DiscoveryBenchmarkSchema || report.Coverage != "curated_docs_only" {
		t.Fatalf("identity=%+v", report)
	}
	if report.Questions < 20 || len(report.Cases) != report.Questions {
		t.Fatalf("questions=%d cases=%d", report.Questions, len(report.Cases))
	}
	if report.Successes == report.Questions {
		t.Fatal("baseline unexpectedly hides every current miss")
	}
	if report.Successes == 0 {
		t.Fatal("baseline has no known-good retrieval canary")
	}
	if report.RenderedBytes <= 0 {
		t.Fatal("rendered byte cost was not measured")
	}
	for _, c := range report.Cases {
		if len(c.TopPaths) > c.TopK {
			t.Fatalf("%s returned %d paths above top-k %d", c.ID, len(c.TopPaths), c.TopK)
		}
		if c.Success && (c.Rank < 1 || c.Rank > c.TopK) {
			t.Fatalf("bad rank: %+v", c)
		}
	}
}
