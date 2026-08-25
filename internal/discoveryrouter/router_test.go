package discoveryrouter

import (
	"errors"
	"testing"
)

type fakeAdapter struct {
	name      string
	relevant  bool
	hits      []Evidence
	watermark string
	err       error
}

func (f fakeAdapter) Name() string         { return f.name }
func (f fakeAdapter) Relevant(string) bool { return f.relevant }
func (f fakeAdapter) Search(string, int) ([]Evidence, string, error) {
	return f.hits, f.watermark, f.err
}

func TestRunKeepsProvenanceAndCoverageHonest(t *testing.T) {
	p := Plan{Adapters: []Adapter{
		fakeAdapter{name: "docs", relevant: true, hits: []Evidence{{Owner: "docs/index.md", Score: 7, Reason: "title match"}}, watermark: "gabc"},
		fakeAdapter{name: "issues", relevant: true, err: errors.New("issue store missing")},
		fakeAdapter{name: "sessions", relevant: false},
		fakeAdapter{name: "memory", relevant: true},
	}}
	got := p.Run("where is the index", 5, map[string]bool{"memory": true})
	if got.Schema != Schema || got.CoverageComplete {
		t.Fatalf("report=%+v", got)
	}
	want := []SourceStatus{Attempted, Unavailable, Irrelevant, Skipped}
	for i, status := range want {
		if got.Coverage[i].Status != status {
			t.Fatalf("coverage[%d]=%+v", i, got.Coverage[i])
		}
	}
	if len(got.Results) != 1 || got.Results[0].Source != "docs" {
		t.Fatalf("results=%+v", got.Results)
	}
}

func TestRunRanksAcrossSourcesAndBoundsOutput(t *testing.T) {
	p := Plan{Adapters: []Adapter{
		fakeAdapter{name: "docs", relevant: true, hits: []Evidence{{Owner: "docs/a.md", Score: 4}}},
		fakeAdapter{name: "code", relevant: true, hits: []Evidence{{Owner: "internal/a.go", Score: 9}, {Owner: "internal/b.go", Score: 3}}},
	}}
	got := p.Run("owner outside docs", 2, nil)
	if !got.CoverageComplete || len(got.Results) != 2 || got.Results[0].Owner != "internal/a.go" {
		t.Fatalf("report=%+v", got)
	}
}
