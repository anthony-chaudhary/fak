package discoveryrouter

import (
	"fmt"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/docsearch"
	"github.com/anthony-chaudhary/fak/internal/fleetsearch"
)

type DocsAdapter struct {
	Catalog  *docsearch.Catalog
	Revision string
}

func (a DocsAdapter) Name() string         { return "docs" }
func (a DocsAdapter) Relevant(string) bool { return true }
func (a DocsAdapter) Search(query string, limit int) ([]Evidence, string, error) {
	if a.Catalog == nil {
		return nil, "", fmt.Errorf("docs catalog unavailable")
	}
	hits := a.Catalog.SearchDocs(query)
	if len(hits) > limit {
		hits = hits[:limit]
	}
	out := make([]Evidence, 0, len(hits))
	for i, hit := range hits {
		reason := "documentation lexical match"
		if hit.Approx {
			reason = "approximate documentation lexical match"
		}
		out = append(out, Evidence{Source: a.Name(), Revision: a.Revision, Owner: hit.Path, Score: limit - i, Reason: reason})
	}
	return out, a.Revision, nil
}

type FleetAdapter struct {
	Config    fleetsearch.Config
	Watermark string
}

func (a FleetAdapter) Name() string { return "sessions" }
func (a FleetAdapter) Relevant(query string) bool {
	q := strings.ToLower(query)
	for _, term := range []string{"session", "worker", "agent", "crash", "active", "stale", "runtime"} {
		if strings.Contains(q, term) {
			return true
		}
	}
	return false
}
func (a FleetAdapter) Search(query string, limit int) ([]Evidence, string, error) {
	cfg := a.Config
	cfg.Limit = limit
	report, err := fleetsearch.Run(query, cfg)
	if err != nil {
		return nil, "", err
	}
	out := make([]Evidence, 0, len(report.Hits))
	for _, hit := range report.Hits {
		out = append(out, Evidence{Source: a.Name(), Revision: a.Watermark, Owner: hit.SessionID, Score: hit.Score, Reason: "joined lifecycle/session evidence"})
	}
	if report.Verdict == fleetsearch.VerdictPartialCoverage {
		return out, a.Watermark, fmt.Errorf("session stores have partial coverage")
	}
	return out, a.Watermark, nil
}
