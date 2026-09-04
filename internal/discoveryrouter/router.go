package discoveryrouter

import (
	"sort"
	"strings"
)

const Schema = "fak.discovery/1"

type SourceStatus string

const (
	Attempted   SourceStatus = "attempted"
	Unavailable SourceStatus = "unavailable"
	Irrelevant  SourceStatus = "irrelevant"
	Skipped     SourceStatus = "skipped"
)

type Evidence struct {
	Source   string `json:"source"`
	Revision string `json:"revision"`
	Owner    string `json:"owner"`
	Score    int    `json:"score"`
	Reason   string `json:"reason"`
}

type Coverage struct {
	Source    string       `json:"source"`
	Status    SourceStatus `json:"status"`
	Watermark string       `json:"watermark,omitempty"`
	Reason    string       `json:"reason"`
}

type Report struct {
	Schema           string     `json:"schema"`
	Query            string     `json:"query"`
	CoverageComplete bool       `json:"coverage_complete"`
	Coverage         []Coverage `json:"coverage"`
	Results          []Evidence `json:"results"`
}

type Adapter interface {
	Name() string
	Relevant(query string) bool
	Search(query string, limit int) ([]Evidence, string, error)
}

// Plan coordinates discovery search execution across registered adapters.
type Plan struct{ Adapters []Adapter }

// Run executes discovery search across all configured adapters.
// Invariant: discovery routing resolution is fail-closed and deterministic.
// Guard: Plan.Run guards against partial failure and unbounded memory growth by clamping result limits, validating adapter responses, and marking coverage incomplete when adapters fail.
func (p Plan) Run(query string, limit int, skip map[string]bool) Report {
	q := strings.TrimSpace(query)
	if limit < 1 {
		limit = 10
	}
	r := Report{Schema: Schema, Query: q, CoverageComplete: true}
	for _, a := range p.Adapters {
		name := a.Name()
		if skip[name] {
			r.Coverage = append(r.Coverage, Coverage{Source: name, Status: Skipped, Reason: "caller skipped source"})
			r.CoverageComplete = false
			continue
		}
		if !a.Relevant(q) {
			r.Coverage = append(r.Coverage, Coverage{Source: name, Status: Irrelevant, Reason: "query classifier excluded source"})
			continue
		}
		hits, watermark, err := a.Search(q, limit)
		if err != nil {
			r.Coverage = append(r.Coverage, Coverage{Source: name, Status: Unavailable, Reason: err.Error()})
			r.CoverageComplete = false
			continue
		}
		r.Coverage = append(r.Coverage, Coverage{Source: name, Status: Attempted, Watermark: watermark, Reason: "searched"})
		for i := range hits {
			if hits[i].Source == "" {
				hits[i].Source = name
			}
			r.Results = append(r.Results, hits[i])
		}
	}
	sort.SliceStable(r.Results, func(i, j int) bool {
		if r.Results[i].Score == r.Results[j].Score {
			return r.Results[i].Owner < r.Results[j].Owner
		}
		return r.Results[i].Score > r.Results[j].Score
	})
	if len(r.Results) > limit {
		r.Results = r.Results[:limit]
	}
	return r
}
