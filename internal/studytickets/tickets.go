// Package studytickets validates conversion of prioritized study clusters into dispatchable issues.
package studytickets

import (
	"errors"
	"sort"
)

type Disposition string

const (
	Selected      Disposition = "selected"
	Matched       Disposition = "matched"
	Deferred      Disposition = "deferred"
	Rejected      Disposition = "rejected"
	Landed        Disposition = "landed"
	BenchmarkOnly Disposition = "benchmark_only"
)

type Ticket struct {
	ClusterID                                                                                                                 string
	Disposition                                                                                                               Disposition
	Issue                                                                                                                     int
	Existing                                                                                                                  bool
	Outcome, Source, For, Problem, Today, BetterBecause, Witness, Centrality, P1P4, Horizon, CloseCondition, NativeConstraint string
	Dependencies                                                                                                              []string
}
type Audit struct {
	Schema, Cutoff, SourceRevision, Checksum                  string
	SourceCount, InaccessibleCount, CreatedCount, ReusedCount int
	RefreshObligations                                        []string
	Tickets                                                   []Ticket
}

var ErrInvalid = errors.New("studytickets: closure audit invalid")

func Validate(a Audit) error {
	if a.Schema != "fak.study-ticket-audit/1" || a.Cutoff == "" || a.SourceRevision == "" || a.Checksum == "" || len(a.Tickets) != a.SourceCount {
		return ErrInvalid
	}
	cluster := map[string]bool{}
	issue := map[int]string{}
	selected := map[string]bool{}
	for _, t := range a.Tickets {
		if t.ClusterID == "" || cluster[t.ClusterID] || t.Disposition == "" {
			return ErrInvalid
		}
		cluster[t.ClusterID] = true
		if t.Disposition == Selected || t.Disposition == Matched {
			if t.Issue <= 0 || t.Outcome == "" || t.Source == "" || t.For == "" || t.Problem == "" || t.Today == "" || t.BetterBecause == "" || t.Witness == "" || t.CloseCondition == "" {
				return ErrInvalid
			}
			if prior := issue[t.Issue]; prior != "" && prior != t.ClusterID {
				return ErrInvalid
			}
			issue[t.Issue] = t.ClusterID
			selected[t.ClusterID] = true
		}
	}
	for _, t := range a.Tickets {
		for _, d := range t.Dependencies {
			if !cluster[d] {
				return ErrInvalid
			}
			if selected[t.ClusterID] && !selected[d] && t.Disposition == Selected {
				return ErrInvalid
			}
		}
	}
	return nil
}
func Queue(a Audit) ([]Ticket, error) {
	if e := Validate(a); e != nil {
		return nil, e
	}
	by := map[string]Ticket{}
	for _, t := range a.Tickets {
		if t.Disposition == Selected || t.Disposition == Matched {
			by[t.ClusterID] = t
		}
	}
	state := map[string]int{}
	var out []Ticket
	var visit func(string) error
	visit = func(id string) error {
		if state[id] == 1 {
			return ErrInvalid
		}
		if state[id] == 2 {
			return nil
		}
		state[id] = 1
		t := by[id]
		deps := append([]string(nil), t.Dependencies...)
		sort.Strings(deps)
		for _, d := range deps {
			if _, ok := by[d]; ok {
				if e := visit(d); e != nil {
					return e
				}
			}
		}
		state[id] = 2
		out = append(out, t)
		return nil
	}
	ids := make([]string, 0, len(by))
	for id := range by {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	for _, id := range ids {
		if e := visit(id); e != nil {
			return nil, e
		}
	}
	return out, nil
}
