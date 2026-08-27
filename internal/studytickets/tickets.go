// Package studytickets validates conversion of prioritized study clusters into dispatchable issues.
package studytickets

import (
	"bytes"
	"embed"
	"encoding/json"
	"fmt"
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

type AuditTicket struct {
	ClusterID        string      `json:"cluster_id"`
	CandidateID      string      `json:"candidate_id,omitempty"`
	Disposition      Disposition `json:"disposition"`
	Issue            int         `json:"issue,omitempty"`
	Existing         bool        `json:"existing,omitempty"`
	Outcome          string      `json:"outcome,omitempty"`
	Source           string      `json:"source,omitempty"`
	For              string      `json:"for,omitempty"`
	Problem          string      `json:"problem,omitempty"`
	Today            string      `json:"today,omitempty"`
	BetterBecause    string      `json:"better_because,omitempty"`
	Witness          string      `json:"witness,omitempty"`
	Centrality       string      `json:"centrality,omitempty"`
	P1P4             string      `json:"p1_p4,omitempty"`
	Horizon          string      `json:"horizon,omitempty"`
	CloseCondition   string      `json:"close_condition,omitempty"`
	NativeConstraint string      `json:"native_constraint,omitempty"`
	Dependencies     []string    `json:"dependencies,omitempty"`
}

type Audit struct {
	Schema             string        `json:"schema"`
	Cutoff             string        `json:"cutoff"`
	SourceRevision     string        `json:"source_revision"`
	Checksum           string        `json:"checksum"`
	PriorityChecksum   string        `json:"priority_checksum,omitempty"`
	SourceCount        int           `json:"source_count"`
	InaccessibleCount  int           `json:"inaccessible_count"`
	CreatedCount       int           `json:"created_count"`
	ReusedCount        int           `json:"reused_count"`
	RefreshObligations []string      `json:"refresh_obligations"`
	Tickets            []AuditTicket `json:"tickets"`
}

type ClosureSummary struct{ SourceClusters, ClassifiedClusters, QueueTickets, Created, Reused int }

var allowed = map[Disposition]bool{Selected: true, Matched: true, Deferred: true, Rejected: true, Landed: true, BenchmarkOnly: true}

//go:embed testdata/closure-ledger.json
var closureFS embed.FS

func LoadClosureLedger() (Audit, error) {
	raw, err := closureFS.ReadFile("testdata/closure-ledger.json")
	if err != nil {
		return Audit{}, err
	}
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.DisallowUnknownFields()
	var a Audit
	if err := dec.Decode(&a); err != nil {
		return Audit{}, fmt.Errorf("decode closure ledger: %w", err)
	}
	if dec.More() {
		return Audit{}, ErrInvalid
	}
	if err := ValidateAudit(a); err != nil {
		return Audit{}, err
	}
	return a, nil
}

func Summary(a Audit) (ClosureSummary, error) {
	if err := ValidateAudit(a); err != nil {
		return ClosureSummary{}, err
	}
	q, err := Queue(a)
	if err != nil {
		return ClosureSummary{}, err
	}
	return ClosureSummary{len(a.Tickets), len(a.Tickets), len(q), a.CreatedCount, a.ReusedCount}, nil
}

func ValidateAudit(a Audit) error {
	if a.Schema != "fak.study-ticket-audit/1" || a.Cutoff == "" || a.SourceRevision == "" || a.Checksum == "" || a.PriorityChecksum == "" || len(a.Tickets) != a.SourceCount || len(a.RefreshObligations) == 0 || a.CreatedCount < 0 || a.ReusedCount < 0 {
		return ErrInvalid
	}
	cluster := map[string]bool{}
	issues := map[int]string{}
	selected := map[string]bool{}
	created, reused := 0, 0
	countedCandidates := map[string]bool{}
	for _, t := range a.Tickets {
		if t.ClusterID == "" || cluster[t.ClusterID] || !allowed[t.Disposition] {
			return ErrInvalid
		}
		cluster[t.ClusterID] = true
		if t.Disposition == Selected || t.Disposition == Matched {
			if t.Issue <= 0 || t.CandidateID == "" || t.Outcome == "" || t.Source == "" || t.For == "" || t.Problem == "" || t.Today == "" || t.BetterBecause == "" || t.Witness == "" || t.Centrality == "" || t.P1P4 == "" || t.Horizon == "" || t.CloseCondition == "" || t.NativeConstraint == "" {
				return ErrInvalid
			}
			if prior := issues[t.Issue]; prior != "" && prior != t.CandidateID {
				return ErrInvalid
			}
			issues[t.Issue] = t.CandidateID
			selected[t.ClusterID] = true
			if !countedCandidates[t.CandidateID] {
				countedCandidates[t.CandidateID] = true
				if t.Existing {
					reused++
				} else {
					created++
				}
			}
		}
	}
	if created != a.CreatedCount || reused != a.ReusedCount {
		return ErrInvalid
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

func Queue(a Audit) ([]AuditTicket, error) {
	if err := ValidateAudit(a); err != nil {
		return nil, err
	}
	by := map[string]AuditTicket{}
	for _, t := range a.Tickets {
		if t.Disposition == Selected || t.Disposition == Matched {
			by[t.ClusterID] = t
		}
	}
	state := map[string]int{}
	var out []AuditTicket
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
				if err := visit(d); err != nil {
					return err
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
	sort.Slice(ids, func(i, j int) bool {
		if by[ids[i]].Issue == by[ids[j]].Issue {
			return ids[i] < ids[j]
		}
		return by[ids[i]].Issue < by[ids[j]].Issue
	})
	for _, id := range ids {
		if err := visit(id); err != nil {
			return nil, err
		}
	}
	return out, nil
}
