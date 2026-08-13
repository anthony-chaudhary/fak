package sessionregistry

import (
	"sort"
	"strings"
)

// IntegrityIssue is one deterministic reason a lineage graph cannot support an
// "all observed executions are registered" fleet claim.
type IntegrityIssue struct {
	Code           string `json:"code"`
	RegistrationID string `json:"registration_id,omitempty"`
	RelatedID      string `json:"related_id,omitempty"`
}

// AuditReport is the bounded machine gate over the latest registration graph and an
// independently observed execution census. Complete is true only when every non-empty
// observed session ID maps to exactly one structurally valid registration root.
type AuditReport struct {
	Registrations      int              `json:"registrations"`
	ObservedSessions   int              `json:"observed_sessions"`
	RegisteredObserved int              `json:"registered_observed"`
	Unregistered       int              `json:"unregistered_observed"`
	Ambiguous          int              `json:"ambiguous_observed"`
	Coverage           float64          `json:"coverage"`
	Complete           bool             `json:"complete"`
	Issues             []IntegrityIssue `json:"issues,omitempty"`
}

// Audit checks root/parent causality, cycles, cross-root session ambiguity, and exact
// registration coverage. It never reconstructs missing lineage or treats an unobserved
// registration as proof that an execution happened.
func Audit(rows []Record, observedSessionIDs []string) AuditReport {
	r := AuditReport{Registrations: len(rows)}
	byID := make(map[string]Record, len(rows))
	sessions := map[string]map[string]struct{}{}
	for _, row := range rows {
		id := strings.TrimSpace(row.RegistrationID)
		if id == "" {
			continue
		}
		if _, exists := byID[id]; exists {
			r.Issues = append(r.Issues, IntegrityIssue{Code: "duplicate_registration", RegistrationID: id})
		}
		byID[id] = row
		sid := strings.TrimSpace(row.Identity.SessionID)
		if sid != "" {
			if sessions[sid] == nil {
				sessions[sid] = map[string]struct{}{}
			}
			sessions[sid][strings.TrimSpace(row.RootRegistrationID)] = struct{}{}
		}
	}
	for id, row := range byID {
		root := strings.TrimSpace(row.RootRegistrationID)
		if _, ok := byID[root]; !ok {
			r.Issues = append(r.Issues, IntegrityIssue{Code: "missing_root", RegistrationID: id, RelatedID: root})
		}
		parent := strings.TrimSpace(row.ParentRegistrationID)
		if parent != "" {
			p, ok := byID[parent]
			if !ok {
				r.Issues = append(r.Issues, IntegrityIssue{Code: "orphan_parent", RegistrationID: id, RelatedID: parent})
			} else if strings.TrimSpace(p.RootRegistrationID) != root {
				r.Issues = append(r.Issues, IntegrityIssue{Code: "cross_root_parent", RegistrationID: id, RelatedID: parent})
			}
		}
		seen := map[string]bool{id: true}
		for cur := parent; cur != ""; {
			if seen[cur] {
				r.Issues = append(r.Issues, IntegrityIssue{Code: "parent_cycle", RegistrationID: id, RelatedID: cur})
				break
			}
			seen[cur] = true
			next, ok := byID[cur]
			if !ok {
				break
			}
			cur = strings.TrimSpace(next.ParentRegistrationID)
		}
	}
	observed := map[string]struct{}{}
	for _, raw := range observedSessionIDs {
		if id := strings.TrimSpace(raw); id != "" {
			observed[id] = struct{}{}
		}
	}
	r.ObservedSessions = len(observed)
	for sid := range observed {
		roots := sessions[sid]
		switch len(roots) {
		case 0:
			r.Unregistered++
			r.Issues = append(r.Issues, IntegrityIssue{Code: "unregistered_observed", RelatedID: sid})
		case 1:
			r.RegisteredObserved++
		default:
			r.Ambiguous++
			r.Issues = append(r.Issues, IntegrityIssue{Code: "ambiguous_session_root", RelatedID: sid})
		}
	}
	if r.ObservedSessions == 0 {
		r.Coverage = 1
	} else {
		r.Coverage = float64(r.RegisteredObserved) / float64(r.ObservedSessions)
	}
	sort.Slice(r.Issues, func(i, j int) bool {
		a, b := r.Issues[i], r.Issues[j]
		if a.Code != b.Code {
			return a.Code < b.Code
		}
		if a.RegistrationID != b.RegistrationID {
			return a.RegistrationID < b.RegistrationID
		}
		return a.RelatedID < b.RelatedID
	})
	r.Complete = len(r.Issues) == 0 && r.Coverage == 1
	return r
}
