// Package sessiondiag projects authoritative inventory evidence into typed watchdog candidates.
package sessiondiag

import "sort"

const WatchdogCandidateSchema = "fak.sessiondiag.watchdog-candidates.v1"

const (
	WatchdogIncludeAbruptInteractive = "ABRUPT_INTERACTIVE_NO_CURRENT_EVIDENCE"
	WatchdogExcludeNoThread          = "NO_THREAD_ID"
	WatchdogExcludeKind              = "UNSUPPORTED_SESSION_KIND"
	WatchdogExcludeTurn              = "TURN_NOT_IN_PROGRESS"
	WatchdogExcludeHealth            = "CURRENT_OR_AMBIGUOUS_HEALTH"
)

type WatchdogCandidate struct {
	Session string `json:"session"`
	Harness string `json:"harness"`
	CWD     string `json:"cwd,omitempty"`
	Reason  string `json:"reason"`
}

type WatchdogCandidateExclusion struct {
	Session string `json:"session,omitempty"`
	Kind    string `json:"kind,omitempty"`
	Health  string `json:"health,omitempty"`
	Reason  string `json:"reason"`
}

type WatchdogCandidateReport struct {
	Schema     string                       `json:"schema"`
	ObservedAt string                       `json:"observed_at,omitempty"`
	Candidates []WatchdogCandidate          `json:"candidates"`
	Exclusions []WatchdogCandidateExclusion `json:"exclusions"`
	Counts     map[string]int               `json:"counts"`
}

// WatchdogPlanRows converts admitted candidates to the common plan wire shape.
// Continuation coordinates are intentionally absent until a harness adapter resolves
// them; this stage owns candidate parity, not launch construction.
func (r WatchdogCandidateReport) WatchdogPlanRows() []map[string]any {
	rows := make([]map[string]any, 0, len(r.Candidates))
	for _, candidate := range r.Candidates {
		row := map[string]any{"session": candidate.Session, "harness": candidate.Harness, "disp": candidate.Reason}
		if candidate.CWD != "" {
			row["cwd"] = candidate.CWD
		}
		rows = append(rows, row)
	}
	return rows
}

// ProjectWatchdogCandidates is the shared classification boundary between
// sessiondiag and recovery planners. It admits only locally resumable interactive
// Codex threads whose last turn is in progress and whose diagnosis proves there is
// no corroborated current process/lock evidence.
func ProjectWatchdogCandidates(in InventoryReport) WatchdogCandidateReport {
	out := WatchdogCandidateReport{Schema: WatchdogCandidateSchema, ObservedAt: in.ObservedAt, Candidates: []WatchdogCandidate{}, Exclusions: []WatchdogCandidateExclusion{}, Counts: map[string]int{}}
	for _, session := range in.Sessions {
		id := ""
		cwd := ""
		if session.Thread != nil {
			id = session.Thread.ID
			cwd = session.Thread.CWD
		}
		reason := watchdogExclusionReason(session, id)
		if reason != "" {
			out.Exclusions = append(out.Exclusions, WatchdogCandidateExclusion{Session: id, Kind: session.Kind, Health: session.Health, Reason: reason})
			out.Counts[reason]++
			continue
		}
		out.Candidates = append(out.Candidates, WatchdogCandidate{Session: id, Harness: "codex", CWD: cwd, Reason: WatchdogIncludeAbruptInteractive})
		out.Counts[WatchdogIncludeAbruptInteractive]++
	}
	sort.Slice(out.Candidates, func(i, j int) bool { return out.Candidates[i].Session < out.Candidates[j].Session })
	sort.Slice(out.Exclusions, func(i, j int) bool {
		if out.Exclusions[i].Reason != out.Exclusions[j].Reason {
			return out.Exclusions[i].Reason < out.Exclusions[j].Reason
		}
		return out.Exclusions[i].Session < out.Exclusions[j].Session
	})
	return out
}

func watchdogExclusionReason(session SessionRecord, id string) string {
	if id == "" {
		return WatchdogExcludeNoThread
	}
	if session.Kind != KindInteractiveTUI && session.Kind != KindGuardedTUI && session.Kind != KindResumeWrapper {
		return WatchdogExcludeKind
	}
	if session.LatestTurn == nil || session.LatestTurn.Status != "inProgress" {
		return WatchdogExcludeTurn
	}
	if session.Health != HealthUnknown || !watchdogContains(session.Reasons, ReasonNoCurrentEvidence) {
		return WatchdogExcludeHealth
	}
	return ""
}

func watchdogContains(values []string, want string) bool {
	for _, value := range values {
		if value == want {
			return true
		}
	}
	return false
}
