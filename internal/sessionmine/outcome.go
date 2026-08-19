package sessionmine

import (
	"sort"
	"strings"
)

const OutcomeSchema = "fak-session-outcomes/1"

const (
	OutcomeShippedCommit = "shipped_commit"
	OutcomeUnknown       = "unknown"
)

// OutcomeEvidence is independently produced evidence that can be joined to one
// registered session. PID and timestamp are deliberately absent: neither is identity.
type OutcomeSession struct {
	RegistrationID string `json:"registration_id"`
	SessionID      string `json:"session_id,omitempty"`
	State          string `json:"state"`
}

type OutcomeEvidence struct {
	SessionID      string `json:"session_id,omitempty"`
	RegistrationID string `json:"registration_id,omitempty"`
	SHA            string `json:"sha,omitempty"`
	Claim          string `json:"claim,omitempty"`
	Verdict        string `json:"verdict,omitempty"`
	Witness        string `json:"witness,omitempty"`
	Issue          int    `json:"issue,omitempty"`
	IssueClosed    bool   `json:"issue_closed,omitempty"`
}

type SessionOutcome struct {
	Schema          string `json:"schema"`
	RegistrationID  string `json:"registration_id"`
	SessionID       string `json:"session_id,omitempty"`
	Outcome         string `json:"outcome"`
	SHA             string `json:"sha,omitempty"`
	Provenance      string `json:"provenance"`
	IssueClosed     bool   `json:"issue_closed,omitempty"`
	IssueProvenance string `json:"issue_provenance,omitempty"`
	Reason          string `json:"reason,omitempty"`
}

// AttributeOutcomes joins terminal registrations to exact-identity witness rows.
// Only a unique green diff witness controls shipped_commit. GitHub closure remains
// observed context and never upgrades an agent-authored claim into success.
func AttributeOutcomes(rows []OutcomeSession, evidence []OutcomeEvidence) []SessionOutcome {
	out := make([]SessionOutcome, 0)
	for _, row := range rows {
		if !terminalState(row.State) {
			continue
		}
		matches := make([]OutcomeEvidence, 0)
		for _, ev := range evidence {
			// Registration identity is stronger. When evidence carries it, a
			// session-id coincidence cannot override a registration mismatch.
			if ev.RegistrationID != "" {
				if ev.RegistrationID == row.RegistrationID {
					matches = append(matches, ev)
				}
				continue
			}
			if ev.SessionID != "" && row.SessionID != "" && ev.SessionID == row.SessionID {
				matches = append(matches, ev)
			}
		}
		result := SessionOutcome{Schema: OutcomeSchema, RegistrationID: row.RegistrationID, SessionID: row.SessionID, Outcome: OutcomeUnknown, Provenance: "independent_witness", Reason: "no_unique_green_diff_witness"}
		shas := map[string]bool{}
		for _, ev := range matches {
			if ev.IssueClosed {
				result.IssueClosed = true
				result.IssueProvenance = "github_observed"
			}
			if strings.EqualFold(strings.TrimSpace(ev.Claim), "CLAIM_WITNESSED") && strings.EqualFold(strings.TrimSpace(ev.Verdict), "OK") && strings.TrimSpace(ev.Witness) == "diff-witnessed" && strings.TrimSpace(ev.SHA) != "" {
				shas[strings.TrimSpace(ev.SHA)] = true
			}
		}
		if len(shas) == 1 {
			for sha := range shas {
				result.SHA = sha
			}
			result.Outcome = OutcomeShippedCommit
			result.Reason = ""
		} else if len(shas) > 1 {
			result.Reason = "conflicting_green_witnesses"
		}
		out = append(out, result)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].RegistrationID < out[j].RegistrationID })
	return out
}

func terminalState(state string) bool {
	switch state {
	case "completed", "failed", "cancelled", "lost", "reaped":
		return true
	default:
		return false
	}
}
