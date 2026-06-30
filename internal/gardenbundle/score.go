package gardenbundle

import (
	"fmt"
	"sort"
)

const topIssueLimit = 5

// GardenIssue is the stable, deduplicable work item projected from one garden
// member. RecurrenceKey is intentionally deterministic: if the same member keeps
// surfacing the same class of condition across runs, downstream issue filers and
// ratchets can update one item instead of opening a fresh "garden red" note.
type GardenIssue struct {
	Key           string `json:"key"`
	RecurrenceKey string `json:"recurrence_key"`
	MemberKey     string `json:"member_key"`
	Label         string `json:"label"`
	State         string `json:"state"`
	Verdict       string `json:"verdict"`
	Gates         bool   `json:"gates"`
	Severity      int    `json:"severity"`
	Debt          int    `json:"debt"`
	Detail        string `json:"detail,omitempty"`
	NextAction    string `json:"next_action"`
}

// GardenScore is the scoreboard view of a garden payload. Score is 100 when the
// garden is clear and falls as debt rises; Debt is the additive unit used for
// trend/ratchet comparisons; TopIssues is bounded so the envelope stays small.
type GardenScore struct {
	Score     int
	Debt      int
	TopIssues []GardenIssue
}

func (p Payload) score() GardenScore {
	return ScoreResults(p.Members)
}

// ScoreResults folds member rows into a deterministic score and a bounded
// worst-first worklist. It is deliberately pure: same member rows in, same score
// out, so scheduled runs can compare score/debt over time.
func ScoreResults(results []MemberResult) GardenScore {
	var debt int
	issues := make([]GardenIssue, 0, len(results))
	for _, r := range results {
		issue := scoreMember(r)
		debt += issue.Debt
		if issue.Severity > 0 {
			issues = append(issues, issue)
		}
	}
	sort.SliceStable(issues, func(i, j int) bool {
		if issues[i].Severity != issues[j].Severity {
			return issues[i].Severity > issues[j].Severity
		}
		if issues[i].Gates != issues[j].Gates {
			return issues[i].Gates
		}
		return issues[i].Key < issues[j].Key
	})
	if len(issues) > topIssueLimit {
		issues = issues[:topIssueLimit]
	}
	score := 100 - debt*5
	if score < 0 {
		score = 0
	}
	return GardenScore{Score: score, Debt: debt, TopIssues: issues}
}

func scoreMember(r MemberResult) GardenIssue {
	severity, debt := baseSeverityDebt(r.State)
	if severity > 0 && r.Gates {
		severity += 20
		debt += 2
	}
	if severity > 0 && r.ExitCode != 0 {
		severity += 10
		debt++
	}
	if r.Counts != nil {
		if broken := r.Counts["broken"]; broken > 0 {
			severity += broken * 20
			debt += broken * 2
		}
		if action := r.Counts["action"]; action > 0 {
			severity += action * 10
			debt += action
		}
	}
	key := fmt.Sprintf("garden/%s/%s", r.Key, normalizedState(r.State))
	return GardenIssue{
		Key:           key,
		RecurrenceKey: key,
		MemberKey:     r.Key,
		Label:         r.Label,
		State:         r.State,
		Verdict:       r.Verdict,
		Gates:         r.Gates,
		Severity:      severity,
		Debt:          debt,
		Detail:        r.Detail,
		NextAction:    memberNextAction(r),
	}
}

func baseSeverityDebt(state string) (severity, debt int) {
	switch state {
	case "errored":
		return 100, 10
	case "red":
		return 90, 8
	case "action":
		return 55, 3
	case "ok", "":
		return 0, 0
	default:
		return 20, 1
	}
}

func normalizedState(state string) string {
	if state == "" {
		return "unknown"
	}
	return state
}

func memberNextAction(r MemberResult) string {
	switch r.State {
	case "ok", "":
		return "hold the line; this member is clear"
	case "errored":
		return "repair the garden member so the pass can measure it: " + r.Detail
	}
	switch r.Key {
	case "scorecard":
		return "retire the scorecard regression worst-first, then rerun `fak garden --check`"
	case "dispatch_plan":
		return "fix dispatch-planning debt, then rerun `fak dispatch scorecard --json`"
	case "fresh_status":
		return "resolve the stale status pane, then rerun `fak garden --check`"
	case "guard_route":
		return "review the routed guard finding and land the smallest durable fix"
	case "sessions_learn":
		return "refresh or repair the session-observability corpus consumed by the learn member"
	case "orphaned_runs":
		return "run `fak loop recover` and re-dispatch or re-verify the surfaced runs"
	case "release_staleness":
		return "run release readiness and ship through the release path if it is green"
	case "stale_leases":
		return "run `fak garden tick` or `fak leaseref reap` to clear expired leases"
	case "loop_audit":
		return "repair the broken loop(s) named by the loop-audit detail"
	default:
		return "address the surfaced garden condition: " + r.Detail
	}
}
