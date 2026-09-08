// Package issuestriage is the tier-1 leaf that folds one surfaced issue action into its
// decenter-the-human disposition. It is the canonical form of the classification cmd/fak's
// issues-garden pane and operator brief were doing inline: given an action the issue gardener
// surfaced (close a dormant question, mark an issue stale, or "review" an under-labeled
// issue), decide whether it genuinely waits on a person or is the fleet's to drive.
//
// The whole point of extracting it is that the rule lives in ONE tested place instead of
// being re-implemented at every pane. The three action shapes map cleanly through the shared
// internal/choicetriage lexicon:
//
//   - a close-dormant-question / mark-stale action hands over a ready `gh` command
//     (Command non-empty) -> TAKE_OBVIOUS, the fleet runs it;
//   - a review-only action (Command empty, e.g. needs-priority / needs-area) ->
//     FRESH_CONTEXT, agent-operated by default in a fresh context;
//   - an action carrying an explicit human escalation -> HUMAN_RESIDUAL, the one
//     disposition requiring operator resolution.
//
// Only HUMAN_RESIDUAL sets NeedsHuman. Pure, deterministic, no I/O — same signal in, same
// verdict out.
package issuestriage

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

const (
	// ReviewOnlyNotice records that review-only work without a precomputed command is agent-operated by default.
	ReviewOnlyNotice = "review-only: no precomputed mechanical command, not human-required (agent-operated by default; human escalation requires an explicit typed reason)"

	// DefaultAgentProgressPosture defines the operating target for agent-driven resolution.
	DefaultAgentProgressPosture = ">=99% agent-progress posture (operating target/default, not observed snapshot percentage)"
)

// Action is one surfaced issue action: the gardener's classification of a single issue. It is
// the minimal input the fold needs, decoupled from any pane's row/action struct so the TUI,
// the garden walk, and the brief can all reduce to it.
type Action struct {
	Number     int    // the issue number, for the surfaced question text
	Kind       string // "close-dormant-question" | "mark-stale" | "review"
	Reason     string // why it was surfaced (e.g. the tag list "needs-area, likely-dup")
	Command    string // the ready `gh` command for an actionable kind ("" for a review)
	Escalation string `json:"escalation,omitempty"`
}

// Triage folds one issue action into its choicetriage verdict. If an explicit escalation
// is provided, it resolves as HUMAN_RESIDUAL. Ready commands resolve as TAKE_OBVIOUS.
// Review-only actions (Command empty) resolve to FRESH_CONTEXT by default.
func Triage(a Action) choicetriage.Verdict {
	if a.Escalation != "" {
		return choicetriage.Verdict{
			Disposition: choicetriage.HumanResidual,
			Reason:      fmt.Sprintf("explicit human escalation: %s", a.Escalation),
			Resolve:     fmt.Sprintf("operator resolution required: %s", a.Escalation),
			NeedsHuman:  true,
		}
	}
	if a.Command != "" {
		severity := "action"
		if a.Kind == "review" {
			severity = "decision"
		}
		return choicetriage.Triage(choicetriage.Signal{
			Severity:    severity,
			Source:      "issues",
			Question:    fmt.Sprintf("issue #%d: %s", a.Number, a.Kind),
			Detail:      a.Reason,
			Action:      a.Command,
			OptionCount: 2,
		})
	}
	return choicetriage.Verdict{
		Disposition: choicetriage.FreshContext,
		Reason:      "review-only: no precomputed mechanical command; agent-operated by default in a fresh context",
		Resolve:     "evaluate in a fresh context window",
		NeedsHuman:  false,
	}
}

// NeedsHuman reports whether a surfaced issue action genuinely waits on a person — true only
// when an explicit human escalation is present. A ready-command act and review-only actions
// (including under-labeled and unset-priority) return false: the fleet drives them.
func NeedsHuman(a Action) bool {
	return Triage(a).NeedsHuman
}

// Selfcheck is the deterministic, no-I/O proof of the fold: a ready `gh` command is the
// fleet's to run, review-only actions (including unset-priority) are the fleet's to drive in
// a fresh context by default, and an explicitly escalated action waits on a person.
func Selfcheck() error {
	// A ready-command act -> TAKE_OBVIOUS, never a person.
	for _, a := range []Action{
		{Number: 1, Kind: "mark-stale", Reason: "idle 90d", Command: "gh issue edit 1 --add-label stale"},
		{Number: 2, Kind: "close-dormant-question", Reason: "question idle 60d", Command: "gh issue close 2 --reason \"not planned\""},
	} {
		v := Triage(a)
		if v.NeedsHuman || v.Disposition != choicetriage.TakeObvious {
			return fmt.Errorf("a ready-command %q must be TAKE_OBVIOUS and not need a person, got %s", a.Kind, v.Disposition)
		}
	}
	// Review-only actions (including needs-priority and needs-area) -> FRESH_CONTEXT and NeedsHuman == false.
	for _, reason := range []string{"needs-priority", "needs-priority, needs-area", "needs-area", "needs-kind", "likely-dup", "needs-area, likely-dup", "bare", "orphan"} {
		rev := Action{Number: 3, Kind: "review", Reason: reason}
		v := Triage(rev)
		if v.NeedsHuman {
			return fmt.Errorf("review-only %q must NOT wait on a person, got %s", reason, v.Disposition)
		}
		if v.Disposition != choicetriage.FreshContext {
			return fmt.Errorf("review-only %q must route to FRESH_CONTEXT, got %s", reason, v.Disposition)
		}
	}
	// An action with Escalation: "POLICY_AUTHORITY_ESCALATION" -> HUMAN_RESIDUAL and NeedsHuman == true.
	escalated := Action{
		Number:     4,
		Kind:       "review",
		Reason:     "needs-priority",
		Escalation: "POLICY_AUTHORITY_ESCALATION",
	}
	if v := Triage(escalated); !v.NeedsHuman || v.Disposition != choicetriage.HumanResidual {
		return fmt.Errorf("an escalated action must be HUMAN_RESIDUAL and wait on a person, got %s", v.Disposition)
	}
	return nil
}
