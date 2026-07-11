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
//   - a "review" whose reason names an unset PRIORITY -> HUMAN_RESIDUAL, the one
//     prioritization authority a person holds;
//   - a "review" that is only under-labeled (needs-area / needs-kind / likely-dup) ->
//     FRESH_CONTEXT, knowable classification an agent drives in a fresh context.
//
// Only HUMAN_RESIDUAL sets NeedsHuman. Pure, deterministic, no I/O — same signal in, same
// verdict out.
package issuestriage

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/choicetriage"
)

// Action is one surfaced issue action: the gardener's classification of a single issue. It is
// the minimal input the fold needs, decoupled from any pane's row/action struct so the TUI,
// the garden walk, and the brief can all reduce to it.
type Action struct {
	Number  int    // the issue number, for the surfaced question text
	Kind    string // "close-dormant-question" | "mark-stale" | "review"
	Reason  string // why it was surfaced (e.g. the tag list "needs-area, likely-dup")
	Command string // the ready `gh` command for an actionable kind ("" for a review)
}

// Triage folds one issue action into its choicetriage verdict. The Signal shape is fixed here
// so every caller classifies identically: a "review" is a genuine decision severity (its
// authority is tested against the Reason), any other kind is an action; the ready Command is
// the strongest TAKE_OBVIOUS signal; and the Source is the neutral pane name "issues" (which
// carries no authority token choicetriage would mistake for human-residual).
func Triage(a Action) choicetriage.Verdict {
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

// NeedsHuman reports whether a surfaced issue action genuinely waits on a person — true only
// for an unset-priority review. A ready-command act and an under-labeled review both return
// false: the fleet drives them.
func NeedsHuman(a Action) bool {
	return Triage(a).NeedsHuman
}

// Selfcheck is the deterministic, no-I/O proof of the fold: a ready `gh` command is the
// fleet's to run, an unset-priority review waits on a person, and an under-labeled review is
// the fleet's to drive in a fresh context. It is the witness a CLI surfaces as a selfcheck.
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
	// An unset-priority review -> HUMAN_RESIDUAL: the prioritization authority a person holds.
	prio := Action{Number: 3, Kind: "review", Reason: "needs-priority, needs-area"}
	if v := Triage(prio); !v.NeedsHuman || v.Disposition != choicetriage.HumanResidual {
		return fmt.Errorf("an unset-priority review must be HUMAN_RESIDUAL and wait on a person, got %s", v.Disposition)
	}
	// An under-labeled review -> FRESH_CONTEXT, never a person.
	for _, reason := range []string{"needs-area", "needs-kind", "likely-dup", "needs-area, likely-dup", "bare", "orphan"} {
		rev := Action{Number: 4, Kind: "review", Reason: reason}
		v := Triage(rev)
		if v.NeedsHuman {
			return fmt.Errorf("under-labeled review %q must NOT wait on a person, got %s", reason, v.Disposition)
		}
		if v.Disposition != choicetriage.FreshContext {
			return fmt.Errorf("under-labeled review %q must route to FRESH_CONTEXT, got %s", reason, v.Disposition)
		}
	}
	return nil
}
