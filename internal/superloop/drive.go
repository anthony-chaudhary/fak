package superloop

// drive.go — the DRIVE rung over the pure WALK (issue #2224, epic #2218 §5). WALK reads
// member status and folds a worst-first worklist but MUTATES NOTHING; DRIVE is the named
// next move: enter EXACTLY ONE member — the worst-first — one member per invocation.
//
// [Drive] is the SELECT step narrowed to one member: a pure fold over a completed
// [WalkReport]. It selects; it does NOT admit or run. The impure shell (cmd/fak) passes
// the chosen member through the SAME admission gate any spawn passes (region admission
// over the live lease fabric — COLLISION_RISK on lease overlap) and re-folds. So the
// super loop keeps its INTERIOR-NODE property (it mutates nothing at its own altitude;
// the effect happens inside the driven member's own machinery) and gets NO private spawn
// path — the band-ladder discipline that B6 may only reach the world through B4's gates.

import "fmt"

// DriveSchema is the versioned payload tag the `--json` drive emits.
const DriveSchema = "fak.superloop-drive.v1"

// DriveDecision is the SINGLE worst-first member a drive enters this invocation — the
// "one member per walk" contract as a pure selection. Enter is false when the walk is
// already satisfied (nothing worst-first to enter): the drive then enters nothing and
// the caller's re-fold is a clean exit check. When Enter is true, Member is the member
// to enter, Action is its front-door hint (the loop's `fak loop drive` / dispatch tick,
// or a scorecard's enter skill — the same text the worklist surfaces), and Rank is its
// 1-based worst-first position (always 1: the drive enters the head).
type DriveDecision struct {
	Intent    string `json:"intent"`
	Enter     bool   `json:"enter"`
	Member    Member `json:"member,omitempty"`
	Rank      int    `json:"rank,omitempty"`
	Action    string `json:"action,omitempty"`
	Debt      int    `json:"debt,omitempty"`
	Dark      bool   `json:"dark,omitempty"`
	Container bool   `json:"container,omitempty"`
	// Satisfied echoes the walk's satisfaction at drive time. A drive that enters
	// nothing (Enter=false) reads clean ONLY when Satisfied is true; an unsatisfied
	// empty-worklist drive is an unmet gate — a declared headline the members did not
	// carry — not a clean read (#3147). Consumers gate their exit on THIS, not on Enter.
	Satisfied bool `json:"satisfied"`
	// IssueShortfall carries the walk's unmet headline issue count (0 = none). It is the
	// magnitude behind an unsatisfied empty-worklist drive: there is no member to enter,
	// yet the declared ~headline is not met — the number the drive must not hide.
	IssueShortfall int    `json:"issue_shortfall,omitempty"`
	Reason         string `json:"reason"`
}

// Drive reduces a completed walk's worst-first worklist to the SINGLE member to enter
// this invocation — the SELECT step narrowed to one member per walk (the drive
// contract). The worklist is already sorted worst-first by [Walk], so the head is the
// member to enter; a satisfied walk (empty worklist) enters nothing. Drive is a pure
// fold: it reads no files and no clock and takes no admission action; the shell gates
// and runs the returned member.
func Drive(rep WalkReport) DriveDecision {
	if len(rep.Worklist) == 0 {
		// An empty MEMBER worklist means no member is worst-first to enter — but that is
		// only a CLEAN read when the walk is satisfied. A declared headline gate (an
		// issue shortfall) can leave the intent UNSATISFIED with zero member debt: every
		// utilization pool used, every member measured and live, yet the ~200-issue
		// headline unmet. A drive that entered nothing must never assert "reads clean"
		// over that gate; it reports the shortfall and stays unsatisfied (#3147).
		if !rep.Satisfied {
			return DriveDecision{
				Intent:         rep.Name,
				Enter:          false,
				Satisfied:      false,
				IssueShortfall: rep.IssueShortfall,
				Reason: fmt.Sprintf("nothing to enter member-first, but %q is UNSATISFIED: issue shortfall %d against headline %d — the declared gate is unmet, not clean",
					rep.Name, rep.IssueShortfall, rep.IssueTarget),
			}
		}
		return DriveDecision{
			Intent:    rep.Name,
			Enter:     false,
			Satisfied: true,
			Reason: fmt.Sprintf("nothing to enter — %q reads clean (debt %d at-or-below floor %d, every member measured and live)",
				rep.Name, rep.TotalDebt, rep.Floor),
		}
	}
	it := rep.Worklist[0]
	return DriveDecision{
		Intent:         rep.Name,
		Enter:          true,
		Satisfied:      rep.Satisfied,
		IssueShortfall: rep.IssueShortfall,
		Member:         it.Member,
		Rank:           it.Rank,
		Action:         it.Action,
		Debt:           it.Debt,
		Dark:           it.Dark,
		Container:      it.Container,
		Reason:         fmt.Sprintf("worst-first: enter %s %s", it.Member.Kind, it.Member.Ref),
	}
}
