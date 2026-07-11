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
	// Allocation is the entered member's divided share of the intent's declared budget,
	// carried from the walk's [WorkItem] so the exec rung can BIND it when it runs the
	// member's front door (#2224). The walk computes it as a reservation; the drive's
	// `--execute` path turns the Time share into a real ceiling — the tighter of the
	// operator's --exec-timeout and this MaxMinutes bounds the front-door deadline, so a
	// member can never outrun its declared time budget. Empty for a non-entering decision.
	Allocation Allocation `json:"allocation,omitempty"`
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
	return enteredDecision(rep, rep.Worklist[0])
}

// enteredDecision builds the entered [DriveDecision] for one worst-first worklist
// member. It is the single place the "enter this member" verdict is shaped, so
// [Drive] (the head) and [DriveBatch] (the top-K) can never drift in what an
// entered decision carries. Pure and total over the item.
func enteredDecision(rep WalkReport, it WorkItem) DriveDecision {
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
		Allocation:     it.Allocation,
		Reason:         fmt.Sprintf("worst-first: enter %s %s", it.Member.Kind, it.Member.Ref),
	}
}

// BatchDriveSchema is the versioned payload tag a batch drive emits.
const BatchDriveSchema = "fak.superloop-drive-batch.v1"

// BatchDriveDecision is [Drive] widened from one member to a bounded worst-first
// fan-out: the top-K members the SELECT step offers this invocation, so throughput
// scales with available non-colliding work instead of one member per walk.
//
// Members is the worst-first PREFIX of the walk's already-ranked worklist (length
// min(K, len(worklist))); each entry is the SAME entered [DriveDecision] [Drive]
// returns for the head, in worst-first (rank) order. This is a PURE SELECTION: it
// reads no leases and admits nothing. The impure shell passes each selected member
// through the SAME admission gate any single drive passes (region admission over
// the live lease fabric — COLLISION_RISK on lease overlap); so "trees mutually
// disjoint (and disjoint from live leases)" is ENFORCED by the gate, member by
// member, never asserted here. A member the gate refuses is simply not entered.
//
// Enter/Satisfied/IssueShortfall carry the SAME honesty as [Drive] when there is
// nothing to enter (an empty worklist): a satisfied walk reads clean, an
// unsatisfied empty worklist is an unmet headline gate (#3147), never clean.
type BatchDriveDecision struct {
	Intent         string          `json:"intent"`
	Enter          bool            `json:"enter"`
	Requested      int             `json:"requested"`
	Members        []DriveDecision `json:"members,omitempty"`
	Satisfied      bool            `json:"satisfied"`
	IssueShortfall int             `json:"issue_shortfall,omitempty"`
	Reason         string          `json:"reason"`
}

// DriveBatch reduces a completed walk's worst-first worklist to the top-K members
// to offer this invocation — the SELECT step widened from one member (see [Drive])
// to a bounded fan-out. The worklist is already sorted worst-first by [Walk], so
// the top-K are its head prefix. k <= 0 means "every worklist member" (no cap);
// k larger than the worklist is clamped to the worklist length. An empty worklist
// enters nothing with the SAME clean-vs-unmet-headline honesty as [Drive] (#3147).
//
// DriveBatch is a pure fold: it reads no files, no clock, and no lease fabric, and
// takes no admission action. The shell gates each returned member through the
// shared region-admission gate and enters only those the gate admits — so the pure
// layer never has to know the live lease geometry to stay honest.
func DriveBatch(rep WalkReport, k int) BatchDriveDecision {
	total := len(rep.Worklist)
	if total == 0 {
		// Reuse Drive's exact empty-worklist verdict (satisfied clean vs. unmet
		// headline shortfall) so the two rungs cannot diverge on the honesty gate.
		d := Drive(rep)
		req := k
		if req <= 0 {
			req = 0
		}
		return BatchDriveDecision{
			Intent:         rep.Name,
			Enter:          false,
			Requested:      req,
			Satisfied:      d.Satisfied,
			IssueShortfall: d.IssueShortfall,
			Reason:         d.Reason,
		}
	}
	n := k
	if n <= 0 || n > total {
		n = total
	}
	members := make([]DriveDecision, 0, n)
	for i := 0; i < n; i++ {
		members = append(members, enteredDecision(rep, rep.Worklist[i]))
	}
	return BatchDriveDecision{
		Intent:         rep.Name,
		Enter:          true,
		Requested:      n,
		Members:        members,
		Satisfied:      rep.Satisfied,
		IssueShortfall: rep.IssueShortfall,
		Reason: fmt.Sprintf("worst-first batch: %d of %d worklist member(s) offered (k=%d) — the shell admits each through the shared gate",
			len(members), total, k),
	}
}
