// Rung E3 (issue #1882): the green-or-parked tree gate — never rotate away from a
// half-written commit. Rung E1 (#1880) composed a SafePoint out of three caller-supplied
// verdicts and rung E2 (#1881) derived the first of them (NoInFlightTurn) from the turn
// boundary; the second — TreeGreenOrParked — was still trusted from the caller, so a relay
// could rotate off a tree carrying a half-written commit and lose the work
// (docs/notes/CONCEPT-PERPETUAL-SESSIONS-2026-07-01.md, "Safe-stop-point detection": "no
// half-written commit / green-or-parked tree — safecommit / guard already know whether the
// working tree is at a committable boundary"). This rung is that consultation: it folds the
// working-tree evidence safecommit's porcelain view already yields into the one axis,
// asserts it onto the E1 SafePoint, and defers to the full conjunction.
//
// Detection only, per the rung's scope: this gate never commits, never parks a path, and
// never shells out. It reads no clock and does no I/O — the tree evidence arrives as a
// value, exactly the way rung E2 takes the boundary bit — so the whole rung stays a pure
// fold that a contract test can drive with canned evidence and no git repo.
package relay

// ReasonTreeDirty is the closed refusal token the gate stamps when the working tree carries
// uncommitted work that was not explicitly parked — a half-written commit at the candidate
// safe point. It joins the Reason* discipline rung E2 established (ReasonInFlight,
// ReasonNotAtSafePoint) so a supervisor reads a checkable cause, never free text.
const ReasonTreeDirty = "TREE_DIRTY_UNPARKED"

// TreeStatus is the working-tree evidence safecommit already produces, narrowed to what the
// safe-point predicate needs. Both fields hold repo-relative, forward-slash paths exactly as
// `git status --porcelain` emits them (safecommit's own convention throughout
// internal/safecommit); membership is decided by exact string equality, so a caller must not
// mix separator styles between the two lists.
type TreeStatus struct {
	// DirtyPaths are the paths carrying uncommitted work — staged, unstaged, or untracked.
	// Empty means a green tree: nothing is half-written.
	DirtyPaths []string
	// ParkedPaths are the dirty paths a leg has EXPLICITLY parked at a committable
	// boundary. "Explicitly" carries the weight: on a shared trunk a sibling's churn is
	// always present, so the gate demands that parking be declared by the leg rather than
	// inferred from the tree — an inferred park would let one leg wave through another
	// leg's half-written commit. A parked path that is not dirty is inert.
	ParkedPaths []string
}

// GreenOrParked derives the E1 TreeGreenOrParked axis from the tree evidence: the tree is
// GREEN when no path is dirty, and PARKED when every dirty path was explicitly parked at a
// committable boundary. A single dirty, unparked path is a half-written commit, so the tree
// is neither — the disjunction the spine names is satisfied only as a whole.
func (t TreeStatus) GreenOrParked() bool {
	if len(t.DirtyPaths) == 0 {
		return true // green: nothing is half-written
	}
	parked := make(map[string]struct{}, len(t.ParkedPaths))
	for _, p := range t.ParkedPaths {
		parked[p] = struct{}{}
	}
	for _, p := range t.DirtyPaths {
		if _, ok := parked[p]; !ok {
			return false // one unparked dirty path is a half-written commit
		}
	}
	return true // parked: every dirty path was explicitly parked
}

// TreeGate is the relay's pre-rotation assertion of the TreeGreenOrParked axis. It derives
// that axis from the safecommit-shaped tree evidence instead of trusting the caller's field,
// refuses with ReasonTreeDirty when the tree carries an unparked half-written commit, and
// otherwise asserts the derived axis onto the E1 SafePoint and defers to the full
// conjunction — stamping ReasonNotAtSafePoint when a DIFFERENT axis fails, and permitting the
// rotation only when every axis holds. It mirrors rung E2's GuardRotation exactly, one axis
// over: deriving the axis here, rather than trusting it, is the whole of rung E3.
func TreeGate(sp SafePoint, tree TreeStatus) GuardVerdict {
	if !tree.GreenOrParked() {
		return GuardVerdict{Permit: false, Reason: ReasonTreeDirty}
	}
	sp.TreeGreenOrParked = true // asserted from the tree evidence, not trusted from the caller
	if !sp.IsSafe() {
		return GuardVerdict{Permit: false, Reason: ReasonNotAtSafePoint}
	}
	return GuardVerdict{Permit: true}
}
