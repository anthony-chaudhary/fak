package ctxknobs

// BaselineUserRequired is the frozen set of user-required context-knob keys
// (kind:name) at the day the R1 ratchet shipped (#2199). It is committed DATA:
// the manual-overlay defects the tree still carries.
//
// The count may only go DOWN. Adding a new user-required overlay reds
// TestNoNewUserRequiredKnobs (run by `make ci`) until this list is updated in
// the SAME commit — and it may only be EXTENDED with an explicit, reviewed
// reason, never to re-admit an overlay a cleaner default could retire. When an
// overlay is genuinely removed from the tree, delete its key here too (the
// ratchet tightening, pythongate-style).
//
// Today the only enumerable user-required overlay is the memory-compact skill:
// a skill whose reason for existing is keeping the auto-memory store under the
// harness context cap. Retiring it (R6 / memview wiring) is the doctrine's
// first scalp; when that lands, this list shrinks to empty.
var BaselineUserRequired = []string{
	"skill:memory-compact",
}

// BaselineCount is the frozen user-required floor the ratchet holds.
func BaselineCount() int { return len(BaselineUserRequired) }
