package adjudicator

// rungprofile.go — the coarse RISK CLASS of a tool call (epic #663). Adjudicate
// runs a fixed sequence of refusal rungs; most are write-shaped (self-modify, the
// shell/synth-tool write floor, the lint-write grammar) and are inert for a plain
// read. riskClass collapses the two ad-hoc shape predicates (writeShaped /
// lowRiskReadShaped) into ONE ordered lattice computed once per call, so a later
// RungProfile (#665/#666) can elide the write-only rungs for the read class without
// re-deriving the shape at each rung. It is pure and reads only the DECODED args —
// never model-controlled Meta — so the class cannot be widened by the caller.

// class is the coarse risk class riskClass assigns to a call: the generalization of
// the writeShaped / lowRiskReadShaped shape predicates into a single ordered lattice.
// classWrite outranks classRead; an ambiguous call — neither clearly read- nor
// write-shaped — classifies to the HIGHER class, so the floor fails closed on
// anything it cannot positively prove is a pure read.
type class uint8

const (
	// classRead is the low-risk read-shaped family: a get_/read_/search_/list_/…
	// tool carrying NO write-capable payload (no path target, no shell command). It
	// is the only class the admit-and-log posture downgrades, and the only class a
	// RungProfile may elide write-only rungs for (they are provably inert here).
	classRead class = iota
	// classWrite is the write / self-modify-adjacent class: a mutating tool name, an
	// ambiguous name, OR a read-shaped name that resolves a write-capable payload (a
	// path target or a shell command). It is the mandatory-floor class — no
	// RungProfile may drop a refusal rung for it (#665).
	classWrite
)

// riskClass computes the coarse risk class of a call ONCE, from the tool name and the
// already-decoded args. It generalizes writeShaped + lowRiskReadShaped and fails
// closed:
//
//   - a write-shaped name is classWrite;
//   - a read-shaped name is classRead UNLESS it resolves a write-capable payload — a
//     non-empty path target or a shell command — in which case it ESCALATES to
//     classWrite (the floor cannot prove a named-target / command-bearing call is a
//     pure read, and the shell/synth-tool write rungs read those exact args, so
//     treating such a call as a write keeps the read class safe to elide them for);
//   - any other (ambiguous) name is classWrite (the higher class, fail-closed).
func riskClass(tool string, args map[string]any) class {
	if writeShaped(tool) {
		return classWrite
	}
	if lowRiskReadShaped(tool) {
		if targetPath(args) != "" || commandText(args) != "" {
			return classWrite
		}
		return classRead
	}
	return classWrite
}

// commandText returns the shell command string a call carries, reading the same arg
// keys the shell write floor inspects ("command" then "cmd"). Used by riskClass to
// escalate a read-shaped name that smuggles a command into the write class. Empty
// when no command-bearing scalar arg is present.
func commandText(args map[string]any) string {
	if s, ok := argString(args, "command"); ok && s != "" {
		return s
	}
	if s, ok := argString(args, "cmd"); ok && s != "" {
		return s
	}
	return ""
}
