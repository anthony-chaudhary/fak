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
		if hasWritePayload(args) {
			return classWrite
		}
		return classRead
	}
	return classWrite
}

// hasWritePayload reports whether the decoded args carry a write-capable value the
// self-modify / shell rungs would inspect: a non-empty path-like target (the keys
// targetPath scans) or a shell command (the keys commandSelfModify reads). It scans
// the args map ONCE rather than probing a fixed series of keys twice. Scanning ALL
// keys (vs targetPath's first-match) only ever escalates MORE calls to the write
// class, the fail-closed direction.
func hasWritePayload(args map[string]any) bool {
	for k, v := range args {
		s, ok := v.(string)
		if !ok || s == "" {
			continue
		}
		switch k {
		case "path", "file_path", "filePath", "filepath", "file", "target", "filename", "dir",
			"command", "cmd":
			return true
		}
	}
	return false
}

// rung enumerates the ordered sub-rungs Adjudicate folds, so a RungProfile can gate
// each one independently (#665/#666). The order MIRRORS the fixed sequence in
// Adjudicate; a profile may only ELIDE a rung for a class, never reorder or add one.
type rung uint8

const (
	rungDenyName      rung = iota // explicit Deny[tool] by name
	rungSelfModify                // write-shaped target hits a SelfModifyGlob
	rungCmdSelfModify             // shell command writes into a guarded tree
	rungSynthTool                 // exec of an agent-authored script into a guarded tree (+ ledger note)
	rungArgPredicate              // per-tool arg-value predicates
	rungEgress                    // tool call reaches a blocked network destination (cloud-metadata SSRF)
	rungLintWrite                 // whole-file write of unparseable code
	rungTransform                 // redact a secret-shaped arg before dispatch
	rungAllow                     // affirmative allow + AllowPrefix (terminal)
	rungDefaultDeny               // fail-closed default deny (terminal)
	numRungs
)

// numClasses is the count of risk classes (classRead, classWrite) — the first
// dimension of a RungProfile's elision table.
const numClasses = int(classWrite) + 1

// RungProfile narrows which of Adjudicate's sub-rungs run, per risk class. It is the
// per-class elision table behind epic #663: most refusal rungs are write-shaped and
// inert for a plain read, so a profile can elide them for classRead to shorten the
// read path WITHOUT changing any verdict (riskClass keeps a write-capable read in
// classWrite, where nothing is elided).
//
// The zero RungProfile — and a nil *RungProfile — elides NOTHING for any class, so it
// reproduces the fixed HEAD sequence byte-for-byte (#666, drop-in safe). A profile may
// only NARROW the floor for the read class; SetPolicy clamps any attempt to elide a
// mandatory write-class rung (see mustRun / sanitizeProfile), so the
// write/self-modify-adjacent floor can never be widened.
type RungProfile struct {
	// elided[class] is a bitmask of rungs SKIPPED for that class (bit r set => rung r
	// is elided). uint16 covers numRungs (≤ 16) with room to spare.
	elided [numClasses]uint16
}

// runs reports whether the profile runs sub-rung r for a call of class cl. A nil
// profile (and the zero RungProfile) runs every rung — the byte-identical HEAD
// sequence. A non-nil profile runs r unless it has explicitly elided r for cl.
func (pr *RungProfile) runs(cl class, r rung) bool {
	if pr == nil {
		return true
	}
	return pr.elided[cl]&(1<<r) == 0
}

// elide marks rungs rs elided for class cl and returns the profile, so callers
// (DefaultRungProfile, tests) can construct a profile fluently. Eliding a mandatory
// rung is a no-op at the floor: SetPolicy clamps it via sanitizeProfile.
func (pr *RungProfile) elide(cl class, rs ...rung) *RungProfile {
	for _, r := range rs {
		pr.elided[cl] |= 1 << r
	}
	return pr
}

// mustRun is the floor invariant: it reports whether sub-rung r is MANDATORY for risk
// class cl and so may not be elided by any RungProfile. The write-shaped refusal rungs
// (self-modify, the shell + synth-tool write floor, the lint-write grammar) are
// mandatory for the write/self-modify-adjacent class — a profile may narrow the floor
// only for the read class, where those rungs are provably inert. Everything else — the
// by-name Deny, the arg-value predicates, the redact transform, and the terminal
// allow / default-deny — is shape-independent and mandatory for EVERY class.
func mustRun(cl class, r rung) bool {
	switch r {
	case rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite:
		return cl == classWrite
	default:
		return true
	}
}

// sanitizeProfile enforces the floor invariant on a policy's RungProfile: it returns a
// copy with every MANDATORY rung's elision bit CLEARED, so a profile can never drop a
// rung mustRun requires — the floor narrows only, never widens. A nil profile stays nil
// (run everything). New and SetPolicy call it, so the LIVE floor always satisfies the
// invariant no matter how a profile was hand-constructed.
func sanitizeProfile(pr *RungProfile) *RungProfile {
	if pr == nil {
		return nil
	}
	out := &RungProfile{elided: pr.elided}
	for cl := class(0); int(cl) < numClasses; cl++ {
		for r := rung(0); r < numRungs; r++ {
			if mustRun(cl, r) {
				out.elided[cl] &^= 1 << r // clear: this rung must run for cl
			}
		}
	}
	return out
}

// DefaultRungProfile is the standard read-class profile (#667): it elides the
// write-only rungs (self-modify, the shell + synth-tool write floor, the lint-write
// grammar) for classRead. Those rungs are PROVABLY inert for a read-class call —
// riskClass puts any write-capable call (a path target or a shell command) in
// classWrite, where this profile elides nothing — so the profile changes no verdict.
// Every rung it names is non-mandatory for classRead (mustRun), so sanitizeProfile
// keeps the elision intact.
func DefaultRungProfile() *RungProfile {
	return (&RungProfile{}).elide(classRead,
		rungSelfModify, rungCmdSelfModify, rungSynthTool, rungLintWrite)
}

// DefaultPolicyWithReadProfile is DefaultPolicy plus the read-class profile. It is
// the EXPLICIT constructor #667 calls for: DefaultPolicy (and the zero Policy) keep
// a nil Profile — the byte-identical HEAD floor — and only an operator who selects
// THIS constructor opts into the read-class rung elision. The write floor is unchanged
// (DefaultRungProfile elides nothing for classWrite).
func DefaultPolicyWithReadProfile() Policy {
	p := DefaultPolicy()
	p.Profile = DefaultRungProfile()
	return p
}
