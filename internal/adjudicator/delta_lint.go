package adjudicator

// delta_lint.go isolates the linter errors a single edit introduces from the
// technical debt that was already present (#10973). A coding agent that runs a
// whole-repo linter after an edit is flooded with pre-existing findings it did
// not cause; this pure kernel gives it only the newly introduced ones so it can
// repair them before proceeding (the SWE-agent/Aider delta-lint worldview).
//
// The join is line-based: pre-edit findings are projected into post-edit line
// space through the one edit's (line, linesDelta), then matched against the
// post-edit findings by (File, Line, Col, Code). A post-edit finding with no
// projected pre-edit twin is newly introduced. It is pure and side-effect-free:
// no external linter is invoked and no ordering is imposed on the output.

// LintError is one diagnostic at one source location, the common shape the
// delta join reads. Code is the linter's finding identity within a file; Msg is
// carried through untouched for the caller to render.
type LintError struct {
	Code string
	File string
	Line int
	Col  int
	Msg  string
}

// ComputeDeltaLint returns the postErrors newly introduced by an edit that
// started at 1-based editLine and changed the file by linesDelta lines (positive
// = inserted, negative = deleted). Pre-edit findings at or after editLine shift
// by linesDelta into post-edit line space; findings before the edit keep their
// line. A post-edit finding matching a projected pre-edit finding on
// (File, Line, Col, Code) is pre-existing and is excluded; every other post-edit
// finding is newly introduced, in its original order.
//
// A nil or empty preErrors means every postErrors entry is newly introduced.
// linesDelta is ignored when editLine <= 0 (no projection target).
func ComputeDeltaLint(preErrors, postErrors []LintError, editLine, linesDelta int) []LintError {
	if len(postErrors) == 0 {
		return nil
	}
	shifted := make([]LintError, 0, len(preErrors))
	for _, e := range preErrors {
		if editLine > 0 && e.Line >= editLine {
			e.Line += linesDelta
		}
		shifted = append(shifted, e)
	}
	var out []LintError
	for _, post := range postErrors {
		if lintErrorPresent(shifted, post) {
			continue
		}
		out = append(out, post)
	}
	return out
}

// lintErrorPresent reports whether want matches an entry of hay on the identity
// tuple (File, Line, Col, Code); Msg is deliberately not part of identity so a
// reworded diagnostic for the same defect is still recognized as pre-existing.
func lintErrorPresent(hay []LintError, want LintError) bool {
	for _, e := range hay {
		if e.File == want.File && e.Line == want.Line && e.Col == want.Col && e.Code == want.Code {
			return true
		}
	}
	return false
}
