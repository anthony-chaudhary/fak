package testquality

import "fmt"

// The finding codes. They are constants because the baseline file is KEYED on
// them: a typo in one place would silently open a hole in the ratchet (the
// mistyped key's floor reads as zero, so every finding of that shape reports as
// new — or, if the typo is in the baseline, the real key's floor vanishes).
const (
	// CodeNoAssertion — a TestXxx with no reachable failure call: no
	// t.Error/t.Fatal, no t.Skip, and the *testing.T is never handed to a helper.
	CodeNoAssertion = "TESTQ_NO_ASSERTION"
	// CodeSelfComparison — an assertion comparing a value to itself, which is true
	// by construction and therefore cannot fail.
	CodeSelfComparison = "TESTQ_SELF_COMPARISON"
	// CodeUncheckedErr — an error captured and never inspected before it is
	// overwritten or the function ends.
	CodeUncheckedErr = "TESTQ_UNCHECKED_ERR"
	// CodeUnreadExpectation — a table-test row field named like an expectation that
	// no code in the test ever reads.
	CodeUnreadExpectation = "TESTQ_UNREAD_EXPECTATION"
)

// Codes is the closed set, in report order. The baseline parser rejects any code
// outside it, so a renamed constant cannot leave orphaned floor rows behind that
// silently absorb a real finding.
var Codes = []string{CodeNoAssertion, CodeSelfComparison, CodeUncheckedErr, CodeUnreadExpectation}

// knownCode reports whether c is one of the four shipped codes.
func knownCode(c string) bool {
	for _, k := range Codes {
		if k == c {
			return true
		}
	}
	return false
}

// Finding is one CANDIDATE — a test-quality defect the analyzer can see the SHAPE
// of. Whether it is a real defect is a judgement this package never makes; see
// the package doc's clause 4.
//
// Func is part of the identity and Line is NOT. Line is carried for the report
// (so an editor and a terminal both make it clickable) and deliberately excluded
// from Key: a line-keyed baseline goes stale the moment anything above the
// finding moves.
type Finding struct {
	Code   string `json:"code"`
	File   string `json:"file"` // slash-separated, relative to the repo root
	Func   string `json:"func"`
	Line   int    `json:"line"`
	Detail string `json:"detail"`
}

// Key is the baseline identity: code, file, function — never the line. Tab-joined
// so it is the baseline row's own first three fields verbatim, which keeps the
// parser and the formatter from having to agree on a second encoding.
func (f Finding) Key() string { return f.Code + "\t" + f.File + "\t" + f.Func }

// String is the report line, `file:line:` first so both an editor and a terminal
// can jump to it.
func (f Finding) String() string {
	return fmt.Sprintf("%s:%d: %s: %s: %s", f.File, f.Line, f.Code, f.Func, f.Detail)
}
