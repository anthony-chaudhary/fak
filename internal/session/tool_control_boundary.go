package session

import "github.com/anthony-chaudhary/fak/internal/abi"

// tool_control_boundary.go holds the compile-time guards that keep the two
// control planes from #2630 (per-call ToolCallOutcome vs turn/session
// SessionControl) from silently collapsing back into one path.
//
// The failure mode this exists to catch: someone "simplifies" the loop by
// copying a per-tool refusal reason into the session-stop reason slot, so a
// single MALFORMED tool call reads as a session-stop reason. The types below
// make that a category error the compiler and one test can see.

// The per-call reason and the session-control reason are deliberately DIFFERENT
// Go types: a tool refusal is an abi.ReasonCode (the closed model-facing
// vocabulary), a session-stop reason is a plain string naming a declared policy
// outcome. Assigning one to the other does not compile without an explicit
// conversion, which is the seam a reviewer can catch.
var (
	_ abi.ReasonCode = ToolCallOutcome{}.Reason
	_ string         = SessionControl{}.Reason
)

// sessionStopReasonIsNotAToolReason reports whether the declared session-stop
// reason token collides with any per-tool refusal name. The load-bearing
// invariant is that it must NOT: a session stop is named by a control-plane
// reason (REPEATED_TOOL_REJECTION), never by a tool refusal token (MALFORMED,
// MISROUTE, ...). Kept as a package function so the paired test can assert it
// over the whole closed abi reason space rather than a hand-picked sample.
func sessionStopReasonIsNotAToolReason(token string) bool {
	if token == "" {
		return false
	}
	for _, name := range abi.ReasonNames() {
		if name == token {
			return false
		}
	}
	return true
}
