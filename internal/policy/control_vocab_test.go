package policy

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// TestControlAndToolVocabulariesAreDisjoint is the core #2633 guarantee: the
// per-tool refusal vocabulary (abi.ReasonCode names) and the turn/session
// control vocabulary (ControlDecision tokens) are DISJOINT string sets. No
// token belongs to both, so a tool rejection name can never resolve as a
// control decision and a control token can never resolve as a tool reason.
func TestControlAndToolVocabulariesAreDisjoint(t *testing.T) {
	toolNames := map[string]bool{}
	for _, n := range abi.ReasonNames() { // the closed core refusal vocabulary
		toolNames[n] = true
	}
	if len(toolNames) == 0 {
		t.Fatal("abi.ReasonNames() is empty; the disjointness proof would be vacuous")
	}

	for _, d := range ControlDecisions() {
		tok := d.String()
		// A control token must NOT be a tool refusal reason...
		if toolNames[tok] {
			t.Errorf("control token %q also names a per-tool refusal reason; the vocabularies overlap", tok)
		}
		if _, ok := abi.ReasonByName(tok); ok {
			t.Errorf("abi.ReasonByName(%q) resolved a control token as a tool reason; the seam leaks", tok)
		}
	}

	// ...and a tool refusal reason must NOT parse as a control decision.
	for name := range toolNames {
		if d, ok := ParseControlDecision(name); ok {
			t.Errorf("ParseControlDecision(%q) resolved a tool reason to control %v; the seam leaks", name, d)
		}
	}

	// The one-way gate holds in the trivial direction too: every real control
	// token round-trips through ParseControlDecision.
	for _, d := range ControlDecisions() {
		got, ok := ParseControlDecision(d.String())
		if !ok || got != d {
			t.Errorf("ParseControlDecision(%q) = %v, %v; want %v, true", d.String(), got, ok, d)
		}
	}
}

// TestOutcomeKeepsTheTwoVocabulariesInSeparateFields proves a tool-call
// rejection cannot be SERIALIZED into the session-stop slot: an Outcome carrying
// a MALFORMED reject and a STOP_SESSION control marshals the reason under
// tool_reject and the control token under control — never the reverse.
func TestOutcomeKeepsTheTwoVocabulariesInSeparateFields(t *testing.T) {
	o := Outcome{Reject: abi.ReasonMalformed, Control: ControlStopSession}
	b, err := json.Marshal(o)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var wire struct {
		ToolReject string `json:"tool_reject"`
		Control    string `json:"control"`
	}
	if err := json.Unmarshal(b, &wire); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if wire.ToolReject != "MALFORMED" {
		t.Errorf("tool_reject = %q, want MALFORMED (the per-tool reason belongs here)", wire.ToolReject)
	}
	if wire.Control != "STOP_SESSION" {
		t.Errorf("control = %q, want STOP_SESSION (the control token belongs here)", wire.Control)
	}
	// The load-bearing assertion: the tool reason never lands in the control slot.
	if wire.Control == "MALFORMED" {
		t.Fatal("a per-tool rejection reason was folded into the session-control slot")
	}
	// An allowed call omits tool_reject entirely — no empty reason masquerading
	// as a control cause.
	allowed, _ := json.Marshal(Outcome{Reject: abi.ReasonNone, Control: ControlContinue})
	if strings.Contains(string(allowed), "tool_reject") {
		t.Errorf("allowed-call Outcome emitted a tool_reject key: %s", allowed)
	}
}

// TestSessionStopReasonReadsControlOnly proves the FOLD the DoD names: reducing
// an Outcome to "the reason the session stopped" reads the control field only —
// a per-tool rejection reason is structurally excluded from the stop reason.
func TestSessionStopReasonReadsControlOnly(t *testing.T) {
	// A malformed tool call escalated to a stop: the stop reason is the declared
	// control token, NEVER the tool reason.
	stop := Outcome{Reject: abi.ReasonMalformed, Control: ControlStopSession}
	tok, stopped := stop.SessionStopReason()
	if !stopped || tok != "STOP_SESSION" {
		t.Fatalf("SessionStopReason() = %q, %v; want STOP_SESSION, true", tok, stopped)
	}
	if tok == abi.ReasonName(abi.ReasonMalformed) {
		t.Fatal("SessionStopReason() returned the per-tool reason as the session-stop cause")
	}

	// A rejected-but-not-escalated call does NOT stop the session, even though it
	// carries a tool reason. A tool rejection alone is not a session stop.
	cont := Outcome{Reject: abi.ReasonMalformed, Control: ControlContinue}
	if tok, stopped := cont.SessionStopReason(); stopped || tok != "" {
		t.Fatalf("a non-escalated rejection reported a session stop: %q, %v", tok, stopped)
	}

	// A pause is not a stop (it is resumable).
	pause := Outcome{Control: ControlPauseSession}
	if _, stopped := pause.SessionStopReason(); stopped {
		t.Error("ControlPauseSession reported StopsSession=true; a pause is resumable, not terminal")
	}
}

// TestEscalateRepeatedRejectionIsADeclaredOutcome proves the repeated-bad-call
// path yields a DECLARED ControlDecision (typed, from the closed set), not a raw
// parser error or a generic exit: below the ladder it stays CONTINUE, a short
// run ends the turn, a long run stops the session.
func TestEscalateRepeatedRejectionIsADeclaredOutcome(t *testing.T) {
	ladder := EscalationLadder{EndTurnAfter: 3, StopAfter: 6}
	cases := []struct {
		consecutive int
		want        ControlDecision
	}{
		{0, ControlContinue},
		{1, ControlContinue},
		{2, ControlContinue},
		{3, ControlEndTurn},
		{5, ControlEndTurn},
		{6, ControlStopSession},
		{99, ControlStopSession},
	}
	for _, tc := range cases {
		got := EscalateRepeatedRejection(abi.ReasonMalformed, tc.consecutive, ladder)
		if got != tc.want {
			t.Errorf("EscalateRepeatedRejection(MALFORMED, %d) = %v, want %v", tc.consecutive, got, tc.want)
		}
		if !got.IsValid() {
			t.Errorf("EscalateRepeatedRejection returned an undeclared decision %v", got)
		}
	}

	// No rejection to escalate => nothing happens, regardless of count.
	if got := EscalateRepeatedRejection(abi.ReasonNone, 100, ladder); got != ControlContinue {
		t.Errorf("EscalateRepeatedRejection(NONE, 100) = %v, want ControlContinue", got)
	}
}

// TestEscalationLadderNormalizeDefaultsAndOrdering proves the declared ladder is
// well-formed: unset thresholds fall to the defaults, and a stop can never be
// reached before (or at the same count as) an end-turn.
func TestEscalationLadderNormalizeDefaultsAndOrdering(t *testing.T) {
	n := EscalationLadder{}.Normalize()
	if n.EndTurnAfter != defaultEndTurnAfter || n.StopAfter != defaultStopAfter {
		t.Fatalf("zero ladder normalized to %+v, want defaults (%d,%d)", n, defaultEndTurnAfter, defaultStopAfter)
	}
	// A stop threshold that is not strictly beyond end-turn is lifted so stop is
	// never the easier escalation.
	bad := EscalationLadder{EndTurnAfter: 5, StopAfter: 2}.Normalize()
	if bad.StopAfter <= bad.EndTurnAfter {
		t.Fatalf("Normalize left StopAfter %d <= EndTurnAfter %d", bad.StopAfter, bad.EndTurnAfter)
	}
}
