package gateway

import (
	"strings"
	"testing"
)

// qadm builds a QUARANTINE ResultAdmission with the given tool_call_id, tool, and reason.
func qadm(id, tool, reason string) ResultAdmission {
	return ResultAdmission{ToolCallID: id, Tool: tool, Verdict: WireVerdict{Kind: "QUARANTINE", Reason: reason}}
}

// TestResultAdmissionNoteShrink pins the B half: the banner is ONE line, names the count
// and the closed-vocabulary reason codes (with per-reason multiplicity), keeps the
// load-bearing "retrievable / not your fault" reassurance, and is far shorter than the old
// per-item paragraph. A clean (no-quarantine) batch produces no note.
func TestResultAdmissionNoteShrink(t *testing.T) {
	if got := resultAdmissionNote(nil); got != "" {
		t.Fatalf("no admissions should yield no note, got %q", got)
	}
	allow := []ResultAdmission{{ToolCallID: "a", Tool: "Read", Verdict: WireVerdict{Kind: "ALLOW"}}}
	if got := resultAdmissionNote(allow); got != "" {
		t.Fatalf("a clean allow should yield no note, got %q", got)
	}

	one := resultAdmissionNote([]ResultAdmission{qadm("tc1", "WebFetch", "TRUST_VIOLATION")})
	if one == "" {
		t.Fatal("a quarantine should yield a note")
	}
	if strings.Contains(one, "\n") {
		t.Errorf("note must be a single line, got:\n%s", one)
	}
	for _, want := range []string{"[fak]", "1 tool result", "TRUST_VIOLATION", "page-in gate", "fak` extension"} {
		if !strings.Contains(one, want) {
			t.Errorf("single-quarantine note missing %q; got: %s", want, one)
		}
	}
	// Regression guard for B: the verbose pre-shrink phrasing must be gone.
	if strings.Contains(one, "Heads up") || strings.Contains(one, "safety precaution") {
		t.Errorf("note still carries the verbose pre-shrink phrasing: %s", one)
	}
	if len(one) > 320 {
		t.Errorf("shrunk note should be short (<=320 chars), got %d: %s", len(one), one)
	}

	multi := resultAdmissionNote([]ResultAdmission{
		qadm("tc1", "WebFetch", "TRUST_VIOLATION"),
		qadm("tc2", "mcp__x", "SECRET_EXFIL"),
		qadm("tc3", "Bash", "TRUST_VIOLATION"),
	})
	if strings.Contains(multi, "\n") {
		t.Errorf("multi note must be a single line, got:\n%s", multi)
	}
	for _, want := range []string{"3 tool results", "TRUST_VIOLATION×2", "SECRET_EXFIL"} {
		if !strings.Contains(multi, want) {
			t.Errorf("multi note missing %q; got: %s", want, multi)
		}
	}
}

// TestResultAdmissionNoteOnceDedup pins the A half: a held result is announced ONCE per
// session (trace), even though the client replays it — and re-quarantines it — every turn.
// A genuinely NEW held result on a later turn still emits; the dedup is per-trace.
func TestResultAdmissionNoteOnceDedup(t *testing.T) {
	s := &Server{}
	adms := []ResultAdmission{qadm("tc1", "WebFetch", "TRUST_VIOLATION")}

	// Turn 1: first sight of tc1 -> emits.
	if got := s.resultAdmissionNoteOnce("sess-A", adms); got == "" {
		t.Fatal("first turn should announce the held result")
	}
	// Turns 2..N: the same replayed result -> suppressed.
	for turn := 2; turn <= 5; turn++ {
		if got := s.resultAdmissionNoteOnce("sess-A", adms); got != "" {
			t.Fatalf("turn %d should suppress the already-announced result, got: %s", turn, got)
		}
	}

	// A NEW held result (tc2) arrives alongside the old tc1: only tc2 is announced.
	two := []ResultAdmission{qadm("tc1", "WebFetch", "TRUST_VIOLATION"), qadm("tc2", "Bash", "SECRET_EXFIL")}
	got := s.resultAdmissionNoteOnce("sess-A", two)
	if got == "" {
		t.Fatal("a new held result should be announced")
	}
	if !strings.Contains(got, "1 tool result") || strings.Contains(got, "2 tool results") {
		t.Errorf("only the new result should be counted; got: %s", got)
	}
	if !strings.Contains(got, "SECRET_EXFIL") || strings.Contains(got, "TRUST_VIOLATION") {
		t.Errorf("only the unseen reason should appear; got: %s", got)
	}

	// A different session sees tc1 for the first time -> emits (dedup is per-trace).
	if got := s.resultAdmissionNoteOnce("sess-B", adms); got == "" {
		t.Fatal("a different session should announce tc1 independently")
	}

	// Empty trace has no session to key on -> always emits (un-deduped fallback).
	if a := s.resultAdmissionNoteOnce("", adms); a == "" {
		t.Fatal("empty trace should fall back to the un-deduped note (first)")
	}
	if b := s.resultAdmissionNoteOnce("", adms); b == "" {
		t.Fatal("empty trace should fall back to the un-deduped note (repeat)")
	}
}

// TestResultNoteKey pins the dedup key: a stable tool_call_id keys the result across
// turns; an idless result falls back to tool|reason so repeats of the same shape collapse.
func TestResultNoteKey(t *testing.T) {
	if k := resultNoteKey(qadm("tc9", "WebFetch", "TRUST_VIOLATION")); k != "tc9" {
		t.Errorf("id-bearing result should key on the id, got %q", k)
	}
	if k := resultNoteKey(qadm("", "tool_result", "SECRET_EXFIL")); k != "tool_result|SECRET_EXFIL" {
		t.Errorf("idless result should key on tool|reason, got %q", k)
	}
}
