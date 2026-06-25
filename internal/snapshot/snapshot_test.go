package snapshot

import (
	"bytes"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/session"
	"github.com/anthony-chaudhary/fak/internal/trajectory"
)

// TestGenericRoundTripAnyBody proves the seam is uniform over ANY primitive: an
// arbitrary struct (here standing in for an RSI loop's keep/revert row, which this
// package ships no typed codec for) dumps and restores through the generic
// Marshal/Parse/Into path with no bespoke code.
func TestGenericRoundTripAnyBody(t *testing.T) {
	type rsiRow struct {
		Candidate string  `json:"candidate"`
		Before    float64 `json:"before"`
		After     float64 `json:"after"`
		Kept      bool    `json:"kept"`
	}
	in := rsiRow{Candidate: "q8FastDecodeOK", Before: 0.62, After: 0.71, Kept: true}

	snap, err := Marshal(KindRSI, "rsi-cycle-7", in, map[string]string{"harness": "rulesynth"}, 1_700_000_000)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	b, err := snap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	got, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	if got.Kind != KindRSI || got.ID != "rsi-cycle-7" || got.Meta["harness"] != "rulesynth" {
		t.Fatalf("header lost: %+v", got)
	}
	var out rsiRow
	if err := got.Into(&out); err != nil {
		t.Fatalf("Into: %v", err)
	}
	if out != in {
		t.Fatalf("body round-trip failed: got %+v want %+v", out, in)
	}
}

// TestParseFailsClosedOnTamper proves integrity: a body whose CONTENT changed after
// Marshal makes Parse refuse the envelope (the sha256 over the canonical body no longer
// matches). The mutation keeps the body valid JSON and the same length, so only a real
// content change — not whitespace — is what trips the check.
func TestParseFailsClosedOnTamper(t *testing.T) {
	snap, err := Marshal(KindTool, "search_kb", map[string]string{"tool": "search_kb"}, nil, 1)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	// Swap one content character (still valid JSON) without touching the recorded digest.
	snap.Body = bytes.Replace(snap.Body, []byte("search_kb"), []byte("search_XX"), 1)
	b, err := snap.Encode()
	if err != nil {
		t.Fatalf("Encode: %v", err)
	}
	if _, err := Parse(b); err == nil || !strings.Contains(err.Error(), "digest mismatch") {
		t.Fatalf("Parse accepted a tampered body or wrong error: %v", err)
	}

	// And whitespace-only reformatting of the body does NOT trip the check — the digest
	// is canonical, so a pretty-print round-trip stays valid.
	good, _ := Marshal(KindTool, "x", map[string]string{"a": "b"}, nil, 1)
	gb, _ := good.Encode()
	if _, err := Parse(gb); err != nil {
		t.Fatalf("canonical body round-trip failed: %v", err)
	}
}

// TestParseRejectsWrongVersion fails closed on an unrecognized envelope version.
func TestParseRejectsWrongVersion(t *testing.T) {
	b := []byte(`{"envelope":"fak.snapshot.v999","kind":"turn","id":"x","body":{},"body_digest":""}`)
	if _, err := Parse(b); err == nil || !strings.Contains(err.Error(), "version") {
		t.Fatalf("Parse accepted a wrong-version envelope or wrong error: %v", err)
	}
}

// TestTraceRoundTrip exercises the turn-level typed codec.
func TestTraceRoundTrip(t *testing.T) {
	turns := []trajectory.Turn{
		{TraceID: "sess-1", Seq: 1, Query: "what refund fee?", Tool: "get_user_details", Verdict: "ALLOW"},
		{TraceID: "sess-1", Seq: 2, Tool: "read_refund_policy", Verdict: "QUARANTINE", Reason: "TRUST_VIOLATION"},
	}
	snap, err := DumpTrace("sess-1", turns, 1)
	if err != nil {
		t.Fatalf("DumpTrace: %v", err)
	}
	b, _ := snap.Encode()
	parsed, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}
	body, err := parsed.RestoreTrace()
	if err != nil {
		t.Fatalf("RestoreTrace: %v", err)
	}
	if body.TraceID != "sess-1" || len(body.Turns) != 2 || body.Turns[1].Verdict != "QUARANTINE" {
		t.Fatalf("trace round-trip failed: %+v", body)
	}
	// Wrong-kind guard.
	if _, err := parsed.RestoreFleet(session.NewTable()); err == nil {
		t.Fatal("RestoreFleet accepted a turn-kind snapshot")
	}
}

// TestFleetRoundTripViaRestore is the fleet-level witness: a whole drive table dumped and
// restored into a FRESH table reproduces every session's drive — including a Stopped one
// restored faithfully stopped (via session.Table.Restore), not revived as Running.
func TestFleetRoundTripViaRestore(t *testing.T) {
	src := session.NewTable()
	src.Transition("a", session.Throttled, "operator-offload")
	src.SetBudget("a", session.Budget{TurnsLeft: 3, TokensLeft: 4096})
	src.SetPriority("a", 5)
	src.Restore("b", session.State{TraceID: "b", Run: session.Stopped, Reason: session.ReasonBudgetTurns, Rev: 9})
	src.Transition("c", session.Paused, "")

	snap, err := DumpFleet("fleet-eu", src, 1)
	if err != nil {
		t.Fatalf("DumpFleet: %v", err)
	}
	b, _ := snap.Encode()
	parsed, err := Parse(b)
	if err != nil {
		t.Fatalf("Parse: %v", err)
	}

	dst := session.NewTable()
	n, err := parsed.RestoreFleet(dst)
	if err != nil {
		t.Fatalf("RestoreFleet: %v", err)
	}
	if n != 3 {
		t.Fatalf("restored %d sessions, want 3", n)
	}
	if got := dst.Get("a"); got.Run != session.Throttled || got.Budget.TokensLeft != 4096 || got.Priority != 5 {
		t.Fatalf("session a not restored: %+v", got)
	}
	if got := dst.Get("b"); got.Run != session.Stopped || got.Reason != session.ReasonBudgetTurns || got.Rev != 9 {
		t.Fatalf("stopped session b not restored faithfully: %+v", got)
	}
	if got := dst.Get("c"); got.Run != session.Paused {
		t.Fatalf("paused session c not restored: %+v", got)
	}
}

// TestRegistryLadder proves the registry enumerates the loops ladder in order and
// recognizes the canonical kinds.
func TestRegistryLadder(t *testing.T) {
	if _, ok := Known(KindSession); !ok {
		t.Fatal("session kind not registered")
	}
	if _, ok := Known("definitely-not-a-kind"); ok {
		t.Fatal("registry recognized a bogus kind")
	}
	ks := Kinds()
	if len(ks) < 5 {
		t.Fatalf("expected at least the 5 canonical kinds, got %d", len(ks))
	}
	// Levels are non-decreasing (sorted by ladder level).
	for i := 1; i < len(ks); i++ {
		if ks[i].Level < ks[i-1].Level {
			t.Fatalf("Kinds not sorted by level: %+v", ks)
		}
	}
}
