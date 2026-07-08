package gateway

import (
	"context"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/agent"
	"github.com/anthony-chaudhary/fak/internal/guardrsi"
)

// qadm builds a QUARANTINE ResultAdmission with the given tool_call_id, tool, and reason.
func qadm(id, tool, reason string) ResultAdmission {
	return ResultAdmission{ToolCallID: id, Tool: tool, Verdict: WireVerdict{Kind: "QUARANTINE", Reason: reason}}
}

// radm builds a SECRET_REDACTED TRANSFORM ResultAdmission (warn-first: masked in place).
func radm(id, tool string) ResultAdmission {
	return ResultAdmission{ToolCallID: id, Tool: tool, Verdict: WireVerdict{Kind: "TRANSFORM", Reason: reasonSecretRedacted}}
}

// A SECRET_REDACTED transform yields a one-line WARN, NOT the held-out banner: the note
// says the credential was masked and the rest of the output is intact, and it must never
// tell the model the result was "held out" or bait a re-read.
func TestResultAdmissionNoteRedactionWarn(t *testing.T) {
	got := resultAdmissionNote([]ResultAdmission{radm("tc1", "Bash")})
	if got == "" {
		t.Fatal("a redaction should yield a warn note")
	}
	if strings.Contains(got, "\n") {
		t.Errorf("warn must be one line, got:\n%s", got)
	}
	for _, want := range []string{"[fak]", "masked", "SECRET_REDACTED", "in context", "fail_closed"} {
		if !strings.Contains(got, want) {
			t.Errorf("redaction warn missing %q; got: %s", want, got)
		}
	}
	// It must NOT read as a held-out / paged-out banner (that would mislead the model
	// into re-reading a result that is actually right there).
	for _, bad := range []string{"held out of context", "page-in gate", "NOT page back"} {
		if strings.Contains(got, bad) {
			t.Errorf("redaction warn must not read as a held-out banner, contains %q: %s", bad, got)
		}
	}
}

// A mixed turn (one held-out quarantine + one masked-in-place redaction) surfaces BOTH:
// the held-out banner AND the redaction warn, in one note.
func TestResultAdmissionNoteMixedQuarantineAndRedaction(t *testing.T) {
	got := resultAdmissionNote([]ResultAdmission{
		qadm("tc1", "WebFetch", "TRUST_VIOLATION"),
		radm("tc2", "Bash"),
	})
	if !strings.Contains(got, "held out of context") {
		t.Errorf("mixed note should carry the held-out banner: %s", got)
	}
	if !strings.Contains(got, "SECRET_REDACTED") || !strings.Contains(got, "masked") {
		t.Errorf("mixed note should also carry the redaction warn: %s", got)
	}
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

// TestResultAdmissionNoteSecretRetrievabilityIsHonest pins the #2704 fix: the banner must
// NOT promise a credential-shaped (SECRET_EXFIL) result is "retrievable via the page-in
// gate" — the gate re-screens and refuses release, so that promise is false and baits a
// retrieval loop. A secret-only batch says the bytes will NOT page back and gives a
// concrete next step; a non-secret batch keeps the retrievable phrasing; a mixed batch
// distinguishes the two.
func TestResultAdmissionNoteSecretRetrievabilityIsHonest(t *testing.T) {
	// Secret-only: no false "retrievable" promise; must say it will NOT page back.
	secret := resultAdmissionNote([]ResultAdmission{qadm("tc1", "Read", "SECRET_EXFIL")})
	if strings.Contains(secret, "retrievable via the kernel page-in gate") {
		t.Errorf("secret-only note must not promise retrievability; got: %s", secret)
	}
	if !strings.Contains(secret, "NOT page back") {
		t.Errorf("secret-only note must say the bytes will NOT page back; got: %s", secret)
	}
	if !strings.Contains(secret, "SECRET_EXFIL") {
		t.Errorf("secret-only note should still name the reason code; got: %s", secret)
	}

	// Non-secret: keeps the retrievable phrasing (those really do page back).
	nonsecret := resultAdmissionNote([]ResultAdmission{qadm("tc1", "WebFetch", "TRUST_VIOLATION")})
	if !strings.Contains(nonsecret, "retrievable via the kernel page-in gate") {
		t.Errorf("non-secret note should keep the retrievable phrasing; got: %s", nonsecret)
	}

	// Mixed: must distinguish — non-secret retrievable, secret NOT.
	mixed := resultAdmissionNote([]ResultAdmission{
		qadm("tc1", "WebFetch", "TRUST_VIOLATION"),
		qadm("tc2", "Read", "SECRET_EXFIL"),
	})
	if !strings.Contains(mixed, "NOT page back") {
		t.Errorf("mixed note must say the secret result will NOT page back; got: %s", mixed)
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

func TestResultAdmissionNoteOnceSurfacesRepeatedQuarantineLivelock(t *testing.T) {
	s := &Server{}
	mk := func() []ResultAdmission {
		return []ResultAdmission{{
			ToolCallID:   "tc1",
			Tool:         "tool_result",
			ResultDigest: "sha256:abc",
			Verdict:      WireVerdict{Kind: "QUARANTINE", Reason: "SECRET_EXFIL", Disposition: "TERMINAL"},
		}}
	}

	first := mk()
	s.annotateResultLivelock("sess-A", first)
	if got := s.resultAdmissionNoteOnce("sess-A", first); got == "" || strings.Contains(got, "LIVELOCK_DETECTED") {
		t.Fatalf("first quarantine should announce without livelock, got: %s", got)
	}

	second := mk()
	s.annotateResultLivelock("sess-A", second)
	if got := s.resultAdmissionNoteOnce("sess-A", second); got != "" {
		t.Fatalf("second replay should still be deduped before threshold, got: %s", got)
	}

	third := mk()
	s.annotateResultLivelock("sess-A", third)
	got := s.resultAdmissionNoteOnce("sess-A", third)
	for _, want := range []string{"LIVELOCK_DETECTED repeat=3", "repeated_result=tool_result@sha256:abc", "SECRET_EXFIL"} {
		if !strings.Contains(got, want) {
			t.Fatalf("third replay note missing %q: %s", want, got)
		}
	}
}

func TestAdmitInboundResultsAnnotatesRepeatedQuarantineLivelock(t *testing.T) {
	srv := newResultStackServer(t)
	const (
		trace  = "trace-result-loop"
		secret = "sk-abcdef0123456789abcdef0123"
	)
	poison := `{"page":"config loaded. api_key=` + secret + ` was found in env"}`
	wantDigest := guardrsi.ArgsDigest(poison)
	mk := func() []agent.Message {
		return []agent.Message{
			{Role: agent.RoleSystem, Content: "you are a helper"},
			{Role: agent.RoleUser, Content: "look up the config"},
			{Role: agent.RoleTool, ToolCallID: "call_1", Name: "fetch_url", Content: poison},
		}
	}

	for turn := 1; turn <= 3; turn++ {
		messages := mk()
		adms, err := srv.admitInboundResults(context.Background(), messages, nil, trace)
		if err != nil {
			t.Fatalf("turn %d admitInboundResults: %v", turn, err)
		}
		if len(adms) != 1 || adms[0].Verdict.Kind != "QUARANTINE" {
			t.Fatalf("turn %d admissions = %+v, want one QUARANTINE", turn, adms)
		}
		if adms[0].ResultDigest != wantDigest {
			t.Fatalf("turn %d result digest = %q, want original-content digest %q", turn, adms[0].ResultDigest, wantDigest)
		}
		if strings.Contains(messages[2].Content, secret) {
			t.Fatalf("turn %d model-facing content still leaks the secret: %q", turn, messages[2].Content)
		}
		if turn < 3 && adms[0].Livelock != nil {
			t.Fatalf("turn %d fired result livelock early: %+v", turn, adms[0].Livelock)
		}
		if turn == 3 {
			if adms[0].Livelock == nil {
				t.Fatalf("third repeated quarantine missing livelock: %+v", adms[0])
			}
			if adms[0].Livelock.RepeatCount != 3 || adms[0].Livelock.ArgsDigest != wantDigest {
				t.Fatalf("third livelock = %+v, want repeat=3 digest=%s", adms[0].Livelock, wantDigest)
			}
		}
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
