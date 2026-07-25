package gateway

import (
	"strings"
	"testing"
)

// piiRadm builds a PII_REDACTED TRANSFORM ResultAdmission (warn-first: masked in place).
func piiRadm(id, tool string) ResultAdmission {
	return ResultAdmission{ToolCallID: id, Tool: tool, Verdict: WireVerdict{Kind: "TRANSFORM", Reason: reasonPIIRedacted}}
}

// A PII_REDACTED transform (#5378) yields a one-line WARN, NOT the held-out banner: the note
// says the PII was masked and the rest of the output is intact, and it must never tell the
// model the result was "held out" or bait a re-read — the PII twin of the secret redaction
// warn.
func TestResultAdmissionNotePIIRedactionWarn(t *testing.T) {
	got := resultAdmissionNote([]ResultAdmission{piiRadm("tc1", "read_webpage")})
	if got == "" {
		t.Fatal("a PII redaction should yield a warn note")
	}
	if strings.Contains(got, "\n") {
		t.Errorf("warn must be one line, got:\n%s", got)
	}
	for _, want := range []string{"[fak]", "masked", "PII_REDACTED", "in context", "fail_closed"} {
		if !strings.Contains(got, want) {
			t.Errorf("PII redaction warn missing %q; got: %s", want, got)
		}
	}
	for _, bad := range []string{"held out of context", "page-in gate", "NOT page back"} {
		if strings.Contains(got, bad) {
			t.Errorf("PII redaction warn must not read as a held-out banner, contains %q: %s", bad, got)
		}
	}
}

// A sealed PII result (PII_EXFIL quarantine, the opt-in fail-closed posture) reads as the
// held-out banner and names the closed-vocabulary reason code — like any other served-path
// quarantine.
func TestResultAdmissionNotePIIExfilBanner(t *testing.T) {
	got := resultAdmissionNote([]ResultAdmission{qadm("tc1", "read_webpage", "PII_EXFIL")})
	if got == "" {
		t.Fatal("a PII_EXFIL quarantine should yield a note")
	}
	if !strings.Contains(got, "held out of context") || !strings.Contains(got, "PII_EXFIL") {
		t.Errorf("PII_EXFIL note should carry the held-out banner naming the reason; got: %s", got)
	}
}

// A mixed turn (a masked-in-place credential AND a masked-in-place PII span) surfaces BOTH
// warn lines, composed on one line with no held-out banner.
func TestResultAdmissionNoteMixedSecretAndPIIRedaction(t *testing.T) {
	got := resultAdmissionNote([]ResultAdmission{
		radm("tc1", "Bash"),
		piiRadm("tc2", "read_webpage"),
	})
	if !strings.Contains(got, "SECRET_REDACTED") {
		t.Errorf("mixed redaction note should carry the secret warn: %s", got)
	}
	if !strings.Contains(got, "PII_REDACTED") {
		t.Errorf("mixed redaction note should carry the PII warn: %s", got)
	}
	if strings.Contains(got, "held out of context") {
		t.Errorf("a redaction-only turn must not read as held-out: %s", got)
	}
}
