package canon

import (
	"strings"
	"testing"
)

// A real credential-shaped span is masked, the surrounding legitimate output is kept
// verbatim, and the redacted body re-screens CLEAN (the property the page-in gate and
// the warn-first normgate path both depend on).
func TestRedactSecretsMasksSpanKeepsRestAndRescreensClean(t *testing.T) {
	body := []byte("HTTP 200\nAuthorization: Bearer xoxb-1234567890-abcdefghijklmnop\n{\"title\":\"Weekly Report\"}")
	red, masked := RedactSecrets(body)
	if masked != 1 {
		t.Fatalf("want 1 masked span, got %d (redacted=%q)", masked, red)
	}
	s := string(red)
	if !strings.Contains(s, "HTTP 200") || !strings.Contains(s, "Weekly Report") {
		t.Errorf("legitimate output was not preserved: %q", s)
	}
	if strings.Contains(s, "xoxb-1234567890-abcdefghijklmnop") {
		t.Errorf("the secret survived redaction: %q", s)
	}
	if Scan(red).Secret {
		t.Errorf("redacted body must re-screen clean, still flags a secret: %q", s)
	}
}

// A STRUCTURAL placeholder (your-…-here / xxxx / changeme, per canon.placeholderHints)
// is NOT masked — redaction reuses the exact detection filter, so it never touches a
// value canon.Scan would not flag. (Note: the canonical AWS example AKIAIOSFODNN7EXAMPLE
// is deliberately treated as a LIVE-key stand-in by canon, so it IS masked — that is
// correct, not a placeholder exception.)
func TestRedactSecretsLeavesStructuralPlaceholders(t *testing.T) {
	body := []byte("token = your-token-here-xxxxxxxxxxxxxxxxxxxx\n")
	if Scan(body).Secret {
		t.Skip("precondition: this placeholder must not scan as a secret; canon changed")
	}
	red, masked := RedactSecrets(body)
	if masked != 0 {
		t.Fatalf("structural placeholder should not be masked, masked=%d (%q)", masked, red)
	}
	if string(red) != string(body) {
		t.Errorf("body changed for a placeholder-only input: %q", red)
	}
}

// A body with no secret is returned unchanged and reports zero masked.
func TestRedactSecretsNoSecretIsNoop(t *testing.T) {
	body := []byte("plain diagnostic output, HTTP 200, all fine\n")
	red, masked := RedactSecrets(body)
	if masked != 0 || string(red) != string(body) {
		t.Fatalf("no-secret body must be untouched: masked=%d red=%q", masked, red)
	}
	if !RawSecretComplete(body) {
		t.Errorf("a no-secret body is trivially raw-complete")
	}
}

// Multiple real secrets are all masked and the result re-screens clean.
func TestRedactSecretsMasksMultiple(t *testing.T) {
	body := []byte("a=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 b=sk-ABCDEFGHIJKLMNOP0123 end")
	red, masked := RedactSecrets(body)
	if masked < 2 {
		t.Fatalf("want >=2 masked, got %d (%q)", masked, red)
	}
	if Scan(red).Secret {
		t.Errorf("redacted multi-secret body still flags: %q", red)
	}
	if !strings.HasPrefix(string(red), "a=") || !strings.HasSuffix(string(red), " end") {
		t.Errorf("structure around the secrets was not preserved: %q", red)
	}
}

// RawSecretComplete is the seam that routes an obfuscated secret to the seal instead
// of the redactor: a raw plaintext secret is raw-complete (redactable), so the
// permissive path may handle it.
func TestRawSecretCompleteForRawPlaintextSecret(t *testing.T) {
	body := []byte("token = xoxb-9999999999-zzzzzzzzzzzz")
	if !Scan(body).Secret {
		t.Fatalf("test precondition: body should scan as a secret")
	}
	if !RawSecretComplete(body) {
		t.Errorf("a raw plaintext secret must be raw-complete so the redactor can mask it")
	}
}
