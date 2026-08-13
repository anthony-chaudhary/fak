package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestContinuitySelfcheckCapturedJourney(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runContinuitySelfcheck(&out, &errout, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	for _, want := range []string{"PASS personal continuity", "3 real objects", "2 isolated homes", "no service", "behavior skill=review-concisely workflow=triage-before-fix policy=deny-destructive", "receipts export=", "rollback restored prior inactive context"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}
func TestContinuityHelpIsTaskOrientedAndDryRunExplicit(t *testing.T) {
	var out bytes.Buffer
	continuityHelp(&out)
	for _, want := range []string{"move a safe managed context between homes", "mutations preview unless --commit", "preview", "export", "apply", "switch", "status", "rollback", "--json", "--select"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q", want)
		}
	}
}

func TestContinuityPublicEgressPreviewDoesNotEchoSensitiveContent(t *testing.T) {
	home := t.TempDir()
	managed := filepath.Join(home, "managed", "skills")
	if err := os.MkdirAll(managed, 0o755); err != nil {
		t.Fatal(err)
	}
	input := `{"name":{"sensitivity":"public","value":"safe"},"token":"ghp_1234567890abcdef","history":"person@example.invalid"}`
	if err := os.WriteFile(filepath.Join(managed, "demo.json"), []byte(input), 0o600); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	code := runContinuity(&out, &stderr, []string{"preview", "--home", home, "--channel", "public", "--json"})
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), `"action": "deny"`) || !strings.Contains(out.String(), "safe") {
		t.Fatalf("missing corpus decisions: %s", out.String())
	}
	for _, secret := range []string{"ghp_1234567890abcdef", "person@example.invalid"} {
		if strings.Contains(out.String(), secret) {
			t.Fatalf("preview leaked %q: %s", secret, out.String())
		}
	}
}
