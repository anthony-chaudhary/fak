package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/portability"
)

func TestContinuitySelfcheckCapturedJourney(t *testing.T) {
	var out, errout bytes.Buffer
	if code := runContinuitySelfcheck(&out, &errout, false); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, errout.String())
	}
	for _, want := range []string{"PASS offline multi-home continuity", "deterministic three-way plan", "conflicts=0", "PASS personal continuity", "3 real objects", "2 isolated homes", "no service", "behavior skill=review-concisely workflow=triage-before-fix policy=deny-destructive", "receipts export=", "rollback restored prior inactive context"} {
		if !strings.Contains(out.String(), want) {
			t.Fatalf("missing %q in:\n%s", want, out.String())
		}
	}
}
func TestContinuityHelpIsTaskOrientedAndDryRunExplicit(t *testing.T) {
	var out bytes.Buffer
	continuityHelp(&out)
	for _, want := range []string{"move a safe managed context between homes", "mutations preview unless --commit", "preview", "export", "sync-plan", "sync-apply", "apply", "switch", "status", "rollback", "--json", "--select"} {
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

func TestContinuitySyncCLIJourneyAndConflictExplanations(t *testing.T) {
	root := t.TempDir()
	export := func(name, body string) string {
		home := filepath.Join(root, name)
		dir := filepath.Join(home, "managed", "skills")
		if err := os.MkdirAll(dir, 0o700); err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(filepath.Join(dir, "demo.json"), []byte(body), 0o600); err != nil {
			t.Fatal(err)
		}
		path := filepath.Join(root, name+".json")
		if _, _, err := portability.New(home).Export(path, nil, true); err != nil {
			t.Fatal(err)
		}
		return path
	}
	base := export("base", `{"tone":"plain","limit":1}`)
	local := export("local", `{"tone":"brief","limit":1}`)
	remote := export("remote", `{"tone":"plain","limit":2}`)
	plan, merged, home := filepath.Join(root, "plan.json"), filepath.Join(root, "merged.json"), filepath.Join(root, "target")
	var out, stderr bytes.Buffer
	if code := runContinuity(&out, &stderr, []string{"sync-plan", "--base", base, "--local", local, "--remote", remote, "--out", plan}); code != 0 {
		t.Fatalf("plan code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), "READY") {
		t.Fatalf("human plan: %s", out.String())
	}
	out.Reset()
	stderr.Reset()
	if code := runContinuity(&out, &stderr, []string{"sync-apply", "--home", home, "--plan", plan, "--out", merged, "--commit"}); code != 0 {
		t.Fatalf("apply code=%d stderr=%s", code, stderr.String())
	}
	if _, err := portability.ReadPackage(merged); err != nil {
		t.Fatal(err)
	}
	out.Reset()
	stderr.Reset()
	conflictRemote := export("conflict-remote", `{"tone":"verbose","limit":1}`)
	if code := runContinuity(&out, &stderr, []string{"sync-plan", "--base", base, "--local", local, "--remote", conflictRemote}); code != 3 {
		t.Fatalf("human conflict code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), "divergent-edit") || !strings.Contains(out.String(), "no last-writer-wins") {
		t.Fatalf("human conflict: %s", out.String())
	}
	out.Reset()
	stderr.Reset()
	if code := runContinuity(&out, &stderr, []string{"sync-plan", "--base", base, "--local", local, "--remote", conflictRemote, "--json"}); code != 3 {
		t.Fatalf("json conflict code=%d stderr=%s", code, stderr.String())
	}
	if !strings.Contains(out.String(), `"kind": "divergent-edit"`) || !strings.Contains(out.String(), `"explanation"`) {
		t.Fatalf("json conflict: %s", out.String())
	}
}
