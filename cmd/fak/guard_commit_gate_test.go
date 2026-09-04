package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestGuardCommitGateEnforceRefusesUnbindableAndAllowsBound(t *testing.T) {
	root := guardCommitGateFixture(t)
	bad := `{"tool_name":"Bash","tool_input":{"command":"git commit -m 'fix(gateway): claim code #3303 (fak gateway)' -- README.md"}}`
	var stdout, stderr strings.Builder
	if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(bad), []string{"--mode", "enforce", "--root", root}); got != 2 {
		t.Fatalf("bad commit exit=%d want 2; stdout=%q stderr=%q", got, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "CLAIM_UNWITNESSED") {
		t.Fatalf("missing typed reason: %q", stderr.String())
	}

	good := `{"tool_name":"Bash","tool_input":{"command":"git commit -m 'feat(gateway): add gate #3303 (fak gateway)' -- internal/gateway/gate.go"}}`
	stdout.Reset()
	stderr.Reset()
	if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(good), []string{"--mode", "enforce", "--root", root}); got != 0 {
		t.Fatalf("bound commit exit=%d want 0; stdout=%q stderr=%q", got, stdout.String(), stderr.String())
	}
}

func TestGuardCommitGateDefaultsToEnforce(t *testing.T) {
	root := guardCommitGateFixture(t)
	bad := `{"tool_name":"Bash","tool_input":{"command":"git commit -m 'fix(gateway): claim code #3303 (fak gateway)' -- README.md"}}`
	var stdout, stderr strings.Builder
	// Omit --mode flag; must default to enforce mode and refuse with exit 2.
	if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(bad), []string{"--root", root}); got != 2 {
		t.Fatalf("bad commit with default mode exit=%d want 2; stdout=%q stderr=%q", got, stdout.String(), stderr.String())
	}
	if !strings.Contains(stderr.String(), "CLAIM_UNWITNESSED") {
		t.Fatalf("missing typed reason: %q", stderr.String())
	}
}

func TestGuardCommitGateShadowAndUnreadableTaxonomyFailOpen(t *testing.T) {
	root := guardCommitGateFixture(t)
	payload := `{"tool_name":"functions.shell_command","tool_input":{"command":"git commit -m 'misc update' -- internal/gateway/gate.go"}}`
	var stdout, stderr strings.Builder
	if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(payload), []string{"--mode", "shadow", "--root", root}); got != 0 || !strings.Contains(stderr.String(), "shadow CLAIM_UNWITNESSED") {
		t.Fatalf("shadow exit=%d stderr=%q", got, stderr.String())
	}
	if err := os.Remove(filepath.Join(root, "dos.toml")); err != nil {
		t.Fatal(err)
	}
	stderr.Reset()
	if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(payload), []string{"--mode", "enforce", "--root", root}); got != 0 || stderr.Len() != 0 {
		t.Fatalf("missing taxonomy must fail open: exit=%d stderr=%q", got, stderr.String())
	}
}

func TestGuardCommitGateNonCommitAndIndirectMessageAreInert(t *testing.T) {
	root := guardCommitGateFixture(t)
	for _, payload := range []string{
		`{"tool_name":"Bash","tool_input":{"command":"git status --short"}}`,
		`{"tool_name":"Read","tool_input":{"file_path":"x"}}`,
		`{"tool_name":"Bash","tool_input":{"command":"git commit -F message.txt -- internal/gateway/gate.go"}}`,
	} {
		var stdout, stderr strings.Builder
		if got := runGuardCommitGate(&stdout, &stderr, strings.NewReader(payload), []string{"--mode", "enforce", "--root", root}); got != 0 || stdout.Len()+stderr.Len() != 0 {
			t.Fatalf("inert payload exit=%d stdout=%q stderr=%q", got, stdout.String(), stderr.String())
		}
	}
}

func guardCommitGateFixture(t *testing.T) string {
	t.Helper()
	root := t.TempDir()
	dos := "[lanes]\nactive = [\"gateway\", \"docs\"]\n[lanes.trees]\n\"internal/gateway/**\" = \"gateway\"\n\"README.md\" = \"docs\"\n"
	if err := os.WriteFile(filepath.Join(root, "dos.toml"), []byte(dos), 0o600); err != nil {
		t.Fatal(err)
	}
	return root
}
