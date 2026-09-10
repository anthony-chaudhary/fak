package devcmd

import (
	"bytes"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestCompanionBridge_RootResolution(t *testing.T) {
	// 1. Explicit FAK_PRIVATE_ROOT environment variable pointing to a valid dir
	tempPrivate := t.TempDir()
	if err := os.WriteFile(filepath.Join(tempPrivate, "go.mod"), []byte("module github.com/anthony-chaudhary/fak-private\n"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_PRIVATE_ROOT", tempPrivate)
	resolved := ResolveCompanionRoot("")
	if resolved != tempPrivate {
		t.Errorf("expected %q, got %q", tempPrivate, resolved)
	}

	// 2. Non-existent FAK_PRIVATE_ROOT should not be accepted
	t.Setenv("FAK_PRIVATE_ROOT", filepath.Join(tempPrivate, "does_not_exist"))
	resolvedBad := ResolveCompanionRoot(t.TempDir())
	if resolvedBad == filepath.Join(tempPrivate, "does_not_exist") {
		t.Errorf("expected non-existent FAK_PRIVATE_ROOT to be rejected, got %q", resolvedBad)
	}

	// 3. Sibling discovery
	tempParent := t.TempDir()
	fakeFak := filepath.Join(tempParent, "fak")
	fakePriv := filepath.Join(tempParent, "fak-private")
	if err := os.MkdirAll(fakeFak, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(fakePriv, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(fakePriv, "go.mod"), []byte("module github.com/anthony-chaudhary/fak-private\n"), 0644); err != nil {
		t.Fatal(err)
	}

	t.Setenv("FAK_PRIVATE_ROOT", "")
	resolvedSibling := ResolveCompanionRoot(fakeFak)
	if filepath.Clean(resolvedSibling) != filepath.Clean(fakePriv) {
		t.Errorf("expected sibling %q, got %q", fakePriv, resolvedSibling)
	}
}

func TestCompanionBridge_MissingCompanion(t *testing.T) {
	// Force companion to be missing
	origOverride := overrideCompanionRoots
	overrideCompanionRoots = func() (string, string) {
		return t.TempDir(), ""
	}
	defer func() {
		overrideCompanionRoots = origOverride
	}()

	// 1. RunCompanionGate without --help
	var stdout, stderr bytes.Buffer
	code := RunCompanionGate(&stdout, &stderr, []string{"check", "--staged"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev gate: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 2. RunCompanionGate with --help (should exit 0 with notice and help)
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionGate(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev gate: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev gate") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 3. RunCompanionProvenance without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"audit", "--commit", "HEAD"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev provenance: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 4. RunCompanionProvenance with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev provenance: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev provenance") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 5. RunCompanionAuditLeak without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--staged"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev audit-leak: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 6. RunCompanionAuditLeak with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev audit-leak: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev audit-leak") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}

	// 7. RunCompanionStrix without --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"key"})
	if code != 1 {
		t.Errorf("expected exit code 1 for missing companion, got %d", code)
	}
	if !strings.Contains(stderr.String(), "fak-dev strix: companion repository (fak-private) not found. Set FAK_PRIVATE_ROOT or clone fak-private as sibling.") {
		t.Errorf("unexpected stderr: %s", stderr.String())
	}

	// 8. RunCompanionStrix with --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Errorf("expected exit code 0 for --help on missing companion, got %d", code)
	}
	if !strings.Contains(stdout.String(), "fak-dev strix: companion repository (fak-private) not found") {
		t.Errorf("expected notice in stdout for --help, got: %s", stdout.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev strix") {
		t.Errorf("expected usage in stdout for --help, got: %s", stdout.String())
	}
}

func TestCompanionBridge_InvocationHelpers(t *testing.T) {
	if !isBoundaryCheckInvocation([]string{"check"}) {
		t.Error("expected 'check' to be a boundary check invocation")
	}
	if !isBoundaryCheckInvocation([]string{"check", "--staged"}) {
		t.Error("expected 'check --staged' to be a boundary check invocation")
	}
	if !isBoundaryCheckInvocation([]string{"--staged"}) {
		t.Error("expected '--staged' to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"--help"}) {
		t.Error("expected '--help' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"-h"}) {
		t.Error("expected '-h' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"query", "concept"}) {
		t.Error("expected 'query' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation([]string{"explain", "path"}) {
		t.Error("expected 'explain' NOT to be a boundary check invocation")
	}
	if isBoundaryCheckInvocation(nil) {
		t.Error("expected empty args NOT to be a boundary check invocation")
	}

	if !hasFlag([]string{"--foo", "--bar"}, "--foo") {
		t.Error("expected hasFlag to find --foo")
	}
	if !hasFlag([]string{"--dir=test"}, "--dir") {
		t.Error("expected hasFlag to find --dir in --dir=test")
	}
	if hasFlag([]string{"--foo"}, "--bar") {
		t.Error("expected hasFlag not to find --bar")
	}
}

func TestCompanionBridge_Live(t *testing.T) {
	_, privRoot := ResolveCompanionRoots()
	if privRoot == "" {
		t.Skip("skipping live companion bridge tests: companion fak-private not present")
	}

	// 1. Gate --help
	var stdout, stderr bytes.Buffer
	code := RunCompanionGate(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("gate --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak-boundary") && !strings.Contains(stdout.String(), "Subcommands:") {
		t.Errorf("unexpected gate --help output: %s", stdout.String())
	}

	// 2. Gate check --staged
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionGate(&stdout, &stderr, []string{"check", "--staged"})
	if code != 0 {
		t.Fatalf("gate check --staged failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "5-Gate Import Encapsulation") {
		t.Errorf("unexpected gate check output: %s", stdout.String())
	}

	// 3. Provenance --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("provenance --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "fak-sync provenance") && !strings.Contains(stdout.String(), "Subcommands:") {
		t.Errorf("unexpected provenance --help output: %s", stdout.String())
	}

	// 4. Provenance audit --commit HEAD
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionProvenance(&stdout, &stderr, []string{"audit", "--commit", "HEAD"})
	if code != 0 {
		t.Fatalf("provenance audit failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Context Provenance Audit Report") {
		t.Errorf("unexpected provenance audit output: %s", stdout.String())
	}

	// 5. Audit-Leak --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("audit-leak --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev audit-leak") {
		t.Errorf("unexpected audit-leak --help output: %s", stdout.String())
	}

	// 6. Audit-Leak --staged
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionAuditLeak(&stdout, &stderr, []string{"--staged"})
	if code != 0 {
		t.Fatalf("audit-leak --staged failed with code %d. stderr: %s", code, stderr.String())
	}

	// 7. Strix --help
	stdout.Reset()
	stderr.Reset()
	code = RunCompanionStrix(&stdout, &stderr, []string{"--help"})
	if code != 0 {
		t.Fatalf("strix --help failed with code %d. stderr: %s", code, stderr.String())
	}
	if !strings.Contains(stdout.String(), "Usage: fak-dev strix") {
		t.Errorf("unexpected strix --help output: %s", stdout.String())
	}
}
