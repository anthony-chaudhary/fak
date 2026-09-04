package main

import (
	"context"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

func TestInspectGuardWitnessedDoneRequiresStampedNamedCommit(t *testing.T) {
	repo := newGuardWitnessRepo(t)
	good := guardWitnessCommit(t, repo, "feat(demo): add witnessed behavior #3302 (fak demo)", "internal/demo/value.txt")
	good = good[:12]

	path := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardWitnessTranscript(t, path, "Shipped the behavior in commit "+good+"; tests pass.")
	got := inspectGuardWitnessedDone(context.Background(), path, repo, nil)
	if !got.Claimed || !got.Witnessed || got.Commit != good {
		t.Fatalf("stamped commit not witnessed: %+v", got)
	}

	bad := guardWitnessCommit(t, repo, "misc update", "internal/demo/other.txt")
	bad = bad[:12]
	writeGuardWitnessTranscript(t, path, "Done in commit "+bad+".")
	got = inspectGuardWitnessedDone(context.Background(), path, repo, nil)
	if !got.Claimed || got.Witnessed || got.Reason != guardClaimUnwitnessedReason {
		t.Fatalf("unstamped commit accepted: %+v", got)
	}
}

func TestRunGuardWitnessedDoneGateShadowEnforceAndBound(t *testing.T) {
	repo := newGuardWitnessRepo(t)
	transcriptPath := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardWitnessTranscript(t, transcriptPath, "Implemented the fix; tests pass.")
	ledger := filepath.Join(t.TempDir(), "stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)

	var stderr strings.Builder
	exit, disp, reason, fired := runGuardWitnessedDoneGate(&stderr, guardPreCompactModeShadow, transcriptPath, repo, "s1", 2)
	if exit != 0 || disp != stopDispClaimWitnessShadow || reason != guardClaimUnwitnessedReason || !fired {
		t.Fatalf("shadow = %d/%s/%s/%v, want allow/shadow/unwitnessed/fired", exit, disp, reason, fired)
	}

	stderr.Reset()
	exit, disp, _, fired = runGuardWitnessedDoneGate(&stderr, guardPreCompactModeEnforce, transcriptPath, repo, "s1", 2)
	if exit != 2 || disp != stopDispClaimUnwitnessedContinue || !fired || !strings.Contains(stderr.String(), guardClaimUnwitnessedReason) {
		t.Fatalf("enforce = %d/%s/%v stderr=%q", exit, disp, fired, stderr.String())
	}
	appendGuardWitnessStopRow(t, ledger, "s1", stopDispClaimUnwitnessedContinue)
	appendGuardWitnessStopRow(t, ledger, "s1", stopDispClaimUnwitnessedContinue)

	stderr.Reset()
	exit, disp, _, fired = runGuardWitnessedDoneGate(&stderr, guardPreCompactModeEnforce, transcriptPath, repo, "s1", 2)
	if exit != 0 || disp != stopDispClaimUnwitnessedGiveUp || !fired {
		t.Fatalf("bounded = %d/%s/%v, want allow/give-up/fired", exit, disp, fired)
	}
}

func TestGuardWitnessedDoneIgnoresNoCompletionClaim(t *testing.T) {
	path := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardWitnessTranscript(t, path, "I am still investigating the parser.")
	got := inspectGuardWitnessedDone(context.Background(), path, t.TempDir(), nil)
	if got.Claimed {
		t.Fatalf("non-completion fired: %+v", got)
	}
}

func newGuardWitnessRepo(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	runGuardWitnessGit(t, dir, "init")
	runGuardWitnessGit(t, dir, "config", "user.email", "test@example.com")
	runGuardWitnessGit(t, dir, "config", "user.name", "Test")
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte("module witness.fixture\\n\\ngo 1.26\\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, "dos.toml"), []byte("[lanes]\nactive = [\"demo\"]\n[lanes.trees]\n\"internal/demo/**\" = \"demo\"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	return dir
}

func guardWitnessCommit(t *testing.T, repo, subject, rel string) string {
	t.Helper()
	path := filepath.Join(repo, filepath.FromSlash(rel))
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, []byte(subject), 0o600); err != nil {
		t.Fatal(err)
	}
	runGuardWitnessGit(t, repo, "add", "--", filepath.ToSlash(rel), "dos.toml", "go.mod")
	runGuardWitnessGit(t, repo, "commit", "-m", subject)
	return strings.TrimSpace(runGuardWitnessGit(t, repo, "rev-parse", "HEAD"))
}

func runGuardWitnessGit(t *testing.T, dir string, args ...string) string {
	t.Helper()
	cmd := exec.Command("git", args...)
	cmd.Dir = dir
	out, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("git %v: %v\n%s", args, err, out)
	}
	return string(out)
}

func writeGuardWitnessTranscript(t *testing.T, path, text string) {
	t.Helper()
	line := fmt.Sprintf(`{"type":"assistant","message":{"role":"assistant","content":%q}}`, text)
	if err := os.WriteFile(path, []byte(line+"\n"), 0o600); err != nil {
		t.Fatal(err)
	}
}

func appendGuardWitnessStopRow(t *testing.T, path, session string, disp guardStopDisposition) {
	t.Helper()
	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	_, err = fmt.Fprintf(f, `{"schema":%q,"session":%q,"disposition":%q}`+"\n", guardStopRecordSchema, session, disp)
	if err != nil {
		t.Fatal(err)
	}
}
