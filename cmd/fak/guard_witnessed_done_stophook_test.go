package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestRunGuardStopHookWitnessedDoneEnforce(t *testing.T) {
	repo := newGuardWitnessRepo(t)
	transcriptPath := filepath.Join(t.TempDir(), "session.jsonl")
	writeGuardWitnessTranscript(t, transcriptPath, "Implemented the fix; tests pass.")
	ledger := filepath.Join(t.TempDir(), "stops.jsonl")
	t.Setenv(guardStopsLedgerEnv, ledger)

	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte("fak_guard_deny_all_consecutive 0\n"))
	}))
	defer srv.Close()

	old, err := os.Getwd()
	if err != nil {
		t.Fatal(err)
	}
	if err := os.Chdir(repo); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = os.Chdir(old) }()
	t.Setenv("GOWORK", "off")

	var stderr strings.Builder
	code := runGuardStopHook(&stderr, strings.NewReader(`{"session_id":"witness-session","transcript_path":"`+filepath.ToSlash(transcriptPath)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardPreCompactModeOff,
		"--hardware-gate", guardPreCompactModeOff,
		"--witnessed-done", guardPreCompactModeEnforce,
		"--task-handoff-mode", guardPreCompactModeOff,
	})
	if code != 2 || !strings.Contains(stderr.String(), guardClaimUnwitnessedReason) {
		t.Fatalf("exit=%d stderr=%q, want blocked %s", code, stderr.String(), guardClaimUnwitnessedReason)
	}

	good := guardWitnessCommit(t, repo, "feat(demo): witness completion #3302 (fak demo)", "internal/demo/done.txt")
	good = good[:12]
	writeGuardWitnessTranscript(t, transcriptPath, "Shipped in commit "+good+"; tests pass.")
	if out := runGuardWitnessGit(t, repo, "cat-file", "-e", good+"^{commit}"); strings.TrimSpace(out) != "" {
		t.Fatalf("unexpected cat-file output: %q", out)
	}
	stderr.Reset()
	code = runGuardStopHook(&stderr, strings.NewReader(`{"session_id":"witness-session","transcript_path":"`+filepath.ToSlash(transcriptPath)+`"}`), []string{
		"--mode", guardPreCompactModeEnforce,
		"--metrics-url", srv.URL + "/metrics",
		"--operator-directed", guardPreCompactModeOff,
		"--hardware-gate", guardPreCompactModeOff,
		"--witnessed-done", guardPreCompactModeEnforce,
		"--task-handoff-mode", guardPreCompactModeOff,
	})
	if code != 0 || !strings.Contains(stderr.String(), "CLAIM_WITNESSED") {
		t.Fatalf("exit=%d stderr=%q, want witnessed clean stop", code, stderr.String())
	}
}
