package main

import (
	"bytes"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"testing"
)

// writeContainJournal writes rows to a temp JSONL file and returns its path.
func writeContainJournal(t *testing.T, rows ...string) string {
	t.Helper()
	p := filepath.Join(t.TempDir(), "console-faults.jsonl")
	if err := os.WriteFile(p, []byte(strings.Join(rows, "\n")+"\n"), 0o644); err != nil {
		t.Fatalf("write journal: %v", err)
	}
	return p
}

// faultRow builds one console-fault JSONL row at a fixed clock.
func faultRow(class, session, surface string, atMS int64) string {
	return `{"class":"` + class + `","at_unix_ms":` + strconv.FormatInt(atMS, 10) +
		`,"session":"` + session + `","surface":"` + surface + `"}`
}

// TestToolprocContainGateVerdicts is the offline witness that the containment
// GATE bites: each recorded-history shape produces its closed verdict and the
// gate-able exit code (0 ADMIT / 3 refuse), so a launcher consulting it before
// spawning is actually stopped during a storm, quarantine, or co-location cap.
func TestToolprocContainGateVerdicts(t *testing.T) {
	const now = int64(1_000_000_000_000)
	nowArg := "--now-ms=" + strconv.FormatInt(now, 10)

	// A cross-session storm: 5 faults across 3 sessions inside the window.
	storm := writeContainJournal(t,
		faultRow("CONSOLE_PIPE_LOST", "s0", "pty", now),
		faultRow("CONSOLE_PIPE_LOST", "s1", "pty", now),
		faultRow("CONSOLE_PIPE_LOST", "s2", "pty", now),
		faultRow("CONSOLE_HOST_FAILFAST", "s0", "stderr", now),
		faultRow("CONSOLE_HANDLE_LOST", "s1", "stdout", now),
	)
	// A surface re-crash loop: 2 faults on pty, one session.
	quar := writeContainJournal(t,
		faultRow("CONSOLE_PIPE_LOST", "a", "pty", now),
		faultRow("CONSOLE_HOST_FAILFAST", "a", "pty", now),
	)

	cases := []struct {
		name     string
		argv     []string
		wantCode int
		wantSub  string
	}{
		{"breaker-storm", []string{"--events=" + storm, "--surface=stdout", nowArg}, 3, "BREAKER_OPEN"},
		{"quarantine-surface", []string{"--events=" + quar, "--surface=pty", nowArg}, 3, "QUARANTINE_SURFACE"},
		{"quarantine-other-surface-admits", []string{"--events=" + quar, "--surface=stdout", nowArg}, 0, "ADMIT"},
		{"colocation-cap", []string{"--events=" + quar, "--surface=stdout", "--live=3", nowArg}, 3, "REFUSE_COLOCATION"},
		{"missing-journal-fail-opens", []string{"--events=" + filepath.Join(t.TempDir(), "absent.jsonl"), "--surface=pty", nowArg}, 0, "ADMIT"},
		{"usage-positional", []string{"--surface=pty", "extra"}, 2, ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			var out, errb bytes.Buffer
			code := runToolprocContain(&out, &errb, tc.argv)
			if code != tc.wantCode {
				t.Fatalf("exit = %d, want %d (stdout=%q stderr=%q)", code, tc.wantCode, out.String(), errb.String())
			}
			if tc.wantSub != "" && !strings.Contains(out.String(), tc.wantSub) {
				t.Fatalf("stdout %q missing %q", out.String(), tc.wantSub)
			}
		})
	}
}

// TestToolprocContainRefusesDriftedJournal proves the gate refuses (exit 1)
// rather than guessing when the journal is unreadable/drifted — a protective
// gate must never silently ADMIT off a row it could not parse.
func TestToolprocContainRefusesDriftedJournal(t *testing.T) {
	bad := writeContainJournal(t, `{"class":"NOT_A_REAL_CLASS","at_unix_ms":1}`)
	var out, errb bytes.Buffer
	if code := runToolprocContain(&out, &errb, []string{"--events=" + bad, "--surface=pty"}); code != 1 {
		t.Fatalf("exit = %d, want 1 (stderr=%q)", code, errb.String())
	}
}

// TestToolprocContainJSON confirms the machine view carries the closed verdict
// and the one-bit admit gate.
func TestToolprocContainJSON(t *testing.T) {
	var out, errb bytes.Buffer
	code := runToolprocContain(&out, &errb, []string{
		"--events=" + filepath.Join(t.TempDir(), "absent.jsonl"), "--surface=pty", "--json",
	})
	if code != 0 {
		t.Fatalf("exit = %d, want 0", code)
	}
	if !strings.Contains(out.String(), `"verdict": "ADMIT"`) || !strings.Contains(out.String(), `"admit": true`) {
		t.Fatalf("json missing admit verdict: %q", out.String())
	}
}
