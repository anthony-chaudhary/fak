package main

import (
	"bytes"
	"encoding/json"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/usagelog"
)

// usagelog_record_test.go exercises the CLI wiring added on top of the
// already-tested internal/usagelog package: the recorder main() calls
// (recordUsage), its path resolution (usageLogPath), the hook exclusion, and
// the `fak usage` verb's flag handling.

func withUsagePath(t *testing.T) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", path)
	return path
}

func TestUsageLogPathOverride(t *testing.T) {
	t.Setenv("FAK_USAGE_LOG_PATH", "")
	if got := usageLogPath(); got != usagelog.DefaultPath() {
		t.Fatalf("usageLogPath() with no override = %q, want usagelog.DefaultPath() %q", got, usagelog.DefaultPath())
	}
	want := filepath.Join(t.TempDir(), "custom-usage.jsonl")
	t.Setenv("FAK_USAGE_LOG_PATH", want)
	if got := usageLogPath(); got != want {
		t.Fatalf("usageLogPath() with FAK_USAGE_LOG_PATH=%q = %q, want %q", want, got, want)
	}
}

func TestRecordUsageWritesOneRow(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	start := time.Now().Add(-5 * time.Millisecond)
	recordUsage("frontierswe", []string{"--suite", "swebench"}, 0, start)

	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 1 {
		t.Fatalf("got %d rows, want 1: %+v", len(rows), rows)
	}
	r := rows[0]
	if r.Verb != "frontierswe" {
		t.Errorf("Verb = %q, want frontierswe", r.Verb)
	}
	if r.Argc != 2 {
		t.Errorf("Argc = %d, want 2", r.Argc)
	}
	if r.ExitCode != 0 {
		t.Errorf("ExitCode = %d, want 0", r.ExitCode)
	}
	if r.ArgsDigest == "" {
		t.Error("ArgsDigest is empty, want a salted digest")
	}
	if r.PID != os.Getpid() {
		t.Errorf("PID = %d, want %d", r.PID, os.Getpid())
	}
	if n, err := usagelog.Verify(path); err != nil || n != 1 {
		t.Fatalf("Verify() = (%d, %v), want (1, nil)", n, err)
	}
}

func TestRecordUsageMultipleInvocationsChain(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	recordUsage("audit", nil, 0, time.Now())
	recordUsage("audit", []string{"verify", "x.jsonl"}, 1, time.Now())
	recordUsage("", nil, 2, time.Now()) // the no-verb help path

	rows, err := usagelog.ReadRows(path)
	if err != nil {
		t.Fatalf("ReadRows: %v", err)
	}
	if len(rows) != 3 {
		t.Fatalf("got %d rows, want 3", len(rows))
	}
	if rows[2].Verb != "" || rows[2].ExitCode != 2 {
		t.Errorf("row 3 = %+v, want verb=\"\" exit_code=2", rows[2])
	}
	if n, err := usagelog.Verify(path); err != nil || n != 3 {
		t.Fatalf("Verify() = (%d, %v), want (3, nil)", n, err)
	}
}

func TestRecordUsageRespectsOptOut(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "off")

	recordUsage("guard", nil, 0, time.Now())

	if _, err := os.Stat(path); err == nil {
		t.Fatalf("usage log %s was created despite FAK_USAGE_LOG=off", path)
	} else if !os.IsNotExist(err) {
		t.Fatalf("stat %s: %v", path, err)
	}
}

func TestRecordUsageExcludesHookVerb(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	recordUsage("hook", nil, 0, time.Now())

	if _, err := os.Stat(path); err == nil {
		t.Fatalf("usage log %s was created for the excluded 'hook' verb", path)
	} else if !os.IsNotExist(err) {
		t.Fatalf("stat %s: %v", path, err)
	}
}

// captureStdout runs fn with os.Stdout redirected to a pipe and returns what
// it wrote.
func captureStdout(t *testing.T, fn func()) string {
	t.Helper()
	orig := os.Stdout
	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("os.Pipe: %v", err)
	}
	os.Stdout = w
	defer func() { os.Stdout = orig }()

	fn()

	_ = w.Close()
	os.Stdout = orig
	out, err := io.ReadAll(r)
	if err != nil {
		t.Fatalf("read pipe: %v", err)
	}
	return string(out)
}

func TestCmdUsageEmptyJournal(t *testing.T) {
	withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")

	out := captureStdout(t, func() { cmdUsage(nil) })
	if !bytes.Contains([]byte(out), []byte("no rows recorded yet")) {
		t.Fatalf("cmdUsage on an empty journal = %q, want a no-rows message", out)
	}
}

func TestCmdUsageTextAndJSON(t *testing.T) {
	path := withUsagePath(t)
	t.Setenv("FAK_USAGE_LOG", "")
	recordUsage("route", []string{"x"}, 0, time.Now())
	recordUsage("route", []string{"y"}, 1, time.Now())
	recordUsage("guard", nil, 0, time.Now())

	text := captureStdout(t, func() { cmdUsage([]string{"--by-verb"}) })
	for _, want := range []string{path, "total: 3", "errors: 1", "route", "guard"} {
		if !bytes.Contains([]byte(text), []byte(want)) {
			t.Errorf("text output missing %q; got:\n%s", want, text)
		}
	}

	jsonOut := captureStdout(t, func() { cmdUsage([]string{"--json"}) })
	var fold usagelog.Fold
	if err := json.Unmarshal([]byte(jsonOut), &fold); err != nil {
		t.Fatalf("unmarshal --json output: %v\noutput: %s", err, jsonOut)
	}
	if fold.Total != 3 || fold.Errors != 1 {
		t.Errorf("fold = %+v, want Total=3 Errors=1", fold)
	}
}
