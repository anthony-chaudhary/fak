package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestWatchdogAuditRunnerPropagatesTypedVerdict(t *testing.T) {
	for _, tc := range []struct {
		verdict string
		status  int
	}{{"GREEN", 0}, {"AMBER", 2}, {"RED", 3}} {
		t.Run(tc.verdict, func(t *testing.T) {
			dir := t.TempDir()
			log := filepath.Join(dir, "audit.jsonl")
			exec := func(string) ([]byte, int, error) {
				return []byte(`{"verdict":"` + tc.verdict + `","reasons":[]}`), tc.status, nil
			}
			var out, stderr bytes.Buffer
			if got := runWatchdogAuditRunnerWith(&out, &stderr, []string{"--script", "audit.ps1", "--log", log, "--max-bytes", "4096"}, exec); got != tc.status {
				t.Fatalf("status=%d want %d stderr=%s", got, tc.status, stderr.String())
			}
			lines := strings.Split(strings.TrimSpace(string(mustRead(t, log))), "\n")
			if len(lines) != 1 {
				t.Fatalf("lines=%d", len(lines))
			}
			var rec watchdogAuditReceipt
			if err := json.Unmarshal([]byte(lines[0]), &rec); err != nil {
				t.Fatal(err)
			}
			if rec.Verdict != tc.verdict || rec.ExitStatus != tc.status || rec.Schema != "fak.watchdog-audit-receipt.v1" {
				t.Fatalf("receipt=%+v", rec)
			}
		})
	}
}

func TestWatchdogAuditRunnerRejectsMalformedAndMismatch(t *testing.T) {
	for _, tc := range []struct {
		name, body string
		status     int
	}{{"malformed", "nope", 0}, {"mismatch", `{"verdict":"GREEN"}`, 3}} {
		t.Run(tc.name, func(t *testing.T) {
			var errout bytes.Buffer
			log := filepath.Join(t.TempDir(), "x.jsonl")
			got := runWatchdogAuditRunnerWith(&bytes.Buffer{}, &errout, []string{"--script", "x", "--log", log}, func(string) ([]byte, int, error) { return []byte(tc.body), tc.status, nil })
			if got != 1 {
				t.Fatalf("got %d stderr=%s", got, errout.String())
			}
			if _, err := os.Stat(log); !os.IsNotExist(err) {
				t.Fatalf("ledger should not exist: %v", err)
			}
		})
	}
}

func TestWatchdogAuditRunnerRejectsDuplicateWriter(t *testing.T) {
	log := filepath.Join(t.TempDir(), "audit.jsonl")
	lock, err := acquireStallWatchLock(log)
	if err != nil {
		t.Fatal(err)
	}
	defer lock.Close()
	var stderr bytes.Buffer
	got := runWatchdogAuditRunnerWith(&bytes.Buffer{}, &stderr, []string{"--script", "x", "--log", log}, func(string) ([]byte, int, error) { return []byte(`{"verdict":"GREEN"}`), 0, nil })
	if got != 1 || !strings.Contains(stderr.String(), "another watcher owns") {
		t.Fatalf("got=%d stderr=%s", got, stderr.String())
	}
}

func TestWatchdogAuditRunnerBoundsCompleteLines(t *testing.T) {
	log := filepath.Join(t.TempDir(), "audit.jsonl")
	exec := func(string) ([]byte, int, error) {
		return []byte(`{"verdict":"GREEN","padding":"` + strings.Repeat("x", 160) + `"}`), 0, nil
	}
	for i := 0; i < 6; i++ {
		if got := runWatchdogAuditRunnerWith(&bytes.Buffer{}, &bytes.Buffer{}, []string{"--script", "x", "--log", log, "--max-bytes", "700"}, exec); got != 0 {
			t.Fatalf("run %d=%d", i, got)
		}
	}
	raw := mustRead(t, log)
	if len(raw) > 700 {
		t.Fatalf("len=%d", len(raw))
	}
	if len(raw) == 0 || raw[len(raw)-1] != '\n' {
		t.Fatal("not complete-line bounded")
	}
	for _, line := range strings.Split(strings.TrimSpace(string(raw)), "\n") {
		var v any
		if json.Unmarshal([]byte(line), &v) != nil {
			t.Fatalf("bad line %q", line)
		}
	}
}
func mustRead(t *testing.T, p string) []byte {
	t.Helper()
	b, e := os.ReadFile(p)
	if e != nil {
		t.Fatal(e)
	}
	return b
}

func TestWatchdogAuditHealthRequiresFreshReceipt(t *testing.T) {
	now := time.Date(2026, 8, 26, 16, 0, 0, 0, time.UTC)
	write := func(t *testing.T, recorded time.Time, verdict string, status int) string {
		t.Helper()
		path := filepath.Join(t.TempDir(), "audit.jsonl")
		receipt := watchdogAuditReceipt{Schema: "fak.watchdog-audit-receipt.v1", RecordedAt: recorded, Verdict: verdict, ExitStatus: status, Audit: json.RawMessage(`{"productivity":{"status_verdict":"green"},"tower_down":1}`)}
		body, err := json.Marshal(receipt)
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, append(body, '\n'), 0o600); err != nil {
			t.Fatal(err)
		}
		return path
	}

	t.Run("fresh preserves typed verdict and fields", func(t *testing.T) {
		path := write(t, now.Add(-15*time.Minute), "RED", 3)
		var out, stderr bytes.Buffer
		if got := runWatchdogAuditHealth(&out, &stderr, []string{"--log", path, "--max-age", "30m"}, func() time.Time { return now }); got != 3 {
			t.Fatalf("status=%d stderr=%s", got, stderr.String())
		}
		if !strings.Contains(out.String(), `"status_verdict":"green"`) || !strings.Contains(out.String(), `"tower_down":1`) {
			t.Fatalf("output lost audit fields: %s", out.String())
		}
	})

	t.Run("stale fails closed", func(t *testing.T) {
		path := write(t, now.Add(-31*time.Minute), "GREEN", 0)
		var stderr bytes.Buffer
		if got := runWatchdogAuditHealth(&bytes.Buffer{}, &stderr, []string{"--log", path, "--max-age", "30m"}, func() time.Time { return now }); got != 1 || !strings.Contains(stderr.String(), "stale") {
			t.Fatalf("status=%d stderr=%s", got, stderr.String())
		}
	})
}
