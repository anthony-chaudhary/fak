package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBoundWatchdogArtifactsRotatesAndPrunes(t *testing.T) {
	d := t.TempDir()
	ledger := filepath.Join(d, "resume_ledger.jsonl")
	t.Setenv("FAK_WATCHDOG_LOG_MAX_BYTES", "4")
	t.Setenv("FAK_RESUME_LEDGER_COMPACT_BYTES", "4")
	t.Setenv("FAK_RESUME_LOG_RETAIN_DAYS", "1")
	for _, p := range []string{filepath.Join(d, "resume_watchdog.log"), filepath.Join(d, "notifications.log"), ledger} {
		if err := os.WriteFile(p, []byte("1234"), 0600); err != nil {
			t.Fatal(err)
		}
	}
	old := filepath.Join(d, "resume-dead-1.log")
	if err := os.WriteFile(old, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	past := time.Now().Add(-48 * time.Hour)
	_ = os.Chtimes(old, past, past)
	rwBoundWatchdogArtifacts(d, ledger, time.Now())
	for _, p := range []string{filepath.Join(d, "resume_watchdog.log.1"), filepath.Join(d, "notifications.log.1"), ledger + ".1"} {
		if _, err := os.Stat(p); err != nil {
			t.Fatalf("missing rotated %s: %v", p, err)
		}
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("expired log remains: %v", err)
	}
}

func TestCompactResumeLedgerKeepsSettledRows(t *testing.T) {
	p := filepath.Join(t.TempDir(), "ledger.jsonl")
	rows := "{\"ts\":\"2020-01-01T00:00:00Z\",\"action\":\"launch\"}\n{\"ts\":\"2020-01-01T00:00:00Z\",\"action\":\"consolidate-settled\"}\n{\"ts\":\"bad\",\"action\":\"launch\"}\n"
	if err := os.WriteFile(p, []byte(rows), 0600); err != nil {
		t.Fatal(err)
	}
	if got := rwCompactResumeLedger(p, 30, 1, time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)); got != 1 {
		t.Fatalf("dropped=%d", got)
	}
	b, _ := os.ReadFile(p)
	text := string(b)
	if strings.Contains(text, `"action":"launch"}`+"\n"+`{"ts":"2020`) || !strings.Contains(text, "consolidate-settled") || !strings.Contains(text, `"ts":"bad"`) {
		t.Fatalf("ledger=%s", text)
	}
}
