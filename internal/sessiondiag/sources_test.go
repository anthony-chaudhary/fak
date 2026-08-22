package sessiondiag

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
	"time"
)

func TestReadCodexInventorySourcesPreservesReadOnlyEvidence(t *testing.T) {
	home := t.TempDir()
	lockDir := filepath.Join(home, "thread-writer-locks")
	receiptDir := filepath.Join(home, "fak-guarded-sessions")
	if err := os.MkdirAll(lockDir, 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(receiptDir, 0o755); err != nil {
		t.Fatal(err)
	}
	threadID := "91000001-0000-4000-8000-000000000001"
	if err := os.WriteFile(filepath.Join(lockDir, threadID+".lock"), nil, 0o600); err != nil {
		t.Fatal(err)
	}
	receipt := `{"schema":"fak.codex_guard_witness.v1","session_id":"` + threadID + `","guarded_at":"2026-08-22T12:00:00Z"}`
	if err := os.WriteFile(filepath.Join(receiptDir, threadID+".json"), []byte(receipt), 0o600); err != nil {
		t.Fatal(err)
	}

	binDir := t.TempDir()
	python := filepath.Join(binDir, "python")
	body := "#!/bin/sh\nprintf '%s' '{\"threads\":[{\"thread_id\":\"" + threadID + "\",\"source\":\"cli\",\"updated_at_ms\":1787414400000}],\"turns\":[],\"spawn_edges\":[],\"source_errors\":[]}'\n"
	if runtime.GOOS == "windows" {
		python += ".bat"
		body = "@echo off\r\necho {\"threads\":[{\"thread_id\":\"" + threadID + "\",\"source\":\"cli\",\"updated_at_ms\":1787414400000}],\"turns\":[],\"spawn_edges\":[],\"source_errors\":[]}\r\n"
	}
	if err := os.WriteFile(python, []byte(body), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", binDir)

	input, err := ReadCodexInventorySources(home, time.Hour, time.Date(2026, 8, 22, 12, 0, 0, 0, time.UTC))
	if err != nil {
		t.Fatal(err)
	}
	if len(input.Threads) != 1 || input.Threads[0].ThreadID != threadID {
		t.Fatalf("threads=%+v", input.Threads)
	}
	if len(input.WriterLocks) != 1 || input.WriterLocks[0].ThreadID != threadID {
		t.Fatalf("writer locks=%+v", input.WriterLocks)
	}
	if len(input.GuardReceipts) != 1 || input.GuardReceipts[0].ThreadID != threadID {
		t.Fatalf("guard receipts=%+v", input.GuardReceipts)
	}
}
