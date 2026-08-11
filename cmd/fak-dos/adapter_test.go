package main

import (
	"path/filepath"
	"testing"
	"time"
)

func TestAddReadIdempotentCleanup(t *testing.T) {
	ws := t.TempDir()
	now := time.Date(2026, 8, 10, 12, 0, 0, 0, time.UTC)
	payload := map[string]any{"case": "blank-reason"}
	first, created, err := Add(ws, "bench-6349-blank", "OPEN_ISSUE", "P1", payload, now)
	if err != nil || !created {
		t.Fatalf("first add: created=%v err=%v", created, err)
	}
	again, created, err := Add(ws, "bench-6349-blank", "OPEN_ISSUE", "P1", payload, now.Add(time.Hour))
	if err != nil || created || again.CreatedAt != first.CreatedAt {
		t.Fatalf("idempotent add: row=%+v created=%v err=%v", again, created, err)
	}
	rows, err := Read(ws)
	if err != nil || len(rows) != 1 {
		t.Fatalf("read: rows=%+v err=%v", rows, err)
	}
	if rows[0].Action != "OPEN_ISSUE" || rows[0].Severity != "P1" || rows[0].SourcePath != filepath.Join(ws, filepath.FromSlash(relativeLog)) {
		t.Fatalf("wrong row: %+v", rows[0])
	}
	removed, err := Remove(ws, first.Key)
	if err != nil || !removed {
		t.Fatalf("remove: removed=%v err=%v", removed, err)
	}
	removed, err = Remove(ws, first.Key)
	if err != nil || removed {
		t.Fatalf("idempotent remove: removed=%v err=%v", removed, err)
	}
	rows, err = Read(ws)
	if err != nil || len(rows) != 0 {
		t.Fatalf("read after cleanup: rows=%+v err=%v", rows, err)
	}
}

func TestAddRejectsConflictingKey(t *testing.T) {
	ws := t.TempDir()
	now := time.Now()
	if _, _, err := Add(ws, "same", "OPEN_ISSUE", "P1", nil, now); err != nil {
		t.Fatal(err)
	}
	if _, _, err := Add(ws, "same", "NOOP", "none", nil, now); err == nil {
		t.Fatal("expected conflicting key error")
	}
	rows, _ := Read(ws)
	if len(rows) != 1 || rows[0].Action != "OPEN_ISSUE" {
		t.Fatalf("conflict changed row: %+v", rows)
	}
}
