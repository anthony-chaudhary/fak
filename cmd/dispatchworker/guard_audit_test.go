package main

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPruneGuardAuditTickCapturesBeforePrune(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(root, "old.jsonl")
	if err := os.WriteFile(journal, []byte("audit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	old := now.Add(-8 * 24 * time.Hour)
	if err := os.Chtimes(journal, old, old); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_LOG_VAULT", filepath.Join(t.TempDir(), "vault"))
	if got := pruneGuardAuditTick(repo, now); got != 1 {
		t.Fatalf("guard_audit_pruned=%d", got)
	}
	if _, err := os.Stat(journal); !os.IsNotExist(err) {
		t.Fatalf("journal remains: %v", err)
	}
}

func TestPruneGuardAuditTickFailsOpenAndRetainsOnVaultFailure(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(root, "old.jsonl")
	if err := os.WriteFile(journal, []byte("audit\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	old := now.Add(-8 * 24 * time.Hour)
	if err := os.Chtimes(journal, old, old); err != nil {
		t.Fatal(err)
	}
	blocked := filepath.Join(t.TempDir(), "not-a-dir")
	if err := os.WriteFile(blocked, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
	t.Setenv("FAK_LOG_VAULT", blocked)
	if got := pruneGuardAuditTick(repo, now); got != 0 {
		t.Fatalf("guard_audit_pruned=%d", got)
	}
	if _, err := os.Stat(journal); err != nil {
		t.Fatalf("journal lost: %v", err)
	}
}
