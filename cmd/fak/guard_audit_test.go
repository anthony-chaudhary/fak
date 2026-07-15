package main

import (
	"bytes"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/logvault"
)

func TestRunGuardAuditPruneApplyUsesVerifiedVault(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(root, "old.jsonl")
	if err := os.WriteFile(journal, []byte("witnessed\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	if err := os.Chtimes(journal, old, old); err != nil {
		t.Fatal(err)
	}
	vaultDir := filepath.Join(t.TempDir(), "vault")
	v := &logvault.Vault{Dir: vaultDir, Sources: []logvault.Source{{ID: "dispatch-runs", Root: filepath.Join(repo, ".dispatch-runs")}}}
	if _, err := v.Capture(); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	now := time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC)
	code := runGuardAuditPrune([]string{"--apply", "--json", "--repo", repo, "--vault", vaultDir}, &out, &stderr, now)
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got struct {
		GuardAuditPruned int `json:"guard_audit_pruned"`
	}
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.GuardAuditPruned != 1 {
		t.Fatalf("report=%s", out.String())
	}
	if _, err := os.Stat(journal); !os.IsNotExist(err) {
		t.Fatalf("journal remains: %v", err)
	}
}

func TestRunGuardAuditPruneRetainsFileWithoutVaultWitness(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	journal := filepath.Join(root, "old.jsonl")
	if err := os.WriteFile(journal, []byte("not captured\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	old := time.Date(2026, 7, 1, 0, 0, 0, 0, time.UTC)
	if err := os.Chtimes(journal, old, old); err != nil {
		t.Fatal(err)
	}
	var out, stderr bytes.Buffer
	code := runGuardAuditPrune([]string{"--apply", "--json", "--repo", repo, "--vault", filepath.Join(t.TempDir(), "empty")}, &out, &stderr, time.Date(2026, 7, 15, 0, 0, 0, 0, time.UTC))
	if code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var got struct{ GuardAuditPruned, Unmirrored int }
	if err := json.Unmarshal(out.Bytes(), &got); err != nil {
		t.Fatal(err)
	}
	if got.GuardAuditPruned != 0 {
		t.Fatalf("report=%s", out.String())
	}
	if _, err := os.Stat(journal); err != nil {
		t.Fatalf("unwitnessed journal removed: %v", err)
	}
}
