package guardaudit

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestPlanRequiresMatchingIndependentWitness(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	old := filepath.Join(root, "old.jsonl")
	changed := filepath.Join(root, "changed.jsonl")
	fresh := filepath.Join(root, "fresh.jsonl")
	for path, body := range map[string]string{old: "old\n", changed: "new bytes\n", fresh: "fresh\n"} {
		if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
			t.Fatal(err)
		}
	}
	now := time.Date(2026, 7, 15, 12, 0, 0, 0, time.UTC)
	past := now.Add(-8 * 24 * time.Hour)
	if err := os.Chtimes(old, past, past); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(changed, past, past); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(fresh, now, now); err != nil {
		t.Fatal(err)
	}
	oldHash, _ := hashPath(old)
	freshHash, _ := hashPath(fresh)
	witnessed := map[string]string{
		"guard-audit/old.jsonl":     oldHash,
		"guard-audit/changed.jsonl": oldHash, // stale capture must not authorize deletion
		"guard-audit/fresh.jsonl":   freshHash,
	}
	rep, err := Plan(repo, "vault", now, DefaultMaxAge, DefaultMaxFiles, witnessed)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Candidates) != 1 || rep.Candidates[0].Path != old || rep.Candidates[0].Reason != "age" {
		t.Fatalf("candidates=%+v", rep.Candidates)
	}
	if err := Apply(&rep); err != nil {
		t.Fatal(err)
	}
	if rep.GuardAuditPruned != 1 {
		t.Fatalf("pruned=%d", rep.GuardAuditPruned)
	}
	if _, err := os.Stat(old); !os.IsNotExist(err) {
		t.Fatalf("old file still exists: %v", err)
	}
	for _, path := range []string{changed, fresh} {
		if _, err := os.Stat(path); err != nil {
			t.Fatalf("retained %s: %v", path, err)
		}
	}
}

func TestPlanCountRetentionKeepsNewest(t *testing.T) {
	repo := t.TempDir()
	root := filepath.Join(repo, ".dispatch-runs", "guard-audit")
	if err := os.MkdirAll(root, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	witnessed := map[string]string{}
	for i, name := range []string{"a.jsonl", "b.jsonl", "c.jsonl"} {
		path := filepath.Join(root, name)
		if err := os.WriteFile(path, []byte(name), 0o644); err != nil {
			t.Fatal(err)
		}
		mt := now.Add(time.Duration(i) * time.Minute)
		if err := os.Chtimes(path, mt, mt); err != nil {
			t.Fatal(err)
		}
		h, _ := hashPath(path)
		witnessed["guard-audit/"+name] = h
	}
	rep, err := Plan(repo, "vault", now, 0, 2, witnessed)
	if err != nil {
		t.Fatal(err)
	}
	if len(rep.Candidates) != 1 || filepath.Base(rep.Candidates[0].Path) != "a.jsonl" || rep.Candidates[0].Reason != "count" {
		t.Fatalf("candidates=%+v", rep.Candidates)
	}
}
