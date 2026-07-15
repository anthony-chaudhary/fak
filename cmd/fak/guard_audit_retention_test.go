package main

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func TestGuardAuditRetentionBoundsDirectory(t *testing.T) {
	root := t.TempDir()
	dir := guardAuditDir(root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < guardAuditRetentionCount+9; i++ {
		path := filepath.Join(dir, fmt.Sprintf("interactive-%04d.jsonl", i))
		if err := os.WriteFile(path, []byte("row\n"), 0o600); err != nil {
			t.Fatal(err)
		}
		mod := now.Add(-time.Duration(i) * time.Minute)
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	res, err := reapGuardAuditJournals(root, now)
	if err != nil {
		t.Fatal(err)
	}
	if res.Before.Files != guardAuditRetentionCount+9 || res.After.Files != guardAuditRetentionCount || res.Removed != 9 {
		t.Fatalf("retention result = %+v", res)
	}
	foot, err := measureGuardAuditJournals(root, now)
	if err != nil {
		t.Fatal(err)
	}
	if foot.Files != guardAuditRetentionCount || foot.Bytes != int64(guardAuditRetentionCount*4) {
		t.Fatalf("footprint = %+v", foot)
	}
}

func TestGuardAuditRetentionDropsExpiredBeforeCount(t *testing.T) {
	root := t.TempDir()
	dir := guardAuditDir(root)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}
	now := time.Now()
	for i, age := range []time.Duration{time.Hour, guardAuditRetentionAge + time.Hour} {
		path := filepath.Join(dir, fmt.Sprintf("row-%d.jsonl", i))
		if err := os.WriteFile(path, []byte("x"), 0o600); err != nil {
			t.Fatal(err)
		}
		mod := now.Add(-age)
		if err := os.Chtimes(path, mod, mod); err != nil {
			t.Fatal(err)
		}
	}
	res, err := reapGuardAuditJournals(root, now)
	if err != nil {
		t.Fatal(err)
	}
	if res.Removed != 1 || res.After.Files != 1 {
		t.Fatalf("retention result = %+v", res)
	}
}
