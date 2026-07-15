package fleetreap

import (
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"
)

func writeAt(t *testing.T, dir, name, body string, mod time.Time) string {
	t.Helper()
	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, []byte(body), 0o600); err != nil {
		t.Fatal(err)
	}
	if err := os.Chtimes(path, mod, mod); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestReapByAgeCountKeepsNewestSurvivors(t *testing.T) {
	dir := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	for i := 0; i < 6; i++ {
		writeAt(t, dir, fmt.Sprintf("row-%d.jsonl", i), "x", now.Add(-time.Duration(i)*time.Hour))
	}
	res, err := ReapByAgeCount(dir, "*.jsonl", 4*time.Hour, 3, now)
	if err != nil {
		t.Fatal(err)
	}
	if res.Before.Files != 6 || res.After.Files != 3 || res.Removed != 3 {
		t.Fatalf("result = %+v", res)
	}
	for _, name := range []string{"row-0.jsonl", "row-1.jsonl", "row-2.jsonl"} {
		if _, err := os.Stat(filepath.Join(dir, name)); err != nil {
			t.Fatalf("new survivor %s missing: %v", name, err)
		}
	}
}

func TestReapByDeadOwnerSparesLiveAndUnknown(t *testing.T) {
	dir := t.TempDir()
	now := time.Now()
	writeAt(t, dir, "seed-11.tmp", "dead", now)
	writeAt(t, dir, "seed-22.tmp", "live", now)
	writeAt(t, dir, "seed-unknown.tmp", "unknown", now)
	res, err := ReapByDeadOwner(dir, "*.tmp", func(pid int) bool { return pid == 22 }, now)
	if err != nil {
		t.Fatal(err)
	}
	if res.Removed != 1 || res.After.Files != 2 {
		t.Fatalf("result = %+v", res)
	}
	if _, err := os.Stat(filepath.Join(dir, "seed-22.tmp")); err != nil {
		t.Fatal("live owner removed")
	}
	if _, err := os.Stat(filepath.Join(dir, "seed-unknown.tmp")); err != nil {
		t.Fatal("unknown owner removed")
	}
}

func TestMeasureFootprint(t *testing.T) {
	dir := t.TempDir()
	now := time.Unix(1_700_000_000, 0)
	oldest := now.Add(-3 * time.Hour)
	writeAt(t, dir, "a.jsonl", "abc", oldest)
	writeAt(t, dir, "b.jsonl", "12345", now.Add(-time.Hour))
	got, err := MeasureFootprint(dir, "*.jsonl", now)
	if err != nil {
		t.Fatal(err)
	}
	if got.Files != 2 || got.Bytes != 8 || !got.Oldest.Equal(oldest) || got.OldestAge != 3*time.Hour {
		t.Fatalf("footprint = %+v", got)
	}
}
