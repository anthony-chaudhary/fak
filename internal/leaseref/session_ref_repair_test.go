package leaseref

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestReapMalformedSessionRefsRemovesOnlyTornSessionRefs(t *testing.T) {
	gitDir := t.TempDir()
	locks := filepath.Join(gitDir, "refs", "fak", "locks")
	if err := os.MkdirAll(filepath.Join(locks, "nested"), 0o755); err != nil {
		t.Fatal(err)
	}
	valid := "0123456789abcdef0123456789abcdef01234567\n"
	files := map[string][]byte{
		"session-good":         []byte(valid),
		"session-torn":         make([]byte, 41),
		"session-short":        []byte("deadbeef\n"),
		"session-writing.lock": []byte("partial"),
		"ordinary-bad":         []byte("not-an-object\n"),
	}
	for name, data := range files {
		if err := os.WriteFile(filepath.Join(locks, name), data, 0o644); err != nil {
			t.Fatal(err)
		}
	}

	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir + "\n", 0, nil
	}}
	got, err := s.ReapMalformedSessionRefs(context.Background())
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 2 || got[0] != refPrefix+"session-short" || got[1] != refPrefix+"session-torn" {
		t.Fatalf("reaped = %#v", got)
	}
	for _, name := range []string{"session-short", "session-torn"} {
		if _, err := os.Stat(filepath.Join(locks, name)); !os.IsNotExist(err) {
			t.Fatalf("%s still exists: %v", name, err)
		}
	}
	for _, name := range []string{"session-good", "session-writing.lock", "ordinary-bad"} {
		if _, err := os.Stat(filepath.Join(locks, name)); err != nil {
			t.Fatalf("%s removed: %v", name, err)
		}
	}
}

func TestReapMalformedSessionRefsDefersToConcurrentGitWriter(t *testing.T) {
	gitDir := t.TempDir()
	locks := filepath.Join(gitDir, "refs", "fak", "locks")
	if err := os.MkdirAll(locks, 0o755); err != nil {
		t.Fatal(err)
	}
	path := filepath.Join(locks, "session-torn")
	if err := os.WriteFile(path, make([]byte, 41), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path+".lock", []byte("writer owns this"), 0o600); err != nil {
		t.Fatal(err)
	}
	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir, 0, nil
	}}
	got, err := s.ReapMalformedSessionRefs(context.Background())
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("concurrent writer's ref was removed: %v", err)
	}
}

func TestReapMalformedSessionRefsMissingDirectoryIsClean(t *testing.T) {
	gitDir := t.TempDir()
	s := &Store{run: func(_ context.Context, _ string, args ...string) (string, int, error) {
		return gitDir, 0, nil
	}}
	got, err := s.ReapMalformedSessionRefs(context.Background())
	if err != nil || len(got) != 0 {
		t.Fatalf("got=%v err=%v", got, err)
	}
}
