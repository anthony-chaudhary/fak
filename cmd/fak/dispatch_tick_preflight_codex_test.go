package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDispatchCodexAmbientAccountHonorsCodexHome(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "auth.json"), []byte("{}"), 0o600); err != nil {
		t.Fatal(err)
	}
	t.Setenv("CODEX_HOME", dir)

	got := dispatchCodexAmbientAccount()
	if !got.Available {
		t.Fatalf("account unavailable: %+v", got)
	}
	if got.Dir != dir {
		t.Fatalf("Dir = %q, want %q", got.Dir, dir)
	}
	if got.Reason != "ambient CODEX_HOME login" {
		t.Fatalf("Reason = %q", got.Reason)
	}
}

func TestDispatchCodexAmbientAccountReportsConfiguredHome(t *testing.T) {
	dir := t.TempDir()
	t.Setenv("CODEX_HOME", dir)

	got := dispatchCodexAmbientAccount()
	if got.Available {
		t.Fatalf("account unexpectedly available: %+v", got)
	}
	if got.Dir != dir {
		t.Fatalf("Dir = %q, want %q", got.Dir, dir)
	}
}
