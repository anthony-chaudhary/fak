package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDefaultResumeRegistryPrefersHostFleetRegistry(t *testing.T) {
	t.Setenv("FLEET_REG_DIR", "")
	t.Setenv("FLEET_STATE_DIR", "")
	local := t.TempDir()
	t.Setenv("LOCALAPPDATA", local)
	want := filepath.Join(local, "Fleet", "registry")
	if err := os.MkdirAll(want, 0o755); err != nil {
		t.Fatal(err)
	}

	if got := resolveSweepRegDir(""); got != want {
		t.Fatalf("resolveSweepRegDir = %q, want %q", got, want)
	}
	if got := defaultResumeLedger(); got != filepath.Join(want, "resume_ledger.jsonl") {
		t.Fatalf("defaultResumeLedger = %q, want under %q", got, want)
	}
}
