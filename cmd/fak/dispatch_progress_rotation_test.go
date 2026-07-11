package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDispatchProgressAppendRotatesAtWriteSite(t *testing.T) {
	path := filepath.Join(t.TempDir(), "progress.jsonl")
	if err := os.WriteFile(path, []byte("old-row\n"), 0644); err != nil {
		t.Fatal(err)
	}
	old := dispatchProgressRotateBytes
	dispatchProgressRotateBytes = 1
	t.Cleanup(func() { dispatchProgressRotateBytes = old })
	if err := dispatchProgressAppend(path, map[string]any{"schema": "new"}); err != nil {
		t.Fatal(err)
	}
	sealed, err := os.ReadFile(path + ".1")
	if err != nil || string(sealed) != "old-row\n" {
		t.Fatalf("sealed=%q err=%v", sealed, err)
	}
	active, err := os.ReadFile(path)
	if err != nil || len(active) == 0 {
		t.Fatalf("active=%q err=%v", active, err)
	}
}
