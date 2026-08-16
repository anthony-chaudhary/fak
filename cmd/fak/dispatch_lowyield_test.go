package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDispatchLowYieldExcludesSkipsMissingHelper(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, "tools"), 0o755); err != nil {
		t.Fatal(err)
	}
	got := dispatchLowYieldExcludes(root)
	if got != nil {
		t.Fatalf("excludes = %v, want nil when helper is absent", got)
	}
}
