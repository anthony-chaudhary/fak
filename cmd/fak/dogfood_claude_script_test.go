package main

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestDogfoodClaudeHeadFallbackStampsCommittedRevision(t *testing.T) {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	body, err := os.ReadFile(filepath.Join(root, "scripts", "dogfood-claude.ps1"))
	if err != nil {
		t.Fatal(err)
	}
	text := string(body)
	for _, want := range []string{
		"git -C $FakDir rev-parse HEAD",
		"internal/appversion.BuildCommit=$headCommit",
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("dogfood committed-HEAD fallback missing %q", want)
		}
	}
}
