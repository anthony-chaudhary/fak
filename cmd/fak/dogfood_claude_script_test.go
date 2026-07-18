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

// TestDogfoodClaudeProvenanceReadBackBeforeSwap guards the fail-closed provenance
// read-back (#5211): before the lock-safe install swap, the script must run both the
// freshly built binary and the staged .new copy and refuse to install unless each
// reports a build: revision equal to the repo HEAD's 12-char prefix.
func TestDogfoodClaudeProvenanceReadBackBeforeSwap(t *testing.T) {
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
		"rev-parse --short=12 HEAD",
		"provenance read-back FAILED",
		`build:\s*([0-9a-fA-F]{12})`,
	} {
		if !strings.Contains(text, want) {
			t.Fatalf("dogfood provenance read-back missing %q", want)
		}
	}
	readBackIdx := strings.Index(text, "provenance read-back OK")
	swapIdx := strings.Index(text, "Move-Item -Force $new")
	if readBackIdx < 0 {
		t.Fatal("dogfood provenance read-back OK marker not found")
	}
	if swapIdx < 0 {
		t.Fatal("dogfood install swap (Move-Item -Force $new) not found")
	}
	if readBackIdx > swapIdx {
		t.Fatalf("provenance read-back (idx %d) must run BEFORE the install swap (idx %d)", readBackIdx, swapIdx)
	}
}
