package issueownerprompt

import (
	"os"
	"path/filepath"
	"runtime"
	"testing"
)

// Invariant: Goal prompts directory validation must enforce canonical lifecycle inclusions.
// Guard: ValidateDir returns an error if lifecycle invariants are missing.

func TestIssueOwnerPromptLifecycle(t *testing.T) {
	t.Parallel()

	_, file, _, ok := runtime.Caller(0)
	if !ok {
		t.Fatal("runtime.Caller failed")
	}
	root := filepath.Clean(filepath.Join(filepath.Dir(file), "..", ".."))
	dir := filepath.Join(root, ".claude", "goal-prompts")

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Skip("goal-prompts dir does not exist")
	}

	if err := ValidateDir(dir); err != nil {
		t.Fatalf("ValidateDir failed: %v", err)
	}
}
