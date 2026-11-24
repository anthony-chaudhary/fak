package issueownerprompt

import (
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/testenv/repotest"
)

// Invariant: Goal prompts directory validation must enforce canonical lifecycle inclusions.
// Guard: ValidateDir returns an error if lifecycle invariants are missing.

func TestIssueOwnerPromptLifecycle(t *testing.T) {
	t.Parallel()

	dir := repotest.Path(t, ".claude/goal-prompts")

	if _, err := os.Stat(dir); os.IsNotExist(err) {
		t.Skip("goal-prompts dir does not exist")
	}

	if err := ValidateDir(dir); err != nil {
		t.Fatalf("ValidateDir failed: %v", err)
	}
}
