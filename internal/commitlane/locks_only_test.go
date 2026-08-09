package commitlane

import (
	"context"
	"path/filepath"
	"strings"
	"testing"
)

// TestStatusLocksOnlySkipsStagedDeletionAudit keeps an unattended lock preflight scoped
// to locks: it must not inspect peers' unrelated staged changes in the shared index.
func TestStatusLocksOnlySkipsStagedDeletionAudit(t *testing.T) {
	root := t.TempDir()
	gitDir := filepath.Join(root, ".git")
	var calls []string
	run := func(_ context.Context, _ string, args ...string) RunResult {
		joined := strings.Join(args, " ")
		calls = append(calls, joined)
		switch {
		case strings.Contains(joined, "rev-parse --show-toplevel"):
			return RunResult{Stdout: root}
		case strings.Contains(joined, "rev-parse --absolute-git-dir"):
			return RunResult{Stdout: gitDir}
		default:
			return RunResult{}
		}
	}

	_, err := Status(context.Background(), Options{
		Dir: root, Runner: run, LocksOnly: true,
		ProcessList: func(context.Context) ([]Process, error) { return nil, nil },
	})
	if err != nil {
		t.Fatal(err)
	}
	for _, call := range calls {
		if strings.Contains(call, " diff ") {
			t.Fatalf("LocksOnly inspected staged changes: %q", call)
		}
	}
}
