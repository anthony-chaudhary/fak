package scoreboard

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestWorktreeStatusRendersAuditorCounts(t *testing.T) {
	ledger := filepath.Join(t.TempDir(), "wave.jsonl")
	if err := os.WriteFile(ledger, []byte("{\"verdict\":\"TREE_POISONED\"}\n"), 0644); err != nil {
		t.Fatal(err)
	}
	git := "worktree C:/tmp/fak-worker-wt-a\nHEAD a\n\nworktree C:/tmp/fak-worker-wt-b\nHEAD b\n\nworktree C:/work/fak\nHEAD c\n"
	status := FoldWorktreeStatus(git, ledger)
	if status.IsolatedWorktrees != 2 || status.PoisonIncidents != 1 {
		t.Fatalf("status=%+v", status)
	}
	text := WithWorktreeStatus(Update{Title: "fleet"}, status).Text()
	if !strings.Contains(text, "2 isolated worktrees, 1 poison incidents") {
		t.Fatalf("text=%q", text)
	}
}
