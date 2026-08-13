package main

import (
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/microagent"
)

func TestDispatchPromptIncludesCollapsedRepoPulseByDefault(t *testing.T) {
	root := dispatchPulseFixture(t)
	got, err := dispatchPrompt(root, nil, 7001, "cmd", dispatchIssueInfo{Number: 7001, Title: "pulse", Body: "## Done condition\nship it", State: "OPEN"})
	if err != nil {
		t.Fatal(err)
	}
	prompt := dispatchMapString(got, "prompt")
	for _, want := range []string{"Repository orientation (governed collapsed child", "repo pulse - status:", "tool_turns_skipped=2", "journal_rows=3"} {
		if !strings.Contains(prompt, want) {
			t.Fatalf("prompt missing %q:\n%s", want, prompt)
		}
	}
	if got["repo_pulse_default"] != true {
		t.Fatalf("repo_pulse_default=%v", got["repo_pulse_default"])
	}
	marker := "Repository orientation (governed collapsed child; do not rerun these reads unless state changes):"
	_, orientation, ok := strings.Cut(prompt, marker)
	if !ok {
		t.Fatal("orientation marker missing")
	}
	added := microagent.NewContext(4096)
	added.Append("user", orientation)
	if added.Tokens() >= 300 {
		t.Fatalf("collapsed startup orientation grew by %d tokens, want <300", added.Tokens())
	}
}

func TestDispatchRepoPulseCanBeDisabled(t *testing.T) {
	t.Setenv(dispatchRepoPulseEnv, "off")
	if got, _, err := dispatchRepoPulseOrientation(t.TempDir()); err != nil || got != "" {
		t.Fatalf("got=%q err=%v", got, err)
	}
}

func dispatchPulseFixture(t *testing.T) string {
	t.Helper()
	dir := t.TempDir()
	run := func(args ...string) {
		cmd := exec.Command("git", args...)
		cmd.Dir = dir
		if out, err := cmd.CombinedOutput(); err != nil {
			t.Fatalf("git %v: %v: %s", args, err, out)
		}
	}
	run("init", "-q")
	run("config", "user.email", "pulse@example.invalid")
	run("config", "user.name", "Pulse Test")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("base\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	run("add", "tracked.txt")
	run("commit", "-q", "-m", "base")
	if err := os.WriteFile(filepath.Join(dir, "tracked.txt"), []byte("base\nchanged\n"), 0o644); err != nil {
		t.Fatal(err)
	}
	return dir
}
