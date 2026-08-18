package main

import (
	"path/filepath"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/dispatchtick"
)

func TestDispatchWorkerEnvCodexStartsFreshThread(t *testing.T) {
	t.Setenv("CODEX_THREAD_ID", "interactive-thread")
	root := t.TempDir()
	account := dispatchtick.Account{Dir: filepath.Join(root, "codex-home"), Tag: "worker-a"}

	env, err := dispatchWorkerEnv("codex", "cmd", root, filepath.Join(root, "runs"), account, "high-priority", "high-priority")
	if err != nil {
		t.Fatal(err)
	}
	if got := env["CODEX_THREAD_ID"]; got != "" {
		t.Fatalf("CODEX_THREAD_ID = %q, want unset for detached worker", got)
	}
	if got := env["CODEX_HOME"]; got != account.Dir {
		t.Fatalf("CODEX_HOME = %q, want %q", got, account.Dir)
	}
}
