package main

import (
	"io"
	"os"
	"path/filepath"
	"strings"
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

// TestStageDispatchPromptStdinSurvivesLauncherExit is the #11491 root-cause witness:
// a detached codex worker must read its prompt from a real file descriptor backed by
// a durable artifact, NOT from an in-memory reader whose os/exec copy goroutine dies
// with the short-lived dispatch tick (which truncated stdin to EOF and produced
// PROMPT_FUEL_MISSING). The staged reader is verified to be a materialized file with
// the exact bytes, so the guard's later stdin drain cannot race the launcher's exit.
func TestStageDispatchPromptStdinSurvivesLauncherExit(t *testing.T) {
	dir := t.TempDir()
	stem := filepath.Join(dir, "resolve-11491-20260913-000000")
	prompt := "issue #11491\r\npreserve prompt fuel λ\n" + strings.Repeat("fuel\x00", 4096)

	f, err := stageDispatchPromptStdin(stem, prompt)
	if err != nil {
		t.Fatalf("stageDispatchPromptStdin: %v", err)
	}
	defer f.Close()

	// The field type is *os.File by construction (see signature); assert the durable
	// artifact exists on disk with the exact bytes so the child fd cannot point at a
	// transient pipe buffer.
	promptPath := stem + dispatchPromptSidecarSuffix
	onDisk, err := os.ReadFile(promptPath)
	if err != nil {
		t.Fatalf("read staged prompt artifact %s: %v", promptPath, err)
	}
	if string(onDisk) != prompt {
		t.Fatal("staged prompt artifact changed bytes")
	}
	got, err := io.ReadAll(f)
	if err != nil {
		t.Fatalf("read staged stdin: %v", err)
	}
	if string(got) != prompt {
		t.Fatalf("staged stdin changed bytes (got %d, want %d)", len(got), len(prompt))
	}
}

// TestStageDispatchPromptStdinRefusesEmptyFuel pins the no-safety-bypass contract:
// staging an empty prompt must refuse with the same closed token the guard uses,
// before any child is constructed — never silently hand a truncated prompt to a
// guarded worker.
func TestStageDispatchPromptStdinRefusesEmptyFuel(t *testing.T) {
	for _, prompt := range []string{"", "   \n\t"} {
		stem := filepath.Join(t.TempDir(), "resolve-11491")
		f, err := stageDispatchPromptStdin(stem, prompt)
		if err == nil || f != nil {
			t.Fatalf("stageDispatchPromptStdin(%q) = file %v err %v, want refusal", prompt, f, err)
		}
		if !strings.Contains(err.Error(), promptFuelMissingReason) {
			t.Fatalf("refusal %q does not name %s", err, promptFuelMissingReason)
		}
	}
}
