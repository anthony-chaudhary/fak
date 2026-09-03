package main

import (
	"bytes"
	"crypto/sha256"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
)

const guardWindowsPromptStdinThreshold = 7 << 10

const (
	promptFuelMissingReason  = "PROMPT_FUEL_MISSING"
	promptFuelTamperedReason = "PROMPT_FUEL_TAMPERED"
)

// promptFuel is the immutable, replayable stdin for a guarded Codex
// `exec -` worker. Dispatch hands the rendered issue prompt to guard through a
// one-shot pipe; persisting it before the first child exists lets every crash or
// budget relaunch verify the same bytes and obtain a fresh reader.
type promptFuel struct {
	path   string
	digest string

	mu       sync.Mutex
	launches int
}

func preparePromptFuel(command []string, stdin io.Reader, runID string) (*promptFuel, error) {
	if !codexPromptFuelRequired(command) {
		return nil, nil
	}
	if stdin == nil {
		return nil, fmt.Errorf("%s: guarded codex exec requires prompt bytes on stdin", promptFuelMissingReason)
	}
	prompt, err := io.ReadAll(stdin)
	if err != nil {
		return nil, fmt.Errorf("read guarded codex prompt fuel: %w", err)
	}
	return persistPromptFuel(promptFuelPath(runID), prompt)
}

func codexPromptFuelRequired(command []string) bool {
	plan := newGuardLaunchPlan(command)
	if plan.agentBaseName() != "codex" || guardCodexSemanticSubcommand(plan.semanticCommand()) != "exec" {
		return false
	}
	for _, arg := range plan.semanticCommand() {
		if arg == "-" {
			return true
		}
	}
	return false
}

func promptFuelPath(runID string) string {
	workspace := strings.TrimSpace(os.Getenv("DISPATCH_WORKSPACE"))
	if workspace == "" {
		workspace = findRepoRoot(".")
	}
	identity := sha256.Sum256([]byte(strings.TrimSpace(runID)))
	return filepath.Join(workspace, ".dispatch-runs", "prompt-fuel", fmt.Sprintf("%x.prompt", identity[:12]))
}

func persistPromptFuel(path string, prompt []byte) (*promptFuel, error) {
	if len(prompt) == 0 {
		return nil, fmt.Errorf("%s: guarded codex exec received empty prompt fuel", promptFuelMissingReason)
	}
	if strings.TrimSpace(path) == "" {
		return nil, fmt.Errorf("%s: prompt fuel path is empty", promptFuelMissingReason)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return nil, fmt.Errorf("create prompt fuel directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".prompt-fuel-*.tmp")
	if err != nil {
		return nil, fmt.Errorf("create prompt fuel artifact: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return nil, fmt.Errorf("protect prompt fuel artifact: %w", err)
	}
	if _, err := tmp.Write(prompt); err != nil {
		_ = tmp.Close()
		return nil, fmt.Errorf("write prompt fuel artifact: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return nil, fmt.Errorf("sync prompt fuel artifact: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return nil, fmt.Errorf("close prompt fuel artifact: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return nil, fmt.Errorf("publish prompt fuel artifact: %w", err)
	}
	return &promptFuel{path: path, digest: promptFuelDigest(prompt)}, nil
}

func promptFuelDigest(prompt []byte) string {
	sum := sha256.Sum256(prompt)
	return fmt.Sprintf("sha256:%x", sum[:])
}

// reader verifies the durable artifact before returning a private reader over
// the verified bytes. Reading before child construction closes the check/use
// race for this launch: later file mutation cannot alter the reader already
// handed to exec.Cmd, and the next relaunch verifies the artifact again.
func (f *promptFuel) reader() (io.Reader, error) {
	if f == nil {
		return nil, nil
	}
	prompt, err := os.ReadFile(f.path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%s: replayable guarded codex prompt fuel is absent", promptFuelMissingReason)
		}
		return nil, fmt.Errorf("read replayable guarded codex prompt fuel: %w", err)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("%s: replayable guarded codex prompt fuel is empty", promptFuelMissingReason)
	}
	if got := promptFuelDigest(prompt); got != f.digest {
		return nil, fmt.Errorf("%s: guarded codex prompt fuel digest mismatch (got %s, want %s)", promptFuelTamperedReason, got, f.digest)
	}
	f.mu.Lock()
	f.launches++
	f.mu.Unlock()
	return bytes.NewReader(prompt), nil
}

func (f *promptFuel) receipt() string {
	if f == nil {
		return ""
	}
	f.mu.Lock()
	restarts := f.launches - 1
	if restarts < 0 {
		restarts = 0
	}
	f.mu.Unlock()
	return fmt.Sprintf("prompt_fuel_digest=%s restart_count=%d", f.digest, restarts)
}

// guardPromptStdinTransport moves a large Claude print prompt off the Windows
// command line. Claude's -p/--print mode reads the prompt from stdin when the
// flag has no value. This keeps every other argument byte-for-byte unchanged.
func guardPromptStdinTransport(command []string) ([]string, string, bool) {
	return guardPromptStdinTransportForOS(command, runtime.GOOS)
}

func guardPromptStdinTransportForOS(command []string, goos string) ([]string, string, bool) {
	if goos != "windows" || len(command) < 3 {
		return command, "", false
	}
	claudeIndex := guardClaudeCommandIndex(command)
	if claudeIndex < 0 {
		return command, "", false
	}
	for i := claudeIndex + 1; i+1 < len(command); i++ {
		if command[i] != "-p" && command[i] != "--print" {
			continue
		}
		prompt := command[i+1]
		if len(prompt) < guardWindowsPromptStdinThreshold {
			return command, "", false
		}
		out := make([]string, 0, len(command)-1)
		out = append(out, command[:i+1]...)
		out = append(out, command[i+2:]...)
		return out, prompt, true
	}
	return command, "", false
}

func guardClaudeCommandIndex(command []string) int {
	for i, arg := range command {
		if i > 0 && command[i-1] != "--" {
			continue
		}
		name := strings.TrimSuffix(strings.ToLower(filepath.Base(arg)), ".exe")
		if name == "claude" {
			return i
		}
	}
	return -1
}

func applyGuardPromptStdinTransport(child *exec.Cmd, command []string, goos string) ([]string, bool) {
	command, prompt, moved := guardPromptStdinTransportForOS(command, goos)
	if moved {
		child.Stdin = strings.NewReader(prompt)
	}
	return command, moved
}
