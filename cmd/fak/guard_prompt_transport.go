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

// promptFuel is the immutable, replayable stdin for a guarded Codex or Claude
// worker. Dispatch hands the rendered issue prompt to guard through a
// one-shot pipe; persisting it before the first child exists lets every crash or
// budget relaunch verify the same bytes and obtain a fresh reader.
type promptFuel struct {
	path   string
	digest string
	mem    []byte

	mu       sync.Mutex
	launches int
}

func preparePromptFuel(command []string, stdin io.Reader, runID string) (*promptFuel, error) {
	if !guardPromptFuelRequired(command) {
		return nil, nil
	}
	if f, ok := stdin.(*os.File); ok && f == os.Stdin && cmdGuardStdinInteractive() {
		return nil, nil
	}
	agent := newGuardLaunchPlan(command).agentBaseName()
	if agent == "" {
		agent = "agent"
	}
	explicitDash := commandHasExplicitDashArg(command)
	if stdin == nil {
		if explicitDash {
			return nil, fmt.Errorf("%s: guarded %s requires prompt bytes on stdin", promptFuelMissingReason, agent)
		}
		return nil, nil
	}
	prompt, err := io.ReadAll(stdin)
	if err != nil {
		return nil, fmt.Errorf("read guarded %s prompt fuel: %w", agent, err)
	}
	if len(prompt) == 0 {
		if explicitDash {
			return nil, fmt.Errorf("%s: guarded %s received empty prompt fuel", promptFuelMissingReason, agent)
		}
		return nil, nil
	}
	return persistPromptFuel(promptFuelPath(runID), prompt)
}

func commandHasExplicitDashArg(command []string) bool {
	for _, arg := range command {
		if arg == "-" {
			return true
		}
	}
	return false
}

func guardPromptFuelRequired(command []string) bool {
	if strings.TrimSpace(os.Getenv("FAK_GUARD_PROMPT_FUEL")) == "1" {
		return true
	}
	if len(command) == 0 {
		return false
	}
	plan := newGuardLaunchPlan(command)
	switch plan.agentBaseName() {
	case "codex":
		if guardCodexSemanticSubcommand(plan.semanticCommand()) == "exec" {
			return codexExecExpectsStdinPrompt(plan.semanticCommand())
		}
		return false
	case "claude":
		return claudePrintExpectsStdinPrompt(plan.semanticCommand())
	default:
		return false
	}
}

func codexPromptFuelRequired(command []string) bool {
	return guardPromptFuelRequired(command)
}

func codexExecExpectsStdinPrompt(command []string) bool {
	for _, arg := range command {
		if arg == "-" {
			return true
		}
	}
	args := command
	for i, arg := range command {
		if arg == "exec" {
			args = command[i+1:]
			break
		}
	}
	inDashDash := false
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if inDashDash {
			if arg != "" && arg != "-" {
				return false
			}
			continue
		}
		if arg == "--" {
			inDashDash = true
			continue
		}
		if isCodexFlagWithValue(arg) {
			i++
			continue
		}
		if strings.HasPrefix(arg, "-") {
			continue
		}
		if arg != "-" {
			return false
		}
	}
	return true
}

func isCodexFlagWithValue(arg string) bool {
	switch arg {
	case "-m", "--model",
		"-c", "--config",
		"-s", "--sandbox",
		"-C", "--cd",
		"--workdir",
		"--profile",
		"-p", "--session",
		"--effort",
		"--output-last-message",
		"--log-level",
		"--env",
		"--target":
		return true
	default:
		return false
	}
}

func claudePrintExpectsStdinPrompt(command []string) bool {
	for _, arg := range command {
		if strings.HasPrefix(arg, "--print=") {
			return false
		}
	}
	for i, arg := range command {
		if arg == "-p" || arg == "--print" {
			if i+1 >= len(command) {
				return true
			}
			next := command[i+1]
			if strings.HasPrefix(next, "-") {
				return true
			}
			return false
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
		return nil, fmt.Errorf("%s: guarded prompt fuel is empty", promptFuelMissingReason)
	}
	digest := promptFuelDigest(prompt)

	if strings.TrimSpace(path) != "" {
		if err := writePromptFuelFile(path, prompt); err == nil {
			return &promptFuel{path: path, digest: digest}, nil
		}
	}

	base := filepath.Base(path)
	if base == "." || base == string(filepath.Separator) || strings.TrimSpace(base) == "" {
		base = fmt.Sprintf("%s.prompt", digest[len(digest)-12:])
	}
	fallbackPath := filepath.Join(os.TempDir(), "fak-prompt-fuel", base)
	if err := writePromptFuelFile(fallbackPath, prompt); err == nil {
		return &promptFuel{path: fallbackPath, digest: digest}, nil
	}

	return &promptFuel{
		digest: digest,
		mem:    append([]byte(nil), prompt...),
	}, nil
}

func writePromptFuelFile(path string, prompt []byte) error {
	if strings.TrimSpace(path) == "" {
		return fmt.Errorf("%s: prompt fuel path is empty", promptFuelMissingReason)
	}
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0o700); err != nil {
		return fmt.Errorf("create prompt fuel directory: %w", err)
	}
	tmp, err := os.CreateTemp(dir, ".prompt-fuel-*.tmp")
	if err != nil {
		return fmt.Errorf("create prompt fuel artifact: %w", err)
	}
	tmpPath := tmp.Name()
	defer os.Remove(tmpPath)
	if err := tmp.Chmod(0o600); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("protect prompt fuel artifact: %w", err)
	}
	if _, err := tmp.Write(prompt); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("write prompt fuel artifact: %w", err)
	}
	if err := tmp.Sync(); err != nil {
		_ = tmp.Close()
		return fmt.Errorf("sync prompt fuel artifact: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return fmt.Errorf("close prompt fuel artifact: %w", err)
	}
	if err := os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("publish prompt fuel artifact: %w", err)
	}
	return nil
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
	var prompt []byte
	if f.path != "" {
		data, err := os.ReadFile(f.path)
		if err == nil {
			prompt = data
		} else if !os.IsNotExist(err) && len(f.mem) == 0 {
			return nil, fmt.Errorf("read replayable guarded prompt fuel: %w", err)
		} else if os.IsNotExist(err) && len(f.mem) == 0 {
			return nil, fmt.Errorf("%s: replayable guarded prompt fuel is absent", promptFuelMissingReason)
		}
	}
	if len(prompt) == 0 && len(f.mem) > 0 {
		prompt = append([]byte(nil), f.mem...)
	}
	if len(prompt) == 0 {
		return nil, fmt.Errorf("%s: replayable guarded prompt fuel is empty", promptFuelMissingReason)
	}
	if got := promptFuelDigest(prompt); got != f.digest {
		return nil, fmt.Errorf("%s: guarded prompt fuel digest mismatch (got %s, want %s)", promptFuelTamperedReason, got, f.digest)
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
