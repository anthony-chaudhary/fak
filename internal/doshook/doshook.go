// Package doshook provides the launcher that binds Claude Code's dos hooks
// to a versioned dos contract instead of reaching into dos.cli internals.
//
// Native-first, fall back to Python, always exit 0.
package doshook

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/internal/windowgate"
)

// Runner turns an argv slice and stdin buffer into (exitCode, stdoutBytes, err).
type Runner func(argv []string, stdin []byte) (exitCode int, stdout []byte, err error)

// RepoRoot returns the workspace root. It prefers CLAUDE_PROJECT_DIR (set by
// Claude Code for hooks) and falls back to walking up from cwd to find .git or go.mod.
func RepoRoot(env map[string]string) string {
	if env != nil {
		if root, ok := env["CLAUDE_PROJECT_DIR"]; ok && strings.TrimSpace(root) != "" {
			return strings.TrimSpace(root)
		}
	} else {
		if root := strings.TrimSpace(os.Getenv("CLAUDE_PROJECT_DIR")); root != "" {
			return root
		}
	}
	return defaultRepoRoot()
}

func defaultRepoRoot() string {
	dir, err := os.Getwd()
	if err != nil {
		return "."
	}
	dir = filepath.Clean(dir)
	cur := dir
	for {
		if _, err := os.Stat(filepath.Join(cur, ".git")); err == nil {
			return cur
		}
		if _, err := os.Stat(filepath.Join(cur, "go.mod")); err == nil {
			return cur
		}
		parent := filepath.Dir(cur)
		if parent == cur {
			break
		}
		cur = parent
	}
	return dir
}

// NativeBinary returns the path to the provisioned native dos-hook binary under
// tools/.bin, or empty string when absent (falling back to the Python path).
func NativeBinary(root string) string {
	if root == "" {
		return ""
	}
	var names []string
	if runtime.GOOS == "windows" {
		names = []string{"dos-hook.exe", "dos-hook"}
	} else {
		names = []string{"dos-hook", "dos-hook.exe"}
	}
	for _, name := range names {
		cand := filepath.Join(root, "tools", ".bin", name)
		if fi, err := os.Stat(cand); err == nil && !fi.IsDir() {
			return cand
		}
	}
	return ""
}

func pythonExe() string {
	if p, err := exec.LookPath("python3"); err == nil {
		return p
	}
	if p, err := exec.LookPath("python"); err == nil {
		return p
	}
	return "python"
}

// DefaultRunner spawns argv feeding stdinData and captures stdout. Diagnostics
// flow to os.Stderr. Non-zero child exit codes return (exitCode, stdout, nil).
func DefaultRunner(argv []string, stdinData []byte) (int, []byte, error) {
	if len(argv) == 0 {
		return 1, nil, errors.New("empty argv")
	}
	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, argv[0], argv[1:]...)
	if len(stdinData) > 0 {
		cmd.Stdin = bytes.NewReader(stdinData)
	}
	var stdout bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = os.Stderr
	windowgate.ConfigureBackgroundCommand(cmd)

	err := cmd.Run()
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return exitErr.ExitCode(), stdout.Bytes(), nil
		}
		return 1, stdout.Bytes(), err
	}
	return 0, stdout.Bytes(), nil
}

// RunHook decides the call and returns stdout to forward.
// Native-first: if native binary present, run [native, verb, "--workspace", workspace]
// with stdinData. If exit code == 0, return stdout (OWNED).
// On non-zero / crash / absent native: delegate to python -m dos.cli hook <verb> --workspace <workspace>.
// Exit code of python fallback is discarded (always succeeds / exit 0 fail-safe).
func RunHook(verb, workspace string, stdinData []byte, nativeBin string, run Runner) ([]byte, error) {
	if run == nil {
		run = DefaultRunner
	}
	if nativeBin != "" {
		rc, out, err := run([]string{nativeBin, verb, "--workspace", workspace}, stdinData)
		if rc == 0 && err == nil {
			return out, nil
		}
	}
	_, out, err := run([]string{pythonExe(), "-m", "dos.cli", "hook", verb, "--workspace", workspace}, stdinData)
	return out, err
}

// Run reads the buffered hook payload, runs the contract, forwards the verdict, and exits 0.
// Always returns 0 -- fail-safe by construction.
func Run(argv []string, stdin io.Reader, stdout, stderr io.Writer, env map[string]string, run Runner) (exitCode int) {
	defer func() {
		if r := recover(); r != nil {
			exitCode = 0
		}
	}()

	verb := "pretool"
	workspace := "."
	for i := 0; i < len(argv); i++ {
		arg := argv[i]
		if arg == "--workspace" {
			if i+1 < len(argv) {
				workspace = argv[i+1]
				i++
			}
		} else if strings.HasPrefix(arg, "--workspace=") {
			workspace = strings.TrimPrefix(arg, "--workspace=")
		} else if !strings.HasPrefix(arg, "-") && verb == "pretool" {
			verb = arg
		}
	}

	var data []byte
	if stdin != nil {
		data, _ = io.ReadAll(stdin)
	}

	if run == nil {
		run = DefaultRunner
	}

	native := NativeBinary(RepoRoot(env))

	out, _ := RunHook(verb, workspace, data, native, run)
	if len(out) > 0 && stdout != nil {
		_, _ = stdout.Write(out)
	}
	return 0
}

// Main is an alias for Run for CLI entry points.
func Main(argv []string, stdin io.Reader, stdout, stderr io.Writer, env map[string]string, run Runner) int {
	return Run(argv, stdin, stdout, stderr, env, run)
}
