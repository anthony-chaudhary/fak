package main

// dev.go keeps the temporary `fak dev ...` compatibility spelling at a process
// boundary. Repository-development commands live in the separately buildable
// fak-dev artifact; runtime fak must never import or relink their implementation
// packages.

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
)

var findFakDev = resolveFakDev

// runDevHandoff executes fak-dev with argv and connects it to the runtime
// process streams. Returning the child's exit code preserves CLI behavior while
// keeping the runtime dependency graph free of development packages.
func runDevHandoff(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	path, err := findFakDev()
	if err != nil {
		fmt.Fprintln(stderr, "fak dev: repository-development commands are provided by the separate 'fak-dev' executable")
		fmt.Fprintln(stderr, "  install or build fak-dev, then run: fak-dev <command> [args...]")
		fmt.Fprintf(stderr, "  lookup: %v\n", err)
		return 2
	}
	cmd := exec.Command(path, argv...)
	cmd.Stdin = stdin
	cmd.Stdout = stdout
	cmd.Stderr = stderr
	if err := cmd.Run(); err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			return exitErr.ExitCode()
		}
		fmt.Fprintf(stderr, "fak dev: launch %s: %v\n", path, err)
		return 2
	}
	return 0
}

// resolveFakDev prefers a sibling artifact so side-by-side installs are
// deterministic, then falls back to PATH for split package installations.
func resolveFakDev() (string, error) {
	name := "fak-dev"
	if runtime.GOOS == "windows" {
		name += ".exe"
	}
	if exe, err := os.Executable(); err == nil {
		candidate := filepath.Join(filepath.Dir(exe), name)
		if info, statErr := os.Stat(candidate); statErr == nil && !info.IsDir() {
			return candidate, nil
		}
	}
	return exec.LookPath(name)
}
