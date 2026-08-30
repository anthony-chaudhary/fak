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

var (
	findFakDev            = resolveFakDev
	findFakSourceCheckout = resolveFakSourceCheckout
	findGoTool            = exec.LookPath
)

// runBuildHandoff preserves the exact top-level `fak build` spelling. It prefers
// the separately installed fak-dev artifact, but a source checkout can bootstrap
// that same process boundary through the local Go toolchain.
func runBuildHandoff(stdin io.Reader, stdout, stderr io.Writer, argv []string) int {
	if path, err := findFakDev(); err == nil {
		return runDevChild(stdin, stdout, stderr, "", path, append([]string{"build"}, argv...))
	}
	root, err := findFakSourceCheckout()
	if err != nil {
		fmt.Fprintf(stderr, "fak build: fak-dev is not installed and no fak source checkout was found: %v\n", err)
		fmt.Fprintln(stderr, "  run from a fak source checkout with Go available, or install fak-dev")
		return 2
	}
	goTool, err := findGoTool("go")
	if err != nil {
		fmt.Fprintf(stderr, "fak build: checkout fallback requires the Go toolchain: %v\n", err)
		return 2
	}
	args := []string{"run", "./cmd/fak-dev", "build"}
	args = append(args, argv...)
	return runDevChild(stdin, stdout, stderr, root, goTool, args)
}

func runDevChild(stdin io.Reader, stdout, stderr io.Writer, dir, path string, argv []string) int {
	cmd := exec.Command(path, argv...)
	cmd.Dir = dir
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

func resolveFakSourceCheckout() (string, error) {
	dir, err := os.Getwd()
	if err != nil {
		return "", err
	}
	for {
		if info, statErr := os.Stat(filepath.Join(dir, "go.mod")); statErr == nil && !info.IsDir() {
			if info, statErr = os.Stat(filepath.Join(dir, "cmd", "fak-dev")); statErr == nil && info.IsDir() {
				return dir, nil
			}
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", fmt.Errorf("walked from %s to filesystem root", dir)
		}
		dir = parent
	}
}

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
	return runDevChild(stdin, stdout, stderr, "", path, argv)
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
