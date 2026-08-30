package main

// dev.go keeps the temporary `fak dev ...` compatibility spelling at a process
// boundary. Repository-development commands live in the separately buildable
// fak-dev artifact; runtime fak must never import or relink their implementation
// packages.

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
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
	if len(argv) > 0 && argv[0] == "availability" {
		return runDevAvailability(stdout, stderr, argv[1:])
	}
	path, err := findFakDev()
	if err != nil {
		fmt.Fprintln(stderr, "fak dev: repository-development commands are provided by the separate 'fak-dev' executable")
		fmt.Fprintln(stderr, "  source checkout: go install ./cmd/fak-dev")
		fmt.Fprintln(stderr, "  released version: go install github.com/anthony-chaudhary/fak/cmd/fak-dev@latest")
		fmt.Fprintln(stderr, "  then run: fak-dev <command> [args...]")
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

type devAvailability struct {
	Schema    string `json:"schema"`
	Available bool   `json:"available"`
	Source    string `json:"source"`
	Path      string `json:"path,omitempty"`
	Recovery  string `json:"recovery,omitempty"`
}

// runDevAvailability reports the companion contract without requiring fak-dev
// to exist. Keeping this probe in runtime fak makes a missing companion
// inspectable rather than turning discovery into another failed handoff.
func runDevAvailability(stdout, stderr io.Writer, argv []string) int {
	jsonOutput := len(argv) == 1 && argv[0] == "--json"
	if len(argv) > 0 && !jsonOutput {
		fmt.Fprintln(stderr, "usage: fak dev availability [--json]")
		return 2
	}

	path, err := findFakDev()
	result := devAvailability{Schema: "fak-dev-availability/1"}
	if err != nil {
		result.Source = "missing"
		result.Recovery = "go install ./cmd/fak-dev"
	} else {
		result.Available = true
		result.Path = path
		result.Source = fakDevSource(path)
	}
	if jsonOutput {
		if err := json.NewEncoder(stdout).Encode(result); err != nil {
			fmt.Fprintf(stderr, "fak dev availability: encode: %v\n", err)
			return 2
		}
		return 0
	}
	if result.Available {
		fmt.Fprintf(stdout, "fak-dev: available (%s) %s\n", result.Source, result.Path)
		return 0
	}
	fmt.Fprintf(stdout, "fak-dev: missing; recover with: %s\n", result.Recovery)
	return 1
}

func fakDevSource(path string) string {
	if exe, err := os.Executable(); err == nil {
		name := "fak-dev"
		if runtime.GOOS == "windows" {
			name += ".exe"
		}
		sibling := filepath.Join(filepath.Dir(exe), name)
		if sameDevPath(path, sibling) {
			return "sibling"
		}
	}
	return "path"
}

func sameDevPath(a, b string) bool {
	a, errA := filepath.Abs(a)
	b, errB := filepath.Abs(b)
	if errA != nil || errB != nil {
		return false
	}
	if runtime.GOOS == "windows" {
		return strings.EqualFold(filepath.Clean(a), filepath.Clean(b))
	}
	return filepath.Clean(a) == filepath.Clean(b)
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
