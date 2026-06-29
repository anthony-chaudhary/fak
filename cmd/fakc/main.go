// Command fakc is the one-word Codex launcher for fak.
//
// It is intentionally a thin wrapper over `fak codex`: the fak binary owns the in-process
// guard gateway, Codex Responses-provider injection, audit journal, and 80/20 fak-info pane.
// Install it beside fak (`go install ./cmd/fak ./cmd/fakc`) and run `fakc ...` instead of
// `fak codex ...`.
package main

import (
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
	fakcRun     = execFakc
	fakcResolve = resolveFakBinary
)

func main() {
	os.Exit(runFakc(os.Stdout, os.Stderr, os.Args[1:]))
}

func runFakc(stdout, stderr io.Writer, args []string) int {
	fakBin, err := fakcResolve(os.Getenv, os.Executable, exec.LookPath, os.Getwd, runtime.GOOS)
	if err != nil && !fakcDryRun(args) {
		fmt.Fprintf(stderr, "fakc: %v\n", err)
		return 1
	}
	if fakBin == "" {
		fakBin = "fak"
	}
	argv := fakcArgv(fakBin, args)
	if fakcDryRun(args) {
		fmt.Fprintf(stderr, "fakc: delegating to `fak codex` (%s)\n", fakBin)
		fmt.Fprintln(stdout, strings.Join(argv, " "))
		return 0
	}
	return fakcRun(stdout, stderr, argv, os.Environ())
}

func fakcArgv(fakBin string, args []string) []string {
	argv := []string{fakBin, "codex"}
	return append(argv, args...)
}

func fakcDryRun(args []string) bool {
	for _, arg := range args {
		if arg == "--dry-run" || arg == "-dry-run" {
			return true
		}
		if arg == "--" {
			return false
		}
	}
	return false
}

func resolveFakBinary(getenv func(string) string, executable func() (string, error), lookPath func(string) (string, error), getwd func() (string, error), goos string) (string, error) {
	if v := strings.TrimSpace(getenv("FAK_BIN")); v != "" {
		return v, nil
	}
	names := []string{"fak"}
	if goos == "windows" {
		names = []string{"fak.exe", "fak.cmd", "fak.bat", "fak"}
	}
	if exe, err := executable(); err == nil {
		dir := filepath.Dir(exe)
		for _, name := range names {
			if p := filepath.Join(dir, name); fileExists(p) {
				return p, nil
			}
		}
	}
	if wd, err := getwd(); err == nil {
		for _, name := range names {
			if p := filepath.Join(wd, name); fileExists(p) {
				return p, nil
			}
		}
	}
	for _, name := range names {
		if p, err := lookPath(name); err == nil && strings.TrimSpace(p) != "" {
			return p, nil
		}
	}
	return "", fmt.Errorf("could not find fak binary; install it beside fakc, put fak on PATH, or set FAK_BIN")
}

func fileExists(path string) bool {
	st, err := os.Stat(path)
	return err == nil && !st.IsDir()
}

func execFakc(stdout, stderr io.Writer, argv, env []string) int {
	if len(argv) == 0 {
		fmt.Fprintln(stderr, "fakc: empty command")
		return 2
	}
	cmd := exec.Command(argv[0], argv[1:]...)
	cmd.Env = env
	cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, stdout, stderr
	if err := cmd.Run(); err != nil {
		var ee *exec.ExitError
		if errors.As(err, &ee) {
			return ee.ExitCode()
		}
		fmt.Fprintf(stderr, "fakc: %v\n", err)
		return 1
	}
	return 0
}
