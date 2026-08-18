// Command fakc is the one-word Codex launcher for fak.
//
// It is intentionally a thin wrapper over `fak codex`: the fak binary owns the in-process
// guard gateway, Codex Responses-provider injection, audit journal, and 80/20 fak-info pane.
// Install it beside fak (`go install ./cmd/fak ./cmd/fakc`) and run `fakc ...` instead of
// `fak codex ...`.
package main

import (
	"fmt"
	"github.com/anthony-chaudhary/fak/internal/childprocess"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
)

var (
	fakcRun     = execFakc
	fakcResolve = resolveFakCommand
)

type fakcCommand struct {
	Argv    []string
	Display string
	Source  string
}

func main() {
	os.Exit(runFakc(os.Stdout, os.Stderr, os.Args[1:]))
}

func runFakc(stdout, stderr io.Writer, args []string) int {
	fakCmd, err := fakcResolve(os.Getenv, os.Executable, exec.LookPath, os.Getwd, runtime.GOOS)
	if err != nil && !fakcDryRun(args) {
		fmt.Fprintf(stderr, "fakc: %v\n", err)
		return 1
	}
	if len(fakCmd.Argv) == 0 {
		fakCmd = fakcCommand{Argv: []string{"fak"}, Display: "fak", Source: "fallback"}
	}
	argv := fakcArgvPrefix(fakCmd.Argv, args)
	if fakCmd.Source == "FAK_BIN" {
		fmt.Fprintf(stderr, "fakc: using explicit FAK_BIN override (%s); unset FAK_BIN to use the installed fak beside fakc or on PATH\n", fakCmd.Display)
	}
	if fakcDryRun(args) {
		fmt.Fprintf(stderr, "fakc: delegating to `fak codex` (%s; source=%s)\n", fakCmd.Display, fakCmd.Source)
		fmt.Fprintln(stdout, strings.Join(argv, " "))
		return 0
	}
	return fakcRun(stdout, stderr, argv, os.Environ())
}

func fakcArgv(fakBin string, args []string) []string {
	return fakcArgvPrefix([]string{fakBin}, args)
}

func fakcArgvPrefix(prefix []string, args []string) []string {
	argv := append([]string{}, prefix...)
	if len(argv) == 0 {
		argv = append(argv, "fak")
	}
	argv = append(argv, "codex")
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

func resolveFakCommand(getenv func(string) string, executable func() (string, error), lookPath func(string) (string, error), getwd func() (string, error), goos string) (fakcCommand, error) {
	fakBin, err := resolveFakBinary(getenv, executable, lookPath, getwd, goos)
	if err != nil {
		return fakcCommand{}, err
	}
	source := "PATH"
	if explicit := strings.TrimSpace(getenv("FAK_BIN")); explicit != "" && filepath.Clean(explicit) == filepath.Clean(fakBin) {
		source = "FAK_BIN"
	} else if exe, exeErr := executable(); exeErr == nil && filepath.Dir(fakBin) == filepath.Dir(exe) {
		source = "sibling"
	} else if wd, wdErr := getwd(); wdErr == nil && filepath.Dir(fakBin) == filepath.Clean(wd) {
		source = "cwd"
	}
	return fakcCommand{Argv: []string{fakBin}, Display: fakBin, Source: source}, nil
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
		code := childprocess.ExitCode(err, 1)
		if code == 1 {
			fmt.Fprintf(stderr, "fakc: %v\n", err)
		}
		return code
	}
	return 0
}
