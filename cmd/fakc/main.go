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
	"io/fs"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

var (
	fakcRun     = execFakc
	fakcResolve = resolveFakCommand
)

type fakcCommand struct {
	Argv    []string
	Display string
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
		fakCmd = fakcCommand{Argv: []string{"fak"}, Display: "fak"}
	}
	argv := fakcArgvPrefix(fakCmd.Argv, args)
	if fakcDryRun(args) {
		fmt.Fprintf(stderr, "fakc: delegating to `fak codex` (%s)\n", fakCmd.Display)
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
	if v := strings.TrimSpace(getenv("FAK_BIN")); v != "" {
		if dev, ok := resolveDevFakCommand(v, executable, lookPath, getwd); ok {
			return dev, nil
		}
		return fakcCommand{Argv: []string{v}, Display: v}, nil
	}
	fakBin, err := resolveFakBinary(func(string) string { return "" }, executable, lookPath, getwd, goos)
	if err != nil {
		return fakcCommand{}, err
	}
	if dev, ok := resolveDevFakCommand(fakBin, executable, lookPath, getwd); ok {
		return dev, nil
	}
	return fakcCommand{Argv: []string{fakBin}, Display: fakBin}, nil
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

func resolveDevFakCommand(fakBin string, executable func() (string, error), lookPath func(string) (string, error), getwd func() (string, error)) (fakcCommand, bool) {
	st, err := os.Stat(fakBin)
	if err != nil || st.IsDir() {
		return fakcCommand{}, false
	}
	root, ok := fakcSourceRoot(executable, getwd)
	if !ok {
		return fakcCommand{}, false
	}
	newest, ok := newestGoSourceModTime(root)
	if !ok || !newest.After(st.ModTime()) {
		return fakcCommand{}, false
	}
	goBin, err := lookPath("go")
	if err != nil || strings.TrimSpace(goBin) == "" {
		return fakcCommand{}, false
	}
	return fakcCommand{Argv: []string{goBin, "run", filepath.Join(root, "cmd", "fak")}, Display: "go run ./cmd/fak"}, true
}

func fakcSourceRoot(executable func() (string, error), getwd func() (string, error)) (string, bool) {
	var starts []string
	if wd, err := getwd(); err == nil {
		starts = append(starts, wd)
	}
	if exe, err := executable(); err == nil {
		starts = append(starts, filepath.Dir(exe))
	}
	for _, start := range starts {
		if root, ok := findSourceRoot(start); ok {
			return root, true
		}
	}
	return "", false
}

func findSourceRoot(start string) (string, bool) {
	dir, err := filepath.Abs(start)
	if err != nil {
		return "", false
	}
	for {
		if fileExists(filepath.Join(dir, "go.mod")) && fileExists(filepath.Join(dir, "cmd", "fak", "main.go")) {
			return dir, true
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "", false
		}
		dir = parent
	}
}

func newestGoSourceModTime(root string) (time.Time, bool) {
	var newest time.Time
	err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}
		if d.IsDir() {
			switch d.Name() {
			case ".git", ".dispatch-runs", "gen":
				return filepath.SkipDir
			}
			return nil
		}
		if filepath.Ext(path) != ".go" && filepath.Base(path) != "go.mod" {
			return nil
		}
		info, err := d.Info()
		if err != nil {
			return nil
		}
		if info.ModTime().After(newest) {
			newest = info.ModTime()
		}
		return nil
	})
	if err != nil || newest.IsZero() {
		return time.Time{}, false
	}
	return newest, true
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
