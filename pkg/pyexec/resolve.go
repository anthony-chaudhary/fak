// Package pyexec resolves the Python interpreter a tool child runs. A bare
// "python" is a PATH lookup, and agent shells and land's publish step often
// run with Python installed but absent from PATH, so the leak audit refused
// every publish. Resolution tries PATH, then the known Windows install
// locations, and accepts a candidate only when `--version` exits 0, which
// rejects the Microsoft Store App Execution Alias stub.
package pyexec

import (
	"context"
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/anthony-chaudhary/fak/pkg/sysproc"
)

// ErrUnavailable is returned when no working Python 3 interpreter resolves.
var ErrUnavailable = errors.New("pyexec: no working python interpreter resolvable")

// Interpreter is a resolved Python: Path plus the arguments that must precede
// the script (the py launcher needs -3).
type Interpreter struct {
	Path string
	Args []string
}

// Argv returns the interpreter argv followed by args.
func (i Interpreter) Argv(args ...string) []string {
	out := append([]string{i.Path}, i.Args...)
	return append(out, args...)
}

const probeTimeout = 10 * time.Second

type host struct {
	goos     string
	getenv   func(string) string
	glob     func(string) ([]string, error)
	isFile   func(string) bool
	lookPath func(string) (string, error)
	probe    func(Interpreter) bool
}

var liveHost = host{
	goos:     runtime.GOOS,
	getenv:   os.Getenv,
	glob:     filepath.Glob,
	isFile:   isRegularFile,
	lookPath: exec.LookPath,
	probe:    probeVersion,
}

// Resolve returns a working Python 3 interpreter or ErrUnavailable.
func Resolve() (Interpreter, error) { return liveHost.resolve() }

// Command rewrites an argv whose first element is a bare python, python3 or
// py into the resolved interpreter. Any other argv, or a miss, is returned
// unchanged so exec reports the start error itself.
func Command(argv []string) []string {
	if len(argv) == 0 || !isBareName(argv[0]) {
		return argv
	}
	interp, err := Resolve()
	if err != nil {
		return argv
	}
	return interp.Argv(argv[1:]...)
}

func isBareName(name string) bool {
	switch strings.ToLower(name) {
	case "python", "python3", "py", "python.exe", "python3.exe", "py.exe":
		return true
	}
	return false
}

func (h host) resolve() (Interpreter, error) {
	for _, c := range h.candidates() {
		if isStoreStub(c.Path) {
			continue
		}
		if h.probe(c) {
			return c, nil
		}
	}
	return Interpreter{}, ErrUnavailable
}

func (h host) candidates() []Interpreter {
	var out []Interpreter
	seen := map[string]bool{}
	add := func(path string, args ...string) {
		key := strings.ToLower(path) + "\x00" + strings.Join(args, " ")
		if path != "" && !seen[key] {
			seen[key] = true
			out = append(out, Interpreter{Path: path, Args: args})
		}
	}
	for _, name := range []string{"python", "python3"} {
		if p, err := h.lookPath(name); err == nil {
			add(p)
		}
	}
	if p, err := h.lookPath("py"); err == nil {
		add(p, "-3")
	}
	if h.goos != "windows" {
		return out
	}
	var roots []string
	if la := h.getenv("LOCALAPPDATA"); strings.TrimSpace(la) != "" {
		roots = append(roots, joinWin(la, `Programs\Python`))
	}
	for _, key := range []string{"ProgramFiles", "ProgramW6432", "ProgramFiles(x86)"} {
		if pf := h.getenv(key); strings.TrimSpace(pf) != "" {
			roots = append(roots, pf)
		}
	}
	for _, root := range roots {
		matches, _ := h.glob(joinWin(root, `Python3*\python.exe`))
		for _, m := range newestFirst(matches) {
			if h.isFile(m) {
				add(m)
			}
		}
	}
	if up := h.getenv("USERPROFILE"); strings.TrimSpace(up) != "" {
		for _, dist := range []string{"miniconda3", "anaconda3"} {
			if p := joinWin(up, dist+`\python.exe`); h.isFile(p) {
				add(p)
			}
		}
	}
	var launchers []string
	if la := h.getenv("LOCALAPPDATA"); strings.TrimSpace(la) != "" {
		launchers = append(launchers, joinWin(la, `Programs\Python\Launcher\py.exe`))
	}
	for _, key := range []string{"SystemRoot", "windir"} {
		if root := h.getenv(key); strings.TrimSpace(root) != "" {
			launchers = append(launchers, joinWin(root, `py.exe`))
		}
	}
	for _, p := range launchers {
		if h.isFile(p) {
			add(p, "-3")
		}
	}
	return out
}

// isStoreStub reports the App Execution Alias under WindowsApps, which opens
// the Store (exit 9009) instead of running Python.
func isStoreStub(path string) bool {
	// Normalize separators by hand: filepath.ToSlash is a no-op for a
	// Windows-style path when the process itself runs on a POSIX host (linux
	// CI, WSL), so the WindowsApps marker check must not depend on the host
	// separator.
	p := strings.ToLower(strings.ReplaceAll(path, "\\", "/"))
	return strings.Contains(p, "/microsoft/windowsapps/")
}

// newestFirst orders Python3NN install directories by minor version, highest
// first, so Python312 wins over Python39.
func newestFirst(paths []string) []string {
	out := append([]string(nil), paths...)
	minor := func(p string) int {
		dir := filepath.Base(filepath.Dir(filepath.FromSlash(strings.ReplaceAll(p, `\`, "/"))))
		rest := strings.TrimPrefix(strings.ToLower(dir), "python3")
		end := 0
		for end < len(rest) && rest[end] >= '0' && rest[end] <= '9' {
			end++
		}
		n, err := strconv.Atoi(rest[:end])
		if err != nil {
			return -1
		}
		return n
	}
	sort.SliceStable(out, func(i, j int) bool { return minor(out[i]) > minor(out[j]) })
	return out
}

func probeVersion(i Interpreter) bool {
	ctx, cancel := context.WithTimeout(context.Background(), probeTimeout)
	defer cancel()
	cmd := sysproc.CommandContext(ctx, i.Path, append(append([]string(nil), i.Args...), "--version")...)
	out, err := cmd.CombinedOutput()
	return err == nil && strings.HasPrefix(strings.TrimSpace(string(out)), "Python 3")
}

func joinWin(root, rel string) string {
	return strings.TrimRight(root, `\/`) + `\` + rel
}

func isRegularFile(p string) bool {
	info, err := os.Stat(filepath.FromSlash(p))
	return err == nil && info.Mode().IsRegular()
}
