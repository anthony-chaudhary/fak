// Package powershell resolves the PowerShell executable a Windows process
// probe runs. A bare "powershell" is a PATH lookup, and a caller whose PATH
// carries System32 but not System32\WindowsPowerShell\v1.0 cannot start it, so
// every CIM process census read the host as unobservable.
package powershell

import (
	"errors"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// ErrUnavailable is returned when no PowerShell executable resolves by
// absolute path or PATH lookup.
var ErrUnavailable = errors.New("powershell: no PowerShell executable resolvable")

type host struct {
	getenv   func(string) string
	isFile   func(string) bool
	lookPath func(string) (string, error)
}

var liveHost = host{getenv: os.Getenv, isFile: isRegularFile, lookPath: exec.LookPath}

// Path returns an absolute path to Windows PowerShell, falling back to
// PowerShell 7 (pwsh), or ErrUnavailable.
func Path() (string, error) { return liveHost.resolve() }

// Exe is Path for exec callers that surface their own start error: when no
// executable resolves it returns the bare name, so exec reports the miss.
func Exe() string {
	if p, err := Path(); err == nil {
		return p
	}
	return "powershell"
}

func (h host) resolve() (string, error) {
	for _, p := range h.candidates() {
		if h.isFile(p) {
			return p, nil
		}
	}
	for _, name := range []string{"powershell.exe", "pwsh.exe"} {
		if p, err := h.lookPath(name); err == nil && p != "" {
			return p, nil
		}
	}
	return "", ErrUnavailable
}

func (h host) candidates() []string {
	var out []string
	seen := map[string]bool{}
	add := func(p string) {
		key := strings.ToLower(p)
		if p != "" && !seen[key] {
			seen[key] = true
			out = append(out, p)
		}
	}
	for _, root := range []string{h.getenv("SystemRoot"), h.getenv("windir"), `C:\Windows`} {
		if strings.TrimSpace(root) != "" {
			add(joinWin(root, `System32\WindowsPowerShell\v1.0\powershell.exe`))
		}
	}
	for _, pf := range []string{h.getenv("ProgramFiles"), `C:\Program Files`} {
		if strings.TrimSpace(pf) != "" {
			add(joinWin(pf, `PowerShell\7\pwsh.exe`))
		}
	}
	return out
}

func joinWin(root, rel string) string {
	return strings.TrimRight(root, `\/`) + `\` + rel
}

func isRegularFile(p string) bool {
	info, err := os.Stat(filepath.FromSlash(p))
	return err == nil && info.Mode().IsRegular()
}
