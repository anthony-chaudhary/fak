package windowgate

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// Reason codes: the closed-vocabulary forms the gate refuses with.
const (
	ReasonInteractiveTask    = "INTERACTIVE_TASK_POPUP" // a .ps1 installs an on-desktop task
	ReasonUnsuppressedSpawn  = "UNSUPPRESSED_SPAWN"     // a suppressing module forgot a flag
	ReasonGoUnsuppressedExec = "UNSUPPRESSED_GO_EXEC"   // a Go dispatch helper forgot the no-window hook
)

// ---- PowerShell scheduled-task installer rules ---------------------------- //

var (
	reCreatesTask    = regexp.MustCompile(`(?i)Register-ScheduledTask\b|schtasks(\.exe)?\b[^\n]*/Create\b`)
	reSchtasksCreate = regexp.MustCompile(`(?i)schtasks(\.exe)?\b[^\n]*/Create\b`)
	reOffDesktop     = regexp.MustCompile(`(?i)-LogonType\s+(S4U|Password|ServiceAccount)\b|-UserId\s+['"]?(SYSTEM|LOCAL\s+SERVICE|NETWORK\s+SERVICE|NT\s+AUTHORITY)`)
	reHeadless       = regexp.MustCompile(`(?i)--headless`)
	reInteractive    = regexp.MustCompile(`(?i)-LogonType\s+Interactive\b|\s/IT\b`)
	reITflag         = regexp.MustCompile(`(?i)\s/IT\b`)
	reSetsPrincipal  = regexp.MustCompile(`(?i)New-ScheduledTaskPrincipal\b|-Principal\b|/RU\b`)
)

// PSInstallerViolation returns a one-line violation for a task-creating .ps1 that
// is neither off-desktop nor headless, and ok=false when the file is clean (or is
// not a task installer at all). Pure: no git, no disk.
func PSInstallerViolation(rel, src string) (string, bool) {
	if !reCreatesTask.MatchString(src) {
		return "", false
	}
	if reHeadless.MatchString(src) {
		return "", false // a fully headless launcher is safe under any principal
	}
	offDesktop := reOffDesktop.MatchString(src)
	// schtasks /Create without /IT runs in session 0 (off-desktop) by default.
	if reSchtasksCreate.MatchString(src) && !reITflag.MatchString(src) {
		offDesktop = true
	}
	interactive := reInteractive.MatchString(src)
	// A Register-ScheduledTask call that never sets a principal inherits the
	// Interactive default — treat as interactive unless it goes headless.
	if strings.Contains(src, "Register-ScheduledTask") && !reSetsPrincipal.MatchString(src) {
		interactive = true
	}
	if interactive {
		return fmt.Sprintf("%s: installs an INTERACTIVE scheduled task without a headless "+
			"(conhost --headless) launcher — it will flash a console window on the desktop; "+
			"use -LogonType S4U (session 0) or wrap the action in conhost.exe --headless (%s)",
			rel, ReasonInteractiveTask), true
	}
	if !offDesktop {
		return fmt.Sprintf("%s: installs a scheduled task with no off-desktop principal "+
			"(-LogonType S4U / a service account / schtasks without /IT) and no conhost "+
			"--headless launcher — it may flash a console window (%s)",
			rel, ReasonInteractiveTask), true
	}
	return "", false
}

// ---- Python window-suppression completeness rules ------------------------ //

var (
	reOptIn     = regexp.MustCompile(`\bno_window_creationflags\b`)
	reSpawnCall = regexp.MustCompile(`subprocess\.(run|Popen|call|check_call|check_output)\s*\(`)
	rePosixHead = regexp.MustCompile(`^\s*[\[(]\s*["']([a-z]+)["']`)
	reFlagWord  = regexp.MustCompile(`\bcreationflags\b`)
)

// posixOnly tools can only run on a non-Windows host, so a spawn of one can never
// pop a Windows console — exempt it (the Windows arm uses tasklist/taskkill).
var posixOnly = map[string]bool{"pgrep": true, "pkill": true, "ps": true, "killall": true}

// PySpawnViolations returns one message per subprocess spawn that lacks a
// creationflags hint, but only for a module that opts into the suppressor. Pure.
func PySpawnViolations(rel, src string) []string {
	if !reOptIn.MatchString(src) {
		return nil
	}
	var out []string
	for _, m := range reSpawnCall.FindAllStringIndex(src, -1) {
		open := m[1] - 1 // index of the '(' that reSpawnCall ends on
		args, ok := callArgs(src, open)
		if !ok {
			continue // unbalanced — skip rather than false-positive
		}
		if strings.Contains(args, "**") { // a **kwargs splat may carry the flag
			continue
		}
		if reFlagWord.MatchString(args) {
			continue
		}
		if g := rePosixHead.FindStringSubmatch(args); g != nil && posixOnly[g[1]] {
			continue
		}
		out = append(out, fmt.Sprintf("%s:%d: subprocess spawn without creationflags in a "+
			"window-suppressing module — pass creationflags=no_window_creationflags() so the "+
			"console child stays windowless on Windows (%s)",
			rel, lineOf(src, m[0]), ReasonUnsuppressedSpawn))
	}
	return out
}

// ---- Go dispatch helper window-suppression rules ------------------------ //

var (
	reGoExecAssign = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*exec\.Command(?:Context)?\s*\(`)
)

// GoExecViolations returns one message per cmd/fak dispatch helper command that
// reaches Run/Output/CombinedOutput/Start before the Windows no-window hook is
// applied. The long-lived worker spawn may use configureDispatchSpawn; short
// helper probes should use configureDispatchHelperCommand.
func GoExecViolations(rel, src string) []string {
	if !strings.HasPrefix(rel, "cmd/fak/dispatch") || strings.HasSuffix(rel, "_test.go") {
		return nil
	}
	lines := strings.Split(src, "\n")
	var out []string
	for i, line := range lines {
		m := reGoExecAssign.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		name := m[1]
		configured := false
		for j := i + 1; j < len(lines) && j <= i+16; j++ {
			text := stripGoLineComment(lines[j])
			if strings.Contains(text, "configureDispatchHelperCommand("+name+")") ||
				strings.Contains(text, "configureDispatchSpawn("+name+")") {
				configured = true
				continue
			}
			if reGoExecAssign.MatchString(text) {
				break
			}
			if goCommandTerminal(name, text) {
				if !configured {
					out = append(out, fmt.Sprintf("%s:%d: exec.Command reaches %s before "+
						"configureDispatchHelperCommand(%s) / configureDispatchSpawn(%s) — a "+
						"windowless Windows parent can flash a console child (%s)",
						rel, i+1, strings.TrimSpace(text), name, name, ReasonGoUnsuppressedExec))
				}
				break
			}
		}
	}
	return out
}

func stripGoLineComment(line string) string {
	if i := strings.Index(line, "//"); i >= 0 {
		return line[:i]
	}
	return line
}

func goCommandTerminal(name, line string) bool {
	for _, method := range []string{".Run(", ".Output(", ".CombinedOutput(", ".Start("} {
		if strings.Contains(line, name+method) {
			return true
		}
	}
	return false
}

// callArgs returns the text between the call's opening paren at openIdx and its
// matching close paren, skipping Python string literals and # comments so that a
// paren inside a string or a nested .join(...) does not confuse the balance.
func callArgs(src string, openIdx int) (string, bool) {
	depth := 0
	i := openIdx
	n := len(src)
	for i < n {
		c := src[i]
		switch {
		case c == '#':
			// comment to end of line
			for i < n && src[i] != '\n' {
				i++
			}
		case c == '\'' || c == '"':
			q, triple := c, false
			if i+2 < n && src[i+1] == c && src[i+2] == c {
				triple = true
				i += 3
			} else {
				i++
			}
			i = skipString(src, i, q, triple)
			continue
		case c == '(' || c == '[' || c == '{':
			depth++
		case c == ')' || c == ']' || c == '}':
			depth--
			if depth == 0 {
				return src[openIdx+1 : i], true
			}
		}
		i++
	}
	return "", false
}

// skipString advances past a Python string body, returning the index just after
// the closing quote(s). Honors backslash escapes.
func skipString(src string, i int, q byte, triple bool) int {
	n := len(src)
	for i < n {
		c := src[i]
		if c == '\\' {
			i += 2
			continue
		}
		if c == q {
			if triple {
				if i+2 < n && src[i+1] == q && src[i+2] == q {
					return i + 3
				}
				i++
				continue
			}
			return i + 1
		}
		i++
	}
	return n
}

// lineOf returns the 1-based line number of byte offset off.
func lineOf(src string, off int) int {
	if off > len(src) {
		off = len(src)
	}
	return strings.Count(src[:off], "\n") + 1
}

// ---- Tree audit ---------------------------------------------------------- //

// Report holds the two violation lists from a tree scan.
type Report struct {
	PSInstallers []string // task installers that flash
	PySpawns     []string // suppressing modules with an unflagged spawn
	GoExecs      []string // Go dispatch helpers with unflagged child commands
}

// OK reports whether the tree is clean.
func (r Report) OK() bool {
	return len(r.PSInstallers) == 0 && len(r.PySpawns) == 0 && len(r.GoExecs) == 0
}

// ScanTree audits the git-tracked .ps1 and .py files under repoRoot.
func ScanTree(repoRoot string) (Report, error) {
	var rep Report
	ps1, err := tracked(repoRoot, "*.ps1")
	if err != nil {
		return rep, err
	}
	for _, rel := range ps1 {
		src, err := readRel(repoRoot, rel)
		if err != nil {
			continue
		}
		if v, bad := PSInstallerViolation(rel, src); bad {
			rep.PSInstallers = append(rep.PSInstallers, v)
		}
	}
	py, err := tracked(repoRoot, "*.py")
	if err != nil {
		return rep, err
	}
	for _, rel := range py {
		src, err := readRel(repoRoot, rel)
		if err != nil {
			continue
		}
		rep.PySpawns = append(rep.PySpawns, PySpawnViolations(rel, src)...)
	}
	goFiles, err := tracked(repoRoot, "cmd/fak/dispatch*.go")
	if err != nil {
		return rep, err
	}
	for _, rel := range goFiles {
		src, err := readRel(repoRoot, rel)
		if err != nil {
			continue
		}
		rep.GoExecs = append(rep.GoExecs, GoExecViolations(rel, src)...)
	}
	sort.Strings(rep.PSInstallers)
	sort.Strings(rep.PySpawns)
	sort.Strings(rep.GoExecs)
	return rep, nil
}

func tracked(repoRoot, glob string) ([]string, error) {
	cmd := exec.Command("git", "ls-files", glob)
	cmd.Dir = repoRoot
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git ls-files %s in %s: %w", glob, repoRoot, err)
	}
	var paths []string
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		if line = strings.TrimSpace(line); line != "" {
			paths = append(paths, strings.ReplaceAll(line, "\\", "/"))
		}
	}
	return paths, nil
}

func readRel(repoRoot, rel string) (string, error) {
	b, err := os.ReadFile(filepath.Join(repoRoot, filepath.FromSlash(rel)))
	if err != nil {
		return "", err
	}
	return string(b), nil
}
