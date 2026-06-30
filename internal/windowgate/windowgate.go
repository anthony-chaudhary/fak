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
	ReasonGoUnsuppressedExec = "UNSUPPRESSED_GO_EXEC"   // a Go background helper forgot the no-window hook
)

// ---- PowerShell scheduled-task installer rules ---------------------------- //

var (
	reCreatesTask    = regexp.MustCompile(`(?i)Register-ScheduledTask\b|schtasks(\.exe)?\b[^\n]*/Create\b`)
	reSchtasksCreate = regexp.MustCompile(`(?i)schtasks(\.exe)?\b[^\n]*/Create\b`)
	reOffDesktop     = regexp.MustCompile(`(?i)-LogonType\s+(S4U|Password|ServiceAccount)\b|-UserId\s+['"]?(SYSTEM|LOCAL\s+SERVICE|NETWORK\s+SERVICE|NT\s+AUTHORITY)|/RU\s+['"]?(SYSTEM|LOCAL\s+SERVICE|NETWORK\s+SERVICE|NT\s+AUTHORITY)`)
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
			"(-LogonType S4U / a service account) and no conhost "+
			"--headless launcher — it may flash a console window (%s)",
			rel, ReasonInteractiveTask), true
	}
	return "", false
}

// ---- Python window-suppression completeness rules ------------------------ //

var (
	reOptIn            = regexp.MustCompile(`\bno_window_creationflags\b`)
	reDefaultInstaller = regexp.MustCompile(`\binstall_no_window_subprocess_defaults\s*\(\s*subprocess\s*\)`)
	reSpawnCall        = regexp.MustCompile(`subprocess\.(run|Popen|call|check_call|check_output)\s*\(`)
	rePosixHead        = regexp.MustCompile(`^\s*[\[(]\s*["']([a-z]+)["']`)
	rePyHead           = regexp.MustCompile(`^\s*(?:[\[(]\s*)?["']([^"']+)["']`)
	reFlagWord         = regexp.MustCompile(`\bcreationflags\b`)
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
		if !pythonCodeOffset(src, m[0]) {
			continue
		}
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

// PySpawnCandidates returns advisory rows for Python modules that have not opted
// into the no-window suppressor but spawn a known console-tool child. These are not
// hard failures yet because many grandfathered scripts are foreground/manual tools,
// but they belong in the strict watchlist for the desktop-popup program.
func PySpawnCandidates(rel, src string) []string {
	if reOptIn.MatchString(src) {
		return nil
	}
	if PyDefaultSuppressorInstalled(src) {
		return nil
	}
	if pythonCandidateIgnoredPath(rel) {
		return nil
	}
	var out []string
	for _, m := range reSpawnCall.FindAllStringIndex(src, -1) {
		if !pythonCodeOffset(src, m[0]) {
			continue
		}
		open := m[1] - 1
		args, ok := callArgs(src, open)
		if !ok {
			continue
		}
		if reFlagWord.MatchString(args) {
			continue
		}
		tool := pythonSpawnTool(args)
		if tool == "" || posixOnly[tool] || !candidateConsoleTools[tool] {
			continue
		}
		out = append(out, fmt.Sprintf("%s:%d: subprocess %s launch is on the desktop-popup watchlist — "+
			"if this can run from background automation, pass creationflags=no_window_creationflags() (%s)",
			rel, lineOf(src, m[0]), tool, ReasonUnsuppressedSpawn))
	}
	return out
}

// PyDefaultSuppressorInstalled reports whether a Python module installs the
// shared subprocess defaults that apply CREATE_NO_WINDOW to helper launches.
func PyDefaultSuppressorInstalled(src string) bool {
	return reDefaultInstaller.MatchString(src)
}

// PyExplicitSuppressorUsed reports whether a Python module references the
// explicit no_window_creationflags helper. The scanner uses this as the hard
// completeness ratchet for modules that have already opted into call-site flags.
func PyExplicitSuppressorUsed(src string) bool {
	return reOptIn.MatchString(src)
}

func pythonSpawnTool(args string) string {
	m := rePyHead.FindStringSubmatch(strings.TrimSpace(args))
	if m == nil {
		return ""
	}
	head := strings.Fields(m[1])
	if len(head) == 0 {
		return ""
	}
	return strings.ToLower(filepath.Base(strings.ReplaceAll(head[0], "\\", "/")))
}

func pythonCodeOffset(src string, off int) bool {
	if off > len(src) {
		off = len(src)
	}
	for i := 0; i < off; {
		c := src[i]
		switch c {
		case '#':
			start := i
			for i < len(src) && src[i] != '\n' {
				i++
			}
			if off >= start && off < i {
				return false
			}
		case '\'', '"':
			q, triple := c, false
			start := i
			if i+2 < len(src) && src[i+1] == c && src[i+2] == c {
				triple = true
				i += 3
			} else {
				i++
			}
			end := skipString(src, i, q, triple)
			if off >= start && off < end {
				return false
			}
			i = end
		default:
			i++
		}
	}
	if off > 0 {
		lineStart := strings.LastIndex(src[:off], "\n") + 1
		if hash := strings.Index(src[lineStart:off], "#"); hash >= 0 {
			return false
		}
	}
	return true
}

func pythonCandidateIgnoredPath(rel string) bool {
	base := filepath.Base(strings.ReplaceAll(rel, "\\", "/"))
	return strings.HasSuffix(base, "_test.py") ||
		strings.HasPrefix(rel, "examples/") ||
		strings.HasPrefix(rel, "testdata/")
}

// ---- Go background helper window-suppression rules ---------------------- //

var (
	reGoExecAssign = regexp.MustCompile(`\b([A-Za-z_][A-Za-z0-9_]*)\s*:?=\s*exec\.Command(?:Context)?\s*\(`)
	reGoCommandLit = regexp.MustCompile(`exec\.Command\s*\(\s*"([^"]+)"|exec\.CommandContext\s*\([^,]+,\s*"([^"]+)"`)
	reGoInlineTerm = regexp.MustCompile(`exec\.Command(?:Context)?\s*\([^)]*\)\.(Run|Output|CombinedOutput|Start)\s*\(`)
)

var hardGoBackgroundFiles = map[string]bool{
	"internal/gardenbundle/gardenbundle.go": true,
	"internal/fleetpane/fleetpane.go":       true,
	"cmd/fak/taskmgr.go":                    true,
	"cmd/fak/tui_issues_garden.go":          true,
	"cmd/fak/steering.go":                   true,
	"cmd/fak/watchdog_autoheal.go":          true,
	"cmd/fak/treedoctor.go":                 true,
	"cmd/fak/release_status.go":             true,
}

var candidateConsoleTools = map[string]bool{
	"cmd": true, "cmd.exe": true,
	"dos": true,
	"fak": true, "fak.exe": true,
	"gh": true, "git": true,
	"go":         true,
	"powershell": true, "powershell.exe": true, "pwsh": true, "pwsh.exe": true,
	"python": true, "python.exe": true, "python3": true, "python3.exe": true,
	"schtasks": true, "schtasks.exe": true,
	"taskkill": true, "taskkill.exe": true, "tasklist": true, "tasklist.exe": true,
	"wsl": true, "wsl.exe": true,
}

// GoExecViolations returns one message per known background Go helper command
// that reaches Run/Output/CombinedOutput/Start before the Windows no-window hook
// is applied. The long-lived dispatch worker spawn may use configureDispatchSpawn;
// short dispatch probes may use configureDispatchHelperCommand; all other
// background helpers use ConfigureBackgroundCommand.
func GoExecViolations(rel, src string) []string {
	if strings.HasSuffix(rel, "_test.go") || !hardGoBackgroundPath(rel) {
		return nil
	}
	return goExecFindings(rel, src, true)
}

// GoExecCandidates returns advisory findings for literal console tools in Go
// files that are not yet part of the hard ratchet. It makes the remaining popup
// surface visible without instantly reding the whole shared tree.
func GoExecCandidates(rel, src string) []string {
	if strings.HasSuffix(rel, "_test.go") || hardGoBackgroundPath(rel) {
		return nil
	}
	return goExecFindings(rel, src, false)
}

func hardGoBackgroundPath(rel string) bool {
	return strings.HasPrefix(rel, "cmd/fak/dispatch") || hardGoBackgroundFiles[rel]
}

func goExecFindings(rel, src string, hard bool) []string {
	lines := strings.Split(src, "\n")
	var out []string
	for i, line := range lines {
		text := stripGoLineComment(line)
		if literalConsoleTool(text) && reGoInlineTerm.MatchString(text) {
			out = append(out, fmt.Sprintf("%s:%d: inline exec.Command reaches %s without "+
				"windowgate.ConfigureBackgroundCommand(cmd) — expand it to a command variable and configure the Windows no-window hook (%s)",
				rel, i+1, strings.TrimSpace(text), ReasonGoUnsuppressedExec))
			continue
		}
		m := reGoExecAssign.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		if !hard && !literalConsoleTool(line) {
			continue
		}
		name := m[1]
		configured := false
		for j := i + 1; j < len(lines) && j <= i+16; j++ {
			text := stripGoLineComment(lines[j])
			if strings.Contains(text, "configureDispatchHelperCommand("+name+")") ||
				strings.Contains(text, "configureDispatchSpawn("+name+")") ||
				strings.Contains(text, "windowgate.ConfigureBackgroundCommand("+name+")") ||
				strings.Contains(text, "ConfigureBackgroundCommand("+name+")") {
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

func literalConsoleTool(line string) bool {
	m := reGoCommandLit.FindStringSubmatch(line)
	if m == nil {
		return false
	}
	name := strings.TrimSpace(m[1])
	if name == "" && len(m) > 2 {
		name = strings.TrimSpace(m[2])
	}
	name = strings.ToLower(filepath.Base(strings.ReplaceAll(name, "\\", "/")))
	return candidateConsoleTools[name]
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

// Report holds the violation and watchlist rows from a tree scan.
type Report struct {
	PSInstallers      []string // task installers that flash
	PySpawns          []string // suppressing modules with an unflagged spawn
	PyCandidates      []string // advisory console-tool spawns in non-suppressing modules
	PyExplicitModules []string // Python modules using explicit no_window_creationflags call-site flags
	PyDefaultModules  []string // Python modules using install_no_window_subprocess_defaults(subprocess)
	GoExecs           []string // hard-ratcheted Go background helpers with unflagged child commands
	GoCandidates      []string // advisory literal console-tool launches not yet hard-ratcheted
}

// OK reports whether the tree is clean.
func (r Report) OK() bool {
	return len(r.PSInstallers) == 0 && len(r.PySpawns) == 0 && len(r.GoExecs) == 0
}

// ScanTree audits the on-disk worktree .ps1, .py, and Go helper files under repoRoot.
// It includes untracked, non-ignored files because a new cmd/fak/*.go file can
// compile into the binary before it is committed.
func ScanTree(repoRoot string) (Report, error) {
	var rep Report
	ps1, err := worktreeFiles(repoRoot, "*.ps1")
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
	py, err := worktreeFiles(repoRoot, "*.py")
	if err != nil {
		return rep, err
	}
	for _, rel := range py {
		src, err := readRel(repoRoot, rel)
		if err != nil {
			continue
		}
		if PyDefaultSuppressorInstalled(src) {
			rep.PyDefaultModules = append(rep.PyDefaultModules, rel)
		} else if PyExplicitSuppressorUsed(src) {
			rep.PyExplicitModules = append(rep.PyExplicitModules, rel)
		}
		rep.PySpawns = append(rep.PySpawns, PySpawnViolations(rel, src)...)
		rep.PyCandidates = append(rep.PyCandidates, PySpawnCandidates(rel, src)...)
	}
	goFiles, err := worktreeFiles(repoRoot, "*.go")
	if err != nil {
		return rep, err
	}
	for _, rel := range goFiles {
		src, err := readRel(repoRoot, rel)
		if err != nil {
			continue
		}
		rep.GoExecs = append(rep.GoExecs, GoExecViolations(rel, src)...)
		rep.GoCandidates = append(rep.GoCandidates, GoExecCandidates(rel, src)...)
	}
	sort.Strings(rep.PSInstallers)
	sort.Strings(rep.PySpawns)
	sort.Strings(rep.PyCandidates)
	sort.Strings(rep.PyExplicitModules)
	sort.Strings(rep.PyDefaultModules)
	sort.Strings(rep.GoExecs)
	sort.Strings(rep.GoCandidates)
	return rep, nil
}

func worktreeFiles(repoRoot string, globs ...string) ([]string, error) {
	args := append([]string{"ls-files", "--cached", "--others", "--exclude-standard", "--"}, globs...)
	cmd := exec.Command("git", args...)
	cmd.Dir = repoRoot
	ConfigureBackgroundCommand(cmd)
	out, err := cmd.Output()
	if err != nil {
		return nil, fmt.Errorf("git %s in %s: %w", strings.Join(args, " "), repoRoot, err)
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
