package windowgate

import (
	"fmt"
	"path/filepath"
	"regexp"
	"strings"
)

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
	"gh": true, "gh.exe": true, "git": true, "git.exe": true,
	"go": true, "go.exe": true,
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
	if strings.HasSuffix(rel, "_test.go") {
		return nil
	}
	if hardGoBackgroundPath(rel) {
		return goExecFindings(rel, src, true, false)
	}
	// Go commands are repository control-plane helpers wherever they live. The Go
	// toolchain is a console executable on Windows, so an unconfigured invocation
	// can allocate a transient black conhost window even outside dispatch code.
	return goExecFindings(rel, src, false, true)
}

// GoExecCandidates returns advisory findings for literal console tools in Go
// files that are not yet part of the hard ratchet. It makes the remaining popup
// surface visible without instantly reding the whole shared tree.
func GoExecCandidates(rel, src string) []string {
	if strings.HasSuffix(rel, "_test.go") || hardGoBackgroundPath(rel) {
		return nil
	}
	return goExecFindings(rel, src, false, false)
}

func hardGoBackgroundPath(rel string) bool {
	return strings.HasPrefix(rel, "cmd/fak/dispatch") || hardGoBackgroundFiles[rel]
}

func goExecFindings(rel, src string, hard, onlyGo bool) []string {
	lines := strings.Split(src, "\n")
	var out []string
	for i, line := range lines {
		text := stripGoLineComment(line)
		if literalConsoleTool(text) && (!onlyGo || literalGoTool(text)) && reGoInlineTerm.MatchString(text) {
			out = append(out, fmt.Sprintf("%s:%d: inline exec.Command reaches %s without "+
				"windowgate.ConfigureBackgroundCommand(cmd) — expand it to a command variable and configure the Windows no-window hook (%s)",
				rel, i+1, strings.TrimSpace(text), ReasonGoUnsuppressedExec))
			continue
		}
		m := reGoExecAssign.FindStringSubmatch(line)
		if m == nil {
			continue
		}
		if onlyGo && !literalGoTool(line) {
			continue
		}
		if !hard && !onlyGo && !literalConsoleTool(line) {
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

func literalGoTool(line string) bool {
	m := reGoCommandLit.FindStringSubmatch(line)
	if m == nil {
		return false
	}
	tool := m[1]
	if tool == "" {
		tool = m[2]
	}
	tool = strings.ToLower(filepath.Base(strings.ReplaceAll(tool, `\`, "/")))
	return tool == "go" || tool == "go.exe"
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
