package adjudicator

import (
	"strings"
)

// defaultGitPushDenyRegex and defaultPSGitPushDenyRegex are the shipped spellings
// of the git push deny_regex in the default policy manifests.
const (
	defaultGitPushDenyRegex   = `\bgit\s+push\b`
	defaultPSGitPushDenyRegex = `(?i)\bgit\s+push\b`
)

// isGitPushArgRule reports whether pr is one of the shipped git push rules on a
// shell command arg, across all recognized shell surfaces.
func isGitPushArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "bash", "powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
	default:
		return false
	}
	switch pr.Re.String() {
	case defaultGitPushDenyRegex, defaultPSGitPushDenyRegex:
		return true
	default:
		return false
	}
}

// commandExecutesGitPush reports whether cmd actually invokes git push at a
// command-execution position. It returns false for mentions in echo, grep,
// commit messages, or inert heredocs.
func commandExecutesGitPush(cmd string) bool {
	return posixGitPush(cmd) || psGitPush(cmd, 0)
}

func posixGitPush(cmd string) bool {
	cleaned := stripCatHeredocs(cmd)
	for _, src := range rceShellSources(cleaned) {
		for _, seg := range rceShellSegments(src) {
			i := rceCommandWord(seg.argv)
			if i < 0 || rceProgramBasename(seg.argv[i]) != "git" {
				continue
			}
			if gitArgvExecutesPush(seg.argv[i+1:]) {
				return true
			}
		}
	}
	return false
}

func psGitPush(src string, depth int) bool {
	cleaned := stripCatHeredocs(src)
	return psSourceMatches(cleaned, depth, func(head string, rest []psToken) bool {
		return head == "git" && gitArgvExecutesPush(psTokenTexts(rest))
	})
}

// gitArgvExecutesPush inspects the arguments following git/git.exe and reports
// whether the subcommand being executed is "push".
func gitArgvExecutesPush(argv []string) bool {
	sub, ok := gitSubcommand(argv)
	if !ok {
		return false
	}
	return sub == "push"
}

// gitSubcommand returns the first positional argument after stepping over git
// global flags (and their separated values, where applicable).
func gitSubcommand(argv []string) (string, bool) {
	skipNext := false
	for i := 0; i < len(argv); i++ {
		tok := argv[i]
		if tok == "" {
			continue
		}
		if skipNext {
			skipNext = false
			continue
		}
		if tok == "--" {
			for j := i + 1; j < len(argv); j++ {
				if argv[j] != "" {
					return strings.ToLower(argv[j]), true
				}
			}
			return "", false
		}
		if strings.HasPrefix(tok, "-") {
			if gitGlobalFlagTakesArg(tok) {
				skipNext = true
			}
			continue
		}
		return strings.ToLower(tok), true
	}
	return "", false
}

// gitGlobalFlagTakesArg reports whether a git global flag consumes the following
// argument token when no "=" is present in the flag token itself.
func gitGlobalFlagTakesArg(tok string) bool {
	if strings.Contains(tok, "=") {
		return false
	}
	switch tok {
	case "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--super-prefix", "--config-env", "--exec-path":
		return true
	default:
		return false
	}
}

// stripCatHeredocs removes the body of cat here-documents where cat only prints
// bytes to stdout or a file and does not execute commands.
func stripCatHeredocs(cmd string) string {
	if !strings.Contains(cmd, "<<") || !strings.Contains(cmd, "\n") {
		return cmd
	}
	lines := strings.Split(strings.ReplaceAll(cmd, "\r\n", "\n"), "\n")
	var out []string
	var delim string
	inHeredoc := false

	for i := 0; i < len(lines); i++ {
		line := lines[i]
		if inHeredoc {
			trimmed := strings.TrimSpace(line)
			if trimmed == delim || strings.TrimLeft(line, "\t") == delim {
				inHeredoc = false
				delim = ""
			}
			continue
		}
		out = append(out, line)
		if d, ok := catHeredocDelim(line); ok {
			inHeredoc = true
			delim = d
		}
	}
	return strings.Join(out, "\n")
}

// catHeredocDelim inspects line for a safe cat here-document opener (e.g. `cat <<EOF`).
// It returns the delimiter and true only when the line cannot execute the body.
func catHeredocDelim(line string) (string, bool) {
	if !strings.Contains(line, "<<") {
		return "", false
	}
	if strings.ContainsAny(line, "|;&$`()") {
		return "", false
	}

	trimmed := strings.TrimSpace(line)
	fields := strings.Fields(trimmed)
	if len(fields) == 0 {
		return "", false
	}
	cmdWord := rceProgramBasename(fields[0])
	if cmdWord != "cat" {
		return "", false
	}

	idx := strings.Index(line, "<<")
	if idx < 0 {
		return "", false
	}
	rest := line[idx+2:]
	if strings.HasPrefix(rest, "<") {
		return "", false
	}
	if strings.HasPrefix(rest, "-") {
		rest = rest[1:]
	}
	rest = strings.TrimSpace(rest)
	if rest == "" {
		return "", false
	}

	var delim string
	if rest[0] == '\'' || rest[0] == '"' {
		q := rest[0]
		end := strings.IndexByte(rest[1:], q)
		if end < 0 {
			return "", false
		}
		delim = rest[1 : end+1]
	} else {
		end := strings.IndexAny(rest, " \t\r\n><")
		if end < 0 {
			delim = rest
		} else {
			delim = rest[:end]
		}
	}
	delim = strings.TrimPrefix(delim, "\\")
	if delim == "" {
		return "", false
	}
	return delim, true
}
