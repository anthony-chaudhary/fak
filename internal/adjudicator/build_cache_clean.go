package adjudicator

import (
	"strings"
)

// defaultBuildCacheCleanDenyRegex is deliberately broad enough to select every
// direct spelling shipped by the default policy. It is not the verdict: the
// structural classifier below resolves executed command words and flags, which
// is what distinguishes a real cache clear from quoted prose.
const defaultBuildCacheCleanDenyRegex = `(?i)\bgo(?:\.exe)?\b[^|;&\n]*\bclean\b[^|;&\n]*(?:[ \t])--?cache(?:=(?:true|1))?(?:[ \t]|$|[;&|])`

func isBuildCacheCleanArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "bash", "powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
		return pr.Re.String() == defaultBuildCacheCleanDenyRegex
	default:
		return false
	}
}

func commandClearsGoBuildCache(command string) bool {
	for _, source := range rceShellSources(command) {
		for _, segment := range rceShellSegments(source) {
			i := rceCommandWord(segment.argv)
			if i < 0 || rceProgramBasename(segment.argv[i]) != "go" {
				continue
			}
			if adjudicatorGoCleanDeletesBuildCache(segment.argv[i+1:]) {
				return true
			}
		}
	}
	return false
}

func adjudicatorGoCleanDeletesBuildCache(args []string) bool {
	i := 0
	for i < len(args) {
		tok := args[i]
		if tok == "--" {
			return false
		}
		if tok == "-C" {
			i += 2
			continue
		}
		if strings.HasPrefix(tok, "-") {
			i++
			continue
		}
		if tok != "clean" {
			return false
		}
		for _, cleanArg := range args[i+1:] {
			if cleanArg == "--" || !strings.HasPrefix(cleanArg, "-") {
				return false
			}
			switch strings.ToLower(cleanArg) {
			case "-cache", "--cache", "-cache=true", "--cache=true", "-cache=1", "--cache=1":
				return true
			}
		}
		return false
	}
	return false
}
