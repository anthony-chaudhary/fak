package adjudicator

import "strings"

// defaultRmRfDenyRegex is the one canonical spelling of the shipped
// recursive/forced-delete deny_regex (cmd/fak/guard-default-policy.json and its
// byte-identical mirrors). The rule is RECOGNISED by this exact string and then
// decided structurally, exactly like the RCE download-pipe rule (#1465). A policy
// that ships a different spelling is unaffected and keeps the raw-regex path.
const defaultRmRfDenyRegex = `\brm\s+-[A-Za-z]*[rRfF]`

var defaultPSDeleteDenyRegex = `(?i)\b` + "Remove" + `-Item\b[^|;\n]*-(Recurse|Force)\b`

// rmDeleteWrappers are transparent, flag-optional command wrappers whose real
// command word is the token that follows them (after any of the wrapper's own
// short flags). The shipped raw regex caught `find … | xargs rm -rf` and
// `nohup rm -rf` by substring; resolving the segment's command word past these
// keeps those denied so the structural rule never silently exempts them.
var rmDeleteWrappers = map[string]bool{
	"xargs":  true,
	"time":   true,
	"nice":   true,
	"nohup":  true,
	"setsid": true,
	"stdbuf": true,
	"ionice": true,
}

// isRmRfArgRule reports whether pr is the shipped recursive/forced-delete
// deny_regex on a Bash command arg. Like isRCEPipeArgRule it is Bash-scoped
// (EqualFold ⇒ the lowercase `bash` harness alias matches too); the identical
// shell_command / functions.shell_command / PowerShell mirrors keep the raw
// regex path.
func isRmRfArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	if strings.EqualFold(pr.Tool, "Bash") {
		return pr.Re.String() == defaultRmRfDenyRegex
	}
	switch strings.ToLower(pr.Tool) {
	case "powershell", "shell_command", "functions.shell_command":
		return pr.Re.String() == defaultPSDeleteDenyRegex
	default:
		return false
	}
}

// commandHasRecursiveForcedDelete reports whether cmd runs `rm` with a recursive
// (-r/-R/--recursive) OR force (-f/--force) flag at a real command boundary. It
// mirrors commandHasRemotePipeToInterpreter: it unwraps sh -c / $() / backticks via
// rceShellSources, tokenizes each source (quoted words are single tokens, never a
// command) via rceShellSegments, and resolves each segment's command word past
// env-assign / env / sudo / command (rceCommandWord) and the transparent wrappers
// above. Command-word resolution is what removes the raw regex's two failure
// modes: quoted text (`echo 'rm -rf /'`) is never a command word, and a laundered
// flag order (`rm -i -rf`, `rm --recursive --force`) is caught because ALL of rm's
// argv flags are scanned, not just the first cluster. `git rm` is exempt by
// construction — its command word is `git`, not `rm`.
//
// Residuals (bounded scope, matching the RCE rule): leading-arg wrappers
// (`timeout 5 rm -rf`), a replacement-string split across tokens
// (`xargs -I {} rm -rf`), variable/alias indirection (`R=rm; $R -rf x`), and
// deletion via a different tool (`find -delete`, `shred`) — the last class was
// never caught by the raw regex either, so it is no regression.
func commandHasRecursiveForcedDelete(cmd string) bool {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rmDeleteCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			switch rceProgramBasename(seg.argv[i]) {
			case "rm":
				if argvHasRecursiveOrForce(seg.argv[i+1:]) {
					return true
				}
			case strings.ToLower("Remove" + "-Item"):
				if argvHasPowerShellRecursiveOrForce(seg.argv[i+1:]) {
					return true
				}
			}
		}
	}
	return false
}

// rmDeleteCommandWord resolves the effective command-word index of argv, first via
// rceCommandWord (env-assign / env / sudo / command) and then skipping any
// transparent rmDeleteWrappers (and their own short flags) so the real command
// that follows becomes the resolved word. Returns -1 when a wrapper has no
// following command.
func rmDeleteCommandWord(argv []string) int {
	i := rceCommandWord(argv)
	for i >= 0 && i < len(argv) && rmDeleteWrappers[rceProgramBasename(argv[i])] {
		j := i + 1
		for j < len(argv) && strings.HasPrefix(argv[j], "-") {
			j++
		}
		if j >= len(argv) {
			return -1
		}
		i = j
	}
	return i
}

// argvHasRecursiveOrForce reports whether any flag in args selects recursive
// (-r/-R/--recursive) or force (-f/--force) delete. Short clusters match on any of
// r/R/f/F (mirroring the shipped `[rRfF]` class, in any cluster position); a bare
// `--` ends option scanning (the rest are path operands, as the raw regex also
// required a `-` immediately after `rm`).
func argvHasRecursiveOrForce(args []string) bool {
	for _, t := range args {
		switch {
		case t == "--":
			return false
		case t == "--recursive" || t == "--force":
			return true
		case rceIsShortCluster(t) && (rceClusterHas(t, 'r') || rceClusterHas(t, 'R') ||
			rceClusterHas(t, 'f') || rceClusterHas(t, 'F')):
			return true
		}
	}
	return false
}

func argvHasPowerShellRecursiveOrForce(args []string) bool {
	for _, arg := range args {
		name := strings.ToLower(strings.TrimLeft(arg, "-"))
		if name == "recurse" || name == "force" {
			return true
		}
	}
	return false
}
