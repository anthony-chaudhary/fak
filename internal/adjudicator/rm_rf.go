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

// isRmRfArgRule reports whether pr is one of the two shipped recursive/forced-delete
// deny_regexes on a shell command arg, on a surface that ships it.
//
// The POSIX spelling used to be recognised on the Bash tool ALONE, even though
// cmd/fak/guard-default-policy.json ships the byte-identical regex on the
// shell_command, functions.shell_command, and exec_command mirrors too. Those
// mirrors therefore
// kept the raw-regex path and lost BOTH carve-outs the structural decision grants:
// a recursive delete confined to a declared scratchpad root, and the force-only
// single-literal-target delete that #4983 degraded to the reversibility confirm
// gate. The same command got two different verdicts depending only on which tool
// NAME the harness happened to use — `rm -f notes.txt` was preview-confirmed on
// Bash and a terminal POLICY_BLOCK on shell_command. A capability floor whose
// strictness tracks a harness's tool naming is not a floor, it is a coin flip, and
// the strict side was refusing routine work.
//
// Recognising both spellings on the generic surfaces closes that. It is safe in the
// same way the rest of this file is: the structural decision is consulted only after
// the raw regex has already matched and can only ever DOWNGRADE that match, and the
// POSIX walk resolves nothing it cannot read — a Windows-spelled path on a
// PowerShell-backed shell_command host loses its separators to POSIX lexing, fails
// to prove containment, and keeps the deny.
func isRmRfArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	re := pr.Re.String()
	switch strings.ToLower(pr.Tool) {
	case "bash":
		return re == defaultRmRfDenyRegex
	case "powershell":
		return re == defaultPSDeleteDenyRegex
	case "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
		// This surface is POSIX on one host and PowerShell on another, so the
		// shipped policy gives it BOTH rules; recognise both.
		return re == defaultRmRfDenyRegex || re == defaultPSDeleteDenyRegex
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
	return commandHasUnsafeRecursiveForcedDelete(cmd, "", nil)
}

// commandHasUnsafeRecursiveForcedDelete keeps the shipped fail-closed verdict
// except in two bounded carve-outs: (1) every recursive/forced delete target is a
// literal path strictly below a declared scratchpad root, and (2) the call is a
// FORCE-ONLY (no -r/-R) delete of a single explicit literal path — see
// rmArgvHardDenied for why that degrades to the reversibility confirm gate (#4983).
// Workspace paths are deliberately NOT scratch-exempt: recursive cleanup there can
// destroy real work even when it is in-tree.
func commandHasUnsafeRecursiveForcedDelete(cmd, ws string, scratch []string) bool {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			i := rmDeleteCommandWord(seg.argv)
			if i < 0 {
				continue
			}
			switch rceProgramBasename(seg.argv[i]) {
			case "rm":
				args := seg.argv[i+1:]
				if rmArgvHardDenied(args) && !rmDeleteTargetsInScratch(args, ws, scratch) {
					return true
				}
			case strings.ToLower("Remove" + "-Item"):
				args := seg.argv[i+1:]
				// psDeleteTargetsInScratch sees POSIX-lexed args, where a Windows
				// path has already lost every backslash — so on the only host this
				// cmdlet runs on it can never prove containment. Re-prove it under
				// PowerShell lexing before refusing. Purely additive: a second way
				// to PROVE the delete is confined, never a new way to deny.
				if argvHasPowerShellRecursiveOrForce(args) &&
					!psDeleteTargetsInScratch(args, ws, scratch) &&
					!psRemoveItemAllTargetsInScratch(src, ws, scratch) {
					return true
				}
			}
		}
	}
	return false
}

func rmDeleteTargetsInScratch(args []string, ws string, scratch []string) bool {
	return deleteTargetsStrictlyInScratch(rmDeleteTargets(args), ws, scratch)
}

// rmDeleteTargets extracts the path operands from an `rm` argv: every non-option
// token, with a bare `--` ending option scanning (the rest are operands). It is the
// shared operand walk behind both the scratch-containment check and the
// single-literal-target check (#4983).
func rmDeleteTargets(args []string) []string {
	var targets []string
	optionsDone := false
	for _, arg := range args {
		if !optionsDone && arg == "--" {
			optionsDone = true
			continue
		}
		if !optionsDone && strings.HasPrefix(arg, "-") {
			continue
		}
		targets = append(targets, arg)
	}
	return targets
}

func psDeleteTargetsInScratch(args []string, ws string, scratch []string) bool {
	var targets []string
	for i := 0; i < len(args); i++ {
		arg := args[i]
		lower := strings.ToLower(arg)
		switch lower {
		case "-path", "-literalpath":
			if i+1 >= len(args) {
				return false
			}
			i++
			targets = append(targets, args[i])
		default:
			if strings.HasPrefix(arg, "-") || strings.Contains(arg, ":") && strings.HasPrefix(lower, "-confirm:") {
				continue
			}
			targets = append(targets, arg)
		}
	}
	return deleteTargetsStrictlyInScratch(targets, ws, scratch)
}

func deleteTargetsStrictlyInScratch(targets []string, ws string, scratch []string) bool {
	if len(targets) == 0 || len(scratch) == 0 {
		return false
	}
	for _, target := range targets {
		ct, ok := canonicalizeArgValue(target)
		if !ok || ct == "" || strings.ContainsAny(ct, "$*?") {
			return false
		}
		abs := ct
		if !isAbsPath(abs) {
			if ws == "" {
				return false
			}
			abs = cleanRooted(ws + "/" + abs)
		} else {
			abs = cleanRooted(abs)
		}
		contained := false
		for _, root := range scratch {
			root = cleanRooted(root)
			// Deleting the scratch root itself is too broad. The carve-out is for
			// per-session children such as a throwaway clone directory.
			if !strings.EqualFold(strings.TrimRight(abs, "/"), strings.TrimRight(root, "/")) && isUnder(abs, root) {
				contained = true
				break
			}
		}
		if !contained {
			return false
		}
	}
	return true
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

// rmArgvHardDenied reports whether an `rm` argv must stay hard-denied
// (POLICY_BLOCK), as opposed to falling through to the reversibility rung's
// preview-confirm gate. Recursive (-r/-R/--recursive) is ALWAYS hard-denied: it
// turns a named delete into an unbounded tree removal. Force (-f/--force) WITHOUT
// recursive adds no blast radius that a plain `rm` — which the reversibility rung
// already preview-confirms, not hard-denies — does not already have: -f only
// suppresses the interactive are-you-sure prompt and the missing-file error,
// neither of which is a floor this guard enforces, and in a headless agent there is
// no prompt to suppress at all. So a force-only delete of a SINGLE explicit literal
// path degrades to that same confirm gate (#4983) — the hard deny was pure friction
// there, forcing a `rm -f foo` -> `rm foo` rewrite that reaches the identical
// deletion through the confirm gate anyway. Force-only stays hard-denied when the
// target set is unbounded or ambiguous — a glob (`rm -f *`), more than one path
// (`rm -f a b`), or a variable/command-substitution target — because there the
// force flag drives a wide, non-interactive sweep the single-target confirm preview
// cannot faithfully surface (the complaint's own requested scope, #4983).
func rmArgvHardDenied(args []string) bool {
	recursive, force := argvRecursiveForce(args)
	if recursive {
		return true
	}
	if !force {
		return false // neither -r nor -f: a plain rm the reversibility rung confirm-gates
	}
	return !rmSingleLiteralTarget(args)
}

// argvRecursiveForce scans an `rm` argv and reports whether it selects recursive
// (-r/-R/--recursive) and/or force (-f/--force) deletion. Short clusters match on
// any of r/R/f/F (mirroring the shipped `[rRfF]` class, in any cluster position);
// a bare `--` ends option scanning (the rest are path operands, as the raw regex
// also required a `-` immediately after `rm`).
func argvRecursiveForce(args []string) (recursive, force bool) {
	for _, t := range args {
		switch {
		case t == "--":
			return recursive, force
		case t == "--recursive":
			recursive = true
		case t == "--force":
			force = true
		case rceIsShortCluster(t):
			if rceClusterHas(t, 'r') || rceClusterHas(t, 'R') {
				recursive = true
			}
			if rceClusterHas(t, 'f') || rceClusterHas(t, 'F') {
				force = true
			}
		}
	}
	return recursive, force
}

// rmSingleLiteralTarget reports whether an `rm` argv names EXACTLY ONE path operand
// that is a plain literal — no glob metacharacter (`*`, `?`, `[`), no shell
// variable/expansion (`$`), and no command substitution (backtick). Those forms
// expand to an unbounded or caller-opaque target set, so a force-only delete naming
// one is NOT the bounded single-file case #4983 carves out.
func rmSingleLiteralTarget(args []string) bool {
	targets := rmDeleteTargets(args)
	if len(targets) != 1 {
		return false
	}
	return !strings.ContainsAny(targets[0], "*?[$`")
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
