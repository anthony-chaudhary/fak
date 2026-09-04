package adjudicator

import "strings"

// defaultSudoDenyRegex is the one canonical spelling of the shipped privilege-
// escalation deny_regex (cmd/fak/guard-default-policy.json and its shell_command
// mirrors). The rule is RECOGNISED by this exact string and then decided
// structurally, exactly like the recursive/forced-delete rule (#4983) and the RCE
// download-pipe rule (#1465). A policy that ships a different spelling is
// unaffected and keeps the raw-regex path.
const defaultSudoDenyRegex = `\bsudo\b`

// isSudoArgRule reports whether pr is the shipped privilege-escalation
// deny_regex on a shell command arg. The Bash tool (and its lowercase harness
// alias) and the shell_command / functions.shell_command / exec_command mirrors
// all ship the identical `\bsudo\b` spelling, so all four take the structural path; the
// PowerShell escalation rule (Start-Process -Verb RunAs) is a different rule and
// is untouched.
func isSudoArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "bash", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
		return pr.Re.String() == defaultSudoDenyRegex
	default:
		return false
	}
}

// commandLocalEscalationWord reports whether cmd runs sudo (or its drop-in
// doas) at a real LOCAL command-word position, returning the resolved word as
// the forensic witness. It mirrors commandHasRecursiveForcedDelete: it unwraps
// sh -c / $() / backticks via rceShellSources, tokenizes each source (quoted
// words are single tokens, never a command) via rceShellSegments, and resolves
// each segment's command word past env-assigns / env / command and the
// transparent wrappers (xargs, nohup, time, …). Command-word resolution is what
// removes the raw regex's two failure modes: quoted text
// (`echo 'sudo make install'`) is never a command word, and a REMOTE escalation
// (`ssh gpu-box 'sudo systemctl restart …'` — the dominant false POLICY_BLOCK
// in real GPU bring-up trajectories) resolves to the command word `ssh`, whose
// remote effects are governed by the remote host's own controls, not this local
// floor. A local escalation stays denied under every launder the walk unwraps:
// `FOO=1 sudo x`, `env sudo x`, `nohup sudo x`, `sh -c 'sudo x'`, `$(sudo x)`,
// `/usr/bin/sudo x`, and the doas spelling the raw regex never caught.
func commandLocalEscalationWord(cmd string) (string, bool) {
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			if word, ok := segmentEscalationWord(seg.argv); ok {
				return word, true
			}
		}
	}
	return "", false
}

// segmentEscalationWord resolves a segment's effective command word — skipping
// env-assigns, env (and its flags), the `command` builtin, and the transparent
// rmDeleteWrappers (and their own short flags) — and reports whether that word
// is a privilege-escalation program.
func segmentEscalationWord(argv []string) (string, bool) {
	for i := 0; i < len(argv); i++ {
		base := rceProgramBasename(argv[i])
		switch {
		case rceIsAssign(argv[i]):
			continue
		case base == "env":
			for i+1 < len(argv) && (rceIsAssign(argv[i+1]) || strings.HasPrefix(argv[i+1], "-")) {
				i++
			}
			continue
		case base == "command":
			continue
		case rmDeleteWrappers[base]:
			for i+1 < len(argv) && strings.HasPrefix(argv[i+1], "-") {
				i++
			}
			continue
		case base == "sudo" || base == "doas":
			return base, true
		default:
			return "", false
		}
	}
	return "", false
}
