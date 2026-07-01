// interactive.go — the would-hang interactive-command rung of the repo-guard
// PreToolUse hook (#2080).
//
// An interactive command assumes a human is watching; under a headless agent
// harness it is a silent hang or a silently-EOF'd no-op — either way a pure
// wasted turn. AGENTS.md warns against several by convention, but convention
// is a landmine an agent must remember across compaction. This rung refuses
// the curated hang-certain forms BEFORE execution, with a structured
// INTERACTIVE_HANG reason whose fix pre-fills the runnable non-interactive
// equivalent.
//
// Curation principle: only invocations that genuinely wedge or no-op without
// a TTY are refused. Deliberately NOT in the set: pagers (`less`, `git log`)
// — they detect the missing TTY and degrade to `cat`; bare REPLs (`python`,
// `node`) — a pipeline segment (`echo x | python`) is indistinguishable from
// a REPL launch, so refusing them would deny benign forms. Pure: no
// filesystem access, hermetically testable like the rest of the core.
package repoguard

import "strings"

// ReasonInteractiveHang is the structured refusal token for would-hang
// interactive invocations.
const ReasonInteractiveHang = "INTERACTIVE_HANG"

// interactiveEditors are full-screen editors that grab the terminal and wait
// for a human. vim draws to the pipe and blocks on stdin when headless.
var interactiveEditors = setOf(
	"vi", "vim", "nvim", "gvim", "vimdiff", "view", "nano", "pico", "emacs", "joe",
)

// editorOverrideVars: an explicit editor override on the invocation itself
// (`GIT_SEQUENCE_EDITOR=: git rebase -i ...`) signals scripted intent — the
// author already routed around the editor — so the segment passes.
var editorOverrideVars = setOf("GIT_EDITOR", "GIT_SEQUENCE_EDITOR", "EDITOR", "VISUAL")

// patchModeSubs are the git subcommands whose -p/--patch enters the
// interactive hunk picker (which EOFs into a no-op with a closed stdin).
var patchModeSubs = setOf("add", "checkout", "reset", "restore", "stash", "commit")

// classifyInteractive returns INTERACTIVE_HANG violations for the curated
// would-hang interactive invocations in a shell command. Pure string work.
func classifyInteractive(command string) []Violation {
	var out []Violation
	for _, seg := range splitSegments(command) {
		toks, ok := shlexSplit(seg)
		if !ok {
			toks = strings.Fields(seg)
		}
		verb, operands, overridden := stripEnvAndEnvVerb(toks)
		if verb == "" || overridden {
			continue
		}
		if v := classifySegment(verb, operands, strings.TrimSpace(seg)); v != nil {
			out = append(out, *v)
		}
	}
	return out
}

// ClassifyInteractive returns would-hang interactive violations for a shell command.
func ClassifyInteractive(command string) []Violation {
	return classifyInteractive(command)
}

// stripEnvAndEnvVerb skips leading NAME=VALUE assignments (and an `env` verb
// with its own assignments) and returns the real verb + its operands.
// overridden is true when an editor-override variable is assigned inline.
func stripEnvAndEnvVerb(toks []string) (verb string, operands []string, overridden bool) {
	i := 0
	for i < len(toks) {
		t := toks[i]
		if name, isAssign := envAssignName(t); isAssign {
			if editorOverrideVars[name] {
				overridden = true
			}
			i++
			continue
		}
		if strings.TrimSuffix(basename(t), ".exe") == "env" {
			i++
			continue
		}
		break
	}
	if i >= len(toks) {
		return "", nil, overridden
	}
	return strings.TrimSuffix(basename(toks[i]), ".exe"), toks[i+1:], overridden
}

// envAssignName parses a leading shell env assignment token (NAME=value).
func envAssignName(tok string) (string, bool) {
	eq := strings.IndexByte(tok, '=')
	if eq <= 0 {
		return "", false
	}
	name := tok[:eq]
	for j := 0; j < len(name); j++ {
		c := name[j]
		if !(c == '_' || isAlpha(c) || (j > 0 && c >= '0' && c <= '9')) {
			return "", false
		}
	}
	return name, true
}

// classifySegment applies the curated rules to one simple command.
func classifySegment(verb string, operands []string, invocation string) *Violation {
	switch {
	case interactiveEditors[verb]:
		if verb == "emacs" && hasAnyToken(operands, "--batch", "-batch", "--script") {
			return nil // scripted emacs
		}
		if hasAnyToken(operands, "-es", "-Es") {
			return nil // vim/nvim silent-ex batch mode
		}
		return interactiveViolation(verb, invocation,
			"launches a full-screen editor that waits for a human",
			"use the harness Edit/Write tools, or a scripted edit: sed -i 's/OLD/NEW/' <file>")
	case verb == "git":
		return classifyGit(operands, invocation)
	case verb == "gh":
		return classifyGh(operands, invocation)
	case verb == "visudo":
		if hasAnyToken(operands, "-c", "--check") {
			return nil // validate-only mode
		}
		return interactiveViolation("visudo", invocation,
			"opens the sudoers editor and waits for a human",
			"validate a candidate file instead: visudo -cf <file>")
	case verb == "crontab":
		if hasAnyToken(operands, "-e") {
			return interactiveViolation("crontab -e", invocation,
				"opens the crontab editor and waits for a human",
				"write the schedule to a file, then: crontab <file>")
		}
	}
	return nil
}

// classifyGit handles the git subcommand rules.
func classifyGit(operands []string, invocation string) *Violation {
	sub, args := gitSubcommand(operands)
	flags := flagsBeforeDashDash(args)
	switch {
	case sub == "rebase" && (hasLongFlag(flags, "--interactive") || clusterHas(flags, 'i')):
		return interactiveViolation("git rebase -i", invocation,
			"interactive rebase opens a sequence editor and waits for a human",
			"GIT_SEQUENCE_EDITOR=: "+invocation)
	case sub == "add" && (hasLongFlag(flags, "--interactive") || clusterHas(flags, 'i')):
		return interactiveViolation("git add -i", invocation,
			"interactive staging prompts a human (a closed stdin EOFs it into a no-op)",
			"stage whole paths: git add -- <paths>")
	case patchModeSubs[sub] && (hasLongFlag(flags, "--patch") || clusterHas(flags, 'p')):
		return interactiveViolation("git "+sub+" -p", invocation,
			"the interactive hunk picker prompts a human (a closed stdin EOFs it into a no-op)",
			"operate on whole paths: git "+sub+" -- <paths>")
	case sub == "commit":
		if hasAnyToken(flags, "--help", "-h", "--dry-run") {
			return nil // no editor opens; help/status output only
		}
		if hasLongFlag(flags, "--edit") || clusterHas(flags, 'e') {
			return commitEditorViolation(invocation)
		}
		if commitHasMessageSource(flags) {
			return nil
		}
		return commitEditorViolation(invocation)
	}
	return nil
}

func commitEditorViolation(invocation string) *Violation {
	return interactiveViolation("git commit", invocation,
		"opens the commit-message editor and waits for a human",
		`git commit -s -m "<type>(<leaf>): <subject> (fak <leaf>)" -- <paths>`)
}

// commitHasMessageSource reports whether a git-commit flag set supplies the
// message without an editor: -m/--message, -F/--file, -C/--reuse-message,
// --no-edit, or an autosquash --fixup. (-c/--reedit-message and bare
// --amend/--squash still open the editor, so they are NOT sources.)
func commitHasMessageSource(flags []string) bool {
	for _, f := range flags {
		switch {
		case f == "--message" || strings.HasPrefix(f, "--message="),
			f == "--file" || strings.HasPrefix(f, "--file="),
			f == "--no-edit",
			strings.HasPrefix(f, "--reuse-message"),
			strings.HasPrefix(f, "--fixup"):
			return true
		case isShortCluster(f) && strings.ContainsAny(f, "mFC"):
			return true
		}
	}
	return false
}

// classifyGh handles gh: `gh auth login` without --with-token opens a
// browser/device-code prompt and waits.
func classifyGh(operands []string, invocation string) *Violation {
	var words []string
	for _, t := range operands {
		if !strings.HasPrefix(t, "-") {
			words = append(words, t)
			if len(words) == 2 {
				break
			}
		}
	}
	if len(words) == 2 && words[0] == "auth" && words[1] == "login" &&
		!hasAnyToken(operands, "--with-token") {
		return interactiveViolation("gh auth login", invocation,
			"opens a browser/device-code prompt and waits for a human",
			"gh auth login --with-token < <token-file>  (or set GH_TOKEN)")
	}
	return nil
}

// gitSubcommand finds the git subcommand, skipping the value-taking global
// flags (`git -C <dir> -c <k=v> commit ...`).
func gitSubcommand(operands []string) (string, []string) {
	i := 0
	for i < len(operands) {
		t := operands[i]
		if !strings.HasPrefix(t, "-") {
			return t, operands[i+1:]
		}
		switch t {
		case "-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path":
			i += 2
		default:
			i++
		}
	}
	return "", nil
}

// flagsBeforeDashDash returns the flag tokens before a bare `--` (after which
// everything is a pathspec, never a flag).
func flagsBeforeDashDash(args []string) []string {
	var flags []string
	for _, t := range args {
		if t == "--" {
			break
		}
		if strings.HasPrefix(t, "-") {
			flags = append(flags, t)
		}
	}
	return flags
}

func hasAnyToken(toks []string, want ...string) bool {
	for _, t := range toks {
		for _, w := range want {
			if t == w {
				return true
			}
		}
	}
	return false
}

func hasLongFlag(flags []string, long string) bool {
	for _, f := range flags {
		if f == long {
			return true
		}
	}
	return false
}

// isShortCluster reports whether a token is a bundled short-flag group (-am),
// as opposed to a long flag (--amend) or a bare `--`.
func isShortCluster(tok string) bool {
	return len(tok) > 1 && tok[0] == '-' && tok[1] != '-'
}

// clusterHas reports whether any short-flag cluster contains the given letter
// (so `git add -up` matches 'p' the way `git add -p` does).
func clusterHas(flags []string, letter byte) bool {
	for _, f := range flags {
		if isShortCluster(f) && strings.IndexByte(f, letter) > 0 {
			return true
		}
	}
	return false
}

func interactiveViolation(op, invocation, why, fix string) *Violation {
	return &Violation{
		Reason:   ReasonInteractiveHang,
		Op:       op,
		Target:   invocation,
		Resolved: "<no-tty>",
		Why:      why,
		Fix:      fix,
	}
}
