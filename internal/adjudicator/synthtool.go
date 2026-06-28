package adjudicator

import (
	"path"
	"strings"
	"sync"
)

// synthtool.go — #543: adjudicate a LIVE self-synthesized tool at the exec
// boundary, not the named-tool boundary.
//
// fak gates every NAMED tool call (Bash, Edit, Read) as a default-deny capability.
// An agent that synthesizes its own helper mid-run defeats that by STRUCTURE: it
// writes `helper.py` to the (unguarded) task workdir, then runs `python helper.py`.
// That one opaque exec collapses N future adjudicated actions into a single call the
// kernel never inspects — the same laundering class as the text-tool-call bypass.
// Live-SWE-agent does exactly this (synthesize a custom script, reuse/refine it
// across later steps) and validates it on execution-feedback alone — no adjudication.
//
// The honesty twist: bind the synthesized tool to the agent's CURRENT envelope so it
// can do no more than a direct adjudicated call could.
//
//   1. DETECT the synth->exec pattern (this file's execsAuthoredScript): an exec of
//      an interpreter (or a `./x` after `chmod +x`) against a path the SAME agent
//      authored earlier this run, tracked in a per-run ledger (the Adjudicator's
//      `authored` set, populated by noteAuthoredScript on each write-shaped call).
//   2. ENVELOPE it (synthToolSelfModify, wired into Adjudicate): a synth-tool exec is
//      treated as WRITE-SHAPED. A `python x.py` carries no write verb / no inline-eval
//      flag / no `>` redirect, so commandWrites (and thus commandSelfModify) waves it
//      through TODAY even when its command line reaches a guarded tree. This routes a
//      synth-tool that reaches a guarded glob through the SAME SELF_MODIFY deny a
//      direct write gets — default-deny still folds, and a self-written tool cannot
//      reach what a direct call could not.
//
// SCOPE (honest): adjudication sees the COMMAND, not the script body. This floor
// closes the command-line laundering case (a self-authored script whose invocation
// names a guarded tree). A script whose BODY writes a guarded tree WITHOUT naming it
// on the command line stays invisible to call-time inspection by construction — that
// residual is the journal/replay witness (issue #543 item 3), not a decide-path
// check. The ledger + detection built here are the machinery that witness builds on.

// scriptExtensions are the interpreter-runnable file suffixes a synthesized tool
// most plausibly carries. A path with one of these, written by the agent, is
// remembered as authored; a `chmod +x` target with no extension is remembered too
// (see noteAuthoredScript). Lower-cased before the suffix test.
var scriptExtensions = []string{
	".py", ".sh", ".bash", ".zsh", ".ksh", ".js", ".mjs", ".cjs", ".ts",
	".rb", ".pl", ".php", ".lua", ".r", ".ps1",
}

// interpreterNames are the runtimes that, in command-head position, run their first
// non-flag operand AS a script — the `<interpreter> <script>` synth-exec form. Keyed
// by the program BASENAME, lower-cased (so `/usr/bin/python3` resolves to `python3`).
var interpreterNames = map[string]bool{
	"python": true, "python2": true, "python3": true, "py": true,
	"sh": true, "bash": true, "dash": true, "zsh": true, "ksh": true,
	"node": true, "nodejs": true, "deno": true, "bun": true, "ts-node": true,
	"ruby": true, "perl": true, "php": true, "lua": true,
	"rscript": true, "pwsh": true, "powershell": true,
}

// normScriptPath cleans a path-like token to the canonical form recorded in /
// looked up from the authored ledger: surrounding quotes stripped, backslashes
// folded to forward slashes, and path.Clean applied (so `./helper.py` and
// `helper.py` are the same key). Returns "" for an empty token.
func normScriptPath(tok string) string {
	tok = strings.Trim(tok, "\"'")
	if tok == "" {
		return ""
	}
	return path.Clean(strings.ReplaceAll(tok, `\`, "/"))
}

// isScriptShaped reports whether a cleaned path has an interpreter-runnable suffix.
func isScriptShaped(p string) bool {
	lc := strings.ToLower(p)
	for _, ext := range scriptExtensions {
		if strings.HasSuffix(lc, ext) {
			return true
		}
	}
	return false
}

// rememberAuthored records p (and its basename) in the authored ledger. Both forms
// are stored so a later exec that names the script differently — `./helper.py` vs an
// absolute `/work/task/helper.py` — still resolves to the same authored entry.
func rememberAuthored(authored *sync.Map, p string) {
	if p == "" {
		return
	}
	authored.Store(p, struct{}{})
	if b := path.Base(p); b != "" && b != p {
		authored.Store(b, struct{}{})
	}
}

// isAuthored reports whether tok (a raw command token) names a script in the ledger,
// matching on the cleaned full path OR the basename.
func isAuthored(authored *sync.Map, tok string) bool {
	n := normScriptPath(tok)
	if n == "" {
		return false
	}
	if _, ok := authored.Load(n); ok {
		return true
	}
	if b := path.Base(n); b != n {
		if _, ok := authored.Load(b); ok {
			return true
		}
	}
	return false
}

// noteAuthoredScript records the script a write-shaped call authored, so the NEXT
// exec of it is recognized as a self-synthesized tool rather than an opaque binary.
// It is the ledger-population half of the #543 floor. It records ONLY script-shaped
// targets that are NOT in a guarded tree — a write into a guarded tree was already
// denied SELF_MODIFY upstream in Adjudicate, so it never lands and must not be
// remembered. Two authoring surfaces:
//
//   - a FILE-WRITE tool (Write/Edit/write_file) whose path arg is script-shaped;
//   - a SHELL write that creates/permits a script: a redirect/tee/cp/mv to a
//     script-shaped path, or `chmod +x <path>` marking a file executable.
//
// Over-recording is fail-SAFE: an extra remembered path only ever tightens a LATER
// exec that also reaches a guarded tree (the synthToolSelfModify deny), and a write
// the policy will deny downstream never produces an exec to catch.
func noteAuthoredScript(args map[string]any, tool string, authored *sync.Map, globs []string) {
	// File-write surface.
	if writeShaped(tool) {
		if n := normScriptPath(targetPath(args)); isScriptShaped(n) && matchGlob(n, globs) == "" {
			rememberAuthored(authored, n)
		}
	}
	// Shell-write surface.
	cmd, ok := commandArg(args)
	if !ok {
		return
	}
	if !commandWrites(cmd) {
		return // a read command authors nothing
	}
	toks := tokenizeCommand(cmd)
	chmodExec := strings.Contains(cmd, "+x") && containsToken(toks, "chmod")
	for _, raw := range toks {
		n := normScriptPath(raw)
		if n == "" || matchGlob(n, globs) != "" {
			continue
		}
		if isScriptShaped(n) || (chmodExec && isPlainOperand(raw)) {
			rememberAuthored(authored, n)
		}
	}
}

// synthToolSelfModify reports the guarded glob a SYNTHESIZED-TOOL invocation would
// reach, or "" if the command is not a synth-tool exec or reaches no guarded tree.
// It is the envelope-binding half of the #543 floor, the synth-exec dual of
// commandSelfModify: where commandSelfModify needs a write VERB to fire,
// synthToolSelfModify fires on an exec of a LEDGER script — a `python helper.py`
// that commandWrites calls read-shaped — once that command line also names a guarded
// glob. Decode-once args in, the offending glob out (deny SELF_MODIFY upstream).
//
// It fires only when BOTH hold: the command execs a ledger script AND the command
// string names a guarded glob. The ledger scoping is load-bearing: it is what keeps
// a legitimate `pytest` / `python manage.py runserver` — a script the agent did NOT
// author — unaffected. Only the agent's OWN mid-run synthesized scripts are
// enveloped; an unguarded synth-tool exec stays allowed (the tool inherits no LESS
// capability than the agent already holds, only no MORE).
func synthToolSelfModify(args map[string]any, authored *sync.Map, globs []string) string {
	cmd, g, ok := guardedCommandTree(args, globs)
	if !ok {
		return "" // no command, or reaches no guarded tree — nothing to envelope
	}
	if !execsAuthoredScript(cmd, authored) {
		return "" // not a self-synthesized-tool exec — leave the command to the other rungs
	}
	return g
}

// execsAuthoredScript reports whether cmd invokes a ledger script as an executable —
// `<interpreter> [flags] <authored-script>` or a direct `./<authored>` /
// `<authored>` in command position. It splits cmd on shell separators and inspects
// each sub-command's head; a script that merely appears as a READ argument
// (`cat helper.py`) is not an exec and does not match.
func execsAuthoredScript(cmd string, authored *sync.Map) bool {
	for _, sub := range splitSubcommands(cmd) {
		if subcmdExecsAuthored(sub, authored) {
			return true
		}
	}
	return false
}

// subcmdExecsAuthored decides one sub-command (already split into tokens). An
// interpreter head runs its first non-flag operand as a script; any other head is a
// direct exec of the program named at position 0.
func subcmdExecsAuthored(sub []string, authored *sync.Map) bool {
	sub = stripExecPrefix(sub)
	if len(sub) == 0 {
		return false
	}
	head := strings.Trim(sub[0], "\"'")
	if interpreterNames[strings.ToLower(path.Base(normScriptPath(head)))] {
		for _, t := range sub[1:] {
			if strings.HasPrefix(t, "-") {
				continue // skip interpreter flags (`-u`, `--`, …)
			}
			return isAuthored(authored, t) // the first operand is the script being run
		}
		return false
	}
	// Direct exec: the program itself is the (authored) script — `./t`, `/abs/t`.
	return isAuthored(authored, head)
}

// stripExecPrefix drops the leading run-wrapper tokens that a real shell consumes
// BEFORE the program word, so the true exec head is what subcmdExecsAuthored
// inspects. Without this an agent launders a synth-tool past the #543 envelope by
// the most ordinary prefix there is: `env python helper.py internal/abi/x.go` (the
// `#!/usr/bin/env python3` shebang shape) parks `env` — not an interpreter — in head
// position, and `FOO=1 python helper.py …` parks an assignment, so the guarded-tree
// reach goes un-enveloped. Two prefix forms are folded, matching shell command-word
// resolution:
//
//   - `VAR=val …` leading assignments (any number) — a token of the shape `name=…`
//     with a non-empty name and no slash before the `=` (so a path like `a=b/c` or
//     `./x` is never mistaken for an assignment);
//   - a leading `env` and its own options/assignments: `env`, `env -i`, `env -u FOO`,
//     `env --`, `env BAR=1` — consume env, then its flags (and `-u`'s argument) and
//     assignments, stopping at the first plain program word, which becomes the head.
//
// Conservative by construction (the commandSelfModify stance): a mis-strip only ever
// exposes a DIFFERENT token as the head, and synthToolSelfModify still fires only when
// that head execs a LEDGER script AND the command names a guarded glob — so widening
// what counts as the head can only tighten a command that already reaches a guarded
// tree, never open an unrelated one.
func stripExecPrefix(sub []string) []string {
	for len(sub) > 0 {
		tok := strings.Trim(sub[0], "\"'")
		if isAssignment(tok) {
			sub = sub[1:]
			continue
		}
		if strings.ToLower(path.Base(normScriptPath(tok))) == "env" {
			sub = stripEnvArgs(sub[1:])
			continue
		}
		break
	}
	return sub
}

// isAssignment reports whether tok is a `name=value` shell assignment prefix: a
// non-empty assignment-name (letters/digits/underscore) followed by `=`. A token
// with a slash before the `=` is a path, not an assignment.
func isAssignment(tok string) bool {
	eq := strings.IndexByte(tok, '=')
	if eq <= 0 {
		return false
	}
	name := tok[:eq]
	for _, r := range name {
		if !(r == '_' || (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9')) {
			return false
		}
	}
	return true
}

// stripEnvArgs consumes `env`'s own arguments — its flags, the operand of a flag that
// takes one (`-u NAME`), assignments, and the `--` end-of-options marker — returning
// the tokens from the program word onward. Unknown flags are skipped as zero-arg
// (env's real flag set is tiny: -i/-0/-u/-C/-S/-v); a `-u` consumes the next token.
func stripEnvArgs(sub []string) []string {
	for len(sub) > 0 {
		tok := strings.Trim(sub[0], "\"'")
		switch {
		case tok == "--":
			return sub[1:]
		case tok == "-u" || tok == "--unset":
			sub = sub[1:]
			if len(sub) > 0 {
				sub = sub[1:] // drop the variable name -u takes
			}
		case strings.HasPrefix(tok, "-"):
			sub = sub[1:]
		case isAssignment(tok):
			sub = sub[1:]
		default:
			return sub // the program word
		}
	}
	return sub
}

// splitSubcommands splits a command string into sub-commands on shell separators
// (`;`, `|`, `&`, newline, parens), each tokenized on whitespace. It is a loose
// tokenizer, not a shell parser — the floor needs only token boundaries to spot an
// interpreter head and its script operand, and the conservative over-broad stance
// commandSelfModify documents applies (a mis-split that flags one extra exec only
// tightens a command that ALSO names a guarded tree).
func splitSubcommands(cmd string) [][]string {
	var subs [][]string
	for _, piece := range strings.FieldsFunc(cmd, isSubcommandSep) {
		if toks := strings.Fields(piece); len(toks) > 0 {
			subs = append(subs, toks)
		}
	}
	return subs
}

func isSubcommandSep(r rune) bool {
	switch r {
	case ';', '|', '&', '\n', '(', ')':
		return true
	}
	return false
}

// tokenizeCommand splits a command string into flat operand tokens for the LEDGER
// pass — broader than splitSubcommands (it also splits on redirects `<`/`>` so a
// `tee >helper.py` target is its own token) with surrounding quotes stripped.
func tokenizeCommand(cmd string) []string {
	fields := strings.FieldsFunc(cmd, func(r rune) bool {
		switch r {
		case ' ', '\t', '\n', '\r', ';', '|', '&', '(', ')', '<', '>':
			return true
		}
		return false
	})
	out := make([]string, 0, len(fields))
	for _, f := range fields {
		out = append(out, strings.Trim(f, "\"'"))
	}
	return out
}

// containsToken reports whether toks contains the exact token t.
func containsToken(toks []string, t string) bool {
	for _, x := range toks {
		if x == t {
			return true
		}
	}
	return false
}

// isPlainOperand reports whether a raw token is a plain path operand (not a flag, a
// mode like `+x`, or the `chmod` verb itself) — the form a `chmod +x <file>` names
// as the file made executable.
func isPlainOperand(tok string) bool {
	if tok == "" || strings.HasPrefix(tok, "-") || strings.HasPrefix(tok, "+") {
		return false
	}
	return tok != "chmod"
}
