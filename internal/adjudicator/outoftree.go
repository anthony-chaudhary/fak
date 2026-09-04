package adjudicator

import (
	"os"
	"path"
	"strings"
	"sync"
)

// outoftree.go — structural decision for the OUT-OF-TREE WRITE arg-rule family
// (the `-o ..`, `--output ..`, `>>? ..`, and `cp|mv|install|tee|rsync|ln … ..`
// deny_regex rules shipped in cmd/fak/guard-default-policy.json and its mirrors).
//
// WHY THIS EXISTS. Those rules are raw substring regexes on a literal `..[\/]`.
// They are simultaneously OVER-broad and UNDER-broad:
//   - over: they DENY a legitimate write reached via `..` even when the resolved
//     target lands INSIDE the working tree (`go build -o build/../bin/fak`), when
//     the `..` is only a READ source (`cp ../vendor/lib.a build/lib.a`), or when
//     it targets the sanctioned harness scratchpad (which lives outside the tree,
//     so it is reachable only via `..`). Under `fak guard -- claude` a false
//     POLICY_BLOCK reads as an agent-chosen end_turn and silently kills the turn —
//     the same failure mode decide.go documents for the rm_rf / rce_pipe rules.
//   - under: because they REQUIRE a `..`, they miss every ABSOLUTE escape
//     (`-o /etc/cron.d/x`, `tee /etc/passwd`) and `$HOME`/`~` write.
//
// THE FIX is a fail-CLOSED, purely SUBTRACTIVE structural decider, mirroring the
// recognise-then-decide shape of rm_rf.go / rce_pipe.go. It changes the verdict in
// exactly ONE direction: it turns a raw-regex DENY into an ALLOW only when it can
// PROVE that every write DESTINATION the command names resolves inside the
// workspace root or an explicitly-declared scratchpad root. In every other case —
// an escaping target, an unresolvable ($VAR / glob) target, an unidentifiable
// target, or a missing workspace root — it keeps the DENY. It never allows
// anything the raw regex would not already have denied, and it never introduces a
// new deny for a command the raw regex would not have matched (the raw match is a
// precondition, threaded in as rawMatches). See the adversarial cases pinned in
// outoftree_test.go.
//
// Containment is decided WITHOUT the filesystem (no stat, no symlink resolution) so
// the decide path stays exec-free and hermetic; symlink/TOCTOU laundering through an
// in-tree link is an explicit non-goal it shares with the rest of the pure floor.

// The four canonical deny_regex spellings of the out-of-tree write family, matched
// by Re.String() exactly (a policy that ships a different spelling keeps the raw
// path, like rm_rf/rce_pipe). Kept in sync with cmd/fak/guard-default-policy.json.
const (
	ootDashORegex    = `-o\s+\.\.[\\/]`
	ootOutputRegex   = `--output[= ]\s*\.\.[\\/]`
	ootRedirectRegex = `>>?\s*\.\.[\\/]`
	ootCopyVerbRegex = `\b(cp|mv|install|tee|rsync|ln)\b[^|;&]*\s\.\.[\\/]`
)

// ootDestVerbs are the copy/move-shaped commands whose LAST operand is the write
// destination (install/ln/rsync included). tee is handled separately (it writes
// EVERY file operand, not just the last).
var ootDestVerbs = map[string]bool{
	"cp": true, "mv": true, "install": true, "rsync": true, "ln": true,
}

// isOutOfTreeWriteArgRule reports whether pr is one of the four shipped out-of-tree
// write deny_regex rules on a POSIX-shell command arg. Scoped to the shell dialects
// the rce tokenizer understands ({Bash, shell_command, functions.shell_command,
// exec_command});
// PowerShell is deliberately excluded (it has no out-of-tree rules today and the
// tokenizer is POSIX — closing the PowerShell gap is tracked separately).
func isOutOfTreeWriteArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "bash", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
	default:
		return false
	}
	if pr.Arg != "command" && pr.Arg != "cmd" {
		return false
	}
	switch pr.Re.String() {
	case ootDashORegex, ootOutputRegex, ootRedirectRegex, ootCopyVerbRegex:
		return true
	default:
		return false
	}
}

// outOfTreeWriteEscapes is the pure decision. It is consulted ONLY when the rule's
// raw regex already matched (rawMatches); the caller passes that in so this
// function never turns a non-matching command into a deny.
//
// Returns escapes=true (keep the DENY) unless it can PROVE the raw match was a
// false positive: every write destination named by cmd resolves, after
// canonicalisation, strictly under ws or one of scratch (or a null device). A
// destination that is absent-from-extraction, unresolvable ($VAR/glob),
// undecodable, or outside every allowed root keeps the deny. An empty ws (root
// unknown) also keeps the deny — it never delegates containment to an empty root.
func outOfTreeWriteEscapes(cmd, ws string, scratch []string, rawMatches bool) (escapes bool) {
	if !rawMatches {
		return false // caller must not deny a command the raw regex would not match
	}
	targets := outOfTreeWriteTargets(cmd)
	if len(targets) == 0 {
		// No destination identified. That is the fail-closed case in general — a
		// shape the extractor does not understand must keep the deny — but it is
		// also what a quoted MENTION of a traversal looks like, and the rules fire
		// on any command that merely names one. Admit only where the absence is
		// PROVEN; see ootMentionOnly.
		return !ootMentionOnly(cmd)
	}
	for _, t := range targets {
		ct, ok := canonicalizeArgValue(t)
		if !ok {
			return true // undecodable destination — keep the deny
		}
		if !targetContained(ct, ws, scratch) {
			return true // escaping or unprovable destination — keep the deny
		}
	}
	return false // every destination provably in-tree/scratchpad — the raw match was a false positive
}

// outOfTreeWriteTargets extracts the write DESTINATIONS a shell command names,
// across every channel the four rules cover: -o / --output / -out flag operands
// (verb-generic, so `curl -o` is included — repoguard's build-verb-gated extractor
// would miss it), redirect (> / >>) targets, and the destination operands of the
// copy-shaped verbs. Read SOURCES (a `..` source of cp/mv) are deliberately NOT
// returned — they are not writes. Quote-aware and sh -c / $() / “ “ -unwrapping
// via the shared rce tokenizer; redirect targets are captured by an ADDITIVE local
// scan so the shared tokenizer stays byte-stable for rm_rf/rce_pipe.
func outOfTreeWriteTargets(cmd string) []string {
	var out []string
	for _, src := range rceShellSources(cmd) {
		for _, seg := range rceShellSegments(src) {
			out = append(out, outOfTreeSegmentWriteTargets(seg.argv)...)
		}
		out = append(out, redirectWriteTargets(src)...)
	}
	return out
}

// outOfTreeSegmentWriteTargets returns the write destinations named within one command
// segment's argv: any -o/--output/-out flag operand (verb-generic), plus the
// destination(s) of a copy-shaped verb. Named distinctly from command_writes.go's
// segmentWriteTargets(string) — that sibling extractor drives the self-modify floor
// over shellWords-tokenized segments; this one drives the out-of-tree write family
// over rce-tokenized argv, so the two coexist in the package without shadowing.
func outOfTreeSegmentWriteTargets(argv []string) []string {
	i := rceCommandWord(argv)
	if i < 0 {
		return nil
	}
	verb := rceProgramBasename(argv[i])
	rest := argv[i+1:]

	var out []string
	out = append(out, flagOutputTargets(rest)...)

	switch {
	case verb == "tee":
		// tee writes to EVERY file operand (last-operand alone would miss the rest).
		out = append(out, nonFlagOperands(rest)...)
	case ootDestVerbs[verb]:
		// A -t/--target-directory value is a destination too (it inverts last-operand);
		// include both so an inverted form cannot launder past a contained last operand.
		out = append(out, targetDirValues(rest)...)
		if d := lastOperand(rest); d != "" {
			out = append(out, d)
		}
	}
	return out
}

// flagOutputTargets returns the operands of -o / --output / -out / --out output
// flags (both space- and `=`-joined). Verb-generic by design.
func flagOutputTargets(rest []string) []string {
	var out []string
	for j := 0; j < len(rest); j++ {
		t := rest[j]
		switch {
		case t == "-o" || t == "--output" || t == "-out" || t == "--out":
			if j+1 < len(rest) {
				out = append(out, rest[j+1])
				j++
			}
		case strings.HasPrefix(t, "--output="):
			out = append(out, t[len("--output="):])
		case strings.HasPrefix(t, "--out="):
			out = append(out, t[len("--out="):])
		case strings.HasPrefix(t, "-o="):
			out = append(out, t[len("-o="):])
		}
	}
	return out
}

// targetDirValues returns the value of a -t / --target-directory flag, if present.
func targetDirValues(rest []string) []string {
	var out []string
	for j := 0; j < len(rest); j++ {
		t := rest[j]
		switch {
		case t == "-t" || t == "--target-directory":
			if j+1 < len(rest) {
				out = append(out, rest[j+1])
				j++
			}
		case strings.HasPrefix(t, "--target-directory="):
			out = append(out, t[len("--target-directory="):])
		}
	}
	return out
}

// lastOperand returns the last non-flag operand of a copy-shaped verb's args (its
// destination), skipping a trailing option that consumes no separate value in the
// shipped set. A bare `--` ends option scanning.
func lastOperand(rest []string) string {
	ops := nonFlagOperands(rest)
	if len(ops) == 0 {
		return ""
	}
	return ops[len(ops)-1]
}

// nonFlagOperands returns the operands (non-flag tokens) of an argv tail, honouring
// a `--` end-of-options marker. It is intentionally simple: a flag that takes a
// separate value (e.g. `-t DIR`) leaves the value looking like an operand, which is
// the SAFE direction here (an extra candidate destination can only add a deny, and
// -t values are captured explicitly by targetDirValues regardless).
func nonFlagOperands(rest []string) []string {
	var out []string
	endOpts := false
	for _, t := range rest {
		if !endOpts && t == "--" {
			endOpts = true
			continue
		}
		if endOpts || !strings.HasPrefix(t, "-") {
			out = append(out, t)
		}
	}
	return out
}

// redirectWriteTargets scans a raw command source for `>` / `>>` (and the
// fd-prefixed `2>` / `&>` spellings, all of which the `>>?\s*\.\.` regex can match)
// and returns each redirect's target token. Quote-aware so a `>` inside a quoted
// word is not mistaken for a redirection. Additive: it does not touch the shared
// rce tokenizer (which drops `>`), so rm_rf/rce_pipe stay byte-stable.
func redirectWriteTargets(src string) []string {
	var out []string
	var quote byte
	for i := 0; i < len(src); i++ {
		ch := src[i]
		if quote != 0 {
			if ch == '\\' && quote == '"' && i+1 < len(src) {
				i++
				continue
			}
			if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\\':
			i++ // skip the escaped char
		case '\'', '"':
			quote = ch
		case '>':
			// consume an optional second '>' (append) then read the target token.
			j := i + 1
			if j < len(src) && src[j] == '>' {
				j++
			}
			for j < len(src) && (src[j] == ' ' || src[j] == '\t') {
				j++
			}
			if tok, end := readRedirectToken(src, j); tok != "" {
				out = append(out, tok)
				i = end - 1
			} else {
				i = j - 1
			}
		}
	}
	return out
}

// readRedirectToken reads one shell token starting at i, stopping at whitespace or
// a shell metacharacter; it unwraps a single layer of surrounding quotes. Returns
// the token and the index just past it. A token that starts with a metacharacter
// (another redirection, a pipe) is not a filename and yields "".
func readRedirectToken(src string, i int) (string, int) {
	if i >= len(src) {
		return "", i
	}
	switch src[i] {
	case '|', '&', ';', '<', '>', '(', ')':
		return "", i
	}
	var b strings.Builder
	var quote byte
	for ; i < len(src); i++ {
		ch := src[i]
		if quote != 0 {
			if ch == quote {
				quote = 0
				continue
			}
			b.WriteByte(ch)
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case ' ', '\t', '\r', '\n', '|', '&', ';', '<', '>', '(', ')':
			return b.String(), i
		case '\\':
			if i+1 < len(src) {
				i++
				b.WriteByte(src[i])
			}
		default:
			b.WriteByte(ch)
		}
	}
	return b.String(), i
}

// targetContained reports whether a canonicalised write target resolves strictly
// under ws or one of scratch (or is a null device). Fail-closed: an empty target,
// an unresolved variable/glob ($ * ?), or an empty ws for a relative target all
// return false (not contained -> the caller keeps the deny). Absolute targets are
// checked directly; relative targets are joined to ws.
func targetContained(target, ws string, scratch []string) bool {
	if target == "" || strings.ContainsAny(target, "$*?") {
		return false
	}
	var abs string
	if isAbsPath(target) {
		abs = cleanRooted(target)
	} else {
		if ws == "" {
			return false // never resolve a relative target against an unknown root
		}
		abs = cleanRooted(ws + "/" + target)
	}
	if isNullDevice(abs) {
		return true
	}
	if ws != "" && isUnder(abs, ws) {
		return true
	}
	for _, r := range scratch {
		if isUnder(abs, r) {
			return true
		}
	}
	return false
}

// isAbsPath reports whether p (already `\`->`/` folded by canonicalisation) is an
// absolute path on either platform: a leading slash (POSIX / UNC) or a Windows
// drive letter.
func isAbsPath(p string) bool {
	if p == "" {
		return false
	}
	if p[0] == '/' {
		return true
	}
	if len(p) >= 2 && p[1] == ':' &&
		((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) {
		return true
	}
	return false
}

// cleanRooted path.Cleans an absolute path while PRESERVING its root, so a `..`
// can never pop above it. path.Clean alone treats a Windows drive letter ("C:") as
// an ordinary poppable segment — so `C:/a/../../../x` would clean to "x" and lose
// the drive — and its POSIX-root clamping does not apply to a drive path. This
// clamps `..` at the root the way a real filesystem does (root's parent is root):
// the drive (or leading slash) is split off, the remainder is cleaned as an
// absolute path, then the root is re-attached.
func cleanRooted(p string) string {
	if len(p) >= 2 && p[1] == ':' &&
		((p[0] >= 'A' && p[0] <= 'Z') || (p[0] >= 'a' && p[0] <= 'z')) {
		drive := p[:2]
		rest := strings.TrimPrefix(p[2:], "/")
		return drive + path.Clean("/"+rest)
	}
	return path.Clean(p)
}

// isUnder reports whether the absolute path p is root itself or strictly inside it.
// Comparison is case-insensitive (Windows paths are), and the trailing-separator
// boundary prevents a sibling-prefix escape (…/fak vs …/fak-evil).
func isUnder(p, root string) bool {
	if root == "" {
		return false
	}
	p = strings.TrimRight(p, "/")
	root = strings.TrimRight(root, "/")
	if strings.EqualFold(p, root) {
		return true
	}
	return strings.HasPrefix(strings.ToLower(p), strings.ToLower(root)+"/")
}

// isNullDevice recognises the two sinks a write may legitimately name that are not
// under any root: POSIX /dev/null and Windows NUL.
func isNullDevice(abs string) bool {
	if strings.EqualFold(abs, "/dev/null") {
		return true
	}
	base := abs
	if k := strings.LastIndexByte(base, '/'); k >= 0 {
		base = base[k+1:]
	}
	return strings.EqualFold(base, "nul")
}

// --- workspace / scratchpad root resolution (the thin, impure wrapper) ----------
//
// The adjudicator runs in the GUARD PARENT process (it decides gateway-stream tool
// calls), so the scratchpad signal must live in the PARENT env, not the child's:
// FAK_GUARD_SCRATCHPAD_ROOTS is an OS-path-list of absolute scratchpad roots the
// guard sets at startup. The workspace root is the guard's own working directory,
// resolved once. Both are canonicalised to the `/`-folded form targetContained
// compares against. Tests exercise the PURE outOfTreeWriteEscapes with injected
// roots, so this wrapper carries no security logic of its own.

var (
	ootRootOnce  sync.Once
	ootWorkspace string
)

func outOfTreeRoots() (string, []string) {
	ootRootOnce.Do(func() {
		if wd, err := os.Getwd(); err == nil {
			if c, ok := canonicalizeArgValue(wd); ok {
				ootWorkspace = c
			}
		}
	})
	return ootWorkspace, scratchpadRoots()
}

// scratchpadRoots parses FAK_GUARD_SCRATCHPAD_ROOTS (OS path-list separated) into
// canonicalised absolute roots. Empty/unset yields nil (no scratchpad carve-out —
// only in-tree writes are then un-blocked, which is the safe default).
func scratchpadRoots() []string {
	raw := strings.TrimSpace(os.Getenv("FAK_GUARD_SCRATCHPAD_ROOTS"))
	if raw == "" {
		return nil
	}
	sep := string(os.PathListSeparator)
	var out []string
	for _, p := range strings.Split(raw, sep) {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		if c, ok := canonicalizeArgValue(p); ok && c != "" {
			out = append(out, c)
		}
	}
	return out
}
