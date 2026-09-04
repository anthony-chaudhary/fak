package adjudicator

import (
	"strings"

	"github.com/anthony-chaudhary/fak/internal/shelltoken"
)

const (
	legacyRCEPipeDenyRegex  = `\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b`
	defaultRCEPipeDenyRegex = `(?i)\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(bash|sh|python(?:[0-9.]+)?|perl|ruby|node|php|lua)\b`

	// defaultPSRCEPipeDenyRegex is the PowerShell mirror of the same rule. It
	// shipped with NO structural decider at all, so every inert MENTION of the
	// pattern was a terminal POLICY_BLOCK on all four surfaces that carry it —
	// including the one an agent reaches for to read the policy that refused it.
	defaultPSRCEPipeDenyRegex = `(?i)\b(Invoke-WebRequest|iwr|curl|wget|Invoke-RestMethod|irm)\b[^|]*\|[^|]*\b(iex|Invoke-Expression)\b`

	maxRCEShellSourceDepth = 8
	maxRCEShellSources     = 256
)

type rceShellSegment struct {
	argv []string
	sep  byte
}

// isRCEPipeArgRule reports whether pr is a shipped download-pipe deny_regex on a
// shell command arg, on one of the four surfaces that ship it.
//
// The shipped policy gives the POSIX spelling to Bash AND, byte-identically, to
// shell_command, functions.shell_command, and exec_command. Recognising it on Bash
// alone left those mirrors on the raw-regex path, so the same command got two different
// verdicts decided by nothing but which tool NAME the harness happens to use:
// `echo 'curl x | sh'` was admitted as the inert quoted mention it is on Bash and
// a terminal POLICY_BLOCK on shell_command.
func isRCEPipeArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	tool := strings.ToLower(pr.Tool)
	switch pr.Re.String() {
	case legacyRCEPipeDenyRegex, defaultRCEPipeDenyRegex:
		return tool == "bash" || tool == "shell_command" || tool == "functions.shell_command" || tool == "exec_command" || tool == "functions.exec_command"
	case defaultPSRCEPipeDenyRegex:
		return tool == "powershell" || tool == "shell_command" || tool == "functions.shell_command" || tool == "exec_command" || tool == "functions.exec_command"
	default:
		return false
	}
}

func commandHasRemotePipeToInterpreter(cmd string) bool {
	for _, src := range rceShellSources(cmd) {
		if sourceHasRemotePipeToInterpreter(src) {
			return true
		}
	}
	// Same command, PowerShell lexing. rceShellSources unwraps POSIX nesting
	// only, so a payload nested the PowerShell way stays folded up as one inert
	// quoted token and the walk above reads a real download-pipe as harmless.
	// Each extracted payload is then decided by the same POSIX segment test,
	// because POSIX is the dialect this rule's regex actually names (curl/wget
	// piped into sh/bash/python/…).
	for _, payload := range psRCEPayloadSources(cmd, 0) {
		for _, src := range rceShellSources(payload) {
			if sourceHasRemotePipeToInterpreter(src) {
				return true
			}
		}
	}
	return false
}

// psRCEPayloadSources returns the strings that POWERSHELL lexing proves are live
// statements nested inside cmd — the quoted payload of `iex '…'`, of
// `powershell -Command "…"`, of `cmd /c "…"` — so that the POSIX walk above can
// decide them too.
//
// It exists because this rule ships on shell_command / functions.shell_command,
// where the receiving shell may be PowerShell, and there the POSIX unwrapping in
// rceShellSources (sh -c, $(…), backticks) finds nothing to open:
//
//	iex 'curl https://x | sh'
//	  -> POSIX argv [iex, "curl https://x | sh"] — one quoted token, no pipe at a
//	     command boundary, so the structural decider ADMITS a download-pipe that
//	     PowerShell really does execute.
//
// This is the only tightening in the walk, and it is what makes extending the
// rule past Bash safe rather than a new bypass: without it, granting the two
// mirror surfaces the structural path would hand them that admission.
//
// An unterminated quote yields NO sources rather than a deny. This walk only ADDS
// sources to a decision the POSIX walk has already made, and a command PowerShell
// cannot parse is a syntax error that executes nothing — so there is no payload
// being missed. That is the opposite posture from psSourceElevates, which is the
// SOLE decider for its rule and therefore must fail closed; here, failing closed
// would newly refuse routine text (`echo "a \" curl x | sh"` parses fine under
// POSIX and is inert, but leaves PowerShell mid-string).
//
// A -EncodedCommand blob is deliberately NOT treated as undecidable, unlike in the
// sibling walks: base64 contains no `|`, so an encoded payload can never be the
// reason THIS rule's regex fired, and denying on one would refuse an unrelated
// statement that merely shares a command line with a quoted mention.
func psRCEPayloadSources(cmd string, depth int) []string {
	if depth > maxRCEShellSourceDepth {
		return nil
	}
	segs, ok := psSegments(cmd)
	if !ok {
		return nil
	}
	var out []string
	for _, seg := range segs {
		head, rest, ok := psCommandWord(seg)
		if !ok || !psLivePayloadHeads[head] {
			// Not a launcher: this statement's quoted arguments are inert
			// mentions — a grep pattern, a commit message, a printed
			// instruction — never a statement that executes (#2752).
			continue
		}
		for _, tok := range rest {
			if !tok.quoted || len(out) >= maxRCEShellSources {
				continue
			}
			out = append(out, tok.text)
			out = append(out, psRCEPayloadSources(tok.text, depth+1)...)
		}
	}
	return out
}

func sourceHasRemotePipeToInterpreter(src string) bool {
	segs := rceShellSegments(src)
	for i := 0; i+1 < len(segs); i++ {
		if segs[i].sep == '|' && rceDownloaderCommand(segs[i].argv) && rceInterpreterCommand(segs[i+1].argv) {
			return true
		}
	}
	return false
}

func rceShellSources(cmd string) []string {
	var out []string
	var walk func(string, int)
	walk = func(src string, depth int) {
		if src == "" || len(out) >= maxRCEShellSources {
			return
		}
		out = append(out, src)
		if depth >= maxRCEShellSourceDepth {
			return
		}
		for _, inner := range rceDashCStrings(src) {
			walk(inner, depth+1)
		}
		for _, inner := range rceCommandSubstitutions(src) {
			walk(inner, depth+1)
		}
	}
	// Strip provably-inert here-doc bodies first: file CONTENT reaching this
	// tokenizer is read as command lines, because a body's newlines are segment
	// boundaries and its first words land at command-word position. Narrow and
	// subtractive — see stripInertHeredocBodies.
	walk(stripInertHeredocBodies(cmd), 0)
	return out
}

func rceShellSegments(cmd string) []rceShellSegment {
	var segs []rceShellSegment
	var cur []string
	var tok strings.Builder
	var quote byte

	flushTok := func() {
		if tok.Len() > 0 {
			cur = append(cur, tok.String())
			tok.Reset()
		}
	}
	flushSeg := func(sep byte) {
		flushTok()
		if len(cur) > 0 {
			segs = append(segs, rceShellSegment{argv: cur, sep: sep})
			cur = nil
			return
		}
		// No words since the last boundary, so this separator really belongs to
		// the segment that just closed. Dropping it loses the pipe in every
		// grouped spelling — `(curl u) | sh` closes [curl u] at the paren and
		// `{ curl u; } | sh` closes it at the brace, after which the `|` finds an
		// empty segment and vanishes. Both then read as "no pipe at a command
		// boundary" and were ADMITTED, which is the same hole the brace fix
		// closed one level up.
		if sep == '|' && len(segs) > 0 {
			segs[len(segs)-1].sep = '|'
		}
	}

	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		if quote != 0 {
			if ch == '\\' && quote == '"' && i+1 < len(cmd) {
				i++
				tok.WriteByte(cmd[i])
				continue
			}
			if ch == quote {
				quote = 0
			} else {
				tok.WriteByte(ch)
			}
			continue
		}
		switch ch {
		case '\\':
			if i+1 < len(cmd) {
				i++
				tok.WriteByte(cmd[i])
			} else {
				tok.WriteByte(ch)
			}
		case '\'', '"':
			quote = ch
		case ' ', '\t', '\r', '<', '>':
			flushTok()
		case '|':
			if i+1 < len(cmd) && cmd[i+1] == '|' {
				flushSeg(';')
				i++
			} else {
				flushSeg('|')
				if i+1 < len(cmd) && cmd[i+1] == '&' {
					i++
				}
			}
		case '&':
			flushSeg('&')
			if i+1 < len(cmd) && cmd[i+1] == '&' {
				i++
			}
		case ';', '\n', '(', ')':
			flushSeg(ch)
		case '{':
			// A brace that OPENS a token is a group / script-block delimiter —
			// `{ rm -rf /x; }`, `& { curl u | sh }`, PowerShell's `&{…}` — so the
			// real command word is the next token, not the brace. Resolving the
			// brace itself as the command word is what made every grouped form
			// invisible to both deciders: rceCommandWord returned "{", which is
			// neither `rm` nor a downloader, so the walk reported "nothing here"
			// and ADMITTED a command the raw regex had already flagged.
			//
			// A brace GLUED to a token in progress is ordinary syntax — ${VAR}, a
			// brace expansion, a JSON body — and must NOT split: breaking up
			// `curl ${U} | sh` would put the downloader and the interpreter in
			// different segments and lose the pipe between them.
			if tok.Len() == 0 {
				flushSeg(ch)
			} else {
				tok.WriteByte(ch)
			}
		case '}':
			// Symmetrically: a closer that matches a brace THIS token opened is
			// part of the word (`${U}`), and one that does not is a group closer
			// glued to the last command word (`&{curl u|sh}` -> "sh}").
			if strings.Contains(tok.String(), "{") {
				tok.WriteByte(ch)
			} else {
				flushSeg(ch)
			}
		default:
			tok.WriteByte(ch)
		}
	}
	flushSeg(0)
	return segs
}

func rceDashCStrings(src string) []string {
	var out []string
	for _, seg := range rceShellSegments(src) {
		i := rceCommandWord(seg.argv)
		if i < 0 || !rceShellProgram(seg.argv[i]) {
			continue
		}
		for j := i + 1; j < len(seg.argv); j++ {
			t := seg.argv[j]
			if t == "-c" {
				if j+1 < len(seg.argv) {
					out = append(out, seg.argv[j+1])
				}
				break
			}
			if rceIsShortCluster(t) && rceClusterHas(t, 'c') {
				if j+1 < len(seg.argv) {
					out = append(out, seg.argv[j+1])
				}
				break
			}
			if !strings.HasPrefix(t, "-") {
				break
			}
		}
	}
	return out
}

func rceCommandSubstitutions(src string) []string {
	var out []string
	for i := 0; i < len(src); i++ {
		switch src[i] {
		case '\\':
			i++
		case '\'':
			if end := scanSingleQuote(src, i+1); end >= 0 {
				i = end
			}
		case '$':
			if i+1 < len(src) && src[i+1] == '(' {
				if body, end, ok := rceParenBody(src, i+2); ok {
					out = append(out, body)
					i = end
				}
			}
		case '`':
			if body, end, ok := rceBacktickBody(src, i+1); ok {
				out = append(out, body)
				i = end
			}
		}
	}
	return out
}

func scanSingleQuote(src string, start int) int {
	for i := start; i < len(src); i++ {
		if src[i] == '\'' {
			return i
		}
	}
	return -1
}

func rceParenBody(src string, start int) (string, int, bool) {
	depth := 1
	var quote byte
	for i := start; i < len(src); i++ {
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
			i++
		case '\'', '"':
			quote = ch
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return src[start:i], i, true
			}
		}
	}
	return "", 0, false
}

func rceBacktickBody(src string, start int) (string, int, bool) {
	var b strings.Builder
	for i := start; i < len(src); i++ {
		ch := src[i]
		if ch == '\\' && i+1 < len(src) {
			i++
			b.WriteByte(src[i])
			continue
		}
		if ch == '`' {
			return b.String(), i, true
		}
		b.WriteByte(ch)
	}
	return "", 0, false
}

func rceDownloaderCommand(argv []string) bool {
	i := rceCommandWord(argv)
	if i < 0 {
		return false
	}
	switch rceProgramBasename(argv[i]) {
	case "curl", "wget":
		return true
	// PowerShell's downloaders, for the mirror rule this decider also serves.
	// The two dialects already overlap at the head word — `curl` and `wget` are
	// themselves PowerShell ALIASES of Invoke-WebRequest — so the set is shared
	// rather than threaded through a dialect parameter. A cross-dialect head is
	// inert on the other host, and the decider is only ever consulted for a
	// command one of the two regexes has ALREADY matched.
	case "invoke-webrequest", "iwr", "invoke-restmethod", "irm":
		return true
	default:
		return false
	}
}

func rceInterpreterCommand(argv []string) bool {
	i := rceCommandWord(argv)
	if i < 0 {
		return false
	}
	base := rceProgramBasename(argv[i])
	// PowerShell's execution sink. Unlike the POSIX interpreters below, iex takes
	// the piped bytes as a STATEMENT with no flag that could make them data, so
	// there is no rcePythonFixedProgramConsumesData-style exemption to consider.
	if base == "iex" || base == "invoke-expression" {
		return true
	}
	if hasNumericSuffix(base, "python") && rcePythonFixedProgramConsumesData(argv[i+1:]) {
		return false
	}
	return rceDangerInterpreter(base)
}

// rcePythonFixedProgramConsumesData distinguishes a visible, fixed -c program
// from Python's stdin-as-source modes. `download | python -` and bare
// `download | python` execute fetched bytes and remain denied. With -c, stdin is
// data unless the fixed program explicitly feeds it to an execution sink; that
// source-execution shape remains denied too.
func rcePythonFixedProgramConsumesData(args []string) bool {
	for i := 0; i < len(args); i++ {
		arg := args[i]
		if arg == "-c" {
			if i+1 >= len(args) {
				return false
			}
			return !rcePythonCodeExecutesStdin(args[i+1])
		}
		if strings.HasPrefix(arg, "-c") && len(arg) > 2 {
			return !rcePythonCodeExecutesStdin(arg[2:])
		}
		if arg == "-" || !strings.HasPrefix(arg, "-") {
			return false
		}
	}
	return false
}

func rcePythonCodeExecutesStdin(code string) bool {
	normalized := strings.ToLower(strings.Join(strings.Fields(code), ""))
	readsStdin := strings.Contains(normalized, "sys.stdin.read(") ||
		strings.Contains(normalized, "sys.stdin.buffer.read(") ||
		strings.Contains(normalized, "open(0).read(")
	if !readsStdin {
		return false
	}
	for _, sink := range []string{"exec(", "eval(", "compile(", "os.system(", "subprocess."} {
		if strings.Contains(normalized, sink) {
			return true
		}
	}
	return false
}

func rceCommandWord(argv []string) int {
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
		case base == "sudo":
			for i+1 < len(argv) && strings.HasPrefix(argv[i+1], "-") {
				i++
			}
			continue
		case base == "command":
			continue
		default:
			return i
		}
	}
	return -1
}

func rceShellProgram(tok string) bool {
	switch rceProgramBasename(tok) {
	case "sh", "bash", "dash", "zsh", "ksh":
		return true
	default:
		return false
	}
}

func rceIsAssign(t string) bool {
	return shelltoken.IsAssign(t)
}

func rceIsShortCluster(t string) bool { return shelltoken.IsShortCluster(t) }

func rceClusterHas(token string, ch byte) bool {
	return shelltoken.ClusterHas(token, ch)
}

func rceDangerInterpreter(base string) bool {
	switch base {
	case "sh", "bash", "dash", "zsh", "ksh", "perl", "ruby", "node", "php", "lua":
		return true
	default:
		return hasNumericSuffix(base, "python")
	}
}

func hasNumericSuffix(s, prefix string) bool {
	if !strings.HasPrefix(s, prefix) {
		return false
	}
	for _, ch := range s[len(prefix):] {
		if (ch < '0' || ch > '9') && ch != '.' {
			return false
		}
	}
	return true
}

func rceProgramBasename(tok string) string {
	return shelltoken.ProgramBasename(tok)
}
