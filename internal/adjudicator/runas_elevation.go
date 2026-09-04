package adjudicator

import "strings"

// defaultRunAsDenyRegex is the one canonical spelling of the shipped WINDOWS
// privilege-elevation deny_regex. cmd/fak/guard-default-policy.json ships it three
// times — once each for the PowerShell, shell_command, and functions.shell_command
// surfaces — byte-identical. The rule is RECOGNISED by this exact string and then
// decided STRUCTURALLY, exactly like its POSIX twin `\bsudo\b` (sudo_local.go), the
// recursive/forced-delete rule (#4983), and the RCE download-pipe rule (#1465). A
// policy that ships a different spelling is unaffected and keeps the raw-regex path.
//
// The asymmetry this closes (#2343) was called out in sudo_local.go's own doc
// comment: the POSIX escalation word got a command-word decision while "the
// PowerShell escalation rule (Start-Process -Verb RunAs) is a different rule and is
// untouched". So `git commit -m "docs: why sudo is refused"` was admitted while
// `git commit -m "docs: why Start-Process -Verb RunAs is refused"` was a
// POLICY_BLOCK — the same quoted-mention false positive, decided two different ways
// depending on which OS the escalation verb came from.
//
// The sharpest case is self-refuting: the refusal's OWN fix text tells the agent to
// "print the exact command and ask the operator to run it in an elevated shell", and
// printing it (`Write-Output "Start-Process -Verb RunAs pwsh"`) tripped the same
// rule. That is the self-refuting-remedy class this package already closed for
// -WhatIf and `git clean -n` (docs/notes/CONFIRM-GATE-DEADLOCK-2026-07-04.md): the
// guard gated its own sanctioned recovery, so the redirect could never be taken.
const defaultRunAsDenyRegex = `(?i)\bStart-Process\b[^|;\n]*-Verb\s+RunAs\b`

// isRunAsArgRule reports whether pr is the shipped Windows privilege-elevation
// deny_regex on a shell command arg. The rule is scoped to the surfaces that ship it
// — the PowerShell tool and the shell_command / functions.shell_command /
// exec_command mirrors —
// so a differently-spelled or differently-scoped rule keeps the raw-regex path.
func isRunAsArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || (pr.Arg != "command" && pr.Arg != "cmd") {
		return false
	}
	switch strings.ToLower(pr.Tool) {
	case "powershell", "shell_command", "functions.shell_command", "exec_command", "functions.exec_command":
		return pr.Re.String() == defaultRunAsDenyRegex
	default:
		return false
	}
}

// commandInvokesRunAsElevation reports whether cmd actually INVOKES an elevated
// launch — `Start-Process` resolved at a statement's command-word position, carrying
// `-Verb RunAs` in that same statement's argv — as opposed to merely naming the
// phrase inside a quoted string.
//
// It is used SUBTRACTIVELY: decide.go only consults it once the raw regex has
// already matched, and a false result downgrades that match to an admit. So this
// function can never introduce a NEW deny, and every ambiguity is resolved by
// returning true (keep the deny). Unparseable input — an unterminated quote — and an
// undecodable `-EncodedCommand` payload both fail CLOSED for that reason.
//
// A genuine elevation stays denied under every launder the walk unwraps: a later
// statement (`Get-Process; Start-Process -Verb RunAs cmd`), a pipeline stage, the
// call operator (`& 'Start-Process' -Verb RunAs cmd`), a nested host payload
// (`powershell -Command "Start-Process -Verb RunAs cmd"`), and a quoted argument to
// any launcher head (`Start-Process pwsh -ArgumentList '-Command','Start-Process
// -Verb RunAs cmd'`).
func commandInvokesRunAsElevation(cmd string) bool {
	return psSourceElevates(cmd, 0)
}

func psSourceElevates(src string, depth int) bool {
	return psSourceMatches(src, depth, func(head string, rest []psToken) bool {
		return head == "start-process" && psArgvHasRunAsVerb(rest)
	})
}

func psSourceMatches(src string, depth int, matches func(string, []psToken) bool) bool {
	if depth > maxRCEShellSourceDepth {
		return true
	}
	segments, ok := psSegments(src)
	if !ok {
		return true
	}
	for _, segment := range segments {
		head, rest, ok := psCommandWord(segment)
		if !ok {
			continue
		}
		if matches(head, rest) {
			return true
		}
		if !psLivePayloadHeads[head] {
			continue
		}
		for _, token := range rest {
			if psEncodedPayloadFlag(token.text) {
				return true
			}
		}
		for _, token := range rest {
			if token.quoted && psSourceMatches(token.text, depth+1, matches) {
				return true
			}
		}
	}
	return false
}

// psLivePayloadHeads are the command words whose QUOTED argument is a statement
// executed elsewhere rather than inert text, mirroring reversibility.go's
// quotedPayloadLiveHeads. `ssh` is deliberately ABSENT for the same reason
// commandLocalEscalationWord excludes it: a remote elevation is governed by the
// remote host's own controls, not this local floor.
var psLivePayloadHeads = map[string]bool{
	"powershell": true, "pwsh": true, "cmd": true, "start-process": true,
	"saps": true, "start": true, "invoke-expression": true, "iex": true,
	"invoke-command": true, "icm": true, "wsl": true,
	"sh": true, "bash": true, "zsh": true, "dash": true, "ksh": true,
}

// psEncodedPayloadFlag reports whether tok is a PowerShell encoded-command flag.
// PowerShell accepts any unambiguous prefix of -EncodedCommand, so the whole
// prefix family is matched rather than the full spelling alone.
func psEncodedPayloadFlag(tok string) bool {
	t := strings.ToLower(tok)
	if i := strings.IndexByte(t, ':'); i >= 0 {
		t = t[:i]
	}
	if !strings.HasPrefix(t, "-e") {
		return false
	}
	return strings.HasPrefix("encodedcommand", strings.TrimPrefix(t, "-"))
}

// psArgvHasRunAsVerb reports whether a Start-Process argv carries `-Verb RunAs`,
// in either the separated (`-Verb RunAs`) or the colon-bound (`-Verb:RunAs`)
// PowerShell parameter spelling.
func psArgvHasRunAsVerb(argv []psToken) bool {
	for i, tok := range argv {
		t := strings.ToLower(tok.text)
		if t == "-verb" {
			if i+1 < len(argv) && strings.EqualFold(argv[i+1].text, "runas") {
				return true
			}
			continue
		}
		if strings.HasPrefix(t, "-verb:") && strings.TrimPrefix(t, "-verb:") == "runas" {
			return true
		}
	}
	return false
}

// psToken is one PowerShell word plus whether it arrived inside a quoted span.
type psToken struct {
	text   string
	quoted bool
}

// psCommandWord resolves a statement's command word — skipping the call operator
// (`&`) and dot-source (`.`) prefixes — and returns its lowered basename plus the
// remaining argv. A statement whose only content is a single QUOTED token is not a
// command invocation in PowerShell (a bare string expression is echoed, not run),
// but that case needs no special handling: such a token keeps its embedded spaces,
// so its basename never equals a bare program name.
func psCommandWord(seg []psToken) (string, []psToken, bool) {
	for i := 0; i < len(seg); i++ {
		if t := seg[i].text; t == "&" || t == "." {
			continue
		}
		return psProgramBasename(seg[i].text), seg[i+1:], true
	}
	return "", nil, false
}

// psProgramBasename lowers a command word and strips its directory and .exe suffix,
// so `C:\Windows\System32\cmd.exe`, `cmd.exe`, and `cmd` all read as "cmd". Unlike
// rceProgramBasename it treats a backslash as a PATH SEPARATOR, never an escape:
// that is the Windows spelling, and reading `C:\work\fak` as an escaped string is
// exactly the mis-parse that would let a real elevation slip past this walk.
func psProgramBasename(tok string) string {
	t := strings.ToLower(tok)
	if i := strings.LastIndexAny(t, `/\`); i >= 0 {
		t = t[i+1:]
	}
	return strings.TrimSuffix(t, ".exe")
}

// psSegments splits a PowerShell command into statement/pipeline segments of words,
// honoring PowerShell's own lexical rules rather than POSIX's: the escape character
// is the BACKTICK (a backslash is an ordinary path byte), a single-quoted span takes
// no escapes, and `”`/`""` inside a span of the same quote is a literal quote. It
// splits on `;`, `|`, `&&`, `||`, newlines, and parenthesis/brace boundaries so a
// sub-expression is decided as its own statement.
//
// It returns ok=false on an UNTERMINATED quoted span. Callers treat that as
// undecidable and keep the deny: a swallowed closing quote would merge the rest of
// the line into one inert-looking token, which is precisely the shape that would let
// a real elevation read as a quoted mention.
func psSegments(cmd string) ([][]psToken, bool) {
	var (
		segs    [][]psToken
		cur     []psToken
		tok     strings.Builder
		started bool // a token is open (possibly empty, e.g. the string "")
		quoted  bool // the open token contained a quoted span
	)
	flushTok := func() {
		if started {
			cur = append(cur, psToken{text: tok.String(), quoted: quoted})
			tok.Reset()
			started = false
			quoted = false
		}
	}
	flushSeg := func() {
		flushTok()
		if len(cur) > 0 {
			segs = append(segs, cur)
			cur = nil
		}
	}
	for i := 0; i < len(cmd); i++ {
		ch := cmd[i]
		switch ch {
		case '`':
			// PowerShell's escape character: the next byte is literal.
			if i+1 < len(cmd) {
				i++
				tok.WriteByte(cmd[i])
				started = true
			}
		case '\'', '"':
			end, ok := psQuotedSpanEnd(cmd, i)
			if !ok {
				return nil, false
			}
			tok.WriteString(psUnquote(cmd[i:end]))
			started = true
			quoted = true
			i = end - 1
		case ' ', '\t', '\r':
			flushTok()
		case ';', '\n', '(', ')', '{', '}', ',':
			flushSeg()
		case '|':
			flushSeg()
			if i+1 < len(cmd) && cmd[i+1] == '|' {
				i++
			}
		case '&':
			// `&&` separates statements; a lone `&` is the call operator and
			// stays a token so psCommandWord can step past it.
			if i+1 < len(cmd) && cmd[i+1] == '&' {
				flushSeg()
				i++
				continue
			}
			flushTok()
			cur = append(cur, psToken{text: "&"})
		default:
			tok.WriteByte(ch)
			started = true
		}
	}
	flushSeg()
	return segs, true
}

// psQuotedSpanEnd returns the index just past the quoted span opening at cmd[i].
// Inside a double-quoted span a BACKTICK escapes the next byte; inside a
// single-quoted span nothing does. In both, a doubled quote (`”` / `""`) is a
// literal quote and does not close the span. An unterminated span reports ok=false.
func psQuotedSpanEnd(cmd string, i int) (int, bool) {
	q := cmd[i]
	for j := i + 1; j < len(cmd); j++ {
		c := cmd[j]
		if c == '`' && q == '"' && j+1 < len(cmd) {
			j++
			continue
		}
		if c != q {
			continue
		}
		if j+1 < len(cmd) && cmd[j+1] == q {
			j++ // a doubled quote is a literal, not the terminator
			continue
		}
		return j + 1, true
	}
	return 0, false
}

// psUnquote returns a quoted span's CONTENT with the delimiters removed, the
// doubled-quote literal collapsed, and (for a double-quoted span) backtick escapes
// resolved — the view a recursive walk must read, since the payload a launcher
// executes is the unquoted text.
func psUnquote(span string) string {
	if len(span) < 2 {
		return span
	}
	q := span[0]
	body := span[1 : len(span)-1]
	var b strings.Builder
	for i := 0; i < len(body); i++ {
		c := body[i]
		if c == '`' && q == '"' && i+1 < len(body) {
			i++
			b.WriteByte(body[i])
			continue
		}
		if c == q && i+1 < len(body) && body[i+1] == q {
			i++
			b.WriteByte(q)
			continue
		}
		b.WriteByte(c)
	}
	return b.String()
}
