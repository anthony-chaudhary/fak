package adjudicator

import "strings"

const (
	legacyRCEPipeDenyRegex  = `\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba)?sh\b`
	defaultRCEPipeDenyRegex = `(?i)\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(bash|sh|python(?:[0-9.]+)?|perl|ruby|node|php|lua)\b`

	maxRCEShellSourceDepth = 8
	maxRCEShellSources     = 256
)

type rceShellSegment struct {
	argv []string
	sep  byte
}

func isRCEPipeArgRule(pr *ArgPredicate) bool {
	if pr == nil || pr.Re == nil || !strings.EqualFold(pr.Tool, "Bash") {
		return false
	}
	if pr.Arg != "command" && pr.Arg != "cmd" {
		return false
	}
	switch pr.Re.String() {
	case legacyRCEPipeDenyRegex, defaultRCEPipeDenyRegex:
		return true
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
	return false
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
	walk(cmd, 0)
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
	default:
		return false
	}
}

func rceInterpreterCommand(argv []string) bool {
	i := rceCommandWord(argv)
	if i < 0 {
		return false
	}
	return rceDangerInterpreter(rceProgramBasename(argv[i]))
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
	eq := strings.IndexByte(t, '=')
	if eq <= 0 {
		return false
	}
	for i := 0; i < eq; i++ {
		ch := t[i]
		ok := ch == '_' ||
			(ch >= 'A' && ch <= 'Z') ||
			(ch >= 'a' && ch <= 'z') ||
			(i > 0 && ch >= '0' && ch <= '9')
		if !ok {
			return false
		}
	}
	return true
}

func rceIsShortCluster(t string) bool { return len(t) >= 2 && t[0] == '-' && t[1] != '-' }

func rceClusterHas(token string, ch byte) bool {
	for i := 1; i < len(token); i++ {
		if token[i] == '=' {
			break
		}
		if token[i] == ch {
			return true
		}
	}
	return false
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
	b := tok
	if k := strings.LastIndexAny(b, `/\`); k >= 0 {
		b = b[k+1:]
	}
	return strings.TrimSuffix(strings.ToLower(b), ".exe")
}
