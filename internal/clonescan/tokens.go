// Package clonescan is the forward, authoring-time half of fak's clone detector.
//
// The batch scorecard (tools/code_slop_scorecard.py, kpi_duplication) grades the
// whole tracked tree a cycle AFTER code lands, as slop-debt. This package inverts
// that engine into a QUERY: given one candidate Go block, which tracked sites hold
// a token-similar block RIGHT NOW — so an author can ask "does this already exist?"
// BEFORE writing it, instead of reading it as a grade afterward. See
// docs/notes/DEDUP-EARLIER-AND-MORE-OFTEN-2026-07-03.md.
//
// The tokenizer here is a faithful port of the Python `go_tokens` so the early
// query and the late scorecard share ONE definition of a clone: a normalized Go
// token window that is whitespace/comment/line-break invariant and (optionally)
// rename invariant. Keeping the two in lockstep is the point — driving the query's
// warnings to zero drives the scorecard's `dup_extractable` to zero by construction.
package clonescan

// Engine constants — kept bit-identical to code_slop_scorecard.py so a window that
// the query flags is the same window the scorecard would count. Do not drift these
// independently; a divergence means "the author was warned" and "CI counts debt"
// stop agreeing.
const (
	// WindowTokens is the clone window length in normalized tokens (~a 6-line block).
	WindowTokens = 34
	// MinLogicTokens is how many computation/control tokens a window must carry to
	// qualify — data/declaration regions (imports, struct fields, composite literals)
	// score zero logic and are never clones.
	MinLogicTokens = 2
	// MinOccurrences is how many DISTINCT locations a window key must appear at to be
	// a clone. For a forward query the candidate is one location and any tracked hit
	// is a second, so a single tracked match already meets this.
	MinOccurrences = 2
)

// goKeywords are the reserved words kept verbatim in the token stream.
var goKeywords = map[string]bool{
	"break": true, "case": true, "chan": true, "const": true, "continue": true,
	"default": true, "defer": true, "else": true, "fallthrough": true, "for": true,
	"func": true, "go": true, "goto": true, "if": true, "import": true,
	"interface": true, "map": true, "package": true, "range": true, "return": true,
	"select": true, "struct": true, "switch": true, "type": true, "var": true,
}

// logicKeywords denote control flow — the signal a window is copied LOGIC.
var logicKeywords = map[string]bool{
	"if": true, "for": true, "switch": true, "select": true, "range": true,
}

// logicOps are the operators that denote computation or control flow.
var logicOps = map[string]bool{
	"+": true, "-": true, "*": true, "/": true, "%": true, "&": true, "|": true,
	"^": true, "<<": true, ">>": true, "&^": true, "&&": true, "||": true, "!": true,
	"==": true, "!=": true, "<": true, "<=": true, ">": true, ">=": true, "=": true,
	":=": true, "+=": true, "-=": true, "*=": true, "/=": true, "%=": true, "&=": true,
	"|=": true, "^=": true, "<<=": true, ">>=": true, "&^=": true, "++": true, "--": true,
}

// assignOps are the bare assignments — logic, but context-gated: they only count
// toward a window's logic when the same window also carries a non-assignment logic
// token (else it is a pure declaration/field-init block, i.e. data not logic).
var assignOps = map[string]bool{"=": true, ":=": true}

// goOps is the operator/punctuation table, longest-first so a greedy prefix match
// never returns a proper prefix (`<<=` before `<<` before `<`).
var goOps = []string{
	"<<=", ">>=", "&^=", "...",
	"<-", ":=", "&&", "||", "++", "--", "==", "!=", "<=", ">=",
	"+=", "-=", "*=", "/=", "%=", "&=", "|=", "^=", "<<", ">>", "&^",
	"+", "-", "*", "/", "%", "&", "|", "^", "<", ">", "=", "!",
	"(", ")", "[", "]", "{", "}", ",", ";", ".", ":",
}

// token is one normalized lexical unit: its symbol, its 1-based source line, and
// whether it is a logic (computation/control) token.
type token struct {
	sym     string
	line    int
	isLogic bool
}

// goTokens lexes Go source into the normalized token stream. Comments and
// whitespace are dropped; every string/rune/number literal collapses to "L";
// identifiers collapse to "I" when normalizeIdents is true (a clone survives a
// rename), else are kept verbatim; keywords, operators and punctuation are kept.
//
// It is a faithful port of code_slop_scorecard.py's go_tokens: best-effort and
// forgiving — an unterminated literal or a stray byte is consumed without error,
// since neither the scorecard nor the query may crash on odd input.
func goTokens(text string, normalizeIdents bool) []token {
	out := make([]token, 0, len(text)/4)
	b := []byte(text)
	n := len(b)
	line := 1
	i := 0
	for i < n {
		c := b[i]
		if c == '\n' {
			line++
			i++
			continue
		}
		if c == ' ' || c == '\t' || c == '\r' || c == '\f' || c == '\v' {
			i++
			continue
		}
		// line comment
		if c == '/' && i+1 < n && b[i+1] == '/' {
			j := indexByteFrom(b, '\n', i)
			if j == -1 {
				break
			}
			i = j
			continue
		}
		// block comment (may span lines)
		if c == '/' && i+1 < n && b[i+1] == '*' {
			j := indexStringFrom(b, "*/", i+2)
			if j == -1 {
				line += countNewlines(b, i, n)
				break
			}
			line += countNewlines(b, i, j+2)
			i = j + 2
			continue
		}
		// raw string literal (may span lines)
		if c == '`' {
			j := indexByteFrom(b, '`', i+1)
			if j == -1 {
				j = n - 1
			}
			line += countNewlines(b, i, j+1)
			out = append(out, token{"L", line, false})
			i = j + 1
			continue
		}
		// interpreted string / rune literal
		if c == '"' || c == '\'' {
			q := c
			j := i + 1
			for j < n {
				if b[j] == '\\' {
					j += 2
					continue
				}
				if b[j] == '\n' {
					break // unterminated — can't span lines
				}
				if b[j] == q {
					j++
					break
				}
				j++
			}
			out = append(out, token{"L", line, false})
			i = j
			continue
		}
		// numeric literal
		if isDigit(c) || (c == '.' && i+1 < n && isDigit(b[i+1])) {
			j := i + 1
			for j < n && (isAlnum(b[j]) || b[j] == '.' || b[j] == '_') {
				if (b[j] == 'e' || b[j] == 'E' || b[j] == 'p' || b[j] == 'P') &&
					j+1 < n && (b[j+1] == '+' || b[j+1] == '-') {
					j += 2 // exponent sign
					continue
				}
				j++
			}
			out = append(out, token{"L", line, false})
			i = j
			continue
		}
		// identifier or keyword
		if isAlpha(c) || c == '_' {
			j := i + 1
			for j < n && (isAlnum(b[j]) || b[j] == '_') {
				j++
			}
			word := string(b[i:j])
			if goKeywords[word] {
				out = append(out, token{word, line, logicKeywords[word]})
			} else if normalizeIdents {
				out = append(out, token{"I", line, false})
			} else {
				out = append(out, token{word, line, false})
			}
			i = j
			continue
		}
		// operator / punctuation (greedy, longest-first)
		matched := false
		for _, op := range goOps {
			if hasPrefixAt(b, op, i) {
				out = append(out, token{op, line, logicOps[op]})
				i += len(op)
				matched = true
				break
			}
		}
		if !matched {
			i++ // unknown byte (e.g. a stray non-ASCII rune) — skip
		}
	}
	return out
}

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
func isAlpha(c byte) bool {
	return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')
}
func isAlnum(c byte) bool { return isDigit(c) || isAlpha(c) }

func indexByteFrom(b []byte, c byte, from int) int {
	for i := from; i < len(b); i++ {
		if b[i] == c {
			return i
		}
	}
	return -1
}

func indexStringFrom(b []byte, s string, from int) int {
	for i := from; i+len(s) <= len(b); i++ {
		if hasPrefixAt(b, s, i) {
			return i
		}
	}
	return -1
}

func hasPrefixAt(b []byte, s string, at int) bool {
	if at+len(s) > len(b) {
		return false
	}
	for k := 0; k < len(s); k++ {
		if b[at+k] != s[k] {
			return false
		}
	}
	return true
}

func countNewlines(b []byte, from, to int) int {
	if to > len(b) {
		to = len(b)
	}
	n := 0
	for i := from; i < to; i++ {
		if b[i] == '\n' {
			n++
		}
	}
	return n
}
