package adjudicator

import (
	"regexp"
	"strings"
	"unicode"
)

var (
	commandSegmentRE = regexp.MustCompile(`\|\||&&|[;|&\r\n]`)
	envAssignmentRE  = regexp.MustCompile(`^[A-Za-z_][A-Za-z0-9_]*=`)
)

var commandWrapperHeads = map[string]bool{
	"sudo":    true,
	"env":     true,
	"nice":    true,
	"time":    true,
	"command": true,
	"xargs":   true,
	"doas":    true,
}

// commandSegments splits a shell command on UNQUOTED sequencing/pipe
// operators and returns each non-empty segment's word list with leading env
// assignments (NAME=value) and wrapper heads (sudo, env, ...) stripped, so
// index 0 is the command actually being invoked. Family matchers anchor here
// instead of scanning the whole token stream, so trigger words inside quoted
// payloads (grep patterns, commit messages) no longer classify — the
// quote-aware split closes the residual #2752 path where an operator byte
// INSIDE a quoted payload manufactured a spurious segment head.
func commandSegments(cmd string) [][]string {
	var segs [][]string
	for _, raw := range quoteAwareSegmentTexts(cmd) {
		text := segmentMentionStrip(raw)
		if quotedPayloadIsLive(raw) {
			text = unquoteSpans(raw)
		}
		fields := strings.Fields(text)
		for len(fields) > 0 &&
			(envAssignmentRE.MatchString(fields[0]) || commandWrapperHeads[strings.ToLower(fields[0])]) {
			fields = fields[1:]
		}
		if len(fields) == 0 {
			continue
		}
		head := strings.ToLower(strings.TrimSuffix(unquoteSpans(fields[0]), ".exe"))
		if i := strings.LastIndexAny(head, `/\`); i >= 0 {
			head = head[i+1:]
		}
		if head == "" {
			continue
		}
		words := []string{head}
		words = append(words, commandWords(strings.Join(fields[1:], " "))...)
		segs = append(segs, words)
	}
	return segs
}

// quotedSpanEnd returns the index just past the quoted span opening at s[i]
// (s[i] is ' or "), honoring backslash escapes inside double quotes and
// inside $'…' (dollar=true for the ANSI-C form; POSIX single quotes take no
// escapes). An unterminated span runs to the end of the string — the shell
// would refuse to run such a command, so hiding its tail fails toward
// reversible on text that cannot execute.
func quotedSpanEnd(s string, i int, dollar bool) int {
	q := s[i]
	j := i + 1
	for j < len(s) {
		c := s[j]
		if c == '\\' && (q == '"' || dollar) && j+1 < len(s) {
			j += 2
			continue
		}
		if c == q {
			return j + 1
		}
		j++
	}
	return j
}

// quoteAwareSegmentTexts splits cmd on unquoted sequencing/pipe operators
// (`;`, `|`, `&`, newlines — `||`/`&&` fall out of the single-byte split) and
// returns each non-blank segment's raw text, quotes intact. A backslash
// escapes the following byte outside quotes, so an unquoted `\|` (a grep
// alternation) does not split either.
func quoteAwareSegmentTexts(cmd string) []string {
	var segs []string
	var b strings.Builder
	flush := func() {
		if strings.TrimSpace(b.String()) != "" {
			segs = append(segs, b.String())
		}
		b.Reset()
	}
	for i := 0; i < len(cmd); {
		c := cmd[i]
		switch {
		case c == '\\' && i+1 < len(cmd):
			b.WriteString(cmd[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && cmd[i-1] == '$'
			end := quotedSpanEnd(cmd, i, dollar)
			b.WriteString(cmd[i:end])
			i = end
		case c == ';' || c == '|' || c == '&' || c == '\n' || c == '\r':
			flush()
			i++
		default:
			b.WriteByte(c)
			i++
		}
	}
	flush()
	return segs
}

// quoteAwareFields splits a segment on unquoted whitespace, keeping quote
// bytes inside each token so a token's quoting is still inspectable after
// the split.
func quoteAwareFields(s string) []string {
	var fields []string
	var b strings.Builder
	flush := func() {
		if b.Len() > 0 {
			fields = append(fields, b.String())
		}
		b.Reset()
	}
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			b.WriteString(s[i:end])
			i = end
		case c == ' ' || c == '\t':
			flush()
			i++
		default:
			b.WriteByte(c)
			i++
		}
	}
	flush()
	return fields
}

// unquoteSpans removes the quote BYTES from every quoted span in s, keeping
// the quoted content in place. Live-payload segments read this view — the
// quoted statement executes, so its text must stay visible with clean word
// boundaries (`bash -c "curl -X POST …"` must still match the curl write
// regex, which anchors on a space before curl, not a quote byte).
func unquoteSpans(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			end := quotedSpanEnd(s, i, dollar)
			stop := end
			if end > i+1 && s[end-1] == c {
				stop = end - 1
			}
			b.WriteString(s[i+1 : stop])
			i = end
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// segmentMentionStrip returns the non-live segment text the matchers read:
// the effective command token (past env assignments and wrapper heads) is
// UNQUOTED — the shell resolves a quoted head ("rm") and runs it exactly
// like an unquoted one, so hiding it would turn quoting the verb into a
// guard bypass — while every other quoted span is blanked as an inert
// mention (#2752).
func segmentMentionStrip(segText string) string {
	toks := quoteAwareFields(segText)
	out := make([]string, 0, len(toks))
	headDone := false
	for _, tok := range toks {
		if !headDone {
			bare := unquoteSpans(tok)
			if envAssignmentRE.MatchString(bare) || commandWrapperHeads[strings.ToLower(bare)] {
				// An env assignment's quoted VALUE is data: blank it, keep the
				// NAME= husk for the callers' existing prefix-strip loop.
				out = append(out, strings.Fields(stripQuotedSpans(tok))...)
				continue
			}
			headDone = true
			out = append(out, strings.Fields(bare)...)
			continue
		}
		out = append(out, strings.Fields(stripQuotedSpans(tok))...)
	}
	return strings.Join(out, " ")
}

// stripQuotedSpans replaces every quoted span in s (single, double, $'…')
// with a single space, leaving the unquoted command skeleton. The payload
// scans read this view so a quoted mention is inert (#2752).
func stripQuotedSpans(s string) string {
	var b strings.Builder
	for i := 0; i < len(s); {
		c := s[i]
		switch {
		case c == '\\' && i+1 < len(s):
			b.WriteString(s[i : i+2])
			i += 2
		case c == '\'' || c == '"':
			dollar := c == '\'' && i > 0 && s[i-1] == '$'
			b.WriteByte(' ')
			i = quotedSpanEnd(s, i, dollar)
		default:
			b.WriteByte(c)
			i++
		}
	}
	return b.String()
}

// quotedPayloadLiveHeads are the client binaries whose QUOTED argument is a
// live statement executed elsewhere — a SQL client's -c/-e payload, a shell's
// -c script, a remote command handed to ssh — never an inert mention. Their
// segments keep quoted contents in the payload-scan view, so
// `psql -c "drop table t"` still escalates while `git commit -m "drop table
// t"` does not (#2752's explicit both-directions rule).
var quotedPayloadLiveHeads = map[string]bool{
	"psql": true, "mysql": true, "mariadb": true, "sqlite3": true, "sqlcmd": true,
	"mongo": true, "mongosh": true, "duckdb": true, "clickhouse-client": true, "redis-cli": true,
	"sh": true, "bash": true, "zsh": true, "dash": true, "ksh": true, "fish": true,
	"pwsh": true, "powershell": true, "cmd": true, "eval": true, "ssh": true,
}

// quotedPayloadIsLive reports whether a raw segment's effective head (env
// assignments and wrapper heads stripped, path and .exe suffix trimmed) is a
// client whose quoted argument executes as a statement.
func quotedPayloadIsLive(segText string) bool {
	fields := quoteAwareFields(segText)
	for len(fields) > 0 {
		bare := unquoteSpans(fields[0])
		if !envAssignmentRE.MatchString(bare) && !commandWrapperHeads[strings.ToLower(bare)] {
			break
		}
		fields = fields[1:]
	}
	if len(fields) == 0 {
		return false
	}
	// A quoted client name still executes: "psql" -c … is as live as psql.
	head := strings.ToLower(unquoteSpans(fields[0]))
	head = strings.TrimSuffix(head, ".exe")
	if k := strings.LastIndexAny(head, `/\`); k >= 0 {
		head = head[k+1:]
	}
	return quotedPayloadLiveHeads[head]
}

// payloadScanView is the whole-command text the payload-shaped scans (curl
// flags, SQL words, cmdContains substrings, the dry-run flag scan) read:
// quoted spans are blanked segment-by-segment so an inert mention cannot
// classify, EXCEPT in segments whose head is a live-payload client, where the
// quoted text is the statement that executes and must stay visible.
func payloadScanView(cmd string) string {
	if cmd == "" {
		return ""
	}
	segs := quoteAwareSegmentTexts(cmd)
	parts := make([]string, 0, len(segs))
	for _, seg := range segs {
		if quotedPayloadIsLive(seg) {
			parts = append(parts, unquoteSpans(seg))
			continue
		}
		parts = append(parts, segmentMentionStrip(seg))
	}
	return strings.Join(parts, " ; ")
}

func segmentHeadIs(segs [][]string, names ...string) bool {
	for _, seg := range segs {
		for _, name := range names {
			if seg[0] == name {
				return true
			}
		}
	}
	return false
}

func segmentHasPrefix(segs [][]string, want ...string) bool {
	for _, seg := range segs {
		if len(seg) < len(want) {
			continue
		}
		match := true
		for i, w := range want {
			if seg[i] != w {
				match = false
				break
			}
		}
		if match {
			return true
		}
	}
	return false
}

func commandWords(cmd string) []string {
	var words []string
	var b strings.Builder
	flush := func() {
		if b.Len() == 0 {
			return
		}
		words = append(words, strings.ToLower(b.String()))
		b.Reset()
	}
	for _, r := range cmd {
		switch {
		case unicode.IsLetter(r) || unicode.IsDigit(r) || r == '_' || r == '-':
			b.WriteRune(unicode.ToLower(r))
		default:
			flush()
		}
	}
	flush()
	for i, w := range words {
		words[i] = strings.TrimLeft(w, "-")
	}
	return words
}

func containsWord(words []string, want string) bool {
	for _, w := range words {
		if w == want {
			return true
		}
	}
	return false
}

func orderedWords(words []string, want ...string) bool {
	pos := 0
	for _, w := range words {
		if w == want[pos] {
			pos++
			if pos == len(want) {
				return true
			}
		}
	}
	return false
}
