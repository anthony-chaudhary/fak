package gitgate

// shellparse.go — the SHELL-GRAMMAR seam of gitgate, split out of gitgate.go (#5849) so that
// file stays under the god-file ceiling. Nothing here knows what git is: it extracts the
// command string from a tool call's args, segments it on the shell operators, strips quoted
// heredoc bodies, and unwraps the `$(...)` / backtick / `bash -c` sources a git call can hide
// inside. The git hazard table and the rung's Classify path stay in gitgate.go.

import (
	"context"
	"encoding/json"
	"strings"

	"github.com/anthony-chaudhary/fak/internal/abi"
)

// shellCommand extracts the shell command string from a tool call's args,
// resolving the args Ref the same way the monitor does and reading the `command`
// then `cmd` scalar key (the two conventions across shell tools). Returns "" when
// there is no command arg — the not-a-shell-call case the rung Defers on.
func shellCommand(ctx context.Context, c *abi.ToolCall) string {
	b := refBytes(ctx, c.Args)
	if len(b) == 0 {
		return ""
	}
	var m map[string]any
	if json.Unmarshal(b, &m) != nil {
		return ""
	}
	if s, ok := m["command"].(string); ok {
		return s
	}
	if s, ok := m["cmd"].(string); ok {
		return s
	}
	return ""
}

// refBytes materializes a Ref's bytes (inline directly, otherwise via the active
// resolver), mirroring internal/adjudicator's decodeArgs read path.
func refBytes(ctx context.Context, r abi.Ref) []byte {
	if r.Kind == abi.RefInline {
		return r.Inline
	}
	if res := abi.ActiveResolver(); res != nil {
		if b, err := res.Resolve(ctx, r); err == nil {
			return b
		}
	}
	return nil
}

// tokenizeSegments splits a shell command into segments at unquoted command
// separators (`;` `|` `&` newline and the subshell-grouping parens `(` `)` — the
// `&&` / `||` chains end in `&` / `|`, so a doubled separator just yields an empty
// segment that is dropped), and tokenizes each segment into words at unquoted
// whitespace and redirection operators (`<` `>`), with surrounding single/double
// quotes stripped. It is a deliberately small shell-ish lexer, NOT a shell parser:
// it does not interpret backslash escapes, `$(...)`/backtick substitution, or
// variable expansion. Those launder a git op past this floor and remain the git
// hooks' job (documented non-goal). Quote stripping is what keeps a flag mentioned
// INSIDE a quoted operand — `git commit -m "always use git push --force"` — from
// being read as a flag: the message is one de-quoted operand token, not `--force`.
func tokenizeSegments(cmd string) [][]string {
	var segs [][]string
	var cur []string
	var tok strings.Builder
	var quote byte // 0, '\'' or '"'

	flushTok := func() {
		if tok.Len() > 0 {
			cur = append(cur, tok.String())
			tok.Reset()
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
		if quote != 0 {
			if ch == quote {
				quote = 0
			} else {
				tok.WriteByte(ch)
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case ' ', '\t', '\r', '<', '>':
			flushTok()
		case ';', '|', '&', '\n', '(', ')':
			flushSeg()
		default:
			tok.WriteByte(ch)
		}
	}
	flushSeg()
	return segs
}

// maxUnwrapDepth bounds the recursion of unwrapShellSources so a pathological input
// (deeply nested `$( $( $( ... )))` or a `bash -c` of a `bash -c` of a ...) cannot make
// the pure decide path blow the stack. 8 levels is far past any real laundering chain.
const maxUnwrapDepth = 8

// maxUnwrapSources bounds the TOTAL number of command strings the unwrap pass yields, so
// a string packed with many substitutions cannot turn one classify() into unbounded work.
const maxUnwrapSources = 256

// unwrapShellSources returns cmd PLUS every command string the shell grammar wraps around a
// git call that the flat tokenizer (tokenizeSegments) cannot see on its own: the body of a
// `$(...)` / backtick command substitution, and the `-c` string of a recognized `bash -c`
// / `sh -c` sub-shell — recursively, so a git call nested inside a `$()` inside a `bash -c`
// is recovered. Pipes / `&&` / `||` / `;` / newline already segment correctly inside
// tokenizeSegments, so they need no extra source here; the recursion only adds the
// substitution bodies and sub-shell strings.
//
// It is the recovery half of the documented honest boundary: it makes pipes, operators,
// command substitution, and `-c` strings VISIBLE to the existing rules. It deliberately
// does NOT resolve EXPANSION — `$VAR`, `alias`, and `eval` require runtime state (the
// variable's value, the alias table, the eval result) a static pre-call pass does not have,
// so `git $CMD --force` is unrecoverable here and DEGRADES to defer/opaque (never to allow):
// we simply cannot see a git call we cannot reconstruct, and the git-hooks floor +
// internal/witness remain the backstop. A malformed / unbalanced / over-deep input yields
// only the sources we could safely extract — it never silently drops cmd itself, so the
// flat-tokenizer floor is preserved as a strict subset of this pass.
// stripQuotedHeredocBodies removes expansion-disabled heredoc payloads
// before command substitutions and command segments are inspected. The header
// and terminator remain visible. Unquoted or unterminated heredocs remain
// untouched so expansion-capable or malformed input is handled conservatively.
func stripQuotedHeredocBodies(cmd string) string {
	parseDelimiter := func(line string) (delim string, stripTabs bool, ok bool) {
		var (
			quote    byte
			tok      strings.Builder
			words    []string
			delims   []string
			tabs     []bool
			redirect bool
		)
		flush := func() {
			if tok.Len() > 0 {
				words = append(words, tok.String())
				tok.Reset()
			}
		}
		for i := 0; i < len(line); i++ {
			ch := line[i]
			if quote != 0 {
				if ch == quote {
					quote = 0
				} else {
					tok.WriteByte(ch)
				}
				continue
			}
			switch ch {
			case '\\':
				if i+1 < len(line) {
					i++
					tok.WriteByte(line[i])
				}
			case '\'', '"':
				quote = ch
			case '>':
				redirect = true
				flush()
			case '<':
				if i+1 >= len(line) || line[i+1] != '<' {
					flush() // a plain stdin redirect reads a FILE, not a body
					continue
				}
				// `<<<` is a here-STRING: its value sits on this line, there is no
				// body to strip, and treating it as one would eat real commands.
				if i+2 < len(line) && line[i+2] == '<' {
					return "", false, false
				}
				flush()
				i += 2
				stripTabs := false
				if i < len(line) && line[i] == '-' {
					stripTabs = true
					i++ // <<- strips leading tabs from the body and terminator
				}
				for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
					i++
				}
				if i >= len(line) || (line[i] != '\'' && line[i] != '"') {
					return "", false, false // unquoted heredocs permit live expansion
				}
				q := line[i]
				i++
				var d strings.Builder
				for i < len(line) && line[i] != q {
					d.WriteByte(line[i])
					i++
				}
				if d.Len() == 0 || i >= len(line) {
					return "", false, false
				}
				delims = append(delims, d.String())
				tabs = append(tabs, stripTabs)
			case ' ', '\t', '\r':
				flush()
			case '|', ';', '&', '(', ')', '{', '}', '`', '$':
				// A second statement, a subshell or a substitution on the opener line
				// could route the body somewhere that runs it. Prove nothing.
				return "", false, false
			default:
				tok.WriteByte(ch)
			}
		}
		flush()
		if quote != 0 || len(delims) != 1 || !redirect || len(words) == 0 {
			return "", false, false
		}
		// cat is the whole allow-list on purpose: it never interprets its input, and
		// with stdout redirected the body lands in a file. tee is excluded because it
		// also writes stdout, so `tee f <<'EOF' | sh` would execute the body.
		if words[0] != "cat" {
			return "", false, false
		}
		return delims[0], tabs[0], true
	}
	// A here-doc needs both an opener and a body on a following line.
	if !strings.Contains(cmd, "<<") || !strings.Contains(cmd, "\n") {
		return cmd
	}
	lines := strings.Split(cmd, "\n")
	out := make([]string, 0, len(lines))
	for i := 0; i < len(lines); i++ {
		out = append(out, lines[i])
		delim, stripTabs, ok := parseDelimiter(lines[i])
		if !ok {
			continue
		}
		end := -1
		for j := i + 1; j < len(lines); j++ {
			candidate := strings.TrimSuffix(lines[j], "\r")
			if stripTabs {
				candidate = strings.TrimLeft(candidate, "\t")
			}
			if candidate == delim {
				end = j
				break
			}
		}
		// A malformed opener is not enough evidence to subtract the rest of
		// the command. Leave it visible so classification remains conservative.
		if end < 0 {
			continue
		}
		i = end
	}
	return strings.Join(out, "\n")
}

func unwrapShellSources(cmd string) []string {
	out := make([]string, 0, 4)
	seen := map[string]bool{}
	var walk func(s string, depth int)
	walk = func(s string, depth int) {
		if len(out) >= maxUnwrapSources {
			return
		}
		if s = strings.TrimSpace(s); s == "" || seen[s] {
			return
		}
		seen[s] = true
		out = append(out, s)
		if depth >= maxUnwrapDepth {
			return
		}
		for _, sub := range commandSubstitutions(s) {
			walk(sub, depth+1)
		}
		for _, inner := range dashCStrings(s) {
			walk(inner, depth+1)
		}
	}
	walk(cmd, 0)
	return out
}

// commandSubstitutions extracts the bodies of every UNQUOTED `$()` and backtick
// command substitution in s. A `$()` is paren-depth-tracked so a nested `$(... $(...) ...)`
// yields the OUTER body (the recursion in unwrapShellSources re-extracts the inner one). A
// substitution inside SINGLE quotes is inert in the shell (no expansion happens there), so
// it is skipped; one inside DOUBLE quotes is active, so it is extracted — matching real shell
// semantics, which keeps a `$(...)` mentioned inside a single-quoted commit message from being
// read as a live call. Backticks do not nest, so the first matching backtick closes.
func commandSubstitutions(s string) []string {
	var subs []string
	var quote byte // 0, '\'' or '"' — the surrounding quote context
	for i := 0; i < len(s); i++ {
		ch := s[i]
		if quote != '\'' && ch == '\\' && i+1 < len(s) {
			// Outside single quotes, a backslash makes the next shell byte
			// literal. In particular, \`...\` is a markdown code span in an
			// unquoted payload, not command substitution.
			i++
			continue
		}
		if quote == '\'' {
			// Single quotes are literal: nothing expands, just find the close.
			if ch == '\'' {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'':
			if quote == 0 {
				quote = '\''
			}
		case '"':
			if quote == '"' {
				quote = 0
			} else if quote == 0 {
				quote = '"'
			}
		case '$':
			if i+1 < len(s) && s[i+1] == '(' {
				body, end, ok := extractParenBody(s, i+1)
				if ok {
					subs = append(subs, body)
					i = end // skip past the closing ')'
				}
			}
		case '`':
			if j := strings.IndexByte(s[i+1:], '`'); j >= 0 {
				subs = append(subs, s[i+1:i+1+j])
				i = i + 1 + j // skip past the closing backtick
			}
		}
	}
	return subs
}

// extractParenBody, given s[open]=='(', returns the substring between the balanced parens,
// the index of the matching ')', and whether the parens balanced. Quote-aware so a ')' inside
// a quoted operand does not close the group prematurely. Unbalanced input returns ok=false
// (the laundering degrades to opaque, not to a mis-parsed allow).
func extractParenBody(s string, open int) (body string, end int, ok bool) {
	depth := 0
	var quote byte
	for i := open; i < len(s); i++ {
		ch := s[i]
		if quote != 0 {
			if ch == quote {
				quote = 0
			}
			continue
		}
		switch ch {
		case '\'', '"':
			quote = ch
		case '(':
			depth++
		case ')':
			depth--
			if depth == 0 {
				return s[open+1 : i], i, true
			}
		}
	}
	return "", 0, false
}

// dashCStrings returns the `-c` operand string of every recognized `bash -c <str>` /
// `sh -c <str>` (also `/bin/bash`, `zsh`, `dash`) sub-shell invocation across the command
// SEGMENTS of s. It reuses tokenizeSegments (which de-quotes), so the recovered string is
// the already-unquoted program text the sub-shell would run — fed back through the unwrap
// recursion as its own program. Only the FIRST non-flag operand after `-c` is taken (the
// shell's command string); trailing operands are $0/positional args, not code.
func dashCStrings(s string) []string {
	var inner []string
	for _, seg := range tokenizeSegments(s) {
		i := skipEnvPrefix(seg)
		if i >= len(seg) || !isShellProgram(seg[i]) {
			continue
		}
		for j := i + 1; j < len(seg); j++ {
			t := seg[j]
			if t == "-c" {
				if j+1 < len(seg) {
					inner = append(inner, seg[j+1])
				}
				break
			}
			// A `-c` bundled in a cluster (`-lc`, `-ic`) still introduces the command
			// string as the next operand.
			if isShortCluster(t) && clusterHas(t, 'c') {
				if j+1 < len(seg) {
					inner = append(inner, seg[j+1])
				}
				break
			}
			if !strings.HasPrefix(t, "-") {
				break // a non-flag operand before -c: not a `-c` sub-shell we recognize
			}
		}
	}
	return inner
}

// isShellProgram reports whether a token names a POSIX shell in command position — the
// program whose `-c` operand is a nested program to unwrap. Mirrors isGitProgram's basename
// normalization (path + .exe stripped, lowercased).
func isShellProgram(tok string) bool {
	switch programBasename(tok) {
	case "sh", "bash", "dash", "zsh", "ksh":
		return true
	}
	return false
}
