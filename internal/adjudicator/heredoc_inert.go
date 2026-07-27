package adjudicator

import "strings"

// stripInertHeredocBodies removes the BODY of a here-document whose bytes provably
// cannot be executed, so that file CONTENT stops being tokenized as if it were a
// command line.
//
// rceShellSegments splits on newlines, so every line of a here-doc body becomes its
// own segment and its first word lands at the command-word position. Writing a file
// therefore gets read as running its contents:
//
//	cat > notes.txt <<'EOF'
//	Get-ChildItem|Get-Content|Select-Object
//	EOF
//
// segments as [cat notes.txt EOF] [Get-ChildItem] [Get-Content] [Select-Object],
// and the third segment is a PowerShell cmdlet at a command word, so the dialect
// rule fires on a plain `cat` that runs nothing. The same hole is shared by every
// family built on this tokenizer: a here-doc body mentioning a recursive delete or
// a download pipe trips those rules too. Writing a file whose content documents the
// guard's own rules — which is what maintaining this package requires — was
// refused, and under `fak guard -- claude` that refusal reads as an agent-chosen
// end_turn rather than a policy decision.
//
// The carve-out is deliberately narrow, because a here-doc body is NOT inert in
// general: `sh <<'EOF'` and `python3 - <<'EOF'` execute theirs, and `cat <<'EOF' |
// sh` pipes its body into one. Stripping those would be a real bypass. So a body is
// dropped only when its opener line is a lone `cat` whose stdout is redirected: cat
// never interprets its input, the redirect sends those bytes to a file, and the line
// carries no `|`, `;`, `&` or subshell that could route them anywhere else. Any
// other shape — a different command word, no redirect, a second statement on the
// line, more than one here-doc, an unterminated quote — keeps today's behaviour
// untouched.
//
// This is purely SUBTRACTIVE in the same sense as the rest of the package: it only
// removes text from the tokenizer's view, so it can turn a deny into an admit but
// never the reverse, and every shape it cannot prove inert is left alone.
func stripInertHeredocBodies(cmd string) string {
	// A here-doc needs both an opener and a body on a following line.
	if !strings.Contains(cmd, "<<") || !strings.Contains(cmd, "\n") {
		return cmd
	}
	lines := strings.Split(cmd, "\n")
	out := make([]string, 0, len(lines))
	for i := 0; i < len(lines); i++ {
		out = append(out, lines[i])
		delim, ok := inertHeredocDelimiter(lines[i])
		if !ok {
			continue
		}
		// Drop the body, and the delimiter line that closes it. An unclosed
		// here-doc runs to end-of-input in the shell too — bash warns and treats
		// the remainder as body — so consuming the rest is what really happens.
		j := i + 1
		for ; j < len(lines); j++ {
			if strings.TrimSpace(lines[j]) == delim {
				break
			}
		}
		i = j
	}
	return strings.Join(out, "\n")
}

// inertHeredocDelimiter returns the here-doc delimiter word of an opener line whose
// body is provably inert, per the narrow shape stripInertHeredocBodies documents.
// It returns ok=false for everything it cannot prove, which leaves the body in view.
func inertHeredocDelimiter(line string) (string, bool) {
	var (
		quote    byte
		tok      strings.Builder
		words    []string
		delims   []string
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
				return "", false
			}
			flush()
			i += 2
			if i < len(line) && line[i] == '-' {
				i++ // <<- strips leading tabs from the body
			}
			for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
				i++
			}
			var q byte
			if i < len(line) && (line[i] == '\'' || line[i] == '"') {
				q = line[i]
				i++
			}
			var d strings.Builder
			for i < len(line) {
				c := line[i]
				if q != 0 {
					if c == q {
						break
					}
				} else if c == ' ' || c == '\t' || c == '>' || c == '<' ||
					c == '|' || c == ';' || c == '&' {
					break
				}
				d.WriteByte(c)
				i++
			}
			if q == 0 {
				i-- // hand the terminator back to the outer scan
			}
			if d.Len() == 0 {
				return "", false
			}
			delims = append(delims, d.String())
		case ' ', '\t', '\r':
			flush()
		case '|', ';', '&', '(', ')', '{', '}', '`', '$':
			// A second statement, a subshell or a substitution on the opener line
			// could route the body somewhere that runs it. Prove nothing.
			return "", false
		default:
			tok.WriteByte(ch)
		}
	}
	flush()
	if quote != 0 || len(delims) != 1 || !redirect || len(words) == 0 {
		return "", false
	}
	// cat is the whole allow-list on purpose: it never interprets its input, and
	// with stdout redirected the body lands in a file. tee is excluded because it
	// also writes stdout, so `tee f <<'EOF' | sh` would execute the body.
	if words[0] != "cat" {
		return "", false
	}
	return delims[0], true
}
