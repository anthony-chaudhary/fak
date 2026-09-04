package adjudicator

import (
	"strings"
)

// HeredocTarget represents a file write target extracted from an inline here-document command.
type HeredocTarget struct {
	Path      string
	Append    bool
	Delimiter string
	TreeKnown bool
}

// ExtractHeredocWriteTarget inspects cmd for a heredoc file write target (e.g.
// `cat << 'EOF' > file.txt` or `cat > file.txt << 'EOF'`), returning the target
// path, whether append mode was used, and ok indicating whether a valid, tree-known
// target was successfully extracted.
func ExtractHeredocWriteTarget(cmd string) (targetPath string, appendMode bool, ok bool) {
	targets := ExtractHeredocTargets(cmd)
	for _, t := range targets {
		if t.TreeKnown && t.Path != "" {
			return t.Path, t.Append, true
		}
	}
	if len(targets) > 0 {
		return targets[0].Path, targets[0].Append, false
	}
	return "", false, false
}

type heredocPendingItem struct {
	delim     string
	stripTabs bool
}

// ExtractHeredocTargets inspects cmd for all inline here-document file write targets,
// reliably parsing variants such as `cat << 'EOF' > file.txt`, `cat > file.txt << 'EOF'`,
// quoted paths, and multi-line heredocs while ensuring that payloads in multi-line
// bodies are not misclassified as command lines.
func ExtractHeredocTargets(cmd string) []HeredocTarget {
	var targets []HeredocTarget
	normalized := strings.ReplaceAll(cmd, "\r\n", "\n")
	lines := strings.Split(normalized, "\n")

	var pending []heredocPendingItem

	for i := 0; i < len(lines); i++ {
		line := strings.TrimRight(lines[i], "\r")

		if len(pending) > 0 {
			cur := pending[0]
			matched := false
			if cur.stripTabs {
				matched = strings.TrimLeft(line, "\t") == cur.delim || strings.TrimSpace(line) == cur.delim
			} else {
				matched = strings.TrimSpace(line) == cur.delim
			}
			if matched {
				pending = pending[1:]
			}
			continue
		}

		trimmed := strings.TrimSpace(line)
		if trimmed == "" || strings.HasPrefix(trimmed, "#") {
			continue
		}

		segments := splitHeredocSegments(line)
		for _, seg := range segments {
			target, delim, stripTabs, ok := parseHeredocSegment(seg)
			if ok {
				targets = append(targets, target)
				pending = append(pending, heredocPendingItem{
					delim:     delim,
					stripTabs: stripTabs,
				})
			}
		}
	}

	return targets
}

func splitHeredocSegments(line string) []string {
	var segs []string
	var b strings.Builder
	var quote byte

	flush := func() {
		s := strings.TrimSpace(b.String())
		if s != "" {
			segs = append(segs, s)
		}
		b.Reset()
	}

	for i := 0; i < len(line); i++ {
		ch := line[i]
		if quote != 0 {
			b.WriteByte(ch)
			if ch == quote {
				quote = 0
			} else if quote == '"' && ch == '\\' && i+1 < len(line) {
				i++
				b.WriteByte(line[i])
			}
			continue
		}

		if ch == '\'' || ch == '"' {
			quote = ch
			b.WriteByte(ch)
			continue
		}

		if ch == '\\' && i+1 < len(line) {
			b.WriteByte(ch)
			i++
			b.WriteByte(line[i])
			continue
		}

		// Don't split on >| or >& or &>
		if ch == '|' && i > 0 && line[i-1] == '>' {
			b.WriteByte(ch)
			continue
		}
		if ch == '&' && i > 0 && line[i-1] == '>' {
			b.WriteByte(ch)
			continue
		}
		if ch == '&' && i+1 < len(line) && line[i+1] == '>' {
			b.WriteByte(ch)
			continue
		}

		if ch == ';' || ch == '|' || ch == '&' {
			flush()
			if (ch == '&' && i+1 < len(line) && line[i+1] == '&') ||
				(ch == '|' && i+1 < len(line) && line[i+1] == '|') {
				i++
			}
			continue
		}

		b.WriteByte(ch)
	}
	flush()
	return segs
}

func parseHeredocSegment(seg string) (target HeredocTarget, delim string, stripTabs bool, ok bool) {
	if !strings.Contains(seg, "<<") || !strings.Contains(seg, ">") {
		return HeredocTarget{}, "", false, false
	}

	var (
		words          []string
		hasHeredoc     bool
		heredocDelim   string
		heredocStrip   bool
		hasRedirect    bool
		redirectPath   string
		redirectRaw    string
		redirectAppend bool
	)

	for i := 0; i < len(seg); {
		ch := seg[i]
		if isWhitespace(ch) {
			i++
			continue
		}
		if ch == '#' {
			break
		}

		// Check for heredoc operator << or <<-
		if ch == '<' && i+1 < len(seg) && seg[i+1] == '<' {
			if i+2 < len(seg) && seg[i+2] == '<' {
				// here-string <<<, not a heredoc
				i += 3
				continue
			}
			i += 2
			strip := false
			if i < len(seg) && seg[i] == '-' {
				strip = true
				i++
			}
			d, next := readDelimiterToken(seg, i)
			if d != "" {
				hasHeredoc = true
				heredocDelim = d
				heredocStrip = strip
			}
			i = next
			continue
		}

		// Check for fd-specified redirect e.g. 1> or 2>
		if (ch == '1' || ch == '2') && i+1 < len(seg) && seg[i+1] == '>' {
			if i == 0 || isWhitespace(seg[i-1]) {
				fd := ch
				i += 2
				isAppend := false
				if i < len(seg) && seg[i] == '>' {
					isAppend = true
					i++
				} else if i < len(seg) && seg[i] == '|' {
					i++
				}
				if i < len(seg) && seg[i] == '&' {
					for i < len(seg) && !isWhitespace(seg[i]) && !isShellMetachar(seg[i]) {
						i++
					}
					continue
				}
				if fd == '2' {
					_, _, next := readTargetToken(seg, i)
					i = next
					continue
				}
				cleaned, raw, next := readTargetToken(seg, i)
				if raw != "" {
					hasRedirect = true
					redirectPath = cleaned
					redirectRaw = raw
					redirectAppend = isAppend
				}
				i = next
				continue
			}
		}

		// Check for output redirect > or >> or >|
		if ch == '>' {
			i++
			isAppend := false
			if i < len(seg) && seg[i] == '>' {
				isAppend = true
				i++
			} else if i < len(seg) && seg[i] == '|' {
				i++
			}
			if i < len(seg) && seg[i] == '&' {
				for i < len(seg) && !isWhitespace(seg[i]) && !isShellMetachar(seg[i]) {
					i++
				}
				continue
			}
			cleaned, raw, next := readTargetToken(seg, i)
			if raw != "" {
				hasRedirect = true
				redirectPath = cleaned
				redirectRaw = raw
				redirectAppend = isAppend
			}
			i = next
			continue
		}

		word, next := readShellWord(seg, i)
		if word != "" {
			words = append(words, word)
			i = next
			continue
		}
		i++
	}

	if !hasHeredoc || !hasRedirect || !isCatCommand(words) {
		return HeredocTarget{}, "", false, false
	}

	treeKnown := isTreeKnown(redirectRaw, redirectPath)
	target = HeredocTarget{
		Path:      redirectPath,
		Append:    redirectAppend,
		Delimiter: heredocDelim,
		TreeKnown: treeKnown,
	}
	return target, heredocDelim, heredocStrip, true
}

func readDelimiterToken(s string, start int) (string, int) {
	i := start
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) {
		return "", i
	}
	var delim strings.Builder
	var quote byte
	if s[i] == '\'' || s[i] == '"' {
		quote = s[i]
		i++
		for i < len(s) && s[i] != quote {
			delim.WriteByte(s[i])
			i++
		}
		if i < len(s) && s[i] == quote {
			i++
		}
		return delim.String(), i
	}
	if s[i] == '\\' {
		i++
	}
	for i < len(s) {
		ch := s[i]
		if isWhitespace(ch) || isShellMetachar(ch) {
			break
		}
		if ch == '\\' && i+1 < len(s) {
			i++
			delim.WriteByte(s[i])
			i++
			continue
		}
		delim.WriteByte(ch)
		i++
	}
	return delim.String(), i
}

func readTargetToken(s string, start int) (string, string, int) {
	i := start
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) {
		return "", "", i
	}
	if isShellMetachar(s[i]) {
		return "", "", i
	}
	var cleaned strings.Builder
	var raw strings.Builder
	var quote byte
	for i < len(s) {
		ch := s[i]
		if quote != 0 {
			raw.WriteByte(ch)
			if ch == quote {
				quote = 0
				i++
				continue
			}
			if quote == '"' && ch == '\\' && i+1 < len(s) {
				next := s[i+1]
				if next == '"' || next == '\\' || next == '$' || next == '`' {
					raw.WriteByte(next)
					cleaned.WriteByte(next)
					i += 2
					continue
				}
			}
			cleaned.WriteByte(ch)
			i++
			continue
		}

		if ch == '\'' || ch == '"' {
			quote = ch
			raw.WriteByte(ch)
			i++
			continue
		}
		if ch == '$' && i+1 < len(s) && s[i+1] == '(' {
			raw.WriteString("$(")
			cleaned.WriteString("$(")
			i += 2
			parenDepth := 1
			var subQuote byte
			for i < len(s) && parenDepth > 0 {
				c := s[i]
				raw.WriteByte(c)
				cleaned.WriteByte(c)
				if subQuote != 0 {
					if c == subQuote {
						subQuote = 0
					}
				} else {
					if c == '\'' || c == '"' {
						subQuote = c
					} else if c == '(' {
						parenDepth++
					} else if c == ')' {
						parenDepth--
					}
				}
				i++
			}
			continue
		}
		if isWhitespace(ch) || isShellMetachar(ch) {
			break
		}
		if ch == '\\' && i+1 < len(s) {
			raw.WriteByte(ch)
			raw.WriteByte(s[i+1])
			cleaned.WriteByte(s[i+1])
			i += 2
			continue
		}
		raw.WriteByte(ch)
		cleaned.WriteByte(ch)
		i++
	}
	return cleaned.String(), raw.String(), i
}

func readShellWord(s string, start int) (string, int) {
	i := start
	for i < len(s) && (s[i] == ' ' || s[i] == '\t') {
		i++
	}
	if i >= len(s) {
		return "", i
	}
	if isShellMetachar(s[i]) {
		return "", i
	}
	var b strings.Builder
	var quote byte
	for i < len(s) {
		ch := s[i]
		if quote != 0 {
			if ch == quote {
				quote = 0
				i++
				continue
			}
			if quote == '"' && ch == '\\' && i+1 < len(s) {
				i++
				b.WriteByte(s[i])
				i++
				continue
			}
			b.WriteByte(ch)
			i++
			continue
		}
		if ch == '\'' || ch == '"' {
			quote = ch
			i++
			continue
		}
		if ch == '$' && i+1 < len(s) && s[i+1] == '(' {
			b.WriteString("$(")
			i += 2
			parenDepth := 1
			var subQuote byte
			for i < len(s) && parenDepth > 0 {
				c := s[i]
				b.WriteByte(c)
				if subQuote != 0 {
					if c == subQuote {
						subQuote = 0
					}
				} else {
					if c == '\'' || c == '"' {
						subQuote = c
					} else if c == '(' {
						parenDepth++
					} else if c == ')' {
						parenDepth--
					}
				}
				i++
			}
			continue
		}
		if isWhitespace(ch) || isShellMetachar(ch) {
			break
		}
		if ch == '\\' && i+1 < len(s) {
			i++
			b.WriteByte(s[i])
			i++
			continue
		}
		b.WriteByte(ch)
		i++
	}
	return b.String(), i
}

func isCatCommand(words []string) bool {
	if len(words) == 0 {
		return false
	}
	idx := 0
	for idx < len(words) && envAssignmentRE.MatchString(words[idx]) {
		idx++
	}
	if idx >= len(words) {
		return false
	}
	base := strings.ToLower(baseCommand(words[idx]))
	if base == "sudo" || base == "env" || base == "doas" {
		idx++
		for idx < len(words) {
			w := words[idx]
			if envAssignmentRE.MatchString(w) {
				idx++
				continue
			}
			if strings.HasPrefix(w, "-") {
				if w == "-u" || w == "-g" || w == "-C" || w == "-D" || w == "-h" || w == "-p" || w == "-r" || w == "-t" || w == "-T" {
					idx += 2
					continue
				}
				idx++
				continue
			}
			break
		}
		if idx >= len(words) {
			return false
		}
		base = strings.ToLower(baseCommand(words[idx]))
	}
	return base == "cat" || base == "cat.exe"
}

func isTreeKnown(raw, cleaned string) bool {
	if cleaned == "" || isNullSink(cleaned) {
		return false
	}
	trimmedRaw := strings.TrimSpace(raw)
	if strings.Count(trimmedRaw, "'")%2 != 0 {
		return false
	}
	if len(trimmedRaw) >= 2 && trimmedRaw[0] == '\'' && trimmedRaw[len(trimmedRaw)-1] == '\'' && strings.Count(trimmedRaw, "'") == 2 {
		return true
	}
	unescapedDQuotes := 0
	for i := 0; i < len(trimmedRaw); i++ {
		if trimmedRaw[i] == '\\' && i+1 < len(trimmedRaw) {
			i++
			continue
		}
		if trimmedRaw[i] == '"' {
			unescapedDQuotes++
		}
	}
	if unescapedDQuotes%2 != 0 {
		return false
	}
	if strings.ContainsAny(cleaned, "$`()") {
		return false
	}
	if strings.ContainsAny(cleaned, "*?") {
		return false
	}
	return true
}

func isWhitespace(ch byte) bool {
	return ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n'
}

func isShellMetachar(ch byte) bool {
	return ch == ';' || ch == '&' || ch == '|' || ch == '<' || ch == '>' || ch == '(' || ch == ')'
}
