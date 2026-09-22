package vdso

import "strings"

// search_replace.go — the tolerant search/replace matcher behind the Edit tool's
// drift recovery path.
//
// The dominant LLM edit failure is not a wrong intent: the model read a block,
// then re-emitted it in old_string with the block re-indented (2 spaces where the
// source uses 4, tabs where the source uses spaces) or with trailing whitespace
// trimmed. A byte-exact search refuses that with EDIT_CONFLICT even though the
// intended target is unambiguous. This matcher recovers that case WITHOUT
// weakening the exact path: exact match is tried first and wins, and a tolerant
// match is only accepted when it is UNIQUE. A tolerant match that is ambiguous
// (>=2 candidate windows) returns ok=false rather than guessing which one the
// model meant — the same do-not-guess discipline the exact path enforces with
// its "want exactly 1" rule.
//
// Everything here is pure: no I/O, no package state, stdlib only.

// MatchAndReplaceRelativeIndent rewrites content by matching oldStr allowing
// relative-indentation / leading-whitespace drift, and applying newStr while
// preserving the target's own indentation. Exact match is tried first and, when
// it succeeds, is returned unchanged (exact-match precedence). Returns ok=false
// when no match, or when the tolerant match is not unique.
func MatchAndReplaceRelativeIndent(content, oldStr, newStr string) (string, bool) {
	// An empty old_string is not a block and must never be "found": every
	// string contains the empty string, so an unguarded exact path would
	// substitute at position 0 and corrupt content.
	if oldStr == "" {
		return "", false
	}
	// Exact-match precedence: a byte-exact hit is authoritative and is never
	// reinterpreted. Replace once, matching the Edit tool's default singleton
	// semantics.
	if strings.Contains(content, oldStr) {
		return strings.Replace(content, oldStr, newStr, 1), true
	}

	oldLines := splitLines(oldStr)
	contentLines := splitLines(content)
	start, count := tolerantWindow(contentLines, oldLines)
	if count != 1 {
		// 0 -> no match; >=2 -> ambiguous, do not guess.
		return "", false
	}

	block := contentLines[start : start+len(oldLines)]
	// The matched block's own base indentation is the leading whitespace of its
	// first non-blank line; the replacement is re-indented onto it. The block's
	// final line keeps its own terminator (including a possible newline).
	base := leadingWhitespace(firstNonBlank(block))
	replacement := reindent(block, splitLines(newStr), base)

	var b strings.Builder
	b.Grow(len(content) + len(replacement))
	for _, l := range contentLines[:start] {
		b.WriteString(l.text)
		b.WriteString(l.end)
	}
	b.WriteString(replacement)
	for _, l := range contentLines[start+len(oldLines):] {
		b.WriteString(l.text)
		b.WriteString(l.end)
	}
	return b.String(), true
}

// RelativeIndentMatchCount reports how many tolerant (relative-indentation
// drift) windows of content match oldStr's trimmed line signature, and whether
// the comparison is meaningful at all. It lets the Edit engine distinguish "no
// tolerant match exists" from "the tolerant match is ambiguous" so its refusal
// can say which. A return of (0, false) means oldStr is empty, blank, or has
// more lines than content — not an anchor to match on.
func RelativeIndentMatchCount(content, oldStr string) (int, bool) {
	if oldStr == "" {
		return 0, false
	}
	oldLines := splitLines(oldStr)
	if len(oldLines) == 0 || allBlank(oldLines) {
		return 0, false
	}
	start, count := tolerantWindow(splitLines(content), oldLines)
	_ = start
	return count, true
}

// tolerantWindow slides a window the length of oldLines over contentLines and
// counts windows whose per-line trimmed content equals oldLines' trimmed
// content. It returns the index of the first match and the total count.
//
// The relative-indent SIGNATURE is deliberately just the trimmed text: it
// ignores leading whitespace and tabs-vs-spaces drift, and trailing whitespace,
// while still requiring the same line count and the same non-whitespace content
// per line.
func tolerantWindow(contentLines, oldLines []line) (int, int) {
	if len(oldLines) == 0 || allBlank(oldLines) || len(contentLines) < len(oldLines) {
		return -1, 0
	}
	oldSig := make([]string, len(oldLines))
	for i, l := range oldLines {
		oldSig[i] = strings.TrimSpace(l.text)
	}
	first, count := -1, 0
	for i := 0; i+len(oldLines) <= len(contentLines); i++ {
		if windowMatches(contentLines[i:i+len(oldLines)], oldSig) {
			if first < 0 {
				first = i
			}
			count++
		}
	}
	return first, count
}

// line is one source line split from its terminator, so a window of lines can be
// concatenated byte-for-byte and the surrounding content's line endings survive
// untouched. `end` is "\n", "\r\n", or "" for a final unterminated line.
type line struct {
	text string
	end  string
}

// splitLines splits s into lines carrying their own terminators. A trailing
// newline yields a final empty-text line with an empty terminator, so join is
// exact. CRLF and lone CR are preserved in `end`.
func splitLines(s string) []line {
	if s == "" {
		return nil
	}
	var out []line
	for len(s) > 0 {
		i := strings.IndexByte(s, '\n')
		if i < 0 {
			// A trailing lone CR is treated as a terminator too.
			if strings.HasSuffix(s, "\r") {
				out = append(out, line{text: s[:len(s)-1], end: "\r"})
			} else {
				out = append(out, line{text: s, end: ""})
			}
			break
		}
		end := "\n"
		text := s[:i]
		if strings.HasSuffix(text, "\r") {
			text = text[:len(text)-1]
			end = "\r\n"
		}
		out = append(out, line{text: text, end: end})
		s = s[i+1:]
	}
	return out
}

// windowMatches reports whether each line of window has the same trimmed content
// as want[i].
func windowMatches(window []line, want []string) bool {
	if len(window) != len(want) {
		return false
	}
	for i := range window {
		if strings.TrimSpace(window[i].text) != want[i] {
			return false
		}
	}
	return true
}

// reindent rewrites newStr so its whole block is shifted from the model's own
// base indentation to base. Relative indentation WITHIN newStr is preserved, and
// each new line keeps the terminator of the corresponding old line when there is
// one (so the replacement cannot drop a newline that separated the block from the
// following content).
func reindent(oldBlock []line, newLines []line, base string) string {
	if len(newLines) == 0 {
		return ""
	}
	modelBase := leadingWhitespace(firstNonBlank(newLines))
	var b strings.Builder
	for i, l := range newLines {
		body := strings.TrimLeft(l.text, " \t")
		b.WriteString(base + relativeIndent(l.text, modelBase) + body)
		// Inherit the old block's terminator where the new block does not carry
		// its own, so a replacement never eats the newline before trailing text.
		if l.end != "" {
			b.WriteString(l.end)
		} else if i == len(newLines)-1 && i < len(oldBlock) {
			b.WriteString(oldBlock[i].end)
		} else if i < len(newLines)-1 {
			b.WriteString("\n")
		}
	}
	return b.String()
}

// relativeIndent returns line's leading whitespace with the model's base prefix
// removed. A line indented no deeper than modelBase contributes no extra
// indentation; a blank line stays blank.
func relativeIndent(line, modelBase string) string {
	lead := leadingWhitespace(line)
	if strings.TrimSpace(line) == "" {
		return ""
	}
	if modelBase != "" && strings.HasPrefix(lead, modelBase) {
		return lead[len(modelBase):]
	}
	return lead
}

// leadingWhitespace returns the run of spaces and tabs at the start of line.
func leadingWhitespace(line string) string {
	i := 0
	for i < len(line) && (line[i] == ' ' || line[i] == '\t') {
		i++
	}
	return line[:i]
}

// firstNonBlank returns the first line with non-whitespace content, or "" when
// every line is blank.
func firstNonBlank(lines []line) string {
	for _, l := range lines {
		if strings.TrimSpace(l.text) != "" {
			return l.text
		}
	}
	return ""
}

// allBlank reports whether every line is empty or whitespace-only.
func allBlank(lines []line) bool {
	for _, l := range lines {
		if strings.TrimSpace(l.text) != "" {
			return false
		}
	}
	return true
}

// NearestLineHint reports the 1-based line number of the content window that best
// matches oldStr by trimmed-line content, together with a coarse 0..100 similarity
// where 100 means every oldStr line appears, in order, somewhere in the window. It
// exists so a terminal EDIT_CONFLICT refusal can tell the model WHERE the intended
// target most plausibly is, without echoing file content back through the refusal
// (the Refusal contract forbids content in a detail).
//
// It returns (0, 0, false) when oldStr is empty/blank or content has no lines. A
// positive score with score < 100 is the common real case: the model's oldString
// is a stale or otherwise-wrong revision of a region that still exists, so the
// remedy is a re-read at the named line, NOT a fuzzy auto-apply.
func NearestLineHint(content, oldStr string) (line int, score int, ok bool) {
	oldLines := splitLines(oldStr)
	if len(oldLines) == 0 || allBlank(oldLines) {
		return 0, 0, false
	}
	contentLines := splitLines(content)
	if len(contentLines) == 0 {
		return 0, 0, false
	}
	want := make([]string, len(oldLines))
	wantCount := 0
	for i, l := range oldLines {
		want[i] = strings.TrimSpace(l.text)
		if want[i] != "" {
			wantCount++
		}
	}
	if wantCount == 0 {
		return 0, 0, false
	}
	// The anchor is oldStr's first non-blank line: a candidate window is most
	// plausibly the intended region when it STARTS on that line. Among windows,
	// prefer (a) those anchored on want[0], then (b) the most ordered hits.
	anchor := ""
	for _, w := range want {
		if w != "" {
			anchor = w
			break
		}
	}
	bestLine, bestScore, bestAnchored := 0, 0, false
	span := len(oldLines)
	for i := 0; i+span <= len(contentLines); i++ {
		wi, hit := 0, 0
		for j := i; j < i+span && wi < len(want); j++ {
			t := strings.TrimSpace(contentLines[j].text)
			if t == "" {
				continue
			}
			if t == want[wi] {
				hit++
				wi++
			}
		}
		sc := hit * 100 / wantCount
		anchored := span > 0 && strings.TrimSpace(contentLines[i].text) == anchor
		better := false
		switch {
		case anchored && !bestAnchored:
			better = true
		case anchored == bestAnchored && sc > bestScore:
			better = true
		}
		if better {
			bestScore, bestLine, bestAnchored = sc, i+1, anchored
		}
	}
	if bestLine == 0 || bestScore == 0 {
		return 0, 0, false
	}
	return bestLine, bestScore, true
}
