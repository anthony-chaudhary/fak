package docrender

import (
	"fmt"
	"regexp"
	"strings"
)

// Unsupported is one construct the bounded subset does not render, located.
type Unsupported struct {
	Line      int    // 1-based line in the source
	Construct string // the construct's name, e.g. "nested list item"
	Text      string // the offending line, trimmed
	Fix       string // what to write instead
}

// String renders one refusal as `path:12: nested list item: ... — <fix>` minus the
// path, which UnsupportedError owns.
func (u Unsupported) String() string {
	s := fmt.Sprintf("%d: %s: %s", u.Line, u.Construct, u.Text)
	if u.Fix != "" {
		s += "\n      fix: " + u.Fix
	}
	return s
}

// UnsupportedError is what Parse returns for a document that uses constructs
// outside the subset. It carries every refusal, not the first, so a document is
// fixed in one pass rather than N.
type UnsupportedError struct {
	Source string
	Items  []Unsupported
}

func (e *UnsupportedError) Error() string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s: %d markdown construct(s) outside the docrender subset", e.Source, len(e.Items))
	for _, it := range e.Items {
		fmt.Fprintf(&b, "\n  %s:%s", e.Source, it.String())
	}
	b.WriteString("\n  The subset is bounded on purpose (see the internal/docrender package doc for the " +
		"full list). It refuses rather than passing an unsupported construct through as literal " +
		"text, because literal text in the PDF reads as a typo in the document rather than a gap " +
		"in the renderer.")
	return b.String()
}

// Constructs returns the names of every construct Scan can refuse, sorted by first
// appearance in the checks below. It backs the doc/test that the enumeration and
// the implementation agree.
func Constructs() []string {
	return []string{
		"heading level 4 or deeper",
		"setext heading",
		"alternate thematic break",
		"'+' bullet",
		"nested list item",
		"indented code block",
		"link reference definition",
		"reference link",
		"footnote",
		"strikethrough",
		"autolink",
		"inline HTML",
		"multi-line HTML comment",
		"inline image",
		"unterminated code fence",
	}
}

var (
	reScanFence   = regexp.MustCompile("^\\s*```")
	reScanH4      = regexp.MustCompile(`^#{4,}(\s|$)`)
	reScanSetext  = regexp.MustCompile(`^=+\s*$`)
	reScanAltRule = regexp.MustCompile(`^\s*(?:(?:\*\s*){3,}|(?:_\s*){3,})$`)
	reScanPlus    = regexp.MustCompile(`^\s*\+\s+\S`)
	reScanNestUL  = regexp.MustCompile(`^\s+[-*+]\s+\S`)
	reScanNestOL  = regexp.MustCompile(`^\s+\d+\.\s+\S`)
	reScanTopUL   = regexp.MustCompile(`^[-*]\s+\S`)
	reScanTopOL   = regexp.MustCompile(`^\d+\.\s+\S`)
	reScanIndent  = regexp.MustCompile(`^(?:\t| {4,})\S`)
	reScanRefDef  = regexp.MustCompile(`^\[[^\]]+\]:\s+\S`)
	reScanRefLink = regexp.MustCompile(`\[[^\]]*\]\[[^\]]*\]`)
	reScanFootnt  = regexp.MustCompile(`\[\^[^\]]+\]`)
	reScanStrike  = regexp.MustCompile(`~~[^~\s][^~]*~~`)
	reScanAuto    = regexp.MustCompile(`<(?:https?|mailto|ftp):[^>\s]+>`)
	reScanTag     = regexp.MustCompile(`</?([a-zA-Z][a-zA-Z0-9]*)(?:\s[^<>]*)?/?>`)
	reScanLineCmt = regexp.MustCompile(`^\s*<!--.*-->\s*$`)
	reScanImgLine = regexp.MustCompile(`^!\[([^\]]*)\]\(([^)]+)\)\s*$`)
	reScanImgAny  = regexp.MustCompile(`!\[[^\]]*\]\([^)]+\)`)
	reScanCodeSpn = regexp.MustCompile("`[^`]*`")
)

// htmlTagNames is a CLOSED list, and closed for a reason. Testing "does this line
// contain <something>" would refuse ordinary prose ("a < b > c", a generic
// `Map<K,V>` outside backticks); testing against the tags a Markdown author
// actually reaches for catches the real cases and leaves arithmetic alone. A tag
// outside this list still escapes safely, it just is not called out.
var htmlTagNames = map[string]bool{
	"a": true, "abbr": true, "b": true, "blockquote": true, "br": true, "code": true,
	"details": true, "div": true, "em": true, "figure": true, "figcaption": true,
	"h1": true, "h2": true, "h3": true, "h4": true, "h5": true, "h6": true, "hr": true,
	"i": true, "iframe": true, "img": true, "kbd": true, "li": true, "ol": true, "p": true,
	"pre": true, "script": true, "section": true, "small": true, "span": true, "strong": true,
	"style": true, "sub": true, "summary": true, "sup": true, "table": true, "tbody": true,
	"td": true, "th": true, "thead": true, "tr": true, "u": true, "ul": true, "video": true,
}

// Scan reports every construct in src that the subset does not render. It is the
// gate Parse runs first; it is exported so a lint verb can report a whole corpus
// without rendering any of it.
//
// The scan is line-oriented and fence-aware: nothing inside a ``` block is checked,
// because a code block's whole job is to hold text that is not Markdown. Inline
// checks additionally mask `code spans` for the same reason — a package doc that
// mentions `[ref][1]` inside backticks is documentation, not a reference link.
func Scan(src string) []Unsupported {
	var out []Unsupported
	add := func(line int, construct, text, fix string) {
		out = append(out, Unsupported{line, construct, strings.TrimSpace(text), fix})
	}

	lines := strings.Split(strings.ReplaceAll(src, "\r\n", "\n"), "\n")
	inFence, fenceLine := false, 0
	listOpen, prevBlank := false, true

	for i, ln := range lines {
		n := i + 1
		if reScanFence.MatchString(ln) {
			if inFence {
				inFence = false
			} else {
				inFence, fenceLine = true, n
			}
			prevBlank, listOpen = false, false
			continue
		}
		if inFence {
			continue
		}
		if strings.TrimSpace(ln) == "" {
			// A blank line closes any open list, exactly as the parser's flush does —
			// so the indented-code check below agrees with what the parser would do.
			prevBlank, listOpen = true, false
			continue
		}

		switch {
		case reScanH4.MatchString(ln):
			add(n, "heading level 4 or deeper", ln, "use ### — three levels is the depth this corpus uses")
		case reScanSetext.MatchString(ln) && !prevBlank:
			add(n, "setext heading", ln, "use # or ## on the heading line itself")
		case reScanAltRule.MatchString(ln):
			add(n, "alternate thematic break", ln, "use --- for a thematic break")
		case reScanPlus.MatchString(ln):
			add(n, "'+' bullet", ln, "use - for a list item")
		case reScanNestUL.MatchString(ln) || reScanNestOL.MatchString(ln):
			add(n, "nested list item", ln, "flatten it, or promote the sub-list to its own ### subsection")
		case reScanIndent.MatchString(ln) && !listOpen:
			add(n, "indented code block", ln, "fence it with ``` — an indented block is indistinguishable from a wrapped line here")
		case reScanRefDef.MatchString(ln):
			add(n, "link reference definition", ln, "write the link inline: [text](url)")
		}

		if reScanTopUL.MatchString(ln) || reScanTopOL.MatchString(ln) {
			listOpen = true
		}
		scanInline(ln, n, listOpen, add)
		prevBlank = false
	}

	if inFence {
		add(fenceLine, "unterminated code fence", lines[fenceLine-1],
			"close it with ``` — everything after an open fence is swallowed into the code block")
	}
	return out
}

// scanInline runs the checks that look inside a line, with code spans masked out.
func scanInline(ln string, n int, listOpen bool, add func(int, string, string, string)) {
	if reScanLineCmt.MatchString(ln) {
		return // a whole-line comment is metadata; the parser drops it
	}
	if strings.Contains(ln, "<!--") || strings.Contains(ln, "-->") {
		add(n, "multi-line HTML comment", ln,
			"keep a comment on one line: <!-- kind: deck -->")
		return
	}
	masked := reScanCodeSpn.ReplaceAllStringFunc(ln, func(m string) string {
		return strings.Repeat(" ", len(m))
	})
	if reScanRefLink.MatchString(masked) {
		add(n, "reference link", ln, "write the link inline: [text](url)")
	}
	if reScanFootnt.MatchString(masked) {
		add(n, "footnote", ln, "fold the note into the sentence, or make it a parenthetical")
	}
	if reScanStrike.MatchString(masked) {
		add(n, "strikethrough", ln, "delete the struck text, or say it in words")
	}
	if reScanAuto.MatchString(masked) {
		add(n, "autolink", ln, "write the link inline: [text](url)")
	}
	for _, m := range reScanTag.FindAllStringSubmatch(masked, -1) {
		if htmlTagNames[strings.ToLower(m[1])] {
			add(n, "inline HTML", ln, "use the markdown construct, or fence the tag as `code`")
			break
		}
	}
	if reScanImgAny.MatchString(masked) && !reScanImgLine.MatchString(strings.TrimSpace(ln)) {
		add(n, "inline image", ln, "put the figure on a line of its own: ![alt](path)")
	}
}
