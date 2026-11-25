package launchvisual

import (
	"fmt"
	"regexp"
	"strings"
)

// Violation is one doc number or table cell that does not trace to the index.
type Violation struct {
	File   string
	Line   int
	Token  string
	Reason string
}

var (
	urlSpan    = regexp.MustCompile(`https?://\S+`)
	linkTarget = regexp.MustCompile(`\]\(([^)\s]*)\)`)
	codeSpan   = regexp.MustCompile("`[^`]*`")
	pathSpan   = regexp.MustCompile(`[\w.\-]*[/\\][\w./\\\-]*`)
	numberTok  = regexp.MustCompile(`\d+(?:\.\d+)?x?`)
	unitAfter  = regexp.MustCompile(`^ ?(?:tok/s|ms|s|%)(?:[^A-Za-z0-9_]|$)`)
	hlCell     = regexp.MustCompile("`hl:([a-z0-9_.]+)`")
	fenceLine  = regexp.MustCompile("^\\s*(```|~~~)")
)

// DocCheck verifies every number in the data file's docs against the index. A number passes
// when it equals a row's display value, is a ratio label (N.NNx for a row with a ratio, or a
// display value of a unit-x row), or is listed in allowed_numbers. A receipt sample display
// is valid only on a table row that cites the row it came from. Integers are checked only
// when a unit word follows them or they sit in a table row citing a headline. A table row
// holding hl:<id> must also show that row's display string. Every row a chart draws must
// be cited by such a cell in some listed doc, and a superseded token may appear only on a
// line that says so. Fenced code, URLs, link targets, code spans and path-like tokens are not
// prose.
func DocCheck(read ReadFunc, cf ChartFile, ix Index) ([]Violation, error) {
	rules := newDocRules(allowedSet(cf), ix)
	var out []Violation
	cited := map[string]bool{}
	for _, doc := range cf.Docs {
		b, err := read(doc)
		if err != nil {
			return nil, fmt.Errorf("read doc %s: %w", doc, err)
		}
		text := string(b)
		sc := scanDoc(doc, text, rules)
		out = append(out, sc.violations...)
		out = append(out, supersededViolations(doc, text, cf.SupersededTokens)...)
		for id := range sc.cited {
			cited[id] = true
		}
	}
	for _, id := range cf.RowIDs() {
		if _, ok := rules.byID[id]; ok && !cited[id] {
			out = append(out, Violation{File: strings.Join(cf.Docs, ", "), Token: "hl:" + id, Reason: "charted row has no hl cell in a listed doc table"})
		}
	}
	return out, nil
}

// docRules is the index view a doc is checked against.
type docRules struct {
	allowed  map[string]bool
	displays map[string]bool
	owners   map[string][]string
	ratios   map[string]bool
	byID     map[string]Row
}

func newDocRules(allowed map[string]bool, ix Index) docRules {
	r := docRules{
		allowed:  allowed,
		displays: map[string]bool{},
		owners:   map[string][]string{},
		ratios:   map[string]bool{},
		byID:     ix.ByID(),
	}
	for _, row := range ix.Rows {
		r.displays[row.Display()] = true
		for _, s := range row.Samples {
			if d := DisplayValue(s, row.DisplayDecimals); d != row.Display() {
				r.owners[d] = append(r.owners[d], row.ID)
			}
		}
		if row.Ratio != nil {
			r.ratios[row.RatioLabel()] = true
		}
		if row.Unit == "x" {
			r.ratios[row.Display()+"x"] = true
		}
	}
	return r
}

// docScan is one doc's findings: its violations and the headline ids its table cells cite.
type docScan struct {
	violations []Violation
	cited      map[string]bool
}

// supersededViolations flags a superseded token on any line that does not also say it is
// superseded, so an old figure can only appear as history.
func supersededViolations(file, text string, tokens []string) []Violation {
	var out []Violation
	for i, line := range strings.Split(text, "\n") {
		if strings.Contains(strings.ToLower(line), "superseded") {
			continue
		}
		for _, tok := range tokens {
			if tok != "" && containsNumber(line, tok) {
				out = append(out, Violation{File: file, Line: i + 1, Token: tok, Reason: "superseded number outside a line that says superseded"})
			}
		}
	}
	return out
}

func scanDoc(file, text string, rules docRules) docScan {
	sc := docScan{cited: map[string]bool{}}
	inFence := false
	for i, line := range strings.Split(text, "\n") {
		lineNo := i + 1
		if fenceLine.MatchString(line) {
			inFence = !inFence
			continue
		}
		if inFence {
			continue
		}
		var ids []string
		if strings.HasPrefix(strings.TrimSpace(line), "|") {
			for _, m := range hlCell.FindAllStringSubmatch(line, -1) {
				ids = append(ids, m[1])
			}
		}
		for _, id := range ids {
			r, ok := rules.byID[id]
			if !ok {
				sc.violations = append(sc.violations, Violation{File: file, Line: lineNo, Token: "hl:" + id, Reason: "unknown headline row"})
				continue
			}
			sc.cited[id] = true
			if !showsDisplay(line, r) {
				sc.violations = append(sc.violations, Violation{File: file, Line: lineNo, Token: r.Display(), Reason: fmt.Sprintf("table cell hl:%s must show %s", id, r.Display())})
			}
		}
		sc.violations = append(sc.violations, rules.numberViolations(file, lineNo, line, maskSpans(line), ids)...)
	}
	return sc
}

// numberViolations checks the numbers of one line. masked is line with non-prose spans blanked,
// at the same byte offsets; unit words are read from line itself.
func (r docRules) numberViolations(file string, lineNo int, line, masked string, lineIDs []string) []Violation {
	var out []Violation
	for _, loc := range numberTok.FindAllStringIndex(masked, -1) {
		if !numberBoundaryOK(masked, loc[0], loc[1]) {
			continue
		}
		tok := masked[loc[0]:loc[1]]
		hasDot := strings.Contains(tok, ".")
		isX := strings.HasSuffix(tok, "x")
		if !hasDot && !isX && !unitAfter.MatchString(line[loc[1]:]) && len(lineIDs) == 0 {
			continue
		}
		if r.allowed[tok] {
			continue
		}
		reason := ""
		switch {
		case isX && !hasDot:
			reason = "integer ratio; a ratio is written N.NNx from an index row"
		case isX:
			if !r.ratios[tok] {
				reason = "not a non-null ratio of any index row"
			}
		default:
			reason = r.displayReason(tok, hasDot, lineIDs)
		}
		if reason != "" {
			out = append(out, Violation{File: file, Line: lineNo, Token: tok, Reason: reason})
		}
	}
	return out
}

// displayReason is empty when tok is a row's display value, or a receipt sample display on a
// table line that cites the row it came from.
func (r docRules) displayReason(tok string, hasDot bool, lineIDs []string) string {
	if r.displays[tok] {
		return ""
	}
	if owners := r.owners[tok]; len(owners) > 0 {
		for _, id := range owners {
			if containsString(lineIDs, id) {
				return ""
			}
		}
		return "receipt sample display is valid only on a table row citing its own hl id"
	}
	if hasDot {
		return "not the display value of any index row"
	}
	return "integer is not an index display value; add it to allowed_numbers if it is not a measurement"
}

// maskSpans blanks URLs, markdown link targets, inline code spans and path-like tokens,
// keeping byte offsets so number positions still match the original line.
func maskSpans(line string) string {
	b := []byte(line)
	blank := func(lo, hi int) {
		for i := lo; i < hi; i++ {
			b[i] = ' '
		}
	}
	for _, loc := range urlSpan.FindAllStringIndex(line, -1) {
		blank(loc[0], loc[1])
	}
	for _, loc := range linkTarget.FindAllStringSubmatchIndex(line, -1) {
		blank(loc[2], loc[3])
	}
	for _, loc := range codeSpan.FindAllStringIndex(line, -1) {
		blank(loc[0]+1, loc[1]-1)
	}
	for _, loc := range pathSpan.FindAllStringIndex(line, -1) {
		if isPathToken(line[loc[0]:loc[1]]) {
			blank(loc[0], loc[1])
		}
	}
	return string(b)
}

// isPathToken is true for a slash- or backslash-joined token with a letter-led segment longer
// than one character, such as docs/launch. A numeric pair such as 117.16/110.67, or a rate
// such as 180.8/s, is not a path and stays checked.
func isPathToken(s string) bool {
	if !strings.ContainsAny(s, `/\`) {
		return false
	}
	for _, seg := range strings.FieldsFunc(s, func(r rune) bool { return r == '/' || r == '\\' }) {
		if len(seg) > 1 && isLetter(seg[0]) {
			return true
		}
	}
	return false
}

// numberBoundaryOK rejects a number match that is part of a larger token, such as an
// identifier or a dotted version, while accepting a number at the end of a sentence.
func numberBoundaryOK(s string, start, end int) bool {
	if start > 0 {
		c := s[start-1]
		if isWordByte(c) || (c == '.' && start > 1 && isDigit(s[start-2])) {
			return false
		}
	}
	if end < len(s) {
		c := s[end]
		if isWordByte(c) {
			return false
		}
		if c == '.' && end+1 < len(s) && isDigit(s[end+1]) {
			return false
		}
	}
	return true
}

// showsDisplay reports whether a table line shows row r's display value. A unit-x row is
// written with its x suffix, so its display may stand as N.NNx.
func showsDisplay(line string, r Row) bool {
	if containsNumber(line, r.Display()) {
		return true
	}
	return r.Unit == "x" && containsNumber(line, r.Display()+"x")
}

func containsNumber(s, want string) bool {
	for off := 0; off < len(s); {
		i := strings.Index(s[off:], want)
		if i < 0 {
			return false
		}
		start := off + i
		if numberBoundaryOK(s, start, start+len(want)) {
			return true
		}
		off = start + 1
	}
	return false
}

func containsString(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}

func isWordByte(c byte) bool {
	return c == '_' || isDigit(c) || isLetter(c)
}

func isLetter(c byte) bool { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') }

func isDigit(c byte) bool { return c >= '0' && c <= '9' }
