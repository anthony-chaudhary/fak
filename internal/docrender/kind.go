package docrender

import (
	"fmt"
	"path/filepath"
	"sort"
	"strings"
)

// Kind is the document class: what `##` and `---` mean, and what page the result
// is laid out for. It is an INPUT to Parse, never something Parse works out. See
// the package doc for why content inference is absent.
type Kind string

const (
	// KindDeck is slide-shaped: 16:9 landscape, one page per `##`, `---` dropped.
	KindDeck Kind = "deck"
	// KindReport is document-shaped: US Letter portrait, continuous flow, rules kept,
	// a table of contents on request. The corpus default.
	KindReport Kind = "report"
	// KindBrief is one-pager-shaped: the report geometry with tighter leading and no
	// per-section page discipline, for something that is meant to be read at once.
	KindBrief Kind = "brief"
)

// Kinds lists every Kind, in a stable order, for flag help and error messages.
func Kinds() []Kind { return []Kind{KindDeck, KindReport, KindBrief} }

// KindNames is Kinds() as a "deck|report|brief" string.
func KindNames() string {
	out := make([]string, 0, len(Kinds()))
	for _, k := range Kinds() {
		out = append(out, string(k))
	}
	return strings.Join(out, "|")
}

// Valid reports whether k is one of the declared kinds. The zero Kind is not:
// defaulting an unset kind inside the parser is exactly the silent classification
// this package refuses to make.
func (k Kind) Valid() bool {
	for _, want := range Kinds() {
		if k == want {
			return true
		}
	}
	return false
}

// ParseKind turns a flag value into a Kind, or explains the choice.
func ParseKind(s string) (Kind, error) {
	k := Kind(strings.ToLower(strings.TrimSpace(s)))
	if !k.Valid() {
		return "", fmt.Errorf("unknown document kind %q: want one of %s", s, KindNames())
	}
	return k, nil
}

// Rule names which of the four ordered rules decided a Kind. It exists so the
// decision is reportable: an operator who thinks a document rendered wrong needs
// to be able to ask "why is this a deck?" and get an answer that is not a guess.
type Rule string

const (
	// RuleOverride: the caller passed an explicit kind (a --kind flag).
	RuleOverride Rule = "override"
	// RuleMarker: the document carries an explicit `<!-- kind: deck -->` marker.
	RuleMarker Rule = "marker"
	// RulePath: a corpus rule keyed on the document's path fired.
	RulePath Rule = "path"
	// RuleDefault: nothing said otherwise, so the corpus default applies.
	RuleDefault Rule = "default"
)

// Decision is a resolved Kind plus the rule that produced it and a sentence a
// human can read.
type Decision struct {
	Kind Kind
	Rule Rule
	Why  string
}

// String renders the decision as one line: `deck (marker): ...`.
func (d Decision) String() string { return fmt.Sprintf("%s (%s): %s", d.Kind, d.Rule, d.Why) }

// pathRule is one entry of the corpus rule: a path fragment (in slash form, matched
// case-sensitively against the whole slashed path) or a basename prefix, and the
// Kind it selects.
type pathRule struct {
	// Segment matches anywhere in the slashed path, e.g. "/decks/".
	Segment string
	// Prefix matches the start of the base name, e.g. "DECK-".
	Prefix string
	Kind   Kind
}

// corpusRules is the path-keyed half of ResolveKind: the conventions this repo's
// docs tree already follows, written down so they stop being folklore.
//
// It is deliberately short and deliberately only names directories and prefixes a
// human chose. Every document that matches nothing is a report, which is the safe
// direction: a deck rendered as a report is a long document with big headings and
// every word legible, while a report rendered as a deck is forty near-empty
// landscape pages that read as a broken renderer.
var corpusRules = []pathRule{
	{Segment: "/decks/", Kind: KindDeck},
	{Segment: "/presentations/", Kind: KindDeck},
	{Prefix: "DECK-", Kind: KindDeck},
	{Prefix: "PRESENTATION-", Kind: KindDeck},
	{Segment: "/briefs/", Kind: KindBrief},
	{Segment: "docs/operator/", Kind: KindBrief},
	{Prefix: "BRIEF-", Kind: KindBrief},
	{Prefix: "operator-brief", Kind: KindBrief},
}

// DefaultKind is the corpus default: what a document that says nothing about
// itself and sits nowhere special is rendered as.
const DefaultKind = KindReport

// ResolveKind picks the Kind for a document. override is the operator's explicit
// answer ("" when there is none); path is the document's repo-relative path; src is
// its Markdown, read ONLY for the explicit marker in its head — never for its shape,
// its length, or how long its sections are.
func ResolveKind(override Kind, path, src string) (Decision, error) {
	if override != "" {
		if !override.Valid() {
			return Decision{}, fmt.Errorf("unknown document kind %q: want one of %s", override, KindNames())
		}
		return Decision{override, RuleOverride, "the caller named it explicitly"}, nil
	}
	if k, line, ok := markerKind(src); ok {
		return Decision{k, RuleMarker, fmt.Sprintf("%s:%d carries <!-- kind: %s -->", path, line, k)}, nil
	}
	if r, ok := matchCorpusRule(path); ok {
		return Decision{r.Kind, RulePath, fmt.Sprintf("path rule %s", r.describe())}, nil
	}
	return Decision{DefaultKind, RuleDefault, fmt.Sprintf(
		"no --kind, no <!-- kind: --> marker, and no path rule matched %s, so the corpus default applies", path)}, nil
}

func (r pathRule) describe() string {
	if r.Segment != "" {
		return fmt.Sprintf("%q in the path -> %s", r.Segment, r.Kind)
	}
	return fmt.Sprintf("base name starts with %q -> %s", r.Prefix, r.Kind)
}

// matchCorpusRule returns the first corpus rule that matches path. Order in
// corpusRules is the tie-break, and it is stable because the slice is a literal.
func matchCorpusRule(path string) (pathRule, bool) {
	p := filepath.ToSlash(path)
	base := filepath.Base(p)
	for _, r := range corpusRules {
		if r.Segment != "" && strings.Contains(p, r.Segment) {
			return r, true
		}
		if r.Prefix != "" && strings.HasPrefix(base, r.Prefix) {
			return r, true
		}
	}
	return pathRule{}, false
}

// markerMaxLines bounds the marker search to the document's head. A marker further
// down is not front matter, and scanning the whole file would let the word appear
// inside a fenced code block — a document ABOUT this package would then change how
// it renders by quoting its own documentation.
const markerMaxLines = 40

// markerKind reads an explicit `<!-- kind: deck -->` from the head of src and
// returns the kind and its 1-based line. A marker naming something that is not a
// kind is ignored rather than fatal: the document still renders, under the next
// rule down, and `fak docrender kind` shows which rule fired.
func markerKind(src string) (Kind, int, bool) {
	for i, ln := range strings.Split(src, "\n") {
		if i >= markerMaxLines {
			break
		}
		ln = strings.TrimSpace(ln)
		if !strings.HasPrefix(ln, "<!--") || !strings.HasSuffix(ln, "-->") {
			continue
		}
		body := strings.TrimSpace(strings.TrimSuffix(strings.TrimPrefix(ln, "<!--"), "-->"))
		key, val, ok := strings.Cut(body, ":")
		if !ok || strings.TrimSpace(strings.ToLower(key)) != "kind" {
			continue
		}
		if k, err := ParseKind(val); err == nil {
			return k, i + 1, true
		}
	}
	return "", 0, false
}

// CorpusRuleLines renders the corpus rule as sorted human-readable lines, so the
// verb can print the table instead of the reader having to find this file.
func CorpusRuleLines() []string {
	out := make([]string, 0, len(corpusRules))
	for _, r := range corpusRules {
		out = append(out, r.describe())
	}
	sort.Strings(out)
	return out
}
