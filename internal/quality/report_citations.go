package quality

import (
	"fmt"
	"regexp"
	"strings"
)

// CitationValidity is the citation MACHINERY oracle for executive reports
// (#4558): every citation marker the engine text carries must resolve to an
// evidence entry the case declares, and no marker may reference a
// non-existent/fabricated source. It is distinct from claim-grounding (#4551):
// that oracle judges whether the report's PROSE is backed by evidence content;
// this one judges whether the report's citation LINKS are wired — a report can
// be perfectly grounded and still cite [9] when only sources 1–3 exist, and a
// dangling citation is exactly how a fabricated source enters a rollup.
//
// Citation markers are parsed from eng.Text by citeMarkerRE: `[` + an id + `]`
// where the id is either bare digits ("[1]") or letters, an optional single
// hyphen, then digits ("[S3]", "[E-12]"). Anything else in brackets ("[sic]",
// "[]", "[1.2]") is not a citation marker and is skipped, never panicked on.
// Ids compare case-insensitively ("[s3]" resolves to evidence "S3").
//
// The allowed evidence ids travel in the case as Rubric.Required: each entry
// is one evidence id, optionally followed by ':' or whitespace and a
// link/description that is carried for humans but ignored for resolution —
// "1: https://ops/rollup-w28" declares id "1". CitationRubric is the helper
// constructor that builds such a spec.
//
// Score = resolved markers / total markers (each occurrence counts). Pass iff
// Score >= Rubric.MinScore (default 1: no dangling marker). On failure Detail
// names the FIRST dangling citation — the fabricated source, localized.
// Declared evidence that is never cited is a soft warning appended to Detail,
// never a failure: the hard failure of this oracle is only the dangling marker.
//
// Edge behavior (defined and tested): a report with zero citation markers has
// nothing to resolve and passes at score 1 with a Detail note (a case that
// REQUIRES citations should declare that via its grounding/omission rubric,
// which own must-appear content); markers judged against an empty evidence set
// all dangle and fail closed at score 0.
type CitationValidity struct{}

func (CitationValidity) Name() string { return "citation-validity" }
func (CitationValidity) Kind() string { return "rubric" }

func init() { Register(CitationValidity{}) }

// citeMarkerRE is the documented citation-marker grammar: "[" then either
// letters + optional single hyphen + digits (S3, E-12) or bare digits (1),
// then "]". Bracketed text outside this grammar is not a citation marker.
var citeMarkerRE = regexp.MustCompile(`\[([A-Za-z]+-?[0-9]+|[0-9]+)\]`)

func (CitationValidity) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "citation-validity", Kind: "rubric", Pass: true, Score: 1}
	markers := citeMarkers(eng.Text)
	known, declared := citeEvidenceIDs(c.Rubric.Required)
	if len(markers) == 0 {
		v.Detail = "report contains no citation markers; nothing to resolve"
		if warn := citeUncitedWarning(declared, nil); warn != "" {
			v.Detail += "; " + warn
		}
		return v
	}
	resolved := 0
	firstDangling := ""
	cited := map[string]bool{}
	for _, m := range markers {
		if known[m.id] {
			resolved++
			cited[m.id] = true
		} else if firstDangling == "" {
			firstDangling = m.raw
		}
	}
	v.Score = float64(resolved) / float64(len(markers))
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: every marker must resolve — no dangling citation
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("citation validity %.2f < %.2f (%d/%d markers resolved); first dangling citation: %s",
			v.Score, min, resolved, len(markers), firstDangling)
		return v
	}
	if firstDangling != "" {
		v.Detail = fmt.Sprintf("citation validity %.2f >= %.2f (%d/%d markers resolved; tolerated dangling citation: %s)",
			v.Score, min, resolved, len(markers), firstDangling)
	} else {
		v.Detail = fmt.Sprintf("all %d citation marker(s) resolved to declared evidence", len(markers))
	}
	if warn := citeUncitedWarning(declared, cited); warn != "" {
		v.Detail += "; " + warn
	}
	return v
}

// CitationRubric is the helper constructor for a case judged by
// citation-validity: it carries the allowed evidence entries in
// Rubric.Required — each entry an evidence id, optionally followed by ':' or
// whitespace and its link/description — and the pass threshold in MinScore
// (0 means the default: no dangling citation). Blank entries are dropped.
func CitationRubric(evidence []string, minScore float64) RubricSpec {
	kept := make([]string, 0, len(evidence))
	for _, e := range evidence {
		if strings.TrimSpace(e) != "" {
			kept = append(kept, e)
		}
	}
	return RubricSpec{Required: kept, MinScore: minScore}
}

// citeMarker is one parsed citation occurrence: the raw marker as it appears
// in the report ("[E-12]", for Detail messages) and its normalized id
// ("E-12", uppercased, for resolution).
type citeMarker struct {
	raw string
	id  string
}

// citeMarkers parses every citation marker occurrence from text, in report
// order. Bracketed text outside the marker grammar is skipped.
func citeMarkers(text string) []citeMarker {
	ms := citeMarkerRE.FindAllStringSubmatch(text, -1)
	out := make([]citeMarker, 0, len(ms))
	for _, m := range ms {
		out = append(out, citeMarker{raw: m[0], id: citeNormalizeID(m[1])})
	}
	return out
}

// citeEvidenceIDs parses Rubric.Required entries into the set of valid
// citation ids plus the ids in declaration order (deduplicated, for the
// uncited-evidence warning). The id is the entry's leading token — everything
// before the first ':' or whitespace — with optional surrounding brackets
// stripped, so "1", "[1]", "1: https://…", and "S3 weekly rollup" all declare
// the same shape of id. Blank entries are dropped.
func citeEvidenceIDs(required []string) (known map[string]bool, declared []string) {
	known = map[string]bool{}
	for _, r := range required {
		id := citeNormalizeID(citeLeadingToken(r))
		if id == "" || known[id] {
			continue
		}
		known[id] = true
		declared = append(declared, id)
	}
	return known, declared
}

// citeLeadingToken returns the entry's id token: leading/trailing space and
// one layer of surrounding brackets trimmed, then cut at the first ':' or
// whitespace (the remainder is the human-facing link/description).
func citeLeadingToken(entry string) string {
	s := strings.TrimSpace(entry)
	s = strings.TrimPrefix(s, "[")
	if i := strings.IndexAny(s, ": \t"); i >= 0 {
		s = s[:i]
	}
	s = strings.TrimSuffix(s, "]")
	return strings.TrimSpace(s)
}

// citeNormalizeID uppercases an id so markers and evidence entries compare
// case-insensitively. The hyphen stays significant: "E-12" and "E12" are
// distinct sources.
func citeNormalizeID(id string) string {
	return strings.ToUpper(strings.TrimSpace(id))
}

// citeUncitedWarning renders the soft (never failing) warning naming declared
// evidence ids no marker cited, in declaration order. Empty when everything
// declared was cited.
func citeUncitedWarning(declared []string, cited map[string]bool) string {
	var un []string
	for _, id := range declared {
		if !cited[id] {
			un = append(un, id)
		}
	}
	if len(un) == 0 {
		return ""
	}
	return fmt.Sprintf("uncited evidence (soft warning): %s", strings.Join(un, ", "))
}
