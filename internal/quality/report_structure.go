package quality

import (
	"fmt"
	"strings"
)

// structWithoutStyle is the executive-report STRUCTURE oracle (#4559): a report
// must carry every required structural section — Summary, Risks, Decisions,
// Next actions, whatever the case declares — WITHOUT the check freezing prose
// style. The case's Rubric.Required entries are read as section ANCHORS, not as
// must-appear phrases: the oracle checks that each anchor exists as a section
// heading in the engine text, and deliberately ignores how the prose inside the
// sections is worded, how the sections are ordered, and how the headings are
// decorated (markdown "## Risks", bold "**Risks**", bare "RISKS", numbered
// "2. Risks", or inline "Risks: ..." all count).
//
// The complementary direction matters just as much: a section anchor mentioned
// mid-prose ("the top risks were reviewed") is NOT a section. A heading is only
// recognized on a line of its own (after decoration stripping the line equals
// the anchor) or as an inline label (the text before the line's first colon
// equals the anchor). That is what lets this oracle fail a report that dissolved
// its Risks section into a paragraph while passing one that merely reworded it.
//
// Score = present sections / required sections; Pass iff Score >= Rubric.MinScore
// (default 1: every declared section must be present). On failure the Detail
// names each missing section — the fix is a list, not a style critique.
type structWithoutStyle struct{}

func (structWithoutStyle) Name() string { return "structure-without-style" }
func (structWithoutStyle) Kind() string { return "rubric" }

func init() { Register(structWithoutStyle{}) }

func (structWithoutStyle) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "structure-without-style", Kind: "rubric", Pass: true, Score: 1}
	anchors := structAnchors(c.Rubric.Required)
	if len(anchors) == 0 {
		v.Detail = "no section anchors declared; nothing to enforce"
		return v
	}
	headings := structHeadingSet(eng.Text)
	present := 0
	var missing []string
	for _, a := range anchors {
		if headings[structNormalizeAnchor(a)] {
			present++
		} else {
			missing = append(missing, a)
		}
	}
	min, short := rubricScore(&v, c, present, len(anchors))
	if short {
		v.Detail = fmt.Sprintf("structure score %.2f < %.2f; missing required section(s): %s",
			v.Score, min, structQuoteList(missing))
		return v
	}
	if len(missing) > 0 {
		v.Detail = fmt.Sprintf("structure score %.2f >= %.2f; tolerated missing section(s): %s",
			v.Score, min, structQuoteList(missing))
		return v
	}
	v.Detail = fmt.Sprintf("all %d required section(s) present (style unconstrained)", len(anchors))
	return v
}

// structAnchors trims and drops empty Rubric.Required entries; an empty entry is
// not a section anchor.
func structAnchors(required []string) []string {
	out := make([]string, 0, len(required))
	for _, r := range required {
		if r = strings.TrimSpace(r); r != "" {
			out = append(out, r)
		}
	}
	return out
}

// structHeadingSet scans the report line by line and collects every normalized
// heading the text carries — the whole-line form ("## Risks", "RISKS",
// "**Risks**", "3) Risks") and the inline-label form ("Risks: nobody owns the
// migration."). Prose sentences produce no anchor-sized keys, so a mid-sentence
// mention of a section word never registers as a section.
func structHeadingSet(text string) map[string]bool {
	set := map[string]bool{}
	for _, line := range strings.Split(text, "\n") {
		for _, k := range structHeadingKeys(line) {
			set[k] = true
		}
	}
	return set
}

// structHeadingKeys derives the candidate heading keys of one line: the full
// decoration-stripped line, and — when the line carries a colon — the label
// before the first colon. Either may match an anchor; both are normalized.
func structHeadingKeys(line string) []string {
	stripped := strings.TrimSpace(line)
	if stripped == "" {
		return nil
	}
	// Strip leading heading decoration: markdown hashes, bullets, quote marks,
	// setext underline characters, then any "1." / "2)" numbering.
	stripped = strings.TrimLeft(stripped, "#>*-=• \t")
	stripped = strings.TrimSpace(structTrimNumbering(stripped))
	if stripped == "" {
		return nil
	}
	var keys []string
	if whole := structNormalizeAnchor(stripped); whole != "" {
		keys = append(keys, whole)
	}
	if i := strings.Index(stripped, ":"); i > 0 {
		if label := structNormalizeAnchor(stripped[:i]); label != "" {
			keys = append(keys, label)
		}
	}
	return keys
}

// structTrimNumbering strips a leading "12." or "12)" ordinal so numbered
// section headings normalize to their anchor text.
func structTrimNumbering(s string) string {
	i := 0
	for i < len(s) && s[i] >= '0' && s[i] <= '9' {
		i++
	}
	if i == 0 || i >= len(s) {
		return s
	}
	if s[i] == '.' || s[i] == ')' {
		return s[i+1:]
	}
	return s
}

// structNormalizeAnchor lowercases, strips surrounding emphasis/heading
// punctuation, and collapses internal whitespace, so "  **Next  Actions:**  "
// and "next actions" compare equal.
func structNormalizeAnchor(s string) string {
	s = strings.ToLower(strings.TrimSpace(s))
	s = strings.Trim(s, ":*_#=- \t")
	return strings.Join(strings.Fields(s), " ")
}

// structQuoteList renders section names for a Detail as a comma-separated list
// of quoted anchors, in the case's declaration order.
func structQuoteList(names []string) string {
	parts := make([]string, len(names))
	for i, n := range names {
		parts[i] = fmt.Sprintf("%q", n)
	}
	return strings.Join(parts, ", ")
}
