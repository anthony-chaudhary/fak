package quality

import (
	"fmt"
	"strings"
)

// drDecisionRelevance is the placement-side complement to MaterialOmission
// (#4553): where omission catches a material decision DROPPED from a report,
// this oracle catches one BURIED. An executive report ranks by decision
// relevance when the decisions a reader must act on lead the report; a report
// that opens with low-signal noise and parks the key decision in an appendix
// reads complete to an omission check yet still fails the executive, so it must
// fail a gate here.
//
// Report model (deterministic, documented): the engine text is split into
// ordered sections. A section starts at a markdown-style header line (leading
// '#'); any content before the first header is an implicit preamble section.
// A report with no header lines at all is split into blank-line-separated
// paragraphs instead, so an unsectioned report still has a "top". The FIRST
// section is the top/priority section.
//
// The material decisions travel on the case as Rubric.Required. Entries with a
// "decision:" prefix (the MaterialItems encoding) carry the decision text after
// the prefix; entries prefixed with the other material categories (win, risk,
// blocker) belong to other oracles and are ignored here; any other non-empty
// entry is a decision taken verbatim, so plain Required lists work unchanged.
// A decision is matched case-insensitively as a substring of a section's
// header + body; its FIRST occurrence decides its placement.
//
// Score = decisions placed in the top section / total decisions; Pass iff
// Score >= Rubric.MinScore (default 1: every material decision must lead). On
// failure Detail names each buried decision and the section it was found in —
// localizing the burial — and a decision absent from the report entirely also
// counts against the score (whatever else it is, it is not in the top section).
type drDecisionRelevance struct{}

func (drDecisionRelevance) Name() string { return "decision-relevance" }
func (drDecisionRelevance) Kind() string { return "rubric" }

func init() { Register(drDecisionRelevance{}) }

func (drDecisionRelevance) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "decision-relevance", Kind: "rubric", Pass: true, Score: 1}
	decisions := drDecisionItems(c.Rubric.Required)
	if len(decisions) == 0 {
		v.Detail = "no material decisions declared; nothing to rank"
		return v
	}
	sections := drParseSections(eng.Text)
	if len(sections) == 0 {
		return rubricFail(v, fmt.Sprintf("empty report: no top section for %d material decision(s); first: %q",
			len(decisions), decisions[0]))
	}
	top := sections[0]
	inTop := 0
	var misplaced []string
	for _, d := range decisions {
		sec, found := drLocate(sections, d)
		switch {
		case found && sec.Position == top.Position:
			inTop++
		case found:
			misplaced = append(misplaced, fmt.Sprintf("decision %q buried in %s, not the top section", d, drDescribe(sec)))
		default:
			misplaced = append(misplaced, fmt.Sprintf("decision %q absent from the report", d))
		}
	}
	min, short := rubricScore(&v, c, inTop, len(decisions))
	if short {
		v.Detail = fmt.Sprintf("decision relevance %.2f < %.2f (%d/%d material decisions lead %s); %s",
			v.Score, min, inTop, len(decisions), drDescribe(top), strings.Join(misplaced, "; "))
		return v
	}
	if len(misplaced) > 0 {
		v.Detail = fmt.Sprintf("decision relevance %.2f >= %.2f (tolerated: %s)",
			v.Score, min, strings.Join(misplaced, "; "))
		return v
	}
	v.Detail = fmt.Sprintf("all %d material decision(s) lead %s", len(decisions), drDescribe(top))
	return v
}

// drSection is one ordered block of a sectioned report: its 1-based position,
// its header title ("" for a preamble or paragraph block), and its body text.
type drSection struct {
	Position int
	Title    string
	Body     string
}

// drParseSections splits report text into ordered sections on markdown-style
// header lines; content before the first header is an implicit preamble
// section. A report with no headers falls back to blank-line-separated
// paragraphs so plain prose still has a defined top section. Blank blocks are
// dropped — an empty report yields no sections.
func drParseSections(text string) []drSection {
	lines := strings.Split(text, "\n")
	hasHeader := false
	for _, l := range lines {
		if drIsHeader(l) {
			hasHeader = true
			break
		}
	}
	if !hasHeader {
		return drParagraphSections(text)
	}
	var out []drSection
	cur := drSection{}
	flush := func() {
		if cur.Title != "" || strings.TrimSpace(cur.Body) != "" {
			cur.Position = len(out) + 1
			out = append(out, cur)
		}
	}
	for _, l := range lines {
		if drIsHeader(l) {
			flush()
			cur = drSection{Title: strings.TrimSpace(strings.TrimLeft(strings.TrimSpace(l), "#"))}
			continue
		}
		cur.Body += l + "\n"
	}
	flush()
	return out
}

// drParagraphSections splits headerless report text into blank-line-separated
// paragraph sections, in order, dropping blank blocks.
func drParagraphSections(text string) []drSection {
	var out []drSection
	for _, block := range strings.Split(text, "\n\n") {
		if strings.TrimSpace(block) == "" {
			continue
		}
		out = append(out, drSection{Position: len(out) + 1, Body: block})
	}
	return out
}

// drIsHeader reports whether a line is a markdown-style section header.
func drIsHeader(line string) bool {
	return strings.HasPrefix(strings.TrimSpace(line), "#")
}

// drDecisionItems extracts the material decisions from Rubric.Required:
// "decision:"-prefixed entries carry the text after the prefix, entries
// prefixed with another known material category (win, risk, blocker) are other
// oracles' items and are skipped, and any other non-empty entry is a decision
// taken verbatim. Prefix matching is case-insensitive; a colon inside ordinary
// decision text stays literal.
func drDecisionItems(required []string) []string {
	var out []string
	for _, r := range required {
		r = strings.TrimSpace(r)
		if r == "" {
			continue
		}
		if i := strings.Index(r, ":"); i >= 0 {
			switch strings.ToLower(strings.TrimSpace(r[:i])) {
			case "decision":
				if text := strings.TrimSpace(r[i+1:]); text != "" {
					out = append(out, text)
				}
				continue
			case "win", "risk", "blocker":
				continue
			}
		}
		out = append(out, r)
	}
	return out
}

// drLocate returns the FIRST section whose header or body contains the
// decision text, case-insensitively. First occurrence decides placement: a
// decision that leads the top section and is also echoed in an appendix is
// placed correctly.
func drLocate(sections []drSection, decision string) (drSection, bool) {
	needle := strings.ToLower(decision)
	for _, s := range sections {
		if strings.Contains(strings.ToLower(s.Title+"\n"+s.Body), needle) {
			return s, true
		}
	}
	return drSection{}, false
}

// drDescribe renders a section for a Detail message: its position and, when it
// has one, its header title.
func drDescribe(s drSection) string {
	if s.Title != "" {
		return fmt.Sprintf("section %d (%q)", s.Position, s.Title)
	}
	return fmt.Sprintf("section %d", s.Position)
}
