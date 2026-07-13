package quality

import (
	"fmt"
	"strings"
	"unicode"
	"unicode/utf8"
)

// ClaimGrounding is the claim-level grounding oracle for executive reports
// (#4551): every claim sentence the engine text asserts must be supported by the
// case's evidence snippets. It inverts GroundingRubric's direction — that oracle
// checks the report CONTAINS the required phrases; this one checks the report
// asserts NOTHING the evidence does not back, which is the direction fabricated
// claims enter from.
//
// Evidence travels in the case as Rubric.Required: this oracle reads each
// required entry as one allowed source snippet (not as a must-appear phrase).
// Claims are the ". "/newline-split sentences of eng.Text.
//
// Grounding rule (deterministic, documented): a claim is GROUNDED iff a single
// evidence snippet contains at least groundingOverlap (60%) of the claim's
// distinct significant tokens. A significant token is a lowercased run of
// letters/digits/'%' that is at least 4 runes long or carries a digit — so
// figures like "12%" and "q3" always count while connective filler ("the",
// "and", "was") never decides grounding. A claim with no significant tokens
// asserts nothing checkable and is trivially grounded.
//
// Score = grounded claims / total claims; Pass iff Score >= Rubric.MinScore
// (default 1: every claim must be grounded). On failure Detail names the FIRST
// ungrounded claim — localizing the hallucination, per the spine contract.
//
// Edge behavior (defined and tested): an empty report asserts no claims and
// passes with score 1; a non-empty report judged against an empty evidence set
// fails closed with score 0 (an unsupported assertion is a hallucination, not a
// skipped check).
type ClaimGrounding struct{}

func (ClaimGrounding) Name() string { return "claim-grounding" }
func (ClaimGrounding) Kind() string { return "rubric" }

func init() { Register(ClaimGrounding{}) }

// groundingOverlap is the fraction of a claim's distinct significant tokens a
// single evidence snippet must contain for the claim to count as grounded.
const groundingOverlap = 0.6

func (ClaimGrounding) Judge(_ Trace, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "claim-grounding", Kind: "rubric", Pass: true, Score: 1}
	claims := splitClaims(eng.Text)
	if len(claims) == 0 {
		v.Detail = "report asserts no claims; nothing to ground"
		return v
	}
	evidence := newEvidenceSet(c.Rubric.Required)
	if len(evidence) == 0 {
		v.Pass = false
		v.Score = 0
		v.Detail = fmt.Sprintf("no evidence snippets declared; first ungrounded claim: %q", claims[0])
		return v
	}
	grounded := 0
	firstUngrounded := ""
	for _, cl := range claims {
		if evidence.grounds(cl) {
			grounded++
		} else if firstUngrounded == "" {
			firstUngrounded = cl
		}
	}
	v.Score = float64(grounded) / float64(len(claims))
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: every claim must be grounded
	}
	if v.Score < min {
		v.Pass = false
		v.Detail = fmt.Sprintf("claim grounding %.2f < %.2f (%d/%d grounded); first ungrounded claim: %q",
			v.Score, min, grounded, len(claims), firstUngrounded)
		return v
	}
	if firstUngrounded != "" {
		v.Detail = fmt.Sprintf("claim grounding %.2f >= %.2f (%d/%d grounded; tolerated ungrounded: %q)",
			v.Score, min, grounded, len(claims), firstUngrounded)
		return v
	}
	v.Detail = fmt.Sprintf("all %d claim(s) grounded in evidence", len(claims))
	return v
}

// splitClaims splits report text into claim sentences on newlines and ". ",
// trimming whitespace and a trailing period and dropping empties. Figures like
// "12.5%" survive intact because the split requires a space after the period.
func splitClaims(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		for _, part := range strings.Split(line, ". ") {
			cl := strings.TrimSpace(strings.TrimSuffix(strings.TrimSpace(part), "."))
			if cl != "" {
				out = append(out, cl)
			}
		}
	}
	return out
}

// evidenceSet is the tokenized allowed source snippets a report's claims are
// checked against. Empty/blank snippets are dropped at construction so they can
// never ground a claim.
type evidenceSet []evidenceSnippet

type evidenceSnippet struct {
	tokens map[string]bool
}

func newEvidenceSet(snippets []string) evidenceSet {
	out := make(evidenceSet, 0, len(snippets))
	for _, s := range snippets {
		toks := groundingTokens(s)
		if len(toks) == 0 {
			continue
		}
		set := make(map[string]bool, len(toks))
		for _, t := range toks {
			set[t] = true
		}
		out = append(out, evidenceSnippet{tokens: set})
	}
	return out
}

// grounds reports whether some single snippet contains at least
// groundingOverlap of the claim's distinct significant tokens.
func (e evidenceSet) grounds(claim string) bool {
	sig := map[string]bool{}
	for _, t := range groundingTokens(claim) {
		if significantToken(t) {
			sig[t] = true
		}
	}
	if len(sig) == 0 {
		return true // nothing checkable asserted
	}
	need := groundingOverlap * float64(len(sig))
	for _, sn := range e {
		hit := 0
		for t := range sig {
			if sn.tokens[t] {
				hit++
			}
		}
		if float64(hit) >= need {
			return true
		}
	}
	return false
}

// groundingTokens lowercases s and splits it into runs of letters, digits, and
// '%' — the comparison alphabet for grounding.
func groundingTokens(s string) []string {
	return strings.FieldsFunc(strings.ToLower(s), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '%'
	})
}

// significantToken reports whether a token is load-bearing for grounding: at
// least 4 runes, or carrying a digit (figures are the most-fabricated content).
func significantToken(t string) bool {
	if utf8.RuneCountInString(t) >= 4 {
		return true
	}
	for _, r := range t {
		if unicode.IsDigit(r) {
			return true
		}
	}
	return false
}
