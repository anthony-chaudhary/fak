package quality

import (
	"fmt"
	"strings"
	"unicode"
)

// mtConsistency is the multi-turn state-consistency oracle (#4549): across a
// dialog, a later turn must not contradict a fact committed by an earlier turn,
// and context carried into the conversation must be respected. A fluent
// assistant that says "the deploy window is tuesday" in turn 1 and "the deploy
// window is friday" in turn 5 must fail a gate, not merely read fine turn by
// turn — per-turn scoring cannot see the pair.
//
// Dialog model (deterministic, documented):
//   - The engine dialog travels in eng.Text: one turn per non-empty line, with
//     an optional single-word speaker tag ("user:", "assistant:", ...) stripped.
//   - Carried context travels in the case's Reference trace Text, same line
//     format. Its facts seed the ledger BEFORE the engine dialog is scanned, so
//     a turn contradicting a carried fact fails exactly like one contradicting
//     an earlier engine turn.
//   - Fact grammar: each turn splits into clauses on ". " and "; ". A clause
//     ending in "?" is a question and asserts nothing. A clause containing
//     " is not " asserts a NEGATED fact; otherwise a clause containing " is "
//     asserts a positive fact. Subject and value are lowercased and trimmed of
//     surrounding space and punctuation, so "Tuesday." re-affirms "tuesday".
//   - Consistency rule: the FIRST commitment for a subject governs. A later
//     positive assertion with a different value, a positive assertion of a value
//     earlier negated, or a negation of the committed value is a contradiction.
//     Re-affirming a committed fact is consistent; a new subject commits freely.
//
// Score = consistent engine assertions / total engine assertions; Pass iff
// Score >= Rubric.MinScore (default 1: no turn may contradict a committed
// fact). On failure, Detail names the contradiction PAIR — the contradicting
// clause and the committed clause it broke, with the subject and where each was
// asserted — and FirstDivergence localizes the contradicting turn (Index = the
// engine turn index, Reference = the committed clause, Engine = the
// contradicting clause), per the spine contract that a defect localizes to a
// step rather than being observed only in prose.
//
// Edge behavior (defined and tested): a dialog asserting no checkable facts has
// nothing to contradict and passes at score 1; carried context is trusted as-is
// (its own first commitment per subject governs; it is never judged).
type mtConsistency struct{}

func (mtConsistency) Name() string { return "multiturn-consistency" }
func (mtConsistency) Kind() string { return "rubric" }

func init() { Register(mtConsistency{}) }

func (mtConsistency) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "multiturn-consistency", Kind: "rubric", Pass: true, Score: 1}
	ledger := mtNewLedger()
	for _, turn := range mtTurns(ref.Text) {
		for _, f := range mtTurnFacts(turn, -1) {
			ledger.commit(f) // carried context is trusted as-is; first commitment governs
		}
	}
	turns := mtTurns(eng.Text)
	total, contradicted := 0, 0
	var firstNew, firstOld *mtFact
	for i, turn := range turns {
		for _, f := range mtTurnFacts(turn, i) {
			total++
			if prior := ledger.commit(f); prior != nil {
				contradicted++
				if firstNew == nil {
					fc, pc := f, *prior
					firstNew, firstOld = &fc, &pc
				}
			}
		}
	}
	if total == 0 {
		v.Detail = fmt.Sprintf("dialog of %d turn(s) asserts no checkable facts; nothing to contradict", len(turns))
		return v
	}
	v.Score = float64(total-contradicted) / float64(total)
	min := c.Rubric.MinScore
	if min == 0 {
		min = 1 // default: no turn may contradict a committed fact
	}
	if v.Score < min {
		v.Pass = false
		v.FirstDivergence = &Divergence{Index: firstNew.Turn, Reference: firstOld.Clause, Engine: firstNew.Clause}
		v.Detail = fmt.Sprintf("multi-turn consistency %.2f < %.2f (%d/%d assertions contradict a committed fact); turn %d asserts %q, contradicting %q committed at %s (subject %q)",
			v.Score, min, contradicted, total, firstNew.Turn, firstNew.Clause, firstOld.Clause, mtFactSite(*firstOld), firstNew.Subject)
		return v
	}
	if firstNew != nil {
		v.Detail = fmt.Sprintf("multi-turn consistency %.2f >= %.2f (tolerated contradiction: turn %d %q vs %s %q)",
			v.Score, min, firstNew.Turn, firstNew.Clause, mtFactSite(*firstOld), firstOld.Clause)
		return v
	}
	v.Detail = fmt.Sprintf("all %d assertion(s) across %d turn(s) consistent with the committed facts", total, len(turns))
	return v
}

// mtFact is one fact assertion a turn makes: the normalized subject it commits,
// the normalized value asserted (or denied), and where it was said — kept with
// the original clause so a contradiction reports the exact words.
type mtFact struct {
	Subject string // normalized subject the fact commits
	Value   string // normalized value asserted (or denied)
	Negated bool   // true for an " is not " assertion
	Turn    int    // engine turn index; -1 = carried context
	Clause  string // the original clause, for Detail/FirstDivergence
}

// mtLedger is the committed-fact state of the dialog so far: for each subject,
// at most one positive committed value and any values explicitly denied. The
// first commitment for a subject governs; a contradiction never overwrites it.
type mtLedger struct {
	positive map[string]mtFact
	negated  map[string]map[string]mtFact
}

func mtNewLedger() *mtLedger {
	return &mtLedger{positive: map[string]mtFact{}, negated: map[string]map[string]mtFact{}}
}

// commit records f in the ledger and returns nil, or returns the previously
// committed fact f contradicts, leaving the ledger unchanged.
func (l *mtLedger) commit(f mtFact) *mtFact {
	if f.Negated {
		if p, ok := l.positive[f.Subject]; ok && p.Value == f.Value {
			return &p // denies the committed value
		}
		if _, ok := l.negated[f.Subject][f.Value]; !ok {
			if l.negated[f.Subject] == nil {
				l.negated[f.Subject] = map[string]mtFact{}
			}
			l.negated[f.Subject][f.Value] = f
		}
		return nil
	}
	if p, ok := l.positive[f.Subject]; ok {
		if p.Value != f.Value {
			return &p // asserts a different value than committed
		}
		return nil // re-affirmation
	}
	if n, ok := l.negated[f.Subject][f.Value]; ok {
		return &n // asserts a value an earlier turn denied
	}
	l.positive[f.Subject] = f
	return nil
}

// mtTurns splits dialog text into turns: one per non-empty line, with a leading
// single-word speaker tag stripped.
func mtTurns(text string) []string {
	var out []string
	for _, line := range strings.Split(text, "\n") {
		t := mtStripSpeaker(strings.TrimSpace(line))
		if t != "" {
			out = append(out, t)
		}
	}
	return out
}

// mtStripSpeaker removes a leading "speaker:" tag when the tag is a single run
// of letters; anything else (including a colon inside ordinary prose with
// spaces before it) stays literal.
func mtStripSpeaker(turn string) string {
	i := strings.Index(turn, ":")
	if i <= 0 {
		return turn
	}
	for _, r := range turn[:i] {
		if !unicode.IsLetter(r) {
			return turn
		}
	}
	return strings.TrimSpace(turn[i+1:])
}

// mtTurnFacts extracts the fact assertions one turn makes, per the documented
// grammar. turnIdx travels onto each fact (-1 for carried context).
func mtTurnFacts(turn string, turnIdx int) []mtFact {
	var out []mtFact
	for _, clause := range mtClauses(turn) {
		if strings.HasSuffix(clause, "?") {
			continue // a question asserts nothing
		}
		lower := strings.ToLower(clause)
		negated := false
		i := strings.Index(lower, " is not ")
		cut := len(" is not ")
		if i < 0 {
			i = strings.Index(lower, " is ")
			cut = len(" is ")
		} else {
			negated = true
		}
		if i < 0 {
			continue
		}
		subject := mtNorm(lower[:i])
		value := mtNorm(lower[i+cut:])
		if subject == "" || value == "" {
			continue
		}
		out = append(out, mtFact{Subject: subject, Value: value, Negated: negated, Turn: turnIdx, Clause: clause})
	}
	return out
}

// mtClauses splits a turn into clauses on ". " and "; ", trimming whitespace
// and dropping empties. Figures like "12.5%" survive because the sentence
// split requires a space after the period.
func mtClauses(turn string) []string {
	var out []string
	for _, part := range strings.Split(turn, ". ") {
		for _, cl := range strings.Split(part, "; ") {
			cl = strings.TrimSpace(cl)
			if cl != "" {
				out = append(out, cl)
			}
		}
	}
	return out
}

// mtNorm lowercases a fact part and trims surrounding space and punctuation,
// so "Tuesday." and "tuesday" commit the same value.
func mtNorm(s string) string {
	return strings.Trim(strings.ToLower(strings.TrimSpace(s)), ".,!?:;")
}

// mtFactSite renders where a fact was committed, for Detail messages.
func mtFactSite(f mtFact) string {
	if f.Turn < 0 {
		return "carried context"
	}
	return fmt.Sprintf("turn %d", f.Turn)
}
