package scorecard

import "math"

// The A-F grade tables, as named functions so a card SELECTS one rather than copy-pasting
// thresholds. The Python family carries two variants in the wild; the third (vCache) lives
// in internal/vcachescore and is documented here but intentionally NOT consolidated in this
// pass (vcachescore is left untouched until the kernel has proven itself on a real card).

// GradeStd is the 90/80/70/60 table used by the 20-card majority
// (e.g. code_quality_scorecard.py:411, guardrsi.go:571). Default when Messages.Grade is nil.
func GradeStd(score float64) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 80:
		return "B"
	case score >= 70:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

// GradeStrict is the 95/85/75/60 table used by the conflation and intent-literal cards
// (conflation_scorecard.py:218). A provenance-honesty card grades on a stricter curve: a
// B there starts at 85, not 80.
func GradeStrict(score float64) string {
	switch {
	case score >= 95:
		return "A"
	case score >= 85:
		return "B"
	case score >= 75:
		return "C"
	case score >= 60:
		return "D"
	default:
		return "F"
	}
}

// GradeVCache is the 90/75/60/40 table internal/vcachescore uses (score.go grade()). It is
// here for documentation + a future consolidation pass; no card in this package selects it.
func GradeVCache(score float64) string {
	switch {
	case score >= 90:
		return "A"
	case score >= 75:
		return "B"
	case score >= 60:
		return "C"
	case score >= 40:
		return "D"
	default:
		return "F"
	}
}

// Round1 rounds to one decimal place, matching Python round(x, 1) for the corpus.score the
// control-pane reads. Python's round is banker's rounding, but the scorecard scores never
// land on an exact .x5 half (they are means of integer/penalty arithmetic), so half-to-even
// vs half-away-from-zero cannot diverge here; math.Round (half away from zero) is faithful.
func Round1(x float64) float64 {
	return math.Round(x*10) / 10
}
