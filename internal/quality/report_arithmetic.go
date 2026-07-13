package quality

import (
	"fmt"
	"math"
	"regexp"
	"sort"
	"strconv"
	"strings"
)

// ReportArithmetic is the numeric-claim validator for executive reports (#4557):
// a rubric oracle that parses the figures a report ASSERTS — "12% week over
// week", "38 of 40", a trend word — and checks each against the period's
// ground-truth numbers instead of trusting the report's own prose. It is the
// arithmetic-layer mirror of the spine's increased-vs-decreased decode defect:
// a report can be perfectly fluent and still state a percentage that does not
// follow from prior/current, a trend word that contradicts the delta's sign, a
// denominator that is not the period's, or a percentage where a raw count
// belongs. Each of those fails here with a Detail pinpointing the first bad
// claim (the claim text plus expected vs stated), so "the numbers looked off"
// localizes to one checkable assertion.
//
// Ground truth travels IN the case, as a small "key: value" block in
// Reference.Text (see ArithmeticGroundTruth):
//
//	prior: 100
//	current: 112
//	denominator: 40
//
// Checks applied to every parsed claim, in report order:
//
//   - Percentage consistency: a stated "P%" must match
//     round((current-prior)/prior*100) within a tolerance. A trend word next to
//     the figure signs it ("decreased 12%" states -12); a bare percentage is
//     compared by magnitude.
//   - Trend-word consistency: "increased" requires current > prior,
//     "decreased" requires current < prior, "flat" requires equality.
//   - Denominator/units: an "N of M" claim must have N <= M and M equal to the
//     ground-truth denominator; a "%" on a ratio operand, or a week-over-week
//     change stated as a raw count instead of a percentage, is a unit mismatch.
//
// The oracle passes iff all parsed claims are consistent; a report with no
// parseable numeric claims passes vacuously (the grounding rubric, not this
// oracle, owns "the required figure is missing"). Numbers are read as plain
// decimals — grouped figures like "1,200" are out of scope for this layer.
type ReportArithmetic struct{}

func (ReportArithmetic) Name() string { return "report-arithmetic" }
func (ReportArithmetic) Kind() string { return "rubric" }

func init() { Register(ReportArithmetic{}) }

// arithPctTolerance is how far a stated percentage may sit from the rounded
// ground-truth delta and still count as consistent: half a point, i.e. the
// stated figure must round to the same integer percentage.
const arithPctTolerance = 0.5

// ArithmeticGroundTruth renders the period's figures as the canonical
// "key: value" block a report-arithmetic case carries in its Reference.Text:
// the prior and current values of the reported metric and the reporting
// denominator an "N of M" claim must agree with. Keeping ground truth inside
// the case preserves the spine's replay contract — the numbers a report is
// judged against travel with the case, not with ambient state.
func ArithmeticGroundTruth(prior, current float64, denominator int) string {
	return "prior: " + arithNum(prior) +
		"\ncurrent: " + arithNum(current) +
		"\ndenominator: " + strconv.Itoa(denominator)
}

// periodFigures is the parsed ground truth for one reporting period. The has*
// flags distinguish "declared as zero" from "not declared" so a claim that
// needs a figure the case never declared fails conservatively instead of being
// checked against a phantom zero.
type periodFigures struct {
	prior, current, denominator          float64
	hasPrior, hasCurrent, hasDenominator bool
}

func (g periodFigures) hasAny() bool { return g.hasPrior || g.hasCurrent || g.hasDenominator }

// parsePeriodFigures reads the "key: value" ground-truth block. Unknown keys
// and non-numeric lines are ignored so the block can sit inside a larger
// reference text.
func parsePeriodFigures(text string) periodFigures {
	var g periodFigures
	for _, line := range strings.Split(text, "\n") {
		key, val, ok := strings.Cut(line, ":")
		if !ok {
			continue
		}
		f, err := strconv.ParseFloat(strings.TrimSpace(val), 64)
		if err != nil {
			continue
		}
		switch strings.ToLower(strings.TrimSpace(key)) {
		case "prior":
			g.prior, g.hasPrior = f, true
		case "current":
			g.current, g.hasCurrent = f, true
		case "denominator":
			g.denominator, g.hasDenominator = f, true
		}
	}
	return g
}

// numericClaim is one parsed assertion from a report: its verbatim text, its
// position (for first-bad-claim ordering), and the decoded fields its kind
// uses. Kinds: "trend" (a trend word, optionally carrying a figure), "pct" (a
// bare percentage), "ratio" (an "N of M" claim).
type numericClaim struct {
	start    int
	kind     string
	text     string
	trend    string // canonical "increased" / "decreased" / "flat"; "" when absent
	value    float64
	hasValue bool
	valuePct bool // "%" present on value
	n, m     float64
	unitBad  bool // "%" attached to a ratio operand
}

var (
	arithRatioRE = regexp.MustCompile(`(?i)(\d+(?:\.\d+)?)(\s*%)?\s+(?:out\s+of|of)\s+(\d+(?:\.\d+)?)(\s*%)?`)
	arithTrendRE = regexp.MustCompile(`(?i)\b(increased|grew|rose|decreased|declined|dropped|fell|flat|unchanged)\b(?:\s+(?:by\s+)?(\d+(?:\.\d+)?)(\s*%)?)?`)
	arithPctRE   = regexp.MustCompile(`(\d+(?:\.\d+)?)\s*%`)
)

// arithTrendWord folds the recognized trend vocabulary onto the three
// canonical directions the ground truth can witness.
func arithTrendWord(w string) string {
	switch w {
	case "increased", "grew", "rose":
		return "increased"
	case "decreased", "declined", "dropped", "fell":
		return "decreased"
	default: // flat, unchanged
		return "flat"
	}
}

// parseNumericClaims extracts the report's numeric assertions in order of
// appearance. Ratio claims are taken first, then trend claims, then leftover
// bare percentages; a span already consumed by an earlier claim is not
// double-counted (the "12%" inside "increased 12%" is one claim, not two).
func parseNumericClaims(text string) []numericClaim {
	var claims []numericClaim
	var spans [][2]int
	take := func(lo, hi int) bool {
		for _, s := range spans {
			if lo < s[1] && s[0] < hi {
				return false
			}
		}
		spans = append(spans, [2]int{lo, hi})
		return true
	}
	grp := func(loc []int, i int) string {
		if loc[2*i] < 0 {
			return ""
		}
		return text[loc[2*i]:loc[2*i+1]]
	}
	for _, loc := range arithRatioRE.FindAllStringSubmatchIndex(text, -1) {
		if !take(loc[0], loc[1]) {
			continue
		}
		n, _ := strconv.ParseFloat(grp(loc, 1), 64)
		m, _ := strconv.ParseFloat(grp(loc, 3), 64)
		claims = append(claims, numericClaim{
			start:   loc[0],
			kind:    "ratio",
			text:    text[loc[0]:loc[1]],
			n:       n,
			m:       m,
			unitBad: grp(loc, 2) != "" || grp(loc, 4) != "",
		})
	}
	for _, loc := range arithTrendRE.FindAllStringSubmatchIndex(text, -1) {
		if !take(loc[0], loc[1]) {
			continue
		}
		cl := numericClaim{
			start: loc[0],
			kind:  "trend",
			text:  text[loc[0]:loc[1]],
			trend: arithTrendWord(strings.ToLower(grp(loc, 1))),
		}
		if s := grp(loc, 2); s != "" {
			cl.value, _ = strconv.ParseFloat(s, 64)
			cl.hasValue = true
			cl.valuePct = grp(loc, 3) != ""
		}
		claims = append(claims, cl)
	}
	for _, loc := range arithPctRE.FindAllStringSubmatchIndex(text, -1) {
		if !take(loc[0], loc[1]) {
			continue
		}
		val, _ := strconv.ParseFloat(grp(loc, 1), 64)
		claims = append(claims, numericClaim{
			start:    loc[0],
			kind:     "pct",
			text:     text[loc[0]:loc[1]],
			value:    val,
			hasValue: true,
			valuePct: true,
		})
	}
	sort.Slice(claims, func(i, j int) bool { return claims[i].start < claims[j].start })
	return claims
}

// check validates one claim against the ground truth. It returns "" when the
// claim is consistent, otherwise a detail string carrying the verbatim claim
// and expected-vs-stated — the localizing evidence a failing Verdict surfaces.
func (cl numericClaim) check(gt periodFigures) string {
	if cl.kind == "ratio" {
		if cl.unitBad {
			return fmt.Sprintf("claim %q: unit mismatch: an \"N of M\" claim takes raw counts, not percentages", cl.text)
		}
		if !gt.hasDenominator {
			return fmt.Sprintf("claim %q: ground truth declares no denominator to check against", cl.text)
		}
		if cl.n > cl.m {
			return fmt.Sprintf("claim %q: numerator %s exceeds denominator %s", cl.text, arithNum(cl.n), arithNum(cl.m))
		}
		if cl.m != gt.denominator {
			return fmt.Sprintf("claim %q: stated denominator %s, ground truth denominator is %s",
				cl.text, arithNum(cl.m), arithNum(gt.denominator))
		}
		return ""
	}
	// Trend and percentage claims ground against the prior -> current delta.
	if !gt.hasPrior || !gt.hasCurrent {
		return fmt.Sprintf("claim %q: ground truth declares no prior/current figures to check against", cl.text)
	}
	if gt.prior == 0 && cl.hasValue {
		return fmt.Sprintf("claim %q: ground-truth prior is 0; a percentage change is undefined", cl.text)
	}
	expectedTrend := "flat"
	switch {
	case gt.current > gt.prior:
		expectedTrend = "increased"
	case gt.current < gt.prior:
		expectedTrend = "decreased"
	}
	var expectedPct float64
	if gt.prior != 0 {
		expectedPct = math.Round((gt.current - gt.prior) / gt.prior * 100)
	}
	if cl.trend != "" && cl.trend != expectedTrend {
		return fmt.Sprintf("claim %q: stated trend %q but ground truth %s (prior %s -> current %s, delta %s%%)",
			cl.text, cl.trend, expectedTrend, arithNum(gt.prior), arithNum(gt.current), arithNum(expectedPct))
	}
	if !cl.hasValue {
		return ""
	}
	if !cl.valuePct {
		return fmt.Sprintf("claim %q: unit mismatch: change stated as a raw count, expected a percentage (%s%%)",
			cl.text, arithNum(math.Abs(expectedPct)))
	}
	stated, expected := cl.value, expectedPct
	switch cl.trend {
	case "decreased":
		stated = -stated // the trend word signs the stated figure
	case "":
		expected = math.Abs(expectedPct) // bare percentage: compare magnitudes
	}
	if math.Abs(stated-expected) > arithPctTolerance {
		return fmt.Sprintf("claim %q: stated %s%%, ground truth delta is %s%% (prior %s -> current %s)",
			cl.text, arithNum(cl.value), arithNum(expectedPct), arithNum(gt.prior), arithNum(gt.current))
	}
	return ""
}

// Judge parses the engine report's numeric claims and validates each against
// the ground-truth figures carried in the reference text. Score is the
// fraction of claims found consistent; on failure Detail pinpoints the FIRST
// inconsistent claim in report order.
func (ReportArithmetic) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "report-arithmetic", Kind: "rubric", Pass: true, Score: 1}
	gt := parsePeriodFigures(ref.Text)
	if !gt.hasAny() {
		gt = parsePeriodFigures(c.Reference.Text)
	}
	claims := parseNumericClaims(eng.Text)
	if len(claims) == 0 {
		v.Detail = "no numeric claims parsed from the report"
		return v
	}
	bad := 0
	var firstBad string
	for _, cl := range claims {
		if why := cl.check(gt); why != "" {
			bad++
			if firstBad == "" {
				firstBad = why
			}
		}
	}
	v.Score = float64(len(claims)-bad) / float64(len(claims))
	if bad > 0 {
		v.Pass = false
		v.Detail = firstBad
		return v
	}
	v.Detail = fmt.Sprintf("%d numeric claim(s) consistent with ground truth", len(claims))
	return v
}

// arithNum renders a figure the way a report states it: no exponent, no
// trailing zeros.
func arithNum(f float64) string { return strconv.FormatFloat(f, 'f', -1, 64) }
