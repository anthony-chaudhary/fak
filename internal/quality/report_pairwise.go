package quality

import (
	"encoding/json"
	"fmt"
	"math"
	"strings"
)

// pairwiseReport is the baseline-versus-candidate pairwise report oracle (#4564,
// under epic #4509). It is the paired-regression axis of the report-quality
// layer: where grounding (#4551) and omission (#4552) judge ONE report against a
// fixed rubric, this judges a CANDIDATE report against a BASELINE report and asks
// the operator's real question — "is the candidate better, worse, or the same as
// the report it replaces?". Executive summaries need grounded and STABLE decision
// support; a candidate that reads fine but silently drops a grounded figure or a
// material category is a regression exact-string correctness cannot see, and a
// coarse end-benchmark sees too late to localize.
//
// The two reports travel on the case as the two decode paths the spine already
// captures: the REFERENCE trace text is the baseline, the ENGINE trace text is the
// candidate. The judge config travels in the Prompt, marked and JSON-encoded so it
// is a machine-checkable order rather than prose (the same convention
// instruction-following (#4561) uses):
//
//	PAIRWISE: {"dimensions":[{"name":"grounding","criteria":["12%","week over week"]}],"expect":"candidate","tie_margin":0}
//
// Each rubric DIMENSION is scored independently for both sides (the fraction of
// its criteria present, matched case-insensitively), the two side scores are
// compared into a per-dimension winner (baseline / candidate / tie — ties are
// allowed within tie_margin), and the weighted net margin across dimensions
// decides the OVERALL winner. The case asserts the expected direction in
// `expect` (default "candidate": the candidate is the proposed improvement); the
// oracle PASSES iff the observed overall winner matches. A known-improved pair
// therefore passes and a known-degraded pair fails — the paired-outcome contract.
//
// Side order is randomized (pairwiseSideOrder, seeded from the case) exactly as a
// pairwise judge presents its A/B pair, but the scorer is per-side and symmetric,
// so the OUTCOME is invariant to the order — the anti-position-bias property the
// side-order-invariance test proves. The presented order is recorded on the
// outcome so a paired verdict stays replayable. On failure the Detail names the
// FIRST dimension whose winner contradicts the expected direction — the first
// actionable divergence — and the spine wraps the full case (baseline text,
// candidate text, spec) into the scrubbed, replay-complete FailureBundle. Every
// per-dimension rationale (which criteria each side covered and missed) is
// preserved on the outcome so a paired verdict is auditable, not a bare number.
type pairwiseReport struct{}

func (pairwiseReport) Name() string { return "report-pairwise" }
func (pairwiseReport) Kind() string { return "rubric" }

func init() { Register(pairwiseReport{}) }

// The pairwise winner vocabulary — a closed set shared by per-dimension and
// overall outcomes and by the `expect` the case declares.
const (
	PairwiseCandidate = "candidate"
	PairwiseBaseline  = "baseline"
	PairwiseTie       = "tie"
)

// pairwiseSpecMarker prefixes the machine-checkable pairwise spec line inside a
// case's Prompt. Everything after the marker on that line is the JSON spec.
const pairwiseSpecMarker = "PAIRWISE:"

// PairwiseDimension is one rubric dimension scored independently for the baseline
// and the candidate report. Score = fraction of Criteria present in that side's
// text. Weight (default 1) sets its pull on the aggregate net margin.
type PairwiseDimension struct {
	Name     string   `json:"name"`
	Criteria []string `json:"criteria"`
	Weight   float64  `json:"weight,omitempty"`
}

// PairwiseSpec is the prompt-carried configuration of a baseline-vs-candidate
// evaluation: the rubric dimensions to score, the expected winner the case
// asserts, and the tie margin below which a score gap is called a tie. A zero
// TieMargin means an exact tie (equal scores) is the only tie.
type PairwiseSpec struct {
	Dimensions []PairwiseDimension `json:"dimensions"`
	Expect     string              `json:"expect,omitempty"`
	TieMargin  float64             `json:"tie_margin,omitempty"`
}

// PairwiseDimensionResult is the preserved rationale for one scored dimension:
// each side's score, the winner (ties allowed), and a human rationale naming
// which criteria each side covered and missed. It is the audit trail that makes
// a paired verdict reviewable rather than a bare number.
type PairwiseDimensionResult struct {
	Name           string  `json:"name"`
	BaselineScore  float64 `json:"baseline_score"`
	CandidateScore float64 `json:"candidate_score"`
	Winner         string  `json:"winner"`
	Rationale      string  `json:"rationale"`
}

// PairwiseOutcome is the full result of one baseline-vs-candidate evaluation: the
// overall winner, the presentation order the sides were judged in (the
// randomized-side-order evidence — the outcome is invariant to it), the weighted
// net margin (candidate minus baseline, positive favors the candidate), and every
// per-dimension rationale.
type PairwiseOutcome struct {
	Winner     string                    `json:"winner"`
	Order      string                    `json:"order"`
	NetMargin  float64                   `json:"net_margin"`
	Dimensions []PairwiseDimensionResult `json:"dimensions"`
}

func (pairwiseReport) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "report-pairwise", Kind: "rubric", Pass: true, Score: 1}
	spec, ok := parsePairwiseSpec(c.Prompt)
	if !ok || len(spec.Dimensions) == 0 {
		v.Detail = "no pairwise spec declared; nothing to compare"
		return v
	}
	// Baseline is the reference trace text; candidate is the engine trace text.
	out := EvaluatePairwise(ref.Text, eng.Text, spec, c.Params.Seed)
	expect := normalizePairwiseExpect(spec.Expect)
	consistent := 0
	for _, d := range out.Dimensions {
		if d.Winner == expect {
			consistent++
		}
	}
	v.Score = round3(float64(consistent) / float64(len(out.Dimensions)))
	if out.Winner != expect {
		v.Pass = false
		v.Detail = fmt.Sprintf("pairwise outcome %q != expected %q (side order %s, net margin %+.3f, %d/%d dimension(s) matched); first actionable divergence: %s",
			out.Winner, expect, out.Order, out.NetMargin, consistent, len(out.Dimensions), pairwiseFirstDivergence(out, expect))
		return v
	}
	v.Detail = fmt.Sprintf("pairwise outcome %q matches expected (side order %s, net margin %+.3f across %d dimension(s))",
		out.Winner, out.Order, out.NetMargin, len(out.Dimensions))
	return v
}

// EvaluatePairwise scores a candidate report against a baseline report across the
// spec's rubric dimensions and folds the per-dimension comparisons into an overall
// paired outcome. It is a pure function of its inputs — the same (baseline,
// candidate, spec, seed) always yields the same outcome — so a paired verdict
// replays. seed selects only the presentation order (recorded, not scored): the
// outcome is invariant to it.
func EvaluatePairwise(baseline, candidate string, spec PairwiseSpec, seed int64) PairwiseOutcome {
	out := PairwiseOutcome{Order: pairwiseSideOrder(seed)}
	// Pairwise evaluation is meaningful only when both sides and a declared rubric
	// are present. Missing evidence is inconclusive, never an implicit win.
	if strings.TrimSpace(baseline) == "" || strings.TrimSpace(candidate) == "" || len(spec.Dimensions) == 0 {
		out.Winner = "inconclusive"
		return out
	}
	tie := spec.TieMargin
	if tie < 0 {
		tie = 0
	}
	var weighted, totalWeight float64
	for _, dim := range spec.Dimensions {
		w := dim.Weight
		if w <= 0 {
			w = 1
		}
		bScore, bCov, bMiss := pairwiseDimScore(baseline, dim.Criteria)
		cScore, cCov, cMiss := pairwiseDimScore(candidate, dim.Criteria)
		winner := pairwiseWinner(bScore, cScore, tie)
		out.Dimensions = append(out.Dimensions, PairwiseDimensionResult{
			Name:           dim.Name,
			BaselineScore:  round3(bScore),
			CandidateScore: round3(cScore),
			Winner:         winner,
			Rationale:      pairwiseRationale(dim.Name, bScore, cScore, winner, bCov, bMiss, cCov, cMiss),
		})
		weighted += w * (cScore - bScore)
		totalWeight += w
	}
	raw := 0.0
	if totalWeight > 0 {
		raw = weighted / totalWeight
	}
	out.NetMargin = round3(raw)
	out.Winner = pairwiseWinner(0, raw, tie)
	return out
}

// pairwiseSideOrder is the deterministic, replayable stand-in for the randomized
// side order a pairwise judge presents its A/B pair in. The scorer is per-side and
// symmetric, so the OUTCOME is invariant to this order (proven by the invariance
// test); recording it keeps the paired verdict replayable and documents which
// side led. It is stable under negative seeds (bit test, not modulo).
func pairwiseSideOrder(seed int64) string {
	if seed&1 == 0 {
		return "baseline-first"
	}
	return "candidate-first"
}

// pairwiseDimScore returns the fraction of a dimension's criteria present in the
// text (case-insensitive substring), plus the covered and missing criteria for the
// rationale. Empty criteria are skipped; a dimension with no usable criteria scores
// zero and reports nothing, so it can never manufacture a spurious win.
func pairwiseDimScore(text string, criteria []string) (score float64, covered, missing []string) {
	low := strings.ToLower(text)
	for _, cr := range criteria {
		cr = strings.TrimSpace(cr)
		if cr == "" {
			continue
		}
		if strings.Contains(low, strings.ToLower(cr)) {
			covered = append(covered, cr)
		} else {
			missing = append(missing, cr)
		}
	}
	n := len(covered) + len(missing)
	if n == 0 {
		return 0, nil, nil
	}
	return float64(len(covered)) / float64(n), covered, missing
}

// pairwiseWinner classifies a baseline-vs-candidate score gap: the candidate wins
// when it clears the baseline by more than the tie margin, the baseline wins on the
// symmetric case, and any gap within the margin (including equal scores) is a tie.
func pairwiseWinner(baseline, candidate, tie float64) string {
	switch d := candidate - baseline; {
	case d > tie:
		return PairwiseCandidate
	case d < -tie:
		return PairwiseBaseline
	default:
		return PairwiseTie
	}
}

// pairwiseRationale renders the preserved, auditable rationale for one dimension:
// each side's score with the criteria it covered and missed, and the winner.
func pairwiseRationale(name string, baseline, candidate float64, winner string, bCov, bMiss, cCov, cMiss []string) string {
	return fmt.Sprintf("%s: baseline %.2f (covered %s; missing %s) vs candidate %.2f (covered %s; missing %s) -> %s",
		name, baseline, joinOrNone(bCov), joinOrNone(bMiss), candidate, joinOrNone(cCov), joinOrNone(cMiss), winner)
}

// pairwiseFirstDivergence localizes the FIRST actionable divergence when the
// overall outcome contradicts the expected direction: the first dimension (in
// declared order) whose winner is not the expected side. If no single dimension
// diverges (a weighting/tie effect), it reports the aggregate net-margin verdict.
func pairwiseFirstDivergence(out PairwiseOutcome, expect string) string {
	for _, d := range out.Dimensions {
		if d.Winner != expect {
			return d.Rationale
		}
	}
	return fmt.Sprintf("net margin %+.3f yields %q, not %q", out.NetMargin, out.Winner, expect)
}

// parsePairwiseSpec extracts the first well-formed pairwise spec from the prompt.
// Lines without the marker, and marker lines whose JSON payload does not parse, are
// skipped deterministically — a malformed order is not an adjudicable comparison.
// The bool reports whether a spec was found at all, so a case with no spec is
// distinguished from one whose spec parsed empty.
func parsePairwiseSpec(prompt string) (PairwiseSpec, bool) {
	for _, line := range strings.Split(prompt, "\n") {
		rest, ok := strings.CutPrefix(strings.TrimSpace(line), pairwiseSpecMarker)
		if !ok {
			continue
		}
		var s PairwiseSpec
		if err := json.Unmarshal([]byte(strings.TrimSpace(rest)), &s); err != nil {
			continue
		}
		return s, true
	}
	return PairwiseSpec{}, false
}

// normalizePairwiseExpect resolves the case's declared expectation to the closed
// winner vocabulary, defaulting to "candidate" (the candidate is the proposed
// improvement, so absent an explicit expectation it must beat the baseline).
func normalizePairwiseExpect(expect string) string {
	switch strings.ToLower(strings.TrimSpace(expect)) {
	case PairwiseBaseline:
		return PairwiseBaseline
	case PairwiseTie:
		return PairwiseTie
	default:
		return PairwiseCandidate
	}
}

func joinOrNone(items []string) string {
	if len(items) == 0 {
		return "none"
	}
	return strings.Join(items, ", ")
}

func round3(f float64) float64 { return math.Round(f*1000) / 1000 }
