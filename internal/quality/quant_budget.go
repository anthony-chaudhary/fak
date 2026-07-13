package quality

import (
	"fmt"
	"strings"
)

// quant_budget.go — #4540: quantized decode quality within a DECLARED budget.
//
// A quantized engine (int8/int4 weights, quantized KV cache) is NOT expected to
// reproduce the full-precision reference token-for-token — rounding error
// legitimately flips an occasional low-margin token. An exact differential gate
// (greedy-token-diff) therefore reds every quantized build, while no gate at all
// lets a broken quantization ship because "some drift is expected". The correct
// gate is a TOLERANCE: measure the token-agreement rate between the quantized
// engine's stream and the full-precision reference over the whole sequence, and
// pass iff agreement >= the case's declared budget (default
// quantDefaultAgreementBudget). A healthy quantization sits above the budget
// with margin; a defective one (bad scales, clipped outliers, a broken kernel)
// drops decisively below it — and the verdict reports the measured agreement vs
// the budget plus the first disagreeing token, so the failure localizes instead
// of being observed only as "the quantized model feels worse".

// quantBudgetName and quantBudgetKind identify the oracle in the registry.
// Kind "statistical" marks the verdict as a tolerance/rate gate over the whole
// sequence rather than an exact comparison or a rubric score.
const (
	quantBudgetName = "quantization-budget"
	quantBudgetKind = "statistical"
)

// quantDefaultAgreementBudget is the pass gate applied when a case declares no
// budget of its own: the quantized engine must agree with the full-precision
// reference on at least this fraction of token positions.
const quantDefaultAgreementBudget = 0.98

// quantBudgetSteps is the sequence length QuantBudgetCase pins. Long enough
// that an agreement rate is a meaningful statistic (a 1% defect is ~2 tokens,
// a 5% defect ~10), small enough to stay a fast hermetic test.
const quantBudgetSteps = 200

// QuantBudget is the quantization-budget oracle (#4540): a statistical
// differential gate that scores the engine's token-agreement rate against the
// reference and passes iff it meets the case's declared budget. The budget is
// declared per case in Rubric.MinScore (a fraction in (0, 1]); a case that
// declares none gets quantDefaultAgreementBudget. Score is the measured
// agreement rate; on failure FirstDivergence pins the first disagreeing
// position and Detail reports measured agreement vs budget.
type QuantBudget struct{}

func (QuantBudget) Name() string { return quantBudgetName }
func (QuantBudget) Kind() string { return quantBudgetKind }

func init() { Register(QuantBudget{}) }

func (QuantBudget) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: quantBudgetName, Kind: quantBudgetKind}
	if len(ref.Tokens) == 0 {
		v.Detail = "reference carries no tokens; an agreement rate cannot be measured"
		return v
	}
	budget := quantBudgetFor(c)
	matches, total, first := quantAgreement(ref.Tokens, eng.Tokens)
	rate := float64(matches) / float64(total)
	v.Score = rate
	if rate >= budget {
		v.Pass = true
		v.Detail = fmt.Sprintf("token agreement %.4f >= declared budget %.4f (%d/%d positions matched)",
			rate, budget, matches, total)
		return v
	}
	if first >= 0 {
		v.FirstDivergence = &Divergence{
			Index:     first,
			Reference: tokenAt(ref.Tokens, first),
			Engine:    tokenAt(eng.Tokens, first),
		}
	}
	v.Detail = fmt.Sprintf("token agreement %.4f < declared budget %.4f (%d/%d positions matched); quantization exceeded its quality budget",
		rate, budget, matches, total)
	return v
}

// quantBudgetFor resolves the declared agreement budget for a case: the case's
// Rubric.MinScore when it is a usable fraction in (0, 1], else the package
// default. The budget travels WITH the case, so a looser int4 corpus and a
// tighter int8 corpus gate under their own declared tolerances.
func quantBudgetFor(c QualityCase) float64 {
	if b := c.Rubric.MinScore; b > 0 && b <= 1 {
		return b
	}
	return quantDefaultAgreementBudget
}

// quantAgreement compares two token streams position-wise. total is the longer
// length — a position present in only one stream counts as a disagreement, so
// a quantized engine cannot buy agreement by truncating. first is the index of
// the first disagreeing position, or -1 when the streams are identical.
func quantAgreement(ref, eng []string) (matches, total, first int) {
	total = len(ref)
	if len(eng) > total {
		total = len(eng)
	}
	first = -1
	n := len(ref)
	if len(eng) < n {
		n = len(eng)
	}
	for i := 0; i < n; i++ {
		if ref[i] == eng[i] {
			matches++
		} else if first < 0 {
			first = i
		}
	}
	if first < 0 && len(ref) != len(eng) {
		first = n
	}
	return matches, total, first
}

// QuantBudgetCase builds a hermetic quantization-budget case: a deterministic
// quantBudgetSteps-token full-precision reference stream, judged only by the
// quantization-budget oracle under the declared budget (0 declares none, so
// the oracle applies its default).
func QuantBudgetCase(id string, budget float64) QualityCase {
	toks := quantBudgetReferenceTokens(quantBudgetSteps)
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  "Decode the pinned sequence under quantized weights.",
		Params:  SamplingParams{Temperature: 0, MaxTokens: quantBudgetSteps},
		Reference: Trace{
			Tokens: toks,
			Text:   strings.Join(toks, " "),
		},
		Oracles: []string{quantBudgetName},
		Rubric:  RubricSpec{MinScore: budget},
	}
}

// quantBudgetReferenceTokens is the deterministic full-precision stream: n
// position-tagged tokens, so any perturbation at any index is visible and the
// stream replays identically everywhere.
func quantBudgetReferenceTokens(n int) []string {
	toks := make([]string, n)
	for i := range toks {
		toks[i] = fmt.Sprintf("tok%03d", i)
	}
	return toks
}

// quantBudgetPerturb models a quantization rounding flip at one position: the
// full-precision token is replaced by a distinct near-neighbor.
func quantBudgetPerturb(t string) string { return t + "~q" }

// QuantBudgetEngine is the engine-side adapter modeling a quantized decode of
// the case's reference: it replays the reference stream with every
// MismatchEvery-th position flipped by quantBudgetPerturb (rounding error).
// MismatchEvery <= 0 is a bit-faithful quantization; a large value models a
// healthy build whose drift sits inside the budget; a small value models a
// defective quantization whose drift exceeds it. This is the deterministic
// mutant source the tests use to prove the gate trips.
type QuantBudgetEngine struct {
	Label         string
	MismatchEvery int
}

func (e QuantBudgetEngine) Name() string {
	if e.Label != "" {
		return e.Label
	}
	return "engine-quantized"
}

func (e QuantBudgetEngine) Run(c QualityCase) (Trace, error) {
	ref := c.Reference.Tokens
	toks := make([]string, len(ref))
	for i, t := range ref {
		if e.MismatchEvery > 0 && (i+1)%e.MismatchEvery == 0 {
			toks[i] = quantBudgetPerturb(t)
		} else {
			toks[i] = t
		}
	}
	return Trace{Runner: e.Name(), Tokens: toks, Text: strings.Join(toks, " ")}, nil
}
