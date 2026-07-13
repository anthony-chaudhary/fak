package quality

import (
	"fmt"
	"math"
	"sort"
)

// topk_topp.go — #4526: exercise top-k and top-p boundary semantics.
//
// The candidate-set truncation math is where fluent-but-wrong sampling starts:
// an off-by-one at the k-th slot, a nondeterministic tie-break, or an exclusive
// p-boundary comparison all change WHICH tokens are even eligible, yet every
// downstream draw still looks plausible. This oracle validates the truncation
// itself: the case's Reference carries the full vocabulary (Reference.Tokens)
// and one aligned logits row (Reference.Logits[0]); the engine emits the
// candidate set it kept as its Trace.Tokens; and Judge compares that against
// the correctly computed boundary set for the case's Params.TopK / Params.TopP.
//
// The truncation contract, stated precisely (every rule below is deterministic
// and closed over the case inputs):
//
//   - Probabilities are the softmax of the logits row. Tokens are ranked by
//     DESCENDING probability; TIES are resolved by ASCENDING vocabulary index
//     (the token appearing earlier in Reference.Tokens wins the higher rank).
//   - top-k keeps EXACTLY the k highest-ranked tokens. TopK <= 0 disables the
//     filter (the zero value of SamplingParams.TopK means unset); TopK >= vocab
//     keeps the entire vocabulary.
//   - top-p keeps the SMALLEST prefix of the (post-top-k, renormalized) ranking
//     whose cumulative probability reaches p; a token landing EXACTLY on the p
//     boundary is included and truncation happens immediately after it (the
//     comparison is >=, with a tiny epsilon so float rounding cannot flip an
//     exact boundary into an exclusion). TopP <= 0 disables the filter (the
//     zero value of SamplingParams.TopP means unset); TopP >= 1 keeps the whole
//     surviving set, since only the full set carries the target mass.
//   - The kept set is an ORDERED trace — the survivors in ranking order — so
//     the comparison is a plain token diff and a defect localizes to the first
//     slot that disagrees, exactly like the greedy differential.
type TopKTopPBoundary struct{}

func (TopKTopPBoundary) Name() string { return "topk-topp-boundary" }
func (TopKTopPBoundary) Kind() string { return "differential" }

func init() { Register(TopKTopPBoundary{}) }

// topkpBoundaryEps absorbs float rounding at the p boundary: a token whose
// cumulative mass lands within this distance of p is treated as reaching it,
// so an exactly-on-boundary token is included rather than dropped to rounding.
const topkpBoundaryEps = 1e-12

func (TopKTopPBoundary) Judge(ref, eng Trace, c QualityCase) Verdict {
	v := Verdict{Oracle: "topk-topp-boundary", Kind: "differential"}
	if len(ref.Tokens) == 0 {
		v.Detail = "reference declares no vocabulary (Reference.Tokens is empty)"
		return v
	}
	if len(ref.Logits) != 1 || len(ref.Logits[0]) != len(ref.Tokens) {
		v.Detail = fmt.Sprintf("reference logits malformed: want one row of %d values aligned with Reference.Tokens", len(ref.Tokens))
		return v
	}
	want := topkpKeptSet(ref.Tokens, ref.Logits[0], c.Params.TopK, c.Params.TopP)
	got := eng.Tokens

	n := len(want)
	if len(got) < n {
		n = len(got)
	}
	for i := 0; i < n; i++ {
		if want[i] != got[i] {
			v.FirstDivergence = &Divergence{Index: i, Reference: want[i], Engine: got[i]}
			v.Detail = fmt.Sprintf("kept set diverged at slot %d: reference keeps %q, engine kept %q (top_k=%d, top_p=%g)",
				i, want[i], got[i], c.Params.TopK, c.Params.TopP)
			return v
		}
	}
	if len(want) != len(got) {
		v.FirstDivergence = &Divergence{Index: n, Reference: tokenAt(want, n), Engine: tokenAt(got, n)}
		v.Detail = fmt.Sprintf("kept-set size diverged: reference keeps %d candidate(s), engine kept %d (top_k=%d, top_p=%g)",
			len(want), len(got), c.Params.TopK, c.Params.TopP)
		return v
	}
	v.Pass = true
	v.Detail = fmt.Sprintf("engine kept the exact %d-candidate boundary set for top_k=%d, top_p=%g",
		len(want), c.Params.TopK, c.Params.TopP)
	return v
}

// TopKTopPCase builds a boundary case for this oracle: the vocabulary goes in
// Reference.Tokens, the aligned logits row as the single row Reference.Logits[0],
// and the k/p under test in Params — no change to the QualityCase struct. The
// engine runner under test must emit the candidate set it kept, in ranking
// order, as its Trace.Tokens.
func TopKTopPCase(id string, vocab []string, logits []float64, topK int, topP float64) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  "Truncate the candidate set at the declared top-k / top-p boundary.",
		Params:  SamplingParams{Temperature: 1, TopK: topK, TopP: topP, MaxTokens: 1},
		Reference: Trace{
			Tokens: append([]string(nil), vocab...),
			Logits: [][]float64{append([]float64(nil), logits...)},
		},
		Oracles: []string{"topk-topp-boundary"},
	}
}

// topkpKeptSet computes the correct truncated candidate set for one logits row
// under the contract documented on TopKTopPBoundary: softmax probabilities,
// descending-probability ranking with ties broken by ascending vocabulary
// index, exact top-k, then the inclusive-boundary nucleus over the renormalized
// survivors. The result is the survivors in ranking order.
func topkpKeptSet(vocab []string, logits []float64, k int, p float64) []string {
	probs := topkpSoftmax(logits)
	order := make([]int, len(probs))
	for i := range order {
		order[i] = i
	}
	// Stable sort by descending probability: equal-probability tokens keep
	// their ascending vocabulary-index order — the documented tie-break.
	sort.SliceStable(order, func(a, b int) bool { return probs[order[a]] > probs[order[b]] })

	if k > 0 && k < len(order) {
		order = order[:k]
	}
	if p > 0 && p < 1 {
		var total float64
		for _, i := range order {
			total += probs[i]
		}
		cut := len(order)
		var cum float64
		for j, i := range order {
			cum += probs[i] / total
			if cum >= p-topkpBoundaryEps {
				cut = j + 1
				break
			}
		}
		order = order[:cut]
	}

	out := make([]string, len(order))
	for j, i := range order {
		out[j] = vocab[i]
	}
	return out
}

// topkpSoftmax converts a logits row to probabilities with the usual max
// subtraction for numeric stability. It is deterministic: same row, same
// probabilities, on every run.
func topkpSoftmax(logits []float64) []float64 {
	max := math.Inf(-1)
	for _, l := range logits {
		if l > max {
			max = l
		}
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, l := range logits {
		e := math.Exp(l - max)
		out[i] = e
		sum += e
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
