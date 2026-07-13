package quality

import (
	"fmt"
	"math"
	"sort"
)

// sampler_distribution.go — #4530: validate a STOCHASTIC sampler by comparing
// its empirical distribution to the reference distribution, not one sample.
//
// Why one sample is insufficient: a single draw from a stochastic sampler is a
// point, not a distribution. A greedy/one-sample comparison collapses the
// sampler to its argmax token, and two very differently skewed samplers can
// share the same argmax — a sampler that puts 0.7 on the top token where the
// reference puts 0.4 emits the identical "most likely" token, so a one-sample
// gate stays green while the sampler is badly biased. Conversely, a faithful
// sampler legitimately varies draw to draw, so an exact one-sample equality
// gate flags healthy variance as failure. The correct gate is distributional:
// draw N samples under a fixed seed, build the empirical histogram, and compare
// it to the target with total-variation distance TV(p, q) = 0.5 * Σ|p_i − q_i|
// — the largest probability-mass disagreement any event can see. With N large
// (20000 draws here) sampling noise keeps a faithful sampler's TV an order of
// magnitude under the threshold, while any real skew exceeds it decisively.
// The fixed-seed PRNG makes the whole check hermetic and replay-identical.

// distributionTVThreshold is the pass gate: the empirical distribution must sit
// within this total-variation distance of the reference distribution. At 20000
// draws over a small vocab, faithful-sampler noise lands well under 0.01, so
// 0.02 separates noise from bias with margin on both sides.
const distributionTVThreshold = 0.02

// DistributionTV is the distribution-comparison oracle (#4530). The case's
// Reference trace carries the target distribution — the vocabulary in
// Reference.Tokens and the aligned target probabilities as the single row
// Reference.Logits[0] — and the engine trace carries the N drawn samples in
// its Tokens. Judge builds the empirical histogram over the reference
// vocabulary, computes the total-variation distance to the target (any
// out-of-vocabulary sample mass counts fully against the engine), and passes
// iff TV <= distributionTVThreshold. Score is 1 - TV.
type DistributionTV struct{}

func (DistributionTV) Name() string { return "distribution-tv" }
func (DistributionTV) Kind() string { return "distribution" }

func (DistributionTV) Judge(ref, eng Trace, _ QualityCase) Verdict {
	v := Verdict{Oracle: "distribution-tv", Kind: "distribution"}
	vocab := ref.Tokens
	if len(vocab) == 0 {
		v.Detail = "reference declares no vocabulary (Reference.Tokens is empty)"
		return v
	}
	if len(ref.Logits) != 1 || len(ref.Logits[0]) != len(vocab) {
		v.Detail = fmt.Sprintf("reference distribution malformed: want Logits as one row of %d probabilities aligned with Tokens", len(vocab))
		return v
	}
	p, err := normalizeProbs(ref.Logits[0])
	if err != nil {
		v.Detail = "reference distribution malformed: " + err.Error()
		return v
	}
	if len(eng.Tokens) == 0 {
		v.Detail = "engine emitted no samples; a distribution cannot be judged from zero draws"
		return v
	}
	q, oov := empiricalDistribution(eng.Tokens, vocab)
	tv := tvDistance(p, q) + 0.5*oov

	// Localize the largest per-token mass disagreement for the failure detail.
	worst := 0
	for i := range p {
		if math.Abs(p[i]-q[i]) > math.Abs(p[worst]-q[worst]) {
			worst = i
		}
	}

	v.Score = 1 - tv
	if v.Score < 0 {
		v.Score = 0
	}
	if tv <= distributionTVThreshold {
		v.Pass = true
		v.Detail = fmt.Sprintf("TV distance %.4f <= threshold %.4f over %d draws across %d-token vocab",
			tv, distributionTVThreshold, len(eng.Tokens), len(vocab))
		return v
	}
	v.Detail = fmt.Sprintf("TV distance %.4f > threshold %.4f over %d draws; worst token %q: reference p=%.4f, empirical q=%.4f",
		tv, distributionTVThreshold, len(eng.Tokens), vocab[worst], p[worst], q[worst])
	if oov > 0 {
		v.Detail += fmt.Sprintf("; out-of-vocab sample mass %.4f", oov)
	}
	return v
}

func init() { Register(DistributionTV{}) }

// DistributionCase builds a stochastic quality case whose reference is a
// DISTRIBUTION rather than a single golden trace: the vocabulary goes in
// Reference.Tokens and the aligned target probabilities as the one row
// Reference.Logits[0] — no change to the QualityCase struct. MaxTokens is the
// number of draws the engine runner must make and Seed pins its PRNG, so the
// case replays identically.
func DistributionCase(id string, vocab []string, probs []float64, draws int, seed int64) QualityCase {
	return QualityCase{
		Schema:  CaseSchema,
		ID:      id,
		Version: 1,
		Prompt:  "Draw from the pinned categorical distribution.",
		Params:  SamplingParams{Temperature: 1, MaxTokens: draws, Seed: seed},
		Reference: Trace{
			Tokens: append([]string(nil), vocab...),
			Logits: [][]float64{append([]float64(nil), probs...)},
		},
		Oracles: []string{"distribution-tv"},
	}
}

// SamplingRunner is the engine-side adapter for distribution cases: it draws
// Params.MaxTokens samples from ITS OWN categorical distribution (Tokens/Probs)
// using the case's fixed seed. A faithful engine carries the same distribution
// as the reference; a biased engine carries a skewed one — the deterministic
// mutant source the tests use to prove the gate trips.
type SamplingRunner struct {
	Label  string
	Tokens []string
	Probs  []float64
}

func (s SamplingRunner) Name() string {
	if s.Label != "" {
		return s.Label
	}
	return "engine-sampler"
}

func (s SamplingRunner) Run(c QualityCase) (Trace, error) {
	smp, err := NewCategoricalSampler(s.Tokens, s.Probs)
	if err != nil {
		return Trace{}, fmt.Errorf("sampling runner %q: %w", s.Name(), err)
	}
	n := c.Params.MaxTokens
	if n <= 0 {
		n = 1
	}
	return Trace{Runner: s.Name(), Tokens: smp.Sample(n, c.Params.Seed)}, nil
}

// CategoricalSampler draws tokens from a fixed categorical distribution with a
// deterministic PRNG, via inverse-CDF lookup over the cumulative probabilities.
type CategoricalSampler struct {
	tokens []string
	cum    []float64
}

// NewCategoricalSampler validates and normalizes the distribution. Negative
// probabilities, a zero-mass distribution, or a token/probability length
// mismatch are refused rather than sampled from.
func NewCategoricalSampler(tokens []string, probs []float64) (CategoricalSampler, error) {
	if len(tokens) == 0 || len(tokens) != len(probs) {
		return CategoricalSampler{}, fmt.Errorf("want equal non-zero token/probability counts, got %d/%d", len(tokens), len(probs))
	}
	p, err := normalizeProbs(probs)
	if err != nil {
		return CategoricalSampler{}, err
	}
	cum := make([]float64, len(p))
	var acc float64
	for i, pi := range p {
		acc += pi
		cum[i] = acc
	}
	cum[len(cum)-1] = 1 // absorb float rounding so every u in [0,1) lands
	return CategoricalSampler{tokens: append([]string(nil), tokens...), cum: cum}, nil
}

// Sample draws n tokens with a PRNG seeded from seed. Same (n, seed), same
// sequence — the hermetic replay contract.
func (s CategoricalSampler) Sample(n int, seed int64) []string {
	rng := splitMix64{state: uint64(seed)}
	out := make([]string, n)
	for i := range out {
		u := rng.float64()
		j := sort.SearchFloat64s(s.cum, u) // first index with cum[j] >= u
		if j >= len(s.tokens) {
			j = len(s.tokens) - 1
		}
		out[i] = s.tokens[j]
	}
	return out
}

// empiricalDistribution folds samples into a histogram over vocab, returning
// the per-token empirical probabilities and the fraction of samples that fell
// outside the vocabulary (which the oracle counts fully against the engine).
func empiricalDistribution(samples, vocab []string) (q []float64, oov float64) {
	idx := make(map[string]int, len(vocab))
	for i, t := range vocab {
		idx[t] = i
	}
	counts := make([]int, len(vocab))
	outside := 0
	for _, s := range samples {
		if i, ok := idx[s]; ok {
			counts[i]++
		} else {
			outside++
		}
	}
	n := float64(len(samples))
	q = make([]float64, len(vocab))
	for i, c := range counts {
		q[i] = float64(c) / n
	}
	return q, float64(outside) / n
}

// tvDistance is total-variation distance between two aligned distributions:
// 0.5 * Σ|p_i − q_i|, the maximum probability-mass disagreement any event can
// see. 0 means identical; 1 means disjoint support. Both slices must have the
// same length (the oracle aligns them over the reference vocabulary).
func tvDistance(p, q []float64) float64 {
	var sum float64
	for i := range p {
		sum += math.Abs(p[i] - q[i])
	}
	return 0.5 * sum
}

// normalizeProbs validates a probability vector (no negatives, positive total
// mass) and rescales it to sum to 1.
func normalizeProbs(probs []float64) ([]float64, error) {
	var sum float64
	for _, p := range probs {
		if p < 0 || math.IsNaN(p) || math.IsInf(p, 0) {
			return nil, fmt.Errorf("probability %v is not a finite non-negative number", p)
		}
		sum += p
	}
	if sum <= 0 {
		return nil, fmt.Errorf("distribution has no mass (sum %v)", sum)
	}
	out := make([]float64, len(probs))
	for i, p := range probs {
		out[i] = p / sum
	}
	return out, nil
}

// splitMix64 is a tiny deterministic PRNG (SplitMix64) kept in this file so the
// sampled sequence is pinned by the seed alone — independent of Go version and
// of any other package's PRNG use. Not cryptographic; this is a test harness.
type splitMix64 struct {
	state uint64
}

func (r *splitMix64) next() uint64 {
	r.state += 0x9e3779b97f4a7c15
	z := r.state
	z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9
	z = (z ^ (z >> 27)) * 0x94d049bb133111eb
	return z ^ (z >> 31)
}

// float64 returns a uniform value in [0, 1) with 53 bits of precision.
func (r *splitMix64) float64() float64 {
	return float64(r.next()>>11) / (1 << 53)
}
