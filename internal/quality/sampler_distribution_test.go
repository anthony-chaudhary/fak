package quality

import (
	"math"
	"reflect"
	"strings"
	"testing"
)

// distTestVocab is the shared 5-token vocabulary and target distribution for
// the distribution-tv tests. The biased mutant keeps the SAME argmax token
// ("alpha") so a one-sample/greedy comparison cannot see the skew — that is the
// exact gap #4530 closes.
func distTestVocab() (vocab []string, target, biased []float64) {
	vocab = []string{"alpha", "beta", "gamma", "delta", "epsilon"}
	target = []float64{0.4, 0.3, 0.15, 0.1, 0.05}
	biased = []float64{0.7, 0.1, 0.1, 0.05, 0.05}
	return vocab, target, biased
}

const (
	distTestSeed  = int64(424242)
	distTestDraws = 20000
)

func distOracles(t *testing.T, c QualityCase) []Oracle {
	t.Helper()
	os, err := Lookup(c.Oracles)
	if err != nil {
		t.Fatalf("Lookup(%v): %v", c.Oracles, err)
	}
	return os
}

// TestDistributionFaithfulSamplerPasses is the happy path: an engine sampling
// from the SAME target distribution stays within the TV threshold over 20000
// fixed-seed draws and passes with a quantified detail.
func TestDistributionFaithfulSamplerPasses(t *testing.T) {
	vocab, target, _ := distTestVocab()
	c := DistributionCase("dist-faithful", vocab, target, distTestDraws, distTestSeed)
	eng := SamplingRunner{Label: "engine-faithful", Tokens: vocab, Probs: target}
	res, err := RunCase(c, ReferenceRunner{}, eng, distOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if !res.Pass {
		t.Fatalf("faithful sampler should pass; got %s", Explain(res))
	}
	if res.FailureBundle != nil {
		t.Fatalf("clean pass must not carry a failure bundle: %+v", res.FailureBundle)
	}
	v := res.Verdicts[0]
	if v.Oracle != "distribution-tv" || v.Kind != "distribution" {
		t.Errorf("verdict identity = %q/%q, want distribution-tv/distribution", v.Oracle, v.Kind)
	}
	// Score = 1 - TV, so a pass means Score >= 1 - threshold.
	if v.Score < 1-distributionTVThreshold {
		t.Errorf("score %.4f < %.4f; TV exceeded the threshold on a faithful sampler", v.Score, 1-distributionTVThreshold)
	}
	if !strings.Contains(v.Detail, "TV distance") || !strings.Contains(v.Detail, "<= threshold 0.0200") {
		t.Errorf("pass detail must quantify TV vs threshold; got %q", v.Detail)
	}
}

// TestDistributionBiasedSamplerFails is the defect witness: an engine sampling
// from a skewed distribution (0.7 on the top token where the reference puts
// 0.4) lands TV ≈ 0.30, an order of magnitude over the 0.02 threshold, and the
// failure detail quantifies both numbers and localizes the worst token.
func TestDistributionBiasedSamplerFails(t *testing.T) {
	vocab, target, biased := distTestVocab()
	c := DistributionCase("dist-biased", vocab, target, distTestDraws, distTestSeed)
	eng := SamplingRunner{Label: "engine-biased", Tokens: vocab, Probs: biased}
	res, err := RunCase(c, ReferenceRunner{}, eng, distOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("biased sampler must not pass; got %s", Explain(res))
	}
	fb := res.FailureBundle
	if fb == nil {
		t.Fatal("failing run must carry a failure bundle")
	}
	if fb.FailingOracle != "distribution-tv" {
		t.Errorf("failing oracle = %q, want distribution-tv", fb.FailingOracle)
	}
	v := res.Verdicts[0]
	if !strings.Contains(v.Detail, "> threshold 0.0200") || !strings.Contains(v.Detail, "worst token") {
		t.Errorf("fail detail must quantify TV > threshold and localize the worst token; got %q", v.Detail)
	}
	// TV = 1 - Score; the injected skew is TV ≈ 0.30, so pin it to a band that
	// proves the measurement, not just the boolean.
	tv := 1 - v.Score
	if tv < 0.25 || tv > 0.35 {
		t.Errorf("measured TV %.4f outside expected band [0.25, 0.35] for the injected skew", tv)
	}
}

// TestOneSampleComparisonIsInsufficient is the load-bearing proof of WHY this
// oracle exists (mirrors TestComparatorIsLoadBearing): the biased distribution
// shares the reference's argmax token, so a greedy one-token comparison — what
// a single-sample gate collapses to — PASSES the biased sampler, while
// distribution-tv over N draws catches it.
func TestOneSampleComparisonIsInsufficient(t *testing.T) {
	vocab, target, biased := distTestVocab()

	// One-sample gate: both samplers' greedy (argmax) decode is "alpha".
	oneRef := Trace{Tokens: []string{vocab[0]}}
	oneEng := Trace{Tokens: []string{vocab[0]}}
	if v := (GreedyTokenDiff{}).Judge(oneRef, oneEng, QualityCase{}); !v.Pass {
		t.Fatalf("setup: the one-sample comparison was expected to (wrongly) pass; got %+v", v)
	}

	// Distribution gate: the same biased sampler fails over 20000 draws.
	c := DistributionCase("dist-one-sample-gap", vocab, target, distTestDraws, distTestSeed)
	eng := SamplingRunner{Label: "engine-biased", Tokens: vocab, Probs: biased}
	res, err := RunCase(c, ReferenceRunner{}, eng, distOracles(t, c))
	if err != nil {
		t.Fatalf("RunCase: %v", err)
	}
	if res.Pass {
		t.Fatalf("distribution-tv must catch the bias the one-sample gate misses; got %s", Explain(res))
	}
}

// TestDistributionJudgeIsDeterministic proves the replay contract: the same
// case and runners produce byte-identical verdicts on every run (fixed-seed
// PRNG, pure Judge).
func TestDistributionJudgeIsDeterministic(t *testing.T) {
	vocab, target, _ := distTestVocab()
	c := DistributionCase("dist-replay", vocab, target, distTestDraws, distTestSeed)
	eng := SamplingRunner{Label: "engine-faithful", Tokens: vocab, Probs: target}
	first, err := RunCase(c, ReferenceRunner{}, eng, distOracles(t, c))
	if err != nil {
		t.Fatalf("first RunCase: %v", err)
	}
	second, err := RunCase(c, ReferenceRunner{}, eng, distOracles(t, c))
	if err != nil {
		t.Fatalf("second RunCase: %v", err)
	}
	if !reflect.DeepEqual(first.Verdicts, second.Verdicts) {
		t.Fatalf("verdicts differ across identical runs:\nfirst:  %+v\nsecond: %+v", first.Verdicts, second.Verdicts)
	}
}

// TestTVDistanceHelper pins the metric itself: identical distributions are at
// 0, disjoint ones at 1, and a known perturbation lands exactly where the
// formula says.
func TestTVDistanceHelper(t *testing.T) {
	p := []float64{0.4, 0.3, 0.15, 0.1, 0.05}
	if got := tvDistance(p, p); got != 0 {
		t.Errorf("tv(p,p) = %v, want 0", got)
	}
	if got := tvDistance([]float64{1, 0}, []float64{0, 1}); got != 1 {
		t.Errorf("tv(disjoint) = %v, want 1", got)
	}
	if got := tvDistance([]float64{0.5, 0.5}, []float64{0.6, 0.4}); math.Abs(got-0.1) > 1e-12 {
		t.Errorf("tv([.5 .5],[.6 .4]) = %v, want 0.1", got)
	}
}

// TestDistributionFailsClosedOnMalformedInput proves the oracle refuses rather
// than passes when the case carries no reference distribution, when the probs
// row is misaligned, or when the engine emitted no samples — and that samples
// outside the reference vocabulary count fully against the engine.
func TestDistributionFailsClosedOnMalformedInput(t *testing.T) {
	vocab, target, _ := distTestVocab()
	o := DistributionTV{}
	goodRef := Trace{Tokens: vocab, Logits: [][]float64{target}}
	samples := Trace{Tokens: []string{"alpha", "beta"}}

	if v := o.Judge(Trace{}, samples, QualityCase{}); v.Pass {
		t.Errorf("empty reference vocabulary must fail closed; got %+v", v)
	}
	if v := o.Judge(Trace{Tokens: vocab, Logits: [][]float64{{0.5, 0.5}}}, samples, QualityCase{}); v.Pass {
		t.Errorf("misaligned probability row must fail closed; got %+v", v)
	}
	if v := o.Judge(goodRef, Trace{}, QualityCase{}); v.Pass {
		t.Errorf("zero engine samples must fail closed; got %+v", v)
	}
	oov := o.Judge(goodRef, Trace{Tokens: []string{"zeta", "zeta", "zeta", "zeta"}}, QualityCase{})
	if oov.Pass {
		t.Errorf("all-out-of-vocab samples must fail; got %+v", oov)
	}
	if !strings.Contains(oov.Detail, "out-of-vocab") {
		t.Errorf("out-of-vocab failure must be named in the detail; got %q", oov.Detail)
	}
}
