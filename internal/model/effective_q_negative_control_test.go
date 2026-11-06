package model

// Audit witness tests for exact proposal-distribution (effective q) acceptance
// and request/position-keyed draws in the ordinary drafting path:
// BuildSpecRejectionPlan over TargetProbabilityRows with UnitCDF24 draft
// evidence is the weighted route the serve-path sampled MTP round uses.
//
// fak-test:runtime medium est=3s lane=default

import (
	"math"
	"reflect"
	"testing"
)

// effectiveQLogProbs mirrors the drafter's own LogProbs construction
// (qwen35_mtp_draft.go proposalTokenEvidence): float32 log of each processed-q
// probability, -Inf where the probability is zero.
func effectiveQLogProbs(q []float32) []float32 {
	out := make([]float32, len(q))
	for i, p := range q {
		if p > 0 {
			out[i] = float32(math.Log(float64(p)))
		} else {
			out[i] = float32(math.Inf(-1))
		}
	}
	return out
}

// effectiveQEvidence builds one admissible stochastic draft evidence row bound
// to token under processed distribution q, exactly as the sampled drafter does.
func effectiveQEvidence(token int, q []float32) DraftTokenEvidence {
	return DraftTokenEvidence{
		Token:       token,
		LogProbs:    effectiveQLogProbs(q),
		Provenance:  DraftEvidenceSampledProcessedQ,
		ProcessedQ:  append([]float32(nil), q...),
		SamplingLaw: DraftSamplingLawQwen38UnitCDF24,
	}
}

// effectiveQTargetRow wraps full-vocabulary target weights with the mass they
// actually sum to, so validateSpecProbabilityRow sees an exact match.
func effectiveQTargetRow(weights []float64) SpecProbabilityRow {
	row := SpecProbabilityRow{Weights: append([]float64(nil), weights...)}
	for _, w := range row.Weights {
		row.Mass += w
	}
	return row
}

// effectiveQKey folds a seed to a non-zero request key the way the planner's
// sampledMTPRequestKey does (zero is the unkeyed sentinel).
func effectiveQKey(seed uint64) uint64 {
	key := SplitMix64(seed)
	if key == 0 {
		key = 1
	}
	return key
}

// TestEffectiveQNegativeControlMisreportedQChangesAcceptance is the negative
// control: a deliberately misreported q must change the acceptance decision,
// the reported acceptance probability, and the outcome distribution. If the
// math ignored q, both plans would agree.
func TestEffectiveQNegativeControlMisreportedQChangesAcceptance(t *testing.T) {
	const vocab = 4
	pBoundary := effectiveQTargetRow([]float64{0.5, 0.3, 0.15, 0.05})
	pBonus := effectiveQTargetRow([]float64{0.2, 0.2, 0.3, 0.3})

	// True effective q: the distribution the drafter actually sampled from.
	// alpha = min(1, 0.5/0.4) = 1, so the drafted token always accepts.
	qTrue := []float32{0.4, 0.3, 0.2, 0.1}
	// Deliberately misreported q: overstates the drafted token.
	// alpha ~= min(1, 0.5/0.9) ~= 0.5556.
	qWrong := []float32{0.9, 0.05, 0.04, 0.01}

	build := func(q []float32, key uint64, step int) (SpecRejectionPlan, error) {
		in := SpecRejectionPlanInput{
			DraftTokens:           []int{0},
			DraftEvidence:         []DraftTokenEvidence{effectiveQEvidence(0, q)},
			TargetProbabilityRows: []SpecProbabilityRow{pBoundary, pBonus},
			RequestKey:            key,
			StepOffset:            step,
		}
		return BuildSpecRejectionPlan(in)
	}

	probe, err := build(qWrong, effectiveQKey(7), 0)
	if err != nil {
		t.Fatalf("probe plan: %v", err)
	}
	alphaWrong := float64(probe.AcceptProb[0])
	if !(alphaWrong > 0.4 && alphaWrong < 0.7) {
		t.Fatalf("misreported-q acceptance probability = %v, want ~0.5556; q is not entering the ratio", alphaWrong)
	}

	// Find a request key whose accept coin falls strictly between the two
	// alphas: accepted under the true q, rejected under the misreported q.
	var key uint64
	for seed := uint64(1); ; seed++ {
		candidate := effectiveQKey(seed)
		coin := KeyedUniform64(candidate, DomainAcceptCoin, 0)
		if coin > alphaWrong+1e-12 && coin < 1.0-1e-12 {
			key = candidate
			break
		}
	}

	planTrue, err := build(qTrue, key, 0)
	if err != nil {
		t.Fatalf("true-q plan: %v", err)
	}
	planWrong, err := build(qWrong, key, 0)
	if err != nil {
		t.Fatalf("wrong-q plan: %v", err)
	}

	if got := float64(planTrue.AcceptProb[0]); math.Abs(got-1.0) > 1e-6 {
		t.Fatalf("true-q acceptance probability = %v, want 1", got)
	}
	if !reflect.DeepEqual(planTrue.AcceptedTokens, []int{0}) || planTrue.RejectedAt != -1 {
		t.Fatalf("true-q plan = %+v, want draft token accepted", planTrue)
	}
	if planWrong.RejectedAt != 0 || len(planWrong.AcceptedTokens) != 0 {
		t.Fatalf("misreported-q plan = %+v, want rejection at position 0", planWrong)
	}
	// The lossless residual (p - q)+ has zero mass on the drafted token under
	// the misreported q, so the correction must come from another token.
	if planWrong.CorrectionToken < 0 || planWrong.CorrectionToken >= vocab || planWrong.CorrectionToken == 0 {
		t.Fatalf("misreported-q correction token = %d, want one of {1,2,3}", planWrong.CorrectionToken)
	}
	if planTrue.AcceptProb[0]-planWrong.AcceptProb[0] < 0.3 {
		t.Fatalf("acceptance probabilities %v vs %v do not diverge; q is not driving the ratio",
			planTrue.AcceptProb[0], planWrong.AcceptProb[0])
	}

	// Distribution check: across many independent positions the accept fraction
	// must move with q (all accepted under the true q, ~alphaWrong under the lie).
	const positions = 512
	acceptedTrue, acceptedWrong := 0, 0
	for i := 0; i < positions; i++ {
		pt, err := build(qTrue, key, i)
		if err != nil {
			t.Fatalf("true-q position %d: %v", i, err)
		}
		if pt.RejectedAt == -1 {
			acceptedTrue++
		}
		pw, err := build(qWrong, key, i)
		if err != nil {
			t.Fatalf("wrong-q position %d: %v", i, err)
		}
		if pw.RejectedAt == -1 {
			acceptedWrong++
		}
	}
	fracTrue := float64(acceptedTrue) / positions
	fracWrong := float64(acceptedWrong) / positions
	if fracTrue != 1.0 {
		t.Fatalf("true-q accept fraction = %v, want 1", fracTrue)
	}
	if fracWrong > 0.85 || fracTrue-fracWrong < 0.2 {
		t.Fatalf("accept fractions true=%v wrong=%v, want a clear gap proving q drives the outcome distribution",
			fracTrue, fracWrong)
	}
}

// TestEffectiveQDrawKeyingPermutationAndBatchWidthInvariance pins the
// request/position keying contract: a request's plan depends only on its own
// request key, step offset, and distributions -- never on which other requests
// share the batch, their order, or the batch width.
func TestEffectiveQDrawKeyingPermutationAndBatchWidthInvariance(t *testing.T) {
	type req struct {
		name string
		key  uint64
		p    []float64
		q    []float32
		tok  int
	}
	a := req{name: "A", key: effectiveQKey(101), p: []float64{0.6, 0.25, 0.1, 0.05}, q: []float32{0.5, 0.3, 0.1, 0.1}, tok: 1}
	b := req{name: "B", key: effectiveQKey(202), p: []float64{0.1, 0.7, 0.15, 0.05}, q: []float32{0.2, 0.4, 0.3, 0.1}, tok: 0}
	dummies := make([]req, 8)
	for i := range dummies {
		dummies[i] = req{name: "D", key: effectiveQKey(uint64(300 + i)), p: a.p, q: a.q, tok: a.tok}
	}

	planFor := func(r req) SpecRejectionPlan {
		in := SpecRejectionPlanInput{
			DraftTokens:           []int{r.tok},
			DraftEvidence:         []DraftTokenEvidence{effectiveQEvidence(r.tok, r.q)},
			TargetProbabilityRows: []SpecProbabilityRow{effectiveQTargetRow(r.p), effectiveQTargetRow(r.p)},
			RequestKey:            r.key,
			StepOffset:            0,
		}
		plan, err := BuildSpecRejectionPlan(in)
		if err != nil {
			t.Fatalf("%s plan: %v", r.name, err)
		}
		return plan
	}

	same := func(x, y SpecRejectionPlan) bool {
		return reflect.DeepEqual(x.AcceptedTokens, y.AcceptedTokens) &&
			x.RejectedAt == y.RejectedAt &&
			x.CorrectionToken == y.CorrectionToken &&
			x.RollbackKVCount == y.RollbackKVCount &&
			x.NextStep == y.NextStep &&
			reflect.DeepEqual(x.AcceptProb, y.AcceptProb)
	}

	layouts := [][]req{
		{a}, // width 1
		{dummies[0], dummies[1], a, dummies[2], dummies[3], dummies[4], dummies[5], dummies[6]}, // width 8, A mid-batch
		{b, a, dummies[0], dummies[1]}, // width 4, B first
		{a, b},                         // pair, A first
		{b, a},                         // pair, permuted
	}
	var plansA, plansB []SpecRejectionPlan
	for _, layout := range layouts {
		var pa, pb SpecRejectionPlan
		haveA, haveB := false, false
		for _, r := range layout {
			plan := planFor(r)
			switch r.name {
			case "A":
				pa, haveA = plan, true
			case "B":
				pb, haveB = plan, true
			}
		}
		if haveA {
			plansA = append(plansA, pa)
		}
		if haveB {
			plansB = append(plansB, pb)
		}
	}
	for i := 1; i < len(plansA); i++ {
		if !same(plansA[0], plansA[i]) {
			t.Fatalf("request A plan changed with batch layout %d: %+v vs %+v", i, plansA[0], plansA[i])
		}
	}
	for i := 1; i < len(plansB); i++ {
		if !same(plansB[0], plansB[i]) {
			t.Fatalf("request B plan changed with batch layout %d: %+v vs %+v", i, plansB[0], plansB[i])
		}
	}
	if same(plansA[0], plansB[0]) {
		t.Fatalf("distinct requests produced identical plans; shared state suspected")
	}

	// Direct coin independence: drawing every one of B's coins in between does
	// not move A's coin.
	coinBefore := KeyedUniform64(a.key, DomainAcceptCoin, 0)
	for domain := DomainTag(0x1); domain <= DomainTag(0x4); domain++ {
		for step := 0; step < 8; step++ {
			_ = KeyedUniform64(b.key, domain, step)
		}
	}
	coinAfter := KeyedUniform64(a.key, DomainAcceptCoin, 0)
	if coinBefore != coinAfter {
		t.Fatalf("A's keyed coin moved after B's draws: %v -> %v", coinBefore, coinAfter)
	}
}

// TestEffectiveQGridLawMatchesProducerDraws sweeps the full 2^24 draw grid and
// proves expandDraftSamplingLaw -- the q the acceptance ratio consumes --
// assigns each token exactly the mass the producer's SampleFromDistribution
// actually realizes over its 24-bit uniform draws.
func TestEffectiveQGridLawMatchesProducerDraws(t *testing.T) {
	v, err := NewQwen38SampledSpeculativeVerifier(Qwen38SamplerConfig{Temperature: 1, TopP: 1})
	if err != nil {
		t.Fatalf("verifier: %v", err)
	}
	const draws = 1 << 24
	rows := [][]float32{
		{0.4, 0.3, 0.2, 0.1},
		{0.1, 0.1, 0.1, 0.7},
	}
	for _, q := range rows {
		counts := make([]int, len(q))
		for k := 0; k < draws; k++ {
			u := float32(k) / 16777216.0
			counts[v.SampleFromDistribution(q, u)]++
		}
		expanded, err := expandDraftSamplingLaw(effectiveQEvidence(0, q), len(q))
		if err != nil {
			t.Fatalf("expandDraftSamplingLaw(%v): %v", q, err)
		}
		for j := range q {
			want := int(expanded[j] * float64(draws))
			if counts[j] != want {
				t.Fatalf("row %v token %d: producer drew %d of %d, law assigns %d", q, j, counts[j], draws, want)
			}
		}
	}
}
