package model

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

func TestMTPDoubleShiftProtection(t *testing.T) {
	// Case 1: Trunk weights without MTP keys
	weightsWithoutMTP := map[string][]float32{
		"model.norm.weight":                      {0.0, 0.0, 0.0},
		"model.layers.0.input_layernorm.weight":  {0.1, -0.2, 0.05},
		"model.layers.0.self_attn.q_norm.weight": {-0.05, 0.1},
		"model.embed_tokens.weight":              {0.5, -0.5, 0.25}, // non-norm, must not shift
	}

	sanitized1 := SanitizeTrunkNorms(weightsWithoutMTP)

	// Verify trunk norm weights shifted by +1.0 exactly once
	norm1 := sanitized1["model.norm.weight"]
	if len(norm1) != 3 || norm1[0] != 1.0 || norm1[1] != 1.0 || norm1[2] != 1.0 {
		t.Fatalf("model.norm.weight without MTP = %v, want [1.0 1.0 1.0]", norm1)
	}
	inputLN1 := sanitized1["model.layers.0.input_layernorm.weight"]
	if math.Abs(float64(inputLN1[0]-1.1)) > 1e-6 || math.Abs(float64(inputLN1[1]-0.8)) > 1e-6 {
		t.Fatalf("input_layernorm without MTP = %v, want [1.1 0.8 ...]", inputLN1)
	}
	qNorm1 := sanitized1["model.layers.0.self_attn.q_norm.weight"]
	if math.Abs(float64(qNorm1[0]-0.95)) > 1e-6 || math.Abs(float64(qNorm1[1]-1.1)) > 1e-6 {
		t.Fatalf("q_norm without MTP = %v, want [0.95 1.1]", qNorm1)
	}
	embed1 := sanitized1["model.embed_tokens.weight"]
	if embed1[0] != 0.5 || embed1[1] != -0.5 || embed1[2] != 0.25 {
		t.Fatalf("embed_tokens unexpectedly modified: %v", embed1)
	}

	// Case 2: Trunk weights WITH MTP keys present
	weightsWithMTP := map[string][]float32{
		"model.norm.weight":                            {0.0, 0.0, 0.0},
		"model.layers.0.input_layernorm.weight":        {0.1, -0.2, 0.05},
		"model.layers.0.self_attn.q_norm.weight":       {-0.05, 0.1},
		"model.embed_tokens.weight":                    {0.5, -0.5, 0.25},
		"mtp.fc.weight":                                {0.3, 0.4},
		"mtp.norm.weight":                              {1.0, 1.0, 1.0}, // MTP norms have baked offset
		"mtp.pre_fc_norm_hidden.weight":                {1.0, 1.0, 1.0},
		"mtp.layers.0.input_layernorm.weight":          {1.0, 1.0, 1.0},
		"mtp.layers.0.post_attention_layernorm.weight": {1.0, 1.0, 1.0},
	}

	sanitized2 := SanitizeTrunkNorms(weightsWithMTP)

	// Verify trunk norms are shifted by +1.0 EXACTLY ONCE (not +2.0 from double-shifting)
	norm2 := sanitized2["model.norm.weight"]
	if len(norm2) != 3 || norm2[0] != 1.0 || norm2[1] != 1.0 || norm2[2] != 1.0 {
		t.Fatalf("model.norm.weight with MTP = %v, want [1.0 1.0 1.0] (double-shifted if 2.0)", norm2)
	}
	inputLN2 := sanitized2["model.layers.0.input_layernorm.weight"]
	if math.Abs(float64(inputLN2[0]-1.1)) > 1e-6 || math.Abs(float64(inputLN2[1]-0.8)) > 1e-6 {
		t.Fatalf("input_layernorm with MTP = %v, want [1.1 0.8 ...]", inputLN2)
	}

	// Verify MTP keys were stripped from the sanitized trunk weights
	for k := range sanitized2 {
		if strings.HasPrefix(k, "mtp.") {
			t.Fatalf("sanitized trunk weights still contain MTP key: %s", k)
		}
	}

	// Case 3: Double-shift defense / Idempotency
	// Running SanitizeTrunkNorms again on already-sanitized weights must NOT re-apply the shift
	sanitized3 := SanitizeTrunkNorms(sanitized2)
	norm3 := sanitized3["model.norm.weight"]
	if norm3[0] != 1.0 || norm3[1] != 1.0 || norm3[2] != 1.0 {
		t.Fatalf("model.norm.weight after second sanitization = %v, want [1.0 1.0 1.0] (double-shift occurred)", norm3)
	}
	inputLN3 := sanitized3["model.layers.0.input_layernorm.weight"]
	if math.Abs(float64(inputLN3[0]-1.1)) > 1e-6 || math.Abs(float64(inputLN3[1]-0.8)) > 1e-6 {
		t.Fatalf("input_layernorm after second sanitization = %v, want [1.1 0.8 ...]", inputLN3)
	}
}

func TestMTPHeadIsolationAndInjection(t *testing.T) {
	weights := map[string][]float32{
		"model.norm.weight":                     {0.0, 0.0},
		"model.layers.0.input_layernorm.weight": {0.0, 0.0},
		"mtp.fc.weight":                         {0.1, 0.2, 0.3, 0.4},
		"mtp.norm.weight":                       {1.0, 1.0},
		"mtp.pre_fc_norm_embedding.weight":      {1.0, 1.0},
		"mtp.layers.0.mlp.down_proj.weight":     {0.7, 0.8},
	}

	head, trunk := IsolateMTPHead(weights)
	if head == nil {
		t.Fatal("IsolateMTPHead returned nil head")
	}
	if len(head.Weights) != 4 {
		t.Fatalf("head.Weights len = %d, want 4", len(head.Weights))
	}
	if len(head.FC) != 4 || head.FC[0] != 0.1 {
		t.Fatalf("head.FC = %v, want [0.1 0.2 0.3 0.4]", head.FC)
	}
	if len(head.Norm) != 2 || head.Norm[0] != 1.0 {
		t.Fatalf("head.Norm = %v, want [1.0 1.0]", head.Norm)
	}
	if len(trunk) != 2 {
		t.Fatalf("trunk weights len = %d, want 2", len(trunk))
	}
	for k := range trunk {
		if strings.HasPrefix(k, "mtp.") {
			t.Fatalf("trunk contains mtp key: %s", k)
		}
	}

	// Sanitize trunk norms
	sanitizedTrunk := SanitizeTrunkNorms(trunk)
	if sanitizedTrunk["model.norm.weight"][0] != 1.0 {
		t.Fatalf("sanitized trunk norm = %v, want 1.0", sanitizedTrunk["model.norm.weight"])
	}

	// Inject MTP head into sanitized trunk
	merged := InjectMTPHead(sanitizedTrunk, head)
	if len(merged) != 6 {
		t.Fatalf("merged weights len = %d, want 6", len(merged))
	}
	// Verify trunk norm remains 1.0
	if merged["model.norm.weight"][0] != 1.0 {
		t.Fatalf("injected merged model.norm.weight = %v, want 1.0", merged["model.norm.weight"])
	}
	// Verify MTP norm was not shifted (remains 1.0)
	if merged["mtp.norm.weight"][0] != 1.0 {
		t.Fatalf("injected merged mtp.norm.weight = %v, want 1.0", merged["mtp.norm.weight"])
	}
	// Verify MTP FC preserved
	if merged["mtp.fc.weight"][0] != 0.1 {
		t.Fatalf("injected merged mtp.fc.weight = %v, want 0.1", merged["mtp.fc.weight"])
	}
}

func TestMTPRejectionSamplingGreedy(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	// Vocab size = 4
	// Greedy Accept Case: Target and Draft both peak at token 2
	pTargetAccept := []float32{0.05, 0.05, 0.85, 0.05}
	pDraftAccept := []float32{0.1, 0.1, 0.7, 0.1}

	// For token 2: P_target(2)/P_draft(2) = 0.85/0.70 > 1.0 => alpha = 1.0
	// Any random value in [0, 1) must result in acceptance!
	for _, r := range []float32{0.0, 0.25, 0.5, 0.75, 0.99} {
		res := sampler.VerifyToken(2, pTargetAccept, pDraftAccept, r)
		if !res.Accepted {
			t.Fatalf("greedy accept: token 2 rejected with r=%g, acceptProb=%g", r, res.AcceptProb)
		}
		if res.DraftToken != 2 {
			t.Fatalf("res.DraftToken = %d, want 2", res.DraftToken)
		}
	}

	// Greedy Reject Case: Draft proposed token 1, but Target peaks at token 3
	pTargetReject := []float32{0.0, 0.0, 0.0, 1.0} // token 3 is target argmax
	pDraftReject := []float32{0.0, 1.0, 0.0, 0.0}  // token 1 is draft argmax

	// For token 1: P_target(1)/P_draft(1) = 0.0/1.0 = 0.0 => alpha = 0.0
	// Any random value r > 0 must result in rejection
	resReject := sampler.VerifyToken(1, pTargetReject, pDraftReject, 0.01)
	if resReject.Accepted {
		t.Fatalf("greedy reject: token 1 unexpectedly accepted")
	}
	// On rejection, residual distribution must place all mass on target argmax (token 3)
	if resReject.ReplacementToken != 3 {
		t.Fatalf("replacement token = %d, want 3 (target argmax)", resReject.ReplacementToken)
	}
	if resReject.ResidualDist[3] != 1.0 {
		t.Fatalf("residual distribution at token 3 = %g, want 1.0", resReject.ResidualDist[3])
	}
}

func TestMTPResidualDistributionSumToOne(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	testCases := []struct {
		name    string
		pTarget []float32
		pDraft  []float32
	}{
		{
			name:    "skewed vs uniform",
			pTarget: []float32{0.7, 0.2, 0.05, 0.05},
			pDraft:  []float32{0.25, 0.25, 0.25, 0.25},
		},
		{
			name:    "disjoint support",
			pTarget: []float32{0.5, 0.5, 0.0, 0.0},
			pDraft:  []float32{0.0, 0.0, 0.5, 0.5},
		},
		{
			name:    "partial overlap",
			pTarget: []float32{0.1, 0.4, 0.3, 0.2},
			pDraft:  []float32{0.3, 0.2, 0.4, 0.1},
		},
		{
			name:    "high dimension random",
			pTarget: makeRandomDist(64, 42),
			pDraft:  makeRandomDist(64, 99),
		},
		{
			name:    "identical distributions",
			pTarget: []float32{0.25, 0.25, 0.25, 0.25},
			pDraft:  []float32{0.25, 0.25, 0.25, 0.25},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			residual := sampler.ResidualDistribution(tc.pTarget, tc.pDraft)
			if len(residual) != len(tc.pTarget) {
				t.Fatalf("residual len = %d, want %d", len(residual), len(tc.pTarget))
			}
			var sum float64
			for i, p := range residual {
				if p < 0 {
					t.Fatalf("negative residual prob at index %d: %g", i, p)
				}
				sum += float64(p)
			}
			if math.Abs(sum-1.0) > 1e-6 {
				t.Fatalf("residual distribution sum = %g, want 1.0 (violated sum-to-one invariant)", sum)
			}
		})
	}
}

func TestMTPSampleAcceptance(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	// pTarget(0) = 0.4, pDraft(0) = 0.8 => alpha = 0.4 / 0.8 = 0.5
	pTarget := []float32{0.4, 0.6}
	pDraft := []float32{0.8, 0.2}

	alpha := sampler.AcceptanceProbability(pTarget[0], pDraft[0])
	if math.Abs(float64(alpha-0.5)) > 1e-6 {
		t.Fatalf("alpha = %g, want 0.5", alpha)
	}

	// Exact check: r <= alpha (0.5) must accept
	res1 := sampler.VerifyToken(0, pTarget, pDraft, 0.49)
	if !res1.Accepted {
		t.Fatalf("expected accept for r=0.49 <= 0.5, got reject")
	}

	// Exact check: r > alpha (0.5) must reject
	res2 := sampler.VerifyToken(0, pTarget, pDraft, 0.51)
	if res2.Accepted {
		t.Fatalf("expected reject for r=0.51 > 0.5, got accept")
	}

	// When pTarget >= pDraft, alpha must be 1.0
	// pTarget(1) = 0.6, pDraft(1) = 0.2 => alpha = min(1, 0.6/0.2 = 3.0) = 1.0
	alpha1 := sampler.AcceptanceProbability(pTarget[1], pDraft[1])
	if alpha1 != 1.0 {
		t.Fatalf("alpha1 = %g, want 1.0", alpha1)
	}
	res3 := sampler.VerifyToken(1, pTarget, pDraft, 0.999)
	if !res3.Accepted {
		t.Fatalf("expected accept for alpha=1.0 with r=0.999, got reject")
	}
}

func TestMTPDistributionConsistency(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	// Analytical verification: P_effective(x) == P_target(x) for all x
	sizes := []int{4, 8, 32, 128}
	for _, size := range sizes {
		pTarget := makeRandomDist(size, int64(size*7))
		pDraft := makeRandomDist(size, int64(size*13))

		if err := sampler.VerifyDistributionConsistency(pTarget, pDraft, 1e-5); err != nil {
			t.Fatalf("size %d distribution consistency failed: %v", size, err)
		}
	}

	// Empirical Monte Carlo verification
	vocab := 4
	pTarget := []float32{0.4, 0.3, 0.2, 0.1}
	pDraft := []float32{0.1, 0.5, 0.2, 0.2}

	rng := rand.New(rand.NewSource(12345))
	mcSampler := NewRejectionSampler(rng.Float32)

	counts := make([]int, vocab)
	const trials = 40000

	for i := 0; i < trials; i++ {
		// Draft model samples a token from pDraft
		draftTok := mcSampler.Sample(pDraft)
		res := mcSampler.VerifyToken(draftTok, pTarget, pDraft, -1)
		if res.Accepted {
			counts[draftTok]++
		} else {
			counts[res.ReplacementToken]++
		}
	}

	// Verify empirical frequency matches target distribution within 3 sigma (~0.01)
	for v := 0; v < vocab; v++ {
		empFreq := float64(counts[v]) / float64(trials)
		wantProb := float64(pTarget[v])
		diff := math.Abs(empFreq - wantProb)
		if diff > 0.015 {
			t.Fatalf("empirical frequency for token %d = %g, want %g (diff %g > 0.015)", v, empFreq, wantProb, diff)
		}
	}
}

func TestMTPAcceptanceRateTracking(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	pTarget := []float32{0.5, 0.5}
	pDraft := []float32{0.5, 0.5}

	// 3 accepts, 1 reject
	sampler.VerifyToken(0, pTarget, pDraft, 0.1)                          // accept
	sampler.VerifyToken(0, pTarget, pDraft, 0.2)                          // accept
	sampler.VerifyToken(0, pTarget, pDraft, 0.3)                          // accept
	sampler.VerifyToken(0, []float32{0.0, 1.0}, []float32{1.0, 0.0}, 0.5) // reject (alpha=0)

	drafted, accepted, rejected, rate := sampler.Stats()
	if drafted != 4 {
		t.Fatalf("drafted = %d, want 4", drafted)
	}
	if accepted != 3 {
		t.Fatalf("accepted = %d, want 3", accepted)
	}
	if rejected != 1 {
		t.Fatalf("rejected = %d, want 1", rejected)
	}
	if math.Abs(rate-0.75) > 1e-6 {
		t.Fatalf("acceptance rate = %g, want 0.75", rate)
	}

	// Test Reset
	sampler.Reset()
	if sampler.AcceptanceRate() != 0.0 {
		t.Fatalf("rate after reset = %g, want 0.0", sampler.AcceptanceRate())
	}
}

func TestMTPDraftSequence(t *testing.T) {
	sampler := NewRejectionSampler(nil)

	// Draft 3 tokens: [1, 2, 3]
	draftTokens := []int{1, 2, 3}
	pTargets := [][]float32{
		{0.1, 0.9, 0.0, 0.0}, // pos 0: token 1 has alpha 1.0
		{0.0, 0.0, 0.9, 0.1}, // pos 1: token 2 has alpha 1.0
		{0.0, 0.0, 0.0, 1.0}, // pos 2: token 3 has alpha 1.0
		{0.5, 0.5, 0.0, 0.0}, // bonus distribution after pos 2
	}
	pDrafts := [][]float32{
		{0.1, 0.9, 0.0, 0.0},
		{0.0, 0.0, 0.9, 0.1},
		{0.0, 0.0, 0.9, 0.1},
	}

	// Case 1: All accepted
	res := sampler.VerifyDraftSequence(draftTokens, pTargets, pDrafts, []float32{0.1, 0.1, 0.1})
	if res.AcceptedCount != 3 {
		t.Fatalf("accepted count = %d, want 3", res.AcceptedCount)
	}
	if res.RejectedAt != -1 {
		t.Fatalf("rejectedAt = %d, want -1", res.RejectedAt)
	}
	if len(res.AcceptedTokens) != 3 {
		t.Fatalf("accepted tokens = %v, want [1 2 3]", res.AcceptedTokens)
	}

	// Case 2: Rejected at position 1
	pTargetsFail := [][]float32{
		{0.1, 0.9, 0.0, 0.0}, // pos 0: token 1 accepted
		{1.0, 0.0, 0.0, 0.0}, // pos 1: token 2 has pTarget=0.0 => alpha=0.0; target is token 0
		{0.0, 0.0, 0.0, 1.0},
	}
	resFail := sampler.VerifyDraftSequence(draftTokens, pTargetsFail, pDrafts, []float32{0.1, 0.5, 0.1})
	if resFail.AcceptedCount != 1 {
		t.Fatalf("fail accepted count = %d, want 1", resFail.AcceptedCount)
	}
	if resFail.RejectedAt != 1 {
		t.Fatalf("fail rejectedAt = %d, want 1", resFail.RejectedAt)
	}
	if resFail.ReplacementToken != 0 {
		t.Fatalf("fail replacementToken = %d, want 0 (target argmax for pos 1)", resFail.ReplacementToken)
	}
}

func makeRandomDist(n int, seed int64) []float32 {
	r := rand.New(rand.NewSource(seed))
	out := make([]float32, n)
	var sum float32
	for i := range out {
		out[i] = r.Float32() + 0.01
		sum += out[i]
	}
	for i := range out {
		out[i] /= sum
	}
	return out
}
