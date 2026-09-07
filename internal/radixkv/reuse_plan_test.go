package radixkv

import (
	"math"
	"math/rand"
	"reflect"
	"testing"
)

func TestReusePlan_PrefixOnly(t *testing.T) {
	promptLen := 100
	prefixLen := 40
	plan, err := ComposeReusePlan(promptLen, prefixLen, nil, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}
	if plan.PromptLen != promptLen || plan.PrefixLen != prefixLen {
		t.Fatalf("unexpected lengths: prompt=%d prefix=%d", plan.PromptLen, plan.PrefixLen)
	}
	if len(plan.Spans) != 2 {
		t.Fatalf("expected 2 spans, got %d", len(plan.Spans))
	}

	// Span 0: prefix direct reuse [0, 40)
	s0 := plan.Spans[0]
	if s0.Start != 0 || s0.End != 40 || s0.Action != ActionDirectReuse || s0.Tier != TierL1GPU || s0.Cost != 0 {
		t.Fatalf("unexpected span 0: %+v", s0)
	}

	// Span 1: compute gap [40, 100)
	s1 := plan.Spans[1]
	if s1.Start != 40 || s1.End != 100 || s1.Action != ActionCompute || s1.Cost != 60 {
		t.Fatalf("unexpected span 1: %+v", s1)
	}

	if plan.DirectTokens != 40 {
		t.Errorf("DirectTokens=%d, want 40", plan.DirectTokens)
	}
	if plan.ComputeTokens != 60 {
		t.Errorf("ComputeTokens=%d, want 60", plan.ComputeTokens)
	}
	if plan.RepairTokens != 0 {
		t.Errorf("RepairTokens=%d, want 0", plan.RepairTokens)
	}
	if plan.EstimatedCost != 60.0 {
		t.Errorf("EstimatedCost=%f, want 60.0", plan.EstimatedCost)
	}
	if plan.SavingsRatio != 0.40 {
		t.Errorf("SavingsRatio=%f, want 0.40", plan.SavingsRatio)
	}
}

func TestReusePlan_WithAddressedCandidates(t *testing.T) {
	promptLen := 100
	prefixLen := 20
	candidates := []ReuseCandidate{
		{
			ID:          "cand-host-1",
			DstStart:    30,
			DstEnd:      50,
			SrcStart:    0,
			SrcEnd:      20,
			Tier:        TierL2Host,
			Cost:        5.0,
			PrefillCost: 20.0,
			Compatible:  true,
			Action:      ActionDirectReuse,
		},
		{
			ID:          "cand-remote-1",
			DstStart:    60,
			DstEnd:      80,
			SrcStart:    100,
			SrcEnd:      120,
			Tier:        TierL3Remote,
			Cost:        8.0,
			PrefillCost: 20.0,
			Compatible:  true,
			Action:      ActionSelectiveRepair,
		},
	}

	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	// Expected spans:
	// 0: [0, 20) DirectReuse (L1) cost 0
	// 1: [20, 30) Compute cost 10
	// 2: [30, 50) DirectReuse (L2) cost 5, CandidateID: "cand-host-1"
	// 3: [50, 60) Compute cost 10
	// 4: [60, 80) SelectiveRepair (L3) cost 8, CandidateID: "cand-remote-1"
	// 5: [80, 100) Compute cost 20
	if len(plan.Spans) != 6 {
		t.Fatalf("expected 6 spans, got %d: %+v", len(plan.Spans), plan.Spans)
	}

	if plan.DirectTokens != 40 {
		t.Errorf("DirectTokens=%d, want 40", plan.DirectTokens)
	}
	if plan.RepairTokens != 20 {
		t.Errorf("RepairTokens=%d, want 20", plan.RepairTokens)
	}
	if plan.ComputeTokens != 40 {
		t.Errorf("ComputeTokens=%d, want 40", plan.ComputeTokens)
	}

	wantCost := 0.0 + 10.0 + 5.0 + 10.0 + 8.0 + 20.0 // 53.0
	if plan.EstimatedCost != wantCost {
		t.Errorf("EstimatedCost=%f, want %f", plan.EstimatedCost, wantCost)
	}

	wantSavings := (100.0 - 53.0) / 100.0 // 0.47
	if math.Abs(plan.SavingsRatio-wantSavings) > 1e-9 {
		t.Errorf("SavingsRatio=%f, want %f", plan.SavingsRatio, wantSavings)
	}
}

func TestReusePlan_ConflictingCandidates(t *testing.T) {
	// Overlapping candidates:
	// cand-a: [20, 60), cost 10, prefill 40 => savings = 30
	// cand-b: [30, 70), cost 25, prefill 40 => savings = 15
	// A and B overlap on [30, 60). A has higher net savings (30 > 15), so A should be selected and B rejected.
	promptLen := 80
	prefixLen := 10
	candidates := []ReuseCandidate{
		{
			ID:          "cand-b",
			DstStart:    30,
			DstEnd:      70,
			Tier:        TierL2Host,
			Cost:        25.0,
			PrefillCost: 40.0,
			Compatible:  true,
		},
		{
			ID:          "cand-a",
			DstStart:    20,
			DstEnd:      60,
			Tier:        TierL2Host,
			Cost:        10.0,
			PrefillCost: 40.0,
			Compatible:  true,
		},
	}

	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	var hasA, hasB bool
	for _, s := range plan.Spans {
		if s.CandidateID == "cand-a" {
			hasA = true
		}
		if s.CandidateID == "cand-b" {
			hasB = true
		}
	}
	if !hasA || hasB {
		t.Fatalf("expected cand-a selected and cand-b rejected; hasA=%v, hasB=%v", hasA, hasB)
	}

	// Tie breaker: same net savings, different span length
	// cand-x: [20, 40), cost 0, prefill 20 => savings = 20, len = 20
	// cand-y: [25, 55), cost 10, prefill 30 => savings = 20, len = 30
	// Both overlap on [25, 40). cand-y has longer length (30 > 20), so cand-y is selected.
	candidates2 := []ReuseCandidate{
		{
			ID:          "cand-x",
			DstStart:    20,
			DstEnd:      40,
			Tier:        TierL2Host,
			Cost:        0.0,
			PrefillCost: 20.0,
			Compatible:  true,
		},
		{
			ID:          "cand-y",
			DstStart:    25,
			DstEnd:      55,
			Tier:        TierL2Host,
			Cost:        10.0,
			PrefillCost: 30.0,
			Compatible:  true,
		},
	}

	plan2, err := ComposeReusePlan(promptLen, prefixLen, candidates2, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	var hasX, hasY bool
	for _, s := range plan2.Spans {
		if s.CandidateID == "cand-x" {
			hasX = true
		}
		if s.CandidateID == "cand-y" {
			hasY = true
		}
	}
	if !hasY || hasX {
		t.Fatalf("expected cand-y selected by span-length tie breaker and cand-x rejected; hasX=%v, hasY=%v", hasX, hasY)
	}
}

func TestReusePlan_NetTrueCostFilter(t *testing.T) {
	promptLen := 100
	prefixLen := 0
	candidates := []ReuseCandidate{
		{
			ID:          "cand-expensive",
			DstStart:    10,
			DstEnd:      30,
			Cost:        25.0,
			PrefillCost: 20.0, // cost > prefill => reject
			Compatible:  true,
		},
		{
			ID:          "cand-break-even",
			DstStart:    40,
			DstEnd:      60,
			Cost:        20.0,
			PrefillCost: 20.0, // cost == prefill => reject
			Compatible:  true,
		},
		{
			ID:          "cand-beneficial",
			DstStart:    70,
			DstEnd:      90,
			Cost:        5.0,
			PrefillCost: 20.0, // cost < prefill => accept
			Compatible:  true,
		},
	}

	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	for _, s := range plan.Spans {
		if s.CandidateID == "cand-expensive" {
			t.Errorf("cand-expensive should have been rejected by net-true cost rule")
		}
		if s.CandidateID == "cand-break-even" {
			t.Errorf("cand-break-even should have been rejected by net-true cost rule")
		}
	}

	var hasBeneficial bool
	for _, s := range plan.Spans {
		if s.CandidateID == "cand-beneficial" {
			hasBeneficial = true
		}
	}
	if !hasBeneficial {
		t.Errorf("cand-beneficial should have been accepted")
	}

	// Gaps [0, 70) and [90, 100) must be ActionCompute
	if plan.ComputeTokens != 80 {
		t.Errorf("ComputeTokens=%d, want 80", plan.ComputeTokens)
	}
	if plan.DirectTokens != 20 {
		t.Errorf("DirectTokens=%d, want 20", plan.DirectTokens)
	}
}

func TestReusePlan_DeterministicShuffling(t *testing.T) {
	promptLen := 200
	prefixLen := 20
	baseCandidates := []ReuseCandidate{
		{ID: "c1", DstStart: 30, DstEnd: 60, Tier: TierL2Host, Cost: 5.0, PrefillCost: 30.0, Compatible: true},
		{ID: "c2", DstStart: 50, DstEnd: 90, Tier: TierL2Host, Cost: 10.0, PrefillCost: 40.0, Compatible: true},
		{ID: "c3", DstStart: 70, DstEnd: 100, Tier: TierL3Remote, Cost: 6.0, PrefillCost: 30.0, Compatible: true},
		{ID: "c4", DstStart: 110, DstEnd: 140, Tier: TierL2Host, Cost: 5.0, PrefillCost: 30.0, Compatible: true},
		{ID: "c5", DstStart: 130, DstEnd: 160, Tier: TierL3Remote, Cost: 5.0, PrefillCost: 30.0, Compatible: true},
		{ID: "c6", DstStart: 170, DstEnd: 190, Tier: TierL2Host, Cost: 3.0, PrefillCost: 20.0, Compatible: true},
		{ID: "c7_incompatible", DstStart: 180, DstEnd: 195, Tier: TierL2Host, Cost: 1.0, PrefillCost: 15.0, Compatible: false},
		{ID: "c8_rejected", DstStart: 180, DstEnd: 195, Tier: TierL2Host, Cost: 1.0, PrefillCost: 15.0, Compatible: true, Action: ActionReject},
	}

	baseline, err := ComposeReusePlan(promptLen, prefixLen, baseCandidates, nil)
	if err != nil {
		t.Fatalf("baseline ComposeReusePlan failed: %v", err)
	}

	r := rand.New(rand.NewSource(42))
	for i := 0; i < 50; i++ {
		shuffled := make([]ReuseCandidate, len(baseCandidates))
		copy(shuffled, baseCandidates)
		r.Shuffle(len(shuffled), func(i, j int) {
			shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
		})

		plan, err := ComposeReusePlan(promptLen, prefixLen, shuffled, nil)
		if err != nil {
			t.Fatalf("shuffle %d failed: %v", i, err)
		}

		if !reflect.DeepEqual(baseline, plan) {
			t.Fatalf("shuffle %d produced divergent plan:\nbaseline: %+v\ngot: %+v", i, baseline, plan)
		}
	}
}

func TestReusePlan_BoundaryConditions(t *testing.T) {
	// 1. Empty prompt
	pEmpty, err := ComposeReusePlan(0, 0, nil, nil)
	if err != nil {
		t.Fatalf("empty prompt failed: %v", err)
	}
	if pEmpty.PromptLen != 0 || pEmpty.PrefixLen != 0 || len(pEmpty.Spans) != 0 ||
		pEmpty.DirectTokens != 0 || pEmpty.ComputeTokens != 0 || pEmpty.RepairTokens != 0 ||
		pEmpty.EstimatedCost != 0 || pEmpty.SavingsRatio != 0 {
		t.Fatalf("unexpected empty plan: %+v", pEmpty)
	}

	// 2. Full prefix reuse
	pFull, err := ComposeReusePlan(50, 50, nil, nil)
	if err != nil {
		t.Fatalf("full prefix reuse failed: %v", err)
	}
	if len(pFull.Spans) != 1 || pFull.Spans[0].Action != ActionDirectReuse || pFull.Spans[0].End != 50 || pFull.DirectTokens != 50 || pFull.SavingsRatio != 1.0 {
		t.Fatalf("unexpected full prefix plan: %+v", pFull)
	}

	// 3. Invalid bounds
	invalidCases := []struct {
		promptLen int
		prefixLen int
	}{
		{-1, 0},
		{10, -1},
		{10, 20},
		{-5, -5},
	}
	for _, tc := range invalidCases {
		_, err := ComposeReusePlan(tc.promptLen, tc.prefixLen, nil, nil)
		if err == nil {
			t.Errorf("expected error for promptLen=%d prefixLen=%d, got nil", tc.promptLen, tc.prefixLen)
		}
	}

	// 4. Out-of-bound / invalid candidates
	oobCandidates := []ReuseCandidate{
		{ID: "dst_start_before_prefix", DstStart: 5, DstEnd: 20, Compatible: true},
		{ID: "dst_end_after_prompt", DstStart: 50, DstEnd: 120, Compatible: true},
		{ID: "zero_length", DstStart: 30, DstEnd: 30, Compatible: true},
		{ID: "negative_length", DstStart: 40, DstEnd: 30, Compatible: true},
		{ID: "incompatible", DstStart: 60, DstEnd: 70, Compatible: false},
		{ID: "reject_action", DstStart: 70, DstEnd: 80, Compatible: true, Action: ActionReject},
	}
	pOOB, err := ComposeReusePlan(100, 10, oobCandidates, nil)
	if err != nil {
		t.Fatalf("OOB test failed: %v", err)
	}
	if len(pOOB.Spans) != 2 {
		t.Fatalf("expected 2 spans for all-OOB candidates, got %d: %+v", len(pOOB.Spans), pOOB.Spans)
	}
	if pOOB.Spans[0].Action != ActionDirectReuse || pOOB.Spans[0].End != 10 {
		t.Errorf("span 0 mismatch: %+v", pOOB.Spans[0])
	}
	if pOOB.Spans[1].Action != ActionCompute || pOOB.Spans[1].Start != 10 || pOOB.Spans[1].End != 100 {
		t.Errorf("span 1 mismatch: %+v", pOOB.Spans[1])
	}
}

type zeroCostModel struct{}

func (zeroCostModel) PrefillCost(spanLen int) float64 {
	return float64(spanLen)
}

func (zeroCostModel) CandidateCost(c ReuseCandidate) float64 {
	return 0.0
}

func TestReusePlan_TrustCandidateCostZero(t *testing.T) {
	promptLen := 100
	prefixLen := 10
	candidates := []ReuseCandidate{
		{
			ID:          "cand-zero-cost",
			DstStart:    20,
			DstEnd:      60,
			Cost:        100.0, // Non-zero candidate declared cost
			PrefillCost: 40.0,
			Compatible:  true,
		},
	}

	// CostModel returns 0.0 for candidate cost. If overridden by c.Cost, cost=100 >= prefill=40 -> rejected.
	// Trusting cm directly means cost=0 < 40 -> accepted with cost=0.
	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, zeroCostModel{})
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	var found bool
	for _, s := range plan.Spans {
		if s.CandidateID == "cand-zero-cost" {
			found = true
			if s.Cost != 0.0 {
				t.Errorf("expected span cost 0.0, got %f", s.Cost)
			}
		}
	}
	if !found {
		t.Fatalf("expected cand-zero-cost to be selected when CostModel returns 0")
	}
}

func TestReusePlan_NaNAndInfHandling(t *testing.T) {
	promptLen := 100
	prefixLen := 0
	candidates := []ReuseCandidate{
		{
			ID:          "cand-nan-cost",
			DstStart:    10,
			DstEnd:      20,
			Cost:        math.NaN(),
			PrefillCost: 10.0,
			Compatible:  true,
		},
		{
			ID:          "cand-inf-cost",
			DstStart:    20,
			DstEnd:      30,
			Cost:        math.Inf(1),
			PrefillCost: 10.0,
			Compatible:  true,
		},
		{
			ID:          "cand-neginf-cost",
			DstStart:    30,
			DstEnd:      40,
			Cost:        math.Inf(-1),
			PrefillCost: 10.0,
			Compatible:  true,
		},
		{
			ID:          "cand-nan-prefill",
			DstStart:    40,
			DstEnd:      50,
			Cost:        1.0,
			PrefillCost: math.NaN(),
			Compatible:  true,
		},
		{
			ID:          "cand-inf-prefill",
			DstStart:    50,
			DstEnd:      60,
			Cost:        1.0,
			PrefillCost: math.Inf(1),
			Compatible:  true,
		},
		{
			ID:          "cand-valid",
			DstStart:    70,
			DstEnd:      90,
			Cost:        2.0,
			PrefillCost: 20.0,
			Compatible:  true,
		},
	}

	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	for _, s := range plan.Spans {
		if s.CandidateID != "" && s.CandidateID != "cand-valid" {
			t.Errorf("expected only cand-valid to be selected, found %s", s.CandidateID)
		}
	}

	if plan.DirectTokens != 20 {
		t.Errorf("DirectTokens=%d, want 20", plan.DirectTokens)
	}
	if plan.ComputeTokens != 80 {
		t.Errorf("ComputeTokens=%d, want 80", plan.ComputeTokens)
	}
}

func TestReusePlan_UnrecognizedActionKind(t *testing.T) {
	promptLen := 100
	prefixLen := 10
	candidates := []ReuseCandidate{
		{
			ID:          "cand-unknown-action",
			DstStart:    20,
			DstEnd:      50,
			Cost:        5.0,
			PrefillCost: 30.0,
			Compatible:  true,
			Action:      ActionKind("custom_unrecognized_action"),
		},
	}

	plan, err := ComposeReusePlan(promptLen, prefixLen, candidates, nil)
	if err != nil {
		t.Fatalf("ComposeReusePlan failed: %v", err)
	}

	var found bool
	for _, s := range plan.Spans {
		if s.CandidateID == "cand-unknown-action" {
			found = true
			if s.Action != ActionDirectReuse {
				t.Errorf("expected ActionDirectReuse for unrecognized action, got %s", s.Action)
			}
		}
	}
	if !found {
		t.Fatalf("cand-unknown-action should have been selected")
	}
	if plan.DirectTokens != 40 { // 10 prefix + 30 candidate
		t.Errorf("DirectTokens=%d, want 40", plan.DirectTokens)
	}
}
