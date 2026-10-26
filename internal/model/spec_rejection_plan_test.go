package model

import (
	"errors"
	"math"
	"reflect"
	"testing"
)

func specLogRow(prob ...float32) []float32 {
	out := make([]float32, len(prob))
	for i, p := range prob {
		if p == 0 {
			out[i] = float32(math.Inf(-1))
		} else {
			out[i] = float32(math.Log(float64(p)))
		}
	}
	return out
}

func sampledEvidence(token int, prob ...float32) DraftTokenEvidence {
	return DraftTokenEvidence{Token: token, LogProbs: specLogRow(prob...), Provenance: DraftEvidenceSampledProcessedQ}
}

func TestSpecRejectionPlanRequiresKPlusOneAndSampledQ(t *testing.T) {
	in := SpecRejectionPlanInput{
		DraftTokens:    []int{0},
		DraftEvidence:  []DraftTokenEvidence{sampledEvidence(0, 0.6, 0.4)},
		TargetLogProbs: [][]float32{specLogRow(0.5, 0.5)},
		RequestKey:     1,
	}
	if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrSpecRejectionTargetRows) {
		t.Fatalf("short target evidence error = %v, want ErrSpecRejectionTargetRows", err)
	}
	in.TargetLogProbs = append(in.TargetLogProbs, specLogRow(0.5, 0.5))
	in.DraftEvidence[0].Provenance = DraftEvidenceModelSoftmaxGreedy
	if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrDraftEvidenceNotProcessedSamplingQ) {
		t.Fatalf("greedy evidence error = %v, want ErrDraftEvidenceNotProcessedSamplingQ", err)
	}
	in.DraftEvidence[0].Provenance = DraftEvidenceSampledProcessedQ
	in.RequestKey = 0
	if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrSpecRejectionUnkeyed) {
		t.Fatalf("unkeyed error = %v, want ErrSpecRejectionUnkeyed", err)
	}
}

func TestSpecRejectionPlanKeyedRoundOffsetAndRollback(t *testing.T) {
	base := SpecRejectionPlanInput{
		DraftTokens: []int{0, 1},
		DraftEvidence: []DraftTokenEvidence{
			sampledEvidence(0, 0.9, 0.1),
			sampledEvidence(1, 0.1, 0.9),
		},
		TargetLogProbs: [][]float32{
			specLogRow(0.45, 0.55),
			specLogRow(0.55, 0.45),
			specLogRow(0.25, 0.75),
		},
		RequestKey: 19,
	}
	first, err := BuildSpecRejectionPlan(base)
	if err != nil {
		t.Fatal(err)
	}
	replay, err := BuildSpecRejectionPlan(base)
	if err != nil || !reflect.DeepEqual(first, replay) {
		t.Fatalf("same keyed round was not replay-stable: first=%+v replay=%+v err=%v", first, replay, err)
	}
	if first.NextStep != 3 || first.RollbackKVCount != len(base.DraftTokens)-len(first.AcceptedTokens) {
		t.Fatalf("plan accounting = %+v", first)
	}
	shifted := base
	shifted.StepOffset = 7
	second, err := BuildSpecRejectionPlan(shifted)
	if err != nil {
		t.Fatal(err)
	}
	if second.NextStep != 10 {
		t.Fatalf("shifted next step = %d, want 10", second.NextStep)
	}
	if reflect.DeepEqual(first.AcceptProb, nil) || len(second.AcceptProb) != 2 {
		t.Fatalf("acceptance evidence missing: first=%+v second=%+v", first, second)
	}
}

func TestSpecRejectionPlanResidualModesAndInputImmutability(t *testing.T) {
	in := SpecRejectionPlanInput{
		DraftTokens:    []int{0},
		DraftEvidence:  []DraftTokenEvidence{sampledEvidence(0, 0.8, 0.2)},
		TargetLogProbs: [][]float32{specLogRow(0.4, 0.6), specLogRow(0.3, 0.7)},
		RequestKey:     77,
	}
	beforeDraft := append([]int(nil), in.DraftTokens...)
	beforeTarget := append([]float32(nil), in.TargetLogProbs[0]...)
	lossless, err := BuildSpecRejectionPlan(in)
	if err != nil {
		t.Fatal(err)
	}
	in.ResidualMode = SpecResidualPenaltyAdjusted
	in.DraftProbabilityPenalty = 0.25
	adjusted, err := BuildSpecRejectionPlan(in)
	if err != nil {
		t.Fatal(err)
	}
	if lossless.AcceptProb[0] == adjusted.AcceptProb[0] {
		t.Fatalf("residual modes produced the same alpha %g", lossless.AcceptProb[0])
	}
	if !reflect.DeepEqual(beforeDraft, in.DraftTokens) || !reflect.DeepEqual(beforeTarget, in.TargetLogProbs[0]) {
		t.Fatal("planner mutated caller-owned evidence")
	}
}

func TestSpecRejectionPlanCompactDraftCoordinates(t *testing.T) {
	in := SpecRejectionPlanInput{
		DraftTokens: []int{2},
		DraftEvidence: []DraftTokenEvidence{{
			Token: 2, TokenIDs: []int{2, 0}, LogProbs: specLogRow(0.75, 0.25), Provenance: DraftEvidenceSampledProcessedQ,
		}},
		TargetLogProbs: [][]float32{specLogRow(0.2, 0.3, 0.5), specLogRow(0.1, 0.2, 0.7)},
		RequestKey:     5,
	}
	plan, err := BuildSpecRejectionPlan(in)
	if err != nil {
		t.Fatal(err)
	}
	if len(plan.AcceptProb) != 1 || plan.AcceptProb[0] <= 0 {
		t.Fatalf("compact q was not expanded: %+v", plan)
	}
}

func TestSpecRejectionPlanValidatesAllDraftRowsBeforeDecision(t *testing.T) {
	cases := map[string]DraftTokenEvidence{
		"full width": sampledEvidence(1, 0.2, 0.3, 0.5),
		"duplicate compact coordinate": {
			Token: 1, TokenIDs: []int{1, 1}, LogProbs: specLogRow(0.5, 0.5), Provenance: DraftEvidenceSampledProcessedQ,
		},
		"out of vocabulary compact coordinate": {
			Token: 1, TokenIDs: []int{1, 2}, LogProbs: specLogRow(0.5, 0.5), Provenance: DraftEvidenceSampledProcessedQ,
		},
	}
	for name, malformed := range cases {
		t.Run(name, func(t *testing.T) {
			in := SpecRejectionPlanInput{
				DraftTokens:   []int{0, 1},
				DraftEvidence: []DraftTokenEvidence{sampledEvidence(0, 1, 0), malformed},
				TargetLogProbs: [][]float32{
					specLogRow(0, 1),
					specLogRow(0.5, 0.5),
					specLogRow(0.5, 0.5),
				},
				RequestKey: 31,
			}
			if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrSpecRejectionProbability) {
				t.Fatalf("malformed later q error = %v, want ErrSpecRejectionProbability", err)
			}
		})
	}
}

func TestSpecRejectionPlanRejectsStepRangeOverflow(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	in := SpecRejectionPlanInput{
		DraftTokens:    []int{0},
		DraftEvidence:  []DraftTokenEvidence{sampledEvidence(0, 1)},
		TargetLogProbs: [][]float32{specLogRow(1), specLogRow(1)},
		RequestKey:     1,
		StepOffset:     maxInt,
	}
	if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrSpecRejectionShape) {
		t.Fatalf("overflow error = %v, want ErrSpecRejectionShape", err)
	}
}
