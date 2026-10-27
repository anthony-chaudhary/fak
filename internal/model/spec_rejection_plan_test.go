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

// fak-test:runtime fast est=1ms lane=default
func TestSpecRejectionPlanCanonicalTargetProbabilityRows(t *testing.T) {
	const key = uint64(19)
	coin := KeyedUniform64(key, DomainBonusToken, 1)
	for _, above := range []bool{false, true} {
		cutoff := math.Nextafter(coin*7, 0)
		want := 1
		if above {
			cutoff = math.Nextafter(coin*7, math.Inf(1))
			want = 0
		}
		q := sampledEvidence(0, 1, 0)
		q.ProcessedQ = []float32{1, 0}
		q.SamplingLaw = DraftSamplingLawQwen38UnitCDF24
		in := SpecRejectionPlanInput{
			DraftTokens: []int{0}, DraftEvidence: []DraftTokenEvidence{q}, RequestKey: key,
			TargetProbabilityRows: []SpecProbabilityRow{
				{Weights: []float64{1, 0}, Mass: 1},
				{Weights: []float64{cutoff, 7 - cutoff}, Mass: 7},
			},
		}
		before := append([]float64(nil), in.TargetProbabilityRows[1].Weights...)
		plan, err := BuildSpecRejectionPlan(in)
		if err != nil || len(plan.AcceptedTokens) != 1 || plan.CorrectionToken != want {
			t.Fatalf("float64 CDF boundary above=%v: plan=%+v err=%v want correction %d", above, plan, err, want)
		}
		if !reflect.DeepEqual(before, in.TargetProbabilityRows[1].Weights) {
			t.Fatal("planner mutated canonical target weights")
		}
		in.DraftEvidence[0].SamplingLaw = DraftSamplingLawUnspecified
		if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrSpecRejectionProbability) {
			t.Fatalf("unspecified draft sampling law admitted: %v", err)
		}
		in.DraftEvidence[0].SamplingLaw = DraftSamplingLawQwen38UnitCDF24
		in.DraftEvidence[0].ProcessedQ = nil
		if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrDraftEvidenceIncomplete) {
			t.Fatalf("log-only q admitted for canonical target: %v", err)
		}
		in.DraftEvidence[0].ProcessedQ = []float32{0.5, 0.5}
		if _, err := BuildSpecRejectionPlan(in); !errors.Is(err, ErrDraftEvidenceIncomplete) {
			t.Fatalf("q inconsistent with original log evidence admitted: %v", err)
		}
	}

	t.Run("actual unit CDF over and under mass", func(t *testing.T) {
		const maxAcceptKey = uint64(0xaf55f34f046b4dbf)
		if got := KeyedUniform64(maxAcceptKey, DomainAcceptCoin, 0); got != 1-0x1p-53 {
			t.Fatalf("boundary key gives %.17g", got)
		}
		for _, raw := range [][]float32{{0.2, 0.8}, {0.1, 0.9}} {
			for _, compact := range []bool{false, true} {
				row := sampledEvidence(0, raw...)
				row.ProcessedQ = append([]float32(nil), raw...)
				row.SamplingLaw = DraftSamplingLawQwen38UnitCDF24
				p := []float64{float64(raw[0]), float64(raw[1])}
				wantCorrection := 1
				if compact {
					row.Token, row.TokenIDs = 1, []int{1, 0}
					p[0], p[1] = p[1], p[0]
					wantCorrection = 0
				}
				q, err := expandDraftSamplingLaw(row, 2)
				if err != nil || q[0]+q[1] != 1 {
					t.Fatalf("derived q = %v, %v", q, err)
				}
				// Check both sides of the only CDF boundary against the actual
				// producer sampler, plus its endpoint/fallback behavior.
				boundary := int(math.Ceil(float64(raw[0]) * (1 << 24)))
				for _, draw := range []int{0, boundary - 1, boundary, (1 << 24) - 1} {
					u := float32(draw) / (1 << 24)
					got := (*Qwen38SampledSpeculativeVerifier)(nil).SampleFromDistribution(raw, u)
					want := 1
					if draw < boundary {
						want = 0
					}
					if got != want {
						t.Fatalf("actual sampler draw %d: %d, want %d", draw, got, want)
					}
				}
				in := SpecRejectionPlanInput{
					DraftTokens: []int{row.Token}, DraftEvidence: []DraftTokenEvidence{row}, RequestKey: maxAcceptKey,
					TargetProbabilityRows: []SpecProbabilityRow{{Weights: p, Mass: p[0] + p[1]}, {Weights: p, Mass: p[0] + p[1]}},
				}
				plan, err := BuildSpecRejectionPlan(in)
				if err != nil || plan.RejectedAt != 0 || plan.CorrectionToken != wantCorrection {
					t.Fatalf("raw=%v compact=%v: plan=%+v err=%v", raw, compact, plan, err)
				}
				// Identical scaled dyadic masses must not round alpha below 1
				// while the consistently scaled residual is exactly zero.
				for i := range in.TargetProbabilityRows {
					in.TargetProbabilityRows[i] = SpecProbabilityRow{Weights: []float64{q[0] * 7, q[1] * 7}, Mass: 7}
				}
				plan, err = BuildSpecRejectionPlan(in)
				if err != nil || len(plan.AcceptedTokens) != 1 {
					t.Fatalf("identical scaled law: plan=%+v err=%v", plan, err)
				}
			}
		}
	})

	t.Run("refuse conditioned or unreachable q", func(t *testing.T) {
		for _, row := range []DraftTokenEvidence{
			{Token: 0, ProcessedQ: []float32{0.1, 0.8999999, 0}},
			{Token: 1, ProcessedQ: []float32{1, math.SmallestNonzeroFloat32}},
		} {
			row.LogProbs = specLogRow(row.ProcessedQ...)
			row.Provenance, row.SamplingLaw = DraftEvidenceSampledProcessedQ, DraftSamplingLawQwen38UnitCDF24
			p := make([]float64, len(row.ProcessedQ))
			p[0] = 1
			_, err := BuildSpecRejectionPlan(SpecRejectionPlanInput{
				DraftTokens: []int{row.Token}, DraftEvidence: []DraftTokenEvidence{row}, RequestKey: 1,
				TargetProbabilityRows: []SpecProbabilityRow{{Weights: p, Mass: 1}, {Weights: p, Mass: 1}},
			})
			if !errors.Is(err, ErrSpecRejectionProbability) {
				t.Fatalf("unsupported actual law admitted: %v, %v", row.ProcessedQ, err)
			}
		}
	})
}
