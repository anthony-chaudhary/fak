package rsiloop

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/shipgate"
)

func multiRegimeFixture(truth bool, scores []float64, completed []int, reps []float64) MultiRegimeSnapshot {
	names := []string{"short", "medium", "long"}
	regimes := make([]RegimeMetrics, len(names))
	for i, name := range names {
		regimes[i] = RegimeMetrics{
			Regime:         name,
			Score:          scores[i],
			Completed:      completed[i],
			RepetitionRate: reps[i],
			Evidence: DOSEvidence{
				ID:        "dos-" + name,
				Subject:   "multi-regime eval " + name,
				Confirmed: true,
				Summary:   "host fixture witness",
			},
		}
	}
	return MultiRegimeSnapshot{
		Ref:        "fixture",
		TruthClean: truth,
		Regimes:    regimes,
	}
}

// fak-test:runtime fast est=1s lane=default
func TestMultiRegimeRSIAcceptanceGate(t *testing.T) {
	t.Parallel()
	policy := DefaultMultiRegimeGatePolicy()
	base := multiRegimeFixture(true, []float64{20, 18, 14}, []int{10, 9, 7}, []float64{0.8, 1.2, 2.0})

	t.Run("strict aggregate gain without regression keeps", func(t *testing.T) {
		cand := multiRegimeFixture(true, []float64{21, 18, 15}, []int{10, 9, 8}, []float64{0.7, 1.1, 2.5})
		dec, err := EvaluateMultiRegimeGate(base, cand, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.KEEP || !dec.Accepted || dec.TieBroken {
			t.Fatalf("got %+v, want KEEP without tie-break", dec)
		}
		if dec.Reason != "aggregate_gain" {
			t.Fatalf("Reason = %q, want aggregate_gain", dec.Reason)
		}
	})

	t.Run("single regime drop past floor reverts even with net gain", func(t *testing.T) {
		// Short +5, long -3 -> net +2, but long dropped > 2.0.
		cand := multiRegimeFixture(true, []float64{25, 18, 11}, []int{12, 9, 6}, []float64{0.5, 1.1, 2.0})
		dec, err := EvaluateMultiRegimeGate(base, cand, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.REVERT || dec.Accepted {
			t.Fatalf("got %+v, want REVERT on single-regime drop floor", dec)
		}
		if dec.Reason != "regime_drop_floor:long" {
			t.Fatalf("Reason = %q, want regime_drop_floor:long", dec.Reason)
		}
	})

	t.Run("per-regime drop at exact floor allowed when aggregate improves", func(t *testing.T) {
		// Short +3, long -2 -> net +1, worst drop == 2.0.
		cand := multiRegimeFixture(true, []float64{23, 18, 12}, []int{11, 9, 6}, []float64{0.8, 1.2, 2.0})
		dec, err := EvaluateMultiRegimeGate(base, cand, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.KEEP || !dec.Accepted {
			t.Fatalf("got %+v, want KEEP at exact drop floor", dec)
		}
	})

	t.Run("repetition rate past max ceiling reverts", func(t *testing.T) {
		// Base max rep = 2.0; limit = max(3.12, 2.0+1.6) = 3.60. Candidate 3.65 must revert.
		cand := multiRegimeFixture(true, []float64{22, 19, 15}, []int{11, 10, 8}, []float64{0.8, 1.2, 3.65})
		dec, err := EvaluateMultiRegimeGate(base, cand, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.REVERT || dec.Reason != "repetition_ceiling" {
			t.Fatalf("got %+v, want REVERT repetition_ceiling", dec)
		}
	})

	t.Run("low-baseline repetition uses RepetitionCeiling floor", func(t *testing.T) {
		lowBase := multiRegimeFixture(true, []float64{20, 18, 14}, []int{10, 9, 7}, []float64{0.5, 0.8, 1.0})
		// Limit = max(3.12, 1.0+1.6) = 3.12. 3.05 passes, 3.20 fails.
		okCand := multiRegimeFixture(true, []float64{21, 18, 14}, []int{10, 9, 7}, []float64{0.5, 0.8, 3.05})
		dec, err := EvaluateMultiRegimeGate(lowBase, okCand, policy)
		if err != nil || !dec.Accepted {
			t.Fatalf("3.05 below 3.12 ceiling should pass, got dec=%+v err=%v", dec, err)
		}
		badCand := multiRegimeFixture(true, []float64{21, 18, 14}, []int{10, 9, 7}, []float64{0.5, 0.8, 3.20})
		dec, err = EvaluateMultiRegimeGate(lowBase, badCand, policy)
		if err != nil || dec.Accepted || dec.Reason != "repetition_ceiling" {
			t.Fatalf("3.20 above 3.12 ceiling should fail, got dec=%+v err=%v", dec, err)
		}
	})

	t.Run("tie-break completion floor rejects below +4 and accepts at +4", func(t *testing.T) {
		// Tie on aggregate score (52 == 52), non-regressing regimes.
		belowTie := multiRegimeFixture(true, []float64{21, 18, 13}, []int{11, 10, 8}, []float64{0.8, 1.2, 2.0}) // +3 completions
		dec, err := EvaluateMultiRegimeGate(base, belowTie, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.REVERT || dec.Reason != "tie_break_completion_floor" {
			t.Fatalf("got %+v, want REVERT tie_break_completion_floor", dec)
		}

		passTie := multiRegimeFixture(true, []float64{21, 18, 13}, []int{12, 10, 8}, []float64{0.8, 1.2, 2.0}) // +4 completions
		dec, err = EvaluateMultiRegimeGate(base, passTie, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec.Decision != shipgate.KEEP || !dec.Accepted || !dec.TieBroken || dec.Reason != "tie_break_completions" {
			t.Fatalf("got %+v, want KEEP tie_break_completions", dec)
		}
	})

	t.Run("shipgate and DOS evidence integration", func(t *testing.T) {
		tiePass := multiRegimeFixture(true, []float64{21, 18, 13}, []int{12, 10, 8}, []float64{0.8, 1.2, 2.0})
		v := LoopVariant{ID: "lv-tie-1", Kind: VariantIterationPolicy, Description: "tie-break completion win"}
		rec, dec, err := EvaluateMultiRegimeLoopVariant(v, base, tiePass, policy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeLoopVariant failed: %v", err)
		}
		if !dec.Accepted || !rec.Kept || rec.Decision != shipgate.KEEP.String() {
			t.Fatalf("expected KEEP LoopVariantRecord, got rec=%+v dec=%+v", rec, dec)
		}

		archPath := filepath.Join(t.TempDir(), "variants.jsonl")
		a, err := NewLoopVariantArchive(archPath)
		if err != nil {
			t.Fatalf("NewLoopVariantArchive: %v", err)
		}
		if err := a.Append(rec); err != nil {
			t.Fatalf("Archive.Append failed: %v", err)
		}
		_ = a.Close()

		// Unconfirmed evidence is rejected when RequireConfirmedEvidence=true.
		strictPolicy := policy
		strictPolicy.RequireConfirmedEvidence = true
		unconfirmed := tiePass
		unconfirmed.Regimes[1].Evidence.Confirmed = false
		dec2, err := EvaluateMultiRegimeGate(base, unconfirmed, strictPolicy)
		if err != nil {
			t.Fatalf("EvaluateMultiRegimeGate failed: %v", err)
		}
		if dec2.Accepted || !strings.HasPrefix(dec2.Reason, "unconfirmed_evidence:") {
			t.Fatalf("expected unconfirmed_evidence revert, got %+v", dec2)
		}
	})

	t.Run("schema validation catches mismatched regimes and invalid policy", func(t *testing.T) {
		badPolicy := policy
		badPolicy.MaxPerRegimeDrop = -0.1
		if _, err := EvaluateMultiRegimeGate(base, base, badPolicy); err == nil {
			t.Fatal("expected error for negative MaxPerRegimeDrop")
		}
		mismatched := multiRegimeFixture(true, []float64{20, 18, 14}, []int{10, 9, 7}, []float64{0.8, 1.2, 2.0})
		mismatched.Regimes[2].Regime = "ultra_long"
		if _, err := EvaluateMultiRegimeGate(base, mismatched, policy); err == nil {
			t.Fatal("expected error for mismatched regime name")
		}
	})
}
