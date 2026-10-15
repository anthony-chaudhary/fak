package model

import (
	"errors"
	"math"
	"testing"
)

// fak-test:runtime fast est=500ms lane=default
func TestV41PrefillDiscardedHeadProjection(t *testing.T) {
	m := v41IncrementalPlainModel(t, 2)
	s := m.NewSession()
	t.Cleanup(s.Close)
	count := func() int { return m.V41ExpertFaultAttribution().Prefill.HeadProjectionHostCalls }
	s.PrefillNoLogits([]int{1, 2, 3})
	if count() != 0 || s.NativeForwardCoverage() != nil {
		t.Fatal("cold NoLogits projected a head or claimed logits coverage")
	}
	for _, noLogits := range []bool{false, true} {
		before := count()
		ids := []int{4, 5, 6}
		var got []float32
		if noLogits {
			s.PrefillNoLogits(ids)
		} else {
			got = s.Prefill(ids)
		}
		want := 1
		if noLogits {
			want = 0
		}
		if count()-before != want {
			t.Fatalf("noLogits=%t head projections=%d want %d", noLogits, count()-before, want)
		}
		if noLogits && s.NativeForwardCoverage() != nil {
			t.Fatal("NoLogits claimed logits coverage")
		}
		if !noLogits {
			ref, err := m.forwardV41(s.v41Forward.history, nil)
			if err != nil {
				t.Fatal(err)
			}
			want := lastLogits(ref)
			if len(got) != len(want) {
				t.Fatal("logits width differs")
			}
			for i := range got {
				if math.IsNaN(float64(got[i])) || math.Abs(float64(got[i]-want[i])) > 1e-6 {
					t.Fatal("final requested row differs from reference")
				}
			}
		}
	}
	before := m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls
	if len(s.Step(7)) != m.Cfg.VocabSize || m.V41ExpertFaultAttribution().Decode.HeadProjectionHostCalls != before+1 {
		t.Fatal("ordinary Step did not project its head")
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41NoLogitsTransactionBoundary(t *testing.T) {
	for _, site := range []string{"success", "normalization", "nonfinite", "selected", "cleanup"} {
		t.Run(site, func(t *testing.T) {
			primary := errors.New("injected boundary failure")
			normalized, projected, scaled, commits, rollbacks, cleanups := 0, 0, 0, 0, 0, 0
			tx := &v41StepTransaction{layers: 2, nextLayer: 2}
			tx.headInput = func() ([]float32, error) {
				normalized++
				switch site {
				case "normalization":
					return nil, primary
				case "selected":
					return nil, &V41ProjectionOperationError{Cause: primary}
				case "nonfinite":
					return []float32{float32(math.NaN())}, nil
				}
				return []float32{1}, nil
			}
			tx.headProject = func([]float32) ([]float32, error) { projected++; return nil, nil }
			tx.headFinish = func([]float32, func() error) ([]float32, error) { scaled++; return nil, nil }
			tx.commit = func(withLogits bool) {
				if withLogits {
					t.Fatal("claimed logits")
				}
				commits++
				tx.stats.Committed = true
			}
			tx.rollback = func() { rollbacks++ }
			tx.cleanup = func() {
				cleanups++
				if site == "cleanup" {
					panic(primary)
				}
			}
			var err error
			var panicked any
			func() { defer func() { panicked = recover() }(); err = tx.finalizeNoLogits() }()
			tx.abort()
			wantCommit, wantRollback := 0, 1
			if site == "success" || site == "cleanup" {
				wantCommit, wantRollback = 1, 0
			}
			if normalized != 1 || projected != 0 || scaled != 0 || commits != wantCommit || rollbacks != wantRollback || cleanups != 1 || !tx.closed {
				t.Fatalf("normalization=%d projection=%d scale=%d commit=%d rollback=%d cleanup=%d closed=%t", normalized, projected, scaled, commits, rollbacks, cleanups, tx.closed)
			}
			switch site {
			case "success":
				if err != nil || panicked != nil {
					t.Fatalf("%v %v", err, panicked)
				}
			case "normalization":
				if !errors.Is(err, primary) || panicked != nil {
					t.Fatal("normalization failure changed")
				}
			case "nonfinite":
				if err == nil || panicked != nil {
					t.Fatal("nonfinite input accepted")
				}
			case "selected":
				e, ok := panicked.(error)
				if !ok || !errors.Is(e, primary) {
					t.Fatal("selected failure changed")
				}
			case "cleanup":
				if panicked != primary {
					t.Fatal("cleanup panic changed")
				}
			}
		})
	}
}

// fak-test:runtime fast est=1ms lane=default
func TestV41NoLogitsCaptureOutcome(t *testing.T) {
	c := &v41StepActivationCapture{state: &v41ForwardState{}}
	if c.committedOutcome(true) {
		t.Fatal("ordinary capture accepted missing logits coverage")
	}
	c.stateOnlyCommit = true
	if !c.committedOutcome(true) || c.committedOutcome(false) {
		t.Fatal("state-only capture lost actual commit outcome")
	}
	if c.state.forwardCoverage != nil {
		t.Fatal("state-only capture invented logits coverage")
	}
}
