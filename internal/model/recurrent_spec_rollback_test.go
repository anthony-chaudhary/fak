package model

import (
	"errors"
	"math"
	"testing"
)

// stepGatedDeltaNet simulates one step of Gated DeltaNet recurrence:
// S'_t = gate_t * S_{t-1}
// kvmem = S'_t * k_t
// delta = beta_t * (v_t - kvmem)
// S_t = S'_t + delta * k_t^T
func stepGatedDeltaNet(state []float32, q, k, v []float32, beta, gate float32, d int) []float32 {
	newState := make([]float32, len(state))
	copy(newState, state)

	// 1. Decay state
	for i := range newState {
		newState[i] *= gate
	}

	// 2. kvmem = S'_t * k_t
	kvmem := make([]float32, d)
	for dv := 0; dv < d; dv++ {
		stRow := newState[dv*d : (dv+1)*d]
		var sum float32
		for dk := 0; dk < d; dk++ {
			sum += stRow[dk] * k[dk]
		}
		kvmem[dv] = sum
	}

	// 3. delta = beta_t * (v_t - kvmem)
	delta := make([]float32, d)
	for dv := 0; dv < d; dv++ {
		delta[dv] = beta * (v[dv] - kvmem[dv])
	}

	// 4. S_t = S'_t + delta * k_t^T
	for dv := 0; dv < d; dv++ {
		dVal := delta[dv]
		stRow := newState[dv*d : (dv+1)*d]
		for dk := 0; dk < d; dk++ {
			stRow[dk] += dVal * k[dk]
		}
	}

	return newState
}

func TestRecurrentSpecRollback(t *testing.T) {
	t.Run("DeltaNetStateParityAndAccumulation", func(t *testing.T) {
		const (
			d     = 16
			steps = 16
		)
		stateSize := d * d

		// Synthetic sequence of true tokens and rejected draft tokens
		type stepInputs struct {
			q, k, v           []float32
			beta, gate        float32
			qDraft, kDraft    []float32
			vDraft            []float32
			betaDraft, gDraft float32
		}

		inputs := make([]stepInputs, steps)
		for s := 0; s < steps; s++ {
			inp := stepInputs{
				q:         make([]float32, d),
				k:         make([]float32, d),
				v:         make([]float32, d),
				beta:      0.75 + 0.01*float32(s%5),
				gate:      0.92 + 0.005*float32(s%3),
				qDraft:    make([]float32, d),
				kDraft:    make([]float32, d),
				vDraft:    make([]float32, d),
				betaDraft: 0.60 + 0.02*float32(s%4),
				gDraft:    0.88 + 0.01*float32(s%2),
			}
			for i := 0; i < d; i++ {
				f := float32(i + 1)
				sf := float32(s + 1)
				inp.q[i] = float32(math.Sin(float64(f*0.3 + sf*0.1)))
				inp.k[i] = float32(math.Cos(float64(f*0.4 + sf*0.2)))
				inp.v[i] = float32(math.Sin(float64(f*0.5 + sf*0.3)))

				// Draft token vectors are divergent
				inp.qDraft[i] = float32(math.Cos(float64(f*0.7 + sf*0.5)))
				inp.kDraft[i] = float32(math.Sin(float64(f*0.8 + sf*0.6)))
				inp.vDraft[i] = float32(math.Cos(float64(f*0.9 + sf*0.7)))
			}
			inputs[s] = inp
		}

		// 1. Reference sequential non-speculative autoregressive decode
		seqState := make([]float32, stateSize)
		for s := 0; s < steps; s++ {
			inp := inputs[s]
			seqState = stepGatedDeltaNet(seqState, inp.q, inp.k, inp.v, inp.beta, inp.gate, d)
		}

		// 2. Speculative decoding WITH snapshot rollback
		rollbackMgr := NewRecurrentStateRollbackManager()
		specRollbackState := make([]float32, stateSize)

		for s := 0; s < steps; s++ {
			inp := inputs[s]

			// Prior to speculative drafting, take an auxiliary snapshot
			rollbackMgr.Checkpoint(s, specRollbackState)

			// Speculatively advance state with draft token
			draftContaminated := stepGatedDeltaNet(specRollbackState, inp.qDraft, inp.kDraft, inp.vDraft, inp.betaDraft, inp.gDraft, d)

			// Target verification rejects the draft token; rollback to verified step
			restored, err := rollbackMgr.Rollback(s)
			if err != nil {
				t.Fatalf("step %d rollback failed: %v", s, err)
			}
			specRollbackState = restored

			// Execute true verified token step on pristine restored state
			specRollbackState = stepGatedDeltaNet(specRollbackState, inp.q, inp.k, inp.v, inp.beta, inp.gate, d)

			// Commit prunes older checkpoints
			rollbackMgr.Commit(s)

			// Verify that draftContaminated was not used
			if len(draftContaminated) != stateSize {
				t.Fatalf("unexpected draft state size %d", len(draftContaminated))
			}
		}

		// 3. Speculative decoding WITHOUT rollback (error accumulates)
		flawedState := make([]float32, stateSize)
		for s := 0; s < steps; s++ {
			inp := inputs[s]

			// Speculative drafting mutates state
			flawedState = stepGatedDeltaNet(flawedState, inp.qDraft, inp.kDraft, inp.vDraft, inp.betaDraft, inp.gDraft, d)

			// Draft token is rejected, but flawed runner fails to rollback,
			// directly appending the true token on top of corrupted state
			flawedState = stepGatedDeltaNet(flawedState, inp.q, inp.k, inp.v, inp.beta, inp.gate, d)
		}

		// Assertion 1: Rollback state must be bit-identical to sequential non-speculative state
		for i := 0; i < stateSize; i++ {
			seqBits := math.Float32bits(seqState[i])
			rbBits := math.Float32bits(specRollbackState[i])
			if seqBits != rbBits {
				t.Fatalf("bit mismatch at element %d: seq=%v (0x%08x) vs rollback=%v (0x%08x)",
					i, seqState[i], seqBits, specRollbackState[i], rbBits)
			}
		}

		// Assertion 2: Without rollback, state error accumulates
		var maxAbsDiff float32
		var l2Sum float64
		for i := 0; i < stateSize; i++ {
			diff := float64(flawedState[i] - seqState[i])
			l2Sum += diff * diff
			if absDiff := float32(math.Abs(diff)); absDiff > maxAbsDiff {
				maxAbsDiff = absDiff
			}
		}
		l2Diff := math.Sqrt(l2Sum)

		t.Logf("State error without rollback: maxAbsDiff=%e, L2Diff=%e", maxAbsDiff, l2Diff)
		if maxAbsDiff < 1e-3 || l2Diff < 1e-3 {
			t.Fatalf("expected substantial accumulated error without rollback, got maxAbsDiff=%v, l2Diff=%v",
				maxAbsDiff, l2Diff)
		}
	})

	t.Run("OutputEntropyMonitorCircuitBreaker", func(t *testing.T) {
		monitor := NewOutputEntropyMonitor()

		// 1. Emitting diverse tokens should not trip circuit breaker
		variedTokens := []int{101, 204, 305, 402, 101, 204, 506, 708}
		for i, tok := range variedTokens {
			if err := monitor.Record(tok); err != nil {
				t.Fatalf("unexpected error on varied token %d (val %d): %v", i, tok, err)
			}
			if monitor.Tripped() {
				t.Fatalf("monitor tripped prematurely on varied token %d", i)
			}
		}
		if monitor.ConsecutiveCount() != 1 {
			t.Fatalf("expected consecutive count 1, got %d", monitor.ConsecutiveCount())
		}
		if ent := monitor.Entropy(); ent <= 0.0 {
			t.Fatalf("expected positive entropy for varied tokens, got %f", ent)
		}

		// 2. Reset and emit exactly 15 consecutive identical tokens (token 0)
		monitor.Reset()
		if monitor.Tripped() {
			t.Fatal("monitor tripped after Reset()")
		}

		for i := 1; i <= 15; i++ {
			if err := monitor.Record(0); err != nil {
				t.Fatalf("step %d (consecutive %d) unexpected error: %v", i, i, err)
			}
			if monitor.Tripped() {
				t.Fatalf("monitor tripped prematurely at %d tokens", i)
			}
			if c := monitor.ConsecutiveCount(); c != i {
				t.Fatalf("expected consecutive count %d, got %d", i, c)
			}
		}

		// 3. Emitting the 16th consecutive identical token MUST trip ErrTokenCollapse
		err := monitor.Record(0)
		if !errors.Is(err, ErrTokenCollapse) {
			t.Fatalf("expected ErrTokenCollapse at 16th token, got: %v", err)
		}
		if !monitor.Tripped() {
			t.Fatal("expected Tripped() == true after 16 consecutive identical tokens")
		}
		if c := monitor.ConsecutiveCount(); c != 16 {
			t.Fatalf("expected consecutive count 16, got %d", c)
		}
		if ent := monitor.Entropy(); ent != 0.0 {
			t.Fatalf("expected 0.0 entropy for collapsed token sequence, got %f", ent)
		}

		// 4. Subsequent token calls while tripped remain tripped
		if err := monitor.Record(0); !errors.Is(err, ErrTokenCollapse) {
			t.Fatalf("expected ErrTokenCollapse while tripped, got: %v", err)
		}
		if err := monitor.Observe(999); !errors.Is(err, ErrTokenCollapse) {
			t.Fatalf("expected ErrTokenCollapse on Observe() while tripped, got: %v", err)
		}

		// 5. Test another token value (e.g. token 42) with custom threshold
		customMonitor := NewOutputEntropyMonitor(16)
		for i := 1; i <= 15; i++ {
			if err := customMonitor.Record(42); err != nil {
				t.Fatalf("custom monitor step %d unexpected error: %v", i, err)
			}
		}
		if customMonitor.Tripped() {
			t.Fatal("custom monitor tripped before 16 tokens")
		}
		if err := customMonitor.Record(42); !errors.Is(err, ErrTokenCollapse) {
			t.Fatalf("expected ErrTokenCollapse on 16th token 42, got: %v", err)
		}
	})

	t.Run("RecurrentStateRollbackManagerLifecycle", func(t *testing.T) {
		mgr := NewRecurrentStateRollbackManager()

		state1 := []float32{1.0, 2.0, 3.0}
		state2 := []float32{4.0, 5.0, 6.0}
		state3 := []float32{7.0, 8.0, 9.0}
		state4 := []float32{10.0, 11.0, 12.0}

		mgr.Checkpoint(1, state1)
		mgr.Checkpoint(2, state2)
		mgr.Checkpoint(3, state3)
		mgr.Checkpoint(4, state4)

		if mgr.CheckpointCount() != 4 {
			t.Fatalf("expected 4 checkpoints, got %d", mgr.CheckpointCount())
		}

		// Commit(3) prunes checkpoints older than step 3 (i.e. steps 1 and 2)
		mgr.Commit(3)

		if mgr.HasCheckpoint(1) {
			t.Fatal("step 1 should have been pruned by Commit(3)")
		}
		if mgr.HasCheckpoint(2) {
			t.Fatal("step 2 should have been pruned by Commit(3)")
		}
		if !mgr.HasCheckpoint(3) {
			t.Fatal("step 3 should be retained")
		}
		if !mgr.HasCheckpoint(4) {
			t.Fatal("step 4 should be retained")
		}

		// Rollback to pruned step should fail with ErrCheckpointNotFound
		if _, err := mgr.Rollback(1); !errors.Is(err, ErrCheckpointNotFound) {
			t.Fatalf("expected ErrCheckpointNotFound for step 1, got: %v", err)
		}

		// Rollback to step 3 succeeds and restores state3
		restored, err := mgr.Rollback(3)
		if err != nil {
			t.Fatalf("rollback to step 3 failed: %v", err)
		}
		for i, v := range state3 {
			if restored[i] != v {
				t.Fatalf("restored element %d mismatch: got %v, want %v", i, restored[i], v)
			}
		}

		// Rollback(3) prunes speculative checkpoints created for steps > 3 (i.e. step 4)
		if mgr.HasCheckpoint(4) {
			t.Fatal("step 4 should have been pruned by Rollback(3)")
		}

		// RestoreTo buffer
		dst := make([]float32, 3)
		if err := mgr.RestoreTo(3, dst); err != nil {
			t.Fatalf("RestoreTo failed: %v", err)
		}
		for i, v := range state3 {
			if dst[i] != v {
				t.Fatalf("dst element %d mismatch: got %v, want %v", i, dst[i], v)
			}
		}
	})
}
