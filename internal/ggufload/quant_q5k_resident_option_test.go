package ggufload

import "testing"

// TestDenseQ5KResidentOptionSelectsOnlyQ5K proves the selective Q5_K option keeps an eligible
// dense Q5_K tensor packed in the raw resident store at file bytes, while a Q6_K tensor under the
// same Q5-only flag stays on the dequant-to-Q8 fallback, with blanket dense residency disabled.
// fak-test:runtime fast est=50ms lane=default
func TestDenseQ5KResidentOptionSelectsOnlyQ5K(t *testing.T) {
	const (
		dim   = 256
		vocab = 4
	)
	q5Path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ5_K, false, false)
	mQ5, err := LoadModelQ4KProfileOptions(q5Path, nil, WithDenseKQuantResident(false), WithDenseQ5KResident(true))
	if err != nil {
		t.Fatalf("load with selective Q5_K residency: %v", err)
	}
	if raw, ok := mQ5.KQuantRaw("lm_head.weight"); !ok || len(raw) != vocab*blockQ5KBytes {
		t.Fatalf("resident Q5_K head bytes = %d, %v; want %d, true", len(raw), ok, vocab*blockQ5KBytes)
	}
	if mQ5.HasQ8("lm_head.weight") {
		t.Fatal("selective Q5_K option also sent the Q5_K head through the Q8 fallback")
	}

	q6Path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ6_K, false, false)
	mQ6, err := LoadModelQ4KProfileOptions(q6Path, nil, WithDenseKQuantResident(false), WithDenseQ5KResident(true))
	if err != nil {
		t.Fatalf("load with selective Q5_K residency over a Q6_K head: %v", err)
	}
	if mQ6.HasKQuant("lm_head.weight") || !mQ6.HasQ8("lm_head.weight") {
		t.Fatal("Q5-only residency changed the Q6_K head's Q8 fallback")
	}

	mDef, err := LoadModelQ4KProfileOptions(q5Path, nil, WithDenseKQuantResident(false))
	if err != nil {
		t.Fatalf("load without selective Q5_K residency: %v", err)
	}
	if mDef.HasKQuant("lm_head.weight") || !mDef.HasQ8("lm_head.weight") {
		t.Fatal("without the option a dense Q5_K head must take the Q8 fallback")
	}
}

// TestEstimateDenseQ5KResidentOptionParity proves the estimate charges the Q5_K head at packed
// bytes exactly when the loader retains it, so admission and residency cannot disagree.
// fak-test:runtime fast est=50ms lane=default
func TestEstimateDenseQ5KResidentOptionParity(t *testing.T) {
	const (
		dim   = 256
		vocab = 4
	)
	path := buildQwen35GGUFFixture(t, "qwen35", dim, vocab, TensorQ2_K, TensorQ5_K, false, false)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	q5Bytes := func(opts ...Q4KLoadOption) int64 {
		plan, err := ws.EstimateQ4KLoadMemoryPlan(opts...)
		if err != nil {
			t.Fatal(err)
		}
		var n int64
		for _, row := range plan {
			if row.DType == "q5_k" {
				n += row.Bytes
			}
		}
		return n
	}
	if got, want := q5Bytes(WithDenseKQuantResident(false), WithDenseQ5KResident(true)), int64(vocab*blockQ5KBytes); got != want {
		t.Fatalf("Q5-retained estimate q5_k bytes = %d, want %d", got, want)
	}
	if got := q5Bytes(WithDenseKQuantResident(false)); got != 0 {
		t.Fatalf("Q5-off estimate q5_k bytes = %d, want 0", got)
	}
}
