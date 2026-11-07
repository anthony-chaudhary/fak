package model

import "testing"

func verifyMoERowsBitEqual(t *testing.T, what string, want, got [][]float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: %d logit rows, want %d", what, len(got), len(want))
	}
	for j := range want {
		if !float32BitsEqual(want[j], got[j]) {
			d, i := maxAbsDiff(want[j], got[j])
			t.Fatalf("%s: row %d differs from serial Step (max abs %.3e at %d)", what, j, d, i)
		}
	}
}

// TestVerifyForwardMoEChainMatchesSerial pins the batched verify lane for the admitted MoE
// subset (prefillBatchedMoEOK) to the serial Step loop bit-for-bit: every draft row's
// logits and every layer's K/Kraw/V. Verify runs as a single pass (no per-token blockStep
// attention), and a tree mask that admits exactly the causal chain reproduces the chain.
// Before this lane, verifyForwardBatchedOK refused every MoE arch, so the chain took the
// sequential fallback and the tree verify returned nil. [SW-VERIFIED]
// fak-test:runtime fast est=100ms lane=default
func TestVerifyForwardMoEChainMatchesSerial(t *testing.T) {
	cfg := prefillBatchedMoETestConfig()
	m := NewSyntheticMoE(cfg)
	prompt := prefillBatchedMoEPrompt(cfg, 9)
	drafts := []int{13, 17, 4, 60, 5, 42}

	ref := m.NewSession()
	ref.Prefill(prompt)
	want := make([][]float32, len(drafts))
	for j, d := range drafts {
		want[j] = append([]float32(nil), ref.Step(d)...)
	}

	ver := m.NewSession()
	ver.Prefill(prompt)
	if !verifyForwardBatchedOK(ver) {
		t.Fatal("premise broken: the admitted MoE session is refused by verifyForwardBatchedOK")
	}
	ver.PhaseProfiler = NewPhaseProfiler()
	got := ver.VerifyForward(drafts, nil, nil)
	if st := ver.PhaseProfiler.stat["full_attn_decode"]; st != nil {
		t.Fatalf("MoE VerifyForward ran blockStep attention %d times, want the single-pass batched lane", st.Calls)
	}
	verifyMoERowsBitEqual(t, "chain verify", want, got)
	assertPrefillCachesBitEqual(t, cfg, ref, ver)

	tree := m.NewSession()
	tree.Prefill(prompt)
	treeRows := tree.VerifyForward(drafts, nil, func(q, k int) bool { return k <= q })
	if treeRows == nil {
		t.Fatal("tree VerifyForward declined an admitted MoE session")
	}
	verifyMoERowsBitEqual(t, "causal-mask tree verify", want, treeRows)
}

// TestVerifyForwardMoEExcludedKeepsSequential pins the admission boundary: an MoE arch
// with an axis the batched verify does not model, and an admitted arch whose session
// carries target-hidden capture or a residual hook, stay on the sequential Step fallback
// (chain still matches Step bit-for-bit; tree verify declines). [SW-VERIFIED]
// fak-test:runtime fast est=100ms lane=default
func TestVerifyForwardMoEExcludedKeepsSequential(t *testing.T) {
	prompt := []int{3, 17, 5, 23, 41}
	drafts := []int{13, 4, 60}
	check := func(t *testing.T, m *Model, setup func(*Session)) {
		t.Helper()
		ref := m.NewSession()
		setup(ref)
		ref.Prefill(prompt)
		want := make([][]float32, len(drafts))
		for j, d := range drafts {
			want[j] = append([]float32(nil), ref.Step(d)...)
		}
		ver := m.NewSession()
		setup(ver)
		ver.Prefill(prompt)
		if verifyForwardBatchedOK(ver) {
			t.Fatal("excluded MoE session admitted to the batched verify lane")
		}
		ver.PhaseProfiler = NewPhaseProfiler()
		got := ver.VerifyForward(drafts, nil, nil)
		if st := ver.PhaseProfiler.stat["full_attn_decode"]; st == nil || st.Calls != len(drafts)*m.Cfg.NumLayers {
			t.Fatalf("excluded MoE verify did not take the sequential fallback (full_attn_decode=%v)", st)
		}
		verifyMoERowsBitEqual(t, "sequential fallback", want, got)
		if rows := ver.VerifyForward(drafts, nil, func(q, k int) bool { return k <= q }); rows != nil {
			t.Fatal("tree verify must decline outside the batched lane")
		}
	}

	t.Run("sandwich-norm", func(t *testing.T) {
		cfg := prefillBatchedMoETestConfig()
		cfg.BlockTopology = SandwichNorm
		check(t, NewSyntheticMoE(cfg), func(*Session) {})
	})
	t.Run("layer-rope-theta", func(t *testing.T) {
		cfg := prefillBatchedMoETestConfig()
		cfg.RopeThetaPerLayer = []float64{10000, 500000}
		check(t, NewSyntheticMoE(cfg), func(*Session) {})
	})
	t.Run("target-hidden-capture", func(t *testing.T) {
		check(t, NewSyntheticMoE(prefillBatchedMoETestConfig()), func(s *Session) { s.captureTargetHidden = true })
	})
	t.Run("residual-hook", func(t *testing.T) {
		cfg := prefillBatchedMoETestConfig()
		cfg.EnableResidualHook = true
		m := NewSyntheticMoE(cfg)
		m.SetResidualHook(func(int, []float32) {})
		check(t, m, func(*Session) {})
	})
}
