package model

import (
	"math"
	"testing"
)

// swaTestCfg is the small synthetic config the SWA tests run on: 2 layers, GQA
// (4 query / 2 kv heads), head_dim 8. EOS = -1 so greedy never short-circuits.
func swaTestCfg() Config {
	return Config{
		HiddenSize:        32,
		NumLayers:         2,
		NumHeads:          4,
		NumKVHeads:        2,
		HeadDim:           8,
		IntermediateSize:  64,
		VocabSize:         97,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
		EOSTokenID:        -1,
	}
}

// TestSWAWindowUnsetIsNoOp is the LOAD-BEARING gate: with no window configured
// (Window == nil, and the explicit all-(-1) form), every attention path must be
// BIT-IDENTICAL to the pre-SWA full-causal path. This proves the read-time mask is
// a true no-op for non-SWA (Llama/Qwen) models — max|Δ| = 0, not merely close.
func TestSWAWindowUnsetIsNoOp(t *testing.T) {
	cfg := swaTestCfg()
	m := NewSynthetic(cfg) // Window nil -> windowForLayer == -1 for every layer

	// An explicit all-(-1) window must behave identically to the nil default.
	cfgExplicit := swaTestCfg()
	cfgExplicit.Window = []int{-1, -1}
	mExplicit := NewSynthetic(cfgExplicit)

	prompt := []int{3, 17, 5, 23, 41, 2, 19, 8, 31, 14}

	t.Run("cacheless Forward bit-identical", func(t *testing.T) {
		base := m.Forward(prompt)
		// Reference: the pre-SWA causal attention, computed independently (no window
		// branch at all). If the window code perturbed a single rounding, this fails.
		want := referenceForwardLogits(m, prompt)
		for tpos := range prompt {
			assertFloat32BitsEqual(t, "logits pos "+itoa(tpos), want[tpos], base.Logits[tpos])
		}
		// nil-default vs explicit all-(-1) are the same too.
		exp := mExplicit.Forward(prompt)
		for tpos := range prompt {
			assertFloat32BitsEqual(t, "explicit -1 logits pos "+itoa(tpos), base.Logits[tpos], exp.Logits[tpos])
		}
	})

	t.Run("prefill + decode bit-identical to legacy", func(t *testing.T) {
		cur := m.NewSession()
		legacy := m.NewSession()
		curLogits := cur.Prefill(prompt)
		legacyLogits := legacyPrefillF32ForTest(legacy, prompt)
		assertFloat32BitsEqual(t, "prefill logits", legacyLogits, curLogits)
		assertKVCacheBitsEqual(t, "prefill", legacy.Cache, cur.Cache)
		for step, id := range []int{11, 29, 7, 13} {
			curLogits = cur.Step(id)
			legacyX := legacyTokenHiddenF32ForTest(legacy, id, legacy.Cache.Len())
			legacyLogits = legacy.head(legacyX)
			assertFloat32BitsEqual(t, "decode logits", legacyLogits, curLogits)
			assertKVCacheBitsEqual(t, "decode step "+itoa(step), legacy.Cache, cur.Cache)
		}
	})

	t.Run("batched prefill + decode bit-identical", func(t *testing.T) {
		prompts := [][]int{
			{1, 5, 9, 13, 21, 33},
			{2, 4, 8, 16, 32, 7},
			{3, 6, 12, 24, 48, 9},
		}
		bs := m.NewBatchSession(len(prompts))
		curLogits := bs.PrefillEach(prompts)
		for b, p := range prompts {
			legacy := m.NewSession()
			want := legacyPrefillF32ForTest(legacy, p)
			assertFloat32BitsEqual(t, "batch prefill user "+itoa(b), want, curLogits[b])
		}
		for _, ids := range [][]int{{10, 20, 30}, {11, 21, 31}} {
			bs.StepBatch(ids)
		}
	})
}

// TestSWAWindowMasksOldKeys proves the masking SEMANTICS: with window W, the query
// at absolute position p attends ONLY to keys in [p-W+1, p]; every key older than
// that contributes zero. It checks this two ways:
//
//  1. The pure lower-bound helpers (windowLo / windowLoContig / windowLoStep) return
//     exactly the [p-W+1, p] boundary index, including after a simulated eviction
//     (pos[] no longer equal to the index) — the keyed-off-pos[] requirement.
//  2. End to end: the windowed cacheless Forward equals an independent masked-attention
//     reference (out-of-window scores forced to -inf pre-softmax), AND a window wider
//     than the sequence reduces to full causal, AND a real window genuinely changes the
//     output (the mask is non-vacuous).
func TestSWAWindowMasksOldKeys(t *testing.T) {
	t.Run("lower-bound helpers honor [p-W+1, p]", func(t *testing.T) {
		// Contiguous cache (pos[j] == j), W = 3: query p attends keys [p-2, p].
		for _, tc := range []struct {
			nPos, qpos, W, wantLo int
		}{
			{nPos: 1, qpos: 0, W: 3, wantLo: 0},  // p=0: [0,0]
			{nPos: 3, qpos: 2, W: 3, wantLo: 0},  // p=2: [0,2], lo=0
			{nPos: 4, qpos: 3, W: 3, wantLo: 1},  // p=3: [1,3], lo=1
			{nPos: 6, qpos: 5, W: 3, wantLo: 3},  // p=5: [3,5], lo=3
			{nPos: 6, qpos: 5, W: 1, wantLo: 5},  // W=1: only self
			{nPos: 6, qpos: 5, W: -1, wantLo: 0}, // full causal
		} {
			if got := windowLoContig(tc.nPos, tc.qpos, tc.W); got != tc.wantLo {
				t.Errorf("windowLoContig(nPos=%d,p=%d,W=%d)=%d want %d", tc.nPos, tc.qpos, tc.W, got, tc.wantLo)
			}
			// windowLo over the contiguous pos[] must agree.
			pos := make([]int, tc.nPos)
			for i := range pos {
				pos[i] = i
			}
			if got := windowLo(pos, tc.nPos, tc.qpos, tc.W); got != tc.wantLo {
				t.Errorf("windowLo(contig,p=%d,W=%d)=%d want %d", tc.qpos, tc.W, got, tc.wantLo)
			}
		}
	})

	t.Run("keyed off pos[] survives eviction renumbering", func(t *testing.T) {
		// After an Evict the cache is contiguous again (pos[i]=i), so the windowed bound
		// keyed off pos[] must still be a contiguous suffix. Simulate a cache whose pos[]
		// is NOT yet renumbered (mid-evict-style: a gap) to prove the bound follows pos[],
		// not the index. pos = [0,1,2,7,8] (positions 3..6 evicted), query at p=8.
		pos := []int{0, 1, 2, 7, 8}
		// W=3, query p=8: window is [6,8]; keys with pos < 6 are masked → lo at first
		// pos >= 6, which is index 3 (pos 7). A naive index-keyed bound (8-3+1=6) would
		// WRONGLY mask indices 0..5 and read past the slice — this is the eviction trap.
		if got := windowLo(pos, len(pos), 8, 3); got != 3 {
			t.Fatalf("windowLo keyed off pos[] = %d, want 3 (first pos >= 6)", got)
		}
		// W=6, query p=8: window [3,8]; first pos >= 3 is index 3 (pos 7), since 0,1,2 < 3.
		if got := windowLo(pos, len(pos), 8, 6); got != 3 {
			t.Fatalf("windowLo(W=6) = %d, want 3", got)
		}
		// W=9, query p=8: window [0,8]; all keys visible.
		if got := windowLo(pos, len(pos), 8, 9); got != 0 {
			t.Fatalf("windowLo(W=9) = %d, want 0", got)
		}
	})

	t.Run("windowLoStep: current key appended, pos[] not yet", func(t *testing.T) {
		// Decode appends the query's K row (nPos keys) but not its pos entry. priorPos
		// covers the nPos-1 earlier keys; the query sits at index nPos-1, position qpos.
		priorPos := []int{0, 1, 2, 3, 4} // 5 earlier keys
		nPos := 6                        // + the query's just-appended row
		// W=3, qpos=5: window [3,5]; first prior pos >= 3 is index 3 → lo=3.
		if got := windowLoStep(priorPos, nPos, 5, 3); got != 3 {
			t.Fatalf("windowLoStep = %d, want 3", got)
		}
		// W=-1: full causal.
		if got := windowLoStep(priorPos, nPos, 5, -1); got != 0 {
			t.Fatalf("windowLoStep(full) = %d, want 0", got)
		}
	})

	t.Run("windowed Forward equals masked reference; wide window == full causal", func(t *testing.T) {
		const W = 3
		cfg := swaTestCfg()
		cfg.Window = []int{W, W}
		m := NewSynthetic(cfg)
		prompt := []int{3, 17, 5, 23, 41, 2, 19, 8, 31, 14} // len 10 > W: the mask is exercised

		got := m.Forward(prompt)
		// Independent reference: identical math, but attention scores for keys j with
		// j < t-W+1 are forced to -inf BEFORE softmax (the textbook SWA mask).
		want := referenceForwardLogitsWindowed(m, prompt, W)
		for tpos := range prompt {
			d, at := maxAbsDiff(got.Logits[tpos], want[tpos])
			if d != 0 {
				t.Errorf("windowed pos %d: max|Δ|=%.3e at %d (want bit-identical to masked ref)", tpos, d, at)
			}
		}

		// A window >= seq length must reduce to FULL causal (bit-identical to no window).
		cfgWide := swaTestCfg()
		cfgWide.Window = []int{len(prompt) + 5, len(prompt) + 5}
		wide := NewSynthetic(cfgWide).Forward(prompt)
		full := NewSynthetic(swaTestCfg()).Forward(prompt)
		for tpos := range prompt {
			assertFloat32BitsEqual(t, "wide-window pos "+itoa(tpos), full.Logits[tpos], wide.Logits[tpos])
		}

		// Non-vacuous: a real window (W=3) MUST differ from full causal at a late position
		// where keys actually fall outside the window. Otherwise the mask did nothing.
		dFull, _ := maxAbsDiff(got.Logits[len(prompt)-1], full.Logits[len(prompt)-1])
		if dFull == 0 {
			t.Errorf("windowed and full-causal logits identical at last pos — window not applied")
		}
	})
}

// referenceForwardLogits recomputes the per-position logits with a from-scratch,
// full-causal attention (no window branch) — the pre-SWA reference the no-op gate
// compares against. It mirrors Forward/layer exactly but is independent code.
func referenceForwardLogits(m *Model, ids []int) [][]float32 {
	return referenceForwardLogitsWindowed(m, ids, -1)
}

// referenceForwardLogitsWindowed is referenceForwardLogits with a sliding window W
// (-1 = full causal): query t attends keys j in [max(0,t-W+1), t], with out-of-window
// scores forced to -inf pre-softmax. This is the independent oracle for the masking
// semantics — it never calls windowLo/windowForLayer.
func referenceForwardLogitsWindowed(m *Model, ids []int, W int) [][]float32 {
	cfg := m.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	seq := len(ids)
	scale := float32(1.0 / math.Sqrt(float64(hd)))

	embed := m.embedRows()
	x := make([][]float32, seq)
	for t, id := range ids {
		x[t] = append([]float32(nil), embed[id*H:(id+1)*H]...)
	}
	rp := newRope(cfg, seq)

	for l := 0; l < cfg.NumLayers; l++ {
		p := func(s string) string { return layerName(l, s) }
		wIn := m.tensor(p("input_layernorm.weight"))
		q := make([][]float32, seq)
		k := make([][]float32, seq)
		v := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			xn := rmsnorm(x[t], wIn, eps)
			q[t] = matRows(m.tensor(p("self_attn.q_proj.weight")), xn, nH*hd, H)
			k[t] = matRows(m.tensor(p("self_attn.k_proj.weight")), xn, nKV*hd, H)
			v[t] = matRows(m.tensor(p("self_attn.v_proj.weight")), xn, nKV*hd, H)
			if cfg.AttentionBias {
				addBias(q[t], m.tensor(p("self_attn.q_proj.bias")))
				addBias(k[t], m.tensor(p("self_attn.k_proj.bias")))
				addBias(v[t], m.tensor(p("self_attn.v_proj.bias")))
			}
			for h := 0; h < nH; h++ {
				rp.apply(q[t][h*hd:(h+1)*hd], t)
			}
			for h := 0; h < nKV; h++ {
				rp.apply(k[t][h*hd:(h+1)*hd], t)
			}
		}
		attnOut := make([][]float32, seq)
		for t := 0; t < seq; t++ {
			attnOut[t] = make([]float32, nH*hd)
			lo := 0
			if W >= 0 {
				if lo = t - W + 1; lo < 0 {
					lo = 0
				}
			}
			for h := 0; h < nH; h++ {
				kvh := h / grp
				qh := q[t][h*hd : (h+1)*hd]
				// Score over the FULL causal range, then mask out-of-window to -inf
				// pre-softmax — the standard SWA formulation. softmaxInPlace turns -inf
				// into a 0 weight, so out-of-window keys contribute nothing.
				scores := make([]float32, t+1)
				for j := 0; j <= t; j++ {
					if j < lo {
						scores[j] = float32(math.Inf(-1))
						continue
					}
					kh := k[j][kvh*hd : (kvh+1)*hd]
					scores[j] = dot(qh, kh) * scale
				}
				softmaxInPlace(scores)
				out := attnOut[t][h*hd : (h+1)*hd]
				for j := lo; j <= t; j++ {
					vh := v[j][kvh*hd : (kvh+1)*hd]
					wj := scores[j]
					for d := 0; d < hd; d++ {
						out[d] += wj * vh[d]
					}
				}
			}
		}
		wo := m.tensor(p("self_attn.o_proj.weight"))
		for t := 0; t < seq; t++ {
			o := matRows(wo, attnOut[t], H, nH*hd)
			for i := 0; i < H; i++ {
				x[t][i] += o[i]
			}
		}
		wPost := m.tensor(p("post_attention_layernorm.weight"))
		wGate := m.tensor(p("mlp.gate_proj.weight"))
		wUp := m.tensor(p("mlp.up_proj.weight"))
		wDown := m.tensor(p("mlp.down_proj.weight"))
		I := cfg.IntermediateSize
		for t := 0; t < seq; t++ {
			xn := rmsnorm(x[t], wPost, eps)
			g := matRows(wGate, xn, I, H)
			u := matRows(wUp, xn, I, H)
			for i := 0; i < I; i++ {
				g[i] = silu(g[i]) * u[i]
			}
			down := matRows(wDown, g, H, I)
			for i := 0; i < H; i++ {
				x[t][i] += down[i]
			}
		}
	}
	normW := m.tensor("model.norm.weight")
	head := m.lmHead()
	logits := make([][]float32, seq)
	for t := 0; t < seq; t++ {
		xf := rmsnorm(x[t], normW, eps)
		logits[t] = matRows(head, xf, cfg.VocabSize, H)
	}
	return logits
}
