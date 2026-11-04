//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

// TestFattnMmaHd256Parity holds the tensor-core hd256 prompt-attention port to the SIMT
// scalar oracle on the dense-HAL panel ABI (fcuda_causal_attention_panel_f32, reached here
// through PromptPanelAttention with an F32 KV store), the route a P>=128 prefill takes.
//
// The fixture gives the greedy decode a closed answer: every visible key scores
// identically (a uniform causal softmax over the window), and every value row carries a
// per-KV-head signature dimension whose strength grows with the KV head index. Both
// kernels must therefore emit, on every output row, the last KV head group's signature
// dimension as the argmax, with a ~0.1 margin no f16 staging noise can flip. Closed
// contract per prompt length P in {128, 512, 2048}:
//   - FAK_CUDA_QWEN38_PREFILL_ATTN=simt never launches the mma kernel;
//   - =mma routes exactly one launch through fattn_mma_hd256 (a silent SIMT fallback is a
//     failure, not a pass);
//   - both modes greedy-decode the identical closed token on every row, and the mma output
//     stays above the recorded mma-vs-simt cosine floor.
//
// Below the P>=128 gate, and outside the 24x4xhd256 instantiation, the panel path keeps the
// SIMT kernel even when mma is selected, bitwise equal to the forced-simt run.
//
// fak-test:runtime slow est=30s
func TestFattnMmaHd256Parity(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	const (
		heads   = qwen38CausalAttentionPanelHeads
		kvHeads = qwen38CausalAttentionPanelKVHeads
		headDim = qwen38CausalAttentionPanelHeadDim
		groups  = heads / kvHeads
		sigDim  = 128 // signature dimension inside each 256-wide KV head block
	)
	scale := float32(1 / math.Sqrt(headDim))
	// Heads (kvHeads-1)*groups..heads-1 share the strongest signature; the smallest output
	// index among them is the closed greedy token for every row.
	wantToken := (kvHeads-1)*groups*headDim + sigDim
	t.Setenv("FAK_CUDA_PAGED_KV", "") // contiguous F32 store, the panel ABI's native shape

	// fixture builds the P-position workload: Q and K are the fixed +/-1 direction tiled
	// across their widths (every visible key scores identically), and V rows carry the
	// per-KV-head signature 1.0+0.1*kvh at dimension sigDim over an LCG +/-0.25 background.
	fixture := func(p int, seed uint64) (q, k, v []float32) {
		dir := make([]float32, headDim)
		for e := range dir {
			dir[e] = 1.0
			if e%2 == 1 {
				dir[e] = -1.0
			}
		}
		tile := func(n int) []float32 {
			out := make([]float32, n)
			for i := range out {
				out[i] = dir[i%headDim]
			}
			return out
		}
		q = tile(p * heads * headDim)
		k = tile(p * kvHeads * headDim)
		rng := qwen35PanelLCG(seed)
		v = rng.values(p*kvHeads*headDim, 0.5)
		width := kvHeads * headDim
		for pos := 0; pos < p; pos++ {
			for kvh := 0; kvh < kvHeads; kvh++ {
				v[pos*width+kvh*headDim+sigDim] = 1.0 + 0.1*float32(kvh)
			}
		}
		return q, k, v
	}

	// fillKV uploads the fixture K/V rows into a single-layer F32 store.
	fillKV := func(t *testing.T, p int, k, v []float32) KVStore {
		t.Helper()
		cfg := KVConfig{NumLayers: 1, NumKVHeads: kvHeads, HeadDim: headDim, Precision: KVPrecisionF32}
		kv := be.NewKV(cfg)
		width := kvHeads * headDim
		for pos := 0; pos < p; pos++ {
			kd := be.Upload(NewF32(be, []int{width}, k[pos*width:(pos+1)*width]), F32)
			vd := be.Upload(NewF32(be, []int{width}, v[pos*width:(pos+1)*width]), F32)
			kv.AppendKV(0, kd, kd, vd, pos)
			be.Free(kd)
			be.Free(vd)
		}
		return kv
	}

	// runPanel executes one panel call under the selected kernel mode and returns the
	// device output plus how many mma launches it caused.
	runPanel := func(t *testing.T, mode string, p, grp int, q []float32, kv KVStore) ([]float32, uint64) {
		t.Helper()
		t.Setenv(cudaQwen38PrefillAttnModeEnv, mode)
		qd := be.Upload(NewF32(be, []int{p, grp * kvHeads * headDim}, q), F32)
		t.Cleanup(func() { be.Free(qd) })
		before := be.qwen38PrefillAttentionMMALaunchesForTest()
		y := be.PromptPanelAttention(qd, kv, 0, p, 0, grp, scale)
		got := be.Read(y) // native checked completion, not enqueue proof
		be.Free(y)
		for i, value := range got {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatalf("%s P=%d grp=%d output[%d] is non-finite: %v", mode, p, grp, i, value)
			}
		}
		return got, be.qwen38PrefillAttentionMMALaunchesForTest() - before
	}

	// greedyTokens maps each output row to the argmax index over its full width, the
	// closed-contract stand-in for the next token a greedy decoder would pick.
	greedyTokens := func(out []float32, p, width int) []int {
		toks := make([]int, p)
		for row := 0; row < p; row++ {
			best := 0
			base := row * width
			for d := 1; d < width; d++ {
				if out[base+d] > out[base+best] {
					best = d
				}
			}
			toks[row] = best
		}
		return toks
	}

	checkTokens := func(mode string, p int, toks []int) {
		for i, tok := range toks {
			if tok != wantToken {
				t.Fatalf("%s P=%d row %d greedy token %d != closed answer %d", mode, p, i, tok, wantToken)
			}
		}
	}

	for _, tc := range []struct {
		p    int
		seed uint64
	}{
		{128, 0x5eed128},
		{512, 0x5eed512},
		{2048, 0x5eed2048},
	} {
		t.Run(fmt.Sprintf("P%d", tc.p), func(t *testing.T) {
			q, k, v := fixture(tc.p, tc.seed)
			kv := fillKV(t, tc.p, k, v)
			defer kv.Free()

			simt, simtLaunches := runPanel(t, "simt", tc.p, groups, q, kv)
			if simtLaunches != 0 {
				t.Fatalf("forced simt launched the mma kernel %d times at P=%d", simtLaunches, tc.p)
			}
			checkTokens("simt", tc.p, greedyTokens(simt, tc.p, heads*headDim))

			mma, mmaLaunches := runPanel(t, "mma", tc.p, groups, q, kv)
			if mmaLaunches != 1 {
				t.Fatalf("mma selection launched the mma kernel %d times at P=%d: the SIMT fallback produced the mma output (device below sm_80?)", mmaLaunches, tc.p)
			}
			checkTokens("mma", tc.p, greedyTokens(mma, tc.p, heads*headDim))

			similarity := cosine(simt, mma)
			if math.IsNaN(similarity) || similarity < cudaQwen38PrefillAttnMMACosineMin {
				t.Fatalf("mma-vs-simt cosine %.8f < %.4f at P=%d (max abs delta %.3e)", similarity, cudaQwen38PrefillAttnMMACosineMin, tc.p, maxAbsDelta(simt, mma))
			}
			t.Logf("P=%d closed token %d on all %d rows in both modes, mma-vs-simt cosine=%.8f max_abs_delta=%.3e", tc.p, wantToken, tc.p, similarity, maxAbsDelta(simt, mma))
		})
	}

	// Below the P>=128 gate the panel path must stay on the SIMT kernel even when mma is
	// selected: zero mma launches and a bitwise-equal output to the forced-simt run.
	t.Run("P64-below-gate-stays-simt", func(t *testing.T) {
		const p = 64
		q, k, v := fixture(p, 0x5eed064)
		kv := fillKV(t, p, k, v)
		defer kv.Free()
		simt, simtLaunches := runPanel(t, "simt", p, groups, q, kv)
		if simtLaunches != 0 {
			t.Fatalf("forced simt launched the mma kernel %d times at P=%d", simtLaunches, p)
		}
		mmaSelected, mmaLaunches := runPanel(t, "mma", p, groups, q, kv)
		if mmaLaunches != 0 {
			t.Fatalf("P=%d is below the P>=128 gate but took the mma kernel %d times", p, mmaLaunches)
		}
		for i := range simt {
			if simt[i] != mmaSelected[i] {
				t.Fatalf("P=%d below-gate output diverged from the SIMT kernel at element %d: %v vs %v", p, i, simt[i], mmaSelected[i])
			}
		}
		t.Logf("P=%d below-gate: mma launches=0, output bitwise equal to forced simt", p)
	})

	// Outside the 24x4xhd256 instantiation the mma port must decline: 32 query heads keep
	// the SIMT kernel even when mma is selected, bitwise equal to the forced-simt run.
	t.Run("N32-outside-instantiation-stays-simt", func(t *testing.T) {
		const (
			p   = 128
			grp = 8 // 8 groups x 4 KV heads = 32 query heads, not the instantiated 24
		)
		q, k, v := fixture(p, 0x5eed32a)
		kv := fillKV(t, p, k, v)
		defer kv.Free()
		simt, simtLaunches := runPanel(t, "simt", p, grp, q, kv)
		if simtLaunches != 0 {
			t.Fatalf("forced simt launched the mma kernel %d times at N=%d", simtLaunches, grp*kvHeads)
		}
		mmaSelected, mmaLaunches := runPanel(t, "mma", p, grp, q, kv)
		if mmaLaunches != 0 {
			t.Fatalf("N=%d is outside the 24x4xhd256 instantiation but took the mma kernel %d times", grp*kvHeads, mmaLaunches)
		}
		for i := range simt {
			if simt[i] != mmaSelected[i] {
				t.Fatalf("N=%d outside-instantiation output diverged from the SIMT kernel at element %d: %v vs %v", grp*kvHeads, i, simt[i], mmaSelected[i])
			}
		}
		t.Logf("N=%d outside-instantiation: mma launches=0, output bitwise equal to forced simt", grp*kvHeads)
	})
}
