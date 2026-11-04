//go:build cuda && cgo

package compute

import (
	"math"
	"testing"
)

// TestFattnMmaHd256NonuniformPrefix complements the uniform-score fixture:
// nonzero prefix, a partial query/KV tile, and nonuniform scores must all survive
// the actual MMA route. Attention coordinates are not model-token evidence.
// fak-test:runtime slow est=10s
func TestFattnMmaHd256NonuniformPrefix(t *testing.T) {
	be := cudaGDNBackend(t)
	t.Cleanup(be.Recycle)
	t.Setenv("FAK_CUDA_PAGED_KV", "")
	const tokens, prefix, heads, kvHeads, hd = 129, 5, 24, 4, 256
	const positions = prefix + tokens
	const width = heads * hd
	const kvWidth = kvHeads * hd
	const scale float32 = 1.0 / 16
	q, k, v := fattnNonuniformFixture(tokens, prefix, heads, kvHeads, hd)
	want := fattnNonuniformOracle(q, k, v, tokens, prefix, heads, kvHeads, hd, scale)
	run := func(t *testing.T, mode string, keys, values []float32) []float32 {
		t.Helper()
		t.Setenv(cudaQwen38PrefillAttnModeEnv, mode)
		kv := be.NewKV(KVConfig{NumLayers: 1, NumKVHeads: kvHeads, HeadDim: hd, Precision: KVPrecisionF32})
		defer kv.Free()
		for pos := 0; pos < positions; pos++ {
			kd := be.Upload(NewF32(be, []int{kvWidth}, keys[pos*kvWidth:(pos+1)*kvWidth]), F32)
			vd := be.Upload(NewF32(be, []int{kvWidth}, values[pos*kvWidth:(pos+1)*kvWidth]), F32)
			kv.AppendKV(0, kd, kd, vd, pos)
			be.Free(kd)
			be.Free(vd)
		}
		qd := be.Upload(NewF32(be, []int{tokens, width}, q), F32)
		defer be.Free(qd)
		before := be.qwen38PrefillAttentionMMALaunchesForTest()
		y := be.PromptPanelAttention(qd, kv, 0, tokens, prefix, heads/kvHeads, scale)
		defer be.Free(y)
		got := be.Read(y)
		launches := be.qwen38PrefillAttentionMMALaunchesForTest() - before
		var expected uint64
		if mode == "mma" {
			expected = 1
		}
		if launches != expected {
			t.Fatalf("%s launches=%d, want %d; fallback is not qualification", mode, launches, expected)
		}
		if len(got) != len(want) {
			t.Fatalf("%s output length=%d, want %d", mode, len(got), len(want))
		}
		for i, x := range got {
			if math.IsNaN(float64(x)) || math.IsInf(float64(x), 0) {
				t.Fatalf("%s output[%d] is not finite", mode, i)
			}
		}
		return got
	}
	for _, mode := range []string{"simt", "mma"} {
		t.Run(mode, func(t *testing.T) {
			got := run(t, mode, k, v)
			// Apply the existing recorded MMA cosine floor independently per row;
			// a whole-panel cosine can hide a broken causal edge or final tile.
			for row := 0; row < tokens; row++ {
				similarity := cosine(got[row*width:(row+1)*width], want[row*width:(row+1)*width])
				if math.IsNaN(similarity) || similarity < cudaQwen38PrefillAttnMMACosineMin {
					t.Fatalf("%s row=%d CPU cosine=%g", mode, row, similarity)
				}
			}
			// Last position is invisible to every query except the final one.
			// Change both score and value so a missing causal mask is observable.
			changedK, changedV := append([]float32(nil), k...), append([]float32(nil), v...)
			for h := 0; h < kvHeads; h++ {
				changedK[(positions-1)*kvWidth+h*hd] = 32
				for d := 0; d < hd; d++ {
					changedV[(positions-1)*kvWidth+h*hd+d] = 8
				}
			}
			changed := run(t, mode, changedK, changedV)
			for i := 0; i < (tokens-1)*width; i++ {
				if got[i] != changed[i] {
					t.Fatalf("%s future KV changed masked output[%d]: %g -> %g", mode, i, got[i], changed[i])
				}
			}
			if maxAbsDelta(got[(tokens-1)*width:], changed[(tokens-1)*width:]) < 1 {
				t.Fatalf("%s visible final KV mutation had no substantial effect", mode)
			}
		})
	}
}

// Exactly representable Q/K inputs give varying scores without staging noise.
func fattnNonuniformFixture(tokens, prefix, heads, kvHeads, hd int) (q, k, v []float32) {
	q = make([]float32, tokens*heads*hd)
	k = make([]float32, (prefix+tokens)*kvHeads*hd)
	v = make([]float32, len(k))
	for row := 0; row < tokens; row++ {
		for h := 0; h < heads; h++ {
			q[(row*heads+h)*hd] = float32(8 + (row+h)%5)
		}
	}
	for pos := 0; pos < prefix+tokens; pos++ {
		for h := 0; h < kvHeads; h++ {
			base := (pos*kvHeads + h) * hd
			k[base] = float32((pos*7+h*3)%17-8) / 2
			for d := 0; d < hd; d++ {
				v[base+d] = float32((pos*11+h*5+d*3)%29-14) / 16
			}
		}
	}
	return
}

// Independent stable float64 CPU softmax for this one-dimensional Q/K fixture.
func fattnNonuniformOracle(q, k, v []float32, tokens, prefix, heads, kvHeads, hd int, scale float32) []float32 {
	out := make([]float32, tokens*heads*hd)
	for row := 0; row < tokens; row++ {
		for h := 0; h < heads; h++ {
			kh := h / (heads / kvHeads)
			scores := make([]float64, prefix+row+1)
			maximum := math.Inf(-1)
			for pos := range scores {
				scores[pos] = float64(q[(row*heads+h)*hd]) * float64(k[(pos*kvHeads+kh)*hd]) * float64(scale)
				maximum = math.Max(maximum, scores[pos])
			}
			denom := 0.0
			for pos := range scores {
				scores[pos] = math.Exp(scores[pos] - maximum)
				denom += scores[pos]
			}
			for d := 0; d < hd; d++ {
				value := 0.0
				for pos, probability := range scores {
					value += probability * float64(v[(pos*kvHeads+kh)*hd+d])
				}
				out[(row*heads+h)*hd+d] = float32(value / denom)
			}
		}
	}
	return out
}
