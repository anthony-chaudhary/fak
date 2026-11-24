//go:build cuda && cgo

package compute

import (
	"fmt"
	"math"
	"testing"
)

const cudaAttnGenericEnv = "FAK_CUDA_ATTN_GENERIC"

type attnGenericGeom struct{ nH, nKV, hd int }

// attnGenericKV builds a single-layer contiguous F32 cache of n positions in one
// panel append and returns it with the host rows it holds.
func attnGenericKV(t *testing.T, cb *cudaBackend, g *lcg, geom attnGenericGeom, n int) KVStore {
	t.Helper()
	w := geom.nKV * geom.hd
	kv := cb.NewKV(KVConfig{NumLayers: 1, NumKVHeads: geom.nKV, HeadDim: geom.hd, RopeTheta: 10000, Precision: KVPrecisionF32})
	k := mkResident(cb, []int{n, w}, rscale(g, n*w, 1.0))
	v := mkResident(cb, []int{n, w}, rscale(g, n*w, 1.0))
	cb.PromptPanelAppendKV(kv, 0, k, k, v, 0, n)
	cb.Free(k)
	cb.Free(v)
	return kv
}

// attnGenericCompare holds got to want within 1e-4 of the output scale (max |want|).
func attnGenericCompare(t *testing.T, label string, want, got []float32) {
	t.Helper()
	if len(want) != len(got) {
		t.Fatalf("%s: len legacy=%d generic=%d", label, len(want), len(got))
	}
	scale := 0.0
	for i, v := range got {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("%s: generic output[%d] non-finite %v", label, i, v)
		}
		scale = math.Max(scale, math.Abs(float64(want[i])))
	}
	tol := 1e-4 * math.Max(scale, 1e-3)
	if d := maxAbsDelta(want, got); d > tol {
		t.Fatalf("%s: generic-vs-legacy max|delta| %.3g > %.3g (output scale %.3g)", label, d, tol, scale)
	}
}

var attnGenericGeoms = []attnGenericGeom{
	{28, 4, 64}, {32, 8, 64}, {8, 8, 64},
	{28, 4, 128}, {32, 8, 128}, {8, 8, 128},
}

// TestCUDAAttentionGenericDecodeMatchesLegacy holds the generic tiled decode kernel
// (including its split-KV merge on long windows) to k_flash_attention, which
// FAK_CUDA_ATTN_GENERIC=legacy restores, and witnesses which kernel actually ran.
//
// fak-test:runtime slow est=30s
func TestCUDAAttentionGenericDecodeMatchesLegacy(t *testing.T) {
	cb := cudaOrSkip(t)
	t.Cleanup(cb.Recycle)
	t.Setenv("FAK_CUDA_PAGED_KV", "")
	geoms := append(append([]attnGenericGeom(nil), attnGenericGeoms...), attnGenericGeom{16, 2, 80}, attnGenericGeom{24, 4, 256})
	for _, geom := range geoms {
		for _, nPos := range []int{1, 7, 255, 256, 1000, 4097} {
			label := fmt.Sprintf("decode nH=%d nKV=%d hd=%d nPos=%d", geom.nH, geom.nKV, geom.hd, nPos)
			var seed lcg = lcg(7001 + nPos + geom.hd)
			kv := attnGenericKV(t, cb, &seed, geom, nPos)
			q := mkResident(cb, []int{geom.nH * geom.hd}, rscale(&seed, geom.nH*geom.hd, 3.0))
			grp := geom.nH / geom.nKV
			scale := float32(1 / math.Sqrt(float64(geom.hd)))

			t.Setenv(cudaAttnGenericEnv, "legacy")
			before := cb.attnGenericLaunchesForTest(0)
			legacy := cb.Read(cb.Attention(q, kv, 0, true, grp, scale))
			if n := cb.attnGenericLaunchesForTest(0) - before; n != 0 {
				t.Fatalf("%s: legacy mode launched the generic kernel %d times", label, n)
			}
			t.Setenv(cudaAttnGenericEnv, "")
			before = cb.attnGenericLaunchesForTest(0)
			generic := cb.Read(cb.Attention(q, kv, 0, true, grp, scale))
			if n := cb.attnGenericLaunchesForTest(0) - before; n != 1 {
				t.Fatalf("%s: default mode generic launches = %d, want 1", label, n)
			}
			attnGenericCompare(t, label, legacy, generic)
			cb.Free(q)
			kv.Free()
			cb.Recycle()
		}
	}
}

// TestCUDAAttentionGenericPanelMatchesLegacy holds the tiled prompt-panel kernel to
// k_prompt_causal_attention_panel over panels with and without a cached prefix,
// including partial row blocks, multi-tile windows and the split-KV path a short
// panel over a long prefix takes.
//
// fak-test:runtime slow est=30s
func TestCUDAAttentionGenericPanelMatchesLegacy(t *testing.T) {
	cb := cudaOrSkip(t)
	t.Cleanup(cb.Recycle)
	t.Setenv("FAK_CUDA_PAGED_KV", "")
	panels := []struct{ prefix, tokens int }{
		{0, 1}, {0, 7}, {3, 4}, {0, 255}, {128, 128}, {0, 300}, {900, 100}, {4000, 97}, {4096, 1},
	}
	geoms := append(append([]attnGenericGeom(nil), attnGenericGeoms...), attnGenericGeom{16, 2, 80}, attnGenericGeom{24, 4, 256})
	for _, geom := range geoms {
		for _, p := range panels {
			label := fmt.Sprintf("panel nH=%d nKV=%d hd=%d prefix=%d tokens=%d", geom.nH, geom.nKV, geom.hd, p.prefix, p.tokens)
			var seed lcg = lcg(9001 + p.prefix + 3*p.tokens + geom.hd)
			kv := attnGenericKV(t, cb, &seed, geom, p.prefix+p.tokens)
			width := geom.nH * geom.hd
			q := mkResident(cb, []int{p.tokens, width}, rscale(&seed, p.tokens*width, 3.0))
			grp := geom.nH / geom.nKV
			scale := float32(1 / math.Sqrt(float64(geom.hd)))
			if geom.nH == 24 && geom.nKV == 4 && geom.hd == 256 {
				t.Setenv(cudaQwen38PrefillAttnModeEnv, "simt") // keep the hd256 mma port out of this comparison
			}

			t.Setenv(cudaAttnGenericEnv, "legacy")
			before := cb.attnGenericLaunchesForTest(1)
			y := cb.PromptPanelAttention(q, kv, 0, p.tokens, p.prefix, grp, scale)
			legacy := cb.Read(y)
			cb.Free(y)
			if n := cb.attnGenericLaunchesForTest(1) - before; n != 0 {
				t.Fatalf("%s: legacy mode launched the generic kernel %d times", label, n)
			}
			t.Setenv(cudaAttnGenericEnv, "")
			before = cb.attnGenericLaunchesForTest(1)
			y = cb.PromptPanelAttention(q, kv, 0, p.tokens, p.prefix, grp, scale)
			generic := cb.Read(y)
			cb.Free(y)
			if n := cb.attnGenericLaunchesForTest(1) - before; n != 1 {
				t.Fatalf("%s: default mode generic launches = %d, want 1", label, n)
			}
			attnGenericCompare(t, label, legacy, generic)
			cb.Free(q)
			kv.Free()
			cb.Recycle()
		}
	}
}

// TestCUDAAttentionGenericPanelRowIsDecode checks the causal offset contract and the
// bitwise row identity the dense HAL prompt panel relies on: panel row t over
// prefix+t+1 keys equals, bit for bit, a decode step over exactly those keys, also
// when the decode spreads its chunks over split blocks and the panel folds them.
//
// fak-test:runtime medium est=10s
func TestCUDAAttentionGenericPanelRowIsDecode(t *testing.T) {
	cb := cudaOrSkip(t)
	t.Cleanup(cb.Recycle)
	t.Setenv("FAK_CUDA_PAGED_KV", "")
	t.Setenv(cudaAttnGenericEnv, "")
	for _, tc := range []struct {
		geom           attnGenericGeom
		prefix, tokens int
	}{
		{attnGenericGeom{28, 4, 128}, 61, 9}, // rows cross the first 64-key chunk boundary
		{attnGenericGeom{32, 8, 64}, 1000, 3},
		{attnGenericGeom{4, 2, 32}, 120, 12},
	} {
		geom, prefix, tokens := tc.geom, tc.prefix, tc.tokens
		var seed lcg = lcg(4242 + prefix)
		w := geom.nKV * geom.hd
		width := geom.nH * geom.hd
		kData := rscale(&seed, (prefix+tokens)*w, 1.0)
		vData := rscale(&seed, (prefix+tokens)*w, 1.0)
		qData := rscale(&seed, tokens*width, 3.0)
		grp := geom.nH / geom.nKV
		scale := float32(1 / math.Sqrt(float64(geom.hd)))
		build := func(n int) KVStore {
			kv := cb.NewKV(KVConfig{NumLayers: 1, NumKVHeads: geom.nKV, HeadDim: geom.hd, RopeTheta: 10000, Precision: KVPrecisionF32})
			k := mkResident(cb, []int{n, w}, kData[:n*w])
			v := mkResident(cb, []int{n, w}, vData[:n*w])
			cb.PromptPanelAppendKV(kv, 0, k, k, v, 0, n)
			cb.Free(k)
			cb.Free(v)
			return kv
		}
		full := build(prefix + tokens)
		q := mkResident(cb, []int{tokens, width}, qData)
		y := cb.PromptPanelAttention(q, full, 0, tokens, prefix, grp, scale)
		panel := cb.Read(y)
		cb.Free(y)
		cb.Free(q)
		full.Free()
		for row := 0; row < tokens; row++ {
			kv := build(prefix + row + 1)
			qr := mkResident(cb, []int{width}, qData[row*width:(row+1)*width])
			want := cb.Read(cb.Attention(qr, kv, 0, true, grp, scale))
			got := panel[row*width : (row+1)*width]
			for i := range want {
				if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
					t.Fatalf("nH=%d nKV=%d hd=%d prefix=%d row %d elem %d: panel %08x != decode %08x",
						geom.nH, geom.nKV, geom.hd, prefix, row, i, math.Float32bits(got[i]), math.Float32bits(want[i]))
				}
			}
			cb.Free(qr)
			kv.Free()
			cb.Recycle()
		}
	}
}
