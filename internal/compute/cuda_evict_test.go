//go:build cuda

package compute

import (
	"math"
	"testing"
)

// cuda_evict_test.go — the `-tags cuda` port of the evict==never-saw witness (#479), run
// ON the GPU against the on-GPU cudaKV.Evict (compact + single-rotation re-RoPE from device
// Kraw, no host round-trip). It reproduces the MIDDLE-span case — a span with BOTH a prefix
// before it AND survivors after it — not just an end trim, because the whole point is the
// asymmetry: the prefix stays byte-for-byte while every suffix survivor is repositioned by a
// single rotation at its new index. The gate is Approx (max|Δ| ≤ tol), matching the cuda
// backend's class (Class()==Approx); device evict vs a device run that never saw the span
// both ride the SAME RoPE kernel at the SAME positions on the SAME Kraw, so they coincide.
//
// Skips cleanly when no CUDA device is registered (cudaOrSkip). The host has no CUDA toolkit;
// this test's actual RUN is the residual handed to a CUDA node via
// tools/run_479_acceptance_on_gpu.sh.

// TestCUDAEvictMiddleSpanEqualsNeverSaw is the device twin of TestEvictEqualsNeverSaw, but
// over a MIDDLE span so survivors must be repositioned. It proves, under the Approx gate:
//
//  1. device middle-span evict == a device cache that NEVER saw the span (K and V, max|Δ|);
//  2. the suffix really moved — a survivor's post-RoPE K at its new index differs from its
//     pre-evict K (re-RoPE happened; an end-span evict would have no suffix to move);
//  3. the prefix is untouched — a prefix survivor's K is byte-identical before and after;
//  4. Host() on a resident KV tensor stays (nil,false) — the cache never leaves VRAM.
func TestCUDAEvictMiddleSpanEqualsNeverSaw(t *testing.T) {
	cb := cudaOrSkip(t)
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 4, RopeTheta: 10000}
	w := cfg.NumKVHeads * cfg.HeadDim

	// deterministic pre-RoPE K (Kraw) and V per (layer,pos), identical across A and B so the
	// only difference that can appear is from the eviction math itself.
	rawK := func(l, p int) []float32 { s := lcg(1000*l + p + 1); return randVec(&s, w) }
	rawV := func(l, p int) []float32 { s := lcg(7000*l + p + 1); return randVec(&s, w) }

	// keep retains every host slice so the per-pointer uploadCache never sees a recycled
	// address return a stale device tensor.
	var keep [][]float32
	appendPos := func(kv KVStore, l, srcPos, atPos int) {
		kr := rawK(l, srcPos)
		vv := rawV(l, srcPos)
		keep = append(keep, kr, vv)
		kRaw := mkResident(cb, []int{w}, kr)                                                              // pre-RoPE -> Kraw
		kRoPE := cb.RoPE(mkResident(cb, []int{w}, kr), atPos, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeTheta) // -> K
		v := mkResident(cb, []int{w}, vv)
		kv.AppendKV(l, kRaw, kRoPE, v, atPos)
	}

	// A: append absolute positions 0..6, then evict the MIDDLE span [from,from+n).
	const total, from, n = 7, 2, 2 // remove indices {2,3}; survivors {0,1,4,5,6}
	A := cb.NewKV(cfg)
	for p := 0; p < total; p++ {
		for l := 0; l < cfg.NumLayers; l++ {
			appendPos(A, l, p, p)
		}
	}

	// snapshot layer-0 K before eviction: one prefix row (index 0, stays put) and one suffix
	// row (last position, a survivor after the span that must be repositioned).
	preK0 := cb.Read(A.KeysView(0))
	suffixOrig := total - 1 // original position 6, a survivor after the span
	preSuffix := append([]float32(nil), preK0[suffixOrig*w:(suffixOrig+1)*w]...)
	prefixIdx := 0 // original position 0, stays at index 0
	prePrefix := append([]float32(nil), preK0[prefixIdx*w:(prefixIdx+1)*w]...)

	if removed := A.Evict(from, n); removed != n || A.Len() != total-n {
		t.Fatalf("evict removed %d (want %d), len %d (want %d)", removed, n, A.Len(), total-n)
	}

	// (2)+(3): the suffix survivor moved (orig pos 6 -> new index total-1-n=4), so its K must
	// change; the prefix survivor (index 0) must be byte-identical.
	postK0 := cb.Read(A.KeysView(0))
	suffixNew := total - 1 - n // 6 -> 4
	postSuffix := postK0[suffixNew*w : (suffixNew+1)*w]
	if evictEqualF32(preSuffix, postSuffix) {
		t.Errorf("suffix survivor K unchanged after middle-span evict — repositioning did not run (end-span behavior)")
	} else {
		t.Logf("✓ suffix survivor repositioned (re-RoPE at new index ran) — this is the MIDDLE-span case, not an end trim")
	}
	postPrefix := postK0[prefixIdx*w : (prefixIdx+1)*w]
	if !evictEqualF32(prePrefix, postPrefix) {
		t.Errorf("prefix survivor K changed after evict — the quarantine asymmetry is broken (prefix must stay byte-for-byte)")
	}

	// B: the never-saw run — survivors appended at their NEW indices.
	survivors := []int{0, 1, 4, 5, 6}
	B := cb.NewKV(cfg)
	for ni, op := range survivors {
		for l := 0; l < cfg.NumLayers; l++ {
			appendPos(B, l, op, ni)
		}
	}
	if B.Len() != len(survivors) {
		t.Fatalf("never-saw cache len %d != %d", B.Len(), len(survivors))
	}

	// (1): device evict == device never-saw, K and V, under the Approx gate.
	const tol = 1e-4 // cuda backend is Approx, not bit-identity; same kernel/pos/Kraw => ~0
	var maxd float64
	for l := 0; l < cfg.NumLayers; l++ {
		ak, bk := cb.Read(A.KeysView(l)), cb.Read(B.KeysView(l))
		av, bv := cb.Read(A.ValuesView(l)), cb.Read(B.ValuesView(l))
		if len(ak) != len(bk) || len(av) != len(bv) {
			t.Fatalf("layer %d length mismatch: K %d/%d V %d/%d", l, len(ak), len(bk), len(av), len(bv))
		}
		maxd = math.Max(maxd, evictMaxAbs(ak, bk))
		maxd = math.Max(maxd, evictMaxAbs(av, bv))
	}
	t.Logf("CUDA middle-span evict==never-saw: max|Δ|=%.3e (Approx gate tol=%.0e, device=%s tier=%s class=%s)",
		maxd, tol, cb.Name(), cb.Tier(), cb.Class())
	if maxd > tol {
		t.Errorf("device middle-span evict != device never-saw: max|Δ|=%.3e > tol=%.0e", maxd, tol)
	}

	// (4): a resident KV tensor is NOT host-addressable.
	if _, ok := cb.Host(A.KeysView(0)); ok {
		t.Errorf("Host() returned addressable data for a resident KV tensor — must stay (nil,false)")
	}
}

func evictEqualF32(a, b []float32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float32bits(a[i]) != math.Float32bits(b[i]) {
			return false
		}
	}
	return true
}

func evictMaxAbs(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}
