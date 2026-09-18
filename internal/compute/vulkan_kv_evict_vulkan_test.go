//go:build vulkan && (windows || linux) && cgo

package compute

import (
	"math"
	"testing"
)

// vulkan_kv_evict_vulkan_test.go - the `-tags vulkan` device twin of
// TestCUDAEvictMiddleSpanEqualsNeverSaw (#479), aimed at the device-resident
// vulkanKV.Evict delivered for #12401.
//
// It proves, ON the device, that eviction no longer crosses the host boundary:
//
//  1. a MIDDLE-span device evict == a device cache that NEVER saw the span
//     (K and V, max|delta| under the Approx gate), so the compaction + single
//     rotation re-RoPE is exactly the never-saw result;
//  2. the suffix really moved - a survivor's post-RoPE K at its new index differs
//     from its pre-evict K (repositioning ran; an end-span trim would not move it);
//  3. the prefix is untouched - a prefix survivor's K is byte-identical, which is
//     the write-time quarantine asymmetry (MODEL-ARCH-SEAM section 3, O1-O3);
//  4. zero D2H and zero H2D bytes are observed across the eviction window - the
//     whole point of the leaf. The pre-change path read K/Kraw/V to the host and
//     rewrote every layer, so this counter is the non-self-referential witness:
//     it is measured by the backend's own live transfer counters, not asserted.
//
// Skips cleanly when no Vulkan device is registered (vk(t)). The host has no
// reachable Vulkan device; the test's actual RUN is the residual handed to a
// Halo node via `go -C <worktree> test -tags vulkan ./internal/compute -run
// TestVulkanKVDeviceResidentEviction`.
func TestVulkanKVDeviceResidentEviction(t *testing.T) {
	v := vk(t)
	c := cpu()
	cfg := KVConfig{NumLayers: 2, NumKVHeads: 2, HeadDim: 4, RopeTheta: 10000}
	w := cfg.NumKVHeads * cfg.HeadDim

	// Deterministic pre-RoPE K (Kraw) and V per (layer,pos), identical across A and
	// B so the only difference that can appear is from the eviction math itself.
	rawK := func(l, p int) []float32 { s := lcg(1000*l + p + 1); return randVec(&s, w) }
	rawV := func(l, p int) []float32 { s := lcg(7000*l + p + 1); return randVec(&s, w) }

	appendPos := func(kv KVStore, l, srcPos, atPos int) {
		kr := rawK(l, srcPos)
		vv := rawV(l, srcPos)
		kRaw := v.Upload(NewF32(c, []int{w}, kr), F32)
		kRoPE := v.RoPE(kRaw, atPos, cfg.NumKVHeads, cfg.HeadDim, cfg.RopeTheta)
		val := v.Upload(NewF32(c, []int{w}, vv), F32)
		kv.AppendKV(l, kRaw, kRoPE, val, atPos)
		v.Free(kRaw)
		v.Free(kRoPE)
		v.Free(val)
	}

	// A: append absolute positions 0..6, then evict the MIDDLE span [from,from+n).
	const total, from, n = 7, 2, 2 // remove indices {2,3}; survivors {0,1,4,5,6}
	A := v.NewKV(cfg)
	for p := 0; p < total; p++ {
		for l := 0; l < cfg.NumLayers; l++ {
			appendPos(A, l, p, p)
		}
	}

	// Snapshot layer-0 K before eviction: one prefix row (index 0, stays put) and one
	// suffix row (last position, a survivor after the span that must be repositioned).
	preK0 := v.Read(A.KeysView(0))
	suffixOrig := total - 1 // original position 6, a survivor after the span
	preSuffix := append([]float32(nil), preK0[suffixOrig*w:(suffixOrig+1)*w]...)
	prefixIdx := 0 // original position 0, stays at index 0
	prePrefix := append([]float32(nil), preK0[prefixIdx*w:(prefixIdx+1)*w]...)

	// (4): bracket ONLY the eviction with the backend's own live transfer counters,
	// so the zero-D2H assertion is measured rather than self-reported.
	window, err := v.BeginBackendExecutionWindow()
	if err != nil {
		A.Free()
		t.Fatalf("begin eviction observation: %v", err)
	}
	if removed := A.Evict(from, n); removed != n || A.Len() != total-n {
		window.End()
		A.Free()
		t.Fatalf("evict removed %d (want %d), len %d (want %d)", removed, n, A.Len(), total-n)
	}
	observation, err := window.End()
	if err != nil {
		A.Free()
		t.Fatalf("end eviction observation: %v", err)
	}
	got := observation.Counters
	t.Logf("device-resident evict transfers: d2h_bytes=%d d2h_count=%d h2d_bytes=%d h2d_count=%d d2d_copies=%d d2d_bytes=%d",
		got.D2HBytes, got.D2HCount, got.H2DBytes, got.H2DCount, got.D2DCopies, got.D2DBytes)
	if got.D2HBytes != 0 || got.D2HCount != 0 {
		t.Errorf("eviction crossed the host boundary on read: d2h_bytes=%d d2h_count=%d (want 0/0)", got.D2HBytes, got.D2HCount)
	}
	if got.H2DBytes != 0 || got.H2DCount != 0 {
		t.Errorf("eviction crossed the host boundary on write: h2d_bytes=%d h2d_count=%d (want 0/0)", got.H2DBytes, got.H2DCount)
	}
	if got.D2DCopies == 0 {
		t.Errorf("eviction performed no device-to-device copies; the compaction did not run on the device")
	}

	// (2)+(3): the suffix survivor moved (orig pos 6 -> new index total-1-n=4), so its K
	// must change; the prefix survivor (index 0) must be byte-identical.
	postK0 := v.Read(A.KeysView(0))
	suffixNew := total - 1 - n // 6 -> 4
	postSuffix := postK0[suffixNew*w : (suffixNew+1)*w]
	if evictEqualF32(preSuffix, postSuffix) {
		t.Errorf("suffix survivor K unchanged after middle-span evict - repositioning did not run (end-span behavior)")
	} else {
		t.Logf("suffix survivor repositioned (re-RoPE at new index ran) - this is the MIDDLE-span case, not an end trim")
	}
	postPrefix := postK0[prefixIdx*w : (prefixIdx+1)*w]
	if !evictEqualF32(prePrefix, postPrefix) {
		t.Errorf("prefix survivor K changed after evict - the quarantine asymmetry is broken (prefix must stay byte-for-byte)")
	}

	// B: the never-saw run - survivors appended at their NEW indices.
	survivors := []int{0, 1, 4, 5, 6}
	B := v.NewKV(cfg)
	for ni, op := range survivors {
		for l := 0; l < cfg.NumLayers; l++ {
			appendPos(B, l, op, ni)
		}
	}
	if B.Len() != len(survivors) {
		t.Fatalf("never-saw cache len %d != %d", B.Len(), len(survivors))
	}

	// (1): device evict == device never-saw, K and V, under the Approx gate.
	const tol = 1e-4 // vulkan backend is Approx, not bit-identity; same kernel/pos/Kraw => ~0
	var maxd float64
	for l := 0; l < cfg.NumLayers; l++ {
		ak, bk := v.Read(A.KeysView(l)), v.Read(B.KeysView(l))
		av, bv := v.Read(A.ValuesView(l)), v.Read(B.ValuesView(l))
		if len(ak) != len(bk) || len(av) != len(bv) {
			t.Fatalf("layer %d length mismatch: K %d/%d V %d/%d", l, len(ak), len(bk), len(av), len(bv))
		}
		maxd = math.Max(maxd, evictMaxAbs(ak, bk))
		maxd = math.Max(maxd, evictMaxAbs(av, bv))
	}
	t.Logf("Vulkan middle-span evict==never-saw: max|delta|=%.3e (Approx gate tol=%.0e, device=%s tier=%s class=%s)",
		maxd, tol, v.Name(), v.Tier(), v.Class())
	if maxd > tol {
		t.Errorf("device middle-span evict != device never-saw: max|delta|=%.3e > tol=%.0e", maxd, tol)
	}

	// A resident KV tensor is NOT host-addressable - the cache never leaves VRAM.
	if _, ok := v.Host(A.KeysView(0)); ok {
		t.Errorf("Host() returned addressable data for a resident KV tensor - must stay (nil,false)")
	}

	A.Free()
	B.Free()
}

// evictEqualF32 reports whether two equal-length float rows hold identical bits - the
// byte-for-byte gate for the untouched-prefix assertion.
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

// evictMaxAbs is the max|delta| the Approx gate compares against tolerance.
func evictMaxAbs(a, b []float32) float64 {
	var m float64
	for i := range a {
		if d := math.Abs(float64(a[i] - b[i])); d > m {
			m = d
		}
	}
	return m
}
