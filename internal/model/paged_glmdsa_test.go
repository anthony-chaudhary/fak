package model

import (
	"math"
	"testing"
)

// TestPagedGLMDsaEvictBitIdenticalToContiguous covers the GLM-DSA acceptance
// edge of #33: the separate DSA attention/index cache can live in paged row blocks
// and still perform the same single-rotation-from-Kraw middle-span Evict as the
// contiguous glmDsaKVCache.
func TestPagedGLMDsaEvictBitIdenticalToContiguous(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensors(t, false)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	all := []int{3, 17, 5, 23, 11, 7}
	const from, n = 2, 2

	contig := m.NewSession()
	contig.PrefillNoLogits(all)
	paged, err := GLMDsaKVCacheToPaged(contig.Cache, 3)
	if err != nil {
		t.Fatalf("GLMDsaKVCacheToPaged: %v", err)
	}
	assertGLMDsaCachesEqual(t, "pre-evict", contig.Cache, paged.ToKVCache())

	rc := contig.Cache.Evict(from, n)
	rp := paged.Evict(from, n)
	if rc != rp || rc != n {
		t.Fatalf("removed mismatch: contiguous=%d paged=%d want=%d", rc, rp, n)
	}
	if got, want := paged.Len(), contig.Cache.Len(); got != want {
		t.Fatalf("paged Len=%d, want contiguous Len=%d", got, want)
	}

	materialized := paged.ToKVCache()
	assertGLMDsaCachesEqual(t, "post-evict", contig.Cache, materialized)
	assertGLMDsaCacheReroped(t, materialized)

	contigStep := &Session{M: m, Cache: contig.Cache}
	pagedStep := &Session{M: m, Cache: materialized}
	assertFloat32BitsEqual(t, "paged GLM-DSA post-evict step", contigStep.Step(31), pagedStep.Step(31))
}

func assertGLMDsaCachesEqual(t *testing.T, tag string, want, got *KVCache) {
	t.Helper()
	if want.Len() != got.Len() {
		t.Fatalf("%s Len=%d, want %d", tag, got.Len(), want.Len())
	}
	for l := 0; l < want.cfg.NumLayers; l++ {
		assertFloat32BitsEqual(t, tag+" K l"+itoa(l), want.glm.K[l], got.glm.K[l])
		assertFloat32BitsEqual(t, tag+" Kraw l"+itoa(l), want.glm.Kraw[l], got.glm.Kraw[l])
		assertFloat32BitsEqual(t, tag+" V l"+itoa(l), want.glm.V[l], got.glm.V[l])
		assertFloat64BitsEqual(t, tag+" IndexK l"+itoa(l), want.glm.IndexK[l], got.glm.IndexK[l])
		assertFloat64BitsEqual(t, tag+" IndexKraw l"+itoa(l), want.glm.IndexKraw[l], got.glm.IndexKraw[l])
	}
}

func assertFloat64BitsEqual(t *testing.T, name string, want, got []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s len=%d, want %d", name, len(got), len(want))
	}
	for i := range want {
		if math.Float64bits(got[i]) != math.Float64bits(want[i]) {
			t.Fatalf("%s[%d] bits=%016x, want %016x (%g vs %g)",
				name, i, math.Float64bits(got[i]), math.Float64bits(want[i]), got[i], want[i])
		}
	}
}
