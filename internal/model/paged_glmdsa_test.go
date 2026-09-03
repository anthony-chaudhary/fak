package model

import (
	"errors"
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

func TestPagedGLMDsaBudgetAndOverflowChecks(t *testing.T) {
	// Base valid config
	baseCfg := Config{
		ModelType:     "glm-dsa",
		NumLayers:     4,
		NumHeads:      8,
		QKNopeHeadDim: 64,
		QKRopeHeadDim: 64,
		VHeadDim:      64,
		IndexHeadDim:  32,
	}

	t.Run("NegativeOrZeroLayers", func(t *testing.T) {
		for _, layers := range []int{0, -1, -100} {
			cfg := baseCfg
			cfg.NumLayers = layers
			_, err := newPagedGLMDsaKVCache(cfg, 16)
			if err == nil {
				t.Fatalf("expected error for NumLayers=%d, got nil", layers)
			}
			var bErr *PagedGLMDsaBudgetError
			if !errors.As(err, &bErr) {
				t.Fatalf("expected *PagedGLMDsaBudgetError, got %T: %v", err, err)
			}
			if bErr.Field != "NumLayers" || bErr.Reason != "non-positive layers" {
				t.Fatalf("unexpected budget error: %+v", bErr)
			}
		}
	})

	t.Run("NegativeOrZeroBlockTokens", func(t *testing.T) {
		for _, bt := range []int{0, -1, -16} {
			_, err := newPagedGLMDsaKVCache(baseCfg, bt)
			if err == nil {
				t.Fatalf("expected error for blockTokens=%d, got nil", bt)
			}
			var bErr *PagedGLMDsaBudgetError
			if !errors.As(err, &bErr) {
				t.Fatalf("expected *PagedGLMDsaBudgetError, got %T: %v", err, err)
			}
			if bErr.Field != "blockTokens" || bErr.Reason != "non-positive block tokens" {
				t.Fatalf("unexpected budget error: %+v", bErr)
			}
		}
	})

	t.Run("NegativeOrZeroStrides", func(t *testing.T) {
		// Zero/negative QK
		cfgQK := baseCfg
		cfgQK.QKNopeHeadDim = 0
		cfgQK.QKRopeHeadDim = 0
		_, err := newPagedGLMDsaKVCache(cfgQK, 16)
		if err == nil {
			t.Fatal("expected error for zero kStride, got nil")
		}
		var bErr *PagedGLMDsaBudgetError
		if !errors.As(err, &bErr) || bErr.Field != "kStride" {
			t.Fatalf("expected kStride budget error, got: %v", err)
		}

		// Zero/negative V
		cfgV := baseCfg
		cfgV.VHeadDim = 0
		_, err = newPagedGLMDsaKVCache(cfgV, 16)
		if err == nil {
			t.Fatal("expected error for zero vStride, got nil")
		}
		if !errors.As(err, &bErr) || bErr.Field != "vStride" {
			t.Fatalf("expected vStride budget error, got: %v", err)
		}

		// Zero/negative Index
		cfgIdx := baseCfg
		cfgIdx.IndexHeadDim = 0
		_, err = newPagedGLMDsaKVCache(cfgIdx, 16)
		if err == nil {
			t.Fatal("expected error for zero IndexHeadDim, got nil")
		}
		if !errors.As(err, &bErr) || bErr.Field != "IndexHeadDim" {
			t.Fatalf("expected IndexHeadDim budget error, got: %v", err)
		}
	})

	t.Run("MultiplicationOverflow", func(t *testing.T) {
		cfgHuge := baseCfg
		cfgHuge.NumHeads = 1 << 25
		cfgHuge.QKNopeHeadDim = 1 << 25
		_, err := newPagedGLMDsaKVCache(cfgHuge, 16)
		if err == nil {
			t.Fatal("expected error for huge overflowing stride, got nil")
		}
		var bErr *PagedGLMDsaBudgetError
		if !errors.As(err, &bErr) {
			t.Fatalf("expected *PagedGLMDsaBudgetError, got %T: %v", err, err)
		}
	})

	t.Run("BudgetLimitExceeded", func(t *testing.T) {
		saved := PagedGLMDsaByteBudget
		defer func() { PagedGLMDsaByteBudget = saved }()

		// Set budget to 10 KiB
		PagedGLMDsaByteBudget = 10 * 1024

		// Base config with 16 blockTokens:
		// kStride = 8 * 128 = 1024 floats = 4096 bytes per block
		// vStride = 8 * 64 = 512 floats = 2048 bytes per block
		// idxStride = 32 floats = 256 bytes per block
		// 1 block = 2*4096 + 2048 + 2*256 = 10752 bytes per layer
		// 4 layers = 43008 bytes > 10240 bytes budget
		_, err := newPagedGLMDsaKVCache(baseCfg, 16)
		if err == nil {
			t.Fatal("expected allocation budget error, got nil")
		}
		var bErr *PagedGLMDsaBudgetError
		if !errors.As(err, &bErr) {
			t.Fatalf("expected *PagedGLMDsaBudgetError, got %T: %v", err, err)
		}
		if bErr.Reason != "allocation budget exceeded" || bErr.Limit != 10*1024 {
			t.Fatalf("unexpected budget error: %+v", bErr)
		}

		// Just-above budget vs just-below budget:
		// 1 layer with small dimensions
		smallCfg := Config{
			ModelType:     "glm-dsa",
			NumLayers:     1,
			NumHeads:      1,
			QKNopeHeadDim: 4,
			QKRopeHeadDim: 4,
			VHeadDim:      4,
			IndexHeadDim:  4,
		}
		// blockTokens = 2
		// kStride = 1 * 8 = 8 floats -> 2 * 8 * 4 = 64 bytes
		// 2*K = 128 bytes
		// vStride = 1 * 4 = 4 floats -> 2 * 4 * 4 = 32 bytes
		// idxStride = 4 floats (f64) -> 2 * 4 * 8 = 64 bytes
		// 2*Index = 128 bytes
		// Total for 1 block = 128 + 32 + 128 = 288 bytes
		PagedGLMDsaByteBudget = 288
		p, err := newPagedGLMDsaKVCache(smallCfg, 2)
		if err != nil {
			t.Fatalf("expected success at exact budget, got %v", err)
		}
		if p == nil {
			t.Fatal("expected non-nil paged cache")
		}

		PagedGLMDsaByteBudget = 287
		_, err = newPagedGLMDsaKVCache(smallCfg, 2)
		if err == nil {
			t.Fatal("expected failure at just-below budget, got nil")
		}
	})
}
