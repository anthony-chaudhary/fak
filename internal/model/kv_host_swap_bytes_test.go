package model

import (
	"bytes"
	"testing"
)

// fak-test:runtime fast est=1s lane=default
func TestKVCacheHostSwapBytes(t *testing.T) {
	for _, hybrid := range []bool{false, true} {
		name := "dense"
		cfg := Config{NumLayers: 2, NumKVHeads: 2, HeadDim: 3}
		if hybrid {
			name, cfg = "qwen-hybrid", qwen35HybridTestCfg()
		}
		t.Run(name, func(t *testing.T) {
			for _, tokens := range []int{0, 1, 4, 5} {
				cache := NewKVCache(cfg)
				if hybrid {
					session := NewSynthetic(cfg).NewSession()
					t.Cleanup(session.Close)
					if tokens > 0 {
						session.Prefill(make([]int, tokens))
					}
					cache = session.Cache
				} else {
					cache.pos = make([]int, tokens)
					for l := 0; l < cfg.NumLayers; l++ {
						cache.K[l] = make([]float32, tokens*cfg.NumKVHeads*cfg.HeadDim)
						cache.Kraw[l] = make([]float32, len(cache.K[l]))
						cache.V[l] = make([]float32, len(cache.K[l]))
					}
				}
				want, err := KVCacheHostSwapBytes(cache, 4)
				if err != nil {
					t.Fatal(err)
				}
				encode := func() []byte {
					t.Helper()
					if hybrid {
						blob, err := QwenHybridKVCacheToHost(cache, 4)
						if err != nil {
							t.Fatal(err)
						}
						return blob
					}
					pool := NewPagedKVPoolWithRaw(cfg, 4)
					seq, err := KVCacheToPaged(pool, cache)
					if err != nil {
						t.Fatal(err)
					}
					defer seq.Free()
					blob, err := seq.SwapToHost()
					if err != nil {
						t.Fatal(err)
					}
					return blob
				}
				first := encode()
				if len(first) != want || cap(first) != want {
					t.Fatalf("tokens=%d extent=%d, actual len/cap=%d/%d", tokens, want, len(first), cap(first))
				}
				if _, err := KVCacheHostSwapBytes(cache, 4); err != nil || !bytes.Equal(first, encode()) {
					t.Fatal("size query changed serializer bytes or source ownership")
				}
			}
		})
	}

	t.Run("invalid-and-overflow", func(t *testing.T) {
		if _, err := KVCacheHostSwapBytes(nil, 4); err == nil {
			t.Fatal("nil cache accepted")
		}
		c := NewKVCache(Config{NumLayers: 1, NumKVHeads: 1, HeadDim: 1})
		if _, err := KVCacheHostSwapBytes(c, 0); err == nil {
			t.Fatal("nonpositive block size accepted")
		}
		c.pos = []int{0}
		if _, err := KVCacheHostSwapBytes(c, 4); err == nil {
			t.Fatal("short rows accepted")
		}
		c.K[0], c.Kraw[0], c.V[0] = []float32{1}, []float32{2}, []float32{3}
		if _, err := KVCacheHostSwapBytes(c, int(^uint(0)>>1)); err == nil {
			t.Fatal("addressable padded extent overflow accepted")
		}
		c.cfg.NumKVHeads, c.cfg.HeadDim = int(^uint(0)>>1), 2
		if _, err := KVCacheHostSwapBytes(c, 4); err == nil {
			t.Fatal("row width overflow accepted")
		}
	})
}
