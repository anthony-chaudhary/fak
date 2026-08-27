package model

import "testing"

var (
	qwenPagedSwapBenchBlob  []byte
	qwenPagedSwapBenchCache *KVCache
)

func BenchmarkQwenHybridPagedSwapCodec(b *testing.B) {
	const (
		promptTokens = 8
		blockTokens  = 4
	)
	cfg := qwen38PagedSwapBenchConfig()
	cache := qwen38PagedSwapBenchCache(cfg, promptTokens)
	blob, err := QwenHybridKVCacheToHost(cache, blockTokens)
	if err != nil {
		b.Fatal(err)
	}
	payloadBytes := int64(len(blob))

	b.Run("Encode", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(payloadBytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qwenPagedSwapBenchBlob, err = QwenHybridKVCacheToHost(cache, blockTokens)
			if err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(payloadBytes), "payload_bytes")
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(int64(b.N)*payloadBytes), "ns/byte")
	})

	b.Run("Decode", func(b *testing.B) {
		b.ReportAllocs()
		b.SetBytes(payloadBytes)
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			qwenPagedSwapBenchCache, err = QwenHybridKVCacheFromHost(cfg, blob)
			if err != nil {
				b.Fatal(err)
			}
		}
		b.ReportMetric(float64(payloadBytes), "payload_bytes")
		b.ReportMetric(float64(b.Elapsed().Nanoseconds())/float64(int64(b.N)*payloadBytes), "ns/byte")
	})
}

func qwen38PagedSwapBenchConfig() Config {
	layerTypes := make([]string, 64)
	for l := range layerTypes {
		if l%4 == 3 {
			layerTypes[l] = "full_attention"
		} else {
			layerTypes[l] = "linear_attention"
		}
	}
	return Config{
		ModelType:             "qwen3_5_text",
		NumLayers:             len(layerTypes),
		NumKVHeads:            4,
		HeadDim:               256,
		LayerTypes:            layerTypes,
		LinearConvKernelDim:   4,
		LinearNumKeyHeads:     16,
		LinearNumValueHeads:   48,
		LinearKeyHeadDim:      128,
		LinearValueHeadDim:    128,
		FullAttentionInterval: 4,
	}
}

func qwen38PagedSwapBenchCache(cfg Config, tokens int) *KVCache {
	cache := NewKVCache(cfg)
	cache.pos = make([]int, tokens)
	for i := range cache.pos {
		cache.pos[i] = i
	}
	stride := cache.kvStride()
	for _, l := range qwenFullAttentionLayers(cfg) {
		cache.K[l] = make([]float32, tokens*stride)
		cache.Kraw[l] = make([]float32, tokens*stride)
		cache.V[l] = make([]float32, tokens*stride)
	}
	_, _, _, _, _, _, convDim := cfg.linearAttnDims()
	for l := range cache.linear.layers {
		if !cfg.isLinearAttnLayer(l) {
			continue
		}
		cache.linear.layers[l].conv = make([][]float32, cfg.LinearConvKernelDim-1)
		for row := range cache.linear.layers[l].conv {
			cache.linear.layers[l].conv[row] = make([]float32, convDim)
		}
	}
	return cache
}
