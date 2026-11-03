package model

import (
	"crypto/sha256"
	"fmt"
)

// KVCacheHostSwapBytes returns the exact serialized extent of the scheduler's
// dense or Qwen hybrid host-swap route, before allocating pages or a blob.
// It is not an RSS estimate: resident KV, temporary pages, logits, Go headers
// and allocator rounding are separate. The caller must hold exclusive cache
// ownership until serialization completes and verify the returned blob extent.
func KVCacheHostSwapBytes(c *KVCache, blockTokens int) (int, error) {
	if c == nil || blockTokens <= 0 {
		return 0, fmt.Errorf("model: host swap size requires a cache and positive block size")
	}
	stride, ok := qwenSwapSizeProduct(c.cfg.NumKVHeads, c.cfg.HeadDim)
	if !ok || stride != c.kvStride() {
		return 0, fmt.Errorf("model: host swap row width overflow")
	}
	if c.cfg.IsQwen35Hybrid() {
		if c.linear == nil || len(c.linear.layers) != c.cfg.NumLayers {
			return 0, fmt.Errorf("model: Qwen hybrid swap linear layer inventory mismatch")
		}
		full := qwenFullAttentionLayers(c.cfg)
		want, ok := qwenSwapSizeProduct(c.Len(), stride)
		if !ok {
			return 0, fmt.Errorf("model: Qwen hybrid swap size overflow")
		}
		for _, layer := range full {
			if layer < 0 || layer >= len(c.K) || layer >= len(c.Kraw) || layer >= len(c.V) ||
				len(c.K[layer]) != want || len(c.Kraw[layer]) != want || len(c.V[layer]) != want {
				return 0, fmt.Errorf("model: Qwen hybrid swap layer %d row geometry mismatch", layer)
			}
		}
		body, err := qwenHybridSwapBodyBytes(c, blockTokens, full, qwenSwapCeilDiv(c.Len(), blockTokens), stride)
		if err != nil {
			return 0, err
		}
		if body > int(^uint(0)>>1)-sha256.Size {
			return 0, fmt.Errorf("model: Qwen hybrid swap size overflow")
		}
		return body + sha256.Size, nil
	}
	if c.cfg.usesMLAMoELayout() || c.cfg.isMiniMaxSparseAttn() {
		return 0, fmt.Errorf("model: KVCacheToPaged supports dense softmax KV only")
	}
	// Use the same raw-pool geometry and checked size primitive as the
	// production snapshot/serializer. No pool blocks are allocated here.
	pool := NewPagedKVPoolWithRaw(c.cfg, blockTokens)
	rows, ok := qwenSwapSizeProduct(c.Len(), pool.stride)
	if !ok || pool.nLayers != c.cfg.NumLayers || pool.stride != c.kvStride() ||
		len(c.K) < pool.nLayers || len(c.Kraw) < pool.nLayers || len(c.V) < pool.nLayers {
		return 0, fmt.Errorf("model: dense host swap geometry mismatch")
	}
	for layer := 0; layer < pool.nLayers; layer++ {
		if len(c.K[layer]) < rows || len(c.Kraw[layer]) < rows || len(c.V[layer]) < rows {
			return 0, fmt.Errorf("model: KVCacheToPaged layer %d has short KV rows", layer)
		}
	}
	blocks := qwenSwapCeilDiv(c.Len(), pool.blockTokens)
	floats, ok := qwenSwapSizeProduct(blocks, pool.nLayers, pool.planes, pool.blockTokens, pool.stride)
	if !ok {
		return 0, fmt.Errorf("model: paged KV swap blob exceeds addressable size")
	}
	return pagedKVSwapGrowSize(len(pagedKVSwapMagic)+6*4, floats)
}
