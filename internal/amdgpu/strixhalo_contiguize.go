// Package amdgpu provides AMD GPU facts probing, hardware governor settings,
// and Strix Halo APU operational serving profiles and optimizations.
package amdgpu

import (
	"errors"
	"fmt"
	"math"
)

const (
	// StrixHaloPhysicalChannels is the number of 16-bit physical LPDDR5X channels on Strix Halo (256-bit total).
	StrixHaloPhysicalChannels = 16

	// ChannelInterleaveAddressMask masks the address bits determining the physical LPDDR5X channel.
	ChannelInterleaveAddressMask = 0x3C0 // Bits [9:6] interleave 64-byte cachelines across 16 channels.

	// MinPrefillTokensForContiguization is the minimum token batch size (neq1) to activate contiguization.
	MinPrefillTokensForContiguization = 64

	// DeepTokenThreshold is the token window threshold where channel camping causes severe collapse.
	DeepTokenThreshold = 32768
)

// F16KVContiguizer coordinates pre-attention KV layout reorganization
// to prevent LPDDR5X memory channel camping on AMD Strix Halo unified memory.
type F16KVContiguizer struct {
	NumKVHeads int
	HeadDim    int
	IsStrix    bool
}

// NewF16KVContiguizer creates an F16KVContiguizer configured for model geometry and hardware target.
func NewF16KVContiguizer(numKVHeads, headDim int, isStrix bool) *F16KVContiguizer {
	return &F16KVContiguizer{
		NumKVHeads: numKVHeads,
		HeadDim:    headDim,
		IsStrix:    isStrix,
	}
}

// ShouldContiguize evaluates whether f16 KV contiguization should be active for a given turn.
// On AMD Strix Halo, contiguization is required during prefill batches (neq1 >= 64) or
// deep token windows (>= 32k tokens) where strided access collapses memory bus throughput.
func (c *F16KVContiguizer) ShouldContiguize(numTokens int) bool {
	if !c.IsStrix {
		return false
	}
	return numTokens >= MinPrefillTokensForContiguization || numTokens >= DeepTokenThreshold
}

// ShouldContiguizeBatch evaluates contiguization given both prefill batch size (neq1) and total context depth.
func (c *F16KVContiguizer) ShouldContiguizeBatch(batchTokens, contextTokens int) bool {
	if !c.IsStrix {
		return false
	}
	return batchTokens >= MinPrefillTokensForContiguization || contextTokens >= DeepTokenThreshold
}

// Contiguize reorganizes token-interleaved f16 KV cache data [numTokens, numKVHeads, headDim]
// into head-contiguous layout [numKVHeads, numTokens, headDim].
// Each f16 element occupies 2 bytes.
func (c *F16KVContiguizer) Contiguize(src []byte, numTokens int) ([]byte, error) {
	if c.NumKVHeads <= 0 || c.HeadDim <= 0 {
		return nil, errors.New("amdgpu: invalid KV geometry (numKVHeads and headDim must be > 0)")
	}
	headBytes := c.HeadDim * 2
	expectedLen := numTokens * c.NumKVHeads * headBytes
	if len(src) != expectedLen {
		return nil, fmt.Errorf("amdgpu: src byte slice length %d does not match expected geometry %d (tokens=%d, heads=%d, dim=%d)",
			len(src), expectedLen, numTokens, c.NumKVHeads, c.HeadDim)
	}

	dst := make([]byte, expectedLen)
	// src is [tok][h][dim]
	// dst is [h][tok][dim]
	for tok := 0; tok < numTokens; tok++ {
		for h := 0; h < c.NumKVHeads; h++ {
			srcOffset := (tok*c.NumKVHeads + h) * headBytes
			dstOffset := (h*numTokens + tok) * headBytes
			copy(dst[dstOffset:dstOffset+headBytes], src[srcOffset:srcOffset+headBytes])
		}
	}
	return dst, nil
}

// Decontiguize reverses head-contiguous KV cache layout [numKVHeads, numTokens, headDim]
// back into token-interleaved layout [numTokens, numKVHeads, headDim].
func (c *F16KVContiguizer) Decontiguize(src []byte, numTokens int) ([]byte, error) {
	if c.NumKVHeads <= 0 || c.HeadDim <= 0 {
		return nil, errors.New("amdgpu: invalid KV geometry (numKVHeads and headDim must be > 0)")
	}
	headBytes := c.HeadDim * 2
	expectedLen := numTokens * c.NumKVHeads * headBytes
	if len(src) != expectedLen {
		return nil, fmt.Errorf("amdgpu: src byte slice length %d does not match expected geometry %d", len(src), expectedLen)
	}

	dst := make([]byte, expectedLen)
	for h := 0; h < c.NumKVHeads; h++ {
		for tok := 0; tok < numTokens; tok++ {
			srcOffset := (h*numTokens + tok) * headBytes
			dstOffset := (tok*c.NumKVHeads + h) * headBytes
			copy(dst[dstOffset:dstOffset+headBytes], src[srcOffset:srcOffset+headBytes])
		}
	}
	return dst, nil
}

// SimulateChannelDistribution models LPDDR5X memory channel distribution efficiency.
// It returns the effective channel utilization ratio (0.0 to 1.0) and the standard deviation
// of accesses across the 16 physical memory channels.
// When accesses camp on 1 or 2 channels, efficiency drops to ~0.06 - 0.20;
// when contiguous bursts spread evenly across all 16 channels, efficiency approaches 1.0.
func SimulateChannelDistribution(accessOffsets []int, lineBytes int) (utilization float64, activeChannels int) {
	if len(accessOffsets) == 0 {
		return 0.0, 0
	}
	if lineBytes <= 0 {
		lineBytes = 64
	}

	channelCounts := make([]int, StrixHaloPhysicalChannels)
	validAccesses := 0
	for _, offset := range accessOffsets {
		if offset < 0 {
			continue
		}
		validAccesses++
		// Line address
		lineIdx := offset / lineBytes
		// Extract channel index from low-order address bits [9:6]
		channelIdx := lineIdx % StrixHaloPhysicalChannels
		channelCounts[channelIdx]++
	}
	if validAccesses == 0 {
		return 0.0, 0
	}

	active := 0
	maxCount := 0
	for _, count := range channelCounts {
		if count > 0 {
			active++
		}
		if count > maxCount {
			maxCount = count
		}
	}

	// Theoretical ideal per channel
	ideal := float64(validAccesses) / float64(StrixHaloPhysicalChannels)
	// Compute variance from ideal
	sumSqDiff := 0.0
	for _, count := range channelCounts {
		diff := float64(count) - ideal
		sumSqDiff += diff * diff
	}
	variance := sumSqDiff / float64(StrixHaloPhysicalChannels)
	stdDev := math.Sqrt(variance)

	// Normalized efficiency: 1.0 = perfect distribution, approaches 0.0 if all hit one channel
	// If all hit one channel: variance = (15*(ideal^2) + (N-ideal)^2)/16 = N^2 * 15/16
	maxPossibleStdDev := float64(validAccesses) * math.Sqrt(float64(StrixHaloPhysicalChannels-1)) / float64(StrixHaloPhysicalChannels)
	if maxPossibleStdDev == 0 {
		return 1.0, active
	}

	eff := 1.0 - (stdDev / maxPossibleStdDev)
	if eff < 0.0 {
		eff = 0.0
	}
	return eff, active
}
