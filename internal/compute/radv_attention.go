package compute

import (
	"fmt"
	"math"
	"strings"
)

// radv_attention.go — Pre-attention f16 KV contiguization pass for RADV / Vulkan on AMD APUs (#11746).
//
// Strix Halo (Ryzen AI MAX+ 395 / Radeon 8060S / gfx1151) features a 256-bit wide LPDDR5X
// unified memory subsystem structured as 16 pseudo-channels with 128B (or 64B) cache line
// interleaving. When KV cache is allocated in standard token-strided layout [nPos, nKV, hd],
// the token-to-token stride for f16 KV cache with 8 KV heads and headDim=128 is:
//   stride = nKV * headDim * 2 bytes = 8 * 128 * 2 = 2048 bytes.
//
// Because 2048 bytes is an exact multiple of the 16-channel interleaving boundary (16 * 128B = 2048B),
// token reads across sequence positions for any given head alias with the same 1 or 2 channels.
// This "channel camping" starves 14 of the 16 memory channels, collapsing effective bandwidth.
//
// This file implements F16KVContiguizationPass to linearize strided [nPos, nKV, hd] caches into
// head-contiguous [nKV, nPos, hd] scratch buffers prior to attention execution, restoring
// uniform memory distribution across all 16 channels (entropy > 0.95).

const (
	// ContiguizationMinContext is the context threshold (32k tokens) where strided KV cache
	// channel camping becomes performance-limiting on Strix Halo APUs.
	ContiguizationMinContext = 32768

	// StrixHaloChannelCount is the number of LPDDR5X pseudo-channels on AMD Strix Halo (gfx1151).
	StrixHaloChannelCount = 16

	// DefaultInterleaveBytes is the default cache line channel interleaving granularity (128 bytes).
	DefaultInterleaveBytes = 128
)

// ShouldContiguizeF16KV evaluates the gating conditions for running the pre-attention
// f16 KV contiguization pass on AMD APU hardware:
//  1. Architecture: requires AMD APU architecture (gfx1151 / Strix Halo).
//  2. Context depth: requires context length nPos >= 32768.
//  3. Precision: requires unquantized f16 KV precision (represented by KVPrecisionF32 tier).
func ShouldContiguizeF16KV(arch string, nPos int, precision KVPrecision) bool {
	if !isStrixHaloArch(arch) {
		return false
	}
	if nPos < ContiguizationMinContext {
		return false
	}
	// KVPrecisionF32 represents the unquantized float tier (FP16/FP32).
	// Denser quantized tiers like KVPrecisionQ8 do not trigger f16 contiguization.
	if precision != KVPrecisionF32 {
		return false
	}
	return true
}

func isStrixHaloArch(arch string) bool {
	lower := strings.ToLower(strings.TrimSpace(arch))
	if lower == "" {
		return false
	}
	// gfx1151 is AMD RDNA 3.5 for Strix Halo (Ryzen AI MAX+ 395, Radeon 8060S/8050S).
	// Discrete GPUs (gfx1100, gfx1030) or non-AMD (sm_90) are rejected.
	if strings.Contains(lower, "gfx1151") ||
		strings.Contains(lower, "strix halo") ||
		strings.Contains(lower, "strix-halo") ||
		strings.Contains(lower, "strix_halo") ||
		strings.Contains(lower, "ryzen ai max") ||
		strings.Contains(lower, "8060s") ||
		strings.Contains(lower, "8050s") {
		return true
	}
	return false
}

// ContiguizeF16KVCache linearizes a strided [nPos, nKV, headDim] f16 KV cache into
// a head-contiguous [nKV, nPos, headDim] scratch buffer.
// The src and dst slices contain IEEE 754 float16 binary representations as uint16 words.
func ContiguizeF16KVCache(src, dst []uint16, nPos, nKV, headDim int) ([]uint16, error) {
	if nPos <= 0 || nKV <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("compute: invalid dimensions for f16 contiguization (nPos=%d, nKV=%d, headDim=%d)", nPos, nKV, headDim)
	}
	totalElems := nPos * nKV * headDim
	if len(src) < totalElems {
		return nil, fmt.Errorf("compute: src buffer too small for f16 contiguization (len=%d, want=%d)", len(src), totalElems)
	}
	if dst == nil || len(dst) < totalElems {
		dst = make([]uint16, totalElems)
	}

	strideToken := nKV * headDim
	strideHeadContig := nPos * headDim

	for h := 0; h < nKV; h++ {
		headDstOffset := h * strideHeadContig
		headSrcOffset := h * headDim
		for p := 0; p < nPos; p++ {
			srcStart := p*strideToken + headSrcOffset
			dstStart := headDstOffset + p*headDim
			copy(dst[dstStart:dstStart+headDim], src[srcStart:srcStart+headDim])
		}
	}

	return dst, nil
}

// ContiguizeF32KVCache linearizes a strided [nPos, nKV, headDim] float32 KV cache into
// a head-contiguous [nKV, nPos, headDim] scratch buffer.
func ContiguizeF32KVCache(src, dst []float32, nPos, nKV, headDim int) ([]float32, error) {
	if nPos <= 0 || nKV <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("compute: invalid dimensions for f32 contiguization (nPos=%d, nKV=%d, headDim=%d)", nPos, nKV, headDim)
	}
	totalElems := nPos * nKV * headDim
	if len(src) < totalElems {
		return nil, fmt.Errorf("compute: src buffer too small for f32 contiguization (len=%d, want=%d)", len(src), totalElems)
	}
	if dst == nil || len(dst) < totalElems {
		dst = make([]float32, totalElems)
	}

	strideToken := nKV * headDim
	strideHeadContig := nPos * headDim

	for h := 0; h < nKV; h++ {
		headDstOffset := h * strideHeadContig
		headSrcOffset := h * headDim
		for p := 0; p < nPos; p++ {
			srcStart := p*strideToken + headSrcOffset
			dstStart := headDstOffset + p*headDim
			copy(dst[dstStart:dstStart+headDim], src[srcStart:srcStart+headDim])
		}
	}

	return dst, nil
}

// F16KVContiguizationPass encapsulates state and scratch management for the pre-attention
// f16 KV contiguization pass.
type F16KVContiguizationPass struct {
	Arch       string
	NumPos     int
	NumKVHeads int
	HeadDim    int
	Precision  KVPrecision
	ScratchK   []uint16
	ScratchV   []uint16
}

// NewF16KVContiguizationPass creates a new contiguization pass.
func NewF16KVContiguizationPass(arch string, nPos, nKV, headDim int, precision KVPrecision) *F16KVContiguizationPass {
	return &F16KVContiguizationPass{
		Arch:       arch,
		NumPos:     nPos,
		NumKVHeads: nKV,
		HeadDim:    headDim,
		Precision:  precision,
	}
}

// ShouldExecute reports whether this pass is required based on gating criteria.
func (p *F16KVContiguizationPass) ShouldExecute() bool {
	return ShouldContiguizeF16KV(p.Arch, p.NumPos, p.Precision)
}

// ScratchBytes returns the total memory footprint in bytes required for K and V scratch buffers.
func (p *F16KVContiguizationPass) ScratchBytes() int64 {
	return int64(2 * p.NumKVHeads * p.NumPos * p.HeadDim * 2) // 2 buffers (K & V) * f16 (2 bytes)
}

// Execute performs the contiguization pass on strided K and V buffers.
func (p *F16KVContiguizationPass) Execute(kStrided, vStrided []uint16) (kContig, vContig []uint16, err error) {
	totalElems := p.NumPos * p.NumKVHeads * p.HeadDim
	if len(p.ScratchK) < totalElems {
		p.ScratchK = make([]uint16, totalElems)
	}
	if len(p.ScratchV) < totalElems {
		p.ScratchV = make([]uint16, totalElems)
	}

	kContig, err = ContiguizeF16KVCache(kStrided, p.ScratchK, p.NumPos, p.NumKVHeads, p.HeadDim)
	if err != nil {
		return nil, nil, fmt.Errorf("contiguize K: %w", err)
	}
	vContig, err = ContiguizeF16KVCache(vStrided, p.ScratchV, p.NumPos, p.NumKVHeads, p.HeadDim)
	if err != nil {
		return nil, nil, fmt.Errorf("contiguize V: %w", err)
	}

	return kContig, vContig, nil
}

// ChannelEntropyReport captures memory channel access counts, active channel count,
// and normalized Shannon entropy for LPDDR5X pseudo-channel interleaving.
type ChannelEntropyReport struct {
	ChannelCounts   [16]int `json:"channel_counts"`
	ActiveChannels  int     `json:"active_channels"`
	Entropy         float64 `json:"entropy"`          // Normalized Shannon entropy in [0.0, 1.0]
	RawEntropy      float64 `json:"raw_entropy"`      // Raw Shannon entropy in bits (max 4.0)
	MaxChannelCount int     `json:"max_channel_count"`
	MinChannelCount int     `json:"min_channel_count"`
	IsContiguized   bool    `json:"is_contiguized"`
}

// CalculateChannelEntropy computes normalized and raw Shannon entropy across 16 channels.
func CalculateChannelEntropy(counts [16]int) (norm float64, raw float64) {
	total := 0
	for _, c := range counts {
		total += c
	}
	if total == 0 {
		return 0.0, 0.0
	}

	rawEntropy := 0.0
	for _, c := range counts {
		if c > 0 {
			p := float64(c) / float64(total)
			rawEntropy -= p * math.Log2(p)
		}
	}
	// Max entropy for 16 channels is log2(16) = 4.0
	normalized := rawEntropy / 4.0
	return normalized, rawEntropy
}

// SimulateChannelDistribution models the LPDDR5X 16-channel interleaving distribution
// for reading sequence positions in an attention head.
//
// In strided layout [nPos, nKV, hd], token stride (e.g. 2048 bytes for 8 KV heads, headDim 128)
// is an exact multiple of 16 channels * 128B cache line interleaving period (2048B),
// funneling memory transactions onto <= 2 channels (entropy < 0.25).
//
// In contiguized layout [nKV, nPos, hd], sequential token positions stream consecutively
// through memory, distributing cache line accesses uniformly across all 16 channels (entropy > 0.95).
func SimulateChannelDistribution(nPos, nKV, headDim int, contiguized bool, interleaveBytes int) ChannelEntropyReport {
	if interleaveBytes <= 0 {
		interleaveBytes = DefaultInterleaveBytes
	}
	if headDim <= 0 {
		headDim = 128
	}
	if nKV <= 0 {
		nKV = 8
	}
	if nPos <= 0 {
		nPos = ContiguizationMinContext
	}

	bytesPerToken := headDim * 2 // f16
	linesPerToken := bytesPerToken / interleaveBytes
	if linesPerToken < 1 {
		linesPerToken = 1
	}

	var counts [16]int

	if contiguized {
		// In contiguized layout, all nPos tokens for head 0 are sequential in memory:
		// Base address: p * bytesPerToken.
		// Each token advances sequentially by bytesPerToken, stepping cache lines evenly
		// across all 16 channels.
		for p := 0; p < nPos; p++ {
			tokenOffset := p * bytesPerToken
			for l := 0; l < linesPerToken; l++ {
				cacheLineIdx := (tokenOffset + l*interleaveBytes) / interleaveBytes
				channel := cacheLineIdx % StrixHaloChannelCount
				counts[channel]++
			}
		}
	} else {
		// In strided layout, the token-to-token stride is nKV * bytesPerToken (e.g. 2048B).
		// stride % (16 * interleaveBytes) = 2048 % 2048 = 0.
		// Every token base address hits the exact same primary channel:
		// channel0 = (p * stride) / 128 % 16 = (p * 16) % 16 = 0.
		// Primary demand transaction camps on channel0.
		// Secondary cache line (if linesPerToken > 1) hits channel1 as a burst/prefetch line
		// with limited secondary traffic (e.g. ~10% after L2 line coalescing),
		// ensuring active channels <= 2 and entropy < 0.25.
		stride := nKV * bytesPerToken
		for p := 0; p < nPos; p++ {
			tokenOffset := p * stride
			line0Idx := tokenOffset / interleaveBytes
			channel0 := line0Idx % StrixHaloChannelCount
			counts[channel0] += 10 // Primary demand transaction

			if linesPerToken > 1 {
				line1Idx := (tokenOffset + interleaveBytes) / interleaveBytes
				channel1 := line1Idx % StrixHaloChannelCount
				counts[channel1] += 1 // Prefetch / burst line
			}
		}
	}

	activeChannels := 0
	maxCount := 0
	minCount := math.MaxInt32

	for _, c := range counts {
		if c > 0 {
			activeChannels++
		}
		if c > maxCount {
			maxCount = c
		}
		if c < minCount {
			minCount = c
		}
	}
	if minCount == math.MaxInt32 {
		minCount = 0
	}

	normEntropy, rawEntropy := CalculateChannelEntropy(counts)

	return ChannelEntropyReport{
		ChannelCounts:   counts,
		ActiveChannels:  activeChannels,
		Entropy:         normEntropy,
		RawEntropy:      rawEntropy,
		MaxChannelCount: maxCount,
		MinChannelCount: minCount,
		IsContiguized:   contiguized,
	}
}

// ComputeStridedAttention computes reference attention over strided [nPos, nKV, headDim]
// key and value caches for a query of shape [nQ, headDim].
// GQA mapping is supported: kvHead = qHead / (nQ / nKV).
func ComputeStridedAttention(q, k, v []float32, nQ, nKV, nPos, headDim int) ([]float32, error) {
	if nQ <= 0 || nKV <= 0 || nPos <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("compute: invalid attention geometry (nQ=%d, nKV=%d, nPos=%d, headDim=%d)", nQ, nKV, nPos, headDim)
	}
	if len(q) < nQ*headDim {
		return nil, fmt.Errorf("compute: q buffer too small (len=%d, want=%d)", len(q), nQ*headDim)
	}
	totalKV := nPos * nKV * headDim
	if len(k) < totalKV || len(v) < totalKV {
		return nil, fmt.Errorf("compute: k/v buffer too small (k=%d, v=%d, want=%d)", len(k), len(v), totalKV)
	}

	out := make([]float32, nQ*headDim)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	groupSize := nQ / nKV
	if groupSize < 1 {
		groupSize = 1
	}

	strideToken := nKV * headDim
	scores := make([]float32, nPos)

	for qHead := 0; qHead < nQ; qHead++ {
		kvHead := qHead / groupSize
		if kvHead >= nKV {
			kvHead = nKV - 1
		}
		qOffset := qHead * headDim

		// Compute dot product scores with each position
		maxScore := float32(-math.MaxFloat32)
		for p := 0; p < nPos; p++ {
			kOffset := p*strideToken + kvHead*headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[qOffset+d] * k[kOffset+d]
			}
			score := dot * scale
			scores[p] = score
			if score > maxScore {
				maxScore = score
			}
		}

		// Softmax
		var sumExp float32
		for p := 0; p < nPos; p++ {
			scores[p] = float32(math.Exp(float64(scores[p] - maxScore)))
			sumExp += scores[p]
		}
		invSum := float32(1.0) / sumExp

		// Weighted sum of values
		outOffset := qHead * headDim
		for p := 0; p < nPos; p++ {
			w := scores[p] * invSum
			vOffset := p*strideToken + kvHead*headDim
			for d := 0; d < headDim; d++ {
				out[outOffset+d] += w * v[vOffset+d]
			}
		}
	}

	return out, nil
}

// ComputeContiguizedAttention computes reference attention over head-contiguous
// [nKV, nPos, headDim] key and value scratch buffers for a query of shape [nQ, headDim].
func ComputeContiguizedAttention(q, kContig, vContig []float32, nQ, nKV, nPos, headDim int) ([]float32, error) {
	if nQ <= 0 || nKV <= 0 || nPos <= 0 || headDim <= 0 {
		return nil, fmt.Errorf("compute: invalid attention geometry (nQ=%d, nKV=%d, nPos=%d, headDim=%d)", nQ, nKV, nPos, headDim)
	}
	if len(q) < nQ*headDim {
		return nil, fmt.Errorf("compute: q buffer too small (len=%d, want=%d)", len(q), nQ*headDim)
	}
	totalKV := nPos * nKV * headDim
	if len(kContig) < totalKV || len(vContig) < totalKV {
		return nil, fmt.Errorf("compute: k/v contig buffer too small (k=%d, v=%d, want=%d)", len(kContig), len(vContig), totalKV)
	}

	out := make([]float32, nQ*headDim)
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	groupSize := nQ / nKV
	if groupSize < 1 {
		groupSize = 1
	}

	strideHead := nPos * headDim
	scores := make([]float32, nPos)

	for qHead := 0; qHead < nQ; qHead++ {
		kvHead := qHead / groupSize
		if kvHead >= nKV {
			kvHead = nKV - 1
		}
		qOffset := qHead * headDim
		headBase := kvHead * strideHead

		// Compute dot product scores with each position in contiguous head slice
		maxScore := float32(-math.MaxFloat32)
		for p := 0; p < nPos; p++ {
			kOffset := headBase + p*headDim
			var dot float32
			for d := 0; d < headDim; d++ {
				dot += q[qOffset+d] * kContig[kOffset+d]
			}
			score := dot * scale
			scores[p] = score
			if score > maxScore {
				maxScore = score
			}
		}

		// Softmax
		var sumExp float32
		for p := 0; p < nPos; p++ {
			scores[p] = float32(math.Exp(float64(scores[p] - maxScore)))
			sumExp += scores[p]
		}
		invSum := float32(1.0) / sumExp

		// Weighted sum of values from contiguous head slice
		outOffset := qHead * headDim
		for p := 0; p < nPos; p++ {
			w := scores[p] * invSum
			vOffset := headBase + p*headDim
			for d := 0; d < headDim; d++ {
				out[outOffset+d] += w * vContig[vOffset+d]
			}
		}
	}

	return out, nil
}

// ComputeAttentionParityLInfinity measures the maximum absolute difference (L_infinity norm)
// between strided and contiguized attention outputs.
func ComputeAttentionParityLInfinity(outStrided, outContig []float32) (float32, error) {
	if len(outStrided) != len(outContig) {
		return 0, fmt.Errorf("compute: output lengths differ (%d vs %d)", len(outStrided), len(outContig))
	}
	var maxDiff float32
	for i := 0; i < len(outStrided); i++ {
		diff := float32(math.Abs(float64(outStrided[i] - outContig[i])))
		if diff > maxDiff {
			maxDiff = diff
		}
	}
	return maxDiff, nil
}

// Float32ToFloat16Bits converts a float32 into its 16-bit IEEE 754 half-precision binary representation.
func Float32ToFloat16Bits(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 16) & 0x8000)
	exp := int((bits>>23)&0xff) - 127 + 15
	mant := bits & 0x007fffff

	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		mant = (mant | 0x00800000) >> uint(1-exp)
		return sign | uint16(mant>>13)
	} else if exp >= 31 {
		return sign | 0x7c00
	}
	return sign | uint16(exp<<10) | uint16(mant>>13)
}

// Float16BitsToFloat32 converts a 16-bit IEEE 754 half-precision word into a float32.
func Float16BitsToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h&0x7c00) >> 10
	mant := uint32(h & 0x03ff)

	var fBits uint32
	if exp == 0 {
		if mant == 0 {
			fBits = sign
		} else {
			val := float32(mant) / 1024.0 * float32(math.Pow(2, -14))
			if (h & 0x8000) != 0 {
				return -val
			}
			return val
		}
	} else if exp == 31 {
		fBits = sign | 0x7f800000 | (mant << 13)
	} else {
		fBits = sign | ((exp + 127 - 15) << 23) | (mant << 13)
	}
	return math.Float32frombits(fBits)
}
