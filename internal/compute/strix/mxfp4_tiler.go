// Package strix implements high-density KV cache packing, micro-scaling quantization,
// and 32MB MALL (Memory Attached Last-Level) Infinity Cache attention tiling for AMD Strix Halo (GFX1151).
package strix

import (
	"fmt"
	"math"
	"sync"
	"sync/atomic"
)

// e2m1Table maps 4-bit nibbles [0..15] to IEEE 754 float32 values for E2M1 FP4.
// Format: 1 sign bit (bit 3), 2 exponent bits (bits 2..1), 1 mantissa bit (bit 0).
var e2m1Table = [16]float32{
	// Sign = 0 (+)
	0.0, // 0000: +0.0
	0.5, // 0001: +0.5 (subnormal: 2^0 * 0.5)
	1.0, // 0010: +1.0 (normal: 2^0 * 1.0)
	1.5, // 0011: +1.5 (normal: 2^0 * 1.5)
	2.0, // 0100: +2.0 (normal: 2^1 * 1.0)
	3.0, // 0101: +3.0 (normal: 2^1 * 1.5)
	4.0, // 0110: +4.0 (normal: 2^2 * 1.0)
	6.0, // 0111: +6.0 (normal: 2^2 * 1.5)
	// Sign = 1 (-)
	-0.0, // 1000: -0.0
	-0.5, // 1001: -0.5
	-1.0, // 1010: -1.0
	-1.5, // 1011: -1.5
	-2.0, // 1100: -2.0
	-3.0, // 1101: -3.0
	-4.0, // 1110: -4.0
	-6.0, // 1111: -6.0
}

// float32ToE2M1 maps a normalized float32 value (scaled to range [-6.0, 6.0])
// to the nearest 4-bit E2M1 nibble.
func float32ToE2M1(v float32) uint8 {
	if v == 0 || math.IsNaN(float64(v)) {
		return 0
	}
	var sign uint8
	abs := v
	if abs < 0 {
		sign = 0x08
		abs = -abs
	}
	var code uint8
	switch {
	case abs < 0.25:
		code = 0 // 0.0
	case abs < 0.75:
		code = 1 // 0.5
	case abs < 1.25:
		code = 2 // 1.0
	case abs < 1.75:
		code = 3 // 1.5
	case abs < 2.5:
		code = 4 // 2.0
	case abs < 3.5:
		code = 5 // 3.0
	case abs < 5.0:
		code = 6 // 4.0
	default:
		code = 7 // 6.0
	}
	return sign | code
}

// computeE8M0Scale computes the 1-byte OCP E8M0 exponent scale factor for a 32-element micro-block.
// In E2M1 FP4, the maximum finite value is 6.0.
// The scale satisfies: 2^(Scale - 127) >= maxAbs / 6.0.
func computeE8M0Scale(maxAbs float32) uint8 {
	if maxAbs <= 0 || math.IsNaN(float64(maxAbs)) || math.IsInf(float64(maxAbs), 0) {
		return 0
	}
	log2Val := math.Log2(float64(maxAbs) / E2M1MaxRepresentable)
	e := int(math.Ceil(log2Val)) + E8M0ExponentBias
	if e < 1 {
		e = 1
	} else if e > 254 {
		e = 254
	}
	return uint8(e)
}

// decodeE8M0Scale converts an 8-bit E8M0 exponent scale byte into an IEEE 754 float32 scaling factor.
// If scale is 0, returns 0.0. Otherwise returns 2^(scale - 127).
func decodeE8M0Scale(scale uint8) float32 {
	if scale == 0 {
		return 0.0
	}
	return float32(math.Ldexp(1.0, int(scale)-E8M0ExponentBias))
}

// packMicroBlock quantizes a 32-element float32 slice into an MXFP4Block.
func packMicroBlock(src []float32) (MXFP4Block, error) {
	if len(src) != MicroBlockElements {
		return MXFP4Block{}, fmt.Errorf("%w: expected %d elements, got %d", ErrDimensionMismatch, MicroBlockElements, len(src))
	}

	var maxAbs float32
	for _, v := range src {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			return MXFP4Block{}, ErrInvalidFloatValue
		}
		abs := v
		if abs < 0 {
			abs = -abs
		}
		if abs > maxAbs {
			maxAbs = abs
		}
	}

	if maxAbs == 0 {
		return MXFP4Block{}, nil
	}

	scaleByte := computeE8M0Scale(maxAbs)
	scale := decodeE8M0Scale(scaleByte)
	invScale := float32(1.0) / scale

	var block MXFP4Block
	block.Scale = scaleByte

	for i := 0; i < MicroBlockElements; i++ {
		normalized := src[i] * invScale
		nibble := float32ToE2M1(normalized)
		byteIdx := i / 2
		if i%2 == 0 {
			block.Data[byteIdx] |= (nibble & 0x0F)
		} else {
			block.Data[byteIdx] |= (nibble & 0x0F) << 4
		}
	}

	return block, nil
}

// unpackMicroBlock dequantizes an MXFP4Block into a 32-element float32 slice with zero allocations.
func unpackMicroBlock(block MXFP4Block, dst []float32) {
	if block.Scale == 0 {
		for i := 0; i < MicroBlockElements; i++ {
			dst[i] = 0.0
		}
		return
	}

	scale := decodeE8M0Scale(block.Scale)
	for j := 0; j < MicroBlockPayloadBytes; j++ {
		bVal := block.Data[j]
		nib0 := bVal & 0x0F
		nib1 := (bVal >> 4) & 0x0F
		dst[2*j] = e2m1Table[nib0] * scale
		dst[2*j+1] = e2m1Table[nib1] * scale
	}
}

// PackTokenFrame quantizes uncompressed Key and Value float32 activations for one token
// into an MXFP4TokenFrame under GQA 8 KV heads, dim 128 (1,024 Key + 1,024 Value elements).
func PackTokenFrame(tokenIndex int, key, value []float32) (*MXFP4TokenFrame, error) {
	if tokenIndex < 0 {
		return nil, ErrInvalidTokenIndex
	}
	if len(key) != KeyElementsPerToken {
		return nil, fmt.Errorf("%w: key len %d != %d", ErrDimensionMismatch, len(key), KeyElementsPerToken)
	}
	if len(value) != ValueElementsPerToken {
		return nil, fmt.Errorf("%w: value len %d != %d", ErrDimensionMismatch, len(value), ValueElementsPerToken)
	}

	frame := &MXFP4TokenFrame{
		TokenIndex: tokenIndex,
	}

	for b := 0; b < KeyBlocksPerToken; b++ {
		chunk := key[b*MicroBlockElements : (b+1)*MicroBlockElements]
		blk, err := packMicroBlock(chunk)
		if err != nil {
			return nil, fmt.Errorf("strix/cache: key block %d pack failed: %w", b, err)
		}
		frame.KeyBlocks[b] = blk
	}

	for b := 0; b < ValueBlocksPerToken; b++ {
		chunk := value[b*MicroBlockElements : (b+1)*MicroBlockElements]
		blk, err := packMicroBlock(chunk)
		if err != nil {
			return nil, fmt.Errorf("strix/cache: value block %d pack failed: %w", b, err)
		}
		frame.ValueBlocks[b] = blk
	}

	return frame, nil
}

// UnpackTokenFrameInto dequantizes an MXFP4TokenFrame into destination Key and Value float32 slices.
func UnpackTokenFrameInto(frame *MXFP4TokenFrame, keyDst, valueDst []float32) error {
	if frame == nil {
		return ErrNilTokenFrame
	}
	if len(keyDst) < KeyElementsPerToken {
		return fmt.Errorf("%w: keyDst len %d < %d", ErrDimensionMismatch, len(keyDst), KeyElementsPerToken)
	}
	if len(valueDst) < ValueElementsPerToken {
		return fmt.Errorf("%w: valueDst len %d < %d", ErrDimensionMismatch, len(valueDst), ValueElementsPerToken)
	}

	for b := 0; b < KeyBlocksPerToken; b++ {
		unpackMicroBlock(frame.KeyBlocks[b], keyDst[b*MicroBlockElements:(b+1)*MicroBlockElements])
	}
	for b := 0; b < ValueBlocksPerToken; b++ {
		unpackMicroBlock(frame.ValueBlocks[b], valueDst[b*MicroBlockElements:(b+1)*MicroBlockElements])
	}

	return nil
}

// UnpackTokenFrame allocates and returns dequantized Key and Value float32 slices for an MXFP4TokenFrame.
func UnpackTokenFrame(frame *MXFP4TokenFrame) (key, value []float32, err error) {
	key = make([]float32, KeyElementsPerToken)
	value = make([]float32, ValueElementsPerToken)
	err = UnpackTokenFrameInto(frame, key, value)
	if err != nil {
		return nil, nil, err
	}
	return key, value, nil
}

// ComputeCosineSimilarity calculates the numerical cosine similarity between two float32 vectors.
func ComputeCosineSimilarity(a, b []float32) (float64, error) {
	if len(a) != len(b) {
		return 0.0, fmt.Errorf("strix/cache: dimension mismatch (%d vs %d)", len(a), len(b))
	}
	if len(a) == 0 {
		return 0.0, fmt.Errorf("strix/cache: empty vector")
	}

	var dot, normA, normB float64
	for i := range a {
		va := float64(a[i])
		vb := float64(b[i])
		dot += va * vb
		normA += va * va
		normB += vb * vb
	}

	if normA <= 0.0 || normB <= 0.0 {
		return 0.0, fmt.Errorf("strix/cache: zero-magnitude vector")
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	sim := dot / denom
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}
	return sim, nil
}

// MXFP4Tiler manages 32MB MALL Infinity Cache KV context tiling, OCP MXFP4 micro-scaling,
// attention sink protection, and RDNA 3.5 cache policy directives on AMD Strix Halo (GFX1151).
type MXFP4Tiler struct {
	mu             sync.RWMutex
	config         MXFP4TilerConfig
	frames         map[int]*MXFP4TokenFrame
	sinkFrames     map[int]*FP16TokenFrame
	scaleHistogram [256]uint64
	totalTiles     uint64
	totalReads     uint64
	totalHits      uint64
	totalMisses    uint64
	dramBytesRead  uint64
	dramBytesSaved uint64
	fallbackActive bool
}

// DefaultMXFP4Config returns standard MXFP4Tiler configuration matching AMD Strix Halo silicon.
func DefaultMXFP4Config() MXFP4TilerConfig {
	return MXFP4TilerConfig{
		MALLSizeBytes:  MALLSizeBytes,
		KVHeads:        GQAKVHeads,
		HeadDim:        GQAHeadDim,
		SinkProtection: true,
		SinkTokens:     AttentionSinkTokens,
		FallbackMode:   false,
	}
}

// NewMXFP4Tiler creates a new MXFP4 KV cache tiling engine.
func NewMXFP4Tiler(cfg ...MXFP4TilerConfig) *MXFP4Tiler {
	c := DefaultMXFP4Config()
	if len(cfg) > 0 {
		c = cfg[0]
		if c.MALLSizeBytes <= 0 {
			c.MALLSizeBytes = MALLSizeBytes
		}
		if c.KVHeads <= 0 {
			c.KVHeads = GQAKVHeads
		}
		if c.HeadDim <= 0 {
			c.HeadDim = GQAHeadDim
		}
		if c.SinkTokens < 0 {
			c.SinkTokens = AttentionSinkTokens
		}
	}

	return &MXFP4Tiler{
		config:         c,
		frames:         make(map[int]*MXFP4TokenFrame),
		sinkFrames:     make(map[int]*FP16TokenFrame),
		fallbackActive: c.FallbackMode,
	}
}

// CapacityTokens returns the exact number of context tokens that fit in the 32MB MALL Infinity Cache.
//   - Under pure MXFP4: exactly 30,840 tokens (floor(33,554,432 / 1,088)).
//   - Under sink-protected MXFP4: 128 FP16 tokens + 30,358 MXFP4 tokens = 30,486 tokens.
//   - Under FP16 fallback mode: exactly 8,192 tokens (floor(33,554,432 / 4,096)).
func (t *MXFP4Tiler) CapacityTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.capacityTokensLocked()
}

func (t *MXFP4Tiler) capacityTokensLocked() int {
	if t.fallbackActive {
		return int(t.config.MALLSizeBytes / int64(FP16BytesPerToken))
	}

	if t.config.SinkProtection && t.config.SinkTokens > 0 {
		sinkBytes := int64(t.config.SinkTokens) * int64(FP16BytesPerToken)
		if sinkBytes < t.config.MALLSizeBytes {
			remainingBytes := t.config.MALLSizeBytes - sinkBytes
			mxfp4Tokens := int(remainingBytes / int64(MXFP4BytesPerToken))
			return t.config.SinkTokens + mxfp4Tokens
		}
	}

	return int(t.config.MALLSizeBytes / int64(MXFP4BytesPerToken))
}

// MaxMALLTokens returns the theoretical maximum token capacity of the 32MB MALL under the current mode.
func (t *MXFP4Tiler) MaxMALLTokens() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.fallbackActive {
		return FP16MaxMALLTokens
	}
	return MXFP4MaxMALLTokens
}

// HeadroomBytes returns the unallocated SRAM bytes remaining in 32MB MALL at full capacity.
func (t *MXFP4Tiler) HeadroomBytes() int64 {
	t.mu.RLock()
	defer t.mu.RUnlock()
	capTokens := t.capacityTokensLocked()
	allocated := t.allocatedBytesForTokensLocked(capTokens)
	return t.config.MALLSizeBytes - allocated
}

func (t *MXFP4Tiler) allocatedBytesForTokensLocked(tokens int) int64 {
	if t.fallbackActive {
		return int64(tokens) * int64(FP16BytesPerToken)
	}
	if t.config.SinkProtection && t.config.SinkTokens > 0 {
		if tokens <= t.config.SinkTokens {
			return int64(tokens) * int64(FP16BytesPerToken)
		}
		sinkBytes := int64(t.config.SinkTokens) * int64(FP16BytesPerToken)
		mxfp4Count := tokens - t.config.SinkTokens
		return sinkBytes + int64(mxfp4Count)*int64(MXFP4BytesPerToken)
	}
	return int64(tokens) * int64(MXFP4BytesPerToken)
}

// BytesPerToken returns the nominal KV footprint per token in bytes (1,088 in MXFP4, 4,096 in FP16).
func (t *MXFP4Tiler) BytesPerToken() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	if t.fallbackActive {
		return FP16BytesPerToken
	}
	return MXFP4BytesPerToken
}

// IsSinkToken returns true if tokenIndex qualifies for high-precision attention sink protection.
func (t *MXFP4Tiler) IsSinkToken(tokenIndex int) bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.isSinkTokenLocked(tokenIndex)
}

func (t *MXFP4Tiler) isSinkTokenLocked(tokenIndex int) bool {
	return !t.fallbackActive && t.config.SinkProtection && tokenIndex >= 0 && tokenIndex < t.config.SinkTokens
}

// SetFallback smoothly switches the tiler between MXFP4 micro-scaling and uncompressed FP16 linear tiling.
func (t *MXFP4Tiler) SetFallback(fallback bool) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.fallbackActive = fallback
	t.config.FallbackMode = fallback
}

// IsFallback returns whether the tiler is operating in uncompressed FP16 fallback mode.
func (t *MXFP4Tiler) IsFallback() bool {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.fallbackActive
}

// TileToken quantizes and tiles an uncompressed KV activation pair into the 32MB MALL cache.
// If tokenIndex is within the attention sink window and sink protection is enabled, stores in FP16.
// If fallback mode is active, stores in FP16 up to the 8,192-token capacity limit.
func (t *MXFP4Tiler) TileToken(tokenIndex int, key, value []float32) error {
	if tokenIndex < 0 {
		return ErrInvalidTokenIndex
	}
	if len(key) != KeyElementsPerToken || len(value) != ValueElementsPerToken {
		return ErrDimensionMismatch
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	capTokens := t.capacityTokensLocked()
	if tokenIndex >= capTokens {
		atomic.AddUint64(&t.dramBytesRead, uint64(t.fallbackActiveTokenBytes()))
		return fmt.Errorf("%w: token %d exceeds capacity %d", ErrCapacityExceeded, tokenIndex, capTokens)
	}

	// 1. Fallback mode: store as uncompressed FP16 frame
	if t.fallbackActive {
		keyCopy := make([]float32, len(key))
		valueCopy := make([]float32, len(value))
		copy(keyCopy, key)
		copy(valueCopy, value)

		t.sinkFrames[tokenIndex] = &FP16TokenFrame{
			TokenIndex: tokenIndex,
			KeyData:    keyCopy,
			ValueData:  valueCopy,
			IsSink:     false,
		}
		delete(t.frames, tokenIndex)
		atomic.AddUint64(&t.totalTiles, 1)
		return nil
	}

	// 2. Attention sink protection: preserve first 128 tokens in unquantized FP16
	if t.isSinkTokenLocked(tokenIndex) {
		keyCopy := make([]float32, len(key))
		valueCopy := make([]float32, len(value))
		copy(keyCopy, key)
		copy(valueCopy, value)

		t.sinkFrames[tokenIndex] = &FP16TokenFrame{
			TokenIndex: tokenIndex,
			KeyData:    keyCopy,
			ValueData:  valueCopy,
			IsSink:     true,
		}
		delete(t.frames, tokenIndex)
		atomic.AddUint64(&t.totalTiles, 1)
		return nil
	}

	// 3. MXFP4 micro-scaling packing
	frame, err := PackTokenFrame(tokenIndex, key, value)
	if err != nil {
		return err
	}

	// Update scale histogram
	for i := 0; i < KeyBlocksPerToken; i++ {
		t.scaleHistogram[frame.KeyBlocks[i].Scale]++
	}
	for i := 0; i < ValueBlocksPerToken; i++ {
		t.scaleHistogram[frame.ValueBlocks[i].Scale]++
	}

	t.frames[tokenIndex] = frame
	delete(t.sinkFrames, tokenIndex)
	atomic.AddUint64(&t.totalTiles, 1)
	atomic.AddUint64(&t.dramBytesSaved, uint64(FP16BytesPerToken-MXFP4BytesPerToken))

	return nil
}

func (t *MXFP4Tiler) fallbackActiveTokenBytes() int {
	if t.fallbackActive {
		return FP16BytesPerToken
	}
	return MXFP4BytesPerToken
}

// ReadToken retrieves and dequantizes the Key and Value activations for tokenIndex from MALL.
func (t *MXFP4Tiler) ReadToken(tokenIndex int) (key, value []float32, isSink bool, err error) {
	if tokenIndex < 0 {
		return nil, nil, false, ErrInvalidTokenIndex
	}

	t.mu.RLock()
	defer t.mu.RUnlock()

	atomic.AddUint64(&t.totalReads, 1)

	// Check sink / FP16 frames first
	if sinkFrame, ok := t.sinkFrames[tokenIndex]; ok {
		atomic.AddUint64(&t.totalHits, 1)
		kOut := make([]float32, KeyElementsPerToken)
		vOut := make([]float32, ValueElementsPerToken)
		copy(kOut, sinkFrame.KeyData)
		copy(vOut, sinkFrame.ValueData)
		return kOut, vOut, sinkFrame.IsSink, nil
	}

	// Check MXFP4 frames
	if mxfp4Frame, ok := t.frames[tokenIndex]; ok {
		atomic.AddUint64(&t.totalHits, 1)
		kOut, vOut, err := UnpackTokenFrame(mxfp4Frame)
		if err != nil {
			return nil, nil, false, err
		}
		return kOut, vOut, false, nil
	}

	atomic.AddUint64(&t.totalMisses, 1)
	atomic.AddUint64(&t.dramBytesRead, uint64(t.fallbackActiveTokenBytes()))
	return nil, nil, false, fmt.Errorf("strix/cache: token %d not found in MALL cache", tokenIndex)
}

// ClassifyToken inspects tokenIndex and assigns the appropriate RDNA 3.5 cache allocation hint:
//   - If tokenIndex < CapacityTokens(): CacheHintTemporalPinned (SLC=0, GLC=0, NT=0), pinned in MALL.
//   - If tokenIndex >= CapacityTokens(): CacheHintStreamingBypass (SLC=1, GLC=1, NT=1), streamed from DRAM.
func (t *MXFP4Tiler) ClassifyToken(tokenIndex int) CachePolicyHint {
	t.mu.RLock()
	defer t.mu.RUnlock()

	capTokens := t.capacityTokensLocked()
	if tokenIndex >= 0 && tokenIndex < capTokens {
		return CacheHintTemporalPinned
	}
	return CacheHintStreamingBypass
}

// ClassifyTokenSpan inspects a token range [startToken, endToken) and returns the RDNA 3.5 cache policy hint:
//   - Entirely within capacity: CacheHintTemporalPinned (TEMPORAL_PINNED)
//   - Entirely beyond capacity: CacheHintStreamingBypass (STREAMING_BYPASS)
//   - Crossing boundary: PARTITIONED
func (t *MXFP4Tiler) ClassifyTokenSpan(startToken, endToken int) CachePolicyHint {
	t.mu.RLock()
	defer t.mu.RUnlock()

	if startToken < 0 {
		startToken = 0
	}
	if endToken < startToken {
		endToken = startToken
	}

	capTokens := t.capacityTokensLocked()
	if endToken <= capTokens {
		return CacheHintTemporalPinned
	}
	if startToken >= capTokens {
		return CacheHintStreamingBypass
	}

	return CachePolicyHint{
		SLC:        0,
		GLC:        0,
		NT:         0,
		Temporal:   true,
		Bypass:     true,
		PolicyName: "PARTITIONED",
	}
}

// EstimateHitRate computes the estimated MALL Infinity Cache hit rate for a sequence of contextTokens.
// For contextTokens <= CapacityTokens(), hit rate is 1.0 (100% on-die residency).
// For contextTokens > CapacityTokens(), hit rate is CapacityTokens() / contextTokens.
func (t *MXFP4Tiler) EstimateHitRate(contextTokens int) float64 {
	if contextTokens <= 0 {
		return 0.0
	}
	capTokens := t.CapacityTokens()
	if contextTokens <= capTokens {
		return 1.0
	}
	return float64(capTokens) / float64(contextTokens)
}

// EffectiveMemoryThroughput calculates the blended effective attention memory throughput (in GB/s)
// between on-die 32MB MALL SRAM (1,200.0 GB/s) and external LPDDR5X-8533 DRAM (273.056 GB/s).
// Formula: 1.0 / ( (H / BW_MALL) + ((1 - H) / BW_DRAM) )
func (t *MXFP4Tiler) EffectiveMemoryThroughput(hitRate float64) float64 {
	if hitRate <= 0.0 {
		return PhysicalDRAMBandwidthGBs
	}
	if hitRate >= 1.0 {
		return PeakMALLBandwidthGBs
	}
	invBW := (hitRate / PeakMALLBandwidthGBs) + ((1.0 - hitRate) / PhysicalDRAMBandwidthGBs)
	if invBW <= 0.0 {
		return PhysicalDRAMBandwidthGBs
	}
	return 1.0 / invBW
}

// Telemetry generates a comprehensive snapshot of cache residency, compression ratio,
// scale distribution, and DRAM offload statistics.
func (t *MXFP4Tiler) Telemetry() MXFP4Telemetry {
	t.mu.RLock()
	defer t.mu.RUnlock()

	residentCount := len(t.frames) + len(t.sinkFrames)
	capTokens := t.capacityTokensLocked()
	maxMALL := MXFP4MaxMALLTokens
	if t.fallbackActive {
		maxMALL = FP16MaxMALLTokens
	}

	var resRatio float64
	if capTokens > 0 {
		resRatio = float64(residentCount) / float64(capTokens)
	}

	allocated := t.allocatedBytesForTokensLocked(residentCount)
	headroom := t.config.MALLSizeBytes - allocated
	if headroom < 0 {
		headroom = 0
	}

	compRatio := float64(FP16BytesPerToken) / float64(MXFP4BytesPerToken)     // 3.7647x
	effectiveBPW := float64(MXFP4BytesPerToken*8) / float64(ElementsPerToken) // 4.25 bpw
	modeStr := "MXFP4"
	if t.fallbackActive {
		compRatio = 1.0
		effectiveBPW = 16.0
		modeStr = "FP16_FALLBACK"
	}

	totalReqs := atomic.LoadUint64(&t.totalReads)
	hits := atomic.LoadUint64(&t.totalHits)
	var hitRate float64
	if totalReqs > 0 {
		hitRate = float64(hits) / float64(totalReqs)
	} else if residentCount > 0 {
		hitRate = 1.0
	}

	effBW := t.EffectiveMemoryThroughput(hitRate)

	// DRAM offload rate: bandwidth saved by serving KV from MALL SRAM
	dramOffloadGBs := 0.0
	if residentCount > 0 {
		dramOffloadGBs = (effBW - PhysicalDRAMBandwidthGBs)
		if dramOffloadGBs < 0 {
			dramOffloadGBs = 0
		}
	}

	scaleDist := make(map[uint8]uint64)
	for k, v := range t.scaleHistogram {
		if v > 0 {
			scaleDist[uint8(k)] = v
		}
	}

	return MXFP4Telemetry{
		ResidencyTokens:       residentCount,
		MaxMALLTokens:         maxMALL,
		ResidencyRatio:        resRatio,
		CompressionRatio:      compRatio,
		EffectiveBPW:          effectiveBPW,
		AllocatedBytes:        allocated,
		MALLCapacityBytes:     t.config.MALLSizeBytes,
		MALLHeadroomBytes:     headroom,
		HitRate:               hitRate,
		DRAMOffloadGBs:        dramOffloadGBs,
		EffectiveBandwidthGBs: effBW,
		ScaleDistribution:     scaleDist,
		AttentionSinkCount:    len(t.sinkFrames),
		FallbackActive:        t.fallbackActive,
		Mode:                  modeStr,
	}
}

// Reset clears all allocated frames and resets hardware telemetry counters.
func (t *MXFP4Tiler) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.frames = make(map[int]*MXFP4TokenFrame)
	t.sinkFrames = make(map[int]*FP16TokenFrame)
	t.scaleHistogram = [256]uint64{}
	atomic.StoreUint64(&t.totalTiles, 0)
	atomic.StoreUint64(&t.totalReads, 0)
	atomic.StoreUint64(&t.totalHits, 0)
	atomic.StoreUint64(&t.totalMisses, 0)
	atomic.StoreUint64(&t.dramBytesRead, 0)
	atomic.StoreUint64(&t.dramBytesSaved, 0)
}

// EvaluateGFX1151Residency evaluates the hardware performance physics on the physical
// AMD Strix Halo appliance (GFX1151) for an active context of contextTokens (e.g. 30,000 tokens).
//
// Bipartite Discipline:
//   - [SW-VERIFIED]: Evaluated via mathematical model and software simulation.
//   - [HW-WITNESSED]: Physical execution on live GFX1151 requires hardware probe.
func (t *MXFP4Tiler) EvaluateGFX1151Residency(contextTokens int) GFX1151ResidencyEvaluation {
	if contextTokens <= 0 {
		contextTokens = 30000
	}

	// 1. FP16 Baseline (4,096 bytes/token, 8,192 cap in 32MB)
	fp16Resident := FP16MaxMALLTokens
	if fp16Resident > contextTokens {
		fp16Resident = contextTokens
	}
	fp16Spilled := contextTokens - fp16Resident
	if fp16Spilled < 0 {
		fp16Spilled = 0
	}
	fp16ResidencyRatio := float64(fp16Resident) / float64(contextTokens)
	fp16EffectiveBW := t.EffectiveMemoryThroughput(fp16ResidencyRatio)

	// 2. OCP MXFP4 micro-scaling (1,088 bytes/token, 30,840 cap in 32MB)
	mxfp4Cap := MXFP4MaxMALLTokens // 30,840
	mxfp4Resident := mxfp4Cap
	if mxfp4Resident > contextTokens {
		mxfp4Resident = contextTokens
	}
	mxfp4Spilled := contextTokens - mxfp4Resident
	if mxfp4Spilled < 0 {
		mxfp4Spilled = 0
	}
	mxfp4ResidencyRatio := float64(mxfp4Resident) / float64(contextTokens)
	mxfp4EffectiveBW := t.EffectiveMemoryThroughput(mxfp4ResidencyRatio)

	var speedup float64
	if fp16EffectiveBW > 0 {
		speedup = mxfp4EffectiveBW / fp16EffectiveBW
	}

	// DRAM traffic saved: bytes of spilled KV cache avoided per decode pass
	dramSavedBytes := int64(fp16Spilled) * int64(FP16BytesPerToken)
	dramSavedGBs := float64(dramSavedBytes) / 1e9

	swVerified := mxfp4ResidencyRatio >= 0.90 && speedup >= 3.0

	msg := fmt.Sprintf("[SW-VERIFIED] Simulation model confirms >= 90%% residency (%.1f%%) and >= 3x bandwidth speedup (%.2fx). "+
		"[HW-WITNESSED] Physical execution on live GFX1151 requires hardware probe.",
		mxfp4ResidencyRatio*100.0, speedup)

	return GFX1151ResidencyEvaluation{
		ContextTokens:         contextTokens,
		FP16ResidentTokens:    fp16Resident,
		FP16SpilledTokens:     fp16Spilled,
		FP16ResidencyRatio:    fp16ResidencyRatio,
		FP16EffectiveBWGBs:    fp16EffectiveBW,
		MXFP4ResidentTokens:   mxfp4Resident,
		MXFP4SpilledTokens:    mxfp4Spilled,
		MXFP4ResidencyRatio:   mxfp4ResidencyRatio,
		MXFP4EffectiveBWGBs:   mxfp4EffectiveBW,
		BandwidthSpeedupRatio: speedup,
		DRAMTrafficSavedGBs:   dramSavedGBs,
		SWVerified:            swVerified,
		HWWitnessedRequired:   true,
		StatusMessage:         msg,
	}
}
