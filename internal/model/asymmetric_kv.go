package model

import (
	"fmt"
	"math"
	"strings"
)

// asymmetric_kv.go — Asymmetric K/V bit allocation codec and memory budgeting for KV caches (#10731).
//
// Worldview & Mathematical Foundation:
// Borrowed from julianmb/q38rocm ("Asymmetric TurboQuant KV Cache: Keys Q8_0, Values turbo4 4-bit").
// Attention mechanisms exhibit an intrinsic mathematical asymmetry in quantization sensitivity:
//
//  1. Softmax Attention Routing:
//     Attention scores are computed as A = softmax(Q * K^T / sqrt(d)).
//     Because of the exponential operator in softmax, quantization noise in Keys causes exponential
//     distortion of the attention distribution, leading to routing collapse and lost recall at long
//     contexts. Keys therefore require higher precision: 8-bit affine (Q8_0) or FP16.
//
//  2. Value Aggregation:
//     Context outputs are formed by linear convex combination: Output = sum_i A_i * V_i.
//     Quantization errors in Values are linear and largely average out across the attention span.
//     Values thus comfortably tolerate aggressive 4-bit compression (e.g. KVQuant4 affine group 32)
//     with near-zero perplexity impact.
//
// Memory Impact:
// At quarter-million (262,144 / 262K) context tokens, KV cache dominates memory:
//   - FP32 baseline: ~61.4–80.3 GB.
//   - Asymmetric K=8, V=4: ~15.6–20.08 GB (~3–4x memory reduction).
// This enables long-context inference on unified-memory APUs and consumer GPUs without OOM.

// KVPrecision defines the precision format for Key or Value vector storage.
type KVPrecision string

const (
	// KVPrecisionFP32 indicates unquantized 32-bit floating point (4 bytes/elem).
	KVPrecisionFP32 KVPrecision = "fp32"

	// KVPrecisionFP16 indicates IEEE 754 binary16 half-precision float (2 bytes/elem).
	KVPrecisionFP16 KVPrecision = "fp16"

	// KVPrecisionQ8_0 indicates 8-bit group-32 affine quantization with per-group f32 scale/min.
	KVPrecisionQ8_0 KVPrecision = "q8_0"

	// KVPrecisionQ4_0 indicates 4-bit group-32 affine quantization with per-group f32 scale/min.
	KVPrecisionQ4_0 KVPrecision = "q4_0"
)

// ParseKVPrecision parses a case-insensitive string into a recognized KVPrecision.
func ParseKVPrecision(s string) (KVPrecision, error) {
	norm := strings.ToLower(strings.TrimSpace(s))
	switch norm {
	case "fp32", "32", "f32":
		return KVPrecisionFP32, nil
	case "fp16", "16", "f16":
		return KVPrecisionFP16, nil
	case "q8_0", "q8", "8":
		return KVPrecisionQ8_0, nil
	case "q4_0", "q4", "4":
		return KVPrecisionQ4_0, nil
	default:
		return "", fmt.Errorf("unknown KV precision %q (must be fp32, fp16, q8_0, or q4_0)", s)
	}
}

// KVVectorBytes calculates the packed storage bytes for a vector of length dim at precision prec,
// including all per-group scale and min metadata overhead for quantized formats.
func KVVectorBytes(dim int, prec KVPrecision) int64 {
	if dim <= 0 {
		return 0
	}
	switch prec {
	case KVPrecisionFP32:
		return int64(dim) * 4
	case KVPrecisionFP16:
		return int64(dim) * 2
	case KVPrecisionQ8_0:
		groups := int64((dim + KVQuant8GroupSize - 1) / KVQuant8GroupSize)
		// 1 byte per code + 8 bytes (f32 scale + f32 min) per group
		return int64(dim) + groups*8
	case KVPrecisionQ4_0:
		groups := int64((dim + KVQuant4GroupSize - 1) / KVQuant4GroupSize)
		// 0.5 bytes per code (packed 2 per byte) + 8 bytes (f32 scale + f32 min) per group
		payloadBytes := int64((dim + 1) / 2)
		return payloadBytes + groups*8
	default:
		// Default to FP32 if unrecognised
		return int64(dim) * 4
	}
}

// AsymmetricKVCacheBytes computes the total byte footprint for a KV cache over contextTokens,
// across numLayers, where each layer has a KV dimension of kvDim (typically numKVHeads * headDim),
// using keyPrecision for Keys and valPrecision for Values.
//
// It accounts honestly for all payload nibbles/bytes and per-group metadata.
func AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim int, keyPrecision, valPrecision KVPrecision) int64 {
	if contextTokens <= 0 || numLayers <= 0 || kvDim <= 0 {
		return 0
	}
	kBytes := KVVectorBytes(kvDim, keyPrecision)
	vBytes := KVVectorBytes(kvDim, valPrecision)
	perTokenBytes := kBytes + vBytes
	return int64(contextTokens) * int64(numLayers) * perTokenBytes
}

// MaxContextTokensForBudget returns the maximum number of context tokens that fit within
// a given memory ceiling budgetBytes for the specified layer count and KV dimension.
func MaxContextTokensForBudget(budgetBytes int64, numLayers, kvDim int, keyPrecision, valPrecision KVPrecision) int {
	if budgetBytes <= 0 || numLayers <= 0 || kvDim <= 0 {
		return 0
	}
	perTokenBytes := int64(numLayers) * (KVVectorBytes(kvDim, keyPrecision) + KVVectorBytes(kvDim, valPrecision))
	if perTokenBytes <= 0 {
		return 0
	}
	tokens := budgetBytes / perTokenBytes
	if tokens > int64(math.MaxInt) {
		return math.MaxInt
	}
	return int(tokens)
}

// AsymmetricKVBudgetReport provides a structured memory accounting comparison across
// precision tiers at a given context length.
type AsymmetricKVBudgetReport struct {
	ContextTokens   int     `json:"context_tokens"`
	NumLayers       int     `json:"num_layers"`
	KVDim           int     `json:"kv_dim"`
	FP32Bytes       int64   `json:"fp32_bytes"`
	FP16Bytes       int64   `json:"fp16_bytes"`
	SymmetricQ8     int64   `json:"symmetric_q8_bytes"`
	SymmetricQ4     int64   `json:"symmetric_q4_bytes"`
	AsymQ8Q4Bytes   int64   `json:"asym_q8_q4_bytes"`
	AsymFP16Q4Bytes int64   `json:"asym_fp16_q4_bytes"`
	SavingsVsFP32   float64 `json:"savings_vs_fp32_ratio"`
	SavingsVsFP16   float64 `json:"savings_vs_fp16_ratio"`
}

// AsymmetricKVCacheBudgetReport generates a comprehensive memory footprint report
// comparing unquantized baselines against symmetric and asymmetric quantization tiers.
func AsymmetricKVCacheBudgetReport(contextTokens, numLayers, kvDim int) AsymmetricKVBudgetReport {
	fp32 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionFP32, KVPrecisionFP32)
	fp16 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionFP16, KVPrecisionFP16)
	q8 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionQ8_0, KVPrecisionQ8_0)
	q4 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionQ4_0, KVPrecisionQ4_0)
	asymQ8Q4 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionQ8_0, KVPrecisionQ4_0)
	asymFP16Q4 := AsymmetricKVCacheBytes(contextTokens, numLayers, kvDim, KVPrecisionFP16, KVPrecisionQ4_0)

	var savingsFP32, savingsFP16 float64
	if asymQ8Q4 > 0 {
		savingsFP32 = float64(fp32) / float64(asymQ8Q4)
		savingsFP16 = float64(fp16) / float64(asymQ8Q4)
	}

	return AsymmetricKVBudgetReport{
		ContextTokens:   contextTokens,
		NumLayers:       numLayers,
		KVDim:           kvDim,
		FP32Bytes:       fp32,
		FP16Bytes:       fp16,
		SymmetricQ8:     q8,
		SymmetricQ4:     q4,
		AsymQ8Q4Bytes:   asymQ8Q4,
		AsymFP16Q4Bytes: asymFP16Q4,
		SavingsVsFP32:   savingsFP32,
		SavingsVsFP16:   savingsFP16,
	}
}

// AsymmetricKVPair holds a single quantized Key and Value vector pair.
// Keys retain high precision (Q8_0 or FP16), while Values are compressed to 4-bit affine.
type AsymmetricKVPair struct {
	KeyPrecision KVPrecision
	ValPrecision KVPrecision
	KeyDim       int
	ValDim       int

	// Key storage:
	KeyQ8   KVQuant8  // set when KeyPrecision == KVPrecisionQ8_0
	KeyFP16 []uint16  // set when KeyPrecision == KVPrecisionFP16 (IEEE binary16 bits)
	KeyFP32 []float32 // set when KeyPrecision == KVPrecisionFP32

	// Value storage: 4-bit group-32 affine
	ValQ4 KVQuant4
}

// Bytes returns the total packed byte size of this Key/Value pair including all metadata.
func (p *AsymmetricKVPair) Bytes() int {
	if p == nil {
		return 0
	}
	var kBytes int
	switch p.KeyPrecision {
	case KVPrecisionFP32:
		kBytes = len(p.KeyFP32) * 4
	case KVPrecisionFP16:
		kBytes = len(p.KeyFP16) * 2
	case KVPrecisionQ8_0:
		kBytes = p.KeyQ8.Bytes()
	default:
		kBytes = len(p.KeyFP32) * 4
	}
	return kBytes + p.ValQ4.Bytes()
}

// KeyErrorBound returns the provable maximum absolute elementwise round-trip error for Key.
func (p *AsymmetricKVPair) KeyErrorBound() float32 {
	if p == nil {
		return 0
	}
	switch p.KeyPrecision {
	case KVPrecisionFP32:
		return 0
	case KVPrecisionFP16:
		// Machine epsilon for IEEE binary16 is 2^-11 ≈ 4.88e-4
		var maxAbs float32
		for _, h := range p.KeyFP16 {
			f := FP16ToFloat32(h)
			if f < 0 {
				f = -f
			}
			if f > maxAbs {
				maxAbs = f
			}
		}
		return maxAbs * float32(math.Pow(2, -11))
	case KVPrecisionQ8_0:
		return p.KeyQ8.ErrorBound()
	default:
		return 0
	}
}

// ValueErrorBound returns the provable maximum absolute elementwise round-trip error for Value.
func (p *AsymmetricKVPair) ValueErrorBound() float32 {
	if p == nil {
		return 0
	}
	return p.ValQ4.ErrorBound()
}

// QuantizeFP16 converts a float32 slice to IEEE 754 binary16 bits.
func QuantizeFP16(src []float32) []uint16 {
	out := make([]uint16, len(src))
	for i, v := range src {
		out[i] = Float32ToFP16(v)
	}
	return out
}

// DequantizeFP16 converts IEEE 754 binary16 bits to a float32 slice.
func DequantizeFP16(src []uint16) []float32 {
	out := make([]float32, len(src))
	for i, h := range src {
		out[i] = FP16ToFloat32(h)
	}
	return out
}

// QuantizeAsymmetricKV quantizes a Key vector and Value vector using asymmetric bit allocation.
// The Key vector is quantized with keyPrecision (default KVPrecisionQ8_0 if empty),
// while the Value vector is compressed to 4-bit affine (KVQuant4).
func QuantizeAsymmetricKV(k, v []float32, keyPrecision KVPrecision) (*AsymmetricKVPair, error) {
	if len(k) == 0 {
		return nil, fmt.Errorf("asymmetric_kv: key vector is empty")
	}
	if len(v) == 0 {
		return nil, fmt.Errorf("asymmetric_kv: value vector is empty")
	}

	if keyPrecision == "" {
		keyPrecision = KVPrecisionQ8_0
	}

	pair := &AsymmetricKVPair{
		KeyPrecision: keyPrecision,
		ValPrecision: KVPrecisionQ4_0,
		KeyDim:       len(k),
		ValDim:       len(v),
		ValQ4:        QuantizeKV4(v),
	}

	switch keyPrecision {
	case KVPrecisionQ8_0:
		pair.KeyQ8 = QuantizeKV8(k)
	case KVPrecisionFP16:
		pair.KeyFP16 = QuantizeFP16(k)
	case KVPrecisionFP32:
		pair.KeyFP32 = append([]float32(nil), k...)
	default:
		return nil, fmt.Errorf("asymmetric_kv: unsupported key precision %q", keyPrecision)
	}

	return pair, nil
}

// DequantizeAsymmetricKV reconstructs float32 Key and Value vectors from an AsymmetricKVPair.
func DequantizeAsymmetricKV(pair *AsymmetricKVPair) ([]float32, []float32, error) {
	if pair == nil {
		return nil, nil, fmt.Errorf("asymmetric_kv: pair is nil")
	}

	var k []float32
	switch pair.KeyPrecision {
	case KVPrecisionQ8_0:
		k = pair.KeyQ8.Dequantize()
	case KVPrecisionFP16:
		k = DequantizeFP16(pair.KeyFP16)
	case KVPrecisionFP32:
		k = append([]float32(nil), pair.KeyFP32...)
	default:
		return nil, nil, fmt.Errorf("asymmetric_kv: unknown key precision %q", pair.KeyPrecision)
	}

	v := pair.ValQ4.Dequantize()
	if pair.ValDim > 0 && len(v) > pair.ValDim {
		v = v[:pair.ValDim]
	}
	return k, v, nil
}

// AsKVQuantAsymmetric converts the pair to a KVQuantAsymmetric if KeyPrecision is Q8_0.
func (p *AsymmetricKVPair) AsKVQuantAsymmetric() (KVQuantAsymmetric, bool) {
	if p == nil || p.KeyPrecision != KVPrecisionQ8_0 {
		return KVQuantAsymmetric{}, false
	}
	return KVQuantAsymmetric{K: p.KeyQ8, V: p.ValQ4}, true
}

// FromKVQuantAsymmetric wraps an existing KVQuantAsymmetric into an AsymmetricKVPair.
func FromKVQuantAsymmetric(a KVQuantAsymmetric) *AsymmetricKVPair {
	valDim := a.V.N
	if valDim == 0 {
		valDim = len(a.V.Codes) * 2
	}
	keyDim := a.K.N
	if keyDim == 0 {
		keyDim = len(a.K.Codes)
	}
	return &AsymmetricKVPair{
		KeyPrecision: KVPrecisionQ8_0,
		ValPrecision: KVPrecisionQ4_0,
		KeyDim:       keyDim,
		ValDim:       valDim,
		KeyQ8:        a.K,
		ValQ4:        a.V,
	}
}

// AsymmetricKVCache provides a multi-layer, sequence-indexed cache for asymmetric KV pairs.
type AsymmetricKVCache struct {
	NumLayers    int
	KVDim        int
	KeyPrecision KVPrecision
	ValPrecision KVPrecision
	Layers       [][]*AsymmetricKVPair // [layer][position]
}

// NewAsymmetricKVCache initializes an empty AsymmetricKVCache.
func NewAsymmetricKVCache(numLayers, kvDim int, keyPrecision KVPrecision) (*AsymmetricKVCache, error) {
	if numLayers <= 0 {
		return nil, fmt.Errorf("asymmetric_kv: numLayers must be positive, got %d", numLayers)
	}
	if kvDim <= 0 {
		return nil, fmt.Errorf("asymmetric_kv: kvDim must be positive, got %d", kvDim)
	}
	if keyPrecision == "" {
		keyPrecision = KVPrecisionQ8_0
	}
	switch keyPrecision {
	case KVPrecisionQ8_0, KVPrecisionFP16, KVPrecisionFP32:
	default:
		return nil, fmt.Errorf("asymmetric_kv: unsupported key precision %q", keyPrecision)
	}

	return &AsymmetricKVCache{
		NumLayers:    numLayers,
		KVDim:        kvDim,
		KeyPrecision: keyPrecision,
		ValPrecision: KVPrecisionQ4_0,
		Layers:       make([][]*AsymmetricKVPair, numLayers),
	}, nil
}

// Append adds a new token's (Key, Value) vector pair to the given layer in the cache.
func (c *AsymmetricKVCache) Append(layer int, k, v []float32) error {
	if c == nil {
		return fmt.Errorf("asymmetric_kv: cache is nil")
	}
	if layer < 0 || layer >= c.NumLayers {
		return fmt.Errorf("asymmetric_kv: layer %d out of bounds [0, %d)", layer, c.NumLayers)
	}
	if len(k) != c.KVDim || len(v) != c.KVDim {
		return fmt.Errorf("asymmetric_kv: vector dimension mismatch: got k=%d v=%d, want %d", len(k), len(v), c.KVDim)
	}

	pair, err := QuantizeAsymmetricKV(k, v, c.KeyPrecision)
	if err != nil {
		return err
	}
	c.Layers[layer] = append(c.Layers[layer], pair)
	return nil
}

// Get retrieves and dequantizes the (Key, Value) vectors for a specific layer and position.
func (c *AsymmetricKVCache) Get(layer, pos int) ([]float32, []float32, error) {
	if c == nil {
		return nil, nil, fmt.Errorf("asymmetric_kv: cache is nil")
	}
	if layer < 0 || layer >= c.NumLayers {
		return nil, nil, fmt.Errorf("asymmetric_kv: layer %d out of bounds [0, %d)", layer, c.NumLayers)
	}
	if pos < 0 || pos >= len(c.Layers[layer]) {
		return nil, nil, fmt.Errorf("asymmetric_kv: pos %d out of bounds [0, %d)", pos, len(c.Layers[layer]))
	}
	return DequantizeAsymmetricKV(c.Layers[layer][pos])
}

// Len returns the current sequence length (token count) for a given layer.
func (c *AsymmetricKVCache) Len(layer int) int {
	if c == nil || layer < 0 || layer >= c.NumLayers {
		return 0
	}
	return len(c.Layers[layer])
}

// TotalBytes returns the total resident packed bytes across all layers and tokens in this cache.
func (c *AsymmetricKVCache) TotalBytes() int64 {
	if c == nil {
		return 0
	}
	var total int64
	for _, l := range c.Layers {
		for _, pair := range l {
			if pair != nil {
				total += int64(pair.Bytes())
			}
		}
	}
	return total
}

// Clear releases all cached token pairs across all layers.
func (c *AsymmetricKVCache) Clear() {
	if c == nil {
		return
	}
	for i := range c.Layers {
		c.Layers[i] = nil
	}
}

// Truncate caps the sequence length of each layer to at most maxLen tokens.
func (c *AsymmetricKVCache) Truncate(maxLen int) {
	if c == nil || maxLen < 0 {
		return
	}
	for l := 0; l < c.NumLayers; l++ {
		if len(c.Layers[l]) > maxLen {
			c.Layers[l] = c.Layers[l][:maxLen]
		}
	}
}

// PrecisionReport captures numerical fidelity metrics comparing original vs dequantized vectors.
type PrecisionReport struct {
	KeyCosineSimilarity float64 `json:"key_cosine_similarity"`
	KeyMaxAbsoluteError float32 `json:"key_max_absolute_error"`
	KeyMSE              float64 `json:"key_mse"`
	KeySNRdB            float64 `json:"key_snr_db"`

	ValCosineSimilarity float64 `json:"val_cosine_similarity"`
	ValMaxAbsoluteError float32 `json:"val_max_absolute_error"`
	ValMSE              float64 `json:"val_mse"`
	ValSNRdB            float64 `json:"val_snr_db"`
}

// computeVectorFidelity calculates cosine similarity, max absolute error, MSE, and SNR (dB).
func computeVectorFidelity(orig, dequant []float32) (cosSim float64, maxErr float32, mse float64, snrDB float64) {
	if len(orig) != len(dequant) || len(orig) == 0 {
		return 0, 0, 0, 0
	}

	var dot, normA, normB, errSqSum, sigSqSum float64
	for i := 0; i < len(orig); i++ {
		a := float64(orig[i])
		b := float64(dequant[i])
		dot += a * b
		normA += a * a
		normB += b * b

		diff := float32(math.Abs(a - b))
		if diff > maxErr {
			maxErr = diff
		}
		errSqSum += float64(diff) * float64(diff)
		sigSqSum += a * a
	}

	denom := math.Sqrt(normA) * math.Sqrt(normB)
	if denom > 0 {
		cosSim = dot / denom
	}

	mse = errSqSum / float64(len(orig))

	if errSqSum > 0 && sigSqSum > 0 {
		snrDB = 10.0 * math.Log10(sigSqSum/errSqSum)
	} else if errSqSum == 0 {
		snrDB = 999.0 // Perfect fidelity
	}

	return cosSim, maxErr, mse, snrDB
}

// EvaluatePrecisionRetention compares original and dequantized Key/Value vectors and returns
// their respective fidelity metrics, proving that Keys retain higher precision than Values.
func EvaluatePrecisionRetention(origK, origV, dequantK, dequantV []float32) (PrecisionReport, error) {
	if len(origK) != len(dequantK) {
		return PrecisionReport{}, fmt.Errorf("asymmetric_kv: key dimension mismatch (%d vs %d)", len(origK), len(dequantK))
	}
	if len(origV) != len(dequantV) {
		return PrecisionReport{}, fmt.Errorf("asymmetric_kv: value dimension mismatch (%d vs %d)", len(origV), len(dequantV))
	}

	kCos, kMax, kMSE, kSNR := computeVectorFidelity(origK, dequantK)
	vCos, vMax, vMSE, vSNR := computeVectorFidelity(origV, dequantV)

	return PrecisionReport{
		KeyCosineSimilarity: kCos,
		KeyMaxAbsoluteError: kMax,
		KeyMSE:              kMSE,
		KeySNRdB:            kSNR,
		ValCosineSimilarity: vCos,
		ValMaxAbsoluteError: vMax,
		ValMSE:              vMSE,
		ValSNRdB:            vSNR,
	}, nil
}
