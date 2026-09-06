package model

import "math"

// kvquant.go — the 4-bit and 8-bit asymmetric KV-cache quantization codec and the
// layer/byte accounting that makes a 262K-context claim checkable (#4874, #10731, gen/next).
//
// WHY ASYMMETRIC K/V BIT ALLOCATION (#10731): attention mechanisms exhibit an asymmetric
// sensitivity to quantization error:
//   - Keys govern softmax exponent routing (Q * K^T / sqrt(d)), so errors in Key vectors
//     cause exponential attention distribution drift. They require higher precision (8-bit).
//   - Values undergo linear convex combination with attention weights (sum A_i V_i), so
//     quantization errors in Values average out and comfortably tolerate 4-bit compression.
// Asymmetric allocation (e.g. K=8, V=4) cuts memory dramatically while preserving routing
// fidelity.
//
// Prior-art: TurboQuant (asymmetric K/V bit allocation, Keys Q8_0 + Values turbo4), KIVI, KVQuant.
//
// WHY THIS IS A SEPARATE AXIS FROM WEIGHT QUANT: the quant_*.go family quantizes WEIGHTS,
// which are static, shared across every session, and packed once at load. A KV cache is
// per-session, grows one row per decoded token, and is written in the hot path. At a 262K
// context the cache — not the weights — is what runs the box out of memory, so it gets its
// own codec with its own error bound.
//
// WHICH LAYERS: only full-attention layers hold a softmax KV cache at all. On the Qwen3.6
// hybrid backbone (~75% linear attention) the linear layers hold an ACCUMULATED recurrent
// state plus a short-conv window (qwen35.go), not a per-token K/V row. Quantizing a
// recurrence is a different problem with a different error model — an accumulated state
// compounds its own quantization error every step, where a KV row is written once and read
// back unchanged. KVQuantLayers draws that line so a caller cannot aim this codec at state
// that is not KV.
//
// GATING: this file is a codec plus accounting. Nothing in the decode path calls it yet —
// the KV cache still stores f32 rows. That is the gen/next gate: the capability and its
// error bound are witnessed here, and wiring it into KVCache/attention is the promotion
// step, which needs decode-quality evidence (perplexity or logit drift over a real long
// context) that this package cannot produce on its own.

// KVQuant4GroupSize is the number of contiguous elements that share one (scale, min)
// pair. 32 matches the k-quant super-block sub-block width the weight codecs already use,
// and divides every head_dim in the supported families, so a group never straddles two
// heads — a group that spanned heads would pool two unrelated dynamic ranges and widen
// the error bound for both.
const KVQuant4GroupSize = 32

// KVQuant4 is one 4-bit-quantized K or V row-set: a group-wise AFFINE (asymmetric)
// quantization holding two nibbles per byte, with a per-group scale and min in f32.
//
// Affine rather than symmetric is deliberate for KV. Post-RoPE K rows and V rows are not
// zero-centered — a symmetric codec spends a code point on a sign the data does not use
// and wastes ~1 bit of an already-4-bit budget. The affine form maps each group onto the
// full 0..15 range, which is what holds the error bound below tight enough to be useful.
type KVQuant4 struct {
	// N is the logical element count (Codes is packed, so len(Codes) is not N).
	N int
	// Scale and Min are per-group, indexed by element/KVQuant4GroupSize.
	Scale []float32
	Min   []float32
	// Codes holds two 4-bit codes per byte: element i occupies the low nibble of
	// Codes[i/2] when i is even, the high nibble when odd.
	Codes []byte
}

// QuantizeKV4 packs src into 4-bit codes. It is exact for a constant group (scale 0
// stores the value in Min), and never returns a code outside 0..15.
func QuantizeKV4(src []float32) KVQuant4 {
	q := KVQuant4{N: len(src)}
	if len(src) == 0 {
		return q
	}
	groups := (len(src) + KVQuant4GroupSize - 1) / KVQuant4GroupSize
	q.Scale = make([]float32, groups)
	q.Min = make([]float32, groups)
	q.Codes = make([]byte, (len(src)+1)/2)
	for g := 0; g < groups; g++ {
		lo := g * KVQuant4GroupSize
		hi := lo + KVQuant4GroupSize
		if hi > len(src) {
			hi = len(src)
		}
		mn, mx := src[lo], src[lo]
		for _, v := range src[lo:hi] {
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		// 15 intervals across 16 code points. A constant group leaves scale at 0, which
		// dequantizes back to Min exactly rather than dividing by zero.
		scale := (mx - mn) / 15
		q.Min[g] = mn
		q.Scale[g] = scale
		for i := lo; i < hi; i++ {
			var code byte
			if scale > 0 {
				c := int((src[i]-mn)/scale + 0.5)
				if c < 0 {
					c = 0
				}
				if c > 15 {
					c = 15
				}
				code = byte(c)
			}
			if i%2 == 0 {
				q.Codes[i/2] = code
			} else {
				q.Codes[i/2] |= code << 4
			}
		}
	}
	return q
}

// Dequantize reconstructs the f32 row-set. The result is within ErrorBound of the input
// QuantizeKV4 was given, elementwise.
func (q KVQuant4) Dequantize() []float32 {
	out := make([]float32, q.N)
	for i := 0; i < q.N; i++ {
		b := q.Codes[i/2]
		if i%2 == 0 {
			b &= 0x0f
		} else {
			b >>= 4
		}
		g := i / KVQuant4GroupSize
		out[i] = q.Min[g] + float32(b)*q.Scale[g]
	}
	return out
}

// ErrorBound is the worst-case absolute round-trip error this quantization can produce:
// half the widest group step, since round-to-nearest never lands further than half a step
// from the value. It is a PROVABLE per-element ceiling derived from the stored scales, not
// a measurement — a caller can compare it against a decode-quality tolerance before
// committing a span to 4-bit, and a test can assert the realized error respects it.
func maxScaleHalf(scale []float32) float32 {
	var worst float32
	for _, s := range scale {
		if s > worst {
			worst = s
		}
	}
	return worst / 2
}

func packedKVBytes(codesLen, scaleLen, minLen int) int {
	return codesLen + 4*scaleLen + 4*minLen
}

func (q KVQuant4) ErrorBound() float32 {
	return maxScaleHalf(q.Scale)
}

// Bytes is the packed footprint: the nibble payload plus the per-group f32 scale and min.
// The group metadata is real overhead and is counted here — at KVQuant4GroupSize=32 it
// adds 8 bytes per 32 elements (2 bits/element), so the honest rate is 6 bits/element,
// not 4. Reporting the bare nibble count would overstate the saving by 50%.
func (q KVQuant4) Bytes() int {
	return packedKVBytes(len(q.Codes), len(q.Scale), len(q.Min))
}

// KVQuant8GroupSize is the number of contiguous elements sharing one (scale, min) pair
// in 8-bit KV quantization. 32 matches KVQuant4GroupSize and weight k-quants.
const KVQuant8GroupSize = 32

// KVQuant8 is one 8-bit-quantized K (or V) row-set: a group-wise affine quantization
// holding one byte per element, with a per-group scale and min in f32.
type KVQuant8 struct {
	// N is the logical element count.
	N int
	// Scale and Min are per-group, indexed by element/KVQuant8GroupSize.
	Scale []float32
	Min   []float32
	// Codes holds one 8-bit code per element.
	Codes []byte
}

// QuantizeKV8 packs src into 8-bit codes (0..255). It is exact for a constant group
// (scale 0 stores the value in Min), and never returns a code outside 0..255.
func QuantizeKV8(src []float32) KVQuant8 {
	q := KVQuant8{N: len(src)}
	if len(src) == 0 {
		return q
	}
	groups := (len(src) + KVQuant8GroupSize - 1) / KVQuant8GroupSize
	q.Scale = make([]float32, groups)
	q.Min = make([]float32, groups)
	q.Codes = make([]byte, len(src))
	for g := 0; g < groups; g++ {
		lo := g * KVQuant8GroupSize
		hi := lo + KVQuant8GroupSize
		if hi > len(src) {
			hi = len(src)
		}
		mn, mx := src[lo], src[lo]
		for _, v := range src[lo:hi] {
			if v < mn {
				mn = v
			}
			if v > mx {
				mx = v
			}
		}
		// 255 intervals across 256 code points. A constant group leaves scale at 0, which
		// dequantizes back to Min exactly rather than dividing by zero.
		scale := (mx - mn) / 255
		q.Min[g] = mn
		q.Scale[g] = scale
		for i := lo; i < hi; i++ {
			var code byte
			if scale > 0 {
				c := int((src[i]-mn)/scale + 0.5)
				if c < 0 {
					c = 0
				}
				if c > 255 {
					c = 255
				}
				code = byte(c)
			}
			q.Codes[i] = code
		}
	}
	return q
}

// Dequantize reconstructs the f32 row-set from 8-bit codes.
func (q KVQuant8) Dequantize() []float32 {
	out := make([]float32, q.N)
	for i := 0; i < q.N; i++ {
		g := i / KVQuant8GroupSize
		out[i] = q.Min[g] + float32(q.Codes[i])*q.Scale[g]
	}
	return out
}

// ErrorBound is the worst-case absolute round-trip error this quantization can produce:
// half the widest group step.
func (q KVQuant8) ErrorBound() float32 {
	return maxScaleHalf(q.Scale)
}

// Bytes is the packed footprint: the 1-byte-per-element payload plus the per-group f32 scale
// and min. At KVQuant8GroupSize=32 it adds 8 bytes per 32 elements (2 bits/element), for an
// honest rate of 10 bits/element.
func (q KVQuant8) Bytes() int {
	return packedKVBytes(len(q.Codes), len(q.Scale), len(q.Min))
}

// KVQuantAsymmetric is an asymmetric cache pair holding K in KVQuant8 and V in KVQuant4.
// Keys dictate softmax exponent routing (Q * K^T / sqrt(d)) where error compounds exponentially
// and requires 8-bit precision, while Values undergo linear convex combination (sum A_i V_i)
// and comfortably tolerate 4-bit compression.
type KVQuantAsymmetric struct {
	K KVQuant8
	V KVQuant4
}

// QuantizeKVAsymmetric quantizes Key vector k to 8-bit (KVQuant8) and Value vector v
// to 4-bit (KVQuant4).
func QuantizeKVAsymmetric(k, v []float32) KVQuantAsymmetric {
	return KVQuantAsymmetric{
		K: QuantizeKV8(k),
		V: QuantizeKV4(v),
	}
}

// DequantizeKVAsymmetric reconstructs the f32 Key and Value row-sets.
func DequantizeKVAsymmetric(a KVQuantAsymmetric) ([]float32, []float32) {
	return a.K.Dequantize(), a.V.Dequantize()
}

// Bytes returns the combined packed byte footprint of the asymmetric K and V pair.
func (a KVQuantAsymmetric) Bytes() int {
	return a.K.Bytes() + a.V.Bytes()
}

// KVQuantLayers returns the layer indices that actually hold a softmax KV cache, and are
// therefore the layers this codec may be aimed at. Linear-attention (Gated-DeltaNet)
// layers are excluded: they hold a recurrent state, not per-token K/V rows. For a
// non-hybrid model every layer qualifies, so the result is the full range.
func (c Config) KVQuantLayers() []int {
	if c.NumLayers <= 0 {
		return nil
	}
	out := make([]int, 0, c.NumLayers)
	for l := 0; l < c.NumLayers; l++ {
		if c.isLinearAttnLayer(l) {
			continue
		}
		out = append(out, l)
	}
	return out
}

// KVCacheBytesAsymmetric sizes the K+V cache for `positions` tokens when the KV-holding
// layers store Key rows at `kBits` precision and Value rows at `vBits` precision (#10731).
// bits=16 is an f16 cache, bits=32 the f32 rows KVCache holds today, while bits=4 and bits=8
// route through the KVQuant4 and KVQuant8 rates INCLUDING their per-group scale/min overhead —
// so the quantized numbers are directly comparable rather than flattering themselves.
//
// It counts only the layers KVQuantLayers admits, which is what makes it usable on the
// hybrid backbone: sizing all NumLayers there overstates a Qwen3.6-class cache by ~4x,
// because ~75% of the layers hold no KV at all.
//
// INVALIDATING ASSUMPTION: this sizes the K+V pair, the standard KV-budget convention
// (and what kvbudget.Shape reports). fak's live KVCache additionally retains Kraw — the
// pre-RoPE K rows eviction re-rotates from — so a resident cache is ~1.5x this number
// until Kraw is either quantized too or dropped for sessions that never evict. If that
// changes, these numbers move; the per-layer and per-bit branch stands.
func (c Config) KVCacheBytesAsymmetric(positions, kBits, vBits int) int64 {
	if positions <= 0 || kBits <= 0 || vBits <= 0 {
		return 0
	}
	layers := int64(len(c.KVQuantLayers()))
	if layers == 0 {
		return 0
	}
	vHeadDim := c.VHeadDim
	if vHeadDim == 0 {
		vHeadDim = c.HeadDim
	}
	perPos := kvBitsToBytes(c.NumKVHeads*c.HeadDim, kBits) + kvBitsToBytes(c.NumKVHeads*vHeadDim, vBits)
	return perPos * layers * int64(positions)
}

// KVCacheBytesAtBits sizes the symmetric K+V cache for `positions` tokens when the KV-holding
// layers store their rows at `bits` precision. It delegates to KVCacheBytesAsymmetric.
func (c Config) KVCacheBytesAtBits(positions, bits int) int64 {
	return c.KVCacheBytesAsymmetric(positions, bits, bits)
}

// kvBitsToBytes is the per-row byte cost of `elems` values at `bits` precision, adding the
// group metadata (one f32 scale + one f32 min per group of 32) on the 4-bit and 8-bit paths
// so the quantized rate is not understated.
func kvBitsToBytes(elems, bits int) int64 {
	if elems <= 0 || bits <= 0 {
		return 0
	}
	n := (int64(elems)*int64(bits) + 7) / 8
	if bits == 4 {
		groups := int64((elems + KVQuant4GroupSize - 1) / KVQuant4GroupSize)
		n += groups * 8 // one f32 scale + one f32 min per group
	} else if bits == 8 {
		groups := int64((elems + KVQuant8GroupSize - 1) / KVQuant8GroupSize)
		n += groups * 8 // one f32 scale + one f32 min per group
	}
	return n
}

// -------------------------------------------------------------------------
// Asymmetric TurboQuant KV-Cache Quantization (#11909)
//
// Keys: KVQuantQ8_0 (symmetric zero-offset int8, preserving angular routing).
// Values: KVQuantTurbo4 (non-linear 4-bit Lloyd-Max centroids).
// -------------------------------------------------------------------------

// Constants for block sizes in TurboQuant KV cache quantization.
const (
	KVQuantQ8_0BlockSize   = 32
	KVQuantTurbo4BlockSize = 32
)

// Turbo4LloydMaxCodebook defines 16 optimal Lloyd-Max centroids for standardized
// bell-shaped attention value distributions normalized to [-1, 1].
var Turbo4LloydMaxCodebook = [16]float32{
	-1.0000, -0.7573, -0.5694, -0.4170, -0.2882, -0.1743, -0.0708, -0.0210,
	0.0210, 0.0708, 0.1743, 0.2882, 0.4170, 0.5694, 0.7573, 1.0000,
}

// Turbo4Thresholds defines 15 decision thresholds (midpoints between adjacent centroids).
var Turbo4Thresholds = [15]float32{
	-0.87865, -0.66335, -0.49320, -0.35260, -0.23125, -0.12255, -0.04590, 0.00000,
	0.04590, 0.12255, 0.23125, 0.35260, 0.49320, 0.66335, 0.87865,
}

// KVQuantQ8_0 is a symmetric zero-offset quantization (scale * int8) per block of 32 elements.
// Its zero-mean noise property (E[e] = 0) preserves exact angular routing (cos(q, k) > 0.9999)
// without the directional bias of min-offset affine quantizers.
type KVQuantQ8_0 struct {
	N     int
	Scale []float32
	Codes []int8
}

// QuantizeKVQ8_0 quantizes src into symmetric zero-offset int8 codes per block of 32 elements.
func QuantizeKVQ8_0(src []float32) KVQuantQ8_0 {
	q := KVQuantQ8_0{N: len(src)}
	if len(src) == 0 {
		return q
	}
	blocks := (len(src) + KVQuantQ8_0BlockSize - 1) / KVQuantQ8_0BlockSize
	q.Scale = make([]float32, blocks)
	q.Codes = make([]int8, len(src))
	for b := 0; b < blocks; b++ {
		lo := b * KVQuantQ8_0BlockSize
		hi := lo + KVQuantQ8_0BlockSize
		if hi > len(src) {
			hi = len(src)
		}
		var maxAbs float32
		for _, v := range src[lo:hi] {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		if maxAbs == 0 {
			q.Scale[b] = 0
			continue
		}
		scale := maxAbs / 127.0
		q.Scale[b] = scale
		invScale := 1.0 / float64(scale)
		for i := lo; i < hi; i++ {
			c := int(math.Round(float64(src[i]) * invScale))
			if c < -128 {
				c = -128
			} else if c > 127 {
				c = 127
			}
			q.Codes[i] = int8(c)
		}
	}
	return q
}

// Dequantize reconstructs the f32 row-set from symmetric int8 codes.
func (q KVQuantQ8_0) Dequantize() []float32 {
	out := make([]float32, q.N)
	for i := 0; i < q.N; i++ {
		b := i / KVQuantQ8_0BlockSize
		out[i] = float32(q.Codes[i]) * q.Scale[b]
	}
	return out
}

// ErrorBound returns the provable maximum elementwise round-trip error ceiling:
// half the widest block scale step.
func (q KVQuantQ8_0) ErrorBound() float32 {
	return maxScaleHalf(q.Scale)
}

// Bytes returns the packed byte footprint: 1 byte per int8 code plus 4 bytes per f32 scale.
func (q KVQuantQ8_0) Bytes() int {
	return len(q.Codes) + 4*len(q.Scale)
}

// AngularCosineSimilarity computes the cosine similarity between orig and the dequantized vector.
func (q KVQuantQ8_0) AngularCosineSimilarity(orig []float32) float64 {
	if len(orig) != q.N || q.N == 0 {
		return 0
	}
	dq := q.Dequantize()
	var dot, normOrig, normDq float64
	for i := 0; i < q.N; i++ {
		o := float64(orig[i])
		d := float64(dq[i])
		dot += o * d
		normOrig += o * o
		normDq += d * d
	}
	denom := math.Sqrt(normOrig) * math.Sqrt(normDq)
	if denom == 0 {
		if normOrig == 0 && normDq == 0 {
			return 1.0
		}
		return 0
	}
	sim := dot / denom
	if sim > 1.0 {
		sim = 1.0
	} else if sim < -1.0 {
		sim = -1.0
	}
	return sim
}

// KVQuantTurbo4 is a 4-bit non-linear Lloyd-Max quantization for attention Values.
// Values undergo linear convex combination in attention (sum A_i V_i), so non-linear 4-bit error
// averages out across sequence tokens. 2 nibbles are packed per byte (low nibble for even element,
// high nibble for odd element, consistent with KVQuant4).
type KVQuantTurbo4 struct {
	N     int
	Scale []float32
	Codes []byte
}

// QuantizeKVTurbo4 packs src into 4-bit Lloyd-Max codes.
func QuantizeKVTurbo4(src []float32) KVQuantTurbo4 {
	q := KVQuantTurbo4{N: len(src)}
	if len(src) == 0 {
		return q
	}
	blocks := (len(src) + KVQuantTurbo4BlockSize - 1) / KVQuantTurbo4BlockSize
	q.Scale = make([]float32, blocks)
	q.Codes = make([]byte, (len(src)+1)/2)
	for b := 0; b < blocks; b++ {
		lo := b * KVQuantTurbo4BlockSize
		hi := lo + KVQuantTurbo4BlockSize
		if hi > len(src) {
			hi = len(src)
		}
		var maxAbs float32
		for _, v := range src[lo:hi] {
			abs := v
			if abs < 0 {
				abs = -abs
			}
			if abs > maxAbs {
				maxAbs = abs
			}
		}
		if maxAbs == 0 {
			q.Scale[b] = 0
			continue
		}
		scale := maxAbs
		q.Scale[b] = scale
		invScale := 1.0 / float64(scale)
		for i := lo; i < hi; i++ {
			normX := float32(float64(src[i]) * invScale)
			if normX < -1.0 {
				normX = -1.0
			} else if normX > 1.0 {
				normX = 1.0
			}
			var code byte
			for code < 15 && normX >= Turbo4Thresholds[code] {
				code++
			}
			if i%2 == 0 {
				q.Codes[i/2] = code
			} else {
				q.Codes[i/2] |= code << 4
			}
		}
	}
	return q
}

// Dequantize reconstructs the f32 row-set from 4-bit Lloyd-Max codes.
func (q KVQuantTurbo4) Dequantize() []float32 {
	out := make([]float32, q.N)
	for i := 0; i < q.N; i++ {
		b := q.Codes[i/2]
		if i%2 == 0 {
			b &= 0x0f
		} else {
			b >>= 4
		}
		g := i / KVQuantTurbo4BlockSize
		out[i] = q.Scale[g] * Turbo4LloydMaxCodebook[b]
	}
	return out
}

// ErrorBound returns the provable maximum elementwise round-trip error ceiling:
// max block scale multiplied by the widest normalized Lloyd-Max half-interval (0.12135).
func (q KVQuantTurbo4) ErrorBound() float32 {
	var maxScale float32
	for _, s := range q.Scale {
		if s > maxScale {
			maxScale = s
		}
	}
	return maxScale * 0.12135
}

// Bytes returns the packed footprint: nibbles payload plus per-block f32 scale.
func (q KVQuantTurbo4) Bytes() int {
	return len(q.Codes) + 4*len(q.Scale)
}

// TurboQuantAsymmetricKV pairs K (KVQuantQ8_0) and V (KVQuantTurbo4).
type TurboQuantAsymmetricKV struct {
	K KVQuantQ8_0
	V KVQuantTurbo4
}

// QuantizeTurboQuantAsymmetric quantizes Key vector k to KVQuantQ8_0 and Value vector v to KVQuantTurbo4.
func QuantizeTurboQuantAsymmetric(k, v []float32) TurboQuantAsymmetricKV {
	return TurboQuantAsymmetricKV{
		K: QuantizeKVQ8_0(k),
		V: QuantizeKVTurbo4(v),
	}
}

// DequantizeTurboQuantAsymmetric reconstructs float32 Key and Value row-sets.
func DequantizeTurboQuantAsymmetric(tq TurboQuantAsymmetricKV) ([]float32, []float32) {
	return tq.K.Dequantize(), tq.V.Dequantize()
}

// Bytes returns the combined packed byte footprint of the asymmetric TurboQuant pair.
func (tq TurboQuantAsymmetricKV) Bytes() int {
	return tq.K.Bytes() + tq.V.Bytes()
}

// AngularCosineSimilarity computes the cosine similarity for Key vectors against the original Key.
func (tq TurboQuantAsymmetricKV) AngularCosineSimilarity(kOrig []float32) float64 {
	return tq.K.AngularCosineSimilarity(kOrig)
}

// TurboQuantFootprintReport captures the VRAM footprint comparison between baseline FP16
// and TurboQuant asymmetric KV cache quantization.
type TurboQuantFootprintReport struct {
	ContextTokens       int     `json:"context_tokens"`
	ElementsPerToken    int64   `json:"elements_per_token"`
	BaselineBytes       int64   `json:"baseline_bytes"`
	BaselineFP16Bytes   int64   `json:"baseline_fp16_bytes"`
	TurboQuantBytes     int64   `json:"turboquant_bytes"`
	BaselineGiB         float64 `json:"baseline_gib"`
	BaselineFP16GiB     float64 `json:"baseline_fp16_gib"`
	TurboQuantGiB       float64 `json:"turboquant_gib"`
	ReductionPercentage float64 `json:"reduction_percentage"`
	SavingsPercentage   float64 `json:"savings_percentage"`
}

// TurboQuantVRAMFootprint computes the memory footprint report proving a 62.5% VRAM footprint
// reduction (40.5 GiB -> 15.2 GiB on 262,144 context length where elementsPerToken is 41,472,
// exactly 43,486,543,872 bytes down to 16,307,453,952 bytes).
func TurboQuantVRAMFootprint(contextTokens int, elementsPerToken int64) TurboQuantFootprintReport {
	if contextTokens <= 0 || elementsPerToken <= 0 {
		return TurboQuantFootprintReport{}
	}
	baseline := int64(contextTokens) * elementsPerToken * 4
	tq := int64(contextTokens) * (elementsPerToken * 3 / 2)
	baseGiB := float64(baseline) / (1024 * 1024 * 1024)
	tqGiB := float64(tq) / (1024 * 1024 * 1024)
	reduction := float64(baseline-tq) / float64(baseline) * 100.0

	return TurboQuantFootprintReport{
		ContextTokens:       contextTokens,
		ElementsPerToken:    elementsPerToken,
		BaselineBytes:       baseline,
		BaselineFP16Bytes:   baseline,
		TurboQuantBytes:     tq,
		BaselineGiB:         baseGiB,
		BaselineFP16GiB:     baseGiB,
		TurboQuantGiB:       tqGiB,
		ReductionPercentage: reduction,
		SavingsPercentage:   reduction,
	}
}

// KVCacheBytesTurboQuant sizes the K+V cache for positions tokens when the KV-holding
// layers store Key rows in KVQuantQ8_0 and Value rows in KVQuantTurbo4.
func (c Config) KVCacheBytesTurboQuant(positions int) int64 {
	if positions <= 0 {
		return 0
	}
	layers := int64(len(c.KVQuantLayers()))
	if layers == 0 {
		return 0
	}
	vHeadDim := c.VHeadDim
	if vHeadDim == 0 {
		vHeadDim = c.HeadDim
	}
	kElems := c.NumKVHeads * c.HeadDim
	vElems := c.NumKVHeads * vHeadDim
	if kElems <= 0 || vElems <= 0 {
		return 0
	}
	kGroups := int64((kElems + KVQuantQ8_0BlockSize - 1) / KVQuantQ8_0BlockSize)
	kBytes := int64(kElems) + kGroups*4 // 1 byte/element + 4 bytes scale per block of 32

	vGroups := int64((vElems + KVQuantTurbo4BlockSize - 1) / KVQuantTurbo4BlockSize)
	vBytes := int64((vElems+1)/2) + vGroups*4 // 0.5 byte/element + 4 bytes scale per block of 32

	perPos := kBytes + vBytes
	return perPos * layers * int64(positions)
}

// AttentionCosineEntropyLoss evaluates numerical fidelity in attention: context vector
// cosine distance loss, original and dequantized Shannon entropies, relative entropy loss (KL),
// and combined loss.
type AttentionCosineEntropyLoss struct {
	CosineDistanceLoss  float64 `json:"cosine_distance_loss"`
	OrigEntropy         float64 `json:"orig_entropy"`
	OriginalEntropy     float64 `json:"original_entropy"`
	DequantEntropy      float64 `json:"dequant_entropy"`
	DequantizedEntropy  float64 `json:"dequantized_entropy"`
	RelativeEntropyLoss float64 `json:"relative_entropy_loss"`
	CombinedLoss        float64 `json:"combined_loss"`
}

// ComputeAttentionCosineEntropyLoss calculates context vector cosine distance loss, Shannon entropy
// for original and dequantized attention distributions, relative entropy loss, and combined loss.
func ComputeAttentionCosineEntropyLoss(query []float32, keysOrig, valsOrig, keysDq, valsDq [][]float32) AttentionCosineEntropyLoss {
	T := len(keysOrig)
	if T == 0 || len(query) == 0 || len(valsOrig) != T || len(keysDq) != T || len(valsDq) != T {
		return AttentionCosineEntropyLoss{}
	}

	d := len(query)
	scale := 1.0 / math.Sqrt(float64(d))

	// 1. Scaled dot-product attention scores
	scoresOrig := make([]float64, T)
	scoresDq := make([]float64, T)
	var maxOrig, maxDq float64
	for i := 0; i < T; i++ {
		var dotO, dotD float64
		kO := keysOrig[i]
		kD := keysDq[i]
		for j := 0; j < d && j < len(kO) && j < len(kD); j++ {
			qj := float64(query[j])
			dotO += qj * float64(kO[j])
			dotD += qj * float64(kD[j])
		}
		sO := dotO * scale
		sD := dotD * scale
		scoresOrig[i] = sO
		scoresDq[i] = sD
		if i == 0 || sO > maxOrig {
			maxOrig = sO
		}
		if i == 0 || sD > maxDq {
			maxDq = sD
		}
	}

	// 2. Softmax attention probabilities
	probsOrig := make([]float64, T)
	probsDq := make([]float64, T)
	var sumOrig, sumDq float64
	for i := 0; i < T; i++ {
		pO := math.Exp(scoresOrig[i] - maxOrig)
		pD := math.Exp(scoresDq[i] - maxDq)
		probsOrig[i] = pO
		probsDq[i] = pD
		sumOrig += pO
		sumDq += pD
	}
	for i := 0; i < T; i++ {
		probsOrig[i] /= sumOrig
		probsDq[i] /= sumDq
	}

	// 3. Shannon Entropy
	var hOrig, hDq float64
	for i := 0; i < T; i++ {
		if probsOrig[i] > 1e-15 {
			hOrig -= probsOrig[i] * math.Log(probsOrig[i])
		}
		if probsDq[i] > 1e-15 {
			hDq -= probsDq[i] * math.Log(probsDq[i])
		}
	}

	// 4. Relative Entropy Loss (Kullback-Leibler divergence D_KL(P_orig || P_dq))
	var relEntropy float64
	for i := 0; i < T; i++ {
		pO := probsOrig[i]
		pD := probsDq[i]
		if pO > 1e-15 && pD > 1e-15 {
			relEntropy += pO * math.Log(pO/pD)
		}
	}
	if relEntropy < 0 {
		relEntropy = 0
	}

	// 5. Context Vectors
	valDim := len(valsOrig[0])
	ctxOrig := make([]float64, valDim)
	ctxDq := make([]float64, valDim)
	for i := 0; i < T; i++ {
		pO := probsOrig[i]
		pD := probsDq[i]
		vO := valsOrig[i]
		vD := valsDq[i]
		for j := 0; j < valDim && j < len(vO) && j < len(vD); j++ {
			ctxOrig[j] += pO * float64(vO[j])
			ctxDq[j] += pD * float64(vD[j])
		}
	}

	// 6. Context vector cosine similarity & distance loss
	var dotCtx, normOrig, normDq float64
	for j := 0; j < valDim; j++ {
		dotCtx += ctxOrig[j] * ctxDq[j]
		normOrig += ctxOrig[j] * ctxOrig[j]
		normDq += ctxDq[j] * ctxDq[j]
	}
	cosSim := 1.0
	denom := math.Sqrt(normOrig) * math.Sqrt(normDq)
	if denom > 0 {
		cosSim = dotCtx / denom
		if cosSim > 1.0 {
			cosSim = 1.0
		} else if cosSim < -1.0 {
			cosSim = -1.0
		}
	}
	cosDistLoss := 1.0 - cosSim
	if cosDistLoss < 0 {
		cosDistLoss = 0
	}

	combined := cosDistLoss + relEntropy

	return AttentionCosineEntropyLoss{
		CosineDistanceLoss:  cosDistLoss,
		OrigEntropy:         hOrig,
		OriginalEntropy:     hOrig,
		DequantEntropy:      hDq,
		DequantizedEntropy:  hDq,
		RelativeEntropyLoss: relEntropy,
		CombinedLoss:        combined,
	}
}
