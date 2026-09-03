package model

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
func (q KVQuant4) ErrorBound() float32 {
	var worst float32
	for _, s := range q.Scale {
		if s > worst {
			worst = s
		}
	}
	return worst / 2
}

// Bytes is the packed footprint: the nibble payload plus the per-group f32 scale and min.
// The group metadata is real overhead and is counted here — at KVQuant4GroupSize=32 it
// adds 8 bytes per 32 elements (2 bits/element), so the honest rate is 6 bits/element,
// not 4. Reporting the bare nibble count would overstate the saving by 50%.
func (q KVQuant4) Bytes() int {
	return len(q.Codes) + 4*len(q.Scale) + 4*len(q.Min)
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
	var worst float32
	for _, s := range q.Scale {
		if s > worst {
			worst = s
		}
	}
	return worst / 2
}

// Bytes is the packed footprint: the 1-byte-per-element payload plus the per-group f32 scale
// and min. At KVQuant8GroupSize=32 it adds 8 bytes per 32 elements (2 bits/element), for an
// honest rate of 10 bits/element.
func (q KVQuant8) Bytes() int {
	return len(q.Codes) + 4*len(q.Scale) + 4*len(q.Min)
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
