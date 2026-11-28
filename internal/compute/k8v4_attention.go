// k8v4_attention.go — Group-32 asymmetric K8V4 KV codec, online-softmax decode
// attention host oracle, and traffic accounting.
//
// ADAPTED from magnitudedev/magnitude (Apache-2.0):
//   inference/engine/kernels/kernels/cuda/attention_decode_k8v4.cu:46@8d7dbcc1ba
//   inference/engine/kernels/kernels/cuda/attention_append_k8v4.cu:11@8d7dbcc1ba
//
// Clean-room Go host implementation (no upstream source bytes copied):
//   1. Keys: group-32 symmetric/affine int8 codes in [-127, 127] with IEEE-754
//      binary16 (scale, zero) coefficients (zero == 0 for symmetric ranges).
//   2. Values: group-32 affine 4-bit codes packed two nibbles per uint8 byte
//      (low nibble = even element, high nibble = odd element) with IEEE-754
//      binary16 (scale, zero) coefficients.
//   3. Attention: causal F32 online-softmax recurrence across prefix + newly
//      appended tokens, supporting GQA (heads % kvHeads == 0) and failing closed
//      on malformed shape, capacity, scale, or non-finite values.
//
// Status: [SW-VERIFIED] host oracle and CUDA source contract only; paired CUDA
// kernels in internal/compute/cuda/kv_fp8_attention_owner.cuh remain guarded by
// #ifdef __CUDACC__ and unwired from cuda_kernels.cu (token_time: unmeasured /
// HW-qualification-pending).

package compute

import (
	"errors"
	"fmt"
	"math"
)

const (
	// K8V4GroupSize is the contiguous element count sharing one f16 (scale, zero) pair.
	K8V4GroupSize = 32
)

var (
	// ErrK8V4InvalidShape refuses invalid head, group, prefix, or buffer geometry.
	ErrK8V4InvalidShape = errors.New("compute: k8v4: invalid group, head, or buffer geometry")
	// ErrK8V4NonFinite refuses NaN/Inf inputs, scores, or outputs.
	ErrK8V4NonFinite = errors.New("compute: k8v4: non-finite input, score, or output")
	// ErrK8V4InvalidScale refuses negative, NaN, or overflowed binary16 quantization coefficients.
	ErrK8V4InvalidScale = errors.New("compute: k8v4: invalid or non-finite f16 quantization coefficient")
	// ErrK8V4Capacity refuses appends exceeding allocated cache capacity.
	ErrK8V4Capacity = errors.New("compute: k8v4: cache capacity exceeded")
)

// K8V4Cache stores an asymmetric group-32 KV cache: int8 Keys + nibble-packed
// 4-bit Values with IEEE-754 binary16 (scale, zero) coefficients.
type K8V4Cache struct {
	Capacity      int
	Positions     int
	KVHeads       int
	HeadDim       int
	GroupsPerHead int

	// KCodes stores [Capacity][KVHeads][HeadDim] int8 codes in [-127, 127].
	KCodes []int8
	// KMeta stores [Capacity][KVHeads][GroupsPerHead][2] f16 bits (scale, zero).
	KMeta []uint16
	// VCodes stores [Capacity][KVHeads][HeadDim/2] nibble-packed 4-bit codes.
	VCodes []uint8
	// VMeta stores [Capacity][KVHeads][GroupsPerHead][2] f16 bits (scale, zero).
	VMeta []uint16
}

// K8V4TrafficReport records long-context per-token and total KV byte traffic
// for K8V4 relative to FP8+scale and FP16 baselines.
type K8V4TrafficReport struct {
	ContextTokens            int
	KVHeads                  int
	HeadDim                  int
	K8V4BytesPerToken        int64
	FP8Group32BytesPerToken  int64
	FP8PerHeadBytesPerToken  int64
	FP16BytesPerToken        int64
	K8V4TotalBytes           int64
	FP8Group32TotalBytes     int64
	FP8PerHeadTotalBytes     int64
	ReductionVsFP8Group32    float64
	ReductionVsFP8PerHead    float64
	PayloadReductionVsRawFP8 float64
}

func k8v4Float32ToFP16RN(f float32) uint16 {
	bits := math.Float32bits(f)
	sign := uint16((bits >> 31) & 0x1)
	exp := int((bits >> 23) & 0xFF)
	mant := bits & 0x7FFFFF

	if exp == 255 {
		if mant != 0 {
			return (sign << 15) | 0x7E00
		}
		return (sign << 15) | 0x7C00
	}

	exp16 := exp - 127 + 15
	if exp16 >= 31 {
		return (sign << 15) | 0x7C00
	}
	if exp16 <= 0 {
		if exp16 < -10 {
			return sign << 15
		}
		mantVal := mant | 0x800000
		shift := uint(14 - exp16)
		mant16 := uint16(mantVal >> shift)
		rem := mantVal & ((uint32(1) << shift) - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && (mant16&1 != 0)) {
			mant16++
		}
		if mant16 == 0x400 {
			return (sign << 15) | (1 << 10)
		}
		return (sign << 15) | mant16
	}

	rem := mant & 0x1FFF
	mant16 := uint16(mant >> 13)
	if rem > 0x1000 || (rem == 0x1000 && (mant16&1 != 0)) {
		mant16++
		if mant16 == 0x400 {
			exp16++
			mant16 = 0
		}
	}
	if exp16 >= 31 {
		return (sign << 15) | 0x7C00
	}
	return (sign << 15) | (uint16(exp16) << 10) | mant16
}

// NewK8V4Cache allocates an empty K8V4 cache for the given capacity and KV head geometry.
// HeadDim must be positive and divisible by K8V4GroupSize (32).
func NewK8V4Cache(capacity, kvHeads, headDim int) (*K8V4Cache, error) {
	if capacity <= 0 || kvHeads <= 0 || headDim <= 0 || headDim%K8V4GroupSize != 0 {
		return nil, fmt.Errorf("%w: capacity=%d kvHeads=%d headDim=%d",
			ErrK8V4InvalidShape, capacity, kvHeads, headDim)
	}
	maxInt := int(^uint(0) >> 1)
	if capacity > maxInt/kvHeads {
		return nil, fmt.Errorf("%w: row count overflow", ErrK8V4InvalidShape)
	}
	totalRows := capacity * kvHeads
	if totalRows > maxInt/headDim {
		return nil, fmt.Errorf("%w: element count overflow", ErrK8V4InvalidShape)
	}
	groupsPerHead := headDim / K8V4GroupSize
	if totalRows > maxInt/(groupsPerHead*2) {
		return nil, fmt.Errorf("%w: metadata count overflow", ErrK8V4InvalidShape)
	}
	totalElems := totalRows * headDim
	totalMeta := totalRows * groupsPerHead * 2
	return &K8V4Cache{
		Capacity:      capacity,
		KVHeads:       kvHeads,
		HeadDim:       headDim,
		GroupsPerHead: groupsPerHead,
		KCodes:        make([]int8, totalElems),
		KMeta:         make([]uint16, totalMeta),
		VCodes:        make([]uint8, totalElems/2),
		VMeta:         make([]uint16, totalMeta),
	}, nil
}

func (c *K8V4Cache) validateLayout() error {
	if c == nil || c.Capacity <= 0 || c.Positions < 0 || c.Positions > c.Capacity ||
		c.KVHeads <= 0 || c.HeadDim <= 0 || c.HeadDim%K8V4GroupSize != 0 ||
		c.GroupsPerHead != c.HeadDim/K8V4GroupSize {
		return ErrK8V4InvalidShape
	}
	maxInt := int(^uint(0) >> 1)
	if c.Capacity > maxInt/c.KVHeads {
		return ErrK8V4InvalidShape
	}
	totalRows := c.Capacity * c.KVHeads
	if totalRows > maxInt/c.HeadDim || totalRows > maxInt/(c.GroupsPerHead*2) {
		return ErrK8V4InvalidShape
	}
	totalElems := totalRows * c.HeadDim
	totalMeta := totalRows * c.GroupsPerHead * 2
	if len(c.KCodes) != totalElems || len(c.KMeta) != totalMeta ||
		len(c.VCodes) != totalElems/2 || len(c.VMeta) != totalMeta {
		return ErrK8V4InvalidShape
	}
	return nil
}

func quantizeK8Group(src []float32, dst []int8) (scaleH, zeroH uint16, err error) {
	mn, mx := src[0], src[0]
	for _, v := range src[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	zeroH = k8v4Float32ToFP16RN(float32(0.5 * (float64(mn) + float64(mx))))
	zeroF32 := fp16ToFloat32(zeroH)
	if !isFiniteF32(zeroF32) {
		return 0, 0, fmt.Errorf("%w: k zero overflow for [%g, %g]", ErrK8V4InvalidScale, mn, mx)
	}
	if mx > mn {
		span := math.Max(math.Abs(float64(mx)-float64(zeroF32)), math.Abs(float64(zeroF32)-float64(mn)))
		scaleH = k8v4Float32ToFP16RN(float32(span / 127.0))
		if (scaleH&0x7fff) == 0 && span > 0 {
			scaleH = 1
		}
		if 127.0*float64(fp16ToFloat32(scaleH)) < span && (scaleH&0x7fff) < 0x7bff {
			scaleH++
		}
	}
	scaleF32 := fp16ToFloat32(scaleH)
	if !isFiniteF32(scaleF32) || scaleF32 < 0 {
		return 0, 0, fmt.Errorf("%w: k scale overflow for [%g, %g]", ErrK8V4InvalidScale, mn, mx)
	}
	for i, v := range src {
		var q int
		if scaleF32 > 0 {
			q = int(math.Round((float64(v) - float64(zeroF32)) / float64(scaleF32)))
			if q < -127 {
				q = -127
			} else if q > 127 {
				q = 127
			}
		}
		dst[i] = int8(q)
	}
	return scaleH, zeroH, nil
}

func quantizeV4Group(src []float32, dstPacked []uint8) (scaleH, zeroH uint16, err error) {
	mn, mx := src[0], src[0]
	for _, v := range src[1:] {
		if v < mn {
			mn = v
		}
		if v > mx {
			mx = v
		}
	}
	zeroH = k8v4Float32ToFP16RN(mn)
	zeroF32 := fp16ToFloat32(zeroH)
	if zeroF32 > mn && isFiniteF32(zeroF32) {
		switch {
		case (zeroH & 0x7fff) == 0:
			zeroH = 0x8001
		case (zeroH & 0x8000) == 0:
			zeroH--
		case (zeroH & 0x7fff) < 0x7bff:
			zeroH++
		}
		zeroF32 = fp16ToFloat32(zeroH)
	}
	if !isFiniteF32(zeroF32) {
		return 0, 0, fmt.Errorf("%w: v zero overflow for [%g, %g]", ErrK8V4InvalidScale, mn, mx)
	}
	if mx > mn {
		span := float64(mx) - float64(zeroF32)
		scaleH = k8v4Float32ToFP16RN(float32(span / 15.0))
		if (scaleH&0x7fff) == 0 && span > 0 {
			scaleH = 1
		}
		if 15.0*float64(fp16ToFloat32(scaleH)) < span && (scaleH&0x7fff) < 0x7bff {
			scaleH++
		}
	}
	scaleF32 := fp16ToFloat32(scaleH)
	if !isFiniteF32(scaleF32) || scaleF32 < 0 {
		return 0, 0, fmt.Errorf("%w: v scale overflow for [%g, %g]", ErrK8V4InvalidScale, mn, mx)
	}
	for j := 0; j < K8V4GroupSize/2; j++ {
		var q0, q1 int
		if scaleF32 > 0 {
			q0 = int(math.Round((float64(src[2*j]) - float64(zeroF32)) / float64(scaleF32)))
			if q0 < 0 {
				q0 = 0
			} else if q0 > 15 {
				q0 = 15
			}
			q1 = int(math.Round((float64(src[2*j+1]) - float64(zeroF32)) / float64(scaleF32)))
			if q1 < 0 {
				q1 = 0
			} else if q1 > 15 {
				q1 = 15
			}
		}
		dstPacked[j] = uint8((q0 & 0x0f) | ((q1 & 0x0f) << 4))
	}
	return scaleH, zeroH, nil
}

// QuantizeAppendK8V4 quantizes `tokens` rows of F32 [tokens, KVHeads, HeadDim]
// Keys and Values into c using group-32 affine quantization with f16
// (scale, zero) coefficients. Atomic fail-closed: c is unmodified on any error.
func QuantizeAppendK8V4(c *K8V4Cache, k, v []float32, tokens int) error {
	if err := c.validateLayout(); err != nil {
		return err
	}
	if tokens <= 0 {
		return fmt.Errorf("%w: tokens=%d must be positive", ErrK8V4InvalidShape, tokens)
	}
	if tokens > c.Capacity-c.Positions {
		return fmt.Errorf("%w: positions=%d + tokens=%d > capacity=%d",
			ErrK8V4Capacity, c.Positions, tokens, c.Capacity)
	}
	rowWidth := c.KVHeads * c.HeadDim
	wantElems := tokens * rowWidth
	if len(k) != wantElems || len(v) != wantElems {
		return fmt.Errorf("%w: k=%d v=%d, want %d", ErrK8V4InvalidShape, len(k), len(v), wantElems)
	}
	for i := 0; i < wantElems; i++ {
		if !isFiniteF32(k[i]) {
			return fmt.Errorf("%w: k[%d]=%v", ErrK8V4NonFinite, i, k[i])
		}
		if !isFiniteF32(v[i]) {
			return fmt.Errorf("%w: v[%d]=%v", ErrK8V4NonFinite, i, v[i])
		}
	}

	rows := tokens * c.KVHeads
	stageKCodes := make([]int8, wantElems)
	stageKMeta := make([]uint16, rows*c.GroupsPerHead*2)
	stageVCodes := make([]uint8, wantElems/2)
	stageVMeta := make([]uint16, rows*c.GroupsPerHead*2)

	for row := 0; row < rows; row++ {
		for g := 0; g < c.GroupsPerHead; g++ {
			elemBase := row*c.HeadDim + g*K8V4GroupSize
			kScaleH, kZeroH, err := quantizeK8Group(k[elemBase:elemBase+K8V4GroupSize], stageKCodes[elemBase:elemBase+K8V4GroupSize])
			if err != nil {
				return err
			}
			vScaleH, vZeroH, err := quantizeV4Group(v[elemBase:elemBase+K8V4GroupSize], stageVCodes[elemBase/2:elemBase/2+K8V4GroupSize/2])
			if err != nil {
				return err
			}
			metaIdx := (row*c.GroupsPerHead + g) * 2
			stageKMeta[metaIdx] = kScaleH
			stageKMeta[metaIdx+1] = kZeroH
			stageVMeta[metaIdx] = vScaleH
			stageVMeta[metaIdx+1] = vZeroH
		}
	}

	startRow := c.Positions * c.KVHeads
	copy(c.KCodes[startRow*c.HeadDim:], stageKCodes)
	copy(c.KMeta[startRow*c.GroupsPerHead*2:], stageKMeta)
	copy(c.VCodes[startRow*(c.HeadDim/2):], stageVCodes)
	copy(c.VMeta[startRow*c.GroupsPerHead*2:], stageVMeta)
	c.Positions += tokens
	return nil
}

func decodeGroupMeta(meta []uint16, idx int) (scale, zero float32, err error) {
	scale = fp16ToFloat32(meta[idx])
	zero = fp16ToFloat32(meta[idx+1])
	if !isFiniteF32(scale) || scale < 0 || !isFiniteF32(zero) {
		return 0, 0, fmt.Errorf("%w: scale=%v zero=%v", ErrK8V4InvalidScale, scale, zero)
	}
	return scale, zero, nil
}

// DequantizeK8V4Row dequantizes the stored Key and Value vectors at (pos, kvHead)
// back into F32 slices of length HeadDim.
func DequantizeK8V4Row(c *K8V4Cache, pos, kvHead int) ([]float32, []float32, error) {
	if err := c.validateLayout(); err != nil {
		return nil, nil, err
	}
	if pos < 0 || pos >= c.Positions || kvHead < 0 || kvHead >= c.KVHeads {
		return nil, nil, fmt.Errorf("%w: pos=%d/%d kvHead=%d/%d",
			ErrK8V4InvalidShape, pos, c.Positions, kvHead, c.KVHeads)
	}

	row := pos*c.KVHeads + kvHead
	kOut := make([]float32, c.HeadDim)
	vOut := make([]float32, c.HeadDim)

	kCodeBase := row * c.HeadDim
	vCodeBase := row * (c.HeadDim / 2)
	metaBase := row * c.GroupsPerHead * 2

	for g := 0; g < c.GroupsPerHead; g++ {
		kScale, kZero, err := decodeGroupMeta(c.KMeta, metaBase+g*2)
		if err != nil {
			return nil, nil, err
		}
		vScale, vZero, err := decodeGroupMeta(c.VMeta, metaBase+g*2)
		if err != nil {
			return nil, nil, err
		}
		gOff := g * K8V4GroupSize
		for i := 0; i < K8V4GroupSize; i++ {
			kOut[gOff+i] = kZero + float32(c.KCodes[kCodeBase+gOff+i])*kScale
			packed := c.VCodes[vCodeBase+(gOff+i)/2]
			var nibble uint8
			if i&1 == 0 {
				nibble = packed & 0x0f
			} else {
				nibble = packed >> 4
			}
			vOut[gOff+i] = vZero + float32(nibble)*vScale
		}
	}
	return kOut, vOut, nil
}

// K8V4DecodeAttentionHost executes causal F32 online-softmax decode attention over
// appended prefix+new tokens in c, supporting GQA (heads % c.KVHeads == 0).
//
// q has shape [tokens, heads, HeadDim]. Query token t in [0, tokens) attends
// causally over KV positions [0, prefix+t], requiring prefix+tokens <= c.Positions.
func K8V4DecodeAttentionHost(c *K8V4Cache, q []float32, tokens, prefix, heads int, scale float32) ([]float32, error) {
	if err := c.validateLayout(); err != nil {
		return nil, err
	}
	maxInt := int(^uint(0) >> 1)
	if tokens <= 0 || prefix < 0 || tokens > maxInt-prefix || prefix+tokens > c.Positions ||
		heads <= 0 || heads%c.KVHeads != 0 || heads > maxInt/c.HeadDim || tokens > maxInt/(heads*c.HeadDim) ||
		!(scale > 0) || !isFiniteF32(scale) {
		return nil, fmt.Errorf("%w: tokens=%d prefix=%d cached=%d heads=%d kvHeads=%d headDim=%d scale=%v",
			ErrK8V4InvalidShape, tokens, prefix, c.Positions, heads, c.KVHeads, c.HeadDim, scale)
	}
	wantQ := tokens * heads * c.HeadDim
	if len(q) != wantQ {
		return nil, fmt.Errorf("%w: q len=%d, want %d", ErrK8V4InvalidShape, len(q), wantQ)
	}
	for i, qv := range q {
		if !isFiniteF32(qv) {
			return nil, fmt.Errorf("%w: q[%d]=%v", ErrK8V4NonFinite, i, qv)
		}
	}

	out := make([]float32, wantQ)
	gqaGroup := heads / c.KVHeads
	acc := make([]float32, c.HeadDim)

	for t := 0; t < tokens; t++ {
		count := prefix + t + 1
		for h := 0; h < heads; h++ {
			kh := h / gqaGroup
			qBase := (t*heads + h) * c.HeadDim
			qRow := q[qBase : qBase+c.HeadDim]

			for d := range acc {
				acc[d] = 0
			}
			var maximum float32
			var denominator float32

			for pos := 0; pos < count; pos++ {
				row := pos*c.KVHeads + kh
				kCodeBase := row * c.HeadDim
				vCodeBase := row * (c.HeadDim / 2)
				metaBase := row * c.GroupsPerHead * 2

				var dot float32
				for g := 0; g < c.GroupsPerHead; g++ {
					kScale, kZero, err := decodeGroupMeta(c.KMeta, metaBase+g*2)
					if err != nil {
						return nil, err
					}
					gOff := g * K8V4GroupSize
					for i := 0; i < K8V4GroupSize; i++ {
						kVal := kZero + float32(c.KCodes[kCodeBase+gOff+i])*kScale
						dot += qRow[gOff+i] * kVal
					}
				}
				score := dot * scale
				if !isFiniteF32(score) {
					return nil, fmt.Errorf("%w: score at token=%d head=%d pos=%d", ErrK8V4NonFinite, t, h, pos)
				}

				var correction, probability float32
				if pos == 0 {
					maximum = score
					denominator = 1.0
				} else {
					nextMax := maximum
					if score > nextMax {
						nextMax = score
					}
					correction = float32(math.Exp(float64(maximum - nextMax)))
					probability = float32(math.Exp(float64(score - nextMax)))
					denominator = denominator*correction + probability
					maximum = nextMax
				}

				for g := 0; g < c.GroupsPerHead; g++ {
					vScale, vZero, err := decodeGroupMeta(c.VMeta, metaBase+g*2)
					if err != nil {
						return nil, err
					}
					gOff := g * K8V4GroupSize
					for i := 0; i < K8V4GroupSize; i++ {
						packed := c.VCodes[vCodeBase+(gOff+i)/2]
						var nibble uint8
						if i&1 == 0 {
							nibble = packed & 0x0f
						} else {
							nibble = packed >> 4
						}
						vVal := vZero + float32(nibble)*vScale
						if pos == 0 {
							acc[gOff+i] = vVal
						} else {
							acc[gOff+i] = acc[gOff+i]*correction + probability*vVal
						}
					}
				}
			}

			if !(denominator > 0) || !isFiniteF32(denominator) {
				return nil, fmt.Errorf("%w: invalid softmax denominator %v", ErrK8V4NonFinite, denominator)
			}
			invDenom := float32(1.0) / denominator
			for d := 0; d < c.HeadDim; d++ {
				val := acc[d] * invDenom
				if !isFiniteF32(val) {
					return nil, fmt.Errorf("%w: output[%d]=%v", ErrK8V4NonFinite, qBase+d, val)
				}
				out[qBase+d] = val
			}
		}
	}
	return out, nil
}

// ExactAttentionF32Host computes unquantized F32 causal attention over k, v of
// shape [prefix+tokens, kvHeads, headDim] and q of shape [tokens, heads, headDim].
func ExactAttentionF32Host(q, k, v []float32, tokens, prefix, heads, kvHeads, headDim int, scale float32) ([]float32, error) {
	maxInt := int(^uint(0) >> 1)
	if tokens <= 0 || prefix < 0 || heads <= 0 || kvHeads <= 0 || headDim <= 0 ||
		heads%kvHeads != 0 || tokens > maxInt-prefix ||
		heads > maxInt/headDim || tokens > maxInt/(heads*headDim) ||
		(prefix+tokens) > maxInt/(kvHeads*headDim) || !(scale > 0) || !isFiniteF32(scale) {
		return nil, ErrK8V4InvalidShape
	}
	wantQ := tokens * heads * headDim
	wantKV := (prefix + tokens) * kvHeads * headDim
	if len(q) != wantQ || len(k) != wantKV || len(v) != wantKV {
		return nil, ErrK8V4InvalidShape
	}
	for i, x := range q {
		if !isFiniteF32(x) {
			return nil, fmt.Errorf("%w: q[%d]=%v", ErrK8V4NonFinite, i, x)
		}
	}
	for i := 0; i < wantKV; i++ {
		if !isFiniteF32(k[i]) || !isFiniteF32(v[i]) {
			return nil, fmt.Errorf("%w: kv[%d]", ErrK8V4NonFinite, i)
		}
	}

	out := make([]float32, wantQ)
	gqaGroup := heads / kvHeads
	scores := make([]float64, prefix+tokens)

	for t := 0; t < tokens; t++ {
		count := prefix + t + 1
		for h := 0; h < heads; h++ {
			kh := h / gqaGroup
			qBase := (t*heads + h) * headDim
			maxScore := math.Inf(-1)
			for pos := 0; pos < count; pos++ {
				kBase := (pos*kvHeads + kh) * headDim
				var dot float64
				for d := 0; d < headDim; d++ {
					dot += float64(q[qBase+d]) * float64(k[kBase+d])
				}
				sc := dot * float64(scale)
				scores[pos] = sc
				if sc > maxScore {
					maxScore = sc
				}
			}
			var denom float64
			for pos := 0; pos < count; pos++ {
				w := math.Exp(scores[pos] - maxScore)
				scores[pos] = w
				denom += w
			}
			for d := 0; d < headDim; d++ {
				var sum float64
				for pos := 0; pos < count; pos++ {
					vBase := (pos*kvHeads + kh) * headDim
					sum += scores[pos] * float64(v[vBase+d])
				}
				out[qBase+d] = float32(sum / denom)
			}
		}
	}
	return out, nil
}

// K8V4KVBytesPerToken returns the exact per-token, per-layer byte footprint of
// group-32 K8V4 storage: 1 B/elem int8 K + 4 B/group f16 (scale, zero), and
// 0.5 B/elem packed int4 V + 4 B/group f16 (scale, zero).
func K8V4KVBytesPerToken(kvHeads, headDim int) (int64, error) {
	if kvHeads <= 0 || headDim <= 0 || headDim%K8V4GroupSize != 0 {
		return 0, fmt.Errorf("%w: kvHeads=%d headDim=%d", ErrK8V4InvalidShape, kvHeads, headDim)
	}
	groups := int64(headDim / K8V4GroupSize)
	kPerHead := int64(headDim) + groups*4   // int8 codes + f16 (scale, zero)
	vPerHead := int64(headDim)/2 + groups*4 // packed 4-bit codes + f16 (scale, zero)
	return int64(kvHeads) * (kPerHead + vPerHead), nil
}

// CompareK8V4VsFP8Traffic computes exact long-context byte traffic for K8V4
// against group-32 FP8+f32-scale (72 B/32 elems), per-head FP8+f32-scale
// (kv_fp8), and unquantized FP16.
func CompareK8V4VsFP8Traffic(contextTokens, kvHeads, headDim int) (K8V4TrafficReport, error) {
	if contextTokens <= 0 {
		return K8V4TrafficReport{}, fmt.Errorf("%w: contextTokens=%d must be positive", ErrK8V4InvalidShape, contextTokens)
	}
	k8v4PerTok, err := K8V4KVBytesPerToken(kvHeads, headDim)
	if err != nil {
		return K8V4TrafficReport{}, err
	}
	groups := int64(headDim / K8V4GroupSize)
	fp8Group32PerTok := int64(kvHeads) * 2 * (int64(headDim) + groups*4)
	fp8PerHeadPerTok := int64(kvHeads) * 2 * (int64(headDim) + 4)
	fp16PerTok := int64(kvHeads) * 2 * int64(headDim) * 2

	ctx := int64(contextTokens)
	k8v4Total := ctx * k8v4PerTok
	fp8Group32Total := ctx * fp8Group32PerTok
	fp8PerHeadTotal := ctx * fp8PerHeadPerTok

	return K8V4TrafficReport{
		ContextTokens:            contextTokens,
		KVHeads:                  kvHeads,
		HeadDim:                  headDim,
		K8V4BytesPerToken:        k8v4PerTok,
		FP8Group32BytesPerToken:  fp8Group32PerTok,
		FP8PerHeadBytesPerToken:  fp8PerHeadPerTok,
		FP16BytesPerToken:        fp16PerTok,
		K8V4TotalBytes:           k8v4Total,
		FP8Group32TotalBytes:     fp8Group32Total,
		FP8PerHeadTotalBytes:     fp8PerHeadTotal,
		ReductionVsFP8Group32:    1.0 - float64(k8v4PerTok)/float64(fp8Group32PerTok),
		ReductionVsFP8PerHead:    1.0 - float64(k8v4PerTok)/float64(fp8PerHeadPerTok),
		PayloadReductionVsRawFP8: 1.0 - float64(int64(headDim)+int64(headDim)/2)/float64(2*int64(headDim)),
	}, nil
}
