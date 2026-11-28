package compute

import (
	"fmt"
	"math"
	"unsafe"
)

// quant_i3gs.go — I3_GS (INT3 gs64 bit-plane) expert quant: a 3.0-bit/weight tier.
//
// I3_GS is a 3.0-bit weight format in which each group of 64 values is stored as a
// 16-byte low plane (the 2 low bits of each code, 4-way byte-interleaved) plus an
// 8-byte high plane (the top bit of each code), with one f32 scale per group; code c in
// [0,7] maps to value c-4 in [-4,3]. It is a 3.0-bit/weight expert tier (3.5 bits/weight
// incl. scale) that sits between the existing Q3_K (3.44 bits/weight, 256-block) and
// IQ2_XXS (2.06 bits/weight) formats, with ~22% fewer expert bytes than INT4 gs64.
//
// The layout and arithmetic are adopted from benmaster82/picchio (MIT, pinned at
// per-group-64 plane layout, the scalar decode, and the quantize mirror (per-group
// amax/3.5 scale, 1e-8 floor, round-to-nearest code clamped to [-4,3]). This is a
// clean-room Go port: no C bytes are vendored; the layout and arithmetic are adopted
// with attribution preserved here.
//
// The scalar path is the always-correct reference every build uses. The AVX2 SIMD decode
// fast path (the plane layout is chosen for it: 4-way byte interleave to low-2-bit codes,
// pshufb bit-expand to the 0/4 high plane, int8 dequant to [-4,3], F32 FMA) is a tracked
// follow-up (the "where built" clause); until it lands, every build takes the scalar path.
// No device backend serves I3_GS yet, so every one must decline cleanly (the same posture
// Q3_K takes today).

// i3gsGroup is the I3_GS group width: 64 values share one f32 scale and one 24-byte plane.
const i3gsGroup = 64

// i3gsGroupBytes is the byte length of one I3_GS group's plane storage:
// 16 (low plane) + 8 (high plane) = 24 bytes = 3 bits per value.
const i3gsGroupBytes = 24

// i3gsLowBytes and i3gsHighBytes are the plane widths within a group.
const (
	i3gsLowBytes  = 16 // 64 values * 2 bits = 128 bits = 16 bytes
	i3gsHighBytes = 8  // 64 values * 1 bit  =  64 bits =  8 bytes
)

// NewI3GS wraps raw I3_GS plane bytes + per-group f32 scales as a host Tensor of dtype
// I3_GS. raw is row-major: row o at raw[o*nGroups*24:], group g at +g*24 (16 B low plane +
// 8 B high plane). shape is [out, in] with in a multiple of 64; len(raw) must be
// out*(in/64)*24 and len(scale) must be out*(in/64). The bytes ride in the HostBuffer.I8()
// view (one int8 per byte, value-preserving two's-complement reinterpret) — the same
// layout cpuBackend.MatMul reads.
func NewI3GS(be Backend, shape []int, raw []byte, scale []float32) Tensor {
	if len(shape) != 2 || shape[0] <= 0 || shape[1] <= 0 || shape[1]%i3gsGroup != 0 {
		panic("compute: invalid I3_GS shape")
	}
	// Bound each multiplication before checking storage: wrapped products can otherwise
	// admit an empty buffer for a large, nonempty tensor.
	groups := shape[1] / i3gsGroup
	maxInt := int(^uint(0) >> 1)
	if groups > maxInt/i3gsGroupBytes || shape[0] > maxInt/(groups*i3gsGroupBytes) ||
		len(raw) != shape[0]*groups*i3gsGroupBytes || len(scale) != shape[0]*groups {
		panic("compute: invalid I3_GS byte length")
	}
	var i8 []int8
	if len(raw) > 0 {
		i8 = unsafe.Slice((*int8)(unsafe.Pointer(&raw[0])), len(raw))
	}
	q := &QuantSpec{Block: i3gsGroup, Axis: 2, Bits: 3, Symmetric: true, Scale: scale}
	return makeTensor(be, I3_GS, RowMajor, append([]int(nil), shape...), q, &hostBuf{i8: i8})
}

// i3gsDequantGroup writes the n (<= 64) values of one I3_GS group into dst (len >= n),
// decoding the two planes and multiplying each by the group scale. This is the scalar
// oracle: the reference every other path is checked against. A partial tail (n < 64) is
// handled by the scalar path.
func i3gsDequantGroup(dst []float32, low, high []byte, scale float32, n int) {
	if len(low) < i3gsLowBytes || len(high) < i3gsHighBytes {
		panic("compute: short I3_GS plane")
	}
	if n < 0 || n > i3gsGroup || len(dst) < n {
		panic("compute: bad I3_GS group count")
	}
	for j := 0; j < n; j++ {
		low2 := int(low[j>>2]>>uint((j&3)*2)) & 3
		high1 := int(high[j>>3]>>uint(j&7)) & 1
		code := low2 | (high1 << 2)
		dst[j] = float32(code-4) * scale
	}
}

// i3gsRowDot computes one output element y[o] = dot(weight row, x) over an I3_GS weight row:
// dequantize per group and FMA-accumulate in f32 with the scale folded per group. raw is the
// [in/64 * 24]-byte weight row; scale is the [in/64] per-group scales; x is the [in] activation.
func i3gsRowDot(raw []byte, scale []float32, x []float32) float32 {
	nGroups := len(raw) / i3gsGroupBytes
	if nGroups*i3gsGroupBytes != len(raw) {
		panic("compute: misaligned I3_GS raw weight row")
	}
	if len(scale) != nGroups || len(x) < nGroups*i3gsGroup {
		panic("compute: short I3_GS scale or input")
	}
	var sum float32
	for g := 0; g < nGroups; g++ {
		grp := raw[g*i3gsGroupBytes : (g+1)*i3gsGroupBytes]
		low := grp[:i3gsLowBytes]
		high := grp[i3gsLowBytes:]
		xs := x[g*i3gsGroup : (g+1)*i3gsGroup]
		var groupSum float32
		for j := 0; j < i3gsGroup; j++ {
			low2 := int(low[j>>2]>>uint((j&3)*2)) & 3
			high1 := int(high[j>>3]>>uint(j&7)) & 1
			code := low2 | (high1 << 2)
			groupSum += float32(code-4) * xs[j]
		}
		sum += groupSum * scale[g]
	}
	return sum
}

// QuantizeRowsI3GS quantizes a row-major [out, in] f32 weight matrix into I3_GS plane bytes
// + per-group f32 scales. Per-group scale = amax/3.5 (1e-8 floor); each value maps to a
// round-to-nearest code clamped to [-4, 3] (the mirror of the upstream Python converter).
// in must be a multiple of 64.
func QuantizeRowsI3GS(w []float32, out, in int) (raw []byte, scale []float32, err error) {
	if out <= 0 || in <= 0 || in%i3gsGroup != 0 {
		return nil, nil, fmt.Errorf("compute: invalid I3_GS quantize shape out=%d in=%d", out, in)
	}
	if len(w) != out*in {
		return nil, nil, fmt.Errorf("compute: I3_GS quantize weight length %d != out*in %d", len(w), out*in)
	}
	nGroups := in / i3gsGroup
	raw = make([]byte, out*nGroups*i3gsGroupBytes)
	scale = make([]float32, out*nGroups)
	for o := 0; o < out; o++ {
		row := w[o*in : (o+1)*in]
		rowRaw := raw[o*nGroups*i3gsGroupBytes : (o+1)*nGroups*i3gsGroupBytes]
		rowScale := scale[o*nGroups : (o+1)*nGroups]
		for g := 0; g < nGroups; g++ {
			grp := row[g*i3gsGroup : (g+1)*i3gsGroup]
			// Per-group amax.
			var amax float32
			for _, v := range grp {
				if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
					return nil, nil, fmt.Errorf("compute: non-finite I3_GS weight at row %d group %d", o, g)
				}
				if a := float32(math.Abs(float64(v))); a > amax {
					amax = a
				}
			}
			// scale = amax/3.5, 1e-8 floor.
			sc := amax / 3.5
			if sc < 1e-8 {
				sc = 1e-8
			}
			rowScale[g] = sc
			inv := 1.0 / sc
			grpRaw := rowRaw[g*i3gsGroupBytes : (g+1)*i3gsGroupBytes]
			low := grpRaw[:i3gsLowBytes]
			high := grpRaw[i3gsLowBytes:]
			for j := 0; j < i3gsGroup; j++ {
				c := int(math.Round(float64(grp[j] * inv)))
				if c < -4 {
					c = -4
				} else if c > 3 {
					c = 3
				}
				code := uint(c + 4) // [0,7]
				low[j>>2] |= byte(code&3) << uint((j&3)*2)
				high[j>>3] |= byte(code>>2) << uint(j&7)
			}
		}
	}
	return raw, scale, nil
}
