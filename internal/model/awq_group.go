package model

// awq_group.go — REAL AutoAWQ group-wise asymmetric 4-bit weight support.
//
// The per-channel symmetric path in awq.go (awqTensor: one scale per output row,
// fixed zero-point 8, plain nibble packing) is a teaching stub. Genuine AutoAWQ
// checkpoints for Llama-2/3 and Qwen2 store weights in a GROUP-WISE ASYMMETRIC
// layout, which this file implements end to end:
//
//	weight[o,i] = (code[o,i] - zero[o,g]) * scale[o,g],   g = i / groupSize
//
//   - code[o,i] ∈ [0,15] is the unsigned 4-bit value for output channel o, input i
//   - scale[o,g] is a float32 per (output channel, input group)
//   - zero[o,g] ∈ [0,15] is the per-group asymmetric zero-point (NOT fixed at 8)
//
// This is the storage+dequant contract AutoAWQ's GEMM kernel consumes. The
// "activation-aware" half of AWQ is an OFFLINE calibration step that chooses a
// per-input-channel scaling to protect salient channels before round-to-nearest
// quantization; awqGroupQuantize implements that scaling (awqActScale) so the
// in-Go quantizer reproduces AWQ's accuracy rather than plain RTN's.
//
// On-disk interop: AutoAWQ packs 8 codes per int32 in the reorder
// [0,2,4,6,1,3,5,7] (awqPackOrder). awqUnpackI32Row inverts that so LoadAWQ can
// read a real qweight/qzeros/scales triple. Byte-equality against a stored HF
// fixture needs an absent fixture and is left to an oracle fixture; what IS
// witnessed on-host is (1) the int32 pack/unpack round-trips bijectively and
// (2) dequant+GEMM agree with an FP32 baseline to cos >= 0.995 with matching
// argmax (awq_group_test.go).

import "math"

// awqPackOrder is AutoAWQ's nibble-reorder map: when packing 8 logical output
// columns into one int32, the column at logical slot p is written to nibble
// position awqPackOrder[p] (bit shift 4*awqPackOrder[p]). Unpacking inverts it.
var awqPackOrder = [8]int{0, 2, 4, 6, 1, 3, 5, 7}

// awqGroupTensor is a resident AutoAWQ group-wise asymmetric 4-bit weight matrix
// [out, in]. Codes are stored output-major and nibble-packed [out][in/2]; the low
// nibble of byte b holds the even input index, the high nibble the odd one.
// scales/zeros are laid out [out][nGroups] (row o, group g at index o*nGroups+g).
type awqGroupTensor struct {
	out, in   int
	groupSize int
	nGroups   int       // in / groupSize
	codes     []byte    // [out][in/2] nibble-packed 4-bit codes
	scales    []float32 // [out*nGroups] per (output channel, input group)
	zeros     []uint8   // [out*nGroups] per-group zero-point, each ∈ [0,15]
}

// awqGroupRowBytes is the packed byte length of one output row (in/2).
func (qt *awqGroupTensor) awqGroupRowBytes() int { return qt.in / 2 }

// awqGroupDequantRow writes the dequantized float32 weights of output channel o
// into dst (len >= in): dst[i] = (code[o,i] - zero[o,g]) * scale[o,g].
func awqGroupDequantRow(dst []float32, qt *awqGroupTensor, o int) {
	rowBytes := qt.awqGroupRowBytes()
	row := qt.codes[o*rowBytes : o*rowBytes+rowBytes]
	sc := qt.scales[o*qt.nGroups : o*qt.nGroups+qt.nGroups]
	zr := qt.zeros[o*qt.nGroups : o*qt.nGroups+qt.nGroups]

	g := -1
	var scale, zero float32
	for i := 0; i < qt.in; i++ {
		if ng := i / qt.groupSize; ng != g {
			g = ng
			scale = sc[g]
			zero = float32(zr[g])
		}
		b := row[i>>1]
		var code uint8
		if i&1 == 0 {
			code = b & 0x0f
		} else {
			code = b >> 4
		}
		dst[i] = (float32(code) - zero) * scale
	}
}

// awqGroupMatRows is the AWQ group-wise GEMV: y[o] = dot(weight row o, x). Each
// row is dequantized on the fly and dotted against x. Row-parallel like the other
// matmul kernels.
func awqGroupMatRows(qt *awqGroupTensor, x []float32) []float32 {
	y := make([]float32, qt.out)
	awqGroupMatRowsInto(qt, x, y)
	return y
}

// awqGroupMatRowsInto computes y = W·x writing into a caller-provided buffer.
func awqGroupMatRowsInto(qt *awqGroupTensor, x, y []float32) {
	y = y[:qt.out]
	parForRange(qt.out, qt.out*qt.in, func(lo, hi int) { awqGroupMatRange(qt, x, y, lo, hi) })
}

// awqGroupMatRange computes y[lo:hi] by dequantizing each row and dotting with x.
func awqGroupMatRange(qt *awqGroupTensor, x, y []float32, lo, hi int) {
	buf := make([]float32, qt.in)
	for o := lo; o < hi; o++ {
		awqGroupDequantRow(buf, qt, o)
		var acc float32
		for i := 0; i < qt.in; i++ {
			acc += buf[i] * x[i]
		}
		y[o] = acc
	}
}

// awqGroupGemm is the AWQ group-wise PREFILL GEMM: Y[t*out+o] = dot(row o,
// X[t*in:(t+1)*in]) for all t in [0,P). Each weight row is dequantized once and
// reused across all P activation rows, amortizing the dequant cost.
func awqGroupGemm(qt *awqGroupTensor, X []float32, P int) []float32 {
	Y := make([]float32, P*qt.out)
	awqGroupGemmInto(qt, X, P, Y)
	return Y
}

// awqGroupGemmInto is awqGroupGemm writing into a caller-provided Y buffer.
func awqGroupGemmInto(qt *awqGroupTensor, X []float32, P int, Y []float32) {
	Y = Y[:P*qt.out]
	parForRange(qt.out, qt.out*qt.in*P, func(lo, hi int) { awqGroupGemmRange(qt, X, P, Y, lo, hi) })
}

// awqGroupGemmRange computes Y[t*out+o] for o in [lo,hi), all t in [0,P).
func awqGroupGemmRange(qt *awqGroupTensor, X []float32, P int, Y []float32, lo, hi int) {
	buf := make([]float32, qt.in)
	for o := lo; o < hi; o++ {
		awqGroupDequantRow(buf, qt, o)
		for t := 0; t < P; t++ {
			xs := X[t*qt.in : t*qt.in+qt.in]
			var acc float32
			for i := 0; i < qt.in; i++ {
				acc += buf[i] * xs[i]
			}
			Y[t*qt.out+o] = acc
		}
	}
}

// ---- Model accessors --------------------------------------------------------

// awqGroup returns the resident AutoAWQ group tensor for a name.
func (m *Model) awqGroup(name string) *awqGroupTensor {
	if m.awqg == nil {
		panic("model: no AWQ group tensors loaded (call Model.LoadAWQ on a real AutoAWQ export)")
	}
	qt, ok := m.awqg[name]
	if !ok {
		panic("model: AWQ group tensor not found: " + name)
	}
	return qt
}

// AWQGroupCount returns how many tensors hold a real AutoAWQ group copy.
func (m *Model) AWQGroupCount() int { return len(m.awqg) }

// AWQGroupShape returns the (out, in, groupSize) of a group tensor, or zeros if absent.
func (m *Model) AWQGroupShape(name string) (out, in, groupSize int) {
	if qt := m.awqg[name]; qt != nil {
		return qt.out, qt.in, qt.groupSize
	}
	return 0, 0, 0
}

// ---- int32 pack / unpack (AutoAWQ disk interop) -----------------------------

// awqUnpackI32Row unpacks one input row of an AutoAWQ qweight/qzeros tensor.
// packed holds out/8 int32 values for this row; dst (len >= out) receives the
// 4-bit code for each output channel, undoing the awqPackOrder nibble reorder.
func awqUnpackI32Row(dst []uint8, packed []uint32, out int) {
	for c := 0; c < out/8; c++ {
		v := packed[c]
		base := c * 8
		for p := 0; p < 8; p++ {
			code := uint8((v >> (4 * awqPackOrder[p])) & 0xf)
			dst[base+awqPackOrder[p]] = code
		}
	}
}

// awqPackI32Row is the inverse of awqUnpackI32Row: it packs out 4-bit codes
// (codes[0:out]) into out/8 int32 values using the awqPackOrder reorder. Used by
// the round-trip witness; AutoAWQ produces this layout offline.
func awqPackI32Row(codes []uint8, out int) []uint32 {
	packed := make([]uint32, out/8)
	for c := 0; c < out/8; c++ {
		base := c * 8
		var v uint32
		for p := 0; p < 8; p++ {
			v |= (uint32(codes[base+awqPackOrder[p]]) & 0xf) << (4 * awqPackOrder[p])
		}
		packed[c] = v
	}
	return packed
}

// ---- in-Go AWQ quantizer (activation-aware) ---------------------------------

// awqActScale computes AWQ's per-input-channel protection scale from a calibration
// salience vector calib[i] (typically the mean |activation| of input channel i)
// and exponent alpha: s[i] = (calib[i]/geomean(calib))^alpha. alpha==0 or calib==nil
// yields the all-ones vector (plain round-to-nearest, no activation awareness).
// The geometric-mean normalization keeps overall weight magnitude stable, exactly
// as AutoAWQ's search does before it sweeps alpha.
func awqActScale(calib []float32, in int, alpha float32) []float32 {
	s := make([]float32, in)
	for i := range s {
		s[i] = 1
	}
	if calib == nil || alpha == 0 {
		return s
	}
	var logSum float64
	for i := 0; i < in; i++ {
		c := float64(calib[i])
		if c < 1e-12 {
			c = 1e-12
		}
		logSum += math.Log(c)
	}
	geomean := math.Exp(logSum / float64(in))
	for i := 0; i < in; i++ {
		c := float64(calib[i])
		if c < 1e-12 {
			c = 1e-12
		}
		s[i] = float32(math.Pow(c/geomean, float64(alpha)))
	}
	return s
}

// awqDefaultAlphas is the scaling-exponent grid AWQ's calibration sweeps.
var awqDefaultAlphas = []float32{0, 0.25, 0.5, 0.75, 1.0}

// awqGroupQuantizeSearch is AWQ's activation-aware calibration: it grid-searches
// the per-input scaling exponent alpha that minimizes the CALIBRATION output error
// ||W·xc - dequant(W∘s)·(xc⊘s)|| over a batch of nCalib activation rows xcal
// ([nCalib*in], row-major), then returns the best group tensor, its per-input
// scaleVec, and the chosen alpha. calib is the per-input salience (mean |activation|
// over the batch). This is exactly the offline per-layer search real AutoAWQ runs;
// alpha=0 (plain round-to-nearest) is always in the grid, so the result is never
// worse than RTN. Pass alphas=nil for awqDefaultAlphas.
func awqGroupQuantizeSearch(w []float32, out, in, groupSize int, calib, xcal []float32, nCalib int, alphas []float32) (*awqGroupTensor, []float32, float32) {
	if alphas == nil {
		alphas = awqDefaultAlphas
	}
	// FP32 reference output for each calibration row.
	ref := make([]float32, nCalib*out)
	for r := 0; r < nCalib; r++ {
		xr := xcal[r*in : r*in+in]
		for o := 0; o < out; o++ {
			wr := w[o*in : o*in+in]
			var acc float32
			for i := 0; i < in; i++ {
				acc += wr[i] * xr[i]
			}
			ref[r*out+o] = acc
		}
	}

	var bestT *awqGroupTensor
	var bestSV []float32
	bestA := float32(0)
	bestMSE := math.MaxFloat64
	buf := make([]float32, in)
	xd := make([]float32, in)
	for _, a := range alphas {
		qt, sv := awqGroupQuantize(w, out, in, groupSize, calib, a)
		var mse float64
		for r := 0; r < nCalib; r++ {
			xr := xcal[r*in : r*in+in]
			for i := 0; i < in; i++ {
				xd[i] = xr[i] / sv[i]
			}
			for o := 0; o < out; o++ {
				awqGroupDequantRow(buf, qt, o)
				var acc float32
				for i := 0; i < in; i++ {
					acc += buf[i] * xd[i]
				}
				d := float64(acc - ref[r*out+o])
				mse += d * d
			}
		}
		if mse < bestMSE {
			bestMSE, bestA, bestT, bestSV = mse, a, qt, sv
		}
	}
	return bestT, bestSV, bestA
}

// awqGroupQuantize quantizes a row-major FP32 weight matrix w [out, in]
// (w[o*in+i]) into the group-wise asymmetric AutoAWQ format. calib is the
// per-input-channel salience (mean |activation|; nil for plain RTN) and alpha is
// the AWQ scaling exponent. It returns the resident tensor and the per-input
// scaleVec the runtime must divide activations by (AWQ folds 1/scaleVec into the
// upstream op so the GEMV stays a plain dequant-matmul): W·x == (W∘scaleVec)·(x⊘scaleVec).
func awqGroupQuantize(w []float32, out, in, groupSize int, calib []float32, alpha float32) (*awqGroupTensor, []float32) {
	if in%2 != 0 {
		panic("model: AWQ input dimension must be even")
	}
	if groupSize <= 0 || in%groupSize != 0 {
		panic("model: AWQ in must be a positive multiple of groupSize")
	}
	nGroups := in / groupSize
	scaleVec := awqActScale(calib, in, alpha)

	qt := &awqGroupTensor{
		out:       out,
		in:        in,
		groupSize: groupSize,
		nGroups:   nGroups,
		codes:     make([]byte, out*(in/2)),
		scales:    make([]float32, out*nGroups),
		zeros:     make([]uint8, out*nGroups),
	}
	rowBytes := in / 2

	for o := 0; o < out; o++ {
		wrow := w[o*in : o*in+in]
		crow := qt.codes[o*rowBytes : o*rowBytes+rowBytes]
		for g := 0; g < nGroups; g++ {
			lo := g * groupSize
			hi := lo + groupSize

			// Find the group's value range AFTER the activation-aware scaling.
			minV := float32(math.MaxFloat32)
			maxV := float32(-math.MaxFloat32)
			for i := lo; i < hi; i++ {
				v := wrow[i] * scaleVec[i]
				if v < minV {
					minV = v
				}
				if v > maxV {
					maxV = v
				}
			}

			// Asymmetric 4-bit: 16 levels span [minV, maxV]. Guard a degenerate
			// (all-equal) group so scale stays finite.
			scale := (maxV - minV) / 15
			if scale <= 0 {
				scale = 1e-8
			}
			zeroF := -minV / scale
			zero := int(math.Round(float64(zeroF)))
			if zero < 0 {
				zero = 0
			}
			if zero > 15 {
				zero = 15
			}

			qt.scales[o*nGroups+g] = scale
			qt.zeros[o*nGroups+g] = uint8(zero)

			for i := lo; i < hi; i++ {
				v := wrow[i] * scaleVec[i]
				code := int(math.Round(float64(v/scale))) + zero
				if code < 0 {
					code = 0
				}
				if code > 15 {
					code = 15
				}
				bi := i >> 1
				if i&1 == 0 {
					crow[bi] = (crow[bi] &^ 0x0f) | byte(code)
				} else {
					crow[bi] = (crow[bi] & 0x0f) | byte(code<<4)
				}
			}
		}
	}
	return qt, scaleVec
}
