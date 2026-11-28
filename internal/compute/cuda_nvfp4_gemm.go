package compute

import (
	"fmt"
	"math"
	"os"
)

// CUDAInk2W4A4GEMMEnv enables the native sm_120f NVFP4 W4A4 Tensor Core GEMM
// path for Inkling-2 batched prefill (T >= 16) and multi-row speculative
// verify (K >= 4). Single-token decode stays on its memory-bound kernel.
const CUDAInk2W4A4GEMMEnv = "FAK_INKLING2_W4A4_GEMM"

// Ink2W4A4Mode distinguishes the two multi-row callers eligible for W4A4 GEMM.
type Ink2W4A4Mode int

const (
	Ink2W4A4Prefill Ink2W4A4Mode = iota
	Ink2W4A4Verify
)

const (
	ink2W4A4PrefillMinM = 16
	ink2W4A4VerifyMinM  = 4
	ink2W4A4TileM       = 64
	ink2W4A4TileN       = 16
	ink2W4A4TileK       = 64
	ink2W4A4MaxE4M3Code = byte(0x7e)
)

// Ink2W4A4GEMMEnabled reports whether FAK_INKLING2_W4A4_GEMM=1 is set.
func Ink2W4A4GEMMEnabled() bool {
	return os.Getenv(CUDAInk2W4A4GEMMEnv) == "1"
}

// Ink2W4A4GEMMEligible reports whether a dense or routed projection of shape
// Y[M, outN] = X[M, inK] * W[outN, inK]^T should dispatch to the native W4A4
// Tensor Core GEMM under the current opt-in gate.
func Ink2W4A4GEMMEligible(mode Ink2W4A4Mode, rowsM, outN, inK int) bool {
	if !Ink2W4A4GEMMEnabled() {
		return false
	}
	if outN <= 0 || inK <= 0 || inK%NVFP4BlockSize != 0 {
		return false
	}
	switch mode {
	case Ink2W4A4Prefill:
		return rowsM >= ink2W4A4PrefillMinM
	case Ink2W4A4Verify:
		return rowsM >= ink2W4A4VerifyMinM
	default:
		return false
	}
}

// Ink2W4A4TilePlan captures the m16n8k64 block-scaled MMA launch geometry,
// padded K stride, and tail counts for arbitrary [M, outN, inK] shapes.
type Ink2W4A4TilePlan struct {
	RowsM, OutN, InK int
	KPad             int
	MTiles, NTiles   int
	KTiles           int
	TailM, TailN     int
	TailK            int
}

// PlanInk2W4A4GEMM computes the tile grid and tail geometry for Y[M, outN] =
// X[M, inK] * W[outN, inK]^T using m16n8k64 block-scaled MMA tiles.
func PlanInk2W4A4GEMM(rowsM, outN, inK int) (Ink2W4A4TilePlan, error) {
	if rowsM <= 0 || outN <= 0 || inK <= 0 || inK%NVFP4BlockSize != 0 {
		return Ink2W4A4TilePlan{}, fmt.Errorf("%w: W4A4 GEMM shape M=%d N=%d K=%d requires positive dims and K divisible by %d",
			ErrNVFP4Shape, rowsM, outN, inK, NVFP4BlockSize)
	}
	kPad := ((inK + ink2W4A4TileK - 1) / ink2W4A4TileK) * ink2W4A4TileK
	return Ink2W4A4TilePlan{
		RowsM:  rowsM,
		OutN:   outN,
		InK:    inK,
		KPad:   kPad,
		MTiles: (outN + ink2W4A4TileM - 1) / ink2W4A4TileM,
		NTiles: (rowsM + ink2W4A4TileN - 1) / ink2W4A4TileN,
		KTiles: kPad / ink2W4A4TileK,
		TailM:  rowsM % ink2W4A4TileN,
		TailN:  outN % ink2W4A4TileM,
		TailK:  inK % ink2W4A4TileK,
	}, nil
}

// Ink2W4A4QuantizedActivations holds the dynamically quantized NVFP4 activation
// buffers for [M, inK], zero-padded along K to KPad for m16n8k64 alignment.
type Ink2W4A4QuantizedActivations struct {
	RowsM     int
	InK       int
	KPad      int
	Packed    []byte
	Scales    []byte
	RowScales []float32
}

// QuantizeInk2W4A4Activations dynamically quantizes finite row-major X[M, inK]
// into block-16 E2M1 codes, UE4M3FN block scales (clamped to 0x7e on overflow),
// and per-row FP32 scales, zero-padding K to a multiple of 64.
func QuantizeInk2W4A4Activations(rowsM, inK int, x []float32) (Ink2W4A4QuantizedActivations, error) {
	if rowsM <= 0 || inK <= 0 || inK%NVFP4BlockSize != 0 {
		return Ink2W4A4QuantizedActivations{}, fmt.Errorf("%w: activation shape [%d %d] requires positive dims and K divisible by %d",
			ErrNVFP4Shape, rowsM, inK, NVFP4BlockSize)
	}
	if len(x) != rowsM*inK {
		return Ink2W4A4QuantizedActivations{}, fmt.Errorf("%w: activations have %d elements, want %d",
			ErrNVFP4Input, len(x), rowsM*inK)
	}
	for i, v := range x {
		if !nvfp4Finite(v) {
			return Ink2W4A4QuantizedActivations{}, fmt.Errorf("%w: activation element %d is non-finite", ErrNVFP4Input, i)
		}
	}

	kPad := ((inK + ink2W4A4TileK - 1) / ink2W4A4TileK) * ink2W4A4TileK
	nb := inK / NVFP4BlockSize
	nbPad := kPad / NVFP4BlockSize
	qRowBytes := kPad / 2
	packed := make([]byte, rowsM*qRowBytes)
	scales := make([]byte, rowsM*nbPad)
	rowScales := make([]float32, rowsM)

	for m := 0; m < rowsM; m++ {
		row := x[m*inK : (m+1)*inK]
		var amax float32
		for _, v := range row {
			if a := float32(math.Abs(float64(v))); a > amax {
				amax = a
			}
		}
		rs := float32(float64(amax) / (nvfp4E2M1Max * nvfp4E4M3Max))
		for rs > 0 && !nvfp4Finite(float32(nvfp4CombinedMax)*rs) {
			rs = math.Nextafter32(rs, 0)
		}
		rowScales[m] = rs
		var ic float32
		if rs > 0 {
			ic = 1.0 / rs
		}

		for b := 0; b < nb; b++ {
			vals := row[b*NVFP4BlockSize : (b+1)*NVFP4BlockSize]
			var v [NVFP4BlockSize]float32
			var amax16 float32
			for k := 0; k < NVFP4BlockSize; k++ {
				v[k] = vals[k] * ic
				if !nvfp4Finite(v[k]) {
					v[k] = float32(math.Copysign(math.MaxFloat32, float64(v[k])))
				}
				if a := float32(math.Abs(float64(v[k]))); a > amax16 {
					amax16 = a
				}
			}

			a6 := float64(amax16) / nvfp4E2M1Max
			var first int
			if a6 > 0 {
				if a6 >= nvfp4E4M3Max {
					first = int(ink2W4A4MaxE4M3Code)
				} else {
					first = int(nvfp4EncodePositiveE4M3RNE(a6))
					if first > int(ink2W4A4MaxE4M3Code) {
						first = int(ink2W4A4MaxE4M3Code)
					}
				}
			}
			code := first
			sc := DecodeE4M3(byte(code))
			best := ink2W4A4BlockErr(&v, sc)
			for _, off := range []int{-1, 1, -2, 2} {
				tc := first + off
				if tc < 0 || tc > int(ink2W4A4MaxE4M3Code) {
					continue
				}
				tsc := DecodeE4M3(byte(tc))
				if cur := ink2W4A4BlockErr(&v, tsc); cur < best {
					best, code, sc = cur, tc, tsc
				}
			}

			var inv float32
			if sc > 0 {
				inv = 1.0 / sc
			}
			s := ic * inv
			if !nvfp4Finite(s) {
				s = 0
			}
			qBase := m*qRowBytes + b*8
			for k := 0; k < NVFP4BlockSize; k += 2 {
				c0 := nvfp4EncodeE2M1RNE(float64(vals[k]) * float64(s))
				c1 := nvfp4EncodeE2M1RNE(float64(vals[k+1]) * float64(s))
				packed[qBase+k/2] = c0 | (c1 << 4)
			}
			scales[m*nbPad+b] = byte(code)
		}
	}

	return Ink2W4A4QuantizedActivations{
		RowsM:     rowsM,
		InK:       inK,
		KPad:      kPad,
		Packed:    packed,
		Scales:    scales,
		RowScales: rowScales,
	}, nil
}

func ink2W4A4BlockErr(v *[NVFP4BlockSize]float32, sc float32) float32 {
	var inv float32
	if sc > 0 {
		inv = 1.0 / sc
	}
	var err float32
	for k := 0; k < NVFP4BlockSize; k++ {
		q := nvfp4EncodeE2M1RNE(float64(v[k] * inv))
		e := float32(math.Abs(float64(v[k]))) - float32(math.Abs(float64(mxE2M1Values[q])))*sc
		err += e * e
	}
	return err
}

// GemmInk2W4A4Reference is the scalar/FP32 reference oracle for the native
// sm_120f NVFP4 W4A4 block-scaled GEMM: Y[M, outN] = X[M, inK] * W[outN, inK]^T.
// It dynamically quantizes X using QuantizeInk2W4A4Activations and mirrors the
// m16n8k64 block_scale accumulation order (block-16 E2M1*E2M1 scaled by
// UE4M3FN(w)*UE4M3FN(a), then multiplied by rowScale[m] * wGlobal).
func GemmInk2W4A4Reference(outN, inK, rowsM int, wPacked, wScales []byte, wGlobal float32, x []float32) ([]float32, error) {
	if _, err := PlanInk2W4A4GEMM(rowsM, outN, inK); err != nil {
		return nil, err
	}
	_, scaleCols, wBlockScales, err := validateNVFP4(outN, inK, wPacked, wScales, wGlobal)
	if err != nil {
		return nil, err
	}
	act, err := QuantizeInk2W4A4Activations(rowsM, inK, x)
	if err != nil {
		return nil, err
	}

	y := make([]float32, rowsM*outN)
	wRowBytes := inK / 2
	qRowBytes := act.KPad / 2
	qScaleCols := act.KPad / NVFP4BlockSize

	for m := 0; m < rowsM; m++ {
		rs := act.RowScales[m]
		for n := 0; n < outN; n++ {
			var acc float32
			for b := 0; b < scaleCols; b++ {
				ws := wBlockScales[n*scaleCols+b]
				as := DecodeE4M3(act.Scales[m*qScaleCols+b])
				var blockDot float32
				for k := 0; k < NVFP4BlockSize; k++ {
					col := b*NVFP4BlockSize + k
					wByte := wPacked[n*wRowBytes+col/2]
					aByte := act.Packed[m*qRowBytes+col/2]
					wCode := wByte & 0x0f
					aCode := aByte & 0x0f
					if col&1 != 0 {
						wCode = wByte >> 4
						aCode = aByte >> 4
					}
					blockDot += mxE2M1Values[wCode] * mxE2M1Values[aCode]
				}
				acc += blockDot * ws * as
			}
			out := acc * rs * wGlobal
			if !nvfp4Finite(out) {
				return nil, fmt.Errorf("%w: W4A4 output overflow at [%d, %d]", ErrNVFP4Scale, m, n)
			}
			y[m*outN+n] = out
		}
	}
	return y, nil
}

// GemmInk2W4A4FP32Reference computes the exact FP32-activation oracle Y[M, outN]
// = X[M, inK] * DecodeNVFP4Reference(W)^T for parity comparison against W4A4.
func GemmInk2W4A4FP32Reference(outN, inK, rowsM int, wPacked, wScales []byte, wGlobal float32, x []float32) ([]float32, error) {
	if _, err := PlanInk2W4A4GEMM(rowsM, outN, inK); err != nil {
		return nil, err
	}
	w, err := DecodeNVFP4Reference(outN, inK, wPacked, wScales, wGlobal)
	if err != nil {
		return nil, err
	}
	if len(x) != rowsM*inK {
		return nil, fmt.Errorf("%w: activations have %d elements, want %d", ErrNVFP4Input, len(x), rowsM*inK)
	}
	for i, v := range x {
		if !nvfp4Finite(v) {
			return nil, fmt.Errorf("%w: activation element %d is non-finite", ErrNVFP4Input, i)
		}
	}
	y := make([]float32, rowsM*outN)
	for m := 0; m < rowsM; m++ {
		xRow := x[m*inK : (m+1)*inK]
		for n := 0; n < outN; n++ {
			wRow := w[n*inK : (n+1)*inK]
			var sum float64
			for k := 0; k < inK; k++ {
				sum += float64(wRow[k]) * float64(xRow[k])
			}
			out := float32(sum)
			if !nvfp4Finite(out) {
				return nil, fmt.Errorf("%w: FP32 oracle overflow at [%d, %d]", ErrNVFP4Scale, m, n)
			}
			y[m*outN+n] = out
		}
	}
	return y, nil
}
