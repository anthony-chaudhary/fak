package compute

import (
	"bytes"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/anthony-chaudhary/fak/internal/computetrace"
)

// V41MHCPrenormBackend folds the V4.1 mHC mix projection of the flattened
// streams and their sum of squares into one projection dispatch ahead of the
// rms-split, the split of DeepGEMM's tf32_hc_prenorm_gemm and TileKernels'
// mhc_reduce_partials_and_rmsnorm_fwd. Unlike that TF32 MMA, every row keeps
// the one-token MatMul's order: an F32 weight sums matmul.comp's 1024-wide
// chunk partials in chunk order and a Q2_K weight runs q2k_matvec.comp's lanes
// and tree. The coefficients are therefore bitwise MatMul(weight, row) for
// each row followed by V41MHCRMSSplit; the serial sum of squares moves off the
// split's critical path and runs beside the projection instead.
//
// Capability absence leaves the caller on its MatMul + V41MHCRMSSplit path;
// after selection an operation error must propagate, never replay.
type V41MHCPrenormBackend interface {
	V41MHCPrenormAvailable() bool
	// SupportsV41MHCPrenormWeight reports whether weight [24, 4*hidden] has a
	// prenorm projection on this device that a once-per-device self-check in
	// the weight's dtype found bit-identical to MatMul + V41MHCRMSSplit.
	SupportsV41MHCPrenormWeight(weight Tensor, hidden int) bool
	// V41MHCPrenormSplit: weight [24,4*hidden], streams [rows,4,hidden],
	// params[27] = scale[3]|base[24] -> fresh caller-owned coeffs[rows,24].
	V41MHCPrenormSplit(weight, streams, params Tensor, rows, hidden, iters int, eps, rmsEps float32) (Tensor, error)
}

// ErrV41MHCPrenormGeometry is the closed refusal for operands the prenorm
// projection does not serve. Nothing is allocated or recorded before it.
var ErrV41MHCPrenormGeometry = errors.New("compute: invalid V4.1 mHC prenorm geometry")

// v41MHCPrenormStride is one scratch row: raw mixes[24] | sum of squares.
const v41MHCPrenormStride = V41MHCCoeffWidth + 1

// A 64-lane workgroup holds every 1024-wide chunk of at least one F32 output.
const v41MHCPrenormMaxF32Chunks = 64

// v41MHCPrenormWeightServed checks the weight's dtype and [24, 4*hidden] shape
// against the two projection orders the module reproduces.
func v41MHCPrenormWeightServed(weight Tensor, hidden int) bool {
	if hidden <= 0 || hidden > (math.MaxUint32/V41MHCCoeffWidth)/V41MHCStreams {
		return false
	}
	width := V41MHCStreams * hidden
	if weight.Layout != RowMajor || !slices.Equal(weight.Shape, []int{V41MHCCoeffWidth, width}) {
		return false
	}
	switch weight.Dtype {
	case F32:
		return weight.Quant == nil && (width+1023)/1024 <= v41MHCPrenormMaxF32Chunks
	case Q2_K:
		return width%q2kSuper == 0
	default:
		// the module serves exactly those two projection orders; every other dtype refuses here
		return false
	}
}

// validateV41MHCPrenorm rejects malformed operands before any native call. It
// reads no tensor data and allocates nothing.
func validateV41MHCPrenorm(weight, streams, params Tensor, rows, hidden, iters int, eps, rmsEps float32) error {
	const maxABIInt = 1<<31 - 1
	if rows <= 0 || rows > maxABIInt || hidden <= 0 || hidden > maxABIInt {
		return fmt.Errorf("%w: rows=%d hidden=%d", ErrV41MHCPrenormGeometry, rows, hidden)
	}
	if iters < 1 || iters > maxABIInt || !(eps > 0) || math.IsInf(float64(eps), 0) || !(rmsEps > 0) || math.IsInf(float64(rmsEps), 0) {
		return fmt.Errorf("%w: iters=%d eps=%g rmsEps=%g", ErrV41MHCPrenormGeometry, iters, eps, rmsEps)
	}
	if uint64(rows)*V41MHCStreams*uint64(hidden) > math.MaxUint32 ||
		uint64(rows)*V41MHCStreams*uint64(hidden) > uint64(int(^uint(0)>>1))/4 ||
		uint64(rows)*v41MHCPrenormStride > math.MaxUint32 {
		return fmt.Errorf("%w: rows=%d hidden=%d exceeds the shader index range", ErrV41MHCPrenormGeometry, rows, hidden)
	}
	if !v41MHCPrenormWeightServed(weight, hidden) {
		return fmt.Errorf("%w: weight %v %v is not a served [24,%d] projection", ErrV41MHCPrenormGeometry, weight.Dtype, weight.Shape, V41MHCStreams*hidden)
	}
	for _, operand := range []struct {
		name  string
		t     Tensor
		shape []int
	}{
		{"streams", streams, []int{rows, V41MHCStreams, hidden}},
		{"params", params, []int{V41MHCParamWidth}},
	} {
		t := operand.t
		if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || !slices.Equal(t.Shape, operand.shape) {
			return fmt.Errorf("%w: %s must be unquantized row-major F32 with shape %v", ErrV41MHCPrenormGeometry, operand.name, operand.shape)
		}
	}
	return nil
}

// v41MHCPrenormProbe is the fixed self-check operand set of one weight dtype.
// Every row count in rows is compared against its own MatMul + rms-split.
type v41MHCPrenormProbe struct {
	dtype   Dtype
	hidden  int
	rows    []int
	weight  []byte    // device bytes of the [24, 4*hidden] weight
	streams []float32 // [max(rows), 4, hidden], BF16-exact like the model's panel
	params  []float32 // scale[3] | base[24]
}

const (
	v41MHCPrenormProbeIters  = 20
	v41MHCPrenormProbeEps    = float32(1e-6)
	v41MHCPrenormProbeRMSEps = float32(1e-6)
)

// newV41MHCPrenormProbe builds the probe for an F32 or Q2_K weight. F32 hidden
// 1000 spans four 1024-wide chunks, the last partial, and two projection
// workgroups, the second partly filled; it runs one row (MatMul) and a two-row
// panel (BatchedMatMul). Q2_K hidden 320 spans five superblocks on one row,
// the only Q2_K shape the model selects. Full-mantissa F32 weights times BF16
// streams round every product, so a contraction difference changes bits.
func newV41MHCPrenormProbe(dtype Dtype) v41MHCPrenormProbe {
	p := v41MHCPrenormProbe{dtype: dtype, hidden: 1000, rows: []int{1, 2}}
	if dtype == Q2_K {
		p.hidden, p.rows = 320, []int{1}
	}
	width := V41MHCStreams * p.hidden
	if dtype == Q2_K {
		p.weight = make([]byte, V41MHCCoeffWidth*(width/q2kSuper)*q2kSuperBlock)
		for i := range p.weight {
			p.weight[i] = byte(math.Float32bits(v41MHCPrenormProbeValue(i, 11)) >> 7)
		}
		for b := 0; b < len(p.weight); b += q2kSuperBlock {
			binary.LittleEndian.PutUint16(p.weight[b+80:], 0x2000) // d = 1/128
			binary.LittleEndian.PutUint16(p.weight[b+82:], 0x1c00) // dmin = 1/256
		}
	} else {
		p.weight = make([]byte, 4*V41MHCCoeffWidth*width)
		for i := 0; i < V41MHCCoeffWidth*width; i++ {
			binary.LittleEndian.PutUint32(p.weight[4*i:], math.Float32bits(0.03*v41MHCPrenormProbeValue(i, 7)))
		}
	}
	p.streams = make([]float32, slices.Max(p.rows)*width)
	for i := range p.streams {
		scale := float32(int(1) << (i % 5))
		p.streams[i] = v41MHCPrenormProbeBF16(scale * v41MHCPrenormProbeValue(i, 3))
	}
	p.params = make([]float32, V41MHCParamWidth)
	for i := range p.params {
		p.params[i] = 0.5 * v41MHCPrenormProbeValue(i, 5)
	}
	copy(p.params, []float32{0.75, 0.5, 0.25})
	return p
}

func v41MHCPrenormProbeValue(i, salt int) float32 {
	h := uint32(i)*2654435761 + uint32(salt)*40503
	h ^= h >> 15
	h *= 2246822519
	h ^= h >> 13
	return float32(int32(h>>8)-(1<<23)) / float32(1<<23)
}

// v41MHCPrenormProbeBF16 rounds a finite value to nearest-even BF16.
func v41MHCPrenormProbeBF16(f float32) float32 {
	bits := math.Float32bits(f)
	bits += 0x7fff + (bits>>16)&1
	return math.Float32frombits(bits &^ 0xffff)
}

// v41MHCPrenormSelfCheckEqual reads the image as four regions per probe row
// count: reference raw [rows,24], reference coeffs [rows,24], prenorm scratch
// [rows,25] and prenorm coeffs [rows,24].
func v41MHCPrenormSelfCheckEqual(probe v41MHCPrenormProbe, image [][]byte) bool {
	const raw = 4 * V41MHCCoeffWidth
	if len(image) != 4*len(probe.rows) {
		return false
	}
	for i, rows := range probe.rows {
		refRaw, refOut, scratch, out := image[4*i], image[4*i+1], image[4*i+2], image[4*i+3]
		if !bytes.Equal(refOut, out) {
			return false
		}
		for r := 0; r < rows; r++ {
			if !bytes.Equal(refRaw[r*raw:(r+1)*raw], scratch[r*4*v41MHCPrenormStride:][:raw]) {
				return false
			}
		}
	}
	return true
}

// v41MHCPrenormKernel is the bounded kernel id of the fused projection, one
// per served weight dtype.
func v41MHCPrenormKernel(dtype Dtype) string { return dtype.String() + "_v41_mhc_prenorm" }

// v41MHCPrenormKernelEvent is the computetrace record of one prenorm call. It
// is the record the replaced MatMul emitted (operation "matmul", phase
// "kernel", shapes [weight, input]) with the input a [4*hidden] row or a
// [rows, 4*hidden] panel, so an operation-filtered trace still finds every mix
// projection; only the kernel id names the module that ran.
func v41MHCPrenormKernelEvent(backendName string, weight Tensor, rows, hidden int, started time.Time, timing vulkanKernelTiming) computetrace.Event {
	input := Tensor{Dtype: F32, Layout: RowMajor, Shape: []int{V41MHCStreams * hidden}}
	if rows != 1 {
		input.Shape = []int{rows, V41MHCStreams * hidden}
	}
	e := vulkanKernelEvent(backendName, weight, input, started, timing)
	e.Kernel = v41MHCPrenormKernel(weight.Dtype)
	return e
}

// recordV41MHCPrenormKernel shares recordVulkanMatMulKernel's Emitting gate.
func recordV41MHCPrenormKernel(backendName string, weight Tensor, rows, hidden int, started time.Time, timing vulkanKernelTiming) {
	if !computetrace.Emitting() {
		return
	}
	computetrace.Record(v41MHCPrenormKernelEvent(backendName, weight, rows, hidden, started, timing))
}
