package compute

import (
	"fmt"
	"math"
	"slices"
)

// V41MHCStreams is the DeepSeek-V4.1 hyper-connection stream count.
const V41MHCStreams = 4

// V41MHCCoeffWidth is one row of split coefficients: pre[4] | post[4] | comb[16],
// with comb row-major [source stream, destination stream].
const V41MHCCoeffWidth = (2 + V41MHCStreams) * V41MHCStreams

// V41MHCBackend is the optional DeepSeek-V4.1 mHC stream primitive. Every
// operand is an unquantized row-major F32 tensor on this backend; every result
// is a fresh caller-owned allocation the caller frees. Residual publications
// (initial streams, pre-collapse, post-mix) are rounded to BF16 with integer
// round-to-nearest-even and widened back to F32, matching the host model.
// Products and sums are separately rounded F32 operations in stream order, so
// initial/pre/post are bit-identical to the host reference. The split computes
// exp and division in F32 (about 1 ulp each) where the host widens to F64, so
// its coefficients agree within a few ulp, not bit-for-bit.
//
// Capability absence permits caller selection of its existing reference; after
// selection an operation error must propagate, never replay on the CPU.
type V41MHCBackend interface {
	SupportsV41MHC() bool
	// V41MHCInitialStreams: x[rows,hidden] -> streams[rows,4,hidden], each stream
	// a BF16 copy of x.
	V41MHCInitialStreams(x Tensor, rows, hidden int) (Tensor, error)
	// V41MHCSplit: mixes[rows,24] (projected and RMS-scaled), scale[3], base[24]
	// -> coeffs[rows,24]: pre = sigmoid(m*s0+b)+eps, post = 2*sigmoid(m*s1+b),
	// comb = Sinkhorn(row softmax(m*s2+b)+eps) over iters normalizations.
	V41MHCSplit(mixes, scale, base Tensor, rows, iters int, eps float32) (Tensor, error)
	// V41MHCPre: streams[rows,4,hidden], coeffs[rows,24] -> BF16(sum_h pre[h]*stream[h]).
	V41MHCPre(streams, coeffs Tensor, rows, hidden int) (Tensor, error)
	// V41MHCPost: x[rows,hidden], streams[rows,4,hidden], coeffs[rows,24] ->
	// next[rows,4,hidden] = BF16(sum_src comb[src,dst]*stream[src] + post[dst]*BF16(x)).
	V41MHCPost(x, streams, coeffs Tensor, rows, hidden int) (Tensor, error)
	// V41MHCRMSSplit: raw mixes[rows,24] (unscaled projection of the flattened
	// streams), streams[rows,4,hidden], params[27] = scale[3]|base[24] ->
	// coeffs[rows,24]. Each row's mixes are first multiplied by
	// 1/sqrt(sum(x^2)/(4*hidden)+rmsEps) over that row's flattened streams, the
	// sum taken serially in index order as the host projection does, then split
	// exactly as V41MHCSplit.
	V41MHCRMSSplit(raw, streams, params Tensor, rows, hidden, iters int, eps, rmsEps float32) (Tensor, error)
}

// V41MHCParamWidth is the rms-split parameter row: scale[3] | base[24].
const V41MHCParamWidth = 3 + V41MHCCoeffWidth

// V41MHCOp names one mode of the shared mHC module (its push-constant mode).
type V41MHCOp int

const (
	V41MHCOpInitial V41MHCOp = iota
	V41MHCOpSplit
	V41MHCOpPre
	V41MHCOpPost
	V41MHCOpRMSSplit
)

func (op V41MHCOp) String() string {
	switch op {
	case V41MHCOpInitial:
		return "initial-streams"
	case V41MHCOpSplit:
		return "split"
	case V41MHCOpPre:
		return "pre-collapse"
	case V41MHCOpPost:
		return "post-mix"
	case V41MHCOpRMSSplit:
		return "rms-split"
	}
	return fmt.Sprintf("mhc-op-%d", int(op))
}

// v41MHCOperand is one validated input and its exact byte extent.
type v41MHCOperand struct {
	name  string
	t     Tensor
	shape []int
}

// validateV41MHC rejects malformed geometry before multiplying or narrowing it
// for the 32-bit shader ABI. It returns the operands in binding order and the
// output shape. It reads no tensor data and allocates nothing.
func validateV41MHC(op V41MHCOp, rows, hidden, iters int, eps float32, operands ...Tensor) ([]v41MHCOperand, []int, error) {
	const maxABIInt = 1<<31 - 1
	if rows <= 0 || rows > maxABIInt {
		return nil, nil, fmt.Errorf("compute: V4.1 mHC %s invalid rows=%d", op, rows)
	}
	var want []v41MHCOperand
	var out []int
	switch op {
	case V41MHCOpInitial, V41MHCOpPre, V41MHCOpPost, V41MHCOpRMSSplit:
		if op == V41MHCOpRMSSplit && (iters < 1 || iters > maxABIInt || eps <= 0 || math.IsInf(float64(eps), 0) || eps != eps) {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC rms-split invalid iters=%d eps=%g", iters, eps)
		}
		if hidden <= 0 || hidden > maxABIInt {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC %s invalid hidden=%d", op, hidden)
		}
		// The shader indexes rows*4*hidden in one uint32 dispatch.
		if uint64(rows)*V41MHCStreams*uint64(hidden) > math.MaxUint32 ||
			uint64(rows)*V41MHCStreams*uint64(hidden) > uint64(int(^uint(0)>>1))/4 {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC %s geometry exceeds shader index or host byte range", op)
		}
	case V41MHCOpSplit:
		if iters < 1 || iters > maxABIInt || eps <= 0 || math.IsInf(float64(eps), 0) || eps != eps {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC split invalid iters=%d eps=%g", iters, eps)
		}
		if uint64(rows)*V41MHCCoeffWidth > math.MaxUint32 {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC split geometry exceeds shader index range")
		}
	default:
		return nil, nil, fmt.Errorf("compute: unknown V4.1 mHC op %d", int(op))
	}
	streams := []int{rows, V41MHCStreams, hidden}
	coeffs := []int{rows, V41MHCCoeffWidth}
	switch op {
	case V41MHCOpInitial:
		want = []v41MHCOperand{{name: "x", shape: []int{rows, hidden}}}
		out = streams
	case V41MHCOpSplit:
		want = []v41MHCOperand{{name: "mixes", shape: coeffs}, {name: "scale", shape: []int{3}}, {name: "base", shape: []int{V41MHCCoeffWidth}}}
		out = coeffs
	case V41MHCOpPre:
		want = []v41MHCOperand{{name: "streams", shape: streams}, {name: "coeffs", shape: coeffs}}
		out = []int{rows, hidden}
	case V41MHCOpPost:
		want = []v41MHCOperand{{name: "x", shape: []int{rows, hidden}}, {name: "streams", shape: streams}, {name: "coeffs", shape: coeffs}}
		out = streams
	case V41MHCOpRMSSplit:
		want = []v41MHCOperand{{name: "raw", shape: coeffs}, {name: "streams", shape: streams}, {name: "params", shape: []int{V41MHCParamWidth}}}
		out = coeffs
	}
	if len(operands) != len(want) {
		return nil, nil, fmt.Errorf("compute: V4.1 mHC %s takes %d operands, got %d", op, len(want), len(operands))
	}
	for i := range want {
		t := operands[i]
		if t.Dtype != F32 || t.Layout != RowMajor || t.Quant != nil || !slices.Equal(t.Shape, want[i].shape) {
			return nil, nil, fmt.Errorf("compute: V4.1 mHC %s %s must be unquantized row-major F32 with shape %v", op, want[i].name, want[i].shape)
		}
		want[i].t = t
	}
	return want, out, nil
}

func v41MHCElements(shape []int) int {
	n := 1
	for _, d := range shape {
		n *= d
	}
	return n
}
