package compute

import (
	"fmt"
	"slices"
)

// V41TailRoPEBackend is the optional one-position DeepSeek-V4.1 tail-adjacent
// rotary primitive. Inputs are immutable, F32, row-major tensors on this backend:
// q[heads,headDim], kv[headDim], sinCos[rotaryDim/2,2], with [sin,cos] table rows.
// The caller supplies the unchanged model table; implementations generate no
// frequencies and apply no additional scaling or BF16 rounding. Only the trailing
// rotaryDim lanes rotate, in adjacent pairs. Prefix bits are copied unchanged.
// Products are separately rounded to F32 before the F32 subtraction/addition.
// Returned tensors are fresh, independently owned allocations; the caller frees
// both. Capability absence permits caller selection of its existing reference;
// after selection an operation error must propagate, never replay on the CPU.
type V41TailRoPEBackend interface {
	SupportsV41TailRoPE() bool
	V41TailRoPEQK(q, kv, sinCos Tensor, heads, headDim, rotaryDim int) (qOut, kvOut Tensor, err error)
}

// validateV41TailRoPE rejects malformed dimensions before multiplying or narrowing
// them for the 32-bit shader ABI. It does not read tensor data or allocate outputs.
func validateV41TailRoPE(q, kv, sinCos Tensor, heads, headDim, rotaryDim int) (qBytes, kvBytes, tableBytes int, err error) {
	const maxABIInt = uint64(1<<31 - 1)
	if heads <= 0 || headDim <= 0 || rotaryDim <= 0 || rotaryDim > headDim || rotaryDim%2 != 0 ||
		uint64(heads) > maxABIInt || uint64(headDim) > maxABIInt || uint64(rotaryDim) > maxABIInt {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 tail RoPE invalid geometry heads=%d headDim=%d rotaryDim=%d", heads, headDim, rotaryDim)
	}
	qElements := uint64(heads) * uint64(headDim)
	// The shader indexes Q followed by one KV row in a single uint32 dispatch.
	if qElements+uint64(headDim) > uint64(1<<32-1) ||
		qElements > uint64(int(^uint(0)>>1))/4 || uint64(headDim) > uint64(int(^uint(0)>>1))/4 {
		return 0, 0, 0, fmt.Errorf("compute: V4.1 tail RoPE geometry exceeds shader index or host byte range")
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		want []int
	}{
		{"q", q, []int{heads, headDim}},
		{"kv", kv, []int{headDim}},
		{"sinCos", sinCos, []int{rotaryDim / 2, 2}},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || !slices.Equal(operand.t.Shape, operand.want) {
			return 0, 0, 0, fmt.Errorf("compute: V4.1 tail RoPE %s must be unquantized row-major F32 with shape %v", operand.name, operand.want)
		}
	}
	return int(qElements) * 4, headDim * 4, rotaryDim * 4, nil
}
