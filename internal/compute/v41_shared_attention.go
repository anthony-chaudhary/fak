package compute

import (
	"fmt"
	"math"
	"slices"
)

// V41SharedAttentionMode preserves the distinct maximum seeds of the two live
// V4.1 host contractions. These values are part of the native/shader ABI.
type V41SharedAttentionMode int32

const (
	V41SharedAttentionPlain      V41SharedAttentionMode = 0
	V41SharedAttentionCompressed V41SharedAttentionMode = 1
)

// V41SharedAttentionBackend is an optional one-position shared-latent, sink
// attention primitive. It does not widen Backend. Inputs are immutable, live,
// unquantized row-major F32 tensors owned by this backend: q[heads,headDim],
// sharedKV[selectedRows,headDim], and sink[heads] when hasSink is true. Otherwise
// sink is a live [1] zero dummy, ignored by the kernel. Each ordered shared row
// supplies both K and V to every head; duplicate rows remain distinct slots.
// The caller validates all original values and selections before gathering.
//
// selectedRows, heads, and headDim must be positive. Empty selections are handled
// by the caller as owned host zeros without upload or dispatch. scale is finite
// and nonzero; the sink is unscaled and contributes only to the denominator.
// Three scalar-order F32 passes use separately rounded products and sums, with
// no FMA, reduction tree, BF16 cast, or online-softmax substitution. Plain seeds
// max with -Inf; Compressed seeds it with -MaxFloat32 and skips when that sentinel
// survives, after checking every scaled score. exp(-Inf) from subtraction of
// finite scores is valid. Other nonfinite intermediates are typed failures.
//
// Output is fresh [heads,headDim], disjoint from inputs and status storage, and
// owned by the caller. A selected upload/dispatch/status/readback/arithmetic
// failure returns no usable output and must never trigger a host retry. Optional
// capability absence permits selection of the unchanged host implementation;
// finite-input overflow rejection deliberately tightens the legacy plain path.
type V41SharedAttentionBackend interface {
	Backend
	SupportsV41SharedAttention() bool
	V41SharedAttention(q, sharedKV, sink Tensor, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool) (Tensor, error)
}

// V41SharedAttentionStage is the producer stage recorded in the device status.
// Keep its values synchronized with v41_shared_attention.comp and the native ABI.
type V41SharedAttentionStage uint32

const (
	V41SharedAttentionStageSuccess     V41SharedAttentionStage = 0
	V41SharedAttentionStageScore       V41SharedAttentionStage = 1
	V41SharedAttentionStageExp         V41SharedAttentionStage = 2
	V41SharedAttentionStageDenominator V41SharedAttentionStage = 3
	V41SharedAttentionStageWeight      V41SharedAttentionStage = 4
	V41SharedAttentionStageValue       V41SharedAttentionStage = 5
	V41SharedAttentionStageOutput      V41SharedAttentionStage = 6
)

func (s V41SharedAttentionStage) String() string {
	switch s {
	case V41SharedAttentionStageSuccess:
		return "success"
	case V41SharedAttentionStageScore:
		return "scaled score"
	case V41SharedAttentionStageExp:
		return "exp term"
	case V41SharedAttentionStageDenominator:
		return "denominator"
	case V41SharedAttentionStageWeight:
		return "normalized weight"
	case V41SharedAttentionStageValue:
		return "weighted-value accumulator"
	case V41SharedAttentionStageOutput:
		return "output validation"
	default:
		return fmt.Sprintf("unknown stage %d", uint32(s))
	}
}

// V41SharedAttentionArithmeticError retains device producer attribution. Head,
// SelectedSlot and Element are zero-based; -1 means not applicable. A slot is a
// gathered slot, not a source row/group: the model owns that ordered mapping.
// ValueBits retains the offending binary32 payload, including a NaN payload.
type V41SharedAttentionArithmeticError struct {
	Stage        V41SharedAttentionStage
	Head         int
	SelectedSlot int
	Element      int
	ValueBits    uint32
}

func (e *V41SharedAttentionArithmeticError) Error() string {
	return fmt.Sprintf("compute: V4.1 shared attention nonfinite %s at head=%d selectedSlot=%d element=%d value=%g bits=0x%08x",
		e.Stage, e.Head, e.SelectedSlot, e.Element, math.Float32frombits(e.ValueBits), e.ValueBits)
}

// V41SharedAttentionProtocolError describes malformed status, not an inferred
// arithmetic producer failure. Head is -1 when the whole buffer is malformed.
type V41SharedAttentionProtocolError struct {
	Head    int
	Message string
}

func (e *V41SharedAttentionProtocolError) Error() string {
	return fmt.Sprintf("compute: V4.1 shared attention status protocol at head=%d: %s", e.Head, e.Message)
}

const (
	v41SharedAttentionStatusWords       = 4
	v41SharedAttentionStatusBytes       = 16
	v41SharedAttentionPushConstantBytes = 24
)

// validateV41SharedAttention checks metadata and byte/index limits before native
// narrowing. Backend-specific code must additionally check ownership, exact live
// buffer extents, and disjoint output/status allocations without reading opaque
// device buffers through host pointers.
func validateV41SharedAttention(q, sharedKV, sink Tensor, selectedRows, heads, headDim int, scale float32, mode V41SharedAttentionMode, hasSink bool) (qBytes, kvBytes, sinkBytes, statusBytes int, err error) {
	const maxABIInt = uint64(1<<31 - 1)
	if selectedRows <= 0 || heads <= 0 || headDim <= 0 ||
		uint64(selectedRows) > maxABIInt || uint64(heads) > maxABIInt || uint64(headDim) > maxABIInt {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 shared attention invalid geometry rows=%d heads=%d headDim=%d", selectedRows, heads, headDim)
	}
	if mode != V41SharedAttentionPlain && mode != V41SharedAttentionCompressed {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 shared attention invalid mode %d", mode)
	}
	if scale == 0 || math.IsNaN(float64(scale)) || math.IsInf(float64(scale), 0) {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 shared attention requires finite nonzero scale")
	}
	qElements := uint64(heads) * uint64(headDim)
	kvElements := uint64(selectedRows) * uint64(headDim)
	maxHostBytes := uint64(int(^uint(0) >> 1))
	if qElements > maxABIInt || kvElements > maxABIInt ||
		qElements > maxHostBytes/4 || kvElements > maxHostBytes/4 ||
		uint64(heads) > maxHostBytes/v41SharedAttentionStatusBytes ||
		uint64(heads)*v41SharedAttentionStatusWords > uint64(1<<32-1) {
		return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 shared attention geometry exceeds native index or host byte range")
	}
	sinkElements := 1
	if hasSink {
		sinkElements = heads
	}
	for _, operand := range []struct {
		name string
		t    Tensor
		want []int
	}{
		{"q", q, []int{heads, headDim}},
		{"sharedKV", sharedKV, []int{selectedRows, headDim}},
		{"sink", sink, []int{sinkElements}},
	} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || !slices.Equal(operand.t.Shape, operand.want) {
			return 0, 0, 0, 0, fmt.Errorf("compute: V4.1 shared attention %s must be unquantized row-major F32 with shape %v", operand.name, operand.want)
		}
	}
	return int(qElements) * 4, int(kvElements) * 4, sinkElements * 4, heads * v41SharedAttentionStatusBytes, nil
}

// decodeV41SharedAttentionStatus validates every head before selecting the first
// arithmetic failure in ascending head order. There is no atomic winner race.
// Each record is stage, slot+1, element+1, bits; zero indices decode to -1. All
// success fields must be initialized to zero on every dispatch, not inherited
// from allocation or a previous successful dispatch.
func decodeV41SharedAttentionStatus(words []uint32, heads, selectedRows, headDim int) error {
	if heads <= 0 || selectedRows <= 0 || headDim <= 0 ||
		uint64(heads) > uint64(1<<31-1) || uint64(selectedRows) > uint64(1<<31-1) || uint64(headDim) > uint64(1<<31-1) ||
		uint64(heads)*v41SharedAttentionStatusWords != uint64(len(words)) {
		return &V41SharedAttentionProtocolError{Head: -1, Message: "invalid geometry or status word count"}
	}
	for h := 0; h < heads; h++ {
		w := words[h*v41SharedAttentionStatusWords : (h+1)*v41SharedAttentionStatusWords]
		stage := V41SharedAttentionStage(w[0])
		bad := func(message string) error { return &V41SharedAttentionProtocolError{Head: h, Message: message} }
		if stage > V41SharedAttentionStageOutput || w[1] > uint32(selectedRows) || w[2] > uint32(headDim) {
			return bad("stage or encoded coordinate is out of range")
		}
		if stage == V41SharedAttentionStageSuccess {
			if w[1] != 0 || w[2] != 0 || w[3] != 0 {
				return bad("success record contains nonzero fields")
			}
			continue
		}
		if w[3]&0x7f800000 != 0x7f800000 {
			return bad("arithmetic failure contains a finite offending value")
		}
		switch stage {
		case V41SharedAttentionStageScore, V41SharedAttentionStageWeight:
			if w[1] == 0 || w[2] != 0 {
				return bad("score/weight requires a slot and no element")
			}
		case V41SharedAttentionStageExp:
			if w[2] != 0 {
				return bad("exp term must not identify an element")
			}
		case V41SharedAttentionStageDenominator:
			if w[1] != 0 || w[2] != 0 {
				return bad("denominator must not identify a slot or element")
			}
		case V41SharedAttentionStageValue:
			if w[1] == 0 || w[2] == 0 {
				return bad("value accumulator requires a slot and element")
			}
		case V41SharedAttentionStageOutput:
			if w[1] != 0 || w[2] == 0 {
				return bad("output requires an element and no slot")
			}
		}
	}
	for h := 0; h < heads; h++ {
		w := words[h*v41SharedAttentionStatusWords : (h+1)*v41SharedAttentionStatusWords]
		if w[0] != 0 {
			return &V41SharedAttentionArithmeticError{
				Stage: V41SharedAttentionStage(w[0]), Head: h,
				SelectedSlot: int(w[1]) - 1, Element: int(w[2]) - 1, ValueBits: w[3],
			}
		}
	}
	return nil
}
