package compute

import (
	"fmt"
	"math"
)

// V41RoutedSwiGLUBackend is the optional fused DeepSeek-V4.1 routed-expert
// activation between the gate/up and down projections (Ref #13815). gate and
// up are live, unquantized row-major F32 [N] tensors on this backend, usually
// the two GEMM outputs. Per lane the result is
//
//	g = bf16(gate), u = bf16(up)            (RNE, widened to F32)
//	g = min(g, limit), u = clamp(u, ±limit) (only when limit > 0)
//	f = (g / (1 + exp(-g))) * u             (F32)
//	out = bf16(f * weight)                  (F32 product, then RNE)
//
// which is the pinned Expert.forward graph through the down-input cast. The
// output is a fresh caller-owned F32 [N] tensor that feeds the down projection
// without a host round trip. Add, multiply and the BF16 bit rounding are
// exact; exp and the division use device precision, so the result carries a
// tolerance against the host formula. A nonfinite or BF16-overflowing value
// is a *V41RoutedSwiGLUFault, a numeric verdict on the inputs; any other error
// is a backend failure. Neither publishes an output, and neither may be
// replayed on the host.
type V41RoutedSwiGLUBackend interface {
	V41RoutedSwiGLUAvailable() bool
	V41RoutedSwiGLU(gate, up Tensor, limit, weight float32) (Tensor, error)
}

// V41RoutedSwiGLUStage is the host stage a routed activation fault belongs to.
type V41RoutedSwiGLUStage uint8

const (
	// V41RoutedSwiGLUStageActivation covers the gate/up BF16 publication and
	// the activation product: a nonfinite gate or up value, a BF16 overflow of
	// either, or a nonfinite silu(g)*u.
	V41RoutedSwiGLUStageActivation V41RoutedSwiGLUStage = 1
	// V41RoutedSwiGLUStageDownOperand covers the weighted product and its BF16
	// down-input publication.
	V41RoutedSwiGLUStageDownOperand V41RoutedSwiGLUStage = 2
)

func (s V41RoutedSwiGLUStage) String() string {
	switch s {
	case V41RoutedSwiGLUStageActivation:
		return "activation"
	case V41RoutedSwiGLUStageDownOperand:
		return "down operand"
	default:
		return fmt.Sprintf("unknown stage %d", uint8(s))
	}
}

// V41RoutedSwiGLUFault names the lowest faulting lane of the earliest stage.
// An activation fault anywhere precedes every down-operand fault, matching the
// host order, which publishes all of gate/up before weighting any lane.
type V41RoutedSwiGLUFault struct {
	Stage V41RoutedSwiGLUStage
	Index int
}

func (f *V41RoutedSwiGLUFault) Error() string {
	return fmt.Sprintf("compute: V4.1 routed SwiGLU %s is nonfinite or overflows BF16 at lane %d", f.Stage, f.Index)
}

const (
	v41RoutedSwiGLUStatusMagic uint32 = 0x52535731
	v41RoutedSwiGLUStatusBytes        = 16
)

// validateV41RoutedSwiGLU checks geometry and scalars before any native call.
// It reads no tensor data and allocates nothing.
func validateV41RoutedSwiGLU(gate, up Tensor, limit, weight float32) (int, error) {
	for _, operand := range []struct {
		name string
		t    Tensor
	}{{"gate", gate}, {"up", up}} {
		if operand.t.Dtype != F32 || operand.t.Layout != RowMajor || operand.t.Quant != nil || len(operand.t.Shape) != 1 {
			return 0, fmt.Errorf("compute: V4.1 routed SwiGLU %s must be an unquantized row-major F32 vector", operand.name)
		}
	}
	count := gate.Shape[0]
	if count <= 0 || up.Shape[0] != count || count > math.MaxInt32 {
		return 0, fmt.Errorf("compute: V4.1 routed SwiGLU gate/up widths %d/%d are empty, differ, or exceed the shader range", count, up.Shape[0])
	}
	if math.IsNaN(float64(limit)) || math.IsInf(float64(limit), 0) || math.IsNaN(float64(weight)) || math.IsInf(float64(weight), 0) {
		return 0, fmt.Errorf("compute: V4.1 routed SwiGLU limit %g and weight %g must be finite", limit, weight)
	}
	return count, nil
}

// decodeV41RoutedSwiGLUStatus consumes the four little-endian status words:
// magic, count, lowest activation fault, lowest down-operand fault. Freshness
// comes from the native all-ones fill recorded before every dispatch.
func decodeV41RoutedSwiGLUStatus(words [4]uint32, count int) error {
	if words[0] != v41RoutedSwiGLUStatusMagic || count <= 0 || count > math.MaxInt32 || words[1] != uint32(count) {
		return fmt.Errorf("compute: V4.1 routed SwiGLU status magic or count does not match the issued operation")
	}
	for _, slot := range []struct {
		word  uint32
		stage V41RoutedSwiGLUStage
	}{{words[2], V41RoutedSwiGLUStageActivation}, {words[3], V41RoutedSwiGLUStageDownOperand}} {
		if slot.word == math.MaxUint32 {
			continue
		}
		if slot.word >= uint32(count) {
			return fmt.Errorf("compute: V4.1 routed SwiGLU fault lane %d exceeds the issued count %d", slot.word, count)
		}
		return &V41RoutedSwiGLUFault{Stage: slot.stage, Index: int(slot.word)}
	}
	return nil
}
