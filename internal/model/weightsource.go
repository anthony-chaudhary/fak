package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// weightsource.go — exposes a loaded Model's weights through the compute.WeightSource
// seam (compute.go §WeightSource: Weight(name, want Dtype) (Tensor, error)). This is the
// type-level closure of the "eager full-RAM residency" assumption for the loader: a
// backend asks for a weight BY NAME at the dtype it wants, instead of reaching into a
// single host-resident f32 blob. A quant-on-load Model (LoadSafetensorsQuant{,Dir}) holds
// the big matmul weights ONLY as Q8_0 — it never materialized the f32 — so a Q8_0 request
// here is served straight from q8w at the quantized footprint, and an f32 request for a
// dropped weight fails loudly rather than silently re-inflating it.

// ModelWeightSource adapts a Model to compute.WeightSource. It is read-only over the
// loaded Model; the Tensors it returns are host views (zero-copy for f32, the prebuilt
// Q8_0 codes/scales for quantized weights), tagged with the backend that will consume
// them so Upload can stage/narrow them.
type ModelWeightSource struct {
	m  *Model
	be compute.Backend
}

// WeightSource returns a compute.WeightSource view of the model's weights, served to be
// (or the registered Default reference backend when be is nil). The returned value
// satisfies compute.WeightSource; the loader stays the sole place weights are named.
func (m *Model) WeightSource(be compute.Backend) *ModelWeightSource {
	if be == nil {
		be = compute.Default()
	}
	return &ModelWeightSource{m: m, be: be}
}

// Weight returns the named weight as a compute.Tensor at the requested dtype, drawing from
// whichever resident form the loader kept. The contract (compute.WeightSource):
//
//   - want == Q8_0: serve the prebuilt Q8_0 tensor from q8w if present (the quant-on-load
//     footprint — int8 codes + per-block f32 scales, no f32 copy). If the model was loaded
//     f32-resident (q8w not built for this name) the f32 is quantized on demand.
//   - want == F32 (or F16/BF16, widened-at-load today): serve the zero-copy f32 view from
//     the packed blob. A weight whose f32 was DROPPED at quant-on-load (isQuantWeight) has
//     no f32 to serve — that is the memory win, so this errors rather than re-inflating.
//
// A name absent from both stores is an error, never a panic, so a backend probing for an
// optional tensor (e.g. a bias) gets a value it can branch on.
func (s *ModelWeightSource) Weight(name string, want compute.Dtype) (compute.Tensor, error) {
	m := s.m
	switch want {
	case compute.Q8_0:
		if qt, ok := m.q8w[name]; ok {
			return s.q8Tensor(qt), nil
		}
		if meta, ok := m.manifest[name]; ok {
			if len(meta.Shape) != 2 {
				return compute.Tensor{}, fmt.Errorf("model: weight %s is not 2-D, cannot quantize to Q8_0", name)
			}
			qt := quantizeQ8(m.tensor(name), meta.Shape[0], meta.Shape[1])
			return s.q8Tensor(qt), nil
		}
		return compute.Tensor{}, fmt.Errorf("model: weight %s not found", name)
	case compute.F32, compute.F16, compute.BF16:
		if meta, ok := m.manifest[name]; ok {
			return compute.NewF32(s.be, append([]int(nil), meta.Shape...), m.tensor(name)), nil
		}
		if _, ok := m.q8w[name]; ok {
			return compute.Tensor{}, fmt.Errorf("model: weight %s is Q8_0-only (f32 dropped at quant-on-load); request want=Q8_0", name)
		}
		return compute.Tensor{}, fmt.Errorf("model: weight %s not found", name)
	default:
		return compute.Tensor{}, fmt.Errorf("model: WeightSource cannot serve weight %s as dtype %s", name, want)
	}
}

// q8Tensor wraps a prebuilt q8Tensor as a compute Q8_0 Tensor. The codes/scales are shared
// (no copy): an immutable loaded weight is never mutated, so the zero-copy view is safe and
// keeps the resident footprint at the quantized size.
func (s *ModelWeightSource) q8Tensor(qt *q8Tensor) compute.Tensor {
	return compute.NewQ8(s.be, []int{qt.out, qt.in}, qt.q, qt.d, qBlk)
}

// static assertion: *ModelWeightSource satisfies the compute.WeightSource seam.
var _ compute.WeightSource = (*ModelWeightSource)(nil)
