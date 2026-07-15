package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// v4CollectiveExpertTransport binds V4 rank dispatch to the existing device
// collective seam. Each rank evaluates only its owned picks; the collective
// sums those rank-local partials in one deterministic reduction.
type v4CollectiveExpertTransport struct {
	backend   compute.Backend
	placement V4ExpertPlacement
}

func newV4CollectiveExpertTransport(be compute.Backend, placement V4ExpertPlacement) (*v4CollectiveExpertTransport, error) {
	if be == nil || placement.WorldSize <= 0 || placement.Rank < 0 || placement.Rank >= placement.WorldSize {
		return nil, fmt.Errorf("%w: invalid collective transport", ErrV4ExpertPlacement)
	}
	if !be.Caps().Collective {
		return nil, fmt.Errorf("%w: backend %q has no collective capability", ErrV4ExpertPlacement, be.Name())
	}
	return &v4CollectiveExpertTransport{backend: be, placement: placement}, nil
}

func (t *v4CollectiveExpertTransport) Forward(dispatch map[int][]V4ExpertDispatch, evaluate func([]routePick) ([]float32, error)) ([]float32, int, error) {
	if t == nil || t.backend == nil {
		return nil, 0, fmt.Errorf("%w: nil collective transport", ErrV4ExpertPlacement)
	}
	parts := make([]compute.Tensor, t.placement.WorldSize)
	allocated := make([]bool, len(parts))
	defer func() {
		for i := range parts {
			if allocated[i] {
				t.backend.Free(parts[i])
			}
		}
	}()
	width := -1
	active := 0
	for rank := 0; rank < t.placement.WorldSize; rank++ {
		work := dispatch[rank]
		picks := make([]routePick, len(work))
		for i, item := range work {
			if item.Rank != rank {
				return nil, 0, fmt.Errorf("%w: dispatch rank %d contains rank %d", ErrV4ExpertPlacement, rank, item.Rank)
			}
			picks[i] = routePick{expert: item.Expert, weight: item.Weight}
		}
		if len(picks) == 0 {
			continue
		}
		partial, err := evaluate(picks)
		if err != nil {
			return nil, 0, err
		}
		if width < 0 {
			width = len(partial)
		} else if len(partial) != width {
			return nil, 0, fmt.Errorf("%w: rank %d partial width %d, want %d", ErrV4ExpertPlacement, rank, len(partial), width)
		}
		parts[rank] = t.backend.Upload(compute.NewF32(t.backend, []int{len(partial)}, partial), compute.F32)
		allocated[rank] = true
		active++
	}
	if width <= 0 {
		return nil, 0, fmt.Errorf("%w: empty collective dispatch", ErrV4ExpertPlacement)
	}
	for rank := range parts {
		if allocated[rank] {
			continue
		}
		parts[rank] = t.backend.Upload(compute.NewF32(t.backend, []int{width}, make([]float32, width)), compute.F32)
		allocated[rank] = true
	}
	collective, ok := t.backend.(compute.CollectiveBackend)
	if !ok {
		return nil, 0, fmt.Errorf("%w: backend %q advertises collective without implementation", ErrV4ExpertPlacement, t.backend.Name())
	}
	reduced, err := collective.AllReduceSum(parts)
	if err != nil {
		return nil, 0, fmt.Errorf("%w: all-reduce: %v", ErrV4ExpertPlacement, err)
	}
	defer t.backend.Free(reduced)
	out := t.backend.Read(reduced)
	if len(out) != width {
		return nil, 0, fmt.Errorf("%w: reduced width %d, want %d", ErrV4ExpertPlacement, len(out), width)
	}
	return out, active, nil
}
