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

// uploadRank places rank r's partial on rank r's device when the backend exposes the
// optional compute.RankUploader seam, exactly as BackendCollective.uploadRankF32 does.
// This is not cosmetic: a real device collective validates that parts[r] is RESIDENT on
// device r (the CUDA AllReduceSum rejects "rank r tensor is resident on device 0, want
// device r"), so the generic device-0 Upload cannot drive a multi-GPU reduction. On
// cpu-ref — which has no RankUploader — Upload is the identity and this is unchanged.
func (t *v4CollectiveExpertTransport) uploadRank(rank int, data []float32) (compute.Tensor, error) {
	host := compute.NewF32(t.backend, []int{len(data)}, data)
	if up, ok := t.backend.(compute.RankUploader); ok {
		return up.UploadRank(host, compute.F32, rank)
	}
	return t.backend.Upload(host, compute.F32), nil
}

// placeRankPart uploads one rank's partial through uploadRank, records it in parts[rank]
// and marks the slot allocated so the deferred Free reclaims it. `what` names the upload in
// the error text ("upload" for a rank that actually had picks, "zero-fill upload" for the
// padding rank), so both callers keep the exact message they printed when each carried its
// own copy of this block.
func (t *v4CollectiveExpertTransport) placeRankPart(parts []compute.Tensor, allocated []bool, rank int, partial []float32, what string) error {
	up, err := t.uploadRank(rank, partial)
	if err != nil {
		return fmt.Errorf("%w: rank %d %s: %v", ErrV4ExpertPlacement, rank, what, err)
	}
	parts[rank] = up
	allocated[rank] = true
	return nil
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
		if err := t.placeRankPart(parts, allocated, rank, partial, "upload"); err != nil {
			return nil, 0, err
		}
		active++
	}
	if width <= 0 {
		return nil, 0, fmt.Errorf("%w: empty collective dispatch", ErrV4ExpertPlacement)
	}
	for rank := range parts {
		if allocated[rank] {
			continue
		}
		if err := t.placeRankPart(parts, allocated, rank, make([]float32, width), "zero-fill upload"); err != nil {
			return nil, 0, err
		}
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
