package model

import "fmt"

// v4ExpertTransport is the fail-closed boundary between deterministic expert
// placement and a rank transport. Production leaves it nil until a transport
// is explicitly configured.
type v4ExpertTransport interface {
	Forward(dispatch map[int][]V4ExpertDispatch, evaluate func([]routePick) ([]float32, error)) (output []float32, partials int, err error)
}

// v4LoopbackExpertTransport exercises the real dispatch/evaluate/recombine
// seam deterministically in one process. It is a parity witness, not a claim
// of inter-process or NCCL transport.
type v4LoopbackExpertTransport struct{}

func (v4LoopbackExpertTransport) Forward(dispatch map[int][]V4ExpertDispatch, evaluate func([]routePick) ([]float32, error)) ([]float32, int, error) {
	var output []float32
	partials := 0
	for rank := 0; rank < len(dispatch); rank++ {
		work, ok := dispatch[rank]
		if !ok {
			return nil, partials, fmt.Errorf("%w: dispatch missing rank %d", ErrV4ExpertPlacement, rank)
		}
		if len(work) == 0 {
			continue
		}
		picks := make([]routePick, len(work))
		for i, item := range work {
			if item.Rank != rank {
				return nil, partials, fmt.Errorf("%w: dispatch rank %d contains rank %d", ErrV4ExpertPlacement, rank, item.Rank)
			}
			picks[i] = routePick{expert: item.Expert, weight: item.Weight}
		}
		partial, err := evaluate(picks)
		if err != nil {
			return nil, partials, err
		}
		if output == nil {
			output = make([]float32, len(partial))
		} else if len(partial) != len(output) {
			return nil, partials, fmt.Errorf("%w: partial width %d, want %d", ErrV4ExpertPlacement, len(partial), len(output))
		}
		for i := range output {
			output[i] += partial[i]
		}
		partials++
	}
	if output == nil {
		return nil, partials, fmt.Errorf("%w: empty dispatch", ErrV4ExpertPlacement)
	}
	return output, partials, nil
}
