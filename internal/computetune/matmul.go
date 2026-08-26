package computetune

import (
	"context"
	"errors"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// MatMulInputs resolves the resident tensors associated with a replay profile.
// Keeping inputs outside Profile makes traces replayable without embedding model data.
type MatMulInputs func(Profile) (weight, input compute.Tensor, err error)

// MatMulImplementation adapts a real compute Backend MatMul implementation to the
// offline candidate contract. Run performs the actual operation and reads its result.
type MatMulImplementation struct {
	CandidateID string
	Backend     compute.Backend
	Inputs      MatMulInputs
}

func (c MatMulImplementation) ID() string { return c.CandidateID }

func (c MatMulImplementation) Run(_ context.Context, profile Profile) ([]float32, error) {
	if profile.Operation != OpMatMul {
		return nil, errors.New("computetune: candidate only supports matmul")
	}
	if c.Backend == nil || c.Inputs == nil {
		return nil, errors.New("computetune: incomplete matmul candidate")
	}
	weight, input, err := c.Inputs(profile)
	if err != nil {
		return nil, err
	}
	output := c.Backend.MatMul(weight, input)
	defer c.Backend.Free(output)
	return c.Backend.Read(output), nil
}
