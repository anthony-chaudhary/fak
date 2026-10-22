package compute

import (
	"errors"
	"fmt"
	"reflect"
)

const Qwen35IndependentDecodeBatchPath = "qwen35/vulkan-independent-decode-batch-v1"

// Qwen35IndependentDecodeBatchRequest advances independent one-token lanes.
// Each lane owns its KV and recurrent state; immutable resident weights must be
// identical so the backend can stream every projection once for the lane panel.
type Qwen35IndependentDecodeBatchRequest struct {
	Path  string
	Lanes []Qwen35SequencePrefillRequest
}

type Qwen35IndependentDecodeBatchResult struct {
	Logits                 Tensor
	Lanes                  int
	SharedProjectionPanels int
	ProjectionMACs         int64
	SerialFallbacks        int
	Transfers              Qwen35SequenceTransferCounters
}

type Qwen35IndependentDecodeBatchBackend interface {
	Qwen35IndependentDecodeBatchPath() string
	Qwen35IndependentDecodeBatch(Qwen35IndependentDecodeBatchRequest) (Qwen35IndependentDecodeBatchResult, error)
}

type Qwen35IndependentDecodeBatchError struct {
	Stage   string
	Mutated bool
	Cause   error
}

func (e *Qwen35IndependentDecodeBatchError) Error() string {
	if e == nil {
		return "compute: nil Qwen3.5 independent decode batch error"
	}
	return fmt.Sprintf("compute: Qwen3.5 independent decode batch failed at %s (mutated=%t): %v", e.Stage, e.Mutated, e.Cause)
}

func (e *Qwen35IndependentDecodeBatchError) Unwrap() error { return e.Cause }

func Qwen35IndependentDecodeBatchMutated(err error) bool {
	var batchErr *Qwen35IndependentDecodeBatchError
	return errors.As(err, &batchErr) && batchErr.Mutated
}

func validateQwen35IndependentDecodeBatchContract(req Qwen35IndependentDecodeBatchRequest) error {
	if req.Path != Qwen35IndependentDecodeBatchPath {
		return &Qwen35IndependentDecodeBatchError{Stage: "request", Cause: errors.New("wrong capability path")}
	}
	if len(req.Lanes) < 2 || len(req.Lanes) > 8 {
		return &Qwen35IndependentDecodeBatchError{Stage: "request", Cause: errors.New("lane count must be in [2,8]")}
	}
	first := req.Lanes[0]
	kvOwners := make([]any, 0, len(req.Lanes))
	for i, lane := range req.Lanes {
		if lane.Path != Qwen35SequencePrefillPath || len(lane.TokenIDs) != 1 || !lane.NeedLogits || lane.NeedAllLogits || lane.CapturePrefixReplay || lane.CaptureRawHidden || len(lane.FeatureLayers) != 0 {
			return &Qwen35IndependentDecodeBatchError{Stage: "lane-contract", Cause: fmt.Errorf("lane %d is not an ordinary one-token decode", i)}
		}
		if lane.Hidden != first.Hidden || lane.Intermediate != first.Intermediate || lane.NumHeads != first.NumHeads || lane.NumKVHeads != first.NumKVHeads || lane.HeadDim != first.HeadDim || lane.RotaryDim != first.RotaryDim || lane.NumKeyHeads != first.NumKeyHeads || lane.NumValueHeads != first.NumValueHeads || lane.KeyHeadDim != first.KeyHeadDim || lane.ValueHeadDim != first.ValueHeadDim || lane.ConvKernel != first.ConvKernel || lane.RMSNormEpsilon != first.RMSNormEpsilon || !equalQwen35DecodeRoPE(lane.RoPEThetaForLayer, first.RoPEThetaForLayer) || len(lane.Layers) != len(first.Layers) || len(lane.States) != len(lane.Layers) {
			return &Qwen35IndependentDecodeBatchError{Stage: "lane-geometry", Cause: fmt.Errorf("lane %d geometry differs", i)}
		}
		if lane.KV == nil {
			return &Qwen35IndependentDecodeBatchError{Stage: "lane-ownership", Cause: fmt.Errorf("lane %d has no KV owner", i)}
		}
		for previous, owner := range kvOwners {
			if sameQwen35DecodeOwner(owner, lane.KV) {
				return &Qwen35IndependentDecodeBatchError{Stage: "lane-ownership", Cause: fmt.Errorf("lane %d aliases lane %d KV owner", i, previous)}
			}
		}
		kvOwners = append(kvOwners, lane.KV)
	}
	return nil
}

func equalQwen35DecodeRoPE(a, b []float64) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sameQwen35DecodeOwner(a, b any) bool {
	av, bv := reflect.ValueOf(a), reflect.ValueOf(b)
	return av.IsValid() && bv.IsValid() && av.Type() == bv.Type() && av.Comparable() && av.Interface() == bv.Interface()
}
