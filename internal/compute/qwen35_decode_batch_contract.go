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

// sameQwen35DecodeLaneWeights reports whether two lanes name the same resident layer,
// output-norm, and head buffers, so one projection stream can serve both. Embedding identity
// is the caller's: a lane carrying its own uploaded embedding row (TokenEmbeddingRows)
// legitimately differs there while sharing every weight.
func sameQwen35DecodeLaneWeights(a, b Qwen35SequencePrefillRequest) bool {
	if a.OutputNorm.Buf() != b.OutputNorm.Buf() || a.Output.Buf() != b.Output.Buf() || len(a.Layers) != len(b.Layers) {
		return false
	}
	for i := range a.Layers {
		x, y := a.Layers[i], b.Layers[i]
		if x.Linear != y.Linear || x.GDNOutTiled != y.GDNOutTiled {
			return false
		}
		for _, pair := range [][2]Tensor{{x.InputNorm, y.InputNorm}, {x.PostNorm, y.PostNorm}, {x.Q, y.Q}, {x.K, y.K}, {x.V, y.V}, {x.O, y.O}, {x.QNorm, y.QNorm}, {x.KNorm, y.KNorm}, {x.GDNInQKV, y.GDNInQKV}, {x.GDNInZ, y.GDNInZ}, {x.GDNInB, y.GDNInB}, {x.GDNInA, y.GDNInA}, {x.GDNConv, y.GDNConv}, {x.GDNALog, y.GDNALog}, {x.GDNDTBias, y.GDNDTBias}, {x.GDNNorm, y.GDNNorm}, {x.GDNOut, y.GDNOut}, {x.Gate, y.Gate}, {x.Up, y.Up}, {x.Down, y.Down}} {
			if pair[0].Buf() != pair[1].Buf() {
				return false
			}
		}
	}
	return true
}

// Qwen35IndependentDecodeBatchProjectionWork is the shared-panel receipt a batch backend
// reports: one panel per projection weight plus the head, and lanes x their MACs.
func Qwen35IndependentDecodeBatchProjectionWork(req Qwen35SequencePrefillRequest, lanes int) (int, int64) {
	panels := 1
	macs := int64(req.Output.Numel())
	for _, layer := range req.Layers {
		weights := []Tensor{layer.Gate, layer.Up, layer.Down}
		if layer.Linear {
			weights = append(weights, layer.GDNInQKV, layer.GDNInZ, layer.GDNInB, layer.GDNInA, layer.GDNOut)
		} else {
			weights = append(weights, layer.Q, layer.K, layer.V, layer.O)
		}
		panels += len(weights)
		for _, weight := range weights {
			macs += int64(weight.Numel())
		}
	}
	return panels, int64(lanes) * macs
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
