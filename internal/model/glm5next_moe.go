package model

import (
	"fmt"

	"github.com/anthony-chaudhary/fak/internal/compute/slotstream"
)

// GLM5NextSlotStreamPlan describes the preallocated expert-slot geometry for a
// GLM5Next MoE layer. It binds the 288-expert / 54-resident-slot topology to the
// generic slotstream engine so the forward pass pages only the top-k routed
// experts instead of requiring all 288 experts resident.
type GLM5NextSlotStreamPlan struct {
	NumLayers     int
	SlotsPerLayer int
	Pool          *slotstream.MultiTensorSlotPool
	Engine        *slotstream.StreamEngine
}

// NewGLM5NextSlotStreamPlan builds the default GLM5Next slot geometry: the
// 288-expert routed topology served from 54 preallocated slots per MoE layer,
// with cold tail experts streamed into a slot on demand.
//
// numMoELayers is the count of sparse MoE layers that carry routed experts
// (GLM-5.3-Flash has 42: layers 3..44). A non-positive count falls back to the
// canonical 42-layer MoE cadence.
func NewGLM5NextSlotStreamPlan(numMoELayers int) (*GLM5NextSlotStreamPlan, error) {
	if numMoELayers <= 0 {
		numMoELayers = 42
	}
	pool, err := slotstream.NewMultiTensorSlotPool(numMoELayers, slotstream.DefaultGLM5NextSlotsPerLayer)
	if err != nil {
		return nil, fmt.Errorf("glm5next slotstream: %w", err)
	}
	return &GLM5NextSlotStreamPlan{
		NumLayers:     numMoELayers,
		SlotsPerLayer: slotstream.DefaultGLM5NextSlotsPerLayer,
		Pool:          pool,
		Engine:        slotstream.NewStreamEngine(pool),
	}, nil
}

// GLM5NextExpertSource resolves the serialized weight bytes for one routed
// expert of one MoE layer. It is the seam between the model's expert store and
// the slotstream pager: the plan calls it only for slot misses, so a warm slot
// never re-reads the tail expert.
type GLM5NextExpertSource interface {
	// ExpertBytes returns the serialized expert record for (layer, expert).
	// The returned slice length must equal slotstream.ExpertChunkBytes.
	ExpertBytes(layer, expert int) ([]byte, error)
}

// GLM5NextExpertCodec converts a serialized slot buffer into f32 SwiGLU weights
// for the requested (inDim, interDim) geometry. Decoding happens from the
// resident slot buffer, never from the original source, so the measured path is
// the one that actually ran.
type GLM5NextExpertCodec interface {
	Decode(slotData []byte, inDim, interDim int) (GLM5NextExpertWeight, error)
}

// GLM5NextStreamedMoEResult carries the executed output plus truthful paging
// counters for the layer's top-k routed experts.
type GLM5NextStreamedMoEResult struct {
	Out    []float32
	Hits   int // routed experts already resident in a slot
	Misses int // routed experts paged in from the source this call
	Slots  int // preallocated slots per layer (the working-set ceiling)
	Layer  int
}

// ExecuteGLM5NextStreamedMoE evaluates the shared expert plus the top-k routed
// experts whose weights are paged through the slot pool.
//
// Only the routed experts named by route are materialized. Each is acquired from
// the pool (a resident slot is a hit; otherwise the LRU slot is evicted and the
// expert's bytes streamed in from source), decoded from the resident slot via
// codec, and folded into the output with its router weight. This is the
// memory-bounded path: the resident working set is SlotsPerLayer * chunk bytes
// rather than NumExperts * chunk bytes.
func ExecuteGLM5NextStreamedMoE(
	layer int,
	x []float32,
	route GLM5NextMoERouteResult,
	shared GLM5NextExpertWeight,
	inDim, interDim int,
	plan *GLM5NextSlotStreamPlan,
	source GLM5NextExpertSource,
	codec GLM5NextExpertCodec,
	now int64,
) (GLM5NextStreamedMoEResult, error) {
	res := GLM5NextStreamedMoEResult{Out: make([]float32, inDim), Layer: layer}
	if plan == nil || plan.Engine == nil {
		return res, fmt.Errorf("glm5next slotstream: nil plan or engine")
	}
	res.Slots = plan.SlotsPerLayer

	// 1. Shared expert stays resident; it is never paged.
	if len(shared.WAct) > 0 {
		sharedOut := executeSwiGLU(x, shared.WAct, shared.WUp, shared.WDown, inDim, interDim)
		for i := 0; i < inDim; i++ {
			res.Out[i] += sharedOut[i]
		}
	}

	// 2. Routed experts: page each into a slot before executing.
	for i, expIdx := range route.ExpertIndices {
		if i >= len(route.Weights) {
			break
		}
		weight := route.Weights[i]
		if weight == 0 {
			continue
		}
		slot, hit := plan.Pool.AcquireSlot(layer, expIdx, now)
		if slot == nil {
			return res, fmt.Errorf("glm5next slotstream: no slot for layer %d expert %d", layer, expIdx)
		}
		if hit {
			res.Hits++
		} else {
			res.Misses++
			if source != nil {
				data, err := source.ExpertBytes(layer, expIdx)
				if err != nil {
					return res, fmt.Errorf("glm5next slotstream: source layer %d expert %d: %w", layer, expIdx, err)
				}
				if len(data) != len(slot.Data) {
					return res, fmt.Errorf("glm5next slotstream: expert record %d bytes, want %d", len(data), len(slot.Data))
				}
				copy(slot.Data, data)
			}
			slot.Resident = true
		}

		if codec == nil {
			continue
		}
		exp, err := codec.Decode(slot.Data, inDim, interDim)
		if err != nil {
			return res, fmt.Errorf("glm5next slotstream: decode layer %d expert %d: %w", layer, expIdx, err)
		}
		if len(exp.WAct) == 0 {
			continue
		}
		expOut := executeSwiGLU(x, exp.WAct, exp.WUp, exp.WDown, inDim, interDim)
		for j := 0; j < inDim; j++ {
			res.Out[j] += weight * expOut[j]
		}
	}

	return res, nil
}
