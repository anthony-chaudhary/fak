//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
void *mg_gdn_graph_encode_batch(void *graph, const int *owners, int batch,
    void *mixed, void *z, void *b, void *a,
    const float *conv, const float *alog, const float *dtbias, const float *norm,
    int nk, int nv, int khd, int vhd, int kernel, float eps);
*/
import "C"

import (
	"errors"
	"fmt"
	"sort"
	"unsafe"
)

const (
	Qwen35DecodeBatchMin = 2
	Qwen35DecodeBatchMax = 8
)

// Qwen35DecodeBatchRequest describes B independent P=1 decode rows. Input is
// packed row-major; Panel supplies the layer constants shared by every row.
// States are independent session owners, never positions in one token scan.
type Qwen35DecodeBatchRequest struct {
	Input                          []float32
	Weights                        Qwen35DecodeWeights
	States                         []*GDNState
	Panel                          GDNPanel
	InjectPostSubmitFailureForTest bool
	afterFirstOwnerRetainForTest   func()
}

// Qwen35DecodeBatchReceipt binds an accepted panel to its shared projection,
// independent-state, transfer, and terminal-fence topology.
type Qwen35DecodeBatchReceipt struct {
	Rows, OwnerCount                                             int
	CommandBuffers, Commits, CompletionWaits                     int
	ProjectionDispatches, Quantizers, GDNDispatches, GDNEncoders int
	InputUploads, FinalReadbacks, IntermediateReadbacks          int
	StateH2DTransfers, StateD2HTransfers, ReplayAttempts         int
	OwnerRetains, OwnerReleases                                  int
	Encoders                                                     int
	Committed, CompletedWait                                     bool
}

// RunQwen35DecodeBatch executes B=2..8 independent Qwen linear-attention
// mixers in one caller-owned command buffer. The projection GEMMs share their
// weight dispatch across all rows, while each row mutates exactly one distinct
// resident GDN owner. Once committed, failures are accepted and must not be
// replayed by a caller because some or all resident owners may have advanced.
func RunQwen35DecodeBatch(req Qwen35DecodeBatchRequest) (out [][]float32, receipt Qwen35DecodeBatchReceipt, accepted bool, err error) {
	batch, hidden, geometry, err := validateQwen35DecodeBatchRequest(req)
	if err != nil {
		return nil, Qwen35DecodeBatchReceipt{}, false, err
	}
	weights := []*Q8Weight{req.Weights.InQKV, req.Weights.InZ, req.Weights.InB, req.Weights.InA, req.Weights.Out}
	for _, weight := range weights {
		if weight == nil {
			return nil, Qwen35DecodeBatchReceipt{}, false, &GDNDeclinedError{Reason: "missing resident Q8 decode weight"}
		}
	}
	for _, weight := range weights {
		weight.mu.RLock()
	}
	defer func() {
		for i := len(weights) - 1; i >= 0; i-- {
			weights[i].mu.RUnlock()
		}
	}()
	if err := validateLockedQwen35DecodeWeights(req.Weights, hidden, geometry); err != nil {
		return nil, Qwen35DecodeBatchReceipt{}, false, err
	}

	graph, err := BeginProjectionGraph(req.Input, nil, nil, batch, hidden)
	if err != nil {
		return nil, Qwen35DecodeBatchReceipt{}, false, err
	}
	defer graph.Free()
	if req.InjectPostSubmitFailureForTest {
		graph.InjectPostSubmitFailureForTest()
	}
	input, err := graph.Input(hidden)
	if err != nil {
		return nil, receipt, false, err
	}
	quantizedInput, err := graph.QuantizeQ8(input)
	if err != nil {
		return nil, receipt, false, err
	}
	projected := make([]*GraphResult, 4)
	for i, weight := range weights[:4] {
		projected[i], err = graph.EncodeQ8From(weight, quantizedInput)
		if err != nil {
			return nil, receipt, false, err
		}
	}
	core, err := graph.encodeGDNBatch(req.States, projected[0], projected[1], projected[2], projected[3], req.Panel, req.afterFirstOwnerRetainForTest)
	if err != nil {
		return nil, receipt, false, err
	}
	quantizedCore, err := graph.QuantizeQ8(core)
	if err != nil {
		return nil, receipt, false, err
	}
	result, err := graph.EncodeQ8From(req.Weights.Out, quantizedCore)
	if err != nil {
		return nil, receipt, false, err
	}
	outputs, graphReceipt, err := graph.FinishRead(result)
	accepted = graphReceipt.Committed
	releases := 0
	if graphReceipt.Committed {
		releases = batch
	}
	receipt = Qwen35DecodeBatchReceipt{
		Rows:           batch,
		OwnerCount:     batch,
		CommandBuffers: 1, Commits: decodeBoolInt(graphReceipt.Committed), CompletionWaits: decodeBoolInt(graphReceipt.CompletedWait),
		ProjectionDispatches: 5, Quantizers: 2, GDNDispatches: 3 * batch, GDNEncoders: 1,
		InputUploads: 1, FinalReadbacks: graphReceipt.HostReadbacks,
		OwnerRetains: batch, OwnerReleases: releases,
		Encoders: graphReceipt.Encoders, Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait,
	}
	if err != nil {
		return nil, receipt, accepted, err
	}
	if len(outputs) != 1 || len(outputs[0]) != batch*req.Weights.Out.Out || !receipt.Committed || !receipt.CompletedWait ||
		receipt.Encoders != 8 || receipt.FinalReadbacks != 1 || receipt.OwnerReleases != batch {
		return nil, receipt, true, fmt.Errorf("metalgemm: incomplete Qwen batch decode receipt: %+v", receipt)
	}
	packed := outputs[0]
	out = make([][]float32, batch)
	for row := range out {
		out[row] = packed[row*req.Weights.Out.Out : (row+1)*req.Weights.Out.Out]
	}
	return out, receipt, true, nil
}

func validateQwen35DecodeBatchRequest(req Qwen35DecodeBatchRequest) (batch, hidden int, geometry GDNGeometry, err error) {
	batch = len(req.States)
	if batch < Qwen35DecodeBatchMin || batch > Qwen35DecodeBatchMax {
		return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode batch=%d, want [%d,%d]", batch, Qwen35DecodeBatchMin, Qwen35DecodeBatchMax)}
	}
	if len(req.Input) == 0 || len(req.Input)%batch != 0 {
		return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: "decode input must contain one equal-width row per owner"}
	}
	hidden = len(req.Input) / batch
	if hidden == 0 || hidden%32 != 0 {
		return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: "decode row width must be a non-empty multiple of 32"}
	}
	if req.Panel.Tokens != 0 && req.Panel.Tokens != 1 {
		return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode tokens=%d, want 1", req.Panel.Tokens)}
	}
	seenStates := make(map[*GDNState]struct{}, batch)
	seenHandles := make(map[GDNStateHandle]struct{}, 2*batch)
	for row, state := range req.States {
		if state == nil {
			return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("nil decode owner at row %d", row)}
		}
		if _, duplicate := seenStates[state]; duplicate {
			return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode owner reused at row %d", row)}
		}
		seenStates[state] = struct{}{}
		rowGeometry, geometryErr := state.graphGeometry()
		if geometryErr != nil {
			return 0, 0, GDNGeometry{}, geometryErr
		}
		if row == 0 {
			geometry = rowGeometry
		} else if rowGeometry != geometry {
			return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode owner geometry differs at row %d", row)}
		}
		conv, recurrent := state.Handles()
		if conv == 0 || recurrent == 0 || conv == recurrent {
			return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("invalid decode owner handles at row %d", row)}
		}
		for _, handle := range []GDNStateHandle{conv, recurrent} {
			if _, duplicate := seenHandles[handle]; duplicate {
				return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode owner backing aliases at row %d", row)}
			}
			seenHandles[handle] = struct{}{}
		}
	}
	if geometry.valueDim()%32 != 0 {
		return 0, 0, GDNGeometry{}, &GDNDeclinedError{Reason: "decode value width must be a multiple of 32"}
	}
	if err := validateGDNConstants(geometry, req.Panel); err != nil {
		return 0, 0, GDNGeometry{}, err
	}
	return batch, hidden, geometry, nil
}

func (g *ProjectionGraph) encodeGDNBatch(states []*GDNState, mixed, z, b, a *GraphResult, panel GDNPanel, afterFirstRetainForTest func()) (*GraphResult, error) {
	batch := len(states)
	if g == nil || g.p != batch || batch < Qwen35DecodeBatchMin || batch > Qwen35DecodeBatchMax {
		return nil, &GDNDeclinedError{Reason: "Qwen batch decode GDN requires B=2..8"}
	}
	geometry, err := states[0].graphGeometry()
	if err != nil {
		return nil, err
	}
	wants := []struct {
		result *GraphResult
		width  int
	}{{mixed, geometry.convDim()}, {z, geometry.valueDim()}, {b, geometry.NumValueHeads}, {a, geometry.NumValueHeads}}
	for _, want := range wants {
		if want.result == nil || want.result.graph != g || want.result.ptr == nil || want.result.p != batch || want.result.out != want.width {
			return nil, &GDNDeclinedError{Reason: "invalid graph-owned batch GDN operand"}
		}
	}
	// Every overlapping call acquires owners by the same stable native-handle
	// identity. Caller row order remains encoded in owners[row], but can never
	// create the [A,B] versus [B,A] hold-and-wait cycle.
	type acquisition struct {
		row       int
		state     *GDNState
		conv      GDNStateHandle
		recurrent GDNStateHandle
	}
	order := make([]acquisition, batch)
	for row, state := range states {
		conv, recurrent := state.Handles()
		if conv == 0 || recurrent == 0 {
			return nil, &GDNDeclinedError{Reason: fmt.Sprintf("decode owner closed before retain at row %d", row)}
		}
		order[row] = acquisition{row: row, state: state, conv: conv, recurrent: recurrent}
	}
	sort.Slice(order, func(i, j int) bool {
		if order[i].conv != order[j].conv {
			return order[i].conv < order[j].conv
		}
		return order[i].recurrent < order[j].recurrent
	})
	owners := make([]C.int, batch)
	retained := make([]gdnGraphLease, 0, batch)
	releaseRetained := func() {
		for i := len(retained) - 1; i >= 0; i-- {
			retained[i].state.releaseGraph(retained[i].done)
		}
	}
	for acquired, entry := range order {
		owner, done, retainErr := entry.state.retainGraph()
		if retainErr != nil {
			releaseRetained()
			return nil, retainErr
		}
		owners[entry.row] = owner
		retained = append(retained, gdnGraphLease{state: entry.state, done: done})
		if acquired == 0 && afterFirstRetainForTest != nil {
			afterFirstRetainForTest()
		}
	}
	ptr := C.mg_gdn_graph_encode_batch(g.ptr, (*C.int)(unsafe.Pointer(&owners[0])), C.int(batch),
		mixed.ptr, z.ptr, b.ptr, a.ptr,
		gdnF32(panel.Conv1D), gdnF32(panel.ALog), gdnF32(panel.DTBias), gdnF32(panel.Norm),
		C.int(geometry.NumKeyHeads), C.int(geometry.NumValueHeads), C.int(geometry.KeyHeadDim), C.int(geometry.ValueHeadDim), C.int(geometry.ConvKernel), C.float(panel.RMSNormEpsilon))
	if ptr == nil {
		releaseRetained()
		return nil, errors.New("metalgemm: Qwen batch decode GDN graph encode failed")
	}
	g.gdnLeases = append(g.gdnLeases, retained...)
	g.encoders++
	return &GraphResult{ptr: ptr, out: geometry.valueDim(), p: batch, graph: g}, nil
}
