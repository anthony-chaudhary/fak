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
	"unsafe"
)

const Qwen35DecodeBatchMax = 8

// Qwen35DecodeBatchRequest describes B independent decode rows which share one
// resident projection set but retain one recurrent-state owner per row.
type Qwen35DecodeBatchRequest struct {
	Input                          []float32
	Weights                        Qwen35DecodeWeights
	States                         []*GDNState
	Panel                          GDNPanel
	InjectPostSubmitFailureForTest bool
}

// Qwen35DecodeBatchReceipt proves the shared submission topology. Projection
// dispatches are B-independent: the five resident weights are each consumed by
// one B-row panel dispatch.
type Qwen35DecodeBatchReceipt struct {
	Batch                                                       int
	CommandBuffers, Commits, CompletionWaits                    int
	ProjectionDispatches, Quantizers, GDNEncoders               int
	InputUploads, FinalReadbacks                                int
	IntermediateReadbacks, StateH2DTransfers, StateD2HTransfers int
	Encoders                                                    int
	Committed, CompletedWait                                    bool
}

// RunQwen35DecodeBatch executes B=2..8 independent-state linear-attention rows
// in one ProjectionGraph. Declines happen before graph submission. Once the
// command buffer commits, every retained owner is released from the graph but
// the operation must not be replayed.
func RunQwen35DecodeBatch(req Qwen35DecodeBatchRequest) (out [][]float32, receipt Qwen35DecodeBatchReceipt, accepted bool, err error) {
	batch, width, geometry, err := validateQwen35DecodeBatchRequest(req)
	if err != nil {
		return nil, receipt, false, err
	}
	weights := []*Q8Weight{req.Weights.InQKV, req.Weights.InZ, req.Weights.InB, req.Weights.InA, req.Weights.Out}
	for _, weight := range weights {
		weight.mu.RLock()
	}
	defer func() {
		for i := len(weights) - 1; i >= 0; i-- {
			weights[i].mu.RUnlock()
		}
	}()
	if err := validateLockedQwen35DecodeWeights(req.Weights, width, geometry); err != nil {
		return nil, receipt, false, err
	}

	graph, err := BeginProjectionGraph(req.Input, nil, nil, batch, width)
	if err != nil {
		return nil, receipt, false, err
	}
	defer graph.Free()
	if req.InjectPostSubmitFailureForTest {
		graph.InjectPostSubmitFailureForTest()
	}
	input, err := graph.Input(width)
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
	core, err := graph.encodeGDNBatch(req.States, projected[0], projected[1], projected[2], projected[3], req.Panel)
	if err != nil {
		return nil, receipt, false, err
	}
	quantizedCore, err := graph.QuantizeQ8(core)
	if err != nil {
		return nil, receipt, false, err
	}
	projectedOut, err := graph.EncodeQ8From(req.Weights.Out, quantizedCore)
	if err != nil {
		return nil, receipt, false, err
	}
	flat, graphReceipt, finishErr := graph.FinishRead(projectedOut)
	receipt = qwen35DecodeBatchReceipt(batch, graphReceipt)
	accepted = graphReceipt.Committed
	if finishErr != nil {
		if accepted {
			return nil, receipt, true, &GDNPostSubmitError{Reason: finishErr.Error()}
		}
		return nil, receipt, false, finishErr
	}
	if len(flat) != 1 || len(flat[0])%batch != 0 {
		return nil, receipt, true, &GDNPostSubmitError{Reason: "invalid packed batch output"}
	}
	rowWidth := len(flat[0]) / batch
	out = make([][]float32, batch)
	for row := range out {
		out[row] = append([]float32(nil), flat[0][row*rowWidth:(row+1)*rowWidth]...)
	}
	return out, receipt, true, nil
}

func validateQwen35DecodeBatchRequest(req Qwen35DecodeBatchRequest) (batch, width int, geometry GDNGeometry, err error) {
	batch = len(req.States)
	if batch < 2 || batch > Qwen35DecodeBatchMax {
		return 0, 0, geometry, &GDNDeclinedError{Reason: fmt.Sprintf("batch=%d outside supported range [2,%d]", batch, Qwen35DecodeBatchMax)}
	}
	if len(req.Input) == 0 || len(req.Input)%batch != 0 {
		return 0, 0, geometry, &GDNDeclinedError{Reason: "input is not a non-empty B-row panel"}
	}
	width = len(req.Input) / batch
	seen := make(map[*GDNState]struct{}, batch)
	for row, state := range req.States {
		if state == nil {
			return 0, 0, geometry, &GDNDeclinedError{Reason: fmt.Sprintf("row %d has nil state", row)}
		}
		if _, exists := seen[state]; exists {
			return 0, 0, geometry, &GDNDeclinedError{Reason: "state owners must be distinct"}
		}
		seen[state] = struct{}{}
		g, stateErr := state.graphGeometry()
		if stateErr != nil {
			return 0, 0, geometry, stateErr
		}
		if row == 0 {
			geometry = g
		} else if g != geometry {
			return 0, 0, geometry, &GDNDeclinedError{Reason: "state geometries differ"}
		}
	}
	if err := validateQwen35DecodePanel(req.Panel, geometry); err != nil {
		return 0, 0, geometry, err
	}
	weights := []*Q8Weight{req.Weights.InQKV, req.Weights.InZ, req.Weights.InB, req.Weights.InA, req.Weights.Out}
	for _, weight := range weights {
		if weight == nil {
			return 0, 0, geometry, &GDNDeclinedError{Reason: "missing resident Q8 decode weight"}
		}
	}
	return batch, width, geometry, nil
}

func validateQwen35DecodePanel(panel GDNPanel, geometry GDNGeometry) error {
	shapes := []struct {
		name      string
		got, want int
	}{
		{"conv1d", len(panel.Conv1D), geometry.convDim() * geometry.ConvKernel},
		{"a_log", len(panel.ALog), geometry.NumValueHeads},
		{"dt_bias", len(panel.DTBias), geometry.NumValueHeads},
		{"norm", len(panel.Norm), geometry.ValueHeadDim},
	}
	for _, shape := range shapes {
		if shape.got != shape.want {
			return &GDNDeclinedError{Reason: fmt.Sprintf("%s elements=%d, want %d", shape.name, shape.got, shape.want)}
		}
	}
	if panel.RMSNormEpsilon <= 0 {
		return &GDNDeclinedError{Reason: "RMSNorm epsilon must be positive"}
	}
	return nil
}

func (g *ProjectionGraph) encodeGDNBatch(states []*GDNState, mixed, z, b, a *GraphResult, panel GDNPanel) (*GraphResult, error) {
	if err := g.open(); err != nil {
		return nil, err
	}
	batch := len(states)
	geometry, err := states[0].graphGeometry()
	if err != nil {
		return nil, err
	}
	wants := []struct {
		r     *GraphResult
		width int
	}{{mixed, geometry.convDim()}, {z, geometry.valueDim()}, {b, geometry.NumValueHeads}, {a, geometry.NumValueHeads}}
	for _, want := range wants {
		if want.r == nil || want.r.graph != g || want.r.ptr == nil || want.r.p != batch || want.r.out != want.width {
			return nil, errors.New("metalgemm: invalid Qwen GDN batch panel")
		}
	}
	owners := make([]C.int, batch)
	leases := make([]gdnGraphLease, 0, batch)
	for row, state := range states {
		owner, done, retainErr := state.retainGraph()
		if retainErr != nil {
			for _, lease := range leases {
				lease.state.releaseGraph(lease.done)
			}
			return nil, retainErr
		}
		owners[row] = owner
		leases = append(leases, gdnGraphLease{state: state, done: done})
	}
	ptr := C.mg_gdn_graph_encode_batch(g.ptr, (*C.int)(unsafe.Pointer(&owners[0])), C.int(batch), mixed.ptr, z.ptr, b.ptr, a.ptr,
		gdnF32(panel.Conv1D), gdnF32(panel.ALog), gdnF32(panel.DTBias), gdnF32(panel.Norm),
		C.int(geometry.NumKeyHeads), C.int(geometry.NumValueHeads), C.int(geometry.KeyHeadDim), C.int(geometry.ValueHeadDim), C.int(geometry.ConvKernel), C.float(panel.RMSNormEpsilon))
	if ptr == nil {
		for _, lease := range leases {
			lease.state.releaseGraph(lease.done)
		}
		return nil, errors.New("metalgemm: Qwen GDN batch encode failed")
	}
	g.gdnLeases = append(g.gdnLeases, leases...)
	g.encoders += 3 * batch
	return &GraphResult{ptr: ptr, out: geometry.valueDim(), p: batch, graph: g}, nil
}

func qwen35DecodeBatchReceipt(batch int, r GraphReceipt) Qwen35DecodeBatchReceipt {
	return Qwen35DecodeBatchReceipt{
		Batch: batch, CommandBuffers: 1, Commits: decodeBoolInt(r.Committed), CompletionWaits: decodeBoolInt(r.CompletedWait),
		ProjectionDispatches: 5, Quantizers: 2, GDNEncoders: 3 * batch,
		InputUploads: 1, FinalReadbacks: r.HostReadbacks,
		IntermediateReadbacks: 0, StateH2DTransfers: 0, StateD2HTransfers: 0,
		Encoders: r.Encoders, Committed: r.Committed, CompletedWait: r.CompletedWait,
	}
}
