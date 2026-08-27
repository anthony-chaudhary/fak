//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
void *mg_gdn_graph_encode(void *graph, int owner, void *mixed, void *z, void *b, void *a,
    const float *conv, const float *alog, const float *dtbias, const float *norm,
    int tokens, int nk, int nv, int khd, int vhd, int kernel, float eps);
*/
import "C"

import (
	"errors"
	"fmt"
)

// Qwen35DecodeWeights is the complete resident-Q8 projection set for one
// linear-attention layer. All five handles are retained for the whole operation.
type Qwen35DecodeWeights struct {
	InQKV, InZ, InB, InA, Out *Q8Weight
}

// Qwen35DecodeRequest describes the exact P=1 Gated-DeltaNet mixer. Input is
// uploaded once; all projection and recurrent intermediates remain graph-owned.
type Qwen35DecodeRequest struct {
	Input                          []float32
	Weights                        Qwen35DecodeWeights
	State                          *GDNState
	Panel                          GDNPanel
	InjectPostSubmitFailureForTest bool
}

// Qwen35DecodeReceipt binds the accepted operation to its transfer and command
// topology. ProjectionDispatches counts only weight projections; Encoders also
// includes the two activation quantizers and resident GDN encoder.
type Qwen35DecodeReceipt struct {
	CommandBuffers, Commits, CompletionWaits                    int
	ProjectionDispatches, Quantizers, GDNEncoders               int
	InputUploads, FinalReadbacks                                int
	IntermediateReadbacks, StateH2DTransfers, StateD2HTransfers int
	Encoders                                                    int
	Committed, CompletedWait                                    bool
}

// RunQwen35Decode executes one Qwen linear-attention mixer in one caller-owned
// ProjectionGraph. Every failure before command-buffer submission declines; no
// resident state has changed and the historical path remains safe. Once the
// command commits, accepted remains true for every failure and callers must not
// replay the operation on the host because resident state may have advanced.
func RunQwen35Decode(req Qwen35DecodeRequest) (out []float32, receipt Qwen35DecodeReceipt, accepted bool, err error) {
	geometry, err := validateQwen35DecodeRequest(req)
	if err != nil {
		return nil, Qwen35DecodeReceipt{}, false, err
	}
	weights := []*Q8Weight{req.Weights.InQKV, req.Weights.InZ, req.Weights.InB, req.Weights.InA, req.Weights.Out}
	for _, weight := range weights {
		if weight == nil {
			return nil, Qwen35DecodeReceipt{}, false, &GDNDeclinedError{Reason: "missing resident Q8 decode weight"}
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
	if err := validateLockedQwen35DecodeWeights(req.Weights, len(req.Input), geometry); err != nil {
		return nil, Qwen35DecodeReceipt{}, false, err
	}

	graph, err := BeginProjectionGraph(req.Input, nil, nil, 1, len(req.Input))
	if err != nil {
		return nil, Qwen35DecodeReceipt{}, false, err
	}
	defer graph.Free()
	if req.InjectPostSubmitFailureForTest {
		graph.InjectPostSubmitFailureForTest()
	}
	input, err := graph.Input(len(req.Input))
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
	core, err := graph.encodeGDN(req.State, projected[0], projected[1], projected[2], projected[3], req.Panel, 1)
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
	receipt = Qwen35DecodeReceipt{
		CommandBuffers: 1, Commits: decodeBoolInt(graphReceipt.Committed), CompletionWaits: decodeBoolInt(graphReceipt.CompletedWait),
		ProjectionDispatches: 5, Quantizers: 2, GDNEncoders: 1,
		InputUploads: 1, FinalReadbacks: graphReceipt.HostReadbacks,
		Encoders: graphReceipt.Encoders, Committed: graphReceipt.Committed, CompletedWait: graphReceipt.CompletedWait,
	}
	if err != nil {
		return nil, receipt, accepted, err
	}
	if len(outputs) != 1 || len(outputs[0]) != req.Weights.Out.Out || !receipt.Committed || !receipt.CompletedWait ||
		receipt.Encoders != 8 || receipt.FinalReadbacks != 1 {
		return nil, receipt, true, fmt.Errorf("metalgemm: incomplete Qwen decode mixer receipt: %+v", receipt)
	}
	return outputs[0], receipt, true, nil
}

func decodeBoolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func validateQwen35DecodeRequest(req Qwen35DecodeRequest) (GDNGeometry, error) {
	if req.State == nil {
		return GDNGeometry{}, &GDNDeclinedError{Reason: "nil decode owner"}
	}
	geometry, err := req.State.graphGeometry()
	if err != nil {
		return GDNGeometry{}, err
	}
	if req.Panel.Tokens != 0 && req.Panel.Tokens != 1 {
		return GDNGeometry{}, &GDNDeclinedError{Reason: fmt.Sprintf("decode tokens=%d, want 1", req.Panel.Tokens)}
	}
	if len(req.Input) == 0 || len(req.Input)%32 != 0 {
		return GDNGeometry{}, &GDNDeclinedError{Reason: "decode input must be a non-empty multiple of 32"}
	}
	if err := validateGDNConstants(geometry, req.Panel); err != nil {
		return GDNGeometry{}, err
	}
	return geometry, nil
}

func validateGDNConstants(geometry GDNGeometry, panel GDNPanel) error {
	if err := geometry.validate(); err != nil {
		return &GDNDeclinedError{Reason: err.Error()}
	}
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

func validateLockedQwen35DecodeWeights(weights Qwen35DecodeWeights, hidden int, geometry GDNGeometry) error {
	wants := []struct {
		name    string
		weight  *Q8Weight
		in, out int
	}{
		{"in_proj_qkv", weights.InQKV, hidden, geometry.convDim()},
		{"in_proj_z", weights.InZ, hidden, geometry.valueDim()},
		{"in_proj_b", weights.InB, hidden, geometry.NumValueHeads},
		{"in_proj_a", weights.InA, hidden, geometry.NumValueHeads},
		{"out_proj", weights.Out, geometry.valueDim(), hidden},
	}
	for _, want := range wants {
		if want.weight == nil {
			return &GDNDeclinedError{Reason: "missing resident Q8 " + want.name}
		}
		if want.weight.id < 0 || want.weight.In != want.in || want.weight.Out != want.out {
			return &GDNDeclinedError{Reason: fmt.Sprintf("resident Q8 %s shape=%dx%d, want %dx%d", want.name, want.weight.Out, want.weight.In, want.out, want.in)}
		}
	}
	return nil
}

func (g *ProjectionGraph) encodeGDN(state *GDNState, mixed, z, b, a *GraphResult, panel GDNPanel, tokens int) (*GraphResult, error) {
	if g == nil || g.p != tokens || tokens != 1 {
		return nil, &GDNDeclinedError{Reason: "Qwen decode GDN requires P=1"}
	}
	geometry, err := state.graphGeometry()
	if err != nil {
		return nil, err
	}
	for _, lease := range g.gdnLeases {
		if lease.state == state {
			return nil, errors.New("metalgemm: GDN owner already retained by graph")
		}
	}
	if err := validateGDNConstants(geometry, panel); err != nil {
		return nil, err
	}
	wants := []struct {
		result *GraphResult
		width  int
	}{{mixed, geometry.convDim()}, {z, geometry.valueDim()}, {b, geometry.NumValueHeads}, {a, geometry.NumValueHeads}}
	for _, want := range wants {
		if want.result == nil || want.result.graph != g || want.result.ptr == nil || want.result.p != tokens || want.result.out != want.width {
			return nil, &GDNDeclinedError{Reason: "invalid graph-owned GDN operand"}
		}
	}
	owner, done, err := state.retainGraph()
	if err != nil {
		return nil, err
	}
	ptr := C.mg_gdn_graph_encode(g.ptr, owner, mixed.ptr, z.ptr, b.ptr, a.ptr,
		gdnF32(panel.Conv1D), gdnF32(panel.ALog), gdnF32(panel.DTBias), gdnF32(panel.Norm),
		C.int(tokens), C.int(geometry.NumKeyHeads), C.int(geometry.NumValueHeads), C.int(geometry.KeyHeadDim), C.int(geometry.ValueHeadDim), C.int(geometry.ConvKernel), C.float(panel.RMSNormEpsilon))
	if ptr == nil {
		state.releaseGraph(done)
		return nil, errors.New("metalgemm: Qwen decode GDN graph encode failed")
	}
	g.gdnLeases = append(g.gdnLeases, gdnGraphLease{state: state, done: done})
	g.encoders++
	return &GraphResult{ptr: ptr, out: geometry.valueDim(), p: tokens, graph: g}, nil
}
