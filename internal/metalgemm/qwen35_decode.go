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
	MLPActivation, MLPUp      *Q4KWeight
	MLPDownQ4                 *Q4KWeight
	MLPDownQ6                 *Q6KWeight
}

// Qwen35DecodeBlock extends the mixer graph across the surrounding pre-norm
// residual block. The two norm vectors and four GDN vectors are immutable
// operation constants; projection weights are already resident device handles.
type Qwen35DecodeBlock struct {
	InputNorm, MLPNorm []float32
	RMSNormEpsilon     float32
	NormGain1p         bool
}

// Qwen35DecodeRequest describes the exact P=1 Gated-DeltaNet mixer. Input is
// uploaded once; all projection and recurrent intermediates remain graph-owned.
type Qwen35DecodeRequest struct {
	Input                          []float32
	Weights                        Qwen35DecodeWeights
	State                          *GDNState
	Panel                          GDNPanel
	Block                          *Qwen35DecodeBlock
	InjectPostSubmitFailureForTest bool
}

// Qwen35DecodeReceipt binds the accepted operation to its transfer and command
// topology. ProjectionDispatches counts only weight projections; Encoders also
// includes the two activation quantizers and resident GDN encoder.
type Qwen35DecodeReceipt struct {
	CommandBuffers, Commits, CompletionWaits                    int
	ProjectionDispatches, MixerProjectionDispatches             int
	MLPProjectionDispatches, Quantizers, GDNEncoders            int
	RMSNormEncoders, ResidualAddEncoders, SwiGLUEncoders        int
	InputUploads, ConstantUploads, FinalReadbacks               int
	IntermediateReadbacks, StateH2DTransfers, StateD2HTransfers int
	Encoders                                                    int
	Committed, CompletedWait                                    bool
}

// EncodeQwen35Decode appends one Qwen linear-attention block to a caller-owned
// ProjectionGraph without submitting or reading it. The caller must keep every
// resident weight alive until the graph finishes. This boundary adapts the
// whole-graph lifetime used by llama.cpp's Qwen3.5 graph builder
// (ggml-org/llama.cpp@9723942adc) and MLX's state-dependency ordering
// (ml-explore/mlx-lm@77c33b1437); it does not import either runtime or source.
//
// Validation completes before the graph gains an encoder or the GDN state is
// retained. Once this function succeeds, callers may append another layer,
// final norm, or LM head and then perform one terminal FinishRead.
func EncodeQwen35Decode(graph *ProjectionGraph, input *GraphResult, req Qwen35DecodeRequest) (result *GraphResult, receipt Qwen35DecodeReceipt, err error) {
	if graph == nil || input == nil || input.graph != graph || input.ptr == nil || input.p != 1 {
		return nil, receipt, &GDNDeclinedError{Reason: "invalid caller-owned Qwen decode graph input"}
	}
	hidden := input.out
	geometry, err := validateQwen35DecodeGeometry(req, hidden)
	if err != nil {
		return nil, receipt, err
	}
	weights := []*Q8Weight{req.Weights.InQKV, req.Weights.InZ, req.Weights.InB, req.Weights.InA, req.Weights.Out}
	for _, weight := range weights {
		if weight == nil {
			return nil, receipt, &GDNDeclinedError{Reason: "missing resident Q8 decode weight"}
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
		return nil, receipt, err
	}
	if err := validateQwen35DecodeBlockHidden(req, hidden); err != nil {
		return nil, receipt, err
	}

	beforeEncoders := graph.encoders
	mixerInput := input
	if req.Block != nil {
		mixerInput, err = graph.RMSNorm(input, req.Block.InputNorm, req.Block.RMSNormEpsilon, req.Block.NormGain1p)
		if err != nil {
			return nil, receipt, err
		}
	}
	quantizedInput, err := graph.QuantizeQ8(mixerInput)
	if err != nil {
		return nil, receipt, err
	}
	projected := make([]*GraphResult, 4)
	for i, weight := range weights[:4] {
		projected[i], err = graph.EncodeQ8From(weight, quantizedInput)
		if err != nil {
			return nil, receipt, err
		}
	}
	core, err := graph.encodeGDN(req.State, projected[0], projected[1], projected[2], projected[3], req.Panel, 1)
	if err != nil {
		return nil, receipt, err
	}
	quantizedCore, err := graph.QuantizeQ8(core)
	if err != nil {
		return nil, receipt, err
	}
	result, err = graph.EncodeQ8From(req.Weights.Out, quantizedCore)
	if err != nil {
		return nil, receipt, err
	}
	projectionDispatches, mlpProjectionDispatches := 5, 0
	rmsNormEncoders, residualAddEncoders, swiGLUEncoders := 0, 0, 0
	constantUploads := 4
	wantEncoders := 8
	if req.Block != nil {
		if err = graph.AddInPlace(input, result); err != nil {
			return nil, receipt, err
		}
		var mlpInput *GraphResult
		mlpInput, err = graph.RMSNorm(input, req.Block.MLPNorm, req.Block.RMSNormEpsilon, req.Block.NormGain1p)
		if err != nil {
			return nil, receipt, err
		}
		var gate, up, down *GraphResult
		gate, err = graph.EncodeQ4KFrom(req.Weights.MLPActivation, mlpInput)
		if err != nil {
			return nil, receipt, err
		}
		up, err = graph.EncodeQ4KFrom(req.Weights.MLPUp, mlpInput)
		if err != nil {
			return nil, receipt, err
		}
		if err = graph.SwiGLUInPlace(gate, up); err != nil {
			return nil, receipt, err
		}
		if req.Weights.MLPDownQ4 != nil {
			down, err = graph.EncodeQ4KFrom(req.Weights.MLPDownQ4, gate)
		} else {
			down, err = graph.EncodeQ6KFrom(req.Weights.MLPDownQ6, gate)
		}
		if err != nil {
			return nil, receipt, err
		}
		if err = graph.AddInPlace(input, down); err != nil {
			return nil, receipt, err
		}
		result = input
		projectionDispatches, mlpProjectionDispatches = 8, 3
		rmsNormEncoders, residualAddEncoders, swiGLUEncoders = 2, 2, 1
		constantUploads, wantEncoders = 6, 16
	}
	receipt = Qwen35DecodeReceipt{
		ProjectionDispatches: projectionDispatches, MixerProjectionDispatches: 5, MLPProjectionDispatches: mlpProjectionDispatches,
		Quantizers: 2, GDNEncoders: 1, RMSNormEncoders: rmsNormEncoders, ResidualAddEncoders: residualAddEncoders, SwiGLUEncoders: swiGLUEncoders,
		ConstantUploads: constantUploads, Encoders: graph.encoders - beforeEncoders,
	}
	if receipt.Encoders != wantEncoders {
		return nil, receipt, fmt.Errorf("metalgemm: incomplete caller-owned Qwen decode encoding: %+v", receipt)
	}
	return result, receipt, nil
}

// RunQwen35Decode preserves the standalone compatibility boundary around
// EncodeQwen35Decode: one input upload, one submission, and one final readback.
// Pre-submit failures decline. Once the command commits, accepted remains true
// and callers must not replay because resident state may have advanced.
func RunQwen35Decode(req Qwen35DecodeRequest) (out []float32, receipt Qwen35DecodeReceipt, accepted bool, err error) {
	if len(req.Input) == 0 || len(req.Input)%32 != 0 {
		return nil, receipt, false, &GDNDeclinedError{Reason: "decode input must be a non-empty multiple of 32"}
	}
	if _, err := validateQwen35DecodeGeometry(req, len(req.Input)); err != nil {
		return nil, receipt, false, err
	}
	graph, err := BeginProjectionGraph(req.Input, nil, nil, 1, len(req.Input))
	if err != nil {
		return nil, receipt, false, err
	}
	defer graph.Free()
	if req.InjectPostSubmitFailureForTest {
		graph.InjectPostSubmitFailureForTest()
	}
	input, err := graph.Input(len(req.Input))
	if err != nil {
		return nil, receipt, false, err
	}
	result, encoded, err := EncodeQwen35Decode(graph, input, req)
	if err != nil {
		return nil, encoded, false, err
	}
	outputs, graphReceipt, err := graph.FinishRead(result)
	accepted = graphReceipt.Committed
	receipt = encoded
	receipt.CommandBuffers = 1
	receipt.Commits = decodeBoolInt(graphReceipt.Committed)
	receipt.CompletionWaits = decodeBoolInt(graphReceipt.CompletedWait)
	receipt.InputUploads = 1
	receipt.FinalReadbacks = graphReceipt.HostReadbacks
	receipt.Encoders = graphReceipt.Encoders
	receipt.Committed = graphReceipt.Committed
	receipt.CompletedWait = graphReceipt.CompletedWait
	if err != nil {
		return nil, receipt, accepted, err
	}
	if len(outputs) != 1 || len(outputs[0]) != req.Weights.Out.Out || !receipt.Committed || !receipt.CompletedWait ||
		receipt.Encoders != encoded.Encoders || receipt.FinalReadbacks != 1 {
		return nil, receipt, true, fmt.Errorf("metalgemm: incomplete Qwen decode mixer receipt: %+v", receipt)
	}
	return outputs[0], receipt, true, nil
}

func validateQwen35DecodeBlock(req Qwen35DecodeRequest) error {
	return validateQwen35DecodeBlockHidden(req, len(req.Input))
}

func validateQwen35DecodeBlockHidden(req Qwen35DecodeRequest, hidden int) error {
	if req.Block == nil {
		return nil
	}
	if len(req.Block.InputNorm) != hidden || len(req.Block.MLPNorm) != hidden ||
		req.Block.RMSNormEpsilon <= 0 || req.Block.RMSNormEpsilon != req.Panel.RMSNormEpsilon {
		return &GDNDeclinedError{Reason: "invalid Qwen decode block normalization"}
	}
	gate, up := req.Weights.MLPActivation, req.Weights.MLPUp
	if gate == nil || up == nil || gate.id < 0 || up.id < 0 || gate.In != hidden || up.In != hidden || gate.Out <= 0 || up.Out != gate.Out {
		return &GDNDeclinedError{Reason: "invalid resident Q4_K gate/up projection"}
	}
	if (req.Weights.MLPDownQ4 == nil) == (req.Weights.MLPDownQ6 == nil) {
		return &GDNDeclinedError{Reason: "Qwen decode block requires exactly one resident down projection"}
	}
	if down := req.Weights.MLPDownQ4; down != nil && (down.id < 0 || down.In != gate.Out || down.Out != hidden) {
		return &GDNDeclinedError{Reason: "invalid resident Q4_K down projection"}
	}
	if down := req.Weights.MLPDownQ6; down != nil && (down.id < 0 || down.In != gate.Out || down.Out != hidden) {
		return &GDNDeclinedError{Reason: "invalid resident Q6_K down projection"}
	}
	return nil
}

func decodeBoolInt(v bool) int {
	if v {
		return 1
	}
	return 0
}

func validateQwen35DecodeRequest(req Qwen35DecodeRequest) (GDNGeometry, error) {
	return validateQwen35DecodeGeometry(req, len(req.Input))
}

func validateQwen35DecodeGeometry(req Qwen35DecodeRequest, hidden int) (GDNGeometry, error) {
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
	if hidden == 0 || hidden%32 != 0 {
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
