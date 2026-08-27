//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
int mg_qwen35_graph_attention_decode_batch(void *graph, void *q, void *k, void *v,
    const float *qnorm, const float *knorm, const float *cosv, const float *sinv,
    const float *prefix_k, const float *prefix_v, const int *prefix_offsets,
    int batch, int nH, int nKV, int hd, int rotary, float scale, float qk_eps,
    int gain1p, int qknorm, int qnorm_elems, int knorm_elems,
    void **out, void **kraw, void **kpost);
int mg_graph_live_owners(void);
int mg_graph_live_buffers(void);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

const (
	Qwen35FullAttentionDecodeBatchMin = 2
	Qwen35FullAttentionDecodeBatchMax = 8
	qwen35FullAttentionScratchTokens  = 4096
	qwen35FullAttentionHidden         = 5120
	qwen35FullAttentionNumHeads       = 24
	qwen35FullAttentionNumKVHeads     = 4
	qwen35FullAttentionHeadDim        = 256
	qwen35FullAttentionRotaryDim      = 64
)

// Qwen35FullAttentionGeometry is the exact full-attention shape shared by all
// independent decode lanes in one panel.
type Qwen35FullAttentionGeometry struct {
	Hidden, NumHeads, NumKVHeads int
	HeadDim, RotaryDim           int
}

// Qwen35FullAttentionDecodeWeights are already-resident fak-native projection
// owners. Q and K use Q8_0 because Qwen stores their reordered forms there; V
// retains the resident Q4_K model payload.
type Qwen35FullAttentionDecodeWeights struct {
	Q, K *Q8Weight
	V    *Q4KWeight
}

// Qwen35FullAttentionDecodeLane carries the metadata that must never cross a
// session boundary. PrefixK is post-RoPE K. Cos and Sin are the rotary row for
// Position, allowing the model layer to preserve its exact scaling policy.
type Qwen35FullAttentionDecodeLane struct {
	Position         int
	PrefixK, PrefixV []float32
	Cos, Sin         []float32
}

// Qwen35FullAttentionDecodeBatchRequest describes B independent decode rows.
// Input is packed [B, Hidden]; every lane owns a distinct logical KV prefix.
type Qwen35FullAttentionDecodeBatchRequest struct {
	Input                          []float32
	Weights                        Qwen35FullAttentionDecodeWeights
	Lanes                          []Qwen35FullAttentionDecodeLane
	Geometry                       Qwen35FullAttentionGeometry
	QNorm                          []float32
	KNorm                          []float32
	Scale                          float32
	QKNormEps                      float32
	NormGain1p, QKNorm             bool
	InjectPostSubmitFailureForTest bool
	afterGraphAllocationForTest    func()
	afterGraphEncodeForTest        func()
}

// Qwen35FullAttentionDecodeRow returns the gated attention output and the
// exact current-token values the caller appends to its lane-local KV owner.
type Qwen35FullAttentionDecodeRow struct {
	Output, KRaw, KPost, VAppend []float32
}

// Qwen35FullAttentionDecodeBatchReceipt binds the result to its one-fence
// projection, attention, transfer, replay, and graph-owner lifecycle.
type Qwen35FullAttentionDecodeBatchReceipt struct {
	Rows                                                      int
	CommandBuffers, Commits, CompletionWaits                  int
	ProjectionDispatches, Quantizers, AttentionDispatches     int
	InputUploads, ConstantUploads                             int
	FinalReadbacks, IntermediateReadbacks, ReplayAttempts     int
	GraphOwners, GraphOwnerReleases, Encoders                 int
	PrefixTokens, Positions, KAppendElements, VAppendElements []int
	Committed, CompletedWait                                  bool
}

// Qwen35FullAttentionDecodeBatchDeclinedError marks a request rejected before
// the command buffer can be submitted. A GraphPostSubmitError is deliberately
// distinct: callers must never replay an accepted lane panel.
type Qwen35FullAttentionDecodeBatchDeclinedError struct{ Reason string }

func (e *Qwen35FullAttentionDecodeBatchDeclinedError) Error() string {
	return "metalgemm: Qwen full-attention decode batch declined: " + e.Reason
}

func qwen35FullAttentionBatchDecline(reason string) error {
	return &Qwen35FullAttentionDecodeBatchDeclinedError{Reason: reason}
}

// RunQwen35FullAttentionDecodeBatch executes B=2..8 independent full-attention
// decode rows in one caller-owned Metal graph. Q/K/V projection weights and the
// submit/wait are shared once; lane offsets keep prefixes and appends isolated.
// Once Committed is true, any error is terminal and no serial replay occurs.
func RunQwen35FullAttentionDecodeBatch(req Qwen35FullAttentionDecodeBatchRequest) (out []Qwen35FullAttentionDecodeRow, receipt Qwen35FullAttentionDecodeBatchReceipt, accepted bool, err error) {
	batch, qwidth, kvwidth, prefixK, prefixV, offsets, cosv, sinv, err := validateQwen35FullAttentionDecodeBatch(req)
	if err != nil {
		return nil, Qwen35FullAttentionDecodeBatchReceipt{}, false, err
	}
	weights := []*Q8Weight{req.Weights.Q, req.Weights.K}
	if !lockQ8Group(weights) {
		return nil, Qwen35FullAttentionDecodeBatchReceipt{}, false, qwen35FullAttentionBatchDecline("Q/K resident handles are unavailable")
	}
	defer unlockQ8Group(weights)
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()
	if req.Weights.Q.In != req.Geometry.Hidden || req.Weights.Q.Out != 2*qwidth ||
		req.Weights.K.In != req.Geometry.Hidden || req.Weights.K.Out != kvwidth ||
		req.Weights.V == nil || req.Weights.V.id < 0 || req.Weights.V.In != req.Geometry.Hidden || req.Weights.V.Out != kvwidth {
		return nil, Qwen35FullAttentionDecodeBatchReceipt{}, false, qwen35FullAttentionBatchDecline("resident Q/K/V shapes do not match the requested geometry")
	}

	graph, err := BeginProjectionGraph(req.Input, nil, nil, batch, req.Geometry.Hidden)
	if err != nil {
		return nil, Qwen35FullAttentionDecodeBatchReceipt{}, false, err
	}
	receipt.GraphOwners = 1
	defer func() {
		graph.Free()
		receipt.GraphOwnerReleases = 1
	}()
	if req.afterGraphAllocationForTest != nil {
		req.afterGraphAllocationForTest()
	}
	if req.InjectPostSubmitFailureForTest {
		graph.InjectPostSubmitFailureForTest()
	}
	input, err := graph.Input(req.Geometry.Hidden)
	if err != nil {
		return nil, receipt, false, err
	}
	quantized, err := graph.QuantizeQ8(input)
	if err != nil {
		return nil, receipt, false, err
	}
	q, err := graph.EncodeQ8From(req.Weights.Q, quantized)
	if err != nil {
		return nil, receipt, false, err
	}
	k, err := graph.EncodeQ8From(req.Weights.K, quantized)
	if err != nil {
		return nil, receipt, false, err
	}
	v, err := graph.EncodeQ4K(req.Weights.V)
	if err != nil {
		return nil, receipt, false, err
	}
	attention, err := graph.qwen35FullAttentionDecodeBatch(q, k, v, req, prefixK, prefixV, offsets, cosv, sinv)
	if err != nil {
		return nil, receipt, false, err
	}
	if req.afterGraphEncodeForTest != nil {
		req.afterGraphEncodeForTest()
	}
	packed, graphReceipt, runErr := graph.FinishRead(attention.Output, attention.KRaw, attention.KPost, v)
	accepted = graphReceipt.Committed
	receipt = qwen35FullAttentionDecodeReceipt(req, graphReceipt, offsets, kvwidth)
	if runErr != nil {
		return nil, receipt, accepted, runErr
	}
	if len(packed) != 4 || len(packed[0]) != batch*qwidth || len(packed[1]) != batch*kvwidth ||
		len(packed[2]) != batch*kvwidth || len(packed[3]) != batch*kvwidth ||
		!receipt.Committed || !receipt.CompletedWait || receipt.Encoders != 6 || receipt.FinalReadbacks != 1 {
		return nil, receipt, true, fmt.Errorf("metalgemm: incomplete Qwen full-attention decode batch receipt: %+v", receipt)
	}
	out = make([]Qwen35FullAttentionDecodeRow, batch)
	for row := range out {
		out[row] = Qwen35FullAttentionDecodeRow{
			Output:  packed[0][row*qwidth : (row+1)*qwidth],
			KRaw:    packed[1][row*kvwidth : (row+1)*kvwidth],
			KPost:   packed[2][row*kvwidth : (row+1)*kvwidth],
			VAppend: packed[3][row*kvwidth : (row+1)*kvwidth],
		}
	}
	return out, receipt, true, nil
}

func validateQwen35FullAttentionDecodeBatch(req Qwen35FullAttentionDecodeBatchRequest) (batch, qwidth, kvwidth int, prefixK, prefixV []float32, offsets []int32, cosv, sinv []float32, err error) {
	batch = len(req.Lanes)
	g := req.Geometry
	if batch < Qwen35FullAttentionDecodeBatchMin || batch > Qwen35FullAttentionDecodeBatchMax {
		err = qwen35FullAttentionBatchDecline(fmt.Sprintf("batch=%d, want [%d,%d]", batch, Qwen35FullAttentionDecodeBatchMin, Qwen35FullAttentionDecodeBatchMax))
		return
	}
	if g.Hidden != qwen35FullAttentionHidden || g.NumHeads != qwen35FullAttentionNumHeads ||
		g.NumKVHeads != qwen35FullAttentionNumKVHeads || g.HeadDim != qwen35FullAttentionHeadDim ||
		g.RotaryDim != qwen35FullAttentionRotaryDim {
		err = qwen35FullAttentionBatchDecline("geometry is not the exact Qwen3.8 full-attention shape")
		return
	}
	qwidth, kvwidth = g.NumHeads*g.HeadDim, g.NumKVHeads*g.HeadDim
	if len(req.Input) != batch*g.Hidden || req.Scale <= 0 || req.QKNormEps <= 0 || req.Weights.Q == nil || req.Weights.K == nil || req.Weights.V == nil {
		err = qwen35FullAttentionBatchDecline("invalid packed input, constants, or resident weights")
		return
	}
	if len(req.QNorm) != g.HeadDim && len(req.QNorm) != qwidth || len(req.KNorm) != g.HeadDim && len(req.KNorm) != kvwidth {
		err = qwen35FullAttentionBatchDecline("invalid Q/K normalization shape")
		return
	}
	halfRotary := g.RotaryDim / 2
	offsets = make([]int32, batch+1)
	totalPrefix := 0
	for row, lane := range req.Lanes {
		if len(lane.PrefixK) == 0 || len(lane.PrefixK)%kvwidth != 0 || len(lane.PrefixV) != len(lane.PrefixK) ||
			len(lane.Cos) != halfRotary || len(lane.Sin) != halfRotary {
			err = qwen35FullAttentionBatchDecline(fmt.Sprintf("invalid lane %d prefix or rotary row", row))
			return
		}
		prefixTokens := len(lane.PrefixK) / kvwidth
		if lane.Position < 0 || prefixTokens+1 > qwen35FullAttentionScratchTokens {
			err = qwen35FullAttentionBatchDecline(fmt.Sprintf("invalid lane %d position=%d prefix=%d", row, lane.Position, prefixTokens))
			return
		}
		totalPrefix += prefixTokens
		offsets[row+1] = int32(totalPrefix)
		prefixK = append(prefixK, lane.PrefixK...)
		prefixV = append(prefixV, lane.PrefixV...)
		cosv = append(cosv, lane.Cos...)
		sinv = append(sinv, lane.Sin...)
	}
	return
}

type qwen35FullAttentionDecodeBatchGraphResult struct {
	Output, KRaw, KPost *GraphResult
}

func (g *ProjectionGraph) qwen35FullAttentionDecodeBatch(q, k, v *GraphResult, req Qwen35FullAttentionDecodeBatchRequest, prefixK, prefixV []float32, offsets []int32, cosv, sinv []float32) (qwen35FullAttentionDecodeBatchGraphResult, error) {
	batch := len(req.Lanes)
	geom := req.Geometry
	qwidth, kvwidth := geom.NumHeads*geom.HeadDim, geom.NumKVHeads*geom.HeadDim
	for _, operand := range []struct {
		result *GraphResult
		width  int
	}{{q, 2 * qwidth}, {k, kvwidth}, {v, kvwidth}} {
		if operand.result == nil || operand.result.graph != g || operand.result.ptr == nil || operand.result.p != batch || operand.result.out != operand.width {
			return qwen35FullAttentionDecodeBatchGraphResult{}, errors.New("metalgemm: invalid graph-owned Qwen attention batch operand")
		}
	}
	prefixElems := int(offsets[len(offsets)-1]) * kvwidth
	if len(prefixK) != prefixElems || len(prefixV) != prefixElems || len(offsets) != batch+1 {
		return qwen35FullAttentionDecodeBatchGraphResult{}, errors.New("metalgemm: invalid packed Qwen attention lane metadata")
	}
	gain, qknorm := C.int(0), C.int(0)
	if req.NormGain1p {
		gain = 1
	}
	if req.QKNorm {
		qknorm = 1
	}
	var outp, krawp, kpostp unsafe.Pointer
	if C.mg_qwen35_graph_attention_decode_batch(g.ptr, q.ptr, k.ptr, v.ptr,
		(*C.float)(unsafe.Pointer(&req.QNorm[0])), (*C.float)(unsafe.Pointer(&req.KNorm[0])),
		(*C.float)(unsafe.Pointer(&cosv[0])), (*C.float)(unsafe.Pointer(&sinv[0])),
		(*C.float)(unsafe.Pointer(&prefixK[0])), (*C.float)(unsafe.Pointer(&prefixV[0])),
		(*C.int)(unsafe.Pointer(&offsets[0])), C.int(batch), C.int(geom.NumHeads), C.int(geom.NumKVHeads),
		C.int(geom.HeadDim), C.int(geom.RotaryDim), C.float(req.Scale), C.float(req.QKNormEps), gain, qknorm,
		C.int(len(req.QNorm)), C.int(len(req.KNorm)), &outp, &krawp, &kpostp) == 0 || outp == nil || krawp == nil || kpostp == nil {
		return qwen35FullAttentionDecodeBatchGraphResult{}, errors.New("metalgemm: Qwen full-attention decode batch encode failed")
	}
	g.encoders += 2
	return qwen35FullAttentionDecodeBatchGraphResult{
		Output: &GraphResult{ptr: outp, out: qwidth, p: batch, graph: g},
		KRaw:   &GraphResult{ptr: krawp, out: kvwidth, p: batch, graph: g},
		KPost:  &GraphResult{ptr: kpostp, out: kvwidth, p: batch, graph: g},
	}, nil
}

func qwen35FullAttentionDecodeReceipt(req Qwen35FullAttentionDecodeBatchRequest, graph GraphReceipt, offsets []int32, kvwidth int) Qwen35FullAttentionDecodeBatchReceipt {
	batch := len(req.Lanes)
	r := Qwen35FullAttentionDecodeBatchReceipt{
		Rows: batch, CommandBuffers: 1, Commits: decodeBoolInt(graph.Committed), CompletionWaits: decodeBoolInt(graph.CompletedWait),
		ProjectionDispatches: 3, Quantizers: 1, AttentionDispatches: 2,
		InputUploads: 1, ConstantUploads: 7, FinalReadbacks: graph.HostReadbacks,
		GraphOwners: 1, Encoders: graph.Encoders, Committed: graph.Committed, CompletedWait: graph.CompletedWait,
		PrefixTokens: make([]int, batch), Positions: make([]int, batch), KAppendElements: make([]int, batch), VAppendElements: make([]int, batch),
	}
	for row, lane := range req.Lanes {
		r.PrefixTokens[row] = int(offsets[row+1] - offsets[row])
		r.Positions[row] = lane.Position
		r.KAppendElements[row], r.VAppendElements[row] = kvwidth, kvwidth
	}
	return r
}

func qwen35FullAttentionDecodeBatchNativeLiveCounts() (owners, buffers int) {
	return int(C.mg_graph_live_owners()), int(C.mg_graph_live_buffers())
}
