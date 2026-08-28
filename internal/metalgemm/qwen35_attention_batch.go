//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
int mg_qwen35_graph_ready(void);
int mg_qwen35_graph_attention_batch(void*,void*,void*,void*,const float*,const float*,const float*,const float*,const int*,const int*,const int*,const float*,const float*,int,int,int,int,int,int,int,int,int,float,float,int,int,int,int,void**,void**,void**,void**);
int mg_graph_live_owners(void);
int mg_graph_live_buffers(void);
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
	"unsafe"
)

const Qwen35FullAttentionBatchMax = 8
const qwen35MaxCInt = int(^uint32(0) >> 1)

type Qwen35FullAttentionWeights struct {
	Q, K *Q8Weight
	V    *Q4KWeight
}
type Qwen35FullAttentionLane struct {
	Position         int
	PrefixK, PrefixV []float32
}
type Qwen35FullAttentionBatchRequest struct {
	Input                                    []float32
	Weights                                  Qwen35FullAttentionWeights
	Lanes                                    []Qwen35FullAttentionLane
	QNorm, KNorm, Cos, Sin                   []float32
	NumHeads, NumKVHeads, HeadDim, RotaryDim int
	Scale, QKNormEpsilon                     float32
	Gain1p, QKNorm                           bool
	InjectPostSubmitFailureForTest           bool
}
type Qwen35FullAttentionBatchResult struct{ Output, KRaw, KPost, V [][]float32 }
type Qwen35FullAttentionBatchReceipt struct {
	Batch, CommandBuffers, Commits, CompletionWaits     int
	ProjectionDispatches, AttentionDispatches           int
	InputUploads, FinalReadbacks, IntermediateReadbacks int
	AppendElements                                      []int
	Committed, CompletedWait                            bool
}

func RunQwen35FullAttentionDecodeBatch(req Qwen35FullAttentionBatchRequest) (result Qwen35FullAttentionBatchResult, receipt Qwen35FullAttentionBatchReceipt, accepted bool, err error) {
	batch, modelWidth, attentionWidth, kvWidth, totalKV, offsets, lengths, packedK, packedV, err := validateQwen35FullAttentionBatch(req)
	if err != nil {
		return result, receipt, false, err
	}
	// Pipeline compilation is an admission prerequisite, not part of command
	// encoding. This keeps a partial native graph fail-closed before a graph owner,
	// projection encoder, or command buffer can be created or committed.
	if C.mg_qwen35_graph_ready() == 0 {
		return result, receipt, false, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "Qwen full-attention graph pipelines unavailable"}
	}
	g, err := BeginProjectionGraph(req.Input, nil, nil, batch, modelWidth)
	if err != nil {
		return result, receipt, false, err
	}
	defer g.Free()
	if req.InjectPostSubmitFailureForTest {
		g.InjectPostSubmitFailureForTest()
	}
	input, err := g.Input(modelWidth)
	if err != nil {
		return result, receipt, false, err
	}
	qi, err := g.QuantizeQ8(input)
	if err != nil {
		return result, receipt, false, err
	}
	qgate, err := g.EncodeQ8From(req.Weights.Q, qi)
	if err != nil {
		return result, receipt, false, err
	}
	k, err := g.EncodeQ8From(req.Weights.K, qi)
	if err != nil {
		return result, receipt, false, err
	}
	v, err := g.EncodeQ4KFrom(req.Weights.V, input)
	if err != nil {
		return result, receipt, false, err
	}
	attn, err := g.encodeQwen35FullAttentionBatch(qgate, k, v, req, modelWidth, attentionWidth, kvWidth, offsets, lengths, packedK, packedV, totalKV)
	if err != nil {
		return result, receipt, false, err
	}
	outs, gr, finishErr := g.FinishRead(attn.Output, attn.KRaw, attn.KPost, attn.V)
	accepted = gr.Committed
	receipt = Qwen35FullAttentionBatchReceipt{Batch: batch, CommandBuffers: 1, Commits: decodeBoolInt(gr.Committed), CompletionWaits: decodeBoolInt(gr.CompletedWait), ProjectionDispatches: 3, AttentionDispatches: 4, InputUploads: 1, FinalReadbacks: gr.HostReadbacks, Committed: gr.Committed, CompletedWait: gr.CompletedWait, AppendElements: make([]int, batch)}
	for i := range receipt.AppendElements {
		receipt.AppendElements[i] = kvWidth
	}
	if finishErr != nil {
		return result, receipt, accepted, finishErr
	}
	if len(outs) != 4 {
		return result, receipt, true, &GraphPostSubmitError{Reason: "incomplete packed attention results"}
	}
	result.Output = splitAttentionRows(outs[0], batch, attentionWidth)
	result.KRaw = splitAttentionRows(outs[1], batch, kvWidth)
	result.KPost = splitAttentionRows(outs[2], batch, kvWidth)
	result.V = splitAttentionRows(outs[3], batch, kvWidth)
	return result, receipt, true, nil
}

func qwen35AttentionBatchGraphLiveCounts() (owners, buffers int) {
	return int(C.mg_graph_live_owners()), int(C.mg_graph_live_buffers())
}

type qwen35FullAttentionBatchSizes struct {
	Input, GatedQ, Attention, KV int
}

func qwen35CheckedCIntProduct(factors ...int) (int, bool) {
	product := 1
	for _, factor := range factors {
		if factor < 1 || product > qwen35MaxCInt/factor {
			return 0, false
		}
		product *= factor
	}
	return product, true
}

func qwen35CheckedFullAttentionBatchSizes(batch, modelWidth, attentionWidth, kvWidth int) (qwen35FullAttentionBatchSizes, bool) {
	input, inputOK := qwen35CheckedCIntProduct(batch, modelWidth)
	gatedQ, gatedQOK := qwen35CheckedCIntProduct(batch, 2, attentionWidth)
	attention, attentionOK := qwen35CheckedCIntProduct(batch, attentionWidth)
	kv, kvOK := qwen35CheckedCIntProduct(batch, kvWidth)
	return qwen35FullAttentionBatchSizes{Input: input, GatedQ: gatedQ, Attention: attention, KV: kv}, inputOK && gatedQOK && attentionOK && kvOK
}

func validateQwen35FullAttentionBatch(req Qwen35FullAttentionBatchRequest) (batch, modelWidth, attentionWidth, kvWidth, totalKV int, offsets, lengths []int, packedK, packedV []float32, err error) {
	batch = len(req.Lanes)
	if batch < 2 || batch > Qwen35FullAttentionBatchMax {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: fmt.Sprintf("batch=%d outside [2,%d]", batch, Qwen35FullAttentionBatchMax)}
	}
	if req.Weights.Q == nil || req.Weights.K == nil || req.Weights.V == nil {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "missing projection weights"}
	}
	if req.NumHeads < 1 || req.NumKVHeads < 1 || req.NumHeads%req.NumKVHeads != 0 || req.HeadDim < 1 || req.HeadDim > 256 ||
		req.NumHeads > qwen35MaxCInt/req.HeadDim || req.NumKVHeads > qwen35MaxCInt/req.HeadDim || req.RotaryDim < 2 || req.RotaryDim > req.HeadDim || req.RotaryDim%2 != 0 ||
		req.Scale <= 0 || math.IsNaN(float64(req.Scale)) || math.IsInf(float64(req.Scale), 0) || req.QKNormEpsilon <= 0 || math.IsNaN(float64(req.QKNormEpsilon)) || math.IsInf(float64(req.QKNormEpsilon), 0) {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "unsupported attention geometry"}
	}
	modelWidth = req.Weights.Q.In
	attentionWidth = req.NumHeads * req.HeadDim
	kvWidth = req.NumKVHeads * req.HeadDim
	sizes, sizesOK := qwen35CheckedFullAttentionBatchSizes(batch, modelWidth, attentionWidth, kvWidth)
	if !sizesOK || modelWidth%256 != 0 || len(req.Input) != sizes.Input {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "missing or malformed model-width panel"}
	}
	if req.Weights.Q.ID() < 0 || req.Weights.K.ID() < 0 || req.Weights.V.ID() < 0 || req.Weights.Q.Out != 2*attentionWidth || batch*req.Weights.Q.Out != sizes.GatedQ || req.Weights.K.In != modelWidth || req.Weights.K.Out != kvWidth || req.Weights.V.In != modelWidth || req.Weights.V.Out != kvWidth {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "model/attention/kv projection shape mismatch"}
	}
	if len(req.QNorm) != req.HeadDim && len(req.QNorm) != attentionWidth {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "Q norm shape"}
	}
	if len(req.KNorm) != req.HeadDim && len(req.KNorm) != kvWidth {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "K norm shape"}
	}
	offsets = make([]int, batch)
	lengths = make([]int, batch)
	for i, l := range req.Lanes {
		if l.Position < 0 || l.Position >= 4096 || l.Position > qwen35MaxCInt/kvWidth || len(l.PrefixK) != l.Position*kvWidth || len(l.PrefixV) != len(l.PrefixK) || totalKV > qwen35MaxCInt-(l.Position+1) {
			return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: fmt.Sprintf("lane %d prefix/position mismatch", i)}
		}
		offsets[i] = totalKV
		lengths[i] = l.Position + 1
		totalKV += lengths[i]
	}
	packedElements, packedOK := qwen35CheckedCIntProduct(totalKV, kvWidth)
	if !packedOK {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "packed KV shape overflow"}
	}
	packedK = make([]float32, packedElements)
	packedV = make([]float32, packedElements)
	for i, l := range req.Lanes {
		off := offsets[i] * kvWidth
		copy(packedK[off:], l.PrefixK)
		copy(packedV[off:], l.PrefixV)
	}
	needRope := 0
	for _, l := range req.Lanes {
		if l.Position+1 > needRope {
			needRope = l.Position + 1
		}
	}
	needRope *= req.RotaryDim / 2
	if len(req.Cos) < needRope || len(req.Sin) < needRope {
		return 0, 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "rotary table too short"}
	}
	return
}

func (g *ProjectionGraph) encodeQwen35FullAttentionBatch(qgate, k, v *GraphResult, req Qwen35FullAttentionBatchRequest, modelWidth, attentionWidth, kvWidth int, offsets, lengths []int, packedK, packedV []float32, totalKV int) (Qwen35GraphAttentionResult, error) {
	batch := len(req.Lanes)
	for _, x := range []struct {
		r *GraphResult
		w int
	}{{qgate, 2 * attentionWidth}, {k, kvWidth}, {v, kvWidth}} {
		if x.r == nil || x.r.graph != g || x.r.p != batch || x.r.out != x.w {
			return Qwen35GraphAttentionResult{}, errors.New("metalgemm: invalid independent-lane projection")
		}
	}
	positions := make([]C.int, batch)
	offs := make([]C.int, batch)
	lens := make([]C.int, batch)
	for i, l := range req.Lanes {
		positions[i] = C.int(l.Position)
		offs[i] = C.int(offsets[i])
		lens[i] = C.int(lengths[i])
	}
	var op, kr, kp, vp unsafe.Pointer
	ok := C.mg_qwen35_graph_attention_batch(g.ptr, qgate.ptr, k.ptr, v.ptr, (*C.float)(unsafe.Pointer(&req.QNorm[0])), (*C.float)(unsafe.Pointer(&req.KNorm[0])), (*C.float)(unsafe.Pointer(&req.Cos[0])), (*C.float)(unsafe.Pointer(&req.Sin[0])), &positions[0], &offs[0], &lens[0], (*C.float)(unsafe.Pointer(&packedK[0])), (*C.float)(unsafe.Pointer(&packedV[0])), C.int(totalKV), C.int(batch), C.int(modelWidth), C.int(attentionWidth), C.int(kvWidth), C.int(req.NumHeads), C.int(req.NumKVHeads), C.int(req.HeadDim), C.int(req.RotaryDim), C.float(req.Scale), C.float(req.QKNormEpsilon), boolInt(req.Gain1p), boolInt(req.QKNorm), C.int(len(req.QNorm)), C.int(len(req.KNorm)), &op, &kr, &kp, &vp)
	if ok == 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: independent-lane full attention encode failed")
	}
	g.encoders += 4
	return Qwen35GraphAttentionResult{Output: &GraphResult{ptr: op, out: attentionWidth, p: batch, graph: g}, KRaw: &GraphResult{ptr: kr, out: kvWidth, p: batch, graph: g}, KPost: &GraphResult{ptr: kp, out: kvWidth, p: batch, graph: g}, V: &GraphResult{ptr: vp, out: kvWidth, p: batch, graph: g}}, nil
}
func splitAttentionRows(flat []float32, batch, width int) [][]float32 {
	if len(flat) != batch*width {
		return nil
	}
	out := make([][]float32, batch)
	for i := range out {
		out[i] = append([]float32(nil), flat[i*width:(i+1)*width]...)
	}
	return out
}
