//go:build darwin && arm64 && cgo

package metalgemm

/*
#include <stdint.h>
int mg_qwen35_graph_attention_batch(void*,void*,void*,void*,const float*,const float*,const float*,const float*,const int*,const int*,const int*,const float*,const float*,int,int,int,int,int,int,float,float,int,int,int,int,void**,void**,void**,void**);
*/
import "C"

import (
	"errors"
	"fmt"
	"unsafe"
)

const Qwen35FullAttentionBatchMax = 8

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
	batch, hidden, kvWidth, totalKV, offsets, lengths, packedK, packedV, err := validateQwen35FullAttentionBatch(req)
	if err != nil {
		return result, receipt, false, err
	}
	g, err := BeginProjectionGraph(req.Input, nil, nil, batch, hidden)
	if err != nil {
		return result, receipt, false, err
	}
	defer g.Free()
	if req.InjectPostSubmitFailureForTest {
		g.InjectPostSubmitFailureForTest()
	}
	input, err := g.Input(hidden)
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
	attn, err := g.encodeQwen35FullAttentionBatch(qgate, k, v, req, offsets, lengths, packedK, packedV, totalKV)
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
	result.Output = splitAttentionRows(outs[0], batch, hidden)
	result.KRaw = splitAttentionRows(outs[1], batch, kvWidth)
	result.KPost = splitAttentionRows(outs[2], batch, kvWidth)
	result.V = splitAttentionRows(outs[3], batch, kvWidth)
	return result, receipt, true, nil
}

func validateQwen35FullAttentionBatch(req Qwen35FullAttentionBatchRequest) (batch, hidden, kvWidth, totalKV int, offsets, lengths []int, packedK, packedV []float32, err error) {
	batch = len(req.Lanes)
	if batch < 2 || batch > Qwen35FullAttentionBatchMax {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: fmt.Sprintf("batch=%d outside [2,%d]", batch, Qwen35FullAttentionBatchMax)}
	}
	if req.NumHeads < 1 || req.NumKVHeads < 1 || req.NumHeads%req.NumKVHeads != 0 || req.HeadDim < 1 || req.HeadDim > 256 || req.RotaryDim < 2 || req.RotaryDim > req.HeadDim || req.RotaryDim%2 != 0 || req.Scale <= 0 || req.QKNormEpsilon <= 0 {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "unsupported attention geometry"}
	}
	hidden = req.NumHeads * req.HeadDim
	kvWidth = req.NumKVHeads * req.HeadDim
	if len(req.Input) != batch*hidden || req.Weights.Q == nil || req.Weights.K == nil || req.Weights.V == nil {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "missing or malformed panel/weights"}
	}
	if req.Weights.Q.In != hidden || req.Weights.Q.Out != 2*hidden || req.Weights.K.In != hidden || req.Weights.K.Out != kvWidth || req.Weights.V.In != hidden || req.Weights.V.Out != kvWidth {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "projection shape mismatch"}
	}
	if len(req.QNorm) != req.HeadDim && len(req.QNorm) != hidden {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "Q norm shape"}
	}
	if len(req.KNorm) != req.HeadDim && len(req.KNorm) != kvWidth {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "K norm shape"}
	}
	offsets = make([]int, batch)
	lengths = make([]int, batch)
	for i, l := range req.Lanes {
		if l.Position < 0 || l.Position >= 4096 || len(l.PrefixK) != l.Position*kvWidth || len(l.PrefixV) != len(l.PrefixK) {
			return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: fmt.Sprintf("lane %d prefix/position mismatch", i)}
		}
		offsets[i] = totalKV
		lengths[i] = l.Position + 1
		totalKV += lengths[i]
		packedK = append(packedK, l.PrefixK...)
		packedK = append(packedK, make([]float32, kvWidth)...)
		packedV = append(packedV, l.PrefixV...)
		packedV = append(packedV, make([]float32, kvWidth)...)
	}
	needRope := 0
	for _, l := range req.Lanes {
		if l.Position+1 > needRope {
			needRope = l.Position + 1
		}
	}
	needRope *= req.RotaryDim / 2
	if len(req.Cos) < needRope || len(req.Sin) < needRope {
		return 0, 0, 0, 0, nil, nil, nil, nil, &MixedQKVError{Stage: MixedQKVDeclined, Detail: "rotary table too short"}
	}
	return
}

func (g *ProjectionGraph) encodeQwen35FullAttentionBatch(qgate, k, v *GraphResult, req Qwen35FullAttentionBatchRequest, offsets, lengths []int, packedK, packedV []float32, totalKV int) (Qwen35GraphAttentionResult, error) {
	batch := len(req.Lanes)
	hidden := req.NumHeads * req.HeadDim
	kvWidth := req.NumKVHeads * req.HeadDim
	for _, x := range []struct {
		r *GraphResult
		w int
	}{{qgate, 2 * hidden}, {k, kvWidth}, {v, kvWidth}} {
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
	ok := C.mg_qwen35_graph_attention_batch(g.ptr, qgate.ptr, k.ptr, v.ptr, (*C.float)(unsafe.Pointer(&req.QNorm[0])), (*C.float)(unsafe.Pointer(&req.KNorm[0])), (*C.float)(unsafe.Pointer(&req.Cos[0])), (*C.float)(unsafe.Pointer(&req.Sin[0])), &positions[0], &offs[0], &lens[0], (*C.float)(unsafe.Pointer(&packedK[0])), (*C.float)(unsafe.Pointer(&packedV[0])), C.int(totalKV), C.int(batch), C.int(req.NumHeads), C.int(req.NumKVHeads), C.int(req.HeadDim), C.int(req.RotaryDim), C.float(req.Scale), C.float(req.QKNormEpsilon), boolInt(req.Gain1p), boolInt(req.QKNorm), C.int(len(req.QNorm)), C.int(len(req.KNorm)), &op, &kr, &kp, &vp)
	if ok == 0 {
		return Qwen35GraphAttentionResult{}, errors.New("metalgemm: independent-lane full attention encode failed")
	}
	g.encoders += 4
	return Qwen35GraphAttentionResult{Output: &GraphResult{ptr: op, out: hidden, p: batch, graph: g}, KRaw: &GraphResult{ptr: kr, out: kvWidth, p: batch, graph: g}, KPost: &GraphResult{ptr: kp, out: kvWidth, p: batch, graph: g}, V: &GraphResult{ptr: vp, out: kvWidth, p: batch, graph: g}}, nil
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
