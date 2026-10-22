//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"errors"
	"fmt"
	"math"
)

var _ Qwen35IndependentDecodeBatchBackend = (*vulkanBackend)(nil)

func (*vulkanBackend) Qwen35IndependentDecodeBatchPath() string {
	return Qwen35IndependentDecodeBatchPath
}

func sameQwen35IndependentDecodeWeights(a, b Qwen35SequencePrefillRequest) bool {
	if a.TokenEmbedding.Buf() != b.TokenEmbedding.Buf() || a.OutputNorm.Buf() != b.OutputNorm.Buf() || a.Output.Buf() != b.Output.Buf() || len(a.Layers) != len(b.Layers) {
		return false
	}
	for i := range a.Layers {
		x, y := a.Layers[i], b.Layers[i]
		if x.Linear != y.Linear {
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

func (v *vulkanBackend) Qwen35IndependentDecodeBatch(req Qwen35IndependentDecodeBatchRequest) (result Qwen35IndependentDecodeBatchResult, err error) {
	if err := validateQwen35IndependentDecodeBatchContract(req); err != nil {
		return result, err
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	start := len(v.transient)
	stage, mutated, completed := "preflight", false, false
	fail := func(cause error) error {
		return &Qwen35IndependentDecodeBatchError{Stage: stage, Mutated: mutated, Cause: cause}
	}
	defer func() {
		if recovered := recover(); recovered != nil {
			cause, ok := recovered.(error)
			if !ok {
				cause = fmt.Errorf("%v", recovered)
			}
			err = fail(cause)
		}
		if !completed && err == nil {
			err = fail(errors.New("batch execution interrupted"))
		}
		if err != nil {
			if mutated {
				C.fvk_batch_flush_status()
				for _, lane := range req.Lanes {
					for i, layer := range lane.Layers {
						if !layer.Linear {
							continue
						}
						for _, state := range []Tensor{lane.States[i].Conv, lane.States[i].Recurrent} {
							if b, ok := state.buf.(*vulkanBuf); ok && b != nil && b.ptr != nil {
								C.fvk_free(b.ptr)
								b.ptr, b.n = nil, 0
							}
						}
					}
				}
			}
			v.qwen35VulkanSequenceReleaseLocked(start)
			result = Qwen35IndependentDecodeBatchResult{}
		}
	}()

	base := req.Lanes[0]
	kvs := make([]*vulkanKV, len(req.Lanes))
	statePtrs := make(map[any]bool)
	for i, lane := range req.Lanes {
		kv, validateErr := v.validateQwen35VulkanSequence(lane)
		if validateErr != nil {
			return result, fail(fmt.Errorf("lane %d: %w", i, validateErr))
		}
		if i != 0 && !sameQwen35IndependentDecodeWeights(base, lane) {
			return result, fail(fmt.Errorf("lane %d does not share resident projection weights", i))
		}
		for layer, desc := range lane.Layers {
			if !desc.Linear {
				continue
			}
			for _, state := range []Tensor{lane.States[layer].Conv, lane.States[layer].Recurrent} {
				key := state.Buf()
				if statePtrs[key] {
					return result, fail(fmt.Errorf("lane %d aliases recurrent state", i))
				}
				statePtrs[key] = true
			}
		}
		kvs[i] = kv
	}
	if status := int(C.fvk_submission_status()); status != 0 {
		return result, fail(fmt.Errorf("Vulkan submission fault %d", status))
	}
	check := func(code C.int) error {
		if code != 0 {
			return fail(fmt.Errorf("Vulkan operation status %d", int(code)))
		}
		return nil
	}
	B, H, I := len(req.Lanes), base.Hidden, base.Intermediate
	qWidth, kvWidth := base.NumHeads*base.HeadDim, base.NumKVHeads*base.HeadDim
	valueDim := base.NumValueHeads * base.ValueHeadDim
	convDim := 2*base.NumKeyHeads*base.KeyHeadDim + valueDim
	row := func(panel Tensor, index, width int) Tensor {
		out, _ := v.devTr([]int{1, width}, F32)
		C.fvk_d2d_range(v.vp(out), 0, v.vp(panel), C.size_t(index*width*4), C.size_t(width*4))
		return out
	}
	putRow := func(panel Tensor, index int, source Tensor, width int) {
		C.fvk_d2d_range(v.vp(panel), C.size_t(index*width*4), v.vp(source), 0, C.size_t(width*4))
	}
	normEpsilon := func(x, weight Tensor, rows int, eps float32) Tensor {
		y, _ := v.devTr(append([]int(nil), x.Shape...), F32)
		C.fvk_rmsnorm_f32(v.vp(x), v.vp(weight), v.vp(y), C.int(rows), C.int(weight.Numel()), C.float(eps))
		return y
	}
	norm := func(x, weight Tensor, rows int) Tensor { return normEpsilon(x, weight, rows, base.RMSNormEpsilon) }
	sharedPanels := 0
	projectionMACs := int64(0)
	mul := func(weight, x Tensor) Tensor {
		sharedPanels++
		projectionMACs += int64(B) * int64(weight.Numel())
		return v.qwen35VulkanSequenceMatMulLocked(weight, x, B)
	}
	stage = "kv-reserve"
	C.fvk_batch_begin()
	for lane, kv := range kvs {
		need := (req.Lanes[lane].StartPos + 1) * kvWidth
		for i := range kv.K {
			for _, cache := range []*vslice{&kv.Kraw[i], &kv.K[i], &kv.V[i]} {
				v.qwen35SequenceReserveGeometric(cache, need, "Qwen independent decode KV reservation")
			}
		}
	}
	h2dStart, d2hStart := uint64(C.fvk_h2d_bytes()), uint64(C.fvk_d2h_bytes())
	stage = "embedding-gather"
	x, _ := v.devTr([]int{B, H}, F32)
	for i, lane := range req.Lanes {
		if lane.TokenEmbeddingRows {
			C.fvk_d2d_range(v.vp(x), C.size_t(i*H*4), v.vp(lane.TokenEmbedding), 0, C.size_t(H*4))
		} else {
			C.fvk_d2d_range(v.vp(x), C.size_t(i*H*4), v.vp(lane.TokenEmbedding), C.size_t(lane.TokenIDs[0]*H*4), C.size_t(H*4))
		}
	}
	attention := 0
	for layerIndex, layer := range base.Layers {
		stage = fmt.Sprintf("layer-%d-input-norm", layerIndex)
		C.fvk_batch_begin()
		n := norm(x, layer.InputNorm, B)
		var branch Tensor
		if layer.Linear {
			stage = fmt.Sprintf("layer-%d-gdn-projections", layerIndex)
			mixed, z, beta, alpha := mul(layer.GDNInQKV, n), mul(layer.GDNInZ, n), mul(layer.GDNInB, n), mul(layer.GDNInA, n)
			cores, _ := v.devTr([]int{B, valueDim}, F32)
			for laneIndex, lane := range req.Lanes {
				mr, zr, br, ar := row(mixed, laneIndex, convDim), row(z, laneIndex, valueDim), row(beta, laneIndex, base.NumValueHeads), row(alpha, laneIndex, base.NumValueHeads)
				core, _ := v.devTr([]int{1, valueDim}, F32)
				state := lane.States[layerIndex]
				mutated = true
				if err = check(C.fvk_qwen35_gdn_preprojected_f32(v.vp(mr), v.vp(zr), v.vp(br), v.vp(ar), v.vp(layer.GDNConv), v.vp(layer.GDNALog), v.vp(layer.GDNDTBias), v.vp(layer.GDNNorm), v.vp(state.Conv), v.vp(state.Recurrent), v.vp(core), 1, C.int(convDim), C.int(base.NumKeyHeads), C.int(base.NumValueHeads), C.int(base.KeyHeadDim), C.int(base.ValueHeadDim), C.int(base.ConvKernel), C.float(base.RMSNormEpsilon))); err != nil {
					return result, err
				}
				putRow(cores, laneIndex, core, valueDim)
			}
			branch = mul(layer.GDNOut, cores)
		} else {
			stage = fmt.Sprintf("layer-%d-attention-projections", layerIndex)
			qg, keys, values := mul(layer.Q, n), mul(layer.K, n), mul(layer.V, n)
			queries, _ := v.devTr([]int{B, qWidth}, F32)
			gates, _ := v.devTr([]int{B, qWidth}, F32)
			if err = check(C.fvk_qwen35_split_qg_panel_f32(v.vp(qg), v.vp(queries), v.vp(gates), C.int(B), C.int(base.NumHeads), C.int(base.HeadDim))); err != nil {
				return result, err
			}
			if layer.QNorm.Buf() != nil {
				queries = normEpsilon(queries, layer.QNorm, B*base.NumHeads, qwen35VulkanSequenceQKEpsilon(base))
				keys = normEpsilon(keys, layer.KNorm, B*base.NumKVHeads, qwen35VulkanSequenceQKEpsilon(base))
			}
			contexts, _ := v.devTr([]int{B, qWidth}, F32)
			for laneIndex, lane := range req.Lanes {
				q, k, value, gate := row(queries, laneIndex, qWidth), row(keys, laneIndex, kvWidth), row(values, laneIndex, kvWidth), row(gates, laneIndex, qWidth)
				qr, _ := v.devTr([]int{1, qWidth}, F32)
				kr, _ := v.devTr([]int{1, kvWidth}, F32)
				if err = check(C.fvk_qwen35_partial_rope_panel_f32(v.vp(q), v.vp(k), v.vp(qr), v.vp(kr), 1, C.int(lane.StartPos), C.int(base.NumHeads), C.int(base.NumKVHeads), C.int(base.HeadDim), C.int(base.RotaryDim), C.double(lane.RoPEThetaForLayer[layerIndex]))); err != nil {
					return result, err
				}
				mutated = true
				v.growAppend(&kvs[laneIndex].Kraw[attention], v.vp(k), kvWidth, "Qwen independent decode raw keys")
				v.growAppend(&kvs[laneIndex].K[attention], v.vp(kr), kvWidth, "Qwen independent decode rotated keys")
				v.growAppend(&kvs[laneIndex].V[attention], v.vp(value), kvWidth, "Qwen independent decode values")
				context, _ := v.devTr([]int{1, qWidth}, F32)
				if err = check(C.fvk_qwen35_causal_attention_panel_f32(v.vp(qr), kvs[laneIndex].K[attention].ptr, kvs[laneIndex].V[attention].ptr, v.vp(context), 1, C.int(lane.StartPos), C.int(base.NumHeads), C.int(base.NumKVHeads), C.int(base.HeadDim), C.float(1/math.Sqrt(float64(base.HeadDim))))); err != nil {
					return result, err
				}
				if err = check(C.fvk_sigmoid_mul_f32(v.vp(context), v.vp(gate), C.int(qWidth))); err != nil {
					return result, err
				}
				putRow(contexts, laneIndex, context, qWidth)
			}
			branch = mul(layer.O, contexts)
			attention++
		}
		C.fvk_add_f32(v.vp(x), v.vp(branch), C.int(B*H))
		post := norm(x, layer.PostNorm, B)
		gate, up := mul(layer.Gate, post), mul(layer.Up, post)
		activated, _ := v.devTr([]int{B, I}, F32)
		C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(activated), C.int(B*I))
		delta := mul(layer.Down, activated)
		C.fvk_add_f32(v.vp(x), v.vp(delta), C.int(B*H))
		v.qwen35VulkanSequenceReleaseLocked(start, x)
	}
	stage = "output"
	C.fvk_batch_begin()
	normalized := norm(x, base.OutputNorm, B)
	logits := mul(base.Output, normalized)
	logits.Shape = []int{B, base.Output.Shape[0]}
	stage = "final-fence"
	if err = check(C.fvk_batch_flush_status()); err != nil {
		return result, err
	}
	h2d, d2h := uint64(C.fvk_h2d_bytes())-h2dStart, uint64(C.fvk_d2h_bytes())-d2hStart
	if h2d != 0 || d2h != 0 {
		return result, fail(fmt.Errorf("resident batch copied H2D=%d D2H=%d bytes", h2d, d2h))
	}
	for i, lane := range req.Lanes {
		kvs[i].pos = append(kvs[i].pos, lane.StartPos)
	}
	v.qwen35VulkanSequenceReleaseLocked(start, logits)
	result = Qwen35IndependentDecodeBatchResult{Logits: logits, Lanes: B, SharedProjectionPanels: sharedPanels, ProjectionMACs: projectionMACs, Transfers: Qwen35SequenceTransferCounters{H2DBytes: h2d, D2HBytes: d2h, ActivationH2DBytes: h2d, ActivationD2HBytes: d2h}}
	completed = true
	return result, nil
}
