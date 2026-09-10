//go:build vulkan && (windows || linux) && cgo

package compute

/*
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"math"
)

var (
	_ Qwen35MTPDraftBackend    = (*vulkanBackend)(nil)
	_ Qwen35MTPTransferCounter = (*vulkanBackend)(nil)
)

func (*vulkanBackend) Qwen35MTPDraftPath() string { return Qwen35MTPDraftPath }

func (v *vulkanBackend) Qwen35MTPTransferBytes() (h2d, d2h uint64) {
	return v.VulkanDebugTransferBytes()
}

// Qwen35MTPFuse normalizes the embedding and prior target hidden state in
// separate device buffers, concatenates [embedding, hidden] by device copies,
// and executes mtp.fc through the resident dtype-dispatched MatMul path.
func (v *vulkanBackend) Qwen35MTPFuse(req Qwen35MTPFuseRequest) (Tensor, error) {
	h := req.HiddenNorm.Numel()
	bad := func(reason string) (Tensor, error) {
		return Tensor{}, &UnsupportedQwen35MTPDraftError{Backend: v.Name(), Path: Qwen35MTPDraftPath, Stage: "pre-layer fusion", Reason: reason}
	}
	if h <= 0 || h > int(^uint(0)>>1)/2 {
		return bad(fmt.Sprintf("invalid hidden dimension %d", h))
	}
	for _, item := range []struct {
		name   string
		tensor Tensor
	}{
		{name: "prior hidden", tensor: req.PriorHidden},
		{name: "current embedding", tensor: req.CurrentEmbedding},
		{name: "hidden norm", tensor: req.HiddenNorm},
		{name: "embedding norm", tensor: req.EmbeddingNorm},
		{name: "mtp.fc", tensor: req.FC},
	} {
		if item.tensor.Backend() != v || item.tensor.Buf() == nil || !item.tensor.Ready() {
			return bad(item.name + " must be a live ready tensor owned by this Vulkan backend")
		}
	}
	if h <= 0 || req.PriorHidden.Numel() != h || req.CurrentEmbedding.Numel() != h || req.EmbeddingNorm.Numel() != h || len(req.FC.Shape) != 2 || req.FC.Shape[0] != h || req.FC.Shape[1] != 2*h {
		return bad(fmt.Sprintf("shape mismatch hidden=%d prior=%d embedding=%d embedding_norm=%d fc=%v", h, req.PriorHidden.Numel(), req.CurrentEmbedding.Numel(), req.EmbeddingNorm.Numel(), req.FC.Shape))
	}
	if req.PriorHidden.Dtype != F32 || req.CurrentEmbedding.Dtype != F32 || req.HiddenNorm.Dtype != F32 || req.EmbeddingNorm.Dtype != F32 {
		return bad("activation and normalization tensors must be F32")
	}
	switch req.FC.Dtype {
	case F32, Q8_0, Q4_K, Q6_K, Q2_K:
	default:
		return bad("mtp.fc dtype " + req.FC.Dtype.String() + " has no resident Vulkan MatMul")
	}
	if !(req.Epsilon >= 0) || math.IsNaN(float64(req.Epsilon)) || math.IsInf(float64(req.Epsilon), 0) {
		return bad("RMS normalization epsilon must be finite and non-negative")
	}

	embedding := v.RMSNorm(req.CurrentEmbedding, req.EmbeddingNorm, req.Epsilon)
	hidden := v.RMSNorm(req.PriorHidden, req.HiddenNorm, req.Epsilon)

	joined, err := func() (joined Tensor, err error) {
		vulkanMu.Lock()
		defer vulkanMu.Unlock()
		defer func() {
			if recovered := recover(); recovered != nil {
				err = fmt.Errorf("Vulkan device concatenation: %v", recovered)
			}
		}()
		joined, joinedBuf := v.devTr([]int{2 * h}, F32)
		C.fvk_d2d_range(joinedBuf.ptr, 0, v.vp(embedding), 0, C.size_t(h*F32.Bytes()))
		C.fvk_d2d_range(joinedBuf.ptr, C.size_t(h*F32.Bytes()), v.vp(hidden), 0, C.size_t(h*F32.Bytes()))
		return joined, nil
	}()
	if err != nil {
		return bad(err.Error())
	}

	return v.MatMul(req.FC, joined), nil
}
