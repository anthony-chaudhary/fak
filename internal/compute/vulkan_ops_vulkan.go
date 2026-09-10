//go:build vulkan && (windows || linux) && cgo

package compute

/*
#cgo CFLAGS: -I${SRCDIR}
#cgo LDFLAGS: -L${SRCDIR} -lfakvulkan
#include <stdlib.h>
#include "vulkan_backend.h"
*/
import "C"

import (
	"fmt"
	"log"
	"math"
	"unsafe"
)

// RMSNorm applies row-wise RMS normalization scaled by weight (eps in the denominator)
// to each row of x, returning a new device tensor of the same shape.
func (v *vulkanBackend) RMSNorm(x, weight Tensor, eps float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := weight.Numel()
	rows := x.Numel() / n
	y, _ := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_rmsnorm_f32(v.vp(x), v.vp(weight), v.vp(y), C.int(rows), C.int(n), C.float(eps))
	return y
}

// RoPE applies rotary position embedding at position pos to each head of x, returning a
// new device tensor (x is copied D2D first so the input is left unmodified).
func (v *vulkanBackend) RoPE(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	y, ybuf := v.devTr(append([]int(nil), x.Shape...), F32)
	C.fvk_d2d(ybuf.ptr, x.buf.(*vulkanBuf).ptr, C.size_t(x.Numel()*4))
	C.fvk_rope_f32(v.vp(y), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return y
}

// RoPEInPlace applies rotary position embedding at position pos to x's buffer directly,
// returning the same tensor (no copy) for the case where x may be overwritten.
func (v *vulkanBackend) RoPEInPlace(x Tensor, pos, nHeads, headDim int, theta float64) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_rope_f32(v.vp(x), C.int(pos), C.int(nHeads), C.int(headDim), C.double(theta))
	return x
}

// SwiGLU computes the elementwise silu(gate)*up activation, returning a new device
// tensor shaped like gate.
func (v *vulkanBackend) SwiGLU(gate, up Tensor) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	n := gate.Numel()
	y, _ := v.devTr(append([]int(nil), gate.Shape...), F32)
	dispatch := recurrentSwiGLUVulkanDispatch
	if dispatch.Kernel != vulkanElementwiseKernelSwiGLU || dispatch.Dispatches != 1 || dispatch.IntermediateBarriers != 0 {
		panic(fmt.Sprintf("compute: invalid recurrent SwiGLU Vulkan lowering %+v", dispatch))
	}
	C.fvk_swiglu_f32(v.vp(gate), v.vp(up), v.vp(y), C.int(n))
	return y
}

// AddInPlace adds src into dst elementwise (dst += src) on the device — the residual add.
func (v *vulkanBackend) AddInPlace(dst, src Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	C.fvk_add_f32(v.vp(dst), v.vp(src), C.int(dst.Numel()))
}

// AddBias adds the width-length bias vector to every row of dst (broadcast over rows).
func (v *vulkanBackend) AddBias(dst, bias Tensor) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	width := bias.Numel()
	rows := dst.Numel() / width
	C.fvk_add_bias_f32(v.vp(dst), v.vp(bias), C.int(rows), C.int(width))
}

// Attention runs the fused scaled-dot-product FlashAttention for one layer over the cached
// keys/values (grp query heads per KV head, scale applied to the scores) via online softmax,
// returning the per-head context vectors as one device tensor. The default path uses no
// global scratch; the explicit context-split candidate uses bounded transient scratch.
func (v *vulkanBackend) Attention(q Tensor, kv KVStore, layer int, causal bool, grp int, scale float32) Tensor {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	vk := kv.(*vulkanKV)
	hd, nKV := vk.cfg.HeadDim, vk.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := vk.K[layer].len / w
	out, outBuf := v.devTr([]int{nH * hd}, F32)
	C.fvk_attention_f32(v.vp(q), vk.K[layer].ptr, vk.V[layer].ptr, v.vp(out),
		C.int(nPos), C.int(nH), C.int(nKV), C.int(hd), C.float(scale))
	if status := int(C.fvk_submission_status()); status != 0 {
		if outBuf != nil && outBuf.ptr != nil {
			C.fvk_free(outBuf.ptr)
			outBuf.ptr = nil
			outBuf.n = 0
		}
		panic(fmt.Errorf("compute: Vulkan attention failed closed (code %d); no fallback", status))
	}
	return out
}

// SpecVerifyAttention executes ordinary linear causal verification with the
// native tree-attention pipeline. qLen is intentionally unrestricted here.
func (be *vulkanBackend) SpecVerifyAttention(q, k, v, out *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	if err := validateVulkanVerifyAttention(q, k, v, qLen, kvLen, nH, nHkv, d); err != nil {
		return fmt.Errorf("compute: SpecVerifyAttention %w", err)
	}
	return be.runVulkanVerifyAttention(q, k, v, out, nil, qLen, kvLen, nH, nHkv, d, 1)
}

// TreeVerifyAttention executes packed-mask tree verification entirely on the
// Vulkan device. Only the compact mask rows cross from host memory.
func (be *vulkanBackend) TreeVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d int) error {
	if err := validateTreeVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d); err != nil {
		return err
	}
	if err := validateVulkanVerifyBuffers(q, k, v); err != nil {
		return fmt.Errorf("compute: TreeVerifyAttention %w", err)
	}
	return be.runVulkanVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d, 0)
}

func validateVulkanVerifyAttention(q, k, v *Tensor, qLen, kvLen, nH, nHkv, d int) error {
	if q == nil || k == nil || v == nil {
		return fmt.Errorf("nil tensor argument")
	}
	if qLen <= 0 || kvLen < qLen {
		return fmt.Errorf("invalid lengths qLen=%d kvLen=%d", qLen, kvLen)
	}
	if nH <= 0 || nHkv <= 0 || nH%nHkv != 0 {
		return fmt.Errorf("invalid heads nH=%d nHkv=%d", nH, nHkv)
	}
	if d <= 0 || d > 1024 {
		return fmt.Errorf("head dim %d outside [1,1024]", d)
	}
	if q.Numel() != qLen*nH*d || k.Numel() != kvLen*nHkv*d || v.Numel() != kvLen*nHkv*d {
		return fmt.Errorf("tensor dimensions do not match qLen=%d kvLen=%d nH=%d nHkv=%d d=%d", qLen, kvLen, nH, nHkv, d)
	}
	return validateVulkanVerifyBuffers(q, k, v)
}

func validateVulkanVerifyBuffers(tensors ...*Tensor) error {
	for _, tensor := range tensors {
		if tensor.Dtype != F32 {
			return fmt.Errorf("tensor dtype %v is not F32", tensor.Dtype)
		}
		buf, ok := tensor.buf.(*vulkanBuf)
		if !ok || buf == nil || buf.ptr == nil {
			return fmt.Errorf("input tensor is not allocated on the Vulkan device")
		}
	}
	return nil
}

func (be *vulkanBackend) runVulkanVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d, mode int) error {
	if out == nil {
		return fmt.Errorf("compute: Vulkan verify attention nil output tensor")
	}
	if err := validateVulkanVerifyOutput(out, qLen, nH, d); err != nil {
		return err
	}
	if C.fvk_have_tree_attention() == 0 {
		log.Printf("compute: Vulkan tree-attention pipeline unavailable; using CPU reference")
		return be.fallbackVulkanVerifyAttention(q, k, v, out, maskRows, qLen, kvLen, nH, nHkv, d, mode)
	}
	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	expected := qLen * nH * d
	allocated := false
	if out.buf == nil {
		devOut, _ := be.devTr([]int{qLen, nH, d}, F32)
		*out = devOut
		allocated = true
	} else {
		buf, ok := out.buf.(*vulkanBuf)
		if !ok || buf == nil || buf.ptr == nil {
			return fmt.Errorf("compute: Vulkan verify attention output tensor is not allocated on the Vulkan device")
		}
		if out.Dtype != F32 || out.Numel() != expected || len(out.Shape) != 3 || out.Shape[0] != qLen || out.Shape[1] != nH || out.Shape[2] != d {
			return fmt.Errorf("compute: Vulkan verify attention output must be F32 [%d,%d,%d]", qLen, nH, d)
		}
	}

	var maskPtr *C.uint32_t
	if len(maskRows) != 0 {
		maskPtr = (*C.uint32_t)(unsafe.Pointer(&maskRows[0]))
	}
	rc := int(C.fvk_tree_attention_f32(
		be.vp(*q), be.vp(*k), be.vp(*v), be.vp(*out), maskPtr,
		C.int(qLen), C.int(kvLen), C.int(nH), C.int(nHkv), C.int(d),
		C.float(1/math.Sqrt(float64(d))), C.int(mode)))
	if rc != 0 {
		if allocated {
			buf := out.buf.(*vulkanBuf)
			C.fvk_free(buf.ptr)
			buf.ptr = nil
			for i := len(be.transient) - 1; i >= 0; i-- {
				if be.transient[i] == buf {
					be.transient = append(be.transient[:i], be.transient[i+1:]...)
					break
				}
			}
			*out = Tensor{}
		}
		return fmt.Errorf("compute: Vulkan verify attention dispatch failed: %d", rc)
	}
	return nil
}

func validateVulkanVerifyOutput(out *Tensor, qLen, nH, d int) error {
	if out.buf == nil {
		if len(out.Shape) != 0 {
			return fmt.Errorf("compute: Vulkan verify attention output without storage must be a zero Tensor")
		}
		return nil
	}
	buf, ok := out.buf.(*vulkanBuf)
	if !ok || buf == nil || buf.ptr == nil {
		return fmt.Errorf("compute: Vulkan verify attention output tensor is not allocated on the Vulkan device")
	}
	if out.Dtype != F32 || out.Layout != RowMajor || len(out.Shape) != 3 || out.Shape[0] != qLen || out.Shape[1] != nH || out.Shape[2] != d {
		return fmt.Errorf("compute: Vulkan verify attention output must be live row-major F32 [%d,%d,%d]", qLen, nH, d)
	}
	return nil
}

func (be *vulkanBackend) fallbackVulkanVerifyAttention(q, k, v, out *Tensor, maskRows []uint32, qLen, kvLen, nH, nHkv, d, mode int) error {
	ref, ok := Default().(*cpuBackend)
	if !ok {
		return fmt.Errorf("compute: CPU reference backend unavailable for Vulkan verify attention")
	}
	qHost := NewF32(ref, append([]int(nil), q.Shape...), be.Read(*q))
	kHost := NewF32(ref, append([]int(nil), k.Shape...), be.Read(*k))
	vHost := NewF32(ref, append([]int(nil), v.Shape...), be.Read(*v))
	var hostOut Tensor
	var err error
	if mode == 0 {
		err = ref.TreeVerifyAttention(&qHost, &kHost, &vHost, &hostOut, maskRows, qLen, kvLen, nH, nHkv, d)
	} else {
		err = ref.SpecVerifyAttention(&qHost, &kHost, &vHost, &hostOut, qLen, kvLen, nH, nHkv, d)
	}
	if err != nil {
		return err
	}
	outValues := ref.Read(hostOut)
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	if out.buf == nil {
		devOut, _ := be.devTr([]int{qLen, nH, d}, F32)
		*out = devOut
	}
	C.fvk_h2d(be.vp(*out), unsafe.Pointer(&outValues[0]), C.size_t(len(outValues)*F32.Bytes()))
	return nil
}

var _ SpecVerifyBackend = (*vulkanBackend)(nil)
var _ TreeVerifyBackend = (*vulkanBackend)(nil)

var _ BatchedPrefillBackend = (*vulkanBackend)(nil)

// PrefillBatch executes batched prompt prefill across a sequence panel (P x D) in 1 pass on Vulkan GPU (#11036, #12181).
func (v *vulkanBackend) PrefillBatch(args PrefillBatchArgs) (PrefillBatchResult, error) {
	P, D, err := validatePrefillBatchArgs(&args)
	if err != nil {
		return PrefillBatchResult{}, err
	}

	nH := args.NumHeads
	nKV := args.NumKVHeads
	hd := args.HeadDim
	qOut := nH * hd
	kvOut := nKV * hd
	startPos := args.StartPos

	// Ensure inputs are resident device tensors
	xDev := args.X
	if _, isHost := v.Host(xDev); isHost {
		xDev = v.Upload(xDev, F32)
	}
	wqDev := args.Wq
	if _, isHost := v.Host(wqDev); isHost {
		wqDev = v.Upload(wqDev, wqDev.Dtype)
	}
	wkDev := args.Wk
	if _, isHost := v.Host(wkDev); isHost {
		wkDev = v.Upload(wkDev, wkDev.Dtype)
	}
	wvDev := args.Wv
	if _, isHost := v.Host(wvDev); isHost {
		wvDev = v.Upload(wvDev, wvDev.Dtype)
	}

	// 1. Batched projections on Vulkan GPU
	qTen := v.BatchedMatMul(wqDev, xDev, P)
	kTen := v.BatchedMatMul(wkDev, xDev, P)
	vTen := v.BatchedMatMul(wvDev, xDev, P)

	vk, isVulkanKV := args.KV.(*vulkanKV)
	canFastPanel := hd%2 == 0 && hd <= 1024 && (args.KV == nil || isVulkanKV)

	if !canFastPanel {
		return v.prefillBatchSerialGPU(args, xDev, wqDev, wkDev, wvDev, P, D, nH, nKV, hd, qOut, kvOut)
	}

	// 2. Rotary position embedding (RoPE) for all P tokens on GPU in one panel dispatch
	vulkanMu.Lock()
	qr, _ := v.devTr([]int{P, qOut}, F32)
	kr, _ := v.devTr([]int{P, kvOut}, F32)
	ropeStatus := int(C.fvk_qwen35_partial_rope_panel_f32(
		v.vp(qTen), v.vp(kTen), v.vp(qr), v.vp(kr),
		C.int(P), C.int(startPos), C.int(nH), C.int(nKV),
		C.int(hd), C.int(hd), C.double(args.RopeTheta),
	))
	vulkanMu.Unlock()
	if ropeStatus != 0 {
		return v.prefillBatchSerialGPU(args, xDev, wqDev, wkDev, wvDev, P, D, nH, nKV, hd, qOut, kvOut)
	}

	// 3. Append to KVStore if provided (all D2D, 0 host copies)
	var kPtr, vPtr unsafe.Pointer
	prefix := 0
	if args.KV != nil {
		for t := 0; t < P; t++ {
			pos := startPos + t
			vulkanMu.Lock()
			kRawRow, _ := v.devTr([]int{kvOut}, F32)
			kRopeRow, _ := v.devTr([]int{kvOut}, F32)
			vRow, _ := v.devTr([]int{kvOut}, F32)
			rowBytes := C.size_t(kvOut * 4)
			srcOff := C.size_t(t * kvOut * 4)
			C.fvk_d2d_range(v.vp(kRawRow), 0, v.vp(kTen), srcOff, rowBytes)
			C.fvk_d2d_range(v.vp(kRopeRow), 0, v.vp(kr), srcOff, rowBytes)
			C.fvk_d2d_range(v.vp(vRow), 0, v.vp(vTen), srcOff, rowBytes)
			vulkanMu.Unlock()
			args.KV.AppendKV(args.Layer, kRawRow, kRopeRow, vRow, pos)
		}
		prefix = startPos
		kPtr = vk.K[args.Layer].ptr
		vPtr = vk.V[args.Layer].ptr
	} else {
		prefix = 0
		kPtr = v.vp(kr)
		vPtr = v.vp(vTen)
	}

	// 4. Causal attention across the prompt panel on GPU (online softmax, zero quadratic scratch)
	vulkanMu.Lock()
	context, _ := v.devTr([]int{P, qOut}, F32)
	causalStatus := int(C.fvk_qwen35_causal_attention_panel_f32(
		v.vp(qr), kPtr, vPtr, v.vp(context),
		C.int(P), C.int(prefix), C.int(nH), C.int(nKV),
		C.int(hd), C.float(args.Scale),
	))
	vulkanMu.Unlock()
	if causalStatus != 0 {
		return PrefillBatchResult{}, fmt.Errorf("compute: vulkan causal attention panel dispatch failed: %d", causalStatus)
	}

	// 5. Output projection if Wo is provided
	var outTen Tensor
	if args.Wo.buf != nil {
		woDev := args.Wo
		if _, isHost := v.Host(woDev); isHost {
			woDev = v.Upload(woDev, woDev.Dtype)
		}
		outTen = v.BatchedMatMul(woDev, context, P)
	} else {
		outTen = context
	}

	return PrefillBatchResult{
		Output:  outTen,
		Context: context,
		Tokens:  P,
	}, nil
}

// prefillBatchSerialGPU executes prompt prefill token-by-token on Vulkan GPU as a safe fallback.
// Activations and KV-cache remain strictly device-resident in VRAM without host transfers.
func (v *vulkanBackend) prefillBatchSerialGPU(args PrefillBatchArgs, xDev, wqDev, wkDev, wvDev Tensor, P, D, nH, nKV, hd, qOut, kvOut int) (PrefillBatchResult, error) {
	grp := nH / nKV
	scale := args.Scale

	vulkanMu.Lock()
	context, _ := v.devTr([]int{P, qOut}, F32)
	vulkanMu.Unlock()

	kvStore := args.KV
	var tempKV KVStore
	if kvStore == nil {
		tempKV = v.NewKV(KVConfig{
			NumLayers:  1,
			NumKVHeads: nKV,
			HeadDim:    hd,
			RopeTheta:  args.RopeTheta,
		})
		defer tempKV.Free()
		kvStore = tempKV
	}

	for t := 0; t < P; t++ {
		pos := args.StartPos + t
		vulkanMu.Lock()
		xRow, _ := v.devTr([]int{D}, F32)
		C.fvk_d2d_range(v.vp(xRow), 0, v.vp(xDev), C.size_t(t*D*4), C.size_t(D*4))
		vulkanMu.Unlock()

		q := v.MatMul(wqDev, xRow)
		kRaw := v.MatMul(wkDev, xRow)
		val := v.MatMul(wvDev, xRow)

		qR := v.RoPE(q, pos, nH, hd, args.RopeTheta)
		kR := v.RoPE(kRaw, pos, nKV, hd, args.RopeTheta)

		layerIdx := args.Layer
		if args.KV == nil {
			layerIdx = 0
		}
		kvStore.AppendKV(layerIdx, kRaw, kR, val, pos)

		attnOut := v.Attention(qR, kvStore, layerIdx, true, grp, scale)

		vulkanMu.Lock()
		C.fvk_d2d_range(v.vp(context), C.size_t(t*qOut*4), v.vp(attnOut), 0, C.size_t(qOut*4))
		vulkanMu.Unlock()
	}

	var outTen Tensor
	if args.Wo.buf != nil {
		woDev := args.Wo
		if _, isHost := v.Host(woDev); isHost {
			woDev = v.Upload(woDev, woDev.Dtype)
		}
		outTen = v.BatchedMatMul(woDev, context, P)
	} else {
		outTen = context
	}

	return PrefillBatchResult{
		Output:  outTen,
		Context: context,
		Tokens:  P,
	}, nil
}

// Argmax returns the index of the largest element of the device logits tensor via the
// scalar-reduction shader, so greedy decode never copies the full vector host-ward.
func (v *vulkanBackend) Argmax(logits Tensor) int {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	return int(C.fvk_argmax_f32(v.vp(logits), C.int(logits.Numel())))
}

// AttentionDequantOnce runs multi-head attention using the dequant-once KV scratchpad pipeline on vulkanBackend.
func (v *vulkanBackend) AttentionDequantOnce(
	q Tensor,
	kv KVStore,
	layer int,
	rawK, rawV []byte,
	format QuantizedKVType,
	causal bool,
	grp int,
	scale float32,
) (Tensor, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()

	vk, ok := kv.(*vulkanKV)
	if !ok {
		return Tensor{}, fmt.Errorf("vulkan_ops: kv store is not *vulkanKV")
	}

	hd, nKV := vk.cfg.HeadDim, vk.cfg.NumKVHeads
	nH := grp * nKV
	w := nKV * hd
	nPos := vk.Len()
	if nPos <= 0 && len(vk.K) > layer && vk.K[layer].len > 0 {
		nPos = vk.K[layer].len / w
	}
	if nPos <= 0 {
		return Tensor{}, fmt.Errorf("vulkan_ops: empty KV cache")
	}

	// Ensure scratchpad is initialized
	sp := vk.scratchpad
	if sp == nil || sp.NumPos != nPos || sp.Format != format {
		var err error
		sp, err = NewVulkanKVScratchpad(v, RADVTargetArchGfx1151, format, nPos, nKV, hd)
		if err != nil {
			return Tensor{}, fmt.Errorf("vulkan_ops: failed to initialize scratchpad: %w", err)
		}
		vk.scratchpad = sp
	}

	// Read query from device if resident
	qHost := v.Read(q)

	// Execute attention using the dequant-once pipeline
	outHost, err := ExecuteVulkanAttentionWithDequantOnce(qHost, sp, rawK, rawV, nH, scale)
	if err != nil {
		return Tensor{}, err
	}

	// Upload result back to device
	out := v.Upload(NewF32(Default(), []int{nH * hd}, outHost), F32)
	return out, nil
}

// AttentionQuantizedKV executes attention for one layer over quantized KV caches using the dequant-once scratchpad.
func (v *vulkanBackend) AttentionQuantizedKV(
	q Tensor,
	rawK, rawV []byte,
	format QuantizedKVType,
	layer, nPos, nQ, nKV, headDim int,
	scale float32,
) (Tensor, *VulkanKVScratchpad, error) {
	vulkanMu.Lock()
	defer vulkanMu.Unlock()
	qHost := v.Read(q)
	arch := "gfx1151"
	scratch, err := NewVulkanKVScratchpad(v, arch, format, nPos, nKV, headDim)
	if err != nil {
		return Tensor{}, nil, fmt.Errorf("vulkan: failed to create dequant scratchpad: %w", err)
	}
	if scale <= 0 {
		scale = float32(1.0 / math.Sqrt(float64(headDim)))
	}
	outHost, err := ExecuteVulkanAttentionWithDequantOnce(qHost, scratch, rawK, rawV, nQ, scale)
	if err != nil {
		return Tensor{}, nil, fmt.Errorf("vulkan: dequant-once attention failed: %w", err)
	}
	outDev, _ := v.devTr([]int{nQ * headDim}, F32)
	C.fvk_h2d(v.vp(outDev), unsafe.Pointer(&outHost[0]), C.size_t(len(outHost)*4))
	return outDev, scratch, nil
}
