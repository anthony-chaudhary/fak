//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
#include "cuda_inkling2_prefill.h"
#include "cuda_inkling2_prefill_dense.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ Inkling2PrefillDevice = (*cudaBackend)(nil)

func (c *cudaBackend) Ink2PMatMul(w Tensor, x, y Ink2Ref, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(w.Shape) != 2 {
		return fmt.Errorf("cuda inkling2 prefill matmul: weight shape %v", w.Shape)
	}
	out, in := w.Shape[0], w.Shape[1]
	wb := c.cudaBufForSubmit(w)
	xp, yp := c.ink2F(x), c.ink2F(y)
	P := C.int(T)
	// Prompt panels run the int8 tensor-core GEMM; it declines (cudaErrorInvalidValue, nothing enqueued)
	// on shapes it does not cover, which then take the legacy launcher below.
	if T >= 16 && ink2PrefillFast("dense") {
		kind, scales := C.int(0), (*C.float)(nil)
		switch w.Dtype {
		case Q8_0:
			if w.Quant != nil && w.Quant.Block == 32 {
				kind, scales = 8, (*C.float)(wb.scales)
			}
		case Q5_K:
			kind = 5
		case Q6_K:
			kind = 6
		}
		if kind != 0 {
			rc := C.fcuda_ink2p_dense_mmq(wb.ptr, scales, xp, yp, C.int(out), C.int(in), P, kind)
			if rc != 1 {
				return ink2Status("prefill matmul mmq", rc)
			}
		}
	}
	switch w.Dtype {
	case F32:
		return ink2Status("prefill matmul f32", ink2PGemmF32((*C.float)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case Q8_0:
		return ink2Status("prefill matmul q8_0", C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), xp, yp,
			C.int(out), C.int(in), P, C.int(w.Quant.Block)))
	case Q4_K:
		return ink2Status("prefill matmul q4_k", C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case Q5_K:
		return ink2Status("prefill matmul q5_k", C.fcuda_q5k_matmul_checked_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case Q6_K:
		return ink2Status("prefill matmul q6_k", C.fcuda_q6k_matmul_checked_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case Q3_K:
		return ink2Status("prefill matmul q3_k", C.fcuda_q3k_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case IQ4_XS:
		return ink2Status("prefill matmul iq4_xs", C.fcuda_iq4xs_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), P))
	case IQ2_XXS, IQ2_XS, IQ1_M, IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S:
		cudaIQGEMV(w.Dtype, wb, xp, yp, out, in, T)
		return nil
	}
	return fmt.Errorf("cuda inkling2 prefill matmul: unsupported weight dtype %s", w.Dtype)
}

func (c *cudaBackend) Ink2PMix(states []Ink2Ref, queries Ink2Ref, nq int, out Ink2Ref, n, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(states) > ink2MaxStates {
		return fmt.Errorf("cuda inkling2 prefill mix: %d states", len(states))
	}
	ptrs := (*[ink2MaxStates]*C.float)(C.calloc(ink2MaxStates, C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(ptrs))
	for i, s := range states {
		ptrs[i] = c.ink2F(s)
	}
	if ink2PrefillFast("mix") {
		return ink2Status("prefill mix", C.fcuda_ink2p_mix_v2((**C.float)(unsafe.Pointer(ptrs)), C.int(len(states)), c.ink2F(queries), C.int(nq),
			c.ink2F(out), C.int(n), C.int(T)))
	}
	return ink2Status("prefill mix", C.fcuda_ink2p_mix((**C.float)(unsafe.Pointer(ptrs)), C.int(len(states)), c.ink2F(queries), C.int(nq),
		c.ink2F(out), C.int(n), C.int(T)))
}

func (c *cudaBackend) Ink2PCtrlDown(cur, msg, w, g Ink2Ref, n, rank, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if ink2PrefillFast("ctrl") {
		return ink2Status("prefill ctrl down", C.fcuda_ink2p_ctrl_down_v2(c.ink2F(cur), c.ink2F(msg), c.ink2F(w), c.ink2F(g), C.int(n), C.int(rank), C.int(T)))
	}
	return ink2Status("prefill ctrl down", C.fcuda_ink2p_ctrl_down(c.ink2F(cur), c.ink2F(msg), c.ink2F(w), c.ink2F(g), C.int(n), C.int(rank), C.int(T)))
}

func (c *cudaBackend) Ink2PCtrlApply(cur, msg, g, gainW, forgetW, out Ink2Ref, n, rank, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if ink2PrefillFast("ctrl") {
		return ink2Status("prefill ctrl apply", C.fcuda_ink2p_ctrl_apply_v2(c.ink2F(cur), c.ink2F(msg), c.ink2F(g), c.ink2F(gainW), c.ink2F(forgetW),
			c.ink2F(out), C.int(n), C.int(rank), C.int(T)))
	}
	return ink2Status("prefill ctrl apply", C.fcuda_ink2p_ctrl_apply(c.ink2F(cur), c.ink2F(msg), c.ink2F(g), c.ink2F(gainW), c.ink2F(forgetW),
		c.ink2F(out), C.int(n), C.int(rank), C.int(T)))
}

func (c *cudaBackend) Ink2PConv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if ink2PrefillFast("conv") {
		return ink2Status("prefill conv", C.fcuda_ink2p_conv_v2(c.ink2F(x), c.ink2F(w), c.ink2F(dyn), c.ink2F(hist), c.ink2F(out), c.ink2F(resid),
			C.int(n), C.int(heads), C.int(kn), C.float(norm), C.int(T)))
	}
	return ink2Status("prefill conv", C.fcuda_ink2p_conv(c.ink2F(x), c.ink2F(w), c.ink2F(dyn), c.ink2F(hist), c.ink2F(out), c.ink2F(resid),
		C.int(n), C.int(heads), C.int(kn), C.float(norm), C.int(T)))
}

// Ink2PRMS is Ink2RMS over the T rows of a chunk with an fp32 reduction; the decode
// path keeps its float64 row sum.
func (c *cudaBackend) Ink2PRMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	if !ink2PrefillFast("rms") {
		return c.Ink2RMS(x, w, y, rows, n, eps)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("prefill rms", C.fcuda_ink2p_rms(c.ink2F(x), c.ink2F(w), c.ink2F(y), C.int(rows), C.int(n), C.float(eps)))
}

// Ink2PSitu is Ink2Situ in fp32 transcendentals for the chunk's routed slots.
func (c *cudaBackend) Ink2PSitu(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	if !ink2PrefillFast("situ") {
		return c.Ink2Situ(gu, act, k, width, actStride, beta, linearBeta)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("prefill situ", C.fcuda_ink2p_situ_v2(c.ink2F(gu), c.ink2F(act), C.int(k), C.int(width), C.int(actStride), C.double(beta), C.double(linearBeta)))
}

func (c *cudaBackend) Ink2PQKVPost(q, r, k, v, qclip, rclip, kcache, vcache Ink2Ref, p Ink2AttnParams, cap, pos0, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("prefill qkv post", C.fcuda_ink2p_qkv_post(c.ink2F(q), c.ink2F(r), c.ink2F(k), c.ink2F(v), c.ink2F(qclip), c.ink2F(rclip),
		c.ink2F(kcache), c.ink2F(vcache), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen),
		C.float(p.QKClip), C.float(p.RelClip), C.float(p.Eps), ink2Bool(p.Global), C.float(p.Alpha), C.float(p.Floor),
		C.int(cap), C.int(pos0), C.int(T)))
}

func (c *cudaBackend) Ink2PAttend(q, r, sinks, relBias, kcache, vcache, out Ink2Ref, p Ink2AttnParams, extent, window, cap, pos0, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if ink2PrefillFast("attn") {
		return ink2Status("prefill attend", C.fcuda_ink2p_attend_v2(c.ink2F(q), c.ink2F(r), c.ink2F(sinks), c.ink2F(relBias), c.ink2F(kcache),
			c.ink2F(vcache), c.ink2F(out), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen), C.int(extent),
			C.int(window), C.int(cap), C.float(p.Scale), C.int(pos0), C.int(T)))
	}
	return ink2Status("prefill attend", C.fcuda_ink2p_attend(c.ink2F(q), c.ink2F(r), c.ink2F(sinks), c.ink2F(relBias), c.ink2F(kcache),
		c.ink2F(vcache), c.ink2F(out), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen), C.int(extent),
		C.int(window), C.int(cap), C.float(p.Scale), C.int(pos0), C.int(T)))
}

func (c *cudaBackend) Ink2PRoute(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("prefill route", C.fcuda_ink2p_route(c.ink2F(logits), c.ink2F(bias), c.ink2I(ids), c.ink2F(wts), C.int(nexp), C.int(k), C.float(scale), C.int(T)))
}

func (c *cudaBackend) Ink2PCombine(down, wts, out Ink2Ref, k, n, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("prefill combine", C.fcuda_ink2p_combine(c.ink2F(down), c.ink2F(wts), c.ink2F(out), C.int(k), C.int(n), C.int(T)))
}

func (c *cudaBackend) Ink2PExpertGrouped(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch Ink2Ref, bucketed bool) (bool, error) {
	f, ok := ink2ExpertFormat[stack.Dtype]
	if !ok || len(stack.Shape) != 2 {
		return false, nil
	}
	if stack.Dtype != NVFP4 {
		cudaMu.Lock()
		defer cudaMu.Unlock()
		b := c.cudaBufForSubmit(stack)
		offsets, order, tileList, ntiles, err := c.ink2pBucketOnce(ids, npairs, nexp, scratch, bucketed)
		if err != nil {
			return true, err
		}
		tiles := int(C.fcuda_ink2p_max_tiles(C.int(npairs), C.int(nexp)))
		if ink2PrefillFast("iq") {
			return true, ink2Status("prefill expert iq", C.fcuda_ink2p_expert_iq_v2(f, (*C.uint8_t)(b.ptr), c.ink2I(offsets), c.ink2I(order),
				c.ink2I(tileList), c.ink2I(ntiles), C.int(tiles), c.ink2F(x), C.int(xDiv), C.int(xStride), c.ink2F(y),
				C.int(expertRows), C.int(stack.Shape[1]), c.ink2F(scale)))
		}
		return true, ink2Status("prefill expert iq", C.fcuda_ink2p_expert_iq(f, (*C.uint8_t)(b.ptr), c.ink2I(offsets), c.ink2I(order),
			c.ink2I(tileList), c.ink2I(ntiles), C.int(tiles), c.ink2F(x), C.int(xDiv), C.int(xStride), c.ink2F(y),
			C.int(expertRows), C.int(stack.Shape[1]), c.ink2F(scale)))
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := stack.Shape[0], stack.Shape[1]
	b := c.cudaBufForSubmit(stack)
	scales := b.scales
	if scales == nil {
		scales = unsafe.Add(b.ptr, out*in/2)
	}
	offsets, order, tileList, ntiles, err := c.ink2pBucketOnce(ids, npairs, nexp, scratch, bucketed)
	if err != nil {
		return true, err
	}
	tiles := int(C.fcuda_ink2p_max_tiles(C.int(npairs), C.int(nexp)))
	if ink2PrefillFast("nvfp4") {
		return true, ink2Status("prefill expert nvfp4", C.fcuda_ink2p_expert_nvfp4_v2((*C.uint8_t)(b.ptr), (*C.uint8_t)(scales),
			c.ink2I(offsets), c.ink2I(order), c.ink2I(tileList), c.ink2I(ntiles), C.int(tiles), c.ink2F(x), C.int(xDiv),
			C.int(xStride), c.ink2F(y), C.int(expertRows), C.int(in), c.ink2F(scale)))
	}
	return true, ink2Status("prefill expert nvfp4", C.fcuda_ink2p_expert_nvfp4((*C.uint8_t)(b.ptr), (*C.uint8_t)(scales),
		c.ink2I(offsets), c.ink2I(order), c.ink2I(tileList), c.ink2I(ntiles), C.int(tiles), c.ink2F(x), C.int(xDiv),
		C.int(xStride), c.ink2F(y), C.int(expertRows), C.int(in), c.ink2F(scale)))
}

// ink2pBucketOnce is ink2pBucket, or only the scratch layout when bucketed says an
// earlier call already filled it for these ids; caller holds cudaMu.
func (c *cudaBackend) ink2pBucketOnce(ids Ink2Ref, npairs, nexp int, scratch Ink2Ref, bucketed bool) (offsets, order, tileList, ntiles Ink2Ref, err error) {
	if !bucketed {
		return c.ink2pBucket(ids, npairs, nexp, scratch)
	}
	tiles := int(C.fcuda_ink2p_max_tiles(C.int(npairs), C.int(nexp)))
	offsets = Ink2At(scratch.T, scratch.Off+nexp)
	order = Ink2At(scratch.T, offsets.Off+nexp+1)
	tileList = Ink2At(scratch.T, order.Off+npairs)
	ntiles = Ink2At(scratch.T, tileList.Off+2*tiles)
	return offsets, order, tileList, ntiles, nil
}

// ink2pBucket buckets ids by expert into scratch (Ink2PGroupScratch layout); caller holds cudaMu.
func (c *cudaBackend) ink2pBucket(ids Ink2Ref, npairs, nexp int, scratch Ink2Ref) (offsets, order, tileList, ntiles Ink2Ref, err error) {
	tiles := int(C.fcuda_ink2p_max_tiles(C.int(npairs), C.int(nexp)))
	counts := scratch
	offsets = Ink2At(scratch.T, counts.Off+nexp)
	order = Ink2At(scratch.T, offsets.Off+nexp+1)
	tileList = Ink2At(scratch.T, order.Off+npairs)
	ntiles = Ink2At(scratch.T, tileList.Off+2*tiles)
	err = ink2Status("prefill bucket", C.fcuda_ink2p_bucket(c.ink2I(ids), C.int(npairs), C.int(nexp), c.ink2I(counts),
		c.ink2I(offsets), c.ink2I(order), c.ink2I(tileList), c.ink2I(ntiles)))
	return offsets, order, tileList, ntiles, err
}

func (c *cudaBackend) Ink2PExpertSorted(stack Tensor, expertRows, nexp int, ids, x Ink2Ref, xDiv, xStride int, y Ink2Ref, npairs int, scale, scratch, sortIds, sortX, sortY Ink2Ref) (bool, error) {
	f, ok := ink2ExpertFormat[stack.Dtype]
	if !ok || len(stack.Shape) != 2 {
		return false, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	out, in := stack.Shape[0], stack.Shape[1]
	b := c.cudaBufForSubmit(stack)
	var scales unsafe.Pointer
	if stack.Dtype == NVFP4 {
		scales = b.scales
		if scales == nil {
			scales = unsafe.Add(b.ptr, out*in/2)
		}
	}
	_, order, _, _, err := c.ink2pBucket(ids, npairs, nexp, scratch)
	if err != nil {
		return true, err
	}
	if err := ink2Status("prefill gather pairs", C.fcuda_ink2p_gather_pairs(c.ink2I(ids), c.ink2I(order), c.ink2F(x), C.int(xDiv), C.int(xStride),
		c.ink2I(sortIds), c.ink2F(sortX), C.int(xStride), C.int(npairs))); err != nil {
		return true, err
	}
	if err := ink2Status("prefill sorted expert gemv", C.fcuda_ink2_expert_gemv(f, (*C.uint8_t)(b.ptr), (*C.uint8_t)(scales), c.ink2I(sortIds),
		c.ink2F(sortX), C.int(xStride), c.ink2F(sortY), C.int(npairs), C.int(expertRows), C.int(in),
		c.ink2F(scale), 0, 0, 0)); err != nil {
		return true, err
	}
	return true, ink2Status("prefill scatter pairs", C.fcuda_ink2p_scatter_pairs(c.ink2I(order), c.ink2F(sortY), c.ink2F(y), C.int(expertRows), C.int(npairs)))
}

var _ Inkling2VerifyDevice = (*cudaBackend)(nil)

func (c *cudaBackend) Ink2PConvLog(x, w, dyn, hist, log, out, resid Ink2Ref, n, heads, kn int, norm float32, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify conv", C.fcuda_ink2p_conv_log(c.ink2F(x), c.ink2F(w), c.ink2F(dyn), c.ink2F(hist), c.ink2F(log),
		c.ink2F(out), c.ink2F(resid), C.int(n), C.int(heads), C.int(kn), C.float(norm), C.int(T)))
}

func (c *cudaBackend) Ink2PCombineConvLog(down, wts Ink2Ref, k int, w, hist, log, out, resid Ink2Ref, n, kn int, norm float32, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify combine conv", C.fcuda_ink2p_combine_conv_log(c.ink2F(down), c.ink2F(wts), C.int(k), c.ink2F(w),
		c.ink2F(hist), c.ink2F(log), c.ink2F(out), c.ink2F(resid), C.int(n), C.int(kn), C.float(norm), C.int(T)))
}

func (c *cudaBackend) Ink2PConvCommit(hist, log Ink2Ref, channels, past, L, keep int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify conv commit", C.fcuda_ink2p_conv_commit(c.ink2F(hist), c.ink2F(log), C.int(channels), C.int(past),
		C.int(L), C.int(keep)))
}

func (c *cudaBackend) Ink2PConvCommitDev(hist, log, keepDev Ink2Ref, channels, past, L int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify conv commit", C.fcuda_ink2p_conv_commit_dev(c.ink2F(hist), c.ink2F(log), C.int(channels), C.int(past),
		C.int(L), c.ink2I(keepDev)))
}

func (c *cudaBackend) Ink2PQKVPostDev(q, r, k, v, qclip, rclip, kcache, vcache, posDev Ink2Ref, p Ink2AttnParams, cap, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify qkv post", C.fcuda_ink2p_qkv_post_dev(c.ink2F(q), c.ink2F(r), c.ink2F(k), c.ink2F(v), c.ink2F(qclip), c.ink2F(rclip),
		c.ink2F(kcache), c.ink2F(vcache), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen),
		C.float(p.QKClip), C.float(p.RelClip), C.float(p.Eps), ink2Bool(p.Global), C.float(p.Alpha), C.float(p.Floor),
		C.int(cap), c.ink2I(posDev), C.int(T)))
}

func (c *cudaBackend) Ink2PAttendDev(q, r, sinks, relBias, kcache, vcache, out, posDev Ink2Ref, p Ink2AttnParams, extent, window, cap, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify attend", C.fcuda_ink2p_attend_dev(c.ink2F(q), c.ink2F(r), c.ink2F(sinks), c.ink2F(relBias), c.ink2F(kcache),
		c.ink2F(vcache), c.ink2F(out), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen), C.int(extent),
		C.int(window), C.int(cap), C.float(p.Scale), c.ink2I(posDev), C.int(T)))
}

func (c *cudaBackend) Ink2GraphEndSlot(slot int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("graph end slot", C.fcuda_ink2_graph_end_slot(C.int(slot)))
}

func (c *cudaBackend) Ink2GraphLaunchSlot(slot int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("graph launch slot", C.fcuda_ink2_graph_launch_slot(C.int(slot)))
}

var _ Inkling2DraftDevice = (*cudaBackend)(nil)

func (c *cudaBackend) Ink2DFuse(taps []Ink2Ref, alpha []float32, w, out Ink2Ref, rows, n int, eps float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(taps) != len(alpha) || len(taps) < 1 || len(taps) > 4 {
		return fmt.Errorf("cuda inkling2 draft fuse: %d taps, %d weights", len(taps), len(alpha))
	}
	ptrs := (*[4]*C.float)(C.malloc(C.size_t(4 * unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(ptrs))
	a := (*[4]C.float)(C.malloc(C.size_t(4 * unsafe.Sizeof(C.float(0)))))
	defer C.free(unsafe.Pointer(a))
	for i, t := range taps {
		ptrs[i] = c.ink2F(t)
		a[i] = C.float(alpha[i])
	}
	return ink2Status("draft fuse", C.fcuda_ink2d_fuse((**C.float)(unsafe.Pointer(ptrs)), (*C.float)(unsafe.Pointer(a)), C.int(len(taps)),
		c.ink2F(w), c.ink2F(out), C.int(rows), C.int(n), C.float(eps)))
}

func (c *cudaBackend) Ink2DQK(x, w, dst, posDev Ink2Ref, rows, heads, dim, cap int, ring bool, eps float32, theta float64) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft qk", C.fcuda_ink2d_qk(c.ink2F(x), c.ink2F(w), c.ink2F(dst), C.int(rows), C.int(heads), C.int(dim),
		c.ink2I(posDev), C.int(cap), ink2Bool(ring), C.float(eps), C.double(theta)))
}

func (c *cudaBackend) Ink2DRingStore(src, ring, posDev Ink2Ref, rows, width, cap int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft ring store", C.fcuda_ink2d_ring_store(c.ink2F(src), c.ink2F(ring), C.int(rows), C.int(width),
		c.ink2I(posDev), C.int(cap)))
}

func (c *cudaBackend) Ink2DAttend(q, kb, vb, kc, vc, out, posDev Ink2Ref, rows, heads, kvHeads, dim, window, cap int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft attend", C.fcuda_ink2d_attend(c.ink2F(q), c.ink2F(kb), c.ink2F(vb), c.ink2F(kc), c.ink2F(vc),
		c.ink2F(out), C.int(rows), C.int(heads), C.int(kvHeads), C.int(dim), c.ink2I(posDev), C.int(window), C.int(cap)))
}

func (c *cudaBackend) Ink2DSiluMul(g, u, out Ink2Ref, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft silu mul", C.fcuda_ink2d_silu_mul(c.ink2F(g), c.ink2F(u), c.ink2F(out), C.int(n)))
}

func (c *cudaBackend) Ink2DTopM(logits Ink2Ref, rows, rowStride, vocab, M int, candV, candI, out Ink2Ref) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft topm", C.fcuda_ink2d_topm(c.ink2F(logits), C.int(rows), C.int(rowStride), C.int(vocab), C.int(M),
		c.ink2F(candV), c.ink2I(candI), c.ink2I(out)))
}

func (c *cudaBackend) Ink2DMarkovIn(state, w1, anchor, idx Ink2Ref, step int, h, frnnRing, posDev Ink2Ref, cap, rank, hidden int, in Ink2Ref) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft markov in", C.fcuda_ink2d_markov_in(c.ink2F(state), (*C.ushort)(c.ink2Ptr(w1)), c.ink2I(anchor), c.ink2I(idx),
		C.int(step), c.ink2F(h), c.ink2F(frnnRing), c.ink2I(posDev), C.int(cap), C.int(rank), C.int(hidden), c.ink2F(in)))
}

func (c *cudaBackend) Ink2DMarkovGate(raw, bias, state, latent Ink2Ref, rank, step int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("draft markov gate", C.fcuda_ink2d_markov_gate(c.ink2F(raw), c.ink2F(bias), c.ink2F(state), c.ink2F(latent),
		C.int(rank), C.int(step)))
}

func (c *cudaBackend) Ink2DMarkovPick(head Tensor, base, cand Ink2Ref, M int, z, zb, idx Ink2Ref, step int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if head.Dtype != Q5_K || len(head.Shape) != 2 {
		return fmt.Errorf("cuda inkling2 draft markov pick: head %s %v, want Q5_K", head.Dtype, head.Shape)
	}
	wb := c.cudaBufForSubmit(head)
	return ink2Status("draft markov pick", C.fcuda_ink2d_markov_pick_q5k((*C.uchar)(wb.ptr), C.int(head.Shape[1]), c.ink2F(base),
		c.ink2I(cand), C.int(M), c.ink2F(z), c.ink2F(zb), c.ink2I(idx), C.int(step)))
}

func (c *cudaBackend) Ink2VRows(w Tensor, x, y Ink2Ref, rows int) (bool, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if (w.Dtype != Q5_K && w.Dtype != Q6_K) || len(w.Shape) != 2 || rows < 1 || rows > 9 {
		return false, nil
	}
	wb := c.cudaBufForSubmit(w)
	if w.Dtype == Q6_K {
		return true, ink2Status("verify q6k rows", C.fcuda_ink2v_q6k_rows((*C.uchar)(wb.ptr), c.ink2F(x), c.ink2F(y), C.int(w.Shape[0]),
			C.int(w.Shape[1]), C.int(rows)))
	}
	return true, ink2Status("verify q5k rows", C.fcuda_ink2v_q5k_rows((*C.uchar)(wb.ptr), c.ink2F(x), c.ink2F(y), C.int(w.Shape[0]),
		C.int(w.Shape[1]), C.int(rows)))
}

func (c *cudaBackend) Ink2VRepeatRows(x, out Ink2Ref, n, rows, k int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("verify repeat rows", C.fcuda_ink2v_repeat_rows(c.ink2F(x), c.ink2F(out), C.int(n), C.int(rows), C.int(k)))
}

func (c *cudaBackend) Ink2VQ8Rows(w Tensor, x, gate, q, y Ink2Ref, rows int, quantize bool) (bool, error) {
	if w.Dtype != Q8_0 || len(w.Shape) != 2 || w.Quant.Block != 32 {
		return false, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	wb := c.cudaBufForSubmit(w)
	rc := C.fcuda_ink2v_q8_rows((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.ink2F(x), c.ink2F(gate), c.ink2Ptr(q), ink2Bool(quantize),
		c.ink2F(y), C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(rows), C.int(w.Quant.Block))
	if rc == -1 {
		return false, nil
	}
	return true, ink2Status("verify q8 rows", rc)
}

func (c *cudaBackend) Ink2VMix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n, T int) (bool, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(states) > ink2MaxStates {
		return false, fmt.Errorf("cuda inkling2 verify mix: %d states", len(states))
	}
	// Same block layout as Ink2Mix: states in [0, ink2MaxStates), norms after; calloc
	// leaves an absent norm null so the kernel skips that weight.
	const slots = ink2MaxStates + ink2MaxQueries
	ptrs := (*[slots]*C.float)(C.calloc(slots, C.size_t(unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(ptrs))
	for i, s := range states {
		ptrs[i] = c.ink2F(s)
	}
	var np **C.float
	if norms != nil {
		for i := 0; i < ink2MaxQueries && i < len(norms); i++ {
			if norms[i].T.Buf() != nil {
				ptrs[ink2MaxStates+i] = c.ink2F(norms[i])
			}
		}
		np = (**C.float)(unsafe.Pointer(&ptrs[ink2MaxStates]))
	}
	rc := C.fcuda_ink2v_mix((**C.float)(unsafe.Pointer(ptrs)), C.int(len(states)), c.ink2F(queries), C.int(nq), np, C.float(eps),
		c.ink2F(out), C.int(n), C.int(T))
	if rc == -1 {
		return false, nil
	}
	return true, ink2Status("verify mix", rc)
}

func (c *cudaBackend) Ink2PControlMulti(cur, msg, down, gain, forget, g, out []Ink2Ref, n, rank, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	J := len(cur)
	if J < 1 || J > 4 || len(msg) != J || len(down) != J || len(gain) != J || len(forget) != J || len(g) != J || len(out) != J {
		return fmt.Errorf("cuda inkling2 control multi: %d controllers", J)
	}
	tab := (*[28]*C.float)(C.malloc(C.size_t(28 * unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(tab))
	for j := 0; j < J; j++ {
		tab[j], tab[4+j], tab[8+j] = c.ink2F(cur[j]), c.ink2F(msg[j]), c.ink2F(down[j])
		tab[12+j], tab[16+j], tab[20+j], tab[24+j] = c.ink2F(gain[j]), c.ink2F(forget[j]), c.ink2F(g[j]), c.ink2F(out[j])
	}
	p := func(i int) **C.float { return (**C.float)(unsafe.Pointer(&tab[i])) }
	return ink2Status("prefill control multi", C.fcuda_ink2p_ctrl_multi(p(0), p(4), p(8), p(12), p(16), p(20), p(24),
		C.int(J), C.int(n), C.int(rank), C.int(T)))
}

func (c *cudaBackend) Ink2PConvPairLog(x, w, dyn, hist, log, out []Ink2Ref, norm []float32, n, heads, kn, T int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(x) != 2 || len(w) != 2 || len(dyn) != 2 || len(hist) != 2 || len(log) != 2 || len(out) != 2 || len(norm) != 2 {
		return fmt.Errorf("cuda inkling2 conv pair: want two convs")
	}
	tab := (*[12]*C.float)(C.malloc(C.size_t(12 * unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(tab))
	nm := (*[2]C.float)(C.malloc(C.size_t(2 * unsafe.Sizeof(C.float(0)))))
	defer C.free(unsafe.Pointer(nm))
	for j := 0; j < 2; j++ {
		tab[j], tab[2+j], tab[4+j] = c.ink2F(x[j]), c.ink2F(w[j]), c.ink2F(dyn[j])
		tab[6+j], tab[8+j], tab[10+j] = c.ink2F(hist[j]), c.ink2F(log[j]), c.ink2F(out[j])
		nm[j] = C.float(norm[j])
	}
	p := func(i int) **C.float { return (**C.float)(unsafe.Pointer(&tab[i])) }
	return ink2Status("verify conv pair", C.fcuda_ink2p_conv_pair_log(p(0), p(2), p(4), p(6), p(8), p(10),
		(*C.float)(unsafe.Pointer(nm)), C.int(n), C.int(heads), C.int(kn), C.int(T)))
}

func ink2PGemmF32(w, x, y *C.float, rows, cols, T C.int) C.int {
	if ink2PrefillFast("dense") {
		return C.fcuda_ink2p_gemm_f32_v2(w, x, y, rows, cols, T)
	}
	return C.fcuda_ink2p_gemm_f32(w, x, y, rows, cols, T)
}
