//go:build cuda && cgo

package compute

/*
#include <stdlib.h>
#include "cuda_backend.h"
*/
import "C"

import (
	"fmt"
	"unsafe"
)

var _ Inkling2Device = (*cudaBackend)(nil)

func ink2Status(op string, rc C.int) error {
	if rc != 0 {
		return fmt.Errorf("cuda inkling2 %s: status %d", op, int(rc))
	}
	return nil
}

// ink2Ptr resolves a Ref to a device address; caller holds cudaMu.
func (c *cudaBackend) ink2Ptr(r Ink2Ref) unsafe.Pointer {
	if r.T.buf == nil {
		return nil
	}
	b := c.cudaBufForSubmit(r.T)
	return unsafe.Add(b.ptr, r.Off*4)
}

func (c *cudaBackend) ink2F(r Ink2Ref) *C.float { return (*C.float)(c.ink2Ptr(r)) }
func (c *cudaBackend) ink2I(r Ink2Ref) *C.int   { return (*C.int)(c.ink2Ptr(r)) }

func (c *cudaBackend) Ink2Alloc(n int) (Tensor, error) {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	b := c.dallocClass(n*4, MemoryActivation, "inkling2-resident")
	if err := ink2Status("zero", C.fcuda_ink2_zero(b.ptr, C.size_t(n*4))); err != nil {
		return Tensor{}, err
	}
	return makeTensor(c, F32, RowMajor, []int{n}, nil, b), nil
}

func (c *cudaBackend) Ink2Write(dst Ink2Ref, src []float32) error {
	if len(src) == 0 {
		return nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("write", C.fcuda_h2d_checked(c.ink2Ptr(dst), unsafe.Pointer(&src[0]), C.size_t(len(src)*4)))
}

func (c *cudaBackend) Ink2Read(src Ink2Ref, n int) ([]float32, error) {
	out := make([]float32, n)
	if n == 0 {
		return out, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return out, ink2Status("read", C.fcuda_d2h_checked(unsafe.Pointer(&out[0]), c.ink2Ptr(src), C.size_t(n*4)))
}

func (c *cudaBackend) Ink2ReadInts(src Ink2Ref, n int) ([]int32, error) {
	out := make([]int32, n)
	if n == 0 {
		return out, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return out, ink2Status("read-ints", C.fcuda_d2h_checked(unsafe.Pointer(&out[0]), c.ink2Ptr(src), C.size_t(n*4)))
}

func (c *cudaBackend) Ink2MatVec(w Tensor, x, y Ink2Ref) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(w.Shape) != 2 {
		return fmt.Errorf("cuda inkling2 matvec: weight shape %v", w.Shape)
	}
	out, in := w.Shape[0], w.Shape[1]
	wb := c.cudaBufForSubmit(w)
	xp, yp := c.ink2F(x), c.ink2F(y)
	switch w.Dtype {
	case F32:
		return ink2Status("matvec f32", C.fcuda_ink2_gemv_f32((*C.float)(wb.ptr), xp, yp, C.int(out), C.int(in)))
	case Q8_0:
		return ink2Status("matvec q8_0", C.fcuda_q8_matmul_f32((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), xp, yp,
			C.int(out), C.int(in), 1, C.int(w.Quant.Block)))
	case Q4_K:
		return ink2Status("matvec q4_k", C.fcuda_q4k_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), 1))
	case Q5_K:
		return ink2Status("matvec q5_k", C.fcuda_q5k_matmul_checked_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), 1))
	case Q6_K:
		return ink2Status("matvec q6_k", C.fcuda_q6k_matmul_checked_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), 1))
	case Q3_K:
		return ink2Status("matvec q3_k", C.fcuda_q3k_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), 1))
	case IQ4_XS:
		return ink2Status("matvec iq4_xs", C.fcuda_iq4xs_matmul_f32((*C.uint8_t)(wb.ptr), xp, yp, C.int(out), C.int(in), 1))
	case IQ2_XXS, IQ2_XS, IQ1_M, IQ4_NL, IQ2_S, IQ3_XXS, IQ3_S:
		cudaIQGEMV(w.Dtype, wb, xp, yp, out, in, 1)
		return nil
	}
	return fmt.Errorf("cuda inkling2 matvec: unsupported weight dtype %s", w.Dtype)
}

func (c *cudaBackend) Ink2MatVecQ8(w Tensor, x, gate, q Ink2Ref, quantize bool, y Ink2Ref) (bool, error) {
	if w.Dtype != Q8_0 || len(w.Shape) != 2 || w.Quant.Block != 32 {
		return false, nil
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	wb := c.cudaBufForSubmit(w)
	return true, ink2Status("matvec q8_0 shared", C.fcuda_ink2_q8_matvec((*C.int8_t)(wb.ptr), (*C.float)(wb.scales), c.ink2F(x), c.ink2F(gate),
		c.ink2Ptr(q), ink2Bool(quantize), c.ink2F(y), C.int(w.Shape[0]), C.int(w.Shape[1]), C.int(w.Quant.Block)))
}

var ink2ExpertFormat = map[Dtype]C.int{NVFP4: 0, IQ2_XS: 1, IQ2_S: 2, IQ3_XXS: 3, IQ3_S: 4, IQ4_NL: 5, Q3_K: 6}

func (c *cudaBackend) Ink2ExpertMatVec(stack Tensor, expertRows int, ids, x Ink2Ref, xStride int, y Ink2Ref, k int, scale Ink2Ref, situ *Ink2Situ) (bool, error) {
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
	sw, beta, lb := 0, 0.0, 0.0
	if situ != nil {
		sw, beta, lb = situ.Width, situ.Beta, situ.LinearBeta
	}
	return true, ink2Status("expert gemv", C.fcuda_ink2_expert_gemv(f, (*C.uint8_t)(b.ptr), (*C.uint8_t)(scales), c.ink2I(ids),
		c.ink2F(x), C.int(xStride), c.ink2F(y), C.int(k), C.int(expertRows), C.int(in),
		c.ink2F(scale), C.int(sw), C.double(beta), C.double(lb)))
}

// ink2SelectNVFP4GEMV picks the NVFP4 expert kernel (true: the bandwidth-shaped
// one, false: the reference) and returns the previous choice.
func ink2SelectNVFP4GEMV(fast bool) bool {
	mode := C.int(0)
	if fast {
		mode = 1
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return C.fcuda_ink2_nvfp4_gemv_select(mode) != 0
}

// ink2SelectQ8GEMV picks the dense Q8_0 GEMV of Ink2MatVecQ8 (true: the
// bandwidth-shaped one, false: k_q8_gemm) and returns the previous choice.
func ink2SelectQ8GEMV(fast bool) bool {
	mode := C.int(0)
	if fast {
		mode = 1
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return C.fcuda_ink2_q8_gemv_select(mode) != 0
}

// ink2SelectIQGEMV is ink2SelectNVFP4GEMV for the IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S kernels.
func ink2SelectIQGEMV(fast bool) bool {
	mode := C.int(0)
	if fast {
		mode = 1
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return C.fcuda_ink2_iq_gemv_select(mode) != 0
}

func (c *cudaBackend) Ink2RMS(x, w, y Ink2Ref, rows, n int, eps float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("rms", C.fcuda_ink2_rms_rows(c.ink2F(x), c.ink2F(w), c.ink2F(y), C.int(rows), C.int(n), C.float(eps)))
}

func (c *cudaBackend) Ink2CtrlPair(cur, msg, pair Ink2Ref, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("ctrl pair", C.fcuda_ink2_ctrl_pair(c.ink2F(cur), c.ink2F(msg), c.ink2F(pair), C.int(n)))
}

func (c *cudaBackend) Ink2Mix(states []Ink2Ref, queries Ink2Ref, nq int, norms []Ink2Ref, eps float32, out Ink2Ref, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	if len(states) > 8 || len(norms) > 4 {
		return fmt.Errorf("cuda inkling2 mix: %d states, %d norms", len(states), len(norms))
	}
	ptrs := (*[12]*C.float)(C.malloc(C.size_t(12 * unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(ptrs))
	for i, s := range states {
		ptrs[i] = c.ink2F(s)
	}
	var np **C.float
	if norms != nil {
		for i, w := range norms {
			ptrs[8+i] = c.ink2F(w)
		}
		np = (**C.float)(unsafe.Pointer(&ptrs[8]))
	}
	return ink2Status("mix", C.fcuda_ink2_mix((**C.float)(unsafe.Pointer(ptrs)), C.int(len(states)), c.ink2F(queries), C.int(nq),
		np, C.float(eps), c.ink2F(out), C.int(n)))
}

func (c *cudaBackend) Ink2Control(ctl Ink2Controls, g, out Ink2Ref, n, rank int, norms []Ink2Ref, eps float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	nj := len(ctl.Cur)
	if nj < 1 || nj > 4 || len(ctl.Msg) != nj || len(ctl.Down) != nj || len(ctl.Gain) != nj || len(ctl.Forget) != nj || len(norms) > nj {
		return fmt.Errorf("cuda inkling2 control: %d controllers", nj)
	}
	ptrs := (*[24]*C.float)(C.malloc(C.size_t(24 * unsafe.Sizeof(uintptr(0)))))
	defer C.free(unsafe.Pointer(ptrs))
	for j := 0; j < nj; j++ {
		ptrs[j], ptrs[4+j], ptrs[8+j] = c.ink2F(ctl.Cur[j]), c.ink2F(ctl.Msg[j]), c.ink2F(ctl.Down[j])
		ptrs[12+j], ptrs[16+j] = c.ink2F(ctl.Gain[j]), c.ink2F(ctl.Forget[j])
	}
	var np **C.float
	if norms != nil {
		for j, w := range norms {
			ptrs[20+j] = c.ink2F(w)
		}
		np = (**C.float)(unsafe.Pointer(&ptrs[20]))
	}
	return ink2Status("control", C.fcuda_ink2_control((**C.float)(unsafe.Pointer(ptrs)), C.int(nj), c.ink2F(g), c.ink2F(out),
		C.int(n), C.int(rank), np, C.float(eps)))
}

func (c *cudaBackend) Ink2Conv(x, w, dyn, hist, out, resid Ink2Ref, n, heads, kn int, norm float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("conv", C.fcuda_ink2_conv(c.ink2F(x), c.ink2F(w), c.ink2F(dyn), c.ink2F(hist), c.ink2F(out), c.ink2F(resid),
		C.int(n), C.int(heads), C.int(kn), C.float(norm)))
}

func (c *cudaBackend) Ink2CombineConv(down, wts Ink2Ref, k int, w, hist, out, resid Ink2Ref, n, kn int, norm float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("combine conv", C.fcuda_ink2_combine_conv(c.ink2F(down), c.ink2F(wts), C.int(k), c.ink2F(w), c.ink2F(hist),
		c.ink2F(out), c.ink2F(resid), C.int(n), C.int(kn), C.float(norm)))
}

func ink2Bool(b bool) C.int {
	if b {
		return 1
	}
	return 0
}

func (c *cudaBackend) Ink2QKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	var conv **C.float
	var kNorm, vNorm C.float
	var kn C.int
	if cv != nil {
		ptrs := (*[6]*C.float)(C.malloc(C.size_t(6 * unsafe.Sizeof(uintptr(0)))))
		defer C.free(unsafe.Pointer(ptrs))
		ptrs[0], ptrs[1], ptrs[2] = c.ink2F(cv.KW), c.ink2F(cv.VW), c.ink2F(cv.KDyn)
		ptrs[3], ptrs[4], ptrs[5] = c.ink2F(cv.VDyn), c.ink2F(cv.KHist), c.ink2F(cv.VHist)
		conv = (**C.float)(unsafe.Pointer(ptrs))
		kNorm, vNorm, kn = C.float(cv.KNorm), C.float(cv.VNorm), C.int(cv.Kn)
	}
	return ink2Status("qkv post", C.fcuda_ink2_qkv_post(c.ink2F(q), c.ink2F(r), c.ink2F(k), c.ink2F(v), c.ink2F(qclip), c.ink2F(rclip),
		c.ink2F(kcache), c.ink2F(vcache), c.ink2F(meta), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen),
		C.float(p.QKClip), C.float(p.RelClip), C.float(p.Eps), ink2Bool(p.Global), C.float(p.Alpha), C.float(p.Floor), C.int(cap),
		conv, kNorm, vNorm, kn))
}

func (c *cudaBackend) Ink2Attend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("attend", C.fcuda_ink2_attend(c.ink2F(q), c.ink2F(r), c.ink2F(sinks), c.ink2F(relBias), c.ink2F(kcache),
		c.ink2F(vcache), (*C.double)(c.ink2Ptr(scores)), c.ink2F(out), c.ink2F(meta), C.int(p.Heads), C.int(p.KVHeads),
		C.int(p.HeadDim), C.int(p.RelLen), C.int(extent), C.int(window), C.int(cap), C.float(p.Scale)))
}

func (c *cudaBackend) Ink2SigmoidMul(x, gate Ink2Ref, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("sigmoid mul", C.fcuda_ink2_sigmoid_mul(c.ink2F(x), c.ink2F(gate), C.int(n)))
}

func (c *cudaBackend) Ink2Route(logits, bias, ids, wts Ink2Ref, nexp, k int, scale float32) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("route", C.fcuda_ink2_route(c.ink2F(logits), c.ink2F(bias), c.ink2I(ids), c.ink2F(wts), C.int(nexp), C.int(k), C.float(scale)))
}

func (c *cudaBackend) Ink2ExpertScale(y, ids, scale Ink2Ref, k, rows int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("expert scale", C.fcuda_ink2_expert_scale(c.ink2F(y), c.ink2I(ids), c.ink2F(scale), C.int(k), C.int(rows)))
}

func (c *cudaBackend) Ink2Situ(gu, act Ink2Ref, k, width, actStride int, beta, linearBeta float64) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("situ", C.fcuda_ink2_situ(c.ink2F(gu), c.ink2F(act), C.int(k), C.int(width), C.int(actStride), C.double(beta), C.double(linearBeta)))
}

func (c *cudaBackend) Ink2Combine(down, wts, out Ink2Ref, k, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("moe combine", C.fcuda_ink2_moe_combine(c.ink2F(down), c.ink2F(wts), c.ink2F(out), C.int(k), C.int(n)))
}

func (c *cudaBackend) Ink2Add(a, b, out Ink2Ref, n int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("add", C.fcuda_ink2_add(c.ink2F(a), c.ink2F(b), c.ink2F(out), C.int(n)))
}

func (c *cudaBackend) Ink2SoftcapArgmax(logits Ink2Ref, n int, softcap float32, outIdx Ink2Ref) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("softcap argmax", C.fcuda_ink2_softcap_argmax(c.ink2F(logits), C.int(n), C.float(softcap), c.ink2I(outIdx)))
}

func (c *cudaBackend) Ink2GraphBegin() error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("graph begin", C.fcuda_ink2_graph_begin())
}

func (c *cudaBackend) Ink2GraphEnd() error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("graph end", C.fcuda_ink2_graph_end())
}

func (c *cudaBackend) Ink2GraphLaunch() error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("graph launch", C.fcuda_ink2_graph_launch())
}

func (c *cudaBackend) Ink2GraphReset() {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	C.fcuda_ink2_graph_reset()
}
