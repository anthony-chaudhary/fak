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

var _ Inkling2BatchDevice = (*cudaBackend)(nil)

// ink2Strides copies st into C memory for one launcher call; the caller frees it.
func ink2Strides(st Ink2SeqStrides) *C.longlong {
	f := st.Fields()
	p := (*[15]C.longlong)(C.malloc(C.size_t(len(f)) * C.size_t(unsafe.Sizeof(C.longlong(0)))))
	for i, v := range f {
		p[i] = C.longlong(v)
	}
	return &p[0]
}

func (c *cudaBackend) Ink2BQKVPost(q, r, k, v, qclip, rclip, kcache, vcache, meta Ink2Ref, p Ink2AttnParams, cap int, cv *Ink2KVConv, st Ink2SeqStrides, n int) error {
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
	sp := ink2Strides(st)
	defer C.free(unsafe.Pointer(sp))
	return ink2Status("qkv post seq", C.fcuda_ink2_qkv_post_seq(c.ink2F(q), c.ink2F(r), c.ink2F(k), c.ink2F(v), c.ink2F(qclip), c.ink2F(rclip),
		c.ink2F(kcache), c.ink2F(vcache), c.ink2F(meta), C.int(p.Heads), C.int(p.KVHeads), C.int(p.HeadDim), C.int(p.RelLen),
		C.float(p.QKClip), C.float(p.RelClip), C.float(p.Eps), ink2Bool(p.Global), C.float(p.Alpha), C.float(p.Floor), C.int(cap),
		conv, kNorm, vNorm, kn, sp, C.int(n)))
}

func (c *cudaBackend) Ink2BAttend(q, r, sinks, relBias, kcache, vcache, scores, out, meta Ink2Ref, p Ink2AttnParams, extent, window, cap int, st Ink2SeqStrides, n int) error {
	if st.Scores%2 != 0 || scores.Off%2 != 0 {
		return fmt.Errorf("cuda inkling2 attend seq: scores offset %d stride %d not float64-aligned", scores.Off, st.Scores)
	}
	cudaMu.Lock()
	defer cudaMu.Unlock()
	sp := ink2Strides(st)
	defer C.free(unsafe.Pointer(sp))
	return ink2Status("attend seq", C.fcuda_ink2_attend_seq(c.ink2F(q), c.ink2F(r), c.ink2F(sinks), c.ink2F(relBias), c.ink2F(kcache),
		c.ink2F(vcache), (*C.double)(c.ink2Ptr(scores)), c.ink2F(out), c.ink2F(meta), C.int(p.Heads), C.int(p.KVHeads),
		C.int(p.HeadDim), C.int(p.RelLen), C.int(extent), C.int(window), C.int(cap), C.float(p.Scale), sp, C.int(n)))
}

func (c *cudaBackend) Ink2BConv(x, w, hist, out, resid Ink2Ref, n, kn int, norm float32, st Ink2SeqStrides, rows int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	sp := ink2Strides(st)
	defer C.free(unsafe.Pointer(sp))
	return ink2Status("conv seq", C.fcuda_ink2_conv_seq(c.ink2F(x), c.ink2F(w), c.ink2F(hist), c.ink2F(out), c.ink2F(resid),
		C.int(n), C.int(kn), C.float(norm), sp, C.int(rows)))
}

func (c *cudaBackend) Ink2BSoftcapArgmax(logits Ink2Ref, n, ld int, softcap float32, outIdx Ink2Ref, rows int) error {
	cudaMu.Lock()
	defer cudaMu.Unlock()
	return ink2Status("softcap argmax seq", C.fcuda_ink2_softcap_argmax_seq(c.ink2F(logits), C.int(n), C.longlong(ld), C.float(softcap),
		c.ink2I(outIdx), C.int(rows)))
}
