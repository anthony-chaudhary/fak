//go:build darwin && cgo && fakmetal

// Package metalgemm is the optional Metal GPU GEMM backend for the prefill projections.
// It is compiled ONLY under `-tags fakmetal` (and only on darwin with cgo), so the default
// `go build` stays pure-Go — the in-kernel trust posture (one static binary, no FFI) is the
// default; Metal is an explicitly-opted-in acceleration lane for reaching llama.cpp-Metal
// prefill parity on Apple Silicon, where prefill is compute-bound and the GPU's FLOP
// advantage is the only way past the hand-tuned-CPU-kernel ceiling.
//
// The backend holds each weight matrix as f16 in unified memory and runs Y = X·Wᵀ through
// MPSMatrixMultiplication. See metal.m for the device/buffer/ARC details.
package metalgemm

/*
#cgo CFLAGS: -x objective-c -fobjc-arc
#cgo LDFLAGS: -framework Metal -framework MetalPerformanceShaders -framework Foundation -framework CoreFoundation -framework Accelerate
#include <stdlib.h>
int  mg_init(void);
int  mg_upload(const float *w, int out, int in);
void mg_matmul(int wid, const float *x, int P, float *y);
void mg_free(int wid);
void mg_reset(void);

int  mg_upload_vec(const float *v, int n);
void mg_fwd_config(int nLayers, int H, int hd, int nH, int nKV, int Im, float eps, float theta, int attnBias);
void mg_fwd_layer(int layer, int q, int k, int v, int o, int gate, int up, int down,
                  int inNorm, int postNorm, int qb, int kb, int vb);
void mg_fwd_final_norm(int id);
int  mg_prefill(const float *X, int P, float *lastPre, float *KrawOut, float *KpostOut, float *Vout);
*/
import "C"

import (
	"sync"
	"unsafe"
)

var (
	initOnce sync.Once
	ready    bool
)

// Available reports whether a Metal device + command queue are present and initialised. MPS is
// a SEPARATE, opt-in capability (FAK_METAL_MPS) — the pure-MSL q4_k prefill path needs no MPS, so
// Available is true on a headless/SSH device where MPSSupportsMTLDevice would otherwise fault.
// Safe to call repeatedly; the device probe runs once.
func Available() bool {
	initOnce.Do(func() { ready = C.mg_init() == 1 })
	return ready
}

// Compiled reports that this binary was built with the Metal backend linked in (the
// `fakmetal` tag). The stub returns false. Distinguishes "not built with Metal" from
// "built with Metal but no usable device".
func Compiled() bool { return true }

// Weight is a handle to an f16 weight matrix [Out, In] resident on the GPU.
type Weight struct {
	id      C.int
	Out, In int
}

// Upload converts a row-major f32 weight [out, in] to f16 on the GPU and returns a handle,
// or nil if the backend is unavailable or out of weight slots. The caller's slice is read
// only during this call (cgo copies it into the device buffer).
func Upload(w []float32, out, in int) *Weight {
	if !Available() || len(w) < out*in {
		return nil
	}
	id := C.mg_upload((*C.float)(unsafe.Pointer(&w[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Weight{id: id, Out: out, In: in}
}

// MatMul computes y[P, Out] = x[P, In] · Wᵀ on the GPU. x and y are f32 row-major; y must
// have length >= P*Out. Both slices are accessed only during the call.
func (w *Weight) MatMul(x []float32, P int, y []float32) {
	C.mg_matmul(w.id, (*C.float)(unsafe.Pointer(&x[0])), C.int(P), (*C.float)(unsafe.Pointer(&y[0])))
}

// Free releases the GPU buffer backing this weight.
func (w *Weight) Free() {
	if w == nil || w.id < 0 {
		return
	}
	C.mg_free(w.id)
	w.id = -1
}

// ID returns the backend weight-table handle for this matrix, so the GPU-resident forward
// (forward.m) can reference an already-uploaded projection by id instead of re-uploading it.
func (w *Weight) ID() int { return int(w.id) }

// UploadVec stores a 1-D f32 vector (an RMSNorm weight or a q/k/v bias) as f16 in the weight
// table and returns its id, or -1 if the backend is unavailable or out of slots. Used by the
// GPU-resident forward to keep norm/bias on-device alongside the projection matrices.
func UploadVec(v []float32) int {
	if !Available() || len(v) == 0 {
		return -1
	}
	return int(C.mg_upload_vec((*C.float)(unsafe.Pointer(&v[0])), C.int(len(v))))
}

// FwdConfig records the model geometry the GPU-resident forward needs (layer count, hidden
// size, head dim, query/KV head counts, MLP intermediate size, RMSNorm eps, RoPE theta, and
// whether the attention projections carry a bias). Call once per model before FwdLayer.
func FwdConfig(nLayers, H, hd, nH, nKV, I int, eps, theta float32, attnBias bool) {
	b := C.int(0)
	if attnBias {
		b = 1
	}
	C.mg_fwd_config(C.int(nLayers), C.int(H), C.int(hd), C.int(nH), C.int(nKV), C.int(I),
		C.float(eps), C.float(theta), b)
}

// FwdLayer registers one transformer layer's resident weight ids: the seven projection
// handles (q/k/v/o, gate/up/down), the two RMSNorm vectors (input + post-attention), and the
// q/k/v bias ids (-1 when the model has no attention bias). Call once per layer after FwdConfig.
func FwdLayer(layer, q, k, v, o, gate, up, down, inNorm, postNorm, qb, kb, vb int) {
	C.mg_fwd_layer(C.int(layer), C.int(q), C.int(k), C.int(v), C.int(o), C.int(gate),
		C.int(up), C.int(down), C.int(inNorm), C.int(postNorm), C.int(qb), C.int(kb), C.int(vb))
}

// FwdFinalNorm records the model.norm weight id (the final RMSNorm).
func FwdFinalNorm(id int) { C.mg_fwd_final_norm(C.int(id)) }

// Reset tears down ALL resident Metal state: it frees every uploaded weight buffer (the
// f16 projection/norm/bias set), the reused matmul scratch, and the GPU-resident
// forward's per-model topology, returning the backend to an empty weight table. The
// device, queue, and compiled kernels stay live. Call ONLY when no *Weight handle is
// still in use — every prior handle (and every UploadVec id) is invalidated. Its purpose
// is to stop the f16 weight set from accumulating across in-process model reloads (the
// per-load leak that, stacked across concurrent processes, helped exhaust unified memory
// on 2026-06-18). A caller in package model that invokes this must also drop its
// per-*Model handle caches (metalWt / metalResidentReady) in the same critical section.
func Reset() { C.mg_reset() }

// Prefill runs the whole fresh prefill on the GPU (one command buffer, one sync). X is the
// f32 token embeddings [P*H]. It returns the last token's pre-final-norm hidden (lastPre,
// [H]) plus the per-layer KV the CPU cache needs — kraw (pre-RoPE K), kpost (post-RoPE K) and
// v, each laid out [nLayers*P*w] where w = nKV*hd. ok is false if the backend declined (e.g.
// pipelines failed to compile), so the caller can fall back to the hybrid/CPU path.
func Prefill(X []float32, P, nLayers, w, H int) (lastPre, kraw, kpost, v []float32, ok bool) {
	if !Available() || len(X) < P*H {
		return nil, nil, nil, nil, false
	}
	lastPre = make([]float32, H)
	n := nLayers * P * w
	kraw = make([]float32, n)
	kpost = make([]float32, n)
	v = make([]float32, n)
	r := C.mg_prefill((*C.float)(unsafe.Pointer(&X[0])), C.int(P),
		(*C.float)(unsafe.Pointer(&lastPre[0])),
		(*C.float)(unsafe.Pointer(&kraw[0])),
		(*C.float)(unsafe.Pointer(&kpost[0])),
		(*C.float)(unsafe.Pointer(&v[0])))
	if r != 1 {
		return nil, nil, nil, nil, false
	}
	return lastPre, kraw, kpost, v, true
}
