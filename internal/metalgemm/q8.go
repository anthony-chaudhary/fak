//go:build darwin && arm64 && cgo

// q8.go — Go side of the Metal Q8_0 dequant-GEMV/GEMM (q8.m). The Q8 twin of q4k.go. It exists
// because the Gated-DeltaNet token mixer's projections (every linear_attn.* weight, plus the
// full-attn q/k) are reordered/unpermuted for qwen35 and so resolve to Q8 (internal/model.q8Tensor)
// rather than raw resident q4_k — the q4_k GPU kernels can't touch them. A GPU-resident decode
// forward (#67) therefore needs this Q8 GEMV to move the GDN projections off the CPU and keep the
// device busy across the whole token. Weight = int8 codes + per-32-block f32 scales; the activation
// is Q8_0-quantized too (codes + per-block scales), so the dot is the same int8×int8→int32 reduction
// the CPU qMatRows takes — see qdot8scalar.

package metalgemm

/*
#include <stdint.h>
typedef struct {
    uintptr_t command_buffer;
    int committed;
    int completed_wait;
    int host_readback;
    int encoders;
    double gpu_milliseconds;
    double wait_milliseconds;
    int timing_available;
} mg_execution_event;
int  mg_q8_upload(const signed char* codes, const float* scales, int out, int in);
int  mg_q8_alias(const signed char* codes, const float* scales, int out, int in);
void mg_q8_release(int wid);
int  mg_q8_live_count(void);
void mg_q8_gemv(int wid, const signed char* xq, const float* xd, float* y, mg_execution_event* event);
void mg_q8_gemv_group(const int* wids, int n, const signed char* xq, const float* xd, float* Ycat, const int* yoff, mg_execution_event* event);
void mg_q8_gemm(int wid, const signed char* Xq, const float* Xd, int P, float* Y, mg_execution_event* event);
void mg_q8_gemm_group(const int* wids, int n, const signed char* Xq, const float* Xd, int P, float* Ycat, const int* yoff, mg_execution_event* event);
void mg_q8_reset(void);
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

func recordQ8Event(observation *ExecutionObservation, event *C.mg_execution_event) {
	if event == nil {
		return
	}
	observation.record(uintptr(event.command_buffer), event.committed != 0, event.completed_wait != 0,
		event.host_readback != 0, int(event.encoders), float64(event.gpu_milliseconds),
		float64(event.wait_milliseconds), event.timing_available != 0)
}

// Q8Weight is a handle to a Q8_0 weight matrix [Out, In] resident on the GPU (int8 codes + per-32
// f32 block scales). In must be a multiple of 32 (the Q8_0 block size); Nblk = In/32.
type Q8Weight struct {
	mu      sync.RWMutex
	id      C.int
	Out, In int
	Nblk    int
	pins    runtime.Pinner
	aliased bool
}

// UploadQ8 copies a Q8_0 payload — codes (out*in int8, row-major) and block scales (out*nblk f32)
// — resident onto the GPU and returns a handle, or nil if the backend is unavailable, in is not a
// multiple of 32, or a slice is short / the table is full. Both slices are read only during the
// call (cgo copies them into device buffers).
func UploadQ8(codes []int8, scales []float32, out, in int) *Q8Weight {
	if !Available() || in <= 0 || in%32 != 0 || out <= 0 {
		return nil
	}
	nblk := in / 32
	if len(codes) < out*in || len(scales) < out*nblk {
		return nil
	}
	id := C.mg_q8_upload((*C.schar)(unsafe.Pointer(&codes[0])), (*C.float)(unsafe.Pointer(&scales[0])),
		C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Q8Weight{id: id, Out: out, In: in, Nblk: nblk}
}

// AliasQ8 exposes page-aligned, model-owned Q8 storage to Metal without copying it. Failure is
// fail-closed: unlike UploadQ8 it never substitutes a copied buffer. The returned handle pins both
// owners until Release has first released the native views.
func AliasQ8(codes []int8, scales []float32, out, in int) *Q8Weight {
	if !Available() || in <= 0 || in%32 != 0 || out <= 0 {
		return nil
	}
	nblk := in / 32
	if len(codes) < out*in || len(scales) < out*nblk {
		return nil
	}
	w := &Q8Weight{id: -1, Out: out, In: in, Nblk: nblk, aliased: true}
	w.pins.Pin(&codes[0])
	w.pins.Pin(&scales[0])
	id := C.mg_q8_alias((*C.schar)(unsafe.Pointer(&codes[0])), (*C.float)(unsafe.Pointer(&scales[0])),
		C.int(out), C.int(in))
	if id < 0 {
		w.pins.Unpin()
		return nil
	}
	w.id = id
	return w
}

// Release drops native buffers before unpinning aliased Go backing. It is idempotent and waits for
// any synchronous GEMV/GEMM call already using this handle.
func (w *Q8Weight) Release() {
	if w == nil {
		return
	}
	w.mu.Lock()
	defer w.mu.Unlock()
	if w.id < 0 {
		return
	}
	C.mg_q8_release(w.id)
	w.id = -1
	if w.aliased {
		w.pins.Unpin()
		w.aliased = false
	}
}

// NoCopy reports whether this live handle aliases its caller-owned payload.
func (w *Q8Weight) NoCopy() bool {
	if w == nil {
		return false
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return w.id >= 0 && w.aliased
}

// GEMV computes y[Out] = W · x for one Q8_0-quantized activation: xq are the in int8 codes, xd the
// nblk per-block f32 scales. y must have length >= Out. All slices are accessed only during the call.
func (w *Q8Weight) GEMVWithEvents(xq []int8, xd []float32, y []float32, observation *ExecutionObservation) {
	if w == nil {
		return
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w == nil || w.id < 0 || len(xq) < w.In || len(xd) < w.Nblk || len(y) < w.Out {
		return
	}
	var event C.mg_execution_event
	C.mg_q8_gemv(w.id, (*C.schar)(unsafe.Pointer(&xq[0])), (*C.float)(unsafe.Pointer(&xd[0])),
		(*C.float)(unsafe.Pointer(&y[0])), &event)
	recordQ8Event(observation, &event)
}

// GEMVGroupQ8 runs one decode GEMV per weight in ws — all reading the SAME Q8_0 activation (xq
// codes [In], xd scales [Nblk], shared) — in a SINGLE Metal command buffer, returning one result
// slice per weight (each length ws[i].Out). Every weight must share the activation's In. This is the
// live GDN decode group (the in_proj quad): it pays the per-command-buffer submit/sync once for the
// whole group and pipelines the dispatches. Returns nil on a shape mismatch or empty input.
func GEMVGroupQ8WithEvents(ws []*Q8Weight, xq []int8, xd []float32, observation *ExecutionObservation) [][]float32 {
	n := len(ws)
	if n == 0 {
		return nil
	}
	if !lockQ8Group(ws) {
		return nil
	}
	defer unlockQ8Group(ws)
	if len(xq) < ws[0].In || len(xd) < ws[0].Nblk {
		return nil
	}
	in := ws[0].In
	wids := make([]C.int, n)
	yoff := make([]C.int, n+1)
	off := 0
	for i, w := range ws {
		if w == nil || w.id < 0 || w.In != in {
			return nil
		}
		wids[i] = w.id
		yoff[i] = C.int(off)
		off += w.Out
	}
	yoff[n] = C.int(off)
	ycat := make([]float32, off)
	var event C.mg_execution_event
	C.mg_q8_gemv_group(&wids[0], C.int(n), (*C.schar)(unsafe.Pointer(&xq[0])), (*C.float)(unsafe.Pointer(&xd[0])),
		(*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0], &event)
	recordQ8Event(observation, &event)
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		out[i] = ycat[o : o+w.Out : o+w.Out]
		o += w.Out
	}
	return out
}

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ for a Q8_0-quantized activation panel: Xq are
// P*In int8 codes, Xd are P*Nblk per-block f32 scales. Y must have length >= P*Out. The
// call is the prefill twin of GEMV and runs the whole panel in one Metal command buffer.
func (w *Q8Weight) GEMMWithEvents(Xq []int8, Xd []float32, P int, Y []float32, observation *ExecutionObservation) {
	if w == nil {
		return
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	if w == nil || w.id < 0 || P <= 0 || len(Xq) < P*w.In || len(Xd) < P*w.Nblk || len(Y) < P*w.Out {
		return
	}
	var event C.mg_execution_event
	C.mg_q8_gemm(w.id, (*C.schar)(unsafe.Pointer(&Xq[0])), (*C.float)(unsafe.Pointer(&Xd[0])),
		C.int(P), (*C.float)(unsafe.Pointer(&Y[0])), &event)
	recordQ8Event(observation, &event)
}

// GEMMGroupQ8 runs one batched prefill GEMM per weight in ws, all reading the SAME Q8_0 activation
// panel X[P, In], in a SINGLE Metal command buffer. It is the prefill twin of GEMVGroupQ8 for
// Qwen3.6's linear_attn in-proj groups: one upload of the activation panel and one submit/sync
// covers several Q8-minority projections.
func GEMMGroupQ8WithEvents(ws []*Q8Weight, Xq []int8, Xd []float32, P int, observation *ExecutionObservation) [][]float32 {
	n := len(ws)
	if n == 0 || P <= 0 {
		return nil
	}
	if !lockQ8Group(ws) {
		return nil
	}
	defer unlockQ8Group(ws)
	if len(Xq) < P*ws[0].In || len(Xd) < P*ws[0].Nblk {
		return nil
	}
	in, nblk := ws[0].In, ws[0].Nblk
	wids := make([]C.int, n)
	yoff := make([]C.int, n+1)
	off := 0
	for i, w := range ws {
		if w == nil || w.id < 0 || w.In != in || w.Nblk != nblk {
			return nil
		}
		wids[i] = w.id
		yoff[i] = C.int(off)
		off += P * w.Out
	}
	yoff[n] = C.int(off)
	ycat := make([]float32, off)
	var event C.mg_execution_event
	C.mg_q8_gemm_group(&wids[0], C.int(n), (*C.schar)(unsafe.Pointer(&Xq[0])), (*C.float)(unsafe.Pointer(&Xd[0])),
		C.int(P), (*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0], &event)
	recordQ8Event(observation, &event)
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		sz := P * w.Out
		out[i] = ycat[o : o+sz : o+sz]
		o += sz
	}
	return out
}

func lockQ8Group(ws []*Q8Weight) bool {
	seen := make(map[*Q8Weight]struct{}, len(ws))
	for _, w := range ws {
		if w == nil {
			for held := range seen {
				held.mu.RUnlock()
			}
			return false
		}
		if _, duplicate := seen[w]; duplicate {
			for held := range seen {
				held.mu.RUnlock()
			}
			return false
		}
		w.mu.RLock()
		seen[w] = struct{}{}
	}
	return true
}

func unlockQ8Group(ws []*Q8Weight) {
	for i := len(ws) - 1; i >= 0; i-- {
		ws[i].mu.RUnlock()
	}
}

// ID returns the backend handle for this matrix.
func (w *Q8Weight) ID() int {
	if w == nil {
		return -1
	}
	w.mu.RLock()
	defer w.mu.RUnlock()
	return int(w.id)
}

// LiveQ8Weights returns the native table's occupied-slot count for lifecycle tests.
func LiveQ8Weights() int { return int(C.mg_q8_live_count()) }

// ResetQ8 releases every resident Q8 weight buffer and the reused scratch (the Q8 twin of
// ResetQ4K). Call only when no Q8Weight handle is still in use — every prior handle is invalidated.
func ResetQ8() { C.mg_q8_reset() }

func (w *Q8Weight) GEMV(xq []int8, xd []float32, y []float32) { w.GEMVWithEvents(xq, xd, y, nil) }
func GEMVGroupQ8(ws []*Q8Weight, xq []int8, xd []float32) [][]float32 {
	return GEMVGroupQ8WithEvents(ws, xq, xd, nil)
}
func (w *Q8Weight) GEMM(Xq []int8, Xd []float32, P int, Y []float32) {
	w.GEMMWithEvents(Xq, Xd, P, Y, nil)
}
func GEMMGroupQ8(ws []*Q8Weight, Xq []int8, Xd []float32, P int) [][]float32 {
	return GEMMGroupQ8WithEvents(ws, Xq, Xd, P, nil)
}
