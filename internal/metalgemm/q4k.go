//go:build darwin && cgo && fakmetal

// q4k.go — Go side of the Metal q4_k dequant-GEMV/GEMM (q4k.m). This is the only resident
// route that fits a 27B model on the 36 GB unified pool (q4_k_m ≈ 16 GB; f16 ≈ 54 GB does
// not), and the only path to the llama.cpp-Metal bar: the CPU int8-SDOT kernel is
// compute-bound at ~23 GB/s and tops out ~1.4 tok/s decode, while the GPU has both the
// bandwidth and the parallel dequant FLOPs (which is how llama.cpp reaches 7.29/51.55 tok/s).
// The raw q4_k super-blocks stay resident on the GPU and each thread dequants its weight row
// on the fly, dotting against the f32 activation.

package metalgemm

/*
int  mg_q4k_upload(const unsigned char* raw, int out, int in);
void mg_q4k_gemv(int wid, const float* x, float* y);
void mg_q4k_gemv_batch(int wid, const float* Xcat, int n, float* Ycat);
void mg_q4k_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff);
void mg_q4k_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y);
void mg_q4k_gemm(int wid, const float* X, int P, float* Y);
void mg_q4k_reset(void);
*/
import "C"

import "unsafe"

// Q4KWeight is a handle to a raw q4_k weight matrix [Out, In] resident on the GPU. In must be
// a multiple of 256 (the q4_k super-block size); the resident byte cost is Out*(In/256)*144.
type Q4KWeight struct {
	id      C.int
	Out, In int
}

// UploadQ4K copies a row-major q4_k payload (the verbatim GGUF super-block bytes, length
// out*(in/256)*144) resident onto the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full. The
// slice is read only during this call (cgo copies it into the device buffer).
func UploadQ4K(raw []byte, out, in int) *Q4KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	if len(raw) < out*(in/256)*144 {
		return nil
	}
	id := C.mg_q4k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Q4KWeight{id: id, Out: out, In: in}
}

// GEMV computes y[Out] = W · x for one f32 activation row x (length In). y must have length
// >= Out. Both slices are accessed only during the call. This is the decode GEMV.
func (w *Q4KWeight) GEMV(x, y []float32) {
	if w == nil || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return
	}
	C.mg_q4k_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
}

// GEMVBatch runs n decode GEMVs of this same weight in ONE command buffer: Xcat is n contiguous
// activation rows (n*In floats), Ycat receives n result rows (n*Out floats). It is a measurement
// primitive for issue #67 — it isolates how much of GEMV's per-call cost is the CPU<->GPU
// submission/sync round-trip (paid once here) vs the kernel (paid n times).
func (w *Q4KWeight) GEMVBatch(Xcat []float32, n int, Ycat []float32) {
	if w == nil || w.id < 0 || n <= 0 || len(Xcat) < n*w.In || len(Ycat) < n*w.Out {
		return
	}
	C.mg_q4k_gemv_batch(w.id, (*C.float)(unsafe.Pointer(&Xcat[0])), C.int(n), (*C.float)(unsafe.Pointer(&Ycat[0])))
}

// GEMVGroup runs one decode GEMV per weight in ws — all reading the SAME activation x (length
// In, shared) — in a SINGLE Metal command buffer, and returns one result slice per weight (each
// length ws[i].Out). Every weight must share x's In. This is the live decode group pattern
// (q/k/v, gate/up, the GDN in_proj quad): it pays the per-command-buffer submit/sync once for the
// whole group and pipelines the dispatches. Returns nil on a shape mismatch or empty input.
func GEMVGroup(ws []*Q4KWeight, x []float32) [][]float32 {
	n := len(ws)
	if n == 0 || ws[0] == nil || len(x) < ws[0].In {
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
	C.mg_q4k_gemv_group(&wids[0], C.int(n), (*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0])
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		out[i] = ycat[o : o+w.Out : o+w.Out]
		o += w.Out
	}
	return out
}

// FusedMLP runs a whole dense SwiGLU MLP for one decode token — y = down( silu(gate·x) * (up·x) )
// — in ONE Metal command buffer, keeping the intermediate-wide gate/up/inter resident on the GPU
// (only x and y cross the boundary). Requires gate.In==up.In==down.Out (=H), gate.Out==up.Out==
// down.In (=I); len(x)>=H, len(y)>=H. Returns false on a shape mismatch (caller uses the per-matmul
// path). The activation is silu — the caller must gate on a non-GELU config.
func FusedMLP(gate, up, down *Q4KWeight, x, y []float32) bool {
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || down.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || down.Out != gate.In {
		return false
	}
	if len(x) < gate.In || len(y) < down.Out {
		return false
	}
	C.mg_q4k_mlp(gate.id, up.id, down.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
	return true
}

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ (batched prefill GEMM). X and Y are f32 row-major;
// Y must have length >= P*Out. Both slices are accessed only during the call.
func (w *Q4KWeight) GEMM(X []float32, P int, Y []float32) {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	C.mg_q4k_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])))
}

// ID returns the backend handle for this matrix.
func (w *Q4KWeight) ID() int { return int(w.id) }

// ResetQ4K releases every resident q4_k weight buffer and the reused scratch (the q4_k twin of
// Reset). Call only when no Q4KWeight handle is still in use — every prior handle is invalidated.
func ResetQ4K() { C.mg_q4k_reset() }
