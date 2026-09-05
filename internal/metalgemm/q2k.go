//go:build darwin && arm64 && cgo

// q2k.go — Go wrapper for Metal Q2_K 2-bit k-quant GEMV/GEMM (q2k.m).

package metalgemm

/*
int  mg_q2k_upload(const unsigned char* raw, int out, int in);
void mg_q2k_gemv(int wid, const float* x, float* y);
void mg_q2k_gemm(int wid, const float* X, int P, float* Y);
void mg_q2k_release(int wid);
void mg_q2k_reset(void);
*/
import "C"

import "unsafe"

// Q2KWeight is a handle to a raw Q2_K weight matrix [Out, In] resident on the GPU (84-byte
// super-blocks). In must be a multiple of Q2KBlockWeights (256); Nblk = In/256.
// The resident byte cost is Out * Nblk * 84 = Out * In * 0.328125 bytes.
type Q2KWeight struct {
	id      C.int
	owned   bool // Only UploadQ2K creates ownership; the zero value is inert.
	Out, In int
	Nblk    int
}

// UploadQ2K copies a raw Q2_K payload (verbatim GGUF 84-byte super-blocks, length
// out*(in/256)*84) resident onto the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full.
func UploadQ2K(raw []byte, out, in int) *Q2KWeight {
	if !Available() || in <= 0 || in%Q2KBlockWeights != 0 || out <= 0 {
		return nil
	}
	need := Q2KPayloadBytes(out, in)
	if len(raw) < need {
		return nil
	}
	id := C.mg_q2k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Q2KWeight{id: id, owned: true, Out: out, In: in, Nblk: in / Q2KBlockWeights}
}

// ID returns the opaque backend handle for this matrix, not a table index.
func (w *Q2KWeight) ID() int {
	if w == nil || !w.owned {
		return -1
	}
	return int(w.id)
}

// GEMV computes y[Out] = W · x for one f32 activation row x (length In). y must have length >= Out.
func (w *Q2KWeight) GEMV(x, y []float32) {
	if w == nil || !w.owned || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return
	}
	C.mg_q2k_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
}

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ for a resident Q2_K matrix over P activation rows.
func (w *Q2KWeight) GEMM(X []float32, P int, Y []float32) {
	if w == nil || !w.owned || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	C.mg_q2k_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])))
}

// Release invalidates this handle and releases only its resident Metal buffer.
// It is nil-safe and idempotent. As with Q4K, callers must serialize backend use
// and ensure no GEMV/GEMM is in flight; Release adds no synchronization.
func (w *Q2KWeight) Release() {
	if w == nil || !w.owned || w.id < 0 {
		return
	}
	C.mg_q2k_release(w.id)
	w.id = -1
	w.owned = false
}

// ResetQ2K releases every resident Q2_K weight buffer and reused scratch.
// Use Release for per-model teardown; ResetQ2K invalidates all owners.
func ResetQ2K() {
	C.mg_q2k_reset()
}
