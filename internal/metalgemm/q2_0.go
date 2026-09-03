//go:build darwin && arm64 && cgo

// q2_0.go — Go side of the Metal ternary (Q2_0) dequant-GEMV (q2_0.m). The 2-bit twin of q8.go,
// and the Apple-side half of Bonsai's on-device headline (epic #4867, issue #4873):
// Ternary-Bonsai-27B keeps the Qwen3.6 architecture but ships its weights as {-1,0,+1}·d, so this
// is the route that makes a 27B usable on Apple Silicon without an FP16 expansion — ~0.375 B per
// weight resident (8 code bytes + one f32 scale per 32-wide block) fits the ~4 GB on-device
// envelope that f16 (~54 GB) and even q4_k_m (~16 GB) do not.
//
// The raw 2-bit blocks stay resident on the GPU and each thread unpacks its own codes in-shader
// (see q2_0.m), so the bandwidth-dominant stream is the code bytes and the weight is never
// materialized wide. Format and math mirror internal/model.quant_q2.go's g32 form exactly:
// in = nblk*32, each 32-wide block is one f32 scale d plus 8 bytes of 2-bit codes (four per byte,
// low code first), and a code c dequantizes to d*(c-2). Because d = amax, quantization only emits
// c in {1,2,3}, so the live code set is the ternary {-1,0,+1}·d.

package metalgemm

/*
int  mg_q2_0_upload(const unsigned char* codes, const float* scales, int out, int in);
void mg_q2_0_gemv(int wid, const float* x, float* y);
void mg_q2_0_gemv_batch(int wid, const float* Xcat, int n, float* Ycat);
void mg_q2_0_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff);
void mg_q2_0_gemm(int wid, const float* X, int p, float* Y);
int  mg_q2_0_mlp(int gate, int up, int down, const float* x, float* y);
void mg_q2_0_reset(void);

int  mg_q2_0_g128_upload(const unsigned char* raw, int out, int in);
void mg_q2_0_g128_gemv(int wid, const float* x, float* y);
void mg_q2_0_g128_gemm(int wid, const float* X, int p, float* Y);
void mg_q2_0_g128_reset(void);
*/
import "C"

import "unsafe"

// Q2_0Weight is a handle to a ternary Q2_0 weight matrix [Out, In] resident on the GPU (2-bit
// codes + per-32 f32 block scales). In must be a multiple of Q2_0BlockWeights; Nblk = In/32. The
// resident byte cost is Out*Nblk*(Q2_0BlockBytes+4) = Out*In*0.375.
type Q2_0Weight struct {
	id      C.int
	Out, In int
	Nblk    int
}

// UploadQ2_0 copies a ternary Q2_0 payload — codes (out*nblk*8 bytes, row-major, four 2-bit codes
// per byte with the low code first) and block scales (out*nblk f32) — resident onto the GPU and
// returns a handle, or nil if the backend is unavailable, in is not a multiple of 32, or a slice is
// short / the table is full. Both slices are read only during the call (cgo copies them into device
// buffers). The packing is internal/model.quant_q2.go's g32 form verbatim: code c means d*(c-2).
func UploadQ2_0(codes []byte, scales []float32, out, in int) *Q2_0Weight {
	if !Available() || in <= 0 || in%Q2_0BlockWeights != 0 || out <= 0 {
		return nil
	}
	nblk := in / Q2_0BlockWeights
	if len(codes) < out*nblk*Q2_0BlockBytes || len(scales) < out*nblk {
		return nil
	}
	id := C.mg_q2_0_upload((*C.uchar)(unsafe.Pointer(&codes[0])), (*C.float)(unsafe.Pointer(&scales[0])),
		C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Q2_0Weight{id: id, Out: out, In: in, Nblk: nblk}
}

// GEMV computes y[Out] = W · x for one f32 activation row x (length In). y must have length >= Out.
// Both slices are accessed only during the call. This is the ternary decode GEMV; the activation is
// plain f32 (unlike the Q8 GEMV's quantized activation), matching the CPU-ref ternary GEMM.
func (w *Q2_0Weight) GEMV(x, y []float32) {
	if w == nil || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return
	}
	C.mg_q2_0_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
}

// GEMVBatch runs n decode GEMVs of this same ternary weight in ONE command buffer: Xcat is n
// contiguous f32 activation rows (n*In floats), Ycat receives n result rows (n*Out floats). It is
// the ternary twin of Q4KWeight.GEMVBatch — it isolates how much of GEMV's per-call cost is the
// CPU<->GPU submission/sync round-trip (paid once here) vs the kernel (paid n times).
func (w *Q2_0Weight) GEMVBatch(Xcat []float32, n int, Ycat []float32) {
	if w == nil || w.id < 0 || n <= 0 || len(Xcat) < n*w.In || len(Ycat) < n*w.Out {
		return
	}
	C.mg_q2_0_gemv_batch(w.id, (*C.float)(unsafe.Pointer(&Xcat[0])), C.int(n),
		(*C.float)(unsafe.Pointer(&Ycat[0])))
}

// GEMM runs ternary prefill Y[P,Out] = X[P,In] * W^T. X and Y are token-major.
func (w *Q2_0Weight) GEMM(X []float32, P int, Y []float32) {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	C.mg_q2_0_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])))
}

// FusedMLPQ2_0 evaluates y = down(silu(gate*x) * (up*x)) in one Metal command buffer.
// Intermediate activations remain device-resident. False reports a shape/backend failure.
func FusedMLPQ2_0(gate, up, down *Q2_0Weight, x, y []float32) bool {
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || down.id < 0 ||
		gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || len(x) < gate.In || len(y) < down.Out {
		return false
	}
	return C.mg_q2_0_mlp(gate.id, up.id, down.id,
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0]))) != 0
}

// GEMVGroupQ2_0 runs one decode GEMV per weight in ws — all reading the SAME f32 activation x
// (length In, shared) — in a SINGLE Metal command buffer, returning one result slice per weight
// (each length ws[i].Out). Every weight must share x's In. This is the live decode group pattern
// (q/k/v, gate/up): it pays the per-command-buffer submit/sync once for the whole group and
// pipelines the dispatches. Returns nil on a shape mismatch or empty input, so the caller falls
// back to per-weight GEMV.
func GEMVGroupQ2_0(ws []*Q2_0Weight, x []float32) [][]float32 {
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
	C.mg_q2_0_gemv_group(&wids[0], C.int(n), (*C.float)(unsafe.Pointer(&x[0])),
		(*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0])
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		out[i] = ycat[o : o+w.Out : o+w.Out]
		o += w.Out
	}
	return out
}

// ID returns the backend handle for this matrix.
func (w *Q2_0Weight) ID() int { return int(w.id) }

// ResetQ2_0 releases every resident ternary weight buffer and the reused scratch (the Q2_0 twin of
// ResetQ8). Call only when no Q2_0Weight handle is still in use — every prior handle is invalidated.
func ResetQ2_0() { C.mg_q2_0_reset() }

// Q2_0G128Weight is a handle to a standard GGUF group-128 Q2_0 weight matrix [Out, In] resident
// on the GPU (34-byte blocks: 2-byte f16 scale + 32-byte codes). In must be a multiple of
// Q2_0G128BlockWeights (128); Nblk = In / 128.
type Q2_0G128Weight struct {
	id      C.int
	Out, In int
	Nblk    int
}

// UploadQ2_0G128 copies a standard GGUF group-128 Q2_0 payload (34 bytes per 128-weight block)
// resident onto the GPU and returns a handle, or nil if the backend is unavailable, in is not
// a multiple of 128, or the raw slice is short / table is full.
func UploadQ2_0G128(raw []byte, out, in int) *Q2_0G128Weight {
	if !Available() || in <= 0 || in%Q2_0G128BlockWeights != 0 || out <= 0 {
		return nil
	}
	need := Q2_0G128PayloadBytes(out, in)
	if len(raw) < need {
		return nil
	}
	id := C.mg_q2_0_g128_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	return &Q2_0G128Weight{id: id, Out: out, In: in, Nblk: in / Q2_0G128BlockWeights}
}

// ID returns the backend handle for this matrix.
func (w *Q2_0G128Weight) ID() int { return int(w.id) }

// GEMV computes y[Out] = W · x for one f32 activation row x (length In). y must have length >= Out.
func (w *Q2_0G128Weight) GEMV(x, y []float32) {
	if w == nil || w.id < 0 || len(x) < w.In || len(y) < w.Out {
		return
	}
	C.mg_q2_0_g128_gemv(w.id, (*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
}

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ for a resident group-128 Q2_0 matrix over P activation rows.
func (w *Q2_0G128Weight) GEMM(X []float32, P int, Y []float32) {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	C.mg_q2_0_g128_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])))
}

// ResetQ2_0G128 releases every resident group-128 Q2_0 weight buffer and reused scratch.
func ResetQ2_0G128() {
	C.mg_q2_0_g128_reset()
}
