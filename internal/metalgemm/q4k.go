//go:build darwin && arm64 && cgo

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
int  mg_q4k_upload_nocopy(const unsigned char* raw, int out, int in);
void mg_q4k_gemv(int wid, const float* x, float* y);
void mg_q4k_gemv_batch(int wid, const float* Xcat, int n, float* Ycat);
void mg_q4k_gemv_group(const int* wids, int n, const float* x, float* Ycat, const int* yoff);
void mg_q4k_mlp(int gate_wid, int up_wid, int down_wid, const float* x, float* y);
int  mg_q6k_upload(const unsigned char* raw, int out, int in);
void mg_q4k_mlp_q6down(int gate_wid, int up_wid, int down_wid, const float* x, float* y);
int  mg_q4k_mlp_q6down_batch(const int* gate_wids, const int* up_wids, const int* down_wids, int n, const float* x, float* Ycat);
void mg_q4k_gemm(int wid, const float* X, int P, float* Y);
void mg_q4k_gemm_group(const int* wids, int n, const float* X, int P, float* Ycat, const int* yoff);
void mg_q4k_set_use_mm(int on);
void mg_q4k_reset(void);
*/
import "C"

import (
	"runtime"
	"sync"
	"unsafe"
)

// Q4KWeight is a handle to a raw q4_k weight matrix [Out, In] resident on the GPU. In must be
// a multiple of 256 (the q4_k super-block size); the resident byte cost is Out*(In/256)*144.
type Q4KWeight struct {
	id      C.int
	Out, In int
	noCopy  bool
}

type q4kPinnedRaw struct {
	pin *runtime.Pinner
	raw []byte
}

var (
	q4kPinMu sync.Mutex
	q4kPins  = map[int]q4kPinnedRaw{}
)

// UploadQ4K makes a row-major q4_k payload (the verbatim GGUF super-block bytes, length
// out*(in/256)*144) resident for the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full.
// On Apple unified memory it first tries newBufferWithBytesNoCopy against the existing resident
// Go bytes, keeping their backing array pinned until ResetQ4K. If Metal rejects the no-copy
// buffer, it falls back to the older shared-buffer copy upload.
func UploadQ4K(raw []byte, out, in int) *Q4KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	need := out * (in / 256) * 144
	if len(raw) < need {
		return nil
	}
	raw = raw[:need]
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()

	pin := new(runtime.Pinner)
	pin.Pin(&raw[0])
	id := C.mg_q4k_upload_nocopy((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id >= 0 {
		q4kPins[int(id)] = q4kPinnedRaw{pin: pin, raw: raw}
		return &Q4KWeight{id: id, Out: out, In: in, noCopy: true}
	}
	pin.Unpin()

	id = C.mg_q4k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	runtime.KeepAlive(raw)
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

// Q6KWeight is a handle to a raw Q6_K weight matrix [Out, In] resident on the GPU (210-B
// super-blocks). In must be a multiple of 256; the resident byte cost is Out*(In/256)*210. It
// backs the fused MLP's down_proj when a q4_k_m GGUF quantizes down_proj to Q6_K. The id is offset
// by the C side's MG_Q6_BASE so it can never alias a Q4KWeight id.
type Q6KWeight struct {
	id      C.int
	Out, In int
}

// UploadQ6K makes a row-major Q6_K payload (verbatim GGUF super-block bytes, length
// out*(in/256)*210) resident for the GPU and returns a handle, or nil if the backend is
// unavailable, in is not a multiple of 256, or the payload is short / the table is full.
func UploadQ6K(raw []byte, out, in int) *Q6KWeight {
	if !Available() || in <= 0 || in%256 != 0 || out <= 0 {
		return nil
	}
	need := out * (in / 256) * 210
	if len(raw) < need {
		return nil
	}
	raw = raw[:need]
	id := C.mg_q6k_upload((*C.uchar)(unsafe.Pointer(&raw[0])), C.int(out), C.int(in))
	if id < 0 {
		return nil
	}
	runtime.KeepAlive(raw)
	return &Q6KWeight{id: id, Out: out, In: in}
}

// ID returns the backend handle for this matrix.
func (w *Q6KWeight) ID() int { return int(w.id) }

// FusedMLPQ6Down runs a whole dense SwiGLU MLP for one decode token — y = down( silu(gate·x) *
// (up·x) ) — in ONE Metal command buffer, exactly like FusedMLP, but with a Q6_K down_proj
// (gate/up stay Q4_K). The intermediate-wide gate/up/inter stays resident (only x and y cross the
// boundary). Requires gate.In==up.In==down.Out (=H), gate.Out==up.Out==down.In (=I); len(x)>=H,
// len(y)>=down.Out. Returns false on a shape mismatch. The activation is silu.
func FusedMLPQ6Down(gate, up *Q4KWeight, down *Q6KWeight, x, y []float32) bool {
	if gate == nil || up == nil || down == nil || gate.id < 0 || up.id < 0 || down.id < 0 {
		return false
	}
	if gate.In != up.In || gate.Out != up.Out || down.In != gate.Out || down.Out != gate.In {
		return false
	}
	if len(x) < gate.In || len(y) < down.Out {
		return false
	}
	C.mg_q4k_mlp_q6down(gate.id, up.id, down.id,
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&y[0])))
	return true
}

// FusedMLPQ6DownBatch runs n experts' fused SwiGLU MLP (Q4_K gate/up, Q6_K down) — each y_e =
// down_e( silu(gate_e·x) * (up_e·x) ) over the SAME token activation x — into ONE Metal command
// buffer, so the top-k experts of a MoE layer pay the submit/sync once instead of n times (issue
// #1382, the mlp_decode decode lever). Ycat receives the n outputs concatenated (row e at e*down.Out);
// the caller applies the gate-weighted sum on the host so the reduction order matches the per-expert
// loop exactly. All experts must share one geometry (gate.In==up.In==down.Out=H, gate.Out==up.Out==
// down.In=I). Returns false if n<=0, len(x)<H, len(Ycat)<n*down.Out, any handle is invalid, or the
// backend declines a shape — the caller then runs the proven per-expert FusedMLPQ6Down loop.
func FusedMLPQ6DownBatch(gate, up []*Q4KWeight, down []*Q6KWeight, x, Ycat []float32) bool {
	n := len(gate)
	if n == 0 || len(up) != n || len(down) != n {
		return false
	}
	H, I, Dout := gate[0].In, gate[0].Out, down[0].Out
	gw := make([]C.int, n)
	uw := make([]C.int, n)
	dw := make([]C.int, n)
	for e := 0; e < n; e++ {
		g, u, d := gate[e], up[e], down[e]
		if g == nil || u == nil || d == nil || g.id < 0 || u.id < 0 || d.id < 0 {
			return false
		}
		if g.In != u.In || g.Out != u.Out || d.In != g.Out || d.Out != g.In {
			return false
		}
		if g.In != H || g.Out != I || d.Out != Dout {
			return false // non-uniform batch geometry — decline (caller uses the per-expert loop)
		}
		gw[e], uw[e], dw[e] = C.int(g.id), C.int(u.id), C.int(d.id)
	}
	if len(x) < H || len(Ycat) < n*Dout {
		return false
	}
	rc := C.mg_q4k_mlp_q6down_batch(&gw[0], &uw[0], &dw[0], C.int(n),
		(*C.float)(unsafe.Pointer(&x[0])), (*C.float)(unsafe.Pointer(&Ycat[0])))
	return rc == 0
}

// GEMM computes Y[P, Out] = X[P, In] · Wᵀ (batched prefill GEMM). X and Y are f32 row-major;
// Y must have length >= P*Out. Both slices are accessed only during the call.
func (w *Q4KWeight) GEMM(X []float32, P int, Y []float32) {
	if w == nil || w.id < 0 || P <= 0 || len(X) < P*w.In || len(Y) < P*w.Out {
		return
	}
	C.mg_q4k_gemm(w.id, (*C.float)(unsafe.Pointer(&X[0])), C.int(P), (*C.float)(unsafe.Pointer(&Y[0])))
}

// GEMMGroup runs one batched prefill GEMM per weight in ws — all reading the SAME activation panel
// X[P, In] (shared) — in a SINGLE Metal command buffer, returning one [P*Out_i] result slice per
// weight (token-major, Y[t*Out_i + o]). Every weight must share X's In. It is the prefill twin of
// GEMVGroup: the live prefill group pattern (a layer's q/k/v, gate/up, or the GDN in_proj quad all
// read the same post-norm panel), paying the per-command-buffer submit/sync once for the whole
// group instead of once per weight — the fix for the ~7-submits-per-layer prefill wall. Returns nil
// on a shape mismatch or empty input, so the caller falls back to per-weight GEMM.
func GEMMGroup(ws []*Q4KWeight, X []float32, P int) [][]float32 {
	n := len(ws)
	if n == 0 || P <= 0 || ws[0] == nil || len(X) < P*ws[0].In {
		return nil
	}
	in := ws[0].In
	wids := make([]C.int, n)
	yoff := make([]C.int, n+1) // yoff[i] = P*Σ_{j<i} out_j (element offset of weight i's [P,out_i] block)
	off := 0
	for i, w := range ws {
		if w == nil || w.id < 0 || w.In != in {
			return nil
		}
		wids[i] = w.id
		yoff[i] = C.int(off)
		off += P * w.Out
	}
	yoff[n] = C.int(off)
	ycat := make([]float32, off)
	C.mg_q4k_gemm_group(&wids[0], C.int(n), (*C.float)(unsafe.Pointer(&X[0])), C.int(P),
		(*C.float)(unsafe.Pointer(&ycat[0])), &yoff[0])
	out := make([][]float32, n)
	o := 0
	for i, w := range ws {
		sz := P * w.Out
		out[i] = ycat[o : o+sz : o+sz]
		o += sz
	}
	return out
}

// ID returns the backend handle for this matrix.
func (w *Q4KWeight) ID() int { return int(w.id) }

// NoCopy reports whether this handle aliases the caller's pinned raw q4_k bytes through
// newBufferWithBytesNoCopy instead of owning a copied Metal buffer.
func (w *Q4KWeight) NoCopy() bool { return w != nil && w.noCopy }

// SetGEMMUseMM selects the batched-GEMM kernel: true prefers the simdgroup-matrix (hardware MMA)
// q4k_gemm_mm when its pipeline compiled, false (default) uses the proven scalar register-tile
// q4k_gemm. Gated so the scalar kernel stays default until the MMA variant is A/B-proven faster on
// the target device; the model layer flips it from FAK_Q4K_MM. No-op if no Metal device.
func SetGEMMUseMM(on bool) {
	v := C.int(0)
	if on {
		v = 1
	}
	C.mg_q4k_set_use_mm(v)
}

// ResetQ4K releases every resident q4_k weight buffer and the reused scratch (the q4_k twin of
// Reset). Call only when no Q4KWeight handle is still in use — every prior handle is invalidated.
func ResetQ4K() {
	q4kPinMu.Lock()
	defer q4kPinMu.Unlock()
	C.mg_q4k_reset()
	for id, pinned := range q4kPins {
		pinned.pin.Unpin()
		delete(q4kPins, id)
	}
}
