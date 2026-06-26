//go:build darwin && cgo && fakmetal

package model

// metal_q4k_on.go — the Metal q4_k prefill GEMM dispatch (built only under -tags fakmetal).
// When s.MetalQ4K is set and a device is present, the resident-Q4_K hybrid prefill's
// q4_k-majority projection/MLP GEMMs run on the GPU via internal/metalgemm's q4_k dequant-GEMM
// (q4k.m) instead of the CPU q4kGemm. Each weight is uploaded once and cached per *Model.
//
// Memory caveat: this keeps the CPU q4kw copy resident AND uploads a GPU copy, so the q4_k
// weight set is held twice (~2× the model's q4 bytes). That fits a small model fine but on the
// 27B/36 GB box it pressures unified memory — the loader change that hands the bytes straight
// to the GPU and frees the CPU copy is the tracked follow-up. Hence MetalQ4K is opt-in.

import (
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalQ4KMu sync.Mutex
	// metalQ4KW caches one GPU q4_k weight handle per (model, weight-name). A nil entry is
	// cached too (upload failed / table full) so we don't retry the upload every token.
	metalQ4KW = map[*Model]map[string]*metalgemm.Q4KWeight{}
)

func (s *Session) q4kGemmDispatch(name string, qt *q4kTensor, Xf []float32, P int) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return q4kGemm(qt, Xf, P)
	}
	w := s.M.metalQ4KWeight(name, qt)
	if w == nil {
		return q4kGemm(qt, Xf, P) // upload declined — stay on the proven CPU path
	}
	Y := make([]float32, P*qt.out)
	w.GEMM(Xf, P, Y)
	return Y
}

// q4kMatRowsDispatch is the decode-GEMV twin of q4kGemmDispatch: under MetalQ4K it runs the q4_k
// GEMV on the GPU (q4k_gemv) instead of the CPU q4kMatRows. Routing BOTH decode and prefill q4_k
// matmuls to the GPU is what lets metalQ4KWeight free the CPU copy (single residency) — the fix
// for the double-residency memory pressure that made the 27B Metal path a regression.
func (s *Session) q4kMatRowsDispatch(name string, qt *q4kTensor, xf []float32) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return q4kMatRows(qt, xf)
	}
	w := s.M.metalQ4KWeight(name, qt)
	if w == nil {
		return q4kMatRows(qt, xf)
	}
	y := make([]float32, qt.out)
	w.GEMV(xf, y)
	return y
}

// q4kGroupDispatch runs a group of matmuls that share one f32 activation xf (a layer's gate/up,
// q/k/v, or the GDN in_proj quad) in ONE Metal command buffer: the q4_k-resident members go through
// metalgemm.GEMVGroup (one commit/waitUntilCompleted, dispatches pipelined), and any Q8/Q6_K
// minority member falls back to the per-call CPU GEMV. Returns nil — so the caller (mulGroup) loops
// the per-call path — unless MetalQ4K is on, a device is present, AND at least two members are
// q4_k-resident (so a command buffer is worth amortizing). Results are bit-identical to calling
// q4kMatRowsDispatch per name.
func (s *Session) q4kGroupDispatch(names []string, xf []float32, outs []int) [][]float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return nil
	}
	n := len(names)
	ws := make([]*metalgemm.Q4KWeight, 0, n)
	pos := make([]int, 0, n) // index in names of each grouped (q4_k-resident, uploaded) member
	for i, name := range names {
		qt := s.M.q4kw[name]
		if qt == nil {
			continue
		}
		w := s.M.metalQ4KWeight(name, qt) // uploads once + frees the CPU copy on success
		if w == nil {
			continue
		}
		ws = append(ws, w)
		pos = append(pos, i)
	}
	if len(ws) < 2 {
		return nil // not enough resident members to amortize a command buffer
	}
	grouped := metalgemm.GEMVGroup(ws, xf)
	if grouped == nil {
		return nil
	}
	out := make([][]float32, n)
	for j, i := range pos {
		out[i] = grouped[j]
	}
	// Fill the non-grouped members (Q8/Q6_K minority) per-call, exactly as sessionQ4KKernel.mul.
	for i, name := range names {
		if out[i] != nil {
			continue
		}
		if qt := s.M.q4kw[name]; qt != nil {
			out[i] = s.q4kMatRowsDispatch(name, qt, xf) // GPU upload declined → its own dispatch
		} else {
			out[i] = qMatRows(s.M.q8(name), quantizeVecQ8(xf))
		}
	}
	return out
}

// q4kFusedMLP runs the dense SwiGLU MLP (gate/up/silu/down) for one decode token entirely on the
// GPU in ONE command buffer (the intermediate-wide buffer stays resident) when MetalQ4K is on and
// all three weights are q4_k-resident + uploaded. Returns nil otherwise so the caller uses the
// per-matmul path. The Metal kernel is silu-only and adds no bias, so the caller must gate on a
// non-GELU activation and bias-free MLP. Bit-identical to the per-matmul path up to GPU float-order.
func (s *Session) q4kFusedMLP(gateName, upName, downName string, x []float32) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return nil
	}
	gt, ut, dt := s.M.q4kw[gateName], s.M.q4kw[upName], s.M.q4kw[downName]
	if gt == nil || ut == nil || dt == nil {
		return nil
	}
	gw := s.M.metalQ4KWeight(gateName, gt)
	uw := s.M.metalQ4KWeight(upName, ut)
	dw := s.M.metalQ4KWeight(downName, dt)
	if gw == nil || uw == nil || dw == nil {
		return nil
	}
	y := make([]float32, dt.out)
	if !metalgemm.FusedMLP(gw, uw, dw, x, y) {
		return nil
	}
	return y
}

// metalQ4KWeight returns this model's GPU q4_k handle for `name`, uploading the raw blocks once.
// On a successful upload it FREES the CPU raw copy (qt.raw = nil): with both decode and prefill
// q4_k matmuls on the GPU, the weight is resident once (~16 GB for 27B, fits 36 GB) instead of
// twice (~30 GB → swap). A failed upload keeps the CPU copy so the q4kMatRows/q4kGemm fallback
// stays valid. Peak overshoot is one weight (its CPU bytes are freed right after its H2D copy).
func (m *Model) metalQ4KWeight(name string, qt *q4kTensor) *metalgemm.Q4KWeight {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ4KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q4KWeight{}
		metalQ4KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	w := metalgemm.UploadQ4K(qt.raw, qt.out, qt.in)
	tbl[name] = w // cache nil too, so a failed upload doesn't retry every token
	if w != nil {
		qt.raw = nil // GPU holds it now; drop the CPU copy → single residency
	}
	return w
}
