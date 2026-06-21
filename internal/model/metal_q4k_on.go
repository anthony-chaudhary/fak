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
