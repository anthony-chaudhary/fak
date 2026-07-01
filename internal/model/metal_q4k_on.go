//go:build darwin && arm64 && cgo

package model

// metal_q4k_on.go — the Metal q4_k prefill GEMM dispatch (built on Apple Silicon with cgo).
// When s.MetalQ4K is set and a device is present, the resident-Q4_K hybrid prefill's
// q4_k-majority projection/MLP GEMMs run on the GPU via internal/metalgemm's q4_k dequant-GEMM
// (q4k.m) instead of the CPU q4kGemm. Each weight is uploaded once and cached per *Model.
//
// On Apple unified memory the q4_k upload path wraps the model's resident raw bytes with a
// no-copy MTLBuffer when Metal accepts the pointer. If it falls back to a copied Metal buffer,
// MetalQ4K can still be paired with FAK_Q4K_FREE_CPU=1 once all q4_k matmuls are GPU-routed.

import (
	"os"
	"sync"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

var (
	metalQ4KMu sync.Mutex
	// metalQ4KW caches one GPU q4_k weight handle per (model, weight-name). A nil entry is
	// cached too (upload failed / table full) so we don't retry the upload every token.
	metalQ4KW = map[*Model]map[string]*metalgemm.Q4KWeight{}
	// metalQ6KW caches one GPU Q6_K weight handle per (model, weight-name) for the fused MLP's
	// Q6_K down_proj. Same nil-caching policy as metalQ4KW. Guarded by metalQ4KMu.
	metalQ6KW = map[*Model]map[string]*metalgemm.Q6KWeight{}
	// metalQ8KW caches one GPU Q8_0 weight handle per (model, weight-name) for Q8-minority
	// prefill projections in the resident-Q4_K lane (full-attn q/k and Qwen3.6 linear_attn.*).
	// Same nil-caching policy as metalQ4KW. Guarded by metalQ4KMu.
	metalQ8KW = map[*Model]map[string]*metalgemm.Q8Weight{}
	// metalQ8Budget caches, per *Model, whether the Q8-minority GPU upload fits the device's
	// working-set budget (computed once by metalQ8UploadAllowed). Guarded by metalQ4KMu.
	metalQ8Budget = map[*Model]bool{}
	// freeCPUCopyAfterUpload, when set, drops qt.raw after a successful GPU upload for single
	// residency. Default OFF: the CPU prefill/decode fallbacks (q4kGemm/q4kMatRows) still read
	// qt.raw and panic on nil when the GPU path isn't taken for some tensor (#1067). Opt in with
	// FAK_Q4K_FREE_CPU=1 only once every q4_k matmul is provably GPU-routed.
	freeCPUCopyAfterUpload = os.Getenv("FAK_Q4K_FREE_CPU") == "1"
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

// q8GemmDispatch is the prefill-GEMM twin for the Q8-minority projections in the resident-Q4_K
// path. Pure CPU builds and non-Metal sessions use qGemm8. With MetalQ4K enabled, the Q8 panel
// runs through metalgemm's batched Q8 GEMM, so Qwen3.6's full-attn q/k and linear_attn.* no longer
// cap prefill on the CPU side (#1087). The caller supplies the already-quantized activation panel.
func (s *Session) q8GemmDispatch(name string, qt *q8Tensor, Xq *q8Panel) []float32 {
	if !s.MetalQ4K || !metalgemm.Available() {
		return qGemm8(qt, Xq)
	}
	w := s.M.metalQ8Weight(name, qt)
	if w == nil {
		return qGemm8(qt, Xq)
	}
	Y := make([]float32, Xq.P*qt.out)
	w.GEMM(Xq.q, Xq.d, Xq.P, Y)
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
	gt, ut := s.M.q4kw[gateName], s.M.q4kw[upName]
	if gt == nil || ut == nil {
		return nil
	}
	gw := s.M.metalQ4KWeight(gateName, gt)
	uw := s.M.metalQ4KWeight(upName, ut)
	if gw == nil || uw == nil {
		return nil
	}
	// Down is Q4_K-resident → the all-Q4_K fused path (unchanged).
	if dt := s.M.q4kw[downName]; dt != nil {
		dw := s.M.metalQ4KWeight(downName, dt)
		if dw == nil {
			return nil
		}
		y := make([]float32, dt.out)
		if !metalgemm.FusedMLP(gw, uw, dw, x, y) {
			return nil
		}
		return y
	}
	// Down is Q6_K-resident (the q4_k_m down_proj case) → the mixed-quant fused path. This is the
	// Stage B cap-lift: previously a Q6_K down made q4kFusedMLP decline and every such expert fell
	// to the per-matmul path; now gate/up stay Q4_K and only stage 3 runs the Q6_K GEMV.
	if dq := s.M.kqw[downName]; dq != nil && dq.kind == kindQ6K {
		dw := s.M.metalQ6KWeight(downName, dq)
		if dw == nil {
			return nil
		}
		y := make([]float32, dq.out)
		if !metalgemm.FusedMLPQ6Down(gw, uw, dw, x, y) {
			return nil
		}
		return y
	}
	return nil
}

// metalQ6KWeight returns this model's GPU Q6_K handle for `name`, uploading the raw 210-B blocks
// once (cached per *Model, nil cached too). The Q6_K resident store backs the fused MLP's down_proj
// when a q4_k_m GGUF quantizes down to Q6_K; gate/up stay Q4_K via metalQ4KWeight.
func (m *Model) metalQ6KWeight(name string, qt *kQuantTensor) *metalgemm.Q6KWeight {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ6KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q6KWeight{}
		metalQ6KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	w := metalgemm.UploadQ6K(qt.raw, qt.out, qt.in)
	tbl[name] = w
	return w
}

// metalQ8UploadAllowed reports (and caches per *Model) whether this model's Q8-minority
// projections may be uploaded to the GPU without breaching the device working-set budget. When it
// returns false, both the bulk pre-upload (metalQ8Weights) and the lazy per-call upload
// (metalQ8Weight) decline, so the Q8 minority stays on the proven CPU qGemm8 path — exactly the
// pre-#1087 behavior, which serves the 27B on a 36 GiB Mac without the OOM. Callers already hold
// no lock; this takes metalQ4KMu to guard the cache.
func (m *Model) metalQ8UploadAllowed() bool {
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	if v, ok := metalQ8Budget[m]; ok {
		return v
	}
	r := m.ResidentReport()
	deviceTotal := int64(0)
	if total, ok := metalgemm.DeviceMemoryTotal(); ok {
		deviceTotal = int64(total)
	}
	allowed := q8UploadFits(r.TotalResidentBytes, r.Q8Bytes, deviceTotal, os.Getenv("FAK_METAL_Q8_UPLOAD"))
	metalQ8Budget[m] = allowed
	return allowed
}

// metalQ8Weight returns this model's GPU Q8 handle for `name`, uploading the Q8_0 codes/scales
// once (cached per *Model, nil cached too). It backs batched Metal prefill for the Q8-minority
// projections in the resident-Q4_K lane. Declines (returns nil, caller stays on CPU qGemm8) when
// the device working-set budget can't absorb the additive Q8 GPU copy (metalQ8UploadAllowed).
func (m *Model) metalQ8Weight(name string, qt *q8Tensor) *metalgemm.Q8Weight {
	// Budget gate BEFORE the lock (metalQ8UploadAllowed takes metalQ4KMu itself; sync.Mutex is
	// not reentrant). On a tight device the additive Q8 GPU copy would OOM the serve, so decline
	// here and let q8GemmDispatch fall back to the CPU qGemm8 — the pre-#1087, non-OOM path.
	if !m.metalQ8UploadAllowed() {
		return nil
	}
	metalQ4KMu.Lock()
	defer metalQ4KMu.Unlock()
	tbl := metalQ8KW[m]
	if tbl == nil {
		tbl = map[string]*metalgemm.Q8Weight{}
		metalQ8KW[m] = tbl
	}
	if w, ok := tbl[name]; ok {
		return w
	}
	w := metalgemm.UploadQ8(qt.q, qt.d, qt.out, qt.in)
	tbl[name] = w
	return w
}

// metalQ4KWeights uploads all Q4_K projection weights for this model to the GPU once,
// caching them per *Model. This is the prefill-weight-upload twin of metalWeights(): it
// uploads every q4_k-resident projection (q/k/v/o, gate/up/down) upfront so the prefill
// loop never incurs a per-call GPU round-trip. The lazy upload path in metalQ4KWeight
// caps warm prefill at ~7x under llama.cpp (#1113); calling this before the layer loop
// restores the full prefill speed by amortizing all H2D copies up front. Returns the map
// (read-only) so the caller can verify upload success; nil on non-Metal builds.
func (m *Model) metalQ4KWeights() map[string]bool {
	if !metalgemm.Available() {
		return nil
	}
	uploaded := map[string]bool{}
	cfg := m.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		for _, name := range []string{
			lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"),
			lp("self_attn.v_proj.weight"), lp("self_attn.o_proj.weight"),
			lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight"), lp("mlp.down_proj.weight"),
		} {
			qt := m.q4kw[name]
			if qt == nil {
				continue // Q8 minority — not a q4_k-resident projection
			}
			// metalQ4KWeight uploads if not already cached and records the result
			w := m.metalQ4KWeight(name, qt)
			uploaded[name] = w != nil
		}
	}
	return uploaded
}

// metalQ8Weights uploads the Q8-minority projection weights for this model to the GPU once. It
// deliberately skips names already present in q4kw or kqw: those route through Q4_K/Q6_K resident
// kernels, and uploading their Q8 copies would waste unified memory.
func (m *Model) metalQ8Weights() map[string]bool {
	if !metalgemm.Available() {
		return nil
	}
	// Skip the whole bulk pre-upload when the device budget can't absorb the additive Q8 GPU
	// copy — otherwise the 7 GiB projection store doubles and the serve is SIGKILLed at first
	// prefill (#1087 OOM). metalQ8Weight would decline each tensor anyway; returning early keeps
	// the intent legible and avoids the pointless per-tensor budget re-checks.
	if !m.metalQ8UploadAllowed() {
		return nil
	}
	uploaded := map[string]bool{}
	add := func(name string) {
		if m.q4kw[name] != nil || m.kqw[name] != nil {
			return
		}
		qt := m.q8w[name]
		if qt == nil {
			return
		}
		w := m.metalQ8Weight(name, qt)
		uploaded[name] = w != nil
	}
	cfg := m.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		for _, name := range []string{
			lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"),
			lp("self_attn.v_proj.weight"), lp("self_attn.o_proj.weight"),
			lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight"), lp("mlp.down_proj.weight"),
		} {
			add(name)
		}
		if cfg.isLinearAttnLayer(l) {
			for _, name := range []string{
				lp("linear_attn.in_proj_qkv.weight"), lp("linear_attn.in_proj_z.weight"),
				lp("linear_attn.in_proj_b.weight"), lp("linear_attn.in_proj_a.weight"),
				lp("linear_attn.out_proj.weight"),
			} {
				add(name)
			}
		}
	}
	return uploaded
}

// metalQ4KWeight returns this model's GPU q4_k handle for `name`, uploading the raw blocks once.
// The normal Apple-unified-memory path aliases qt.raw with a no-copy MTLBuffer, so the GPU and
// CPU fallback read the same resident bytes. If Metal falls back to a copied buffer, the
// FAK_Q4K_FREE_CPU opt-in may still drop qt.raw after upload for single residency; failed uploads
// always keep the CPU copy so q4kMatRows/q4kGemm remain valid.
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
	if w != nil && freeCPUCopyAfterUpload && !w.NoCopy() {
		// Drop the CPU copy → single residency (~16 GB for 27B vs ~30 GB doubled). UNSAFE
		// unless EVERY q4_k matmul for this weight — decode GEMV *and* batched prefill GEMM —
		// is guaranteed to run on the GPU: the CPU fallbacks q4kGemm/q4kMatRows read qt.raw and
		// panic on a nil slice (#1067, the multi-K-prompt prefill crash). Gated OFF by default;
		// FAK_Q4K_FREE_CPU=1 opts back into single residency once the prefill path is fully
		// GPU-routed and the CPU fallback is provably unreachable. A no-copy Metal buffer already
		// aliases qt.raw, so keeping the slice costs no duplicate storage and preserves fallback.
		qt.raw = nil
	}
	return w
}
