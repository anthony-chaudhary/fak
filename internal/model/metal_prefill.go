//go:build darwin && arm64 && cgo

package model

// metal_prefill.go — the Metal GPU prefill twin. Built by default on Apple Silicon with cgo. It is a
// structural copy of prefillBatched (the f32 lane): every elementwise op — RMSNorm, RoPE,
// the GQA attention over the f32 KV cache, SwiGLU, the residuals — is the identical f32 math,
// and ONLY the seven per-layer projection GEMMs (q/k/v/o, gate/up/down) are routed to the GPU
// via MPSMatrixMultiplication on f16 weights. Prefill is compute-bound, so moving just the
// matmuls to the GPU's FLOP-rich path is what closes the gap to llama.cpp-Metal; the cheap
// elementwise work stays on the CPU where the f32 reference already lives.
//
// Weights: the GPU holds an f16 copy of each projection, dequantized once from the Q8_0 store
// (so it is the SAME Q8_0 weight values llama.cpp's GGUF uses, just half-precision on-device).
// They are uploaded lazily on the first Metal prefill and cached per *Model.

import (
	"fmt"
	"os"
	"sync"
	"time"

	"github.com/anthony-chaudhary/fak/internal/metalgemm"
)

// metalProf enables coarse phase timing in prefillBatchedMetal (FAK_QPROFILE=1): how the
// prefill wall-clock splits across the GPU matmuls (incl. f16 round-trip), the CPU attention,
// and the CPU elementwise rest (norm/rope/silu/residual) — the map for where the gap to
// llama.cpp-Metal still lives.
var metalProf = os.Getenv("FAK_QPROFILE") != ""
var metalResidentDisabled = os.Getenv("FAK_METAL_RESIDENT") == "0"

// metalProjNames are the seven projection weights uploaded to the GPU, in upload order.
var metalProjNames = []string{
	"self_attn.q_proj.weight", "self_attn.k_proj.weight",
	"self_attn.v_proj.weight", "self_attn.o_proj.weight",
	"mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight",
}

var (
	metalMu sync.Mutex
	metalWt = map[*Model]map[string]*metalgemm.Weight{} // per-Model name -> GPU f16 weight
)

// dequantQ8 reconstructs the f32 weight matrix [out, in] from a Q8_0 tensor (code*scale per
// block) — the values the GPU then stores as f16. Parallel across output rows (one-time, off
// the hot path).
func dequantQ8(qt *q8Tensor) []float32 {
	out, in, nblk := qt.out, qt.in, qt.nblk
	f := make([]float32, out*in)
	parFor(out, numWorkers, func(lo, hi int) {
		for o := lo; o < hi; o++ {
			for b := 0; b < nblk; b++ {
				d := qt.d[o*nblk+b]
				base := o*in + b*qBlk
				for i := 0; i < qBlk; i++ {
					f[base+i] = float32(qt.q[base+i]) * d
				}
			}
		}
	})
	return f
}

// metalWeights returns this model's GPU weight table, uploading it once. The big f32 dequant
// buffer is freed after each upload (the GPU keeps its own f16 copy), so peak host overhead is
// one matrix, not the whole model.
func (m *Model) metalWeights() map[string]*metalgemm.Weight {
	metalMu.Lock()
	defer metalMu.Unlock()
	if w, ok := metalWt[m]; ok {
		return w
	}
	w := make(map[string]*metalgemm.Weight, len(metalProjNames)*m.Cfg.NumLayers)
	for l := 0; l < m.Cfg.NumLayers; l++ {
		for _, s := range metalProjNames {
			name := "model.layers." + itoa(l) + "." + s
			qt := m.q8(name)
			h := metalgemm.Upload(dequantQ8(qt), qt.out, qt.in)
			if h == nil {
				panic("model: metal weight upload failed for " + name)
			}
			w[name] = h
		}
	}
	metalWt[m] = w
	return w
}

// metalResidentReady tracks which models have had their GPU-resident-forward topology
// (norm/bias vectors + layer wiring) registered with the backend — a one-time step per *Model.
var metalResidentReady = map[*Model]bool{}

// metalResidentConfig registers this model with the GPU-resident forward (forward.m): it
// reuses the already-uploaded f16 projection handles from metalWeights(), uploads the per-layer
// RMSNorm vectors + the q/k/v biases (Qwen carries attention bias), and records the geometry.
// Runs once per *Model; cheap, off the hot path.
func (m *Model) metalResidentConfig() {
	gw := m.metalWeights() // locks/unlocks metalMu itself; must be called before we take it
	metalMu.Lock()
	defer metalMu.Unlock()
	if metalResidentReady[m] {
		return
	}
	cfg := m.Cfg
	metalgemm.FwdConfig(cfg.NumLayers, cfg.HiddenSize, cfg.HeadDim, cfg.NumHeads, cfg.NumKVHeads,
		cfg.IntermediateSize, float32(cfg.RMSNormEps), float32(cfg.RopeTheta), cfg.AttentionBias)
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return "model.layers." + itoa(l) + "." + str }
		inNorm := metalgemm.UploadVec(m.tensor(lp("input_layernorm.weight")))
		postNorm := metalgemm.UploadVec(m.tensor(lp("post_attention_layernorm.weight")))
		qb, kb, vb := -1, -1, -1
		if cfg.AttentionBias {
			qb = metalgemm.UploadVec(m.tensor(lp("self_attn.q_proj.bias")))
			kb = metalgemm.UploadVec(m.tensor(lp("self_attn.k_proj.bias")))
			vb = metalgemm.UploadVec(m.tensor(lp("self_attn.v_proj.bias")))
		}
		metalgemm.FwdLayer(l,
			gw[lp("self_attn.q_proj.weight")].ID(), gw[lp("self_attn.k_proj.weight")].ID(),
			gw[lp("self_attn.v_proj.weight")].ID(), gw[lp("self_attn.o_proj.weight")].ID(),
			gw[lp("mlp.gate_proj.weight")].ID(), gw[lp("mlp.up_proj.weight")].ID(),
			gw[lp("mlp.down_proj.weight")].ID(), inNorm, postNorm, qb, kb, vb)
	}
	metalgemm.FwdFinalNorm(metalgemm.UploadVec(m.tensor("model.norm.weight")))
	metalResidentReady[m] = true
}

// PrepareMetalResidency moves the one-time Metal weight residency work out of the first
// served request. Dense Q8 models register both the resident prefill topology and the Q8
// decode topology; Qwen3.5/3.6 hybrids prepare their hybrid prefill weight table; resident
// Q4_K models upload their q4_k-majority table. The hot Prefill/Step calls then reuse the
// existing per-model handles instead of paying a first-call upload.
func (m *Model) PrepareMetalResidency(q4k bool) bool {
	if !metalgemm.Available() {
		return false
	}
	if q4k {
		uploaded := m.metalQ4KWeights()
		if uploaded == nil {
			return false
		}
		ok := false
		for _, v := range uploaded {
			if !v {
				return false
			}
			ok = true
		}
		return ok
	}
	if m.Cfg.IsQwen35Hybrid() {
		m.metalWeightsQwen35Hybrid()
		return true
	}
	m.metalResidentConfig()
	return m.metalDecodeConfig()
}

// prefillMetalResident runs the WHOLE fresh prefill on the GPU (one command buffer, one sync):
// every layer's projections, RMSNorm, RoPE, causal GQA attention, SwiGLU and residuals stay
// f16 on-device. It fills the same f32 KV cache the CPU/hybrid paths build (pre-RoPE Kraw,
// post-RoPE K, V, pos) so decode/Evict/Clone stay valid, and returns the last token's
// post-final-norm hidden (caller applies the head). Returns nil if the backend declined, so
// prefillBatchedMetal can fall back to the hybrid path. Assumes base==0 (fresh prefill).
func (s *Session) prefillMetalResident(ids []int) []float32 {
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	w := cfg.NumKVHeads * hd
	eps := float32(cfg.RMSNormEps)
	P := len(ids)

	m.metalResidentConfig()

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
	}

	var t0 time.Time
	if metalProf {
		t0 = time.Now()
	}
	lastPre, kraw, kpost, v, ok := metalgemm.Prefill(X, P, cfg.NumLayers, w, H)
	if !ok {
		return nil
	}
	if metalProf {
		fmt.Fprintf(os.Stderr, "[metal-resident P=%d] device prefill+sync=%.1f ms\n",
			P, float64(time.Since(t0).Nanoseconds())/1e6)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		off := l * P * w
		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], kraw[off:off+P*w]...)
		s.Cache.K[l] = append(s.Cache.K[l], kpost[off:off+P*w]...)
		s.Cache.V[l] = append(s.Cache.V[l], v[off:off+P*w]...)
	}
	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, t) // base == 0
	}
	return rmsnorm(lastPre, m.tensor("model.norm.weight"), eps)
}

// prefillBatchedMetal is the GPU prefill. For a fresh prefill (base == 0) it runs the fully
// GPU-resident forward (prefillMetalResident — one command buffer, no per-matmul round-trips);
// otherwise (or if the resident path declines) it falls back to the hybrid path below, which
// is identical to prefillBatched except the seven projection GEMMs run on the Metal device.
// Both fill the same f32 KV cache the CPU paths build and return the last token's
// post-final-norm hidden (caller applies the head).
func (s *Session) prefillBatchedMetal(ids []int) []float32 {
	if s.Cache.Len() == 0 && !metalResidentDisabled {
		if out := s.prefillMetalResident(ids); out != nil {
			return out
		}
		// Resident path declined (e.g. MSL pipelines failed to build) — fall back to hybrid.
	}
	m, cfg := s.M, s.M.Cfg
	H, hd := cfg.HiddenSize, cfg.HeadDim
	nH, nKV := cfg.NumHeads, cfg.NumKVHeads
	grp := cfg.GroupSize()
	eps := float32(cfg.RMSNormEps)
	w := nKV * hd
	scale := cfg.attnScale()
	attnCap := float32(cfg.AttnSoftcap)
	P := len(ids)
	base := s.Cache.Len()

	gw := m.metalWeights()
	var tGemm, tAttn time.Duration
	t0 := time.Now()
	// mm runs Y[P,out] = X[P,in] * W[name]^T on the GPU into a fresh buffer.
	mm := func(name string, X []float32, out, in int) []float32 {
		Y := make([]float32, P*out)
		if metalProf {
			t := time.Now()
			gw[name].MatMul(X, P, Y)
			tGemm += time.Since(t)
			return Y
		}
		gw[name].MatMul(X, P, Y)
		return Y
	}

	embed := m.embedRows()
	X := make([]float32, P*H)
	for t, id := range ids {
		copy(X[t*H:(t+1)*H], embed[id*H:(id+1)*H])
	}

	cosP := make([][]float32, P)
	sinP := make([][]float32, P)
	for t := 0; t < P; t++ {
		cosP[t], sinP[t] = ropeRow(cfg, base+t)
	}

	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return "model.layers." + itoa(l) + "." + str }

		Xn := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wIn := m.tensor(lp("input_layernorm.weight"))
			for t := lo; t < hi; t++ {
				rmsnormInto(Xn[t*H:(t+1)*H], X[t*H:(t+1)*H], wIn, eps)
			}
		})

		Q := mm(lp("self_attn.q_proj.weight"), Xn, nH*hd, H)
		K := mm(lp("self_attn.k_proj.weight"), Xn, w, H)
		V := mm(lp("self_attn.v_proj.weight"), Xn, w, H)
		if cfg.AttentionBias {
			bq, bk, bv := m.tensor(lp("self_attn.q_proj.bias")), m.tensor(lp("self_attn.k_proj.bias")), m.tensor(lp("self_attn.v_proj.bias"))
			for t := 0; t < P; t++ {
				addBias(Q[t*nH*hd:(t+1)*nH*hd], bq)
				addBias(K[t*w:(t+1)*w], bk)
				addBias(V[t*w:(t+1)*w], bv)
			}
		}

		s.Cache.Kraw[l] = append(s.Cache.Kraw[l], K...)
		parFor(P, numWorkers, func(lo, hi int) {
			for t := lo; t < hi; t++ {
				for h := 0; h < nH; h++ {
					applyRopeRow(Q[t*nH*hd+h*hd:t*nH*hd+(h+1)*hd], cosP[t], sinP[t])
				}
				for h := 0; h < nKV; h++ {
					applyRopeRow(K[t*w+h*hd:t*w+(h+1)*hd], cosP[t], sinP[t])
				}
			}
		})

		s.Cache.K[l] = append(s.Cache.K[l], K...)
		s.Cache.V[l] = append(s.Cache.V[l], V...)
		Kl, Vl := s.Cache.K[l], s.Cache.V[l]

		attnOut := make([]float32, P*nH*hd)
		var tA time.Time
		if metalProf {
			tA = time.Now()
		}
		attnPrefillInto(attnOut, Q, Kl, Vl, P, base, nH, hd, w, grp, cfg.windowForLayer(l), l, scale, attnCap, fdot, nil)
		if metalProf {
			tAttn += time.Since(tA)
		}

		O := mm(lp("self_attn.o_proj.weight"), attnOut, H, nH*hd)
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += O[i]
			}
		})

		Xn2 := make([]float32, P*H)
		parFor(P, numWorkers, func(lo, hi int) {
			wPost := m.tensor(lp("post_attention_layernorm.weight"))
			for t := lo; t < hi; t++ {
				rmsnormInto(Xn2[t*H:(t+1)*H], X[t*H:(t+1)*H], wPost, eps)
			}
		})
		I := cfg.IntermediateSize
		G := mm(lp("mlp.gate_proj.weight"), Xn2, I, H)
		U := mm(lp("mlp.up_proj.weight"), Xn2, I, H)
		parFor(len(G), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				G[i] = silu(G[i]) * U[i]
			}
		})
		Down := mm(lp("mlp.down_proj.weight"), G, H, I)
		parFor(len(X), numWorkers, func(lo, hi int) {
			for i := lo; i < hi; i++ {
				X[i] += Down[i]
			}
		})
	}

	for t := 0; t < P; t++ {
		s.Cache.pos = append(s.Cache.pos, base+t)
	}
	if metalProf {
		total := time.Since(t0)
		rest := total - tGemm - tAttn
		ms := func(d time.Duration) float64 { return float64(d.Nanoseconds()) / 1e6 }
		fmt.Fprintf(os.Stderr, "[metalprof P=%d] total=%.1f  gemm+roundtrip=%.1f  attn=%.1f  rest(norm/rope/silu/resid)=%.1f ms\n",
			P, ms(total), ms(tGemm), ms(tAttn), ms(rest))
	}
	last := X[(P-1)*H : P*H]
	return rmsnorm(last, m.tensor("model.norm.weight"), eps)
}
