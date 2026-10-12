//go:build darwin && arm64 && cgo

package model

import "github.com/anthony-chaudhary/fak/internal/metalgemm"

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
	// Opt into the simdgroup-matrix (hardware MMA) batched GEMM when FAK_Q4K_MM=1. Measured ~1.3x
	// the scalar register-tile GEMM at the Qwen3.6-27B gate/up prefill shape (1465 vs 1113 GFLOP/s
	// on the M3 Pro), cosine 1.0 vs the CPU f32 reference. Default OFF (the scalar kernel stays the
	// proven path) until the MMA variant earns auto-enable. Set once per process — cheap and
	// idempotent on the metalgemm side.
	// Since fak#13692 the mul_mm default outranks both the MM32 and the M5 candidates below; they
	// only select a kernel when SetKQuantMulMMOptIn(false) turns the default off (or for P below
	// metalgemm.Q4KMulMMMinPrompt).
	kquantMulMMOnce.Do(func() { metalgemm.SetGEMMUseMulMM(!kquantMulMMDisabled) })
	q4kMMOnce.Do(func() { metalgemm.SetGEMMUseMM(q4kMMOptIn()) })
	// Default ON the wide-tile cooperative-SMEM candidate for the widened panel regime (P>=64,
	// fak#13041). The sanctioned on-silicon M3 Pro receipt now pins a row in metalgemm's
	// device/version-pinned crossover table (fak#13124 / 1ebb4a9c6), where the median candidate/
	// scalar on-GPU ratio measured 1.57x at P=64 and 1.46x at P=128, every sample clearing the
	// fak#9937 >=1.10x gate. The encode-time crossover gate is the real safety mechanism: on any
	// device/OS with no pinned row q4kGEMMModeForPrompt still requests the scalar identity, so
	// default-on is inert off the receipted box. SetQ4KM5OptIn(false) (formerly FAK_Q4K_M5=0) forces
	// the scalar kernel when the mul_mm default is also off. Set once per process (cheap, idempotent on the metalgemm side).
	q4kM5Once.Do(func() { metalgemm.SetGEMMUseM5(q4kM5OptIn()) })
	uploaded := map[string]bool{}
	cfg := m.Cfg
	for l := 0; l < cfg.NumLayers; l++ {
		lp := func(str string) string { return layerName(l, str) }
		names := denseProjectionNames(lp)
		if cfg.isLinearAttnLayer(l) {
			// fak#13567: qwen35 GDN in_proj rows may be held native Q4_K after a raw row
			// reorder; bulk-upload them too so their first prefill use pays no lazy H2D.
			names = append(names, lp("linear_attn.in_proj_qkv.weight"), lp("linear_attn.in_proj_z.weight"),
				lp("linear_attn.in_proj_b.weight"), lp("linear_attn.in_proj_a.weight"))
		}
		for _, name := range names {
			qt := m.q4kw[name]
			if qt == nil {
				continue // Q8 minority — not a q4_k-resident projection
			}
			if qt.lazy != nil {
				uploaded[name] = true // lazy weights stage at the operation seam
				continue
			}
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
	if names, err := m.metalQ8BandNames(); err == nil {
		if m.promoteMetalQ8Residency() != nil {
			return nil
		}
		uploaded := make(map[string]bool, len(names))
		for _, name := range names {
			uploaded[name] = true
		}
		return uploaded
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
		for _, name := range denseProjectionNames(lp) {
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

func denseProjectionNames(lp func(string) string) []string {
	return []string{
		lp("self_attn.q_proj.weight"), lp("self_attn.k_proj.weight"),
		lp("self_attn.v_proj.weight"), lp("self_attn.o_proj.weight"),
		lp("mlp.gate_proj.weight"), lp("mlp.up_proj.weight"), lp("mlp.down_proj.weight"),
	}
}
