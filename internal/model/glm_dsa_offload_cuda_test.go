//go:build cuda

package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAGLMDsaCPUOffloadHybrid is the on-DEVICE witness for the --n-cpu-moe CPU-offload hybrid
// (Session.CPUOffloadExperts, moe_offload.go): with the cuda backend attached AND offload on, the
// ROUTED MoE expert GEMMs must run HOST-resident (the CPU Q8 kernel) while the dense projections +
// router + DSA attention — and the always-on shared expert (#1304) — run on the GPU pure kernels
// (k_q8_gemm + k_dsa_sparse_attend). This is the real 753B serve split — routed experts in host
// RAM, the every-token dense FLOPs (plus the shared expert) on the device — proven on real sm_80,
// not just the cpu-ref stand-in the moe_offload_test.go suite uses on the agent-host.
//
// A recordingBackend over the cuda backend makes "which GEMMs reached the GPU" directly observable:
//   - the all-device baseline (offload off) records the expert shapes [I,H]/[H,I] — routed + shared
//     experts on GPU;
//   - the hybrid (offload on) records STRICTLY FEWER expert-shaped GEMMs — the routed experts moved
//     to host — but still MORE THAN ZERO, because the shared expert stays device-resident (#1304);
//     it also records the router [NumExperts,H], an MLA projection [QLoraRank,H], and DSA
//     sparse-attention calls, so the dense path stayed on the device.
//
// The shared and routed experts share the [I,H]/[H,I] shapes, so the routed-vs-shared split is
// witnessed by the differential COUNT rather than by shape alone (the Q8/Q4_K staged residency
// reaches the backend through a direct Upload with no upload site/name). A GPU MatMul at [I,H] or
// [H,I] is therefore no longer unambiguous routed work — the shared expert contributes it by
// design. The hybrid forward must still match the all-host CPU Q8 reference argmax-exact within the
// recorded Approx cosine floor (the GPU reduction order differs from the host). A skip (no
// reachable GPU) is NOT a pass; run on an sm_80+ node via tools/dgx_glm_gpu_witness.sh.
func TestCUDAGLMDsaCPUOffloadHybrid(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok {
		t.Skip("cuda backend not registered (no reachable CUDA device)")
	}
	if _, ok := be.(compute.DSASparseBackend); !ok {
		t.Fatalf("cuda backend does not implement DSASparseBackend — DSA sparse attention would run host-resident")
	}
	// MoE + shared experts so the offload predicate (hostOffloadWeight) actually fires on real tensors.
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true /*tied*/, false /*omitSharedIndexer*/, true /*withMoE*/, true /*withSharedExperts*/)
	lean, err := LoadSafetensorsQuant(path, cfg) // q8-resident GLM-DSA: dense -> k_q8_gemm, experts -> host qMatRows
	if err != nil {
		t.Fatalf("LoadSafetensorsQuant: %v", err)
	}
	if !lean.Cfg.isGLMMoeDsa() || !lean.Cfg.IsMoE() {
		t.Fatalf("fixture family=%q IsMoE=%v, want glm_moe_dsa MoE", lean.Cfg.archFamilyKey(), lean.Cfg.IsMoE())
	}
	prompt := []int{3, 17, 5, 23}
	I, H := cfg.IntermediateSize, cfg.HiddenSize

	// All-host CPU Q8 reference.
	sCPU := lean.NewSession()
	sCPU.Quant = true
	lCPU := sCPU.Prefill(prompt)

	// All-device baseline (offload OFF): routed + shared experts reach the GPU.
	recAll := newRecordingBackend(be)
	sAll := lean.NewBackendSession(recAll)
	sAll.Quant = true
	_ = sAll.Prefill(prompt)
	sAll.Close()
	baseGateUp, baseDown := recAll.count(I, H), recAll.count(H, I)
	if baseGateUp == 0 || baseDown == 0 {
		t.Fatalf("all-device baseline did not run expert GEMMs on the GPU (gate/up [%d,%d] count=%d, down [%d,%d] count=%d) - probe is broken",
			I, H, baseGateUp, H, I, baseDown)
	}

	// Hybrid (offload ON): ROUTED experts host-resident; the SHARED expert stays device-resident
	// (#1304), as do dense + router + DSA attention. The Q8/Q4_K staged residency reaches the
	// backend through a direct Upload (no upload site/name), so the routed-vs-shared split is
	// witnessed by the differential COUNT of expert-shaped GEMMs: the hybrid runs strictly FEWER
	// than the baseline (the routed experts left the device) yet still MORE than zero (the shared
	// expert stayed). This replaces the pre-#1304 assertion that NO expert-shaped GEMM reached the
	// GPU, which is now false by design because the shared expert shares those shapes.
	recOff := newRecordingBackend(be)
	sHy := lean.NewBackendSession(recOff)
	sHy.Quant = true
	sHy.CPUOffloadExperts = true
	lHy := sHy.Prefill(prompt)
	sHy.Close()

	hyGateUp, hyDown := recOff.count(I, H), recOff.count(H, I)
	if hyGateUp >= baseGateUp || hyDown >= baseDown {
		t.Errorf("CPU-offload hybrid ran as many expert GEMMs on the GPU as the all-device baseline (gate/up %d vs %d, down %d vs %d) - routed experts did NOT offload to host",
			hyGateUp, baseGateUp, hyDown, baseDown)
	}
	if hyGateUp == 0 || hyDown == 0 {
		t.Errorf("CPU-offload hybrid ran NO expert-shaped GEMM on the GPU (gate/up=%d down=%d) - the always-on shared expert must stay DEVICE-resident (#1304)", hyGateUp, hyDown)
	}
	if !recOff.saw(cfg.QLoraRank, H) {
		t.Errorf("CPU-offload hybrid did not run q_a_proj [%d,%d] on the GPU - attention wrongly offloaded", cfg.QLoraRank, H)
	}
	if !recOff.saw(cfg.NumExperts, H) {
		t.Errorf("CPU-offload hybrid did not run the router [%d,%d] on the GPU - dense path wrongly offloaded", cfg.NumExperts, H)
	}
	if calls, sel := recOff.sparse(); calls == 0 || sel == 0 {
		t.Errorf("CPU-offload hybrid: DSA sparse attention did not run on the GPU (calls=%d sel=%d) - should stay on the device", calls, sel)
	}

	// Correctness: the hybrid forward matches the all-host CPU Q8 reference argmax-exact within the
	// recorded Approx cosine floor (same floor the all-device GLM-DSA cuda witness uses).
	if len(lCPU) != cfg.VocabSize || len(lHy) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d hybrid=%d want vocab=%d", len(lCPU), len(lHy), cfg.VocabSize)
	}
	c := glmDsaCosine(lCPU, lHy)
	const floor = 0.99 // tiny synthetic + Q8-on-device dense vs Q8-on-host reduction-order Approx
	if c < floor {
		t.Fatalf("GLM-DSA CPU-offload hybrid forward cosine %.6f < %.4f vs CPU Q8", c, floor)
	}
	aCPU, aHy := glmDsaArgmax(lCPU), glmDsaArgmax(lHy)
	t.Logf("GLM-DSA --n-cpu-moe hybrid on cuda: experts host-offloaded (OFF the GPU); router+MLA projections+DSA sparse attention on k_q8_gemm/k_dsa_sparse_attend; cosine=%.6f argmax cpu=%d hybrid=%d tier=%s class=%s",
		c, aCPU, aHy, be.Tier(), be.Class())
	if aCPU != aHy {
		t.Fatalf("GLM-DSA CPU-offload hybrid argmax %d != CPU Q8 argmax %d (cosine=%.6f)", aHy, aCPU, c)
	}
}
