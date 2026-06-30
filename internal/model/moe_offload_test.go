package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// moe_offload_test.go — witnesses for the CPU-offload hybrid (the --n-cpu-moe split kernel).
// All run on CPU (no GPU, no CUDA): the device side is the cpu-ref backend, so the whole suite
// builds and passes on the Windows/AMD agent-host. The honesty gates are (1) the split is a PURE
// placement decision — with no device it is BIT-EXACT with the plain host forward — and (2) the
// expert GEMMs really do leave the device when offload is on, proven by a backend shape probe.

// TestIsExpertWeight pins the offload predicate to the canonical HF tensor names the GLM-DSA
// forward actually reads. The expert/shared-expert weights (the bulk a --n-cpu-moe split sends to
// host) must match; the router, the MLA/attention projections, the learned-index projections, and
// the LM head (the every-token dense GEMMs that stay on the device) must NOT. These literal names
// are produced by expertName / routerName (moe.go) and the GLM-DSA projection paths — a wrong
// predicate (e.g. matching the router, or missing shared experts) is caught here structurally.
func TestIsExpertWeight(t *testing.T) {
	onHost := []string{
		expertName(0, 0, "gate_proj.weight"),                 // model.layers.0.mlp.experts.0.gate_proj.weight
		expertName(3, 7, "up_proj.weight"),                   // a different layer/expert
		expertName(1, 2, "down_proj.weight"),                 //
		expertName(0, 0, "gate_proj.bias"),                   // an expert bias rides with its GEMM
		"model.layers.5.mlp.shared_experts.gate_proj.weight", // shared expert
		"model.layers.5.mlp.shared_experts.down_proj.weight",
	}
	onDevice := []string{
		routerName(0),     // model.layers.0.mlp.gate.weight — the router stays on device
		routerBiasName(0), // mlp.gate.bias
		"model.layers.0.self_attn.q_a_proj.weight", // MLA projections
		"model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
		"model.layers.0.self_attn.kv_b_proj.weight",
		"model.layers.0.self_attn.o_proj.weight",
		"model.layers.0.self_attn.indexer.wk.weight", // learned-index projection
		"model.layers.0.self_attn.indexer.weights_proj.weight",
		"lm_head.weight",
		"model.embed_tokens.weight",
		"model.layers.0.mlp.gate_proj.weight", // a DENSE (non-MoE) MLP weight — not an expert
	}
	for _, n := range onHost {
		if !isExpertWeight(n) {
			t.Errorf("isExpertWeight(%q) = false, want true (expert weight should offload to host)", n)
		}
	}
	for _, n := range onDevice {
		if isExpertWeight(n) {
			t.Errorf("isExpertWeight(%q) = true, want false (dense weight must stay on the device)", n)
		}
	}
}

// recordingKernel is a matKernel that records every weight name routed to it and delegates the
// arithmetic to an inner kernel unchanged. It makes WHICH names a splitKernel routes to host vs
// device directly observable, independent of shapes or the author's bookkeeping.
type recordingKernel struct {
	inner matKernel
	names map[string]int
}

func newRecordingKernel(inner matKernel) *recordingKernel {
	return &recordingKernel{inner: inner, names: map[string]int{}}
}
func (k *recordingKernel) prep(x []float32) any { return k.inner.prep(x) }
func (k *recordingKernel) mul(name string, x any, out, in int) []float32 {
	k.names[name]++
	return k.inner.mul(name, x, out, in)
}
func (k *recordingKernel) saw(name string) bool { return k.names[name] > 0 }

// TestSplitKernelRoutesByPredicate proves splitKernel.mul sends a weight to host iff onHost(name),
// and that it computes the SAME bytes the routed sub-kernel would — the routing is a pure dispatch.
func TestSplitKernelRoutesByPredicate(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	host := newRecordingKernel(residentKernel{m})
	device := newRecordingKernel(residentKernel{m})
	k := splitKernel{host: host, device: device, onHost: isExpertWeight}

	// A representative activation; the exact values do not matter — we assert routing + equality.
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32(i)*0.01 - 0.1
	}
	prepped := k.prep(x)

	cases := []struct {
		name       string
		out, in    int
		wantOnHost bool
	}{
		{expertName(0, 0, "gate_proj.weight"), m.Cfg.IntermediateSize, H, true},
		{expertName(0, 1, "up_proj.weight"), m.Cfg.IntermediateSize, H, true},
		{"model.layers.0.mlp.shared_experts.gate_proj.weight", m.Cfg.MoEIntermediateSize * m.Cfg.NSharedExperts, H, true},
		{routerName(0), m.Cfg.NumExperts, H, false},
		{"model.layers.0.self_attn.q_a_proj.weight", m.Cfg.QLoraRank, H, false},
	}
	for _, c := range cases {
		got := k.mul(c.name, prepped, c.out, c.in)
		want := residentKernel{m}.mul(c.name, prepped, c.out, c.in)
		if len(got) != len(want) {
			t.Fatalf("%s: split result len %d != reference %d", c.name, len(got), len(want))
		}
		for i := range got {
			if got[i] != want[i] {
				t.Fatalf("%s: split routed result differs from direct residentKernel at %d (%v != %v)", c.name, i, got[i], want[i])
			}
		}
		if c.wantOnHost && !host.saw(c.name) {
			t.Errorf("%s: expected on HOST kernel, but host never saw it", c.name)
		}
		if c.wantOnHost && device.saw(c.name) {
			t.Errorf("%s: expert weight reached the DEVICE kernel (should be host-offloaded)", c.name)
		}
		if !c.wantOnHost && !device.saw(c.name) {
			t.Errorf("%s: expected on DEVICE kernel, but device never saw it", c.name)
		}
		if !c.wantOnHost && host.saw(c.name) {
			t.Errorf("%s: dense weight reached the HOST kernel (should stay on device)", c.name)
		}
	}
}

// TestGLMDsaCPUOffloadPlacementInvariance is the load-bearing honesty witness: with no device
// backend, CPUOffloadExperts swaps in split(host=resident, device=resident) — every GEMM still runs
// on host — so the full GLM-DSA forward must be BIT-EXACT (max|Δ|==0, argmax-exact) with the plain
// resident forward. This proves the split changes only WHERE a GEMM runs, never WHAT it computes:
// the offload routing threads operands through the real MoE+attention+down-proj interleaving without
// perturbing a single bit. It is the --n-cpu-moe analogue of ForwardTP(ranks=1)==Forward.
func TestGLMDsaCPUOffloadPlacementInvariance(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	if !m.Cfg.isGLMMoeDsa() || !m.Cfg.IsMoE() {
		t.Fatalf("fixture is not a glm_moe_dsa MoE model (isGLMMoeDsa=%v IsMoE=%v)", m.Cfg.isGLMMoeDsa(), m.Cfg.IsMoE())
	}
	prompt := []int{3, 17, 5, 23}

	lPlain := m.NewSession().Prefill(prompt)
	sOff := m.NewSession()
	sOff.CPUOffloadExperts = true
	lOff := sOff.Prefill(prompt)

	if len(lPlain) != cfg.VocabSize || len(lOff) != cfg.VocabSize {
		t.Fatalf("logits shape plain=%d off=%d want vocab=%d", len(lPlain), len(lOff), cfg.VocabSize)
	}
	d, at := maxAbsDiff(lPlain, lOff)
	if d != 0 {
		t.Fatalf("CPU-offload (no backend) forward max|Δ|=%.3e at %d != 0 — the split is NOT a pure placement decision", d, at)
	}
	if a1, a2 := glmDsaArgmax(lPlain), glmDsaArgmax(lOff); a1 != a2 {
		t.Fatalf("CPU-offload argmax %d != plain argmax %d", a2, a1)
	}
	t.Logf("GLM-DSA CPU-offload placement-invariance: bit-exact vs plain host forward (max|Δ|=0, argmax-exact)")
}

// TestGLMDsaCPUOffloadHybridOverBackend is the real-hybrid witness: with a device backend attached
// AND CPUOffloadExperts on, the expert GEMMs must LEAVE the device (stay on host RAM) while the
// dense projections + router + attention stay on it — and the forward must still be correct. A
// recordingBackend over cpu-ref makes "which weights reached the device" directly observable. The
// proof is differential: an all-device session (offload off) records the expert shapes; the hybrid
// session (offload on) does NOT — the experts moved to host — while both record the dense shapes.
// The hybrid forward stays argmax-exact and within the f32-reduction-order floor vs the all-host
// reference, so the offload is correct, not just present. On the GPU server the device side is the cuda
// Q4_K kernel and the host side is the resident Q4_K GEMM; here both are cpu-ref, which is what
// makes the routing — the only new logic — witnessable without the A100s.
func TestGLMDsaCPUOffloadHybridOverBackend(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	prompt := []int{3, 17, 5, 23}

	// Expert tensor shapes in this fixture: gate/up = [I,H], down = [H,I]; shared experts share
	// these dims (MoEIntermediateSize==IntermediateSize). None of the dense/attention/router/head
	// weights share either shape (verified by the fixture geometry), so seeing [I,H] or [H,I] on the
	// backend is unambiguous evidence an EXPERT GEMM ran there.
	I, H := cfg.IntermediateSize, cfg.HiddenSize
	expertGateUp := [2]int{I, H}
	expertDown := [2]int{H, I}

	// All-device baseline: experts DO reach the backend.
	recAll := newRecordingBackend(compute.Default())
	sAll := m.NewBackendSession(recAll)
	lAll := sAll.Prefill(prompt)
	if !recAll.saw(expertGateUp[0], expertGateUp[1]) || !recAll.saw(expertDown[0], expertDown[1]) {
		t.Fatalf("all-device baseline did not run expert GEMMs on the backend (gate/up [%d,%d] seen=%v, down [%d,%d] seen=%v) — probe is broken",
			expertGateUp[0], expertGateUp[1], recAll.saw(expertGateUp[0], expertGateUp[1]),
			expertDown[0], expertDown[1], recAll.saw(expertDown[0], expertDown[1]))
	}

	// Hybrid: experts must NOT reach the backend; dense projections + router still must.
	recOff := newRecordingBackend(compute.Default())
	sOff := m.NewBackendSession(recOff)
	sOff.CPUOffloadExperts = true
	lOff := sOff.Prefill(prompt)

	if recOff.saw(expertGateUp[0], expertGateUp[1]) || recOff.saw(expertDown[0], expertDown[1]) {
		t.Errorf("CPU-offload hybrid still ran expert GEMMs on the backend (gate/up [%d,%d] seen=%v, down [%d,%d] seen=%v) — experts did NOT offload to host",
			expertGateUp[0], expertGateUp[1], recOff.saw(expertGateUp[0], expertGateUp[1]),
			expertDown[0], expertDown[1], recOff.saw(expertDown[0], expertDown[1]))
	}
	// The router ([NumExperts,H]) and an MLA projection ([QLoraRank,H]) prove the dense path stayed
	// on the device under offload.
	if !recOff.saw(cfg.NumExperts, H) {
		t.Errorf("CPU-offload hybrid did not run the router [%d,%d] on the backend — dense path wrongly offloaded", cfg.NumExperts, H)
	}
	if !recOff.saw(cfg.QLoraRank, H) {
		t.Errorf("CPU-offload hybrid did not run q_a_proj [%d,%d] on the backend — attention wrongly offloaded", cfg.QLoraRank, H)
	}

	// Correctness: the hybrid forward matches the all-host reference argmax-exact, within the same
	// f32-reduction-order floor the all-device GLM-DSA route tests use (1e-3). It routes a SUBSET of
	// GEMMs to the cpu-ref device that the all-device path routes, so its divergence cannot exceed it.
	lCPU := m.NewSession().Prefill(prompt)
	if len(lCPU) != cfg.VocabSize || len(lOff) != cfg.VocabSize || len(lAll) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d off=%d all=%d want vocab=%d", len(lCPU), len(lOff), len(lAll), cfg.VocabSize)
	}
	if a := glmDsaArgmax(lOff); a != glmDsaArgmax(lCPU) {
		t.Fatalf("CPU-offload hybrid argmax %d != all-host argmax %d", a, glmDsaArgmax(lCPU))
	}
	d, at := maxAbsDiff(lCPU, lOff)
	if d > 1e-3 {
		t.Fatalf("CPU-offload hybrid forward max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — routing bug", d, at)
	}
	t.Logf("GLM-DSA --n-cpu-moe hybrid on backend %q: experts host-offloaded (off the device), router+attention on device; argmax-exact, max|Δ|=%.3e vs all-host", recOff.Name(), d)
}

// TestGLMDsaCPUOffloadRoutesIndexSelection witnesses that the learned-indexer SCORE + top-k SELECTION
// reaches the DEVICE through the --n-cpu-moe split (splitKernel.indexSelect), not just through the plain
// all-device backendKernel. isExpertWeight keeps the index PROJECTIONS on the device under offload, so
// the selection compute they feed must run there too; without the splitKernel forwarder the hybrid would
// silently keep the indexer host-resident while every other DSA op ran on the kernel. A DECODE step
// (where glmDsaIndexStep runs — Prefill uses the host-only whole-sequence index helper) over an
// offload-backed recording session must increase DSAIndexSelect calls and stay argmax-exact vs all-host.
func TestGLMDsaCPUOffloadRoutesIndexSelection(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	prompt := []int{3, 17, 5, 23}
	const next = 11 // the token decoded after the prompt — this is where glmDsaIndexStep runs

	rec := newRecordingBackend(compute.Default())
	if _, ok := compute.Backend(rec).(compute.DSAIndexBackend); !ok {
		t.Fatalf("recordingBackend over %q is not a DSAIndexBackend — the split's device side could not route index selection", rec.Name())
	}
	sOff := m.NewBackendSession(rec)
	sOff.CPUOffloadExperts = true // drives the splitKernel path, not the plain backendKernel
	sOff.Prefill(prompt)
	callsBefore, _ := rec.index()
	lOff := sOff.Step(next)

	calls, keys := rec.index()
	if calls == callsBefore {
		t.Fatalf("GLM-DSA index selection never reached the backend through the --n-cpu-moe split during decode (splitKernel did not forward indexSelect — still host-resident)")
	}
	if keys == 0 {
		t.Fatalf("GLM-DSA index selection reached the split backend but scored 0 keys (empty routing)")
	}

	sCPU := m.NewSession()
	sCPU.Prefill(prompt)
	lCPU := sCPU.Step(next)
	if len(lCPU) != cfg.VocabSize || len(lOff) != cfg.VocabSize {
		t.Fatalf("logits shape cpu=%d off=%d want vocab=%d", len(lCPU), len(lOff), cfg.VocabSize)
	}
	if a, ac := glmDsaArgmax(lOff), glmDsaArgmax(lCPU); a != ac {
		t.Fatalf("GLM-DSA offload decode (incl. index selection through the split) argmax %d != CPU argmax %d (selection flipped a key)", a, ac)
	}
	d, at := maxAbsDiff(lCPU, lOff)
	if d > 1e-3 {
		t.Fatalf("GLM-DSA offload decode max|Δ|=%.3e at %d (> 1e-3 f32-order floor) — split index-selection routing bug", d, at)
	}
	t.Logf("GLM-DSA index selection through --n-cpu-moe split on %q: %d DSAIndexSelect calls over %d scored keys during decode; argmax-exact, max|Δ|=%.3e vs all-host",
		rec.Name(), calls-callsBefore, keys, d)
}
