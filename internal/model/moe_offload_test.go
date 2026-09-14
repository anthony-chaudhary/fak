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

// TestExpertWeightClassification pins the two disjoint expert-weight classes the offload predicate is
// built from: isRoutedExpertWeight matches the per-token routed experts (.mlp.experts.<e>.*),
// isSharedExpertWeight matches the always-on shared experts (.mlp.shared_experts.*), the two never
// overlap, isExpertWeight is exactly their OR, and NEITHER class ever matches the router/gate weight
// (routerName) — the router is a dense every-token GEMM that stays on the device. A predicate that
// classified the router as an expert, or that let the shared form leak into the routed class, is
// caught here structurally.
func TestExpertWeightClassification(t *testing.T) {
	router := routerName(0) // model.layers.0.mlp.gate.weight — the every-token dense router
	cases := []struct {
		desc                   string
		weight                 string
		wantRouted, wantShared bool
	}{
		{"routed gate", expertName(0, 0, "gate_proj.weight"), true, false},
		{"routed up (other layer/expert)", expertName(3, 7, "up_proj.weight"), true, false},
		{"routed down", expertName(1, 2, "down_proj.weight"), true, false},
		{"routed bias rides with its GEMM", expertName(0, 0, "gate_proj.bias"), true, false},
		{"shared gate", "model.layers.5.mlp.shared_experts.gate_proj.weight", false, true},
		{"shared down", "model.layers.5.mlp.shared_experts.down_proj.weight", false, true},
		{"router/gate weight", router, false, false},
		{"router bias", routerBiasName(0), false, false},
		{"dense (non-MoE) mlp weight", "model.layers.0.mlp.gate_proj.weight", false, false},
	}
	for _, c := range cases {
		gotRouted, gotShared := isRoutedExpertWeight(c.weight), isSharedExpertWeight(c.weight)
		if gotRouted != c.wantRouted {
			t.Errorf("%s: isRoutedExpertWeight(%q) = %v, want %v", c.desc, c.weight, gotRouted, c.wantRouted)
		}
		if gotShared != c.wantShared {
			t.Errorf("%s: isSharedExpertWeight(%q) = %v, want %v", c.desc, c.weight, gotShared, c.wantShared)
		}
		if gotRouted && gotShared {
			t.Errorf("%s: %q classified as BOTH routed and shared — the classes must be disjoint", c.desc, c.weight)
		}
		if got, want := isExpertWeight(c.weight), gotRouted || gotShared; got != want {
			t.Errorf("%s: isExpertWeight(%q) = %v, want OR of routed|shared = %v", c.desc, c.weight, got, want)
		}
	}
	// The task's load-bearing assertion, stated directly: the router/gate weight is neither class.
	if isRoutedExpertWeight(router) || isSharedExpertWeight(router) {
		t.Fatalf("router weight %q must classify as neither routed nor shared (it stays on the device), got routed=%v shared=%v",
			router, isRoutedExpertWeight(router), isSharedExpertWeight(router))
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
// It exercises the RUNTIME placement predicate hostOffloadWeight (not the accounting union
// isExpertWeight), so the shared-expert case pins the #1304 contract: the always-on shared expert
// stays on the DEVICE while the routed experts offload to host.
func TestSplitKernelRoutesByPredicate(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	host := newRecordingKernel(residentKernel{m})
	device := newRecordingKernel(residentKernel{m})
	k := splitKernel{host: host, device: device, onHost: hostOffloadWeight}

	// A representative activation; the exact values do not matter - we assert routing + equality.
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
		{"model.layers.0.mlp.shared_experts.gate_proj.weight", m.Cfg.MoEIntermediateSize * m.Cfg.NSharedExperts, H, false}, // shared expert is device-pinned (#1304)
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
// AND CPUOffloadExperts on, the ROUTED expert GEMMs must LEAVE the device (stay on host RAM) while
// the dense projections + router + attention — and the always-on shared expert (#1304) — stay on it,
// and the forward must still be correct. A recordingBackend over cpu-ref makes "which weights
// reached the device" directly observable BY NAME: the shared expert and the routed experts share
// [I,H]/[H,I] shapes in this fixture, so a shape-only probe cannot distinguish them (the pre-#1304
// bug this test encoded). recordingBackend.UploadClass pairs each weight's upload site
// ("hal-weight <name>") with its buffer, so MatMul resolves the NAME. The proof is differential: an
// all-device session (offload off) runs routed + shared expert names on the device; the hybrid
// session (offload on) runs the SHARED expert name but NONE of the routed ones — the routed experts
// moved to host — while both run the dense/router names. The hybrid forward stays argmax-exact and
// within the f32-reduction-order floor vs the all-host reference.
func TestGLMDsaCPUOffloadHybridOverBackend(t *testing.T) {
	path, cfg := writeTinyGLMDsaSafetensorsFixture(t, "F32", true, false, true /*withMoE*/, true /*withSharedExperts*/)
	m, err := LoadSafetensors(path, cfg)
	if err != nil {
		t.Fatalf("LoadSafetensors: %v", err)
	}
	prompt := []int{3, 17, 5, 23}

	routedOnBackend := func(rec *recordingBackend) []string {
		var out []string
		for _, name := range rec.namesSeen() {
			if isRoutedExpertWeight(name) {
				out = append(out, name)
			}
		}
		return out
	}
	sharedOnBackend := func(rec *recordingBackend) []string {
		var out []string
		for _, name := range rec.namesSeen() {
			if isSharedExpertWeight(name) {
				out = append(out, name)
			}
		}
		return out
	}
	const sharedGate = "model.layers.0.mlp.shared_experts.gate_proj.weight"

	// All-device baseline: BOTH routed and shared experts DO reach the backend.
	recAll := newRecordingBackend(compute.Default())
	sAll := m.NewBackendSession(recAll)
	lAll := sAll.Prefill(prompt)
	if len(routedOnBackend(recAll)) == 0 {
		t.Fatalf("all-device baseline ran NO routed expert on the backend — name probe is broken (seen=%v)", recAll.namesSeen())
	}
	if len(sharedOnBackend(recAll)) == 0 {
		t.Fatalf("all-device baseline ran NO shared expert on the backend — name probe is broken (seen=%v)", recAll.namesSeen())
	}

	// Hybrid: the ROUTED experts must NOT reach the backend; the SHARED expert must.
	recOff := newRecordingBackend(compute.Default())
	sOff := m.NewBackendSession(recOff)
	sOff.CPUOffloadExperts = true
	lOff := sOff.Prefill(prompt)

	if routed := routedOnBackend(recOff); len(routed) != 0 {
		t.Errorf("CPU-offload hybrid ran routed experts on the backend (should be host-offloaded): %v", routed)
	}
	if !recOff.sawName(sharedGate) {
		t.Errorf("CPU-offload hybrid did NOT run the shared expert %q on the backend — it must stay DEVICE-resident (#1304); shared seen=%v", sharedGate, sharedOnBackend(recOff))
	}
	// The router and an MLA projection prove the dense path stayed on the device under offload.
	if !recOff.saw(cfg.NumExperts, cfg.HiddenSize) {
		t.Errorf("CPU-offload hybrid did not run the router [%d,%d] on the backend — dense path wrongly offloaded", cfg.NumExperts, cfg.HiddenSize)
	}
	if !recOff.saw(cfg.QLoraRank, cfg.HiddenSize) {
		t.Errorf("CPU-offload hybrid did not run q_a_proj [%d,%d] on the backend — attention wrongly offloaded", cfg.QLoraRank, cfg.HiddenSize)
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
	t.Logf("GLM-DSA --n-cpu-moe hybrid on backend %q: routed experts host-offloaded (off the device), shared expert DEVICE-resident, router+attention on device; argmax-exact, max|Δ|=%.3e vs all-host", recOff.Name(), d)
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
