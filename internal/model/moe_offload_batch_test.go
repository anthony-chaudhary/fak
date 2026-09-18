package model

import (
	"math"
	"math/rand"
	"strings"
	"testing"
)

// moe_offload_batch_test.go — witness for #12972: the --n-cpu-moe split must reach the SAME
// batched host-expert primitive (hostBatchedGLMExperts) the bare resident kernel does, instead of
// falling through to the ~3*K-dispatch per-expert loop.
//
// The split the live GLM-DSA forward builds is splitKernel{host: residentKernel, device: ...,
// onHost: isExpertWeight}. Its host side IS a residentKernel, but the concrete type threaded
// through apply is splitKernel, so the bare `mat.(residentKernel)` gate (glmMoeFFN.apply) and the
// type switch (moeFFN.apply) both MISS it and the per-expert loop runs. splitHostResidentExperts
// closes that gap — and, critically, only for an ALL-host-routed pick set, so a GRADED split
// (ExpertSpillLayers) that keeps some experts on the device declines rather than stealing them.
//
// WITNESS SPLIT (why there are two assertions per test): the batched delta and the per-expert
// loop are BIT-IDENTICAL by design, so the numeric max|Δ|==0 check is the bit-identity witness
// ONLY — it passes even when the `apply` wiring is deleted and the loop runs. The DISPATCH
// witness is the hostBatchedExpertDispatch counter: it advances exactly once per layer only
// when apply reaches hostBatchedGLMExperts, so removing that wiring makes the counter assertion
// FAIL. A scale test that changes the wiring (e.g. reverts the split arm) must break the counter
// assertion; a scale test that changes only arithmetic must break the numeric one.

// TestGLMMoeFFNSplitHostBatchedMatchesLoop drives glmMoeFFN.apply with the degenerate no-backend
// split (host==device==residentKernel, the exact kernel glmDsaMatKernel builds when offload is on
// and no device is attached) and pins BOTH the numeric bit-identity vs the per-expert reference
// AND the dispatch: splitHostResidentExperts must admit the split, the batched primitive must
// record exactly one layer / len(picks) experts through hostBatchedGLMExperts, and the result must
// equal it.
func TestGLMMoeFFNSplitHostBatchedMatchesLoop(t *testing.T) {
	const H, MI, E, K = 256, 256, 6, 2 // dims in multiples of qkK (256); realistic top-k
	cfg := Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: MI, MoEIntermediateSize: MI, VocabSize: 4,
		RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: E, NumExpertsPerTok: K, NormTopKProb: true, EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(12972))

	mkQ4K := func(out, in int) *q4kTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q4kBlockBytes)
		blk := make([]byte, q4kBlockBytes)
		for o := 0; o < out; o++ {
			for b := 0; b < nblk; b++ {
				randQ4KBlock(rng, blk)
				copy(raw[(o*nblk+b)*q4kBlockBytes:], blk)
			}
		}
		return quantizeQ4KFromRaw(raw, out, in)
	}
	mkQ6K := func(out, in int) *kQuantTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q6kBlockBytes)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		return quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	}

	// Resident experts in the q4_k_m shape (gate/up Q4_K -> q4kw, down Q6_K -> kqw) plus a resident
	// E-row router, exactly as TestMoEFFNBatchedMatchesLoop builds them.
	m := &Model{Cfg: cfg}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	m.q4kw[routerName(0)] = mkQ4K(E, H)
	for e := 0; e < E; e++ {
		m.q4kw[expertName(0, e, "gate_proj.weight")] = mkQ4K(MI, H)
		m.q4kw[expertName(0, e, "up_proj.weight")] = mkQ4K(MI, H)
		m.kqw[expertName(0, e, "down_proj.weight")] = mkQ6K(H, MI)
	}

	// The offload-hybrid split, driven through the REAL production seam: glmDsaMatKernel builds
	// splitKernel{host: residentKernel, device, onHost: expertSpillOnHost()} for `--n-cpu-moe`
	// (no device attached here -> device==host==residentKernel). Constructing it via the session
	// (not by hand) pins that the PRODUCTION predicate — expertSpillOnHost, whose default grade is
	// isExpertWeight — is what the batch admits, the exact bug class #12972 fixes.
	s := &Session{M: m, CPUOffloadExperts: true}
	mat := s.glmDsaMatKernel()
	sk, ok := mat.(splitKernel)
	if !ok {
		t.Fatalf("glmDsaMatKernel returned %T, want splitKernel for CPUOffloadExperts", mat)
	}
	if _, ok := sk.host.(residentKernel); !ok {
		t.Fatalf("glmDsaMatKernel split host is %T, want residentKernel", sk.host)
	}
	if sk.onHost == nil {
		t.Fatalf("glmDsaMatKernel split carries a nil onHost predicate")
	}

	xn := make([]float32, H)
	for i := range xn {
		xn[i] = float32(rng.NormFloat64())
	}
	picks := glmRoute(m, 0, xn, mat)
	if len(picks) != K {
		t.Fatalf("router returned %d picks, want top-k=%d", len(picks), K)
	}

	// The dispatch helper must ADMIT this production split — this is the load-bearing assertion.
	// On the pre-fix code the symbol does not exist (build failure); a future regression that drops
	// the split arm leaves it declining here and the loop silently returning.
	if !splitHostResidentExperts(mat, 0, picks) {
		t.Fatalf("splitHostResidentExperts declined the all-host-routed --n-cpu-moe split — the batch wiring is not reached")
	}

	// Reference: the exact per-expert loop glmMoeFFN.apply ran before the fix.
	wantLoop := make([]float32, H)
	for _, pk := range picks {
		out := expertSwiGLU(m, 0, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			wantLoop[i] += pk.weight * out[i]
		}
	}

	// The batched primitive is available for this resident config (the lever can fire). This
	// call itself advances the dispatch counter, so RESET it afterwards and measure only the
	// real apply below.
	probe := make([]float32, H)
	if !m.hostBatchedGLMExperts(0, xn, probe, picks) {
		t.Fatalf("hostBatchedGLMExperts declined for a resident q4_k_m split — the lever did not fire")
	}
	resetHostBatchedExpertDispatch()

	// The live path.
	got := glmMoeFFN{}.apply(m, 0, xn, mat)

	// DISPATCH WITNESS (the assertion that fails on pre-fix wiring): apply must have reached
	// hostBatchedGLMExperts exactly once for this layer, with len(picks) experts. On the pre-fix
	// code the split misses the residentKernel gate, the per-expert loop runs, and this counter
	// stays at zero — while the numeric check below still passes (loop == batch bit-for-bit).
	layers, experts := hostBatchedExpertDispatch()
	if layers != 1 {
		t.Fatalf("glmMoeFFN split apply fired the batched host-experts path %d times, want exactly 1 — the #12972 wiring is not reached", layers)
	}
	if experts != int64(len(picks)) {
		t.Fatalf("glmMoeFFN split batched dispatch counted %d experts, want len(picks)=%d", experts, len(picks))
	}

	var maxAbs float64
	for i := 0; i < H; i++ {
		if d := math.Abs(float64(got[i] - wantLoop[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs != 0 {
		t.Fatalf("glmMoeFFN split batched vs per-expert loop max|Δ|=%g, want 0 (bit-identity)", maxAbs)
	}
	var norm float64
	for _, v := range got {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		t.Fatalf("delta is all-zero; test is vacuous (experts contributed nothing)")
	}
	t.Logf("glmMoeFFN --n-cpu-moe split batched == per-expert loop, max|Δ|=0 over %d picks", len(picks))
}

// TestMoEFFNSplitHostBatchedMatchesLoop is the Mixtral/Qwen3-MoE counterpart: moeFFN.apply's type
// switch must also admit the host-resident split and reach hostBatchedGLMExperts. As in the GLM
// test, the numeric bit-identity is the bit-identity witness and the dispatch counter is the
// wiring witness — the latter fails when the split arm is removed from moeFFN.apply's switch.
func TestMoEFFNSplitHostBatchedMatchesLoop(t *testing.T) {
	const H, MI, E, K = 256, 256, 6, 2
	cfg := Config{
		HiddenSize: H, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: MI, MoEIntermediateSize: MI, VocabSize: 4,
		RMSNormEps: 1e-5, RopeTheta: 10000,
		NumExperts: E, NumExpertsPerTok: K, NormTopKProb: true, EOSTokenID: -1,
	}
	rng := rand.New(rand.NewSource(12973))

	mkQ4K := func(out, in int) *q4kTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q4kBlockBytes)
		blk := make([]byte, q4kBlockBytes)
		for o := 0; o < out; o++ {
			for b := 0; b < nblk; b++ {
				randQ4KBlock(rng, blk)
				copy(raw[(o*nblk+b)*q4kBlockBytes:], blk)
			}
		}
		return quantizeQ4KFromRaw(raw, out, in)
	}
	mkQ6K := func(out, in int) *kQuantTensor {
		nblk := in / qkK
		raw := make([]byte, out*nblk*q6kBlockBytes)
		for i := range raw {
			raw[i] = byte(rng.Intn(256))
		}
		return quantizeKQuantFromRaw(raw, out, in, kindQ6K)
	}

	m := &Model{Cfg: cfg, q4kw: map[string]*q4kTensor{}, kqw: map[string]*kQuantTensor{}}
	m.q4kw[routerName(0)] = mkQ4K(E, H)
	for e := 0; e < E; e++ {
		m.q4kw[expertName(0, e, "gate_proj.weight")] = mkQ4K(MI, H)
		m.q4kw[expertName(0, e, "up_proj.weight")] = mkQ4K(MI, H)
		m.kqw[expertName(0, e, "down_proj.weight")] = mkQ6K(H, MI)
	}

	mat := splitKernel{host: residentKernel{m}, device: residentKernel{m}, onHost: isExpertWeight}
	xn := make([]float32, H)
	for i := range xn {
		xn[i] = float32(rng.NormFloat64())
	}
	picks := route(m, 0, xn, mat)

	wantLoop := make([]float32, H)
	for _, pk := range picks {
		out := expertSwiGLU(m, 0, pk.expert, xn, mat)
		for i := 0; i < H; i++ {
			wantLoop[i] += pk.weight * out[i]
		}
	}
	resetHostBatchedExpertDispatch()
	got := moeFFN{}.apply(m, 0, xn, mat)

	// DISPATCH WITNESS: moeFFN.apply's split arm must have fired the batched host primitive
	// exactly once for this layer. On pre-fix wiring the type switch misses splitKernel, the
	// per-expert loop runs, and this stays zero (while the numeric check below still passes).
	layers, experts := hostBatchedExpertDispatch()
	if layers != 1 {
		t.Fatalf("moeFFN split apply fired the batched host-experts path %d times, want exactly 1 — the #12972 wiring is not reached", layers)
	}
	if experts != int64(len(picks)) {
		t.Fatalf("moeFFN split batched dispatch counted %d experts, want len(picks)=%d", experts, len(picks))
	}

	var maxAbs float64
	for i := 0; i < H; i++ {
		if d := math.Abs(float64(got[i] - wantLoop[i])); d > maxAbs {
			maxAbs = d
		}
	}
	if maxAbs != 0 {
		t.Fatalf("moeFFN split batched vs per-expert loop max|Δ|=%g, want 0 (bit-identity)", maxAbs)
	}
	var norm float64
	for _, v := range got {
		norm += float64(v) * float64(v)
	}
	if norm == 0 {
		t.Fatalf("delta is all-zero; test is vacuous")
	}
}

// TestSplitHostResidentExpertsDeclinesGradedSpill pins the mixed-placement safety property: when
// the split's predicate keeps even ONE picked expert weight on the device (the ExpertSpillLayers
// grade, expert_spill_placement.go), splitHostResidentExperts must return ok=false so the caller
// takes the per-expert loop. The batched host primitive only models host-resident experts, so
// admitting a mixed layer would silently steal the device-kept experts to the host — exactly the
// placement the operator's `--n-cpu-moe N` did NOT budget.
func TestSplitHostResidentExpertsDeclinesGradedSpill(t *testing.T) {
	const H = 256
	cfg := Config{HiddenSize: H, NumLayers: 1, NumHeads: 1, NumKVHeads: 1, HeadDim: 2,
		IntermediateSize: H, MoEIntermediateSize: H, VocabSize: 4, NumExperts: 4, NumExpertsPerTok: 2}
	m := &Model{Cfg: cfg}
	allHost := func(string) bool { return true }
	picks := []routePick{{expert: 0, weight: 0.6}, {expert: 1, weight: 0.4}}

	// Every pick host-routed -> admitted.
	if !splitHostResidentExperts(splitKernel{host: residentKernel{m}, device: residentKernel{m}, onHost: allHost}, 0, picks) {
		t.Fatalf("all-host picks declined; want admitted")
	}

	// Graded: expert 1 is device-kept, so the split declines (and the loop runs).
	graded := func(name string) bool {
		return isExpertWeight(name) && !strings.Contains(name, ".mlp.experts.1.")
	}
	if splitHostResidentExperts(splitKernel{host: residentKernel{m}, device: residentKernel{m}, onHost: graded}, 0, picks) {
		t.Fatalf("graded split with a device-kept picked expert was ADMITTED — it would steal that expert to the host")
	}

	// A non-split kernel never admits.
	if splitHostResidentExperts(residentKernel{m}, 0, picks) {
		t.Fatalf("bare residentKernel admitted by splitHostResidentExperts; want decline")
	}
	// A split whose host side is not a residentKernel never admits.
	if splitHostResidentExperts(splitKernel{host: f32Kernel{m}, device: residentKernel{m}, onHost: allHost}, 0, picks) {
		t.Fatalf("non-resident host admitted; want decline")
	}
	// A hand-built split with a nil predicate never admits (no host routing at all).
	if splitHostResidentExperts(splitKernel{host: residentKernel{m}, device: residentKernel{m}}, 0, picks) {
		t.Fatalf("nil onHost admitted; want decline")
	}
}
