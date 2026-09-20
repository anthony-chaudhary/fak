package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestRoutedExpertReachableFromGLMDsaMatKernel is the #5111 regression witness.
//
// It exercises the REAL kernel-construction path — Session.glmDsaMatKernel, the
// single place decodeBandGLMDsa selects its matKernel — and asserts the routed
// resident-k-quant device path is reachable from it. The existing HAL tests
// (moe_expert_hal_test.go) hand-build sessionQ4KKernel{s: sess} directly and so
// CANNOT catch this class: before the fix, glmDsaMatKernel builds backendKernel
// (never sessionQ4KKernel), so routedExpertKQuantActive's old type-assert on
// sessionQ4KKernel always returned false and expert_parallel.go's host scalar
// batch always won on the live GLM-5.2 serve — the ~0.09-0.21 tok/s cause (#4784).
//
// The predicate is capability-keyed, so this witness needs no GPU: it proves the
// gate now opens for a routed-capable backend and stays closed for every other
// configuration (cpu-ref/Metal, host-only, and --cpu-offload-experts where the
// experts are intentionally host-pinned).
func TestRoutedExpertReachableFromGLMDsaMatKernel(t *testing.T) {
	cfg := expertHALTestConfig(64)
	m := NewSyntheticMoE(cfg)

	// expertHALRecordingBackend advertises SupportsRoutedExpertKQuant()=true; the
	// base cpu-ref backend (compute.Default) does not, which is why the recording
	// backend overrides it.
	capable := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}

	cases := []struct {
		name string
		s    *Session
		want bool
	}{
		// THE fix: a routed-capable backend, experts resident (no offload) — the
		// live EP serve config (#4777) — now reaches the device routed path.
		{"capable-backend-resident", &Session{M: m, Backend: capable, halW: map[string]compute.Tensor{}}, true},
		// --cpu-offload-experts pins experts on host by design: splitKernel must NOT
		// claim the device routed capability.
		{"capable-backend-offload-host-pinned", &Session{M: m, Backend: capable, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}, false},
		// cpu-ref / Metal: capability absent, path byte-identical to before the seam.
		{"non-routed-cpuref-backend", &Session{M: m, Backend: compute.Default(), halW: map[string]compute.Tensor{}}, false},
		// Pure host path: residentKernel, no session capability.
		{"host-only-no-backend", &Session{M: m}, false},
	}

	for _, tc := range cases {
		mat := tc.s.glmDsaMatKernel()
		if got := routedExpertKQuantActive(mat); got != tc.want {
			t.Errorf("%s: routedExpertKQuantActive(glmDsaMatKernel()=%T) = %v, want %v", tc.name, mat, got, tc.want)
		}
	}
}

// ---------------------------------------------------------------------------
// #5111 residual: EXECUTION + PARITY, not just the predicate.
//
// The predicate witness above proves the gate OPENS for the live decode kernel.
// It does not prove the device path then RUNS, nor that it computes the right
// number. The three tests below close that: they drive expertSwiGLU (and the raw
// k-quant mul) through s.glmDsaMatKernel() — the kernel decodeBandGLMDsa actually
// builds — count the device ops that executed, and pin the result against the host
// oracle. Every other routed-expert HAL test hand-builds sessionQ4KKernel{s: sess},
// a kernel the live decode NEVER constructs, so none of them can witness this.
//
// No GPU is needed: compute.Default() is the cpu-ref backend, which natively
// executes Q4_K, Q5_K and Q6_K MatMul (internal/compute/cpuref.go). These are
// reachability/execution/parity witnesses only — they say NOTHING about tok/s,
// which needs the multi-GPU run tracked by the parent throughput issue.

// q4kmRoutedExpertModel builds the exact q4_k_m routed-expert residency a live
// GLM-5.2 serve loads for one expert, covering BOTH halves of the q4_k_m MIXTURE:
// gate/up land in m.q4kw as raw Q4_K super-blocks (staged by weightHALQ4K) while
// down lands in m.kqw as raw Q6_K (staged by weightHALKQuant). That is the same
// split hostBatchedGLMExperts keys on (moe_host_batch.go), so a change that fixes
// only "the Q4_K path" half-applies here and shows up instead of passing silently.
//
// The f32 manifest copies NewSyntheticMoE synthesizes for those three names are
// dropped so the quantized stores are the ONLY residency — exactly the state after
// a q4_k_m GGUF load, where no f32 copy of an expert weight exists. This is load
// bearing twice over: with the f32 copies present, residentKernel would read them
// instead of the quantized bytes (making the parity check compare two different
// weights), and glmDsaWeightHAL's m.has(name) branch would win over its kqw branch
// so weightHALKQuant — the seam #5111 restored — would never be reached at all.
func q4kmRoutedExpertModel(t *testing.T) (*Model, [3]string) {
	t.Helper()
	const H = 256
	m := NewSyntheticMoE(expertHALTestConfig(H))
	names := [3]string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, name := range names[:2] {
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 401+i), nblk: 1}
	}
	m.kqw[names[2]] = expertHALQ6KTensor(H, H, 403)
	for _, name := range names {
		delete(m.manifest, name)
	}
	return m, names
}

// routedExpertRawQ4K is buildRawQ4K with the super-block f16 scale exponents pinned
// to one realistic magnitude. The shared generator leaves the exponent random over
// [1,30], so d spans 2^-14..2^15; run through gate·up and then a down projection that
// lands the expert output near 1e16 — finite and deterministic, but a scale at which
// nothing but a RELATIVE bound is left to assert. Pinning the exponent (sign and
// mantissa stay random, so the weights are still adversarial in every other respect)
// keeps the fixture in the magnitude band a real q4_k_m checkpoint occupies.
func routedExpertRawQ4K(t *testing.T, out, in, seed int) []byte {
	t.Helper()
	raw := buildRawQ4K(t, out, in, seed)
	for b := 0; b+q4kBlockBytes <= len(raw); b += q4kBlockBytes {
		// f16 little-endian: byte1 = sign(1) | exp(5) | frac-high(2). Super-block
		// bytes 0..1 are d and 2..3 are dmin; exponent 9 => a 2^-6 magnitude.
		for s := 0; s < 2; s++ {
			raw[b+s*2+1] = (raw[b+s*2+1] & 0x83) | byte(9<<2)
		}
	}
	return raw
}

// routedExpertParity fails unless got tracks the host oracle want within the same
// scale-relative bound TestExpertSwiGLUHALParity uses (1e-5·max(1,‖want‖∞)) — the
// honest tolerance for a device that reassociates the same f32 reduction over the
// same resident bytes. The all-zero refusal is deliberate: a Q6_K fixture of zero
// blocks dequantizes to a zero projection, which would make any parity assertion
// pass vacuously, so an oracle with no signal is a test bug, not a pass.
func routedExpertParity(t *testing.T, label string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: output len=%d, want %d", label, len(got), len(want))
	}
	var maxAbs, scale float64
	for i := range want {
		if v := float64(got[i]); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("%s: output[%d]=%v", label, i, got[i])
		}
		if d := math.Abs(float64(got[i] - want[i])); d > maxAbs {
			maxAbs = d
		}
		if a := math.Abs(float64(want[i])); a > scale {
			scale = a
		}
	}
	if scale == 0 {
		t.Fatalf("%s: host oracle is all-zero — a vacuous parity comparison", label)
	}
	const relTol = 1e-5
	rel := maxAbs / math.Max(1, scale)
	if rel > relTol {
		t.Fatalf("%s: parity rel=%g > %g (maxAbs=%g, scale=%g)", label, rel, relTol, maxAbs, scale)
	}
	t.Logf("%s: parity rel=%g (bound %g) maxAbs=%g scale=%g", label, rel, relTol, maxAbs, scale)
}

// TestGLMDsaMatKernelRoutedExpertRunsOnDevice is the EXECUTION half of the #5111
// witness. It runs one routed expert through s.glmDsaMatKernel() — asserting first
// that the kernel really is backendKernel, the type the old predicate could never
// match — and requires that all three expert GEMMs plus the SwiGLU actually
// executed on the backend, with both halves of the q4_k_m mixture staged resident
// (Q4_K×2 for gate/up, Q6_K×1 for down). Pre-#5111 this identical call ran 2
// MatMuls and 0 SwiGLUs: expertSwiGLU's HAL branch type-asserted sessionQ4KKernel
// so it fell through, and backendKernel.mul returned kQuantMatRows for the kqw
// down_proj, keeping it and the activation on the host.
func TestGLMDsaMatKernelRoutedExpertRunsOnDevice(t *testing.T) {
	// residentKernel is the host oracle below. Keep it on the exact f32 Q4_K
	// path rather than arm64's approximate activation-quantized SDOT path.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, names := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	// Host oracle: residentKernel is precisely what glmDsaMatKernel returns with no
	// Backend, so want is the number this expert produced on the host path — the
	// number the device path must reproduce now that it wins.
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	mat := s.glmDsaMatKernel()
	if _, ok := mat.(backendKernel); !ok {
		t.Fatalf("live decode matKernel = %T, want backendKernel (the kernel decodeBandGLMDsa builds)", mat)
	}

	got := expertSwiGLU(m, 0, 0, x, mat)

	if be.matmuls != 3 || be.swiglu != 1 {
		t.Fatalf("device expert ops matmul=%d swiglu=%d, want 3/1 (pre-#5111 this exact call ran 2/0: down_proj and the SwiGLU stayed host)", be.matmuls, be.swiglu)
	}
	if be.uploads[compute.Q4_K] != 2 || be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("resident expert uploads q4_k=%d q6_k=%d, want 2/1 (gate+up from q4kw, down from kqw)", be.uploads[compute.Q4_K], be.uploads[compute.Q6_K])
	}
	if be.uploads[compute.F16] != 0 {
		t.Fatalf("expanded F16 uploads=%d, want 0 — k-quant residency must stay at checkpoint size", be.uploads[compute.F16])
	}
	for _, key := range []string{"q4k:" + names[0], "q4k:" + names[1], "kquant-raw:" + names[2]} {
		if _, ok := s.halW[key]; !ok {
			t.Fatalf("expert weight %q is not device-resident; halW holds %d entries", key, len(s.halW))
		}
	}
	routedExpertParity(t, "glm_moe_dsa routed expert (Q4_K gate/up + Q6_K down)", got, want)

	// A second token reuses every resident weight; only the activation crosses again.
	expertSwiGLU(m, 0, 0, x, mat)
	if be.uploads[compute.Q4_K] != 2 || be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("warm token restaged resident weights: q4_k=%d q6_k=%d, want 2/1", be.uploads[compute.Q4_K], be.uploads[compute.Q6_K])
	}
	if be.matmuls != 6 || be.swiglu != 2 {
		t.Fatalf("two-token device ops matmul=%d swiglu=%d, want 6/2", be.matmuls, be.swiglu)
	}
}

// TestGLMDsaMatKernelStagesRoutedKQuantDownProj pins the kqw branch of
// backendKernel.mul (kernel.go) directly, the one #5111 seam the fused
// expertSwiGLUHAL route above skips over. It is the second half of the q4_k_m
// mixture: the Q6_K down_proj, reached through the generic named mul the GLM-DSA
// forward uses for every other weight. Pre-#5111 that branch unconditionally
// returned kQuantMatRows (0 device MatMuls, 0 uploads) and glmDsaWeightHAL PANICKED
// on a kqw weight that reached the device upload path; now it stages the GGUF bytes
// verbatim through weightHALKQuant and computes the same number on the backend.
func TestGLMDsaMatKernelStagesRoutedKQuantDownProj(t *testing.T) {
	m, names := q4kmRoutedExpertModel(t)
	dn := names[2]
	H, I := m.Cfg.HiddenSize, m.Cfg.expertIntermediate()

	// The SwiGLU intermediate the down projection consumes.
	g := make([]float32, I)
	for i := range g {
		g[i] = float32((i%23)-11) / 96
	}
	host := residentKernel{m}
	want := host.mul(dn, host.prep(g), H, I)

	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	mat := s.glmDsaMatKernel()
	got := mat.mul(dn, mat.prep(g), H, I)

	if be.matmuls != 1 || be.uploads[compute.Q6_K] != 1 {
		t.Fatalf("kqw branch of backendKernel.mul: matmul=%d q6_k uploads=%d, want 1/1 (pre-#5111 it returned the host kQuantMatRows at 0/0)", be.matmuls, be.uploads[compute.Q6_K])
	}
	if _, ok := s.halW["kquant-raw:"+dn]; !ok {
		t.Fatalf("down_proj was not staged verbatim by weightHALKQuant; halW holds %d entries", len(s.halW))
	}
	if be.uploads[compute.F16] != 0 {
		t.Fatalf("expanded F16 uploads=%d, want 0 — the raw Q6_K bytes stage as-is", be.uploads[compute.F16])
	}
	routedExpertParity(t, "backendKernel.mul Q6_K down_proj", got, want)

	// Warm call: the resident weight is reused, only the GEMM repeats.
	mat.mul(dn, mat.prep(g), H, I)
	if be.uploads[compute.Q6_K] != 1 || be.matmuls != 2 {
		t.Fatalf("warm k-quant mul: q6_k uploads=%d matmul=%d, want 1/2", be.uploads[compute.Q6_K], be.matmuls)
	}
}

// routedIncapableBackend is the SAME recording backend with the ONE capability
// #5111 keys on switched off. It is how the pre-fix behaviour is witnessed without
// reverting production code: kernel type, residency and activation are all held
// fixed, so the op counts in the test below isolate exactly what
// supportsRoutedExpertKQuant buys — and prove the counts asserted above are not a
// constant the fixture would produce either way.
type routedIncapableBackend struct{ *expertHALRecordingBackend }

func (routedIncapableBackend) SupportsRoutedExpertKQuant() bool { return false }

// TestGLMDsaMatKernelRoutedExpertStaysHostWithoutCapability is the negative control
// for the two witnesses above AND the no-regression gate for every non-CUDA backend:
// cpu-ref and Metal do not advertise the routed k-quant capability, so their routed
// expert must still run the host k-quant GEMV, byte-for-byte as before the seam.
func TestGLMDsaMatKernelRoutedExpertStaysHostWithoutCapability(t *testing.T) {
	// This is an exact path-parity witness, so select the same f32 Q4_K
	// numerical contract as the backend under comparison.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: &routedIncapableBackend{rec}, Q4K: true, halW: map[string]compute.Tensor{}}
	mat := s.glmDsaMatKernel()
	if _, ok := mat.(backendKernel); !ok {
		t.Fatalf("matKernel = %T, want backendKernel", mat)
	}
	if routedExpertKQuantActive(mat) {
		t.Fatal("routedExpertKQuantActive is true for a backend that declines the capability")
	}

	got := expertSwiGLU(m, 0, 0, x, mat)

	// Identical fixture, identical kernel TYPE, capability off => the pre-#5111 shape:
	// gate/up on the device, the Q6_K down_proj through the host k-quant GEMV, and the
	// SwiGLU host-side. This is the 2/0 the capable case above must NOT produce.
	if rec.matmuls != 2 || rec.swiglu != 0 {
		t.Fatalf("capability-off ops matmul=%d swiglu=%d, want 2/0", rec.matmuls, rec.swiglu)
	}
	if rec.uploads[compute.Q6_K] != 0 {
		t.Fatalf("capability-off staged the k-quant down_proj on the device: q6_k uploads=%d, want 0", rec.uploads[compute.Q6_K])
	}
	routedExpertParity(t, "capability-off host fallback", got, want)
}
