package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// moe_device_engine_test.go — #13128: the MoE offload split selects a COMPUTE ENGINE per expert
// encoding instead of assuming the host CPU for every routed expert.
//
// The three witnesses below are deliberately split by what they can prove:
//
//  1. TestDeviceKernelForExpertEncoding — the PER-ENCODING predicate itself (AC #1). Q4_K has a
//     device kernel; Q5_K/Q6_K down does not; a nil model / nil q4kw answers host.
//  2. TestSplitKernelDeviceEngineMatchesHostCPU — the parity witness (AC #3): a routed expert whose
//     bytes stay host-resident on the split executes gate/up + SwiGLU on the DEVICE backend and
//     matches the host-CPU oracle within the existing scale-relative bound. It also counts the
//     device ops, so a regression that silently keeps the expert on the host fails on the counter
//     even though the numeric comparison would still pass (host == host).
//  3. TestSplitKernelDeviceEngineFailClosed — the negative test (AC #4): with NO device kernel for
//     the encoding, or NO device backend, the host-CPU arm is byte-for-byte unchanged.
//
// The engine is witnessed through the REAL production seam: s.glmDsaMatKernel() builds the
// splitKernel the live forward threads (moe_offload.go:glmDsaMatKernel), not a hand-built kernel.

// TestDeviceKernelForExpertEncoding pins the per-encoding predicate: the split consults it rather
// than assuming host-CPU compute for every routed expert (AC #1).
func TestDeviceKernelForExpertEncoding(t *testing.T) {
	const H = 256
	m := NewSyntheticMoE(expertHALTestConfig(H))

	// Q4_K resident gate/up => a device kernel exists (the Q4_K MatMul + SwiGLU seam).
	m.q4kw = map[string]*q4kTensor{
		expertName(0, 0, "gate_proj.weight"): &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 901), nblk: 1},
		expertName(0, 0, "up_proj.weight"):   &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 902), nblk: 1},
	}
	// A Q6_K down projection: NO device kernel yet (fak#13129), so it must answer host.
	m.kqw = map[string]*kQuantTensor{
		expertName(0, 0, "down_proj.weight"): expertHALQ6KTensor(H, H, 903),
	}

	cases := []struct {
		desc string
		name string
		want expertEngine
	}{
		{"Q4_K gate has a device kernel", expertName(0, 0, "gate_proj.weight"), expertEngineDevice},
		{"Q4_K up has a device kernel", expertName(0, 0, "up_proj.weight"), expertEngineDevice},
		{"Q6_K down has no device kernel yet (fak#13129)", expertName(0, 0, "down_proj.weight"), expertEngineHost},
		{"an absent expert weight answers host", expertName(0, 7, "gate_proj.weight"), expertEngineHost},
	}
	for _, c := range cases {
		if got := deviceKernelForExpertEncoding(m, c.name); got != c.want {
			t.Errorf("%s: deviceKernelForExpertEncoding(%q) = %v, want %v", c.desc, c.name, got, c.want)
		}
	}

	// The predicate keys on the encoding the model actually carries: a nil model or a model with no
	// resident Q4_K store answers host rather than panicking.
	if got := deviceKernelForExpertEncoding(nil, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
		t.Errorf("nil model = %v, want expertEngineHost", got)
	}
	if got := deviceKernelForExpertEncoding(&Model{Cfg: m.Cfg}, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
		t.Errorf("model with nil q4kw = %v, want expertEngineHost", got)
	}

	// The ADMISSION wrapper additionally requires a device-capable session. Without a backend, even a
	// Q4_K weight answers host — the fail-closed default is structural.
	sNoBackend := &Session{M: m}
	if got := expertEngineForWeight(sNoBackend, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
		t.Errorf("session with no backend = %v, want expertEngineHost", got)
	}
	sBackend := &Session{M: m, Backend: compute.Default()}
	if got := expertEngineForWeight(sBackend, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
		t.Errorf("cpu-ref backend (no DeviceMemory cap) = %v, want expertEngineHost", got)
	}
	if got := expertEngineForWeight(nil, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
		t.Errorf("nil session = %v, want expertEngineHost", got)
	}
}

// splitDeviceEngineBackend is the fixture's KEY DISCRIMINATOR. It advertises the two compute.Caps
// the #13128 device engine needs (DeviceMemory + UploadDtype) but does NOT implement
// SupportsRoutedExpertKQuant, so the PRE-EXISTING expertSwiGLUHAL device route (#5111) declines and
// the new split engine branch is the ONLY device route that can fire. Without this the HAL route
// returns first and a non-discriminating test passes with the new branch deleted.
type splitDeviceEngineBackend struct {
	expertHALRecordingBackend
}

// SupportsRoutedExpertKQuant keeps the LEGACY capability OFF: expertHALRecordingBackend implements it
// returning true, so delegating here would re-arm the #5111 HAL route and mask the new branch.
func (b *splitDeviceEngineBackend) SupportsRoutedExpertKQuant() bool { return false }

// TestSplitKernelDeviceEngineMatchesHostCPU is the parity + execution witness. A routed expert whose
// bytes stay HOST-resident under the --n-cpu-moe split is executed by the DEVICE engine (gate/up +
// SwiGLU on the backend) and must reproduce the host-CPU result within the existing scale-relative
// tolerance (routedExpertParity, 1e-5), because host and device read the SAME f32/Q4_K bytes.
//
// The dispatch counters are the load-bearing non-numeric half: the device route returns the same
// numbers the host arm computes whenever the fixture weights agree, so a wiring regression that never
// reaches the device would still pass a numeric-only check. The backend declines the LEGACY HAL
// capability, so only the NEW split branch can produce the 2-MatMul/1-SwiGLU shape.
func TestSplitKernelDeviceEngineMatchesHostCPU(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	// Host-CPU oracle: the residentKernel path IS the pre-#13128 arm for the split's host side.
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	// A session that is BOTH --n-cpu-moe (CPUOffloadExperts) AND device-capable. This is the
	// Strix-Halo-shaped configuration #13128 targets: expert bytes host-resident in the unified
	// pool, a device backend present, and a device kernel for the Q4_K encoding.
	be := &splitDeviceEngineBackend{expertHALRecordingBackend: expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}}
	s := &Session{M: m, Backend: be, Q4K: true, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
	mat := s.glmDsaMatKernel()
	sk, ok := mat.(splitKernel)
	if !ok {
		t.Fatalf("glmDsaMatKernel = %T, want splitKernel for CPUOffloadExperts", mat)
	}
	// The split's device side must carry the session's backend, or no engine choice is reachable.
	// glmDsaMatKernel builds backendKernel for the device half (moe_offload.go:194-209).
	if _, ok := sk.device.(backendKernel); !ok {
		t.Fatalf("split device side = %T, want backendKernel (the kernel owning the backend)", sk.device)
	}
	// The engine predicate must ADMIT this expert's encoding on this session — the load-bearing gate.
	if got := expertEngineForWeight(s, expertName(0, 0, "gate_proj.weight")); got != expertEngineDevice {
		t.Fatalf("expertEngineForWeight(Q4_K gate on a device-capable offload session) = %v, want expertEngineDevice", got)
	}
	// Isolation proof: the legacy HAL route must DECLINE for this backend, so the only way the
	// backend sees ops is the new split engine branch.
	if _, ok := s.expertSwiGLUHAL(expertName(0, 0, "gate_proj.weight"), expertName(0, 0, "up_proj.weight"), expertName(0, 0, "down_proj.weight"), x); ok {
		t.Fatalf("fixture backend admitted expertSwiGLUHAL — the new split engine branch is not isolated")
	}

	got := expertSwiGLU(m, 0, 0, x, mat)

	// DISPATCH WITNESS through the SPLIT engine: gate/up on the device (2 MatMuls) + SwiGLU, and the
	// Q6_K down must NOT be uploaded — it stays on the split's host side until fak#13129.
	if be.matmuls != 2 || be.swiglu != 1 {
		t.Fatalf("split device engine ops matmul=%d swiglu=%d, want exactly 2/1 (down must stay host until fak#13129)", be.matmuls, be.swiglu)
	}
	if be.uploads[compute.Q6_K] != 0 {
		t.Fatalf("split device engine staged a Q6_K down projection (uploads=%d), want 0 (down stays host-resident)", be.uploads[compute.Q6_K])
	}
	// The routed expert's weights must be staged on the device under the Q4_K descriptor prefix.
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight"} {
		if _, ok := s.halW["q4k:"+expertName(0, 0, proj)]; !ok {
			t.Errorf("device engine did not stage %s on the backend", proj)
		}
	}
	// The reachability counter lands on the same session, so an operator surface can attribute the
	// expert to the device rather than the host.
	if st := s.Q4KExpertStats(); st.RoutedExpertsDeviceHAL != 1 || st.RoutedExpertsOffHAL != 0 {
		t.Errorf("counters device=%d off=%d, want 1/0 (a routed expert executed device-side)", st.RoutedExpertsDeviceHAL, st.RoutedExpertsOffHAL)
	}

	routedExpertParity(t, "split device engine (Q4_K gate/up + SwiGLU device; Q6_K down host)", got, want)
}

// TestSplitKernelDeviceEngineFailClosed is the negative test (AC #4): no device kernel for the
// encoding, or no device backend, MUST keep the current host-CPU split byte-for-byte.
func TestSplitKernelDeviceEngineFailClosed(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}
	hostOracle := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	t.Run("no device backend", func(t *testing.T) {
		// The degenerate split glmDsaMatKernel builds with offload on and NO backend: the engine
		// predicate must decline and the host arm must be BIT-EXACT (max|Δ|==0).
		s := &Session{M: m, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
		mat := s.glmDsaMatKernel()
		sk, ok := mat.(splitKernel)
		if !ok {
			t.Fatalf("glmDsaMatKernel = %T, want splitKernel", mat)
		}
		if _, ok := sk.device.(sessionQ4KKernel); ok {
			t.Fatalf("no-backend split device side unexpectedly carries a session")
		}
		if got := expertEngineForWeight(s, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
			t.Fatalf("no-backend engine = %v, want expertEngineHost", got)
		}
		got := expertSwiGLU(m, 0, 0, x, mat)
		for i := range got {
			if got[i] != hostOracle[i] {
				t.Fatalf("no-backend offload expert diverged at %d (%v != %v) — host arm not byte-for-byte", i, got[i], hostOracle[i])
			}
		}
	})

	t.Run("no device kernel for the encoding", func(t *testing.T) {
		// A device backend IS attached, but the model carries NO resident Q4_K representation for
		// this expert (the f32 manifest is the only copy). The per-encoding predicate answers host
		// and the host arm runs; the backend sees no expert MatMul.
		mF32 := NewSyntheticMoE(expertHALTestConfig(H))
		be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		s := &Session{M: mF32, Backend: be, Q4K: true, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
		mat := s.glmDsaMatKernel()
		if got := expertEngineForWeight(s, expertName(0, 0, "gate_proj.weight")); got != expertEngineHost {
			t.Fatalf("engine for an f32-only expert = %v, want expertEngineHost", got)
		}
		ref := expertSwiGLU(mF32, 0, 0, x, residentKernel{mF32})
		before := be.matmuls
		got := expertSwiGLU(mF32, 0, 0, x, mat)
		if be.matmuls != before {
			t.Errorf("device engine fired for an encoding with no device kernel: matmul %d -> %d", before, be.matmuls)
		}
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("no-device-kernel expert diverged at %d (%v != %v) — host arm not byte-for-byte", i, got[i], ref[i])
			}
		}
	})

	t.Run("residency is unchanged by the engine choice", func(t *testing.T) {
		// The engine choice moves COMPUTE, never residency: the split's host-side routing must be
		// identical with and without a device backend, so the byte accounting the load-time planner
		// sizes against is untouched.
		hostName := expertName(0, 0, "gate_proj.weight")
		sPlain := &Session{M: m, CPUOffloadExperts: true}
		sDevice := &Session{M: m, Backend: &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}, Q4K: true, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
		plain := sPlain.glmDsaMatKernel().(splitKernel)
		dev := sDevice.glmDsaMatKernel().(splitKernel)
		if plain.onHost(hostName) != dev.onHost(hostName) {
			t.Fatalf("the engine choice changed host-side residency routing for %q", hostName)
		}
		if !plain.onHost(hostName) {
			t.Fatalf("%q is expected to stay host-ROUTED under offload (bytes host-resident)", hostName)
		}
	})
}

// TestSplitDeviceEngineDoesNotUploadOffloadedExperts is the RESIDENCY regression witness. A split
// over a device-capable session must NOT resolve a Session for the expert path in a way that re-arms
// the legacy expertSwiGLUHAL route (#5111): that route uploads all THREE expert projections,
// including the down projection the operator deliberately moved off the device with --n-cpu-moe.
//
// The bug this pins: an earlier revision of #13128 resolved `sess` from the split's device side, so
// expertSwiGLUHAL fired on the offload path and staged host-offloaded expert bytes to device memory.
// The #13128 seam moves the COMPUTE (gate/up) only; residency stays exactly as the split decided.
func TestSplitDeviceEngineDoesNotUploadOffloadedExperts(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	// A backend that WOULD admit the legacy HAL route (SupportsRoutedExpertKQuant=true), so the only
	// thing keeping the expert bytes off the device is the offload split itself.
	be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: be, Q4K: true, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
	mat := s.glmDsaMatKernel()

	// Prove this backend COULD have taken the HAL route: excluding offload, expertSwiGLUHAL admits it.
	// This probe runs on its own backend so its uploads do not contaminate the measurement below.
	probeBE := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	capable := &Session{M: m, Backend: probeBE, Q4K: true, halW: map[string]compute.Tensor{}}
	if _, ok := capable.expertSwiGLUHAL(expertName(0, 0, "gate_proj.weight"), expertName(0, 0, "up_proj.weight"), expertName(0, 0, "down_proj.weight"), x); !ok {
		t.Fatalf("fixture backend does not admit expertSwiGLUHAL even without offload — the regression check would be vacuous")
	}
	if probeBE.uploads[compute.Q6_K] == 0 {
		t.Fatalf("HAL probe did not stage the Q6_K down — the isolation proof is not exercising the route it claims")
	}

	expertSwiGLU(m, 0, 0, x, mat)

	// The DOWN projection must never have been staged to the device: it is host-routed under offload.
	if be.uploads[compute.Q6_K] != 0 {
		t.Fatalf("offload path uploaded the Q6_K down projection=%d to the device — host-offloaded expert bytes reached device memory", be.uploads[compute.Q6_K])
	}
	for _, key := range []string{"kquant-raw:" + expertName(0, 0, "down_proj.weight")} {
		if _, ok := s.halW[key]; ok {
			t.Fatalf("offload path staged %q resident on the device — residency must stay host-routed", key)
		}
	}
	// The gate/up weights ARE staged: the #13128 engine intentionally computes them device-side.
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight"} {
		if _, ok := s.halW["q4k:"+expertName(0, 0, proj)]; !ok {
			t.Errorf("split device engine did not stage %s (the compute did not move)", proj)
		}
	}
}

// TestSplitDeviceEngineEdgeCases covers the guard edges the engine predicate and the split branch
// must both handle: a bias, a GELU activation, and a session that is not Q4_K-qualified. Each MUST
// leave the host arm byte-for-byte, because q4kExpertInputHAL cannot model them.
func TestSplitDeviceEngineEdgeCases(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	H := 256
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	newOffloadSession := func(t *testing.T, cfg Config, mutate func(*Model)) (*Model, *Session, *expertHALRecordingBackend) {
		t.Helper()
		m := NewSyntheticMoE(cfg)
		m.q4kw = map[string]*q4kTensor{}
		m.kqw = map[string]*kQuantTensor{}
		m.q4kw[expertName(0, 0, "gate_proj.weight")] = &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 411), nblk: 1}
		m.q4kw[expertName(0, 0, "up_proj.weight")] = &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 412), nblk: 1}
		m.kqw[expertName(0, 0, "down_proj.weight")] = expertHALQ6KTensor(H, H, 413)
		if mutate != nil {
			mutate(m)
		}
		be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		s := &Session{M: m, Backend: be, Q4K: true, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
		return m, s, be
	}

	t.Run("per-expert bias keeps the host arm", func(t *testing.T) {
		cfg := expertHALTestConfig(H)
		m, s, be := newOffloadSession(t, cfg, func(m *Model) {
			// A down_proj.bias is the case splitDeviceExpertInput itself does NOT check; the outer
			// expertSwiGLU guard must keep the whole expert on the host. m.has reads the manifest,
			// so presence alone is what the guard keys on.
			m.manifest[expertName(0, 0, "down_proj.bias")] = tensorMeta{Dtype: "F32", Shape: []int{H}, Nbytes: H * 4}
		})
		ref := expertSwiGLU(m, 0, 0, x, residentKernel{m})
		got := expertSwiGLU(m, 0, 0, x, s.glmDsaMatKernel())
		if be.matmuls != 0 || be.swiglu != 0 {
			t.Fatalf("bias expert took the device engine: matmul=%d swiglu=%d, want 0/0", be.matmuls, be.swiglu)
		}
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("bias expert diverged at %d — host arm not byte-for-byte", i)
			}
		}
	})

	t.Run("GELU activation keeps the host arm", func(t *testing.T) {
		cfg := expertHALTestConfig(H)
		cfg.ActGeluTanh = true
		m, s, be := newOffloadSession(t, cfg, nil)
		ref := expertSwiGLU(m, 0, 0, x, residentKernel{m})
		got := expertSwiGLU(m, 0, 0, x, s.glmDsaMatKernel())
		if be.swiglu != 0 {
			t.Fatalf("GELU expert took the device SwiGLU: swiglu=%d, want 0", be.swiglu)
		}
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("GELU expert diverged at %d — host arm not byte-for-byte", i)
			}
		}
	})

	t.Run("session without the Q4_K flag takes the host arm", func(t *testing.T) {
		cfg := expertHALTestConfig(H)
		m := NewSyntheticMoE(cfg)
		m.q4kw = map[string]*q4kTensor{
			expertName(0, 0, "gate_proj.weight"): &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 421), nblk: 1},
			expertName(0, 0, "up_proj.weight"):   &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 422), nblk: 1},
		}
		m.kqw = map[string]*kQuantTensor{expertName(0, 0, "down_proj.weight"): expertHALQ6KTensor(H, H, 423)}
		be := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		// Q4K deliberately false: the executor's useHALQ4KWeights gate declines, so the host arm runs.
		s := &Session{M: m, Backend: be, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}
		ref := expertSwiGLU(m, 0, 0, x, residentKernel{m})
		got := expertSwiGLU(m, 0, 0, x, s.glmDsaMatKernel())
		if be.matmuls != 0 || be.swiglu != 0 {
			t.Fatalf("s.Q4K=false expert took the device engine: matmul=%d swiglu=%d, want 0/0", be.matmuls, be.swiglu)
		}
		for i := range got {
			if got[i] != ref[i] {
				t.Fatalf("s.Q4K=false expert diverged at %d — host arm not byte-for-byte", i)
			}
		}
	})
}
