package model

import (
	"encoding/binary"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type deviceExpertRouteBackend struct {
	compute.Backend
	matmul int
	swiglu int
}

func (b *deviceExpertRouteBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = true
	return c
}

func (b *deviceExpertRouteBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmul++
	return b.Backend.MatMul(w, x)
}

func (b *deviceExpertRouteBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}

// SupportsDeviceWeightDtype delegates to the wrapped backend (a pass-through recorder over
// compute.Default()/cpu-ref, which serves Q4_K/Q5_K/Q6_K in MatMul). This models a backend
// whose own dtype capability is that of the backend it forwards to.
func (b *deviceExpertRouteBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return compute.BackendSupportsDeviceWeightDtype(b.Backend, dt)
}

// deviceExpertRouteBackend deliberately does NOT advertise SupportsRoutedExpertKQuant:
// expertSwiGLUHAL declines for it, so expertSwiGLU reaches the incremental
// q4kExpertGateUpDownHAL seam — exactly the path this file witnesses.
func (b *deviceExpertRouteBackend) deviceMatmuls() int { return b.matmul }

// TestExpertSwiGLUUsesResidentDeviceQ4KGateUp is the original #4843 witness, now
// extended to the #13129 down projection: with all three projections resident Q4_K on
// a DeviceMemory backend, the incremental seam runs gate, up, SwiGLU AND down on the
// device (3 MatMuls / 1 SwiGLU). Before #13129 this same call ran 2/1 — the down
// projection was dispatched to the host because the seam only knew Q4_K gate/up.
func TestExpertSwiGLUUsesResidentDeviceQ4KGateUp(t *testing.T) {
	const h, i = 256, 256
	cfg := Config{HiddenSize: h, IntermediateSize: i, MoEIntermediateSize: i}
	m := &Model{Cfg: cfg, q4kw: map[string]*q4kTensor{}}
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		name := expertName(0, 0, proj)
		m.q4kw[name] = quantizeQ4KFromRaw(make([]byte, (h/256)*i*q4kBlockBytes), i, h)
	}
	be := &deviceExpertRouteBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	x := make([]float32, h)
	x[0] = 1

	got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
	if len(got) != h {
		t.Fatalf("expert output len=%d want %d", len(got), h)
	}
	if be.matmul != 3 || be.swiglu != 1 {
		t.Fatalf("device route calls: MatMul=%d SwiGLU=%d, want 3/1 (gate+up+down resident on device, #13129)", be.matmul, be.swiglu)
	}
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight", "down_proj.weight"} {
		if _, ok := s.halW["q4k:"+expertName(0, 0, proj)]; !ok {
			t.Errorf("resident device route did not stage %s", proj)
		}
	}
}

// TestExpertSwiGLUDeviceDownKQuantParity is the #13129 parity witness for the missing
// Q5_K/Q6_K routed-expert DOWN device kernel. Q4_K gate/up + a k-quant down projection
// is the q4_k_m mixture a live serve loads; the incremental seam must keep the fused
// SwiGLU intermediate resident and run the down GEMM on the backend, matching the host
// oracle and staging the raw k-quant bytes verbatim (no expanded F16).
//
// A recording DeviceMemory backend that does NOT advertise the routed capability is
// used deliberately: that is the exact configuration for which the incremental seam
// exists (a capable backend is already served by expertSwiGLUHAL, #5111).
func TestExpertSwiGLUDeviceDownKQuantParity(t *testing.T) {
	// Select the exact f32 Q4_K numerical contract so the oracle and the backend
	// agree to the same tolerance.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	for _, tc := range []struct {
		name string
		kind kQuantKind
	}{
		{name: "Q5_K", kind: kindQ5K},
		{name: "Q6_K", kind: kindQ6K},
	} {
		t.Run(tc.name, func(t *testing.T) {
			const H, I = 256, 256
			m, names := deviceDownExpertModel(t, tc.kind, H, I)
			x := make([]float32, H)
			for i := range x {
				x[i] = float32((i%19)-9) / 64
			}

			// Host oracle: the same expert through the pure host kernel.
			want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

			be := &deviceExpertRouteBackend{Backend: compute.Default()}
			s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
			rec := &expertHALRecordingBackend{Backend: be, uploads: map[compute.Dtype]int{}}
			s.Backend = rec

			got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})

			// (a) gate + up + down all executed on the backend.
			if rec.matmuls != 3 || rec.swiglu != 1 {
				t.Fatalf("device ops matmul=%d swiglu=%d, want 3/1 (#13129 must retain down on device)", rec.matmuls, rec.swiglu)
			}
			// (c) the down weight staged as raw k-quant, never expanded F16.
			desc, ok := LookupQuantDescriptor(tc.kind)
			if !ok {
				t.Fatalf("no descriptor for %s", tc.name)
			}
			if rec.uploads[desc.Dtype()] != 1 {
				t.Fatalf("down %s uploads=%d, want 1", tc.name, rec.uploads[desc.Dtype()])
			}
			if rec.uploads[compute.F16] != 0 {
				t.Fatalf("expanded F16 uploads=%d, want 0 — k-quant residency stays at checkpoint size", rec.uploads[compute.F16])
			}
			if _, ok := s.halW["kquant-raw:"+names[2]]; !ok {
				t.Fatalf("down_proj not staged verbatim by weightHALKQuant; halW holds %d entries", len(s.halW))
			}
			// (b) device result matches the host oracle within the shared tolerance.
			routedExpertParity(t, "incremental device down "+tc.name, got, want)

			// (d) a warm token reuses every resident weight — only the activation uploads.
			downUploads := rec.uploads[desc.Dtype()]
			expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
			if rec.matmuls != 6 || rec.swiglu != 2 {
				t.Fatalf("two-token device ops matmul=%d swiglu=%d, want 6/2", rec.matmuls, rec.swiglu)
			}
			if rec.uploads[desc.Dtype()] != downUploads {
				t.Fatalf("warm token re-uploaded the down weight: %s uploads %d -> %d", tc.name, downUploads, rec.uploads[desc.Dtype()])
			}
		})
	}
}

// deviceDownExpertModel builds one routed expert with resident Q4_K gate/up and a
// k-quant down projection of the requested kind — the q4_k_m mixture. The f32
// manifest copies of the expert projections are dropped so the quantized stores are
// the only residency, exactly the state after a GGUF load (residentKernel would
// otherwise read the f32 copies, making the parity compare two different weights).
func deviceDownExpertModel(t *testing.T, kind kQuantKind, H, I int) (*Model, [3]string) {
	t.Helper()
	cfg := expertHALTestConfig(H)
	cfg.IntermediateSize = I
	cfg.MoEIntermediateSize = I
	m := NewSyntheticMoE(cfg)
	names := [3]string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, name := range names[:2] {
		m.q4kw[name] = &q4kTensor{out: I, in: H, raw: routedExpertRawQ4K(t, I, H, 401+i), nblk: I / qkK}
	}
	m.kqw[names[2]] = deviceDownKQuant(t, kind, I, H, 403)
	for _, name := range names {
		delete(m.manifest, name)
	}
	return m, names
}

func deviceDownKQuant(t *testing.T, kind kQuantKind, out, in int, seed int64) *kQuantTensor {
	t.Helper()
	switch kind {
	case kindQ5K:
		return expertHALQ5KTensor(out, in, uint64(seed))
	case kindQ6K:
		return expertHALQ6KTensor(out, in, seed)
	default:
		t.Fatalf("unsupported fixture kind %s", kind)
		return nil
	}
}

// TestExpertSwiGLUDeviceDownStaysHostWithoutDeviceKernel is the #13129 negative control.
// Two independent declines must BOTH keep the host path byte-for-byte:
//
//  1. a backend with no DeviceMemory (cpu-ref) — the seam never admits; and
//  2. a down weight whose k-quant kind has no device HAL kernel (Q3_K).
//
// In both cases the down projection must run the host k-quant GEMV, no device MatMul
// may serve it, and the result must reproduce the host oracle exactly.
func TestExpertSwiGLUDeviceDownStaysHostWithoutDeviceKernel(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	const H, I = 256, 256

	t.Run("no-device-memory", func(t *testing.T) {
		m, _ := deviceDownExpertModel(t, kindQ6K, H, I)
		x := make([]float32, H)
		for i := range x {
			x[i] = float32((i%19)-9) / 64
		}
		want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

		rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		// A plain cpu-ref caps are used for admission, but wrap the recorder so any
		// device GEMM would be counted — proving none runs.
		s := &Session{M: m, Backend: rec, Q4K: true, halW: map[string]compute.Tensor{}}
		s.Backend = &nonDeviceSeamBackend{expertHALRecordingBackend: rec}

		got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
		if rec.matmuls != 0 || rec.swiglu != 0 {
			t.Fatalf("host fallback ran device ops matmul=%d swiglu=%d, want 0/0", rec.matmuls, rec.swiglu)
		}
		if rec.uploads[compute.Q6_K] != 0 {
			t.Fatalf("host fallback staged the k-quant down weight: q6_k uploads=%d, want 0", rec.uploads[compute.Q6_K])
		}
		routedExpertParity(t, "no-device-memory host fallback", got, want)
	})

	t.Run("kind-without-device-kernel", func(t *testing.T) {
		m, names := deviceDownExpertModel(t, kindQ6K, H, I)
		// Replace the down weight with a kind whose descriptor exposes no HAL kernel.
		if SupportsHALKQuant(kindQ3K) {
			t.Fatalf("fixture assumption broken: %s advertises HAL support", kindQ3K)
		}
		m.kqw[names[2]] = q3kFixtureTensor(I, H)
		x := make([]float32, H)
		for i := range x {
			x[i] = float32((i%19)-9) / 64
		}
		want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

		be := &deviceExpertRouteBackend{Backend: compute.Default()}
		s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
		got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})

		// gate+up still device-side through the narrow seam, but the down projection
		// has no device kernel, so it must stay on the host.
		if be.matmul != 2 || be.swiglu != 1 {
			t.Fatalf("seam ops matmul=%d swiglu=%d, want 2/1 (down must NOT run on device)", be.matmul, be.swiglu)
		}
		if _, ok := s.halW["kquant-raw:"+names[2]]; ok {
			t.Fatalf("down weight with no device kernel was staged on the backend")
		}
		routedExpertParity(t, "kind-without-device-kernel host fallback", got, want)
	})

	// #13129 adversarial follow-up: a Q5_K down projection on a Vulkan-like backend. The
	// MODEL descriptor admits Q5_K (SupportsHALKQuant is true) but the BACKEND's MatMul has
	// no Q5_K case (vulkan.go:1462 panics in default). Before the backend-dtype predicate
	// this path called MatMul(Q5_K) and panicked; now it must cleanly decline to the host,
	// while the SAME backend still runs Q6_K on device — proving the predicate discriminates
	// by dtype rather than blanket-refusing.
	t.Run("vulkan-q5k-down-has-no-kernel", func(t *testing.T) {
		m, names := deviceDownExpertModel(t, kindQ5K, H, I)
		if !SupportsHALKQuant(kindQ5K) {
			t.Fatalf("fixture assumption broken: %s does not advertise HAL support", kindQ5K)
		}
		x := make([]float32, H)
		for i := range x {
			x[i] = float32((i%19)-9) / 64
		}
		want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

		rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		be := &vulkanLikeSeamBackend{expertHALRecordingBackend: rec}
		s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}

		var got []float32
		func() {
			defer func() {
				if r := recover(); r != nil {
					t.Fatalf("Q5_K down on a Vulkan-like backend PANICKED instead of declining: %v", r)
				}
			}()
			got = expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})
		}()

		// gate + up ran on device through the narrow seam; the down projection did NOT.
		if be.matmuls != 2 || be.swiglu != 1 {
			t.Fatalf("seam ops matmul=%d swiglu=%d, want 2/1 (Q5_K down must NOT run on device)", be.matmuls, be.swiglu)
		}
		if be.uploads[compute.Q5_K] != 0 {
			t.Fatalf("Q5_K down weight was staged on the backend: q5_k uploads=%d, want 0", be.uploads[compute.Q5_K])
		}
		if _, ok := s.halW["kquant-raw:"+names[2]]; ok {
			t.Fatalf("Q5_K down weight with no device kernel was staged on the backend")
		}
		routedExpertParity(t, "vulkan-q5k-down host fallback", got, want)

		// The predicate was consulted for Q4_K (gate/up) and the resolved down dtype Q5_K.
		if len(be.weights) == 0 {
			t.Fatalf("backend dtype predicate was never consulted")
		}
	})

	// The discrimination arm: the same Vulkan-like backend, but a Q6_K down projection —
	// which its MatMul DOES handle — must run the down GEMM on device.
	t.Run("vulkan-q6k-down-runs-on-device", func(t *testing.T) {
		m, names := deviceDownExpertModel(t, kindQ6K, H, I)
		x := make([]float32, H)
		for i := range x {
			x[i] = float32((i%19)-9) / 64
		}
		want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

		rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
		be := &vulkanLikeSeamBackend{expertHALRecordingBackend: rec}
		s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}

		got := expertSwiGLU(m, 0, 0, x, sessionQ4KKernel{s: s})

		if be.matmuls != 3 || be.swiglu != 1 {
			t.Fatalf("device ops matmul=%d swiglu=%d, want 3/1 (Q6_K down must run on device)", be.matmuls, be.swiglu)
		}
		desc, ok := LookupQuantDescriptor(kindQ6K)
		if !ok {
			t.Fatalf("no descriptor for Q6_K")
		}
		if be.uploads[desc.Dtype()] != 1 {
			t.Fatalf("Q6_K down uploads=%d, want 1", be.uploads[desc.Dtype()])
		}
		if _, ok := s.halW["kquant-raw:"+names[2]]; !ok {
			t.Fatalf("Q6_K down_proj not staged verbatim by weightHALKQuant")
		}
		routedExpertParity(t, "vulkan-q6k-down device", got, want)
	})
}

// nonDeviceSeamBackend presents recorder counters but floors the Caps the seam keys on
// (DeviceMemory) and the routed-expert capability, isolating exactly what the device
// admission buys: a backend that can neither serve the full expertSwiGLUHAL route nor the
// incremental seam.
type nonDeviceSeamBackend struct{ *expertHALRecordingBackend }

func (b *nonDeviceSeamBackend) Caps() compute.Caps {
	c := b.expertHALRecordingBackend.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = false
	return c
}

func (nonDeviceSeamBackend) SupportsRoutedExpertKQuant() bool { return false }

// vulkanLikeSeamBackend models the one-Halo Vulkan target for the #13129 device seam:
// DeviceMemory=true, no SupportsRoutedExpertKQuant (so expertSwiGLUHAL declines and the
// incremental q4kExpertGateUpDownHAL seam is reached), and a MatMul whose resident dtype
// set matches vulkan.go's switch — Q4_K/Q6_K runnable, Q5_K NOT (vulkan.go's MatMul has no
// Q5_K case and panics in its default branch). MatMul itself forwards to cpu-ref so a
// genuinely-admitted dtype produces a real result to compare against the host oracle.
//
// This is the faithful negative fixture: it reports the Vulkan capability, not a blanket
// "no quantized support" and not a lie that Q5_K runs.
type vulkanLikeSeamBackend struct {
	*expertHALRecordingBackend
	weights []compute.Dtype
}

func (b *vulkanLikeSeamBackend) Caps() compute.Caps {
	c := b.expertHALRecordingBackend.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = true
	return c
}

func (b *vulkanLikeSeamBackend) SupportsRoutedExpertKQuant() bool { return false }

// SupportsDeviceWeightDtype reports vulkan's real MatMul dtype set: Q4_K and Q6_K are
// runnable, Q5_K is not. weightDtypes records every dtype probed so a test can assert the
// seam consulted the predicate for the actual down dtype.
func (b *vulkanLikeSeamBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	b.weights = append(b.weights, dt)
	switch dt {
	case compute.F32, compute.Q8_0, compute.Q4_K, compute.Q6_K, compute.Q2_K:
		return true
	default:
		return false
	}
}

// q3kFixtureTensor builds a resident Q3_K down projection (a kind whose descriptor has no
// device HAL kernel) with a nonzero f16 super-block scale, so the seam must decline it and
// the host path must still produce a signal (never stage, never panic).
func q3kFixtureTensor(out, in int) *kQuantTensor {
	nblk := in / qkK
	raw := make([]byte, out*nblk*q3kBlockBytes)
	lcgBytes(raw, 0x13129)
	for o := 0; o < out; o++ {
		for b := 0; b < nblk; b++ {
			blk := raw[(o*nblk+b)*q3kBlockBytes:]
			binary.LittleEndian.PutUint16(blk[q3kBlockBytes-2:], 0x2800) // d = 2^-6, keeps the oracle finite and nonzero
		}
	}
	return &kQuantTensor{out: out, in: in, nblk: nblk, kind: kindQ3K, raw: raw}
}
