package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// moe_device_engine_test.go — the #13128 witness that a routed expert whose bytes
// are resident on the DEVICE side of a --n-cpu-moe splitKernel executes there,
// instead of being forced to host CPU by construction.
//
// The defect: splitKernel carries no *Session, so expertSwiGLU resolved sess==nil
// and the already-correct device route sess.expertSwiGLUHAL was UNREACHABLE under
// the split. splitKernel.expertSession() now exposes the device side's session and
// splitDeviceExpertInput(name) is the per-encoding placement predicate the host
// guard consults. The invariant these tests pin: the split's OWN onHost predicate
// decides placement and is never overridden — a device-owned expert with a
// device-kernel encoding runs on the device, and every other split (host-pinned,
// sessionless device side, or a backend without the encoding) keeps the host arm
// byte-for-byte.

// capableDeviceExpertBackend is the recording backend whose ONE added capability
// (SupportsRoutedExpertKQuant) is the per-encoding device-kernel token #13128 keys
// on. It records the device ops so a test can PROVE the device route ran rather
// than infer it from throughput.
type capableDeviceExpertBackend struct {
	*expertHALRecordingBackend
}

func (b *capableDeviceExpertBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	c.UploadDtype = true
	return c
}

func (b *capableDeviceExpertBackend) SupportsRoutedExpertKQuant() bool { return true }

// incapableDeviceExpertBackend is the same recording backend with the ONE encoding
// capability switched off: no device kernel exists for the k-quant routed expert, so
// the split must decline and leave the host arm exactly as it was.
type incapableDeviceExpertBackend struct {
	*expertHALRecordingBackend
}

func (b *incapableDeviceExpertBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory = true
	c.UploadDtype = true
	return c
}

func (b *incapableDeviceExpertBackend) SupportsRoutedExpertKQuant() bool { return false }

// q4kmTwoMoELayerRoutedExpertModel is q4kmRoutedExpertModel extended to TWO MoE
// layers, with the q4_k_m mixture populated for layer 1 expert 0 (q4kw gate/up, kqw
// down) identically to layer 0. Two layers is the minimum for a GRADED spill to leave
// a routed expert DEVICE-owned: `spilledExpertLayers(n)` always spills the model's
// FIRST n MoE layers, so on a single-MoE-layer model every non-zero grade either
// spills layer 0 or (n >= len) is the ungraded hostOffloadWeight — neither opens the
// device path. With layers {0,1} graded at N=1, layer 0 spills to host and layer 1
// stays device-resident under the SHIPPED predicate.
//
// Layer 0's residency is preserved so the SPILLED arm (and every other test that
// probes layer 0) still sees the same bytes; layer 1's quantized stores are the ONLY
// residency for that layer's expert (the synthesized f32 copies are deleted), exactly
// as q4kmRoutedExpertModel does for layer 0.
func q4kmTwoMoELayerRoutedExpertModel(t *testing.T) (*Model, [3]string) {
	t.Helper()
	const H = 256
	cfg := expertHALTestConfig(H)
	cfg.NumLayers = 2
	m := NewSyntheticMoE(cfg)
	names := [3]string{
		expertName(1, 0, "gate_proj.weight"),
		expertName(1, 0, "up_proj.weight"),
		expertName(1, 0, "down_proj.weight"),
	}
	// Preserve layer 0's routed-expert residency from the single-layer fixture so both
	// MoE layers have real routed weights: MoEExpertLayers() must see {0,1}, otherwise
	// a graded spill of 1 would degenerate to the ungraded predicate (nil spilled set).
	layer0 := [3]string{
		expertName(0, 0, "gate_proj.weight"),
		expertName(0, 0, "up_proj.weight"),
		expertName(0, 0, "down_proj.weight"),
	}
	m.q4kw = map[string]*q4kTensor{}
	m.kqw = map[string]*kQuantTensor{}
	for i, name := range layer0[:2] {
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 401+i), nblk: 1}
	}
	m.kqw[layer0[2]] = expertHALQ6KTensor(H, H, 403)
	for i, name := range names[:2] {
		m.q4kw[name] = &q4kTensor{out: H, in: H, raw: routedExpertRawQ4K(t, H, H, 421+i), nblk: 1}
	}
	m.kqw[names[2]] = expertHALQ6KTensor(H, H, 423)
	for _, name := range append(append([]string{}, layer0[:]...), names[:]...) {
		delete(m.manifest, name)
	}
	return m, names
}

// TestMoEOffloadSplitDeviceExpertEngineReachable is the #13128 positive witness: a
// routed expert the split owns on the DEVICE (onHost==false) reaches the device
// engine when the backend advertises a device kernel for the encoding, and computes
// the same number the host oracle does.
//
// The placement here is NOT a test-local predicate: it is the SHIPPED production
// graded-spill predicate `Session.expertSpillOnHost()` (#5612) — the same function
// `Session.glmDsaMatKernel` hands to `splitKernel.onHost`. A two-MoE-layer model with
// `ExpertSpillLayers = 1` spills the first MoE layer (layer 0) to host and leaves
// layer 1's routed experts DEVICE-owned, so the probed expert at layer 1 is a real
// device-explosion candidate from a real production config. The witness cannot pass
// unless that shipped placement actually opens the device path.
func TestMoEOffloadSplitDeviceExpertEngineReachable(t *testing.T) {
	// The host oracle below runs the exact f32 Q4_K path; the device path must agree
	// with it, not with arm64's approximate activation-quantized SDOT path.
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmTwoMoELayerRoutedExpertModel(t)
	const deviceLayer = 1 // layer 1 is the non-spilled, device-owned MoE layer
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	// Host oracle: residentKernel is what glmDsaMatKernel returns with no Backend.
	want := expertSwiGLU(m, deviceLayer, 0, x, residentKernel{m})

	rec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	s := &Session{M: m, Backend: &capableDeviceExpertBackend{rec}, Q4K: true, halW: map[string]compute.Tensor{}}
	// The REAL production placement: a graded spill that spills layer 0 but NOT the
	// expert's own layer, so this routed expert is device-owned by the shipped
	// predicate — not a test-local stand-in.
	s.ExpertSpillLayers = 1 // spill the first MoE layer only
	onHost := s.expertSpillOnHost()
	// The --n-cpu-moe split shape (host=resident, device=the backend session) driven by
	// the shipped graded predicate: this routed expert is the DEVICE side's to run.
	mat := splitKernel{host: residentKernel{m}, device: backendKernel{s}, onHost: onHost}

	// The graded predicate must be honored in BOTH directions (see the assertion after
	// the device run for the complementary spilled-layer arm).
	gn := expertName(deviceLayer, 0, "gate_proj.weight")
	if onHost(expertName(0, 0, "gate_proj.weight")) != true {
		t.Fatal("graded spill did not route the SPILLED layer 0 expert to host")
	}
	if onHost(gn) {
		t.Fatalf("onHost(%q) = true for a non-spilled layer, want false: the graded predicate must leave layer %d device-owned", gn, deviceLayer)
	}
	if !mat.expertSession().supportsRoutedExpertKQuant() {
		t.Fatal("split did not resolve its device-side session")
	}
	if !mat.splitDeviceExpertInput(gn) {
		t.Fatalf("splitDeviceExpertInput(%q) = false, want true: a device-owned routed expert with a device-kernel encoding is a device candidate", gn)
	}

	got := expertSwiGLU(m, deviceLayer, 0, x, mat)

	if rec.matmuls != 3 || rec.swiglu != 1 {
		t.Fatalf("device expert ops matmul=%d swiglu=%d, want 3/1 (pre-#13128 this exact call ran 0/0: the split pinned the expert to host CPU)", rec.matmuls, rec.swiglu)
	}
	if s.q4kExpertStats.RoutedExpertsDeviceHAL != 1 {
		t.Fatalf("RoutedExpertsDeviceHAL=%d, want 1 (the device route must be admitted)", s.q4kExpertStats.RoutedExpertsDeviceHAL)
	}
	if s.q4kExpertStats.RoutedExpertsOffHAL != 0 {
		t.Fatalf("RoutedExpertsOffHAL=%d, want 0 (a device-eligible expert that ran device-side must not count as off-HAL)", s.q4kExpertStats.RoutedExpertsOffHAL)
	}
	routedExpertParity(t, "split device routed expert (Q4_K gate/up + Q6_K down)", got, want)

	// The complementary real-config arm: the SPILLED layer 0 expert is host-owned by the
	// same shipped predicate, so the split declines it for the device and keeps the host
	// arm — the graded placement is honored in both directions, not just the device one.
	spilledGn := expertName(0, 0, "gate_proj.weight")
	if !onHost(spilledGn) {
		t.Fatalf("onHost(%q) = false for the spilled layer, want true", spilledGn)
	}
	if mat.splitDeviceExpertInput(spilledGn) {
		t.Fatalf("splitDeviceExpertInput(%q) = true for a spilled host-owned expert, want false", spilledGn)
	}
}

// TestMoEOffloadSplitDeviceExpertEngineFailClosed is the #13128 NEGATIVE witness.
// Two independent declines must each leave the host-CPU arm byte-for-byte identical:
// a backend that carries no device kernel for the encoding, and a split whose own
// onHost predicate sends the expert to HOST. Neither may be overridden.
func TestMoEOffloadSplitDeviceExpertEngineFailClosed(t *testing.T) {
	setQ4KSDOTForTest(false)
	t.Cleanup(func() { setQ4KSDOTForTest(true) })

	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}
	gn := expertName(0, 0, "gate_proj.weight")

	// (1) Device-owned, but the backend advertises no device kernel for the encoding.
	// The DECLINE must come from the missing encoding capability, not from placement, so
	// this arm must use a placement that genuinely owns the expert on the DEVICE. The
	// real such placement is a GRADED spill whose set excludes this expert's layer
	// (ExpertSpillLayers==1 spills layer 0, so this layer-1 expert is device-owned); an
	// ungraded `hostOffloadWeight` would instead route every routed expert to HOST, and
	// would make the predicate decline on placement alone — passing the assertion for
	// the WRONG reason and never exercising the capability check.
	m2, _ := q4kmTwoMoELayerRoutedExpertModel(t)
	const deviceLayer = 1
	dGn := expertName(deviceLayer, 0, "gate_proj.weight")
	incapableSession := &Session{M: m2, Backend: &incapableDeviceExpertBackend{
		&expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}},
		Q4K: true, halW: map[string]compute.Tensor{}}
	incapableSession.ExpertSpillLayers = 1
	if incapableSession.expertSpillOnHost()(dGn) {
		t.Fatalf("fixture error: layer %d must be DEVICE-owned so the capability check is reached", deviceLayer)
	}
	incapable := splitKernel{
		host:   residentKernel{m2},
		device: backendKernel{incapableSession},
		onHost: incapableSession.expertSpillOnHost(),
	}
	if incapable.splitDeviceExpertInput(dGn) {
		t.Fatalf("splitDeviceExpertInput(%q) = true for a DEVICE-owned expert on a backend with no device kernel for the encoding, want false", dGn)
	}

	// (2) A device-kernel-capable backend, but the split itself routes the expert to HOST.
	// `hostOffloadWeight` IS the ungraded host-pinned placement: every routed expert goes to
	// host RAM by design, so this is the real `--n-cpu-moe` arm.
	hostRec := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	hostPinned := splitKernel{
		host:   residentKernel{m},
		device: backendKernel{&Session{M: m, Backend: &capableDeviceExpertBackend{hostRec}, Q4K: true, halW: map[string]compute.Tensor{}}},
		onHost: hostOffloadWeight,
	}
	if hostPinned.splitDeviceExpertInput(gn) {
		t.Fatalf("splitDeviceExpertInput(%q) = true for a host-pinned expert, want false: the split's own placement must not be overridden", gn)
	}

	// Each arm runs on its OWN fixture/layer, so the two decline reasons are genuinely
	// isolated and each is byte-for-byte checked against the plain host resident arm on
	// the SAME model+layer.
	for _, tc := range []struct {
		name  string
		mdl   *Model
		layer int
		mat   splitKernel
		rec   *expertHALRecordingBackend
	}{
		{"no-device-kernel-encoding", m2, deviceLayer, incapable, incapableSession.Backend.(*incapableDeviceExpertBackend).expertHALRecordingBackend},
		{"host-pinned-by-split", m, 0, hostPinned, hostRec},
	} {
		hostWant := expertSwiGLU(tc.mdl, tc.layer, 0, x, residentKernel{tc.mdl})
		got := expertSwiGLU(tc.mdl, tc.layer, 0, x, tc.mat)
		if len(got) != len(hostWant) {
			t.Fatalf("%s: output len=%d, want %d", tc.name, len(got), len(hostWant))
		}
		// Byte-for-byte: zero tolerance. The declined expert must run the EXACT host
		// resident arithmetic, not merely something close to it.
		for i := range hostWant {
			if got[i] != hostWant[i] {
				t.Fatalf("%s: output[%d]=%v, want %v (exact host parity broken — the split stole a host expert)", tc.name, i, got[i], hostWant[i])
			}
		}
		if tc.rec.swiglu != 0 {
			t.Fatalf("%s: device SwiGLU ran %d times, want 0 (the host arm must be unchanged)", tc.name, tc.rec.swiglu)
		}
	}
}

// TestMoEOffloadSplitDeclinesWhenDeviceOwnedButNoKernel pins the last fail-closed
// arm: a split whose device side is ANOTHER residentKernel (the honest degenerate
// config — offload requested but no backend) has no device session at all, so
// expertSession()==nil and the predicate declines. The result must equal the plain
// host resident path exactly.
func TestMoEOffloadSplitDeclinesWhenDeviceOwnedButNoKernel(t *testing.T) {
	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	mat := splitKernel{host: residentKernel{m}, device: residentKernel{m}, onHost: hostOffloadWeight}
	gn := expertName(0, 0, "gate_proj.weight")
	if mat.expertSession() != nil {
		t.Fatal("expertSession() must be nil when the device side is a sessionless residentKernel")
	}
	if mat.splitDeviceExpertInput(gn) {
		t.Fatalf("splitDeviceExpertInput(%q) = true with no device session, want false", gn)
	}

	got := expertSwiGLU(m, 0, 0, x, mat)
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})
	if len(got) != len(want) {
		t.Fatalf("output len=%d, want %d", len(got), len(want))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("output[%d]=%v, want %v (device-owned expert with no device kernel must keep the host arm byte-for-byte)", i, got[i], want[i])
		}
	}
}
