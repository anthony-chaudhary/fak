package ggufload

// deepseek41_expert_hal_integration_test.go — the acceptance witness for
// fak#13359 ("DeepSeek V4.1 mixed-quant loader reaches the shared expert HAL").
//
// The named integration gap: the resident/streamed mixed-quant DeepSeek-V4.1
// Q2_K(gate/up)+Q3_K(down) expert slate must reach the DEVICE expert gate/up HAL
// when it enters through the REAL production loader. The #13358 witness proved
// the device gate/up seam fires, but built its model by private resident-map
// assignment (m.kqw[...] = ...) — it never exercised the loader -> checkpoint
// tier -> model path. This test closes that: it loads a real deepseek41 GGUF
// through WeightSource.QuantModelQ4KProfileOptionsContext(WithStreamedExperts),
// so each routed expert is STAGED from the fused on-disk slab by the R5
// ExpertCheckpointTier exactly as a serve loads the published artifact, then
// drives one decode Step on a recording device backend.
//
// What is asserted (all from exported API + the recording backend, no new public
// testing hooks):
//   - the loader attached a checkpoint tier and indexed exactly the E*3 routed
//     projections (streamed, not resident);
//   - on a backend that serves Q2_K but not Q3_K, gate MatMul + up MatMul +
//     SwiGLU run ON THE DEVICE through the shared q4kExpertInputHAL seam, and the
//     device MatMul count is exactly 2 per routed pick — NOT 3, so the Q3_K down
//     contraction stayed on the host;
//   - the checkpoint tier actually FAULTED the activated experts (Reads grew);
//   - the device arm reproduces the host arm's logits within the f32 reduction
//     tolerance, so the streamed device seam is numerically the streamed host
//     triple;
//   - the negative control: a session with no device backend runs the host triple
//     with zero device ops, byte-identically to another host session.
//
// [SW-VERIFIED]: the device backend forwards to cpu-ref (compute.Default()); this
// is a deterministic software contract, NOT a physical GPU qualification. No
// [HW-WITNESSED] criterion is claimed.

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/model"
)

// ds41HALRecordingBackend forwards to a cpu-ref backend while advertising
// DeviceMemory and counting the device MatMul/SwiGLU ops. It serves Q2_K gate/up
// but NOT Q3_K, so the shared gate/up seam activates and the down projection must
// stay on the host — the exact DeepSeek-V4.1 mixed-quant contract.
type ds41HALRecordingBackend struct {
	compute.Backend
	// deviceMemory is what the wrapper advertises. The positive witness sets it
	// true; the negative control leaves it false so the seam must decline while
	// the wrapper still counts any op that (wrongly) runs.
	deviceMemory bool
	matmuls      int
	swiglu       int
}

func (b *ds41HALRecordingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	c.DeviceMemory = b.deviceMemory
	return c
}

func (b *ds41HALRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	b.matmuls++
	return b.Backend.MatMul(w, x)
}

func (b *ds41HALRecordingBackend) SwiGLU(g, u compute.Tensor) compute.Tensor {
	b.swiglu++
	return b.Backend.SwiGLU(g, u)
}

// SupportsDeviceWeightDtype reports the dtype set the one-Halo Vulkan target
// serves for this slate: Q2_K gate/up are runnable, Q3_K down is NOT (its
// descriptor has no HAL kernel), so the down projection stays on the host.
func (b *ds41HALRecordingBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	switch dt {
	case compute.F32, compute.Q8_0, compute.Q4_K, compute.Q6_K, compute.Q2_K:
		return true
	default:
		return false
	}
}

// ds41HALLoadStreamed loads the mixed-quant fixture through the REAL production
// loader with the streamed-experts option. The WeightSource is retained by the
// model (it owns the tier's readers), so it is closed only at test cleanup —
// after the model is done.
func ds41HALLoadStreamed(t *testing.T) *model.Model {
	t.Helper()
	path := writeDeepSeek41ExpertHALFile(t)
	ws, err := OpenWeights(path)
	if err != nil {
		t.Fatalf("OpenWeights: %v", err)
	}
	t.Cleanup(func() { _ = ws.Close() })
	m, err := ws.QuantModelQ4KProfileOptionsContext(t.Context(), nil, WithStreamedExperts(0))
	if err != nil {
		t.Fatalf("QuantModelQ4KProfileOptionsContext(WithStreamedExperts): %v", err)
	}
	if m == nil {
		t.Fatal("streamed-experts load returned a nil model")
	}
	return m
}

// TestDeepSeek41MixedQuantExpertHALIntegration is the positive witness, named
// exactly as the fak#13359 witness command (`go test ./internal/ggufload -run
// TestDeepSeek41MixedQuantExpertHALIntegration`).
func TestDeepSeek41MixedQuantExpertHALIntegration(t *testing.T) {
	m := ds41HALLoadStreamed(t)
	cfg := m.Cfg

	// (a) The loader attached a checkpoint tier that indexes exactly the routed
	// projections. A model with no tier — or a tier missing a slab — means the
	// streamed expert seam under test was never reached.
	stats := m.ExpertCheckpointStats()
	if !stats.Enabled {
		t.Fatal("the loader attached no checkpoint tier; the streamed expert seam was never reached")
	}
	wantIndexed := cfg.NumLayers * cfg.NumExperts * 3
	if stats.Tensors != wantIndexed {
		t.Fatalf("checkpoint tier indexed %d tensors, want %d (E*3 per MoE layer)", stats.Tensors, wantIndexed)
	}

	// Host oracle: the SAME model and tier, no device backend, so the callback is
	// nil and the historical host gate/up/down triple runs. It reads the same
	// streamed experts through the same tier, isolating the device seam (device
	// gate/up + host down) from the weight source.
	hostSess := &model.Session{M: m}
	want := hostSess.Step(1)
	afterHost := m.ExpertCheckpointStats()
	if afterHost.Reads == 0 {
		t.Fatal("the host oracle read no expert from the streamed tier; the fixture experts are not streamed")
	}

	// Device arm: a DeviceMemory backend whose MatMul serves Q2_K but not Q3_K.
	be := &ds41HALRecordingBackend{Backend: compute.Default(), deviceMemory: true}
	devSess := &model.Session{M: m, Backend: be}
	got := devSess.Step(1)
	afterDev := m.ExpertCheckpointStats()

	// (b) The device seam fired: exactly two MatMuls (gate, up) + one SwiGLU per
	// routed pick across the model's layers. Three MatMuls per pick would mean the
	// Q3_K down was (wrongly) staged device-resident.
	picks := cfg.NumExpertsPerTok * cfg.NumLayers
	if be.matmuls != 2*picks {
		t.Fatalf("device MatMul count = %d, want %d (gate+up per routed pick, never the Q3_K down)", be.matmuls, 2*picks)
	}
	if be.swiglu != picks {
		t.Fatalf("device SwiGLU count = %d, want %d (one fused SwiGLU per routed pick)", be.swiglu, picks)
	}

	// (c) The tier faulted the activated experts on the DEVICE arm too — the
	// device gate/up weights were served from the streamed checkpoint, not a
	// resident store.
	if afterDev.Reads <= afterHost.Reads {
		t.Fatalf("device step read %d expert faults, want more than the host step's %d (the streamed tier must serve the device gate/up from the checkpoint)", afterDev.Reads, afterHost.Reads)
	}

	// (d) Logit parity: the streamed device gate/up + host Q3_K down reproduces
	// the streamed host triple within the f32 reduction-order tolerance.
	if len(got) != len(want) {
		t.Fatalf("device arm returned %d logits, want %d", len(got), len(want))
	}
	for i := range got {
		bound := 1e-4 * float64(maxf32(1, absf32(want[i])))
		if d := math.Abs(float64(got[i] - want[i])); d > bound {
			t.Fatalf("logit[%d] = %v, want %v (|d|=%v > %v): the streamed device seam diverged from the streamed host triple", i, got[i], want[i], d, bound)
		}
	}
}

// TestDeepSeek41MixedQuantLoaderDeclinesWithoutDevice is the negative control:
// the SAME streamed model on a session with NO device backend issues zero device
// ops and runs the host triple byte-identically to another host session, proving
// the device seam is a strict addition.
func TestDeepSeek41MixedQuantLoaderDeclinesWithoutDevice(t *testing.T) {
	m := ds41HALLoadStreamed(t)

	// A non-device backend (cpu-ref) must not trigger the gate/up seam: the
	// recording wrapper (advertising DeviceMemory=false) counts zero device ops
	// and its output matches a plain host session.
	ref := compute.Default()
	if ref.Caps().DeviceMemory {
		t.Skip("cpu-ref unexpectedly advertises DeviceMemory; the negative control cannot be exercised")
	}
	probe := &ds41HALRecordingBackend{Backend: ref, deviceMemory: false}
	nonDevice := &model.Session{M: m, Backend: probe}

	base := (&model.Session{M: m}).Step(1)
	alt := nonDevice.Step(1)
	if probe.matmuls != 0 || probe.swiglu != 0 {
		t.Fatalf("non-device backend ran device ops matmul=%d swiglu=%d, want 0/0 (the seam must decline)", probe.matmuls, probe.swiglu)
	}
	if len(base) != len(alt) {
		t.Fatalf("non-device arm returned %d logits, want %d", len(alt), len(base))
	}
	for i := range base {
		if base[i] != alt[i] {
			t.Fatalf("non-device arm not byte-identical at %d: %v != %v (the device seam must be a strict addition)", i, base[i], alt[i])
		}
	}
}

func absf32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func maxf32(a, b float32) float32 {
	if a > b {
		return a
	}
	return b
}
