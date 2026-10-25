package model

import (
	"errors"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type qwen35QKNormOutputBackend struct {
	*qkNormRecordingBackend
	failAt int
	calls  int
}

func (b *qwen35QKNormOutputBackend) RMSNorm(x, weight compute.Tensor, eps float32) compute.Tensor {
	b.calls++
	if b.calls == b.failAt {
		return compute.Tensor{}
	}
	return b.qkNormRecordingBackend.RMSNorm(x, weight, eps)
}

func newQwen35QKNormOutputSession(t *testing.T, failAt int) (*Session, *qwen35QKNormOutputBackend, compute.Tensor, compute.Tensor, int) {
	t.Helper()
	cfg := qwen35HybridTestCfg()
	cfg.QKNorm = true
	m := NewSynthetic(cfg)
	fullLayer := 3
	tpInjectTensors(m, map[string]tpTensor{
		layerName(fullLayer, "self_attn.q_norm.weight"): {shape: []int{cfg.HeadDim}, vals: make([]float32, cfg.HeadDim)},
		layerName(fullLayer, "self_attn.k_norm.weight"): {shape: []int{cfg.HeadDim}, vals: make([]float32, cfg.HeadDim)},
	})
	be := &qwen35QKNormOutputBackend{qkNormRecordingBackend: newQKNormRecordingBackend(m), failAt: failAt}
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked: %v", err)
	}
	t.Cleanup(s.Close)
	q := be.Upload(compute.NewF32(compute.Default(), []int{cfg.NumHeads * cfg.HeadDim}, make([]float32, cfg.NumHeads*cfg.HeadDim)), compute.F32)
	k := be.Upload(compute.NewF32(compute.Default(), []int{cfg.NumKVHeads * cfg.HeadDim}, make([]float32, cfg.NumKVHeads*cfg.HeadDim)), compute.F32)
	return s, be, q, k, fullLayer
}

// fak-test:runtime fast est=20ms lane=default
func TestQwen35ResidentQKNormChainsResidentOutputs(t *testing.T) {
	s, be, q, k, layer := newQwen35QKNormOutputSession(t, 0)
	qNorm, kNorm, err := s.qwen35ResidentQKNorm(layer, q, k)
	if err != nil {
		t.Fatalf("qwen35ResidentQKNorm: %v", err)
	}
	if be.calls != 2 || qNorm.Buf() == nil || kNorm.Buf() == nil {
		t.Fatalf("RMSNorm chain calls=%d q=%v k=%v, want 2/non-nil/non-nil", be.calls, qNorm.Buf(), kNorm.Buf())
	}
	if be.readCalls != 0 || be.hostCalls != 0 {
		t.Fatalf("resident RMSNorm chain crossed host boundary: reads=%d hosts=%d", be.readCalls, be.hostCalls)
	}
}

// fak-test:runtime fast est=20ms lane=default
func TestQwen35ResidentQKNormRejectsMissingBuffers(t *testing.T) {
	for _, tc := range []struct {
		name   string
		failAt int
	}{
		{name: "query", failAt: 1},
		{name: "key", failAt: 2},
	} {
		t.Run(tc.name, func(t *testing.T) {
			s, be, q, k, layer := newQwen35QKNormOutputSession(t, tc.failAt)
			_, _, err := s.qwen35ResidentQKNorm(layer, q, k)
			var residentErr *Qwen35QKNormResidencyError
			if !errors.As(err, &residentErr) {
				t.Fatalf("error=%T, want typed Qwen35QKNormResidencyError", err)
			}
			if residentErr.Layer != layer {
				t.Fatalf("residency error layer=%d, want %d", residentErr.Layer, layer)
			}
			if be.calls != tc.failAt {
				t.Fatalf("RMSNorm calls=%d, want %d", be.calls, tc.failAt)
			}
			if be.readCalls != 0 || be.hostCalls != 0 {
				t.Fatalf("missing-buffer refusal crossed host boundary: reads=%d hosts=%d", be.readCalls, be.hostCalls)
			}
		})
	}
}

// TestQwen35ResidentQKNormCUDAAcceptsUnfencedOutputs is hardware-required. A
// non-CUDA build reports UNRUN by skipping; acceptance runs it with -tags cuda on
// a registered device so RMSNorm returns genuine asynchronous CUDA tensors.
// fak-test:runtime slow est=2s lane=hardware
func TestQwen35ResidentQKNormCUDAAcceptsUnfencedOutputs(t *testing.T) {
	be, ok := compute.Lookup("cuda")
	if !ok || be == nil || be.Name() != "cuda" {
		t.Skip("hardware-required: exact CUDA backend is not compiled and registered")
	}

	cfg := qwen35HybridTestCfg()
	cfg.QKNorm = true
	m := NewSynthetic(cfg)
	fullLayer := 3
	tpInjectTensors(m, map[string]tpTensor{
		layerName(fullLayer, "self_attn.q_norm.weight"): {shape: []int{cfg.HeadDim}, vals: make([]float32, cfg.HeadDim)},
		layerName(fullLayer, "self_attn.k_norm.weight"): {shape: []int{cfg.HeadDim}, vals: make([]float32, cfg.HeadDim)},
	})
	s, err := m.NewBackendSessionChecked(be)
	if err != nil {
		t.Fatalf("NewBackendSessionChecked(cuda): %v", err)
	}
	t.Cleanup(s.Close)

	q := uploadHostF32Class(be, []int{cfg.NumHeads * cfg.HeadDim}, make([]float32, cfg.NumHeads*cfg.HeadDim), compute.MemoryActivation, "qknorm-test-query")
	k := uploadHostF32Class(be, []int{cfg.NumKVHeads * cfg.HeadDim}, make([]float32, cfg.NumKVHeads*cfg.HeadDim), compute.MemoryActivation, "qknorm-test-key")
	t.Cleanup(func() {
		be.Free(q)
		be.Free(k)
	})

	qNorm, kNorm, err := s.qwen35ResidentQKNorm(fullLayer, q, k)
	if err != nil {
		t.Fatalf("qwen35ResidentQKNorm(cuda): %v", err)
	}
	t.Cleanup(func() {
		be.Free(qNorm)
		be.Free(kNorm)
	})
	if qNorm.Buf() == nil || kNorm.Buf() == nil {
		t.Fatalf("CUDA RMSNorm returned missing resident tensors: q=%v k=%v", qNorm.Buf(), kNorm.Buf())
	}
	if qNorm.Ready() || kNorm.Ready() {
		t.Fatalf("CUDA RMSNorm unexpectedly fenced before its same-stream consumer: q_ready=%t k_ready=%t", qNorm.Ready(), kNorm.Ready())
	}

	// Fence only after proving qwen35ResidentQKNorm accepted both live async results.
	if got := be.Read(kNorm); len(got) != cfg.NumKVHeads*cfg.HeadDim {
		t.Fatalf("fenced key result length=%d, want %d", len(got), cfg.NumKVHeads*cfg.HeadDim)
	}
}
