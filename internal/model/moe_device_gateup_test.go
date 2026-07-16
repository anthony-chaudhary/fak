package model

import (
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
	if be.matmul != 2 || be.swiglu != 1 {
		t.Fatalf("device route calls: MatMul=%d SwiGLU=%d, want 2/1 (gate+up resident; down remains k-quant fallback)", be.matmul, be.swiglu)
	}
	for _, proj := range []string{"gate_proj.weight", "up_proj.weight"} {
		if _, ok := s.halW["q4k:"+expertName(0, 0, proj)]; !ok {
			t.Errorf("resident device route did not stage %s", proj)
		}
	}
}
