package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type q2kUploadRecordingBackend struct {
	compute.Backend
	uploads []compute.Dtype
}

type q2kArgmaxRecordingBackend struct {
	*q2kUploadRecordingBackend
	matMulArgmaxCalls int
	matMulArgmaxDtype compute.Dtype
	matMulCalls       int
	argmaxCalls       int
}

func (r *q2kUploadRecordingBackend) Caps() compute.Caps {
	c := r.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (r *q2kUploadRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	r.uploads = append(r.uploads, as)
	return r.Backend.Upload(t, as)
}

func (r *q2kArgmaxRecordingBackend) MatMul(w, x compute.Tensor) compute.Tensor {
	r.matMulCalls++
	return r.Backend.MatMul(w, x)
}

func (r *q2kArgmaxRecordingBackend) Argmax(x compute.Tensor) int {
	r.argmaxCalls++
	return r.Backend.Argmax(x)
}

func (r *q2kArgmaxRecordingBackend) MatMulArgmax(w, x compute.Tensor) int {
	r.matMulArgmaxCalls++
	r.matMulArgmaxDtype = w.Dtype
	logits := r.Backend.MatMul(w, x)
	defer r.Backend.Free(logits)
	return r.Backend.Argmax(logits)
}

func TestWeightHALKQuantQ2K(t *testing.T) {
	const out, in = 4, 256
	raw := make([]byte, out*(in/256)*84)
	for i := range raw {
		raw[i] = byte(i*13 + 7)
	}

	qt := quantizeKQuantFromRaw(raw, out, in, kindQ2K)
	rec := &q2kUploadRecordingBackend{Backend: compute.Default()}
	s := &Session{
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	tensor := s.weightHALKQuant("test_q2k", qt)
	if tensor.Dtype != compute.Q2_K {
		t.Fatalf("expected tensor dtype Q2_K, got %s", tensor.Dtype)
	}
	if len(rec.uploads) != 1 || rec.uploads[0] != compute.Q2_K {
		t.Fatalf("expected 1 upload with dtype Q2_K, got %v", rec.uploads)
	}

	// Verify caching on second call
	cached := s.weightHALKQuant("test_q2k", qt)
	if len(rec.uploads) != 1 {
		t.Fatalf("expected second call to be cached, got %d uploads", len(rec.uploads))
	}
	if cached.Dtype != compute.Q2_K {
		t.Fatalf("expected cached tensor dtype Q2_K, got %s", cached.Dtype)
	}
}

func TestMatWeightHALRoutesQ2K(t *testing.T) {
	const out, in = 4, 256
	raw := make([]byte, out*(in/256)*84)
	name := "model.layers.0.self_attn.q_proj.weight"

	m := &Model{
		kqw: map[string]*kQuantTensor{
			name: quantizeKQuantFromRaw(raw, out, in, kindQ2K),
		},
	}
	rec := &q2kUploadRecordingBackend{Backend: compute.Default()}
	s := &Session{
		M:       m,
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	tensor := s.matWeightHAL(name)
	if tensor.Dtype != compute.Q2_K {
		t.Fatalf("expected matWeightHAL to return Q2_K, got %s", tensor.Dtype)
	}
	if len(rec.uploads) != 1 || rec.uploads[0] != compute.Q2_K {
		t.Fatalf("expected 1 upload with Q2_K, got %v", rec.uploads)
	}
}

func TestLMHeadMatHALRoutesQ2K(t *testing.T) {
	const out, in = 4, 256
	raw := make([]byte, out*(in/256)*84)
	name := "lm_head.weight"

	m := &Model{
		kqw: map[string]*kQuantTensor{
			name: quantizeKQuantFromRaw(raw, out, in, kindQ2K),
		},
	}
	rec := &q2kUploadRecordingBackend{Backend: compute.Default()}
	s := &Session{
		M:       m,
		Backend: rec,
		halW:    map[string]compute.Tensor{},
	}

	tensor := s.lmHeadMatHAL()
	if tensor.Dtype != compute.Q2_K {
		t.Fatalf("expected lmHeadMatHAL to return Q2_K, got %s", tensor.Dtype)
	}
	if len(rec.uploads) != 1 || rec.uploads[0] != compute.Q2_K {
		t.Fatalf("expected 1 upload with Q2_K, got %v", rec.uploads)
	}
}

func TestHALQ2KFinalArgmaxStaysOnBackend(t *testing.T) {
	const out, in = 4, 256
	raw := make([]byte, out*(in/256)*84)
	head := compute.NewQ2K(compute.Default(), []int{out, in}, raw)
	norm := compute.NewF32(compute.Default(), []int{in}, make([]float32, in))
	x := compute.NewF32(compute.Default(), []int{in}, make([]float32, in))
	base := &q2kUploadRecordingBackend{Backend: compute.Default()}
	rec := &q2kArgmaxRecordingBackend{q2kUploadRecordingBackend: base}
	s := &Session{
		M: &Model{
			Cfg: Config{RMSNormEps: 1e-6},
			kqw: map[string]*kQuantTensor{
				"lm_head.weight": quantizeKQuantFromRaw(raw, out, in, kindQ2K),
			},
		},
		Backend: rec,
		halW: map[string]compute.Tensor{
			"model.norm.weight":         norm,
			"kquant-raw:lm_head.weight": head,
		},
	}

	_, next := s.halFinalLogits(x, halArgmax, false, false, func() {})
	if next != 0 {
		t.Fatalf("tied Q2_K logits chose index %d, want first index 0", next)
	}
	if rec.matMulArgmaxCalls != 1 || rec.matMulArgmaxDtype != compute.Q2_K {
		t.Fatalf("MatMulArgmax calls=%d dtype=%s, want one Q2_K call", rec.matMulArgmaxCalls, rec.matMulArgmaxDtype)
	}
	if rec.matMulCalls != 0 || rec.argmaxCalls != 0 {
		t.Fatalf("generic fallback ran: MatMul=%d Argmax=%d", rec.matMulCalls, rec.argmaxCalls)
	}
}
