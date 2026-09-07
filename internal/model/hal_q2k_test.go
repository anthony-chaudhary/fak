package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

type q2kUploadRecordingBackend struct {
	compute.Backend
	uploads []compute.Dtype
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
