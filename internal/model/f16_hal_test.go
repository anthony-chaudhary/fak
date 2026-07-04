package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// f16_hal_test.go — the off-GPU witness for the F16 Session-forward dtype select (issue #1481,
// epic #1476 C4). It has NO build tag, so it runs in the default non-cuda `go test` the way
// tf32_enable_test.go guards the TF32 seam: the routing decision is testable without a GPU. The
// device F16 GEMM itself (#484, cudaFP16CosineMin=0.997) is exercised by the cuda-tagged tests on
// hardware; here we pin only that Session.matWeightHAL/lmHeadMatHAL route an F16-tagged session's
// weight upload through compute.F16 (not Q8/Q4_K/F32).

// f16UploadRecordingBackend embeds a reference backend but advertises Caps().UploadDtype (so the
// Session F16 select fires) and records the dtype each weightHALStaged upload requests. The
// embedded cpu-ref ignores the narrowing numerically; the recorded `as` dtype is the routing
// witness. It deliberately does NOT implement UploadClass, so the f32 weightHAL path (which prefers
// UploadClass) is observable here only when it falls through to plain Upload.
type f16UploadRecordingBackend struct {
	compute.Backend
	uploads []compute.Dtype
}

func (r *f16UploadRecordingBackend) Caps() compute.Caps {
	c := r.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (r *f16UploadRecordingBackend) Upload(t compute.Tensor, as compute.Dtype) compute.Tensor {
	r.uploads = append(r.uploads, as)
	return r.Backend.Upload(t, as)
}

func recordedDtype(uploads []compute.Dtype, want compute.Dtype) bool {
	for _, d := range uploads {
		if d == want {
			return true
		}
	}
	return false
}

func f16SyntheticModel() *Model {
	return NewSynthetic(Config{
		HiddenSize:        8,
		NumLayers:         1,
		NumHeads:          2,
		NumKVHeads:        1,
		HeadDim:           4,
		IntermediateSize:  16,
		VocabSize:         16,
		RMSNormEps:        1e-5,
		RopeTheta:         10000,
		TieWordEmbeddings: true,
	})
}

// TestSessionF16SelectRoutesWeightUploadToF16 is the DoD witness for #1481: with s.F16 set on an
// UploadDtype-capable backend, matWeightHAL narrows the weight upload to compute.F16 and caches it
// under the f16 key — and without s.F16 it does not request F16 (the f32 path is unchanged).
func TestSessionF16SelectRoutesWeightUploadToF16(t *testing.T) {
	m := f16SyntheticModel()
	name := layerName(0, "self_attn.q_proj.weight")

	// F16-tagged session: the select must route through the F16 upload.
	beF16 := &f16UploadRecordingBackend{Backend: compute.Default()}
	s := &Session{M: m, Backend: beF16, F16: true, halW: map[string]compute.Tensor{}}
	_ = s.matWeightHAL(name)
	if !recordedDtype(beF16.uploads, compute.F16) {
		t.Fatalf("matWeightHAL with s.F16 requested upload dtypes %v, want one to be F16", beF16.uploads)
	}
	if _, ok := s.halW["f16:"+name]; !ok {
		t.Errorf("matWeightHAL(F16) did not cache the staged weight under f16:%s", name)
	}

	// Untagged session (same backend caps): must NOT narrow to F16 — the f32 weightHAL path.
	beBase := &f16UploadRecordingBackend{Backend: compute.Default()}
	base := &Session{M: m, Backend: beBase, halW: map[string]compute.Tensor{}}
	_ = base.matWeightHAL(name)
	if recordedDtype(beBase.uploads, compute.F16) {
		t.Fatalf("matWeightHAL without s.F16 requested F16 upload (uploads=%v); want the f32 path", beBase.uploads)
	}
	if _, ok := base.halW["f16:"+name]; ok {
		t.Errorf("matWeightHAL without s.F16 cached an f16-keyed weight; want the f32 path")
	}
}

// TestUseHALF16WeightsGatesOnUploadDtype pins the gate: the F16 select fires only for an
// F16-tagged session on a backend that honors the upload dtype. cpu-ref reports
// UploadDtype=false, so a Reference session stays on the bit-identical f32 path.
func TestUseHALF16WeightsGatesOnUploadDtype(t *testing.T) {
	m := f16SyntheticModel()

	// cpu-ref (UploadDtype=false): even with s.F16 set, the select is inert.
	ref := &Session{M: m, Backend: compute.Default(), F16: true, halW: map[string]compute.Tensor{}}
	if compute.Default().Caps().UploadDtype {
		t.Skip("reference backend unexpectedly reports UploadDtype; the inert-path assertion does not apply")
	}
	if ref.useHALF16Weights() {
		t.Fatalf("useHALF16Weights true on a non-UploadDtype backend; want inert (f32) on the Reference")
	}

	// UploadDtype backend but s.F16 unset: still inert (no F16 mode requested).
	up := &f16UploadRecordingBackend{Backend: compute.Default()}
	off := &Session{M: m, Backend: up, halW: map[string]compute.Tensor{}}
	if off.useHALF16Weights() {
		t.Fatalf("useHALF16Weights true with s.F16 unset")
	}

	// Both conditions met: the select fires.
	on := &Session{M: m, Backend: up, F16: true, halW: map[string]compute.Tensor{}}
	if !on.useHALF16Weights() {
		t.Fatalf("useHALF16Weights false with s.F16 set on an UploadDtype backend")
	}
}
