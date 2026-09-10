//go:build linux && rocm && cgo

package model

import (
	"math"
	"os"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func requireROCmModelBackend(t *testing.T) compute.Backend {
	t.Helper()
	be, ok := compute.Lookup("rocm")
	if !ok {
		if os.Getenv("FAK_ROCM_REQUIRE_DEVICE") == "1" {
			t.Fatal("FAK_ROCM_REQUIRE_DEVICE=1: native ROCm backend unavailable")
		}
		t.Skip("native ROCm backend unavailable")
	}
	return be
}

func TestHALROCmForwardMatchesNative(t *testing.T) {
	be := requireROCmModelBackend(t)
	if be.Class() != compute.Approx || !be.Caps().DeviceMemory {
		t.Fatalf("ROCm contract: class=%s caps=%+v", be.Class(), be.Caps())
	}
	cfg := Config{HiddenSize: 96, NumLayers: 4, NumHeads: 6, NumKVHeads: 2, HeadDim: 16,
		IntermediateSize: 256, VocabSize: 128, RMSNormEps: 1e-5, RopeTheta: 10000,
		TieWordEmbeddings: true, EOSTokenID: -1}
	m := NewSynthetic(cfg)
	prompt := []int{3, 9, 44, 1, 77, 22}
	native := m.NewSession().Prefill(prompt)
	devSession := m.NewBackendSession(be)
	defer devSession.Close()
	device := devSession.Prefill(prompt)
	if len(native) != cfg.VocabSize || len(device) != cfg.VocabSize {
		t.Fatalf("logit count native=%d rocm=%d want=%d", len(native), len(device), cfg.VocabSize)
	}
	var dot, nn, nd float64
	for i := range native {
		if math.IsNaN(float64(native[i])) || math.IsNaN(float64(device[i])) || math.IsInf(float64(native[i]), 0) || math.IsInf(float64(device[i]), 0) {
			t.Fatalf("non-finite logit at %d: native=%v rocm=%v", i, native[i], device[i])
		}
		dot += float64(native[i] * device[i])
		nn += float64(native[i] * native[i])
		nd += float64(device[i] * device[i])
	}
	if nn <= 0 || nd <= 0 || math.IsNaN(nn) || math.IsNaN(nd) || math.IsInf(nn, 0) || math.IsInf(nd, 0) {
		t.Fatalf("invalid logit norms native=%v rocm=%v", nn, nd)
	}
	cos := dot / (math.Sqrt(nn) * math.Sqrt(nd))
	if math.IsNaN(cos) || math.IsInf(cos, 0) || cos < 0.999 {
		t.Fatalf("prefill cosine %.7f < 0.999", cos)
	}
	want := m.NewSession().Generate(prompt, 10)
	gotSession := m.NewBackendSession(be)
	defer gotSession.Close()
	got := gotSession.Generate(prompt, 10)
	if len(got) != 10 || len(want) != len(got) {
		t.Fatalf("token count native=%d rocm=%d", len(want), len(got))
	}
	for i := range want {
		if got[i] != want[i] {
			t.Fatalf("token %d: native=%d rocm=%d; native=%v rocm=%v", i, want[i], got[i], want, got)
		}
	}
}
