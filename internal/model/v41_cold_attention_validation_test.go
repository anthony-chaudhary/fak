package model

import (
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// An installed clock runs between cold positions and may mutate the original
// rows, so cold prefill must not mint proofs while one is installed: every
// position falls back to the strict preparation with identical output.
// fak-test:runtime fast est=3s lane=default
func TestV41ColdAttentionProofDeclinesInstalledClock(t *testing.T) {
	if ref := compute.Default(); ref == nil || ref.Name() != "cpu-ref" || ref.Caps().DeviceMemory {
		t.Fatal("cold proof software witness requires cpu-ref")
	}
	prior := v41ResidentKVDisabled
	v41ResidentKVDisabled = false
	defer func() { v41ResidentKVDisabled = prior }()
	m := v41ReaderWidthFixture(t)
	newSession := func() (*Session, *v41ValidatedAttentionRecorder) {
		b := &v41ValidatedAttentionRecorder{v41DenseTestBackend: newV41DenseTestBackend(), live: map[compute.Buffer]bool{}}
		s := v41DenseTestSession(t, m, b)
		v41SharedAttentionOnly(s)
		return s, b
	}
	s, b := newSession()
	strict, control := newSession()
	strict.v41State().setSharedAttention(strict.v41State().sharedAttentionFunc())
	core := s.v41State().sharedAttentionBinding.native.core
	validated, consumed := core.validated, 0
	core.validated = func(layer int, request *v41SharedAttentionRequest, proof *v41SharedAttentionValidation) (v41SharedAttentionResult, error) {
		consumed++
		return validated(layer, request, proof)
	}
	ticks := int64(0)
	m.v41SetExpertTimeClock(func() int64 { ticks++; return ticks })
	defer m.v41SetExpertTimeClock(nil)
	scans := 0
	v41SharedAttentionPrepareScanProbe = func(elements int) { scans += elements }
	defer func() { v41SharedAttentionPrepareScanProbe = nil }()
	got := s.Prefill([]int{1, 2})
	if consumed != 0 || scans == 0 || ticks == 0 {
		t.Fatalf("installed clock: proofs=%d prepare scans=%d clock ticks=%d", consumed, scans, ticks)
	}
	v41SharedAttentionSessionBits(t, "cold with clock", got, strict.Prefill([]int{1, 2}))
	if len(b.calls) != len(control.calls) || b.uploads != control.uploads || len(b.live) != 0 ||
		!reflect.DeepEqual(captureV41ForwardSnapshot(s.v41Forward), captureV41ForwardSnapshot(strict.v41Forward)) {
		t.Fatal("declined cold proofs changed work, lifetime or state")
	}
}
