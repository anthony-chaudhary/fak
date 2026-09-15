package model

import (
	"bytes"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// tierRecordingBackend is the smallest device stand-in for the Q4_K single-residency witness: the
// cpu-ref oracle for every op, but with UploadDtype advertised (so useHALQ4KWeights selects the
// device Q4_K path) and a settable Tier() (so the test can present an integrated: APU or a
// discrete: GPU without hardware).
type tierRecordingBackend struct {
	compute.Backend
	tier string
}

func (b *tierRecordingBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.UploadDtype = true
	return c
}

func (b *tierRecordingBackend) Tier() string { return b.tier }

// TestWeightHALQ4KDropsHostCopyOnlyOnIntegratedSingleResidency proves the release gate: the host
// packed Q4_K bytes are dropped ONLY when the session is HAL-Q4K routed AND the device reports an
// integrated: (unified-memory) tier AND the FAK_Q4K_FREE_CPU knob is on. A cpu-ref / discrete /
// lazy tensor keeps its host copy, and the release is idempotent (a second call cannot resurrect
// a stale copy).
func TestWeightHALQ4KDropsHostCopyOnlyOnIntegratedSingleResidency(t *testing.T) {
	const out, in = 8, 256
	newResident := func() *q4kTensor {
		qt := quantizeQ4KFromRaw(buildRawQ4K(t, out, in, 61517), out, in)
		if len(qt.raw) == 0 {
			t.Fatal("fixture has no resident raw bytes")
		}
		return qt
	}
	newSession := func(be compute.Backend) *Session {
		m := &Model{q4kw: map[string]*q4kTensor{}}
		return &Session{Q4K: true, M: m, Backend: be, halW: map[string]compute.Tensor{}}
	}

	t.Run("integrated enabled drops", func(t *testing.T) {
		t.Setenv("FAK_Q4K_FREE_CPU", "1")
		qt := newResident()
		s := newSession(&tierRecordingBackend{Backend: compute.Default(), tier: "integrated:strix-halo"})
		if wt := s.weightHALQ4K("dense.weight", qt); !wt.Ready() {
			t.Fatal("upload returned an unready device tensor")
		}
		if len(qt.raw) != 0 {
			t.Fatalf("host raw retained (%d bytes) on integrated single residency", len(qt.raw))
		}
		// Idempotent: a second staging (cache hit) leaves nil raw and never re-materializes.
		s.weightHALQ4K("dense.weight", qt)
		if len(qt.raw) != 0 {
			t.Fatalf("second call resurrected %d host bytes", len(qt.raw))
		}
	})

	t.Run("integrated disabled keeps", func(t *testing.T) {
		qt := newResident()
		s := newSession(&tierRecordingBackend{Backend: compute.Default(), tier: "integrated:strix-halo"})
		s.weightHALQ4K("dense.weight", qt)
		if len(qt.raw) == 0 {
			t.Fatal("host raw dropped with the release knob off")
		}
	})

	t.Run("discrete enabled keeps", func(t *testing.T) {
		t.Setenv("FAK_Q4K_FREE_CPU", "1")
		qt := newResident()
		s := newSession(&tierRecordingBackend{Backend: compute.Default(), tier: "discrete:nvidia-4090"})
		s.weightHALQ4K("dense.weight", qt)
		if len(qt.raw) == 0 {
			t.Fatal("host raw dropped on a discrete device")
		}
	})

	t.Run("cpu-ref keeps", func(t *testing.T) {
		t.Setenv("FAK_Q4K_FREE_CPU", "1")
		qt := newResident()
		s := newSession(compute.Default())
		s.weightHALQ4K("dense.weight", qt)
		if len(qt.raw) == 0 {
			t.Fatal("host raw dropped on a cpu-ref backend")
		}
	})

	t.Run("lazy keeps", func(t *testing.T) {
		t.Setenv("FAK_Q4K_FREE_CPU", "1")
		raw := buildRawQ4K(t, out, in, 22)
		qt := &q4kTensor{out: out, in: in, nblk: in / qkK, lazy: &LazyQ4KRange{Reader: bytes.NewReader(raw), Bytes: len(raw)}}
		s := newSession(&tierRecordingBackend{Backend: compute.Default(), tier: "integrated:strix-halo"})
		s.weightHALQ4K("dense.weight", qt)
		if qt.lazy == nil {
			t.Fatal("lazy descriptor cleared by the release")
		}
	})
}
