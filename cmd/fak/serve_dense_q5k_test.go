package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

type q5kCapBackend struct {
	compute.Backend
	deviceMemory, uploadDtype, q5k bool
}

func (b q5kCapBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = b.deviceMemory, b.uploadDtype
	return c
}
func (b q5kCapBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q5_K && b.q5k
}

type optionalQ5KCapBackend struct {
	q5kCapBackend
	native bool
}

func (b optionalQ5KCapBackend) SupportsQ5KMatMul() bool { return b.native }

// TestServeDenseKQuantOptionsQ5KAdmission pins when serve keeps dense Q5_K packed: only a
// device-memory backend with quantized upload and a Q5_K device matmul (and, when it exposes the
// optional execution probe, a passing one). No other residency arm moves with it.
// fak-test:runtime fast est=5ms lane=default
func TestServeDenseKQuantOptionsQ5KAdmission(t *testing.T) {
	full := q5kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q5k: true}
	cases := []struct {
		name    string
		backend compute.Backend
		want    bool
	}{
		{"common-dtype-contract", full, true},
		{"optional-native-contract", optionalQ5KCapBackend{q5kCapBackend: full, native: true}, true},
		{"nil", nil, false},
		{"cpu", compute.Default(), false},
		{"upload-only", q5kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true}, false},
		{"execution-only", q5kCapBackend{Backend: compute.Default(), deviceMemory: true, q5k: true}, false},
		{"host-memory", q5kCapBackend{Backend: compute.Default(), uploadDtype: true, q5k: true}, false},
		{"optional-bundle-missing", optionalQ5KCapBackend{q5kCapBackend: full}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(tc.backend))
			if effects.DenseQ5KResident != tc.want {
				t.Fatalf("DenseQ5KResident = %v, want %v", effects.DenseQ5KResident, tc.want)
			}
			if effects.DenseKQuantResident || effects.DenseQ2KResident || effects.DenseQ6KResident {
				t.Fatalf("Q5_K admission changed other residency: %+v", effects)
			}
		})
	}
}
