package main

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
	"github.com/anthony-chaudhary/fak/internal/ggufload"
)

// q2kCapBackend is a minimal backend stub whose two capability axes (native packed
// Q2_K matmul and quantized upload) are independently settable, so the serve dense
// K-quant option builder can be exercised against every combination.
type q2kCapBackend struct {
	compute.Backend
	uploadDtype bool
	q2k         bool
}

func (b q2kCapBackend) Caps() compute.Caps {
	return compute.Caps{UploadDtype: b.uploadDtype}
}

func (b q2kCapBackend) SupportsQ2K() bool { return b.q2k }

func TestServeDenseKQuantOptionsNilBackendIsEmpty(t *testing.T) {
	if got := serveDenseKQuantOptions(nil); got != nil {
		t.Fatalf("nil backend options = %v, want nil", got)
	}
}

func TestServeDenseKQuantOptionsDisableRawResidencyForDeviceBackend(t *testing.T) {
	opts := serveDenseKQuantOptions(compute.Default())
	effects := ggufload.ApplyQ4KLoadOptions(opts)
	if effects.DenseKQuantResident {
		t.Fatal("device backend must route dense k-quant through Q8 (DenseKQuantResident must be false)")
	}
	if effects.DenseQ2KResident {
		t.Fatal("backend without native Q2_K must not enable dense Q2_K residency")
	}
}

func TestServeDenseKQuantOptionsRetainQ2KOnlyWhenBackendAdvertisesBoth(t *testing.T) {
	cases := []struct {
		name        string
		q2k         bool
		uploadDtype bool
		wantDenseK  bool
		wantQ2K     bool
	}{
		// #12757: the 246 GiB UD-Q2_K_XL artifact must stay packed on a backend that
		// advertises BOTH native packed Q2_K matmul and quantized upload. Expanding it
		// to Q8/F32 at load is exactly what makes a 128 GB Halo refuse the model.
		{name: "q2k+native+upload", q2k: true, uploadDtype: true, wantDenseK: false, wantQ2K: true},
		// Either capability alone is NOT enough: without native Q2_K the packed tensor
		// is unreachable at decode; without UploadDtype there is no quantized device
		// seam to hold it.
		{name: "q2k-only", q2k: true, uploadDtype: false, wantDenseK: false, wantQ2K: false},
		{name: "upload-only", q2k: false, uploadDtype: true, wantDenseK: false, wantQ2K: false},
		{name: "neither", q2k: false, uploadDtype: false, wantDenseK: false, wantQ2K: false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			backend := q2kCapBackend{Backend: compute.Default(), uploadDtype: tc.uploadDtype, q2k: tc.q2k}
			// Drive the options through the loader's own option-application path so the
			// assertion is on the EFFECT of the option list, not its slice length.
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(backend))
			if effects.DenseKQuantResident != tc.wantDenseK {
				t.Fatalf("DenseKQuantResident = %v, want %v", effects.DenseKQuantResident, tc.wantDenseK)
			}
			if effects.DenseQ2KResident != tc.wantQ2K {
				t.Fatalf("DenseQ2KResident = %v, want %v", effects.DenseQ2KResident, tc.wantQ2K)
			}
		})
	}
}

type q6kCapBackend struct {
	compute.Backend
	deviceMemory, uploadDtype, q6k bool
}

func (b q6kCapBackend) Caps() compute.Caps {
	c := b.Backend.Caps()
	c.DeviceMemory, c.UploadDtype = b.deviceMemory, b.uploadDtype
	return c
}
func (b q6kCapBackend) SupportsDeviceWeightDtype(dt compute.Dtype) bool {
	return dt == compute.Q6_K && b.q6k
}

type optionalQ6KCapBackend struct {
	q6kCapBackend
	native bool
}

func (b optionalQ6KCapBackend) SupportsQ6KMatMul() bool { return b.native }

func TestServeDenseKQuantOptionsEnableQ6KOnlyWhenBackendAdvertisesBoth(t *testing.T) {
	backends := []struct {
		name    string
		backend compute.Backend
	}{
		{name: "common-dtype-contract", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}},
		{name: "optional-native-contract", backend: optionalQ6KCapBackend{q6kCapBackend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}, native: true}},
	}
	for _, tc := range backends {
		t.Run(tc.name, func(t *testing.T) {
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(tc.backend))
			if !effects.DenseQ6KResident {
				t.Fatal("DenseQ6KResident = false, want true")
			}
			if effects.DenseKQuantResident || effects.DenseQ2KResident {
				t.Fatalf("Q6_K admission changed other residency: %+v", effects)
			}
		})
	}
}

func TestServeDenseKQuantOptionsDeclinesQ6KPartialCapability(t *testing.T) {
	backends := []struct {
		name    string
		backend compute.Backend
	}{
		{name: "nil", backend: nil},
		{name: "cpu", backend: compute.Default()},
		{name: "upload-only", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true}},
		{name: "execution-only", backend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, q6k: true}},
		{name: "host-memory", backend: q6kCapBackend{Backend: compute.Default(), uploadDtype: true, q6k: true}},
		{name: "optional-bundle-missing", backend: optionalQ6KCapBackend{q6kCapBackend: q6kCapBackend{Backend: compute.Default(), deviceMemory: true, uploadDtype: true, q6k: true}}},
	}
	for _, tc := range backends {
		t.Run(tc.name, func(t *testing.T) {
			effects := ggufload.ApplyQ4KLoadOptions(serveDenseKQuantOptions(tc.backend))
			if effects.DenseQ6KResident {
				t.Fatal("DenseQ6KResident = true, want false")
			}
			if effects.DenseKQuantResident || effects.DenseQ2KResident {
				t.Fatalf("declined Q6_K changed other residency: %+v", effects)
			}
		})
	}
}
