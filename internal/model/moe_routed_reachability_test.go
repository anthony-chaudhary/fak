package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestRoutedExpertReachableFromGLMDsaMatKernel is the #5111 regression witness.
//
// It exercises the REAL kernel-construction path — Session.glmDsaMatKernel, the
// single place decodeBandGLMDsa selects its matKernel — and asserts the routed
// resident-k-quant device path is reachable from it. The existing HAL tests
// (moe_expert_hal_test.go) hand-build sessionQ4KKernel{s: sess} directly and so
// CANNOT catch this class: before the fix, glmDsaMatKernel builds backendKernel
// (never sessionQ4KKernel), so routedExpertKQuantActive's old type-assert on
// sessionQ4KKernel always returned false and expert_parallel.go's host scalar
// batch always won on the live GLM-5.2 serve — the ~0.09-0.21 tok/s cause (#4784).
//
// The predicate is capability-keyed, so this witness needs no GPU: it proves the
// gate now opens for a routed-capable backend and stays closed for every other
// configuration (cpu-ref/Metal, host-only, and --cpu-offload-experts where the
// experts are intentionally host-pinned).
func TestRoutedExpertReachableFromGLMDsaMatKernel(t *testing.T) {
	cfg := expertHALTestConfig(64)
	m := NewSyntheticMoE(cfg)

	// expertHALRecordingBackend advertises SupportsRoutedExpertKQuant()=true; the
	// base cpu-ref backend (compute.Default) does not, which is why the recording
	// backend overrides it.
	capable := &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}

	cases := []struct {
		name string
		s    *Session
		want bool
	}{
		// THE fix: a routed-capable backend, experts resident (no offload) — the
		// live EP serve config (#4777) — now reaches the device routed path.
		{"capable-backend-resident", &Session{M: m, Backend: capable, halW: map[string]compute.Tensor{}}, true},
		// --cpu-offload-experts pins experts on host by design: splitKernel must NOT
		// claim the device routed capability.
		{"capable-backend-offload-host-pinned", &Session{M: m, Backend: capable, CPUOffloadExperts: true, halW: map[string]compute.Tensor{}}, false},
		// cpu-ref / Metal: capability absent, path byte-identical to before the seam.
		{"non-routed-cpuref-backend", &Session{M: m, Backend: compute.Default(), halW: map[string]compute.Tensor{}}, false},
		// Pure host path: residentKernel, no session capability.
		{"host-only-no-backend", &Session{M: m}, false},
	}

	for _, tc := range cases {
		mat := tc.s.glmDsaMatKernel()
		if got := routedExpertKQuantActive(mat); got != tc.want {
			t.Errorf("%s: routedExpertKQuantActive(glmDsaMatKernel()=%T) = %v, want %v", tc.name, mat, got, tc.want)
		}
	}
}
