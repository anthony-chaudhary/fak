package model

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestRoutedExpertDeviceHALCountersDiscriminateTheLiveRoute is the instrumentation
// witness behind the #5111 hardware evidence gap.
//
// The reachability and execution tests next door prove the device routed path is
// reachable and correct on THIS host. They cannot travel: a GPU-server witness run
// reports throughput, and throughput alone cannot answer "did the device path run".
// That is exactly how #4843's resident routed-expert path stayed unreachable dead
// code on the live glm_moe_dsa decode for three weeks — the hardware evidence showed
// only tok/s, a number equally consistent with a slow device path and a host scalar
// batch, so nothing in it contradicted the dead path.
//
// RoutedExpertsDeviceHAL/RoutedExpertsOffHAL close that by recording the outcome at
// expertSwiGLU's single branch point. This test pins the property a remote witness
// actually leans on: the pair REPORTS THE ROUTE rather than a fixture constant. Model,
// residency, kernel type and activation are held byte-identical across the two cases;
// only supportsRoutedExpertKQuant flips. A counter that could not tell those apart
// would be worthless on the hardware run, which is the only place it has to work.
func TestRoutedExpertDeviceHALCountersDiscriminateTheLiveRoute(t *testing.T) {
	const tokens = 3
	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	newRec := func() *expertHALRecordingBackend {
		return &expertHALRecordingBackend{Backend: compute.Default(), uploads: map[compute.Dtype]int{}}
	}

	cases := []struct {
		name string
		// wrap adapts the recorder into the backend the session sees, so the recorder
		// stays reachable for the independent cross-check below.
		wrap       func(*expertHALRecordingBackend) compute.Backend
		wantDevice int64
		wantOff    int64
		wantFrac   float64
	}{
		// The live CUDA EP serve config: a routed-capable backend with experts resident.
		{"routed-capable", func(r *expertHALRecordingBackend) compute.Backend { return r }, tokens, 0, 1},
		// cpu-ref / Metal decline the capability, so every expert takes a non-HAL route.
		// This is the shape a post-fix GPU witness must NOT report.
		{"capability-off", func(r *expertHALRecordingBackend) compute.Backend { return &routedIncapableBackend{r} }, 0, tokens, 0},
	}

	for _, tc := range cases {
		rec := newRec()
		s := &Session{M: m, Backend: tc.wrap(rec), Q4K: true, halW: map[string]compute.Tensor{}}
		mat := s.glmDsaMatKernel()
		if _, ok := mat.(backendKernel); !ok {
			t.Fatalf("%s: matKernel = %T, want backendKernel (the kernel decodeBandGLMDsa builds)", tc.name, mat)
		}
		for i := 0; i < tokens; i++ {
			expertSwiGLU(m, 0, 0, x, mat)
		}

		got := s.Q4KExpertStats()
		if got.RoutedExpertsDeviceHAL != tc.wantDevice || got.RoutedExpertsOffHAL != tc.wantOff {
			t.Errorf("%s: device_hal=%d off_hal=%d, want %d/%d",
				tc.name, got.RoutedExpertsDeviceHAL, got.RoutedExpertsOffHAL, tc.wantDevice, tc.wantOff)
		}
		if f := got.RoutedExpertDeviceHALFraction(); f != tc.wantFrac {
			t.Errorf("%s: device-HAL fraction=%v, want %v", tc.name, f, tc.wantFrac)
		}
		// Every expert is accounted for exactly once, so a future route added without a
		// recorder shows up as a shortfall here rather than silently deflating the
		// fraction a hardware witness reads.
		if total := got.RoutedExpertsDeviceHAL + got.RoutedExpertsOffHAL; total != tokens {
			t.Errorf("%s: device_hal+off_hal=%d, want %d — an expert took an unrecorded route", tc.name, total, tokens)
		}
		// Cross-check against the backend's OWN op tally: the counter must track what
		// the device actually executed, not merely what the branch intended. One fused
		// SwiGLU per expert is the device route's signature.
		if int64(rec.swiglu) != got.RoutedExpertsDeviceHAL {
			t.Errorf("%s: backend recorded %d device SwiGLUs but the counter claims %d",
				tc.name, rec.swiglu, got.RoutedExpertsDeviceHAL)
		}
	}
}

// TestRoutedExpertDeviceHALCountersIgnoreSessionlessKernels pins the honest limit of
// the pair. residentKernel and splitKernel carry no session — the pure-host and
// --cpu-offload-experts regimes — so there is nothing to record on and both recorders
// no-op rather than panic. The counters therefore describe session-bearing kernels
// only; a zero/zero reading means "this route was never observed", NOT "the host path
// ran". A GPU witness reads them alongside the backend it configured, where the kernel
// is always backendKernel, so the distinction never bites there — but it is the reason
// the fraction is not a standalone claim about the whole decode.
func TestRoutedExpertDeviceHALCountersIgnoreSessionlessKernels(t *testing.T) {
	m, _ := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	s := &Session{M: m} // no Backend => glmDsaMatKernel returns the sessionless host kernel
	mat := s.glmDsaMatKernel()
	if _, ok := mat.(backendKernel); ok {
		t.Fatalf("matKernel = %T, want a sessionless host kernel for a Session with no Backend", mat)
	}
	expertSwiGLU(m, 0, 0, x, mat) // must not panic on the nil-session recorders

	got := s.Q4KExpertStats()
	if got.RoutedExpertsDeviceHAL != 0 || got.RoutedExpertsOffHAL != 0 {
		t.Errorf("sessionless kernel recorded device_hal=%d off_hal=%d, want 0/0",
			got.RoutedExpertsDeviceHAL, got.RoutedExpertsOffHAL)
	}
	if f := got.RoutedExpertDeviceHALFraction(); f != 0 {
		t.Errorf("empty-denominator fraction=%v, want 0 (not NaN)", f)
	}
}
