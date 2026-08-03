//go:build cuda

package model

import (
	"math"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

// TestCUDAGLMDsaRoutedExpertsExecuteOnDevice is the HARDWARE half of the #5111 witness:
// it is the one routed-expert test on this node that neither hand-builds the kernel nor
// hand-grants the capability.
//
// Every other CUDA expert test wraps the backend in cudaExpertRecordingBackend, which
// OVERRIDES SupportsRoutedExpertKQuant to return true, and drives sessionQ4KKernel{s}
// directly — a kernel decodeBandGLMDsa never constructs. Both substitutions are exactly
// the two things #5111 turned out to be about, so a cuda backend that stopped advertising
// the capability, or a glmDsaMatKernel that stopped reaching the device route, would leave
// every one of those tests green while the live GLM-5.2 serve ran its experts on the host.
// This test removes both substitutions: the raw registered backend, and the kernel the
// live decode actually builds.
//
// It then asserts execution rather than inferring it. RoutedExpertsDeviceHAL is recorded
// at expertSwiGLU's single branch point, so a full device-HAL fraction is positive proof
// the resident routed path ran on this GPU — the evidence the 2026-07-15 GPU-server run
// could not produce, because tok/s alone is equally consistent with a slow device path and
// a host scalar batch. That ambiguity is what let #4843's routed path sit unreachable for
// three weeks with a hardware witness already on file.
//
// The fixture is the q4_k_m MIXTURE a real GLM-5.2 checkpoint loads — Q4_K gate/up out of
// q4kw, raw Q6_K down out of kqw — because that split is what #5111 restored, and a
// Q4_K-only fixture would pass with the kqw seam still broken.
//
// A skip (no reachable GPU) is NOT a pass; set FAK_CUDA_EXPERT_REQUIRED=1 on an acceptance
// node to make the skip a failure. Run via tools/dgx_witness_run.sh.
func TestCUDAGLMDsaRoutedExpertsExecuteOnDevice(t *testing.T) {
	be := cudaExpertBackend(t)

	// The capability on the REAL backend, unwrapped. supportsRoutedExpertKQuant is the
	// single predicate the live decode keys on (hal.go); if the registered cuda backend
	// stops implementing it, routed experts silently return to the host GEMV and only
	// throughput would ever show it.
	routed, ok := be.(interface{ SupportsRoutedExpertKQuant() bool })
	if !ok || !routed.SupportsRoutedExpertKQuant() {
		t.Fatalf("registered cuda backend does not advertise SupportsRoutedExpertKQuant (implements=%v) — routed GLM-5.2 experts would run host-resident, the #5111 dead-path condition", ok)
	}

	const tokens = 4
	m, names := q4kmRoutedExpertModel(t)
	H := m.Cfg.HiddenSize
	x := make([]float32, H)
	for i := range x {
		x[i] = float32((i%19)-9) / 64
	}

	// Host oracle: residentKernel is what glmDsaMatKernel returns with no Backend, so this
	// is the same expert computed entirely on the host from the same resident bytes.
	want := expertSwiGLU(m, 0, 0, x, residentKernel{m})

	s := &Session{M: m, Backend: be, Q4K: true, halW: map[string]compute.Tensor{}}
	defer s.Close()
	mat := s.glmDsaMatKernel()
	if _, ok := mat.(backendKernel); !ok {
		t.Fatalf("live decode matKernel = %T, want backendKernel (the kernel decodeBandGLMDsa builds)", mat)
	}

	var got []float32
	for i := 0; i < tokens; i++ {
		got = expertSwiGLU(m, 0, 0, x, mat)
	}

	// Both halves of the mixture must be device-resident at checkpoint size. A staged
	// weight that fell back to an expanded copy would still compute the right number, so
	// the residency keys are checked rather than inferred from the output.
	for _, key := range []string{"q4k:" + names[0], "q4k:" + names[1], "kquant-raw:" + names[2]} {
		if _, ok := s.halW[key]; !ok {
			t.Fatalf("expert weight %q is not device-resident; halW holds %d entries", key, len(s.halW))
		}
	}

	st := s.Q4KExpertStats()
	if st.RoutedExpertsDeviceHAL != tokens || st.RoutedExpertsOffHAL != 0 {
		t.Fatalf("routed experts on device: device_hal=%d off_hal=%d, want %d/0 — the resident routed path did not run on this GPU (#5111 dead-path condition; tok/s alone would not have shown it)",
			st.RoutedExpertsDeviceHAL, st.RoutedExpertsOffHAL, tokens)
	}

	// Parity against the host oracle, at the same cosine floor the other CUDA expert
	// witness uses: the device reassociates the same f32 reduction over the same bytes.
	cos := glmDsaCosine(got, want)
	const floor = 0.997
	if math.IsNaN(cos) || math.IsInf(cos, 0) || cos < floor {
		t.Fatalf("routed expert device/host cosine %.8f < %.3f", cos, floor)
	}

	t.Logf("GLM-MoE-DSA routed experts through the LIVE decode kernel (glmDsaMatKernel -> backendKernel -> expertSwiGLUHAL; q4_k_m mixture: Q4_K gate/up + raw Q6_K down) on cuda backend: cosine=%.6f routed_experts_device_hal=%d routed_experts_off_hal=%d device_hal_fraction=%.6f tier=%s class=%s",
		cos, st.RoutedExpertsDeviceHAL, st.RoutedExpertsOffHAL, st.RoutedExpertDeviceHALFraction(), be.Tier(), be.Class())
}
