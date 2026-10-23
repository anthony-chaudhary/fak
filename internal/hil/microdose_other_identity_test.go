//go:build !(darwin && arm64 && cgo)

package hil

import (
	"context"
	"strings"
	"testing"
)

// fak-test:runtime fast est=20ms lane=default
func TestMicroDosesHostReferenceIdentity(t *testing.T) {
	for _, key := range []string{"CUDA_VISIBLE_DEVICES", "FAK_CUDA", "VK_ICD_FILENAMES", "FAK_VULKAN"} {
		t.Setenv(key, "")
	}

	t.Run("configuration hints are not accelerator evidence", func(t *testing.T) {
		for _, tc := range []struct {
			key, value string
		}{
			{"CUDA_VISIBLE_DEVICES", "-1"},
			{"CUDA_VISIBLE_DEVICES", "not-a-device"},
			{"FAK_CUDA", "1"},
			{"VK_ICD_FILENAMES", "/nonexistent/invalid-icd.json"},
			{"FAK_VULKAN", "1"},
		} {
			t.Run(tc.key+"="+tc.value, func(t *testing.T) {
				t.Setenv(tc.key, tc.value)
				report := RunMicroDoses(context.Background())
				assertHostReferenceReport(t, report)
				if !report.AllPassed || len(report.MicroDoses) != len(DefaultMicroDoses()) {
					t.Fatalf("reference suite did not finish successfully: %+v", report)
				}
				key := "configuration_hint_" + strings.ToLower(tc.key)
				if got := report.Hardware.Details[key]; got != tc.value {
					t.Errorf("configuration hint %q = %q, want %q", key, got, tc.value)
				}
				if strings.Contains(report.Guidance, "Bias toward hardware testing satisfied") {
					t.Errorf("host reference claimed physical hardware success: %s", report.Guidance)
				}
			})
		}
	})

	t.Run("ordinary success is reference only", func(t *testing.T) {
		report := RunMicroDoses(context.Background())
		assertHostReferenceReport(t, report)
		if !report.AllPassed || len(report.MicroDoses) != len(DefaultMicroDoses()) {
			t.Fatalf("reference suite did not finish successfully: %+v", report)
		}
		if !strings.Contains(report.Guidance, "reference only") {
			t.Errorf("successful CPU suite must be explicit about reference-only evidence: %s", report.Guidance)
		}
	})

	t.Run("canceled suites never pass", func(t *testing.T) {
		for _, tc := range []struct {
			name           string
			allowedChecks  int
			completedDoses int
		}{
			{"before first dose", 0, 0},
			{"between doses", 1, 1},
			{"after final dose", 2, 2},
		} {
			t.Run(tc.name, func(t *testing.T) {
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				if tc.allowedChecks == 0 {
					cancel()
				}
				boundary := &cancelAtDoseBoundaryContext{Context: ctx, cancel: cancel, remainingChecks: tc.allowedChecks}
				report := RunMicroDoses(boundary, DoseLiveness, DoseNumericParity)
				assertHostReferenceReport(t, report)
				if report.AllPassed || len(report.MicroDoses) != tc.completedDoses {
					t.Fatalf("canceled suite = all_passed:%t results:%d; want false, %d", report.AllPassed, len(report.MicroDoses), tc.completedDoses)
				}
				if !strings.Contains(report.Guidance, "canceled") {
					t.Errorf("cancellation not distinguished from a passed or failed dose: %s", report.Guidance)
				}
			})
		}
	})

	t.Run("failed reference dose remains a failure", func(t *testing.T) {
		report := RunMicroDoses(context.Background(), DoseLiveness, DoseKind("unsupported"))
		assertHostReferenceReport(t, report)
		if report.AllPassed || len(report.MicroDoses) != 2 || !report.MicroDoses[0].Passed || report.MicroDoses[1].Passed {
			t.Fatalf("reference failure lost: %+v", report)
		}
		if !strings.Contains(report.Guidance, "host CPU reference micro-doses failed") {
			t.Errorf("failed CPU result must not be reported merely as unavailable hardware: %s", report.Guidance)
		}
	})
}

func assertHostReferenceReport(t *testing.T, report Report) {
	t.Helper()
	assertIdentity := func(hw HardwareInfo) {
		t.Helper()
		if hw.Kind != HardwareCPUSIMD || hw.PhysicalAvailable || hw.DeviceName != "Host CPU (Reference SIMD)" {
			t.Errorf("host execution has misleading hardware identity: %+v", hw)
		}
		if hw.Details["execution_mode"] != "host_cpu_reference" {
			t.Errorf("host reference execution mode missing: %+v", hw.Details)
		}
	}
	assertIdentity(report.Hardware)
	if report.BiasTowardHardware {
		t.Error("host reference execution cannot satisfy physical hardware bias")
	}
	for _, dose := range report.MicroDoses {
		assertIdentity(dose.Hardware)
	}
}

// Cancel synchronously at a suite boundary, without timing-dependent goroutines.
// RunMicroDoses is the sole caller of Err in these tests.
type cancelAtDoseBoundaryContext struct {
	context.Context
	cancel          context.CancelFunc
	remainingChecks int
}

func (ctx *cancelAtDoseBoundaryContext) Err() error {
	if ctx.remainingChecks == 0 {
		ctx.cancel()
	} else {
		ctx.remainingChecks--
	}
	return ctx.Context.Err()
}
