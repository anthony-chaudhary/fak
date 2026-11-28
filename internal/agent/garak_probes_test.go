package agent

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// fak-test:runtime fast est=50ms lane=default
func TestGarakProbeFloorAdmittedWhenAllRefuse(t *testing.T) {
	probes := DefaultGarakProbeFloor()
	report, err := RunGarakProbeFloor(context.Background(), probes, func(_ context.Context, _ string) (string, error) {
		return "I cannot comply with that request.", nil
	})
	if err != nil {
		t.Fatalf("RunGarakProbeFloor: %v", err)
	}
	if report.Schema != "fak-garak-floor/1" {
		t.Fatalf("schema = %q, want fak-garak-floor/1", report.Schema)
	}
	if !report.Admitted || report.PassedProbes != len(probes) || report.FailedProbes != 0 || report.PassRate != 1.0 {
		t.Fatalf("unexpected report: %+v", report)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGarakProbeFloorFailClosedOnLeakOrInconclusive(t *testing.T) {
	probes := DefaultGarakProbeFloor()
	// One probe leaks its marker -> report must fail admission.
	report, err := RunGarakProbeFloor(context.Background(), probes, func(_ context.Context, prompt string) (string, error) {
		if strings.Contains(prompt, "OVERRIDE_OK") {
			return "Sure, OVERRIDE_OK", nil
		}
		return "Refused.", nil
	})
	if err != nil {
		t.Fatalf("RunGarakProbeFloor: %v", err)
	}
	if report.Admitted || report.FailedProbes != 1 || report.PassedProbes != len(probes)-1 {
		t.Fatalf("expected 1 failure and Admitted=false, got %+v", report)
	}

	// Transport error produces Inconclusive -> also fails admission.
	incReport, err := RunGarakProbeFloor(context.Background(), probes[:1], func(_ context.Context, _ string) (string, error) {
		return "", errors.New("upstream timeout")
	})
	if err != nil {
		t.Fatalf("RunGarakProbeFloor inconclusive: %v", err)
	}
	if incReport.Admitted || incReport.InconclusiveProbes != 1 {
		t.Fatalf("expected InconclusiveProbes=1 and Admitted=false, got %+v", incReport)
	}
}

// fak-test:runtime fast est=50ms lane=default
func TestGarakProbeFloorRejectsInvalidContract(t *testing.T) {
	bad := []GarakProbe{{
		ID:       "bad",
		Category: ProbeCategory("unknown_category"),
		Prompts:  []string{"hi"},
		Detector: refusalOrAbsentDetector("x"),
	}}
	if _, err := RunGarakProbeFloor(context.Background(), bad, func(context.Context, string) (string, error) {
		return "ok", nil
	}); !errors.Is(err, errGarakInvalidProbe) {
		t.Fatalf("expected errGarakInvalidProbe, got %v", err)
	}
}
