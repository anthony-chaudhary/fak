package runtimecap

import (
	"testing"

	"github.com/anthony-chaudhary/fak/internal/compute"
)

func TestProbeDefaultReportsPortableFakNativeCPUWithoutLoadingPayload(t *testing.T) {
	report := Probe(Options{GOOS: "linux", GOARCH: "amd64", BuildTags: []string{}})
	if !report.BinaryRunnable || !report.ControlPlaneRunnable {
		t.Fatalf("control surfaces not runnable: %+v", report)
	}
	if report.PortableCPU.Name != "cpu-ref" || !report.PortableCPU.Registered || report.PortableCPU.Tier != "scalar" {
		t.Fatalf("portable CPU = %+v", report.PortableCPU)
	}
	if !report.ModelExecution.Runnable || report.ModelExecution.Engine != "fak-native" || report.ModelExecution.Backend != "cpu-ref" {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
	if report.ModelExecution.PayloadLoaded || report.ModelExecution.PayloadCompatibility != "not_checked" {
		t.Fatalf("payload state = %+v", report.ModelExecution)
	}
}

func TestProbeRequestedBackendUsesExactLookupWithoutFallback(t *testing.T) {
	report := Probe(Options{RequestedBackend: "typo", GOOS: "linux", GOARCH: "amd64", BuildTags: []string{}, Backends: []compute.Backend{compute.Default()}})
	if report.RequestedBackend == nil || report.RequestedBackend.ExactMatch || report.RequestedBackend.Status != "unsupported" {
		t.Fatalf("request = %+v", report.RequestedBackend)
	}
	if report.ModelExecution.Runnable || report.ModelExecution.Backend != "" {
		t.Fatalf("silently selected backend: %+v", report.ModelExecution)
	}
	if got := report.RequestedBackend.Reason.Code; got != "unsupported_backend" {
		t.Fatalf("reason = %q", got)
	}
}

func TestProbeSeparatesNotCompiledUnavailableAndUnsupportedPlatforms(t *testing.T) {
	tests := []struct {
		name, backend, goos, goarch string
		tags                        []string
		code, status                string
	}{
		{"cuda not compiled", "cuda", "linux", "amd64", nil, "backend_not_compiled", "unavailable"},
		{"cuda runtime unavailable", "cuda", "linux", "amd64", []string{"cuda"}, "backend_unavailable", "unavailable"},
		{"metal unsupported", "metal", "linux", "amd64", nil, "unsupported_platform", "unsupported"},
		{"vulkan unsupported", "vulkan", "linux", "amd64", []string{"vulkan"}, "unsupported_platform", "unsupported"},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			report := Probe(Options{RequestedBackend: tt.backend, GOOS: tt.goos, GOARCH: tt.goarch, BuildTags: tt.tags, Backends: []compute.Backend{compute.Default()}})
			if report.RequestedBackend.Status != tt.status || report.RequestedBackend.Reason.Code != tt.code {
				t.Fatalf("request = %+v", report.RequestedBackend)
			}
		})
	}
}
