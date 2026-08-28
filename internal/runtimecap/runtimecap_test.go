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
	if report.CPUFallbackPolicy != FallbackPolicyPinOrRefuse || len(report.SupportedCPUEnvelopes) == 0 {
		t.Fatalf("fallback surface = %+v", report)
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

func TestProbeExactBackendStillRefusesWhenCPUFallbackPolicyIsPresent(t *testing.T) {
	report := Probe(Options{
		RequestedBackend:   "vulkan",
		CPUFallbackPolicy:  FallbackPolicyLocalCPUDegrade,
		CPUEnvelope:        "qwen25-1p5b-q8-windows-amd64",
		GOOS:               "windows",
		GOARCH:             "amd64",
		BuildTags:          []string{},
		Backends:           []compute.Backend{compute.Default()},
		HostMemory:         HostMemory{Known: true, TotalBytes: 16 << 30, FreeBytes: 12 << 30, FreeKnown: true},
		HostMemoryOverride: true,
	})
	if report.RequestedBackend == nil || report.RequestedBackend.ExactMatch || report.RequestedBackend.Reason == nil {
		t.Fatalf("request = %+v", report.RequestedBackend)
	}
	if got := report.RequestedBackend.Reason.Code; got != "backend_not_compiled" {
		t.Fatalf("reason = %q", got)
	}
	if report.ModelExecution.Runnable || report.ModelExecution.Backend != "" || report.ModelExecution.LocalCPUDegraded != nil {
		t.Fatalf("exact request silently fell back: %+v", report.ModelExecution)
	}
}

func TestProbePreferredBackendCanSelectExplicitLocalCPUDegradedFallback(t *testing.T) {
	report := Probe(Options{
		PreferredBackend:   "vulkan",
		CPUFallbackPolicy:  FallbackPolicyLocalCPUDegrade,
		CPUEnvelope:        "qwen25-1p5b-q8-windows-amd64",
		GOOS:               "windows",
		GOARCH:             "amd64",
		BuildTags:          []string{},
		Backends:           []compute.Backend{compute.Default()},
		HostMemory:         HostMemory{Known: true, TotalBytes: 16 << 30, FreeBytes: 12 << 30, FreeKnown: true},
		HostMemoryOverride: true,
	})
	if report.PreferredBackend == nil || report.PreferredBackend.Reason == nil {
		t.Fatalf("preferred backend = %+v", report.PreferredBackend)
	}
	if report.PreferredBackend.Selected != "cpu-ref" {
		t.Fatalf("selected backend = %q, want cpu-ref", report.PreferredBackend.Selected)
	}
	if !report.ModelExecution.Runnable || report.ModelExecution.Backend != "cpu-ref" || report.ModelExecution.Mode != FallbackPolicyLocalCPUDegrade {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
	if report.ModelExecution.PayloadCompatibility != payloadCompatibilitySupported || report.ModelExecution.PayloadLoaded {
		t.Fatalf("payload state = %+v", report.ModelExecution)
	}
	receipt := report.ModelExecution.LocalCPUDegraded
	if receipt == nil || receipt.RequestedBackend != "vulkan" || receipt.SelectedBackend != "cpu-ref" {
		t.Fatalf("receipt = %+v", receipt)
	}
	if receipt.Reason == nil || receipt.Reason.Code != "backend_not_compiled" {
		t.Fatalf("receipt reason = %+v", receipt)
	}
}

func TestProbePreferredBackendRefusesOverBudgetCPUFallbackBeforePayloadLoad(t *testing.T) {
	report := Probe(Options{
		PreferredBackend:   "vulkan",
		CPUFallbackPolicy:  FallbackPolicyLocalCPUDegrade,
		CPUEnvelope:        "qwen25-1p5b-q8-windows-amd64",
		GOOS:               "windows",
		GOARCH:             "amd64",
		BuildTags:          []string{},
		Backends:           []compute.Backend{compute.Default()},
		HostMemory:         HostMemory{Known: true, TotalBytes: 16 << 30, FreeBytes: 2 << 30, FreeKnown: true},
		HostMemoryOverride: true,
	})
	if report.ModelExecution.Runnable {
		t.Fatalf("over-budget fallback must refuse: %+v", report.ModelExecution)
	}
	if report.ModelExecution.Backend != "cpu-ref" {
		t.Fatalf("fallback candidate backend = %q, want cpu-ref", report.ModelExecution.Backend)
	}
	if report.ModelExecution.PayloadCompatibility != payloadCompatibilityRefused || report.ModelExecution.PayloadLoaded {
		t.Fatalf("payload state = %+v", report.ModelExecution)
	}
	if report.ModelExecution.Reason == nil || report.ModelExecution.Reason.Code != "cpu_fallback_over_budget" {
		t.Fatalf("reason = %+v", report.ModelExecution.Reason)
	}
	if report.ModelExecution.LocalCPUDegraded != nil {
		t.Fatalf("refused fallback must not mint receipt: %+v", report.ModelExecution.LocalCPUDegraded)
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
