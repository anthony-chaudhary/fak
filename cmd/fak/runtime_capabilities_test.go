package main

import (
	"bytes"
	"encoding/json"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/runtimecap"
)

func TestRunRuntimeCapabilitiesJSONWitness(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, nil); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if report.Schema != runtimecap.Schema || report.GOOS == "" || report.GOARCH == "" {
		t.Fatalf("identity fields = %+v", report)
	}
	if !report.BinaryRunnable || !report.ControlPlaneRunnable || report.ModelExecution.Engine != "fak-native" || report.ModelExecution.Backend != "cpu-ref" {
		t.Fatalf("runtime report = %+v", report)
	}
	if report.ModelExecution.PayloadLoaded {
		t.Fatal("runtime diagnostics loaded a model payload")
	}
}

func TestRunRuntimeCapabilitiesRequestedBackendRefusesSilentFallback(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, []string{"--backend", "definitely-not-a-backend"}); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.RequestedBackend == nil || report.RequestedBackend.ExactMatch || report.ModelExecution.Runnable || report.ModelExecution.Backend != "" {
		t.Fatalf("silent fallback: %+v", report)
	}
}
