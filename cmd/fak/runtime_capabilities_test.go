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

func TestRunRuntimeCapabilitiesPreferredBackendCanEmitLocalCPUDegradedReceipt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{
		"--prefer-backend", "vulkan",
		"--fallback-policy", runtimecap.FallbackPolicyLocalCPUDegrade,
		"--cpu-envelope", "qwen25-1p5b-q8-windows-amd64",
		"--goos", "windows",
		"--goarch", "amd64",
		"--host-total-ram-bytes", "17179869184",
		"--host-free-ram-bytes", "12884901888",
	}
	if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatalf("decode: %v\n%s", err, stdout.String())
	}
	if report.PreferredBackend == nil || report.PreferredBackend.Selected != "cpu-ref" {
		t.Fatalf("preferred backend = %+v", report.PreferredBackend)
	}
	if !report.ModelExecution.Runnable || report.ModelExecution.Mode != runtimecap.FallbackPolicyLocalCPUDegrade {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
	if report.ModelExecution.LocalCPUDegraded == nil || report.ModelExecution.LocalCPUDegraded.RequestedBackend != "vulkan" {
		t.Fatalf("receipt = %+v", report.ModelExecution.LocalCPUDegraded)
	}
}

func TestRunRuntimeCapabilitiesRejectsConflictingBackendFlags(t *testing.T) {
	var stdout, stderr bytes.Buffer
	code := runRuntimeCapabilities(&stdout, &stderr, []string{"--backend", "cpu-ref", "--prefer-backend", "vulkan"})
	if code != 2 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
}

func remoteCLIArgs() []string {
	return []string{
		"--prefer-backend", "vulkan",
		"--placement", "remote_allowed",
		"--remote-target", "research-west",
		"--authorize-remote-target", "research-west",
		"--remote-provider", "acme-cloud",
		"--remote-engine", "acme-inference",
		"--remote-model", "qwen3.8-4b",
		"--remote-endpoint-class", "managed_api",
		"--remote-region", "us-west",
		"--remote-state-boundary", "prompt,tool_results",
		"--remote-egress", "allowed",
		"--remote-credential-name", "acme-runtime-token",
		"--remote-credential-present",
		"--remote-tls", "verified",
		"--remote-proxy", "none",
		"--remote-reachability", "reachable",
		"--remote-timeout-ms", "30000",
		"--remote-retry-ceiling", "1",
		"--remote-budget-microusd", "250000",
		"--goos", "windows", "--goarch", "amd64",
	}
}

func TestRunRuntimeCapabilitiesEmitsRemotePlacementReceipt(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, remoteCLIArgs()); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if !report.ModelExecution.Runnable || report.ModelExecution.Mode != "remote" || report.ModelExecution.PayloadLoaded || report.ModelExecution.RemotePlacement == nil {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
	if report.ModelExecution.RemotePlacement.ControlPlaneOwner != "fak-local" || report.ModelExecution.RemotePlacement.Target != "research-west" {
		t.Fatalf("receipt = %+v", report.ModelExecution.RemotePlacement)
	}
}

func TestRunRuntimeCapabilitiesRemotePlacementDenialIsMachineReadable(t *testing.T) {
	args := remoteCLIArgs()
	for i := range args {
		if args[i] == "--remote-egress" {
			args[i+1] = "denied"
		}
	}
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var report runtimecap.Report
	if err := json.Unmarshal(stdout.Bytes(), &report); err != nil {
		t.Fatal(err)
	}
	if report.ModelExecution.Runnable || report.ModelExecution.PayloadLoaded || report.ModelExecution.Reason == nil || report.ModelExecution.Reason.Code != "remote_egress_denied" {
		t.Fatalf("execution = %+v", report.ModelExecution)
	}
}

func TestRunRuntimeCapabilitiesExecutionModeReceiptIsOptIn(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--receipt-schema", runtimecap.ExecutionModeReceiptSchema, "--backend", "definitely-not-a-backend"}
	if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 0 {
		t.Fatalf("code=%d stderr=%s", code, stderr.String())
	}
	var receipt runtimecap.ExecutionModeReceipt
	if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
		t.Fatal(err)
	}
	if receipt.Schema != runtimecap.ExecutionModeReceiptSchema || receipt.Mode != runtimecap.ExecutionModeRefused || !receipt.Valid {
		t.Fatalf("receipt = %+v", receipt)
	}
	if receipt.Identity.Engine != runtimecap.EvidenceUnknown || receipt.Witness.Status != runtimecap.EvidenceObserved {
		t.Fatalf("explicit evidence = %+v / %+v", receipt.Identity, receipt.Witness)
	}
}

func TestRunRuntimeCapabilitiesExecutionModeFixturesCoverSevenStates(t *testing.T) {
	modes := []string{
		runtimecap.ExecutionModeLocalAccelerator, runtimecap.ExecutionModeLocalCPUDegraded, runtimecap.ExecutionModeRemoteBacked,
		runtimecap.ExecutionModeOfflineControlMock, runtimecap.ExecutionModeOfflineModelBacked, runtimecap.ExecutionModeControlOnly, runtimecap.ExecutionModeRefused,
	}
	for _, mode := range modes {
		var stdout, stderr bytes.Buffer
		args := []string{"--receipt-schema", runtimecap.ExecutionModeReceiptSchema, "--execution-mode-fixture", mode}
		if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 0 {
			t.Fatalf("%s: code=%d stderr=%s", mode, code, stderr.String())
		}
		var receipt runtimecap.ExecutionModeReceipt
		if err := json.Unmarshal(stdout.Bytes(), &receipt); err != nil {
			t.Fatal(err)
		}
		if receipt.Mode != mode || !receipt.Valid || receipt.Witness.Certification != runtimecap.EvidenceUnwitnessed {
			t.Fatalf("%s = %+v", mode, receipt)
		}
	}
}

func TestRunRuntimeCapabilitiesRejectsFixtureOnLegacySchema(t *testing.T) {
	var stdout, stderr bytes.Buffer
	if code := runRuntimeCapabilities(&stdout, &stderr, []string{"--execution-mode-fixture", runtimecap.ExecutionModeRefused}); code != 2 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}

func TestRunRuntimeCapabilitiesRejectsUnknownExecutionModeFixture(t *testing.T) {
	var stdout, stderr bytes.Buffer
	args := []string{"--receipt-schema", runtimecap.ExecutionModeReceiptSchema, "--execution-mode-fixture", "not_a_mode"}
	if code := runRuntimeCapabilities(&stdout, &stderr, args); code != 2 {
		t.Fatalf("code=%d stdout=%s stderr=%s", code, stdout.String(), stderr.String())
	}
}
