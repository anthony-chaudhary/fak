package runtimecap

import "testing"

func TestExecutionModeFixturesCoverClosedModes(t *testing.T) {
	modes := []string{
		ExecutionModeLocalAccelerator, ExecutionModeLocalCPUDegraded, ExecutionModeRemoteBacked,
		ExecutionModeOfflineControlMock, ExecutionModeOfflineModelBacked, ExecutionModeControlOnly, ExecutionModeRefused,
	}
	for _, mode := range modes {
		r := ExecutionModeFixture(mode)
		if r.Schema != ExecutionModeReceiptSchema || r.Mode != mode || !r.Valid {
			t.Fatalf("fixture %s = %+v", mode, r)
		}
		if r.Witness.Status != EvidenceFixture || r.Witness.Certification != EvidenceUnwitnessed {
			t.Fatalf("fixture %s masquerades as certified: %+v", mode, r.Witness)
		}
		if r.Status.Mode != r.Audit.Mode || r.Status.Health != r.Audit.Health || r.Status.Evidence != EvidenceUnwitnessed || r.Audit.Evidence != EvidenceUnwitnessed {
			t.Fatalf("fixture status/audit evidence for %s: status=%+v audit=%+v", mode, r.Status, r.Audit)
		}
		if r.ModelBacked && r.FallbackReason == nil {
			t.Fatalf("model-backed fixture %s omitted fallback semantics", mode)
		}
	}
}

func TestExecutionModeProjectionMatchesCPUDegradedReport(t *testing.T) {
	report := Probe(Options{
		PreferredBackend: "vulkan", CPUFallbackPolicy: FallbackPolicyLocalCPUDegrade,
		CPUEnvelope: "qwen25-1p5b-q8-windows-amd64", GOOS: "windows", GOARCH: "amd64",
		HostMemory: HostMemory{Known: true, TotalBytes: 16 << 30, FreeKnown: true, FreeBytes: 12 << 30}, HostMemoryOverride: true,
	})
	r := ExecutionModeReceiptFromReport(report)
	if !r.Valid || r.Mode != ExecutionModeLocalCPUDegraded || r.Health != ExecutionHealthDegraded {
		t.Fatalf("receipt = %+v", r)
	}
	if r.Identity.Engine != "fak-native" || r.Identity.Backend != "cpu-ref" || r.Identity.Model == EvidenceUnknown {
		t.Fatalf("identity = %+v", r.Identity)
	}
	if r.FallbackReason == nil || r.OperatingEnvelope == EvidenceUnknown {
		t.Fatalf("fallback/envelope = %+v / %q", r.FallbackReason, r.OperatingEnvelope)
	}
}

func TestExecutionModeProjectionMatchesRemoteReport(t *testing.T) {
	report := Probe(Options{
		PreferredBackend: "vulkan", PlacementMode: PlacementRemoteAllowed,
		GOOS: "linux", GOARCH: "amd64", RemoteTarget: "research-west", AuthorizedTarget: "research-west",
		RemoteProvider: "provider-a", RemoteEngine: "remote-engine", RemoteModel: "qwen3.8-remote",
		RemoteStateBoundary: []string{"prompt"}, RemoteEgress: "allowed", RemoteCredentialName: "runtime-token",
		RemoteCredentialPresent: true, RemoteTLS: "verified", RemoteProxy: "none", RemoteReachability: "reachable",
		RemoteTimeoutMilliseconds: 30000, RemoteRetryCeiling: 1, RemoteBudgetMicroUSD: 1000,
	})
	r := ExecutionModeReceiptFromReport(report)
	if !r.Valid || r.Mode != ExecutionModeRemoteBacked || r.Identity.Provider != "provider-a" || r.Identity.Model != "qwen3.8-remote" {
		t.Fatalf("receipt = %+v", r)
	}
	if r.ControlPlaneOwner != "fak-local" || r.FallbackReason == nil || r.Egress != "allowed" {
		t.Fatalf("boundary = %+v", r)
	}
}

func TestLocalModelModeRejectsSilentEngineSubstitutionWithoutCallerAssertion(t *testing.T) {
	r := NewExecutionModeReceipt(ExecutionModeOptions{
		Mode: ExecutionModeLocalAccelerator, Health: ExecutionHealthReady,
		Engine: "substituted-engine", Backend: "vulkan", Device: "fixture", Model: "qwen3.8",
		WitnessStatus: EvidenceObserved, WitnessSource: "test", Certification: EvidenceUnwitnessed,
	})
	if r.Valid || r.ValidationReason == nil || r.ValidationReason.Code != "native_engine_substitution" {
		t.Fatalf("substitution admitted without assertion: %+v", r)
	}
}

func TestNativePerformanceReceiptRejectsSilentEngineSubstitution(t *testing.T) {
	r := NewExecutionModeReceipt(ExecutionModeOptions{
		Mode: ExecutionModeLocalAccelerator, Health: ExecutionHealthReady,
		Engine: "substituted-engine", Backend: "vulkan", Device: "fixture", Model: "qwen3.8",
		NativePerformance: true, WitnessStatus: EvidenceObserved, WitnessSource: "test", Certification: EvidenceUnwitnessed,
	})
	if r.Valid || r.ValidationReason == nil || r.ValidationReason.Code != "native_engine_substitution" {
		t.Fatalf("substitution admitted: %+v", r)
	}
}

func TestModelBackedReceiptRequiresExactIdentity(t *testing.T) {
	r := NewExecutionModeReceipt(ExecutionModeOptions{Mode: ExecutionModeOfflineModelBacked, Health: ExecutionHealthOffline, Engine: "fak-native", Backend: "cpu-ref"})
	if r.Valid || r.ValidationReason == nil || r.ValidationReason.Code != "model_identity_unwitnessed" {
		t.Fatalf("missing model admitted: %+v", r)
	}
}

func TestInvalidReceiptKeepsStatusAndAuditConsistent(t *testing.T) {
	r := NewExecutionModeReceipt(ExecutionModeOptions{
		Mode: ExecutionModeLocalAccelerator, Health: ExecutionHealthReady,
		Engine: "substituted-engine", Backend: "vulkan", Device: "fixture", Model: "qwen3.8", NativePerformance: true,
	})
	if r.Health != ExecutionHealthRefused || r.Status.Health != r.Health || r.Audit.Health != r.Health {
		t.Fatalf("invalid receipt views diverged: %+v", r)
	}
}

func TestAcceleratorProjectionDoesNotInventDeviceIdentity(t *testing.T) {
	report := Report{Schema: Schema, GOOS: "linux", GOARCH: "amd64", BinaryRunnable: true, ControlPlaneRunnable: true, ModelExecution: Execution{Runnable: true, Engine: "fak-native", Backend: "vulkan", Mode: executionModeStandard}}
	r := ExecutionModeReceiptFromReport(report)
	if r.Identity.Device != EvidenceUnknown {
		t.Fatalf("device identity invented from backend: %+v", r.Identity)
	}
}

func TestObservedStatusAuditMismatchRefuses(t *testing.T) {
	r := NewExecutionModeReceipt(ExecutionModeOptions{
		Mode: ExecutionModeControlOnly, Health: ExecutionHealthControlOnly,
		StatusMode: ExecutionModeControlOnly, StatusHealth: ExecutionHealthControlOnly, StatusEvidence: EvidenceObserved,
		AuditMode: ExecutionModeRefused, AuditHealth: ExecutionHealthRefused, AuditEvidence: EvidenceObserved,
	})
	if r.Valid || r.ValidationReason == nil || r.ValidationReason.Code != "execution_view_mismatch" {
		t.Fatalf("observed mismatch admitted: %+v", r)
	}
}
