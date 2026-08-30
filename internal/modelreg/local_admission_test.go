package modelreg

import (
	"encoding/json"
	"errors"
	"reflect"
	"strings"
	"testing"
)

const admissionGiB = int64(1 << 30)

func validLocalAdmissionRequest() LocalAdmissionRequest {
	sha := strings.Repeat("a", 64)
	return LocalAdmissionRequest{
		Declaration: LocalAdmissionDeclaration{
			ModelID: "tiny-gguf", ArtifactSHA256: sha, ArtifactBytes: admissionGiB,
			RuntimeID: "llama.cpp", RuntimeVersion: "b1234", RequiredRuntimeCapability: "gguf",
			Requested: LocalDeviceTarget{DeviceKind: LocalDeviceCPU, Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: 2 * admissionGiB}},
		},
		Artifact: LocalVerifiedArtifactFacts{Path: "/cache/aaaaaaaa.gguf", SHA256: sha, Bytes: admissionGiB, Verified: true},
		Runtime:  LocalRuntimeFacts{ID: "llama.cpp", Version: "b1234", Capabilities: []string{"openai-http", "gguf", "cpu", "cuda"}, Verified: true},
		Host:     LocalHostFacts{DiskKnown: true, FreeDiskBytes: 8 * admissionGiB, RAMKnown: true, FreeRAMBytes: 4 * admissionGiB},
	}
}

func refusalCode(t *testing.T, got LocalAdmissionDecision) string {
	t.Helper()
	if got.Verdict != LocalAdmissionRefuse || got.Plan != nil || len(got.Refusals) == 0 {
		t.Fatalf("decision = %+v, want typed refusal and no launch plan", got)
	}
	var typed *LocalAdmissionRefusalError
	if err := got.RefusalError(); !errors.As(err, &typed) || typed.Decision.Refusals[0].Code != got.Refusals[0].Code {
		t.Fatalf("RefusalError() = %T %v, want decision-preserving typed error", err, err)
	}
	return got.Refusals[0].Code
}

func TestEvaluateLocalAdmissionAdmitsSufficientCPU(t *testing.T) {
	req := validLocalAdmissionRequest()
	wantInput := req
	got := EvaluateLocalAdmission(req)
	if got.Verdict != LocalAdmissionAdmit || got.Plan == nil || len(got.Refusals) != 0 {
		t.Fatalf("decision = %+v, want admitted plan", got)
	}
	if got.Plan.DeviceKind != LocalDeviceCPU || got.Plan.CPUFallback || got.Plan.ArtifactSHA256 != req.Declaration.ArtifactSHA256 || got.Plan.Required.RAMBytes != 2*admissionGiB {
		t.Fatalf("CPU launch plan lost joined declaration/artifact/resources: %+v", got.Plan)
	}
	if got.RefusalError() != nil {
		t.Fatalf("admitted decision returned refusal error: %v", got.RefusalError())
	}
	if !reflect.DeepEqual(req, wantInput) {
		t.Fatalf("pure planner mutated its input\nbefore=%+v\nafter=%+v", wantInput, req)
	}
	first, _ := json.Marshal(got)
	second, _ := json.Marshal(EvaluateLocalAdmission(req))
	if string(first) != string(second) || !strings.Contains(string(first), `"schema":"`+LocalAdmissionDecisionSchema+`"`) {
		t.Fatalf("decision JSON is not deterministic/typed:\n%s\n%s", first, second)
	}
}

func TestEvaluateLocalAdmissionRefusesUnavailableRequestedGPU(t *testing.T) {
	req := validLocalAdmissionRequest()
	req.Declaration.Requested = LocalDeviceTarget{DeviceKind: "cuda", DeviceID: "0", Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: admissionGiB, VRAMBytes: 2 * admissionGiB}}
	if got := refusalCode(t, EvaluateLocalAdmission(req)); got != LocalRefusalDeviceUnavailable {
		t.Fatalf("refusal code = %q, want %q", got, LocalRefusalDeviceUnavailable)
	}
}

func TestEvaluateLocalAdmissionRefusesInsufficientMeasuredResources(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate func(*LocalAdmissionRequest)
		code   string
	}{
		"disk": {
			mutate: func(req *LocalAdmissionRequest) { req.Host.FreeDiskBytes = req.Declaration.ArtifactBytes - 1 },
			code:   LocalRefusalDiskInsufficient,
		},
		"ram": {
			mutate: func(req *LocalAdmissionRequest) {
				req.Host.FreeRAMBytes = req.Declaration.Requested.Resources.RAMBytes - 1
			},
			code: LocalRefusalRAMInsufficient,
		},
		"vram": {
			mutate: func(req *LocalAdmissionRequest) {
				req.Declaration.Requested = LocalDeviceTarget{DeviceKind: "cuda", DeviceID: "0", Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: admissionGiB, VRAMBytes: 3 * admissionGiB}}
				req.Host.Devices = []LocalDeviceFacts{{Kind: "cuda", ID: "0", Available: true, VRAMKnown: true, FreeVRAMBytes: 3*admissionGiB - 1}}
			},
			code: LocalRefusalVRAMInsufficient,
		},
	} {
		t.Run(name, func(t *testing.T) {
			req := validLocalAdmissionRequest()
			tc.mutate(&req)
			got := EvaluateLocalAdmission(req)
			if code := refusalCode(t, got); code != tc.code {
				t.Fatalf("refusal code = %q, want %q: %+v", code, tc.code, got)
			}
			if got.Refusals[0].Required <= got.Refusals[0].Available {
				t.Fatalf("resource refusal lacks falsifiable required/available evidence: %+v", got.Refusals[0])
			}
		})
	}
}

func TestEvaluateLocalAdmissionRefusesUnverifiedArtifact(t *testing.T) {
	req := validLocalAdmissionRequest()
	req.Artifact.Verified = false
	if got := refusalCode(t, EvaluateLocalAdmission(req)); got != LocalRefusalArtifactUnverified {
		t.Fatalf("refusal code = %q, want %q", got, LocalRefusalArtifactUnverified)
	}

	req = validLocalAdmissionRequest()
	req.Artifact.SHA256 = strings.Repeat("b", 64)
	if got := refusalCode(t, EvaluateLocalAdmission(req)); got != LocalRefusalArtifactMismatch {
		t.Fatalf("identity refusal code = %q, want %q", got, LocalRefusalArtifactMismatch)
	}
}

func TestEvaluateLocalAdmissionRefusesIncompatibleRuntime(t *testing.T) {
	for name, tc := range map[string]struct {
		mutate func(*LocalAdmissionRequest)
		code   string
	}{
		"runtime":    {func(req *LocalAdmissionRequest) { req.Runtime.ID = "other" }, LocalRefusalRuntimeIncompatible},
		"version":    {func(req *LocalAdmissionRequest) { req.Runtime.Version = "b9999" }, LocalRefusalRuntimeVersion},
		"capability": {func(req *LocalAdmissionRequest) { req.Runtime.Capabilities = []string{"openai-http"} }, LocalRefusalCapabilityMissing},
		"device-capability": {
			func(req *LocalAdmissionRequest) {
				req.Declaration.Requested = LocalDeviceTarget{DeviceKind: "cuda", DeviceID: "0", Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: admissionGiB, VRAMBytes: admissionGiB}}
				req.Runtime.Capabilities = []string{"gguf", "cpu"}
			},
			LocalRefusalCapabilityMissing,
		},
	} {
		t.Run(name, func(t *testing.T) {
			req := validLocalAdmissionRequest()
			tc.mutate(&req)
			if got := refusalCode(t, EvaluateLocalAdmission(req)); got != tc.code {
				t.Fatalf("refusal code = %q, want %q", got, tc.code)
			}
		})
	}
}

func TestEvaluateLocalAdmissionUsesOnlyDeclaredCPUFallback(t *testing.T) {
	req := validLocalAdmissionRequest()
	req.Declaration.Requested = LocalDeviceTarget{DeviceKind: "cuda", DeviceID: "0", Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: admissionGiB, VRAMBytes: 2 * admissionGiB}}
	fallback := LocalDeviceTarget{DeviceKind: LocalDeviceCPU, Resources: LocalResourceRequirements{DiskBytes: admissionGiB, RAMBytes: 3 * admissionGiB}}
	req.Declaration.CPUFallback = &fallback

	got := EvaluateLocalAdmission(req)
	if got.Verdict != LocalAdmissionAdmit || got.Plan == nil || !got.Plan.CPUFallback || got.Plan.DeviceKind != LocalDeviceCPU || got.Plan.Required.RAMBytes != 3*admissionGiB || got.Plan.FallbackReason == nil || got.Plan.FallbackReason.Code != LocalRefusalDeviceUnavailable {
		t.Fatalf("declared CPU fallback was not selected exactly: %+v", got)
	}

	req.Declaration.CPUFallback = nil
	if code := refusalCode(t, EvaluateLocalAdmission(req)); code != LocalRefusalDeviceUnavailable {
		t.Fatalf("missing fallback invented a CPU path: code=%q", code)
	}
}
