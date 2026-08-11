package fp4runtime

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

const fixtureDigest = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

func fixtureBytes(t *testing.T, name string) []byte {
	t.Helper()
	raw, err := os.ReadFile(filepath.Join("testdata", name))
	if err != nil {
		t.Fatal(err)
	}
	return raw
}

func fixtureRequest(t *testing.T, name string) Request {
	t.Helper()
	var request Request
	if err := json.Unmarshal(fixtureBytes(t, name), &request); err != nil {
		t.Fatal(err)
	}
	return request
}

func fixtureMatrix(t *testing.T) Matrix {
	t.Helper()
	var matrix Matrix
	if err := json.Unmarshal(fixtureBytes(t, "compatibility-matrix-v1.json"), &matrix); err != nil {
		t.Fatal(err)
	}
	return matrix
}

func TestCompatibilityMatrixEvidence(t *testing.T) {
	tests := []struct {
		name    string
		fixture string
		outcome Outcome
		reason  ReasonCode
	}{
		{
			name:    "NVFP4 delegates only on an exact Blackwell runtime profile",
			fixture: "nvfp4-blackwell-delegate.json",
			outcome: OutcomeDelegate,
			reason:  ReasonRuntimeDelegationRequired,
		},
		{
			name:    "OCP MXFP4 remains a distinct exact profile",
			fixture: "mxfp4-blackwell-delegate.json",
			outcome: OutcomeDelegate,
			reason:  ReasonRuntimeDelegationRequired,
		},
		{
			name:    "known Hopper architecture is unavailable for the NVFP4 profile",
			fixture: "nvfp4-hopper-refuse.json",
			outcome: OutcomeRefuse,
			reason:  ReasonGPUArchitectureUnavailable,
		},
		{
			name:    "known accumulator semantics cannot silently substitute",
			fixture: "nvfp4-accumulator-refuse.json",
			outcome: OutcomeRefuse,
			reason:  ReasonAccumulatorUnavailable,
		},
		{
			name:    "unknown artifact version abstains",
			fixture: "unknown-artifact-version.json",
			outcome: OutcomeAbstain,
			reason:  ReasonUnknownArtifactVersion,
		},
		{
			name:    "unknown runtime version abstains",
			fixture: "unknown-runtime-version.json",
			outcome: OutcomeAbstain,
			reason:  ReasonUnknownRuntimeVersion,
		},
		{
			name:    "known runtime without an FP4 profile refuses",
			fixture: "fak-native-refuse.json",
			outcome: OutcomeRefuse,
			reason:  ReasonRuntimeUnavailable,
		},
	}
	matrixRaw := fixtureBytes(t, "compatibility-matrix-v1.json")
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			got, err := ParseAndNegotiate(fixtureBytes(t, test.fixture), matrixRaw)
			if err != nil {
				t.Fatal(err)
			}
			if got.Outcome != test.outcome || got.Reason != test.reason {
				t.Fatalf("got %s/%s (%s), want %s/%s", got.Outcome, got.Reason, got.Detail, test.outcome, test.reason)
			}
		})
	}
}

func TestUnknownSchemaAndFieldAreTypedAbstentions(t *testing.T) {
	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	request.Schema = "fak.fp4runtime/v99"
	got := Negotiate(request, matrix)
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownSchema {
		t.Fatalf("unknown schema: got %#v", got)
	}

	raw := fixtureBytes(t, "nvfp4-blackwell-delegate.json")
	raw = []byte(strings.Replace(string(raw), `"schema":`, `"future_field":true,"schema":`, 1))
	got, err := ParseAndNegotiate(raw, fixtureBytes(t, "compatibility-matrix-v1.json"))
	if err == nil {
		t.Fatal("strict parser accepted an unknown field")
	}
	if got.Outcome != OutcomeAbstain || got.Reason != ReasonUnknownField {
		t.Fatalf("unknown field: got %#v err=%v", got, err)
	}
}

func TestUnknownVocabularyAbstainsWithoutGuessing(t *testing.T) {
	matrix := fixtureMatrix(t)
	base := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	tests := []struct {
		name   string
		mutate func(*Request)
		reason ReasonCode
	}{
		{
			name:   "artifact",
			mutate: func(request *Request) { request.Artifact.Pin.ID = "future-fp4" },
			reason: ReasonUnknownArtifact,
		},
		{
			name:   "runtime",
			mutate: func(request *Request) { request.Runtime.ID = "future-runtime" },
			reason: ReasonUnknownRuntime,
		},
		{
			name:   "architecture",
			mutate: func(request *Request) { request.GPU.Architecture = "sm_999" },
			reason: ReasonUnknownGPUArchitecture,
		},
		{
			name:   "accumulator",
			mutate: func(request *Request) { request.Accumulator.ID = "future-accumulator" },
			reason: ReasonUnknownAccumulator,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			request := base
			test.mutate(&request)
			got := Negotiate(request, matrix)
			if got.Outcome != OutcomeAbstain || got.Reason != test.reason {
				t.Fatalf("got %#v", got)
			}
		})
	}
}

func TestMalformedJSONAndInvalidPinsAreTypedRefusals(t *testing.T) {
	got, err := ParseAndNegotiate([]byte(`{"schema":`), fixtureBytes(t, "compatibility-matrix-v1.json"))
	if err == nil || got.Outcome != OutcomeRefuse || got.Reason != ReasonInvalidJSON {
		t.Fatalf("malformed request: got %#v err=%v", got, err)
	}

	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	request.Recipe.SHA256 = "not-a-digest"
	got = Negotiate(request, matrix)
	if got.Outcome != OutcomeRefuse || got.Reason != ReasonInvalidRequest {
		t.Fatalf("invalid recipe pin: got %#v", got)
	}
}

func TestExactNativeProfileCanAllowWithoutBecomingADelegation(t *testing.T) {
	matrix := fixtureMatrix(t)
	matrix.Profiles[0].Mode = ModeNative
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	got := Negotiate(request, matrix)
	if got.Outcome != OutcomeAllow || got.Reason != ReasonAdmitted {
		t.Fatalf("got %#v", got)
	}
	if got.Claims.Runtime.External {
		t.Fatalf("native profile became a delegation: %#v", got.Claims.Runtime)
	}
}

func TestArtifactSemanticsAreExactNotNameBased(t *testing.T) {
	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	request.Artifact.ScaleFormat = "e8m0"
	request.Artifact.BlockSize = 32
	got := Negotiate(request, matrix)
	if got.Outcome != OutcomeRefuse || got.Reason != ReasonArtifactSemanticsMismatch {
		t.Fatalf("got %#v", got)
	}
}

func TestPublicIDsMatchTheCompatibilityFixture(t *testing.T) {
	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	if ArtifactID(request.Artifact.Pin.ID) != ArtifactNVIDIANVFP4 ||
		RuntimeID(request.Runtime.ID) != RuntimeTensorRTLLMPyTorch ||
		request.GPU.Architecture != ArchitectureSM100 ||
		request.Accumulator.ID != AccumulatorFP32BF16RNE {
		t.Fatalf("public IDs drifted from fixture: %#v", request)
	}
	if matrix.Artifacts[1].ID != ArtifactOCPMXFP4 || matrix.Runtimes[1].ID != RuntimeFakNative {
		t.Fatalf("matrix public IDs drifted: %#v", matrix)
	}
}

func TestClaimsKeepArtifactRecipeRuntimeAndHardwareSeparate(t *testing.T) {
	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	got := Negotiate(request, matrix)
	if got.Outcome != OutcomeDelegate {
		t.Fatalf("got %#v", got)
	}
	if got.Claims.Artifact != request.Artifact.Pin {
		t.Fatalf("artifact claim collapsed: %#v", got.Claims.Artifact)
	}
	if got.Claims.Recipe != request.Recipe {
		t.Fatalf("recipe claim collapsed: %#v", got.Claims.Recipe)
	}
	if got.Claims.Runtime.Pin != request.Runtime || !got.Claims.Runtime.External {
		t.Fatalf("runtime claim collapsed: %#v", got.Claims.Runtime)
	}
	if got.Claims.Hardware.Observed {
		t.Fatalf("compatibility-only output masqueraded as measured hardware: %#v", got.Claims.Hardware)
	}
	if got.Claims.Hardware.Architecture != request.GPU.Architecture {
		t.Fatalf("hardware envelope lost: %#v", got.Claims.Hardware)
	}
}

func TestObservedHardwareRequiresIndependentMatchingEvidence(t *testing.T) {
	matrix := fixtureMatrix(t)
	request := fixtureRequest(t, "nvfp4-blackwell-delegate.json")
	request.HardwareEvidence = &HardwareEvidence{
		Source:            "sanctioned-gpu:nvidia-smi",
		RunSHA256:         fixtureDigest,
		RuntimeSHA256:     request.Runtime.SHA256,
		Architecture:      request.GPU.Architecture,
		AccumulatorID:     request.Accumulator.ID,
		DeviceFingerprint: "vendor=nvidia;arch=sm_100;class=blackwell",
		Command:           SanctionedGPUEvidenceCommand,
	}
	got := Negotiate(request, matrix)
	if got.Outcome != OutcomeDelegate || !got.Claims.Hardware.Observed || got.Claims.Hardware.EvidenceSHA256 != fixtureDigest {
		t.Fatalf("complete evidence rejected or lost: %#v", got)
	}

	request.HardwareEvidence.RuntimeSHA256 = strings.Repeat("a", 64)
	got = Negotiate(request, matrix)
	if got.Outcome != OutcomeRefuse || got.Reason != ReasonHardwareEvidenceMismatch {
		t.Fatalf("mismatched runtime evidence: got %#v", got)
	}
}

func TestInvalidMatrixCannotCreateAnAmbiguousAdmission(t *testing.T) {
	matrix := fixtureMatrix(t)
	matrix.Profiles = append(matrix.Profiles, matrix.Profiles[0])
	got := Negotiate(fixtureRequest(t, "nvfp4-blackwell-delegate.json"), matrix)
	if got.Outcome != OutcomeRefuse || got.Reason != ReasonInvalidMatrix {
		t.Fatalf("duplicate profile: got %#v", got)
	}
}

func TestSanctionedCommandNamesTheRealHardwareEvidence(t *testing.T) {
	for _, want := range []string{
		"FAK_FP4_GPU_EVIDENCE=1",
		"FAK_FP4_EXPECT_OUTCOME=delegate",
		"tensorrt_llm.__version__",
		"TestSanctionedGPUEvidence",
	} {
		if !strings.Contains(SanctionedGPUEvidenceCommand, want) {
			t.Fatalf("sanctioned command missing %q: %s", want, SanctionedGPUEvidenceCommand)
		}
	}
}
