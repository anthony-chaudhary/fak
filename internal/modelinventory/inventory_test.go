package modelinventory_test

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/anthony-chaudhary/fak/internal/modelinventory"
)

var (
	asOf      = time.Date(2026, 8, 19, 18, 0, 0, 0, time.UTC)
	observed  = "2026-08-19T17:00:00Z"
	expires   = "2026-08-20T17:00:00Z"
	digestA   = "sha256:" + strings.Repeat("a", 64)
	digestB   = "sha256:" + strings.Repeat("b", 64)
	revisionA = strings.Repeat("c", 40)
)

func witness(kind modelinventory.WitnessKind, source string) modelinventory.Witness {
	return modelinventory.Witness{Kind: kind, Source: source, ObservedAt: observed, ExpiresAt: expires}
}

func fact(name string, value modelinventory.Value, source string) modelinventory.Fact {
	return modelinventory.Fact{Name: name, Value: value, Witnesses: []modelinventory.Witness{witness(modelinventory.EvidenceProbe, source)}}
}

func evidence(prefix, accelerator string, memory int64) modelinventory.EvidenceSet {
	return modelinventory.EvidenceSet{
		Availability: fact("available", modelinventory.Bool(true), prefix+"/availability"),
		Serving: []modelinventory.Fact{
			fact("runtime", modelinventory.Text("vllm"), prefix+"/runtime"),
			fact("protocol", modelinventory.Text("openai-compatible"), prefix+"/protocol"),
		},
		Platform: []modelinventory.Fact{
			fact("os", modelinventory.Text("linux"), prefix+"/os"),
			fact("architecture", modelinventory.Text("amd64"), prefix+"/arch"),
			fact("accelerator", modelinventory.Text(accelerator), prefix+"/accelerator"),
			fact("accelerator_memory_bytes", modelinventory.Integer(memory), prefix+"/memory"),
		},
		Policy: []modelinventory.Fact{
			fact("locality", modelinventory.Text("local"), prefix+"/locality"),
			fact("license", modelinventory.Text("apache-2.0"), prefix+"/license"),
		},
		Capabilities: []modelinventory.Fact{
			fact("tool_calling", modelinventory.Bool(true), prefix+"/tools"),
			fact("structured_json", modelinventory.Bool(true), prefix+"/json"),
			fact("context_tokens", modelinventory.Integer(131072), prefix+"/context"),
		},
	}
}

func fixture() modelinventory.Observations {
	return modelinventory.Observations{
		Providers: []modelinventory.ProviderObservation{{
			ID:         "hub-planner",
			Provider:   "huggingface-hub",
			Repository: "acme/planner-model",
			Revision:   revisionA,
			Digest:     digestA,
			Format:     "safetensors",
			IdentityEvidence: []modelinventory.Witness{
				witness(modelinventory.EvidenceArtifactMetadata, "hf://acme/planner-model@"+revisionA),
			},
			Evidence: evidence("probe://planner", "cuda", 24<<30),
		}},
		Locals: []modelinventory.LocalObservation{{
			ID:       "local-executor",
			Artifact: "models/executor.gguf",
			Digest:   digestB,
			Format:   "gguf",
			IdentityEvidence: []modelinventory.Witness{
				witness(modelinventory.EvidenceProbe, "file-digest://models/executor.gguf"),
			},
			Evidence: evidence("probe://executor", "cpu", 0),
		}},
	}
}

func TestNormalizeProviderAndLocalIsByteStableAndReadable(t *testing.T) {
	first, diagnostics := modelinventory.Normalize(fixture(), asOf)
	if len(diagnostics) != 0 {
		t.Fatalf("normalize diagnostics:\n%s", diagnostics)
	}
	input := fixture()
	input.Providers[0].Evidence.Capabilities[0], input.Providers[0].Evidence.Capabilities[2] = input.Providers[0].Evidence.Capabilities[2], input.Providers[0].Evidence.Capabilities[0]
	second, diagnostics := modelinventory.Normalize(input, asOf)
	if len(diagnostics) != 0 {
		t.Fatalf("second normalize diagnostics:\n%s", diagnostics)
	}
	firstJSON, diagnostics := first.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatalf("first canonical JSON diagnostics:\n%s", diagnostics)
	}
	secondJSON, diagnostics := second.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatalf("second canonical JSON diagnostics:\n%s", diagnostics)
	}
	if !bytes.Equal(firstJSON, secondJSON) {
		t.Fatalf("canonical reruns differ:\n%s\n%s", firstJSON, secondJSON)
	}
	if bytes.Contains(firstJSON, []byte("authorization")) || bytes.Contains(firstJSON, []byte("credential")) {
		t.Fatalf("inventory exposed credential surface: %s", firstJSON)
	}
	readBack, diagnostics := modelinventory.ParseJSON(firstJSON, asOf)
	if len(diagnostics) != 0 {
		t.Fatalf("read-back diagnostics:\n%s", diagnostics)
	}
	if len(readBack.Candidates) != 2 || readBack.Candidates[0].ID != "hub-planner" || readBack.Candidates[1].ID != "local-executor" {
		t.Fatalf("read-back candidates = %+v", readBack.Candidates)
	}
	if got := readBack.Candidates[0].Identity.Witnesses[0].Source; got != "hf://acme/planner-model@"+revisionA {
		t.Fatalf("provider identity provenance = %q", got)
	}
}

func TestNormalizeFailsClosedWithStableTypedDiagnostics(t *testing.T) {
	cases := []struct {
		name string
		edit func(*modelinventory.Observations)
		code modelinventory.Code
	}{
		{
			name: "missing identity",
			edit: func(in *modelinventory.Observations) { in.Providers[0].Revision = "" },
			code: modelinventory.CodeMissingIdentity,
		},
		{
			name: "expired evidence",
			edit: func(in *modelinventory.Observations) {
				in.Locals[0].Evidence.Capabilities[0].Witnesses[0].ExpiresAt = "2026-08-19T17:30:00Z"
			},
			code: modelinventory.CodeEvidenceStale,
		},
		{
			name: "unknown evidence",
			edit: func(in *modelinventory.Observations) { in.Locals[0].Evidence.Capabilities[0].Witnesses = nil },
			code: modelinventory.CodeEvidenceUnknown,
		},
		{
			name: "platform contradiction",
			edit: func(in *modelinventory.Observations) {
				in.Locals[0].Evidence.Platform[3].Value = modelinventory.Integer(1024)
			},
			code: modelinventory.CodePlatformContradiction,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			input := fixture()
			tc.edit(&input)
			inventory, diagnostics := modelinventory.Normalize(input, asOf)
			if len(inventory.Candidates) != 0 {
				t.Fatalf("invalid input returned candidates: %+v", inventory.Candidates)
			}
			if !hasCode(diagnostics, tc.code) {
				t.Fatalf("diagnostics = %s, want code %s", diagnostics, tc.code)
			}
			first := diagnostics.CanonicalJSON()
			second := diagnostics.CanonicalJSON()
			if !bytes.Equal(first, second) {
				t.Fatalf("diagnostics are unstable:\n%s\n%s", first, second)
			}
			if !bytes.Contains(first, []byte(`"remediation"`)) || !bytes.Contains(first, []byte(`"field"`)) {
				t.Fatalf("diagnostic omits actionable fields: %s", first)
			}
		})
	}
}

func TestCredentialMaterialCannotSerialize(t *testing.T) {
	input := fixture()
	input.Providers[0].Evidence.Serving[0].Witnesses[0].Source = "https://models.example/evidence?api_key=do-not-serialize"
	inventory, diagnostics := modelinventory.Normalize(input, asOf)
	if !hasCode(diagnostics, modelinventory.CodeCredentialMaterial) || len(inventory.Candidates) != 0 {
		t.Fatalf("credential input did not fail closed: inventory=%+v diagnostics=%s", inventory, diagnostics)
	}
	if bytes.Contains(diagnostics.CanonicalJSON(), []byte("do-not-serialize")) {
		t.Fatalf("diagnostic leaked credential: %s", diagnostics.CanonicalJSON())
	}

	valid, diagnostics := modelinventory.Normalize(fixture(), asOf)
	if len(diagnostics) != 0 {
		t.Fatalf("valid fixture diagnostics: %s", diagnostics)
	}
	valid.Candidates[0].Evidence.Policy[0].Value = modelinventory.Text("Bearer do-not-serialize")
	if encoded, err := json.Marshal(valid); err == nil {
		t.Fatalf("mutated credential-bearing inventory serialized: %s", encoded)
	}
}

func TestParseJSONIsStrictAndRechecksFreshness(t *testing.T) {
	inventory, diagnostics := modelinventory.Normalize(fixture(), asOf)
	if len(diagnostics) != 0 {
		t.Fatalf("fixture diagnostics: %s", diagnostics)
	}
	encoded, diagnostics := inventory.CanonicalJSON()
	if len(diagnostics) != 0 {
		t.Fatalf("encode diagnostics: %s", diagnostics)
	}
	withUnknown := bytes.Replace(encoded, []byte(`"schema":`), []byte(`"unknown":true,"schema":`), 1)
	if _, diagnostics := modelinventory.ParseJSON(withUnknown, asOf); !hasCode(diagnostics, modelinventory.CodeInvalidJSON) {
		t.Fatalf("unknown field diagnostics = %s", diagnostics)
	}
	if _, diagnostics := modelinventory.ParseJSON(encoded, asOf.Add(48*time.Hour)); !hasCode(diagnostics, modelinventory.CodeEvidenceStale) {
		t.Fatalf("stale read-back diagnostics = %s", diagnostics)
	}
}

func TestRepositoryMetadataCannotWitnessCapabilities(t *testing.T) {
	input := fixture()
	input.Providers[0].Evidence.Capabilities[0].Witnesses[0].Kind = modelinventory.EvidenceArtifactMetadata
	inventory, diagnostics := modelinventory.Normalize(input, asOf)
	if len(inventory.Candidates) != 0 || !hasCode(diagnostics, modelinventory.CodeEvidenceScopeMismatch) {
		t.Fatalf("repository metadata was accepted as capability proof: inventory=%+v diagnostics=%s", inventory, diagnostics)
	}
}

func hasCode(diagnostics modelinventory.Diagnostics, code modelinventory.Code) bool {
	for _, diagnostic := range diagnostics {
		if diagnostic.Code == code {
			return true
		}
	}
	return false
}
