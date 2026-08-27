package newmodel

import (
	"encoding/json"
	"os"
	"reflect"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/modeldescriptor"
)

func TestCompilePinnedReleaseIsDeterministicAndRefusesSemanticDelta(t *testing.T) {
	valid, err := os.ReadFile("testdata/qwen38-release.json")
	if err != nil {
		t.Fatal(err)
	}
	first, err := CompileReleaseManifest(valid)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileReleaseManifest(valid)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatalf("packet changed across identical compiles:\nfirst=%+v\nsecond=%+v", first, second)
	}
	firstJSON, err := json.Marshal(first)
	if err != nil {
		t.Fatal(err)
	}
	secondJSON, err := json.Marshal(second)
	if err != nil {
		t.Fatal(err)
	}
	if string(firstJSON) != string(secondJSON) {
		t.Fatal("packet JSON is not byte deterministic")
	}
	if first.Schema != OnboardingPacketSchema || first.Execution.Engine != "fak-native" || first.Execution.ExternalFallback {
		t.Fatalf("execution identity = %+v", first.Execution)
	}
	if first.ManifestSHA256 == "" || first.Source.SHA256 == "" || first.Artifact.SHA256 == "" || first.DescriptorSHA256 == "" {
		t.Fatalf("packet lost pinned digests: %+v", first)
	}
	if first.Descriptor.Aliases[0] != "qwen35forconditionalgeneration" || first.Descriptor.Aliases[1] != "qwen38" {
		t.Fatalf("aliases were not normalized: %v", first.Descriptor.Aliases)
	}
	if len(first.OpenObligations) != 5 || len(first.SupportLadder) != 6 || first.SupportLadder[5].Status != "blocked" {
		t.Fatalf("onboarding debt is incomplete: obligations=%v ladder=%v", first.OpenObligations, first.SupportLadder)
	}
	if err := modeldescriptor.Validate(first.Descriptor); err != nil {
		t.Fatal(err)
	}

	mutated, err := os.ReadFile("testdata/qwen38-semantic-delta.json")
	if err != nil {
		t.Fatal(err)
	}
	_, err = CompileReleaseManifest(mutated)
	refusal, ok := RefusalFor(err)
	if !ok {
		t.Fatalf("mutation error = %v, want typed refusal", err)
	}
	if refusal.Code != RefusalUnresolvedSemanticAxis || refusal.Phase != "pre-allocation" || refusal.Axis != "rotary_scaling_semantics" {
		t.Fatalf("refusal = %+v", refusal)
	}
}
