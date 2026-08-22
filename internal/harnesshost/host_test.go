package harnesshost

import (
	"strings"
	"testing"

	"github.com/anthony-chaudhary/fak/internal/harnessprofile"
)

func TestResolvedProfileArtifactsRefuseDescriptorDrift(t *testing.T) {
	profiles, err := harnessprofile.Resolve([]byte(`{"harnesses":[{"name":"acme","adapter_version":"2.1.0","names":["acme-cli"],"wire":"openai","repoint":["env"],"credential":{"kind":"env-key","env_key":"ACME_API_KEY"},"identity":"env-key"}]}`))
	if err != nil {
		t.Fatal(err)
	}
	binding, ok, err := harnessprofile.ResolveBinding(profiles, "acme-cli")
	if err != nil || !ok {
		t.Fatalf("resolve binding: ok=%v err=%v", ok, err)
	}
	artifacts, err := BuildResolved(binding, "fixture:harnessprofile", "fixture:harnessprofile/acme@2.1.0", "v1alpha1")
	if err != nil {
		t.Fatal(err)
	}
	if err := VerifyResolved(binding, artifacts); err != nil {
		t.Fatal(err)
	}

	mutated := binding
	mutated.Repoint = append(mutated.Repoint, harnessprofile.RepointCLIConfig)
	if err := VerifyResolved(mutated, artifacts); err == nil || !strings.Contains(err.Error(), "stale") {
		t.Fatalf("old lock accepted after descriptor mutation: %v", err)
	}
}
